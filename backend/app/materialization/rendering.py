"""Map a persisted Stage 6 materialization into a typed Stage 5.2 render request.

This module owns no FFmpeg and no provider work: it reads the accepted authored
timeline and audio recipe from the durable envelope, binds them to the exact
current Stage 5.0 contract / Stage 5.1 composition, builds the deterministic
authored ASS in output time, and returns a :class:`RenderSpec` whose optional
``authored`` payload the frozen renderer executes.

The builder loads the *exact* materialization bound to the queued render
execution (never whichever row is currently newest) and fails closed when the
materialization is not current, the expected output fingerprint does not match,
or a source mapping is inconsistent with the frozen Stage 5.0 rebound bindings.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.materialization.overlays import build_authored_ass
from app.render.execution.authored import (
    ASSET_ROLE_NARRATION,
    ASSET_ROLE_SOURCE_VIDEO,
    ASSET_ROLE_SUPPORTING_IMAGE,
    ASSET_ROLE_SUPPORTING_VIDEO,
    AUTHORED_SCHEMA_VERSION,
    MASTERING_MODE_LINEAR,
    UNIT_CARD,
    UNIT_SOURCE,
    AssAssetRef,
    AudioTrackSpec,
    AuthoredRenderPayload,
    GainPoint,
    MasteringSpec,
    MediaAssetRef,
    RenderUnit,
)
from app.render.execution.policy import (
    AUTHORED_MATERIALIZATION_BINDING_INVALID,
    AUTHORED_SOURCE_MAPPING_INVALID,
    CORE_SOURCE_VALIDATION,
    TRANSFORMED_SHORT,
)
from app.render.execution.service import RenderExecutionError, build_render_spec
from app.render.execution.types import RenderSpec
from app.services.storage import StorageService

_SOURCE_ASSET_ID = "source"
_SOURCE_SPAN_TOLERANCE = 0.06
_RENDER_GUARD_DBTP = -1.8


def _as_uuid(value: object) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str) and value:
        try:
            return uuid.UUID(value)
        except ValueError:
            return None
    return None


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _segments(timeline_payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = timeline_payload.get("segments")
    if not isinstance(raw, list):
        raise RenderExecutionError(AUTHORED_MATERIALIZATION_BINDING_INVALID)
    return [item for item in raw if isinstance(item, Mapping)]


def _entries(audio_recipe: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = audio_recipe.get("entries")
    if not isinstance(raw, list):
        raise RenderExecutionError(AUTHORED_MATERIALIZATION_BINDING_INVALID)
    return [item for item in raw if isinstance(item, Mapping)]


def build_transformed_render_spec(
    session: Session,
    candidate: Any,
    *,
    materialization_id: str | uuid.UUID,
    expected_output_fingerprint: str,
    delivery_profile_key: str,
    settings: Settings,
    storage: StorageService,
    stream_probe: Any | None = None,
) -> RenderSpec:
    """Assemble a TRANSFORMED_SHORT render request from persisted Stage 6 output."""

    from app.models import ClipMaterialization

    materialization_uuid = _as_uuid(materialization_id)
    if materialization_uuid is None:
        raise RenderExecutionError(AUTHORED_MATERIALIZATION_BINDING_INVALID)
    row = session.get(ClipMaterialization, materialization_uuid, populate_existing=True)
    if row is None or not row.is_current:
        raise RenderExecutionError(AUTHORED_MATERIALIZATION_BINDING_INVALID)
    if not row.output_fingerprint or row.output_fingerprint != expected_output_fingerprint:
        raise RenderExecutionError(AUTHORED_MATERIALIZATION_BINDING_INVALID)
    timeline_payload = dict(row.timeline_payload or {})
    audio_recipe = dict(row.audio_recipe or {})
    if not timeline_payload or not audio_recipe:
        raise RenderExecutionError(AUTHORED_MATERIALIZATION_BINDING_INVALID)

    base = build_render_spec(
        session,
        candidate,
        artifact_purpose=CORE_SOURCE_VALIDATION,
        delivery_profile_key=delivery_profile_key,
        settings=settings,
        storage=storage,
        stream_probe=stream_probe,
    )
    frame_rate = float(base.output_frame_rate)
    if frame_rate <= 0:
        raise RenderExecutionError(AUTHORED_SOURCE_MAPPING_INVALID)
    sample_rate = 48000

    base_occurrences = {occurrence.block_index: occurrence for occurrence in base.occurrences}
    units: list[RenderUnit] = []
    authored_occurrences: list[Any] = []
    cursor_frame = 0
    for segment in _segments(timeline_payload):
        kind = str(segment.get("kind") or "")
        if kind == "VERIFICATION_DEPENDENCY":
            continue
        block_index = _int(segment.get("plan_block_index"), -1)
        start_frame = int(round(_float(segment.get("output_start")) * frame_rate))
        end_frame = int(round(_float(segment.get("output_end")) * frame_rate))
        if end_frame < start_frame or start_frame != cursor_frame:
            raise RenderExecutionError(AUTHORED_SOURCE_MAPPING_INVALID)
        cursor_frame = end_frame
        unit_id = f"u-{str(segment.get('segment_id') or block_index)}"
        if kind == "SOURCE":
            occurrence = base_occurrences.get(block_index)
            if occurrence is None:
                raise RenderExecutionError(AUTHORED_SOURCE_MAPPING_INVALID)
            seg_start = _optional_float(segment.get("source_start"))
            seg_end = _optional_float(segment.get("source_end"))
            if (
                seg_start is None
                or seg_end is None
                or abs(seg_start - occurrence.source_start) > _SOURCE_SPAN_TOLERANCE
                or abs(seg_end - occurrence.source_end) > _SOURCE_SPAN_TOLERANCE
            ):
                raise RenderExecutionError(AUTHORED_SOURCE_MAPPING_INVALID)
            units.append(
                RenderUnit(
                    unit_id=unit_id,
                    kind=UNIT_SOURCE,
                    block_index=block_index,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    occurrence_id=occurrence.occurrence_id,
                )
            )
            authored_occurrences.append(
                replace(
                    occurrence,
                    output_start=_float(segment.get("output_start")),
                    output_end=_float(segment.get("output_end")),
                )
            )
        elif kind in {"AUTHORED_CARD", "TRANSITION"}:
            units.append(
                RenderUnit(
                    unit_id=unit_id,
                    kind=UNIT_CARD,
                    block_index=block_index,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )
        else:
            raise RenderExecutionError(AUTHORED_SOURCE_MAPPING_INVALID)
    if not units:
        raise RenderExecutionError(AUTHORED_SOURCE_MAPPING_INVALID)
    frame_count = _int(timeline_payload.get("output_frame_count"), cursor_frame)
    if frame_count != cursor_frame:
        raise RenderExecutionError(AUTHORED_SOURCE_MAPPING_INVALID)

    assets, narration_assets = _assets(row, base, sample_rate)
    tracks = _tracks(audio_recipe, narration_assets, sample_rate)
    authored_ass = _build_and_store_authored_ass(
        session, row, timeline_payload, base, settings, storage
    )
    mastering = MasteringSpec(
        target_lufs=_float(audio_recipe.get("target_integrated_lufs"), -16.0),
        ceiling_dbtp=_float(audio_recipe.get("true_peak_ceiling_dbtp"), -1.5),
        render_guard_dbtp=_RENDER_GUARD_DBTP,
        sample_rate=sample_rate,
        channels=2,
        mode=MASTERING_MODE_LINEAR,
        repair_peak_limit_db=0.0,
    )
    payload = AuthoredRenderPayload(
        schema_version=AUTHORED_SCHEMA_VERSION,
        materialization_id=str(row.id),
        materialization_output_fingerprint=row.output_fingerprint,
        repair_revision=int(row.repair_count or 0),
        units=tuple(units),
        assets=tuple(assets),
        tracks=tuple(tracks),
        authored_ass=authored_ass,
        mastering=mastering,
        frame_count=frame_count,
        sample_count=_int(timeline_payload.get("output_sample_count"), cursor_frame),
        source_protection_evidence=dict(row.metrics.get("source_protection", {}) or {}),
    )
    spec = replace(
        base,
        render_execution_id="",
        artifact_purpose=TRANSFORMED_SHORT,
        authored=payload,
        occurrences=tuple(authored_occurrences),
    )
    return spec


def _assets(
    row: Any, base: RenderSpec, sample_rate: int
) -> tuple[list[MediaAssetRef], dict[str, MediaAssetRef]]:
    source_asset = MediaAssetRef(
        asset_id=_SOURCE_ASSET_ID,
        role=ASSET_ROLE_SOURCE_VIDEO,
        storage_relative_path=base.source_media_relative_path,
        sha256=base.source_content_hash or "0" * 64,
        size_bytes=max(1, int(base.source_size_bytes)),
        duration_seconds=max(0.0, float(base.source_duration)),
        sample_rate=sample_rate,
        channels=max(1, int(base.audio_channels)),
        width=base.display_width,
        height=base.display_height,
    )
    assets: list[MediaAssetRef] = [source_asset]
    narration_assets: dict[str, MediaAssetRef] = {}
    for reference in row.narration_references or []:
        if not isinstance(reference, Mapping):
            continue
        chunk_id = str(reference.get("chunk_id") or "")
        relative = str(reference.get("relative_path") or "")
        sha = str(reference.get("sha256") or "")
        if not chunk_id or not relative or not sha:
            continue
        asset = MediaAssetRef(
            asset_id=f"narr-{chunk_id}",
            role=ASSET_ROLE_NARRATION,
            storage_relative_path=relative,
            sha256=sha,
            size_bytes=max(1, _int(reference.get("size_bytes"), 1)),
            duration_seconds=max(0.0, _float(reference.get("duration_seconds"))),
            sample_rate=_int(reference.get("sample_rate"), 24000) or 24000,
            channels=_int(reference.get("channels"), 1) or 1,
        )
        assets.append(asset)
        narration_assets[chunk_id] = asset
    for reference in row.asset_references or []:
        if not isinstance(reference, Mapping):
            continue
        asset_id = str(reference.get("asset_id") or "")
        role = str(reference.get("role") or "")
        if role not in {ASSET_ROLE_SUPPORTING_IMAGE, ASSET_ROLE_SUPPORTING_VIDEO}:
            continue
        if not asset_id:
            continue
        assets.append(
            MediaAssetRef(
                asset_id=asset_id,
                role=role,
                storage_relative_path=str(reference.get("relative_path") or ""),
                sha256=str(reference.get("sha256") or ""),
                size_bytes=max(1, _int(reference.get("size_bytes"), 1)),
                duration_seconds=max(0.0, _float(reference.get("duration_seconds"))),
            )
        )
    return assets, narration_assets


def _tracks(
    audio_recipe: Mapping[str, object],
    narration_assets: Mapping[str, MediaAssetRef],
    sample_rate: int,
) -> list[AudioTrackSpec]:
    tracks: list[AudioTrackSpec] = []
    for entry in _entries(audio_recipe):
        kind = str(entry.get("kind") or "")
        output_start_sample = int(round(_float(entry.get("output_start")) * sample_rate))
        output_end_sample = int(round(_float(entry.get("output_end")) * sample_rate))
        if output_end_sample <= output_start_sample:
            raise RenderExecutionError(AUTHORED_MATERIALIZATION_BINDING_INVALID)
        if kind == "SOURCE":
            tracks.append(
                AudioTrackSpec(
                    track_id=f"src-{str(entry.get('entry_id') or len(tracks))}",
                    asset_id=_SOURCE_ASSET_ID,
                    source_start_seconds=_float(entry.get("source_start")),
                    source_end_seconds=_float(entry.get("source_end")),
                    output_start_sample=output_start_sample,
                    output_end_sample=output_end_sample,
                    gain_points=(GainPoint(sample_index=output_start_sample, gain_db=0.0),),
                )
            )
        elif kind == "NARRATION":
            chunk_id = str(entry.get("narration_chunk_id") or "")
            asset = narration_assets.get(chunk_id)
            if asset is None:
                raise RenderExecutionError(AUTHORED_MATERIALIZATION_BINDING_INVALID)
            tracks.append(
                AudioTrackSpec(
                    track_id=f"narr-{chunk_id}",
                    asset_id=asset.asset_id,
                    source_start_seconds=0.0,
                    source_end_seconds=max(0.0, asset.duration_seconds),
                    output_start_sample=output_start_sample,
                    output_end_sample=output_end_sample,
                    gain_points=(GainPoint(sample_index=output_start_sample, gain_db=0.0),),
                )
            )
    if not tracks:
        raise RenderExecutionError(AUTHORED_MATERIALIZATION_BINDING_INVALID)
    return tracks


def _build_and_store_authored_ass(
    session: Session,
    row: Any,
    timeline_payload: Mapping[str, object],
    base: RenderSpec,
    settings: Settings,
    storage: StorageService,
) -> AssAssetRef:
    from app.materialization.types import AuthoredTimeline, TimelineSegment

    segments = tuple(
        TimelineSegment(
            segment_id=str(segment.get("segment_id") or ""),
            plan_block_index=_int(segment.get("plan_block_index"), -1),
            kind=str(segment.get("kind") or ""),
            output_start=_float(segment.get("output_start")),
            output_end=_float(segment.get("output_end")),
            text=str(segment.get("text") or ""),
        )
        for segment in _segments(timeline_payload)
    )
    timeline = AuthoredTimeline(
        segments=segments,
        frame_rate=base.output_frame_rate,
        sample_rate=48000,
        output_duration=_float(timeline_payload.get("output_duration")),
        output_frame_count=_int(timeline_payload.get("output_frame_count")),
        output_sample_count=_int(timeline_payload.get("output_sample_count")),
        source_caption_asset=base.ass.relative_path,
    )
    result = build_authored_ass(
        timeline,
        play_res_x=base.output_width,
        play_res_y=base.output_height,
        font_family="DejaVu Sans",
    )
    directory = storage.source_directory(row.source_video_id) / "materializations" / str(row.id)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "authored.ass"
    storage.atomic_write(destination, [result.data])
    return AssAssetRef(
        relative_path=storage.storage_relative(destination),
        sha256=result.sha256,
        policy_version=result.policy_version,
    )


__all__ = ["build_transformed_render_spec"]
