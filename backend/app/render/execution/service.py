"""DB adaptation for Stage 5.2 render execution.

This module builds the immutable :class:`RenderSpec` from live rows, validates
currentness/readiness/ASS/source identity, and persists the durable execution
envelope. Media components never touch the database; all discovery lives here.

Currentness is recomputed from live dependencies. The full accepted Stage 5.1
plan input fingerprint is recomputed through the frozen composition service
(stat-only, never a probe), so a changed source, contract, selection, plan, or
FINAL_CLIP binding invalidates the request.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from sqlalchemy import exists, select, update
from sqlalchemy.orm import Session

from app.composition.geometry import FFprobeDisplayProbe
from app.composition.policy import FramingMode
from app.composition.service import read_visual_composition
from app.core.enums import (
    CandidateDisposition,
    JobStatus,
    RenderArtifactPurpose,
    RenderExecutionLifecycle,
    RenderQCStatus,
)
from app.core.settings import Settings, get_settings
from app.models import ClipCandidate, ProcessingJob, VisualCompositionPlan
from app.models.render_contract import RenderContract
from app.models.render_execution import RenderExecution
from app.render.execution.fingerprints import (
    render_compiled_fingerprint,
    render_output_fingerprint,
    render_qc_fingerprint,
    render_request_fingerprint,
    render_runtime_fingerprint,
)
from app.render.execution.policy import (
    CORE_SOURCE_VALIDATION,
    EXECUTION_FINGERPRINT_VERSION,
    EXECUTION_POLICY_VERSION,
    EXECUTION_SCHEMA_VERSION,
    SOURCE_STREAMS_UNSUPPORTED,
    SUPPORTED_ARTIFACT_PURPOSES,
    Stage52Config,
    delivery_profile_for,
    qc_payload,
    stage52_config_payload,
)
from app.render.execution.qc import check_render_artifact
from app.render.execution.timeline import build_timeline
from app.render.execution.types import (
    AssAsset,
    CaptionEventSpec,
    CropKeyframeSpec,
    OmittedRequirement,
    RenderArtifacts,
    RenderSpec,
    RuntimeIdentity,
    SceneSpec,
    TechnicalQCResult,
    TimelineOccurrence,
)
from app.render.execution.validation import (
    RenderValidationError,
    ass_file_facts,
    validate_ass_asset,
    validate_spec,
)
from app.render.policy import EXECUTABLE_STATUSES
from app.render.service import read_render_contract
from app.services.hashing import sha256_file
from app.services.storage import StorageService

_VALID_DISPOSITIONS = {
    CandidateDisposition.CANDIDATE,
    CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
}

_CROP_MODES = frozenset(
    {
        FramingMode.STATIC_CROP.value,
        FramingMode.TRACKED_CROP.value,
        FramingMode.MULTI_SUBJECT_FIT.value,
        FramingMode.CENTER_FALLBACK.value,
    }
)
_ALL_MODES = frozenset(mode.value for mode in FramingMode)


class RenderExecutionError(ValueError):
    """A Stage 5.2 request is stale, blocked, or missing a prerequisite."""

    def __init__(self, reason_code: str, *, blocked: bool = True) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.blocked = blocked


@dataclass(frozen=True)
class RenderExecutionView:
    row: RenderExecution
    live_freshness: str
    effective: bool


@dataclass(frozen=True)
class RenderPrerequisites:
    candidate: ClipCandidate
    contract: RenderContract
    plan: VisualCompositionPlan


@dataclass(frozen=True)
class SourceStreamFacts:
    """Resolved source stream origin used to build the shared output timeline."""

    video_start_seconds: float
    audio_start_seconds: float
    audio_channels: int
    verified: bool


# Explicit test seam. Production leaves this ``None`` so stream timing evidence
# is never invented; hermetic tests inject deterministic facts rather than
# relying on a silent production failure fallback.
_SOURCE_STREAM_PROBE_OVERRIDE: Callable[[Path, Settings], Any] | None = None


def _resolve_source_streams(
    source_path: Path,
    settings: Settings,
    stream_probe: Any | None,
) -> SourceStreamFacts:
    probe = stream_probe if stream_probe is not None else _SOURCE_STREAM_PROBE_OVERRIDE
    if probe is not None:
        resolved = probe(source_path, settings)
        if isinstance(resolved, SourceStreamFacts):
            return resolved
        channels = resolved if isinstance(resolved, int) and resolved > 0 else 2
        return SourceStreamFacts(0.0, 0.0, channels, True)
    facts = _probe_source_stream_facts(source_path, settings)
    if facts is None:
        # Missing/unusable stream timing evidence must fail closed, never be
        # replaced with an invented zero origin that silently shifts content.
        raise RenderExecutionError(SOURCE_STREAMS_UNSUPPORTED)
    return facts


def _probe_source_stream_facts(source_path: Path, settings: Settings) -> SourceStreamFacts | None:
    """Probe per-stream start offsets and audio channels; ``None`` when unavailable."""

    command = [
        settings.ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,start_time,channels",
        "-of",
        "json",
        str(source_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        import json

        payload = json.loads(completed.stdout or "{}")
    except ValueError:
        return None
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    if not isinstance(streams, list):
        return None
    video_start: float | None = None
    audio_start: float | None = None
    audio_channels = 2
    for stream in streams:
        if not isinstance(stream, Mapping):
            continue
        kind = stream.get("codec_type")
        start = stream.get("start_time")
        start_value = _as_float(start, 0.0) if start is not None else 0.0
        if kind == "video" and video_start is None:
            video_start = start_value
        elif kind == "audio" and audio_start is None:
            audio_start = start_value
            channels = _as_int(stream.get("channels"), 2)
            audio_channels = channels if channels > 0 else 2
    if video_start is None or audio_start is None:
        return None
    if video_start < -0.05 or audio_start < -0.05:
        return None
    # Normalize to the shared source-local origin used across ingestion,
    # refinement, contracts, and FFmpeg: FFmpeg resets each input stream's
    # timestamps to its own first packet, so the authoritative source-local time
    # zero is the earliest stream start. The *relative* A/V offset survives as a
    # genuine leading gap; the common container offset is never treated as
    # missing content.
    origin = min(video_start, audio_start)
    return SourceStreamFacts(video_start - origin, audio_start - origin, audio_channels, True)


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _strict_float(value: object, reason_code: str) -> float:
    """Return a finite float or fail closed; never invent a replacement value."""

    if isinstance(value, bool):
        raise RenderExecutionError(reason_code)
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError as error:
            raise RenderExecutionError(reason_code) from error
    else:
        raise RenderExecutionError(reason_code)
    if not math.isfinite(parsed):
        raise RenderExecutionError(reason_code)
    return parsed


def _strict_str(value: object, reason_code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenderExecutionError(reason_code)
    return value


def _strict_int(value: object, reason_code: str, default: int | None = None) -> int:
    if isinstance(value, bool) or value is None:
        if default is not None:
            return default
        raise RenderExecutionError(reason_code)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        try:
            parsed = int(float(value))
        except ValueError as error:
            raise RenderExecutionError(reason_code) from error
        return parsed
    if default is not None:
        return default
    raise RenderExecutionError(reason_code)


# Read


def get_current_render_execution(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    artifact_purpose: str = CORE_SOURCE_VALIDATION,
    delivery_profile_key: str = "",
) -> RenderExecution | None:
    candidate_uuid = _as_uuid(candidate_id)
    if candidate_uuid is None:
        return None
    query = (
        select(RenderExecution)
        .where(RenderExecution.clip_candidate_id == candidate_uuid)
        .where(RenderExecution.is_current.is_(True))
        .where(RenderExecution.artifact_purpose == RenderArtifactPurpose(artifact_purpose))
    )
    if delivery_profile_key:
        query = query.where(RenderExecution.delivery_profile_key == delivery_profile_key)
    return cast(
        "RenderExecution | None",
        session.scalars(query.order_by(RenderExecution.created_at.desc())).first(),
    )


def get_render_execution(
    session: Session, render_execution_id: uuid.UUID | str
) -> RenderExecution | None:
    execution_uuid = _as_uuid(render_execution_id)
    if execution_uuid is None:
        return None
    return cast("RenderExecution | None", session.get(RenderExecution, execution_uuid))


def read_render_execution(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    settings: Settings | None = None,
) -> RenderExecutionView | None:
    candidate_uuid = _as_uuid(candidate_id)
    if candidate_uuid is None:
        return None
    row = get_current_render_execution(session, candidate_uuid)
    if row is None:
        return None
    return _view_for_row(session, row, settings or get_settings())


def read_render_execution_by_id(
    session: Session,
    render_execution_id: uuid.UUID | str,
    *,
    settings: Settings | None = None,
) -> RenderExecutionView | None:
    row = get_render_execution(session, render_execution_id)
    if row is None:
        return None
    return _view_for_row(session, row, settings or get_settings())


def _view_for_row(
    session: Session, row: RenderExecution, settings: Settings
) -> RenderExecutionView:
    freshness = _live_request_freshness(session, row, settings)
    effective = bool(row.is_current) and freshness == "CURRENT"
    return RenderExecutionView(row=row, live_freshness=freshness, effective=effective)


def _live_request_freshness(session: Session, row: RenderExecution, settings: Settings) -> str:
    if not row.input_fingerprint:
        return "UNVERIFIABLE"
    candidate = session.get(ClipCandidate, row.clip_candidate_id)
    if candidate is None:
        return "STALE"
    try:
        spec = build_render_spec(
            session,
            candidate,
            artifact_purpose=row.artifact_purpose.value,
            delivery_profile_key=row.delivery_profile_key,
            settings=settings,
            storage=StorageService(settings.storage_root),
            stream_probe=None,
        )
        recomputed = request_input_fingerprint(spec, settings.stage52_config())
    except Exception:
        return "UNVERIFIABLE"
    return "CURRENT" if recomputed == row.input_fingerprint else "STALE"


# Prerequisites


def validate_candidate_for_render(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    settings: Settings | None = None,
) -> RenderPrerequisites:
    resolved = settings or get_settings()
    if not resolved.render_execution_enabled:
        raise RenderExecutionError("RENDER_DISABLED")
    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        raise RenderExecutionError("CANDIDATE_NOT_CURRENT")
    if not candidate.is_current:
        raise RenderExecutionError("CANDIDATE_NOT_CURRENT")
    if candidate.disposition not in _VALID_DISPOSITIONS:
        raise RenderExecutionError("CANDIDATE_NOT_RETAINED")
    contract = session.scalars(
        select(RenderContract)
        .where(RenderContract.clip_candidate_id == candidate.id)
        .where(RenderContract.is_current.is_(True))
        .order_by(RenderContract.created_at.desc())
    ).first()
    if contract is None:
        raise RenderExecutionError("CONTRACT_NOT_FOUND")
    view = read_render_contract(session, candidate.id, settings=resolved)
    if view is None or not view.effective:
        raise RenderExecutionError("CONTRACT_NOT_CURRENT")
    if not contract.contract_ready or contract.status.value not in EXECUTABLE_STATUSES:
        raise RenderExecutionError("CONTRACT_NOT_EXECUTABLE")
    plan = session.scalars(
        select(VisualCompositionPlan)
        .where(VisualCompositionPlan.clip_candidate_id == candidate.id)
        .where(VisualCompositionPlan.is_current.is_(True))
        .order_by(VisualCompositionPlan.created_at.desc())
    ).first()
    if plan is None:
        raise RenderExecutionError("VISUAL_PLAN_NOT_FOUND")
    plan_view = read_visual_composition(
        session,
        candidate.id,
        display_probe=FFprobeDisplayProbe(binary=resolved.ffprobe_binary),
        config=resolved.stage51_config(),
    )
    if plan_view is None:
        raise RenderExecutionError("VISUAL_PLAN_NOT_FOUND")
    if not plan_view.effective:
        raise RenderExecutionError("VISUAL_PLAN_NOT_CURRENT")
    if not plan.plan_ready or plan.status.value != "READY_FOR_VISUAL_EXECUTION":
        raise RenderExecutionError("VISUAL_PLAN_NOT_READY")
    if plan.contract_input_fingerprint != contract.input_fingerprint:
        raise RenderExecutionError("VISUAL_PLAN_FINGERPRINT_MISMATCH")
    if plan.contract_output_fingerprint != contract.output_fingerprint:
        raise RenderExecutionError("VISUAL_PLAN_FINGERPRINT_MISMATCH")
    return RenderPrerequisites(candidate=candidate, contract=contract, plan=plan)


# Spec building


def build_render_spec(
    session: Session,
    candidate: ClipCandidate,
    *,
    artifact_purpose: str,
    delivery_profile_key: str,
    settings: Settings,
    storage: StorageService,
    stream_probe: Any | None = None,
) -> RenderSpec:
    if artifact_purpose not in SUPPORTED_ARTIFACT_PURPOSES:
        raise RenderExecutionError("UNSUPPORTED_ARTIFACT_PURPOSE")
    profile = delivery_profile_for(delivery_profile_key)
    if profile is None:
        raise RenderExecutionError("UNSUPPORTED_DELIVERY_PROFILE")
    config = settings.stage52_config()

    prerequisites = validate_candidate_for_render(session, candidate.id, settings=settings)
    contract = prerequisites.contract
    plan = prerequisites.plan
    contract_payload = dict(contract.contract_payload or {})
    plan_payload = dict(plan.plan_payload or {})

    output_profile = _mapping(contract_payload.get("output_profile"))
    geometry = _mapping(plan_payload.get("geometry"))
    source_media = _mapping(contract_payload.get("source_media"))
    identity = _mapping(source_media.get("identity"))
    probe = _mapping(source_media.get("probe"))

    source_relative = str(
        source_media.get("managed_relative_path") or identity.get("relative_path") or ""
    )
    if not source_relative:
        raise RenderExecutionError("SOURCE_MEDIA_MISSING")
    source_path = (storage.storage_root / source_relative).resolve()
    try:
        source_path.relative_to(storage.storage_root)
    except ValueError as error:
        raise RenderExecutionError("SOURCE_MEDIA_UNMANAGED") from error
    if not source_path.is_file():
        raise RenderExecutionError("SOURCE_MEDIA_MISSING")
    stat = source_path.stat()
    if stat.st_size <= 0:
        raise RenderExecutionError("SOURCE_MEDIA_ZERO_BYTES")
    if stat.st_size != _as_int(identity.get("size_bytes"), stat.st_size) or (
        identity.get("mtime_ns") is not None
        and stat.st_mtime_ns != _as_int(identity.get("mtime_ns"), stat.st_mtime_ns)
    ):
        raise RenderExecutionError("SOURCE_MEDIA_CHANGED")

    width = _as_int(output_profile.get("width"), 1080)
    height = _as_int(output_profile.get("height"), 1920)
    target_frame_rate = _as_float(output_profile.get("target_frame_rate"), 0.0)
    source_frame_rate = _as_float(
        output_profile.get("source_frame_rate"), _as_float(probe.get("frames_per_second"), 30.0)
    )
    if target_frame_rate <= 0:
        target_frame_rate = source_frame_rate if source_frame_rate > 0 else 30.0

    source_streams = _resolve_source_streams(source_path, settings, stream_probe)

    ass_meta = _mapping(plan_payload.get("ass"))
    ass_relative = str(ass_meta.get("asset_path") or "")
    if not ass_relative:
        raise RenderExecutionError("ASS_MISSING")
    ass_path = (storage.storage_root / ass_relative).resolve()
    try:
        ass_path.relative_to(storage.storage_root)
    except ValueError as error:
        raise RenderExecutionError("ASS_MISSING") from error
    if not ass_path.is_file():
        raise RenderExecutionError("ASS_MISSING")
    ass_data = ass_path.read_bytes()
    facts = ass_file_facts(ass_data)
    asset = AssAsset(
        relative_path=ass_relative,
        sha256=str(ass_meta.get("sha256") or ""),
        event_count=_as_int(ass_meta.get("event_count"), 0),
        line_count=_as_int(ass_meta.get("line_count"), 0),
        policy_version=str(ass_meta.get("policy_version") or ""),
    )

    caption_events = _caption_events(plan_payload)
    try:
        validate_ass_asset(asset, facts, event_count=len(caption_events))
    except RenderValidationError as error:
        raise RenderExecutionError(error.reason_code) from error

    occurrences, omitted = _occurrences(contract_payload, plan_payload)
    source_duration = _as_float(probe.get("duration_seconds"), 0.0)
    if source_duration > config.max_source_duration_seconds + 1e-6:
        raise RenderExecutionError("SOURCE_DURATION_EXCEEDS_LIMIT")
    output_duration = sum(max(0.0, occ.source_end - occ.source_start) for occ in occurrences)
    if output_duration > config.max_output_duration_seconds + 1e-6:
        raise RenderExecutionError("OUTPUT_DURATION_EXCEEDS_LIMIT")
    spec = RenderSpec(
        render_execution_id="",  # filled by the caller (job-scoped identity)
        candidate_id=str(candidate.id),
        source_id=str(candidate.source_video_id),
        render_contract_id=str(contract.id),
        visual_plan_id=str(plan.id),
        artifact_purpose=artifact_purpose,
        source_media_relative_path=source_relative,
        source_content_hash=str(
            source_media.get("content_hash") or identity.get("content_hash") or ""
        ),
        source_size_bytes=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        source_duration=source_duration,
        source_frame_rate=_fraction(source_frame_rate, 30.0),
        display_width=_as_int(geometry.get("display_width"), width),
        display_height=_as_int(geometry.get("display_height"), height),
        encoded_width=_as_int(geometry.get("encoded_width"), width),
        encoded_height=_as_int(geometry.get("encoded_height"), height),
        rotation_degrees=_as_int(geometry.get("rotation_degrees"), 0),
        pixel_aspect_ratio=_as_float(geometry.get("pixel_aspect_ratio"), 1.0),
        output_width=width,
        output_height=height,
        output_frame_rate=_fraction(target_frame_rate, 30.0),
        output_profile=dict(output_profile),
        delivery_profile_key=delivery_profile_key,
        plan_input_fingerprint=plan.input_fingerprint,
        plan_output_fingerprint=plan.output_fingerprint,
        ass=asset,
        caption_events=caption_events,
        occurrences=occurrences,
        omitted=omitted,
        audio_channels=source_streams.audio_channels,
        source_video_start_seconds=source_streams.video_start_seconds,
        source_audio_start_seconds=source_streams.audio_start_seconds,
    )
    try:
        validate_spec(spec)
    except RenderValidationError as error:
        raise RenderExecutionError(error.reason_code) from error
    return spec


def with_execution_id(spec: RenderSpec, render_execution_id: str) -> RenderSpec:
    from dataclasses import replace

    return replace(spec, render_execution_id=render_execution_id)


def _fraction(value: float, fallback: float) -> Fraction:
    resolved = value if value > 0 else fallback
    return Fraction(resolved).limit_denominator(1001)


def _caption_events(plan_payload: Mapping[str, object]) -> tuple[CaptionEventSpec, ...]:
    captions = _mapping(plan_payload.get("captions"))
    events: list[CaptionEventSpec] = []
    for index, raw in enumerate(_sequence(captions.get("events"))):
        event = _mapping(raw)
        events.append(
            CaptionEventSpec(
                event_id=_strict_str(event.get("event_id"), "ASS_METADATA_MISMATCH")
                if event.get("event_id")
                else f"event-{index}",
                block_index=_strict_int(
                    event.get("block_index"), "ASS_METADATA_MISMATCH", default=0
                ),
                start=_strict_float(event.get("start"), "ASS_METADATA_MISMATCH"),
                end=_strict_float(event.get("end"), "ASS_METADATA_MISMATCH"),
            )
        )
    return tuple(events)


def _occurrences(
    contract_payload: Mapping[str, object],
    plan_payload: Mapping[str, object],
) -> tuple[tuple[TimelineOccurrence, ...], tuple[OmittedRequirement, ...]]:
    scenes_by_block: dict[int, list[SceneSpec]] = {}
    for index, raw in enumerate(_sequence(plan_payload.get("scenes"))):
        if not isinstance(raw, Mapping):
            raise RenderExecutionError("SCENE_BOUNDS_INVALID")
        scene = raw
        block_index = _strict_int(scene.get("block_index"), "SCENE_BLOCK_MISMATCH", default=index)
        source_start = _strict_float(scene.get("source_start"), "SCENE_BOUNDS_INVALID")
        source_end = _strict_float(scene.get("source_end"), "SCENE_BOUNDS_INVALID")
        mode = _strict_str(scene.get("framing_mode"), "UNSUPPORTED_FRAMING_MODE")
        interpolation = _strict_str(scene.get("interpolation_policy"), "UNSUPPORTED_INTERPOLATION")
        keyframes = tuple(
            _keyframe(frame, mode)
            for frame in _sequence(scene.get("crop_keyframes"))
            if isinstance(frame, Mapping)
        )
        if len(keyframes) != len(_sequence(scene.get("crop_keyframes"))):
            raise RenderExecutionError("KEYFRAME_GEOMETRY_INVALID")
        if not keyframes and mode in _CROP_MODES:
            raise RenderExecutionError("CONTRADICTORY_FRAMING_EVIDENCE")
        scenes_by_block.setdefault(block_index, []).append(
            SceneSpec(
                scene_index=_strict_int(
                    scene.get("scene_index"), "SCENE_BOUNDS_INVALID", default=index
                ),
                block_index=block_index,
                source_start=source_start,
                source_end=source_end,
                framing_mode=mode,
                interpolation_policy=interpolation,
                crop_keyframes=keyframes,
            )
        )
    for block_scene_list in scenes_by_block.values():
        block_scene_list.sort(key=lambda item: item.source_start)

    occurrences: list[TimelineOccurrence] = []
    omitted: list[OmittedRequirement] = []
    cursor = 0.0
    for raw in _sequence(contract_payload.get("blocks")):
        if not isinstance(raw, Mapping):
            raise RenderExecutionError("OCCURRENCE_BOUNDS_INVALID")
        block = raw
        block_index = _strict_int(block.get("block_index"), "OCCURRENCE_BOUNDS_INVALID", default=0)
        block_type = str(block.get("block_type") or "")
        slot_kind = str(block.get("slot_kind") or "")
        if block_type == "SOURCE_EXCERPT":
            binding = _mapping(block.get("source_binding"))
            start = binding.get("final_clip_start")
            end = binding.get("final_clip_end")
            if not bool(binding.get("rebind_valid", True)) or start is None or end is None:
                raise RenderExecutionError("OCCURRENCE_BOUNDS_INVALID")
            start_f = _strict_float(start, "OCCURRENCE_BOUNDS_INVALID")
            end_f = _strict_float(end, "OCCURRENCE_BOUNDS_INVALID")
            scenes = tuple(scenes_by_block.get(block_index, ()))
            occurrences.append(
                TimelineOccurrence(
                    occurrence_id=f"block-{block_index}",
                    block_index=block_index,
                    source_start=start_f,
                    source_end=end_f,
                    output_start=cursor,
                    output_end=cursor + max(0.0, end_f - start_f),
                    source_role=str(binding.get("source_role") or "") or None,
                    is_hero=bool(binding.get("is_hero")),
                    scenes=scenes,
                )
            )
            cursor += max(0.0, end_f - start_f)
        else:
            omitted.append(
                OmittedRequirement(
                    block_index=block_index,
                    block_type=block_type,
                    slot_kind=slot_kind,
                    reason_code=_omission_reason(block_type, slot_kind),
                )
            )
    materialization = _mapping(contract_payload.get("materialization"))
    for raw in _sequence(materialization.get("slots")):
        slot = _mapping(raw)
        slot_block = slot.get("block_index")
        omitted.append(
            OmittedRequirement(
                block_index=slot_block
                if isinstance(slot_block, int) and not isinstance(slot_block, bool)
                else None,
                block_type=str(slot.get("block_type") or ""),
                slot_kind=str(slot.get("slot_kind") or ""),
                reason_code=str(
                    slot.get("reason_code") or "AUTHORED_SLOT_MATERIALIZATION_REQUIRED"
                ),
            )
        )
    from app.render.execution.timeline import TimelineError, validate_occurrence_order

    try:
        validate_occurrence_order(tuple(occurrences))
    except TimelineError as error:
        raise RenderExecutionError(error.reason_code) from error
    return tuple(occurrences), tuple(omitted)


def _keyframe(frame: Mapping[str, object], scene_mode: str) -> CropKeyframeSpec:
    keyframe_mode = _strict_str(frame.get("mode"), "CONTRADICTORY_FRAMING_EVIDENCE")
    if keyframe_mode not in _ALL_MODES:
        raise RenderExecutionError("CONTRADICTORY_FRAMING_EVIDENCE")
    both_crop = scene_mode in _CROP_MODES and keyframe_mode in _CROP_MODES
    if keyframe_mode != scene_mode and not both_crop:
        raise RenderExecutionError("CONTRADICTORY_FRAMING_EVIDENCE")
    return CropKeyframeSpec(
        t=_strict_float(frame.get("t"), "KEYFRAME_GEOMETRY_INVALID"),
        cx=_strict_float(frame.get("cx"), "KEYFRAME_GEOMETRY_INVALID"),
        cy=_strict_float(frame.get("cy"), "KEYFRAME_GEOMETRY_INVALID"),
        height_fraction=_strict_float(frame.get("height_fraction"), "KEYFRAME_GEOMETRY_INVALID"),
        mode=keyframe_mode,
        confidence=_as_float(frame.get("confidence"), 0.0),
    )


def _omission_reason(block_type: str, slot_kind: str) -> str:
    if block_type == "ORIGINAL_VALUE":
        return "AUTHORED_NARRATION_MATERIALIZATION_REQUIRED"
    if block_type == "TEXTUAL_ANNOTATION":
        return "AUTHORED_TEXT_MATERIALIZATION_REQUIRED"
    if block_type == "FACT_VERIFICATION_PLACEHOLDER":
        return "VERIFICATION_EVIDENCE_MATERIALIZATION_REQUIRED"
    if block_type == "TRANSITION":
        return "TRANSITION_MATERIALIZATION_REQUIRED"
    return "AUTHORED_SLOT_MATERIALIZATION_REQUIRED"


# Runtime + fingerprints


# Cache keyed by a cheap dependency *signature* (binary/font/library path + stat),
# never by path alone, so a changed dependency under an unchanged path cannot be
# concealed by a stale cache entry.
_RUNTIME_VERSION_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


def _stat_signature(path: str) -> tuple[int, int]:
    if not path:
        return (0, 0)
    try:
        stat = Path(path).stat()
    except OSError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


def _resolved_binary(binary: str) -> str:
    return shutil.which(binary) or binary


def _file_sha256(path: str) -> str:
    if not path:
        return ""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _font_match_path(family: str) -> str:
    """Resolve the concrete font file fontconfig selects for ``family``."""

    try:
        completed = subprocess.run(
            ["fc-match", "-f", "%{file}", family],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "").strip()


def _resolve_effective_fonts(family: str) -> tuple[str, ...]:
    """Concrete files for the primary family plus its Latin fallback.

    The canonical ASS places mixed Arabic/Latin text through one family, and
    libass resolves missing glyphs through fontconfig fallback; both the primary
    and the generic fallback file participate in the runtime content identity.
    """

    paths: list[str] = []
    for name in (family, "sans-serif"):
        path = _font_match_path(name)
        if path and path not in paths and Path(path).is_file():
            paths.append(path)
    return tuple(paths)


def _font_sha256(font_paths: Sequence[str]) -> str:
    """Composite content digest over the effective font files."""

    digest = hashlib.sha256()
    hashed_any = False
    for path in font_paths:
        data_digest = _file_sha256(path)
        if not data_digest:
            continue
        hashed_any = True
        digest.update(path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(bytes.fromhex(data_digest))
    return digest.hexdigest() if hashed_any else ""


def _shared_library_path(binary: str, name: str) -> str:
    """Resolve the concrete loaded shared object for ``name`` (via ldd, then ctypes)."""

    resolved_binary = _resolved_binary(binary)
    try:
        completed = subprocess.run(
            ["ldd", resolved_binary], check=False, capture_output=True, text=True, timeout=30
        )
        output = completed.stdout or ""
    except (OSError, subprocess.SubprocessError):
        output = ""
    for line in output.splitlines():
        left, separator, right = line.partition("=>")
        if separator and name in left and right.strip():
            candidate = right.strip().split(" ")[0]
            if candidate.startswith("/") and Path(candidate).is_file():
                return str(Path(candidate).resolve())
    import ctypes.util

    found = ctypes.util.find_library(name)
    if found:
        try:
            resolved = Path(found)
        except (OSError, ValueError):
            resolved = None
        if resolved is not None and resolved.is_file():
            return str(resolved.resolve())
    return ""


def _libass_identity(ffmpeg_binary: str) -> tuple[str, str]:
    """Truthful identity of the loaded libass: real filename + content digest."""

    path = _shared_library_path(ffmpeg_binary, "ass")
    if not path:
        return ("", "")
    return (Path(path).name, _file_sha256(path))


def _runtime_signature(
    ffmpeg_binary: str, ffprobe_binary: str, font_family: str
) -> tuple[object, ...]:
    ffmpeg_path = _resolved_binary(ffmpeg_binary)
    ffprobe_path = _resolved_binary(ffprobe_binary)
    fonts = _resolve_effective_fonts(font_family)
    libass_path = _shared_library_path(ffmpeg_binary, "ass")
    return (
        ffmpeg_path,
        _stat_signature(ffmpeg_path),
        ffprobe_path,
        _stat_signature(ffprobe_path),
        font_family,
        tuple((path, _stat_signature(path)) for path in fonts),
        libass_path,
        _stat_signature(libass_path),
    )


def _runtime_version_fields(
    ffmpeg_binary: str, ffprobe_binary: str, font_family: str
) -> dict[str, object]:
    key = _runtime_signature(ffmpeg_binary, ffprobe_binary, font_family)
    cached = _RUNTIME_VERSION_CACHE.get(key)
    if cached is not None:
        return cached
    ffmpeg_text = _binary_output(ffmpeg_binary, "-version")
    ffprobe_text = _binary_output(ffprobe_binary, "-version")
    buildconf_text = _binary_output(ffmpeg_binary, "-buildconf")
    ffmpeg_version = (ffmpeg_text.splitlines()[0] if ffmpeg_text else "") or ffmpeg_binary
    ffprobe_version = (ffprobe_text.splitlines()[0] if ffprobe_text else "") or ffprobe_binary
    configuration = _configuration_line(buildconf_text) or _configuration_line(ffmpeg_text)
    fonts = _resolve_effective_fonts(font_family)
    libass_version, libass_sha256 = _libass_identity(ffmpeg_binary)
    fields: dict[str, object] = {
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_version": ffprobe_version,
        "libavformat_version": _lib_version(ffmpeg_text, "libavformat") or ffmpeg_version,
        "libavcodec_version": _lib_version(ffmpeg_text, "libavcodec") or ffmpeg_version,
        "libass_version": libass_version or "unavailable",
        "libass_sha256": libass_sha256,
        "font_match": fonts[0] if fonts else "",
        "font_sha256": _font_sha256(fonts),
        "effective_fonts": list(fonts),
        "build_config_sha256": (
            hashlib.sha256(configuration.encode("utf-8")).hexdigest() if configuration else ""
        ),
    }
    _RUNTIME_VERSION_CACHE[key] = fields
    return fields


def resolve_runtime_identity(
    settings: Settings,
    *,
    source_absolute_path: str = "",
    attempt_directory: str = "",
) -> RuntimeIdentity:
    ffmpeg_binary = settings.ffmpeg_binary
    ffprobe_binary = settings.ffprobe_binary
    font_family = settings.visual_caption_font_family
    config = settings.stage52_config()
    fields = _runtime_version_fields(ffmpeg_binary, ffprobe_binary, font_family)
    identity = RuntimeIdentity(
        ffmpeg_version=cast(str, fields["ffmpeg_version"]),
        ffprobe_version=cast(str, fields["ffprobe_version"]),
        libavformat_version=cast(str, fields["libavformat_version"]),
        libavcodec_version=cast(str, fields["libavcodec_version"]),
        libass_version=cast(str, fields["libass_version"]),
        libass_sha256=cast(str, fields["libass_sha256"]),
        font_family=font_family,
        font_match=cast(str, fields["font_match"]),
        font_sha256=cast(str, fields["font_sha256"]),
        build_config_sha256=cast(str, fields["build_config_sha256"]),
        compiler_version=EXECUTION_POLICY_VERSION,
        policy_version=EXECUTION_POLICY_VERSION,
        ffmpeg_binary=ffmpeg_binary,
        ffprobe_binary=ffprobe_binary,
        encoder_threads=config.encoder_threads,
        filter_threads=config.filter_threads,
        filter_complex_threads=config.filter_complex_threads,
        source_absolute_path=source_absolute_path,
        attempt_directory=attempt_directory,
    )
    return identity


def _binary_output(binary: str, flag: str) -> str:
    try:
        completed = subprocess.run(
            [binary, flag], check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout or ""


def _configuration_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith("configuration:"):
            return line.strip()
        if line.strip().startswith("--"):
            return line.strip()
    return ""


def _lib_version(text: str, name: str) -> str:
    """Full, whitespace-collapsed version (never a truncated fragment)."""

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(name):
            remainder = stripped[len(name) :].strip()
            version = "".join(remainder.split("/", 1)[0].split())
            if version:
                return f"{name} {version}"
    return ""


def request_input_fingerprint(spec: RenderSpec, config: Stage52Config) -> str:
    profile = delivery_profile_for(spec.delivery_profile_key)
    payload = {
        "spec": spec.as_dict(),
        "delivery_profile": profile.as_dict() if profile is not None else {},
        "policy": stage52_config_payload(config),
        "qc_policy": qc_payload(config),
    }
    return render_request_fingerprint(payload)


# Persistence


def request_payload(spec: RenderSpec) -> dict[str, object]:
    return {
        "spec": spec.as_dict(),
        "timeline": build_timeline(spec).as_dict(),
        "omitted": [item.as_dict() for item in spec.omitted],
    }


def persist_envelope(
    session: Session,
    candidate: ClipCandidate,
    *,
    spec: RenderSpec,
    input_fingerprint: str,
    delivery_profile_key: str,
) -> RenderExecution:
    existing = session.scalars(
        select(RenderExecution)
        .where(RenderExecution.clip_candidate_id == candidate.id)
        .where(RenderExecution.input_fingerprint == input_fingerprint)
    ).first()
    if existing is not None:
        return cast(RenderExecution, existing)
    row = RenderExecution(
        id=uuid.uuid4(),
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        render_contract_id=_as_uuid(spec.render_contract_id),
        visual_composition_plan_id=_as_uuid(spec.visual_plan_id),
        artifact_purpose=RenderArtifactPurpose(spec.artifact_purpose),
        lifecycle=RenderExecutionLifecycle.QUEUED,
        is_current=True,
        input_fingerprint=input_fingerprint,
        delivery_profile_key=delivery_profile_key,
        request_payload=request_payload(spec),
        omitted_requirements=[item.as_dict() for item in spec.omitted],
        policy_version=EXECUTION_POLICY_VERSION,
        schema_version=EXECUTION_SCHEMA_VERSION,
        fingerprint_version=EXECUTION_FINGERPRINT_VERSION,
    )
    session.add(row)
    session.flush()
    return row


def _as_uuid(value: object) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, str) and value:
        try:
            return uuid.UUID(value)
        except ValueError:
            return None
    return None


def execution_artifact_reference(
    storage: StorageService, artifacts: RenderArtifacts
) -> dict[str, object]:
    relative = storage.storage_relative(artifacts.output_path)
    return {
        "relative_path": relative,
        "sha256": artifacts.sha256,
        "size_bytes": artifacts.size_bytes,
        "content_type": "video/mp4",
    }


def runtime_fingerprint(identity: RuntimeIdentity) -> str:
    return render_runtime_fingerprint(identity.as_dict())


def compiled_fingerprint(compiled_fingerprint_value: str) -> str:
    return render_compiled_fingerprint({"compiled": compiled_fingerprint_value})


def output_fingerprint(
    row: RenderExecution, artifacts: RenderArtifacts, qc: TechnicalQCResult
) -> str:
    return render_output_fingerprint(
        {
            "artifact": artifacts.as_dict(),
            "qc": qc.as_dict(),
            "input_fingerprint": row.input_fingerprint,
        }
    )


def qc_fingerprint(config: Stage52Config) -> str:
    return render_qc_fingerprint(qc_payload(config))


def _job_owns(job_id: uuid.UUID, claim_version: int, statuses: Sequence[JobStatus]) -> Any:
    """Correlated EXISTS predicate: the executing job still owns its claim."""

    return exists(
        select(ProcessingJob.id).where(
            ProcessingJob.id == job_id,
            ProcessingJob.claim_version == claim_version,
            ProcessingJob.status.in_(tuple(statuses)),
        )
    )


def _execution_fence(row_id: uuid.UUID, job_id: uuid.UUID, claim_version: int) -> tuple[Any, ...]:
    return (
        RenderExecution.id == row_id,
        RenderExecution.active_job_id == job_id,
        _job_owns(job_id, claim_version, (JobStatus.RUNNING,)),
    )


def _job_status_is(job_id: uuid.UUID, statuses: Sequence[JobStatus]) -> Any:
    return exists(
        select(ProcessingJob.id).where(
            ProcessingJob.id == job_id, ProcessingJob.status.in_(tuple(statuses))
        )
    )


def begin_attempt(
    session: Session, row_id: uuid.UUID, *, job_id: uuid.UUID, claim_version: int
) -> bool:
    """Owned atomic transition into a fresh attempt, resetting cache/attempt state.

    Resets cache eligibility, QC verdict, and the previously published artifact
    pointer in one fenced UPDATE so an earlier success can never be exposed as
    the authoritative result of the new attempt, and the
    ``ck_render_executions_cache_consistency`` check always holds.
    """

    result = session.execute(
        update(RenderExecution)
        .where(*_execution_fence(row_id, job_id, claim_version))
        .values(
            lifecycle=RenderExecutionLifecycle.RENDERING,
            cache_eligible=False,
            qc_status=None,
            artifact_reference={},
            execution_manifest={},
            qc_result={},
            output_fingerprint="",
            reason_codes=[],
            error_code=None,
            error_message=None,
        )
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def set_lifecycle_fenced(
    session: Session,
    row_id: uuid.UUID,
    lifecycle: RenderExecutionLifecycle,
    *,
    job_id: uuid.UUID,
    claim_version: int,
) -> bool:
    result = session.execute(
        update(RenderExecution)
        .where(*_execution_fence(row_id, job_id, claim_version))
        .values(lifecycle=lifecycle)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def release_active_job(session: Session, row_id: uuid.UUID, *, job_id: uuid.UUID) -> None:
    session.execute(
        update(RenderExecution)
        .where(RenderExecution.id == row_id, RenderExecution.active_job_id == job_id)
        .values(active_job_id=None)
        .execution_options(synchronize_session=False)
    )


#: Job statuses a running cancellation may observe. An API/session cancel flips
#: the job to CANCELLED before the worker notices, so a cancellation fence that
#: required RUNNING would match zero rows and strand the owned execution.
_CANCELLATION_JOB_STATUSES = (JobStatus.RUNNING, JobStatus.CANCELLED)


def finalize_cancellation(
    session: Session, row_id: uuid.UUID, *, job_id: uuid.UUID, claim_version: int
) -> bool:
    """Atomically finalize an owned running execution as CANCELLED.

    Requires the exact execution, the authoritative ``active_job_id``, the exact
    executing job, and the exact ``claim_version`` -- but accepts a job already
    flipped to CANCELLED as well as one still RUNNING. A superseded worker (newer
    claim or reassigned active job) therefore matches zero rows and can never
    cancel or release a newer run's result. Clears cache eligibility and the
    published-artifact pointer and releases ownership coherently.
    """

    result = session.execute(
        update(RenderExecution)
        .where(
            RenderExecution.id == row_id,
            RenderExecution.active_job_id == job_id,
            _job_owns(job_id, claim_version, _CANCELLATION_JOB_STATUSES),
        )
        .values(
            lifecycle=RenderExecutionLifecycle.CANCELLED,
            cache_eligible=False,
            qc_status=None,
            artifact_reference={},
            execution_manifest={},
            qc_result={},
            output_fingerprint="",
            reason_codes=["RENDER_CANCELLED"],
            error_code="RENDER_CANCELLED",
            error_message=None,
            active_job_id=None,
        )
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def cancel_execution(session: Session, row_id: uuid.UUID, *, job_id: uuid.UUID) -> bool:
    """Fenced cancellation: only the authoritative active job cancels the row.

    Deliberately keys on ``active_job_id`` (not job status) so a queued job that
    was cancelled through the API is still able to finalize its own row, while a
    superseded job whose row is owned by a newer attempt cannot.
    """

    result = session.execute(
        update(RenderExecution)
        .where(
            RenderExecution.id == row_id,
            RenderExecution.active_job_id == job_id,
            _job_status_is(job_id, (JobStatus.CANCELLED,)),
        )
        .values(
            lifecycle=RenderExecutionLifecycle.CANCELLED,
            cache_eligible=False,
            qc_status=None,
            artifact_reference={},
            execution_manifest={},
            qc_result={},
            output_fingerprint="",
            reason_codes=[],
            error_code=None,
            active_job_id=None,
        )
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def finalize_success(
    session: Session,
    row: RenderExecution,
    *,
    artifacts: RenderArtifacts,
    qc: TechnicalQCResult,
    compiled_fingerprint_value: str,
    runtime_fp: str,
    storage: StorageService,
    config: Stage52Config,
    job_id: uuid.UUID | None = None,
    claim_version: int = -1,
) -> bool:
    """Publish a successful artifact under an optional ownership fence."""

    reference = execution_artifact_reference(storage, artifacts)
    qc_fp = qc_fingerprint(config)
    input_fp = row.input_fingerprint
    output_fp = render_output_fingerprint(
        {
            "artifact": artifacts.as_dict(),
            "qc": qc.as_dict(),
            "input_fingerprint": input_fp,
            "qc_fingerprint": qc_fp,
        }
    )
    values: dict[str, Any] = dict(
        lifecycle=RenderExecutionLifecycle.COMPLETE,
        qc_status=RenderQCStatus(qc.status),
        execution_manifest=dict(artifacts.manifest),
        qc_result=qc.as_dict(),
        artifact_reference=reference,
        compiler_fingerprint=compiled_fingerprint_value,
        runtime_fingerprint=runtime_fp,
        qc_fingerprint=qc_fp,
        output_fingerprint=output_fp,
        cache_eligible=qc.status in {"PASS", "WARN"},
        reason_codes=list(qc.reason_codes),
        error_code=None,
        error_message=None,
        active_job_id=None,
    )
    if job_id is None:
        for key, value in values.items():
            setattr(row, key, value)
        if row.output_fingerprint != output_fp:
            row.output_fingerprint = output_fp
        session.flush()
        return True
    result = session.execute(
        update(RenderExecution)
        .where(*_execution_fence(row.id, job_id, claim_version))
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def mark_blocked(
    session: Session,
    row: RenderExecution,
    reason_code: str,
    *,
    job_id: uuid.UUID | None = None,
    claim_version: int = -1,
) -> bool:
    return _mark_terminal(
        session,
        row,
        RenderExecutionLifecycle.BLOCKED,
        reason_code,
        job_id=job_id,
        claim_version=claim_version,
    )


def mark_failed(
    session: Session,
    row: RenderExecution,
    reason_code: str,
    *,
    qc: TechnicalQCResult | None = None,
    job_id: uuid.UUID | None = None,
    claim_version: int = -1,
) -> bool:
    return _mark_terminal(
        session,
        row,
        RenderExecutionLifecycle.FAILED,
        reason_code,
        qc=qc,
        job_id=job_id,
        claim_version=claim_version,
    )


def mark_cancelled(
    session: Session,
    row: RenderExecution,
    *,
    job_id: uuid.UUID | None = None,
    claim_version: int = -1,
) -> bool:
    return _mark_terminal(
        session,
        row,
        RenderExecutionLifecycle.CANCELLED,
        "RENDER_CANCELLED",
        job_id=job_id,
        claim_version=claim_version,
    )


def _mark_terminal(
    session: Session,
    row: RenderExecution,
    lifecycle: RenderExecutionLifecycle,
    reason_code: str,
    *,
    qc: TechnicalQCResult | None = None,
    job_id: uuid.UUID | None = None,
    claim_version: int = -1,
) -> bool:
    values: dict[str, Any] = dict(
        lifecycle=lifecycle,
        cache_eligible=False,
        qc_status=(RenderQCStatus(qc.status) if qc is not None and qc.status == "FAIL" else None),
        artifact_reference={},
        execution_manifest=dict(qc.measured) if qc is not None else {},
        qc_result=qc.as_dict() if qc is not None else {},
        output_fingerprint="",
        reason_codes=[reason_code],
        error_code=reason_code,
        active_job_id=None,
    )
    if job_id is None:
        for key, value in values.items():
            setattr(row, key, value)
        session.flush()
        return True
    result = session.execute(
        update(RenderExecution)
        .where(*_execution_fence(row.id, job_id, claim_version))
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount) == 1


def cache_hit(
    session: Session,
    candidate_id: uuid.UUID,
    settings: Settings,
    *,
    expected_input_fingerprint: str | None = None,
    runtime_identity: RuntimeIdentity | None = None,
) -> RenderExecution | None:
    """Return a current, valid, cache-eligible COMPLETE execution when one exists.

    Validates artifact existence/size/digest, lifecycle/QC, current upstream
    dependencies, rendering runtime identity, and QC identity. Any mismatch
    returns ``None`` so the request re-enters execution rather than reusing a
    stale verdict or a missing/corrupt artifact.
    """

    config = settings.stage52_config()
    row = get_current_render_execution(session, candidate_id)
    if row is None:
        return None
    if (
        expected_input_fingerprint is not None
        and row.input_fingerprint != expected_input_fingerprint
    ):
        return None
    if row.lifecycle is not RenderExecutionLifecycle.COMPLETE:
        return None
    if not row.cache_eligible:
        return None
    if row.qc_status not in {RenderQCStatus.PASS, RenderQCStatus.WARN}:
        return None
    if row.qc_fingerprint and row.qc_fingerprint != qc_fingerprint(config):
        return None
    reference = _mapping(row.artifact_reference)
    relative = reference.get("relative_path")
    if not isinstance(relative, str) or not relative:
        return None
    storage = StorageService(settings.storage_root)
    path = (storage.storage_root / relative).resolve()
    try:
        path.relative_to(storage.storage_root)
    except ValueError:
        return None
    if not path.is_file():
        return None
    size = path.stat().st_size
    if size <= 0:
        return None
    expected_size = reference.get("size_bytes")
    if isinstance(expected_size, int) and expected_size > 0 and size != expected_size:
        return None
    expected_hash = str(reference.get("sha256") or "")
    if not expected_hash:
        return None
    if sha256_file(path).lower() != expected_hash.lower():
        return None

    candidate = session.get(ClipCandidate, candidate_id)
    if candidate is None:
        return None
    try:
        spec = build_render_spec(
            session,
            candidate,
            artifact_purpose=row.artifact_purpose.value,
            delivery_profile_key=row.delivery_profile_key,
            settings=settings,
            storage=storage,
        )
    except Exception:
        return None
    if request_input_fingerprint(spec, config) != row.input_fingerprint:
        return None
    if row.runtime_fingerprint:
        resolved = runtime_identity or resolve_runtime_identity(settings)
        if runtime_fingerprint(resolved) != row.runtime_fingerprint:
            return None
    return row


def now() -> datetime:
    return datetime.now(timezone.utc)


# Re-exports used by the queue/executor for a single import surface.


def validate_spec_or_blocked(spec: RenderSpec) -> None:
    try:
        validate_spec(spec)
    except RenderValidationError as error:
        raise RenderExecutionError(error.reason_code) from error


__all__ = [
    "RenderExecutionError",
    "RenderExecutionView",
    "RenderPrerequisites",
    "build_render_spec",
    "cache_hit",
    "check_render_artifact",
    "begin_attempt",
    "execution_artifact_reference",
    "finalize_cancellation",
    "finalize_success",
    "get_current_render_execution",
    "get_render_execution",
    "mark_blocked",
    "mark_cancelled",
    "mark_failed",
    "now",
    "persist_envelope",
    "read_render_execution",
    "read_render_execution_by_id",
    "release_active_job",
    "request_input_fingerprint",
    "request_payload",
    "resolve_runtime_identity",
    "runtime_fingerprint",
    "set_lifecycle_fenced",
    "validate_candidate_for_render",
    "with_execution_id",
]
