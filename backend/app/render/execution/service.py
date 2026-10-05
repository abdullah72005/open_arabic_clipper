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

import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.composition.geometry import FFprobeDisplayProbe
from app.composition.policy import FramingMode
from app.composition.service import read_visual_composition
from app.core.enums import (
    CandidateDisposition,
    RenderArtifactPurpose,
    RenderExecutionLifecycle,
    RenderQCStatus,
)
from app.core.settings import Settings, get_settings
from app.models import ClipCandidate, VisualCompositionPlan
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

_CROP_MODES = frozenset(mode.value for mode in FramingMode)


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
    return session.scalars(query.order_by(RenderExecution.created_at.desc())).first()


def get_render_execution(
    session: Session, render_execution_id: uuid.UUID | str
) -> RenderExecution | None:
    execution_uuid = _as_uuid(render_execution_id)
    if execution_uuid is None:
        return None
    return session.get(RenderExecution, execution_uuid)


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

    audio_channels = 2
    if stream_probe is not None:
        channels = stream_probe(source_path, settings)
        if isinstance(channels, int) and channels > 0:
            audio_channels = channels

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
        source_duration=_as_float(probe.get("duration_seconds"), 0.0),
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
        audio_channels=audio_channels,
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
                event_id=str(event.get("event_id") or f"event-{index}"),
                block_index=_as_int(event.get("block_index"), 0),
                start=_as_float(event.get("start"), 0.0),
                end=_as_float(event.get("end"), 0.0),
            )
        )
    return tuple(events)


def _occurrences(
    contract_payload: Mapping[str, object],
    plan_payload: Mapping[str, object],
) -> tuple[tuple[TimelineOccurrence, ...], tuple[OmittedRequirement, ...]]:
    scenes_by_block: dict[int, list[SceneSpec]] = {}
    for index, raw in enumerate(_sequence(plan_payload.get("scenes"))):
        scene = _mapping(raw)
        block_index = _as_int(scene.get("block_index"), 0)
        keyframes = tuple(
            CropKeyframeSpec(
                t=_as_float(frame.get("t"), 0.0),
                cx=_as_float(frame.get("cx"), 0.5),
                cy=_as_float(frame.get("cy"), 0.5),
                height_fraction=_as_float(frame.get("height_fraction"), 1.0),
            )
            for frame in _sequence(scene.get("crop_keyframes"))
            if isinstance(frame, Mapping)
        )
        mode = str(scene.get("framing_mode") or FramingMode.CENTER_FALLBACK.value)
        if not keyframes and mode in _CROP_MODES:
            raise RenderExecutionError("CONTRADICTORY_FRAMING_EVIDENCE")
        scenes_by_block.setdefault(block_index, []).append(
            SceneSpec(
                scene_index=_as_int(scene.get("scene_index"), index),
                block_index=block_index,
                source_start=_as_float(scene.get("source_start"), 0.0),
                source_end=_as_float(scene.get("source_end"), 0.0),
                framing_mode=mode,
                interpolation_policy=str(scene.get("interpolation_policy") or "smoothstep-ease"),
                crop_keyframes=keyframes,
            )
        )
    for block_scene_list in scenes_by_block.values():
        block_scene_list.sort(key=lambda item: item.source_start)

    occurrences: list[TimelineOccurrence] = []
    omitted: list[OmittedRequirement] = []
    cursor = 0.0
    for raw in _sequence(contract_payload.get("blocks")):
        block = _mapping(raw)
        block_index = _as_int(block.get("block_index"), 0)
        block_type = str(block.get("block_type") or "")
        slot_kind = str(block.get("slot_kind") or "")
        if block_type == "SOURCE_EXCERPT":
            binding = _mapping(block.get("source_binding"))
            start = binding.get("final_clip_start")
            end = binding.get("final_clip_end")
            if not bool(binding.get("rebind_valid", True)) or start is None or end is None:
                raise RenderExecutionError("OCCURRENCE_BOUNDS_INVALID")
            start_f = _as_float(start)
            end_f = _as_float(end)
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


def resolve_runtime_identity(
    settings: Settings,
    *,
    source_absolute_path: str = "",
    attempt_directory: str = "",
) -> RuntimeIdentity:
    ffmpeg_binary = settings.ffmpeg_binary
    ffprobe_binary = settings.ffprobe_binary
    ffmpeg_version = _first_line([ffmpeg_binary, "-version"])
    ffprobe_version = _first_line([ffprobe_binary, "-version"])
    libass_version = _version_token(ffmpeg_version, "--enable-libass")
    config = settings.stage52_config()
    font_family = settings.visual_caption_font_family
    return RuntimeIdentity(
        ffmpeg_version=ffmpeg_version or ffmpeg_binary,
        ffprobe_version=ffprobe_version or ffprobe_binary,
        libavformat_version=_version_token(ffmpeg_version, "libavformat"),
        libass_version=libass_version,
        font_family=font_family,
        font_match=_font_match(font_family),
        compiler_version=EXECUTION_POLICY_VERSION,
        policy_version=EXECUTION_POLICY_VERSION,
        ffmpeg_binary=ffmpeg_binary,
        ffprobe_binary=ffprobe_binary,
        encoder_threads=config.encoder_threads,
        filter_threads=config.filter_threads,
        source_absolute_path=source_absolute_path,
        attempt_directory=attempt_directory,
    )


def request_input_fingerprint(spec: RenderSpec, config: Stage52Config) -> str:
    profile = delivery_profile_for(spec.delivery_profile_key)
    payload = {
        "spec": spec.as_dict(),
        "delivery_profile": profile.as_dict() if profile is not None else {},
        "policy": stage52_config_payload(config),
    }
    return render_request_fingerprint(payload)


def _first_line(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command), check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "").splitlines()[0] if completed.stdout else ""


def _version_token(text: str, token: str) -> str:
    for part in text.split():
        if token in part:
            return part
    return ""


def _font_match(family: str) -> str:
    try:
        completed = subprocess.run(
            ["fc-match", family], check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or "").strip()


def _probe_streams(source_path: Path, settings: Settings) -> int:
    command = [
        settings.ffprobe_binary,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "default=nw=1:nk=1",
        str(source_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return 2
    try:
        return int((completed.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 2


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
        return existing
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


def cache_hit(
    session: Session, candidate_id: uuid.UUID, settings: Settings
) -> RenderExecution | None:
    """Return a current, valid, cache-eligible COMPLETE execution when one exists."""

    row = get_current_render_execution(session, candidate_id)
    if row is None:
        return None
    if row.lifecycle is not RenderExecutionLifecycle.COMPLETE:
        return None
    if not row.cache_eligible:
        return None
    if row.qc_status not in {RenderQCStatus.PASS, RenderQCStatus.WARN}:
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
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    expected_hash = str(reference.get("sha256") or "")
    if expected_hash and sha256_file(path).lower() != expected_hash.lower():
        return None
    return row


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
) -> None:
    reference = execution_artifact_reference(storage, artifacts)
    row.lifecycle = RenderExecutionLifecycle.COMPLETE
    row.qc_status = RenderQCStatus(qc.status)
    row.execution_manifest = dict(artifacts.manifest)
    row.qc_result = qc.as_dict()
    row.artifact_reference = reference
    row.compiler_fingerprint = compiled_fingerprint_value
    row.runtime_fingerprint = runtime_fp
    row.qc_fingerprint = qc_fingerprint(config)
    row.output_fingerprint = output_fingerprint(row, artifacts, qc)
    row.cache_eligible = qc.status in {"PASS", "WARN"}
    row.reason_codes = list(qc.reason_codes)
    row.error_code = None
    row.error_message = None
    row.active_job_id = None
    session.flush()


def mark_blocked(session: Session, row: RenderExecution, reason_code: str) -> None:
    row.lifecycle = RenderExecutionLifecycle.BLOCKED
    row.reason_codes = [reason_code]
    row.error_code = reason_code
    row.cache_eligible = False
    row.active_job_id = None
    session.flush()


def mark_failed(session: Session, row: RenderExecution, reason_code: str) -> None:
    row.lifecycle = RenderExecutionLifecycle.FAILED
    row.reason_codes = [reason_code]
    row.error_code = reason_code
    row.cache_eligible = False
    row.qc_status = RenderQCStatus.FAIL
    row.active_job_id = None
    session.flush()


def mark_cancelled(session: Session, row: RenderExecution) -> None:
    row.lifecycle = RenderExecutionLifecycle.CANCELLED
    row.cache_eligible = False
    row.active_job_id = None
    session.flush()


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
    "execution_artifact_reference",
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
    "request_input_fingerprint",
    "request_payload",
    "resolve_runtime_identity",
    "runtime_fingerprint",
    "validate_candidate_for_render",
    "with_execution_id",
]
