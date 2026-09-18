"""Stage 5.1 visual-composition service: input resolution, persistence, reads.

Deterministic and provider-free. The only external processes are the injected
bounded Stage 5.1 seams (one display-geometry ffprobe, FFmpeg frame sampling,
FFmpeg scene-cut detection) and the CPU face detector. This module never makes a
network call, never calls a hosted provider, and never renders a video artifact.

Input resolution and freshness both pass through :func:`resolve_planner_inputs`
so a persisted plan can be re-fingerprinted from live rows without re-probing
geometry: the persisted plan payload is the geometry source, and a changed source
stat short-circuits to ``STALE`` before any probe.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.composition.analysis import (
    FFmpegFrameSampler,
    FFmpegSceneCutDetector,
    merge_spans,
    plan_analysis,
    scaled_dimensions,
)
from app.composition.detector import FaceDetector, detector_from_config
from app.composition.fingerprints import (
    analysis_fingerprint,
    build_stage51_input_payload,
    visual_composition_input_fingerprint,
)
from app.composition.geometry import DisplayProbe, FFprobeDisplayProbe
from app.composition.planner import build_visual_composition_plan
from app.composition.policy import (
    DETECTOR_IDENTITY,
    FINGERPRINT_VERSION,
    SCHEMA_VERSION,
    VISUAL_COMPOSITION_POLICY_VERSION,
    Stage51Config,
    VisualCompositionExecutionStatus,
    VisualCompositionStatus,
)
from app.composition.types import (
    BoundSpan,
    DisplayGeometry,
    PlannerInputs,
    VisualCompositionPlan,
)
from app.core.settings import Settings, get_settings
from app.models import ClipCandidate
from app.models.visual_composition_plan import VisualCompositionPlan as VisualCompositionPlanRow
from app.pipeline.executor import StageCancelled
from app.render.handoff import build_stage5_1_handoff
from app.render.media import SourceMediaFailure, source_media_identity
from app.render.policy import EXECUTABLE_STATUSES
from app.render.service import get_current_render_contract
from app.services.storage import StorageService

CURRENT = "CURRENT"
STALE = "STALE"
UNVERIFIABLE = "UNVERIFIABLE"

_CACHEABLE_STATUSES = {VisualCompositionStatus.READY_FOR_VISUAL_EXECUTION.value}


class CompositionError(RuntimeError):
    """Base Stage 5.1 visual-composition error."""


class CompositionInputError(CompositionError):
    """A prerequisite input is missing, stale, or unusable (repository-owned)."""

    retryable = False


class CompositionUnavailableError(CompositionError):
    """A required local tool (detector/FFmpeg) is unavailable for execution."""

    retryable = False


class CompositionCancelled(StageCancelled):
    """Cooperative cancellation while Stage 5.1 visual composition was running."""


class PreviewError(CompositionError):
    """A preview render was rejected (disabled or FFmpeg unavailable)."""

    retryable = False


@dataclass(frozen=True)
class VisualCompositionView:
    """A durable Stage 5.1 plan row plus live effectiveness information."""

    row: VisualCompositionPlanRow
    live_freshness: str
    effective: bool


class _PersistedDisplayProbe:
    """Replay a persisted display geometry without touching the filesystem."""

    def __init__(self, geometry: DisplayGeometry) -> None:
        self._geometry = geometry

    def probe(self, path: Path) -> DisplayGeometry:
        return self._geometry


# ---------------------------------------------------------------------------
# Input resolution


def resolve_planner_inputs(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    display_probe: DisplayProbe,
    config: Stage51Config,
) -> PlannerInputs | None:
    """Build the exact :class:`PlannerInputs` from the current Stage 5.0 contract.

    Returns ``None`` when the candidate has no current, live-effective, executable
    render contract. Display geometry is obtained through the injected
    ``display_probe`` (one bounded ffprobe at execution time; a persisted replay
    probe when re-fingerprinting an existing plan).
    """

    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        return None
    handoff = build_stage5_1_handoff(session, candidate.id)
    if handoff is None:
        return None
    contract = handoff.get("contract")
    if not isinstance(contract, Mapping):
        return None
    if contract.get("live_freshness") != CURRENT or not bool(contract.get("effective")):
        return None
    if str(contract.get("status") or "") not in EXECUTABLE_STATUSES:
        return None

    row = get_current_render_contract(session, candidate.id)
    if row is None:
        return None
    payload = dict(row.contract_payload or {})
    contract_id = str(contract.get("id") or "")
    source_media = handoff.get("source_media")
    if not isinstance(source_media, Mapping):
        return None
    identity = _as_mapping(source_media.get("identity"))
    relative_path = str(
        source_media.get("managed_relative_path") or identity.get("relative_path") or ""
    )
    video_stream = _as_mapping(source_media.get("video_stream"))
    frames_per_second = _as_float(video_stream.get("frame_rate"))
    output_profile = _mapping_dict(payload.get("output_profile"))
    safe_zone_key = str(output_profile.get("safe_zone_profile") or config.safe_zone_profile_key)
    caption_input = _mapping_dict(payload.get("caption_input"))
    caption_source_fingerprint = str(
        row.caption_source_fingerprint or caption_input.get("caption_source_fingerprint") or ""
    )
    blocks = _mapping_sequence(payload.get("blocks"))
    hero = payload.get("hero")
    hero_block_index = _hero_index(hero)
    spans = _bound_spans(blocks, hero_block_index)
    language_and_dialect = _mapping_dict(payload.get("language_and_dialect"))
    display_geometry = display_probe.probe(_source_path(relative_path))

    return PlannerInputs(
        candidate_id=str(candidate.id),
        source_id=str(candidate.source_video_id),
        contract_id=contract_id,
        contract_input_fingerprint=row.input_fingerprint,
        contract_output_fingerprint=row.output_fingerprint,
        contract_status=str(contract.get("status") or row.status.value),
        contract_ready=bool(contract.get("contract_ready")),
        selected_plan_id=_optional_str(handoff.get("selected_plan"), "id"),
        selection_id=_optional_str(handoff.get("selection"), "id"),
        final_refinement_id=_optional_str_value(caption_input.get("refinement_id")),
        source_media_relative_path=relative_path,
        source_media_identity=dict(identity),
        display_geometry=display_geometry,
        frames_per_second=frames_per_second,
        spans=spans,
        blocks=tuple(blocks),
        hero_block_index=hero_block_index,
        preservation_constraints=_str_tuple(payload.get("preservation_constraints")),
        retention=_mapping_dict(payload.get("retention")),
        governance=_mapping_dict(payload.get("governance")),
        narration=_mapping_dict(payload.get("narration")),
        materialization_slots=tuple(
            _mapping_sequence(_as_mapping(payload.get("materialization")).get("slots"))
        ),
        caption_input=caption_input,
        caption_source_fingerprint=caption_source_fingerprint,
        output_profile=output_profile,
        safe_zone_key=safe_zone_key,
        language=_optional_str_value(language_and_dialect.get("language")),
        dialect_profile=_optional_str_value(
            caption_input.get("dialect_profile")
            if caption_input.get("dialect_profile") is not None
            else language_and_dialect.get("dialect_profile")
        ),
        dialect_confidence=_as_float(
            caption_input.get("dialect_confidence")
            if caption_input.get("dialect_confidence") is not None
            else language_and_dialect.get("dialect_confidence")
        ),
        code_switch_evidence=_mapping_dict(caption_input.get("code_switch_evidence")),
        warnings=_str_tuple(payload.get("warnings")),
    )


def composition_input_fingerprint(inputs: PlannerInputs, config: Stage51Config) -> str:
    """Recompute the Stage 5.1 input fingerprint exactly as the planner does."""

    selected = merge_spans([(span.block_index, span.start, span.end) for span in inputs.spans])
    analysis_plan = plan_analysis(
        selected,
        analysis_fps=config.analysis_fps,
        max_analysis_seconds=config.max_analysis_seconds,
        max_analysis_frames=config.max_analysis_frames,
        scene_context_seconds=config.scene_context_seconds,
    )
    analysis_fp = analysis_fingerprint(
        {
            "spans": [span.as_dict() for span in inputs.spans],
            "effective_fps": analysis_plan.effective_fps,
            "effective_fps_reduced": analysis_plan.reduced_fps,
            "scene_context_seconds": config.scene_context_seconds,
            "min_scene_seconds": config.min_scene_seconds,
            "scene_cut_threshold": config.scene_cut_threshold,
            "display_geometry": inputs.display_geometry.as_dict(),
            "detector_identity": dict(DETECTOR_IDENTITY),
        }
    )
    payload = build_stage51_input_payload(
        candidate_id=inputs.candidate_id,
        candidate_key=inputs.candidate_id,
        candidate_is_current=True,
        disposition=inputs.contract_status,
        analysis_fingerprint=analysis_fp,
        source_id=inputs.source_id,
        source_media_identity=dict(inputs.source_media_identity),
        contract_input_fingerprint=inputs.contract_input_fingerprint,
        contract_output_fingerprint=inputs.contract_output_fingerprint,
        contract_status=inputs.contract_status,
        contract_ready=inputs.contract_ready,
        contract_is_current=True,
        contract_live_freshness=CURRENT,
        contract_effective=True,
        selected_plan_id=inputs.selected_plan_id,
        selection_id=inputs.selection_id,
        final_refinement_id=inputs.final_refinement_id,
        display_geometry=inputs.display_geometry.as_dict(),
        rotation_degrees=inputs.display_geometry.rotation_degrees,
        bound_spans=[span.as_dict() for span in inputs.spans],
        caption_source_fingerprint=inputs.caption_source_fingerprint,
        caption_payload=dict(inputs.caption_input),
        output_profile=dict(inputs.output_profile),
        config=config,
        detector_identity=dict(DETECTOR_IDENTITY),
        safe_zone_key=inputs.safe_zone_key,
    )
    return visual_composition_input_fingerprint(payload)


# ---------------------------------------------------------------------------
# Execution orchestration


def execute_visual_composition(
    session: Session,
    plan_id: uuid.UUID | str,
    *,
    storage: StorageService | None = None,
    settings: Settings | None = None,
    force: bool = False,
    display_probe: DisplayProbe | None = None,
    frame_sampler: object | None = None,
    scene_cut_detector: object | None = None,
    detector: FaceDetector | None = None,
    cancel_check: Any | None = None,
    persist_guard: Any | None = None,
) -> VisualCompositionPlanRow | None:
    """Resolve inputs, run the pure planner with real seams, and persist the plan.

    ``persist_guard`` is an optional claim-fencing predicate evaluated immediately
    before persistence so the executor can reject a superseded or cancelled run.
    ``cancel_check`` is polled after planning and before persistence; the planner
    itself polls it during frame sampling.
    """

    resolved = settings or get_settings()
    config = resolved.stage51_config()
    storage_service = storage or StorageService(resolved.storage_root)
    row = session.get(VisualCompositionPlanRow, _as_uuid(plan_id))
    if row is None:
        raise CompositionInputError("visual-composition plan is missing")
    candidate = session.get(ClipCandidate, row.clip_candidate_id)
    if candidate is None:
        raise CompositionInputError("candidate is missing")

    probe = display_probe or FFprobeDisplayProbe(binary=resolved.ffprobe_binary)
    inputs = resolve_planner_inputs(session, candidate.id, display_probe=probe, config=config)
    if inputs is None:
        raise CompositionInputError(
            "candidate has no current live-effective executable Stage 5.0 contract"
        )
    sampler = frame_sampler or _default_frame_sampler(inputs, config, storage_service, resolved)
    scene_detector = scene_cut_detector or FFmpegSceneCutDetector(
        ffmpeg_binary=resolved.ffmpeg_binary
    )
    face_detector = detector or detector_from_config(config)

    plan = build_visual_composition_plan(
        inputs=inputs,
        config=config,
        storage=storage_service,
        frame_sampler=sampler,  # type: ignore[arg-type]
        scene_cut_detector=scene_detector,  # type: ignore[arg-type]
        detector=face_detector,
        cancel_check=cancel_check,
    )
    if cancel_check is not None and cancel_check():
        raise CompositionCancelled("visual composition cancelled before persistence")
    if persist_guard is not None and not persist_guard():
        return None
    return persist_plan(session, inputs, plan, row_id=row.id)


def _default_frame_sampler(
    inputs: PlannerInputs,
    config: Stage51Config,
    storage: StorageService,
    settings: Settings,
) -> FFmpegFrameSampler:
    geometry = inputs.display_geometry
    width, height = scaled_dimensions(
        int(geometry.display_width),
        int(geometry.display_height),
        config.analysis_frame_max_dimension,
    )
    return FFmpegFrameSampler(
        _source_path(inputs.source_media_relative_path),
        frame_size=(width, height),
        ffmpeg_binary=settings.ffmpeg_binary,
        storage=storage,
    )


# ---------------------------------------------------------------------------
# Persistence


def persist_plan(
    session: Session,
    inputs: PlannerInputs,
    plan: VisualCompositionPlan,
    *,
    row_id: uuid.UUID | str | None = None,
) -> VisualCompositionPlanRow:
    """Persist one plan as the current row, preserving every historical row.

    The durable queue envelope row (``row_id``) is reused in place when its input
    fingerprint already matches (or is still empty); a genuinely new input
    fingerprint versions a new row and demotes the previous current row. A
    uniqueness race is recovered inside a savepoint.
    """

    candidate_id = _as_uuid(inputs.candidate_id)
    envelope = session.get(VisualCompositionPlanRow, _as_uuid(row_id)) if row_id else None
    if envelope is not None and (
        not envelope.input_fingerprint or envelope.input_fingerprint == plan.input_fingerprint
    ):
        other = _row_by_fingerprint(session, candidate_id, plan.input_fingerprint)
        target = other if (other is not None and other.id != envelope.id) else envelope
        _demote_others(session, candidate_id, target.id)
        _apply(target, inputs, plan, is_current=True, execution_status=_execution_status(plan))
        session.flush()
        return target

    existing = _row_by_fingerprint(session, candidate_id, plan.input_fingerprint)
    if existing is not None:
        _demote_others(session, candidate_id, existing.id)
        _apply(existing, inputs, plan, is_current=True, execution_status=_execution_status(plan))
        session.flush()
        return existing

    _demote_others(session, candidate_id, None)
    new_row = _build_row(inputs, plan)
    try:
        with session.begin_nested():
            session.add(new_row)
            session.flush()
    except IntegrityError:
        concurrent = _row_by_fingerprint(session, candidate_id, plan.input_fingerprint)
        if concurrent is None:
            raise
        _demote_others(session, candidate_id, concurrent.id)
        _apply(concurrent, inputs, plan, is_current=True, execution_status=_execution_status(plan))
        session.flush()
        return concurrent
    return new_row


def _execution_status(plan: VisualCompositionPlan) -> VisualCompositionExecutionStatus:
    if plan.status == VisualCompositionStatus.FAILED.value:
        return VisualCompositionExecutionStatus.FAILED
    if plan.status == VisualCompositionStatus.BLOCKED.value:
        return VisualCompositionExecutionStatus.COMPLETE
    return VisualCompositionExecutionStatus.COMPLETE


def _apply(
    row: VisualCompositionPlanRow,
    inputs: PlannerInputs,
    plan: VisualCompositionPlan,
    *,
    is_current: bool,
    execution_status: VisualCompositionExecutionStatus,
) -> None:
    row.render_contract_id = _as_uuid(inputs.contract_id)
    row.transformation_selection_id = _optional_uuid(inputs.selection_id)
    row.selected_plan_id = _optional_uuid(inputs.selected_plan_id)
    row.final_refinement_id = _optional_uuid(inputs.final_refinement_id)
    row.status = VisualCompositionStatus(plan.status)
    row.execution_status = execution_status
    row.plan_ready = plan.plan_ready
    row.is_current = is_current
    row.reason_codes = list(plan.reason_codes)
    row.input_fingerprint = plan.input_fingerprint
    row.output_fingerprint = plan.output_fingerprint
    row.contract_input_fingerprint = plan.contract_input_fingerprint
    row.contract_output_fingerprint = plan.contract_output_fingerprint
    row.caption_source_fingerprint = plan.caption_source_fingerprint
    row.source_media_fingerprint = plan.source_media_fingerprint
    row.source_media_identity = dict(plan.source_media_identity)
    row.analysis_fingerprint = plan.analysis_fingerprint
    row.framing_fingerprint = plan.framing_fingerprint
    row.ass_fingerprint = plan.ass_fingerprint
    row.policy_version = VISUAL_COMPOSITION_POLICY_VERSION
    row.schema_version = SCHEMA_VERSION
    row.fingerprint_version = FINGERPRINT_VERSION
    row.plan_payload = dict(plan.payload)
    row.readiness = _mapping_dict(plan.payload.get("readiness"))
    row.metrics = dict(plan.metrics)
    row.cache_eligible = plan.cache_eligible
    row.active_job_id = None


def _build_row(inputs: PlannerInputs, plan: VisualCompositionPlan) -> VisualCompositionPlanRow:
    row = VisualCompositionPlanRow(
        source_video_id=_as_uuid(inputs.source_id),
        clip_candidate_id=_as_uuid(inputs.candidate_id),
    )
    _apply(row, inputs, plan, is_current=True, execution_status=_execution_status(plan))
    return row


def _demote_others(session: Session, candidate_id: uuid.UUID, keep_id: uuid.UUID | None) -> None:
    rows = session.scalars(
        select(VisualCompositionPlanRow).where(
            VisualCompositionPlanRow.clip_candidate_id == candidate_id,
            VisualCompositionPlanRow.is_current.is_(True),
        )
    ).all()
    for row in rows:
        if keep_id is not None and row.id == keep_id:
            continue
        row.is_current = False
    session.flush()


def _row_by_fingerprint(
    session: Session, candidate_id: uuid.UUID, fingerprint: str
) -> VisualCompositionPlanRow | None:
    if not fingerprint:
        return None
    return session.scalars(
        select(VisualCompositionPlanRow).where(
            VisualCompositionPlanRow.clip_candidate_id == candidate_id,
            VisualCompositionPlanRow.input_fingerprint == fingerprint,
        )
    ).first()


# ---------------------------------------------------------------------------
# Reads


def get_current_visual_composition(
    session: Session, candidate_id: uuid.UUID | str
) -> VisualCompositionPlanRow | None:
    candidate_uuid = _as_uuid(candidate_id)
    return session.scalars(
        select(VisualCompositionPlanRow)
        .where(VisualCompositionPlanRow.clip_candidate_id == candidate_uuid)
        .where(VisualCompositionPlanRow.is_current.is_(True))
        .order_by(VisualCompositionPlanRow.created_at.desc())
    ).first()


def read_visual_composition(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    display_probe: DisplayProbe,
    config: Stage51Config,
) -> VisualCompositionView | None:
    """Read the current plan and its live freshness without re-probing geometry."""

    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        return None
    row = get_current_visual_composition(session, candidate.id)
    if row is None:
        return None
    return _view_for_row(session, candidate, row, config)


def read_visual_composition_by_id(
    session: Session,
    plan_id: uuid.UUID | str,
    *,
    display_probe: DisplayProbe,
    config: Stage51Config,
) -> VisualCompositionView | None:
    """Read one historical plan row with its own current/freshness/effective state."""

    row = session.get(VisualCompositionPlanRow, _as_uuid(plan_id))
    if row is None:
        return None
    candidate = session.get(ClipCandidate, row.clip_candidate_id)
    if candidate is None:
        return None
    return _view_for_row(session, candidate, row, config)


def _view_for_row(
    session: Session,
    candidate: ClipCandidate,
    row: VisualCompositionPlanRow,
    config: Stage51Config,
) -> VisualCompositionView:
    freshness = _live_plan_freshness(session, candidate, row, config)
    effective = bool(row.is_current) and freshness == CURRENT and bool(row.plan_ready)
    return VisualCompositionView(row=row, live_freshness=freshness, effective=effective)


def _live_plan_freshness(
    session: Session,
    candidate: ClipCandidate,
    row: VisualCompositionPlanRow,
    config: Stage51Config,
) -> str:
    """Recompute the plan input fingerprint from live rows (stat only, no probe)."""

    if not row.input_fingerprint:
        return UNVERIFIABLE
    try:
        live_identity = source_media_identity(
            candidate.source_video_id,
            candidate.source_video.source_uri if candidate.source_video is not None else None,
            storage=StorageService(get_settings().storage_root),
        )
    except (SourceMediaFailure, OSError, ValueError):
        return STALE
    if _identity_changed(_as_mapping(row.source_media_identity), live_identity.as_dict()):
        return STALE
    try:
        geometry = _geometry_from_payload(row.plan_payload)
        inputs = resolve_planner_inputs(
            session,
            candidate.id,
            display_probe=_PersistedDisplayProbe(geometry),
            config=config,
        )
        if inputs is None:
            return STALE
        recomputed = composition_input_fingerprint(inputs, config)
    except Exception:
        return UNVERIFIABLE
    return CURRENT if recomputed == row.input_fingerprint else STALE


def _identity_changed(stored: Mapping[str, object], live: Mapping[str, object]) -> bool:
    keys = ("source_id", "relative_path", "size_bytes", "mtime_ns")
    return any(stored.get(key) != live.get(key) for key in keys)


def _geometry_from_payload(payload: Mapping[str, object]) -> DisplayGeometry:
    geometry = _as_mapping(payload.get("geometry"))
    if not geometry:
        raise CompositionInputError("plan payload has no persisted display geometry")
    return DisplayGeometry(
        encoded_width=_required_int(geometry.get("encoded_width"), "encoded_width"),
        encoded_height=_required_int(geometry.get("encoded_height"), "encoded_height"),
        rotation_degrees=_required_int(geometry.get("rotation_degrees") or 0, "rotation_degrees"),
        display_width=_required_int(geometry.get("display_width"), "display_width"),
        display_height=_required_int(geometry.get("display_height"), "display_height"),
        pixel_aspect_ratio=_as_float(geometry.get("pixel_aspect_ratio"), 1.0),
        square_pixels_applied=bool(geometry.get("square_pixels_applied", True)),
        exotic_pixel_aspect=bool(geometry.get("exotic_pixel_aspect", False)),
    )


# ---------------------------------------------------------------------------
# Helpers


def _bound_spans(
    blocks: Sequence[Mapping[str, object]], hero_block_index: int | None
) -> tuple[BoundSpan, ...]:
    spans: list[BoundSpan] = []
    for block in blocks:
        if str(block.get("block_type") or "") != "SOURCE_EXCERPT":
            continue
        if str(block.get("slot_kind") or "") != "SOURCE_MEDIA":
            continue
        binding = block.get("source_binding")
        if not isinstance(binding, Mapping) or not bool(binding.get("rebind_valid")):
            continue
        start = binding.get("final_clip_start")
        end = binding.get("final_clip_end")
        if start is None or end is None:
            continue
        block_index = _optional_int(block.get("block_index")) or 0
        spans.append(
            BoundSpan(
                block_index=block_index,
                start=float(start),
                end=float(end),
                word_start_index=_optional_int(binding.get("final_clip_word_start")),
                word_end_index=_optional_int(binding.get("final_clip_word_end")),
                source_role=_optional_str_value(binding.get("source_role")),
                is_hero=hero_block_index is not None and hero_block_index == block_index,
                caption_text=str(binding.get("final_clip_text") or ""),
            )
        )
    return tuple(spans)


def _hero_index(hero: object) -> int | None:
    if not isinstance(hero, Mapping):
        return None
    return _optional_int(hero.get("block_index"))


def _source_path(relative_path: str) -> Path:
    root = get_settings().storage_root
    return Path(root) / relative_path


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _mapping_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_sequence(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _optional_str(container: object, key: str) -> str | None:
    return _optional_str_value(_as_mapping(container).get(key))


def _optional_str_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _required_int(value: object, field: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise CompositionInputError(f"plan payload geometry is missing {field}")
    return parsed


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _as_uuid(value: object) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as error:
        raise CompositionInputError("invalid identifier") from error


def _optional_uuid(value: object) -> uuid.UUID | None:
    if value is None:
        return None
    return _as_uuid(value)


__all__ = [
    "CURRENT",
    "STALE",
    "UNVERIFIABLE",
    "CompositionError",
    "CompositionInputError",
    "CompositionUnavailableError",
    "CompositionCancelled",
    "PreviewError",
    "VisualCompositionView",
    "composition_input_fingerprint",
    "execute_visual_composition",
    "get_current_visual_composition",
    "persist_plan",
    "read_visual_composition",
    "read_visual_composition_by_id",
    "resolve_planner_inputs",
]
