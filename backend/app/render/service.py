"""Deterministic, synchronous, provider-free Stage 5.0 preflight service.

Artificial-intelligence-free by design: no Celery task, no ``ProcessingJob``, no
queue/executor, no ``PipelineStage``/``PipelineRun``, no automatic Stage 4 rerun,
no automatic FINAL_CLIP refinement, no rendering, no captions, no TTS, and no
Gemini/Qwen/Whisper/network path. The only external process is a bounded
read-only ffprobe metadata probe through an injectable seam.

Concurrent POST requests converge on one authoritative current contract through
candidate-row locking, a partial unique index on ``is_current``, savepoint
uniqueness recovery, and post-lock freshness revalidation.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import (
    ExecutionSlotKind,
    FinalClipCompatibilityOutcome,
    RefinementPriority,
    RefinementStatus,
    RenderContractStatus,
    TransformationSelectionStatus,
)
from app.core.settings import Settings, get_settings
from app.media.ffprobe import FFprobe
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    RenderContract,
    TransformationPlan,
)
from app.render.binding import build_bound_spans
from app.render.compatibility import evaluate_compatibility
from app.render.fingerprints import (
    build_caption_source_payload,
    build_render_contract_input_payload,
    caption_source_fingerprint,
    probe_fingerprint,
    render_contract_input_fingerprint,
    render_contract_output_fingerprint,
    source_media_identity_fingerprint,
)
from app.render.media import (
    SourceMediaFailure,
    run_source_media_preflight,
    source_media_identity,
)
from app.render.policy import (
    AUTHORED_TEXT_MATERIALIZATION_REQUIRED,
    BLOCKED,
    COMPATIBILITY_POLICY_VERSION,
    EXECUTABLE_STATUSES,
    FINAL_CLIP_REFINEMENT_REQUIRED,
    INVALID_SOURCE_BINDING,
    MATERIALIZATION_REQUIRED,
    NARRATION_MATERIALIZATION_REQUIRED,
    NO_SELECTED_PLAN,
    NO_USABLE_FINAL_CLIP_REFINEMENT,
    PLAN_NOT_CURRENT,
    READY_FOR_RENDER_PLANNING,
    RENDER_CONTRACT_FINGERPRINT_VERSION,
    RENDER_CONTRACT_POLICY_VERSION,
    RENDER_CONTRACT_SCHEMA_VERSION,
    SELECTED_PLAN_NOT_FOUND,
    SELECTION_NOT_SELECTED,
    SOURCE_MEDIA_UNAVAILABLE,
    SPAN_OUT_OF_MEDIA_BOUNDS,
    SPAN_REVERSED_OR_NEGATIVE,
    STALE_SELECTION_INPUT,
    UNRESOLVED_REQUIRED_VERIFICATION,
    UPSTREAM_CHAIN_STALE,
    UPSTREAM_REVALIDATION_REQUIRED,
    Stage50Config,
    render_profile_for,
    stage50_config_payload,
)
from app.render.types import (
    ContractBlock,
    ContractDraft,
    FinalClipEvidence,
    MaterializationSlot,
    RenderContractPreflight,
)
from app.services.storage import StorageService, StorageValidationError
from app.transformation.governance.handoff import FRESHNESS_VERIFIED_CURRENT
from app.transformation.selection.handoff import build_execution_handoff
from app.transformation.selection.service import as_uuid, read_selection

_FINAL_READY = {RefinementStatus.FINAL_TRANSCRIPT_READY.value}
_SELECTED_STATUSES = {
    TransformationSelectionStatus.PLAN_SELECTED.value,
    TransformationSelectionStatus.PLAN_SELECTED_WITH_CAUTION.value,
}
_RESOLVED_VERIFICATION_STATES = {"GROUNDED_IN_SOURCE", "NOT_APPLICABLE"}


@dataclass(frozen=True)
class RenderContractView:
    """A durable render-contract row plus live effectiveness information."""

    row: RenderContract
    live_freshness: str
    effective: bool


@dataclass
class _Gathered:
    candidate: ClipCandidate
    handoff: Mapping[str, Any]
    selection: Any
    selection_view: Any
    selected_plan: TransformationPlan | None
    planning_refinement: CandidateRefinement | None
    final_refinement: CandidateRefinement | None
    source_segments: Sequence[Mapping[str, object]]
    governance_snapshot: Mapping[str, object]


@dataclass
class _Assembly:
    payload: dict[str, object]
    readiness: dict[str, object]
    blocks: list[ContractBlock]
    slots: list[MaterializationSlot]
    materialization_required: bool
    extra_reasons: list[str]


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _as_sequence(value: object) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _as_str_list(value: object) -> list[str]:
    return [str(item) for item in _as_sequence(value)]


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


# ---------------------------------------------------------------------------
# Public service API
# ---------------------------------------------------------------------------


def get_render_contract(session: Session, contract_id: uuid.UUID | str) -> RenderContract | None:
    contract_uuid = as_uuid(contract_id)
    if contract_uuid is None:
        return None
    return session.get(RenderContract, contract_uuid)


def list_render_contracts(session: Session, candidate_id: uuid.UUID | str) -> list[RenderContract]:
    candidate_uuid = as_uuid(candidate_id)
    if candidate_uuid is None:
        return []
    return list(
        session.scalars(
            select(RenderContract)
            .where(RenderContract.clip_candidate_id == candidate_uuid)
            .order_by(RenderContract.created_at.desc())
        ).all()
    )


def get_current_render_contract(
    session: Session, candidate_id: uuid.UUID | str
) -> RenderContract | None:
    candidate_uuid = as_uuid(candidate_id)
    if candidate_uuid is None:
        return None
    return session.scalars(
        select(RenderContract)
        .where(RenderContract.clip_candidate_id == candidate_uuid)
        .where(RenderContract.is_current.is_(True))
        .order_by(RenderContract.created_at.desc())
    ).first()


def read_render_contract(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    settings: Settings | None = None,
) -> RenderContractView | None:
    """Read-only: recompute live freshness (stat only, no probe, no provider)."""

    candidate = session.get(ClipCandidate, as_uuid(candidate_id))
    if candidate is None:
        return None
    row = get_current_render_contract(session, candidate.id)
    if row is None:
        return None
    resolved = settings or get_settings()
    freshness = _live_freshness(session, candidate, row, resolved)
    return RenderContractView(
        row=row,
        live_freshness=freshness,
        effective=bool(row.is_current) and freshness == "CURRENT",
    )


def create_render_contract(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    settings: Settings | None = None,
    storage: StorageService | None = None,
    prober: Any | None = None,
) -> RenderContractView | None:
    """Synchronously run preflight, then persist or reuse one current contract."""

    candidate = session.get(ClipCandidate, as_uuid(candidate_id))
    if candidate is None:
        return None
    _lock_candidate(session, candidate.id)
    candidate = session.get(ClipCandidate, candidate.id)
    if candidate is None:  # pragma: no cover - deleted under lock
        return None

    resolved_settings = settings or get_settings()
    storage_service = storage or StorageService(resolved_settings.storage_root)
    prober_service = prober or FFprobe(binary=resolved_settings.ffprobe_binary)

    preflight = _run_preflight(
        session,
        candidate,
        settings=resolved_settings,
        storage=storage_service,
        prober=prober_service,
    )
    row = _persist(session, candidate, preflight)
    session.flush()
    session.refresh(row)
    return RenderContractView(row=row, live_freshness="CURRENT", effective=True)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _run_preflight(
    session: Session,
    candidate: ClipCandidate,
    *,
    settings: Settings,
    storage: StorageService,
    prober: Any,
) -> RenderContractPreflight:
    started = time.monotonic()
    gathered = _gather(session, candidate)
    identity_payload = _best_effort_identity(candidate, storage)
    config = settings.stage50_config()

    selection = gathered.selection
    selected_plan = gathered.selected_plan
    final_row = gathered.final_refinement

    if selection is None or selection.selected_plan_id is None:
        return _finalize(
            session,
            candidate,
            gathered,
            identity_payload,
            config,
            status=BLOCKED,
            reasons=[NO_SELECTED_PLAN],
            outcome=None,
            compatibility=None,
            started=started,
        )
    if selection.status.value not in _SELECTED_STATUSES:
        return _finalize(
            session,
            candidate,
            gathered,
            identity_payload,
            config,
            status=BLOCKED,
            reasons=[SELECTION_NOT_SELECTED],
            outcome=None,
            compatibility=None,
            started=started,
        )
    if selected_plan is None:
        return _finalize(
            session,
            candidate,
            gathered,
            identity_payload,
            config,
            status=BLOCKED,
            reasons=[SELECTED_PLAN_NOT_FOUND],
            outcome=None,
            compatibility=None,
            started=started,
        )
    if not bool(selected_plan.is_current):
        return _finalize(
            session,
            candidate,
            gathered,
            identity_payload,
            config,
            status=BLOCKED,
            reasons=[PLAN_NOT_CURRENT],
            outcome=None,
            compatibility=None,
            started=started,
        )
    readiness_state = str(gathered.handoff.get("execution_readiness") or "")
    effective = gathered.selection_view is not None and bool(gathered.selection_view.effective)
    freshness = (
        gathered.selection_view.live_freshness
        if gathered.selection_view is not None
        else FRESHNESS_VERIFIED_CURRENT
    )
    if not effective or freshness != FRESHNESS_VERIFIED_CURRENT:
        # A newer usable FINAL_CLIP invalidates the frozen upstream chain by
        # design; Stage 5.0 reconciles it through deterministic compatibility
        # instead of blocking. Every other stale/ineffective input fails closed.
        if readiness_state != "REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK":
            reason = STALE_SELECTION_INPUT if freshness == "STALE" else UPSTREAM_CHAIN_STALE
            return _finalize(
                session,
                candidate,
                gathered,
                identity_payload,
                config,
                status=BLOCKED,
                reasons=[reason],
                outcome=None,
                compatibility=None,
                started=started,
            )

    if _required_verification_unresolved(selected_plan, gathered.governance_snapshot):
        return _finalize(
            session,
            candidate,
            gathered,
            identity_payload,
            config,
            status=BLOCKED,
            reasons=[UNRESOLVED_REQUIRED_VERIFICATION],
            outcome=None,
            compatibility=None,
            started=started,
        )

    if final_row is None or not _is_usable_final(final_row):
        return _finalize(
            session,
            candidate,
            gathered,
            identity_payload,
            config,
            status=FINAL_CLIP_REFINEMENT_REQUIRED,
            reasons=[NO_USABLE_FINAL_CLIP_REFINEMENT],
            outcome=None,
            compatibility=None,
            started=started,
        )

    final = _final_evidence(final_row)
    caption_payload = build_caption_source_payload(
        final_transcript=final.final_transcript,
        word_timestamps=final_row.word_timestamps or [],
        dialect_profile=final.dialect_profile,
        dialect_confidence=final.dialect_confidence,
        code_switch_evidence=final.code_switch_evidence,
        final_refinement_output_fingerprint=final.output_fingerprint,
    )
    caption_fp = caption_source_fingerprint(caption_payload)

    planning_identity = _planning_identity(selection, gathered.planning_refinement)
    exact_identity = _exact_identity_match(selection, final_row)

    planning_start = (
        gathered.planning_refinement.refined_start
        if gathered.planning_refinement is not None
        else None
    )
    planning_end = (
        gathered.planning_refinement.refined_end
        if gathered.planning_refinement is not None
        else None
    )
    blocks = [dict(block) for block in _as_sequence(selected_plan.blocks)]
    compatibility = evaluate_compatibility(
        blocks=blocks,
        hero_block_index=selected_plan.hero_block_index,
        hook_payoff_evidence=dict(selected_plan.hook_payoff_evidence or {}),
        final=final,
        source_segments=gathered.source_segments,
        planning_refined_start=planning_start,
        planning_refined_end=planning_end,
        exact_identity_match=exact_identity,
        caption_source_fingerprint=caption_fp,
        planning_refinement_id=str(planning_identity.get("id") or ""),
        planning_refinement_priority=str(planning_identity.get("priority") or ""),
        planning_refinement_quality_level=str(planning_identity.get("quality_level") or ""),
        planning_output_fingerprint=str(planning_identity.get("output_fingerprint") or ""),
    )
    outcome = compatibility.outcome

    if outcome == FinalClipCompatibilityOutcome.SOURCE_SPAN_NO_LONGER_VALID.value:
        return _finalize(
            session,
            candidate,
            gathered,
            identity_payload,
            config,
            status=INVALID_SOURCE_BINDING,
            reasons=[
                reason for verdict in compatibility.per_block for reason in verdict.reason_codes
            ]
            or [SPAN_REVERSED_OR_NEGATIVE],
            outcome=outcome,
            compatibility=compatibility,
            started=started,
        )
    if outcome in {
        FinalClipCompatibilityOutcome.MATERIAL_SEMANTIC_CHANGE.value,
        FinalClipCompatibilityOutcome.MATERIAL_TIMING_CHANGE.value,
        FinalClipCompatibilityOutcome.UNRESOLVED_COMPATIBILITY.value,
    }:
        return _finalize(
            session,
            candidate,
            gathered,
            identity_payload,
            config,
            status=UPSTREAM_REVALIDATION_REQUIRED,
            reasons=list(compatibility.reason_codes)
            or [reason for verdict in compatibility.per_block for reason in verdict.reason_codes],
            outcome=outcome,
            compatibility=compatibility,
            started=started,
        )

    # Compatible: run the bounded source-media preflight.
    cached_identity, cached_facts = _probe_reuse(session, candidate, identity_payload)
    preflight = run_source_media_preflight(
        candidate.source_video_id,
        candidate.source_video.source_uri if candidate.source_video is not None else None,
        storage=storage,
        prober=prober,
        probe_reuse_enabled=config.probe_reuse_enabled,
        cached_identity=cached_identity,
        cached_facts=cached_facts,
    )
    if not preflight.ok or preflight.identity is None or preflight.facts is None:
        return _finalize(
            session,
            candidate,
            gathered,
            identity_payload,
            config,
            status=SOURCE_MEDIA_UNAVAILABLE,
            reasons=[preflight.reason_code or "SOURCE_MEDIA_UNAVAILABLE"],
            outcome=outcome,
            compatibility=compatibility,
            started=started,
            probe_reused=preflight.probe_reused,
        )

    media_identity = _identity_payload_with_hash(candidate, preflight.identity.as_dict())
    bound_spans = build_bound_spans(
        blocks,
        {verdict.block_index: verdict for verdict in compatibility.per_block},
        hero_block_index=selected_plan.hero_block_index,
        words=final.words,
    )
    bounds_reason = _validate_media_bounds(bound_spans, preflight.facts.duration_seconds)
    if bounds_reason is not None:
        return _finalize(
            session,
            candidate,
            gathered,
            media_identity,
            config,
            status=INVALID_SOURCE_BINDING,
            reasons=[bounds_reason],
            outcome=outcome,
            compatibility=compatibility,
            started=started,
            probe_reused=preflight.probe_reused,
            facts=preflight.facts,
        )

    return _finalize(
        session,
        candidate,
        gathered,
        media_identity,
        config,
        status=None,
        reasons=[],
        outcome=outcome,
        compatibility=compatibility,
        started=started,
        probe_reused=preflight.probe_reused,
        facts=preflight.facts,
        bound_spans=bound_spans,
    )


# ---------------------------------------------------------------------------
# Draft finalization / persistence
# ---------------------------------------------------------------------------


def _finalize(
    session: Session,
    candidate: ClipCandidate,
    gathered: _Gathered,
    identity_payload: Mapping[str, object],
    config: Stage50Config,
    *,
    status: str | None,
    reasons: Sequence[str],
    outcome: str | None,
    compatibility: Any,
    started: float,
    probe_reused: bool = False,
    facts: Any = None,
    bound_spans: Sequence[Any] = (),
) -> RenderContractPreflight:
    selection = gathered.selection
    selected_plan = gathered.selected_plan
    final_row = gathered.final_refinement
    profile = render_profile_for(config.profile_key)

    compatibility_result = compatibility
    # Fingerprint inputs are computed from live rows only, independent of the
    # preflight exit path, so read-time freshness can recompute them exactly.
    final_fp = final_row.output_fingerprint or "" if final_row is not None else ""
    planning_fp = (selection.refinement_output_fingerprint if selection is not None else "") or ""
    caption_fp = ""
    if final_row is not None and _is_usable_final(final_row):
        caption_fp = _caption_source_fingerprint(final_row)
    verification_state, verification_unresolved = _verification_flags(gathered.governance_snapshot)

    media_fp = source_media_identity_fingerprint(identity_payload) if identity_payload else ""
    facts_payload = facts.as_dict() if facts is not None else {}
    probe_fp = probe_fingerprint(facts_payload, identity_payload) if facts_payload else ""

    input_fp = _input_fingerprint(
        candidate=candidate,
        gathered=gathered,
        identity_payload=identity_payload,
        caption_fingerprint=caption_fp,
        final_refinement_output_fingerprint=final_fp,
        planning_refinement_output_fingerprint=planning_fp,
        verification_state=verification_state,
        verification_unresolved=verification_unresolved,
        profile=profile,
        config=config,
    )

    executable = status in EXECUTABLE_STATUSES
    if status is None:
        # Compatible: assemble the contract and decide readiness.
        assembly = _assemble_contract(
            candidate=candidate,
            gathered=gathered,
            final_row=final_row,
            compatibility=compatibility_result,
            bound_spans=bound_spans,
            facts=facts,
            media_identity=identity_payload,
            caption_fingerprint=caption_fp,
            profile=profile,
            config=config,
        )
        contract_payload = assembly.payload
        readiness = assembly.readiness
        slots = list(assembly.slots)
        materialization_required = assembly.materialization_required
        status = MATERIALIZATION_REQUIRED if materialization_required else READY_FOR_RENDER_PLANNING
        executable = True
        reasons = [*reasons, *assembly.extra_reasons]
    else:
        contract_payload = {}
        readiness = _non_ready_readiness(status, list(reasons))
        slots = []
    assert status is not None

    draft = ContractDraft(
        status=status,
        contract_ready=executable,
        reason_codes=tuple(dict.fromkeys(str(reason) for reason in reasons)),
        compatibility_outcome=outcome,
        compatibility_evidence=_compatibility_evidence(compatibility_result),
        selection_id=str(selection.id) if selection is not None else None,
        selected_plan_id=str(selected_plan.id) if selected_plan is not None else None,
        final_refinement_id=str(final_row.id) if final_row is not None else None,
        source_media_identity=dict(identity_payload),
        source_probe={
            "facts": facts_payload,
            "probe_reused": probe_reused,
            "probe_fingerprint": probe_fp,
        },
        readiness=readiness,
        contract_payload=dict(contract_payload),
        selected_plan_fingerprint=(
            selected_plan.plan_output_fingerprint if selected_plan is not None else ""
        ),
        planning_refinement_output_fingerprint=planning_fp,
        final_refinement_output_fingerprint=final_fp,
        caption_source_fingerprint=caption_fp,
        source_media_fingerprint=media_fp,
        probe_fingerprint=probe_fp,
        input_fingerprint=input_fp,
        output_fingerprint=render_contract_output_fingerprint(
            {"payload": dict(contract_payload), "status": status}
        ),
        profile_key=profile.key,
        profile_version=profile.semantic_version,
        metrics={
            "preflight_runs": 1,
            "source_probe_reused": 1 if probe_reused else 0,
            "compatibility_checks": 1 if compatibility_result is not None else 0,
            "compatible_non_material_changes": (
                1
                if outcome == FinalClipCompatibilityOutcome.COMPATIBLE_NON_MATERIAL_CHANGE.value
                else 0
            ),
            "upstream_revalidation_blocks": (1 if status == UPSTREAM_REVALIDATION_REQUIRED else 0),
            "source_media_failures": 1 if status == SOURCE_MEDIA_UNAVAILABLE else 0,
            "bound_source_spans": len(bound_spans),
            "materialization_slots": len(slots),
            "preflight_wall_seconds": round(time.monotonic() - started, 6),
        },
    )
    return RenderContractPreflight(
        status=status,
        reason_codes=draft.reason_codes,
        draft=draft,
        source_probe_reused=probe_reused,
    )


def _persist(
    session: Session, candidate: ClipCandidate, preflight: RenderContractPreflight
) -> RenderContract:
    draft = preflight.draft
    input_fp = draft.input_fingerprint
    current = get_current_render_contract(session, candidate.id)
    if current is not None and current.input_fingerprint == input_fp:
        current.metrics = _merge_cache_metrics(current.metrics, draft.metrics)
        session.flush()
        return current
    existing = _row_by_fingerprint(session, candidate.id, input_fp)
    if current is not None:
        current.is_current = False
        session.flush()
    if existing is not None:
        existing.is_current = True
        existing.metrics = _merge_cache_metrics(existing.metrics, draft.metrics)
        session.flush()
        return existing
    row = _build_row(candidate, draft)
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
    except IntegrityError:
        concurrent = _row_by_fingerprint(session, candidate.id, input_fp)
        if concurrent is not None:
            concurrent.is_current = True
            session.flush()
            return concurrent
        raise
    return row


def _merge_cache_metrics(
    existing: Mapping[str, object], incoming: Mapping[str, object]
) -> dict[str, object]:
    merged: dict[str, object] = dict(existing)
    runs = _as_int(merged.get("preflight_runs"))
    merged["preflight_runs"] = runs + 1
    merged["contract_cache_hits"] = _as_int(merged.get("contract_cache_hits")) + 1
    merged["preflight_wall_seconds"] = incoming.get("preflight_wall_seconds")
    return merged


def _build_row(candidate: ClipCandidate, draft: ContractDraft) -> RenderContract:
    return RenderContract(
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        transformation_selection_id=as_uuid(draft.selection_id),
        selected_plan_id=as_uuid(draft.selected_plan_id),
        final_refinement_id=as_uuid(draft.final_refinement_id),
        status=RenderContractStatus(draft.status),
        compatibility_outcome=(
            FinalClipCompatibilityOutcome(draft.compatibility_outcome)
            if draft.compatibility_outcome is not None
            else None
        ),
        is_current=True,
        contract_ready=draft.contract_ready,
        reason_codes=list(draft.reason_codes),
        compatibility_evidence=dict(draft.compatibility_evidence),
        source_media_identity=dict(draft.source_media_identity),
        source_probe=dict(draft.source_probe),
        readiness=dict(draft.readiness),
        contract_payload=dict(draft.contract_payload),
        selected_plan_fingerprint=draft.selected_plan_fingerprint,
        planning_refinement_output_fingerprint=draft.planning_refinement_output_fingerprint,
        final_refinement_output_fingerprint=draft.final_refinement_output_fingerprint,
        caption_source_fingerprint=draft.caption_source_fingerprint,
        source_media_fingerprint=draft.source_media_fingerprint,
        probe_fingerprint=draft.probe_fingerprint,
        input_fingerprint=draft.input_fingerprint,
        output_fingerprint=draft.output_fingerprint,
        profile_key=draft.profile_key,
        profile_version=draft.profile_version,
        policy_version=RENDER_CONTRACT_POLICY_VERSION,
        schema_version=RENDER_CONTRACT_SCHEMA_VERSION,
        fingerprint_version=RENDER_CONTRACT_FINGERPRINT_VERSION,
        metrics=dict(draft.metrics),
    )


# ---------------------------------------------------------------------------
# Contract assembly
# ---------------------------------------------------------------------------


def _assemble_contract(
    *,
    candidate: ClipCandidate,
    gathered: _Gathered,
    final_row: CandidateRefinement | None,
    compatibility: Any,
    bound_spans: Sequence[Any],
    facts: Any,
    media_identity: Mapping[str, object],
    caption_fingerprint: str,
    profile: Any,
    config: Stage50Config,
) -> _Assembly:
    selected_plan = gathered.selected_plan
    if selected_plan is None or final_row is None or compatibility is None:
        return _Assembly(
            payload={},
            readiness=_non_ready_readiness(BLOCKED, []),
            blocks=[],
            slots=[],
            materialization_required=False,
            extra_reasons=[],
        )

    plan_blocks = [dict(block) for block in _as_sequence(selected_plan.blocks)]
    span_by_index = {span.block_index: span for span in bound_spans}
    governance = gathered.governance_snapshot or {}

    blocks: list[ContractBlock] = []
    slots: list[MaterializationSlot] = []
    cursor = 0.0
    materialization_required = False
    extra_reasons: list[str] = []

    for block in plan_blocks:
        index = int(block.get("index", 0) or 0)
        block_type = str(block.get("block_type") or "")
        estimated = _as_float(block.get("estimated_duration"))
        slot_kind, requires_materialization, materialization_payload = _slot_for_block(block)
        source_binding: dict[str, object] | None = None
        duration = estimated
        authoritative = False
        if block_type == "SOURCE_EXCERPT":
            span = span_by_index.get(index)
            if span is not None:
                source_binding = span.as_dict()
                if (
                    span.rebind_valid
                    and span.final_clip_start is not None
                    and span.final_clip_end is not None
                ):
                    duration = max(0.0, span.final_clip_end - span.final_clip_start)
                    authoritative = True
        timeline_start = round(cursor, 3)
        cursor += max(0.0, duration)
        timeline_end = round(cursor, 3)

        if requires_materialization:
            materialization_required = True
            slots.append(
                MaterializationSlot(
                    slot_id=f"block-{index}",
                    slot_kind=slot_kind,
                    block_index=index,
                    block_type=block_type,
                    required=True,
                    reason_code=str(materialization_payload.get("reason_code") or ""),
                    payload=materialization_payload,
                )
            )

        blocks.append(
            ContractBlock(
                block_index=index,
                block_type=block_type,
                purpose=str(block.get("purpose") or ""),
                placement=str(block.get("placement") or ""),
                interrupts_source=bool(block.get("interrupts_source")),
                estimated_duration_seconds=round(estimated, 3),
                timeline_start=timeline_start,
                timeline_end=timeline_end,
                timeline_authoritative=authoritative,
                preservation_constraints=tuple(_as_str_list(block.get("preservation_constraints"))),
                dependency_ids=tuple(_as_str_list(block.get("dependency_ids"))),
                slot_kind=slot_kind,
                source_binding=source_binding,
                materialization=materialization_payload or None,
                verification=_verification_payload(block),
            )
        )

    narration = _narration_requirement(selected_plan)
    narration_slot = _narration_slot(selected_plan, narration)
    if narration_slot is not None:
        slots.append(narration_slot)
        if narration_slot.required:
            materialization_required = True
            extra_reasons.append(NARRATION_MATERIALIZATION_REQUIRED)

    readiness = _readiness(
        status=READY_FOR_RENDER_PLANNING
        if not materialization_required
        else MATERIALIZATION_REQUIRED,
        executable=True,
        materialization_required=materialization_required,
        source_media_ready=True,
        transcript_ready=True,
        compatibility_ready=True,
        verification_ready=True,
        reason_codes=extra_reasons,
    )

    hero_block_index = selected_plan.hero_block_index
    hero_span = span_by_index.get(hero_block_index)
    payload = _contract_payload(
        candidate=candidate,
        gathered=gathered,
        final_row=final_row,
        compatibility=compatibility,
        facts=facts,
        media_identity=media_identity,
        caption_fingerprint=caption_fingerprint,
        profile=profile,
        config=config,
        blocks=blocks,
        hero_span=hero_span,
        narration=narration,
        slots=slots,
        readiness=readiness,
        governance=governance,
        selected_plan=selected_plan,
    )
    return _Assembly(
        payload=payload,
        readiness=readiness,
        blocks=blocks,
        slots=slots,
        materialization_required=materialization_required,
        extra_reasons=extra_reasons,
    )


def _slot_for_block(
    block: Mapping[str, object],
) -> tuple[str, bool, dict[str, object]]:
    block_type = str(block.get("block_type") or "")
    if block_type == "SOURCE_EXCERPT":
        return ExecutionSlotKind.SOURCE_MEDIA.value, False, {}
    if block_type == "TRANSITION":
        return ExecutionSlotKind.TRANSITION.value, False, {}
    if block_type == "FACT_VERIFICATION_PLACEHOLDER":
        return ExecutionSlotKind.VERIFICATION_EVIDENCE.value, False, {}
    delivery = block.get("delivery_intent")
    if block_type == "TEXTUAL_ANNOTATION":
        return (
            ExecutionSlotKind.AUTHORED_TEXT.value,
            True,
            _authored_payload(block, AUTHORED_TEXT_MATERIALIZATION_REQUIRED),
        )
    if block_type == "ORIGINAL_VALUE" and delivery == "NARRATION":
        return (
            ExecutionSlotKind.AUTHORED_NARRATION.value,
            True,
            _authored_payload(block, NARRATION_MATERIALIZATION_REQUIRED),
        )
    return (
        ExecutionSlotKind.AUTHORED_TEXT.value,
        True,
        _authored_payload(block, AUTHORED_TEXT_MATERIALIZATION_REQUIRED),
    )


def _authored_payload(block: Mapping[str, object], reason_code: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "reason_code": reason_code,
        "purpose": str(block.get("purpose") or ""),
        "semantic_intent": block.get("semantic_intent"),
        "required_information": {
            "why_unavailable": block.get("why_unavailable"),
            "sub_reason": reason_code,
        },
        "substantive_value_kind": block.get("substantive_value_kind"),
        "delivery_intent": block.get("delivery_intent"),
        "estimated_duration_seconds": _as_float(block.get("estimated_duration")),
        "placement": str(block.get("placement") or ""),
        "verification_dependency_ids": _as_str_list(block.get("dependency_ids")),
        "grounding_refs": _as_str_list(block.get("grounding_refs")),
    }
    draft_line = block.get("draft_line")
    if isinstance(draft_line, str) and draft_line.strip():
        payload["authoring_reference"] = {
            "draft_line": draft_line,
            "draft_only": True,
            "authoritative": False,
        }
    return payload


def _verification_payload(block: Mapping[str, object]) -> dict[str, object] | None:
    if str(block.get("block_type") or "") != "FACT_VERIFICATION_PLACEHOLDER":
        return None
    return {
        "claim_id": block.get("claim_dependency"),
        "rationale": block.get("verification_rationale"),
        "intended_use": block.get("intended_use"),
        "must_verify_before_execution": bool(block.get("must_verify_before_execution")),
        "dependent_block_indexes": [
            item for item in _as_sequence(block.get("dependent_block_ids")) if isinstance(item, int)
        ],
        "resolved": True,
    }


def _narration_requirement(selected_plan: TransformationPlan) -> dict[str, object]:
    requirements = dict(selected_plan.narration_requirements or {})
    requirements["need"] = selected_plan.narration_need
    return requirements


def _narration_slot(
    selected_plan: TransformationPlan, narration: Mapping[str, object]
) -> MaterializationSlot | None:
    need = str(narration.get("need") or selected_plan.narration_need or "NONE")
    if need == "NONE":
        return None
    essential = bool(narration.get("essential"))
    required = need in {"RECOMMENDED", "REQUIRED"} or essential
    return MaterializationSlot(
        slot_id="narration",
        slot_kind=ExecutionSlotKind.AUTHORED_NARRATION.value,
        block_index=None,
        block_type="NARRATION_REQUIREMENT",
        required=required,
        reason_code=NARRATION_MATERIALIZATION_REQUIRED if required else "",
        payload=dict(narration),
    )


def _contract_payload(
    *,
    candidate: ClipCandidate,
    gathered: _Gathered,
    final_row: CandidateRefinement,
    compatibility: Any,
    facts: Any,
    media_identity: Mapping[str, object],
    caption_fingerprint: str,
    profile: Any,
    config: Stage50Config,
    blocks: Sequence[ContractBlock],
    hero_span: Any,
    narration: Mapping[str, object],
    slots: Sequence[MaterializationSlot],
    readiness: Mapping[str, object],
    governance: Mapping[str, object],
    selected_plan: TransformationPlan,
) -> dict[str, object]:
    selection = gathered.selection
    final = _final_evidence(final_row)
    facts_payload = facts.as_dict() if facts is not None else {}
    source_language = _source_language(candidate)
    caption_language = _caption_language(final_row, source_language)
    target_frame_rate = _target_frame_rate(facts.frames_per_second if facts else None, config)

    identity_section = {
        "candidate": {
            "id": str(candidate.id),
            "candidate_key": candidate.candidate_key,
            "source_id": str(candidate.source_video_id),
        },
        "selection": {
            "id": str(selection.id),
            "status": selection.status.value,
            "selected_with_caution": bool(selection.selected_with_caution),
        },
        "selected_plan": {
            "id": str(selected_plan.id),
            "plan_key": selected_plan.plan_key,
            "plan_output_fingerprint": selected_plan.plan_output_fingerprint,
        },
        "strategy": {
            "type": selected_plan.strategy_type.value,
            "intensity": selected_plan.intensity.value,
        },
        "governance": {
            "set_id": str(selection.transformation_governance_set_id)
            if selection.transformation_governance_set_id
            else None,
            "input_fingerprint": selection.governance_input_fingerprint,
            "output_fingerprint": selection.governance_output_fingerprint,
            "policy_version": selection.governor_policy_version,
            "validation_version": selection.governor_validation_version,
            "platform_policy_profile_version": selection.platform_policy_profile_version,
        },
        "planning_refinement": {
            "id": str(selection.refinement_id) if selection.refinement_id else None,
            "priority": selection.refinement_priority,
            "quality_level": selection.refinement_quality_level,
            "output_fingerprint": selection.refinement_output_fingerprint,
        },
        "final_clip_refinement": {
            "id": final.refinement_id,
            "priority": final.priority,
            "quality_level": final.quality_level,
            "status": final.status,
            "output_fingerprint": final.output_fingerprint,
        },
        "compatibility_outcome": compatibility.outcome,
        "render_profile": {"key": profile.key, "version": profile.semantic_version},
        "policy_version": RENDER_CONTRACT_POLICY_VERSION,
        "schema_version": RENDER_CONTRACT_SCHEMA_VERSION,
        "fingerprint_version": RENDER_CONTRACT_FINGERPRINT_VERSION,
        "compatibility_policy_version": COMPATIBILITY_POLICY_VERSION,
    }

    source_media = {
        "identity": dict(media_identity),
        "managed_relative_path": str(media_identity.get("relative_path") or ""),
        "content_hash": str(media_identity.get("content_hash") or ""),
        "probe": facts_payload,
        "probe_fingerprint": (
            probe_fingerprint(facts_payload, media_identity) if facts_payload else ""
        ),
    }

    output_profile = {
        "profile_key": profile.key,
        "profile_semantic_version": profile.semantic_version,
        "aspect_ratio": profile.aspect_ratio,
        "width": profile.width,
        "height": profile.height,
        "frame_rate_policy": profile.frame_rate_policy,
        "source_frame_rate": facts.frames_per_second if facts is not None else None,
        "target_frame_rate": target_frame_rate,
        "audio_output_policy": "SOURCE_AUDIO_PRESERVED",
        "safe_zone_profile": profile.safe_zone_profile,
    }

    hero = None
    if hero_span is not None:
        hero = {
            "block_index": hero_span.block_index,
            "rebound_span": {
                "start": hero_span.final_clip_start,
                "end": hero_span.final_clip_end,
                "word_start": hero_span.final_clip_word_start,
                "word_end": hero_span.final_clip_word_end,
            },
            "appearance_time": selected_plan.hero_appearance_time,
            "is_identified": True,
        }

    dimensions = governance.get("dimensions")
    retention = {
        "hook_payoff_evidence": dict(selected_plan.hook_payoff_evidence or {}),
        "derived_durations": dict(selected_plan.derived_durations or {}),
        "governance_dimensions": {
            key: dimensions.get(key)
            for key in ("retention_preservation", "source_moment_damage", "moment_density")
            if isinstance(dimensions, Mapping)
        },
        "narration_interruption_limits": {
            "max_source_interruption_seconds": narration.get(
                "max_source_interruption_seconds", 0.0
            ),
            "overlaps_source_audio": bool(narration.get("overlaps_source_audio")),
            "replaces_silence": bool(narration.get("replaces_silence")),
        },
    }

    caption_input = {
        "refinement_id": final.refinement_id,
        "priority": "FINAL_CLIP",
        "quality_level": final.quality_level,
        "status": final.status,
        "output_fingerprint": final.output_fingerprint,
        "transcript_text": final.final_transcript,
        "word_timestamps": [word.as_dict() for word in final.words],
        "language": caption_language,
        "dialect_profile": final.dialect_profile,
        "dialect_confidence": final.dialect_confidence,
        "code_switch_evidence": dict(final.code_switch_evidence),
        "entity_evidence": [dict(item) for item in final.entity_evidence],
        "protected_tokens": _protected_tokens(final.final_transcript),
        "logical_order_preserved": True,
        "caption_source_fingerprint": caption_fingerprint,
        "rendered_assets": None,
        # No subtitle rendering happens in Stage 5.0.
        "rendered": False,
    }

    framing_input = {
        "requires_visual_treatment": True,
        "requires_speaker_tracking": True,
        "face_metadata_available": False,
        "segments": [
            {
                "block_index": block.block_index,
                "tracking_required": True,
                "source_binding": block.source_binding,
            }
            for block in blocks
            if block.block_type == "SOURCE_EXCERPT"
        ],
    }

    compatibility_section = {
        "outcome": compatibility.outcome,
        "exact_match": bool(compatibility.exact_match),
        "planning_refinement": {
            "id": compatibility.planning_refinement_id,
            "priority": compatibility.planning_refinement_priority,
            "quality_level": compatibility.planning_refinement_quality_level,
            "output_fingerprint": compatibility.planning_output_fingerprint,
        },
        "final_refinement": {
            "id": compatibility.final_refinement_id,
            "output_fingerprint": compatibility.final_output_fingerprint,
        },
        "per_block_verdicts": compatibility.verdicts(),
        "unresolved_spans": [dict(span) for span in compatibility.unresolved_spans],
        "recovered_code_switch_tokens": list(compatibility.recovered_code_switch_tokens),
        "compatibility_policy_version": compatibility.compatibility_policy_version,
    }

    verification_evidence = _as_mapping(governance.get("verification"))
    verification = {
        "claim_state": verification_evidence.get("claim_state"),
        "claims": _as_sequence(verification_evidence.get("claims")),
        "unresolved": False,
        "plan_dependencies": [
            dict(item) for item in selected_plan.external_fact_dependencies or []
        ],
        "placeholder_resolutions": [
            block.verification
            for block in blocks
            if block.block_type == "FACT_VERIFICATION_PLACEHOLDER"
        ],
    }

    governance_section = {
        "status": governance.get("status"),
        "severity": governance.get("severity"),
        "dimensions": dict(_as_mapping(governance.get("dimensions"))),
        "warnings": _as_sequence(governance.get("warnings")),
        "reason_codes": _as_sequence(governance.get("reason_codes")),
        "provider_evidence": dict(_as_mapping(governance.get("provider_evidence"))),
        "remediation": _as_sequence(governance.get("remediation")),
        "platform_risk": dict(_as_mapping(governance.get("platform_risk"))),
    }

    payload: dict[str, object] = {
        "identity": identity_section,
        "source_media": source_media,
        "output_profile": output_profile,
        "blocks": [block.as_dict() for block in blocks],
        "hero": hero,
        "preservation_constraints": list(selected_plan.preservation_constraints or []),
        "retention": retention,
        "narration": dict(narration),
        "materialization": {
            "required": any(slot.required for slot in slots),
            "slots": [slot.as_dict() for slot in slots],
        },
        "caption_input": caption_input,
        "framing_input": framing_input,
        "compatibility": compatibility_section,
        "verification": verification,
        "governance": governance_section,
        "readiness": dict(readiness),
        "warnings": _as_sequence(governance.get("warnings")),
        "language_and_dialect": {
            "language": caption_language,
            "source_language": source_language,
            "dialect_profile": final.dialect_profile,
            "dialect_confidence": final.dialect_confidence,
            "code_switch_suspected": bool(final.code_switch_evidence.get("suspected")),
        },
    }
    return payload


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def _readiness(
    *,
    status: str,
    executable: bool,
    materialization_required: bool,
    source_media_ready: bool,
    transcript_ready: bool,
    compatibility_ready: bool,
    verification_ready: bool,
    reason_codes: Sequence[str],
) -> dict[str, object]:
    final_render_ready = executable and not materialization_required
    if not executable:
        downstream = "NOT_READY"
        next_action = "NONE"
    elif materialization_required:
        downstream = "READY_FOR_DOWNSTREAM_COMPOSITION"
        next_action = "STAGE_6_MATERIALIZATION_REQUIRED"
    else:
        downstream = "READY_FOR_DOWNSTREAM_COMPOSITION"
        next_action = "PROCEED_TO_STAGE_5_1"
    return {
        "preflight": "PASSED" if executable else "FAILED",
        "source_media_ready": source_media_ready,
        "transcript_ready": transcript_ready,
        "compatibility_ready": compatibility_ready,
        "verification_ready": verification_ready,
        "materialization_ready": not materialization_required,
        "final_render_ready": final_render_ready,
        "downstream_stage_eligibility": downstream,
        "next_action": next_action,
        "blocking_reason_codes": list(reason_codes),
        "status": status,
    }


def _non_ready_readiness(status: str, reason_codes: Sequence[str]) -> dict[str, object]:
    return _readiness(
        status=status,
        executable=False,
        materialization_required=False,
        source_media_ready=False,
        transcript_ready=False,
        compatibility_ready=False,
        verification_ready=False,
        reason_codes=reason_codes,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _gather(session: Session, candidate: ClipCandidate) -> _Gathered:
    handoff = build_execution_handoff(session, candidate.id) or {}
    selection_view = read_selection(session, candidate.id)
    selection = selection_view.row if selection_view is not None else None
    selected_plan: TransformationPlan | None = None
    if selection is not None and selection.selected_plan_id is not None:
        selected_plan = session.get(TransformationPlan, selection.selected_plan_id)
    planning_refinement = None
    if selection is not None and selection.refinement_id is not None:
        planning_refinement = session.get(CandidateRefinement, selection.refinement_id)
    final_refinement = _usable_final(session, candidate.id)
    source_segments: list[Mapping[str, object]] = []
    if candidate.source_video is not None and candidate.source_video.transcript is not None:
        source_segments = [
            segment
            for segment in (candidate.source_video.transcript.segments or [])
            if isinstance(segment, Mapping)
        ]
    governance_snapshot = dict(
        selection.selected_governance_snapshot or {} if selection is not None else {}
    )
    return _Gathered(
        candidate=candidate,
        handoff=handoff,
        selection=selection,
        selection_view=selection_view,
        selected_plan=selected_plan,
        planning_refinement=planning_refinement,
        final_refinement=final_refinement,
        source_segments=source_segments,
        governance_snapshot=governance_snapshot,
    )


def _usable_final(session: Session, candidate_id: uuid.UUID) -> CandidateRefinement | None:
    rows = list(
        session.scalars(
            select(CandidateRefinement).where(CandidateRefinement.clip_candidate_id == candidate_id)
        ).all()
    )
    finals = [row for row in rows if row.priority is RefinementPriority.FINAL_CLIP]
    if not finals:
        return None
    finals.sort(key=lambda row: row.updated_at, reverse=True)
    for row in finals:
        if _is_usable_final(row):
            return row
    return None


def _is_usable_final(row: CandidateRefinement) -> bool:
    return bool((row.final_transcript or "").strip()) and row.status.value in _FINAL_READY


def _final_evidence(row: CandidateRefinement) -> FinalClipEvidence:
    words: list[Any] = []
    for index, item in enumerate(row.word_timestamps or []):
        if not isinstance(item, Mapping):
            continue
        text = item.get("text", item.get("word", ""))
        start = item.get("start")
        end = item.get("end")
        if not isinstance(text, str) or not text.strip():
            continue
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            continue
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            continue
        probability = item.get("probability")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            probability = None
        words.append(
            _word(
                index=len(words),
                text=text.strip(),
                start=float(start),
                end=float(end),
                probability=float(probability) if probability is not None else None,
            )
        )
    from app.render.types import ClipWord

    typed_words = tuple(
        ClipWord(
            index=word["index"],
            text=word["text"],
            start=word["start"],
            end=word["end"],
            probability=word["probability"],
        )
        for word in words
    )
    return FinalClipEvidence(
        refinement_id=str(row.id),
        priority=row.priority.value,
        quality_level=row.quality_level,
        status=row.status.value,
        final_transcript=row.final_transcript or "",
        refined_start=row.refined_start,
        refined_end=row.refined_end,
        words=typed_words,
        unresolved_spans=tuple(
            span for span in (row.unresolved_spans or []) if isinstance(span, Mapping)
        ),
        entity_evidence=tuple(
            item for item in (row.entity_evidence or []) if isinstance(item, Mapping)
        ),
        code_switch_evidence=dict(row.code_switch_evidence or {}),
        dialect_profile=row.dialect_profile,
        dialect_confidence=float(row.dialect_confidence or 0.0),
        output_fingerprint=row.output_fingerprint or "",
    )


def _word(
    *, index: int, text: str, start: float, end: float, probability: float | None
) -> dict[str, Any]:
    return {
        "index": index,
        "text": text,
        "start": start,
        "end": end,
        "probability": probability,
    }


def _planning_identity(
    selection: Any, planning_refinement: CandidateRefinement | None
) -> dict[str, object]:
    if selection is not None:
        return {
            "id": str(selection.refinement_id) if selection.refinement_id else None,
            "priority": selection.refinement_priority,
            "quality_level": selection.refinement_quality_level,
            "output_fingerprint": selection.refinement_output_fingerprint,
        }
    return {"id": None, "priority": "", "quality_level": "", "output_fingerprint": ""}


def _exact_identity_match(selection: Any, final_row: CandidateRefinement) -> bool:
    if selection is None or selection.refinement_id is None:
        return False
    if selection.refinement_id != final_row.id:
        return False
    if selection.refinement_priority != RefinementPriority.FINAL_CLIP.value:
        return False
    fingerprint = selection.refinement_output_fingerprint or ""
    return bool(fingerprint) and fingerprint == (final_row.output_fingerprint or "")


def _required_verification_unresolved(
    selected_plan: TransformationPlan, governance: Mapping[str, object]
) -> bool:
    verification = governance.get("verification")
    claim_state = verification.get("claim_state") if isinstance(verification, Mapping) else None
    unresolved = bool(verification.get("unresolved")) if isinstance(verification, Mapping) else True
    resolved = claim_state in _RESOLVED_VERIFICATION_STATES and not unresolved
    requires = selected_plan.status.value == "PLAN_GENERATED_WITH_VERIFICATION_REQUIRED"
    dependencies = list(selected_plan.external_fact_dependencies or [])
    for dependency in dependencies:
        if isinstance(dependency, Mapping) and dependency.get("must_verify_before_execution"):
            requires = True
    essential_placeholders = [
        block
        for block in _as_sequence(selected_plan.blocks)
        if isinstance(block, Mapping)
        and block.get("block_type") == "FACT_VERIFICATION_PLACEHOLDER"
        and block.get("must_verify_before_execution")
    ]
    if essential_placeholders:
        requires = True
    if not requires:
        return False
    return not resolved


def _validate_media_bounds(spans: Sequence[Any], duration: float) -> str | None:
    for span in spans:
        start = span.final_clip_start
        end = span.final_clip_end
        if start is None or end is None:
            continue
        if start < 0 or end <= start:
            return SPAN_REVERSED_OR_NEGATIVE
        if end > duration + 0.75:
            return SPAN_OUT_OF_MEDIA_BOUNDS
    return None


def _best_effort_identity(candidate: ClipCandidate, storage: StorageService) -> dict[str, object]:
    try:
        source = candidate.source_video
        identity = source_media_identity(
            candidate.source_video_id,
            source.source_uri if source is not None else None,
            storage=storage,
        )
        return _identity_payload_with_hash(candidate, identity.as_dict())
    except (SourceMediaFailure, StorageValidationError, OSError):
        return {}


def _identity_payload_with_hash(
    candidate: ClipCandidate, payload: Mapping[str, object]
) -> dict[str, object]:
    enriched = dict(payload)
    source = candidate.source_video
    enriched["content_hash"] = str((source.content_hash if source is not None else "") or "")
    return enriched


def _probe_reuse(
    session: Session, candidate: ClipCandidate, identity_payload: Mapping[str, object]
) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
    if not identity_payload:
        return None, None
    rows = session.scalars(
        select(RenderContract)
        .where(RenderContract.source_video_id == candidate.source_video_id)
        .order_by(RenderContract.created_at.desc())
    ).all()
    for row in rows:
        stored = row.source_media_identity or {}
        if stored and dict(stored) == dict(identity_payload) and row.source_probe:
            cached_facts = (row.source_probe or {}).get("facts")
            if isinstance(cached_facts, Mapping):
                return dict(stored), dict(cached_facts)
    return None, None


def _input_fingerprint(
    *,
    candidate: ClipCandidate,
    gathered: _Gathered,
    identity_payload: Mapping[str, object],
    caption_fingerprint: str,
    final_refinement_output_fingerprint: str,
    planning_refinement_output_fingerprint: str,
    verification_state: str,
    verification_unresolved: bool,
    profile: Any,
    config: Stage50Config,
) -> str:
    selection = gathered.selection
    selected_plan = gathered.selected_plan
    final_row = gathered.final_refinement
    return render_contract_input_fingerprint(
        build_render_contract_input_payload(
            candidate_id=str(candidate.id),
            candidate_key=candidate.candidate_key,
            candidate_is_current=bool(candidate.is_current),
            disposition=candidate.disposition.value,
            analysis_fingerprint=candidate.analysis_fingerprint or "",
            source_id=str(candidate.source_video_id),
            source_media_identity=identity_payload,
            selection_id=str(selection.id) if selection is not None else None,
            selection_status=selection.status.value if selection is not None else None,
            selection_with_caution=bool(selection.selected_with_caution)
            if selection is not None
            else False,
            selection_input_fingerprint=selection.input_fingerprint
            if selection is not None
            else "",
            selection_output_fingerprint=(
                selection.output_fingerprint if selection is not None else ""
            ),
            governance_input_fingerprint=(
                selection.governance_input_fingerprint if selection is not None else ""
            ),
            governance_output_fingerprint=(
                selection.governance_output_fingerprint if selection is not None else ""
            ),
            governor_policy_version=(
                selection.governor_policy_version if selection is not None else ""
            ),
            governor_validation_version=(
                selection.governor_validation_version if selection is not None else ""
            ),
            platform_policy_profile_version=(
                selection.platform_policy_profile_version if selection is not None else ""
            ),
            selected_plan_id=str(selected_plan.id) if selected_plan is not None else None,
            selected_plan_fingerprint=(
                selected_plan.plan_output_fingerprint if selected_plan is not None else ""
            ),
            selected_plan_output_fingerprint=(
                selected_plan.plan_output_fingerprint if selected_plan is not None else ""
            ),
            selected_plan_is_current=bool(selected_plan.is_current)
            if selected_plan is not None
            else False,
            planning_refinement_id=(
                str(selection.refinement_id) if selection and selection.refinement_id else None
            ),
            planning_refinement_priority=(
                selection.refinement_priority if selection is not None else ""
            ),
            planning_refinement_quality_level=(
                selection.refinement_quality_level if selection is not None else ""
            ),
            planning_refinement_output_fingerprint=planning_refinement_output_fingerprint,
            final_refinement_id=str(final_row.id) if final_row is not None else "",
            final_refinement_status=final_row.status.value if final_row is not None else "",
            final_refinement_quality_level=(
                final_row.quality_level if final_row is not None else ""
            ),
            final_refinement_output_fingerprint=final_refinement_output_fingerprint,
            live_caption_source_fingerprint=caption_fingerprint,
            verification_state=verification_state,
            verification_unresolved=verification_unresolved,
            profile_key=profile.key,
            profile_version=profile.semantic_version,
            stage50_config=stage50_config_payload(config),
        )
    )


def _caption_source_fingerprint(final_row: CandidateRefinement) -> str:
    """Recompute the caption-source fingerprint from the live FINAL_CLIP row."""

    return caption_source_fingerprint(
        build_caption_source_payload(
            final_transcript=final_row.final_transcript or "",
            word_timestamps=final_row.word_timestamps or [],
            dialect_profile=final_row.dialect_profile,
            dialect_confidence=float(final_row.dialect_confidence or 0.0),
            code_switch_evidence=dict(final_row.code_switch_evidence or {}),
            final_refinement_output_fingerprint=final_row.output_fingerprint or "",
        )
    )


def _verification_flags(governance: Mapping[str, object]) -> tuple[str, bool]:
    """Live governance verification state used by the fingerprint and gate."""

    verification = governance.get("verification")
    if isinstance(verification, Mapping):
        claim_state = verification.get("claim_state")
        return (
            str(claim_state) if claim_state is not None else "",
            bool(verification.get("unresolved")),
        )
    return "", False


def _compatibility_evidence(compatibility: Any) -> dict[str, object]:
    if compatibility is None:
        return {}
    return {
        "outcome": compatibility.outcome,
        "exact_match": bool(compatibility.exact_match),
        "planning_refinement_id": compatibility.planning_refinement_id,
        "planning_output_fingerprint": compatibility.planning_output_fingerprint,
        "final_refinement_id": compatibility.final_refinement_id,
        "final_output_fingerprint": compatibility.final_output_fingerprint,
        "caption_source_fingerprint": compatibility.caption_source_fingerprint,
        "per_block_verdicts": compatibility.verdicts(),
        "reason_codes": list(compatibility.reason_codes),
        "recovered_code_switch_tokens": list(compatibility.recovered_code_switch_tokens),
        "compatibility_policy_version": compatibility.compatibility_policy_version,
    }


def _target_frame_rate(source_fps: float | None, config: Stage50Config) -> float:
    if source_fps is not None and source_fps > 0 and 23.976 <= source_fps <= config.max_frame_rate:
        return round(source_fps, 6)
    return 30.0


def _source_language(candidate: ClipCandidate) -> str | None:
    transcript = candidate.source_video.transcript if candidate.source_video is not None else None
    if transcript is not None and transcript.language:
        return str(transcript.language)
    return None


def _caption_language(final_row: CandidateRefinement, source_language: str | None) -> str | None:
    evidence = final_row.provider_evidence or {}
    language = evidence.get("language")
    if isinstance(language, str) and language.strip():
        return language
    return source_language


def _protected_tokens(text: str) -> list[str]:
    from app.transcription.dialect import extract_protected_tokens

    return list(extract_protected_tokens(text))


def _live_freshness(
    session: Session, candidate: ClipCandidate, row: RenderContract, settings: Settings
) -> str:
    """Recompute the full stored input fingerprint from live rows.

    Uses database reads plus the stat-only ``_best_effort_identity``: never
    ffprobe, never a provider. Any recomputation failure is UNVERIFIABLE; only an
    exact match to the persisted input fingerprint is CURRENT.
    """

    if not row.input_fingerprint:
        return "UNVERIFIABLE"
    try:
        gathered = _gather(session, candidate)
        identity = _best_effort_identity(candidate, StorageService(settings.storage_root))
        config = settings.stage50_config()
        profile = render_profile_for(config.profile_key)
        final_row = gathered.final_refinement
        final_fp = (final_row.output_fingerprint or "") if final_row is not None else ""
        planning_fp = (
            (gathered.selection.refinement_output_fingerprint or "")
            if gathered.selection is not None
            else ""
        )
        caption_fp = (
            _caption_source_fingerprint(final_row)
            if final_row is not None and _is_usable_final(final_row)
            else ""
        )
        verification_state, verification_unresolved = _verification_flags(
            gathered.governance_snapshot
        )
        recomputed = _input_fingerprint(
            candidate=candidate,
            gathered=gathered,
            identity_payload=identity,
            caption_fingerprint=caption_fp,
            final_refinement_output_fingerprint=final_fp,
            planning_refinement_output_fingerprint=planning_fp,
            verification_state=verification_state,
            verification_unresolved=verification_unresolved,
            profile=profile,
            config=config,
        )
    except Exception:
        return "UNVERIFIABLE"
    return "CURRENT" if recomputed == row.input_fingerprint else "STALE"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def _lock_candidate(session: Session, candidate_id: uuid.UUID) -> None:
    bind = session.get_bind()
    dialect = bind.dialect.name if bind is not None else ""
    statement = select(ClipCandidate.id).where(ClipCandidate.id == candidate_id)
    if dialect == "postgresql":
        statement = statement.with_for_update()
    session.execute(statement)


def _row_by_fingerprint(
    session: Session, candidate_id: uuid.UUID, input_fingerprint: str
) -> RenderContract | None:
    return session.scalars(
        select(RenderContract)
        .where(RenderContract.clip_candidate_id == candidate_id)
        .where(RenderContract.input_fingerprint == input_fingerprint)
    ).first()


__all__ = [
    "RenderContractView",
    "create_render_contract",
    "get_current_render_contract",
    "get_render_contract",
    "list_render_contracts",
    "read_render_contract",
]
