"""Deterministic, synchronous, provider-free Stage 4.3 selection service.

Artificial-intelligence-free by design: no Celery task, no ``ProcessingJob``, no
queue/executor, no ``PipelineStage``/``PipelineRun``, no cancellation/worker
fencing, and no Gemini/Qwen/Whisper/FFmpeg/network path. The work is bounded
arbitration over at most three persisted Stage 4.1 plans governed by Stage 4.2.

Concurrent POST requests converge on one authoritative current outcome through
candidate-row locking, a partial unique index on ``is_current``, savepoint
uniqueness recovery, and post-lock freshness revalidation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import TransformationSelectionStatus
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    TransformationGovernanceSet,
    TransformationPlanSelection,
)
from app.transformation.governance.handoff import (
    FRESHNESS_NOT_CURRENT,
    FRESHNESS_STALE,
    FRESHNESS_UNVERIFIABLE,
    build_stage4_3_handoff,
)
from app.transformation.planning.queue import get_plan_set_for_candidate, list_plans
from app.transformation.selection import policy
from app.transformation.selection.fingerprints import (
    build_selection_input_payload,
    build_selection_output_payload,
    selection_input_fingerprint,
    selection_output_fingerprint,
)
from app.transformation.selection.policy import (
    CAUTION_STATUS,
    CLEAN_STATUS,
    COMPARISON_DIMENSIONS,
    DEFAULT_POLICY_SUMMARY,
)
from app.transformation.selection.types import (
    TIER_CAUTION,
    TIER_CLEAN,
    TIER_EXCLUDED,
    PlanAlternative,
    SelectionDecision,
)


@dataclass(frozen=True)
class SelectionView:
    """A durable selection row plus live effectiveness information."""

    row: TransformationPlanSelection
    live_freshness: str
    effective: bool


def as_uuid(value: object) -> uuid.UUID | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


def get_current_selection(
    session: Session, candidate_id: uuid.UUID | str
) -> TransformationPlanSelection | None:
    candidate_uuid = as_uuid(candidate_id)
    if candidate_uuid is None:
        return None
    row: TransformationPlanSelection | None = session.scalars(
        select(TransformationPlanSelection)
        .where(TransformationPlanSelection.clip_candidate_id == candidate_uuid)
        .where(TransformationPlanSelection.is_current.is_(True))
        .order_by(TransformationPlanSelection.created_at.desc())
    ).first()
    return row


def get_selection(
    session: Session, selection_id: uuid.UUID | str
) -> TransformationPlanSelection | None:
    selection_uuid = as_uuid(selection_id)
    if selection_uuid is None:
        return None
    row: TransformationPlanSelection | None = session.get(
        TransformationPlanSelection, selection_uuid
    )
    return row


def list_selections(
    session: Session, candidate_id: uuid.UUID | str
) -> list[TransformationPlanSelection]:
    candidate_uuid = as_uuid(candidate_id)
    if candidate_uuid is None:
        return []
    return list(
        session.scalars(
            select(TransformationPlanSelection)
            .where(TransformationPlanSelection.clip_candidate_id == candidate_uuid)
            .order_by(TransformationPlanSelection.created_at.desc())
        ).all()
    )


def read_selection(session: Session, candidate_id: uuid.UUID | str) -> SelectionView | None:
    """Read-only: recompute effective freshness without mutating any row."""

    candidate = session.get(ClipCandidate, as_uuid(candidate_id))
    if candidate is None:
        return None
    row = get_current_selection(session, candidate.id)
    if row is None:
        return None
    fingerprint, freshness = _current_input_fingerprint(session, candidate)
    return SelectionView(
        row=row,
        live_freshness=freshness,
        effective=row.input_fingerprint == fingerprint,
    )


def select_transformation_plan(
    session: Session, candidate_id: uuid.UUID | str
) -> SelectionView | None:
    """Synchronously select or reuse one authoritative current outcome."""

    candidate = session.get(ClipCandidate, as_uuid(candidate_id))
    if candidate is None:
        return None
    _lock_candidate(session, candidate.id)
    handoff, decision = _build_decision(session, candidate)
    row = _persist(session, candidate, handoff, decision)
    session.flush()
    session.refresh(row)
    return SelectionView(row=row, live_freshness=decision.freshness, effective=True)


def evaluate_without_persisting(
    session: Session, candidate_id: uuid.UUID | str
) -> tuple[SelectionDecision, str] | None:
    """Pure evaluation used by tests and validation; never writes."""

    candidate = session.get(ClipCandidate, as_uuid(candidate_id))
    if candidate is None:
        return None
    handoff, decision = _build_decision(session, candidate)
    return decision, decision.freshness


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _build_decision(
    session: Session, candidate: ClipCandidate
) -> tuple[dict[str, Any], SelectionDecision]:
    handoff = build_stage4_3_handoff(session, candidate.id) or {}
    plan_set = get_plan_set_for_candidate(session, candidate.id)
    plan_rows = [row for row in (list_plans(session, plan_set.id) if plan_set else [])]
    plan_rows = [row for row in plan_rows if row.is_current]
    governance_set = _governance_set(session, handoff)
    refinement = _refinement(session, handoff)

    input_fp = selection_input_fingerprint(
        build_selection_input_payload(
            candidate_id=str(candidate.id),
            candidate_key=candidate.candidate_key,
            source_id=str(candidate.source_video_id),
            disposition=candidate.disposition.value,
            is_current=bool(candidate.is_current),
            analysis_fingerprint=candidate.analysis_fingerprint or "",
            handoff=handoff,
            plan_rows=plan_rows,
            planning_refinement_output_fingerprint=(
                (refinement.output_fingerprint or "") if refinement is not None else ""
            ),
        )
    )
    decision = _evaluate(
        candidate=candidate,
        handoff=handoff,
        plan_rows=plan_rows,
        governance_set=governance_set,
        refinement=refinement,
        input_fp=input_fp,
    )
    return handoff, decision


def _current_input_fingerprint(session: Session, candidate: ClipCandidate) -> tuple[str, str]:
    handoff = build_stage4_3_handoff(session, candidate.id) or {}
    plan_set = get_plan_set_for_candidate(session, candidate.id)
    plan_rows = [row for row in (list_plans(session, plan_set.id) if plan_set else [])]
    plan_rows = [row for row in plan_rows if row.is_current]
    refinement = _refinement(session, handoff)
    payload = build_selection_input_payload(
        candidate_id=str(candidate.id),
        candidate_key=candidate.candidate_key,
        source_id=str(candidate.source_video_id),
        disposition=candidate.disposition.value,
        is_current=bool(candidate.is_current),
        analysis_fingerprint=candidate.analysis_fingerprint or "",
        handoff=handoff,
        plan_rows=plan_rows,
        planning_refinement_output_fingerprint=(
            (refinement.output_fingerprint or "") if refinement is not None else ""
        ),
    )
    governance_set = handoff.get("governance_set")
    freshness = (
        str(governance_set.get("freshness"))
        if isinstance(governance_set, dict)
        else FRESHNESS_NOT_CURRENT
    )
    return selection_input_fingerprint(payload), freshness


def _evaluate(
    *,
    candidate: ClipCandidate,
    handoff: dict[str, Any],
    plan_rows: list[Any],
    governance_set: TransformationGovernanceSet | None,
    refinement: CandidateRefinement | None,
    input_fp: str,
) -> SelectionDecision:
    gov = handoff.get("governance_set")
    freshness = _freshness(gov)
    identity = _identity(handoff, governance_set, refinement)
    rows_by_id = {str(row.id): row for row in plan_rows}

    if gov is None:
        return _no_selection_decision(
            status=TransformationSelectionStatus.SELECTION_DEFERRED,
            reason=policy.GOVERNANCE_NOT_AVAILABLE,
            freshness=freshness,
            identity=identity,
            input_fp=input_fp,
            alternatives=(),
        )
    if freshness == FRESHNESS_STALE:
        return _no_selection_decision(
            status=TransformationSelectionStatus.STALE_SELECTION_INPUT,
            reason=policy.GOVERNANCE_STALE,
            freshness=freshness,
            identity=identity,
            input_fp=input_fp,
            alternatives=_build_alternatives(handoff, rows_by_id),
        )
    if freshness == FRESHNESS_NOT_CURRENT:
        return _no_selection_decision(
            status=TransformationSelectionStatus.SELECTION_DEFERRED,
            reason=policy.GOVERNANCE_NOT_CURRENT,
            freshness=freshness,
            identity=identity,
            input_fp=input_fp,
            alternatives=_build_alternatives(handoff, rows_by_id),
        )
    if freshness == FRESHNESS_UNVERIFIABLE:
        return _no_selection_decision(
            status=TransformationSelectionStatus.SELECTION_DEFERRED,
            reason=policy.GOVERNANCE_UNVERIFIABLE,
            freshness=freshness,
            identity=identity,
            input_fp=input_fp,
            alternatives=_build_alternatives(handoff, rows_by_id),
        )

    alternatives = _build_alternatives(handoff, rows_by_id)
    if any(alt.governance_result_id is None for alt in alternatives):
        return _no_selection_decision(
            status=TransformationSelectionStatus.SELECTION_DEFERRED,
            reason=policy.SEMANTIC_GOVERNANCE_UNFINISHED,
            freshness=freshness,
            identity=identity,
            input_fp=input_fp,
            alternatives=alternatives,
        )
    for alt in alternatives:
        inconsistent, _codes = policy.governance_evidence_inconsistent(
            alt.result_view, alt.plan_output_fingerprint
        )
        if inconsistent:
            return _no_selection_decision(
                status=TransformationSelectionStatus.SELECTION_DEFERRED,
                reason=policy.INCONSISTENT_GOVERNANCE_EVIDENCE,
                freshness=freshness,
                identity=identity,
                input_fp=input_fp,
                alternatives=alternatives,
            )

    classified = _classify(alternatives)
    clean = [alt for alt in classified if alt.inclusion == TIER_CLEAN]
    caution = [alt for alt in classified if alt.inclusion == TIER_CAUTION]

    if clean:
        return _arbitrate(
            tier=TIER_CLEAN,
            pool=clean,
            classified=classified,
            freshness=freshness,
            identity=identity,
            input_fp=input_fp,
        )
    if caution:
        return _arbitrate(
            tier=TIER_CAUTION,
            pool=caution,
            classified=classified,
            freshness=freshness,
            identity=identity,
            input_fp=input_fp,
        )

    if any(alt.status == policy.DEFERRED_STATUS for alt in classified):
        return _no_selection_decision(
            status=TransformationSelectionStatus.SELECTION_DEFERRED,
            reason=policy.SEMANTIC_GOVERNANCE_UNFINISHED,
            freshness=freshness,
            identity=identity,
            input_fp=input_fp,
            alternatives=classified,
        )
    return _no_selection_decision(
        status=TransformationSelectionStatus.NO_SELECTABLE_PLAN,
        reason=policy.NO_SELECTABLE_PLAN,
        freshness=freshness,
        identity=identity,
        input_fp=input_fp,
        alternatives=classified,
    )


def _build_alternatives(
    handoff: dict[str, Any], rows_by_id: dict[str, Any]
) -> list[PlanAlternative]:
    alternatives: list[PlanAlternative] = []
    plans = handoff.get("plans")
    if not isinstance(plans, list):
        return alternatives
    for entry in plans:
        if not isinstance(entry, dict):
            continue
        plan_id = str(entry.get("plan_id") or "")
        if not plan_id:
            continue
        row = rows_by_id.get(plan_id)
        governance = entry.get("governance")
        plan_fingerprint = (
            str(row.plan_output_fingerprint)
            if row is not None
            else str(governance.get("plan_output_fingerprint") if governance else "")
        )
        if not isinstance(governance, dict):
            alternatives.append(
                PlanAlternative(
                    plan_id=plan_id,
                    plan_output_fingerprint=plan_fingerprint,
                    governance_result_id=None,
                    governance_output_fingerprint="",
                    status="MISSING",
                    eligible=False,
                    generation_rank=int(getattr(row, "generation_rank", 0) or 0),
                    planner_confidence=float(getattr(row, "planner_confidence", 0.0) or 0.0),
                    strategy_type=str(entry.get("strategy", {}).get("type", "")),
                    intensity=str(entry.get("intensity") or ""),
                    hard_gates=(),
                    dimensions={},
                    verification={},
                    platform_risk={},
                    warnings=(),
                    reason_codes=(),
                    remediation=(),
                    provider_evidence={},
                    governance_input_fingerprint="",
                    inclusion=TIER_EXCLUDED,
                    exclusion_codes=(policy.MISSING_GOVERNANCE_RESULT,),
                )
            )
            continue
        alternatives.append(
            PlanAlternative(
                plan_id=plan_id,
                plan_output_fingerprint=plan_fingerprint,
                governance_result_id=str(governance.get("result_id") or "") or None,
                governance_output_fingerprint=str(governance.get("output_fingerprint") or ""),
                status=str(governance.get("status") or ""),
                eligible=bool(governance.get("eligible_for_stage4_3")),
                generation_rank=int(getattr(row, "generation_rank", 0) or 0),
                planner_confidence=float(getattr(row, "planner_confidence", 0.0) or 0.0),
                strategy_type=str(entry.get("strategy", {}).get("type", "")),
                intensity=str(entry.get("intensity") or ""),
                hard_gates=tuple(governance.get("hard_gates") or ()),
                dimensions=dict(governance.get("dimensions") or {}),
                verification=dict(governance.get("verification") or {}),
                platform_risk=dict(governance.get("platform_risk") or {}),
                warnings=tuple(governance.get("warnings") or ()),
                reason_codes=tuple(str(code) for code in (governance.get("reason_codes") or ())),
                remediation=tuple(governance.get("remediation") or ()),
                provider_evidence=dict(governance.get("provider_evidence") or {}),
                governance_input_fingerprint=str(governance.get("input_fingerprint") or ""),
                snapshot=dict(governance),
            )
        )
    return alternatives


def _classify(alternatives: list[PlanAlternative]) -> list[PlanAlternative]:
    classified: list[PlanAlternative] = []
    for alt in alternatives:
        if alt.status == CLEAN_STATUS:
            ok, codes = policy.clean_selectability(alt.result_view, alt.plan_output_fingerprint)
            if ok:
                classified.append(_with_inclusion(alt, TIER_CLEAN, ()))
            else:
                classified.append(_with_inclusion(alt, TIER_EXCLUDED, codes))
        elif alt.status == CAUTION_STATUS:
            ok, codes = policy.caution_selectability(alt.result_view, alt.plan_output_fingerprint)
            if ok:
                classified.append(_with_inclusion(alt, TIER_CAUTION, ()))
            else:
                classified.append(_with_inclusion(alt, TIER_EXCLUDED, codes))
        else:
            classified.append(_with_inclusion(alt, TIER_EXCLUDED, (policy.STATUS_NOT_APPROVED,)))
    return classified


def _with_inclusion(alt: PlanAlternative, tier: str, codes: tuple[str, ...]) -> PlanAlternative:
    return PlanAlternative(
        plan_id=alt.plan_id,
        plan_output_fingerprint=alt.plan_output_fingerprint,
        governance_result_id=alt.governance_result_id,
        governance_output_fingerprint=alt.governance_output_fingerprint,
        status=alt.status,
        eligible=alt.eligible,
        generation_rank=alt.generation_rank,
        planner_confidence=alt.planner_confidence,
        strategy_type=alt.strategy_type,
        intensity=alt.intensity,
        hard_gates=alt.hard_gates,
        dimensions=alt.dimensions,
        verification=alt.verification,
        platform_risk=alt.platform_risk,
        warnings=alt.warnings,
        reason_codes=alt.reason_codes,
        remediation=alt.remediation,
        provider_evidence=alt.provider_evidence,
        governance_input_fingerprint=alt.governance_input_fingerprint,
        inclusion=tier,
        exclusion_codes=codes,
        snapshot=alt.snapshot,
    )


def _arbitrate(
    *,
    tier: str,
    pool: list[PlanAlternative],
    classified: list[PlanAlternative],
    freshness: str,
    identity: dict[str, object],
    input_fp: str,
) -> SelectionDecision:
    def vector(alt: PlanAlternative) -> tuple[object, ...]:
        return policy.comparison_vector(
            alt.result_view,
            plan_fingerprint=alt.plan_output_fingerprint,
            generation_rank=alt.generation_rank,
            planner_confidence=alt.planner_confidence,
        )

    ordered = sorted(pool, key=vector)
    winner = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    winner_vector = vector(winner)
    distinction = None
    tie_break_used = False
    if runner_up is not None:
        distinction = policy.first_material_distinction(
            winner.result_view,
            runner_up.result_view,
            left_vector=winner_vector,
            right_vector=vector(runner_up),
        )
        tie_break_used = bool(
            distinction and distinction.get("dimension") == COMPARISON_DIMENSIONS[-1]
        )

    if runner_up is None:
        competition_rule = "SINGLE_ELIGIBLE_PLAN"
    elif tie_break_used:
        competition_rule = "STABLE_PLAN_ID"
    else:
        competition_rule = "LEXICOGRAPHIC_DIMENSION"

    dispositions: list[dict[str, object]] = []
    loser_ids = {alt.plan_id for alt in ordered[1:]}
    for alt in classified:
        if alt.plan_id == winner.plan_id:
            dispositions.append(
                {
                    "plan_id": alt.plan_id,
                    "governance_result_id": alt.governance_result_id,
                    "status": alt.status,
                    "eligible_for_stage4_3": alt.eligible,
                    "inclusion": TIER_CLEAN if tier == TIER_CLEAN else TIER_CAUTION,
                    "exclusion_reason_codes": [],
                    "selected": True,
                }
            )
        elif alt.plan_id in loser_ids:
            dispositions.append(
                {
                    "plan_id": alt.plan_id,
                    "governance_result_id": alt.governance_result_id,
                    "status": alt.status,
                    "eligible_for_stage4_3": alt.eligible,
                    "inclusion": tier,
                    "exclusion_reason_codes": [policy.LOST_IN_ARBITRATION],
                    "selected": False,
                }
            )
        else:
            codes = list(alt.exclusion_codes)
            if not codes and alt.inclusion != tier:
                codes = [policy.NOT_IN_ACTIVE_TIER]
            dispositions.append(
                {
                    "plan_id": alt.plan_id,
                    "governance_result_id": alt.governance_result_id,
                    "status": alt.status,
                    "eligible_for_stage4_3": alt.eligible,
                    "inclusion": alt.inclusion,
                    "exclusion_reason_codes": codes,
                    "selected": False,
                }
            )

    eligible_ids = [
        alt.plan_id for alt in classified if alt.inclusion in {TIER_CLEAN, TIER_CAUTION}
    ]
    excluded_ids = [alt.plan_id for alt in classified if alt.inclusion == TIER_EXCLUDED]
    arbitration = {
        "approval_tier": tier,
        "eligible_plan_ids": eligible_ids,
        "excluded_plan_ids": excluded_ids,
        "comparison_dimensions": list(COMPARISON_DIMENSIONS),
        "winner_plan_id": winner.plan_id,
        "winner_comparison_vector": list(winner_vector),
        "first_material_distinction": distinction,
        "tie_break_used": tie_break_used,
        "competition_rule": competition_rule,
        "alternative_dispositions": dispositions,
    }
    snapshot = _governance_snapshot(winner, identity)
    reason = policy.SELECTED_CLEAN if tier == TIER_CLEAN else policy.SELECTED_CAUTION
    status = (
        TransformationSelectionStatus.PLAN_SELECTED
        if tier == TIER_CLEAN
        else TransformationSelectionStatus.PLAN_SELECTED_WITH_CAUTION
    )
    output_fp = selection_output_fingerprint(
        build_selection_output_payload(
            status=status.value,
            selected_plan_id=winner.plan_id,
            selected_governance_result_id=winner.governance_result_id,
            selected_with_caution=tier == TIER_CAUTION,
            selected_plan_fingerprint=winner.plan_output_fingerprint,
            selection_reason_codes=(reason,),
            selected_governance_snapshot=snapshot,
            arbitration_evidence=arbitration,
            alternative_dispositions=dispositions,
        )
    )
    return SelectionDecision(
        status=status,
        reason_codes=(reason,),
        selected_plan_id=winner.plan_id,
        selected_governance_result_id=winner.governance_result_id,
        selected_with_caution=tier == TIER_CAUTION,
        selected_plan_fingerprint=winner.plan_output_fingerprint,
        arbitration_evidence=arbitration,
        selected_governance_snapshot=snapshot,
        alternative_dispositions=tuple(dispositions),
        governance_input_fingerprint=str(identity["governance_input_fingerprint"]),
        governance_output_fingerprint=str(identity["governance_output_fingerprint"]),
        governor_policy_version=str(identity["governor_policy_version"]),
        governor_validation_version=str(identity["governor_validation_version"]),
        platform_policy_profile_version=str(identity["platform_policy_profile_version"]),
        transformation_analysis_id=_opt_str(identity["transformation_analysis_id"]),
        transformation_plan_set_id=_opt_str(identity["transformation_plan_set_id"]),
        transformation_governance_set_id=_opt_str(identity["transformation_governance_set_id"]),
        refinement_id=_opt_str(identity["refinement_id"]),
        refinement_priority=str(identity["refinement_priority"]),
        refinement_quality_level=str(identity["refinement_quality_level"]),
        refinement_output_fingerprint=str(identity["refinement_output_fingerprint"]),
        freshness=freshness,
        input_fingerprint=input_fp,
        output_fingerprint=output_fp,
    )


def _no_selection_decision(
    *,
    status: TransformationSelectionStatus,
    reason: str,
    freshness: str,
    identity: dict[str, object],
    input_fp: str,
    alternatives: tuple[PlanAlternative, ...] | list[PlanAlternative],
) -> SelectionDecision:
    dispositions = tuple(alt.disposition() for alt in alternatives)
    arbitration: dict[str, object] = {
        "approval_tier": "NONE",
        "eligible_plan_ids": [],
        "excluded_plan_ids": [alt.plan_id for alt in alternatives],
        "comparison_dimensions": list(COMPARISON_DIMENSIONS),
        "winner_plan_id": None,
        "first_material_distinction": None,
        "tie_break_used": False,
        "competition_rule": "NO_ACTIVE_POOL",
        "alternative_dispositions": [dict(item) for item in dispositions],
        "policy_summary": dict(DEFAULT_POLICY_SUMMARY),
    }
    output_fp = selection_output_fingerprint(
        build_selection_output_payload(
            status=status.value,
            selected_plan_id=None,
            selected_governance_result_id=None,
            selected_with_caution=False,
            selected_plan_fingerprint="",
            selection_reason_codes=(reason,),
            selected_governance_snapshot={},
            arbitration_evidence=arbitration,
            alternative_dispositions=dispositions,
        )
    )
    return SelectionDecision(
        status=status,
        reason_codes=(reason,),
        selected_plan_id=None,
        selected_governance_result_id=None,
        selected_with_caution=False,
        selected_plan_fingerprint="",
        arbitration_evidence=arbitration,
        selected_governance_snapshot={},
        alternative_dispositions=dispositions,
        governance_input_fingerprint=str(identity["governance_input_fingerprint"]),
        governance_output_fingerprint=str(identity["governance_output_fingerprint"]),
        governor_policy_version=str(identity["governor_policy_version"]),
        governor_validation_version=str(identity["governor_validation_version"]),
        platform_policy_profile_version=str(identity["platform_policy_profile_version"]),
        transformation_analysis_id=_opt_str(identity["transformation_analysis_id"]),
        transformation_plan_set_id=_opt_str(identity["transformation_plan_set_id"]),
        transformation_governance_set_id=_opt_str(identity["transformation_governance_set_id"]),
        refinement_id=_opt_str(identity["refinement_id"]),
        refinement_priority=str(identity["refinement_priority"]),
        refinement_quality_level=str(identity["refinement_quality_level"]),
        refinement_output_fingerprint=str(identity["refinement_output_fingerprint"]),
        freshness=freshness,
        input_fingerprint=input_fp,
        output_fingerprint=output_fp,
    )


def _governance_snapshot(winner: PlanAlternative, identity: dict[str, object]) -> dict[str, object]:
    snapshot = dict(winner.snapshot)
    snapshot["governor_policy_version"] = identity["governor_policy_version"]
    snapshot["governor_validation_version"] = identity["governor_validation_version"]
    snapshot["platform_policy_profile_version"] = identity["platform_policy_profile_version"]
    snapshot["governor_schema_version"] = identity["governor_schema_version"]
    return snapshot


# ---------------------------------------------------------------------------
# Identity / freshness
# ---------------------------------------------------------------------------


def _identity(
    handoff: dict[str, Any],
    governance_set: TransformationGovernanceSet | None,
    refinement: CandidateRefinement | None,
) -> dict[str, object]:
    stage40 = handoff.get("stage40") or {}
    stage41 = handoff.get("stage41") or {}
    selected_refinement = handoff.get("selected_refinement") or {}
    gov = handoff.get("governance_set") or {}
    return {
        "transformation_analysis_id": stage40.get("analysis_id"),
        "transformation_plan_set_id": stage41.get("plan_set_id"),
        "transformation_governance_set_id": gov.get("id"),
        "refinement_id": selected_refinement.get("id"),
        "refinement_priority": selected_refinement.get("priority") or "",
        "refinement_quality_level": selected_refinement.get("quality_level") or "",
        "refinement_output_fingerprint": (
            refinement.output_fingerprint if refinement is not None else ""
        ),
        "governance_input_fingerprint": gov.get("input_fingerprint") or "",
        "governance_output_fingerprint": gov.get("output_fingerprint") or "",
        "governor_policy_version": (
            governance_set.policy_version if governance_set is not None else ""
        ),
        "governor_schema_version": (
            governance_set.schema_version if governance_set is not None else ""
        ),
        "governor_validation_version": (
            governance_set.validation_version if governance_set is not None else ""
        ),
        "platform_policy_profile_version": (
            governance_set.platform_policy_profile_version
            if governance_set is not None
            else str(gov.get("platform_policy_profile_version") or "")
        ),
    }


def _governance_set(
    session: Session, handoff: dict[str, Any]
) -> TransformationGovernanceSet | None:
    gov = handoff.get("governance_set")
    if not isinstance(gov, dict):
        return None
    set_id = as_uuid(gov.get("id"))
    if set_id is None:
        return None
    row: TransformationGovernanceSet | None = session.get(TransformationGovernanceSet, set_id)
    return row


def _refinement(session: Session, handoff: dict[str, Any]) -> CandidateRefinement | None:
    selected = handoff.get("selected_refinement")
    if not isinstance(selected, dict):
        return None
    refinement_id = as_uuid(selected.get("id"))
    if refinement_id is None:
        return None
    row: CandidateRefinement | None = session.get(CandidateRefinement, refinement_id)
    return row


def _freshness(gov: object) -> str:
    if not isinstance(gov, dict):
        return FRESHNESS_NOT_CURRENT
    return str(gov.get("freshness") or FRESHNESS_NOT_CURRENT)


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Persistence / concurrency
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
) -> TransformationPlanSelection | None:
    row: TransformationPlanSelection | None = session.scalars(
        select(TransformationPlanSelection)
        .where(TransformationPlanSelection.clip_candidate_id == candidate_id)
        .where(TransformationPlanSelection.input_fingerprint == input_fingerprint)
    ).first()
    return row


def _current_row(session: Session, candidate_id: uuid.UUID) -> TransformationPlanSelection | None:
    return get_current_selection(session, candidate_id)


def _persist(
    session: Session,
    candidate: ClipCandidate,
    handoff: dict[str, Any],
    decision: SelectionDecision,
) -> TransformationPlanSelection:
    input_fp = decision.input_fingerprint
    current = _current_row(session, candidate.id)
    if current is not None and current.input_fingerprint == input_fp:
        return current
    existing = _row_by_fingerprint(session, candidate.id, input_fp)
    if current is not None:
        current.is_current = False
        session.flush()
    if existing is not None:
        existing.is_current = True
        session.flush()
        return existing
    row = _build_row(candidate, decision)
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


def _build_row(
    candidate: ClipCandidate, decision: SelectionDecision
) -> TransformationPlanSelection:
    return TransformationPlanSelection(
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        transformation_analysis_id=as_uuid(decision.transformation_analysis_id),
        transformation_plan_set_id=as_uuid(decision.transformation_plan_set_id),
        transformation_governance_set_id=as_uuid(decision.transformation_governance_set_id),
        selected_plan_id=as_uuid(decision.selected_plan_id),
        selected_governance_result_id=as_uuid(decision.selected_governance_result_id),
        refinement_id=as_uuid(decision.refinement_id),
        refinement_priority=decision.refinement_priority,
        refinement_quality_level=decision.refinement_quality_level,
        refinement_output_fingerprint=decision.refinement_output_fingerprint,
        status=decision.status,
        is_current=True,
        selected_with_caution=decision.selected_with_caution,
        selection_reason_codes=list(decision.reason_codes),
        arbitration_evidence=dict(decision.arbitration_evidence),
        selected_governance_snapshot=dict(decision.selected_governance_snapshot),
        alternative_dispositions=[dict(item) for item in decision.alternative_dispositions],
        governance_input_fingerprint=decision.governance_input_fingerprint,
        governance_output_fingerprint=decision.governance_output_fingerprint,
        governor_policy_version=decision.governor_policy_version,
        governor_validation_version=decision.governor_validation_version,
        platform_policy_profile_version=decision.platform_policy_profile_version,
        selected_plan_fingerprint=decision.selected_plan_fingerprint,
        input_fingerprint=decision.input_fingerprint,
        output_fingerprint=decision.output_fingerprint,
        policy_version=policy.SELECTION_POLICY_VERSION,
        schema_version=policy.SELECTION_SCHEMA_VERSION,
        fingerprint_version=policy.SELECTION_FINGERPRINT_VERSION,
        metrics={"selection_runs": 1},
    )


__all__ = [
    "SelectionView",
    "evaluate_without_persisting",
    "get_current_selection",
    "get_selection",
    "list_selections",
    "read_selection",
    "select_transformation_plan",
]
