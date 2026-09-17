"""Bounded Stage 4.2 input assembly from a current Stage 4.1 plan set.

Reloads the plan set, its current plans, the selected refinement, and the Stage
4.0 analysis from persistence. Refuses stale input before any provider work.
Never accepts a client-supplied plan body, never rebuilds Stage 4.1, and never
mutates a plan.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from sqlalchemy.orm import Session

from app.core.enums import CandidateDisposition
from app.core.settings import Settings
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    TransformationEligibilityAnalysis,
    TransformationPlan,
    TransformationPlanSet,
)
from app.transformation.governance.policy import Stage42Config
from app.transformation.governance.types import GovernanceInputs, PlanEvidence
from app.transformation.inputs import resolve_effective_refinement
from app.transformation.planning.executor import build_transformation_planning_executor
from app.transformation.planning.inputs import PlanningInputError, build_planning_inputs
from app.transformation.planning.policy import Stage41Config
from app.transformation.planning.queue import list_plans

_VALID_DISPOSITIONS = {
    CandidateDisposition.CANDIDATE,
    CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
}
_READY_PLAN_SET_STATUS = {"COMPLETE", "PROVIDER_DEGRADED"}


class GovernanceInputError(ValueError):
    """A Stage 4.2 input prerequisite is missing or stale."""


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as error:
        raise GovernanceInputError("invalid identifier") from error


def _plan_evidence(row: TransformationPlan) -> PlanEvidence:
    snapshot = dict(row.strategy_snapshot or {})
    return PlanEvidence(
        plan_id=str(row.id),
        plan_key=row.plan_key,
        plan_output_fingerprint=row.plan_output_fingerprint,
        provider_input_fingerprint=row.provider_input_fingerprint,
        strategy_id=str(row.strategy_candidate_id),
        strategy_key=str(snapshot.get("strategy_key") or ""),
        strategy_type=row.strategy_type.value,
        strategy_fingerprint=row.strategy_fingerprint,
        strategy_rank=int(snapshot.get("rank") or row.generation_rank or 1),
        intensity=row.intensity.value,
        status=row.status.value,
        generation_rank=row.generation_rank,
        is_current=bool(row.is_current),
        planner_confidence=float(row.planner_confidence or 0.0),
        blocks=tuple(dict(block) for block in (row.blocks or []) if isinstance(block, Mapping)),
        hero_block_index=int(row.hero_block_index or 0),
        hero_source_start=row.hero_source_start,
        hero_source_end=row.hero_source_end,
        hero_appearance_time=float(row.hero_appearance_time or 0.0),
        narration={
            "need": row.narration_need,
            "requirements": dict(row.narration_requirements or {}),
        },
        verification_dependencies=tuple(
            dict(item)
            for item in (row.external_fact_dependencies or [])
            if isinstance(item, Mapping)
        ),
        derived_durations=dict(row.derived_durations or {}),
        hook_payoff_evidence=dict(row.hook_payoff_evidence or {}),
        original_value_kinds=tuple(str(item) for item in (row.original_value_kinds or [])),
        original_value_reasons=tuple(str(item) for item in (row.original_value_reasons or [])),
        preservation_constraints=tuple(str(item) for item in (row.preservation_constraints or [])),
        required_context=tuple(str(item) for item in (row.required_context or [])),
        degraded_rules=tuple(str(item) for item in (row.degraded_rules or [])),
        stage40_risk=dict(row.stage40_risk or {}),
        source_dialect=dict(row.source_dialect or {}),
        target_audience=dict(row.target_audience or {}),
        strategy_snapshot=snapshot,
    )


def get_plan_set_for_candidate(
    session: Session, candidate_id: uuid.UUID | str
) -> TransformationPlanSet | None:
    from sqlalchemy import select

    return session.scalar(
        select(TransformationPlanSet).where(
            TransformationPlanSet.clip_candidate_id == _as_uuid(candidate_id)
        )
    )


def validate_candidate_for_governance(
    session: Session, candidate_id: uuid.UUID | str
) -> tuple[ClipCandidate, TransformationPlanSet]:
    """Gate queueing on a current plan set with at least one current plan."""

    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        raise GovernanceInputError("candidate does not exist")
    if not candidate.is_current:
        raise GovernanceInputError("candidate is stale and cannot be governed")
    if candidate.disposition not in _VALID_DISPOSITIONS:
        raise GovernanceInputError("candidate was not retained for transformation")
    plan_set = get_plan_set_for_candidate(session, candidate.id)
    if plan_set is None:
        raise GovernanceInputError("candidate has no Stage 4.1 plan set")
    if plan_set.execution_status.value not in _READY_PLAN_SET_STATUS:
        raise GovernanceInputError("Stage 4.1 plan set is not complete")
    if plan_set.planning_outcome is None:
        raise GovernanceInputError("Stage 4.1 plan set has no semantic outcome")
    current = [row for row in list_plans(session, plan_set.id) if row.is_current]
    if not current:
        raise GovernanceInputError("Stage 4.1 plan set has no current plan")
    return candidate, plan_set


def _staleness(session: Session, plan_set: TransformationPlanSet, settings: Settings) -> bool:
    if not plan_set.input_fingerprint:
        return True
    try:
        executor = build_transformation_planning_executor(session, settings)
        current = executor.input_fingerprint(plan_set)
    except Exception:
        return True
    if not current:
        return True
    return current != plan_set.input_fingerprint


def build_governance_inputs(
    session: Session,
    candidate: ClipCandidate,
    plan_set: TransformationPlanSet,
    settings: Settings,
    config: Stage42Config,
    stage41_config: Stage41Config,
) -> GovernanceInputs:
    """Assemble bounded, output-relevant Stage 4.2 inputs or raise."""

    if _staleness(session, plan_set, settings):
        raise GovernanceInputError("STALE_STAGE41")

    analysis = session.get(TransformationEligibilityAnalysis, plan_set.transformation_analysis_id)
    if analysis is None:
        raise GovernanceInputError("Stage 4.0 analysis is missing")
    refinement: CandidateRefinement | None = resolve_effective_refinement(session, candidate)
    if refinement is None:
        raise GovernanceInputError("candidate has no usable Stage 3.5 refinement")

    try:
        planning = build_planning_inputs(
            session, candidate, analysis, refinement, settings, stage41_config
        )
    except PlanningInputError as error:
        raise GovernanceInputError(str(error)) from error

    plans = tuple(_plan_evidence(row) for row in list_plans(session, plan_set.id) if row.is_current)
    if not plans:
        raise GovernanceInputError("Stage 4.1 plan set has no current plan")

    return GovernanceInputs(
        candidate_id=str(candidate.id),
        candidate_key=candidate.candidate_key,
        source_id=str(candidate.source_video_id),
        disposition=candidate.disposition.value,
        content_type=planning.content_type.value,
        source_moment_structure=planning.source_moment_structure.value,
        transcript=planning.transcript[: config.provider_max_input_characters],
        transcript_confidence=planning.transcript_confidence,
        refined_start=planning.refined_start,
        refined_end=planning.refined_end,
        context_segments=tuple(planning.context_segments),
        dialect_profile=planning.dialect_profile,
        dialect_confidence=planning.dialect_confidence,
        code_switch=dict(planning.code_switch),
        idea_summary=planning.idea_summary,
        topic_summary=planning.topic_summary,
        hooks=tuple(planning.hooks),
        rights_risk=planning.rights_risk,
        originality_risk=planning.originality_risk,
        rights_status=planning.rights_status,
        media_origin=planning.media_origin,
        provenance_snapshot=dict(planning.provenance_snapshot),
        stage3_risk=dict(planning.stage3_risk),
        stage40_assessments=dict(planning.stage40_assessments),
        stage40_platform_risk=dict(planning.stage40_platform_risk),
        stage40_analysis_id=planning.stage40_analysis_id,
        stage40_input_fingerprint=planning.stage40_input_fingerprint,
        stage40_output_fingerprint=planning.stage40_output_fingerprint,
        stage40_policy_version=planning.stage40_policy_version,
        plan_set_id=str(plan_set.id),
        plan_set_input_fingerprint=plan_set.input_fingerprint,
        plan_set_output_fingerprint=plan_set.output_fingerprint,
        plan_set_semantic_outcome=(
            plan_set.planning_outcome.value if plan_set.planning_outcome else ""
        ),
        plan_set_provider_identity=dict(plan_set.provider_identity or {}),
        plan_set_stage40_snapshot=dict(plan_set.stage40_snapshot or {}),
        refinement_id=str(refinement.id),
        refinement_priority=refinement.priority.value,
        refinement_quality_level=refinement.quality_level,
        refinement_status=refinement.status.value,
        refinement_output_fingerprint=refinement.output_fingerprint or "",
        target_context=planning.planning_context,
        plans=plans,
    )
