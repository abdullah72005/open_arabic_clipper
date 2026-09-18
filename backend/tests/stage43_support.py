"""Shared Stage 4.3 test helpers: persisted governed fixtures.

Builds a real Stage 4.0 analysis and Stage 4.1 plan set, then persists a
controlled, fingerprint-current Stage 4.2 governance set so Stage 4.3 can be
tested deterministically without fighting the governor. It never touches the
network and never calls a provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stage41_support import (
    FakePlanningProvider,
    run_planning,
    seed_stage41,
)
from stage41_support import (
    make_source_value_plan as _make_stage41_plan,
)
from stage42_support import (
    FakeGovernanceSettings,
    install_stage42_settings,
    make_source_value_plan,
    with_review_narration,
)

from app.core.enums import (
    ExternalFactRequirement,
    GovernanceExecutionStatus,
    GovernancePlanStatus,
    GovernanceSemanticOutcome,
    RefinementPriority,
    RefinementStatus,
    SemanticProviderMode,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    TransformationEligibilityAnalysis,
    TransformationGovernanceResult,
    TransformationGovernanceSet,
    TransformationPlan,
    TransformationPlanSet,
)
from app.transformation.governance.executor import (
    build_transformation_governance_executor,
)
from app.transformation.governance.policy import (
    GOVERNOR_POLICY_VERSION,
    GOVERNOR_SCHEMA_VERSION,
    GOVERNOR_VALIDATION_VERSION,
    PLATFORM_POLICY_CHECKED_AT,
    PLATFORM_POLICY_PROFILE_VERSION,
)
from app.transformation.planning.queue import list_plans

_ELIGIBLE_STATUSES = {
    GovernancePlanStatus.APPROVED_FOR_SELECTION.value,
    GovernancePlanStatus.APPROVED_WITH_CAUTION.value,
}


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_list_str(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


DEFAULT_DIMENSIONS: dict[str, str] = {
    "retention_preservation": "STRONG",
    "source_moment_damage": "LOW",
    "substantive_originality": "STRONG",
    "source_dominance": "LOW",
    "semantic_fidelity": "STRONG",
    "generic_filler_risk": "LOW",
    "redundant_commentary_risk": "LOW",
    "narration_burden": "APPROPRIATE",
    "verification_completeness": "GROUNDED_IN_SOURCE",
    "template_mass_produced_feel": "LOW",
    "substantive_transformation": "LOW",
    "plan_coherence": "STRONG",
    "transformation_proportionality": "LOW",
}

DEFAULT_VERIFICATION: dict[str, object] = {
    "claim_state": "GROUNDED_IN_SOURCE",
    "claims": [],
    "unresolved": False,
    "reason_codes": [],
}


@dataclass
class SelectionFixture:
    source: Any
    candidate: ClipCandidate
    refinement: CandidateRefinement
    analysis: TransformationEligibilityAnalysis
    plan_set: TransformationPlanSet
    plans: list[TransformationPlan]
    governance_set: TransformationGovernanceSet
    settings: FakeGovernanceSettings


def install_selection_settings(monkeypatch: Any, settings: FakeGovernanceSettings) -> None:
    install_stage42_settings(monkeypatch, settings)


def platform_risk(
    *,
    youtube_reused: str = "LOW",
    youtube_inauthentic: str = "LOW",
    youtube_spam: str = "LOW",
    facebook_unoriginal: str = "LOW",
    facebook_spam: str = "LOW",
    source_dominance: str = "LOW",
) -> dict[str, object]:
    def level(value: str) -> dict[str, object]:
        return {"level": value, "reason_codes": [], "evidence": []}

    return {
        "policy_profile_version": PLATFORM_POLICY_PROFILE_VERSION,
        "policy_checked_at": PLATFORM_POLICY_CHECKED_AT,
        "youtube": {
            "reused_content": level(youtube_reused),
            "inauthentic_mass_produced": level(youtube_inauthentic),
            "spam_deceptive_practices": level(youtube_spam),
        },
        "facebook": {
            "unoriginal_content": level(facebook_unoriginal),
            "spam_repetitive": level(facebook_spam),
        },
        "generic": {
            "source_dominance": level(source_dominance),
            "substantive_transformation": level("LOW"),
            "template_mass_produced_feel": level("LOW"),
        },
        "account_level_repetition": "DEFERRED_TO_STAGE_7",
        "limitations": ["Decision support only; not a guarantee."],
    }


def make_result_spec(
    *,
    status: str = GovernancePlanStatus.APPROVED_FOR_SELECTION.value,
    dimensions: dict[str, str] | None = None,
    warnings: list[dict[str, object]] | None = None,
    verification: dict[str, object] | None = None,
    risk: dict[str, object] | None = None,
    hard_gates: list[dict[str, object]] | None = None,
    reason_codes: list[str] | None = None,
    remediation: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "dimensions": dimensions,
        "warnings": warnings or [],
        "verification": verification,
        "platform_risk": risk,
        "hard_gates": hard_gates or [],
        "reason_codes": reason_codes or [],
        "remediation": remediation or [],
    }


def warning(code: str, severity: str = "WARNING") -> dict[str, object]:
    return {"severity": severity, "code": code, "detail": code, "block_indexes": []}


def gate(code: str) -> dict[str, object]:
    return {"severity": "HARD_FAIL", "code": code, "detail": code, "block_indexes": []}


def seed_selection_fixture(
    session: Any,
    *,
    settings: FakeGovernanceSettings | None = None,
    result_specs: list[dict[str, object]],
    refinement_status: RefinementStatus = RefinementStatus.CANDIDATE_REFINED,
    final_transcript: str | None = None,
    overall_outcome: GovernanceSemanticOutcome | None = None,
    planning_on_final: bool = False,
    plan_narration: str = "NONE",
    rights_risk: Any = None,
    originality_risk: Any = None,
    rights_status: Any = None,
) -> SelectionFixture:
    settings = settings or FakeGovernanceSettings()
    second = (
        (TransformationStrategyType.ANALYSIS, "Compare the drop with the baseline")
        if len(result_specs) > 1
        else None
    )
    seed = seed_stage41(
        session,
        settings=settings,
        second_strategy=second,
        external=ExternalFactRequirement.NOT_REQUIRED,
    )
    source, candidate, refinement, analysis, strategies = seed
    if refinement_status is not RefinementStatus.CANDIDATE_REFINED or final_transcript is not None:
        refinement.status = refinement_status
        if final_transcript is not None:
            refinement.final_transcript = final_transcript
        session.flush()
    if rights_risk is not None or originality_risk is not None or rights_status is not None:
        from app.core.enums import OriginalityRisk, RightsRisk, RightsStatus
        from app.transformation.executor import build_transformation_executor

        if rights_risk is not None:
            candidate.rights_risk = RightsRisk(rights_risk)
        if originality_risk is not None:
            candidate.originality_risk = OriginalityRisk(originality_risk)
        if rights_status is not None:
            source.rights_status = RightsStatus(rights_status)
        session.flush()
        analysis.input_fingerprint = build_transformation_executor(
            session, settings
        ).input_fingerprint(candidate)
        session.flush()
    if planning_on_final:
        final = CandidateRefinement(
            source_video_id=source.id,
            clip_candidate_id=candidate.id,
            priority=RefinementPriority.FINAL_CLIP,
            status=RefinementStatus.FINAL_TRANSCRIPT_READY,
            coarse_start=refinement.coarse_start,
            coarse_end=refinement.coarse_end,
            context_start=refinement.context_start,
            context_end=refinement.context_end,
            refined_start=refinement.refined_start,
            refined_end=refinement.refined_end,
            automatic_transcript=refinement.final_transcript,
            final_transcript=refinement.final_transcript,
            word_timestamps=list(refinement.word_timestamps or []),
            confidence=0.95,
            quality_level="FINAL_CLIP",
            dialect_profile=refinement.dialect_profile,
            dialect_confidence=refinement.dialect_confidence,
            output_fingerprint="final-planning-output-fp",
        )
        session.add(final)
        session.flush()
        analysis.refinement_id = final.id
        analysis.refinement_priority = RefinementPriority.FINAL_CLIP.value
        analysis.refinement_quality_level = "FINAL_CLIP"
        session.flush()
        from app.transformation.executor import build_transformation_executor

        analysis.input_fingerprint = build_transformation_executor(
            session, settings
        ).input_fingerprint(candidate)
        session.flush()
        refinement = final

    provider_plans: list[Any] = []
    for index, strategy in enumerate(strategies[: len(result_specs)]):
        if index == 0:
            first_plan = make_source_value_plan(
                str(strategy.id), strategy.strategy_key, narration_need=plan_narration
            )
            if plan_narration == "RECOMMENDED":
                first_plan = with_review_narration(first_plan)
            provider_plans.append(first_plan)
        else:
            provider_plans.append(
                make_source_value_plan(
                    str(strategy.id),
                    strategy.strategy_key,
                    kind=SubstantiveValueKind.AUTHORED_THESIS,
                    intent=(
                        "Compare the promotion-rate drop with the pre-pandemic baseline "
                        "to argue remote work changed how mentors are found"
                    ),
                )
            )
    plan_set = run_planning(
        session,
        settings,
        seed,
        FakePlanningProvider(provider_plans),
        mode=SemanticProviderMode.ADAPTIVE,
    )
    plans = [row for row in list_plans(session, plan_set.id) if row.is_current]
    assert len(plans) == len(result_specs), (len(plans), len(result_specs))

    governance_set = TransformationGovernanceSet(
        source_video_id=source.id,
        clip_candidate_id=candidate.id,
        transformation_plan_set_id=plan_set.id,
        transformation_analysis_id=analysis.id,
        refinement_id=refinement.id,
        refinement_priority=refinement.priority.value,
        refinement_quality_level=refinement.quality_level,
        execution_status=GovernanceExecutionStatus.COMPLETE,
        governance_outcome=overall_outcome
        or _overall_outcome([str(spec["status"]) for spec in result_specs]),
        outcome_reasons=[],
        summary_counts=_summary_counts([str(spec["status"]) for spec in result_specs]),
        stage40_snapshot=dict(plan_set.stage40_snapshot or {}),
        target_context=dict(plan_set.target_context or {}),
        provider_mode=SemanticProviderMode.DETERMINISTIC,
        provider_identity={"provider": "deterministic"},
        provider_status="DETERMINISTIC",
        platform_policy_profile_version=PLATFORM_POLICY_PROFILE_VERSION,
        platform_policy_checked_at=PLATFORM_POLICY_CHECKED_AT,
        policy_version=GOVERNOR_POLICY_VERSION,
        schema_version=GOVERNOR_SCHEMA_VERSION,
        validation_version=GOVERNOR_VALIDATION_VERSION,
        cache_eligible=True,
    )
    session.add(governance_set)
    session.flush()

    for plan, spec in zip(plans, result_specs):
        status = str(spec["status"])
        dimensions = {**DEFAULT_DIMENSIONS, **_as_dict(spec.get("dimensions"))}
        verification = {**DEFAULT_VERIFICATION, **_as_dict(spec.get("verification"))}
        session.add(
            TransformationGovernanceResult(
                governance_set_id=governance_set.id,
                transformation_plan_id=plan.id,
                plan_output_fingerprint=plan.plan_output_fingerprint,
                status=GovernancePlanStatus(status),
                eligible_for_stage4_3=status in _ELIGIBLE_STATUSES,
                severity=str(spec.get("severity") or "ADVISORY"),
                hard_gates=_as_list(spec.get("hard_gates")),
                dimensions=dimensions,
                verification=verification,
                platform_risk=_as_dict(spec.get("platform_risk")) or platform_risk(),
                reason_codes=_as_list_str(spec.get("reason_codes")),
                warnings=_as_list(spec.get("warnings")),
                remediation=_as_list(spec.get("remediation")),
                governance_provider_evidence={"state": "NOT_REQUIRED", "critique": None},
                input_fingerprint=governance_set.input_fingerprint,
                output_fingerprint=f"gov-result-out-{plan.generation_rank}",
            )
        )
    session.flush()
    session.refresh(governance_set)
    executor = build_transformation_governance_executor(session, settings)
    governance_set.input_fingerprint = executor.input_fingerprint(governance_set)
    governance_set.output_fingerprint = f"gov-out-{governance_set.id}"
    session.flush()
    session.refresh(governance_set)
    return SelectionFixture(
        source=source,
        candidate=candidate,
        refinement=refinement,
        analysis=analysis,
        plan_set=plan_set,
        plans=plans,
        governance_set=governance_set,
        settings=settings,
    )


def _overall_outcome(statuses: list[str]) -> GovernanceSemanticOutcome:
    if any(status in _ELIGIBLE_STATUSES for status in statuses):
        return GovernanceSemanticOutcome.PLANS_ELIGIBLE_FOR_SELECTION
    if GovernancePlanStatus.GOVERNANCE_DEFERRED.value in statuses:
        return GovernanceSemanticOutcome.GOVERNANCE_DEFERRED
    return GovernanceSemanticOutcome.NO_GOVERNOR_APPROVED_PLAN


def _summary_counts(statuses: list[str]) -> dict[str, int]:
    counts = {
        "plans_total": len(statuses),
        "approved": 0,
        "caution": 0,
        "verification_blocked": 0,
        "revision_required": 0,
        "rejected": 0,
        "deferred": 0,
    }
    mapping = {
        GovernancePlanStatus.APPROVED_FOR_SELECTION.value: "approved",
        GovernancePlanStatus.APPROVED_WITH_CAUTION.value: "caution",
        GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION.value: "verification_blocked",
        GovernancePlanStatus.REVISION_REQUIRED.value: "revision_required",
        GovernancePlanStatus.REJECTED_BY_GOVERNOR.value: "rejected",
        GovernancePlanStatus.GOVERNANCE_DEFERRED.value: "deferred",
    }
    for status in statuses:
        key = mapping.get(status)
        if key is not None:
            counts[key] += 1
    return counts


__all__ = [
    "DEFAULT_DIMENSIONS",
    "DEFAULT_VERIFICATION",
    "SelectionFixture",
    "FakeGovernanceSettings",
    "gate",
    "install_selection_settings",
    "make_result_spec",
    "platform_risk",
    "seed_selection_fixture",
    "warning",
    "_make_stage41_plan",
]
