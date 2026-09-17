"""Deterministic+provider Stage 4.2 governance orchestration.

Deterministic logic owns integrity, evidence, dimensions, hard gates, status
precedence, platform-risk interpretation, and fingerprint composition. An
optional provider may only supply bounded observable semantic findings; it can
never assign a final status or platform classification, and its failure never
fails the candidate/source.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from app.candidates.providers import ProviderErrorCategory
from app.core.enums import (
    GovernancePlanStatus,
    GovernanceSemanticOutcome,
    RetentionEffectFinding,
    SemanticFidelityFinding,
    SubstantiveValueFinding,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.pipeline.executor import StageCancelled
from app.transformation.governance.fingerprints import (
    build_governance_output_payload,
    build_plan_governance_output_payload,
    build_provider_input_payload,
    governance_output_fingerprint,
    plan_governance_fingerprint,
)
from app.transformation.governance.fingerprints import (
    provider_input_fingerprint as governance_provider_input_fingerprint,
)
from app.transformation.governance.policy import DEFAULT_CONFIG, Stage42Config
from app.transformation.governance.providers import (
    DeterministicGovernanceProvider,
    GovernanceProvider,
    GovernanceProviderError,
    GovernanceRequest,
    build_governance_request,
    deserialize_critique,
    serialize_critique,
)
from app.transformation.governance.types import (
    GovernanceAttempt,
    GovernanceInputs,
    GovernanceOutcome,
    PlanEvidence,
    PlanGovernance,
    ProviderCritique,
)
from app.transformation.governance.validation import (
    SEMANTIC_AVAILABLE,
    SEMANTIC_INVALID,
    SEMANTIC_NOT_REQUIRED,
    SEMANTIC_UNAVAILABLE,
    PlanEvaluation,
    apply_semantic_review,
    build_plan_governance,
    evaluate_plan,
)

_STRONG_STRATEGIES = frozenset(
    {
        TransformationStrategyType.ANALYSIS,
        TransformationStrategyType.COUNTERPOINT,
        TransformationStrategyType.COMPARISON,
        TransformationStrategyType.NEWS_CONTEXT,
        TransformationStrategyType.DEBATE_CONTEXT,
        TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION,
    }
)

_STATUS_COMPLETE = "COMPLETE"
_STATUS_DEGRADED = "PROVIDER_DEGRADED"
_STATUS_DETERMINISTIC = "DETERMINISTIC"
_STATUS_PROVIDER_OK = "PROVIDER_OK"
_STATUS_RATE_LIMITED = "RATE_LIMITED"

_ATTEMPT_REUSED = "REUSED"
_ATTEMPT_REVIEWED = "REVIEWED"
_ATTEMPT_DEFERRED = "DEFERRED"
_ATTEMPT_INVALID = "INVALID"
_ATTEMPT_NOT_REQUIRED = "NOT_REQUIRED"

_ROUTINE = "ROUTINE"
_STRONG = "STRONG"


class _Cancelled(StageCancelled):
    """Internal cooperative cancellation marker."""


class GovernanceService:
    """Govern current Stage 4.1 plans for one candidate independently."""

    def __init__(
        self,
        *,
        config: Stage42Config = DEFAULT_CONFIG,
        provider: GovernanceProvider | None = None,
        provider_identity: Mapping[str, object] | None = None,
        mode: object | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._configured_identity = (
            dict(provider_identity) if provider_identity is not None else None
        )
        self._mode = str(getattr(mode, "value", mode) or "deterministic")
        self._is_cancelled = is_cancelled or (lambda: False)
        self._provider_status = _STATUS_DETERMINISTIC
        self._provider_attempted = False
        self._provider_failed = False
        self._provider_error_category: str | None = None
        self._provider_evidence: dict[str, object] = {}
        self._raw_calls = 0
        self._routine_calls = 0
        self._strong_calls = 0
        self._unfinished = False

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise _Cancelled()

    def provider_identity(self) -> dict[str, object]:
        if self._configured_identity is not None:
            return dict(self._configured_identity)
        if self._provider is None:
            return DeterministicGovernanceProvider().runtime_identity()
        identity = getattr(self._provider, "runtime_identity", None)
        return dict(identity()) if callable(identity) else {"provider": "unknown"}

    def govern(
        self,
        inputs: GovernanceInputs,
        *,
        input_fingerprint: str,
        checkpoints: Mapping[str, Mapping[str, object]] | None = None,
    ) -> GovernanceOutcome:
        self._check_cancelled()
        checkpoints = checkpoints or {}
        request = build_governance_request(inputs, self._config)
        deterministic = self._mode == "deterministic"

        evaluations: list[PlanEvaluation] = []
        attempts: list[GovernanceAttempt] = []
        pending: list[PlanEvaluation] = []
        fingerprints: dict[str, str] = {}
        reused_fingerprints: dict[str, str] = {}

        for plan in inputs.plans:
            self._check_cancelled()
            evaluation = evaluate_plan(
                plan,
                inputs,
                self._config,
                provider_mode_deterministic=deterministic,
            )
            fingerprint = self._plan_provider_fingerprint(request, plan)
            fingerprints[plan.plan_id] = fingerprint
            reused = self._reuse(plan, checkpoints, fingerprint)
            if reused is not None:
                apply_semantic_review(evaluation, reused, SEMANTIC_AVAILABLE)
                reused_fingerprints[plan.plan_id] = fingerprint
                attempts.append(_attempt(plan, _ATTEMPT_REUSED, (), fingerprint, reused))
                evaluations.append(evaluation)
                continue
            if evaluation.requires_semantic_review:
                pending.append(evaluation)
            evaluator_state = (
                SEMANTIC_NOT_REQUIRED
                if not evaluation.requires_semantic_review
                else SEMANTIC_UNAVAILABLE
            )
            if not evaluation.requires_semantic_review:
                attempts.append(_attempt(plan, _ATTEMPT_NOT_REQUIRED, (), ""))
            evaluations.append(evaluation)
            evaluation.semantic_state = evaluator_state

        if pending and self._provider is not None and not deterministic:
            self._run_provider(request, inputs, pending, fingerprints, attempts)
        elif pending:
            for evaluation in pending:
                apply_semantic_review(evaluation, None, SEMANTIC_UNAVAILABLE)
                attempts.append(
                    _attempt(
                        evaluation.plan,
                        _ATTEMPT_DEFERRED,
                        (ProviderErrorCategory.MISSING_KEY.value,),
                        fingerprints[evaluation.plan.plan_id],
                    )
                )

        plan_governances = self._finalize_plans(inputs, input_fingerprint, evaluations)
        attempts_by_plan = {attempt.plan_id: attempt for attempt in attempts}
        ordered_attempts: list[GovernanceAttempt] = []
        for governance in plan_governances:
            attempt = attempts_by_plan.get(governance.plan_id)
            if attempt is None:
                attempt = _attempt(None, _ATTEMPT_REUSED, (), "")  # pragma: no cover
            ordered_attempts.append(attempt)

        return self._finalize(
            inputs,
            input_fingerprint,
            plan_governances,
            ordered_attempts,
        )

    def _plan_provider_fingerprint(self, request: GovernanceRequest, plan: PlanEvidence) -> str:
        return governance_provider_input_fingerprint(
            build_provider_input_payload(
                request=_plan_request_payload(request, plan),
                provider_identity=self.provider_identity(),
                route=_ROUTINE,
            )
        )

    def _reuse(
        self,
        plan: PlanEvidence,
        checkpoints: Mapping[str, Mapping[str, object]],
        fingerprint: str,
    ) -> ProviderCritique | None:
        stored = checkpoints.get(plan.plan_id)
        if not stored:
            return None
        if str(stored.get("provider_input_fingerprint", "")) != fingerprint:
            return None
        critique = deserialize_critique(stored.get("critique"))
        if critique is None:
            return None
        if critique.plan_id != plan.plan_id:
            return None
        if critique.plan_output_fingerprint != plan.plan_output_fingerprint:
            return None
        return critique

    def _run_provider(
        self,
        request: GovernanceRequest,
        inputs: GovernanceInputs,
        pending: Sequence[PlanEvaluation],
        fingerprints: Mapping[str, str],
        attempts: list[GovernanceAttempt],
    ) -> None:
        pending_ids = {evaluation.plan.plan_id for evaluation in pending}
        tier = _STRONG if any(_plan_requires_strong(item.plan) for item in pending) else _ROUTINE
        self._provider_attempted = True
        critiques = self._call_provider(request, pending_ids, tier, inputs)

        unresolved: list[PlanEvaluation] = []
        for evaluation in pending:
            plan_id = evaluation.plan.plan_id
            critique = critiques.get(plan_id)
            if critique is None:
                if self._provider_failed:
                    apply_semantic_review(evaluation, None, SEMANTIC_UNAVAILABLE)
                    status = _ATTEMPT_DEFERRED
                    reasons = (
                        self._provider_error_category or ProviderErrorCategory.PROVIDER_ERROR.value,
                    )
                else:
                    apply_semantic_review(evaluation, None, SEMANTIC_INVALID)
                    status = _ATTEMPT_INVALID
                    reasons = (ProviderErrorCategory.MALFORMED_OUTPUT.value,)
                attempts.append(_attempt(evaluation.plan, status, reasons, fingerprints[plan_id]))
                continue
            if _critique_unknown(critique):
                unresolved.append(evaluation)
            apply_semantic_review(evaluation, critique, SEMANTIC_AVAILABLE)
            attempts.append(
                _attempt(evaluation.plan, _ATTEMPT_REVIEWED, (), fingerprints[plan_id], critique)
            )

        if unresolved and self._raw_calls < self._config.max_hosted_raw_calls:
            self._run_second_strong_call(request, unresolved, fingerprints, attempts, inputs)

    def _run_second_strong_call(
        self,
        request: GovernanceRequest,
        unresolved: Sequence[PlanEvaluation],
        fingerprints: Mapping[str, str],
        attempts: list[GovernanceAttempt],
        inputs: GovernanceInputs,
    ) -> None:
        self._check_cancelled()
        target_ids = {evaluation.plan.plan_id for evaluation in unresolved}
        critiques = self._call_provider(request, target_ids, _STRONG, inputs)
        for evaluation in unresolved:
            plan_id = evaluation.plan.plan_id
            critique = critiques.get(plan_id)
            if critique is None:
                # Preserve the first accepted critique checkpoint (if any) and
                # mark this plan truthfully deferred.
                apply_semantic_review(evaluation, evaluation.critique, SEMANTIC_UNAVAILABLE)
                attempts.append(
                    _attempt(
                        evaluation.plan,
                        _ATTEMPT_DEFERRED,
                        (ProviderErrorCategory.PROVIDER_ERROR.value,),
                        fingerprints[plan_id],
                        evaluation.critique,
                    )
                )
                continue
            # The bounded strong call resolved the ambiguity: apply it and persist
            # it as the plan checkpoint so a forced rerun with unchanged inputs
            # reuses it and makes zero additional hosted calls.
            apply_semantic_review(evaluation, critique, SEMANTIC_AVAILABLE)
            attempts.append(
                _attempt(
                    evaluation.plan,
                    _ATTEMPT_REVIEWED,
                    (),
                    fingerprints[plan_id],
                    critique,
                )
            )

    def _call_provider(
        self,
        request: GovernanceRequest,
        plan_ids: set[str],
        tier: str,
        inputs: GovernanceInputs,
    ) -> dict[str, ProviderCritique]:
        subset = GovernanceRequest(
            plan_set_id=request.plan_set_id,
            candidate_id=request.candidate_id,
            content_type=request.content_type,
            source_moment_structure=request.source_moment_structure,
            reflection_start=request.reflection_start,
            reflection_end=request.reflection_end,
            dialect_profile=request.dialect_profile,
            transcript=request.transcript,
            context_text=request.context_text,
            idea_summary=request.idea_summary,
            topic_summary=request.topic_summary,
            rights_risk=request.rights_risk,
            originality_risk=request.originality_risk,
            plans=tuple(plan for plan in request.plans if str(plan.get("plan_id")) in plan_ids),
        )
        self._raw_calls += 1
        if tier == _STRONG:
            self._strong_calls += 1
        else:
            self._routine_calls += 1
        try:
            result = self._provider.govern(subset, tier)  # type: ignore[union-attr]
            self._provider_status = _STATUS_PROVIDER_OK
        except _Cancelled:
            raise
        except StageCancelled:
            raise
        except GovernanceProviderError as error:
            self._provider_failed = True
            self._unfinished = True
            self._provider_error_category = error.category
            self._provider_status = (
                _STATUS_RATE_LIMITED
                if error.category == ProviderErrorCategory.RATE_LIMITED.value
                else _STATUS_DEGRADED
            )
            return {}
        except Exception:
            self._provider_failed = True
            self._unfinished = True
            self._provider_error_category = ProviderErrorCategory.PROVIDER_ERROR.value
            self._provider_status = _STATUS_DEGRADED
            return {}
        return {critique.plan_id: critique for critique in result.critiques}

    def _finalize_plans(
        self,
        inputs: GovernanceInputs,
        input_fingerprint: str,
        evaluations: Sequence[PlanEvaluation],
    ) -> list[PlanGovernance]:
        ordered = sorted(evaluations, key=lambda item: item.plan.generation_rank)
        results: list[PlanGovernance] = []
        for evaluation in ordered:
            output_fp = plan_governance_fingerprint(
                build_plan_governance_output_payload(
                    build_plan_governance(
                        evaluation,
                        input_fingerprint=input_fingerprint,
                        output_fingerprint="",
                    )
                )
            )
            results.append(
                build_plan_governance(
                    evaluation,
                    input_fingerprint=input_fingerprint,
                    output_fingerprint=output_fp,
                )
            )
            if evaluation.requires_semantic_review and evaluation.semantic_state in {
                SEMANTIC_UNAVAILABLE,
                SEMANTIC_INVALID,
            }:
                self._unfinished = True
        return results

    def _finalize(
        self,
        inputs: GovernanceInputs,
        input_fingerprint: str,
        plans: Sequence[PlanGovernance],
        attempts: Sequence[GovernanceAttempt],
    ) -> GovernanceOutcome:
        counts = _summary_counts(plans)
        if counts["approved"] + counts["caution"] > 0:
            outcome = GovernanceSemanticOutcome.PLANS_ELIGIBLE_FOR_SELECTION
        elif counts["deferred"] > 0:
            outcome = GovernanceSemanticOutcome.GOVERNANCE_DEFERRED
        else:
            outcome = GovernanceSemanticOutcome.NO_GOVERNOR_APPROVED_PLAN

        if self._provider_failed or self._provider_status == _STATUS_RATE_LIMITED:
            execution_status = _STATUS_DEGRADED
        else:
            execution_status = _STATUS_COMPLETE

        output_fp = governance_output_fingerprint(
            build_governance_output_payload(
                semantic_outcome=outcome.value,
                plans=plans,
                summary_counts=counts,
                attempts=attempts,
            )
        )
        metrics = {
            "governance_sets_completed": 1,
            "plans_evaluated": len(plans),
            "plans_per_governance": len(plans),
            "deterministic_hard_rejects": counts["rejected"],
            "verification_blocked_plans": counts["verification_blocked"],
            "revision_required_plans": counts["revision_required"],
            "plans_semantically_reviewed": sum(
                1 for attempt in attempts if attempt.status == _ATTEMPT_REVIEWED
            ),
            "approvals": counts["approved"],
            "cautions": counts["caution"],
            "revisions": counts["revision_required"],
            "rejections": counts["rejected"],
            "deferred_plans": counts["deferred"],
            "no_survivor_candidates": 1
            if outcome is GovernanceSemanticOutcome.NO_GOVERNOR_APPROVED_PLAN
            else 0,
            "platform_risk_distribution": _platform_distribution(plans),
            "narration_excessive_findings": _count_reason(plans, "NARRATION_EXCESSIVE"),
            "semantic_fidelity_failures": _count_reason(plans, "SEMANTIC_DISTORTION"),
            "gemini_calls": self._raw_calls,
            "routine_calls": self._routine_calls,
            "strong_calls": self._strong_calls,
            "hosted_raw_calls": self._raw_hosted_calls(),
            "plans_per_provider_call": _plans_per_call(len(plans), self._raw_calls),
            "checkpoint_reuses": sum(
                1 for attempt in attempts if attempt.status == _ATTEMPT_REUSED
            ),
            "provider_failures": 1 if self._provider_failed else 0,
            "provider_failure_category": self._provider_error_category,
            "provider_status": self._provider_status,
        }
        return GovernanceOutcome(
            execution_status=execution_status,
            semantic_outcome=outcome,
            outcome_reasons=tuple(
                dict.fromkeys(reason for attempt in attempts for reason in attempt.reasons)
            ),
            plans=tuple(plans),
            attempts=tuple(attempts),
            summary_counts=counts,
            provider_status=self._provider_status,
            provider_evidence=dict(self._provider_evidence),
            provider_mode=self._mode,
            provider_identity=self.provider_identity(),
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fp,
            cache_eligible=not self._unfinished,
            metrics=metrics,
        )

    def _raw_hosted_calls(self) -> int:
        if self._provider is None or not getattr(self._provider, "hosted_provider", False):
            return 0
        counter = getattr(self._provider, "raw_call_count", None)
        if callable(counter):
            try:
                return int(counter())
            except Exception:
                return 0
        return 0


def _plan_request_payload(request: GovernanceRequest, plan: PlanEvidence) -> dict[str, object]:
    payload = request.to_payload()
    plans = payload.get("plans")
    if isinstance(plans, list):
        payload["plans"] = [
            item
            for item in plans
            if isinstance(item, dict) and str(item.get("plan_id")) == plan.plan_id
        ]
    return payload


def _plan_requires_strong(plan: PlanEvidence) -> bool:
    try:
        strategy = TransformationStrategyType(plan.strategy_type)
    except ValueError:
        return True
    if strategy in _STRONG_STRATEGIES:
        return True
    if plan.intensity == TransformationIntensity.STRONG.value:
        return True
    return bool(plan.verification_dependencies)


def _critique_unknown(critique: ProviderCritique) -> bool:
    return (
        critique.fidelity == SemanticFidelityFinding.UNKNOWN
        or critique.value == SubstantiveValueFinding.UNKNOWN
        or critique.retention == RetentionEffectFinding.UNKNOWN
    )


def _attempt(
    plan: PlanEvidence | None,
    status: str,
    reasons: Sequence[str],
    provider_input_fp: str,
    critique: ProviderCritique | None = None,
) -> GovernanceAttempt:
    plan_id = plan.plan_id if plan is not None else ""
    plan_fp = plan.plan_output_fingerprint if plan is not None else ""
    return GovernanceAttempt(
        plan_id=plan_id,
        plan_output_fingerprint=plan_fp,
        status=status,
        reasons=tuple(reason for reason in reasons if reason),
        provider_input_fingerprint=provider_input_fp,
        checkpoint=(
            {
                "provider_input_fingerprint": provider_input_fp,
                "critique": serialize_critique(critique),
            }
            if critique is not None
            else None
        ),
    )


def _summary_counts(plans: Sequence[PlanGovernance]) -> dict[str, int]:
    counts = {
        "plans_total": len(plans),
        "approved": 0,
        "caution": 0,
        "verification_blocked": 0,
        "revision_required": 0,
        "rejected": 0,
        "deferred": 0,
    }
    for plan in plans:
        if plan.status is GovernancePlanStatus.APPROVED_FOR_SELECTION:
            counts["approved"] += 1
        elif plan.status is GovernancePlanStatus.APPROVED_WITH_CAUTION:
            counts["caution"] += 1
        elif plan.status is GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION:
            counts["verification_blocked"] += 1
        elif plan.status is GovernancePlanStatus.REVISION_REQUIRED:
            counts["revision_required"] += 1
        elif plan.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR:
            counts["rejected"] += 1
        elif plan.status is GovernancePlanStatus.GOVERNANCE_DEFERRED:
            counts["deferred"] += 1
    return counts


def _platform_distribution(plans: Sequence[PlanGovernance]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for plan in plans:
        youtube = plan.platform_risk.get("youtube")
        if isinstance(youtube, Mapping):
            reused = youtube.get("reused_content")
            if isinstance(reused, Mapping):
                level = str(reused.get("level", "UNDETERMINED"))
                distribution[level] = distribution.get(level, 0) + 1
    return distribution


def _count_reason(plans: Sequence[PlanGovernance], code: str) -> int:
    return sum(1 for plan in plans if code in plan.reason_codes)


def _plans_per_call(plans: int, calls: int) -> float:
    if calls <= 0:
        return 0.0
    return round(plans / calls, 4)


__all__ = [
    "DeterministicGovernanceProvider",
    "GovernanceService",
]
