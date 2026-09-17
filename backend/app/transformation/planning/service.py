"""Deterministic+provider Stage 4.1 planning orchestration.

Deterministic logic owns readiness, bounds, routing, source-span resolution,
hero placement, duration arithmetic, substantive-value validation, narration/TTS
separation, verification enforcement, material distinction, persistence
eligibility, and fingerprint composition. An optional provider may only turn an
already-approved Stage 4.0 strategy into a concrete plan; it can never override a
hard deterministic gate and its failure never fails the source.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from app.candidates.providers import ProviderErrorCategory
from app.core.enums import (
    PlanSemanticOutcome,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.pipeline.executor import StageCancelled
from app.transformation.planning.fingerprints import (
    build_plan_set_output_payload,
    build_provider_input_payload,
    planning_output_fingerprint,
)
from app.transformation.planning.fingerprints import (
    provider_input_fingerprint as planning_provider_input_fingerprint,
)
from app.transformation.planning.policy import (
    DEFAULT_CONFIG,
    Stage41Config,
    is_complex_strategy,
)
from app.transformation.planning.providers import (
    DeterministicPlanningProvider,
    PlanningProvider,
    PlanningProviderError,
    build_planning_request,
    deserialize_provider_result,
    serialize_provider_result,
)
from app.transformation.planning.types import (
    PlanningInputs,
    PlanningOutcome,
    PlanProviderResult,
    StrategyAttempt,
    ValidatedPlan,
)
from app.transformation.planning.validation import (
    ValidationResult,
    deterministic_provider_plan,
    validate_provider_plan,
)

_STATUS_COMPLETE = "COMPLETE"
_STATUS_DEGRADED = "PROVIDER_DEGRADED"
_STATUS_DETERMINISTIC = "DETERMINISTIC"
_STATUS_REUSED = "REUSED"
_STATUS_PROVIDER_OK = "PROVIDER_OK"
_STATUS_RATE_LIMITED = "RATE_LIMITED"
_STATUS_DEFERRED = "DEFERRED"
_STATUS_NO_VALID_PLAN = "NO_VALID_PLAN"
_STATUS_INVALID = "INVALID"

_ATTEMPT_GENERATED = "GENERATED"
_ATTEMPT_VERIFICATION = "GENERATED_VERIFICATION_REQUIRED"
_ATTEMPT_REUSED = "REUSED"
_ATTEMPT_NO_VALID_PLAN = "NO_VALID_PLAN"
_ATTEMPT_INVALID = "INVALID"
_ATTEMPT_DEFERRED = "DEFERRED"

_ROUTINE = "ROUTINE"
_STRONG = "STRONG"


class PlanningService:
    """Turn current recommended Stage 4.0 strategies into concrete plans."""

    def __init__(
        self,
        *,
        config: Stage41Config = DEFAULT_CONFIG,
        provider: PlanningProvider | None = None,
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
        self._provider_calls = 0
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
            return DeterministicPlanningProvider().runtime_identity()
        identity = getattr(self._provider, "runtime_identity", None)
        return dict(identity()) if callable(identity) else {"provider": "unknown"}

    def plan(
        self,
        inputs: PlanningInputs,
        *,
        input_fingerprint: str,
        checkpoints: Mapping[str, Mapping[str, object]] | None = None,
    ) -> PlanningOutcome:
        self._check_cancelled()
        checkpoints = checkpoints or {}
        strategies = list(inputs.stage40_strategies)
        accepted: list[ValidatedPlan] = []
        attempts: list[StrategyAttempt] = []
        pending: list[tuple[Mapping[str, object], str, str, str]] = []
        # (strategy, provider_input_fp, route, reason)

        for strategy in strategies:
            self._check_cancelled()
            strategy_key = str(strategy.get("strategy_key", ""))
            request = build_planning_request(strategy, inputs, self._config)
            route = self._route(strategy)
            fingerprint = planning_provider_input_fingerprint(
                build_provider_input_payload(
                    request=request.to_payload(),
                    provider_identity=self.provider_identity(),
                    route=route,
                )
            )
            reused = self._reuse(strategy, request, checkpoints, fingerprint)
            if reused is not None:
                result, cached_fp = reused
                plan, attempt = self._consume_result(
                    strategy,
                    inputs,
                    result,
                    provider_evidence={"reused": True},
                    provider_input_fp=cached_fp,
                    reused=True,
                )
                if plan is not None:
                    accepted.append(plan)
                attempts.append(attempt)
                continue
            deterministic = deterministic_provider_plan(strategy, inputs, self._config)
            if deterministic is not None and self._allows_deterministic(strategy, inputs):
                validation = self._validate(
                    strategy,
                    deterministic,
                    inputs,
                    provider_evidence={"origin": "deterministic"},
                    provider_input_fp="",
                )
                if validation.plan is not None:
                    accepted.append(validation.plan)
                    attempts.append(_attempt(strategy, _ATTEMPT_GENERATED, (), validation.plan, ""))
                    continue
                attempts.append(
                    _attempt(
                        strategy,
                        _ATTEMPT_INVALID,
                        validation.reasons,
                        None,
                        fingerprint,
                    )
                )
                continue
            if self._mode == "deterministic":
                attempts.append(
                    _attempt(
                        strategy,
                        _ATTEMPT_DEFERRED,
                        ("DETERMINISTIC_MODE",),
                        None,
                        fingerprint,
                    )
                )
                continue
            pending.append((strategy, fingerprint, route, strategy_key))

        if pending and self._provider is not None and self._mode != "deterministic":
            self._run_provider(inputs, pending, accepted, attempts)
        elif pending:
            for strategy, fingerprint, _route, _key in pending:
                attempts.append(
                    _attempt(
                        strategy,
                        _ATTEMPT_DEFERRED,
                        ("PROVIDER_UNAVAILABLE",),
                        None,
                        fingerprint,
                    )
                )

        return self._finalize(
            inputs,
            input_fingerprint,
            accepted,
            attempts,
        )

    def _allows_deterministic(self, strategy: Mapping[str, object], inputs: PlanningInputs) -> bool:
        # Deterministic fallback is conservative and never used in local_only
        # (which must exercise the explicit local provider). In adaptive mode a
        # configured provider is preferred for everything except a genuinely
        # strict source-led minimal case; deterministic fallback is otherwise the
        # safe behavior when no provider is available.
        if self._mode == "local_only":
            return False
        try:
            strategy_type = TransformationStrategyType(str(strategy.get("strategy_type")))
        except ValueError:
            return False
        if self._mode == "adaptive" and self._provider is not None:
            return strategy_type is TransformationStrategyType.SOURCE_LED_MINIMAL
        return True

    def _route(self, strategy: Mapping[str, object]) -> str:
        try:
            strategy_type = TransformationStrategyType(str(strategy.get("strategy_type")))
        except ValueError:
            return _ROUTINE
        try:
            intensity = TransformationIntensity(str(strategy.get("intensity")))
        except ValueError:
            intensity = TransformationIntensity.MODERATE
        requires = (
            str(strategy.get("external_verification_requirement", "NOT_REQUIRED"))
            == "REQUIRES_EXTERNAL_FACT_VERIFICATION"
        )
        return _STRONG if is_complex_strategy(strategy_type, intensity, requires) else _ROUTINE

    def _reuse(
        self,
        strategy: Mapping[str, object],
        request: object,
        checkpoints: Mapping[str, Mapping[str, object]],
        fingerprint: str,
    ) -> tuple[PlanProviderResult, str] | None:
        strategy_id = str(strategy.get("id", ""))
        stored = checkpoints.get(strategy_id)
        if not stored:
            return None
        if str(stored.get("provider_input_fingerprint", "")) != fingerprint:
            return None
        raw = stored.get("result")
        result = deserialize_provider_result(raw, request)  # type: ignore[arg-type]
        if result is None:
            return None
        return result, fingerprint

    def _run_provider(
        self,
        inputs: PlanningInputs,
        pending: Sequence[tuple[Mapping[str, object], str, str, str]],
        accepted: list[ValidatedPlan],
        attempts: list[StrategyAttempt],
    ) -> None:
        groups: dict[str, list[tuple[Mapping[str, object], str, str, str]]] = {
            _ROUTINE: [],
            _STRONG: [],
        }
        for item in pending:
            groups[item[2]].append(item)
        for tier in (_ROUTINE, _STRONG):
            group = groups[tier]
            if not group:
                continue
            self._check_cancelled()
            requests = [build_planning_request(item[0], inputs, self._config) for item in group]
            self._provider_attempted = True
            self._provider_calls += 1
            if tier == _STRONG:
                self._strong_calls += 1
            else:
                self._routine_calls += 1
            try:
                results = self._provider.plan(requests, tier)  # type: ignore[union-attr]
                self._provider_status = _STATUS_PROVIDER_OK
            except _Cancelled:
                raise
            except PlanningProviderError as error:
                self._provider_failed = True
                self._unfinished = True
                self._provider_error_category = error.category
                self._provider_status = (
                    _STATUS_RATE_LIMITED
                    if error.category == ProviderErrorCategory.RATE_LIMITED.value
                    else _STATUS_DEGRADED
                )
                for strategy, fingerprint, _route, _key in group:
                    attempts.append(
                        _attempt(strategy, _ATTEMPT_DEFERRED, (error.category,), None, fingerprint)
                    )
                continue
            self._check_cancelled()
            self._consume_group(group, results, inputs, accepted, attempts)

    def _consume_group(
        self,
        group: Sequence[tuple[Mapping[str, object], str, str, str]],
        results: Mapping[str, PlanProviderResult],
        inputs: PlanningInputs,
        accepted: list[ValidatedPlan],
        attempts: list[StrategyAttempt],
    ) -> None:
        for strategy, fingerprint, _route, key in group:
            result = results.get(key)
            if result is None or not result.plans:
                # Omission is not an explicit no-plan result: retryable.
                self._unfinished = True
                attempts.append(
                    _attempt(strategy, _ATTEMPT_DEFERRED, ("PROVIDER_OMITTED",), None, fingerprint)
                )
                continue
            plan, attempt = self._consume_result(
                strategy,
                inputs,
                result,
                provider_evidence={**self._provider_evidence, "tier": _route},
                provider_input_fp=fingerprint,
                reused=False,
            )
            if plan is not None:
                accepted.append(plan)
            attempts.append(attempt)

    def _consume_result(
        self,
        strategy: Mapping[str, object],
        inputs: PlanningInputs,
        result: PlanProviderResult,
        *,
        provider_evidence: Mapping[str, object],
        provider_input_fp: str,
        reused: bool,
    ) -> tuple[ValidatedPlan | None, StrategyAttempt]:
        provider_plan = result.plans[0]
        # Every parsed provider result — including an explicit ``no_valid_plan``
        # — must pass the complete deterministic provider-text/boundary
        # validation before any of its fields can be persisted, checkpointed,
        # used as an attempt/outcome reason, or reused from cache. The validator
        # checks forbidden text before its own no-valid early return; branching
        # on ``no_valid_plan`` first would let adversarial reasons bypass it.
        validation = self._validate(
            strategy,
            provider_plan,
            inputs,
            provider_evidence=provider_evidence,
            provider_input_fp=provider_input_fp,
        )
        if validation.plan is not None:
            plan = validation.plan
            attempt_status = (
                _ATTEMPT_VERIFICATION
                if plan.status.value == "PLAN_GENERATED_WITH_VERIFICATION_REQUIRED"
                else _ATTEMPT_GENERATED
            )
        elif validation.explicit_no_plan:
            # A clean, bounded explicit decline is a legitimate no-plan result.
            plan = None
            attempt_status = _ATTEMPT_NO_VALID_PLAN
        else:
            # Forbidden/malformed provider output (including inside an explicit
            # no-valid payload) is a safe invalid result, never an explicit
            # no-plan, and persists no raw provider text.
            self._unfinished = True
            return None, _attempt(
                strategy, _ATTEMPT_INVALID, validation.reasons, None, provider_input_fp
            )
        checkpoint = {
            "provider_input_fingerprint": provider_input_fp,
            "result": serialize_provider_result(result),
        }
        attempt = StrategyAttempt(
            strategy_id=str(strategy.get("id", "")),
            strategy_key=str(strategy.get("strategy_key", "")),
            status=attempt_status,
            reasons=tuple(validation.reasons) if validation.explicit_no_plan else (),
            provider_input_fingerprint=provider_input_fp,
            # Always persist the checkpoint, including on reuse, so an accepted
            # provider result stays durable across repeated forced reruns and
            # does not require another hosted call.
            checkpoint=checkpoint,
        )
        return plan, attempt

    def _raw_hosted_calls(self) -> int:
        """Actual raw hosted generate_content calls, never generic invocations.

        Local-only/Qwen, deterministic, and any non-hosted provider report zero;
        only a provider that declares itself hosted contributes its real raw-call
        count.
        """

        if self._provider is None or not getattr(self._provider, "hosted_provider", False):
            return 0
        counter = getattr(self._provider, "raw_call_count", None)
        if callable(counter):
            try:
                return int(counter())
            except Exception:
                return 0
        return 0

    def _validate(
        self,
        strategy: Mapping[str, object],
        provider_plan: object,
        inputs: PlanningInputs,
        *,
        provider_evidence: Mapping[str, object],
        provider_input_fp: str,
    ) -> ValidationResult:
        return validate_provider_plan(
            provider_plan,  # type: ignore[arg-type]
            strategy,
            inputs,
            self._config,
            provider_evidence=provider_evidence,
            provider_input_fingerprint=provider_input_fp,
        )

    def _finalize(
        self,
        inputs: PlanningInputs,
        input_fingerprint: str,
        accepted: Sequence[ValidatedPlan],
        attempts: Sequence[StrategyAttempt],
    ) -> PlanningOutcome:
        ordered = sorted(
            accepted,
            key=lambda item: (item.strategy_rank, item.strategy_type.value),
        )
        deduped: list[ValidatedPlan] = []
        seen_signatures: set[str] = set()
        for plan in ordered:
            if plan.structure_signature in seen_signatures:
                continue
            seen_signatures.add(plan.structure_signature)
            deduped.append(plan)
        ranked = [_with_generation_rank(plan, index) for index, plan in enumerate(deduped, start=1)]

        has_verification = any(
            plan.status.value == "PLAN_GENERATED_WITH_VERIFICATION_REQUIRED" for plan in ranked
        )
        deferred = any(attempt.status == _ATTEMPT_DEFERRED for attempt in attempts)
        if ranked and has_verification:
            outcome = PlanSemanticOutcome.PLANS_GENERATED_WITH_VERIFICATION_REQUIRED
        elif ranked:
            outcome = PlanSemanticOutcome.PLANS_GENERATED
        elif self._provider_failed and self._mode != "deterministic":
            outcome = PlanSemanticOutcome.PROVIDER_UNAVAILABLE
        elif deferred:
            outcome = PlanSemanticOutcome.PLANNING_DEFERRED
        elif self._mode != "deterministic" and self._provider is None and attempts:
            outcome = PlanSemanticOutcome.PROVIDER_UNAVAILABLE
        else:
            outcome = PlanSemanticOutcome.NO_VALID_PLAN_FROM_STRATEGY

        cache_eligible = not self._unfinished

        execution_status = (
            _STATUS_DEGRADED if self._provider_failed and ranked else _STATUS_COMPLETE
        )

        output_fp = planning_output_fingerprint(
            build_plan_set_output_payload(
                semantic_outcome=outcome.value,
                plans=ranked,
                attempts=attempts,
            )
        )
        metrics = {
            "candidates_planned": 1,
            "plans_generated": len(ranked),
            "plans_per_candidate": len(ranked),
            "strategies_attempted": len(attempts),
            "explicit_no_valid_plan_strategies": sum(
                1 for attempt in attempts if attempt.status == _ATTEMPT_NO_VALID_PLAN
            ),
            "invalid_malformed_strategies": sum(
                1 for attempt in attempts if attempt.status == _ATTEMPT_INVALID
            ),
            "verification_required_plans": sum(
                1
                for plan in ranked
                if plan.status.value == "PLAN_GENERATED_WITH_VERIFICATION_REQUIRED"
            ),
            "narration_need_distribution": _narration_distribution(ranked),
            "gemini_calls": self._provider_calls,
            "routine_calls": self._routine_calls,
            "strong_calls": self._strong_calls,
            "hosted_raw_calls": self._raw_hosted_calls(),
            "cache_hits": sum(1 for attempt in attempts if attempt.status == _ATTEMPT_REUSED),
            "checkpoint_reuses": sum(
                1 for attempt in attempts if attempt.status == _ATTEMPT_REUSED
            ),
            "provider_failures": 1 if self._provider_failed else 0,
            "provider_failure_category": self._provider_error_category,
            "provider_status": self._provider_status,
        }
        return PlanningOutcome(
            execution_status=execution_status,
            semantic_outcome=outcome,
            outcome_reasons=tuple(
                dict.fromkeys(reason for attempt in attempts for reason in attempt.reasons)
            ),
            plans=tuple(ranked),
            attempts=tuple(attempts),
            provider_status=self._provider_status,
            provider_evidence=dict(self._provider_evidence),
            provider_mode=self._mode,
            provider_identity=self.provider_identity(),
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fp,
            cache_eligible=cache_eligible,
            metrics=metrics,
        )


def _attempt(
    strategy: Mapping[str, object],
    status: str,
    reasons: Sequence[str],
    plan: ValidatedPlan | None,
    provider_input_fp: str,
) -> StrategyAttempt:
    return StrategyAttempt(
        strategy_id=str(strategy.get("id", "")),
        strategy_key=str(strategy.get("strategy_key", "")),
        status=status,
        reasons=tuple(reason for reason in reasons if reason),
        provider_input_fingerprint=provider_input_fp,
    )


def _with_generation_rank(plan: ValidatedPlan, rank: int) -> ValidatedPlan:
    from dataclasses import replace

    return replace(plan, generation_rank=rank)


def _narration_distribution(plans: Sequence[ValidatedPlan]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for plan in plans:
        key = plan.narration.need.value
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


class _Cancelled(StageCancelled):
    """Internal cooperative cancellation marker."""


__all__ = [
    "DeterministicPlanningProvider",
    "PlanningService",
]
