"""Deterministic+provider Stage 4.1 planning orchestration.

Deterministic logic owns readiness, bounds, routing, source-span resolution,
hero placement, duration arithmetic, substantive-value validation, narration/TTS
separation, verification enforcement, material distinction, persistence
eligibility, and fingerprint composition. An optional provider may only turn an
already-approved Stage 4.0 strategy into a concrete plan; it can never override a
hard deterministic gate and its failure never fails the source.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence

from app.candidates.providers import ProviderErrorCategory
from app.core.enums import (
    PlanSemanticOutcome,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.pipeline.executor import StageCancelled
from app.transformation.grounding import parse_reference
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
    POLICY_VERSION,
    SCHEMA_VERSION,
    Stage41Config,
    is_complex_strategy,
)
from app.transformation.planning.providers import (
    DeterministicPlanningProvider,
    PlanningProvider,
    PlanningProviderError,
    PlanningRequest,
    build_planning_request,
    deserialize_provider_result,
    planning_prompt_hash,
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
    resolve_source_span,
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
        job_id: object | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._configured_identity = (
            dict(provider_identity) if provider_identity is not None else None
        )
        self._mode = str(getattr(mode, "value", mode) or "deterministic")
        self._is_cancelled = is_cancelled or (lambda: False)
        self._job_id = job_id
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
        pending: list[tuple[Mapping[str, object], str, str, str, PlanningRequest]] = []
        # (strategy, provider_input_fp, route, strategy_key, request)

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
                    request=request,
                    route=route,
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
            pending.append((strategy, fingerprint, route, strategy_key, request))

        if pending and self._provider is not None and self._mode != "deterministic":
            self._run_provider(inputs, pending, accepted, attempts)
        elif pending:
            for strategy, fingerprint, _route, _key, _request in pending:
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
        if result is None or not result.plans:
            return None
        return result, fingerprint

    def _run_provider(
        self,
        inputs: PlanningInputs,
        pending: Sequence[tuple[Mapping[str, object], str, str, str, PlanningRequest]],
        accepted: list[ValidatedPlan],
        attempts: list[StrategyAttempt],
    ) -> None:
        groups: dict[str, list[tuple[Mapping[str, object], str, str, str, PlanningRequest]]] = {
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
            requests = [item[4] for item in group]
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
                phase = (
                    "MALFORMED_CONTENT"
                    if error.category == ProviderErrorCategory.MALFORMED_OUTPUT.value
                    else "TRANSPORT"
                )
                for strategy, fingerprint, route, _key, request in group:
                    attempts.append(
                        _attempt(
                            strategy,
                            _ATTEMPT_DEFERRED,
                            (error.category,),
                            None,
                            fingerprint,
                            diagnostics=self._diagnostics(
                                strategy,
                                request,
                                fingerprint,
                                route,
                                outcome="DEFERRED",
                                phase=phase,
                                reasons=(error.category,),
                                response_evidence=error.response_evidence or None,
                                schema_rejections=error.rejections,
                            ),
                        )
                    )
                continue
            self._check_cancelled()
            self._consume_group(group, results, inputs, accepted, attempts)

    def _consume_group(
        self,
        group: Sequence[tuple[Mapping[str, object], str, str, str, PlanningRequest]],
        results: Mapping[str, PlanProviderResult],
        inputs: PlanningInputs,
        accepted: list[ValidatedPlan],
        attempts: list[StrategyAttempt],
    ) -> None:
        from dataclasses import replace as _replace

        for strategy, fingerprint, route, key, request in group:
            result = results.get(key)
            if result is None or not result.plans:
                # A true omission is distinct from a returned response whose
                # requested identity was rejected by the parser. Both are
                # retryable, but only recorded parser evidence can explain the
                # latter without treating it as an absent plan.
                self._unfinished = True
                parser_rejections = tuple(result.rejections) if result is not None else ()
                schema_rejections = tuple(result.schema_rejections) if result is not None else ()
                if parser_rejections:
                    reason = str(parser_rejections[0].get("code") or "PARSER_REJECTION")
                    phase = "PARSER_REJECTION"
                elif schema_rejections:
                    reason = str(schema_rejections[0].get("code") or "SCHEMA_REJECTION")
                    phase = "SDK_SCHEMA_REJECTION"
                else:
                    reason = "PROVIDER_OMITTED"
                    phase = "OMISSION"
                attempts.append(
                    _attempt(
                        strategy,
                        _ATTEMPT_DEFERRED,
                        (reason,),
                        None,
                        fingerprint,
                        diagnostics=self._diagnostics(
                            strategy,
                            request,
                            fingerprint,
                            route,
                            outcome="DEFERRED",
                            phase=phase,
                            reasons=(reason,),
                            response_evidence=(
                                result.response_evidence if result is not None else None
                            ),
                            parser_rejections=parser_rejections,
                            schema_rejections=schema_rejections,
                        ),
                    )
                )
                continue
            plan, attempt = self._consume_result(
                strategy,
                inputs,
                result,
                provider_evidence={**self._provider_evidence, "tier": route},
                provider_input_fp=fingerprint,
                reused=False,
                request=request,
                route=route,
            )
            if result.rejections or result.schema_rejections:
                # Accepted-sibling isolation: the accepted plan remains valid,
                # while bounded evidence about a rejected sibling/duplicate or a
                # schema-level rejection is retained independently.
                rejection_codes = tuple(
                    str(item.get("code") or "PARSER_REJECTION")
                    for item in (*result.rejections, *result.schema_rejections)
                )
                if attempt.diagnostics:
                    diagnostics = dict(attempt.diagnostics)
                    diagnostics.setdefault(
                        "parser_rejections", [dict(item) for item in result.rejections[:8]]
                    )
                    diagnostics.setdefault(
                        "schema_rejections", [dict(item) for item in result.schema_rejections[:8]]
                    )
                    diagnostics.setdefault("response_evidence", dict(result.response_evidence))
                else:
                    diagnostics = self._diagnostics(
                        strategy,
                        request,
                        fingerprint,
                        route,
                        outcome="ACCEPTED",
                        phase=(
                            "PARSER_SIBLING_REJECTION"
                            if result.rejections
                            else "SDK_SCHEMA_REJECTION"
                        ),
                        reasons=rejection_codes,
                        response_evidence=result.response_evidence,
                        parser_rejections=result.rejections,
                        schema_rejections=result.schema_rejections,
                    )
                attempt = _replace(attempt, diagnostics=diagnostics)
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
        request: PlanningRequest,
        route: str,
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
            diagnostics = self._diagnostics(
                strategy,
                request,
                provider_input_fp,
                route,
                outcome="NO_VALID_PLAN",
                phase="EXPLICIT_DECLINE",
                reasons=("EXPLICIT_NO_VALID_PLAN",),
                response_evidence=result.response_evidence,
                provider_plan=provider_plan,
                inputs=inputs,
            )
            attempt = StrategyAttempt(
                strategy_id=str(strategy.get("id", "")),
                strategy_key=str(strategy.get("strategy_key", "")),
                status=attempt_status,
                reasons=tuple(validation.reasons),
                provider_input_fingerprint=provider_input_fp,
                checkpoint={
                    "provider_input_fingerprint": provider_input_fp,
                    "result": serialize_provider_result(result),
                },
                diagnostics=diagnostics,
            )
            return plan, attempt
        else:
            # Forbidden/malformed provider output (including inside an explicit
            # no-valid payload) is a safe invalid result, never an explicit
            # no-plan, and persists no raw provider text.
            self._unfinished = True
            return None, _attempt(
                strategy,
                _ATTEMPT_INVALID,
                validation.reasons,
                None,
                provider_input_fp,
                diagnostics=self._diagnostics(
                    strategy,
                    request,
                    provider_input_fp,
                    route,
                    outcome="INVALID",
                    phase="CONTRACT_REJECTION",
                    reasons=validation.reasons,
                    response_evidence=result.response_evidence,
                    schema_rejections=result.schema_rejections,
                    validation_rejections=validation.rejection_details,
                    provider_plan=provider_plan,
                    inputs=inputs,
                ),
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

    def _diagnostics(
        self,
        strategy: Mapping[str, object],
        request: PlanningRequest,
        provider_input_fp: str,
        route: str,
        *,
        outcome: str,
        phase: str,
        reasons: Sequence[str],
        response_evidence: Mapping[str, object] | None = None,
        parser_rejections: Sequence[Mapping[str, object]] = (),
        schema_rejections: Sequence[Mapping[str, object]] = (),
        validation_rejections: Sequence[Mapping[str, object]] = (),
        provider_plan: object | None = None,
        inputs: PlanningInputs | None = None,
    ) -> dict[str, object]:
        """Bounded, structural rejection evidence. Never raw provider/transcript text."""

        selectors = _selector_diagnostics(provider_plan, inputs, self._config)
        return {
            "contract_version": f"{POLICY_VERSION}/{SCHEMA_VERSION}",
            "job_id": str(self._job_id) if self._job_id is not None else None,
            "strategy_id": str(strategy.get("id", "")),
            "strategy_key": str(strategy.get("strategy_key", "")),
            "route": route,
            "outcome": outcome,
            "phase": phase,
            "provider_input_fingerprint": provider_input_fp,
            "prompt_hash": planning_prompt_hash(),
            "schema_version": SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "provider_identity": self.provider_identity(),
            "failure_codes": _failure_codes(reasons),
            "failures": _failure_details(reasons),
            "selectors": selectors,
            "mandatory_evidence": dict(request.mandatory_evidence),
            "source_selection": dict(request.source_selection),
            "response_evidence": dict(response_evidence or {}),
            "parser_rejections": [
                dict(item) for item in list(parser_rejections)[:8] if isinstance(item, Mapping)
            ],
            "schema_rejections": [
                dict(item) for item in list(schema_rejections)[:8] if isinstance(item, Mapping)
            ],
            "validation_rejections": [
                dict(item) for item in list(validation_rejections)[:8] if isinstance(item, Mapping)
            ],
            "rejection_origins": list(
                dict.fromkeys(
                    (["SDK_PARSING"] if schema_rejections else [])
                    + (["APPLICATION_VALIDATION"] if validation_rejections else [])
                    + (["PARSER"] if parser_rejections else [])
                )
            ),
        }

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
        invalid = any(attempt.status == _ATTEMPT_INVALID for attempt in attempts)
        # A truthful "no strategy works" conclusion requires every attempted
        # strategy to be a clean, validated explicit no-valid outcome. Any
        # invalid/malformed/deferred/omitted/provider-failure work makes the
        # run retryable instead of asserting a conclusion the provider never
        # validly reached.
        truthful_no_plan = bool(attempts) and all(
            attempt.status == _ATTEMPT_NO_VALID_PLAN for attempt in attempts
        )
        if ranked and has_verification:
            outcome = PlanSemanticOutcome.PLANS_GENERATED_WITH_VERIFICATION_REQUIRED
        elif ranked:
            outcome = PlanSemanticOutcome.PLANS_GENERATED
        elif self._provider_failed and self._mode != "deterministic":
            outcome = PlanSemanticOutcome.PROVIDER_UNAVAILABLE
        elif deferred or invalid or self._unfinished:
            outcome = PlanSemanticOutcome.PLANNING_DEFERRED
        elif self._mode != "deterministic" and self._provider is None and attempts:
            outcome = PlanSemanticOutcome.PROVIDER_UNAVAILABLE
        elif truthful_no_plan:
            outcome = PlanSemanticOutcome.NO_VALID_PLAN_FROM_STRATEGY
        else:
            # Defensive: never assert a no-valid conclusion for unknown work.
            outcome = PlanSemanticOutcome.PLANNING_DEFERRED

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


def _safe_reference(ref: object) -> dict[str, object]:
    parsed = parse_reference(ref)
    if parsed.kind == "QUOTE":
        raw = str(parsed.quote or "")
        return {
            "kind": "QUOTE",
            "digest": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "characters": len(raw),
        }
    if parsed.kind == "INVALID":
        return {
            "kind": "INVALID",
            "error": (parsed.error or "invalid_reference")[:80],
            "digest": hashlib.sha256(str(parsed.raw).encode("utf-8")).hexdigest(),
        }
    payload: dict[str, object] = {"kind": parsed.kind}
    if parsed.block_index is not None:
        payload["block_index"] = parsed.block_index
    if parsed.word_start_index is not None:
        payload["word_start_index"] = parsed.word_start_index
    if parsed.word_end_index is not None:
        payload["word_end_index"] = parsed.word_end_index
    if parsed.source_start is not None:
        payload["source_start"] = parsed.source_start
    if parsed.source_end is not None:
        payload["source_end"] = parsed.source_end
    return payload


def _selector_diagnostics(
    provider_plan: object | None,
    inputs: PlanningInputs | None,
    config: Stage41Config,
) -> list[dict[str, object]]:
    blocks = getattr(provider_plan, "blocks", None)
    if not isinstance(blocks, (list, tuple)):
        return []
    selectors: list[dict[str, object]] = []
    for index, block in enumerate(list(blocks)[:8]):
        block_type = getattr(block, "block_type", None)
        item: dict[str, object] = {
            "block_index": index,
            "block_type": block_type.value if hasattr(block_type, "value") else str(block_type),
            "source_role": (
                block.source_role.value if getattr(block, "source_role", None) is not None else None
            ),
            "word_start_index": getattr(block, "word_start_index", None),
            "word_end_index": getattr(block, "word_end_index", None),
            "use_full_window": bool(getattr(block, "use_full_window", False)),
            "grounding_refs": [
                _safe_reference(item) for item in list(getattr(block, "grounding_refs", ()))[:8]
            ],
            "resolved": None,
        }
        if inputs is not None and getattr(block, "block_type", None) is not None:
            try:
                span, error = resolve_source_span(block, inputs, config)  # type: ignore[arg-type]
            except Exception:
                span, error = None, "resolution_failed"
            if span is not None:
                item["resolved"] = {
                    "source_start": round(span.start, 6),
                    "source_end": round(span.end, 6),
                    "word_start_index": span.word_start_index,
                    "word_end_index": span.word_end_index,
                }
            else:
                item["resolve_error"] = str(error or "invalid_source_span")[:80]
        selectors.append(item)
    return selectors


def _failure_codes(reasons: Sequence[str]) -> list[str]:
    codes: list[str] = []
    for reason in reasons:
        code = str(reason).split("[", 1)[0].split(":", 1)[0].strip()
        if code and code not in codes:
            codes.append(code[:80])
        if len(codes) >= 8:
            break
    return codes


def _failure_details(reasons: Sequence[str]) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for reason in reasons[:8]:
        text = str(reason)
        code = text.split("[", 1)[0].split(":", 1)[0].strip()
        block_index: int | None = None
        marker = "block="
        if marker in text:
            tail = text.split(marker, 1)[1]
            digits = ""
            for character in tail:
                if character.isdigit():
                    digits += character
                else:
                    break
            if digits:
                block_index = int(digits)
        declaration_digest = None
        declaration_characters = None
        if "declaration=" in text:
            raw_declaration = text.split("declaration=", 1)[1].split("]", 1)[0].strip("'\"")
            encoded = raw_declaration.encode("utf-8")
            declaration_digest = hashlib.sha256(encoded).hexdigest()
            declaration_characters = len(raw_declaration)
        # Never persist the declaration body: it can contain source text.
        detail_text = text.split("]: ", 1)[1] if "]: " in text else text
        details.append(
            {
                "code": code[:80],
                "block_index": block_index,
                "declaration_digest": declaration_digest,
                "declaration_characters": declaration_characters,
                "detail": detail_text[:200],
            }
        )
    return details


def _attempt(
    strategy: Mapping[str, object],
    status: str,
    reasons: Sequence[str],
    plan: ValidatedPlan | None,
    provider_input_fp: str,
    *,
    diagnostics: Mapping[str, object] | None = None,
    request: PlanningRequest | None = None,
    route: str | None = None,
) -> StrategyAttempt:
    return StrategyAttempt(
        strategy_id=str(strategy.get("id", "")),
        strategy_key=str(strategy.get("strategy_key", "")),
        status=status,
        reasons=tuple(reason for reason in reasons if reason),
        provider_input_fingerprint=provider_input_fp,
        diagnostics=dict(diagnostics) if diagnostics is not None else None,
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
