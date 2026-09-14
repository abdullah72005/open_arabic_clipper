"""Deterministic+provider Stage 4.0 eligibility and strategy orchestration.

Deterministic logic owns blockers, transformation necessity, source-moment
structure, suitability, hard gates, validation, final eligibility, fallback, and
ranking. An optional provider may only assess candidates that survived those
deterministic gates and may never override a hard blocker. Provider failure
never fails the analysis; it degrades truthfully and preserves deterministic
results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from app.candidates.providers import ProviderErrorCategory
from app.core.enums import (
    ContentType,
    ExternalFactRequirement,
    OriginalityRisk,
    RightsRisk,
    SemanticProviderMode,
    StrategyDisposition,
    StrategyOrigin,
    TransformationEligibilityOutcome,
)
from app.transformation import strategies as strategy_ops
from app.transformation.eligibility import (
    REASON_ELEVATED_PROVENANCE_RISK,
    REASON_EXTERNAL_VERIFICATION_REQUIRED,
    REASON_HIGH_DAMAGE_RISK,
    REASON_NO_SUBSTANTIVE_STRATEGY,
    REASON_THIRD_PARTY_TRANSFORMATION_REQUIRED,
    context_blocker,
    derive_source_moment,
    is_third_party,
    platform_risk_snapshot,
    policy_provenance_blocker,
    transcript_blocker,
    transformation_necessity,
)
from app.transformation.fingerprints import (
    build_output_fingerprint_payload,
    provider_input_fingerprint,
    strategy_fingerprint,
    transformation_output_fingerprint,
)
from app.transformation.policy import (
    DEFAULT_CONFIG,
    Stage40Config,
    is_structure_complex,
)
from app.transformation.providers import (
    DeterministicTransformationProvider,
    TransformationProvider,
    TransformationProviderError,
    TransformationStrategyRequest,
    serialize_provider_result,
    transformation_prompt_hash,
)
from app.transformation.types import (
    AnalysisAssessments,
    SourceMoment,
    StrategyAssessments,
    StrategyDraft,
    TransformationInputs,
    TransformationOutcome,
    TransformationProviderResult,
    TransformationProviderStrategy,
    clamp,
)

_STATUS_DETERMINISTIC = "DETERMINISTIC"
_STATUS_NO_PROVIDER = "NO_PROVIDER"
_STATUS_PROVIDER_OK = "PROVIDER_OK"
_STATUS_PROVIDER_DEGRADED = "PROVIDER_DEGRADED"
_STATUS_RATE_LIMITED = "RATE_LIMITED"
_STATUS_REUSED = "REUSED"
_STATUS_SKIPPED = "SKIPPED_LOW_QUALITY"
_STATUS_NOT_CALLED = "NOT_CALLED"

_COMPLEX_CONTENT_TYPES = {
    ContentType.DEBATE,
    ContentType.NEWS_CURRENT_EVENT,
    ContentType.ANALYSIS,
    ContentType.CONTROVERSIAL_OPINION,
}


def compute_provider_route(
    inputs: TransformationInputs, structure: SourceMoment, necessity: float
) -> str:
    """Pure deterministic tier route: 'STRONG' or 'ROUTINE'."""

    complex_case = (
        is_third_party(inputs)
        or inputs.content_type in _COMPLEX_CONTENT_TYPES
        or is_structure_complex(structure.structure)
        or necessity >= 0.60
    )
    return "STRONG" if complex_case else "ROUTINE"


def _is_complex_case(
    inputs: TransformationInputs, structure: SourceMoment, necessity: float
) -> bool:
    return compute_provider_route(inputs, structure, necessity) == "STRONG"


class TransformationEligibilityService:
    """Evaluate one candidate's Stage 4.0 eligibility and strategy directions."""

    def __init__(
        self,
        *,
        config: Stage40Config = DEFAULT_CONFIG,
        provider: TransformationProvider | None = None,
        provider_identity: Mapping[str, object] | None = None,
        mode: SemanticProviderMode = SemanticProviderMode.DETERMINISTIC,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        # Stable configured identity, independent of transient availability.
        self._configured_identity = (
            dict(provider_identity) if provider_identity is not None else None
        )
        self._mode = mode
        self._is_cancelled = is_cancelled or (lambda: False)

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise _Cancelled()

    def evaluate(
        self,
        inputs: TransformationInputs,
        *,
        input_fingerprint: str,
        provider_route: str | None = None,
        reuse: tuple[str, TransformationProviderResult] | None = None,
    ) -> TransformationOutcome:
        self._check_cancelled()
        structure = derive_source_moment(inputs, self._config)
        necessity = transformation_necessity(inputs)

        blocked = self._blocker_outcome(inputs, structure, necessity, input_fingerprint)
        if blocked is not None:
            return blocked

        route = provider_route or compute_provider_route(inputs, structure, necessity)
        deterministic = strategy_ops.discover_strategies(inputs, self._config, structure)
        provider_status = _STATUS_DETERMINISTIC
        provider_evidence: dict[str, object] = {}
        provider_fp = ""

        request = self._build_request(inputs, structure, necessity)
        configured_identity = self._provider_identity()
        if self._mode is not SemanticProviderMode.DETERMINISTIC and reuse is not None:
            # Reuse accepted hosted work when the stable configured identity
            # matches, even if the provider is temporarily unavailable now.
            reuse_fp = provider_input_fingerprint(
                _request_fingerprint_payload(request, configured_identity, route)
            )
            if reuse[0] == reuse_fp:
                merged = self._merge_with_provider(
                    deterministic, reuse[1].strategies, inputs, structure
                )
                return self._finalize(
                    inputs,
                    structure,
                    necessity,
                    merged,
                    _STATUS_REUSED,
                    {"reused": True},
                    reuse_fp,
                    input_fingerprint,
                    cache_eligible=True,
                )
        if self._mode is not SemanticProviderMode.DETERMINISTIC and self._provider is None:
            # Provider unavailable (e.g. missing key): deterministic results stand.
            return self._finalize(
                inputs,
                structure,
                necessity,
                deterministic,
                _STATUS_NO_PROVIDER,
                {"reason": "provider unavailable"},
                provider_fp,
                input_fingerprint,
                cache_eligible=True,
            )
        if not self._should_call_provider(inputs, deterministic, structure, necessity):
            if self._mode is not SemanticProviderMode.DETERMINISTIC:
                provider_status = _STATUS_SKIPPED
            return self._finalize(
                inputs,
                structure,
                necessity,
                deterministic,
                provider_status,
                provider_evidence,
                provider_fp,
                input_fingerprint,
                cache_eligible=True,
            )

        # One hosted call at most; tier chosen deterministically before the call.
        assert self._provider is not None
        tier = getattr(self._provider, "select_tier", None)
        if callable(tier):
            tier([request])
        provider_fp = provider_input_fingerprint(
            _request_fingerprint_payload(request, configured_identity, route)
        )
        provider_attempted = True
        try:
            self._check_cancelled()
            results = self._provider.discover([request])
            self._check_cancelled()
            provider_status = _STATUS_PROVIDER_OK
            result = results.get(inputs.candidate_id)
            if result is not None:
                provider_evidence = {
                    "provider_notes": result.notes,
                    "provider_confidence": result.confidence,
                    "strategy_count": len(result.strategies),
                    "provider_result": serialize_provider_result(result),
                }
                deterministic = self._merge_with_provider(
                    deterministic, result.strategies, inputs, structure
                )
        except _Cancelled:
            raise
        except TransformationProviderError as error:
            provider_status = (
                _STATUS_RATE_LIMITED
                if error.category == ProviderErrorCategory.RATE_LIMITED.value
                else _STATUS_PROVIDER_DEGRADED
            )
            provider_evidence = {"error_category": error.category}

        cache_eligible = not provider_attempted or provider_status in {
            _STATUS_PROVIDER_OK,
            _STATUS_REUSED,
        }
        return self._finalize(
            inputs,
            structure,
            necessity,
            deterministic,
            provider_status,
            provider_evidence,
            provider_fp,
            input_fingerprint,
            cache_eligible=cache_eligible,
        )

    # ---- blockers ---------------------------------------------------------

    def _blocker_outcome(
        self,
        inputs: TransformationInputs,
        structure: SourceMoment,
        necessity: float,
        input_fingerprint: str,
    ) -> TransformationOutcome | None:
        reason = transcript_blocker(inputs, self._config)
        outcome: TransformationEligibilityOutcome | None = None
        if reason is not None:
            outcome = TransformationEligibilityOutcome.INSUFFICIENT_TRANSCRIPT_CONFIDENCE
        else:
            reason = context_blocker(inputs, self._config)
            if reason is not None:
                outcome = TransformationEligibilityOutcome.INSUFFICIENT_CONTEXT
            else:
                reason = policy_provenance_blocker(inputs)
                if reason is not None:
                    outcome = TransformationEligibilityOutcome.UNRESOLVED_POLICY_OR_PROVENANCE_RISK
        if outcome is None:
            return None
        return self._finalize(
            inputs,
            structure,
            necessity,
            [],
            _STATUS_NOT_CALLED,
            {},
            "",
            input_fingerprint,
            cache_eligible=True,
            forced_outcome=outcome,
            forced_reasons=(reason,),
        )

    # ---- provider ---------------------------------------------------------

    def _provider_identity(self) -> dict[str, object]:
        if self._configured_identity is not None:
            return dict(self._configured_identity)
        if self._provider is None:
            return DeterministicTransformationProvider().runtime_identity()
        identity = getattr(self._provider, "runtime_identity", None)
        return dict(identity()) if callable(identity) else {"provider": "unknown"}

    def _should_call_provider(
        self,
        inputs: TransformationInputs,
        deterministic: Sequence[StrategyDraft],
        structure: SourceMoment,
        necessity: float,
    ) -> bool:
        if self._mode is SemanticProviderMode.DETERMINISTIC or self._provider is None:
            return False
        recommended = [
            item for item in deterministic if item.disposition is StrategyDisposition.RECOMMENDED
        ]
        complex_case = _is_complex_case(inputs, structure, necessity)
        if not recommended and not complex_case:
            # Obvious no-strategy / low-value cases never spend provider quota.
            return False
        return True

    def _build_request(
        self,
        inputs: TransformationInputs,
        structure: SourceMoment,
        necessity: float,
    ) -> TransformationStrategyRequest:
        return TransformationStrategyRequest(
            candidate_id=inputs.candidate_id,
            content_type=inputs.content_type.value,
            source_moment_structure=structure.structure.value,
            refined_transcript=inputs.transcript[: self._config.provider_max_input_characters],
            context_text=" ".join(inputs.context_segments)[: self._config.max_context_characters],
            hooks=tuple(
                str(hook.get("text", ""))[: self._config.max_context_characters]
                for hook in inputs.hooks
                if isinstance(hook, Mapping)
            ),
            idea_summary=inputs.idea_summary[: self._config.max_context_characters],
            topic_summary=inputs.topic_summary[: self._config.max_context_characters],
            dialect_profile=inputs.dialect_profile,
            code_switch_tokens=_code_switch_tokens(inputs.code_switch),
            rights_risk=inputs.rights_risk.value,
            originality_risk=inputs.originality_risk.value,
            transformation_required=is_third_party(inputs)
            or inputs.originality_risk is OriginalityRisk.TRANSFORMATION_REQUIRED,
            complex_case=_is_complex_case(inputs, structure, necessity),
        )

    def _merge_with_provider(
        self,
        deterministic: Sequence[StrategyDraft],
        provider_strategies: Sequence[TransformationProviderStrategy],
        inputs: TransformationInputs,
        structure: SourceMoment,
    ) -> list[StrategyDraft]:
        drafts = list(deterministic)
        existing_types = {item.strategy_type for item in drafts}
        for provider_strategy in provider_strategies:
            if provider_strategy.strategy_type in existing_types:
                continue
            candidate = self._provider_to_draft(provider_strategy, inputs, structure)
            if candidate is None:
                continue
            validated = strategy_ops.apply_hard_gates(
                candidate, inputs, structure.structure, self._config
            )
            drafts.append(validated)
            existing_types.add(provider_strategy.strategy_type)
        return drafts

    def _provider_to_draft(
        self,
        provider_strategy: TransformationProviderStrategy,
        inputs: TransformationInputs,
        structure: SourceMoment,
    ) -> StrategyDraft | None:
        value_kind = provider_strategy.substantive_value_kind
        if value_kind is None:
            return None
        assessments = StrategyAssessments(
            retention_preservation=_or_default(provider_strategy.retention_preservation, 0.5),
            source_moment_damage_risk=_or_default(provider_strategy.source_moment_damage_risk, 0.5),
            added_value_density=_or_default(provider_strategy.added_value_density, 0.4),
            originality_potential=_or_default(provider_strategy.originality_potential, 0.4),
            source_dominance_risk=_or_default(provider_strategy.source_dominance_risk, 0.5),
            generic_filler_risk=_or_default(provider_strategy.generic_filler_risk, 0.5),
            redundant_commentary_risk=_or_default(provider_strategy.redundant_commentary_risk, 0.5),
            template_staleness_risk=_or_default(provider_strategy.template_staleness_risk, 0.5),
        )
        requirements = provider_strategy.verification_requirements
        external = provider_strategy.external_verification_requirement
        if external is ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION and not (
            requirements
        ):
            requirements = (
                "Provider flagged an external factual dependency; Stage 4.1/4.2 must verify it "
                "with an authoritative source before presentation.",
            )
        return StrategyDraft(
            strategy_type=provider_strategy.strategy_type,
            disposition=provider_strategy.disposition,
            rank=0,
            intensity=provider_strategy.intensity,
            direction_summary=provider_strategy.direction_summary,
            added_value_focus=provider_strategy.added_value_focus,
            substantive_value_kind=value_kind,
            source_moment_role="HERO",
            preservation_requirements=provider_strategy.preservation_requirements
            or ("Keep the strongest source moment as the hero.",),
            assessments=assessments,
            external_verification_requirement=external,
            verification_requirements=requirements,
            rejection_reasons=provider_strategy.rejection_reasons,
            confidence=provider_strategy.confidence,
            origin=StrategyOrigin.PROVIDER,
        )

    # ---- finalization -----------------------------------------------------

    def _finalize(
        self,
        inputs: TransformationInputs,
        structure: SourceMoment,
        necessity: float,
        drafts: Sequence[StrategyDraft],
        provider_status: str,
        provider_evidence: Mapping[str, object],
        provider_fp: str,
        input_fingerprint: str,
        *,
        cache_eligible: bool,
        forced_outcome: TransformationEligibilityOutcome | None = None,
        forced_reasons: tuple[str | None, ...] = (),
    ) -> TransformationOutcome:
        recommended = [
            item for item in drafts if item.disposition is StrategyDisposition.RECOMMENDED
        ]
        rejected = [item for item in drafts if item.disposition is StrategyDisposition.REJECTED]
        recommended.sort(
            key=lambda item: (item.rank if item.rank > 0 else 999, item.strategy_type.value)
        )
        recommended = recommended[: self._config.max_recommended_strategies]
        rejected = rejected[: self._config.max_rejected_strategies]
        ranked = [_rank(item, index) for index, item in enumerate(recommended, start=1)] + rejected

        if forced_outcome is not None:
            eligibility = forced_outcome
            reasons = tuple(reason for reason in forced_reasons if reason)
        elif not recommended:
            eligibility = TransformationEligibilityOutcome.NO_TRANSFORMATION_STRATEGY_WORTH_USING
            reasons = (REASON_NO_SUBSTANTIVE_STRATEGY,)
        elif is_third_party(inputs):
            eligibility = TransformationEligibilityOutcome.TRANSFORMATION_REQUIRED
            reasons = (REASON_THIRD_PARTY_TRANSFORMATION_REQUIRED,)
        else:
            caution = self._caution_reasons(inputs, recommended)
            if caution:
                eligibility = TransformationEligibilityOutcome.ELIGIBLE_WITH_CAUTION
                reasons = tuple(caution)
            else:
                eligibility = TransformationEligibilityOutcome.ELIGIBLE_FOR_TRANSFORMATION
                reasons = ()

        assessments = self._analysis_assessments(inputs, structure, necessity, recommended)
        intensity = recommended[0].intensity if recommended else None
        platform_risk = platform_risk_snapshot(
            inputs,
            source_dominance_risk=assessments.source_dominance_risk,
            template_staleness_risk=assessments.template_staleness_risk,
            presentation_only=not recommended,
        )
        if is_third_party(inputs):
            platform_risk["transformation_required"] = True
        output_fp = transformation_output_fingerprint(
            build_output_fingerprint_payload(
                eligibility_outcome=eligibility.value,
                intensity=intensity.value if intensity else None,
                strategies=ranked,
                source_moment=structure.as_dict(),
                platform_risk=platform_risk,
                assessments=assessments.as_dict(),
            )
        )
        metrics = {
            "candidates_evaluated": 1,
            "deterministic_rejects": len(rejected),
            "recommended_count": len(recommended),
            "rejected_count": len(rejected),
            "strategy_types": sorted({item.strategy_type.value for item in ranked}),
            "transformation_necessity": round(necessity, 4),
            "provider_calls": 1 if provider_status in {_STATUS_PROVIDER_OK} else 0,
        }
        return TransformationOutcome(
            eligibility_outcome=eligibility,
            eligibility_reasons=tuple(reasons),
            assessments=assessments,
            source_moment=structure,
            platform_risk=platform_risk,
            intensity=intensity,
            strategies=tuple(ranked),
            provider_status=provider_status,
            provider_evidence=dict(provider_evidence),
            provider_input_fingerprint=provider_fp,
            cache_eligible=cache_eligible,
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fp,
            metrics=metrics,
        )

    def _caution_reasons(
        self, inputs: TransformationInputs, recommended: Sequence[StrategyDraft]
    ) -> list[str]:
        reasons: list[str] = []
        if inputs.rights_risk is RightsRisk.ELEVATED:
            reasons.append(REASON_ELEVATED_PROVENANCE_RISK)
        if inputs.originality_risk is OriginalityRisk.TRANSFORMATION_REQUIRED:
            reasons.append(REASON_THIRD_PARTY_TRANSFORMATION_REQUIRED)
        if any(
            item.external_verification_requirement
            is ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION
            for item in recommended
        ):
            reasons.append(REASON_EXTERNAL_VERIFICATION_REQUIRED)
        if recommended and recommended[0].assessments.source_moment_damage_risk >= 0.6:
            reasons.append(REASON_HIGH_DAMAGE_RISK)
        return list(dict.fromkeys(reasons))

    def _analysis_assessments(
        self,
        inputs: TransformationInputs,
        structure: SourceMoment,
        necessity: float,
        recommended: Sequence[StrategyDraft],
    ) -> AnalysisAssessments:
        if not recommended:
            return AnalysisAssessments(
                retention_preservation=0.5,
                source_moment_damage_risk=0.5,
                added_value_density=0.0,
                transformation_necessity=necessity,
                transformation_potential=0.0,
                originality_potential=0.0,
                source_dominance_risk=0.5,
                generic_ai_filler_risk=0.5,
                redundant_commentary_risk=0.5,
                template_staleness_risk=0.5,
            )
        top = recommended[0]
        return AnalysisAssessments(
            retention_preservation=max(
                item.assessments.retention_preservation for item in recommended
            ),
            source_moment_damage_risk=top.assessments.source_moment_damage_risk,
            added_value_density=max(item.assessments.added_value_density for item in recommended),
            transformation_necessity=necessity,
            transformation_potential=max(
                item.assessments.originality_potential for item in recommended
            ),
            originality_potential=max(
                item.assessments.originality_potential for item in recommended
            ),
            source_dominance_risk=top.assessments.source_dominance_risk,
            generic_ai_filler_risk=min(
                item.assessments.generic_filler_risk for item in recommended
            ),
            redundant_commentary_risk=min(
                item.assessments.redundant_commentary_risk for item in recommended
            ),
            template_staleness_risk=top.assessments.template_staleness_risk,
        )


def _or_default(value: float | None, default: float) -> float:
    if value is None:
        return default
    return clamp(value)


def _code_switch_tokens(code_switch: Mapping[str, object]) -> tuple[str, ...]:
    raw = code_switch.get("tokens")
    if not isinstance(raw, list):
        return ()
    return tuple(str(token) for token in raw if isinstance(token, str))


def _rank(draft: StrategyDraft, rank: int) -> StrategyDraft:
    return StrategyDraft(
        strategy_type=draft.strategy_type,
        disposition=draft.disposition,
        rank=rank,
        intensity=draft.intensity,
        direction_summary=draft.direction_summary,
        added_value_focus=draft.added_value_focus,
        substantive_value_kind=draft.substantive_value_kind,
        source_moment_role=draft.source_moment_role,
        preservation_requirements=draft.preservation_requirements,
        assessments=draft.assessments,
        external_verification_requirement=draft.external_verification_requirement,
        verification_requirements=draft.verification_requirements,
        rejection_reasons=draft.rejection_reasons,
        confidence=draft.confidence,
        origin=draft.origin,
        provider_evidence=draft.provider_evidence,
    )


def _request_fingerprint_payload(
    request: TransformationStrategyRequest,
    provider_identity: Mapping[str, object],
    route: str | None,
) -> dict[str, object]:
    return {
        "request": request.to_payload(),
        "provider_identity": dict(provider_identity),
        "route": route,
    }


class _Cancelled(RuntimeError):
    """Internal cooperative cancellation marker."""


def build_strategy_fingerprint(
    draft: StrategyDraft, analysis_output_fingerprint: str, policy_version: str
) -> str:
    """Stable per-strategy identity for Stage 4.1 change detection."""

    payload = {
        "analysis_output_fingerprint": analysis_output_fingerprint,
        "policy_version": policy_version,
        "strategy": draft.fingerprint_payload(),
    }
    return strategy_fingerprint(payload)


__all__ = [
    "TransformationEligibilityService",
    "build_strategy_fingerprint",
    "compute_provider_route",
    "transformation_prompt_hash",
]
