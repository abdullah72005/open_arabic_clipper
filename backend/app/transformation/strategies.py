"""Deterministic Stage 4.0 strategy discovery, hard validation, and ranking.

Discovery is a pure deterministic pass over content suitability and bounded
evidence. Hard constraints precede ranking: a direction is rejected when it has
no substantive value, relies on presentation, is mostly paraphrase, is generic
filler, distorts the source, fakes a dramatic hook, damages retention, needs
unavailable context, cannot reach the required originality, or depends on an
unverified external fact without marking it. There is no single
"transformation score"; ranking uses ordered transparent criteria.
"""

from __future__ import annotations

from app.core.enums import (
    ExternalFactRequirement,
    SourceMomentStructure,
    StrategyDisposition,
    StrategyOrigin,
    SubstantiveValueKind,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.transformation.eligibility import required_originality
from app.transformation.policy import (
    CONTENT_SUITABILITY,
    DISTORTION_MARKERS,
    FAKE_HOOK_MARKERS,
    NAME_ONLY_STRATEGIES,
    PARAPHRASE_MARKERS,
    PRESENTATION_ONLY_CHANGES,
    SCRIPT_SHAPE_MARKERS,
    STRATEGY_VALUE_KINDS,
    Stage40Config,
)
from app.transformation.types import (
    SourceMoment,
    StrategyAssessments,
    StrategyDraft,
    TransformationInputs,
    clamp,
)

# Rejection reason codes (bounded).
REJECT_NO_SUBSTANTIVE_VALUE = "NO_SUBSTANTIVE_VALUE"
REJECT_PRESENTATION_ONLY = "PRESENTATION_ONLY"
REJECT_PARAPHRASE = "PARAPHRASE_ONLY"
REJECT_GENERIC_FILLER = "GENERIC_FILLER"
REJECT_DISTORTION = "SOURCE_DISTORTION"
REJECT_FAKE_HOOK = "FAKE_DRAMATIC_HOOK"
REJECT_RETENTION_DAMAGE = "HOOK_PAYOFF_DAMAGE"
REJECT_MISSING_CONTEXT = "MISSING_CONTEXT_UNSAFE"
REJECT_INSUFFICIENT_ORIGINALITY = "INSUFFICIENT_ORIGINALITY"
REJECT_EXTERNAL_FACT = "UNVERIFIED_EXTERNAL_FACT"
REJECT_SCRIPT_SHAPED = "SCRIPT_OR_TIMELINE_SHAPED"
REJECT_LOW_VALUE_DENSITY = "LOW_ADDED_VALUE_DENSITY"
REJECT_TEMPLATE_STALENESS = "TEMPLATE_STALENESS"

_INTENSITY: dict[TransformationStrategyType, TransformationIntensity] = {
    TransformationStrategyType.SOURCE_LED_MINIMAL: TransformationIntensity.MINIMAL,
    TransformationStrategyType.CONTEXT_HOOK: TransformationIntensity.MINIMAL,
    TransformationStrategyType.HOOK_PLUS_TAKEAWAY: TransformationIntensity.MODERATE,
    TransformationStrategyType.EXPLANATORY: TransformationIntensity.MODERATE,
    TransformationStrategyType.COMPARISON: TransformationIntensity.MODERATE,
    TransformationStrategyType.COUNTERPOINT: TransformationIntensity.MODERATE,
    TransformationStrategyType.SOURCE_AS_EVIDENCE: TransformationIntensity.MODERATE,
    TransformationStrategyType.NEWS_CONTEXT: TransformationIntensity.MODERATE,
    TransformationStrategyType.REACTION_FRAMING: TransformationIntensity.MODERATE,
    TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY: TransformationIntensity.MODERATE,
    TransformationStrategyType.ANALYSIS: TransformationIntensity.STRONG,
    TransformationStrategyType.COMMENTARY: TransformationIntensity.STRONG,
    TransformationStrategyType.SUMMARY: TransformationIntensity.STRONG,
    TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION: TransformationIntensity.STRONG,
    TransformationStrategyType.DEBATE_CONTEXT: TransformationIntensity.STRONG,
}
_INTENSITY_ORDER = {
    TransformationIntensity.MINIMAL: 0,
    TransformationIntensity.MODERATE: 1,
    TransformationIntensity.STRONG: 2,
}

_BASE_ADDED_VALUE: dict[TransformationStrategyType, float] = {
    TransformationStrategyType.SOURCE_LED_MINIMAL: 0.42,
    TransformationStrategyType.CONTEXT_HOOK: 0.55,
    TransformationStrategyType.HOOK_PLUS_TAKEAWAY: 0.62,
    TransformationStrategyType.EXPLANATORY: 0.65,
    TransformationStrategyType.COMPARISON: 0.62,
    TransformationStrategyType.COUNTERPOINT: 0.60,
    TransformationStrategyType.SOURCE_AS_EVIDENCE: 0.58,
    TransformationStrategyType.NEWS_CONTEXT: 0.60,
    TransformationStrategyType.REACTION_FRAMING: 0.44,
    TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY: 0.60,
    TransformationStrategyType.ANALYSIS: 0.62,
    TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION: 0.58,
    TransformationStrategyType.DEBATE_CONTEXT: 0.55,
    TransformationStrategyType.COMMENTARY: 0.40,
    TransformationStrategyType.SUMMARY: 0.36,
}
_BASE_ORIGINALITY: dict[TransformationStrategyType, float] = {
    TransformationStrategyType.SOURCE_LED_MINIMAL: 0.30,
    TransformationStrategyType.CONTEXT_HOOK: 0.55,
    TransformationStrategyType.HOOK_PLUS_TAKEAWAY: 0.58,
    TransformationStrategyType.EXPLANATORY: 0.62,
    TransformationStrategyType.COMPARISON: 0.60,
    TransformationStrategyType.COUNTERPOINT: 0.65,
    TransformationStrategyType.SOURCE_AS_EVIDENCE: 0.66,
    TransformationStrategyType.NEWS_CONTEXT: 0.58,
    TransformationStrategyType.REACTION_FRAMING: 0.40,
    TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY: 0.58,
    TransformationStrategyType.ANALYSIS: 0.70,
    TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION: 0.60,
    TransformationStrategyType.DEBATE_CONTEXT: 0.58,
    TransformationStrategyType.COMMENTARY: 0.60,
    TransformationStrategyType.SUMMARY: 0.34,
}
_BASE_FILLER: dict[TransformationStrategyType, float] = {
    TransformationStrategyType.SOURCE_LED_MINIMAL: 0.20,
    TransformationStrategyType.CONTEXT_HOOK: 0.20,
    TransformationStrategyType.HOOK_PLUS_TAKEAWAY: 0.28,
    TransformationStrategyType.EXPLANATORY: 0.25,
    TransformationStrategyType.COMPARISON: 0.28,
    TransformationStrategyType.COUNTERPOINT: 0.30,
    TransformationStrategyType.SOURCE_AS_EVIDENCE: 0.22,
    TransformationStrategyType.NEWS_CONTEXT: 0.25,
    TransformationStrategyType.REACTION_FRAMING: 0.45,
    TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY: 0.28,
    TransformationStrategyType.ANALYSIS: 0.28,
    TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION: 0.30,
    TransformationStrategyType.DEBATE_CONTEXT: 0.32,
    TransformationStrategyType.COMMENTARY: 0.52,
    TransformationStrategyType.SUMMARY: 0.58,
}
_BASE_REDUNDANCY: dict[TransformationStrategyType, float] = {
    TransformationStrategyType.SOURCE_LED_MINIMAL: 0.15,
    TransformationStrategyType.CONTEXT_HOOK: 0.20,
    TransformationStrategyType.HOOK_PLUS_TAKEAWAY: 0.30,
    TransformationStrategyType.EXPLANATORY: 0.32,
    TransformationStrategyType.COMPARISON: 0.35,
    TransformationStrategyType.COUNTERPOINT: 0.35,
    TransformationStrategyType.SOURCE_AS_EVIDENCE: 0.30,
    TransformationStrategyType.NEWS_CONTEXT: 0.30,
    TransformationStrategyType.REACTION_FRAMING: 0.50,
    TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY: 0.32,
    TransformationStrategyType.ANALYSIS: 0.35,
    TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION: 0.38,
    TransformationStrategyType.DEBATE_CONTEXT: 0.40,
    TransformationStrategyType.COMMENTARY: 0.58,
    TransformationStrategyType.SUMMARY: 0.62,
}
_BASE_TEMPLATE: dict[TransformationStrategyType, float] = {
    TransformationStrategyType.SOURCE_LED_MINIMAL: 0.20,
    TransformationStrategyType.CONTEXT_HOOK: 0.22,
    TransformationStrategyType.HOOK_PLUS_TAKEAWAY: 0.30,
    TransformationStrategyType.EXPLANATORY: 0.28,
    TransformationStrategyType.COMPARISON: 0.30,
    TransformationStrategyType.COUNTERPOINT: 0.30,
    TransformationStrategyType.SOURCE_AS_EVIDENCE: 0.25,
    TransformationStrategyType.NEWS_CONTEXT: 0.30,
    TransformationStrategyType.REACTION_FRAMING: 0.46,
    TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY: 0.32,
    TransformationStrategyType.ANALYSIS: 0.30,
    TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION: 0.34,
    TransformationStrategyType.DEBATE_CONTEXT: 0.34,
    TransformationStrategyType.COMMENTARY: 0.50,
    TransformationStrategyType.SUMMARY: 0.55,
}
_BASE_RETENTION: dict[TransformationStrategyType, float] = {
    TransformationStrategyType.SOURCE_LED_MINIMAL: 0.90,
    TransformationStrategyType.CONTEXT_HOOK: 0.82,
    TransformationStrategyType.HOOK_PLUS_TAKEAWAY: 0.78,
    TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY: 0.72,
    TransformationStrategyType.SOURCE_AS_EVIDENCE: 0.75,
    TransformationStrategyType.COUNTERPOINT: 0.68,
    TransformationStrategyType.NEWS_CONTEXT: 0.68,
    TransformationStrategyType.ANALYSIS: 0.66,
    TransformationStrategyType.EXPLANATORY: 0.64,
    TransformationStrategyType.DEBATE_CONTEXT: 0.65,
    TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION: 0.63,
    TransformationStrategyType.COMPARISON: 0.60,
    TransformationStrategyType.COMMENTARY: 0.60,
    TransformationStrategyType.SUMMARY: 0.56,
    TransformationStrategyType.REACTION_FRAMING: 0.55,
}
_BASE_DOMINANCE: dict[TransformationStrategyType, float] = {
    TransformationStrategyType.SOURCE_LED_MINIMAL: 0.80,
    TransformationStrategyType.CONTEXT_HOOK: 0.62,
    TransformationStrategyType.HOOK_PLUS_TAKEAWAY: 0.55,
    TransformationStrategyType.EXPLANATORY: 0.45,
    TransformationStrategyType.COMPARISON: 0.42,
    TransformationStrategyType.COUNTERPOINT: 0.40,
    TransformationStrategyType.SOURCE_AS_EVIDENCE: 0.58,
    TransformationStrategyType.NEWS_CONTEXT: 0.55,
    TransformationStrategyType.REACTION_FRAMING: 0.52,
    TransformationStrategyType.QUESTION_EXPLANATION_TAKEAWAY: 0.48,
    TransformationStrategyType.ANALYSIS: 0.38,
    TransformationStrategyType.CLAIM_CONTEXT_CONCLUSION: 0.42,
    TransformationStrategyType.DEBATE_CONTEXT: 0.40,
    TransformationStrategyType.COMMENTARY: 0.44,
    TransformationStrategyType.SUMMARY: 0.60,
}
_VALUE_KIND_FOR: dict[str, SubstantiveValueKind] = {
    "MISSING_CONTEXT": SubstantiveValueKind.MISSING_CONTEXT,
    "INFERENCE": SubstantiveValueKind.INFERENCE,
    "EXPLANATION": SubstantiveValueKind.EXPLANATION,
    "COMPARISON": SubstantiveValueKind.COMPARISON,
    "COUNTERPOINT": SubstantiveValueKind.COUNTERPOINT,
    "VERIFICATION": SubstantiveValueKind.VERIFICATION_CORRECTION,
    "SYNTHESIS": SubstantiveValueKind.SYNTHESIS,
    "AUTHORED_THESIS": SubstantiveValueKind.AUTHORED_THESIS,
    "USEFUL_TAKEAWAY": SubstantiveValueKind.USEFUL_TAKEAWAY,
    "SOURCE_AS_EVIDENCE": SubstantiveValueKind.SOURCE_AS_EVIDENCE,
}


def _substance(inputs: TransformationInputs) -> float:
    """Candidate-specific evidence substance in [0, 1] (never a universal template)."""

    signal = 0.0
    combined = f"{inputs.idea_summary} {inputs.topic_summary}".strip()
    if len(combined.split()) >= 6:
        signal += 0.35
    if len(inputs.idea_summary.split()) >= 4:
        signal += 0.20
    if inputs.hooks:
        signal += 0.15
    if inputs.duration >= 15.0:
        signal += 0.15
    if inputs.moment_density_score >= 0.4:
        signal += 0.10
    if inputs.entity_evidence:
        signal += 0.10
    return clamp(signal)


def _value_kind(strategy: TransformationStrategyType) -> SubstantiveValueKind:
    kinds = STRATEGY_VALUE_KINDS.get(strategy, ("SYNTHESIS",))
    return _VALUE_KIND_FOR.get(kinds[0], SubstantiveValueKind.SYNTHESIS)


def _excerpt(inputs: TransformationInputs, limit: int = 120) -> str:
    lead = (inputs.idea_summary or inputs.topic_summary).strip()
    transcript = inputs.transcript.strip()
    combined = f"{lead} {transcript}".strip()
    return combined[:limit].rstrip()


def _added_value_focus(
    strategy: TransformationStrategyType, kind: SubstantiveValueKind, inputs: TransformationInputs
) -> str:
    excerpt = _excerpt(inputs)
    templates = {
        SubstantiveValueKind.MISSING_CONTEXT: "Supply the context a viewer needs: {e}",
        SubstantiveValueKind.INFERENCE: "State the original inference here: {e}",
        SubstantiveValueKind.EXPLANATION: "Explain the reasoning behind the claim: {e}",
        SubstantiveValueKind.COMPARISON: "Compare this case to a concrete alternative: {e}",
        SubstantiveValueKind.COUNTERPOINT: "Add a specific counterpoint: {e}",
        SubstantiveValueKind.VERIFICATION_CORRECTION: "Verify or correct a checkable detail: {e}",
        SubstantiveValueKind.SYNTHESIS: "Draw the synthesis the source leaves implicit: {e}",
        SubstantiveValueKind.AUTHORED_THESIS: "Advance an authored thesis: {e}",
        SubstantiveValueKind.USEFUL_TAKEAWAY: "Give one actionable takeaway: {e}",
        SubstantiveValueKind.SOURCE_AS_EVIDENCE: "Use this moment as evidence: {e}",
    }
    template = templates.get(kind, templates[SubstantiveValueKind.SYNTHESIS])
    return template.format(e=excerpt)


def _direction_summary(strategy: TransformationStrategyType, inputs: TransformationInputs) -> str:
    moment = inputs.transcript.strip()[:80]
    return f"{strategy.value.replace('_', ' ').title()} built on the moment: {moment}"


def _moment_quality(inputs: TransformationInputs) -> float:
    """Bounded moment-quality signal from Stage 3 scores, in [0, 1]."""

    values = (
        inputs.clip_score,
        inputs.short_form_score,
        inputs.moment_density_score,
        inputs.ending_quality_score,
        inputs.loopability_score,
    )
    return clamp(sum(clamp(value) for value in values) / len(values))


def _assessments(
    strategy: TransformationStrategyType,
    inputs: TransformationInputs,
    config: Stage40Config,
) -> StrategyAssessments:
    substance = _substance(inputs)
    quality = _moment_quality(inputs)
    intensity = _INTENSITY[strategy]
    value_factor = 0.40 + 0.60 * quality
    added = clamp((_BASE_ADDED_VALUE[strategy] + 0.10 * substance) * value_factor)
    originality = clamp((_BASE_ORIGINALITY[strategy] + 0.12 * substance) * (0.60 + 0.40 * quality))
    filler = clamp(_BASE_FILLER[strategy] - 0.15 * substance + 0.25 * (1.0 - quality))
    redundancy = clamp(_BASE_REDUNDANCY[strategy] - 0.12 * substance + 0.20 * (1.0 - quality))
    template = clamp(_BASE_TEMPLATE[strategy] - 0.08 * substance + 0.20 * (1.0 - quality))
    retention = clamp(_BASE_RETENTION[strategy] + 0.05 * substance)
    base_damage = {
        TransformationIntensity.MINIMAL: 0.15,
        TransformationIntensity.MODERATE: 0.35,
        TransformationIntensity.STRONG: 0.55,
    }[intensity]
    short = inputs.duration <= config.short_moment_seconds
    if short and intensity is TransformationIntensity.STRONG:
        base_damage += 0.25
    elif short and intensity is TransformationIntensity.MODERATE:
        base_damage += 0.10
    damage = clamp(base_damage - 0.05 * substance)
    dominance = clamp(_BASE_DOMINANCE[strategy] + 0.05 * substance)
    return StrategyAssessments(
        retention_preservation=retention,
        source_moment_damage_risk=damage,
        added_value_density=added,
        originality_potential=originality,
        source_dominance_risk=dominance,
        generic_filler_risk=filler,
        redundant_commentary_risk=redundancy,
        template_staleness_risk=template,
    )


def is_presentation_only(text: str) -> bool:
    lowered = text.casefold()
    return any(change in lowered for change in PRESENTATION_ONLY_CHANGES)


def has_fake_hook(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in FAKE_HOOK_MARKERS)


def has_distortion(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in DISTORTION_MARKERS)


def is_paraphrase(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in PARAPHRASE_MARKERS)


def looks_like_script(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in SCRIPT_SHAPE_MARKERS)


def _external_requirement(
    strategy: TransformationStrategyType,
    structure: SourceMomentStructure,
) -> tuple[ExternalFactRequirement, tuple[str, ...]]:
    if (
        strategy is TransformationStrategyType.NEWS_CONTEXT
        or structure is SourceMomentStructure.NEWS
    ):
        return (
            ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION,
            (
                "Verify the referenced current-event fact with an authoritative source in "
                "Stage 4.1/4.2 before presenting it; Stage 4.0 does no research.",
            ),
        )
    return ExternalFactRequirement.NOT_REQUIRED, ()


def apply_hard_gates(
    draft: StrategyDraft,
    inputs: TransformationInputs,
    structure: SourceMomentStructure,
    config: Stage40Config,
) -> StrategyDraft:
    """Apply hard gates. Reject with bounded reasons; never silently keep."""

    reasons: list[str] = []
    combined = f"{draft.direction_summary} {draft.added_value_focus}"
    if is_presentation_only(combined):
        reasons.append(REJECT_PRESENTATION_ONLY)
    if looks_like_script(combined):
        reasons.append(REJECT_SCRIPT_SHAPED)
    if has_fake_hook(combined):
        reasons.append(REJECT_FAKE_HOOK)
    if has_distortion(combined):
        reasons.append(REJECT_DISTORTION)
    if is_paraphrase(combined):
        reasons.append(REJECT_PARAPHRASE)
    if draft.strategy_type in NAME_ONLY_STRATEGIES and _substance(inputs) < 0.35:
        reasons.append(REJECT_NO_SUBSTANTIVE_VALUE)
    if draft.assessments.added_value_density < config.min_added_value_density:
        reasons.append(REJECT_LOW_VALUE_DENSITY)
    if draft.assessments.source_moment_damage_risk > config.max_source_moment_damage:
        reasons.append(REJECT_RETENTION_DAMAGE)
    if draft.assessments.generic_filler_risk > config.max_generic_filler:
        reasons.append(REJECT_GENERIC_FILLER)
    if draft.assessments.redundant_commentary_risk > config.max_redundant_commentary:
        reasons.append(REJECT_PARAPHRASE)
    if draft.assessments.originality_potential < required_originality(inputs, config):
        reasons.append(REJECT_INSUFFICIENT_ORIGINALITY)
    if draft.assessments.template_staleness_risk > config.max_template_staleness:
        reasons.append(REJECT_TEMPLATE_STALENESS)
    if (
        inputs.duration <= config.short_moment_seconds
        and draft.intensity is not TransformationIntensity.MINIMAL
        and draft.assessments.source_moment_damage_risk >= config.short_moment_setup_damage_risk
    ):
        reasons.append(REJECT_RETENTION_DAMAGE)
    requirement = draft.external_verification_requirement
    if requirement is ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION and not (
        draft.verification_requirements
    ):
        reasons.append(REJECT_EXTERNAL_FACT)
    if reasons:
        unique = tuple(dict.fromkeys(reasons))
        return StrategyDraft(
            strategy_type=draft.strategy_type,
            disposition=StrategyDisposition.REJECTED,
            rank=0,
            intensity=draft.intensity,
            direction_summary=draft.direction_summary,
            added_value_focus=draft.added_value_focus,
            substantive_value_kind=draft.substantive_value_kind,
            source_moment_role=draft.source_moment_role,
            preservation_requirements=draft.preservation_requirements,
            assessments=draft.assessments,
            external_verification_requirement=draft.external_verification_requirement,
            verification_requirements=draft.verification_requirements,
            rejection_reasons=unique,
            confidence=clamp(draft.confidence * 0.5),
            origin=draft.origin,
            provider_evidence=draft.provider_evidence,
        )
    return draft


def _preservation_requirements(
    strategy: TransformationStrategyType, inputs: TransformationInputs, config: Stage40Config
) -> tuple[str, ...]:
    requirements = ["Keep the strongest source moment as the hero."]
    if inputs.duration <= config.short_moment_seconds:
        requirements.append("Do not add a long preamble before the source moment.")
        requirements.append("Preserve the payoff timing; keep the source hook early.")
    else:
        requirements.append("Keep the key source claim intact.")
    if strategy in {
        TransformationStrategyType.SOURCE_LED_MINIMAL,
        TransformationStrategyType.REACTION_FRAMING,
        TransformationStrategyType.CONTEXT_HOOK,
    }:
        requirements.append("Do not narrate what is already on screen.")
    if strategy is TransformationStrategyType.SUMMARY:
        requirements.append("Do not simply restate the source in fewer words.")
    requirements.append("Place concise context or implication after the source moment.")
    return tuple(dict.fromkeys(requirements))


def _candidate_drafts(
    inputs: TransformationInputs, config: Stage40Config, structure: SourceMomentStructure
) -> list[StrategyDraft]:
    suitability = CONTENT_SUITABILITY.get(inputs.content_type, ())
    drafts: list[StrategyDraft] = []
    for strategy in suitability:
        kind = _value_kind(strategy)
        requirement, details = _external_requirement(strategy, structure)
        base = StrategyDraft(
            strategy_type=strategy,
            disposition=StrategyDisposition.RECOMMENDED,
            rank=0,
            intensity=_INTENSITY[strategy],
            direction_summary=_direction_summary(strategy, inputs),
            added_value_focus=_added_value_focus(strategy, kind, inputs),
            substantive_value_kind=kind,
            source_moment_role="HERO",
            preservation_requirements=_preservation_requirements(strategy, inputs, config),
            assessments=_assessments(strategy, inputs, config),
            external_verification_requirement=requirement,
            verification_requirements=details,
            confidence=clamp(0.55 + 0.35 * _substance(inputs)),
            origin=StrategyOrigin.DETERMINISTIC,
        )
        drafts.append(apply_hard_gates(base, inputs, structure, config))
    return drafts


def _rank_key(draft: StrategyDraft) -> tuple[float, ...]:
    """Least-intrusive sufficient ordering with a stable enum tie-break."""

    assessments = draft.assessments
    return (
        -assessments.added_value_density,
        -assessments.originality_potential,
        -assessments.retention_preservation,
        assessments.source_moment_damage_risk,
        assessments.generic_filler_risk,
        assessments.redundant_commentary_risk,
        float(_INTENSITY_ORDER[draft.intensity]),
        -assessments.template_staleness_risk,
    )


def discover_strategies(
    inputs: TransformationInputs, config: Stage40Config, source_moment: SourceMoment
) -> list[StrategyDraft]:
    """Discover, hard-filter, and deterministically rank candidate directions."""

    drafts = _candidate_drafts(inputs, config, source_moment.structure)
    recommended = [item for item in drafts if item.disposition is StrategyDisposition.RECOMMENDED]
    rejected = [item for item in drafts if item.disposition is StrategyDisposition.REJECTED]
    recommended.sort(key=lambda item: (_rank_key(item), item.strategy_type.value))
    ranked: list[StrategyDraft] = []
    for index, draft in enumerate(recommended[: config.max_recommended_strategies], start=1):
        ranked.append(_with_rank(draft, index))
    rejected.sort(key=lambda item: (item.strategy_type.value,))
    for draft in rejected[: config.max_rejected_strategies]:
        ranked.append(draft)
    return ranked


def _with_rank(draft: StrategyDraft, rank: int) -> StrategyDraft:
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
