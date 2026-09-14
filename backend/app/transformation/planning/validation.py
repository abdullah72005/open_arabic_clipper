"""Deterministic Stage 4.1 plan validation and source-span resolution.

Pure functions only: no network, no model loading, no audio decoding. Owns
source-span resolution, hero placement, duration arithmetic, substantive-value
validation, paraphrase/cosmetic rejection, narration/TTS separation,
verification dependency enforcement, material-distinction signatures, and
persistence eligibility.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from app.core.enums import (
    NarrationNeed,
    PlanBlockType,
    PlanStatus,
    SourceExcerptRole,
    StrategyOrigin,
    SubstantiveValueKind,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.transformation.planning.fingerprints import (
    build_plan_output_payload,
    plan_fingerprint,
)
from app.transformation.planning.policy import (
    DISTORTION_MARKERS,
    FAKE_HOOK_MARKERS,
    GENERIC_VALUE_MARKERS,
    PARAPHRASE_SCAFFOLD_MARKERS,
    PRESENTATION_ONLY_TERMS,
    VERIFIED_CLAIM_MARKERS,
    Stage41Config,
    additive_value_kinds,
    is_strict_hero_window,
    strategy_value_kinds,
)
from app.transformation.planning.types import (
    NarrationRequirement,
    PlanBlock,
    PlanningInputs,
    PlanProviderBlock,
    PlanProviderPlan,
    ValidatedPlan,
)

# Bounded rejection reason codes.
REJECT_TOO_MANY_BLOCKS = "TOO_MANY_BLOCKS"
REJECT_NO_SOURCE = "NO_SOURCE_EXCERPT"
REJECT_INVALID_SPAN = "INVALID_SOURCE_SPAN"
REJECT_HERO_PLACEMENT = "HERO_PLACEMENT_INVALID"
REJECT_LATE_HERO = "HERO_TOO_LATE"
REJECT_LONG_PREAMBLE = "LONG_PREAMBLE"
REJECT_TOO_LONG = "PLAN_TOO_LONG"
REJECT_EMPTY_INTENT = "EMPTY_OR_GENERIC_INTENT"
REJECT_PRESENTATION_ONLY = "PRESENTATION_ONLY"
REJECT_FAKE_HOOK = "FAKE_HOOK"
REJECT_DISTORTION = "SOURCE_DISTORTION"
REJECT_PARAPHRASE = "PARAPHRASE_ONLY"
REJECT_UNSUPPORTED_FACT = "UNSUPPORTED_FACT"
REJECT_VALUE_KIND_MISMATCH = "VALUE_KIND_INCONSISTENT_WITH_STRATEGY"
REJECT_NON_CHRONOLOGICAL = "NON_CHRONOLOGICAL_SOURCE"
REJECT_DUPLICATE_EXCERPT = "DUPLICATE_EXCERPT"
REJECT_VERIFICATION_MISSING = "VERIFICATION_DEPENDENCY_MISSING"
REJECT_VERIFICATION_UNLINKED = "VERIFICATION_DEPENDENCY_UNLINKED"
REJECT_NARRATION_ESSENTIAL = "NARRATION_MUST_BE_ESSENTIAL"
REJECT_NARRATION_UNSPECIFIED = "NARRATION_REQUIREMENT_INCOMPLETE"
REJECT_NO_VALUE = "NO_SUBSTANTIVE_VALUE"

_VERIFICATION_BLOCK_TYPES = {PlanBlockType.FACT_VERIFICATION_PLACEHOLDER}
_SUBSTANTIVE_TYPES = {PlanBlockType.ORIGINAL_VALUE, PlanBlockType.TEXTUAL_ANNOTATION}
_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
_NUMBER_RE = re.compile(r"\d[\d.,%:/-]*")
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "that",
        "this",
        "is",
        "are",
        "was",
        "were",
        "as",
        "by",
        "من",
        "في",
        "على",
        "عن",
        "الى",
        "إلى",
        "هو",
        "هي",
        "هذا",
        "هذه",
        "و",
        "او",
        "أو",
    }
)


@dataclass(frozen=True)
class SpanResolution:
    start: float
    end: float
    text: str
    word_start_index: int | None
    word_end_index: int | None


@dataclass(frozen=True)
class ValidationResult:
    plan: ValidatedPlan | None
    reasons: tuple[str, ...] = ()
    explicit_no_plan: bool = False


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text) if len(token) >= 2]


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(text) if token not in _STOPWORDS]


def containment(block_text: str, source_text: str) -> float:
    block = _content_tokens(block_text)
    if not block:
        return 0.0
    source = set(_content_tokens(source_text))
    if not source:
        return 0.0
    shared = sum(1 for token in block if token in source)
    return shared / len(block)


def _has_marker(text: str, markers: Sequence[str]) -> bool:
    lowered = f" {text.casefold()} "
    return any(marker in lowered for marker in markers)


def _is_presentation_only(text: str) -> bool:
    lowered = text.casefold()
    if not any(term in lowered for term in PRESENTATION_ONLY_TERMS):
        return False
    tokens = _content_tokens(text)
    non_presentation = [
        token for token in tokens if not any(term in token for term in PRESENTATION_ONLY_TERMS)
    ]
    return len(non_presentation) < 2


def _is_generic(text: str) -> bool:
    tokens = _content_tokens(text)
    if not tokens:
        return True
    if all(token in {"good", "bad", "nice", "great", "amazing"} for token in tokens):
        return True
    return False


def resolve_source_span(
    block: PlanProviderBlock,
    inputs: PlanningInputs,
    config: Stage41Config,
) -> tuple[SpanResolution | None, str | None]:
    """Resolve provider word-index selection to real Stage 3.5 source evidence.

    The provider never supplies timestamps or excerpt text. Invalid indices or
    out-of-window spans reject only that provider item. When word coverage is
    insufficient, only the safe full refined window is allowed.
    """

    duration = inputs.duration
    if duration <= 0:
        return None, REJECT_INVALID_SPAN
    if block.use_full_window:
        start, end = inputs.refined_start, inputs.refined_end
        text = inputs.transcript
        return SpanResolution(start, end, text, None, None), None
    if block.word_start_index is None or block.word_end_index is None:
        return None, REJECT_INVALID_SPAN
    if not inputs.word_coverage_sufficient or not inputs.words:
        return None, REJECT_INVALID_SPAN
    start_index = int(block.word_start_index)
    end_index = int(block.word_end_index)
    if start_index < 0 or end_index < start_index or end_index >= len(inputs.words):
        return None, REJECT_INVALID_SPAN
    selected = inputs.words[start_index : end_index + 1]
    start = selected[0].start
    end = selected[-1].end
    tolerance = 0.75
    if end <= start:
        return None, REJECT_INVALID_SPAN
    if start < inputs.refined_start - tolerance or end > inputs.refined_end + tolerance:
        return None, REJECT_INVALID_SPAN
    text = " ".join(word.text for word in selected)
    return SpanResolution(start, end, text, start_index, end_index), None


def _resolve_block(
    raw: PlanProviderBlock,
    index: int,
    inputs: PlanningInputs,
    config: Stage41Config,
) -> tuple[PlanBlock | None, str | None]:
    if raw.block_type is PlanBlockType.SOURCE_EXCERPT:
        span, error = resolve_source_span(raw, inputs, config)
        if span is None or error is not None:
            return None, error or REJECT_INVALID_SPAN
        role = raw.source_role or SourceExcerptRole.SUPPORT
        duration = max(0.0, span.end - span.start)
        return (
            PlanBlock(
                index=index,
                block_type=PlanBlockType.SOURCE_EXCERPT,
                purpose=raw.purpose,
                estimated_duration=duration,
                interrupts_source=False,
                preservation_constraints=raw.preservation_constraints,
                dependency_ids=raw.dependency_ids,
                source_role=role,
                word_start_index=span.word_start_index,
                word_end_index=span.word_end_index,
                source_start=span.start,
                source_end=span.end,
                source_text=span.text,
                continuity_rationale=raw.continuity_rationale,
            ),
            None,
        )
    if raw.block_type is PlanBlockType.FACT_VERIFICATION_PLACEHOLDER:
        if not raw.claim_dependency or not raw.must_verify_before_execution:
            return None, REJECT_VERIFICATION_MISSING
        return (
            PlanBlock(
                index=index,
                block_type=PlanBlockType.FACT_VERIFICATION_PLACEHOLDER,
                purpose=raw.purpose,
                estimated_duration=max(0.0, float(raw.estimated_duration)),
                interrupts_source=raw.interrupts_source,
                preservation_constraints=raw.preservation_constraints,
                dependency_ids=raw.dependency_ids,
                claim_dependency=raw.claim_dependency,
                verification_rationale=raw.verification_rationale,
                intended_use=raw.intended_use,
                must_verify_before_execution=True,
                dependent_block_ids=raw.dependent_block_ids,
            ),
            None,
        )
    # ORIGINAL_VALUE, TEXTUAL_ANNOTATION, TRANSITION.
    return (
        PlanBlock(
            index=index,
            block_type=raw.block_type,
            purpose=raw.purpose,
            estimated_duration=max(0.0, float(raw.estimated_duration)),
            interrupts_source=raw.interrupts_source,
            preservation_constraints=raw.preservation_constraints,
            dependency_ids=raw.dependency_ids,
            substantive_value_kind=raw.substantive_value_kind,
            semantic_intent=raw.semantic_intent,
            why_unavailable=raw.why_unavailable,
            grounding_refs=raw.grounding_refs,
            delivery_intent=raw.delivery_intent,
            draft_line=raw.draft_line,
            draft_only=bool(raw.draft_line),
        ),
        None,
    )


def _validate_substantive_block(
    block: PlanBlock,
    inputs: PlanningInputs,
    strategy_type: TransformationStrategyType,
    config: Stage41Config,
) -> str | None:
    if block.block_type is PlanBlockType.TRANSITION:
        return None
    if block.block_type not in _SUBSTANTIVE_TYPES:
        return None
    intent = (block.semantic_intent or "").strip()
    why = (block.why_unavailable or "").strip()
    draft = (block.draft_line or "").strip()
    combined = f"{intent} {draft}".strip()
    if len(intent) < config.min_substantive_intent_characters or not why:
        return REJECT_EMPTY_INTENT
    if _is_generic(intent) and not draft:
        return REJECT_EMPTY_INTENT
    if _is_presentation_only(combined):
        return REJECT_PRESENTATION_ONLY
    if _has_marker(combined, FAKE_HOOK_MARKERS):
        return REJECT_FAKE_HOOK
    if _has_marker(combined, DISTORTION_MARKERS):
        return REJECT_DISTORTION
    if _has_marker(combined, PARAPHRASE_SCAFFOLD_MARKERS):
        return REJECT_PARAPHRASE
    if _has_marker(combined, VERIFIED_CLAIM_MARKERS):
        return REJECT_UNSUPPORTED_FACT
    if any(marker in intent.casefold() for marker in GENERIC_VALUE_MARKERS) and not why:
        return REJECT_EMPTY_INTENT
    allowed = strategy_value_kinds(strategy_type)
    if block.substantive_value_kind is None or block.substantive_value_kind not in allowed:
        return REJECT_VALUE_KIND_MISMATCH
    if block.substantive_value_kind not in additive_value_kinds():
        return REJECT_NO_VALUE
    if block.draft_line:
        draft_numbers = {token for token in _NUMBER_RE.findall(draft)}
        unsupported = {token for token in draft_numbers if token not in inputs.transcript}
        if unsupported and not block.dependency_ids:
            return REJECT_UNSUPPORTED_FACT
        if (
            containment(combined, inputs.transcript) >= config.containment_reject_ratio
            and len(_content_tokens(combined)) >= 5
        ):
            return REJECT_PARAPHRASE
    return None


def _resolve_narration(
    plan: PlanProviderPlan, inputs: PlanningInputs
) -> tuple[NarrationRequirement, str | None]:
    narration = plan.narration
    if narration.need is NarrationNeed.NONE:
        return NarrationRequirement(need=NarrationNeed.NONE), None
    if not inputs.planning_context.narration_allowed:
        if narration.need is NarrationNeed.REQUIRED or narration.essential:
            return narration, REJECT_NARRATION_UNSPECIFIED
    if not narration.purposes:
        return narration, REJECT_NARRATION_UNSPECIFIED
    if narration.estimated_duration <= 0:
        return narration, REJECT_NARRATION_UNSPECIFIED
    if narration.language is None:
        narration = replace(narration, language=inputs.planning_context.output_language_policy)
    if narration.register is None:
        narration = replace(narration, register=inputs.planning_context.register_intent)
    return narration, None


def _structure_signature(
    strategy_type: TransformationStrategyType, blocks: Sequence[PlanBlock]
) -> str:
    parts = [strategy_type.value]
    for block in blocks:
        kind = block.substantive_value_kind.value if block.substantive_value_kind else ""
        parts.append(f"{block.block_type.value}:{kind}")
    return "|".join(parts)


def _durations(blocks: Sequence[PlanBlock], narration: NarrationRequirement) -> dict[str, object]:
    source = sum(
        b.estimated_duration for b in blocks if b.block_type is PlanBlockType.SOURCE_EXCERPT
    )
    original = sum(
        b.estimated_duration
        for b in blocks
        if b.block_type in {PlanBlockType.ORIGINAL_VALUE, PlanBlockType.TEXTUAL_ANNOTATION}
    )
    transition = sum(
        b.estimated_duration for b in blocks if b.block_type is PlanBlockType.TRANSITION
    )
    narration_duration = float(narration.estimated_duration)
    total = source + original + transition
    safe_total = total if total > 0 else 1.0
    return {
        "source_seconds": round(source, 3),
        "original_seconds": round(original, 3),
        "transition_seconds": round(transition, 3),
        "narration_seconds": round(narration_duration, 3),
        "total_seconds": round(total, 3),
        "source_ratio": round(source / safe_total, 4),
        "original_ratio": round(original / safe_total, 4),
        "narration_ratio": round(narration_duration / safe_total, 4),
    }


def validate_provider_plan(
    provider_plan: PlanProviderPlan,
    strategy: Mapping[str, object],
    inputs: PlanningInputs,
    config: Stage41Config,
    *,
    provider_evidence: Mapping[str, object],
    provider_input_fingerprint: str,
) -> ValidationResult:
    """Validate one provider plan against deterministic Stage 4.1 rules."""

    if provider_plan.no_valid_plan:
        reason = provider_plan.no_valid_reason or "PROVIDER_DECLINED"
        return ValidationResult(plan=None, reasons=(reason,), explicit_no_plan=True)

    strategy_type_raw = strategy.get("strategy_type")
    try:
        strategy_type = TransformationStrategyType(str(strategy_type_raw))
    except ValueError:
        return ValidationResult(plan=None, reasons=("UNKNOWN_STRATEGY",))
    try:
        intensity = TransformationIntensity(str(strategy.get("intensity")))
    except ValueError:
        intensity = TransformationIntensity.MODERATE
    external = str(strategy.get("external_verification_requirement", "NOT_REQUIRED"))

    raw_blocks = list(provider_plan.blocks)
    if not raw_blocks:
        return ValidationResult(plan=None, reasons=(REJECT_NO_SOURCE,))
    if len(raw_blocks) > config.max_blocks_per_plan:
        return ValidationResult(plan=None, reasons=(REJECT_TOO_MANY_BLOCKS,))

    blocks: list[PlanBlock] = []
    for index, raw in enumerate(raw_blocks):
        resolved, error = _resolve_block(raw, index, inputs, config)
        if resolved is None or error is not None:
            return ValidationResult(plan=None, reasons=(error or REJECT_INVALID_SPAN,))
        blocks.append(resolved)

    source_blocks = [b for b in blocks if b.block_type is PlanBlockType.SOURCE_EXCERPT]
    if not source_blocks:
        return ValidationResult(plan=None, reasons=(REJECT_NO_SOURCE,))
    heroes = [b for b in source_blocks if b.source_role is SourceExcerptRole.HERO]
    if len(heroes) != 1:
        return ValidationResult(plan=None, reasons=(REJECT_HERO_PLACEMENT,))
    hero = heroes[0]
    hero_index = hero.index
    if hero_index not in (0, 1):
        return ValidationResult(plan=None, reasons=(REJECT_HERO_PLACEMENT,))
    if hero.source_start is None or hero.source_end is None or hero.source_start >= hero.source_end:
        return ValidationResult(plan=None, reasons=(REJECT_INVALID_SPAN,))
    if (
        hero.source_start < inputs.refined_start - 0.75
        or hero.source_end > inputs.refined_end + 0.75
    ):
        return ValidationResult(plan=None, reasons=(REJECT_INVALID_SPAN,))

    authored_before = sum(b.estimated_duration for b in blocks[:hero_index] if not b.is_source)
    strict = is_strict_hero_window(
        inputs.source_moment_structure,
        inputs.duration,
        float(inputs.stage3_risk.get("moment_density_score", 0.0) or 0.0),
    )
    cap = (
        config.strict_authored_before_hero_seconds
        if strict
        else config.max_authored_before_hero_seconds
    )
    if authored_before > cap:
        return ValidationResult(
            plan=None,
            reasons=(REJECT_LONG_PREAMBLE if authored_before >= 10.0 else REJECT_LATE_HERO,),
        )

    derived = _durations(blocks, provider_plan.narration)
    if float(derived["total_seconds"]) > config.max_plan_duration_seconds:
        return ValidationResult(plan=None, reasons=(REJECT_TOO_LONG,))

    # Chronology and duplicate excerpts.
    previous_start: float | None = None
    seen_spans: set[tuple[float, float]] = set()
    for block in source_blocks:
        assert block.source_start is not None and block.source_end is not None
        key = (round(block.source_start, 3), round(block.source_end, 3))
        if key in seen_spans:
            return ValidationResult(plan=None, reasons=(REJECT_DUPLICATE_EXCERPT,))
        seen_spans.add(key)
        if previous_start is not None and block.source_start < previous_start - 0.5:
            return ValidationResult(plan=None, reasons=(REJECT_NON_CHRONOLOGICAL,))
        previous_start = block.source_start

    narration, narration_error = _resolve_narration(provider_plan, inputs)
    if narration_error is not None:
        return ValidationResult(plan=None, reasons=(narration_error,))

    substantive = [b for b in blocks if b.is_substantive]
    for block in substantive:
        error = _validate_substantive_block(block, inputs, strategy_type, config)
        if error is not None:
            return ValidationResult(plan=None, reasons=(error,))

    if not substantive and not _verification_blocks(blocks):
        return ValidationResult(plan=None, reasons=(REJECT_NO_VALUE,))

    non_narration_substantive = [
        b
        for b in substantive
        if b.delivery_intent is None or b.delivery_intent.value != "NARRATION"
    ]
    if (
        not narration.is_none
        and not narration.essential
        and substantive
        and not non_narration_substantive
    ):
        return ValidationResult(plan=None, reasons=(REJECT_NARRATION_ESSENTIAL,))

    verification_blocks = _verification_blocks(blocks)
    requires_verification = external == "REQUIRES_EXTERNAL_FACT_VERIFICATION"
    if requires_verification and not verification_blocks:
        return ValidationResult(plan=None, reasons=(REJECT_VERIFICATION_MISSING,))
    if verification_blocks and requires_verification:
        linked = False
        for placeholder in verification_blocks:
            if placeholder.dependent_block_ids or any(
                placeholder.claim_dependency in block.dependency_ids for block in substantive
            ):
                linked = True
        if not linked:
            return ValidationResult(plan=None, reasons=(REJECT_VERIFICATION_UNLINKED,))

    status = (
        PlanStatus.PLAN_GENERATED_WITH_VERIFICATION_REQUIRED
        if verification_blocks or requires_verification
        else PlanStatus.PLAN_GENERATED
    )
    external_dependencies: list[dict[str, object]] = []
    for placeholder in verification_blocks:
        external_dependencies.append(
            {
                "dependency": placeholder.claim_dependency,
                "rationale": placeholder.verification_rationale,
                "intended_use": placeholder.intended_use,
                "must_verify_before_execution": True,
                "dependent_block_ids": list(placeholder.dependent_block_ids),
            }
        )

    hero_appearance = sum(b.estimated_duration for b in blocks[:hero_index])
    structure = _structure_signature(strategy_type, blocks)
    plan = ValidatedPlan(
        strategy_id=str(strategy.get("id", "")),
        strategy_key=str(strategy.get("strategy_key", "")),
        strategy_type=strategy_type,
        strategy_rank=int(strategy.get("rank", 0) or 0),
        intensity=intensity,
        strategy_fingerprint=str(strategy.get("strategy_fingerprint", "")),
        plan_key=f"{strategy.get('strategy_key', strategy_type.value)}:plan",
        status=status,
        generation_rank=0,
        blocks=tuple(blocks),
        hero_block_index=hero_index,
        hero_source_start=float(hero.source_start),
        hero_source_end=float(hero.source_end),
        hero_appearance_time=hero_appearance,
        preservation_constraints=tuple(
            dict.fromkeys(
                [*provider_plan.preservation_constraints, *(hero.preservation_constraints)]
            )
        ),
        original_value_kinds=tuple(
            dict.fromkeys(
                b.substantive_value_kind
                for b in substantive
                if b.substantive_value_kind is not None
            )
        ),
        original_value_reasons=tuple(
            b.why_unavailable or "" for b in substantive if b.why_unavailable
        ),
        narration=narration,
        external_fact_dependencies=tuple(external_dependencies),
        required_context=tuple(inputs.context_segments),
        derived_durations=derived,
        hook_payoff_evidence={
            "hook_index": inputs.source_moment.get("hook_index"),
            "payoff_index": inputs.source_moment.get("payoff_index"),
            "hero_appearance_time": round(hero_appearance, 3),
            "authored_before_hero_seconds": round(authored_before, 3),
        },
        degraded_rules=_degraded_rules(strategy_type),
        stage40_risk={
            "platform_risk": dict(inputs.stage40_platform_risk),
            "assessments": dict(inputs.stage40_assessments),
            "rights_risk": inputs.rights_risk,
            "originality_risk": inputs.originality_risk,
        },
        source_dialect={
            "profile": inputs.dialect_profile,
            "confidence": inputs.dialect_confidence,
        },
        target_intent=inputs.planning_context.as_dict(),
        planner_confidence=raw_confidence(provider_plan, strategy),
        generation_origin=_origin(strategy),
        provider_evidence=dict(provider_evidence),
        provider_input_fingerprint=provider_input_fingerprint,
        plan_output_fingerprint="",
        structure_signature=structure,
    )
    plan = replace(
        plan,
        plan_output_fingerprint=plan_fingerprint(build_plan_output_payload(plan)),
    )
    return ValidationResult(plan=plan)


def _verification_blocks(blocks: Sequence[PlanBlock]) -> list[PlanBlock]:
    return [b for b in blocks if b.block_type in _VERIFICATION_BLOCK_TYPES]


def _degraded_rules(strategy_type: TransformationStrategyType) -> tuple[str, ...]:
    if strategy_type in {
        TransformationStrategyType.SOURCE_LED_MINIMAL,
        TransformationStrategyType.SOURCE_AS_EVIDENCE,
    }:
        return ("If the added value cannot be produced, degrade to the source excerpt alone.",)
    return ("If the provider plan is unavailable, defer rather than emit template filler.",)


def raw_confidence(plan: PlanProviderPlan, strategy: Mapping[str, object]) -> float:
    from app.transformation.types import clamp

    base = plan.confidence if plan.confidence > 0 else float(strategy.get("confidence", 0.0) or 0.0)
    return clamp(base)


def _origin(strategy: Mapping[str, object]) -> StrategyOrigin:
    raw = str(strategy.get("origin", StrategyOrigin.DETERMINISTIC.value))
    try:
        return StrategyOrigin(raw)
    except ValueError:
        return StrategyOrigin.DETERMINISTIC


def deterministic_provider_plan(
    strategy: Mapping[str, object],
    inputs: PlanningInputs,
    config: Stage41Config,
) -> PlanProviderPlan | None:
    """Conservative deterministic structuring of an already-concrete intent.

    Only obvious, grounded Stage 4.0 strategies qualify. It structures existing
    approved intent and never invents research, claims, names, numbers, context,
    or generic commentary.
    """

    from app.transformation.planning.policy import DETERMINISTIC_FALLBACK_STRATEGIES

    try:
        strategy_type = TransformationStrategyType(str(strategy.get("strategy_type")))
    except ValueError:
        return None
    if strategy_type not in DETERMINISTIC_FALLBACK_STRATEGIES:
        return None
    focus = str(strategy.get("added_value_focus", "")).strip()
    if not focus:
        return None
    kind_raw = str(strategy.get("substantive_value_kind", ""))
    try:
        kind = SubstantiveValueKind(kind_raw)
    except ValueError:
        return None
    source = PlanProviderBlock(
        block_type=PlanBlockType.SOURCE_EXCERPT,
        purpose="Hero source moment",
        use_full_window=True,
        source_role=SourceExcerptRole.HERO,
        preservation_constraints=tuple(strategy.get("preservation_requirements") or []),
    )
    value = PlanProviderBlock(
        block_type=PlanBlockType.ORIGINAL_VALUE,
        purpose="Substantive value after the source",
        estimated_duration=4.0,
        interrupts_source=False,
        substantive_value_kind=kind,
        semantic_intent=focus,
        why_unavailable=(
            "The source excerpt alone does not provide this specific added dimension."
        ),
        grounding_refs=(str(strategy.get("strategy_key", "")),),
        delivery_intent=None,
    )
    return PlanProviderPlan(
        strategy_id=str(strategy.get("id", "")),
        strategy_key=str(strategy.get("strategy_key", strategy_type.value)),
        confidence=0.5,
        blocks=(source, value),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )
