"""Deterministic Stage 4.2 evidence, dimensions, hard gates, and status mapping.

The governor never creates an overall score. It persists independent categorical
dimensions with bounded evidence/reason codes and maps validated semantic
findings to statuses deterministically. A provider can never assign the final
governor status or a platform-risk classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.enums import (
    ClaimGroundingState,
    CoherenceFinding,
    GovernanceEvidenceStrength,
    GovernanceLevel,
    GovernancePlanStatus,
    GovernanceReasonCode,
    GovernanceRemediationAction,
    GovernanceRemediationPriority,
    GovernanceSeverityClass,
    NarrationBurdenFinding,
    NarrationNeed,
    PlanBlockType,
    PlatformRiskLevel,
    RetentionEffectFinding,
    SemanticFidelityFinding,
    SubstantiveValueFinding,
    SubstantiveValueKind,
    TransformationIntensity,
    TransformationStrategyType,
    UnsupportedClaimFinding,
)
from app.transformation.governance.policy import (
    ADDITIVE_VALUE_KINDS,
    FAKE_HOOK_MARKERS,
    FORBIDDEN_STATUS_MARKERS,
    PLATFORM_EVASION_MARKERS,
    PLATFORM_GUARANTEE_MARKERS,
    SOURCE_AS_EVIDENCE_STRATEGIES,
    Stage42Config,
    is_strict_hero_window,
)
from app.transformation.governance.types import (
    GovernanceInputs,
    PlanEvidence,
    PlanGovernance,
    ProviderCritique,
)
from app.transformation.planning.policy import (
    DISTORTION_MARKERS,
    GENERIC_VALUE_MARKERS,
    PARAPHRASE_SCAFFOLD_MARKERS,
    PRESENTATION_ONLY_CHANGES,
    PRESENTATION_ONLY_TERMS,
    VERIFIED_CLAIM_MARKERS,
)
from app.transformation.planning.validation import boundary_violation, containment

_R = GovernanceReasonCode

# Semantic-review lifecycle states.
SEMANTIC_NOT_REQUIRED = "NOT_REQUIRED"
SEMANTIC_NOT_ATTEMPTED = "NOT_ATTEMPTED"
SEMANTIC_AVAILABLE = "AVAILABLE"
SEMANTIC_UNAVAILABLE = "UNAVAILABLE"
SEMANTIC_INVALID = "INVALID"

_NUMERIC_RE = re.compile(r"[0-9\u0660-\u0669][0-9\u0660-\u0669.,%\u066b\u066c]*")

_PRESENTATION_WORDS = frozenset(
    word.casefold() for word in (*PRESENTATION_ONLY_TERMS, *PRESENTATION_ONLY_CHANGES)
)
_PRESENTATION_FILLER = frozenset(
    {
        "add",
        "adds",
        "adding",
        "use",
        "uses",
        "using",
        "put",
        "make",
        "makes",
        "the",
        "a",
        "an",
        "and",
        "with",
        "only",
        "just",
        "video",
        "videos",
        "clip",
        "clips",
        "this",
        "that",
        "to",
        "of",
        "it",
        "on",
        "in",
        "very",
    }
)

_BOUNDARY_REASON = {
    "TTS_SELECTION_FORBIDDEN": _R.TTS_IDENTITY_FORBIDDEN,
    "SPEAKER_SELECTION_FORBIDDEN": _R.TTS_IDENTITY_FORBIDDEN,
    "RENDERING_INSTRUCTION_FORBIDDEN": _R.PLAN_INTEGRITY_INVALID,
    "PLATFORM_EVASION_FORBIDDEN": _R.PLATFORM_EVASION_TACTIC,
    "COSMETIC_CLAIM_FORBIDDEN": _R.PLATFORM_EVASION_TACTIC,
}


@dataclass
class PlanEvaluation:
    """Mutable accumulator for one plan's governance evaluation."""

    plan: PlanEvidence
    integrity_ok: bool = True
    integrity_reasons: tuple[str, ...] = ()
    evidence: dict[str, object] = field(default_factory=dict)
    dimensions: dict[str, object] = field(default_factory=dict)
    platform_risk: dict[str, object] = field(default_factory=dict)
    verification: dict[str, object] = field(default_factory=dict)
    hard_failures: list[dict[str, object]] = field(default_factory=list)
    blocking: list[dict[str, object]] = field(default_factory=list)
    revisions: list[dict[str, object]] = field(default_factory=list)
    warnings: list[dict[str, object]] = field(default_factory=list)
    advisories: list[dict[str, object]] = field(default_factory=list)
    requires_semantic_review: bool = False
    semantic_state: str = SEMANTIC_NOT_ATTEMPTED
    critique: ProviderCritique | None = None
    provider_evidence: dict[str, object] = field(default_factory=dict)


def _text(value: object) -> str:
    return str(value or "")


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[\w\u0600-\u06ff]+", text.casefold()) if token]


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in _tokens(text)
        if token not in _PRESENTATION_WORDS and token not in _PRESENTATION_FILLER
    ]


def _has_marker(text: str, markers: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(marker.casefold() in folded for marker in markers)


def _is_presentation_only(text: str) -> bool:
    return len(_content_tokens(text)) < 2


def _is_generic(text: str) -> bool:
    if not _content_tokens(text):
        return True
    return _has_marker(text, GENERIC_VALUE_MARKERS) or all(
        token in {"interesting", "insightful", "useful", "important", "value"}
        for token in _content_tokens(text)
    )


def _is_paraphrase(text: str, transcript: str, config: Stage42Config) -> bool:
    if _has_marker(text, PARAPHRASE_SCAFFOLD_MARKERS):
        return True
    if not text.strip() or not transcript.strip():
        return False
    return containment(text, transcript) >= config.containment_reject_ratio


def _add(
    bucket: list[dict[str, object]],
    severity: GovernanceSeverityClass,
    code: GovernanceReasonCode,
    detail: str,
    *,
    block_indexes: tuple[int, ...] = (),
) -> None:
    bucket.append(
        {
            "severity": severity.value,
            "code": code.value,
            "detail": detail[:240],
            "block_indexes": list(block_indexes),
        }
    )


def _blocks(plan: PlanEvidence) -> list[dict[str, object]]:
    return [dict(block) for block in plan.blocks]


def _block_type(block: dict[str, object]) -> str:
    return _text(block.get("block_type"))


def _duration(block: dict[str, object]) -> float:
    value = block.get("estimated_duration", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, float(value))


def _int(value: object, default: int = -1) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


def _num(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _as_text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


# ---------------------------------------------------------------------------
# Integrity revalidation
# ---------------------------------------------------------------------------


def revalidate_integrity(plan: PlanEvidence, config: Stage42Config) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    try:
        TransformationStrategyType(plan.strategy_type)
    except ValueError:
        reasons.append("INVALID_STRATEGY_TYPE")
    try:
        TransformationIntensity(plan.intensity)
    except ValueError:
        reasons.append("INVALID_INTENSITY")
    if not plan.plan_output_fingerprint:
        reasons.append("MISSING_PLAN_FINGERPRINT")
    try:
        NarrationNeed(_text(plan.narration.get("need", "NONE")))
    except ValueError:
        reasons.append("INVALID_NARRATION_NEED")

    blocks = _blocks(plan)
    if not blocks:
        return False, tuple(reasons + ["NO_BLOCKS"])
    for index, block in enumerate(blocks):
        if _int(block.get("index")) != index:
            reasons.append("BLOCK_INDEX_MISMATCH")
        try:
            PlanBlockType(_block_type(block))
        except ValueError:
            reasons.append("INVALID_BLOCK_TYPE")
    if reasons:
        return False, tuple(dict.fromkeys(reasons))

    hero = plan.hero_block_index
    if (
        not (0 <= hero < len(blocks))
        or _block_type(blocks[hero]) != PlanBlockType.SOURCE_EXCERPT.value
    ):
        reasons.append("HERO_NOT_SOURCE")
    hero_roles = [
        index
        for index, block in enumerate(blocks)
        if _block_type(block) == PlanBlockType.SOURCE_EXCERPT.value
        and _text(block.get("source_role")) == "HERO"
    ]
    if len(hero_roles) != 1 or (hero_roles and hero_roles[0] != hero):
        reasons.append("HERO_ROLE_MISMATCH")

    spans: list[tuple[float, float]] = []
    for index, block in enumerate(blocks):
        if _block_type(block) != PlanBlockType.SOURCE_EXCERPT.value:
            continue
        start = block.get("source_start")
        end = block.get("source_end")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            reasons.append("SOURCE_SPAN_MISSING")
            continue
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            reasons.append("SOURCE_SPAN_MISSING")
            continue
        if float(end) <= float(start):
            reasons.append("SOURCE_SPAN_INVALID")
            continue
        if (float(end) - float(start)) + 1e-9 < config.min_source_excerpt_seconds:
            reasons.append("SOURCE_SPAN_TOO_SHORT")
        spans.append((float(start), float(end)))
    for (prev_start, prev_end), (next_start, _next_end) in zip(spans, spans[1:]):
        if next_start < prev_start:
            reasons.append("SOURCE_SPAN_NON_CHRONOLOGICAL")
        if next_start < prev_end - 1e-6:
            reasons.append("SOURCE_SPAN_OVERLAP")

    for block in blocks:
        if _block_type(block) != PlanBlockType.FACT_VERIFICATION_PLACEHOLDER.value:
            continue
        claim = _text(block.get("claim_dependency"))
        dependents = block.get("dependent_block_ids")
        if not claim or not isinstance(dependents, list) or not dependents:
            reasons.append("VERIFICATION_MISSING_LINKAGE")
            continue
        for dependent in dependents:
            if not isinstance(dependent, int) or isinstance(dependent, bool):
                reasons.append("VERIFICATION_INVALID_LINKAGE")
                continue
            if not (0 <= dependent < len(blocks)):
                reasons.append("VERIFICATION_INVALID_LINKAGE")
                continue
            target = blocks[dependent]
            if _block_type(target) not in {
                PlanBlockType.ORIGINAL_VALUE.value,
                PlanBlockType.TEXTUAL_ANNOTATION.value,
            }:
                reasons.append("VERIFICATION_NON_SUBSTANTIVE_LINKAGE")
                continue
            dependencies = target.get("dependency_ids")
            if not isinstance(dependencies, list) or claim not in dependencies:
                reasons.append("VERIFICATION_UNLINKED")

    for kind in plan.original_value_kinds:
        try:
            SubstantiveValueKind(kind)
        except ValueError:
            reasons.append("INVALID_VALUE_KIND")

    return (not reasons), tuple(dict.fromkeys(reasons))


# ---------------------------------------------------------------------------
# Evidence derivation
# ---------------------------------------------------------------------------


def derive_evidence(
    plan: PlanEvidence, inputs: GovernanceInputs, config: Stage42Config
) -> dict[str, object]:
    blocks = _blocks(plan)
    source = [block for block in blocks if _block_type(block) == PlanBlockType.SOURCE_EXCERPT.value]
    original = [
        block for block in blocks if _block_type(block) == PlanBlockType.ORIGINAL_VALUE.value
    ]
    annotation = [
        block for block in blocks if _block_type(block) == PlanBlockType.TEXTUAL_ANNOTATION.value
    ]
    transition = [block for block in blocks if _block_type(block) == PlanBlockType.TRANSITION.value]
    verification = [
        block
        for block in blocks
        if _block_type(block) == PlanBlockType.FACT_VERIFICATION_PLACEHOLDER.value
    ]
    substantive = original + annotation

    total_duration = sum(_duration(block) for block in blocks)
    hero_index = plan.hero_block_index
    elapsed_before_hero = sum(
        _duration(block) for index, block in enumerate(blocks) if index < hero_index
    )
    authored_before_hero = sum(
        _duration(block)
        for index, block in enumerate(blocks)
        if index < hero_index
        and _block_type(block)
        in {PlanBlockType.ORIGINAL_VALUE.value, PlanBlockType.TEXTUAL_ANNOTATION.value}
    )
    source_duration = sum(_duration(block) for block in source)
    original_duration = sum(_duration(block) for block in substantive)
    transition_duration = sum(_duration(block) for block in transition)
    narration_requirements = plan.narration.get("requirements") or {}
    if not isinstance(narration_requirements, dict):
        narration_requirements = {}
    narration_duration = narration_requirements.get("estimated_duration", 0.0)
    if isinstance(narration_duration, bool) or not isinstance(narration_duration, (int, float)):
        narration_duration = 0.0
    narration_duration = max(0.0, float(narration_duration))

    intents = [_text(block.get("semantic_intent")) for block in substantive]
    grounding = [
        tuple(str(item) for item in (block.get("grounding_refs") or [])) for block in substantive
    ]
    value_kinds = [_text(block.get("substantive_value_kind")) for block in substantive]
    additive = [kind for kind in value_kinds if kind in ADDITIVE_VALUE_KINDS]
    interrupting = [
        index for index, block in enumerate(blocks) if bool(block.get("interrupts_source"))
    ]

    structure = inputs.source_moment_structure
    density = 0.0
    moment = inputs.stage40_assessments.get("moment_density")
    if isinstance(moment, (int, float)) and not isinstance(moment, bool):
        density = float(moment)
    else:
        source_moment = inputs.stage40_assessments
        value = source_moment.get("source_moment_density")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            density = float(value)
    strict = is_strict_hero_window(structure, inputs.duration, density)
    elapsed_cap = (
        config.strict_elapsed_before_hero_seconds
        if strict
        else config.max_elapsed_before_hero_seconds
    )
    authored_cap = (
        config.strict_authored_before_hero_seconds
        if strict
        else config.max_authored_before_hero_seconds
    )

    generic_intents = sum(1 for intent in intents if _is_generic(intent))
    paraphrase_intents = sum(
        1 for intent in intents if _is_paraphrase(intent, inputs.transcript, config)
    )
    presentation_only = sum(1 for intent in intents if _is_presentation_only(intent))

    return {
        "block_types": [_block_type(block) for block in blocks],
        "total_blocks": len(blocks),
        "source_fragment_count": len(source),
        "substantive_count": len(substantive),
        "transition_count": len(transition),
        "verification_count": len(verification),
        "interrupting_block_indexes": interrupting,
        "hero_block_index": hero_index,
        "stored_hero_appearance_time": plan.hero_appearance_time,
        "hero_appearance_time": round(elapsed_before_hero, 3),
        "elapsed_before_hero": round(elapsed_before_hero, 3),
        "authored_before_hero": round(authored_before_hero, 3),
        "total_duration": round(total_duration, 3),
        "source_duration": round(source_duration, 3),
        "original_duration": round(original_duration, 3),
        "transition_duration": round(transition_duration, 3),
        "narration_duration": round(narration_duration, 3),
        "source_ratio": round(source_duration / total_duration, 4) if total_duration else 0.0,
        "original_ratio": round(original_duration / total_duration, 4) if total_duration else 0.0,
        "narration_ratio": round(narration_duration / total_duration, 4) if total_duration else 0.0,
        "intents": intents,
        "grounding": grounding,
        "value_kinds": value_kinds,
        "has_additive_value": bool(additive),
        "generic_intents": generic_intents,
        "paraphrase_intents": paraphrase_intents,
        "presentation_only_intents": presentation_only,
        "strict_hero_window": strict,
        "elapsed_cap": elapsed_cap,
        "authored_cap": authored_cap,
        "source_moment_structure": structure,
        "narration_need": _text(plan.narration.get("need", "NONE")),
        "narration_requirements": narration_requirements,
        "verification_placeholders": [
            {
                "claim_dependency": _text(block.get("claim_dependency")),
                "dependent_block_ids": list(block.get("dependent_block_ids") or []),
                "must_verify_before_execution": bool(block.get("must_verify_before_execution")),
            }
            for block in verification
        ],
    }


# ---------------------------------------------------------------------------
# Dimensions and deterministic findings
# ---------------------------------------------------------------------------


def _numeric_tokens(text: str) -> set[str]:
    return {match.group(0) for match in _NUMERIC_RE.finditer(text)}


def _fabricated_numeric_claim(
    plan: PlanEvidence, inputs: GovernanceInputs, evidence: dict[str, object]
) -> tuple[int, ...]:
    transcript_numbers = _numeric_tokens(inputs.transcript)
    indexes: list[int] = []
    for block in _blocks(plan):
        if _block_type(block) not in {
            PlanBlockType.ORIGINAL_VALUE.value,
            PlanBlockType.TEXTUAL_ANNOTATION.value,
        }:
            continue
        grounding = tuple(str(item) for item in (block.get("grounding_refs") or []))
        intent = _text(block.get("semantic_intent"))
        numbers = _numeric_tokens(intent)
        if (
            numbers
            and not grounding
            and any(number not in transcript_numbers for number in numbers)
        ):
            indexes.append(_int(block.get("index")))
    return tuple(indexes)


def _template_level(evidence: dict[str, object], plan: PlanEvidence) -> str:
    substantive = int(evidence["substantive_count"])
    intents = [str(item) for item in evidence["intents"]]
    kinds = [str(item) for item in evidence["value_kinds"]]
    if (
        substantive >= 2
        and intents
        and all(intent == "" or _is_generic(intent) for intent in intents)
    ):
        return "HIGH"
    if substantive >= 3 and len(set(kinds)) == 1:
        return "MODERATE"
    if (
        substantive >= 2
        and plan.intensity == TransformationIntensity.STRONG.value
        and int(evidence["source_fragment_count"]) <= 1
    ):
        return "MODERATE"
    return "LOW"


def _source_dominance_level(evidence: dict[str, object], plan: PlanEvidence) -> str:
    ratio = float(evidence["source_ratio"])
    has_value = bool(evidence["has_additive_value"])
    if plan.strategy_type in SOURCE_AS_EVIDENCE_STRATEGIES and has_value:
        return "LOW"
    if not has_value:
        return "HIGH"
    if ratio >= 0.85:
        return "MODERATE"
    return "LOW"


def _narration_burden(
    evidence: dict[str, object], plan: PlanEvidence, config: Stage42Config
) -> str:
    need = str(evidence["narration_need"])
    if need == NarrationNeed.NONE.value:
        return NarrationBurdenFinding.APPROPRIATE.value
    requirements = evidence["narration_requirements"]
    if not isinstance(requirements, dict):
        requirements = {}
    placement = requirements.get("placement_block_index")
    overlaps = bool(requirements.get("overlaps_source_audio"))
    duration = float(evidence["narration_duration"])
    ratio = float(evidence["narration_ratio"])
    hero = int(evidence["hero_block_index"])
    if isinstance(placement, int) and not isinstance(placement, bool) and placement <= hero:
        if overlaps or str(evidence["source_moment_structure"]) in {"JOKE", "PAYOFF"}:
            return NarrationBurdenFinding.POSITION_DAMAGING.value
    if duration > config.excessive_narration_seconds or ratio > config.narration_share_concern:
        return NarrationBurdenFinding.EXCESSIVE.value
    return NarrationBurdenFinding.APPROPRIATE.value


def _verification(
    plan: PlanEvidence, inputs: GovernanceInputs, evidence: dict[str, object]
) -> dict[str, object]:
    placeholders = evidence["verification_placeholders"]
    claims: list[dict[str, object]] = []
    for placeholder in placeholders:
        if not isinstance(placeholder, dict):
            continue
        claims.append(
            {
                "claim_id": placeholder.get("claim_dependency"),
                "state": ClaimGroundingState.EXTERNAL_REQUIRED_UNRESOLVED.value,
                "dependent_block_indexes": placeholder.get("dependent_block_ids"),
                "must_verify_before_execution": placeholder.get("must_verify_before_execution"),
            }
        )
    if claims:
        state = ClaimGroundingState.EXTERNAL_REQUIRED_UNRESOLVED.value
    else:
        fabricated = _fabricated_numeric_claim(plan, inputs, evidence)
        if fabricated:
            state = ClaimGroundingState.UNSUPPORTED_OR_FABRICATED.value
            claims = [
                {
                    "claim_id": None,
                    "state": state,
                    "dependent_block_indexes": list(fabricated),
                    "must_verify_before_execution": False,
                }
            ]
        elif evidence["substantive_count"]:
            state = ClaimGroundingState.GROUNDED_IN_SOURCE.value
        else:
            state = ClaimGroundingState.NOT_APPLICABLE.value
    return {
        "claim_state": state,
        "claims": claims,
        "unresolved": state
        in {
            ClaimGroundingState.EXTERNAL_REQUIRED_UNRESOLVED.value,
            ClaimGroundingState.UNSUPPORTED_OR_FABRICATED.value,
        },
        "reason_codes": (
            [_R.EXTERNAL_VERIFICATION_REQUIRED.value]
            if state == ClaimGroundingState.EXTERNAL_REQUIRED_UNRESOLVED.value
            else (
                [_R.UNSUPPORTED_CRITICAL_CLAIM.value]
                if state == ClaimGroundingState.UNSUPPORTED_OR_FABRICATED.value
                else []
            )
        ),
    }


def _boundary_findings(plan: PlanEvidence, evaluation: PlanEvaluation) -> None:
    texts: list[tuple[str, tuple[int, ...]]] = []
    for block in _blocks(plan):
        index = _int(block.get("index"))
        for value in (
            block.get("purpose"),
            block.get("semantic_intent"),
            block.get("why_unavailable"),
            block.get("draft_line"),
            block.get("continuity_rationale"),
            block.get("verification_rationale"),
            block.get("intended_use"),
            block.get("claim_dependency"),
        ):
            if isinstance(value, str) and value.strip():
                texts.append((value, (index,)))
        for key in ("preservation_constraints", "dependency_ids", "grounding_refs"):
            items = block.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str) and item.strip():
                        texts.append((item, (index,)))
    requirements = plan.narration.get("requirements")
    if isinstance(requirements, dict):
        for key in ("language", "register"):
            value = requirements.get(key)
            if isinstance(value, str) and value.strip():
                texts.append((value, ()))
        for key in ("verification_dependency_ids",):
            items = requirements.get(key)
            if isinstance(items, list):
                texts.extend(
                    (str(item), ()) for item in items if isinstance(item, str) and item.strip()
                )
    for key in ("strategy_key", "direction_summary", "added_value_focus"):
        value = plan.strategy_snapshot.get(key)
        if isinstance(value, str) and value.strip():
            texts.append((value, ()))

    seen: set[str] = set()
    for text, indexes in texts:
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        violation = boundary_violation(text)
        if violation is not None:
            reason = _BOUNDARY_REASON.get(violation, _R.PLAN_INTEGRITY_INVALID)
            _add(
                evaluation.hard_failures,
                GovernanceSeverityClass.HARD_FAIL,
                reason,
                f"prohibited plan text ({violation})",
                block_indexes=indexes,
            )
        if _has_marker(text, PLATFORM_GUARANTEE_MARKERS):
            _add(
                evaluation.hard_failures,
                GovernanceSeverityClass.HARD_FAIL,
                _R.PLATFORM_EVASION_TACTIC,
                "platform-safety guarantee text",
                block_indexes=indexes,
            )
        if any(marker in folded for marker in FORBIDDEN_STATUS_MARKERS):
            _add(
                evaluation.hard_failures,
                GovernanceSeverityClass.HARD_FAIL,
                _R.PLAN_INTEGRITY_INVALID,
                "provider-supplied governor status",
                block_indexes=indexes,
            )
        if _has_marker(text, PLATFORM_EVASION_MARKERS):
            _add(
                evaluation.hard_failures,
                GovernanceSeverityClass.HARD_FAIL,
                _R.PLATFORM_EVASION_TACTIC,
                "platform-detection evasion instruction",
                block_indexes=indexes,
            )


def _substantive_findings(
    plan: PlanEvidence,
    inputs: GovernanceInputs,
    evidence: dict[str, object],
    evaluation: PlanEvaluation,
    config: Stage42Config,
) -> None:
    blocks = _blocks(plan)
    substantive = [
        block
        for block in blocks
        if _block_type(block)
        in {PlanBlockType.ORIGINAL_VALUE.value, PlanBlockType.TEXTUAL_ANNOTATION.value}
    ]
    if not substantive:
        _add(
            evaluation.hard_failures,
            GovernanceSeverityClass.HARD_FAIL,
            _R.NO_SUBSTANTIVE_VALUE,
            "plan contains no authored substantive contribution",
        )
        return

    additive_count = 0
    hard_hook = False
    hard_distortion = False
    hard_verified_claim = False
    for block in substantive:
        index = _int(block.get("index"))
        intent = _text(block.get("semantic_intent"))
        kind = _text(block.get("substantive_value_kind"))
        if kind in ADDITIVE_VALUE_KINDS and not _is_presentation_only(intent):
            additive_count += 1
        if _has_marker(f"{intent} {_text(block.get('purpose'))}", FAKE_HOOK_MARKERS):
            hard_hook = True
        if _has_marker(intent, DISTORTION_MARKERS):
            hard_distortion = True
        if _has_marker(intent, VERIFIED_CLAIM_MARKERS):
            hard_verified_claim = True
        if _is_paraphrase(intent, inputs.transcript, config):
            _add(
                evaluation.revisions,
                GovernanceSeverityClass.REVISION,
                _R.REDUNDANT_PARAPHRASE_ONLY,
                "authored contribution only paraphrases the source",
                block_indexes=(index,),
            )
        elif _is_presentation_only(intent):
            _add(
                evaluation.hard_failures,
                GovernanceSeverityClass.HARD_FAIL,
                _R.PRESENTATION_ONLY_TRANSFORMATION,
                "authored block is a presentation-only change",
                block_indexes=(index,),
            )
        elif _is_generic(intent):
            _add(
                evaluation.revisions,
                GovernanceSeverityClass.REVISION,
                _R.GENERIC_FILLER_ONLY,
                "authored contribution is generic filler",
                block_indexes=(index,),
            )

    if additive_count == 0:
        _add(
            evaluation.hard_failures,
            GovernanceSeverityClass.HARD_FAIL,
            _R.NO_SUBSTANTIVE_VALUE,
            "no authored block provides a substantive, additive value kind",
        )
    if hard_hook:
        _add(
            evaluation.hard_failures,
            GovernanceSeverityClass.HARD_FAIL,
            _R.FAKE_OR_MISLEADING_HOOK,
            "misleading or fake hook framing",
        )
    if hard_distortion:
        _add(
            evaluation.hard_failures,
            GovernanceSeverityClass.HARD_FAIL,
            _R.SEMANTIC_DISTORTION,
            "sensational or distorting framing",
        )
    if hard_verified_claim:
        _add(
            evaluation.hard_failures,
            GovernanceSeverityClass.HARD_FAIL,
            _R.UNSUPPORTED_CRITICAL_CLAIM,
            "plan asserts a verification that did not occur",
        )


def _retention_findings(
    plan: PlanEvidence,
    evidence: dict[str, object],
    evaluation: PlanEvaluation,
    config: Stage42Config,
) -> None:
    elapsed = float(evidence["elapsed_before_hero"])
    authored = float(evidence["authored_before_hero"])
    if elapsed > float(evidence["elapsed_cap"]):
        if authored > float(evidence["authored_cap"]):
            generic_preamble = all(
                _is_generic(intent) or _is_presentation_only(intent)
                for intent in [str(item) for item in evidence["intents"]]
                if intent
            )
            if generic_preamble:
                _add(
                    evaluation.hard_failures,
                    GovernanceSeverityClass.HARD_FAIL,
                    _R.NO_SUBSTANTIVE_VALUE,
                    "delayed hero is preceded only by generic authored material",
                )
            else:
                _add(
                    evaluation.revisions,
                    GovernanceSeverityClass.REVISION,
                    _R.EXCESSIVE_PREAMBLE,
                    "authored preamble delays the hero beyond the cap",
                )
        else:
            _add(
                evaluation.revisions,
                GovernanceSeverityClass.REVISION,
                _R.SOURCE_MOMENT_SEVERELY_DAMAGED,
                "hero does not appear early enough",
            )

    structure = str(evidence["source_moment_structure"])
    hero = int(evidence["hero_block_index"])
    if structure in {"JOKE", "PAYOFF", "QUESTION_ANSWER", "STORY"}:
        for index in evidence["interrupting_block_indexes"]:
            if isinstance(index, int) and index <= hero:
                _add(
                    evaluation.revisions,
                    GovernanceSeverityClass.REVISION,
                    _R.PAYOFF_INTERRUPTED,
                    "an interrupting block splits the source payoff",
                    block_indexes=(index,),
                )
                break

    if int(evidence["source_fragment_count"]) > config.over_fragmented_source_count:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.OVER_FRAGMENTED,
            "source moment is fragmented into too many excerpts",
        )


def _narration_findings(evidence: dict[str, object], evaluation: PlanEvaluation) -> None:
    finding = str(evidence["narration_finding"])
    if finding == NarrationBurdenFinding.EXCESSIVE.value:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.NARRATION_EXCESSIVE,
            "narration is excessive relative to the source moment",
        )
    elif finding == NarrationBurdenFinding.POSITION_DAMAGING.value:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.NARRATION_POSITION_DAMAGING,
            "narration is positioned where it damages the source moment",
        )
    elif finding == NarrationBurdenFinding.REDUNDANT.value:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.NARRATION_REDUNDANT,
            "narration repeats the source",
        )


def _proportionality_findings(
    plan: PlanEvidence, evidence: dict[str, object], evaluation: PlanEvaluation
) -> None:
    source_duration = float(evidence["source_duration"])
    original_duration = float(evidence["original_duration"])
    structure = str(evidence["source_moment_structure"])
    if structure in {"JOKE", "PAYOFF"} and original_duration > source_duration + 1.0:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.TRANSFORMATION_OVER_EDIT,
            "authored material exceeds and overwhelms the source moment",
        )
    elif int(evidence["source_fragment_count"]) <= 1 and original_duration > source_duration * 3:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.TRANSFORMATION_OVER_EDIT,
            "authored material is disproportionate to the source excerpt",
        )
    if plan.intensity == TransformationIntensity.STRONG.value and structure in {"JOKE", "PAYOFF"}:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.TRANSFORMATION_OVER_EDIT,
            "a strong transformation is applied to a fragile source moment",
        )


def _caution_dimensions(
    plan: PlanEvidence, evidence: dict[str, object], evaluation: PlanEvaluation
) -> None:
    template = str(evidence["template_level"])
    template_risk = _R.TEMPLATE_MASS_PRODUCED_FEEL if template in {"MODERATE", "HIGH"} else None
    if template_risk is not None and template == "HIGH":
        _add(
            evaluation.warnings,
            GovernanceSeverityClass.WARNING,
            template_risk,
            "plan structure reads as a mechanically imposed template",
        )
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.TEMPLATE_MASS_PRODUCED_FEEL,
            "template feel is high enough to require revision",
        )
    elif template_risk is not None:
        _add(
            evaluation.warnings,
            GovernanceSeverityClass.WARNING,
            template_risk,
            "plan structure shows some template/mass-produced feel",
        )

    dominance = str(evidence["source_dominance_level"])
    if dominance in {"MODERATE", "HIGH"}:
        _add(
            evaluation.warnings,
            GovernanceSeverityClass.WARNING,
            _R.SOURCE_DOMINANCE_CONCERN,
            "source material dominates the plan; verify the authored contribution is distinct",
        )


def build_dimensions(
    plan: PlanEvidence,
    inputs: GovernanceInputs,
    evidence: dict[str, object],
    evaluation: PlanEvaluation,
) -> None:
    retention = GovernanceEvidenceStrength.STRONG.value
    if evidence["elapsed_before_hero"] and float(evidence["elapsed_before_hero"]) > float(
        evidence["elapsed_cap"]
    ):
        retention = GovernanceEvidenceStrength.WEAK.value
    if any(
        finding.get("code") == _R.PAYOFF_INTERRUPTED.value
        for finding in evaluation.revisions + evaluation.hard_failures
    ) or any(
        finding.get("code") == _R.NARRATION_POSITION_DAMAGING.value
        for finding in evaluation.revisions + evaluation.hard_failures
    ):
        retention = GovernanceEvidenceStrength.NONE.value

    damage = GovernanceLevel.LOW.value
    if retention == GovernanceEvidenceStrength.WEAK.value:
        damage = GovernanceLevel.MODERATE.value
    elif retention == GovernanceEvidenceStrength.NONE.value:
        damage = GovernanceLevel.HIGH.value

    if not evidence["has_additive_value"]:
        originality = GovernanceEvidenceStrength.NONE.value
    elif _int(evidence["paraphrase_intents"], 0) == _int(evidence["substantive_count"], 0):
        originality = GovernanceEvidenceStrength.WEAK.value
    elif _int(evidence["generic_intents"], 0) or _int(evidence["presentation_only_intents"], 0):
        originality = GovernanceEvidenceStrength.ADEQUATE.value
    else:
        originality = GovernanceEvidenceStrength.STRONG.value

    filler = GovernanceLevel.LOW.value
    if _int(evidence["generic_intents"], 0):
        filler = GovernanceLevel.MODERATE.value
    if _int(evidence["generic_intents"], 0) >= 2:
        filler = GovernanceLevel.HIGH.value

    redundancy = GovernanceLevel.LOW.value
    if _int(evidence["paraphrase_intents"], 0):
        redundancy = GovernanceLevel.MODERATE.value
    if _int(evidence["paraphrase_intents"], 0) >= 2:
        redundancy = GovernanceLevel.HIGH.value

    template = str(evidence["template_level"])
    coherence = (
        GovernanceEvidenceStrength.WEAK.value
        if template == "HIGH"
        else GovernanceEvidenceStrength.ADEQUATE.value
    )

    evidence["narration_finding"] = str(evidence.get("narration_finding", "UNKNOWN"))
    evidence["source_dominance_level"] = str(evidence.get("source_dominance_level", "UNKNOWN"))

    proportionality = GovernanceLevel.LOW.value
    if any(
        finding.get("code") == _R.TRANSFORMATION_OVER_EDIT.value for finding in evaluation.revisions
    ):
        proportionality = GovernanceLevel.MODERATE.value
    if _num(evidence["source_duration"]) > 0 and _num(evidence["original_duration"]) > 2 * _num(
        evidence["source_duration"]
    ):
        proportionality = GovernanceLevel.HIGH.value

    evaluation.dimensions = {
        "retention_preservation": retention,
        "source_moment_damage": damage,
        "substantive_originality": originality,
        "source_dominance": evidence["source_dominance_level"],
        "semantic_fidelity": GovernanceEvidenceStrength.UNKNOWN.value,
        "generic_filler_risk": filler,
        "redundant_commentary_risk": redundancy,
        "narration_burden": evidence["narration_finding"],
        "verification_completeness": evaluation.verification.get("claim_state"),
        "template_mass_produced_feel": template,
        "substantive_transformation": (
            GovernanceLevel.LOW.value
            if evidence["has_additive_value"]
            else GovernanceLevel.HIGH.value
        ),
        "plan_coherence": coherence,
        "transformation_proportionality": proportionality,
    }


def build_platform_risk(
    plan: PlanEvidence,
    inputs: GovernanceInputs,
    evidence: dict[str, object],
    evaluation: PlanEvaluation,
) -> None:
    from app.transformation.governance.policy import (
        ACCOUNT_LEVEL_REPETITION,
        PLATFORM_LIMITATIONS,
        PLATFORM_POLICY_CHECKED_AT,
        PLATFORM_POLICY_PROFILE_VERSION,
    )

    third_party = inputs.rights_risk != "LOW" or inputs.originality_risk not in {
        "NOT_INDICATED",
    }
    has_value = bool(evidence["has_additive_value"])
    source_ratio = _num(evidence["source_ratio"])
    template = str(evidence["template_level"])
    presentation_only = _int(evidence["presentation_only_intents"], 0) > 0
    generic = _int(evidence["generic_intents"], 0) > 0

    def level_reasons(level: str, reasons: list[str], texts: list[str]) -> dict[str, object]:
        return {
            "level": level,
            "reason_codes": reasons,
            "evidence": texts,
        }

    if not has_value or presentation_only:
        reused = PlatformRiskLevel.HIGH.value
        reused_reasons = [_R.PLATFORM_REUSE_RISK.value, _R.PRESENTATION_ONLY_TRANSFORMATION.value]
    elif third_party and source_ratio >= 0.85:
        reused = PlatformRiskLevel.MODERATE.value
        reused_reasons = [_R.PLATFORM_REUSE_RISK.value, _R.SOURCE_DOMINANCE_CONCERN.value]
    elif third_party and plan.intensity == TransformationIntensity.MINIMAL.value:
        reused = PlatformRiskLevel.MODERATE.value
        reused_reasons = [_R.PLATFORM_REUSE_RISK.value]
    else:
        reused = PlatformRiskLevel.LOW.value
        reused_reasons = []

    inauthentic = (
        PlatformRiskLevel.HIGH.value
        if template == "HIGH" and generic
        else PlatformRiskLevel.MODERATE.value
        if template in {"MODERATE", "HIGH"}
        else PlatformRiskLevel.LOW.value
    )
    deceptive_codes: list[str] = []
    if any(
        finding.get("code") in {_R.FAKE_OR_MISLEADING_HOOK.value, _R.SEMANTIC_DISTORTION.value}
        for finding in evaluation.hard_failures
    ):
        spam = PlatformRiskLevel.HIGH.value
        deceptive_codes = [_R.FAKE_OR_MISLEADING_HOOK.value]
    elif _R.UNSUPPORTED_CRITICAL_CLAIM.value in [
        finding.get("code") for finding in evaluation.warnings
    ]:
        spam = PlatformRiskLevel.MODERATE.value
        deceptive_codes = [_R.UNSUPPORTED_CRITICAL_CLAIM.value]
    else:
        spam = PlatformRiskLevel.LOW.value

    if not has_value or presentation_only:
        facebook = PlatformRiskLevel.HIGH.value
        facebook_reasons = [_R.PLATFORM_REUSE_RISK.value]
    elif (
        third_party
        and source_ratio >= 0.85
        and plan.intensity == TransformationIntensity.MINIMAL.value
    ):
        facebook = PlatformRiskLevel.MODERATE.value
        facebook_reasons = [_R.PLATFORM_REUSE_RISK.value]
    else:
        facebook = PlatformRiskLevel.LOW.value
        facebook_reasons = []

    facebook_spam = (
        PlatformRiskLevel.HIGH.value
        if template == "HIGH" and generic
        else PlatformRiskLevel.MODERATE.value
        if template in {"MODERATE", "HIGH"}
        else PlatformRiskLevel.LOW.value
    )

    evaluation.platform_risk = {
        "policy_profile_version": PLATFORM_POLICY_PROFILE_VERSION,
        "policy_checked_at": PLATFORM_POLICY_CHECKED_AT,
        "youtube": {
            "reused_content": level_reasons(
                reused,
                reused_reasons,
                [f"source_ratio={source_ratio}", f"intensity={plan.intensity}"],
            ),
            "inauthentic_mass_produced": level_reasons(
                inauthentic,
                [_R.TEMPLATE_MASS_PRODUCED_FEEL.value] if template != "LOW" else [],
                [f"template_feel={template}"],
            ),
            "spam_deceptive_practices": level_reasons(spam, deceptive_codes, []),
        },
        "facebook": {
            "unoriginal_content": level_reasons(facebook, facebook_reasons, []),
            "spam_repetitive": level_reasons(
                facebook_spam,
                [_R.TEMPLATE_MASS_PRODUCED_FEEL.value] if template != "LOW" else [],
                [],
            ),
        },
        "generic": {
            "source_dominance": level_reasons(
                str(evidence["source_dominance_level"]),
                [_R.SOURCE_DOMINANCE_CONCERN.value]
                if evidence["source_dominance_level"] != "LOW"
                else [],
                [f"source_ratio={source_ratio}"],
            ),
            "substantive_transformation": level_reasons(
                "LOW" if has_value else "HIGH",
                [] if has_value else [_R.NO_SUBSTANTIVE_VALUE.value],
                [f"value_kinds={_as_text_list(evidence['value_kinds'])}"],
            ),
            "template_mass_produced_feel": level_reasons(template, [], []),
        },
        "account_level_repetition": ACCOUNT_LEVEL_REPETITION,
        "limitations": list(PLATFORM_LIMITATIONS),
    }


def evaluate_plan(
    plan: PlanEvidence,
    inputs: GovernanceInputs,
    config: Stage42Config,
    *,
    provider_mode_deterministic: bool,
) -> PlanEvaluation:
    evaluation = PlanEvaluation(plan=plan)
    integrity_ok, reasons = revalidate_integrity(plan, config)
    evaluation.integrity_ok = integrity_ok
    evaluation.integrity_reasons = reasons
    evidence = derive_evidence(plan, inputs, config)
    evaluation.evidence = evidence

    if not integrity_ok:
        _add(
            evaluation.hard_failures,
            GovernanceSeverityClass.HARD_FAIL,
            _R.PLAN_INTEGRITY_INVALID,
            "persisted plan integrity is invalid: " + ",".join(reasons[:4]),
        )
        evidence["narration_finding"] = (
            NarrationBurdenFinding.UNKNOWN.value
            if _text(plan.narration.get("need", "NONE")) != NarrationNeed.NONE.value
            else NarrationBurdenFinding.APPROPRIATE.value
        )
        evidence["template_level"] = "UNKNOWN"
        evidence["source_dominance_level"] = "UNKNOWN"
        evaluation.verification = {
            "claim_state": ClaimGroundingState.NOT_APPLICABLE.value,
            "claims": [],
            "unresolved": False,
            "reason_codes": [],
        }
        build_dimensions(plan, inputs, evidence, evaluation)
        build_platform_risk(plan, inputs, evidence, evaluation)
        evaluation.requires_semantic_review = False
        evaluation.semantic_state = SEMANTIC_NOT_REQUIRED
        return evaluation

    evidence["template_level"] = _template_level(evidence, plan)
    evidence["source_dominance_level"] = _source_dominance_level(evidence, plan)
    evidence["narration_finding"] = _narration_burden(evidence, plan, config)

    _boundary_findings(plan, evaluation)
    _substantive_findings(plan, inputs, evidence, evaluation, config)
    _retention_findings(plan, evidence, evaluation, config)
    _narration_findings(evidence, evaluation)
    _proportionality_findings(plan, evidence, evaluation)
    _caution_dimensions(plan, evidence, evaluation)

    evaluation.verification = _verification(plan, inputs, evidence)
    if (
        evaluation.verification["claim_state"]
        == ClaimGroundingState.EXTERNAL_REQUIRED_UNRESOLVED.value
    ):
        _add(
            evaluation.blocking,
            GovernanceSeverityClass.BLOCKING_CONDITION,
            _R.EXTERNAL_VERIFICATION_REQUIRED,
            "essential external verification is unresolved",
        )
    elif (
        evaluation.verification["claim_state"]
        == ClaimGroundingState.UNSUPPORTED_OR_FABRICATED.value
    ):
        _add(
            evaluation.hard_failures,
            GovernanceSeverityClass.HARD_FAIL,
            _R.UNSUPPORTED_CRITICAL_CLAIM,
            "plan depends on an unsupported or fabricated critical claim",
        )

    if not (evaluation.hard_failures or evaluation.blocking or evaluation.revisions):
        evaluation.requires_semantic_review = _requires_semantic_review(
            plan, evidence, evaluation, config, provider_mode_deterministic
        )
    evaluation.semantic_state = (
        SEMANTIC_NOT_ATTEMPTED if evaluation.requires_semantic_review else SEMANTIC_NOT_REQUIRED
    )

    build_dimensions(plan, inputs, evidence, evaluation)
    build_platform_risk(plan, inputs, evidence, evaluation)
    return evaluation


def _requires_semantic_review(
    plan: PlanEvidence,
    evidence: dict[str, object],
    evaluation: PlanEvaluation,
    config: Stage42Config,
    provider_mode_deterministic: bool,
) -> bool:
    if provider_mode_deterministic:
        return False
    strong = (
        bool(evidence["has_additive_value"])
        and not _int(evidence["generic_intents"], 0)
        and not _int(evidence["paraphrase_intents"], 0)
        and not _int(evidence["presentation_only_intents"], 0)
        and _num(evidence["authored_before_hero"]) <= _num(evidence["authored_cap"])
        and _num(evidence["elapsed_before_hero"]) <= _num(evidence["elapsed_cap"])
        and str(evidence["template_level"]) == "LOW"
        and str(evidence["narration_need"]) in {"NONE", "OPTIONAL"}
        and evaluation.verification.get("claim_state")
        == ClaimGroundingState.GROUNDED_IN_SOURCE.value
    )
    return not strong


# ---------------------------------------------------------------------------
# Semantic review application
# ---------------------------------------------------------------------------


def apply_semantic_review(
    evaluation: PlanEvaluation,
    critique: ProviderCritique | None,
    state: str,
) -> None:
    evaluation.semantic_state = state
    if state == SEMANTIC_NOT_REQUIRED:
        return
    if critique is None:
        evaluation.provider_evidence = {"state": state}
        return
    evaluation.critique = critique
    evaluation.provider_evidence = {"state": state, "critique": critique.as_dict()}

    dimensions = evaluation.dimensions
    dimensions["semantic_fidelity"] = {
        SemanticFidelityFinding.PASS: GovernanceEvidenceStrength.STRONG.value,
        SemanticFidelityFinding.CONCERN: GovernanceEvidenceStrength.WEAK.value,
        SemanticFidelityFinding.FAIL: GovernanceEvidenceStrength.NONE.value,
        SemanticFidelityFinding.UNKNOWN: GovernanceEvidenceStrength.UNKNOWN.value,
    }[critique.fidelity]

    if critique.fidelity == SemanticFidelityFinding.FAIL:
        _add(
            evaluation.hard_failures,
            GovernanceSeverityClass.HARD_FAIL,
            _R.SEMANTIC_DISTORTION,
            "semantic review found a fidelity failure",
        )
    elif critique.fidelity == SemanticFidelityFinding.CONCERN:
        _add(
            evaluation.warnings,
            GovernanceSeverityClass.WARNING,
            _R.SEMANTIC_FIDELITY_CONCERN,
            "semantic review found a fidelity concern",
        )

    if critique.value == SubstantiveValueFinding.NONE:
        _add(
            evaluation.hard_failures,
            GovernanceSeverityClass.HARD_FAIL,
            _R.NO_SUBSTANTIVE_VALUE,
            "semantic review found no substantive value",
        )
    elif critique.value == SubstantiveValueFinding.REDUNDANT:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.REDUNDANT_PARAPHRASE_ONLY,
            "semantic review found the contribution redundant",
        )
    elif critique.value == SubstantiveValueFinding.GENERIC:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.GENERIC_FILLER_ONLY,
            "semantic review found generic filler",
        )

    if critique.unsupported_claim == UnsupportedClaimFinding.CLEAR:
        _add(
            evaluation.hard_failures,
            GovernanceSeverityClass.HARD_FAIL,
            _R.UNSUPPORTED_CRITICAL_CLAIM,
            "semantic review found a clear unsupported claim",
        )
    elif critique.unsupported_claim == UnsupportedClaimFinding.POSSIBLE:
        _add(
            evaluation.warnings,
            GovernanceSeverityClass.WARNING,
            _R.UNSUPPORTED_CRITICAL_CLAIM,
            "semantic review flagged a possibly unsupported claim",
        )

    if critique.retention == RetentionEffectFinding.DAMAGED:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.SOURCE_MOMENT_SEVERELY_DAMAGED,
            "semantic review found the source moment damaged",
        )
    elif critique.retention == RetentionEffectFinding.MIXED:
        _add(
            evaluation.warnings,
            GovernanceSeverityClass.WARNING,
            _R.SOURCE_MOMENT_SEVERELY_DAMAGED,
            "semantic review found mixed retention",
        )

    if critique.coherence == CoherenceFinding.INCOHERENT:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.PLAN_INCOHERENT,
            "semantic review found the plan incoherent",
        )
    elif critique.coherence == CoherenceFinding.MIXED:
        _add(
            evaluation.warnings,
            GovernanceSeverityClass.WARNING,
            _R.PLAN_INCOHERENT,
            "semantic review found mixed coherence",
        )

    if critique.narration == NarrationBurdenFinding.EXCESSIVE:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.NARRATION_EXCESSIVE,
            "semantic review found narration excessive",
        )
    elif critique.narration == NarrationBurdenFinding.REDUNDANT:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.NARRATION_REDUNDANT,
            "semantic review found narration redundant",
        )
    elif critique.narration == NarrationBurdenFinding.POSITION_DAMAGING:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.NARRATION_POSITION_DAMAGING,
            "semantic review found narration position damaging",
        )

    if critique.template_feel == GovernanceLevel.HIGH:
        _add(
            evaluation.revisions,
            GovernanceSeverityClass.REVISION,
            _R.TEMPLATE_MASS_PRODUCED_FEEL,
            "semantic review found strong template/mass-produced feel",
        )
    elif critique.template_feel == GovernanceLevel.MODERATE:
        _add(
            evaluation.warnings,
            GovernanceSeverityClass.WARNING,
            _R.TEMPLATE_MASS_PRODUCED_FEEL,
            "semantic review found moderate template feel",
        )


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def _remediation(evaluation: PlanEvaluation) -> tuple[dict[str, object], ...]:
    actions: list[dict[str, object]] = []
    codes = {str(finding.get("code")) for finding in evaluation.revisions + evaluation.blocking}
    indexes_for_preamble = [
        index
        for index, block in enumerate(_blocks(evaluation.plan))
        if _block_type(block)
        in {PlanBlockType.ORIGINAL_VALUE.value, PlanBlockType.TEXTUAL_ANNOTATION.value}
        and index < evaluation.plan.hero_block_index
    ]
    if _R.EXCESSIVE_PREAMBLE.value in codes:
        actions.append(
            {
                "action_code": GovernanceRemediationAction.MOVE_EXPLANATION_AFTER_HERO.value,
                "target_block_indexes": indexes_for_preamble,
                "priority": GovernanceRemediationPriority.REQUIRED.value,
                "note": "Move authored explanation after the hero source moment.",
            }
        )
    if _R.REDUNDANT_PARAPHRASE_ONLY.value in codes:
        actions.append(
            {
                "action_code": GovernanceRemediationAction.REMOVE_PARAPHRASE.value,
                "target_block_indexes": [],
                "priority": GovernanceRemediationPriority.REQUIRED.value,
                "note": "Remove paraphrase-only commentary; keep a distinct contribution.",
            }
        )
    if _R.GENERIC_FILLER_ONLY.value in codes:
        actions.append(
            {
                "action_code": GovernanceRemediationAction.REPLACE_GENERIC_TAKEAWAY.value,
                "target_block_indexes": [],
                "priority": GovernanceRemediationPriority.REQUIRED.value,
                "note": "Replace generic takeaway intent with genuine analysis.",
            }
        )
    if _R.NARRATION_EXCESSIVE.value in codes or _R.NARRATION_REDUNDANT.value in codes:
        actions.append(
            {
                "action_code": GovernanceRemediationAction.REDUCE_NARRATION.value,
                "target_block_indexes": [],
                "priority": GovernanceRemediationPriority.REQUIRED.value,
                "note": "Reduce or remove narration that does not add value.",
            }
        )
    if _R.NARRATION_POSITION_DAMAGING.value in codes:
        actions.append(
            {
                "action_code": GovernanceRemediationAction.PRESERVE_PAYOFF.value,
                "target_block_indexes": [],
                "priority": GovernanceRemediationPriority.REQUIRED.value,
                "note": "Do not interrupt the source payoff with narration.",
            }
        )
    if _R.PAYOFF_INTERRUPTED.value in codes or _R.SOURCE_MOMENT_SEVERELY_DAMAGED.value in codes:
        actions.append(
            {
                "action_code": GovernanceRemediationAction.PRESERVE_PAYOFF.value,
                "target_block_indexes": [],
                "priority": GovernanceRemediationPriority.REQUIRED.value,
                "note": "Preserve the source payoff and pacing.",
            }
        )
    if _R.TEMPLATE_MASS_PRODUCED_FEEL.value in codes:
        actions.append(
            {
                "action_code": GovernanceRemediationAction.REMOVE_REDUNDANT_INTRO.value,
                "target_block_indexes": [],
                "priority": GovernanceRemediationPriority.RECOMMENDED.value,
                "note": "Remove template scaffolding to reduce mass-produced feel.",
            }
        )
    if _R.TRANSFORMATION_OVER_EDIT.value in codes:
        actions.append(
            {
                "action_code": GovernanceRemediationAction.PRESERVE_PAYOFF.value,
                "target_block_indexes": [],
                "priority": GovernanceRemediationPriority.REQUIRED.value,
                "note": "Reduce transformation intensity to preserve the source moment.",
            }
        )
    if _R.EXTERNAL_VERIFICATION_REQUIRED.value in codes:
        actions.append(
            {
                "action_code": GovernanceRemediationAction.RESOLVE_VERIFICATION.value,
                "target_block_indexes": [],
                "priority": GovernanceRemediationPriority.REQUIRED.value,
                "note": "Resolve the essential external verification dependency.",
            }
        )
    return tuple(actions)


def _all_reason_codes(evaluation: PlanEvaluation) -> tuple[str, ...]:
    ordered: list[str] = []
    for bucket in (
        evaluation.hard_failures,
        evaluation.blocking,
        evaluation.revisions,
        evaluation.warnings,
        evaluation.advisories,
    ):
        for finding in bucket:
            code = str(finding.get("code"))
            if code and code not in ordered:
                ordered.append(code)
    return tuple(ordered)


def finalize(evaluation: PlanEvaluation) -> tuple[GovernancePlanStatus, GovernanceSeverityClass]:
    if evaluation.hard_failures:
        return GovernancePlanStatus.REJECTED_BY_GOVERNOR, GovernanceSeverityClass.HARD_FAIL
    if evaluation.blocking:
        return (
            GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION,
            GovernanceSeverityClass.BLOCKING_CONDITION,
        )
    if evaluation.revisions:
        return GovernancePlanStatus.REVISION_REQUIRED, GovernanceSeverityClass.REVISION
    if evaluation.requires_semantic_review and evaluation.semantic_state in {
        SEMANTIC_UNAVAILABLE,
        SEMANTIC_INVALID,
        SEMANTIC_NOT_ATTEMPTED,
    }:
        return GovernancePlanStatus.GOVERNANCE_DEFERRED, GovernanceSeverityClass.WARNING
    if evaluation.critique is not None:
        if evaluation.critique.fidelity == SemanticFidelityFinding.UNKNOWN or (
            evaluation.critique.value == SubstantiveValueFinding.UNKNOWN
        ):
            return GovernancePlanStatus.GOVERNANCE_DEFERRED, GovernanceSeverityClass.WARNING
        if evaluation.critique.retention == RetentionEffectFinding.UNKNOWN:
            return GovernancePlanStatus.GOVERNANCE_DEFERRED, GovernanceSeverityClass.WARNING
    if evaluation.warnings:
        return GovernancePlanStatus.APPROVED_WITH_CAUTION, GovernanceSeverityClass.WARNING
    return GovernancePlanStatus.APPROVED_FOR_SELECTION, GovernanceSeverityClass.ADVISORY


def build_plan_governance(
    evaluation: PlanEvaluation,
    *,
    input_fingerprint: str,
    output_fingerprint: str,
) -> PlanGovernance:
    status, severity = finalize(evaluation)
    eligible = status in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }
    criticisms = (
        {
            "state": evaluation.semantic_state,
            "critique": evaluation.critique.as_dict() if evaluation.critique else None,
        }
        if evaluation.requires_semantic_review
        else {"state": SEMANTIC_NOT_REQUIRED, "critique": None}
    )
    return PlanGovernance(
        plan_id=evaluation.plan.plan_id,
        plan_output_fingerprint=evaluation.plan.plan_output_fingerprint,
        status=status,
        eligible_for_stage4_3=eligible,
        severity=severity,
        hard_gates=tuple(evaluation.hard_failures + evaluation.blocking),
        dimensions=dict(evaluation.dimensions),
        verification=dict(evaluation.verification),
        platform_risk=dict(evaluation.platform_risk),
        reason_codes=_all_reason_codes(evaluation),
        warnings=tuple(evaluation.warnings + evaluation.advisories),
        remediation=_remediation(evaluation),
        provider_evidence=criticisms,
        input_fingerprint=input_fingerprint,
        output_fingerprint=output_fingerprint,
    )


__all__ = [
    "PlanEvaluation",
    "SEMANTIC_AVAILABLE",
    "SEMANTIC_INVALID",
    "SEMANTIC_NOT_ATTEMPTED",
    "SEMANTIC_NOT_REQUIRED",
    "SEMANTIC_UNAVAILABLE",
    "apply_semantic_review",
    "build_dimensions",
    "build_plan_governance",
    "build_platform_risk",
    "derive_evidence",
    "evaluate_plan",
    "finalize",
    "revalidate_integrity",
]
