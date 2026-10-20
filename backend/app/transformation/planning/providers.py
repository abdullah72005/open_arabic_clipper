"""Stage 4.1 provider protocol, strict structured schema, and deterministic provider.

Separate from the Stage 4.0 directions-only provider: Stage 4.1 needs a complete
structured plan (blocks, source selections, narration semantics, verification
placeholders). Providers never return a finished script, timeline, TTS text,
voice/provider selection, or rendering instruction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

from pydantic import BaseModel, Field

from app.candidates.providers import ProviderErrorCategory  # noqa: F401  (re-exported)
from app.core.enums import (
    DeliveryIntent,
    NarrationNeed,
    NarrationPurpose,
    PlanBlockType,
    SourceExcerptRole,
    SubstantiveValueKind,
)
from app.transformation.hook_payoff import (
    MANDATORY_COVERAGE_TOLERANCE_SECONDS,
    spans_cover_endpoints,
)
from app.transformation.planning.policy import (
    SCHEMA_VERSION,
    is_strict_hero_window,
    strategy_value_kinds,
)
from app.transformation.planning.types import (
    NarrationRequirement,
    PlanningInputs,
    PlanProviderBlock,
    PlanProviderPlan,
    PlanProviderResult,
)
from app.transformation.types import clamp

PLANNING_PRIORITY = "HIGH"

_BLOCK_TYPES = [item.value for item in PlanBlockType]
_SOURCE_ROLES = [item.value for item in SourceExcerptRole]
_VALUE_KINDS = [item.value for item in SubstantiveValueKind]
_NARRATION_NEEDS = [item.value for item in NarrationNeed]
_NARRATION_PURPOSES = [item.value for item in NarrationPurpose]
_DELIVERY = [item.value for item in DeliveryIntent]

_BlockTypeLiteral = Literal[
    "SOURCE_EXCERPT",
    "ORIGINAL_VALUE",
    "TRANSITION",
    "TEXTUAL_ANNOTATION",
    "FACT_VERIFICATION_PLACEHOLDER",
]
_SourceRoleLiteral = Literal["HERO", "HOOK", "PAYOFF", "SUPPORT"]
_ValueKindLiteral = Literal[
    "MISSING_CONTEXT",
    "INFERENCE",
    "EXPLANATION",
    "COMPARISON",
    "COUNTERPOINT",
    "VERIFICATION_CORRECTION",
    "SYNTHESIS",
    "AUTHORED_THESIS",
    "USEFUL_TAKEAWAY",
    "SOURCE_AS_EVIDENCE",
]
_DeliveryIntentLiteral = Literal["ON_SCREEN_TEXT", "NARRATION", "FLEXIBLE"]
_NarrationNeedLiteral = Literal["NONE", "OPTIONAL", "RECOMMENDED", "REQUIRED"]
_NarrationPurposeLiteral = Literal[
    "CONTEXT",
    "ANALYSIS",
    "COUNTERPOINT",
    "EXPLANATION",
    "TAKEAWAY",
    "HOOK",
    "TRANSITION",
]


class PlanningNarrationSchema(BaseModel):
    """Narration is a semantic requirement, never a TTS/voice/render decision."""

    need: _NarrationNeedLiteral = Field(
        description="NONE unless narration is genuinely necessary for the plan."
    )
    purposes: list[_NarrationPurposeLiteral] = Field(
        description="Closed narration purposes. Provide an empty list when need is NONE."
    )
    language: str | None = Field(
        default=None, description="Output language policy; never the source dialect."
    )
    register_intent: str | None = Field(
        default=None, description="Register intent for any narration wording."
    )
    estimated_duration: float = Field(
        description="Positive estimated narration seconds; 0.0 when need is NONE."
    )
    placement_block_index: int | None = Field(
        default=None, description="Block index where narration should be placed."
    )
    max_source_interruption_seconds: float = 0.0
    overlaps_source_audio: bool = False
    replaces_silence: bool = False
    essential: bool = Field(
        default=False,
        description="True only when removing narration would make the plan invalid.",
    )
    verification_dependency_ids: list[str] = Field(
        description="Claim ids from FACT_VERIFICATION_PLACEHOLDER blocks that narration uses."
    )


class PlanningBlockSchema(BaseModel):
    """One plan block. Conditional requirements are stated in descriptions."""

    block_type: _BlockTypeLiteral = Field(
        description=(
            "SOURCE_EXCERPT, ORIGINAL_VALUE, TRANSITION, TEXTUAL_ANNOTATION, or "
            "FACT_VERIFICATION_PLACEHOLDER. Unknown values are rejected."
        )
    )
    purpose: str = Field(description="Short purpose of this block.")
    estimated_duration: float = Field(
        description="Non-negative estimated seconds contributed by this block."
    )
    interrupts_source: bool = Field(
        default=False, description="True only when this block interrupts source audio."
    )
    preservation_constraints: list[str] = Field(
        description="Explicit source-preservation constraints for this block."
    )
    dependency_ids: list[str] = Field(
        description=(
            "Claim ids from FACT_VERIFICATION_PLACEHOLDER blocks that this block "
            "depends on. Empty when the block has no external dependency."
        )
    )
    source_role: _SourceRoleLiteral | None = Field(
        default=None,
        description=(
            "SOURCE_EXCERPT only. Exactly one excerpt is HERO and it must be block 0 "
            "or block 1. HERO/HOOK/PAYOFF/SUPPORT labels alone never prove mandatory "
            "hook/payoff coverage."
        ),
    )
    word_start_index: int | None = Field(
        default=None,
        description=(
            "SOURCE_EXCERPT only. First inclusive zero-based planning-word index from "
            "the provided words list. These are planning indexes, not FINAL_CLIP indexes."
        ),
    )
    word_end_index: int | None = Field(
        default=None,
        description=(
            "SOURCE_EXCERPT only. Last inclusive planning-word index; must be >= "
            "word_start_index and inside the offered words."
        ),
    )
    use_full_window: bool = Field(
        default=False,
        description="Only when word evidence is unavailable: select the whole refined window.",
    )
    continuity_rationale: str | None = None
    substantive_value_kind: _ValueKindLiteral | None = Field(
        default=None,
        description=(
            "ORIGINAL_VALUE/TEXTUAL_ANNOTATION only. Must be one of "
            "strategy_constraints.allowed_value_kinds."
        ),
    )
    semantic_intent: str = Field(
        description=(
            "Required for ORIGINAL_VALUE/TEXTUAL_ANNOTATION: the specific value the "
            "viewer gains beyond the selected source excerpt. Empty string otherwise."
        )
    )
    why_unavailable: str = Field(
        description=(
            "Required for ORIGINAL_VALUE/TEXTUAL_ANNOTATION: why the selected source "
            "excerpt alone does not already provide that value. Empty string otherwise."
        )
    )
    grounding_refs: list[str] = Field(
        description=(
            "Required non-empty for every ORIGINAL_VALUE/TEXTUAL_ANNOTATION unless a "
            "genuine FACT_VERIFICATION_PLACEHOLDER dependency covers it. Every supplied "
            "declaration must resolve against the selected SOURCE_EXCERPT evidence. "
            "Grammar: 'block:N' where N is the zero-based index of a SOURCE_EXCERPT in "
            "the complete returned blocks array; 'words:N-M' with inclusive planning-word "
            "indexes offered in the words list; 'source:A-B' with absolute seconds covered "
            "exactly by selected evidence; or a plain quote equal to an entire selected "
            "SOURCE_EXCERPT text after strip/casefold. A bare decimal such as 793.32 is a "
            "quote, not a timestamp. A substring of an excerpt is not a valid quote. Do not "
            "use object references or final_clip."
        )
    )
    delivery_intent: _DeliveryIntentLiteral | None = Field(
        default=None,
        description="ON_SCREEN_TEXT, NARRATION, or FLEXIBLE. Must agree with narration.need.",
    )
    draft_line: str | None = Field(
        default=None,
        description=(
            "Optional short drafted wording. Any number in it must already appear in the "
            "source transcript; otherwise provide a verification placeholder dependency."
        ),
    )
    claim_dependency: str | None = Field(
        default=None,
        description="FACT_VERIFICATION_PLACEHOLDER only: bounded claim id.",
    )
    verification_rationale: str | None = None
    intended_use: str | None = None
    must_verify_before_execution: bool = Field(
        default=False,
        description="FACT_VERIFICATION_PLACEHOLDER only: must be true.",
    )
    dependent_block_ids: list[int] = Field(
        description=(
            "FACT_VERIFICATION_PLACEHOLDER only: zero-based indexes of the substantive "
            "blocks that depend on this claim. Empty for other block types."
        )
    )


class PlanningPlanSchema(BaseModel):
    """One plan for exactly one requested strategy identity."""

    strategy_id: str = Field(description="Exact requested strategy_id.")
    strategy_key: str = Field(description="Exact requested strategy_key.")
    confidence: float = Field(description="Planner confidence between 0.0 and 1.0.")
    no_valid_plan: bool = Field(
        description=(
            "True only when no valid plan can be produced. Then blocks must be empty and "
            "no_valid_reason must be a short bounded reason."
        )
    )
    no_valid_reason: str = Field(
        description="Short reason when no_valid_plan is true; empty string otherwise."
    )
    planner_notes: str = Field(description="Short bounded notes; never a script or status.")
    preservation_constraints: list[str] = Field(
        description="Plan-level source-preservation constraints."
    )
    blocks: list[PlanningBlockSchema] = Field(
        description=(
            "Ordered blocks. A valid plan has at least one SOURCE_EXCERPT and exactly one "
            "HERO at block 0 or 1. Selected SOURCE_EXCERPT spans must cover every resolved "
            "mandatory hook/payoff segment endpoint. Nonoverlapping chronological excerpts."
        )
    )
    narration: PlanningNarrationSchema = Field(
        description="Narration semantic requirement. Use need=NONE when not needed."
    )


class PlanningOutputSchema(BaseModel):
    """Authoritative hosted structured-output schema for Stage 4.1 planning."""

    plans: list[PlanningPlanSchema] = Field(
        description="Exactly one plan per requested strategy; no unrequested identities."
    )


# One authoritative contract: the same Pydantic model Google receives and the
# schema hashed into provider/prompt identity. A separate documentary schema is
# deliberately not permitted.
PLANNING_OUTPUT_SCHEMA: dict[str, object] = PlanningOutputSchema.model_json_schema()


PLANNING_SYSTEM_INSTRUCTION = (
    "You produce concrete structured transformation plans for one short Arabic "
    "source moment, exactly one plan per requested Stage 4.0 strategy. "
    "Return only the strict JSON schema. Each plan must repeat the exact requested "
    "strategy_id and strategy_key; never invent, rename, or omit a requested identity. "
    "If a strategy cannot produce a valid substantive plan, set no_valid_plan=true, "
    "leave blocks empty, and give a short bounded no_valid_reason. Use the "
    "no_valid_plan shape only for a genuine decline, never for malformed output. "
    "A valid plan has at least one SOURCE_EXCERPT and exactly one HERO source excerpt "
    "placed as block 0 or block 1. Do not delay the source behind a long preamble. "
    "SOURCE_EXCERPT selection uses only the provided planning-word indexes: "
    "word_start_index and word_end_index are inclusive zero-based planning indexes from "
    "the words list. Never invent timestamps or source text. Selected excerpts must be "
    "chronological, nonoverlapping, and at least min_source_excerpt_seconds long. "
    "Every selected excerpt must stay inside the selectable window in "
    "source_selection. Never treat planning indexes as FINAL_CLIP indexes. "
    "Read mandatory_evidence. Each of hook and payoff reports state ABSENT, RESOLVED, "
    "MALFORMED, or UNRESOLVED, plus the authoritative source segment index and absolute "
    "bounds. If state is RESOLVED, the union of selected SOURCE_EXCERPT spans must cover "
    "both the segment start and end (within the stated tolerance). Role labels such as "
    "HERO, HOOK, PAYOFF, or SUPPORT do not prove coverage. A null payoff does not waive "
    "a hook. If a declared hook or payoff is MALFORMED or UNRESOLVED, decline with "
    "no_valid_plan rather than guessing. For a RESOLVED segment, mandatory_coverage_feasible "
    "states whether a supported source selection can preserve both endpoints; "
    "supported_selection reports the method (WORD_SELECTION or FULL_WINDOW) and the "
    "selectable planning-word range. When word evidence is insufficient, the supported "
    "full-window fallback is a valid selection. Decline with no_valid_plan only when "
    "mandatory_coverage_feasible is false. covered_by_offered_words is descriptive word "
    "mapping only; a false value does not require declining when a supported selection "
    "covers the segment. "
    "Every ORIGINAL_VALUE and TEXTUAL_ANNOTATION requires at least one grounding "
    "declaration in grounding_refs, except a genuine two-way external-verification "
    "linkage. Every supplied declaration must resolve against the selected "
    "SOURCE_EXCERPT evidence; one valid declaration never excuses another invalid one. "
    "The grounding grammar is: block:N identifies the zero-based index N in the complete "
    "returned blocks array and N must be a SOURCE_EXCERPT; words:N-M uses inclusive "
    "planning-word indexes and every index must belong to a selected SOURCE_EXCERPT; "
    "source:A-B uses absolute seconds and the whole range must be covered by selected "
    "evidence; a plain quote must equal an entire selected SOURCE_EXCERPT text after "
    "strip and casefold. A bare decimal such as 793.32 is a quote declaration, not a "
    "timestamp. A transcript substring is not a whole-excerpt quote. Do not use object "
    "references, final_clip, or unselected context. context_text and suggested_hook_copy "
    "are nonselectable guidance, never evidence. "
    "Choose substantive_value_kind only from strategy_constraints.allowed_value_kinds. "
    "Use an additive value kind: state what the viewer learns that the selected excerpt "
    "alone did not provide in semantic_intent, and state why the excerpt alone does not "
    "provide it in why_unavailable. No generic filler, no paraphrase-only restatement, no "
    "fabricated facts, numbers, names, or quotes, no rewritten source speech, no source "
    "dialect shifting, no publication-ready full scripts, no voice/provider/model "
    "selection, no frame-level rendering instructions, no final-plan approval, and no "
    "platform-detection evasion. "
    "If a plan depends on an external fact, emit a FACT_VERIFICATION_PLACEHOLDER with "
    "claim_dependency and must_verify_before_execution=true; do not supply the supposed "
    "fact and do not research it. List the integer block indexes of the dependent "
    "substantive blocks in dependent_block_ids and that claim id in those blocks' "
    "dependency_ids (or in narration verification_dependency_ids when narration delivers "
    "the dependent content). Linkage is bidirectional. "
    "Narration is optional: use need=NONE unless narration is genuinely necessary. If "
    "narration.need is NONE, no block may set delivery_intent=NARRATION. Never make a "
    "plan invalid only because optional narration was removed. Do not select a TTS "
    "provider, model, or voice."
)

_BOUNDED_NOTES = 600
_BOUNDED_FIELD = 800
_BOUNDED_LIST = 12


def planning_prompt_hash() -> str:
    return hashlib.sha256(
        (PLANNING_SYSTEM_INSTRUCTION + json.dumps(PLANNING_OUTPUT_SCHEMA, sort_keys=True)).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class PlanningRequest:
    """One bounded strategy request with indexed word evidence (never timestamps)."""

    strategy_id: str
    strategy_key: str
    strategy_type: str
    intensity: str
    direction_summary: str
    added_value_focus: str
    substantive_value_kind: str
    preservation_requirements: tuple[str, ...]
    external_verification_requirement: str
    verification_requirements: tuple[str, ...]
    content_type: str
    source_moment_structure: str
    dialect_profile: str | None
    code_switch_tokens: tuple[str, ...]
    target_market: str
    output_language_policy: str
    register_intent: str
    narration_allowed: bool
    max_blocks: int
    strict_hero_cap_seconds: float
    refined_transcript: str
    refined_start: float
    refined_end: float
    context_text: str
    idea_summary: str
    topic_summary: str
    hooks: tuple[str, ...]
    words: tuple[dict[str, object], ...]
    priority: str = PLANNING_PRIORITY
    mandatory_evidence: Mapping[str, object] = field(default_factory=dict)
    source_selection: Mapping[str, object] = field(default_factory=dict)
    strategy_constraints: Mapping[str, object] = field(default_factory=dict)
    nonselectable_context: Mapping[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_key": self.strategy_key,
            "strategy_type": self.strategy_type,
            "intensity": self.intensity,
            "direction_summary": self.direction_summary,
            "added_value_focus": self.added_value_focus,
            "substantive_value_kind": self.substantive_value_kind,
            "preservation_requirements": list(self.preservation_requirements),
            "external_verification_requirement": self.external_verification_requirement,
            "verification_requirements": list(self.verification_requirements),
            "content_type": self.content_type,
            "source_moment_structure": self.source_moment_structure,
            "dialect_profile": self.dialect_profile,
            "code_switch_tokens": list(self.code_switch_tokens),
            "target_market": self.target_market,
            "output_language_policy": self.output_language_policy,
            "register_intent": self.register_intent,
            "narration_allowed": self.narration_allowed,
            "max_blocks": self.max_blocks,
            "strict_hero_cap_seconds": self.strict_hero_cap_seconds,
            "refined_transcript": self.refined_transcript,
            "refined_start": self.refined_start,
            "refined_end": self.refined_end,
            "context_text": self.context_text,
            "idea_summary": self.idea_summary,
            "topic_summary": self.topic_summary,
            "words": list(self.words),
            "priority": self.priority,
            "mandatory_evidence": dict(self.mandatory_evidence),
            "source_selection": dict(self.source_selection),
            "strategy_constraints": dict(self.strategy_constraints),
            "nonselectable_context": dict(self.nonselectable_context),
        }


class PlanningProviderError(Exception):
    """Sanitized provider failure that never carries credentials or transcript data."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"transformation_planning_{category}")
        self.category = category
        self.detail = detail


class PlanningProvider(Protocol):
    def plan(
        self, requests: Sequence[PlanningRequest], tier: str = "ROUTINE"
    ) -> dict[str, PlanProviderResult]: ...

    def release(self) -> None: ...

    def runtime_identity(self) -> dict[str, object]: ...

    def refresh_runtime_identity(self) -> dict[str, object]: ...


class DeterministicPlanningProvider:
    """Default provider: zero network calls, zero model loads, no results."""

    provider_name = "deterministic"
    model = None
    hosted_provider = False

    def plan(
        self, requests: Sequence[PlanningRequest], tier: str = "ROUTINE"
    ) -> dict[str, PlanProviderResult]:
        return {}

    def release(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "deterministic",
            "model": None,
            "prompt_hash": planning_prompt_hash(),
            "schema_version": SCHEMA_VERSION,
            "temperature": 0.0,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


def build_planning_request(
    strategy: Mapping[str, object],
    inputs: PlanningInputs,
    config: object,
) -> PlanningRequest:
    max_blocks = int(getattr(config, "max_blocks_per_plan", 8))
    strict_cap = float(getattr(config, "strict_authored_before_hero_seconds", 1.5))
    words = tuple(word.as_dict() for word in inputs.words)
    strict_hero = is_strict_hero_window(
        inputs.source_moment_structure,
        inputs.duration,
        float(inputs.stage3_risk.get("moment_density_score", 0.0) or 0.0),
    )
    return PlanningRequest(
        strategy_id=str(strategy.get("id", "")),
        strategy_key=str(strategy.get("strategy_key", "")),
        strategy_type=str(strategy.get("strategy_type", "")),
        intensity=str(strategy.get("intensity", "")),
        direction_summary=str(strategy.get("direction_summary", "")),
        added_value_focus=str(strategy.get("added_value_focus", "")),
        substantive_value_kind=str(strategy.get("substantive_value_kind", "")),
        preservation_requirements=tuple(
            str(item) for item in (strategy.get("preservation_requirements") or [])
        ),
        external_verification_requirement=str(
            strategy.get("external_verification_requirement", "NOT_REQUIRED")
        ),
        verification_requirements=tuple(
            str(item) for item in (strategy.get("verification_requirements") or [])
        ),
        content_type=inputs.content_type.value,
        source_moment_structure=inputs.source_moment_structure.value,
        dialect_profile=inputs.dialect_profile,
        code_switch_tokens=_code_switch_tokens(inputs.code_switch),
        target_market=inputs.planning_context.target_market,
        output_language_policy=inputs.planning_context.output_language_policy,
        register_intent=inputs.planning_context.register_intent,
        narration_allowed=inputs.planning_context.narration_allowed,
        max_blocks=max_blocks,
        strict_hero_cap_seconds=strict_cap,
        refined_transcript=inputs.transcript[
            : int(getattr(config, "provider_max_input_characters", 12000))
        ],
        refined_start=inputs.refined_start,
        refined_end=inputs.refined_end,
        context_text=" ".join(inputs.context_segments),
        idea_summary=inputs.idea_summary,
        topic_summary=inputs.topic_summary,
        hooks=tuple(
            str(hook.get("text", "")) for hook in inputs.hooks if isinstance(hook, Mapping)
        ),
        words=words,
        mandatory_evidence=_mandatory_request_evidence(inputs, config, words),
        source_selection=_source_selection_payload(inputs, config, words, strict_hero, strict_cap),
        strategy_constraints=_strategy_constraints_payload(strategy),
        nonselectable_context={
            "context_text": " ".join(inputs.context_segments),
            "suggested_hook_copy": [
                str(hook.get("text", ""))
                for hook in inputs.hooks
                if isinstance(hook, Mapping) and str(hook.get("text", "")).strip()
            ],
            "idea_summary": inputs.idea_summary,
            "topic_summary": inputs.topic_summary,
            "note": (
                "Nonselectable guidance only. Never cite context_text or suggested hook "
                "copy as source evidence."
            ),
        },
    )


def _word_bounds(word: Mapping[str, object]) -> tuple[float, float] | None:
    start = word.get("start")
    end = word.get("end")
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return None
    if isinstance(end, bool) or not isinstance(end, (int, float)):
        return None
    if float(end) < float(start):
        return None
    return float(start), float(end)


def _planning_word_identity(
    words: Sequence[dict[str, object]], start: float, end: float
) -> dict[str, object]:
    """Descriptive mapping of a mandatory segment onto offered planning words.

    This is word overlap only. It is deliberately not the feasibility decision;
    mandatory coverage may still be satisfiable by a supported word selection
    when an endpoint falls in an inter-word gap, or by the supported full-window
    fallback when word evidence is insufficient.
    """

    overlapping = [
        word
        for word in words
        if (bounds := _word_bounds(word)) is not None and bounds[1] > start and bounds[0] < end
    ]
    start_in_word = any(
        bounds[0] <= start <= bounds[1]
        for word in words
        if (bounds := _word_bounds(word)) is not None
    )
    end_in_word = any(
        bounds[0] <= end <= bounds[1]
        for word in words
        if (bounds := _word_bounds(word)) is not None
    )
    if not overlapping:
        return {
            "planning_word_start_index": None,
            "planning_word_end_index": None,
            "planning_word_count": 0,
            "overlaps_offered_words": False,
            "start_endpoint_in_word": bool(start_in_word),
            "end_endpoint_in_word": bool(end_in_word),
            "covered_by_offered_words": False,
        }
    return {
        "planning_word_start_index": int(overlapping[0]["index"]),
        "planning_word_end_index": int(overlapping[-1]["index"]),
        "planning_word_count": len(overlapping),
        "overlaps_offered_words": True,
        "start_endpoint_in_word": bool(start_in_word),
        "end_endpoint_in_word": bool(end_in_word),
        "covered_by_offered_words": bool(start_in_word and end_in_word),
    }


def _supported_selection(
    words: Sequence[dict[str, object]],
    start: float,
    end: float,
    inputs: PlanningInputs,
    config: object,
) -> dict[str, object] | None:
    """Whether an allowed source selection can preserve both mandatory endpoints.

    Uses the same resolution and endpoint-coverage semantics as validation: a
    contiguous planning-word range resolves to ``[first.start, last.end]`` and
    both endpoints must fall within the authoritative coverage tolerance. Only
    when word evidence is insufficient may the documented full-window fallback
    be offered.
    """

    tolerance = MANDATORY_COVERAGE_TOLERANCE_SECONDS
    if inputs.word_coverage_sufficient:
        indexes: list[int] = []
        for index, word in enumerate(words):
            bounds = _word_bounds(word)
            if bounds is None:
                continue
            if bounds[1] >= start - tolerance and bounds[0] <= end + tolerance:
                indexes.append(index)
        if indexes:
            low, high = indexes[0], indexes[-1]
            minimum = max(0.0, float(getattr(config, "min_source_excerpt_seconds", 0.4)))
            # Grow the range only when a minimum-duration excerpt is required.
            while high - low + 1 < len(words):
                low_bounds = _word_bounds(words[low])
                high_bounds = _word_bounds(words[high])
                if low_bounds is None or high_bounds is None:
                    break
                if (high_bounds[1] - low_bounds[0]) + 1e-9 >= minimum:
                    break
                if low > 0:
                    low -= 1
                elif high + 1 < len(words):
                    high += 1
                else:
                    break
            low_bounds = _word_bounds(words[low])
            high_bounds = _word_bounds(words[high])
            if low_bounds is not None and high_bounds is not None:
                span = (low_bounds[0], high_bounds[1])
                if (
                    span[1] > span[0]
                    and span[0] >= inputs.refined_start - tolerance
                    and span[1] <= inputs.refined_end + tolerance
                    and spans_cover_endpoints([span], start, end, tolerance=tolerance)
                ):
                    return {
                        "method": "WORD_SELECTION",
                        "planning_word_start_index": low,
                        "planning_word_end_index": high,
                        "planning_word_count": high - low + 1,
                        "source_start": round(span[0], 6),
                        "source_end": round(span[1], 6),
                    }
    if not inputs.word_coverage_sufficient:
        full = (inputs.refined_start, inputs.refined_end)
        if full[1] > full[0] and spans_cover_endpoints([full], start, end, tolerance=tolerance):
            return {
                "method": "FULL_WINDOW",
                "planning_word_start_index": None,
                "planning_word_end_index": None,
                "planning_word_count": 0,
                "source_start": round(full[0], 6),
                "source_end": round(full[1], 6),
            }
    return None


def _mandatory_side(
    data: Mapping[str, object],
    key: str,
    words: Sequence[dict[str, object]],
    inputs: PlanningInputs,
    config: object,
) -> dict[str, object]:
    declared = bool(data.get(f"{key}_declared"))
    segment = data.get(f"{key}_segment")
    error = data.get(f"{key}_error")
    if not declared:
        state = "ABSENT"
    elif error == "MALFORMED":
        state = "MALFORMED"
    elif error == "UNRESOLVED" or not isinstance(segment, Mapping):
        state = "UNRESOLVED"
    else:
        state = "RESOLVED"
    payload: dict[str, object] = {
        "state": state,
        "declared": declared,
        "required": state == "RESOLVED" or state in {"MALFORMED", "UNRESOLVED"},
        "segment_index": None,
        "source_start": None,
        "source_end": None,
        "planning_word_start_index": None,
        "planning_word_end_index": None,
        "planning_word_count": 0,
        "overlaps_offered_words": False,
        "start_endpoint_in_word": False,
        "end_endpoint_in_word": False,
        "covered_by_offered_words": False,
        "mandatory_coverage_feasible": False,
        "supported_selection": {"method": "NONE"},
    }
    if isinstance(segment, Mapping):
        start = segment.get("start")
        end = segment.get("end")
        index = segment.get("index")
        if (
            isinstance(start, (int, float))
            and not isinstance(start, bool)
            and isinstance(end, (int, float))
            and not isinstance(end, bool)
            and isinstance(index, int)
            and not isinstance(index, bool)
        ):
            payload.update(
                {
                    "segment_index": int(index),
                    "source_start": float(start),
                    "source_end": float(end),
                    **_planning_word_identity(words, float(start), float(end)),
                }
            )
            selection = _supported_selection(words, float(start), float(end), inputs, config)
            payload["mandatory_coverage_feasible"] = selection is not None
            payload["supported_selection"] = selection or {"method": "NONE"}
    return payload


def _mandatory_request_evidence(
    inputs: PlanningInputs, config: object, words: Sequence[dict[str, object]]
) -> dict[str, object]:
    source = dict(inputs.mandatory_evidence)
    hook = _mandatory_side(source, "hook", words, inputs, config)
    payoff = _mandatory_side(source, "payoff", words, inputs, config)
    return {
        "contract_version": source.get("contract_version"),
        "source_id": source.get("source_id") or inputs.source_id,
        "hook": hook,
        "payoff": payoff,
        "coverage_tolerance_seconds": MANDATORY_COVERAGE_TOLERANCE_SECONDS,
        "coverage_requirement": (
            "A resolved hook/payoff segment is satisfied only when the union of selected "
            "SOURCE_EXCERPT spans covers both its start and end within the stated tolerance. "
            "Role labels do not satisfy it, and a null payoff never waives a hook. "
            "mandatory_coverage_feasible reports whether a supported word selection or the "
            "supported full-window fallback can preserve both endpoints; "
            "covered_by_offered_words is descriptive mapping only."
        ),
    }


def _source_selection_payload(
    inputs: PlanningInputs,
    config: object,
    words: Sequence[dict[str, object]],
    strict_hero: bool,
    strict_cap: float,
) -> dict[str, object]:
    authored_cap = float(
        getattr(
            config,
            "strict_authored_before_hero_seconds"
            if strict_hero
            else "max_authored_before_hero_seconds",
            1.5,
        )
    )
    elapsed_cap = float(
        getattr(
            config,
            "strict_elapsed_before_hero_seconds"
            if strict_hero
            else "max_elapsed_before_hero_seconds",
            1.5,
        )
    )
    return {
        "word_index_scope": (
            "Inclusive, zero-based planning-word indexes after filtering and capping. "
            "They are not source segment positions and not FINAL_CLIP word indexes."
        ),
        "word_range_rule": (
            "words:N-M uses inclusive planning-word indexes; every index must belong to a "
            "selected SOURCE_EXCERPT. Bare decimals are quote declarations, not timestamps."
        ),
        "words_offered": len(words),
        "first_planning_word": dict(words[0]) if words else None,
        "last_planning_word": dict(words[-1]) if words else None,
        "word_coverage_sufficient": inputs.word_coverage_sufficient,
        "refined_start": inputs.refined_start,
        "refined_end": inputs.refined_end,
        "selectable_window_start": round(inputs.refined_start - 0.75, 6),
        "selectable_window_end": round(inputs.refined_end + 0.75, 6),
        "min_source_excerpt_seconds": float(getattr(config, "min_source_excerpt_seconds", 0.4)),
        "max_blocks": int(getattr(config, "max_blocks_per_plan", 8)),
        "max_plan_duration_seconds": float(getattr(config, "max_plan_duration_seconds", 180.0)),
        "hero_rule": "Exactly one HERO at block 0 or block 1.",
        "strict_hero_cap_seconds": strict_cap,
        "hero_cap_seconds": strict_cap
        if strict_hero
        else float(getattr(config, "max_authored_before_hero_seconds", 3.0)),
        "elapsed_before_hero_cap_seconds": elapsed_cap,
        "authored_before_hero_cap_seconds": authored_cap,
    }


def _strategy_constraints_payload(strategy: Mapping[str, object]) -> dict[str, object]:
    from app.core.enums import TransformationStrategyType

    try:
        strategy_type = TransformationStrategyType(str(strategy.get("strategy_type")))
        allowed = tuple(item.value for item in strategy_value_kinds(strategy_type))
    except ValueError:
        allowed = ()
    return {
        "strategy_type": strategy.get("strategy_type"),
        "offered_value_kind": strategy.get("substantive_value_kind"),
        "allowed_value_kinds": list(allowed),
        "external_verification_requirement": strategy.get(
            "external_verification_requirement", "NOT_REQUIRED"
        ),
        "source_moment_role": strategy.get("source_moment_role"),
    }


def _code_switch_tokens(code_switch: Mapping[str, object]) -> tuple[str, ...]:
    raw = code_switch.get("tokens")
    if not isinstance(raw, list):
        return ()
    return tuple(str(token) for token in raw if isinstance(token, str))


def _enum(value: object, enum_type: type) -> object | None:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return cast(object, enum_type(value))
        except ValueError:
            return None
    return None


def _bounded_text(value: object, limit: int) -> str:
    return str(value)[:limit] if isinstance(value, str) else ""


def _bounded_list(value: object, limit: int = _BOUNDED_LIST) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item[:_BOUNDED_FIELD])
        if len(items) >= limit:
            break
    return tuple(items)


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return max(0.0, number)


def _parse_grounding_refs(
    value: object,
) -> tuple[tuple[str, ...], bool, tuple[str, ...]]:
    """Preserve every string declaration and flag malformed containers/items."""

    if value is None:
        return (), False, ()
    if not isinstance(value, (list, tuple)):
        return (
            (),
            True,
            (f"invalid_reference_container:{type(value).__name__}",),
        )
    refs: list[str] = []
    errors: list[str] = []
    for item in list(value)[:64]:
        if isinstance(item, str):
            refs.append(item)
            continue
        errors.append(f"unsupported_reference_type:{type(item).__name__}:{repr(item)[:120]}")
    if len(value) > 64:
        errors.append("too_many_reference_declarations")
    return tuple(refs), bool(errors), tuple(errors)


def _bounded_int_list(
    value: object, *, present: bool = False, limit: int = _BOUNDED_LIST
) -> tuple[tuple[int, ...], bool]:
    """Parse dependent block references into validated integer block indexes.

    A supplied non-list value (string, boolean, object, null, float, nested) is
    flagged invalid rather than treated as empty, and every item is inspected
    even past the bounded valid-item cap. Malformed/non-integer references are
    never silently dropped: the second return value flags them so the
    deterministic boundary rejects the plan.
    """

    if not present:
        return (), False
    if not isinstance(value, list):
        return (), True
    items: list[int] = []
    invalid = False
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            invalid = True
            continue
        candidate = int(item)
        if candidate in items:
            continue
        if len(items) < limit:
            items.append(candidate)
    return tuple(items), invalid


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _parse_block(entry: Mapping[str, object]) -> PlanProviderBlock | None:
    block_type = _enum(entry.get("block_type"), PlanBlockType)
    if block_type is None:
        return None
    dependent_block_ids, dependent_block_ids_invalid = _bounded_int_list(
        entry.get("dependent_block_ids"),
        present="dependent_block_ids" in entry,
    )
    grounding_refs, grounding_refs_invalid, grounding_refs_errors = _parse_grounding_refs(
        entry.get("grounding_refs")
    )
    return PlanProviderBlock(
        block_type=block_type,  # type: ignore[arg-type]
        purpose=_bounded_text(entry.get("purpose"), _BOUNDED_FIELD),
        estimated_duration=_float(entry.get("estimated_duration")),
        interrupts_source=bool(entry.get("interrupts_source", False)),
        preservation_constraints=_bounded_list(entry.get("preservation_constraints")),
        dependency_ids=_bounded_list(entry.get("dependency_ids")),
        source_role=_enum(entry.get("source_role"), SourceExcerptRole),  # type: ignore[arg-type]
        word_start_index=_int_or_none(entry.get("word_start_index")),
        word_end_index=_int_or_none(entry.get("word_end_index")),
        use_full_window=bool(entry.get("use_full_window", False)),
        continuity_rationale=_bounded_text(entry.get("continuity_rationale"), _BOUNDED_FIELD),
        substantive_value_kind=_enum(entry.get("substantive_value_kind"), SubstantiveValueKind),  # type: ignore[arg-type]
        semantic_intent=_bounded_text(entry.get("semantic_intent"), _BOUNDED_FIELD),
        why_unavailable=_bounded_text(entry.get("why_unavailable"), _BOUNDED_FIELD),
        grounding_refs=grounding_refs,
        grounding_refs_present="grounding_refs" in entry,
        grounding_refs_invalid=grounding_refs_invalid,
        grounding_refs_errors=grounding_refs_errors,
        delivery_intent=_enum(entry.get("delivery_intent"), DeliveryIntent),  # type: ignore[arg-type]
        draft_line=_bounded_text(entry.get("draft_line"), _BOUNDED_FIELD) or None,
        claim_dependency=_bounded_text(entry.get("claim_dependency"), _BOUNDED_FIELD) or None,
        verification_rationale=_bounded_text(entry.get("verification_rationale"), _BOUNDED_FIELD)
        or None,
        intended_use=_bounded_text(entry.get("intended_use"), _BOUNDED_FIELD) or None,
        must_verify_before_execution=bool(entry.get("must_verify_before_execution", False)),
        dependent_block_ids=dependent_block_ids,
        dependent_block_ids_invalid=(
            dependent_block_ids_invalid or bool(entry.get("dependent_block_ids_invalid", False))
        ),
    )


def _parse_narration(value: object) -> NarrationRequirement:
    if not isinstance(value, Mapping):
        return NarrationRequirement(need=NarrationNeed.NONE)
    need = _enum(value.get("need"), NarrationNeed)
    if need is None:
        need = NarrationNeed.NONE
    purposes: list[NarrationPurpose] = []
    raw_purposes = value.get("purposes")
    if isinstance(raw_purposes, list):
        for item in raw_purposes:
            parsed = _enum(item, NarrationPurpose)
            if parsed is not None and parsed not in purposes:
                purposes.append(parsed)  # type: ignore[arg-type]
    return NarrationRequirement(
        need=need,  # type: ignore[arg-type]
        purposes=tuple(purposes),
        language=_bounded_text(value.get("language"), 64) or None,
        register=_bounded_text(value.get("register", value.get("register_intent")), 64) or None,
        estimated_duration=_float(value.get("estimated_duration")),
        placement_block_index=(
            value.get("placement_block_index")
            if isinstance(value.get("placement_block_index"), int)
            and not isinstance(value.get("placement_block_index"), bool)
            else None
        ),
        max_source_interruption_seconds=_float(value.get("max_source_interruption_seconds")),
        overlaps_source_audio=bool(value.get("overlaps_source_audio", False)),
        replaces_silence=bool(value.get("replaces_silence", False)),
        essential=bool(value.get("essential", False)),
        verification_dependency_ids=_bounded_list(value.get("verification_dependency_ids")),
    )


def _bounded_identity(value: object) -> tuple[str | None, int | None, str]:
    """Safe structural identity evidence: digest, length, and type name."""

    if not isinstance(value, str):
        return None, None, type(value).__name__
    return hashlib.sha256(value.encode("utf-8")).hexdigest(), len(value), "str"


def _rejection(
    code: str,
    *,
    entry_index: int | None = None,
    block_index: int | None = None,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {"code": code}
    if entry_index is not None:
        payload["entry_index"] = entry_index
    if block_index is not None:
        payload["block_index"] = block_index
    payload.update(extra)
    return payload


def _response_evidence_with_parser_context(
    content: Mapping[str, object],
    supplied: Mapping[str, object] | None,
    unmatched: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    evidence = dict(supplied or {})
    if "content_digest" not in evidence or "content_bytes" not in evidence:
        encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
        evidence.setdefault("content_digest", hashlib.sha256(encoded).hexdigest())
        evidence.setdefault("content_bytes", len(encoded))
    entries = content.get("plans")
    evidence.setdefault("returned_plan_count", len(entries) if isinstance(entries, list) else 0)
    evidence.setdefault("truncated", False)
    evidence["unmatched_entry_count"] = len(unmatched)
    evidence["unmatched_entries"] = [dict(item) for item in list(unmatched)[:8]]
    return evidence


def parse_plan_results(
    content: Mapping[str, object],
    requests: Sequence[PlanningRequest],
    *,
    response_evidence: Mapping[str, object] | None = None,
) -> dict[str, PlanProviderResult]:
    """Strictly parse provider plans and retain bounded rejection evidence.

    Unknown, duplicate, stale, or mismatched strategy identities are dropped
    from executable results. Their bounded parser evidence is retained even when
    no plan was collected, so a caller can distinguish a true omission from a
    mismatched, unknown, duplicate, or malformed entry. One malformed item never
    invalidates an accepted sibling.
    """

    entries = content.get("plans")
    if not isinstance(entries, list):
        raise PlanningProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
    requested = {request.strategy_key: request for request in requests}
    collected: dict[str, list[PlanProviderPlan]] = {}
    rejections: dict[str, list[dict[str, object]]] = {key: [] for key in requested}
    unmatched: list[dict[str, object]] = []
    seen: set[str] = set()
    recognized_requested_entry = False
    for entry_index, entry in enumerate(entries[:256]):
        if not isinstance(entry, Mapping):
            unmatched.append(
                _rejection(
                    "MALFORMED_ENTRY",
                    entry_index=entry_index,
                    actual_type=type(entry).__name__,
                )
            )
            continue
        key = entry.get("strategy_key")
        if not isinstance(key, str) or key not in requested:
            key_digest, key_characters, _key_type = _bounded_identity(key)
            id_digest, id_characters, _id_type = _bounded_identity(entry.get("strategy_id"))
            unmatched.append(
                _rejection(
                    "UNKNOWN_STRATEGY_KEY" if isinstance(key, str) else "MALFORMED_STRATEGY_KEY",
                    entry_index=entry_index,
                    strategy_key_digest=key_digest,
                    strategy_key_characters=key_characters,
                    strategy_id_digest=id_digest,
                    strategy_id_characters=id_characters,
                )
            )
            continue
        recognized_requested_entry = True
        request = requested[key]
        if entry.get("strategy_id") != request.strategy_id:
            id_digest, id_characters, _id_type = _bounded_identity(entry.get("strategy_id"))
            rejections[key].append(
                _rejection(
                    "STRATEGY_ID_MISMATCH",
                    entry_index=entry_index,
                    requested_strategy_id=request.strategy_id,
                    requested_strategy_key=request.strategy_key,
                    returned_strategy_id_digest=id_digest,
                    returned_strategy_id_characters=id_characters,
                )
            )
            continue
        if key in seen:
            rejections[key].append(
                _rejection(
                    "DUPLICATE_STRATEGY_KEY",
                    entry_index=entry_index,
                    requested_strategy_id=request.strategy_id,
                    requested_strategy_key=request.strategy_key,
                )
            )
            continue
        seen.add(key)
        blocks: list[PlanProviderBlock] = []
        raw_blocks = entry.get("blocks")
        if isinstance(raw_blocks, list):
            for block_index, item in enumerate(raw_blocks[:256]):
                if not isinstance(item, Mapping):
                    rejections[key].append(
                        _rejection(
                            "MALFORMED_BLOCK_ENTRY",
                            entry_index=entry_index,
                            block_index=block_index,
                            actual_type=type(item).__name__,
                        )
                    )
                    continue
                if _enum(item.get("block_type"), PlanBlockType) is None:
                    type_digest, type_characters, _type_name = _bounded_identity(
                        item.get("block_type")
                    )
                    rejections[key].append(
                        _rejection(
                            "UNKNOWN_BLOCK_TYPE",
                            entry_index=entry_index,
                            block_index=block_index,
                            block_type_digest=type_digest,
                            block_type_characters=type_characters,
                        )
                    )
                    continue
                parsed = _parse_block(item)
                if parsed is None:
                    rejections[key].append(
                        _rejection(
                            "MALFORMED_BLOCK_ENTRY",
                            entry_index=entry_index,
                            block_index=block_index,
                        )
                    )
                    continue
                blocks.append(parsed)
        else:
            rejections[key].append(
                _rejection(
                    "MALFORMED_BLOCKS_CONTAINER",
                    entry_index=entry_index,
                    requested_strategy_id=request.strategy_id,
                    requested_strategy_key=request.strategy_key,
                )
            )
        plan = PlanProviderPlan(
            strategy_id=request.strategy_id,
            strategy_key=key,
            confidence=clamp(_float(entry.get("confidence"))),
            no_valid_plan=bool(entry.get("no_valid_plan", False)),
            no_valid_reason=_bounded_text(entry.get("no_valid_reason"), 240),
            blocks=tuple(blocks),
            narration=_parse_narration(entry.get("narration")),
            preservation_constraints=_bounded_list(entry.get("preservation_constraints")),
            planner_notes=_bounded_text(entry.get("planner_notes"), _BOUNDED_NOTES),
        )
        collected.setdefault(key, []).append(plan)
    if not recognized_requested_entry and unmatched and requested:
        # The response contained no requested identity at all. Keep this
        # response-level rejection visible for every requested strategy without
        # inventing a placeholder plan or attaching it as a key-specific match.
        unmatched_codes = {str(item.get("code")) for item in unmatched}
        shared_code = (
            "MALFORMED_ENTRY" if unmatched_codes <= {"MALFORMED_ENTRY"} else "UNKNOWN_STRATEGY_KEY"
        )
        shared = _rejection(
            shared_code,
            scope="RESPONSE",
            unmatched_entry_count=len(unmatched),
        )
        for key in requested:
            rejections[key].append(dict(shared))
    evidence = _response_evidence_with_parser_context(content, response_evidence, unmatched)
    parsed_results: dict[str, PlanProviderResult] = {}
    for key in requested:
        plans = tuple(collected.get(key, ()))
        candidate_rejections = tuple(rejections.get(key, ()))
        if not plans and not candidate_rejections:
            continue
        parsed_results[key] = PlanProviderResult(
            plans=plans,
            confidence=plans[0].confidence if plans else 0.0,
            response_evidence=dict(evidence),
            rejections=candidate_rejections,
        )
    return parsed_results


def serialize_provider_result(result: PlanProviderResult) -> dict[str, object]:
    """Serialize a parsed provider result for durable checkpoint reuse."""

    return {
        "plans": [
            {
                "strategy_id": plan.strategy_id,
                "strategy_key": plan.strategy_key,
                "confidence": plan.confidence,
                "no_valid_plan": plan.no_valid_plan,
                "no_valid_reason": plan.no_valid_reason,
                "planner_notes": plan.planner_notes,
                "preservation_constraints": list(plan.preservation_constraints),
                "blocks": [_block_payload(block) for block in plan.blocks],
                "narration": plan.narration.as_dict(),
            }
            for plan in result.plans
        ]
    }


def _block_payload(block: PlanProviderBlock) -> dict[str, object]:
    return {
        "block_type": block.block_type.value,
        "purpose": block.purpose,
        "estimated_duration": block.estimated_duration,
        "interrupts_source": block.interrupts_source,
        "preservation_constraints": list(block.preservation_constraints),
        "dependency_ids": list(block.dependency_ids),
        "source_role": block.source_role.value if block.source_role else None,
        "word_start_index": block.word_start_index,
        "word_end_index": block.word_end_index,
        "use_full_window": block.use_full_window,
        "continuity_rationale": block.continuity_rationale,
        "substantive_value_kind": (
            block.substantive_value_kind.value if block.substantive_value_kind else None
        ),
        "semantic_intent": block.semantic_intent,
        "why_unavailable": block.why_unavailable,
        "grounding_refs": list(block.grounding_refs),
        "grounding_refs_present": block.grounding_refs_present,
        "delivery_intent": block.delivery_intent.value if block.delivery_intent else None,
        "draft_line": block.draft_line,
        "claim_dependency": block.claim_dependency,
        "verification_rationale": block.verification_rationale,
        "intended_use": block.intended_use,
        "must_verify_before_execution": block.must_verify_before_execution,
        "dependent_block_ids": list(block.dependent_block_ids),
        "dependent_block_ids_invalid": block.dependent_block_ids_invalid,
    }


def deserialize_provider_result(
    data: object, request: PlanningRequest
) -> PlanProviderResult | None:
    """Rebuild a checkpointed provider plan from persisted evidence."""

    if not isinstance(data, Mapping):
        return None
    try:
        return parse_plan_results(data, [request]).get(request.strategy_key)
    except PlanningProviderError:
        return None
