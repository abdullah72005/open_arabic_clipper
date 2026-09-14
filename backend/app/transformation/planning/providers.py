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
from dataclasses import dataclass
from typing import Protocol, cast

from app.candidates.providers import ProviderErrorCategory  # noqa: F401  (re-exported)
from app.core.enums import (
    DeliveryIntent,
    NarrationNeed,
    NarrationPurpose,
    PlanBlockType,
    SourceExcerptRole,
    SubstantiveValueKind,
)
from app.transformation.planning.policy import SCHEMA_VERSION
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


PLANNING_SYSTEM_INSTRUCTION = (
    "You produce concrete structured transformation plans for one short Arabic "
    "source moment, one plan per requested Stage 4.0 strategy. "
    "Return only the strict JSON schema. Each plan identifies the exact requested "
    "strategy_id and strategy_key. "
    "You never invent source timestamps or quotes: select source spans by the "
    "provided word indexes, or use the full refined window when word evidence is "
    "unavailable. "
    "Every valid plan has at least one source excerpt and exactly one hero source "
    "excerpt placed as block 0 or block 1. Do not delay the source behind a long "
    "preamble. "
    "Every original-value block must state what the viewer learns that the source "
    "excerpt alone did not provide, as a specific, source-grounded semantic intent. "
    "No long generic introductions, no paraphrase presented as originality, no "
    "fabricated facts, numbers, names, or quotes, no rewritten source speech, no "
    "source dialect shifting, no publication-ready full scripts, no voice/provider/"
    "model selection, no frame-level rendering instructions, no final-plan "
    "approval, and no platform-detection evasion. "
    "If a plan depends on an external fact, emit a FACT_VERIFICATION_PLACEHOLDER "
    "that names what must be verified and marks must_verify_before_execution=true; "
    "do not supply the supposed fact and do not research it. Dependent content must "
    "reference the requirement: list the integer block indexes of the dependent "
    "substantive blocks in dependent_block_ids, and list the placeholder's "
    "claim_dependency value in those blocks' dependency_ids (or in the narration "
    "verification_dependency_ids when narration delivers the dependent content). "
    "Narration is optional: use NONE unless narration is genuinely necessary. "
    "Never cause a plan to be required solely because narration was removed. "
    "Do not select a TTS provider, TTS model, or voice; do not give frame-level "
    "rendering instructions; do not propose mirroring, pitch shifting, speed "
    "tricks, watermark removal, or any platform-detection evasion. "
    "If a strategy cannot produce a valid substantive plan, set no_valid_plan=true "
    "with a short bounded reason instead of emitting filler."
)

PLANNING_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "plans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "strategy_id": {"type": "string"},
                    "strategy_key": {"type": "string"},
                    "confidence": {"type": "number"},
                    "no_valid_plan": {"type": "boolean"},
                    "no_valid_reason": {"type": "string"},
                    "planner_notes": {"type": "string"},
                    "preservation_constraints": {"type": "array"},
                    "blocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "block_type": {"type": "string", "enum": _BLOCK_TYPES},
                                "purpose": {"type": "string"},
                                "estimated_duration": {"type": "number"},
                                "interrupts_source": {"type": "boolean"},
                                "preservation_constraints": {"type": "array"},
                                "dependency_ids": {"type": "array"},
                                "source_role": {"type": "string", "enum": _SOURCE_ROLES},
                                "word_start_index": {"type": "integer"},
                                "word_end_index": {"type": "integer"},
                                "use_full_window": {"type": "boolean"},
                                "continuity_rationale": {"type": "string"},
                                "substantive_value_kind": {
                                    "type": "string",
                                    "enum": _VALUE_KINDS,
                                },
                                "semantic_intent": {"type": "string"},
                                "why_unavailable": {"type": "string"},
                                "grounding_refs": {"type": "array"},
                                "delivery_intent": {"type": "string", "enum": _DELIVERY},
                                "draft_line": {"type": "string"},
                                "claim_dependency": {"type": "string"},
                                "verification_rationale": {"type": "string"},
                                "intended_use": {"type": "string"},
                                "must_verify_before_execution": {"type": "boolean"},
                                "dependent_block_ids": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                            },
                            "required": ["block_type"],
                        },
                    },
                    "narration": {
                        "type": "object",
                        "properties": {
                            "need": {"type": "string", "enum": _NARRATION_NEEDS},
                            "purposes": {"type": "array"},
                            "language": {"type": "string"},
                            "register": {"type": "string"},
                            "estimated_duration": {"type": "number"},
                            "placement_block_index": {"type": "integer"},
                            "max_source_interruption_seconds": {"type": "number"},
                            "overlaps_source_audio": {"type": "boolean"},
                            "replaces_silence": {"type": "boolean"},
                            "essential": {"type": "boolean"},
                            "verification_dependency_ids": {"type": "array"},
                        },
                    },
                },
                "required": ["strategy_id", "strategy_key"],
            },
        }
    },
    "required": ["plans"],
}

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
            "hooks": list(self.hooks),
            "words": list(self.words),
            "priority": self.priority,
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
    )


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


def _bounded_int_list(value: object, limit: int = _BOUNDED_LIST) -> tuple[int, ...]:
    """Parse dependent block references into validated integer block indexes."""

    if not isinstance(value, list):
        return ()
    items: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        candidate: int | None = None
        if isinstance(item, int):
            candidate = int(item)
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            candidate = int(item.strip())
        if candidate is None or candidate in items:
            continue
        items.append(candidate)
        if len(items) >= limit:
            break
    return tuple(items)


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _parse_block(entry: Mapping[str, object]) -> PlanProviderBlock | None:
    block_type = _enum(entry.get("block_type"), PlanBlockType)
    if block_type is None:
        return None
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
        grounding_refs=_bounded_list(entry.get("grounding_refs")),
        delivery_intent=_enum(entry.get("delivery_intent"), DeliveryIntent),  # type: ignore[arg-type]
        draft_line=_bounded_text(entry.get("draft_line"), _BOUNDED_FIELD) or None,
        claim_dependency=_bounded_text(entry.get("claim_dependency"), _BOUNDED_FIELD) or None,
        verification_rationale=_bounded_text(entry.get("verification_rationale"), _BOUNDED_FIELD)
        or None,
        intended_use=_bounded_text(entry.get("intended_use"), _BOUNDED_FIELD) or None,
        must_verify_before_execution=bool(entry.get("must_verify_before_execution", False)),
        dependent_block_ids=_bounded_int_list(entry.get("dependent_block_ids")),
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


def parse_plan_results(
    content: Mapping[str, object],
    requests: Sequence[PlanningRequest],
) -> dict[str, PlanProviderResult]:
    """Strictly parse provider plans, isolating malformed items.

    Unknown, duplicate, stale, or mismatched strategy identities are dropped.
    One malformed item never invalidates accepted siblings.
    """

    entries = content.get("plans")
    if not isinstance(entries, list):
        raise PlanningProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
    requested = {request.strategy_key: request for request in requests}
    collected: dict[str, list[PlanProviderPlan]] = {}
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        key = entry.get("strategy_key")
        if not isinstance(key, str) or key not in requested:
            continue
        request = requested[key]
        if entry.get("strategy_id") != request.strategy_id:
            continue
        if key in seen:
            continue
        seen.add(key)
        blocks: list[PlanProviderBlock] = []
        raw_blocks = entry.get("blocks")
        if isinstance(raw_blocks, list):
            for item in raw_blocks:
                if not isinstance(item, Mapping):
                    continue
                parsed = _parse_block(item)
                if parsed is not None:
                    blocks.append(parsed)
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
    return {
        key: PlanProviderResult(plans=tuple(plans), confidence=plans[0].confidence)
        for key, plans in collected.items()
    }


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
        "delivery_intent": block.delivery_intent.value if block.delivery_intent else None,
        "draft_line": block.draft_line,
        "claim_dependency": block.claim_dependency,
        "verification_rationale": block.verification_rationale,
        "intended_use": block.intended_use,
        "must_verify_before_execution": block.must_verify_before_execution,
        "dependent_block_ids": list(block.dependent_block_ids),
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
