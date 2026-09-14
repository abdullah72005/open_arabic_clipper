"""Stage 4.0 provider protocol, strict schema, and deterministic provider.

Deliberately separate from the Stage 3 semantic prompt/schema and from the
Stage 2.7 reconstruction schema: transformation strategy discovery is neither
candidate scoring nor transcript reconstruction. Providers return directions
only - never scripts, timelines, TTS text, or rewritten source speech.
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
    ExternalFactRequirement,
    StrategyDisposition,
    SubstantiveValueKind,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.transformation.policy import SCHEMA_VERSION
from app.transformation.types import (
    TransformationProviderResult,
    TransformationProviderStrategy,
    clamp,
)

TRANSFORMATION_PRIORITY = "HIGH"

TRANSFORMATION_SYSTEM_INSTRUCTION = (
    "You assess whether one short Arabic source moment has a credible substantive "
    "transformation direction for short-form video. "
    "Return directions only: an angle or thesis seed, never a finished narration, "
    "hook text, script, shot list, timeline, TTS text, or rendering instruction. "
    "Never rewrite, translate, localize, or dialect-shift the source speech. "
    "Preserve any dialect and code-switched terms exactly as evidenced; do not fake "
    "Gulf or Saudi slang and never treat dialect as a target market. "
    "The strongest source moment must stay the hero; do not delay or damage its hook "
    "or payoff. "
    "Do not invent facts, numbers, names, quotes, or context. If a direction depends "
    "on an outside fact, mark REQUIRES_EXTERNAL_FACT_VERIFICATION and describe only "
    "what must be verified; do not research it. "
    "Presentation changes (captions, crop, zoom, borders, emoji, gameplay, B-roll, "
    "background loops, music, speed changes, simple cuts, filters) are never "
    "substantive value. Reject directions that are only paraphrase, generic filler, "
    "fake drama, distortion, or interchangeable templates. "
    "Do not make legal, copyright, or platform-approval guarantees. "
    "Score only the requested candidate IDs. Output only the strict JSON schema."
)


_STRATEGY_TYPES = [item.value for item in TransformationStrategyType]
_VALUE_KINDS = [item.value for item in SubstantiveValueKind]
_INTENSITIES = [item.value for item in TransformationIntensity]
_DISPOSITIONS = [item.value for item in StrategyDisposition]
_EXTERNAL = [item.value for item in ExternalFactRequirement]

TRANSFORMATION_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "confidence": {"type": "number"},
                    "notes": {"type": "string"},
                    "strategies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "strategy_type": {"type": "string", "enum": _STRATEGY_TYPES},
                                "disposition": {"type": "string", "enum": _DISPOSITIONS},
                                "intensity": {"type": "string", "enum": _INTENSITIES},
                                "direction_summary": {"type": "string"},
                                "added_value_focus": {"type": "string"},
                                "substantive_value_kind": {
                                    "type": "string",
                                    "enum": _VALUE_KINDS,
                                },
                                "preservation_requirements": {"type": "array"},
                                "retention_preservation": {"type": "number"},
                                "source_moment_damage_risk": {"type": "number"},
                                "added_value_density": {"type": "number"},
                                "originality_potential": {"type": "number"},
                                "source_dominance_risk": {"type": "number"},
                                "generic_filler_risk": {"type": "number"},
                                "redundant_commentary_risk": {"type": "number"},
                                "template_staleness_risk": {"type": "number"},
                                "external_verification_requirement": {
                                    "type": "string",
                                    "enum": _EXTERNAL,
                                },
                                "verification_requirements": {"type": "array"},
                                "rejection_reasons": {"type": "array"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["strategy_type", "disposition", "added_value_focus"],
                        },
                    },
                },
                "required": ["candidate_id"],
            },
        }
    },
    "required": ["candidates"],
}

_BOUNDED_SUMMARY = 480
_BOUNDED_FOCUS = 600
_BOUNDED_REQUIREMENTS = 8
_BOUNDED_REQUIREMENT_CHARS = 240


def transformation_prompt_hash() -> str:
    return hashlib.sha256(
        (
            TRANSFORMATION_SYSTEM_INSTRUCTION
            + json.dumps(TRANSFORMATION_OUTPUT_SCHEMA, sort_keys=True)
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TransformationStrategyRequest:
    """One bounded serious-candidate summary sent to a provider."""

    candidate_id: str
    content_type: str = "OTHER"
    source_moment_structure: str = "UNKNOWN"
    refined_transcript: str = ""
    context_text: str = ""
    hooks: tuple[str, ...] = ()
    idea_summary: str = ""
    topic_summary: str = ""
    dialect_profile: str | None = None
    code_switch_tokens: tuple[str, ...] = ()
    rights_risk: str = "UNDETERMINED"
    originality_risk: str = "UNDETERMINED"
    transformation_required: bool = False
    complex_case: bool = False
    priority: str = TRANSFORMATION_PRIORITY

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "content_type": self.content_type,
            "source_moment_structure": self.source_moment_structure,
            "refined_transcript": self.refined_transcript,
            "context_text": self.context_text,
            "hooks": list(self.hooks),
            "idea_summary": self.idea_summary,
            "topic_summary": self.topic_summary,
            "dialect_profile": self.dialect_profile,
            "code_switch_tokens": list(self.code_switch_tokens),
            "rights_risk": self.rights_risk,
            "originality_risk": self.originality_risk,
            "transformation_required": self.transformation_required,
            "priority": self.priority,
        }


class TransformationProviderError(Exception):
    """Sanitized provider failure that never carries credentials or transcript data."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"transformation_{category}")
        self.category = category
        self.detail = detail


class TransformationProvider(Protocol):
    def discover(
        self, requests: Sequence[TransformationStrategyRequest]
    ) -> dict[str, TransformationProviderResult]: ...

    def release(self) -> None: ...

    def runtime_identity(self) -> dict[str, object]: ...

    def refresh_runtime_identity(self) -> dict[str, object]: ...


class DeterministicTransformationProvider:
    """Default provider: zero network calls, zero model loads, no results."""

    provider_name = "deterministic"
    model = None

    def discover(
        self, requests: Sequence[TransformationStrategyRequest]
    ) -> dict[str, TransformationProviderResult]:
        return {}

    def release(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "deterministic",
            "model": None,
            "prompt_hash": transformation_prompt_hash(),
            "schema_version": SCHEMA_VERSION,
            "temperature": 0.0,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return clamp(number)


def _bounded_text(value: object, limit: int) -> str:
    return str(value)[:limit] if isinstance(value, str) else ""


def _bounded_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        items.append(item[:_BOUNDED_REQUIREMENT_CHARS])
        if len(items) >= _BOUNDED_REQUIREMENTS:
            break
    return tuple(items)


def _enum(value: object, enum_type: type) -> object | None:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return cast(object, enum_type(value))
        except ValueError:
            return None
    return None


def _parse_strategy(entry: Mapping[str, object]) -> TransformationProviderStrategy | None:
    strategy_type = _enum(entry.get("strategy_type"), TransformationStrategyType)
    if strategy_type is None:
        return None
    disposition = _enum(entry.get("disposition"), StrategyDisposition)
    if disposition is None:
        return None
    intensity = _enum(entry.get("intensity"), TransformationIntensity)
    if intensity is None:
        intensity = TransformationIntensity.MODERATE
    value_kind = _enum(entry.get("substantive_value_kind"), SubstantiveValueKind)
    external = _enum(entry.get("external_verification_requirement"), ExternalFactRequirement)
    if external is None:
        external = ExternalFactRequirement.NOT_REQUIRED
    confidence = _finite_or_none(entry.get("confidence")) or 0.0
    return TransformationProviderStrategy(
        strategy_type=strategy_type,  # type: ignore[arg-type]
        disposition=disposition,  # type: ignore[arg-type]
        intensity=intensity,  # type: ignore[arg-type]
        direction_summary=_bounded_text(entry.get("direction_summary"), _BOUNDED_SUMMARY),
        added_value_focus=_bounded_text(entry.get("added_value_focus"), _BOUNDED_FOCUS),
        substantive_value_kind=value_kind,  # type: ignore[arg-type]
        preservation_requirements=_bounded_list(entry.get("preservation_requirements")),
        external_verification_requirement=external,  # type: ignore[arg-type]
        verification_requirements=_bounded_list(entry.get("verification_requirements")),
        rejection_reasons=_bounded_list(entry.get("rejection_reasons")),
        confidence=confidence,
        retention_preservation=_finite_or_none(entry.get("retention_preservation")),
        source_moment_damage_risk=_finite_or_none(entry.get("source_moment_damage_risk")),
        added_value_density=_finite_or_none(entry.get("added_value_density")),
        originality_potential=_finite_or_none(entry.get("originality_potential")),
        source_dominance_risk=_finite_or_none(entry.get("source_dominance_risk")),
        generic_filler_risk=_finite_or_none(entry.get("generic_filler_risk")),
        redundant_commentary_risk=_finite_or_none(entry.get("redundant_commentary_risk")),
        template_staleness_risk=_finite_or_none(entry.get("template_staleness_risk")),
    )


def parse_strategy_results(
    content: Mapping[str, object],
    requests: Sequence[TransformationStrategyRequest],
) -> dict[str, TransformationProviderResult]:
    """Strictly parse provider results, isolating malformed entries.

    Unknown/duplicate candidate IDs are ignored. Within a candidate, unknown
    strategy types and duplicate strategy types are dropped; accepted siblings
    survive a malformed item.
    """

    entries = content.get("candidates")
    if not isinstance(entries, list):
        raise TransformationProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
    requested = {request.candidate_id for request in requests}
    results: dict[str, TransformationProviderResult] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        candidate_id = entry.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in requested:
            continue
        if candidate_id in results:
            continue
        raw_strategies = entry.get("strategies")
        strategies: list[TransformationProviderStrategy] = []
        seen: set[TransformationStrategyType] = set()
        if isinstance(raw_strategies, list):
            for item in raw_strategies:
                if not isinstance(item, Mapping):
                    continue
                parsed = _parse_strategy(item)
                if parsed is None or parsed.strategy_type in seen:
                    continue
                seen.add(parsed.strategy_type)
                strategies.append(parsed)
        results[candidate_id] = TransformationProviderResult(
            candidate_id=candidate_id,
            strategies=tuple(strategies),
            notes=_bounded_text(entry.get("notes"), _BOUNDED_SUMMARY),
            confidence=_finite_or_none(entry.get("confidence")) or 0.0,
        )
    return results


def _strategy_payload(strategy: TransformationProviderStrategy) -> dict[str, object]:
    return {
        "strategy_type": strategy.strategy_type.value,
        "disposition": strategy.disposition.value,
        "intensity": strategy.intensity.value,
        "direction_summary": strategy.direction_summary,
        "added_value_focus": strategy.added_value_focus,
        "substantive_value_kind": (
            strategy.substantive_value_kind.value if strategy.substantive_value_kind else None
        ),
        "preservation_requirements": list(strategy.preservation_requirements),
        "retention_preservation": strategy.retention_preservation,
        "source_moment_damage_risk": strategy.source_moment_damage_risk,
        "added_value_density": strategy.added_value_density,
        "originality_potential": strategy.originality_potential,
        "source_dominance_risk": strategy.source_dominance_risk,
        "generic_filler_risk": strategy.generic_filler_risk,
        "redundant_commentary_risk": strategy.redundant_commentary_risk,
        "template_staleness_risk": strategy.template_staleness_risk,
        "external_verification_requirement": strategy.external_verification_requirement.value,
        "verification_requirements": list(strategy.verification_requirements),
        "rejection_reasons": list(strategy.rejection_reasons),
        "confidence": strategy.confidence,
    }


def serialize_provider_result(result: TransformationProviderResult) -> dict[str, object]:
    """Serialize a parsed provider result for durable checkpoint reuse."""

    return {
        "candidates": [
            {
                "candidate_id": result.candidate_id,
                "confidence": result.confidence,
                "notes": result.notes,
                "strategies": [_strategy_payload(item) for item in result.strategies],
            }
        ]
    }


def deserialize_provider_result(
    data: object, candidate_id: str
) -> TransformationProviderResult | None:
    """Rebuild a checkpointed provider result from persisted evidence."""

    if not isinstance(data, Mapping):
        return None
    request = TransformationStrategyRequest(
        candidate_id=candidate_id, content_type="OTHER", source_moment_structure="UNKNOWN"
    )
    try:
        return parse_strategy_results(data, [request]).get(candidate_id)
    except TransformationProviderError:
        return None
