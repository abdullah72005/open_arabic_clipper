"""Strict, bounded Stage 4.2 provider boundary.

A provider may only return bounded observable semantic findings for a known
current plan. It can never assign a final governor status or a platform-risk
classification, never claim a platform outcome, and never supply TTS identity or
evasion instructions. One malformed plan critique never invalidates valid
siblings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from app.candidates.providers import ProviderErrorCategory
from app.core.enums import (
    CoherenceFinding,
    GovernanceLevel,
    NarrationBurdenFinding,
    RetentionEffectFinding,
    SemanticFidelityFinding,
    SubstantiveValueFinding,
    UnsupportedClaimFinding,
)
from app.transformation.governance.policy import (
    GOVERNOR_SCHEMA_VERSION,
    PLATFORM_GUARANTEE_MARKERS,
    Stage42Config,
)
from app.transformation.governance.types import (
    GovernanceInputs,
    PlanEvidence,
    ProviderCritique,
    ProviderGovernanceResult,
)
from app.transformation.planning.policy import (
    TTS_PROVIDER_TOKENS,
    VOICE_SELECTION_CONTEXT_MARKERS,
)
from app.transformation.planning.validation import boundary_violation
from app.transformation.types import clamp

GOVERNANCE_PRIORITY = "HIGH"
_ROUTINE = "ROUTINE"
_STRONG = "STRONG"

_BOUNDED_SUMMARY = 600
_BOUNDED_CODES = 12
_BOUNDED_INDEXES = 12

_FORBIDDEN_STATUS_TOKENS = (
    "APPROVED_FOR_SELECTION",
    "APPROVED_WITH_CAUTION",
    "BLOCKED_PENDING_VERIFICATION",
    "REVISION_REQUIRED",
    "REJECTED_BY_GOVERNOR",
    "GOVERNANCE_DEFERRED",
    "ELIGIBLE_FOR_STAGE4_3",
    "ELIGIBLE_FOR_STAGE_4_3",
)

GOVERNANCE_SYSTEM_INSTRUCTION = (
    "You are a conservative editorial governor for short-form video transformation "
    "plans. You review ONE candidate's concrete Stage 4.1 plans and report only "
    "observable semantic characteristics. You never decide whether a plan is "
    "approved, eligible, safe, monetizable, recommended, or enforceable, and you "
    "never mention a platform outcome. You never select or rank a plan, never "
    "rewrite or generate plan text, never choose a text-to-speech provider, model, "
    "voice, or speaker identity, and never suggest detection evasion. Judge plan "
    "intent, not final script naturalness. Preserve the source speaker's meaning: "
    "flag distortion, decontextualization, sarcasm treated literally, speculation "
    "presented as fact, false attribution, and fake hooks. Report substantive value "
    "only for genuine context, explanation, inference, comparison, counterpoint, "
    "synthesis, verification/correction, authored thesis, useful takeaway, or "
    "source-as-evidence framing; presentation-only changes such as captions, crop, "
    "zoom, borders, music, B-roll, filters, or speed receive zero credit. Narration "
    'NONE is valid. Return strict JSON: {"critiques": [...]} with exactly one item '
    "per requested plan, echoing its plan_id and plan_output_fingerprint, using only "
    "the closed values defined by the schema."
)

GOVERNANCE_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "critiques": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "plan_id": {"type": "string"},
                    "plan_output_fingerprint": {"type": "string"},
                    "fidelity": {
                        "type": "string",
                        "enum": ["PASS", "CONCERN", "FAIL", "UNKNOWN"],
                    },
                    "value": {
                        "type": "string",
                        "enum": [
                            "DISTINCT",
                            "ADEQUATE",
                            "REDUNDANT",
                            "GENERIC",
                            "NONE",
                            "UNKNOWN",
                        ],
                    },
                    "retention": {
                        "type": "string",
                        "enum": ["PRESERVED", "MIXED", "DAMAGED", "UNKNOWN"],
                    },
                    "coherence": {
                        "type": "string",
                        "enum": ["COHERENT", "MIXED", "INCOHERENT", "UNKNOWN"],
                    },
                    "narration": {
                        "type": "string",
                        "enum": [
                            "APPROPRIATE",
                            "EXCESSIVE",
                            "REDUNDANT",
                            "POSITION_DAMAGING",
                            "NOT_NEEDED",
                            "UNKNOWN",
                        ],
                    },
                    "template_feel": {
                        "type": "string",
                        "enum": ["LOW", "MODERATE", "HIGH", "UNKNOWN"],
                    },
                    "unsupported_claim": {
                        "type": "string",
                        "enum": ["NONE", "POSSIBLE", "CLEAR", "UNKNOWN"],
                    },
                    "finding_codes": {"type": "array", "items": {"type": "string"}},
                    "block_indexes": {"type": "array", "items": {"type": "integer"}},
                    "summary": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "plan_id",
                    "plan_output_fingerprint",
                    "fidelity",
                    "value",
                    "retention",
                    "coherence",
                    "narration",
                    "template_feel",
                    "unsupported_claim",
                ],
            },
        }
    },
    "required": ["critiques"],
}


def governance_prompt_hash() -> str:
    return hashlib.sha256(
        (
            GOVERNANCE_SYSTEM_INSTRUCTION + json.dumps(GOVERNANCE_OUTPUT_SCHEMA, sort_keys=True)
        ).encode()
    ).hexdigest()


class GovernanceProviderError(Exception):
    """A Stage 4.2 provider failure with a bounded category."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"transformation_governance_{category}")
        self.category = category
        self.detail = detail


class GovernanceProvider(Protocol):
    def govern(
        self, request: "GovernanceRequest", tier: str = _ROUTINE
    ) -> ProviderGovernanceResult: ...

    def release(self) -> None: ...

    def runtime_identity(self) -> dict[str, object]: ...

    def refresh_runtime_identity(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class GovernanceRequest:
    plan_set_id: str
    candidate_id: str
    content_type: str
    source_moment_structure: str
    reflection_start: float
    reflection_end: float
    dialect_profile: str | None
    transcript: str
    context_text: str
    idea_summary: str
    topic_summary: str
    rights_risk: str
    originality_risk: str
    plans: tuple[dict[str, object], ...]
    priority: str = GOVERNANCE_PRIORITY

    def to_payload(self) -> dict[str, object]:
        return {
            "plan_set_id": self.plan_set_id,
            "candidate_id": self.candidate_id,
            "content_type": self.content_type,
            "source_moment_structure": self.source_moment_structure,
            "refined_start": self.reflection_start,
            "refined_end": self.reflection_end,
            "dialect_profile": self.dialect_profile,
            "refined_transcript": self.transcript,
            "context_text": self.context_text,
            "idea_summary": self.idea_summary,
            "topic_summary": self.topic_summary,
            "rights_risk": self.rights_risk,
            "originality_risk": self.originality_risk,
            "plans": [dict(plan) for plan in self.plans],
            "priority": self.priority,
        }


def _plan_payload(plan: PlanEvidence) -> dict[str, object]:
    return {
        "plan_id": plan.plan_id,
        "plan_output_fingerprint": plan.plan_output_fingerprint,
        "strategy_type": plan.strategy_type,
        "intensity": plan.intensity,
        "strategy_fingerprint": plan.strategy_fingerprint,
        "blocks": [dict(block) for block in plan.blocks],
        "narration": dict(plan.narration),
        "verification_dependencies": [dict(item) for item in plan.verification_dependencies],
        "stage40_risk": dict(plan.stage40_risk),
        "preservation_constraints": list(plan.preservation_constraints),
        "original_value_kinds": list(plan.original_value_kinds),
    }


def build_governance_request(inputs: GovernanceInputs, config: Stage42Config) -> GovernanceRequest:
    return GovernanceRequest(
        plan_set_id=inputs.plan_set_id,
        candidate_id=inputs.candidate_id,
        content_type=inputs.content_type,
        source_moment_structure=inputs.source_moment_structure,
        reflection_start=inputs.refined_start,
        reflection_end=inputs.refined_end,
        dialect_profile=inputs.dialect_profile,
        transcript=inputs.transcript[: config.provider_max_input_characters],
        context_text=" ".join(inputs.context_segments)[: config.provider_max_input_characters],
        idea_summary=inputs.idea_summary,
        topic_summary=inputs.topic_summary,
        rights_risk=inputs.rights_risk,
        originality_risk=inputs.originality_risk,
        plans=tuple(_plan_payload(plan) for plan in inputs.plans),
    )


def coerce_enum(value: object, enum_type: Any) -> Any | None:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            return None
    return None


def _bounded_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _provider_text_forbidden(text: str) -> bool:
    if not text.strip():
        return False
    if boundary_violation(text) is not None:
        return True
    folded = text.casefold()
    if any(marker.casefold() in folded for marker in PLATFORM_GUARANTEE_MARKERS):
        return True
    if any(token.casefold() in folded for token in _FORBIDDEN_STATUS_TOKENS):
        return True
    # Provider voice/speaker selection (provider token + voice context).
    if any(token in folded for token in TTS_PROVIDER_TOKENS) and any(
        cue in folded for cue in VOICE_SELECTION_CONTEXT_MARKERS
    ):
        return True
    return False


def _valid_indexes(value: object, limit: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    indexes: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        indexes.append(int(item))
        if len(indexes) >= limit:
            break
    return tuple(indexes)


def _parse_critique(
    entry: Mapping[str, object],
    requested: Mapping[str, Mapping[str, object]],
) -> ProviderCritique | None:
    plan_id = entry.get("plan_id")
    if not isinstance(plan_id, str) or plan_id not in requested:
        return None
    plan = requested[plan_id]
    fingerprint = entry.get("plan_output_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != plan.get("plan_output_fingerprint"):
        return None
    fidelity = coerce_enum(entry.get("fidelity"), SemanticFidelityFinding)
    value = coerce_enum(entry.get("value"), SubstantiveValueFinding)
    retention = coerce_enum(entry.get("retention"), RetentionEffectFinding)
    coherence = coerce_enum(entry.get("coherence"), CoherenceFinding)
    narration = coerce_enum(entry.get("narration"), NarrationBurdenFinding)
    template = coerce_enum(entry.get("template_feel"), GovernanceLevel)
    unsupported = coerce_enum(entry.get("unsupported_claim"), UnsupportedClaimFinding)
    if None in (fidelity, value, retention, coherence, narration, template, unsupported):
        return None

    codes_raw = entry.get("finding_codes")
    codes: list[str] = []
    if isinstance(codes_raw, list):
        for item in codes_raw:
            if isinstance(item, str) and item.strip():
                codes.append(item.strip()[:64])
            if len(codes) >= _BOUNDED_CODES:
                break
    summary = _bounded_text(entry.get("summary"), _BOUNDED_SUMMARY)
    if _provider_text_forbidden(summary) or any(_provider_text_forbidden(code) for code in codes):
        return None
    confidence = entry.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = 0.0

    return ProviderCritique(
        plan_id=plan_id,
        plan_output_fingerprint=fingerprint,
        fidelity=fidelity,
        value=value,
        retention=retention,
        coherence=coherence,
        narration=narration,
        template_feel=template,
        unsupported_claim=unsupported,
        finding_codes=tuple(codes),
        block_indexes=_valid_indexes(entry.get("block_indexes"), _BOUNDED_INDEXES),
        summary=summary,
        confidence=clamp(float(confidence)),
    )


def parse_governance_result(
    content: Mapping[str, object],
    request: GovernanceRequest,
) -> ProviderGovernanceResult:
    """Tolerant per-item parsing: one malformed critique never blocks siblings."""

    entries = content.get("critiques")
    if not isinstance(entries, list):
        raise GovernanceProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
    requested = {str(plan["plan_id"]): plan for plan in request.plans}
    critiques: list[ProviderCritique] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        parsed = _parse_critique(entry, requested)
        if parsed is None or parsed.plan_id in seen:
            continue
        seen.add(parsed.plan_id)
        critiques.append(parsed)
    confidence = max((item.confidence for item in critiques), default=0.0)
    return ProviderGovernanceResult(critiques=tuple(critiques), notes="", confidence=confidence)


def serialize_critique(critique: ProviderCritique) -> dict[str, object]:
    return critique.as_dict()


def deserialize_critique(data: object) -> ProviderCritique | None:
    if not isinstance(data, Mapping):
        return None
    return ProviderCritique.from_dict(data)


class DeterministicGovernanceProvider:
    """No-op provider used by deterministic mode; makes zero network calls."""

    provider_name = "deterministic"
    model = None
    hosted_provider = False

    def govern(self, request: GovernanceRequest, tier: str = _ROUTINE) -> ProviderGovernanceResult:
        return ProviderGovernanceResult()

    def release(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "deterministic",
            "prompt_hash": governance_prompt_hash(),
            "schema_version": GOVERNOR_SCHEMA_VERSION,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


__all__ = [
    "DeterministicGovernanceProvider",
    "GOVERNANCE_OUTPUT_SCHEMA",
    "GOVERNANCE_PRIORITY",
    "GOVERNANCE_SYSTEM_INSTRUCTION",
    "GovernanceProvider",
    "GovernanceProviderError",
    "GovernanceRequest",
    "build_governance_request",
    "coerce_enum",
    "deserialize_critique",
    "governance_prompt_hash",
    "parse_governance_result",
    "serialize_critique",
]
