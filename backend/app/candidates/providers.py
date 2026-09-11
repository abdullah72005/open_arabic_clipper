"""Stage 3 semantic-provider protocol, schema, and deterministic provider.

Deliberately separate from the Stage 2.7 reconstruction prompt and response
schema: candidate evaluation is not transcript reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from app.candidates.policy import SEMANTIC_SCHEMA_VERSION
from app.candidates.types import HookRecord
from app.core.enums import ContentType

if TYPE_CHECKING:
    from app.candidates.policy import Stage3Config

SEMANTIC_PRIORITY = "MEDIUM"

SEMANTIC_SYSTEM_INSTRUCTION = (
    "You evaluate short candidate moments from an imperfect Arabic transcript for "
    "short-form clip potential. "
    "Judge moment and content quality separately from transcript confidence. "
    "An uncertain or imperfect transcript is never by itself a reason to reject a moment. "
    "Preserve the dialect and any code-switched terms exactly as evidenced; do not "
    "translate, localize, formalize, or Egyptianize anything. "
    "Do not invent facts, numbers, names, quotes, or context that is not present in the "
    "candidate or its bounded context. "
    "Score only the requested candidate IDs. Return at most three essential hooks. "
    "Output only the strict JSON schema with no extra text."
)


def semantic_prompt_hash() -> str:
    return hashlib.sha256(
        (SEMANTIC_SYSTEM_INSTRUCTION + json.dumps(_SEMANTIC_OUTPUT_SCHEMA, sort_keys=True)).encode(
            "utf-8"
        )
    ).hexdigest()


_SEMANTIC_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "primary_content_type": {
                        "type": "string",
                        "enum": [item.value for item in ContentType],
                    },
                    "secondary_content_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [item.value for item in ContentType],
                        },
                    },
                    "score_adjustments": {"type": "object"},
                    "idea_summary": {"type": "string"},
                    "topic_summary": {"type": "string"},
                    "hooks": {"type": "array"},
                    "confidence": {"type": "number"},
                    "explanation": {"type": "string"},
                },
                "required": ["candidate_id"],
            },
        }
    },
    "required": ["evaluations"],
}


@dataclass(frozen=True)
class SemanticEvaluationRequest:
    candidate_key: str
    text: str = ""
    previous_context: str = ""
    following_context: str = ""
    start: float = 0.0
    end: float = 0.0
    feature_summary: Mapping[str, object] = field(default_factory=dict)
    uncertainty_summary: Mapping[str, object] = field(default_factory=dict)
    dialect_profile: str | None = None
    dialect_confidence: float = 0.0
    protected_tokens: tuple[str, ...] = ()
    code_switch_tokens: tuple[str, ...] = ()
    priority: str = SEMANTIC_PRIORITY

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_key,
            "text": self.text,
            "previous_context": self.previous_context,
            "following_context": self.following_context,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "feature_summary": dict(self.feature_summary or {}),
            "uncertainty_summary": dict(self.uncertainty_summary or {}),
            "dialect_profile": self.dialect_profile,
            "dialect_confidence": self.dialect_confidence,
            "protected_tokens": list(self.protected_tokens),
            "code_switch_tokens": list(self.code_switch_tokens),
            "priority": self.priority,
        }


@dataclass(frozen=True)
class SemanticEvaluationResult:
    candidate_key: str
    primary_content_type: ContentType | None
    secondary_content_types: tuple[ContentType, ...]
    score_adjustments: Mapping[str, float]
    idea_summary: str
    topic_summary: str
    hooks: tuple[Mapping[str, object], ...]
    confidence: float
    explanation: str


class SemanticProviderError(Exception):
    """Sanitized provider failure that never carries credentials or transcript data."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"semantic_{category}")
        self.category = category
        self.detail = detail


class ProviderErrorCategory(str, Enum):
    MISSING_KEY = "MISSING_KEY"
    AUTHENTICATION = "AUTHENTICATION"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    INVALID_REQUEST = "INVALID_REQUEST"
    CONNECTION = "CONNECTION"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    SAFETY_REFUSAL = "SAFETY_REFUSAL"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"


RequestAdmission = Callable[[SemanticEvaluationRequest], bool]


class SemanticProvider(Protocol):
    """Minimal Stage 3-semantic provider boundary with a future budget seam."""

    def evaluate(
        self, requests: Sequence[SemanticEvaluationRequest]
    ) -> dict[str, SemanticEvaluationResult]: ...

    def release(self) -> None: ...

    def runtime_identity(self) -> dict[str, object]: ...

    def refresh_runtime_identity(self) -> dict[str, object]: ...


class DeterministicSemanticProvider:
    """Default provider: zero network calls, zero model loads, no results.

    Missing or misconfigured providers always degrade to this behavior.
    """

    provider_name = "deterministic"
    model = None

    def evaluate(
        self, requests: Sequence[SemanticEvaluationRequest]
    ) -> dict[str, SemanticEvaluationResult]:
        return {}

    def release(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "deterministic",
            "model": None,
            "prompt_hash": semantic_prompt_hash(),
            "schema_version": SEMANTIC_SCHEMA_VERSION,
            "temperature": 0.0,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


def parse_semantic_entries(
    content: Mapping[str, object],
    requests: Sequence[SemanticEvaluationRequest],
) -> dict[str, SemanticEvaluationResult]:
    """Strictly parse/evaluate provider entries, isolating malformed items.

    Accepted results survive unrelated malformed items. Unknown candidate IDs are
    ignored; a duplicate ID keeps only the first valid evaluation.
    """

    entries = content.get("evaluations")
    if not isinstance(entries, list):
        raise SemanticProviderError(ProviderErrorCategory.MALFORMED_OUTPUT)
    requested = {request.candidate_key for request in requests}
    results: dict[str, SemanticEvaluationResult] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        candidate_id = entry.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in requested:
            continue
        if candidate_id in results:
            continue
        parsed = _parse_entry(candidate_id, entry)
        if parsed is not None:
            results[candidate_id] = parsed
    return results


def _parse_entry(candidate_id: str, entry: Mapping[str, object]) -> SemanticEvaluationResult | None:
    raw_primary = entry.get("primary_content_type")
    if raw_primary is not None and _content_type(raw_primary) is None:
        return None
    primary = _content_type(raw_primary)
    secondary = tuple(
        item for item in _content_types(entry.get("secondary_content_types")) if item is not primary
    )
    adjustments = _score_adjustments(entry.get("score_adjustments"))
    confidence = _confidence(entry.get("confidence"))
    hooks = tuple(item for item in _as_mapping_list(entry.get("hooks")) if _hook_shaped(item))
    return SemanticEvaluationResult(
        candidate_key=candidate_id,
        primary_content_type=primary,
        secondary_content_types=secondary,
        score_adjustments=adjustments,
        idea_summary=_bounded_text(entry.get("idea_summary"), 500),
        topic_summary=_bounded_text(entry.get("topic_summary"), 500),
        hooks=hooks,
        confidence=confidence,
        explanation=_bounded_text(entry.get("explanation"), 1000),
    )


def _content_type(value: object) -> ContentType | None:
    if isinstance(value, ContentType):
        return value
    if isinstance(value, str):
        try:
            return ContentType(value)
        except ValueError:
            return None
    return None


def _content_types(value: object) -> list[ContentType]:
    if not isinstance(value, list):
        return []
    found: list[ContentType] = []
    for item in value:
        parsed = _content_type(item)
        if parsed is not None and parsed not in found:
            found.append(parsed)
    return found


def _score_adjustments(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    allowed = {
        "moment_density_score",
        "short_form_score",
        "ending_quality_score",
        "loopability_score",
        "boredom_risk_score",
    }
    adjustments: dict[str, float] = {}
    for key, raw in value.items():
        if key not in allowed:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        number = float(raw)
        if not math.isfinite(number):
            continue
        adjustments[str(key)] = max(-0.2, min(0.2, number))
    return adjustments


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _hook_shaped(value: Mapping[str, object]) -> bool:
    from app.core.enums import HookType

    hook_type = value.get("type")
    if isinstance(hook_type, HookType):
        return True
    if isinstance(hook_type, str):
        try:
            HookType(hook_type)
            return True
        except ValueError:
            return False
    return False


def _as_mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _bounded_text(value: object, limit: int) -> str:
    return str(value)[:limit] if isinstance(value, str) else ""


def rebuild_hooks(
    result: SemanticEvaluationResult,
    *,
    proposal_text: str,
    context_text: str,
    config: "Stage3Config",
) -> list[HookRecord]:
    """Expose validated provider hooks via the strict hook validator."""

    from app.candidates.hooks import validate_provider_hooks
    from app.candidates.types import Proposal

    proposal = Proposal(
        start_segment_index=0,
        end_segment_index=0,
        start_time=0.0,
        end_time=0.0,
        text=proposal_text,
        boundary_reason="provider",
        segment_indexes=(),
    )
    return validate_provider_hooks(result.hooks, proposal, context_text=context_text, config=config)
