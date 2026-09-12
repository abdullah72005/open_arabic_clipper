"""Bounded value objects shared across Stage 3.5 candidate refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.enums import AdmissionPriority, EvidenceKind, EvidenceState, RefinementPriority

VALID_STAGE35_PRIORITIES = (RefinementPriority.CANDIDATE, RefinementPriority.FINAL_CLIP)


class RefinementConfigurationError(ValueError):
    """A Stage 3.5 request referenced an unsupported priority or input."""


class RefinementCancelled(RuntimeError):
    """Cooperative cancellation of a Stage 3.5 refinement."""

    retryable = False


@dataclass(frozen=True)
class WordTimestamp:
    """One word with source-time bounds after context-offset conversion."""

    text: str
    start: float
    end: float
    probability: float | None = None

    @property
    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class EvidenceRecord:
    """One bounded piece of transcript evidence.

    Records are deduplicated by ``fingerprint`` and persisted as JSON. They never
    contain API keys, remote URIs, raw provider objects, or unbounded prompts.
    """

    kind: EvidenceKind
    fingerprint: str
    provider: str
    model: str | None
    settings: dict[str, object]
    window_start: float
    window_end: float
    transcript: str
    confidence: float
    state: EvidenceState
    word_timestamps: tuple[WordTimestamp, ...] = ()
    reason: str | None = None
    entity_metadata: dict[str, object] = field(default_factory=dict)
    code_switch_tokens: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "fingerprint": self.fingerprint,
            "provider": self.provider,
            "model": self.model,
            "settings": dict(self.settings),
            "window_start": self.window_start,
            "window_end": self.window_end,
            "transcript": self.transcript,
            "confidence": self.confidence,
            "state": self.state.value,
            "word_timestamps": [word.as_dict for word in self.word_timestamps],
            "reason": self.reason,
            "entity_metadata": dict(self.entity_metadata),
            "code_switch_tokens": list(self.code_switch_tokens),
        }


@dataclass(frozen=True)
class EntityMention:
    """A practical important entity observed in candidate evidence."""

    text: str
    normalized: str
    entity_type: str
    start: float | None
    end: float | None
    evidence_fingerprints: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnresolvedSpan:
    """Bounded meaning-critical or flagged ambiguity for manual review."""

    span_id: str
    start: float | None
    end: float | None
    context: str
    readings: tuple[str, ...]
    evidence_fingerprints: tuple[str, ...]
    providers: tuple[str, ...]
    confidence: float
    reason: str
    entity_type: str | None
    meaning_critical: bool
    resolution_state: str = "UNRESOLVED"
    operator_resolution: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "span_id": self.span_id,
            "start": self.start,
            "end": self.end,
            "context": self.context,
            "readings": list(self.readings),
            "evidence_fingerprints": list(self.evidence_fingerprints),
            "providers": list(self.providers),
            "confidence": self.confidence,
            "reason": self.reason,
            "entity_type": self.entity_type,
            "meaning_critical": self.meaning_critical,
            "resolution_state": self.resolution_state,
            "operator_resolution": self.operator_resolution,
        }


@dataclass(frozen=True)
class BoundaryResult:
    """Deterministic refined boundaries plus their evidence."""

    start: float
    end: float
    confidence: float
    reasons: tuple[str, ...] = ()
    evidence: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class RefinementAudioWindow:
    """The bounded extracted candidate audio interval."""

    source_id: str
    candidate_id: str
    priority: RefinementPriority
    coarse_start: float
    coarse_end: float
    context_start: float
    context_end: float
    relative_path: str
    content_hash: str
    duration: float
    input_fingerprint: str


@dataclass(frozen=True)
class TargetASRResult:
    """One audio-backed ASR reading of a candidate window."""

    provider: str
    model: str
    language: str | None
    language_probability: float | None
    transcript: str
    word_timestamps: tuple[WordTimestamp, ...]
    confidence: float
    runtime_identity: dict[str, object] = field(default_factory=dict)
    fingerprint: str = ""
    settings: dict[str, object] = field(default_factory=dict)
    rejected_reason: str | None = None


@dataclass(frozen=True)
class AdjudicationRequest:
    """One bounded ambiguity offered to the adjudication provider."""

    ambiguity_id: str
    context: str
    candidate_readings: tuple[str, ...]
    evidence_summary: tuple[dict[str, object], ...]
    dialect_profile: str | None
    meaning_critical: bool


@dataclass(frozen=True)
class AdjudicationResult:
    """Provider adjudication outcome; never free-form transcript rewriting."""

    ambiguity_id: str
    selected_reading: str | None
    confidence: float
    reason: str
    rejected_reason: str | None = None


@dataclass
class RefinementContext:
    """Mutable per-run accumulator passed through the refinement phases."""

    metrics: dict[str, int] = field(default_factory=dict)
    provider_evidence: dict[str, object] = field(default_factory=dict)
    routing_evidence: dict[str, object] = field(default_factory=dict)
    component_fingerprints: dict[str, object] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def bump(self, key: str, amount: int = 1) -> None:
        self.metrics[key] = self.metrics.get(key, 0) + amount


@dataclass(frozen=True)
class RefinementOutcome:
    """The persisted result of one candidate-scoped refinement run."""

    source_id: str
    candidate_id: str
    priority: RefinementPriority
    quality_level: str
    status: str
    coarse_start: float
    coarse_end: float
    context_start: float
    context_end: float
    refined_start: float
    refined_end: float
    audio_relative_path: str
    audio_content_hash: str
    audio_input_fingerprint: str
    automatic_transcript: str
    manual_transcript: str | None
    final_transcript: str
    word_timestamps: tuple[WordTimestamp, ...]
    confidence: float
    dialect_profile: str | None
    dialect_confidence: float
    code_switch_evidence: dict[str, object]
    transcript_evidence: tuple[EvidenceRecord, ...]
    entity_evidence: tuple[EntityMention, ...]
    unresolved_spans: tuple[UnresolvedSpan, ...]
    provider_evidence: dict[str, object]
    routing_evidence: dict[str, object]
    input_fingerprint: str
    output_fingerprint: str
    component_fingerprints: dict[str, object]
    cache_eligible: bool
    metrics: dict[str, int]
    processing_duration: float


def priority_is_stage35(priority: RefinementPriority) -> bool:
    """Stage 3.5 only accepts CANDIDATE and FINAL_CLIP priorities."""

    return priority in VALID_STAGE35_PRIORITIES


def admission_priority_for(priority: RefinementPriority) -> AdmissionPriority:
    """Map a refinement quality level to the shared Gemini admission class."""

    if priority is RefinementPriority.FINAL_CLIP:
        return AdmissionPriority.CRITICAL
    return AdmissionPriority.MEDIUM
