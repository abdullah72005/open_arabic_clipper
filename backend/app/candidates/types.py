"""Typed Stage 3 candidate-search structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from app.core.enums import (
    CandidateDisposition,
    ContentType,
    HookOrigin,
    HookType,
    OriginalityRisk,
    RefinementReason,
    RightsRisk,
)


@dataclass(frozen=True)
class Proposal:
    """A deterministic coarse clip proposal over contiguous segment indexes."""

    start_segment_index: int
    end_segment_index: int
    start_time: float
    end_time: float
    text: str
    boundary_reason: str
    segment_indexes: tuple[int, ...]

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass(frozen=True)
class ContentClassification:
    primary: ContentType
    secondary: tuple[ContentType, ...]
    scores: Mapping[str, float]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class UncertaintyEvidence:
    """Bounded transcript/audio/boundary uncertainty derived from candidate evidence."""

    transcript_confidence: float
    low_confidence_spans: tuple[dict[str, object], ...]
    unresolved_segment_indexes: tuple[int, ...]
    low_confidence_word_span_ratio: float
    unresolved_ratio: float
    protected_tokens: tuple[str, ...]
    code_switch_tokens: tuple[str, ...]
    code_switch_uncertainty: bool
    reasons: tuple[RefinementReason, ...]


@dataclass(frozen=True)
class CandidateScores:
    clip_score: float
    short_form_score: float
    moment_density_score: float
    boredom_risk_score: float
    ending_quality_score: float
    loopability_score: float
    engagement_confidence: float
    transcript_confidence: float
    audio_confidence: float
    boundary_confidence: float
    uncertainty_severity: float
    idea_novelty_score: float
    topic_novelty_score: float
    recent_semantic_similarity_risk: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class HookRecord:
    type: HookType
    text: str | None
    source_segment_indexes: tuple[int, ...]
    source_evidence: tuple[str, ...]
    strength: float
    faithfulness: float
    naturalness: float
    audience_suitability: float
    policy_risk: float
    missing_context_dependency: float
    origin: HookOrigin

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "text": self.text,
            "source_segment_indexes": list(self.source_segment_indexes),
            "source_evidence": list(self.source_evidence),
            "strength": self.strength,
            "faithfulness": self.faithfulness,
            "naturalness": self.naturalness,
            "audience_suitability": self.audience_suitability,
            "policy_risk": self.policy_risk,
            "missing_context_dependency": self.missing_context_dependency,
            "origin": self.origin.value,
        }


@dataclass
class CandidateDraft:
    """A fully-scored candidate ready for atomic persistence."""

    candidate_key: str
    proposal: Proposal
    content: ContentClassification
    scores: CandidateScores
    disposition: CandidateDisposition
    refinement_reasons: tuple[RefinementReason, ...] = ()
    refinement_evidence: Mapping[str, object] = field(default_factory=dict)
    idea_summary: str = ""
    topic_summary: str = ""
    idea_signature: str = ""
    topic_signature: str = ""
    hooks: list[HookRecord] = field(default_factory=list)
    provenance_snapshot: Mapping[str, object] = field(default_factory=dict)
    rights_risk: RightsRisk = RightsRisk.UNDETERMINED
    originality_risk: OriginalityRisk = OriginalityRisk.UNDETERMINED
    dialect_profile: str | None = None
    dialect_confidence: float = 0.0
    code_switch_suspected: bool = False
    provider_input_fingerprint: str = ""
    provider_evidence: Mapping[str, object] = field(default_factory=dict)
    analysis_fingerprint: str = ""
    transcript_excerpt: str = ""
    evidence_snapshot: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateAnalysisOutcome:
    """Result returned by the Stage 3 service/executor."""

    input_fingerprint: str
    output_fingerprint: str
    provider_status: str
    semantic_provider_mode: str
    cache_eligible: bool
    metrics: Mapping[str, object]
    candidates: tuple[CandidateDraft, ...]
