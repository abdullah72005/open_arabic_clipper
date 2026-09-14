"""Typed Stage 4.0 inputs, independent assessments, and strategy outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.core.enums import (
    ContentType,
    ExternalFactRequirement,
    OriginalityRisk,
    RightsRisk,
    SourceMomentStructure,
    StrategyDisposition,
    StrategyOrigin,
    SubstantiveValueKind,
    TransformationEligibilityOutcome,
    TransformationIntensity,
    TransformationStrategyType,
)


def clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class StrategyAssessments:
    """Independent strategy-level risk/value assessments, each in [0, 1]."""

    retention_preservation: float
    source_moment_damage_risk: float
    added_value_density: float
    originality_potential: float
    source_dominance_risk: float
    generic_filler_risk: float
    redundant_commentary_risk: float
    template_staleness_risk: float

    def as_dict(self) -> dict[str, float]:
        return {
            "retention_preservation": clamp(self.retention_preservation),
            "source_moment_damage_risk": clamp(self.source_moment_damage_risk),
            "added_value_density": clamp(self.added_value_density),
            "originality_potential": clamp(self.originality_potential),
            "source_dominance_risk": clamp(self.source_dominance_risk),
            "generic_filler_risk": clamp(self.generic_filler_risk),
            "redundant_commentary_risk": clamp(self.redundant_commentary_risk),
            "template_staleness_risk": clamp(self.template_staleness_risk),
        }


@dataclass(frozen=True)
class AnalysisAssessments:
    """Independent analysis-level assessments, each in [0, 1]."""

    retention_preservation: float
    source_moment_damage_risk: float
    added_value_density: float
    transformation_necessity: float
    transformation_potential: float
    originality_potential: float
    source_dominance_risk: float
    generic_ai_filler_risk: float
    redundant_commentary_risk: float
    template_staleness_risk: float

    def as_dict(self) -> dict[str, float]:
        return {
            "retention_preservation": clamp(self.retention_preservation),
            "source_moment_damage_risk": clamp(self.source_moment_damage_risk),
            "added_value_density": clamp(self.added_value_density),
            "transformation_necessity": clamp(self.transformation_necessity),
            "transformation_potential": clamp(self.transformation_potential),
            "originality_potential": clamp(self.originality_potential),
            "source_dominance_risk": clamp(self.source_dominance_risk),
            "generic_ai_filler_risk": clamp(self.generic_ai_filler_risk),
            "redundant_commentary_risk": clamp(self.redundant_commentary_risk),
            "template_staleness_risk": clamp(self.template_staleness_risk),
        }


@dataclass(frozen=True)
class SourceMoment:
    """Bounded deterministic description of the source moment's structure."""

    structure: SourceMomentStructure
    hook_index: int | None
    payoff_index: int | None
    duration: float
    moment_density: float
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "structure": self.structure.value,
            "hook_index": self.hook_index,
            "payoff_index": self.payoff_index,
            "duration": round(float(self.duration), 3),
            "moment_density": clamp(self.moment_density),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class StrategyDraft:
    """One validated (recommended or rejected) Stage 4.0 strategy direction."""

    strategy_type: TransformationStrategyType
    disposition: StrategyDisposition
    rank: int
    intensity: TransformationIntensity
    direction_summary: str
    added_value_focus: str
    substantive_value_kind: SubstantiveValueKind
    source_moment_role: str
    preservation_requirements: tuple[str, ...]
    assessments: StrategyAssessments
    external_verification_requirement: ExternalFactRequirement = (
        ExternalFactRequirement.NOT_REQUIRED
    )
    verification_requirements: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    confidence: float = 0.0
    origin: StrategyOrigin = StrategyOrigin.DETERMINISTIC
    provider_evidence: Mapping[str, object] = field(default_factory=dict)

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "strategy_type": self.strategy_type.value,
            "disposition": self.disposition.value,
            "rank": self.rank,
            "intensity": self.intensity.value,
            "direction_summary": self.direction_summary,
            "added_value_focus": self.added_value_focus,
            "substantive_value_kind": self.substantive_value_kind.value,
            "source_moment_role": self.source_moment_role,
            "preservation_requirements": list(self.preservation_requirements),
            "assessments": self.assessments.as_dict(),
            "external_verification_requirement": self.external_verification_requirement.value,
            "verification_requirements": list(self.verification_requirements),
            "rejection_reasons": list(self.rejection_reasons),
            "confidence": clamp(self.confidence),
            "origin": self.origin.value,
        }


@dataclass(frozen=True)
class TransformationOutcome:
    """Complete deterministic+provider Stage 4.0 analysis outcome."""

    eligibility_outcome: TransformationEligibilityOutcome
    eligibility_reasons: tuple[str, ...]
    assessments: AnalysisAssessments
    source_moment: SourceMoment
    platform_risk: Mapping[str, object]
    intensity: TransformationIntensity | None
    strategies: tuple[StrategyDraft, ...]
    provider_status: str
    provider_evidence: Mapping[str, object]
    provider_input_fingerprint: str
    cache_eligible: bool
    input_fingerprint: str
    output_fingerprint: str
    metrics: Mapping[str, object]

    @property
    def recommended(self) -> tuple[StrategyDraft, ...]:
        return tuple(
            item for item in self.strategies if item.disposition is StrategyDisposition.RECOMMENDED
        )

    @property
    def rejected(self) -> tuple[StrategyDraft, ...]:
        return tuple(
            item for item in self.strategies if item.disposition is StrategyDisposition.REJECTED
        )


@dataclass(frozen=True)
class TransformationInputs:
    """Immutable, output-relevant Stage 4.0 input evidence (bounded only)."""

    candidate_id: str
    candidate_key: str
    source_id: str
    disposition: str
    content_type: ContentType
    secondary_content_types: tuple[ContentType, ...]
    coarse_start: float
    coarse_end: float
    refined_start: float | None
    refined_end: float | None
    transcript: str
    transcript_confidence: float
    refinement_confidence: float
    word_timestamps: tuple[Mapping[str, object], ...]
    unresolved_spans: tuple[Mapping[str, object], ...]
    entity_evidence: tuple[Mapping[str, object], ...]
    dialect_profile: str | None
    dialect_confidence: float
    code_switch: Mapping[str, object]
    context_segments: tuple[str, ...]
    clip_score: float
    short_form_score: float
    moment_density_score: float
    ending_quality_score: float
    loopability_score: float
    idea_summary: str
    topic_summary: str
    hooks: tuple[Mapping[str, object], ...]
    rights_risk: RightsRisk
    originality_risk: OriginalityRisk
    rights_status: str
    media_origin: str
    provenance_snapshot: Mapping[str, object]
    refinement_priority: str
    refinement_quality_level: str
    refinement_status: str
    refinement_output_fingerprint: str
    stage3_analysis_fingerprint: str
    stage3_policy_version: str

    @property
    def effective_start(self) -> float:
        return self.refined_start if self.refined_start is not None else self.coarse_start

    @property
    def effective_end(self) -> float:
        return self.refined_end if self.refined_end is not None else self.coarse_end

    @property
    def duration(self) -> float:
        return max(0.0, self.effective_end - self.effective_start)


@dataclass(frozen=True)
class TransformationProviderStrategy:
    """One provider-discovered strategy after strict schema parsing."""

    strategy_type: TransformationStrategyType
    disposition: StrategyDisposition
    intensity: TransformationIntensity
    direction_summary: str
    added_value_focus: str
    substantive_value_kind: SubstantiveValueKind | None
    preservation_requirements: tuple[str, ...]
    external_verification_requirement: ExternalFactRequirement
    verification_requirements: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    confidence: float
    retention_preservation: float | None
    source_moment_damage_risk: float | None
    added_value_density: float | None
    originality_potential: float | None
    source_dominance_risk: float | None
    generic_filler_risk: float | None
    redundant_commentary_risk: float | None
    template_staleness_risk: float | None


@dataclass(frozen=True)
class TransformationProviderResult:
    candidate_id: str
    strategies: tuple[TransformationProviderStrategy, ...]
    notes: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class TransformationVerification:
    """Bounded external-fact dependency flagged for Stage 4.1/4.2."""

    requirement: ExternalFactRequirement
    details: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {"requirement": self.requirement.value, "details": list(self.details)}


def assessments_from_mapping(value: Mapping[str, object]) -> StrategyAssessments:
    def get(key: str) -> float:
        raw = value.get(key, 0.0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return 0.0
        return clamp(float(raw))

    return StrategyAssessments(
        retention_preservation=get("retention_preservation"),
        source_moment_damage_risk=get("source_moment_damage_risk"),
        added_value_density=get("added_value_density"),
        originality_potential=get("originality_potential"),
        source_dominance_risk=get("source_dominance_risk"),
        generic_filler_risk=get("generic_filler_risk"),
        redundant_commentary_risk=get("redundant_commentary_risk"),
        template_staleness_risk=get("template_staleness_risk"),
    )


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return clamp(sum(clamp(value) for value in values) / len(values))
