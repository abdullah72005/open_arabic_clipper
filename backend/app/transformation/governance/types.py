"""Typed Stage 4.2 governance inputs, provider critiques, and outcomes.

Everything here is immutable. The governor consumes the immutable Stage 4.1 plan
representation and never mutates, rewrites, or repairs it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from app.core.enums import (
    ClaimGroundingState,
    CoherenceFinding,
    GovernanceLevel,
    GovernancePlanStatus,
    GovernanceSemanticOutcome,
    GovernanceSeverityClass,
    NarrationBurdenFinding,
    RetentionEffectFinding,
    SemanticFidelityFinding,
    SubstantiveValueFinding,
    UnsupportedClaimFinding,
)
from app.transformation.planning.types import PlanningContext

_EMPTY_CONTEXT = PlanningContext()


@dataclass(frozen=True)
class PlanEvidence:
    """Immutable governance-relevant representation of one Stage 4.1 plan."""

    plan_id: str
    plan_key: str
    plan_output_fingerprint: str
    provider_input_fingerprint: str
    strategy_id: str
    strategy_key: str
    strategy_type: str
    strategy_fingerprint: str
    strategy_rank: int
    intensity: str
    status: str
    generation_rank: int
    is_current: bool
    planner_confidence: float
    blocks: tuple[Mapping[str, object], ...]
    hero_block_index: int
    hero_source_start: float | None
    hero_source_end: float | None
    hero_appearance_time: float
    narration: Mapping[str, object]
    verification_dependencies: tuple[Mapping[str, object], ...]
    derived_durations: Mapping[str, object]
    hook_payoff_evidence: Mapping[str, object]
    original_value_kinds: tuple[str, ...]
    original_value_reasons: tuple[str, ...]
    preservation_constraints: tuple[str, ...]
    required_context: tuple[str, ...]
    degraded_rules: tuple[str, ...]
    stage40_risk: Mapping[str, object]
    source_dialect: Mapping[str, object]
    target_audience: Mapping[str, object]
    strategy_snapshot: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "plan_key": self.plan_key,
            "plan_output_fingerprint": self.plan_output_fingerprint,
            "provider_input_fingerprint": self.provider_input_fingerprint,
            "strategy_id": self.strategy_id,
            "strategy_key": self.strategy_key,
            "strategy_type": self.strategy_type,
            "strategy_fingerprint": self.strategy_fingerprint,
            "strategy_rank": self.strategy_rank,
            "intensity": self.intensity,
            "status": self.status,
            "generation_rank": self.generation_rank,
            "is_current": self.is_current,
            "planner_confidence": self.planner_confidence,
            "blocks": [dict(block) for block in self.blocks],
            "hero_block_index": self.hero_block_index,
            "hero_source_start": self.hero_source_start,
            "hero_source_end": self.hero_source_end,
            "hero_appearance_time": self.hero_appearance_time,
            "narration": dict(self.narration),
            "verification_dependencies": [dict(item) for item in self.verification_dependencies],
            "derived_durations": dict(self.derived_durations),
            "hook_payoff_evidence": dict(self.hook_payoff_evidence),
            "original_value_kinds": list(self.original_value_kinds),
            "original_value_reasons": list(self.original_value_reasons),
            "preservation_constraints": list(self.preservation_constraints),
            "required_context": list(self.required_context),
            "degraded_rules": list(self.degraded_rules),
            "stage40_risk": dict(self.stage40_risk),
            "source_dialect": dict(self.source_dialect),
            "target_audience": dict(self.target_audience),
            "strategy_snapshot": dict(self.strategy_snapshot),
        }


@dataclass(frozen=True)
class GovernanceInputs:
    """Immutable, bounded Stage 4.2 input evidence assembled from Stage 4.1."""

    candidate_id: str
    candidate_key: str
    source_id: str
    disposition: str
    content_type: str
    source_moment_structure: str
    transcript: str
    transcript_confidence: float
    refined_start: float
    refined_end: float
    context_segments: tuple[str, ...]
    dialect_profile: str | None
    dialect_confidence: float
    code_switch: Mapping[str, object]
    idea_summary: str
    topic_summary: str
    hooks: tuple[Mapping[str, object], ...]
    rights_risk: str
    originality_risk: str
    rights_status: str
    media_origin: str
    provenance_snapshot: Mapping[str, object]
    stage3_risk: Mapping[str, object]
    stage40_assessments: Mapping[str, object]
    stage40_platform_risk: Mapping[str, object]
    stage40_analysis_id: str
    stage40_input_fingerprint: str
    stage40_output_fingerprint: str
    stage40_policy_version: str
    plan_set_id: str
    plan_set_input_fingerprint: str
    plan_set_output_fingerprint: str
    plan_set_semantic_outcome: str
    plan_set_provider_identity: Mapping[str, object]
    plan_set_stage40_snapshot: Mapping[str, object]
    refinement_id: str | None
    refinement_priority: str
    refinement_quality_level: str
    refinement_status: str
    refinement_output_fingerprint: str
    target_context: PlanningContext
    plans: tuple[PlanEvidence, ...]

    @property
    def duration(self) -> float:
        return max(0.0, self.refined_end - self.refined_start)


@dataclass(frozen=True)
class ProviderCritique:
    """One strictly validated provider semantic critique for a single plan."""

    plan_id: str
    plan_output_fingerprint: str
    fidelity: SemanticFidelityFinding
    value: SubstantiveValueFinding
    retention: RetentionEffectFinding
    coherence: CoherenceFinding
    narration: NarrationBurdenFinding
    template_feel: GovernanceLevel
    unsupported_claim: UnsupportedClaimFinding
    finding_codes: tuple[str, ...]
    block_indexes: tuple[int, ...]
    summary: str
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "plan_output_fingerprint": self.plan_output_fingerprint,
            "fidelity": self.fidelity.value,
            "value": self.value.value,
            "retention": self.retention.value,
            "coherence": self.coherence.value,
            "narration": self.narration.value,
            "template_feel": self.template_feel.value,
            "unsupported_claim": self.unsupported_claim.value,
            "finding_codes": list(self.finding_codes),
            "block_indexes": list(self.block_indexes),
            "summary": self.summary,
            "confidence": round(self.confidence, 4),
        }

    @staticmethod
    def from_dict(data: Mapping[str, object]) -> "ProviderCritique | None":
        return _critique_from_dict(data)


@dataclass(frozen=True)
class ProviderGovernanceResult:
    """Parsed provider output for one request (one plan set)."""

    critiques: tuple[ProviderCritique, ...] = ()
    notes: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class PlanGovernance:
    """Deterministic+provider governance result for one current plan."""

    plan_id: str
    plan_output_fingerprint: str
    status: GovernancePlanStatus
    eligible_for_stage4_3: bool
    severity: GovernanceSeverityClass
    hard_gates: tuple[Mapping[str, object], ...]
    dimensions: Mapping[str, object]
    verification: Mapping[str, object]
    platform_risk: Mapping[str, object]
    reason_codes: tuple[str, ...]
    warnings: tuple[Mapping[str, object], ...]
    remediation: tuple[Mapping[str, object], ...]
    provider_evidence: Mapping[str, object]
    input_fingerprint: str
    output_fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "plan_output_fingerprint": self.plan_output_fingerprint,
            "status": self.status.value,
            "eligible_for_stage4_3": self.eligible_for_stage4_3,
            "severity": self.severity.value,
            "hard_gates": [dict(item) for item in self.hard_gates],
            "dimensions": dict(self.dimensions),
            "verification": dict(self.verification),
            "platform_risk": dict(self.platform_risk),
            "reason_codes": list(self.reason_codes),
            "warnings": [dict(item) for item in self.warnings],
            "remediation": [dict(item) for item in self.remediation],
            "provider_evidence": dict(self.provider_evidence),
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
        }


@dataclass(frozen=True)
class GovernanceAttempt:
    """Bounded per-plan provider attempt/checkpoint state for a governance set."""

    plan_id: str
    plan_output_fingerprint: str
    status: str
    reasons: tuple[str, ...] = ()
    provider_input_fingerprint: str = ""
    checkpoint: Mapping[str, object] | None = None


@dataclass(frozen=True)
class GovernanceOutcome:
    """Complete deterministic+provider Stage 4.2 governance outcome."""

    execution_status: str
    semantic_outcome: GovernanceSemanticOutcome
    outcome_reasons: tuple[str, ...]
    plans: tuple[PlanGovernance, ...]
    attempts: tuple[GovernanceAttempt, ...]
    summary_counts: Mapping[str, int]
    provider_status: str
    provider_evidence: Mapping[str, object]
    provider_mode: str
    provider_identity: Mapping[str, object]
    input_fingerprint: str
    output_fingerprint: str
    cache_eligible: bool
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class PlanDimension:
    """One bounded categorical dimension finding with bounded evidence."""

    level: str
    reason_codes: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "reason_codes": list(self.reason_codes),
            "evidence": list(self.evidence),
        }


_EMPTY_DIMENSION = PlanDimension(level="UNKNOWN")


def _critique_from_dict(data: Mapping[str, object]) -> ProviderCritique | None:
    from app.transformation.governance.providers import coerce_enum

    try:
        plan_id = str(data["plan_id"])
        fingerprint = str(data["plan_output_fingerprint"])
    except KeyError:
        return None
    fidelity = coerce_enum(data.get("fidelity"), SemanticFidelityFinding)
    value = coerce_enum(data.get("value"), SubstantiveValueFinding)
    retention = coerce_enum(data.get("retention"), RetentionEffectFinding)
    coherence = coerce_enum(data.get("coherence"), CoherenceFinding)
    narration = coerce_enum(data.get("narration"), NarrationBurdenFinding)
    template = coerce_enum(data.get("template_feel"), GovernanceLevel)
    unsupported = coerce_enum(data.get("unsupported_claim"), UnsupportedClaimFinding)
    if None in (fidelity, value, retention, coherence, narration, template, unsupported):
        return None
    codes = data.get("finding_codes")
    indexes = data.get("block_indexes")
    confidence = data.get("confidence", 0.0)
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
        finding_codes=tuple(str(code) for code in codes) if isinstance(codes, list) else (),
        block_indexes=(
            tuple(
                int(index)
                for index in indexes
                if isinstance(index, int) and not isinstance(index, bool)
            )
            if isinstance(indexes, list)
            else ()
        ),
        summary=str(data.get("summary", ""))[:600],
        confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.0,
    )


__all__ = [
    "ClaimGroundingState",
    "GovernanceAttempt",
    "GovernanceInputs",
    "GovernanceOutcome",
    "PlanDimension",
    "PlanEvidence",
    "PlanGovernance",
    "ProviderCritique",
    "ProviderGovernanceResult",
]
