"""Bounded immutable Stage 4.3 selection evidence types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from app.core.enums import TransformationSelectionStatus

# Alternative inclusion tiers.
TIER_CLEAN = "CLEAN"
TIER_CAUTION = "CAUTION"
TIER_EXCLUDED = "EXCLUDED"


@dataclass(frozen=True)
class PlanAlternative:
    """Deterministic view of one current Stage 4.1 plan and its Stage 4.2 result."""

    plan_id: str
    plan_output_fingerprint: str
    governance_result_id: str | None
    governance_output_fingerprint: str
    status: str
    eligible: bool
    generation_rank: int
    planner_confidence: float
    strategy_type: str
    intensity: str
    hard_gates: tuple[Mapping[str, object], ...]
    dimensions: Mapping[str, object]
    verification: Mapping[str, object]
    platform_risk: Mapping[str, object]
    warnings: tuple[Mapping[str, object], ...]
    reason_codes: tuple[str, ...]
    remediation: tuple[Mapping[str, object], ...]
    provider_evidence: Mapping[str, object]
    governance_input_fingerprint: str
    inclusion: str = TIER_EXCLUDED
    exclusion_codes: tuple[str, ...] = ()
    snapshot: Mapping[str, object] = field(default_factory=dict)

    @property
    def result_view(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "plan_output_fingerprint": self.plan_output_fingerprint,
            "result_id": self.governance_result_id,
            "status": self.status,
            "eligible_for_stage4_3": self.eligible,
            "hard_gates": [dict(item) for item in self.hard_gates],
            "dimensions": dict(self.dimensions),
            "verification": dict(self.verification),
            "platform_risk": dict(self.platform_risk),
            "warnings": [dict(item) for item in self.warnings],
            "reason_codes": list(self.reason_codes),
            "remediation": [dict(item) for item in self.remediation],
            "provider_evidence": dict(self.provider_evidence),
            "output_fingerprint": self.governance_output_fingerprint,
            "intensity": self.intensity,
        }

    def disposition(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "governance_result_id": self.governance_result_id,
            "status": self.status,
            "eligible_for_stage4_3": self.eligible,
            "inclusion": self.inclusion,
            "exclusion_reason_codes": list(self.exclusion_codes),
        }


@dataclass(frozen=True)
class SelectionDecision:
    """The complete deterministic Stage 4.3 outcome before persistence."""

    status: TransformationSelectionStatus
    reason_codes: tuple[str, ...]
    selected_plan_id: str | None
    selected_governance_result_id: str | None
    selected_with_caution: bool
    selected_plan_fingerprint: str
    arbitration_evidence: Mapping[str, object]
    selected_governance_snapshot: Mapping[str, object]
    alternative_dispositions: tuple[Mapping[str, object], ...]
    governance_input_fingerprint: str
    governance_output_fingerprint: str
    governor_policy_version: str
    governor_validation_version: str
    platform_policy_profile_version: str
    transformation_analysis_id: str | None
    transformation_plan_set_id: str | None
    transformation_governance_set_id: str | None
    refinement_id: str | None
    refinement_priority: str
    refinement_quality_level: str
    refinement_output_fingerprint: str
    freshness: str
    input_fingerprint: str
    output_fingerprint: str


__all__ = [
    "PlanAlternative",
    "SelectionDecision",
    "TIER_CAUTION",
    "TIER_CLEAN",
    "TIER_EXCLUDED",
]
