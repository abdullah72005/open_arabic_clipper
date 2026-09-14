"""Typed Stage 4.1 planning inputs, block/value objects, and outcomes.

Everything here is immutable. Persisted blocks are strict value objects that
serialize to validated JSON; they are never an untyped script blob.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.enums import (
    ContentType,
    DeliveryIntent,
    NarrationNeed,
    NarrationPurpose,
    PlanBlockType,
    PlanSemanticOutcome,
    PlanStatus,
    SourceExcerptRole,
    SourceMomentStructure,
    StrategyOrigin,
    SubstantiveValueKind,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.transformation.planning.policy import (
    CONTEXT_POLICY_VERSION,
    DEFAULT_OUTPUT_LANGUAGE_POLICY,
    DEFAULT_REGISTER_INTENT,
    DEFAULT_TARGET_MARKET,
)
from app.transformation.types import clamp


@dataclass(frozen=True)
class PlanningContext:
    """Typed target/audience/narration semantic context (no channel schema).

    Neutral production defaults apply until future channel configuration
    supplies target-market semantics through the resolver seam. This never
    changes source transcript, quoted speech, or source dialect.
    """

    target_market: str = DEFAULT_TARGET_MARKET
    output_language_policy: str = DEFAULT_OUTPUT_LANGUAGE_POLICY
    register_intent: str = DEFAULT_REGISTER_INTENT
    narration_allowed: bool = True
    context_policy_version: str = CONTEXT_POLICY_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "target_market": self.target_market,
            "output_language_policy": self.output_language_policy,
            "register_intent": self.register_intent,
            "narration_allowed": self.narration_allowed,
            "context_policy_version": self.context_policy_version,
        }

    def semantic_payload(self) -> dict[str, object]:
        """Only planning-semantic fields participate in fingerprints."""

        return {
            "target_market": self.target_market,
            "output_language_policy": self.output_language_policy,
            "register_intent": self.register_intent,
            "narration_allowed": self.narration_allowed,
            "context_policy_version": self.context_policy_version,
        }


@dataclass(frozen=True)
class WordEvidence:
    """One indexed word with source-time bounds from Stage 3.5 evidence."""

    index: int
    text: str
    start: float
    end: float
    probability: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "probability": self.probability,
        }


@dataclass(frozen=True)
class NarrationRequirement:
    """Abstract narration semantic requirement. Never a TTS decision."""

    need: NarrationNeed = NarrationNeed.NONE
    purposes: tuple[NarrationPurpose, ...] = ()
    language: str | None = None
    register: str | None = None
    estimated_duration: float = 0.0
    placement_block_index: int | None = None
    max_source_interruption_seconds: float = 0.0
    overlaps_source_audio: bool = False
    replaces_silence: bool = False
    essential: bool = False
    verification_dependency_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "need": self.need.value,
            "purposes": [item.value for item in self.purposes],
            "language": self.language,
            "register": self.register,
            "estimated_duration": round(float(self.estimated_duration), 3),
            "placement_block_index": self.placement_block_index,
            "max_source_interruption_seconds": round(
                float(self.max_source_interruption_seconds), 3
            ),
            "overlaps_source_audio": self.overlaps_source_audio,
            "replaces_silence": self.replaces_silence,
            "essential": self.essential,
            "verification_dependency_ids": list(self.verification_dependency_ids),
        }

    @property
    def is_none(self) -> bool:
        return self.need is NarrationNeed.NONE


_EMPTY_NARRATION = NarrationRequirement()


@dataclass(frozen=True)
class PlanBlock:
    """One validated structured plan block."""

    index: int
    block_type: PlanBlockType
    purpose: str = ""
    estimated_duration: float = 0.0
    placement: str = ""
    interrupts_source: bool = False
    preservation_constraints: tuple[str, ...] = ()
    dependency_ids: tuple[str, ...] = ()
    # SOURCE_EXCERPT fields.
    source_role: SourceExcerptRole | None = None
    word_start_index: int | None = None
    word_end_index: int | None = None
    source_start: float | None = None
    source_end: float | None = None
    source_text: str | None = None
    continuity_rationale: str | None = None
    # ORIGINAL_VALUE / TEXTUAL_ANNOTATION fields.
    substantive_value_kind: SubstantiveValueKind | None = None
    semantic_intent: str | None = None
    why_unavailable: str | None = None
    grounding_refs: tuple[str, ...] = ()
    delivery_intent: DeliveryIntent | None = None
    draft_line: str | None = None
    draft_only: bool = False
    # FACT_VERIFICATION_PLACEHOLDER fields.
    claim_dependency: str | None = None
    verification_rationale: str | None = None
    intended_use: str | None = None
    must_verify_before_execution: bool = False
    dependent_block_ids: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "index": self.index,
            "block_type": self.block_type.value,
            "purpose": self.purpose,
            "estimated_duration": round(float(self.estimated_duration), 3),
            "placement": self.placement,
            "interrupts_source": self.interrupts_source,
            "preservation_constraints": list(self.preservation_constraints),
            "dependency_ids": list(self.dependency_ids),
        }
        if self.block_type is PlanBlockType.SOURCE_EXCERPT:
            payload.update(
                {
                    "source_role": self.source_role.value if self.source_role else None,
                    "word_start_index": self.word_start_index,
                    "word_end_index": self.word_end_index,
                    "source_start": _round_opt(self.source_start),
                    "source_end": _round_opt(self.source_end),
                    "source_text": self.source_text,
                    "continuity_rationale": self.continuity_rationale,
                }
            )
        if self.block_type in {
            PlanBlockType.ORIGINAL_VALUE,
            PlanBlockType.TEXTUAL_ANNOTATION,
        }:
            payload.update(
                {
                    "substantive_value_kind": (
                        self.substantive_value_kind.value if self.substantive_value_kind else None
                    ),
                    "semantic_intent": self.semantic_intent,
                    "why_unavailable": self.why_unavailable,
                    "grounding_refs": list(self.grounding_refs),
                    "delivery_intent": (
                        self.delivery_intent.value if self.delivery_intent else None
                    ),
                    "draft_line": self.draft_line,
                    "draft_only": self.draft_only,
                }
            )
        if self.block_type is PlanBlockType.FACT_VERIFICATION_PLACEHOLDER:
            payload.update(
                {
                    "claim_dependency": self.claim_dependency,
                    "verification_rationale": self.verification_rationale,
                    "intended_use": self.intended_use,
                    "must_verify_before_execution": self.must_verify_before_execution,
                    "dependent_block_ids": list(self.dependent_block_ids),
                }
            )
        return payload

    @property
    def is_source(self) -> bool:
        return self.block_type is PlanBlockType.SOURCE_EXCERPT

    @property
    def is_substantive(self) -> bool:
        return self.block_type in {
            PlanBlockType.ORIGINAL_VALUE,
            PlanBlockType.TEXTUAL_ANNOTATION,
        }


def _round_opt(value: float | None) -> float | None:
    return round(float(value), 3) if value is not None else None


@dataclass(frozen=True)
class PlanProviderBlock:
    """One provider-proposed block before deterministic source resolution."""

    block_type: PlanBlockType
    purpose: str = ""
    estimated_duration: float = 0.0
    interrupts_source: bool = False
    preservation_constraints: tuple[str, ...] = ()
    dependency_ids: tuple[str, ...] = ()
    source_role: SourceExcerptRole | None = None
    word_start_index: int | None = None
    word_end_index: int | None = None
    use_full_window: bool = False
    continuity_rationale: str | None = None
    substantive_value_kind: SubstantiveValueKind | None = None
    semantic_intent: str | None = None
    why_unavailable: str | None = None
    grounding_refs: tuple[str, ...] = ()
    delivery_intent: DeliveryIntent | None = None
    draft_line: str | None = None
    claim_dependency: str | None = None
    verification_rationale: str | None = None
    intended_use: str | None = None
    must_verify_before_execution: bool = False
    dependent_block_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlanProviderPlan:
    """One strict parsed provider plan for a single requested strategy."""

    strategy_id: str
    strategy_key: str
    confidence: float = 0.0
    no_valid_plan: bool = False
    no_valid_reason: str = ""
    blocks: tuple[PlanProviderBlock, ...] = ()
    narration: NarrationRequirement = field(default_factory=lambda: _EMPTY_NARRATION)
    preservation_constraints: tuple[str, ...] = ()
    planner_notes: str = ""


@dataclass(frozen=True)
class PlanProviderResult:
    """Parsed provider output for one candidate's requested strategies."""

    plans: tuple[PlanProviderPlan, ...] = ()
    notes: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class ValidatedPlan:
    """One fully deterministic-validated concrete plan, ready for persistence."""

    strategy_id: str
    strategy_key: str
    strategy_type: TransformationStrategyType
    strategy_rank: int
    intensity: TransformationIntensity
    strategy_fingerprint: str
    plan_key: str
    status: PlanStatus
    generation_rank: int
    blocks: tuple[PlanBlock, ...]
    hero_block_index: int
    hero_source_start: float
    hero_source_end: float
    hero_appearance_time: float
    preservation_constraints: tuple[str, ...]
    original_value_kinds: tuple[SubstantiveValueKind, ...]
    original_value_reasons: tuple[str, ...]
    narration: NarrationRequirement
    external_fact_dependencies: tuple[dict[str, object], ...]
    required_context: tuple[str, ...]
    derived_durations: Mapping[str, object]
    hook_payoff_evidence: Mapping[str, object]
    degraded_rules: tuple[str, ...]
    stage40_risk: Mapping[str, object]
    source_dialect: Mapping[str, object]
    target_intent: Mapping[str, object]
    planner_confidence: float
    generation_origin: StrategyOrigin
    provider_evidence: Mapping[str, object]
    provider_input_fingerprint: str
    plan_output_fingerprint: str
    structure_signature: str

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_key": self.strategy_key,
            "strategy_type": self.strategy_type.value,
            "strategy_rank": self.strategy_rank,
            "intensity": self.intensity.value,
            "strategy_fingerprint": self.strategy_fingerprint,
            "plan_key": self.plan_key,
            "status": self.status.value,
            "generation_rank": self.generation_rank,
            "blocks": [block.as_dict() for block in self.blocks],
            "hero_block_index": self.hero_block_index,
            "hero_source_start": round(self.hero_source_start, 3),
            "hero_source_end": round(self.hero_source_end, 3),
            "hero_appearance_time": round(self.hero_appearance_time, 3),
            "preservation_constraints": list(self.preservation_constraints),
            "original_value_kinds": [item.value for item in self.original_value_kinds],
            "original_value_reasons": list(self.original_value_reasons),
            "narration": self.narration.as_dict(),
            "external_fact_dependencies": [dict(item) for item in self.external_fact_dependencies],
            "required_context": list(self.required_context),
            "derived_durations": dict(self.derived_durations),
            "hook_payoff_evidence": dict(self.hook_payoff_evidence),
            "degraded_rules": list(self.degraded_rules),
            "stage40_risk": dict(self.stage40_risk),
            "source_dialect": dict(self.source_dialect),
            "target_intent": dict(self.target_intent),
            "planner_confidence": round(clamp(self.planner_confidence), 4),
            "generation_origin": self.generation_origin.value,
            "provider_evidence": dict(self.provider_evidence),
            "provider_input_fingerprint": self.provider_input_fingerprint,
            "plan_output_fingerprint": self.plan_output_fingerprint,
            "structure_signature": self.structure_signature,
        }


@dataclass(frozen=True)
class StrategyAttempt:
    """Bounded per-strategy attempt/checkpoint state for a plan set."""

    strategy_id: str
    strategy_key: str
    status: str
    reasons: tuple[str, ...] = ()
    provider_input_fingerprint: str = ""
    checkpoint: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_key": self.strategy_key,
            "status": self.status,
            "reasons": list(self.reasons),
            "provider_input_fingerprint": self.provider_input_fingerprint,
            "checkpoint": dict(self.checkpoint) if self.checkpoint is not None else None,
        }


@dataclass(frozen=True)
class PlanningOutcome:
    """Complete deterministic+provider Stage 4.1 planning outcome."""

    execution_status: str
    semantic_outcome: PlanSemanticOutcome
    outcome_reasons: tuple[str, ...]
    plans: tuple[ValidatedPlan, ...]
    attempts: tuple[StrategyAttempt, ...]
    provider_status: str
    provider_evidence: Mapping[str, object]
    provider_mode: str
    provider_identity: Mapping[str, object]
    input_fingerprint: str
    output_fingerprint: str
    cache_eligible: bool
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class PlanningInputs:
    """Immutable, bounded Stage 4.1 input evidence assembled from Stage 4.0."""

    candidate_id: str
    candidate_key: str
    source_id: str
    content_type: ContentType
    source_moment_structure: SourceMomentStructure
    source_moment: Mapping[str, object]
    transcript: str
    transcript_confidence: float
    refined_start: float
    refined_end: float
    words: tuple[WordEvidence, ...]
    word_coverage_sufficient: bool
    context_segments: tuple[str, ...]
    dialect_profile: str | None
    dialect_confidence: float
    code_switch: Mapping[str, object]
    entities: tuple[Mapping[str, object], ...]
    unresolved_spans: tuple[Mapping[str, object], ...]
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
    stage40_strategies: tuple[Mapping[str, object], ...]
    planning_context: PlanningContext
    refinement_priority: str
    refinement_quality_level: str
    refinement_status: str
    refinement_output_fingerprint: str

    @property
    def duration(self) -> float:
        return max(0.0, self.refined_end - self.refined_start)


def word_evidence_from_mapping(item: Mapping[str, object], index: int) -> WordEvidence | None:
    text = item.get("text", item.get("word", ""))
    if not isinstance(text, str) or not text.strip():
        return None
    start = item.get("start")
    end = item.get("end")
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return None
    if isinstance(end, bool) or not isinstance(end, (int, float)):
        return None
    if float(end) < float(start):
        return None
    probability = item.get("probability")
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        probability = None
    return WordEvidence(
        index=index,
        text=text.strip(),
        start=float(start),
        end=float(end),
        probability=float(probability) if probability is not None else None,
    )


def coerce_enum(value: object, enum_type: type[Any]) -> Any | None:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            return None
    return None


def positive_durations(values: Sequence[float]) -> float:
    return sum(max(0.0, float(value)) for value in values)
