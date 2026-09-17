"""Shared Stage 4.2 test helpers: settings injection and fake providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from stage41_support import (
    FakeHeavyModelLeaseFactory,
    FakePlanningProvider,
    FakeStage41Settings,
    install_stage41_settings,
)

from app.core.enums import (
    CoherenceFinding,
    GovernanceLevel,
    NarrationBurdenFinding,
    RetentionEffectFinding,
    SemanticFidelityFinding,
    SubstantiveValueFinding,
    UnsupportedClaimFinding,
)
from app.transformation.governance.policy import DEFAULT_CONFIG as STAGE42_CONFIG
from app.transformation.governance.policy import Stage42Config
from app.transformation.governance.types import (
    GovernanceInputs,
    PlanEvidence,
    ProviderCritique,
    ProviderGovernanceResult,
)
from app.transformation.planning.policy import DEFAULT_CONFIG as STAGE41_CONFIG
from app.transformation.planning.policy import Stage41Config
from app.transformation.planning.types import PlanningContext


class FakeGovernanceProvider:
    """Deterministic fake governance provider that never touches the network."""

    provider_name = "fake-governance"

    def __init__(
        self,
        critiques: Mapping[str, ProviderCritique] | None = None,
        *,
        auto: bool = False,
        hosted: bool = True,
        model: str = "fake-governance-model",
    ) -> None:
        self.model = model
        self.hosted_provider = hosted
        self.calls = 0
        self.tiers: list[str] = []
        self.behavior = "ok"
        self.released = 0
        self._critiques = dict(critiques or {})
        self._auto = auto

    def govern(self, request: object, tier: str = "ROUTINE") -> ProviderGovernanceResult:
        self.calls += 1
        self.tiers.append(tier)
        if self.behavior == "rate_limited":
            from app.transformation.governance.providers import GovernanceProviderError

            raise GovernanceProviderError("RATE_LIMITED")
        if self.behavior == "outage":
            from app.transformation.governance.providers import GovernanceProviderError

            raise GovernanceProviderError("PROVIDER_ERROR")
        if self.behavior == "malformed":
            from app.candidates.providers import ProviderErrorCategory
            from app.transformation.governance.providers import GovernanceProviderError

            raise GovernanceProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        plan_ids = [str(item.get("plan_id")) for item in request.plans]  # type: ignore[attr-defined]
        found: list[ProviderCritique] = []
        for plan_id in plan_ids:
            if plan_id in self._critiques:
                found.append(self._critiques[plan_id])
            elif self._auto:
                fingerprint = next(
                    str(item.get("plan_output_fingerprint"))
                    for item in request.plans  # type: ignore[attr-defined]
                    if str(item.get("plan_id")) == plan_id
                )
                found.append(
                    ProviderCritique(
                        plan_id=plan_id,
                        plan_output_fingerprint=fingerprint,
                        fidelity=SemanticFidelityFinding.PASS,
                        value=SubstantiveValueFinding.DISTINCT,
                        retention=RetentionEffectFinding.PRESERVED,
                        coherence=CoherenceFinding.COHERENT,
                        narration=NarrationBurdenFinding.APPROPRIATE,
                        template_feel=GovernanceLevel.LOW,
                        unsupported_claim=UnsupportedClaimFinding.NONE,
                        finding_codes=(),
                        block_indexes=(),
                        summary="bounded review",
                        confidence=0.7,
                    )
                )
        return ProviderGovernanceResult(critiques=tuple(found))

    def release(self) -> None:
        self.released += 1
        return None

    def raw_call_count(self) -> int:
        return self.calls

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "fake-governance", "model": self.model}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


class FakeGovernanceSettings(FakeStage41Settings):
    """Duck-typed settings covering Stage 4.0/4.1 seeding plus Stage 4.2."""

    def __init__(
        self,
        *,
        planning_provider: object | None = None,
        governance_provider: object | None = None,
        provider_identity: dict[str, object] | None = None,
        planning_mode: str = "adaptive",
        governance_mode: str = "adaptive",
        stage41_config: Stage41Config = STAGE41_CONFIG,
        stage42_config: Stage42Config = STAGE42_CONFIG,
        admission: object | None = None,
        lease_factory: object | None = None,
    ) -> None:
        super().__init__(
            provider=planning_provider,
            provider_identity=provider_identity,
            config=stage41_config,
            admission=admission,
            lease_factory=lease_factory,
        )
        from app.core.enums import SemanticProviderMode

        self._planning_mode = SemanticProviderMode(planning_mode)
        self._governance_mode = SemanticProviderMode(governance_mode)
        self._governance_provider = governance_provider
        self._stage42_config = stage42_config

    # Stage 4.0/4.1 mode overrides.
    def transformation_semantic_mode(self) -> object:
        return self._planning_mode

    def transformation_planning_semantic_mode(self) -> object:
        return self._planning_mode

    # Stage 4.2 accessors.
    def stage42_config(self) -> Stage42Config:
        return self._stage42_config

    def transformation_governance_semantic_mode(self) -> object:
        return self._governance_mode

    def transformation_governance_provider(self) -> object | None:
        return self._governance_provider

    def transformation_governance_provider_identity(self) -> dict[str, object]:
        if self._governance_provider is not None:
            identity = getattr(self._governance_provider, "runtime_identity", None)
            if callable(identity):
                return dict(identity())
        from app.transformation.governance.providers import (
            DeterministicGovernanceProvider,
        )

        return DeterministicGovernanceProvider().runtime_identity()


def install_stage42_settings(monkeypatch: Any, settings: FakeGovernanceSettings) -> None:
    install_stage41_settings(monkeypatch, settings)
    monkeypatch.setattr("app.transformation.governance.queue.get_settings", lambda: settings)
    monkeypatch.setattr("app.transformation.governance.handoff.get_settings", lambda: settings)


def with_review_narration(plan: Any) -> Any:
    """Attach a Stage 4.1-valid RECOMMENDED narration requirement to a plan."""

    from dataclasses import replace

    from app.core.enums import NarrationNeed, NarrationPurpose
    from app.transformation.planning.types import NarrationRequirement

    return replace(
        plan,
        narration=NarrationRequirement(
            need=NarrationNeed.RECOMMENDED,
            purposes=(NarrationPurpose.EXPLANATION,),
            estimated_duration=4.0,
            placement_block_index=1,
        ),
    )


def make_critique(
    plan: PlanEvidence,
    *,
    fidelity: str = "PASS",
    value: str = "DISTINCT",
    retention: str = "PRESERVED",
    coherence: str = "COHERENT",
    narration: str = "APPROPRIATE",
    template_feel: str = "LOW",
    unsupported_claim: str = "NONE",
    finding_codes: Sequence[str] = (),
    summary: str = "bounded review",
    confidence: float = 0.7,
) -> ProviderCritique:
    return ProviderCritique(
        plan_id=plan.plan_id,
        plan_output_fingerprint=plan.plan_output_fingerprint,
        fidelity=SemanticFidelityFinding(fidelity),
        value=SubstantiveValueFinding(value),
        retention=RetentionEffectFinding(retention),
        coherence=CoherenceFinding(coherence),
        narration=NarrationBurdenFinding(narration),
        template_feel=GovernanceLevel(template_feel),
        unsupported_claim=UnsupportedClaimFinding(unsupported_claim),
        finding_codes=tuple(finding_codes),
        block_indexes=(),
        summary=summary,
        confidence=confidence,
    )


def source_block(
    index: int,
    *,
    role: str = "HERO",
    duration: float = 6.0,
    start: float = 21.0,
    interrupts: bool = False,
) -> dict[str, object]:
    return {
        "index": index,
        "block_type": "SOURCE_EXCERPT",
        "purpose": "source moment",
        "estimated_duration": duration,
        "placement": "sequential",
        "interrupts_source": interrupts,
        "preservation_constraints": [],
        "dependency_ids": [],
        "source_role": role,
        "word_start_index": 0,
        "word_end_index": 10,
        "source_start": start,
        "source_end": start + duration,
        "source_text": "the source speaker said something",
        "continuity_rationale": "keeps context",
    }


def original_block(
    index: int,
    *,
    intent: str = "Explain the promotion-rate drop as evidence for the remote-work debate",
    kind: str = "SOURCE_AS_EVIDENCE",
    duration: float = 5.0,
    interrupts: bool = False,
    grounding: Sequence[str] = ("strategy",),
    dependency_ids: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "index": index,
        "block_type": "ORIGINAL_VALUE",
        "purpose": "authored contribution",
        "estimated_duration": duration,
        "placement": "sequential",
        "interrupts_source": interrupts,
        "preservation_constraints": [],
        "dependency_ids": list(dependency_ids),
        "substantive_value_kind": kind,
        "semantic_intent": intent,
        "why_unavailable": "the raw excerpt states the fact but not why it matters",
        "grounding_refs": list(grounding),
        "delivery_intent": "NARRATION",
        "draft_line": None,
        "draft_only": False,
    }


def verification_block(
    index: int,
    *,
    claim_id: str = "claim-1",
    dependent: Sequence[int] = (),
) -> dict[str, object]:
    return {
        "index": index,
        "block_type": "FACT_VERIFICATION_PLACEHOLDER",
        "purpose": "verification placeholder",
        "estimated_duration": 0.0,
        "placement": "inline",
        "interrupts_source": False,
        "preservation_constraints": [],
        "dependency_ids": [],
        "claim_dependency": claim_id,
        "verification_rationale": "external fact required",
        "intended_use": "support the claim",
        "must_verify_before_execution": True,
        "dependent_block_ids": list(dependent),
    }


def make_plan(
    *,
    plan_id: str = "00000000-0000-0000-0000-000000000001",
    fingerprint: str = "plan-fingerprint-1",
    strategy_type: str = "SOURCE_AS_EVIDENCE",
    intensity: str = "MODERATE",
    blocks: Sequence[Mapping[str, object]] | None = None,
    hero_index: int = 0,
    narration_need: str = "NONE",
    narration: Mapping[str, object] | None = None,
    strategy_rank: int = 1,
    original_value_kinds: Sequence[str] = ("SOURCE_AS_EVIDENCE",),
    verification_dependencies: Sequence[Mapping[str, object]] = (),
    stage40_risk: Mapping[str, object] | None = None,
    strategy_snapshot: Mapping[str, object] | None = None,
    is_current: bool = True,
    generation_rank: int = 1,
) -> PlanEvidence:
    chosen_blocks = list(blocks) if blocks is not None else [source_block(0), original_block(1)]
    hero_start = None
    hero_end = None
    hero_appearance = 0.0
    for block in chosen_blocks[:hero_index]:
        if block.get("block_type") == "SOURCE_EXCERPT":
            pass
        duration = block.get("estimated_duration", 0.0)
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            hero_appearance += float(duration)
    hero = chosen_blocks[hero_index] if 0 <= hero_index < len(chosen_blocks) else {}
    if isinstance(hero.get("source_start"), (int, float)):
        hero_start = float(hero["source_start"])
    if isinstance(hero.get("source_end"), (int, float)):
        hero_end = float(hero["source_end"])
    requirements = dict(narration) if narration is not None else {}
    return PlanEvidence(
        plan_id=plan_id,
        plan_key=f"plan-key-{plan_id}",
        plan_output_fingerprint=fingerprint,
        provider_input_fingerprint=f"provider-fp-{plan_id}",
        strategy_id=f"strategy-{plan_id}",
        strategy_key=f"strategy-key-{plan_id}",
        strategy_type=strategy_type,
        strategy_fingerprint=f"strategy-fp-{plan_id}",
        strategy_rank=strategy_rank,
        intensity=intensity,
        status="PLAN_GENERATED",
        generation_rank=generation_rank,
        is_current=is_current,
        planner_confidence=0.7,
        blocks=tuple(dict(block) for block in chosen_blocks),
        hero_block_index=hero_index,
        hero_source_start=hero_start,
        hero_source_end=hero_end,
        hero_appearance_time=round(hero_appearance, 3),
        narration={"need": narration_need, "requirements": requirements},
        verification_dependencies=tuple(dict(item) for item in verification_dependencies),
        derived_durations={},
        hook_payoff_evidence={},
        original_value_kinds=tuple(original_value_kinds),
        original_value_reasons=("grounded in the source",),
        preservation_constraints=("keep the hero moment intact",),
        required_context=("remote work debate",),
        degraded_rules=(),
        stage40_risk=dict(stage40_risk or {}),
        source_dialect={"profile": "EGYPTIAN", "confidence": 0.85},
        target_audience={"target_market": "UNSPECIFIED"},
        strategy_snapshot=dict(strategy_snapshot or {}),
    )


def make_inputs(
    plans: Sequence[PlanEvidence],
    *,
    transcript: str = (
        "The guest argues that remote work collapsed productivity because managers lost "
        "the ability to mentor junior staff and promotion rates fell sharply."
    ),
    source_moment_structure: str = "CLAIM",
    rights_risk: str = "LOW",
    originality_risk: str = "NOT_INDICATED",
    rights_status: str = "OWNED",
    media_origin: str = "PODCAST_INTERVIEW",
    stage40_assessments: Mapping[str, object] | None = None,
    context_segments: Sequence[str] = ("prior context", "following context"),
) -> GovernanceInputs:
    return GovernanceInputs(
        candidate_id="00000000-0000-0000-0000-0000000000aa",
        candidate_key="candidate-key",
        source_id="00000000-0000-0000-0000-0000000000bb",
        disposition="CANDIDATE",
        content_type="INTERVIEW_INSIGHT",
        source_moment_structure=source_moment_structure,
        transcript=transcript,
        transcript_confidence=0.9,
        refined_start=20.5,
        refined_end=44.5,
        context_segments=tuple(context_segments),
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.85,
        code_switch={},
        idea_summary="Remote work hurts junior mentorship",
        topic_summary="future of remote work",
        hooks=({"type": "DIRECT_CLAIM", "text": "Remote work collapse"},),
        rights_risk=rights_risk,
        originality_risk=originality_risk,
        rights_status=rights_status,
        media_origin=media_origin,
        provenance_snapshot={"source": "operator"},
        stage3_risk={"clip_score": 0.8},
        stage40_assessments=dict(stage40_assessments or {"moment_density": 0.65}),
        stage40_platform_risk={"transformation_required": False},
        stage40_analysis_id="00000000-0000-0000-0000-0000000000cc",
        stage40_input_fingerprint="stage40-input-fp",
        stage40_output_fingerprint="stage40-output-fp",
        stage40_policy_version="stage4.0-v1",
        plan_set_id="00000000-0000-0000-0000-0000000000dd",
        plan_set_input_fingerprint="stage41-input-fp",
        plan_set_output_fingerprint="stage41-output-fp",
        plan_set_semantic_outcome="PLANS_GENERATED",
        plan_set_provider_identity={"provider": "gemini"},
        plan_set_stage40_snapshot={"analysis_id": "00000000-0000-0000-0000-0000000000cc"},
        refinement_id="00000000-0000-0000-0000-0000000000ee",
        refinement_priority="CANDIDATE",
        refinement_quality_level="CANDIDATE",
        refinement_status="CANDIDATE_REFINED",
        refinement_output_fingerprint="refinement-output-fp",
        target_context=PlanningContext(),
        plans=tuple(plans),
    )


__all__ = [
    "FakeGovernanceProvider",
    "FakeGovernanceSettings",
    "FakeHeavyModelLeaseFactory",
    "FakePlanningProvider",
    "STAGE41_CONFIG",
    "STAGE42_CONFIG",
    "install_stage42_settings",
    "make_critique",
    "make_inputs",
    "make_plan",
    "original_block",
    "source_block",
    "verification_block",
    "with_review_narration",
]
