"""Shared Stage 4.1 test helpers: settings injection and a fake planning provider.

Keeps every Stage 4.1 test hermetic: queue/handoff/executor are always driven
through an injected settings object and a fake provider, so a Gemini key present
in the environment can never trigger live work.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

from app.core.enums import (
    ContentType,
    ExternalFactRequirement,
    PlanBlockType,
    PlanSemanticOutcome,
    SemanticProviderMode,
    SourceExcerptRole,
    StrategyDisposition,
    SubstantiveValueKind,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.runtime.heavy_model_lease import NoopHeavyModelLeaseFactory
from app.transformation.planning.policy import DEFAULT_CONFIG, Stage41Config
from app.transformation.planning.types import (
    PlanProviderBlock,
    PlanProviderPlan,
    PlanProviderResult,
)
from app.transformation.policy import DEFAULT_CONFIG as STAGE40_CONFIG


class FakeHeavyModelLease:
    """Tracks entry/exit and refuses a second concurrent acquisition."""

    def __init__(self, factory: "FakeHeavyModelLeaseFactory") -> None:
        self._factory = factory
        self.active = False
        self.ownership_lost = False

    def __enter__(self) -> "FakeHeavyModelLease":
        if self._factory.held:
            self._factory.blocked += 1
            raise RuntimeError("heavy-model lease already held")
        self._factory.held = True
        self.active = True
        self._factory.entered += 1
        return self

    def __exit__(self, *exc: object) -> bool:
        if self.active:
            self.active = False
            self._factory.held = False
            self._factory.exited += 1
        return False


class FakeHeavyModelLeaseFactory:
    def __init__(self) -> None:
        self.held = False
        self.acquired = 0
        self.entered = 0
        self.exited = 0
        self.blocked = 0

    def acquire(self, purpose: str = "ollama") -> FakeHeavyModelLease:
        self.acquired += 1
        return FakeHeavyModelLease(self)


class FakeStage41Settings:
    """Minimal duck-typed settings for Stage 4.1 executor/queue/handoff."""

    def __init__(
        self,
        *,
        provider: object | None = None,
        provider_identity: dict[str, object] | None = None,
        mode: SemanticProviderMode = SemanticProviderMode.ADAPTIVE,
        config: Stage41Config = DEFAULT_CONFIG,
        admission: object | None = None,
        lease_factory: object | None = None,
    ) -> None:
        self._provider = provider
        self._provider_identity = dict(provider_identity) if provider_identity is not None else None
        self._mode = mode
        self._config = config
        self._admission = admission
        self._lease_factory = lease_factory

    def stage41_config(self) -> Stage41Config:
        return self._config

    def stage40_config(self) -> object:
        return STAGE40_CONFIG

    # Stage 4.0 executor compatibility so the handoff freshness gate can run.
    def transformation_semantic_mode(self) -> SemanticProviderMode:
        return self._mode

    def transformation_provider(self) -> object | None:
        return None

    def transformation_provider_identity(self) -> dict[str, object]:
        from app.transformation.providers import DeterministicTransformationProvider

        return DeterministicTransformationProvider().runtime_identity()

    def transformation_planning_semantic_mode(self) -> SemanticProviderMode:
        return self._mode

    def transformation_planning_provider(self) -> object | None:
        return self._provider

    def transformation_planning_provider_identity(self) -> dict[str, object]:
        if self._provider_identity is not None:
            return dict(self._provider_identity)
        if self._provider is not None:
            identity = getattr(self._provider, "runtime_identity", None)
            if callable(identity):
                return dict(identity())
        from app.transformation.planning.providers import DeterministicPlanningProvider

        return DeterministicPlanningProvider().runtime_identity()

    def heavy_model_lease_factory(self) -> object:
        return self._lease_factory or NoopHeavyModelLeaseFactory()

    def gemini_admission_controller(self) -> object | None:
        return self._admission


class FakePlanningProvider:
    """Deterministic fake planning provider that never touches the network."""

    provider_name = "fake"

    def __init__(
        self, plans: Sequence[PlanProviderPlan] = (), *, model: str = "fake-model"
    ) -> None:
        self.model = model
        self.calls = 0
        self.tiers: list[str] = []
        self.behavior = "ok"
        self.released = 0
        self._plans = tuple(plans)

    def plan(self, requests: Sequence[Any], tier: str = "ROUTINE") -> dict[str, PlanProviderResult]:
        self.calls += 1
        self.tiers.append(tier)
        if self.behavior in {"rate_limited", "outage"}:
            from app.transformation.planning.providers import PlanningProviderError

            category = "RATE_LIMITED" if self.behavior == "rate_limited" else "PROVIDER_ERROR"
            raise PlanningProviderError(category)
        if self.behavior == "malformed":
            from app.candidates.providers import ProviderErrorCategory
            from app.transformation.planning.providers import PlanningProviderError

            raise PlanningProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        results: dict[str, PlanProviderResult] = {}
        by_key = {plan.strategy_key: plan for plan in self._plans}
        for request in requests:
            plan = by_key.get(request.strategy_key)
            if plan is None:
                continue
            results[request.strategy_key] = PlanProviderResult(
                plans=(plan,), confidence=plan.confidence
            )
        return results

    def release(self) -> None:
        self.released += 1
        return None

    def raw_call_count(self) -> int:
        return self.calls

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "fake", "model": self.model}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


def make_source_value_plan(
    strategy_id: str,
    strategy_key: str,
    *,
    kind: SubstantiveValueKind = SubstantiveValueKind.SOURCE_AS_EVIDENCE,
    intent: str = "Frame the promotion-rate drop as evidence for the remote-work debate",
    why: str = "The raw excerpt states the drop but not why it matters for the debate",
    use_full_window: bool = True,
    hero_role: SourceExcerptRole = SourceExcerptRole.HERO,
    narration_need: str = "NONE",
    confidence: float = 0.7,
) -> PlanProviderPlan:
    from app.core.enums import NarrationNeed
    from app.transformation.planning.types import NarrationRequirement

    blocks = [
        PlanProviderBlock(
            block_type=PlanBlockType.SOURCE_EXCERPT,
            purpose="Hero source moment",
            use_full_window=use_full_window,
            source_role=hero_role,
        ),
        PlanProviderBlock(
            block_type=PlanBlockType.ORIGINAL_VALUE,
            purpose="Substantive value after the source",
            estimated_duration=4.0,
            substantive_value_kind=kind,
            semantic_intent=intent,
            why_unavailable=why,
            grounding_refs=("strategy",),
        ),
    ]
    return PlanProviderPlan(
        strategy_id=strategy_id,
        strategy_key=strategy_key,
        confidence=confidence,
        blocks=tuple(blocks),
        narration=NarrationRequirement(need=NarrationNeed(narration_need)),
    )


def plan_status_outcome(outcome: PlanSemanticOutcome) -> str:
    return outcome.value


_SEED_COUNTER = itertools.count()

TRANSCRIPT = (
    "The guest argues that remote work collapsed productivity because managers lost "
    "the ability to mentor junior staff and promotion rates fell sharply."
)


def build_words(
    transcript: str = TRANSCRIPT, start: float = 21.0, span: float = 23.0
) -> list[dict[str, object]]:
    tokens = transcript.split()
    if not tokens:
        return []
    step = span / max(1, len(tokens))
    return [
        {
            "text": token,
            "start": round(start + index * step, 3),
            "end": round(start + (index + 1) * step, 3),
            "probability": 0.9,
        }
        for index, token in enumerate(tokens)
    ]


def seed_stage41(
    session: Any,
    *,
    content_type: ContentType = ContentType.INTERVIEW_INSIGHT,
    strategy_type: TransformationStrategyType = TransformationStrategyType.SOURCE_AS_EVIDENCE,
    external: ExternalFactRequirement = ExternalFactRequirement.NOT_REQUIRED,
    added_value_focus: str = (
        "Frame the promotion-rate drop as evidence for the remote-work debate"
    ),
    value_kind: SubstantiveValueKind = SubstantiveValueKind.SOURCE_AS_EVIDENCE,
    dialect_profile: str | None = "EGYPTIAN",
    transcript: str = TRANSCRIPT,
    refined: tuple[float, float] = (20.5, 44.5),
    words: list[dict[str, object]] | None = None,
    second_strategy: tuple[TransformationStrategyType, str] | None = None,
    settings: FakeStage41Settings | None = None,
) -> tuple[Any, Any, Any, Any, list[Any]]:
    """Create a source/candidate/refinement plus a current Stage 4.0 analysis."""

    from app.core.enums import (
        CandidateDisposition,
        OriginalityRisk,
        RefinementPriority,
        RefinementStatus,
        RightsRisk,
        TransformationEligibilityOutcome,
        TransformationExecutionStatus,
    )
    from app.models import (
        CandidateRefinement,
        ClipCandidate,
        SourceVideo,
        Transcript,
        TransformationEligibilityAnalysis,
        TransformationStrategyCandidate,
    )

    settings = settings or FakeStage41Settings(config=DEFAULT_CONFIG)
    seed_index = next(_SEED_COUNTER)
    source = SourceVideo(
        source_uri=f"/tmp/stage41-{seed_index}.mp4", content_hash=f"stage41-hash-{seed_index}"
    )
    session.add(source)
    session.flush()
    session.add(
        Transcript(
            source_video_id=source.id,
            whisper_model="small",
            transcription_options={},
            input_fingerprint="t" * 64,
            raw_text=transcript,
            normalized_text=transcript,
            corrected_text=transcript,
            final_text=transcript,
            segments=[
                {"start": 0.0, "end": 20.0, "text": "Intro"},
                {"start": 20.0, "end": 45.0, "text": transcript, "corrected_text": transcript},
                {"start": 45.0, "end": 90.0, "text": "Outro"},
            ],
            duration=90.0,
            language="ar",
        )
    )
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key=f"stage41-candidate-{seed_index}",
        disposition=CandidateDisposition.CANDIDATE,
        start_time=20.0,
        end_time=45.0,
        start_segment_index=1,
        end_segment_index=1,
        segment_indexes=[1],
        primary_content_type=content_type,
        idea_summary="Remote work hurts junior mentorship",
        topic_summary="future of remote work",
        hooks=[{"type": "DIRECT_CLAIM", "text": "Remote work collapse"}],
        clip_score=0.8,
        short_form_score=0.75,
        moment_density_score=0.65,
        ending_quality_score=0.7,
        loopability_score=0.5,
        rights_risk=RightsRisk.LOW,
        originality_risk=OriginalityRisk.NOT_INDICATED,
        dialect_profile=dialect_profile,
        dialect_confidence=0.85,
        analysis_fingerprint="stage3-analysis-fp",
        policy_version="stage3-v1",
    )
    session.add(candidate)
    session.flush()
    refinement = CandidateRefinement(
        source_video_id=source.id,
        clip_candidate_id=candidate.id,
        priority=RefinementPriority.CANDIDATE,
        status=RefinementStatus.CANDIDATE_REFINED,
        coarse_start=20.0,
        coarse_end=45.0,
        context_start=15.0,
        context_end=50.0,
        refined_start=refined[0],
        refined_end=refined[1],
        automatic_transcript=transcript,
        final_transcript=transcript,
        word_timestamps=words if words is not None else build_words(transcript),
        confidence=0.9,
        quality_level="CANDIDATE",
        dialect_profile=dialect_profile,
        dialect_confidence=0.85,
        output_fingerprint="refinement-output-fp",
    )
    session.add(refinement)
    session.flush()
    analysis = TransformationEligibilityAnalysis(
        source_video_id=source.id,
        clip_candidate_id=candidate.id,
        refinement_id=refinement.id,
        refinement_priority=refinement.priority.value,
        refinement_quality_level=refinement.quality_level,
        execution_status=TransformationExecutionStatus.COMPLETE,
        eligibility_outcome=TransformationEligibilityOutcome.ELIGIBLE_FOR_TRANSFORMATION,
        assessments={
            "retention_preservation": 0.7,
            "source_moment_damage_risk": 0.3,
            "added_value_density": 0.6,
            "transformation_necessity": 0.4,
            "transformation_potential": 0.6,
            "originality_potential": 0.65,
            "source_dominance_risk": 0.4,
            "generic_ai_filler_risk": 0.3,
            "redundant_commentary_risk": 0.3,
            "template_staleness_risk": 0.3,
        },
        source_moment={
            "structure": "CLAIM",
            "hook_index": 1,
            "payoff_index": None,
            "duration": refined[1] - refined[0],
            "moment_density": 0.65,
            "evidence": ["content_type"],
        },
        platform_risk={"transformation_required": False, "dimensions": []},
        policy_version="stage4.0-v1",
        output_fingerprint="stage40-output-fp",
        cache_eligible=True,
    )
    session.add(analysis)
    session.flush()
    strategies = [
        TransformationStrategyCandidate(
            analysis_id=analysis.id,
            strategy_key=f"{analysis.id}:{strategy_type.value}",
            strategy_type=strategy_type,
            is_current=True,
            disposition=StrategyDisposition.RECOMMENDED,
            rank=1,
            intensity=TransformationIntensity.MODERATE,
            direction_summary="Use the moment as evidence",
            added_value_focus=added_value_focus,
            substantive_value_kind=value_kind,
            source_moment_role="HERO",
            preservation_requirements=["Keep the strongest source moment as the hero."],
            external_verification_requirement=external,
            verification_requirements=(
                ["Verify the referenced fact before presenting it."]
                if external is ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION
                else []
            ),
            confidence=0.6,
            strategy_fingerprint="stage40-strategy-fp",
            policy_version="stage4.0-v1",
        )
    ]
    if second_strategy is not None:
        second_type, second_focus = second_strategy
        strategies.append(
            TransformationStrategyCandidate(
                analysis_id=analysis.id,
                strategy_key=f"{analysis.id}:{second_type.value}",
                strategy_type=second_type,
                is_current=True,
                disposition=StrategyDisposition.RECOMMENDED,
                rank=2,
                intensity=TransformationIntensity.MODERATE,
                direction_summary="Second direction",
                added_value_focus=second_focus,
                substantive_value_kind=SubstantiveValueKind.AUTHORED_THESIS,
                source_moment_role="HERO",
                preservation_requirements=["Keep the strongest source moment as the hero."],
                external_verification_requirement=ExternalFactRequirement.NOT_REQUIRED,
                verification_requirements=[],
                confidence=0.6,
                strategy_fingerprint="stage40-strategy-fp-2",
                policy_version="stage4.0-v1",
            )
        )
    session.add_all(strategies)
    session.flush()
    from app.transformation.executor import build_transformation_executor

    analysis.input_fingerprint = build_transformation_executor(session, settings).input_fingerprint(
        candidate
    )
    session.commit()
    return source, candidate, refinement, analysis, strategies


def run_planning(
    session: Any,
    settings: FakeStage41Settings,
    seed: tuple[Any, ...],
    provider: object | None = None,
    *,
    mode: SemanticProviderMode = SemanticProviderMode.ADAPTIVE,
) -> Any:
    """Run one planning executor pass over a seed and return the refreshed plan set."""

    from app.transformation.planning.executor import TransformationPlanningExecutor
    from app.transformation.planning.queue import get_or_create_plan_set

    _source, candidate, _refinement, analysis, _strategies = seed
    plan_set = get_or_create_plan_set(session, candidate, analysis)
    executor = TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=provider,  # type: ignore[arg-type]
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=mode,
        config=DEFAULT_CONFIG,
        lease_factory=settings.heavy_model_lease_factory(),  # type: ignore[arg-type]
    )
    executor.execute(plan_set.id)
    session.refresh(plan_set)
    return plan_set


def install_stage41_settings(monkeypatch: Any, settings: FakeStage41Settings) -> None:
    """Route queue/handoff/planning-input runtime identity through one settings."""

    monkeypatch.setattr("app.transformation.planning.queue.get_settings", lambda: settings)
    monkeypatch.setattr("app.transformation.planning.handoff.get_settings", lambda: settings)
    monkeypatch.setattr("app.transformation.handoff.get_settings", lambda: settings)
    monkeypatch.setattr("app.transformation.queue.get_settings", lambda: settings)


__all__ = [
    "FakeHeavyModelLease",
    "FakeHeavyModelLeaseFactory",
    "FakePlanningProvider",
    "FakeStage41Settings",
    "ContentType",
    "ExternalFactRequirement",
    "StrategyDisposition",
    "TransformationStrategyType",
    "TransformationIntensity",
    "install_stage41_settings",
    "make_source_value_plan",
    "plan_status_outcome",
    "build_words",
    "seed_stage41",
    "run_planning",
]
