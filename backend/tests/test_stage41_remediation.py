"""Stage 4.1 review-remediation regression tests (findings 1-10)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage41_support import (
    FakeHeavyModelLeaseFactory,
    FakePlanningProvider,
    FakeStage41Settings,
    install_stage41_settings,
    make_source_value_plan,
    run_planning,
    seed_stage41,
)

from app.core.enums import (
    ExternalFactRequirement,
    JobKind,
    JobStatus,
    NarrationNeed,
    NarrationPurpose,
    PlanBlockType,
    PlanSemanticOutcome,
    SemanticProviderMode,
    SourceExcerptRole,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.db.base import Base
from app.models import ProcessingJob, TransformationPlanSet
from app.transformation.planning.executor import TransformationPlanningExecutor
from app.transformation.planning.gemini import GeminiPlanningProvider
from app.transformation.planning.inputs import build_planning_inputs
from app.transformation.planning.policy import DEFAULT_CONFIG
from app.transformation.planning.providers import (
    PlanningProviderError,
    PlanningRequest,
)
from app.transformation.planning.queue import get_or_create_plan_set, list_plans
from app.transformation.planning.types import (
    NarrationRequirement,
    PlanProviderBlock,
    PlanProviderPlan,
)
from app.transformation.planning.validation import (
    REJECT_EVASION,
    REJECT_LATE_HERO,
    REJECT_NARRATION_CONTRADICTION,
    REJECT_OVERLAPPING_EXCERPT,
    REJECT_RENDERING_INSTRUCTION,
    REJECT_SHORT_EXCERPT,
    REJECT_TTS_SELECTION,
    REJECT_VERIFICATION_BLOCK_REFERENCE,
    REJECT_VERIFICATION_UNLINKED,
    validate_provider_plan,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _install(settings: FakeStage41Settings) -> None:
    from _pytest.monkeypatch import MonkeyPatch

    install_stage41_settings(MonkeyPatch(), settings)


def _seed(session: Session, settings: FakeStage41Settings, **kwargs: Any) -> tuple[Any, ...]:
    _install(settings)
    return seed_stage41(session, settings=settings, **kwargs)


def _inputs(session: Session, settings: FakeStage41Settings, **kwargs: Any):
    seed = _seed(session, settings, **kwargs)
    inputs = build_planning_inputs(
        session,
        seed[1],
        seed[3],
        seed[2],
        settings,
        DEFAULT_CONFIG,  # type: ignore[arg-type]
    )
    return seed, inputs


def _validate(inputs: Any, plan: PlanProviderPlan):
    return validate_provider_plan(
        plan,
        inputs.stage40_strategies[0],
        inputs,
        DEFAULT_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="fp",
    )


def _source(**overrides: Any) -> PlanProviderBlock:
    base: dict[str, Any] = {
        "block_type": PlanBlockType.SOURCE_EXCERPT,
        "use_full_window": True,
        "source_role": SourceExcerptRole.HERO,
    }
    base.update(overrides)
    return PlanProviderBlock(**base)


def _value(kind: SubstantiveValueKind, intent: str, **overrides: Any) -> PlanProviderBlock:
    base: dict[str, Any] = {
        "block_type": PlanBlockType.ORIGINAL_VALUE,
        "estimated_duration": 4.0,
        "substantive_value_kind": kind,
        "semantic_intent": intent,
        "why_unavailable": "The excerpt alone does not provide this added dimension",
    }
    base.update(overrides)
    return PlanProviderBlock(**base)


def _plan(key: str, blocks: list[PlanProviderBlock], narration: Any = None) -> PlanProviderPlan:
    return PlanProviderPlan(
        strategy_id="id",
        strategy_key=key,
        confidence=0.7,
        blocks=tuple(blocks),
        narration=narration or NarrationRequirement(need=NarrationNeed.NONE),
    )


# --- Finding 1: local-only heavy-model lease release -------------------------


def test_local_only_releases_lease_on_success(session: Session) -> None:
    factory = FakeHeavyModelLeaseFactory()
    settings = FakeStage41Settings(mode=SemanticProviderMode.LOCAL_ONLY, lease_factory=factory)
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [
            make_source_value_plan(
                str(strategy.id), strategy.strategy_key, kind=SubstantiveValueKind.AUTHORED_THESIS
            )
        ]
    )
    plan_set = run_planning(session, settings, seed, provider, mode=SemanticProviderMode.LOCAL_ONLY)
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANS_GENERATED
    assert factory.held is False
    assert factory.exited >= 1
    assert factory.blocked == 0
    # A subsequent heavy-model lease is not blocked by the released one.
    lease = factory.acquire(purpose="ollama")
    lease.__enter__()
    assert factory.held is True
    lease.__exit__(None, None, None)
    assert factory.blocked == 0


def test_local_only_releases_lease_on_provider_failure(session: Session) -> None:
    factory = FakeHeavyModelLeaseFactory()
    settings = FakeStage41Settings(mode=SemanticProviderMode.LOCAL_ONLY, lease_factory=factory)
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [
            make_source_value_plan(
                str(strategy.id), strategy.strategy_key, kind=SubstantiveValueKind.AUTHORED_THESIS
            )
        ]
    )
    provider.behavior = "outage"
    run_planning(session, settings, seed, provider, mode=SemanticProviderMode.LOCAL_ONLY)
    assert factory.held is False
    assert factory.exited >= 1


def test_local_only_releases_lease_on_cache_hit(session: Session) -> None:
    factory = FakeHeavyModelLeaseFactory()
    settings = FakeStage41Settings(mode=SemanticProviderMode.LOCAL_ONLY, lease_factory=factory)
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [
            make_source_value_plan(
                str(strategy.id), strategy.strategy_key, kind=SubstantiveValueKind.AUTHORED_THESIS
            )
        ]
    )
    plan_set = run_planning(session, settings, seed, provider, mode=SemanticProviderMode.LOCAL_ONLY)
    session.refresh(plan_set)
    run_planning(session, settings, seed, provider, mode=SemanticProviderMode.LOCAL_ONLY)
    assert factory.held is False
    assert provider.calls == 1


# --- Finding 2: duplicate Celery delivery fencing ----------------------------


def _job(
    session: Session, plan_set: TransformationPlanSet, status: JobStatus, **kwargs: Any
) -> ProcessingJob:
    job = ProcessingJob(
        source_video_id=plan_set.source_video_id,
        kind=JobKind.TRANSFORMATION_PLANNING,
        transformation_plan_set_id=plan_set.id,
        status=status,
        **kwargs,
    )
    session.add(job)
    session.commit()
    return job


def _executor(
    session: Session,
    settings: FakeStage41Settings,
    provider: object | None,
    *,
    mode: SemanticProviderMode = SemanticProviderMode.ADAPTIVE,
) -> TransformationPlanningExecutor:
    return TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=provider,  # type: ignore[arg-type]
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=mode,
        config=DEFAULT_CONFIG,
        lease_factory=settings.heavy_model_lease_factory(),  # type: ignore[arg-type]
    )


def test_duplicate_delivery_does_not_repeat_provider_work(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    plan_set = get_or_create_plan_set(session, seed[1], seed[3])
    job = _job(session, plan_set, JobStatus.QUEUED)
    plan = make_source_value_plan(
        str(strategy.id), strategy.strategy_key, kind=SubstantiveValueKind.AUTHORED_THESIS
    )
    first = FakePlanningProvider([plan])
    executor_a = _executor(session, settings, first)
    executor_a.set_active_job(job.id)
    executor_a.execute(plan_set.id)
    assert first.calls == 1

    second = FakePlanningProvider([plan])
    executor_b = _executor(session, settings, second)
    executor_b.set_active_job(job.id)
    executor_b.execute(plan_set.id)
    assert executor_b.skipped_duplicate is True
    assert second.calls == 0
    session.refresh(job)
    assert job.status is JobStatus.SUCCEEDED


def test_failed_job_can_be_retried(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    plan_set = get_or_create_plan_set(session, seed[1], seed[3])
    job = _job(session, plan_set, JobStatus.FAILED)
    provider = FakePlanningProvider(
        [
            make_source_value_plan(
                str(strategy.id), strategy.strategy_key, kind=SubstantiveValueKind.AUTHORED_THESIS
            )
        ]
    )
    executor = _executor(session, settings, provider)
    executor.set_active_job(job.id)
    executor.execute(plan_set.id)
    assert executor.skipped_duplicate is False
    assert provider.calls == 1


def test_abandoned_running_job_is_reclaimed(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    plan_set = get_or_create_plan_set(session, seed[1], seed[3])
    job = _job(
        session,
        plan_set,
        JobStatus.RUNNING,
        started_at=datetime.now(timezone.utc) - timedelta(seconds=7200),
    )
    provider = FakePlanningProvider(
        [
            make_source_value_plan(
                str(strategy.id), strategy.strategy_key, kind=SubstantiveValueKind.AUTHORED_THESIS
            )
        ]
    )
    executor = _executor(session, settings, provider)
    executor.set_active_job(job.id)
    executor.execute(plan_set.id)
    assert executor.skipped_duplicate is False
    assert provider.calls == 1


# --- Finding 3: hero elapsed-time cap ----------------------------------------


def test_long_source_support_preamble_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _inputs(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(
                use_full_window=False,
                word_start_index=0,
                word_end_index=5,
                source_role=SourceExcerptRole.SUPPORT,
            ),
            _source(
                use_full_window=False,
                word_start_index=6,
                word_end_index=10,
                source_role=SourceExcerptRole.HERO,
            ),
            _value(SubstantiveValueKind.AUTHORED_THESIS, "Analyze the drop mechanism"),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_LATE_HERO,)


def test_hero_appearance_time_uses_elapsed_blocks(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _inputs(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _value(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Frame the claim before the source",
                estimated_duration=2.0,
            ),
            _source(),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is not None
    assert result.plan.hero_appearance_time == pytest.approx(2.0)
    assert result.plan.hook_payoff_evidence["elapsed_before_hero_seconds"] == pytest.approx(2.0)


# --- Finding 4: minimum source excerpt duration ------------------------------


def test_too_short_hero_excerpt_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    words = [
        {"text": "alpha", "start": 21.0, "end": 22.0, "probability": 0.9},
        {"text": "beta", "start": 22.0, "end": 23.0, "probability": 0.9},
        {"text": "gamma", "start": 23.0, "end": 25.0, "probability": 0.9},
        {"text": "short", "start": 25.0, "end": 25.05, "probability": 0.9},
        {"text": "delta", "start": 26.0, "end": 28.0, "probability": 0.9},
        {"text": "epsilon", "start": 28.0, "end": 31.0, "probability": 0.9},
        {"text": "zeta", "start": 31.0, "end": 35.0, "probability": 0.9},
        {"text": "eta", "start": 35.0, "end": 40.0, "probability": 0.9},
        {"text": "theta", "start": 40.0, "end": 44.0, "probability": 0.9},
    ]
    seed, inputs = _inputs(session, settings, words=words)
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(use_full_window=False, word_start_index=3, word_end_index=3),
            _value(SubstantiveValueKind.SOURCE_AS_EVIDENCE, "Frame the moment as debate evidence"),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_SHORT_EXCERPT,)


# --- Finding 5: verification dependency linkage ------------------------------


def _verification_inputs(session: Session, settings: FakeStage41Settings):
    return _inputs(
        session,
        settings,
        strategy_type=TransformationStrategyType.COUNTERPOINT,
        external=ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION,
        value_kind=SubstantiveValueKind.COUNTERPOINT,
    )


def test_verification_dependency_valid_linkage(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _verification_inputs(session, settings)
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(),
            _value(
                SubstantiveValueKind.COUNTERPOINT,
                "Add the opposing productivity finding",
                dependency_ids=("meta-analysis",),
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.FACT_VERIFICATION_PLACEHOLDER,
                claim_dependency="meta-analysis",
                verification_rationale="The opposing figure needs an authoritative source",
                intended_use="Balance the claim",
                must_verify_before_execution=True,
                dependent_block_ids=(1,),
            ),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is not None
    assert result.plan.external_fact_dependencies[0]["dependency"] == "meta-analysis"


def test_nonexistent_dependency_reference_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _verification_inputs(session, settings)
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(),
            _value(
                SubstantiveValueKind.COUNTERPOINT,
                "Add the opposing productivity finding",
                dependency_ids=("meta-analysis",),
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.FACT_VERIFICATION_PLACEHOLDER,
                claim_dependency="meta-analysis",
                must_verify_before_execution=True,
                dependent_block_ids=(9,),
            ),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_VERIFICATION_BLOCK_REFERENCE,)


def test_unlinked_dependency_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _verification_inputs(session, settings)
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(),
            _value(
                SubstantiveValueKind.COUNTERPOINT,
                "Add the opposing productivity finding",
                dependency_ids=("other-claim",),
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.FACT_VERIFICATION_PLACEHOLDER,
                claim_dependency="meta-analysis",
                must_verify_before_execution=True,
                dependent_block_ids=(1,),
            ),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_VERIFICATION_UNLINKED,)


def test_narration_verification_linkage_accepted(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _verification_inputs(session, settings)
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(),
            _value(
                SubstantiveValueKind.COUNTERPOINT,
                "Add the opposing productivity finding",
                dependency_ids=("meta-analysis",),
                delivery_intent=None,
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.FACT_VERIFICATION_PLACEHOLDER,
                claim_dependency="meta-analysis",
                must_verify_before_execution=True,
            ),
        ],
        NarrationRequirement(
            need=NarrationNeed.REQUIRED,
            purposes=(NarrationPurpose.COUNTERPOINT,),
            language="ar",
            register="broadly_understandable",
            estimated_duration=4.0,
            essential=True,
            verification_dependency_ids=("meta-analysis",),
        ),
    )
    result = _validate(inputs, plan)
    assert result.plan is not None
    assert result.plan.status.value == "PLAN_GENERATED_WITH_VERIFICATION_REQUIRED"


# --- Finding 6: TTS/rendering/evasion boundary -------------------------------


def test_tts_selection_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _inputs(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(),
            _value(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Narrate this in a female voice",
            ),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_TTS_SELECTION,)


def test_rendering_instruction_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _inputs(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(),
            _value(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Add a keyframe at the timeline start",
            ),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_RENDERING_INSTRUCTION,)


def test_evasion_tactic_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _inputs(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(),
            _value(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Mirror the video to avoid detection by the platform",
            ),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_EVASION,)


def test_non_tts_model_word_is_allowed(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _inputs(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(),
            _value(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Explain the economic model behind the productivity drop",
            ),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is not None


# --- Finding 7: narration-contract contradictions ----------------------------


def test_narration_delivery_block_with_none_need_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _inputs(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    key = inputs.stage40_strategies[0]["strategy_key"]
    from app.core.enums import DeliveryIntent

    plan = _plan(
        key,
        [
            _source(),
            _value(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Explain the economic model behind the drop",
                delivery_intent=DeliveryIntent.NARRATION,
            ),
        ],
        NarrationRequirement(need=NarrationNeed.NONE),
    )
    result = _validate(inputs, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_NARRATION_CONTRADICTION,)


def test_valid_none_narration_plan_accepted(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _inputs(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key, [_source(), _value(SubstantiveValueKind.AUTHORED_THESIS, "Explain the model")]
    )
    result = _validate(inputs, plan)
    assert result.plan is not None
    assert result.plan.narration.is_none


# --- Finding 8: checkpoint durability across forced reruns -------------------


def test_two_consecutive_forced_reruns_reuse_checkpoint(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    plan_set = get_or_create_plan_set(session, seed[1], seed[3])
    plan = make_source_value_plan(
        str(strategy.id), strategy.strategy_key, kind=SubstantiveValueKind.AUTHORED_THESIS
    )
    provider = FakePlanningProvider([plan])
    executor = _executor(session, settings, provider)
    executor.execute(plan_set.id)
    assert provider.calls == 1
    assert list_plans(session, plan_set.id)

    second = FakePlanningProvider([plan])
    _executor(session, settings, second).execute(plan_set.id, force=True)
    session.refresh(plan_set)
    assert second.calls == 0
    assert list_plans(session, plan_set.id)
    assert any(attempt.get("checkpoint") for attempt in (plan_set.strategy_attempts or []))

    third = FakePlanningProvider([plan])
    _executor(session, settings, third).execute(plan_set.id, force=True)
    session.refresh(plan_set)
    assert third.calls == 0
    assert list_plans(session, plan_set.id)
    assert any(attempt.get("checkpoint") for attempt in (plan_set.strategy_attempts or []))


# --- Finding 9: raw hosted call budget ---------------------------------------


def _gemini_request() -> PlanningRequest:
    return PlanningRequest(
        strategy_id="id",
        strategy_key="k",
        strategy_type="ANALYSIS",
        intensity="MODERATE",
        direction_summary="d",
        added_value_focus="f",
        substantive_value_kind="AUTHORED_THESIS",
        preservation_requirements=(),
        external_verification_requirement="NOT_REQUIRED",
        verification_requirements=(),
        content_type="ANALYSIS",
        source_moment_structure="CLAIM",
        dialect_profile="EGYPTIAN",
        code_switch_tokens=(),
        target_market="UNSPECIFIED",
        output_language_policy="SOURCE_LANGUAGE",
        register_intent="SOURCE_COMPATIBLE",
        narration_allowed=True,
        max_blocks=8,
        strict_hero_cap_seconds=1.5,
        refined_transcript="text",
        refined_start=0.0,
        refined_end=10.0,
        context_text="",
        idea_summary="",
        topic_summary="",
        hooks=(),
        words=(),
    )


class _TimeoutModels:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, **kwargs: object) -> object:
        self.calls += 1
        raise TimeoutError("simulated timeout")


class _TimeoutClient:
    def __init__(self) -> None:
        self.models = _TimeoutModels()

    def close(self) -> None:
        return None


def test_retryable_failure_does_not_exceed_raw_budget() -> None:
    client = _TimeoutClient()
    provider = GeminiPlanningProvider(
        api_key="key",
        client_factory=lambda: client,
        retry_attempts=5,
        max_raw_calls=2,
    )
    with pytest.raises(PlanningProviderError):
        provider.plan([_gemini_request()], "ROUTINE")
    assert provider.raw_call_count() == 1
    with pytest.raises(PlanningProviderError):
        provider.plan([_gemini_request()], "STRONG")
    assert provider.raw_call_count() == 2
    # The hard two-call ceiling blocks a third raw call without retrying.
    with pytest.raises(PlanningProviderError):
        provider.plan([_gemini_request()], "ROUTINE")
    assert provider.raw_call_count() == 2
    assert client.models.calls == 2


def test_metrics_report_raw_hosted_calls(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [
            make_source_value_plan(
                str(strategy.id), strategy.strategy_key, kind=SubstantiveValueKind.AUTHORED_THESIS
            )
        ]
    )
    plan_set = run_planning(session, settings, seed, provider)
    assert plan_set.metrics["hosted_raw_calls"] == 1
    assert plan_set.metrics["gemini_calls"] == 1


# --- Finding 10: overlapping source excerpts ---------------------------------


def test_partially_overlapping_excerpts_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _inputs(session, settings)
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(use_full_window=False, word_start_index=0, word_end_index=5),
            _source(
                use_full_window=False,
                word_start_index=3,
                word_end_index=8,
                source_role=SourceExcerptRole.SUPPORT,
            ),
            _value(SubstantiveValueKind.SOURCE_AS_EVIDENCE, "Frame the moment as evidence"),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_OVERLAPPING_EXCERPT,)


def test_distinct_non_overlapping_excerpts_accepted(session: Session) -> None:
    settings = FakeStage41Settings()
    seed, inputs = _inputs(session, settings)
    key = inputs.stage40_strategies[0]["strategy_key"]
    plan = _plan(
        key,
        [
            _source(use_full_window=False, word_start_index=0, word_end_index=3),
            _source(
                use_full_window=False,
                word_start_index=6,
                word_end_index=9,
                source_role=SourceExcerptRole.SUPPORT,
            ),
            _value(SubstantiveValueKind.SOURCE_AS_EVIDENCE, "Frame the moment as evidence"),
        ],
    )
    result = _validate(inputs, plan)
    assert result.plan is not None
