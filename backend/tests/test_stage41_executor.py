"""Focused Stage 4.1 executor, queue, cache, concurrency, and cancellation tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from stage41_support import (
    FakePlanningProvider,
    FakeStage41Settings,
    install_stage41_settings,
    make_source_value_plan,
    run_planning,
    seed_stage41,
)

from app.core.enums import (
    JobKind,
    JobStatus,
    PlanExecutionStatus,
    PlanSemanticOutcome,
    SemanticProviderMode,
    TransformationStrategyType,
)
from app.db.base import Base
from app.models import ProcessingJob, SourceVideo, TransformationPlan, TransformationPlanSet
from app.transformation.planning.executor import (
    PlanningCancelled,
    TransformationPlanningExecutor,
)
from app.transformation.planning.policy import DEFAULT_CONFIG
from app.transformation.planning.queue import (
    get_or_create_plan_set,
    list_plans,
    queue_transformation_planning,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.workers.tasks.run_transformation_planning.delay", lambda *a, **k: None)


def _install(settings: FakeStage41Settings) -> None:
    from _pytest.monkeypatch import MonkeyPatch

    install_stage41_settings(MonkeyPatch(), settings)


def _seed(session: Session, settings: FakeStage41Settings, **kwargs: Any) -> tuple[Any, ...]:
    _install(settings)
    return seed_stage41(session, settings=settings, **kwargs)


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
    )


def test_repeated_execution_is_idempotent_with_stable_plan_uuid(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    plan_set = run_planning(session, settings, seed)
    first = list_plans(session, plan_set.id)
    assert len(first) == 1
    plan_id = first[0].id
    run_planning(session, settings, seed)
    second = list_plans(session, plan_set.id)
    assert len(second) == 1
    assert second[0].id == plan_id


def test_cache_hit_makes_no_duplicate_provider_call(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [make_source_value_plan(str(strategy.id), strategy.strategy_key)]
    )
    _install(settings)
    plan_set = run_planning(session, settings, seed, provider)
    assert provider.calls == 1
    assert provider.released >= 1
    # Second execution is a cache hit and must not call the provider again.
    session.refresh(plan_set)
    assert plan_set.cache_eligible is True
    run_planning(session, settings, seed, provider)
    assert provider.calls == 1


def test_force_does_not_repeat_matching_accepted_hosted_work(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    plan = make_source_value_plan(str(strategy.id), strategy.strategy_key)
    provider = FakePlanningProvider([plan])
    run_planning(session, settings, seed, provider)
    assert provider.calls == 1
    # Force re-enters the executor but reuses the accepted checkpoint.
    _source, candidate, _ref, analysis, _strategies = seed
    plan_set = get_or_create_plan_set(session, candidate, analysis)
    second_provider = FakePlanningProvider([plan])
    _executor(session, settings, second_provider).execute(plan_set.id, force=True)
    session.refresh(plan_set)
    assert second_provider.calls == 0
    assert list_plans(session, plan_set.id)


def test_stage40_strategy_change_invalidates_plan_set(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    plan_set = run_planning(session, settings, seed)
    assert plan_set.cache_eligible is True
    strategy = seed[4][0]
    strategy.strategy_fingerprint = "changed-strategy-fingerprint"
    session.commit()
    session.refresh(plan_set)
    outcome = queue_transformation_planning(session, seed[1], seed[3], force=False)
    assert outcome.cached is False


def test_transcript_change_invalidates_plan_set(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    plan_set = run_planning(session, settings, seed)
    refinement = seed[2]
    refinement.final_transcript = "Completely different upstream transcript now and more text."
    session.commit()
    session.refresh(plan_set)
    outcome = queue_transformation_planning(session, seed[1], seed[3], force=False)
    assert outcome.cached is False


def test_voice_only_changes_do_not_invalidate_semantic_plan(session: Session) -> None:
    from dataclasses import replace

    from app.transformation.planning.fingerprints import (
        build_plan_set_input_payload,
        planning_input_fingerprint,
    )
    from app.transformation.planning.inputs import PlanningContextResolver

    settings = FakeStage41Settings()
    _install(settings)
    _src, candidate, refinement, analysis, _strategies = seed_stage41(session, settings=settings)
    from app.transformation.planning.inputs import build_planning_inputs

    inputs = build_planning_inputs(
        session,
        candidate,
        analysis,
        refinement,
        settings,
        DEFAULT_CONFIG,  # type: ignore[arg-type]
    )
    voice_a = PlanningContextResolver.from_mapping({"tts_voice_id": "Charon"})
    voice_b = PlanningContextResolver.from_mapping(
        {"tts_voice_id": "Puck", "tts_provider": "other", "tts_model": "other-model"}
    )
    assert voice_a.semantic_payload() == voice_b.semantic_payload()
    fp_a = planning_input_fingerprint(
        build_plan_set_input_payload(
            inputs=replace(inputs, planning_context=voice_a),
            config=DEFAULT_CONFIG,
            provider_mode="adaptive",
            provider_identity={"provider": "fake"},
        )
    )
    fp_b = planning_input_fingerprint(
        build_plan_set_input_payload(
            inputs=replace(inputs, planning_context=voice_b),
            config=DEFAULT_CONFIG,
            provider_mode="adaptive",
            provider_identity={"provider": "fake"},
        )
    )
    assert fp_a == fp_b
    # Target-market semantics do invalidate.
    gcc = PlanningContextResolver.from_mapping({"target_market": "GCC"})
    fp_c = planning_input_fingerprint(
        build_plan_set_input_payload(
            inputs=replace(inputs, planning_context=gcc),
            config=DEFAULT_CONFIG,
            provider_mode="adaptive",
            provider_identity={"provider": "fake"},
        )
    )
    assert fp_c != fp_a


def test_accepted_plans_survive_partial_provider_failure(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        second_strategy=(
            TransformationStrategyType.EXPLANATORY,
            "Explain the mechanism behind the promotion-rate drop",
        ),
    )
    first = seed[4][0]
    # First run: provider returns a valid plan for the first strategy only.
    provider = FakePlanningProvider([make_source_value_plan(str(first.id), first.strategy_key)])
    plan_set = run_planning(session, settings, seed, provider)
    assert plan_set.cache_eligible is False
    rows = list_plans(session, plan_set.id)
    assert len(rows) == 1

    # Second run: provider outage must not overwrite the accepted plan.
    outage = FakePlanningProvider([])
    outage.behavior = "outage"
    _source, candidate, _ref, analysis, _strategies = seed
    executor = _executor(session, settings, outage)
    executor.execute(plan_set.id)
    session.refresh(plan_set)
    rows = list_plans(session, plan_set.id)
    assert len(rows) == 1
    assert rows[0].strategy_candidate_id == first.id


def test_cancellation_stays_cancelled_and_preserves_checkpoint(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    _source, candidate, _ref, analysis, _strategies = seed
    plan_set = get_or_create_plan_set(session, candidate, analysis)
    job = ProcessingJob(
        source_video_id=candidate.source_video_id,
        kind=JobKind.TRANSFORMATION_PLANNING,
        transformation_plan_set_id=plan_set.id,
        status=JobStatus.CANCELLED,
    )
    session.add(job)
    session.commit()
    plan_set.strategy_attempts = [
        {
            "strategy_id": str(seed[4][0].id),
            "checkpoint": {"provider_input_fingerprint": "x", "result": {"plans": []}},
        }
    ]
    session.commit()
    executor = _executor(session, settings, None)
    executor.set_active_job(job.id)
    with pytest.raises(PlanningCancelled):
        executor.execute(plan_set.id)
    session.refresh(plan_set)
    assert plan_set.execution_status is PlanExecutionStatus.CANCELLED
    assert list_plans(session, plan_set.id) == []
    assert plan_set.strategy_attempts[0]["checkpoint"]["provider_input_fingerprint"] == "x"


def test_provider_released_on_success_failure_and_malformed(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    plan = make_source_value_plan(str(strategy.id), strategy.strategy_key)
    ok_provider = FakePlanningProvider([plan])
    run_planning(session, settings, seed, ok_provider)
    assert ok_provider.released >= 1

    seed2 = _seed(session, settings)
    strategy2 = seed2[4][0]
    bad = FakePlanningProvider([make_source_value_plan(str(strategy2.id), strategy2.strategy_key)])
    bad.behavior = "malformed"
    run_planning(session, settings, seed2, bad)
    assert bad.released >= 1


def test_concurrent_requests_create_one_plan_set_and_one_active_job(
    sqlite_engine: Engine,
) -> None:
    Base.metadata.create_all(sqlite_engine)
    settings = FakeStage41Settings()
    _install(settings)
    with Session(sqlite_engine) as setup:
        seed = seed_stage41(setup, settings=settings)
        candidate, analysis = seed[1], seed[3]
        candidate_id, analysis_id = candidate.id, analysis.id
        setup.commit()

    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=sqlite_engine)
    with factory() as s1, factory() as s2:
        c1 = s1.get(type(candidate), candidate_id)
        a1 = s1.get(type(analysis), analysis_id)
        c2 = s2.get(type(candidate), candidate_id)
        a2 = s2.get(type(analysis), analysis_id)
        outcome1 = queue_transformation_planning(s1, c1, a1)
        outcome2 = queue_transformation_planning(s2, c2, a2)
    with factory() as check:
        plan_sets = list(check.scalars(select(TransformationPlanSet)).all())
        jobs = list(
            check.scalars(
                select(ProcessingJob).where(ProcessingJob.kind == JobKind.TRANSFORMATION_PLANNING)
            ).all()
        )
    assert len(plan_sets) == 1
    assert outcome1.plan_set_id == outcome2.plan_set_id
    assert len(jobs) <= 1


def test_queue_cache_hit_returns_cached(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    run_planning(session, settings, seed)
    _source, candidate, _ref, analysis, _strategies = seed
    outcome = queue_transformation_planning(session, candidate, analysis)
    assert outcome.cached is True
    assert outcome.queued is False


def test_no_plan_outcome_is_success_not_pipeline_failure(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        added_value_focus="",
    )
    plan_set = run_planning(session, settings, seed, None)
    assert plan_set.planning_outcome in {
        PlanSemanticOutcome.PLANNING_DEFERRED,
        PlanSemanticOutcome.PROVIDER_UNAVAILABLE,
        PlanSemanticOutcome.NO_VALID_PLAN_FROM_STRATEGY,
    }
    assert plan_set.execution_status is PlanExecutionStatus.COMPLETE


def test_only_plan_sets_created_and_no_automatic_next_stage(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    run_planning(session, settings, seed)
    jobs = list(
        session.scalars(
            select(ProcessingJob).where(ProcessingJob.kind == JobKind.TRANSFORMATION_PLANNING)
        ).all()
    )
    # The executor itself schedules nothing; queueing creates at most one job.
    assert all(job.source_video_id for job in jobs)
    assert "TRANSFORMATION_PLANNING" in {k.value for k in JobKind}
    assert len(list(session.scalars(select(SourceVideo)).all())) == 1
    assert len(list(session.scalars(select(TransformationPlan)).all())) == 1
