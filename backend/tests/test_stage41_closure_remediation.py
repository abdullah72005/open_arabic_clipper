"""Stage 4.1 final-closure remediation tests (findings 1-6).

Focused, hermetic tests. No live provider calls: providers are fakes and the
one threaded case only blocks a fake provider.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from stage41_support import (
    FakePlanningProvider,
    FakeStage41Settings,
    install_stage41_settings,
    make_source_value_plan,
    seed_stage41,
)

from app.core.enums import (
    ExternalFactRequirement,
    JobKind,
    JobStatus,
    PlanBlockType,
    PlanExecutionStatus,
    SemanticProviderMode,
    SourceExcerptRole,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.db.base import Base
from app.models import ProcessingJob, TransformationPlanSet
from app.transformation.planning.executor import TransformationPlanningExecutor
from app.transformation.planning.inputs import build_planning_inputs
from app.transformation.planning.policy import DEFAULT_CONFIG
from app.transformation.planning.providers import (
    build_planning_request,
    parse_plan_results,
)
from app.transformation.planning.queue import get_or_create_plan_set, list_plans
from app.transformation.planning.types import (
    NarrationNeed,
    NarrationRequirement,
    PlanProviderBlock,
    PlanProviderPlan,
)
from app.transformation.planning.validation import (
    REJECT_SPEAKER_SELECTION,
    REJECT_VERIFICATION_BLOCK_REFERENCE,
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


def _job(
    session: Session,
    plan_set: TransformationPlanSet,
    *,
    status: JobStatus = JobStatus.QUEUED,
    **kwargs: Any,
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
    job_claim_stale_seconds: float = 3600.0,
    heartbeat_interval_seconds: float = 30.0,
) -> TransformationPlanningExecutor:
    return TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=provider,  # type: ignore[arg-type]
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=mode,
        config=DEFAULT_CONFIG,
        job_claim_stale_seconds=job_claim_stale_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


# --- Finding 1: cancellation fences persistence ------------------------------


class _CancelBeforePersist(TransformationPlanningExecutor):
    """Cancels the job after the execute cancellation check, before the fence."""

    def _persist(self, plan_set: Any, refinement: Any, inputs: Any, outcome: Any, duration: float):
        job = self._session.get(ProcessingJob, self._active_job_id)
        job.status = JobStatus.CANCELLED
        self._session.commit()
        return super()._persist(plan_set, refinement, inputs, outcome, duration)


def test_cancellation_between_check_and_persist_rejects_persistence(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    plan_set = get_or_create_plan_set(session, seed[1], seed[3])
    job = _job(session, plan_set)
    provider = FakePlanningProvider(
        [make_source_value_plan(str(strategy.id), strategy.strategy_key)]
    )
    executor = _CancelBeforePersist(
        session=session,
        settings=settings,
        provider=provider,
        provider_identity=settings.transformation_planning_provider_identity(),
        config=DEFAULT_CONFIG,
    )
    executor.set_active_job(job.id)
    executor.execute(plan_set.id)

    session.expire_all()
    refreshed_job = session.get(ProcessingJob, job.id)
    refreshed_plan_set = session.get(TransformationPlanSet, plan_set.id)
    assert refreshed_job.status is JobStatus.CANCELLED
    assert refreshed_plan_set.execution_status is PlanExecutionStatus.CANCELLED
    assert list_plans(session, plan_set.id) == []


# --- Finding 2: renewable liveness heartbeat ---------------------------------


class _BlockingProvider(FakePlanningProvider):
    def __init__(self, plans: Any, entered: threading.Event, release: threading.Event) -> None:
        super().__init__(plans)
        self._entered = entered
        self._release = release

    def plan(self, requests: Any, tier: str = "ROUTINE") -> Any:
        self._entered.set()
        if not self._release.wait(30):
            raise RuntimeError("provider was never released")
        return super().plan(requests, tier)


def test_live_provider_worker_is_not_reclaimed(sqlite_engine: Engine) -> None:
    Base.metadata.create_all(sqlite_engine)
    settings = FakeStage41Settings()
    _install(settings)
    with Session(sqlite_engine) as setup:
        seed = seed_stage41(setup, settings=settings)
        plan_set = get_or_create_plan_set(setup, seed[1], seed[3])
        job = _job(setup, plan_set)
        plan_set_id, job_id = plan_set.id, job.id
        strategy = seed[4][0]
        strategy_id, strategy_key = strategy.id, strategy.strategy_key

    factory = sessionmaker(bind=sqlite_engine)
    entered = threading.Event()
    release = threading.Event()
    outcomes: dict[str, Any] = {}

    def run_worker_a() -> None:
        with factory() as session_a:
            provider_a = _BlockingProvider(
                [make_source_value_plan(str(strategy_id), strategy_key)], entered, release
            )
            executor_a = _executor(
                session_a,
                settings,
                provider_a,
                job_claim_stale_seconds=0.3,
                heartbeat_interval_seconds=0.05,
            )
            executor_a.set_active_job(job_id)
            executor_a.execute(plan_set_id)
            outcomes["a_calls"] = provider_a.calls

    worker = threading.Thread(target=run_worker_a, daemon=True)
    worker.start()
    try:
        assert entered.wait(10)
        # Hold A inside provider.plan well past the configured stale threshold.
        time.sleep(0.9)
        with factory() as session_b:
            provider_b = FakePlanningProvider(
                [make_source_value_plan(str(strategy_id), strategy_key)]
            )
            executor_b = _executor(
                session_b,
                settings,
                provider_b,
                job_claim_stale_seconds=0.3,
                heartbeat_interval_seconds=0.05,
            )
            executor_b.set_active_job(job_id)
            executor_b.execute(plan_set_id)
            assert executor_b.skipped_duplicate is True
            assert provider_b.calls == 0
    finally:
        release.set()
        worker.join(30)
    assert not worker.is_alive()
    assert outcomes.get("a_calls") == 1

    with factory() as check:
        final_job = check.get(ProcessingJob, job_id)
        assert final_job.status is JobStatus.SUCCEEDED
        assert final_job.claim_version == 1
        assert len(list_plans(check, plan_set_id)) == 1


def test_abandoned_heartbeat_job_is_reclaimed(sqlite_engine: Engine) -> None:
    Base.metadata.create_all(sqlite_engine)
    settings = FakeStage41Settings()
    _install(settings)
    with Session(sqlite_engine) as setup:
        seed = seed_stage41(setup, settings=settings)
        plan_set = get_or_create_plan_set(setup, seed[1], seed[3])
        old = datetime.now(timezone.utc) - timedelta(seconds=7200)
        job = _job(
            setup,
            plan_set,
            status=JobStatus.RUNNING,
            started_at=old,
            heartbeat_at=old,
            claim_version=1,
        )
        plan_set_id, job_id = plan_set.id, job.id
        strategy = seed[4][0]
        strategy_id, strategy_key = strategy.id, strategy.strategy_key

    with Session(sqlite_engine) as session_b:
        provider_b = FakePlanningProvider([make_source_value_plan(str(strategy_id), strategy_key)])
        executor_b = _executor(session_b, settings, provider_b)
        executor_b.set_active_job(job_id)
        executor_b.execute(plan_set_id)
        assert executor_b.skipped_duplicate is False
        assert provider_b.calls == 1
        reclaimed = session_b.get(ProcessingJob, job_id)
        assert reclaimed.claim_version == 2
        assert reclaimed.status is JobStatus.SUCCEEDED
        assert len(list_plans(session_b, plan_set_id)) == 1


# --- Finding 3: speaker-identity bypasses ------------------------------------


def _plan_inputs(session: Session, settings: FakeStage41Settings):
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    inputs = build_planning_inputs(
        session,
        seed[1],
        seed[3],
        seed[2],
        settings,
        DEFAULT_CONFIG,  # type: ignore[arg-type]
    )
    return seed, inputs


def _boundary_value(intent: str, **overrides: Any) -> PlanProviderBlock:
    base: dict[str, Any] = {
        "block_type": PlanBlockType.ORIGINAL_VALUE,
        "estimated_duration": 4.0,
        "substantive_value_kind": SubstantiveValueKind.AUTHORED_THESIS,
        "semantic_intent": intent,
        "why_unavailable": "The excerpt alone does not provide this added dimension",
    }
    base.update(overrides)
    return PlanProviderBlock(**base)


def _boundary_validate(session: Session, intent: str):
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    strategy = inputs.stage40_strategies[0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy["id"]),
        strategy_key=str(strategy["strategy_key"]),
        confidence=0.7,
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                use_full_window=True,
                source_role=SourceExcerptRole.HERO,
            ),
            _boundary_value(intent),
        ),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )
    return validate_provider_plan(
        plan,
        strategy,
        inputs,
        DEFAULT_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="fp",
    )


def test_sound_like_imitation_rejected(session: Session) -> None:
    result = _boundary_validate(session, "Make it sound like Morgan Freeman")
    assert result.plan is None
    assert result.reasons == (REJECT_SPEAKER_SELECTION,)


def test_in_style_of_person_rejected(session: Session) -> None:
    result = _boundary_validate(session, "In the style of Morgan Freeman")
    assert result.plan is None
    assert result.reasons == (REJECT_SPEAKER_SELECTION,)


def test_legitimate_named_person_reference_allowed(session: Session) -> None:
    result = _boundary_validate(
        session, "Reference Morgan Freeman's career as context for the claim"
    )
    assert result.plan is not None


def test_explain_economic_model_still_allowed(session: Session) -> None:
    result = _boundary_validate(session, "Explain the economic model behind the productivity drop")
    assert result.plan is not None


# --- Finding 4: malformed verification-reference shapes ----------------------


def _verification_inputs(session: Session, settings: FakeStage41Settings):
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.COUNTERPOINT,
        external=ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION,
        value_kind=SubstantiveValueKind.COUNTERPOINT,
    )
    inputs = build_planning_inputs(
        session,
        seed[1],
        seed[3],
        seed[2],
        settings,
        DEFAULT_CONFIG,  # type: ignore[arg-type]
    )
    return seed, inputs


def _verification_content(dependent_ids: object) -> dict[str, object]:
    return {
        "plans": [
            {
                "strategy_id": "id",
                "strategy_key": "key",
                "confidence": 0.7,
                "blocks": [
                    {
                        "block_type": "SOURCE_EXCERPT",
                        "source_role": "HERO",
                        "use_full_window": True,
                    },
                    {
                        "block_type": "ORIGINAL_VALUE",
                        "estimated_duration": 4,
                        "substantive_value_kind": "COUNTERPOINT",
                        "semantic_intent": "Add the opposing productivity finding",
                        "why_unavailable": "The excerpt states the claim without the counterpoint",
                        "dependency_ids": ["meta-analysis"],
                    },
                    {
                        "block_type": "FACT_VERIFICATION_PLACEHOLDER",
                        "claim_dependency": "meta-analysis",
                        "must_verify_before_execution": True,
                        "dependent_block_ids": dependent_ids,
                    },
                ],
            }
        ]
    }


def _validate_parsed(session: Session, settings: FakeStage41Settings, dependent_ids: object):
    seed, inputs = _verification_inputs(session, settings)
    strategy = inputs.stage40_strategies[0]
    request = build_planning_request(strategy, inputs, DEFAULT_CONFIG)
    content = _verification_content(dependent_ids)
    content["plans"][0]["strategy_id"] = request.strategy_id  # type: ignore[index]
    content["plans"][0]["strategy_key"] = request.strategy_key  # type: ignore[index]
    results = parse_plan_results(content, [request])
    assert request.strategy_key in results
    provider_plan = results[request.strategy_key].plans[0]
    result = validate_provider_plan(
        provider_plan,
        strategy,
        inputs,
        DEFAULT_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="fp",
    )
    return provider_plan, result


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "shape",
    ["not-a-list", "", True, 123, 1.5, {"nested": 1}, None, [[1]], [1, "bad"]],
)
def test_malformed_dependent_block_ids_shapes_rejected(session: Session, shape: object) -> None:
    provider_plan, result = _validate_parsed(session, FakeStage41Settings(), shape)
    assert provider_plan.blocks[2].dependent_block_ids_invalid is True
    assert result.plan is None
    assert result.reasons == (REJECT_VERIFICATION_BLOCK_REFERENCE,)


def test_malformed_item_after_valid_cap_rejected(session: Session) -> None:
    ids = list(range(1, 13)) + ["later-bad-shape"]
    provider_plan, result = _validate_parsed(session, FakeStage41Settings(), ids)
    assert provider_plan.blocks[2].dependent_block_ids_invalid is True
    assert len(provider_plan.blocks[2].dependent_block_ids) <= 12
    assert result.plan is None
    assert result.reasons == (REJECT_VERIFICATION_BLOCK_REFERENCE,)


def test_valid_integer_dependent_block_ids_preserved(session: Session) -> None:
    provider_plan, result = _validate_parsed(session, FakeStage41Settings(), [1])
    assert provider_plan.blocks[2].dependent_block_ids == (1,)
    assert result.plan is not None


# --- Finding 5: clear stale failure metadata on retry/success ----------------


def test_failed_job_retry_clears_stale_failure_metadata(session: Session) -> None:
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
        status=JobStatus.FAILED,
        error_code="OLD_FAILURE",
        error_message="old failure message",
        completed_at=datetime.now(timezone.utc),
    )
    provider = FakePlanningProvider(
        [make_source_value_plan(str(strategy.id), strategy.strategy_key)]
    )
    executor = _executor(session, settings, provider)
    executor.set_active_job(job.id)
    executor.execute(plan_set.id)
    session.refresh(job)
    assert job.status is JobStatus.SUCCEEDED
    assert job.error_code is None
    assert job.error_message is None
    assert job.completed_at is not None


# --- Finding 6: record_failure safe after database errors --------------------


def test_record_failure_recovers_after_commit_failure(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    plan_set = get_or_create_plan_set(session, seed[1], seed[3])
    job = _job(session, plan_set)
    provider = FakePlanningProvider(
        [make_source_value_plan(str(strategy.id), strategy.strategy_key)]
    )
    executor = _executor(session, settings, provider)
    executor.set_active_job(job.id)

    real_commit = session.commit
    calls = {"count": 0}

    def flaky_commit() -> None:
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("simulated persistence commit failure")
        real_commit()

    monkeypatch.setattr(session, "commit", flaky_commit)
    with pytest.raises(RuntimeError):
        executor.execute(plan_set.id)

    # The session is left needing rollback; record_failure must recover safely.
    executor.record_failure(RuntimeError("simulated persistence commit failure"))

    session.rollback()
    recovered = session.get(ProcessingJob, job.id)
    assert recovered.status is JobStatus.FAILED
    assert recovered.error_message is not None
    assert "RuntimeError" in recovered.error_message
