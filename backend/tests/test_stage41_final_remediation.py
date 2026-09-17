"""Stage 4.1 final re-review remediation tests (durable fencing, boundary, metrics).

Focused, hermetic tests for the five remaining re-review findings. No live
provider calls: Gemini is exercised only through an injected fake client.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from sqlalchemy import Engine
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
from app.pipeline.executor import StageCancelled
from app.transformation.planning.executor import TransformationPlanningExecutor
from app.transformation.planning.gemini import GeminiPlanningProvider
from app.transformation.planning.inputs import (
    PlanningInputError,
    build_planning_inputs,
)
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
    REJECT_TTS_SELECTION,
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


# --- Finding 4: hosted raw-call metrics --------------------------------------


def test_local_only_reports_zero_hosted_raw_calls(session: Session) -> None:
    settings = FakeStage41Settings(mode=SemanticProviderMode.LOCAL_ONLY)
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
                str(strategy.id),
                strategy.strategy_key,
                kind=SubstantiveValueKind.AUTHORED_THESIS,
            )
        ],
        hosted=False,
    )
    plan_set = run_planning(session, settings, seed, provider, mode=SemanticProviderMode.LOCAL_ONLY)
    assert plan_set.metrics["hosted_raw_calls"] == 0
    # Local work still happened.
    assert provider.calls >= 1


def test_deterministic_reports_zero_hosted_raw_calls(session: Session) -> None:
    settings = FakeStage41Settings(mode=SemanticProviderMode.DETERMINISTIC)
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.SOURCE_LED_MINIMAL,
        value_kind=SubstantiveValueKind.INFERENCE,
        added_value_focus="State the inference behind the clip",
    )
    plan_set = run_planning(session, settings, seed, None, mode=SemanticProviderMode.DETERMINISTIC)
    assert plan_set.metrics["hosted_raw_calls"] == 0


def test_non_hosted_provider_reports_zero_hosted_raw_calls(session: Session) -> None:
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
                str(strategy.id),
                strategy.strategy_key,
                kind=SubstantiveValueKind.AUTHORED_THESIS,
            )
        ],
        hosted=False,
    )
    plan_set = run_planning(session, settings, seed, provider)
    assert provider.calls == 1
    assert plan_set.metrics["hosted_raw_calls"] == 0


class _GeminiModels:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, **kwargs: object) -> object:
        self.calls += 1
        payload = json.loads(str(kwargs["contents"]))
        requested = payload["plans_requested"]
        plans = [
            {
                "strategy_id": item["strategy_id"],
                "strategy_key": item["strategy_key"],
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
                        "substantive_value_kind": "AUTHORED_THESIS",
                        "semantic_intent": "Analyze the causal mechanism behind the drop",
                        "why_unavailable": "The excerpt states the drop without the cause",
                    },
                ],
            }
            for item in requested
        ]
        return type(
            "_Response",
            (),
            {
                "parsed": {"plans": plans},
                "usage_metadata": None,
                "prompt_feedback": None,
                "candidates": [type("_C", (), {"finish_reason": None})()],
            },
        )()


class _GeminiClient:
    def __init__(self) -> None:
        self.models = _GeminiModels()

    def close(self) -> None:
        return None


def test_gemini_reports_actual_raw_hosted_calls(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    client = _GeminiClient()
    provider = GeminiPlanningProvider(api_key="key", client_factory=lambda: client)
    plan_set = run_planning(session, settings, seed, provider)
    assert provider.raw_call_count() == 1
    assert plan_set.metrics["hosted_raw_calls"] == 1
    assert plan_set.metrics["gemini_calls"] == 1


# --- Finding 3: malformed verification references ----------------------------


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


def test_valid_integer_dependent_block_ids_survive_parse_path(session: Session) -> None:
    provider_plan, result = _validate_parsed(session, FakeStage41Settings(), [1])
    assert provider_plan.blocks[2].dependent_block_ids == (1,)
    assert result.plan is not None


def test_numeric_string_dependent_block_ids_rejected_through_parse_path(
    session: Session,
) -> None:
    _provider_plan, result = _validate_parsed(session, FakeStage41Settings(), ["1"])
    assert result.plan is None
    assert result.reasons == (REJECT_VERIFICATION_BLOCK_REFERENCE,)


def test_fractional_dependent_block_ids_rejected_through_parse_path(
    session: Session,
) -> None:
    _provider_plan, result = _validate_parsed(session, FakeStage41Settings(), [1.5])
    assert result.plan is None
    assert result.reasons == (REJECT_VERIFICATION_BLOCK_REFERENCE,)


def test_boolean_dependent_block_ids_rejected_through_parse_path(
    session: Session,
) -> None:
    _provider_plan, result = _validate_parsed(session, FakeStage41Settings(), [True])
    assert result.plan is None
    assert result.reasons == (REJECT_VERIFICATION_BLOCK_REFERENCE,)


# --- Finding 2: complete deterministic provider-text boundary ----------------


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


def _boundary_source() -> PlanProviderBlock:
    return PlanProviderBlock(
        block_type=PlanBlockType.SOURCE_EXCERPT,
        use_full_window=True,
        source_role=SourceExcerptRole.HERO,
    )


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


def _boundary_plan(key: str, blocks: list[PlanProviderBlock], **overrides: Any) -> PlanProviderPlan:
    return PlanProviderPlan(
        strategy_id="id",
        strategy_key=key,
        confidence=0.7,
        blocks=tuple(blocks),
        narration=overrides.pop("narration", NarrationRequirement(need=NarrationNeed.NONE)),
        **overrides,
    )


def _boundary_validate(session: Session, plan: PlanProviderPlan):
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    strategy = inputs.stage40_strategies[0]
    plan = replace(
        plan, strategy_id=str(strategy["id"]), strategy_key=str(strategy["strategy_key"])
    )
    return validate_provider_plan(
        plan,
        strategy,
        inputs,
        DEFAULT_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="fp",
    )


def test_provider_plus_voice_selection_rejected(session: Session) -> None:
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    key = str(inputs.stage40_strategies[0]["strategy_key"])
    plan = _boundary_plan(key, [_boundary_source(), _boundary_value("Use Gemini voice Charon")])
    result = _boundary_validate(session, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_TTS_SELECTION,)


def test_named_speaker_selection_rejected(session: Session) -> None:
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    key = str(inputs.stage40_strategies[0]["strategy_key"])
    plan = _boundary_plan(
        key,
        [_boundary_source(), _boundary_value("Have Morgan Freeman narrate the analysis")],
    )
    result = _boundary_validate(session, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_SPEAKER_SELECTION,)


def test_forbidden_text_in_planner_notes_rejected(session: Session) -> None:
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    key = str(inputs.stage40_strategies[0]["strategy_key"])
    plan = _boundary_plan(
        key,
        [_boundary_source(), _boundary_value("Explain the causal mechanism behind the drop")],
        planner_notes="Then use Gemini voice Charon for the read",
    )
    result = _boundary_validate(session, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_TTS_SELECTION,)


def test_forbidden_text_in_no_valid_reason_rejected(session: Session) -> None:
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    key = str(inputs.stage40_strategies[0]["strategy_key"])
    plan = _boundary_plan(
        key,
        [],
        no_valid_plan=True,
        no_valid_reason="Use Gemini voice Charon to read it",
    )
    result = _boundary_validate(session, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_TTS_SELECTION,)
    assert result.explicit_no_plan is False


def test_forbidden_text_in_dependency_ids_rejected(session: Session) -> None:
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    key = str(inputs.stage40_strategies[0]["strategy_key"])
    plan = _boundary_plan(
        key,
        [
            _boundary_source(),
            _boundary_value(
                "Explain the causal mechanism behind the drop",
                dependency_ids=("Use Gemini voice Charon",),
            ),
        ],
    )
    result = _boundary_validate(session, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_TTS_SELECTION,)


def test_forbidden_text_in_verification_rationale_rejected(session: Session) -> None:
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    key = str(inputs.stage40_strategies[0]["strategy_key"])
    plan = _boundary_plan(
        key,
        [
            _boundary_source(),
            _boundary_value("Explain the causal mechanism behind the drop"),
            PlanProviderBlock(
                block_type=PlanBlockType.FACT_VERIFICATION_PLACEHOLDER,
                claim_dependency="claim-1",
                verification_rationale="Have Morgan Freeman narrate the correction",
                must_verify_before_execution=True,
                dependent_block_ids=(1,),
            ),
        ],
    )
    result = _boundary_validate(session, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_SPEAKER_SELECTION,)


def test_forbidden_text_in_narration_language_rejected(session: Session) -> None:
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    key = str(inputs.stage40_strategies[0]["strategy_key"])
    plan = _boundary_plan(
        key,
        [_boundary_source(), _boundary_value("Explain the causal mechanism behind the drop")],
        narration=NarrationRequirement(
            need=NarrationNeed.OPTIONAL,
            language="Gemini voice Charon",
            register="broadly_understandable",
            estimated_duration=3.0,
        ),
    )
    result = _boundary_validate(session, plan)
    assert result.plan is None
    assert result.reasons == (REJECT_TTS_SELECTION,)


def test_legitimate_model_reference_is_allowed(session: Session) -> None:
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    key = str(inputs.stage40_strategies[0]["strategy_key"])
    plan = _boundary_plan(
        key,
        [
            _boundary_source(),
            _boundary_value("Explain the economic model behind the productivity drop"),
        ],
    )
    result = _boundary_validate(session, plan)
    assert result.plan is not None


# --- Finding 5: failed-job diagnostics ---------------------------------------


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


def _job(session: Session, plan_set: TransformationPlanSet) -> ProcessingJob:
    job = ProcessingJob(
        source_video_id=plan_set.source_video_id,
        kind=JobKind.TRANSFORMATION_PLANNING,
        transformation_plan_set_id=plan_set.id,
        status=JobStatus.QUEUED,
    )
    session.add(job)
    session.commit()
    return job


class _CancellingProvider(FakePlanningProvider):
    """Cancels the current job from inside the provider call."""

    def __init__(self, plans: Any, session: Session, job_id: object) -> None:
        super().__init__(plans)
        self._session = session
        self._job_id = job_id

    def plan(self, requests: object, tier: str = "ROUTINE") -> object:
        result = super().plan(requests, tier)
        job = self._session.get(ProcessingJob, self._job_id)
        job.status = JobStatus.CANCELLED
        self._session.commit()
        return result


def test_current_claim_cancellation_still_marks_plan_set_cancelled(session: Session) -> None:
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
    provider = _CancellingProvider(
        [make_source_value_plan(str(strategy.id), strategy.strategy_key)],
        session,
        job.id,
    )
    executor = _executor(session, settings, provider)
    executor.set_active_job(job.id)
    with pytest.raises(StageCancelled):
        executor.execute(plan_set.id)
    session.refresh(plan_set)
    session.refresh(job)
    assert plan_set.execution_status is PlanExecutionStatus.CANCELLED
    assert job.status is JobStatus.CANCELLED


def test_planning_input_failure_records_sanitized_diagnostics(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    plan_set = get_or_create_plan_set(session, seed[1], seed[3])
    job = _job(session, plan_set)
    # Remove the prerequisite refinement: input resolution now fails.
    session.delete(seed[2])
    session.commit()
    executor = _executor(session, settings, None)
    executor.set_active_job(job.id)
    with pytest.raises(PlanningInputError):
        executor.execute(plan_set.id)
    session.refresh(job)
    assert job.status is JobStatus.FAILED
    assert job.error_message
    assert "refinement" in job.error_message
    assert "refinement" in (job.error_code or "").casefold() or job.error_code


class _ExplodingProvider:
    hosted_provider = False

    def plan(self, requests: object, tier: str = "ROUTINE") -> object:
        raise RuntimeError("secret key sk-live-123 transcript body leaked")

    def release(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "explode", "model": "boom"}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def raw_call_count(self) -> int:
        return 0


def test_unexpected_service_failure_records_sanitized_diagnostics(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    plan_set = get_or_create_plan_set(session, seed[1], seed[3])
    job = _job(session, plan_set)
    executor = _executor(session, settings, _ExplodingProvider())
    executor.set_active_job(job.id)
    with pytest.raises(RuntimeError):
        executor.execute(plan_set.id)
    session.refresh(job)
    assert job.status is JobStatus.FAILED
    assert job.error_message == "RuntimeError"
    assert "secret" not in (job.error_message or "").casefold()
    assert "sk-live" not in (job.error_message or "")
    assert "transcript" not in (job.error_message or "").casefold()
    session.refresh(plan_set)
    assert plan_set.execution_status is PlanExecutionStatus.FAILED


# --- Finding 1: durable claim-token fencing ----------------------------------


def test_stale_worker_claim_cannot_persist_or_finalize(sqlite_engine: Engine) -> None:
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select
    from sqlalchemy.orm import sessionmaker

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
    with factory() as session_a:
        # Worker A claims the job, then stalls (no provider work yet).
        provider_a = FakePlanningProvider([make_source_value_plan(str(strategy_id), strategy_key)])
        executor_a = _executor(session_a, settings, provider_a)
        executor_a.set_active_job(job_id)
        assert executor_a._claim_job() is True
        version_a = session_a.scalar(
            select(ProcessingJob.claim_version).where(ProcessingJob.id == job_id)
        )
        assert version_a == 1

        with factory() as session_b:
            # A is held past the reclaim threshold; B reclaims and finishes.
            stale_job = session_b.get(ProcessingJob, job_id)
            stale_job.started_at = datetime.now(timezone.utc) - timedelta(seconds=7200)
            # No heartbeat was started (A stalled before provider work), so the
            # abandoned liveness signal is what makes the claim reclaimable.
            stale_job.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=7200)
            session_b.commit()
            provider_b = FakePlanningProvider(
                [make_source_value_plan(str(strategy_id), strategy_key)]
            )
            executor_b = _executor(session_b, settings, provider_b)
            executor_b.set_active_job(job_id)
            executor_b.execute(plan_set_id)
            assert executor_b.skipped_duplicate is False
            assert provider_b.calls == 1
            assert len(list_plans(session_b, plan_set_id)) == 1
            version_b = session_b.scalar(
                select(ProcessingJob.claim_version).where(ProcessingJob.id == job_id)
            )
            assert version_b == 2

        # A's token is now stale: it neither owns the claim nor may mutate.
        assert executor_a._fence_claim() is False
        assert executor_a._claim_is_current() is False
        executor_a.record_failure(RuntimeError("secret leaked transcript"))
        session_a.expire_all()
        resumed = session_a.get(ProcessingJob, job_id)
        assert resumed.status is JobStatus.SUCCEEDED
        assert resumed.error_message is None
        assert len(list_plans(session_a, plan_set_id)) == 1

        # A resuming as a delivery performs zero provider work and persists nothing.
        executor_a.execute(plan_set_id)
        assert executor_a.skipped_duplicate is True
        assert provider_a.calls == 0
        assert len(list_plans(session_a, plan_set_id)) == 1
