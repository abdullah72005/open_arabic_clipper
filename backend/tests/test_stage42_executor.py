"""Stage 4.2 executor, queue, cache, concurrency, and cancellation tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from stage41_support import FakePlanningProvider, run_planning, seed_stage41
from stage42_support import (
    FakeGovernanceProvider,
    FakeGovernanceSettings,
    install_stage42_settings,
    make_source_value_plan,
    with_review_narration,
)

from app.core.enums import (
    GovernanceExecutionStatus,
    GovernanceSemanticOutcome,
    JobKind,
    JobStatus,
    SemanticProviderMode,
)
from app.db.base import Base
from app.models import (
    ProcessingJob,
    TransformationGovernanceResult,
    TransformationGovernanceSet,
)
from app.transformation.governance.executor import (
    GovernanceCancelled,
    build_transformation_governance_executor,
)
from app.transformation.governance.inputs import GovernanceInputError
from app.transformation.governance.queue import (
    get_governance_set_for_candidate,
    get_or_create_governance_set,
    list_results,
    queue_transformation_governance,
    validate_candidate_for_governance,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.workers.tasks.run_transformation_planning.delay", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.workers.tasks.run_transformation_governance.delay", lambda *a, **k: None
    )


def _install(settings: FakeGovernanceSettings) -> None:
    from _pytest.monkeypatch import MonkeyPatch

    install_stage42_settings(MonkeyPatch(), settings)


def _setup(
    session: Session,
    *,
    narration: str = "NONE",
    governance_provider: Any = None,
) -> tuple[FakeGovernanceSettings, tuple[Any, ...], Any]:
    settings = FakeGovernanceSettings(governance_provider=governance_provider)
    _install(settings)
    seed = seed_stage41(session, settings=settings)
    strategy = seed[4][0]
    plan = make_source_value_plan(str(strategy.id), strategy.strategy_key, narration_need=narration)
    if narration == "RECOMMENDED":
        plan = with_review_narration(plan)
    planning_provider = FakePlanningProvider([plan])
    plan_set = run_planning(
        session, settings, seed, planning_provider, mode=SemanticProviderMode.ADAPTIVE
    )
    return settings, seed, plan_set


def _run(
    session: Session,
    settings: FakeGovernanceSettings,
    seed: tuple[Any, ...],
    plan_set: Any,
    *,
    force: bool = False,
) -> TransformationGovernanceSet:
    candidate = seed[1]
    governance_set = get_or_create_governance_set(session, candidate, plan_set)
    executor = build_transformation_governance_executor(session, settings)
    executor.execute(governance_set.id, force=force)
    session.refresh(governance_set)
    return governance_set


def test_strong_plan_governed_without_provider_call(session: Session) -> None:
    provider = FakeGovernanceProvider(auto=True)
    settings, seed, plan_set = _setup(session, governance_provider=provider)
    governance_set = _run(session, settings, seed, plan_set)
    assert provider.calls == 0
    assert governance_set.execution_status is GovernanceExecutionStatus.COMPLETE
    assert (
        governance_set.governance_outcome is GovernanceSemanticOutcome.PLANS_ELIGIBLE_FOR_SELECTION
    )
    results = list_results(session, governance_set.id)
    assert results and all(row.eligible_for_stage4_3 for row in results)


def test_cache_hit_makes_no_repeated_provider_call(session: Session) -> None:
    provider = FakeGovernanceProvider(auto=True)
    settings, seed, plan_set = _setup(
        session, narration="RECOMMENDED", governance_provider=provider
    )
    governance_set = _run(session, settings, seed, plan_set)
    assert provider.calls == 1
    assert governance_set.cache_eligible is True
    # Force re-enters the executor but reuses the accepted checkpoint.
    _run(session, settings, seed, plan_set, force=True)
    assert provider.calls == 1


def test_provider_outage_does_not_fail_candidate(session: Session) -> None:
    provider = FakeGovernanceProvider(auto=True)
    provider.behavior = "outage"
    settings, seed, plan_set = _setup(
        session, narration="RECOMMENDED", governance_provider=provider
    )
    governance_set = _run(session, settings, seed, plan_set)
    assert governance_set.execution_status is GovernanceExecutionStatus.PROVIDER_DEGRADED
    assert governance_set.governance_outcome is GovernanceSemanticOutcome.GOVERNANCE_DEFERRED
    assert governance_set.cache_eligible is False


def test_concurrent_queue_yields_one_active_job(session: Session) -> None:
    provider = FakeGovernanceProvider(auto=True)
    settings, seed, plan_set = _setup(session, governance_provider=provider)
    candidate, resolved_set = validate_candidate_for_governance(session, seed[1].id)
    first = queue_transformation_governance(session, candidate, resolved_set)
    second = queue_transformation_governance(session, candidate, resolved_set)
    assert first.queued is True
    assert second.active is True
    jobs = session.scalars(
        select(ProcessingJob).where(ProcessingJob.kind == JobKind.TRANSFORMATION_GOVERNANCE)
    ).all()
    assert len(jobs) == 1
    assert provider.calls == 0


def test_stale_stage41_input_is_refused_before_provider_work(session: Session) -> None:
    provider = FakeGovernanceProvider(auto=True)
    settings, seed, plan_set = _setup(session, governance_provider=provider)
    assert provider.calls == 0
    plan_set.input_fingerprint = "stale-fingerprint"
    session.commit()
    candidate = seed[1]
    governance_set = get_or_create_governance_set(session, candidate, plan_set)
    executor = build_transformation_governance_executor(session, settings)
    with pytest.raises(GovernanceInputError):
        executor.execute(governance_set.id)
    assert provider.calls == 0


def test_cancelled_job_cannot_persist_final_result(session: Session) -> None:
    provider = FakeGovernanceProvider(auto=True)
    settings, seed, plan_set = _setup(
        session, narration="RECOMMENDED", governance_provider=provider
    )
    candidate, resolved_set = validate_candidate_for_governance(session, seed[1].id)
    outcome = queue_transformation_governance(session, candidate, resolved_set)
    job = session.get(ProcessingJob, outcome.job_id)
    assert job is not None
    job.status = JobStatus.CANCELLED
    session.commit()
    governance_set = get_governance_set_for_candidate(session, seed[1].id)
    executor = build_transformation_governance_executor(session, settings)
    executor.set_active_job(job.id)
    with pytest.raises(GovernanceCancelled):
        executor.execute(governance_set.id)
    session.refresh(governance_set)
    assert governance_set.execution_status is GovernanceExecutionStatus.CANCELLED
    assert list_results(session, governance_set.id) == []
    assert provider.calls == 0


def test_force_reuses_accepted_checkpoint(session: Session) -> None:
    provider = FakeGovernanceProvider(auto=True)
    settings, seed, plan_set = _setup(
        session, narration="RECOMMENDED", governance_provider=provider
    )
    _run(session, settings, seed, plan_set)
    assert provider.calls == 1
    second = FakeGovernanceProvider(auto=True)
    settings2 = FakeGovernanceSettings(governance_provider=second)
    _install(settings2)
    _run(session, settings2, seed, plan_set, force=True)
    assert second.calls == 0


def test_governance_set_is_unique_per_plan_set(session: Session) -> None:
    settings, seed, plan_set = _setup(session)
    candidate = seed[1]
    first = get_or_create_governance_set(session, candidate, plan_set)
    second = get_or_create_governance_set(session, candidate, plan_set)
    assert first.id == second.id


def test_results_are_independent_per_plan(session: Session) -> None:
    settings, seed, plan_set = _setup(session)
    governance_set = _run(session, settings, seed, plan_set)
    results = list_results(session, governance_set.id)
    plan_ids = {row.transformation_plan_id for row in results}
    assert len(plan_ids) == len(results)
    assert all(isinstance(row, TransformationGovernanceResult) for row in results)


def test_status_eligibility_constraint_is_enforced(session: Session) -> None:
    from sqlalchemy.exc import IntegrityError

    from app.core.enums import GovernancePlanStatus

    settings, seed, plan_set = _setup(session)
    governance_set = _run(session, settings, seed, plan_set)
    row = list_results(session, governance_set.id)[0]
    row.status = GovernancePlanStatus.APPROVED_FOR_SELECTION
    row.eligible_for_stage4_3 = False
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_platform_policy_profile_change_invalidates_governance(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.transformation.governance import fingerprints as fingerprints_module

    settings, seed, plan_set = _setup(session)
    governance_set = _run(session, settings, seed, plan_set)
    executor = build_transformation_governance_executor(session, settings)
    before = governance_set.input_fingerprint
    assert executor.input_fingerprint(governance_set) == before
    monkeypatch.setattr(
        fingerprints_module,
        "platform_policy_payload",
        lambda: {"policy_profile_version": "changed-profile"},
    )
    assert executor.input_fingerprint(governance_set) != before


def test_governance_fingerprint_excludes_tts_and_rendering_settings(session: Session) -> None:
    import json

    from app.transformation.governance.fingerprints import build_governance_input_payload
    from app.transformation.governance.inputs import build_governance_inputs

    settings, seed, plan_set = _setup(session)
    inputs = build_governance_inputs(
        session,
        seed[1],
        plan_set,
        settings,
        settings.stage42_config(),
        settings.stage41_config(),
    )
    payload = build_governance_input_payload(
        inputs=inputs,
        config=settings.stage42_config(),
        provider_mode="adaptive",
        provider_identity={"provider": "fake"},
    )
    blob = json.dumps(payload).casefold()
    for token in ("tts", "voice", "render", "publish", "caption_font", "resolution"):
        assert token not in blob


def _expire_running_job(session: Session, job: ProcessingJob, *, stale: bool) -> None:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    age = timedelta(seconds=7200) if stale else timedelta(seconds=0)
    job.status = JobStatus.RUNNING
    job.claim_version = 3
    job.started_at = now - age
    job.heartbeat_at = now - age
    session.commit()


def test_queue_recovers_abandoned_running_job(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeGovernanceProvider(auto=True)
    settings, seed, plan_set = _setup(session, governance_provider=provider)
    candidate, resolved_set = validate_candidate_for_governance(session, seed[1].id)
    first = queue_transformation_governance(session, candidate, resolved_set)
    job = session.get(ProcessingJob, first.job_id)
    assert job is not None
    _expire_running_job(session, job, stale=True)

    dispatched: list[tuple[object, object]] = []
    monkeypatch.setattr(
        "app.transformation.governance.queue._dispatch",
        lambda sid, jid, force: dispatched.append((sid, jid)),
    )
    outcome = queue_transformation_governance(session, candidate, resolved_set)
    assert outcome.skipped_reason == "RECOVERED_STALE_RUNNING"
    assert outcome.active is True
    assert dispatched == [(first.governance_set_id, job.id)]
    assert provider.calls == 0


def test_queue_does_not_recover_live_running_job(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeGovernanceProvider(auto=True)
    settings, seed, plan_set = _setup(session, governance_provider=provider)
    candidate, resolved_set = validate_candidate_for_governance(session, seed[1].id)
    first = queue_transformation_governance(session, candidate, resolved_set)
    job = session.get(ProcessingJob, first.job_id)
    assert job is not None
    _expire_running_job(session, job, stale=False)

    dispatched: list[object] = []
    monkeypatch.setattr(
        "app.transformation.governance.queue._dispatch",
        lambda sid, jid, force: dispatched.append((sid, jid)),
    )
    outcome = queue_transformation_governance(session, candidate, resolved_set)
    assert outcome.active is True
    assert outcome.skipped_reason is None
    assert dispatched == []


def test_redelivered_job_makes_no_duplicate_provider_call(session: Session) -> None:
    provider = FakeGovernanceProvider(auto=True)
    settings, seed, plan_set = _setup(
        session, narration="RECOMMENDED", governance_provider=provider
    )
    candidate, resolved_set = validate_candidate_for_governance(session, seed[1].id)
    first = queue_transformation_governance(session, candidate, resolved_set)
    job = session.get(ProcessingJob, first.job_id)
    assert job is not None

    executor = build_transformation_governance_executor(session, settings)
    executor.set_active_job(job.id)
    executor.execute(first.governance_set_id)
    assert provider.calls == 1

    redelivered = build_transformation_governance_executor(session, settings)
    redelivered.set_active_job(job.id)
    redelivered.execute(first.governance_set_id)
    assert redelivered.skipped_duplicate is True
    assert provider.calls == 1


def test_stale_worker_cannot_overwrite_newer_claim(session: Session) -> None:
    provider = FakeGovernanceProvider(auto=True)
    settings, seed, plan_set = _setup(
        session, narration="RECOMMENDED", governance_provider=provider
    )
    candidate, resolved_set = validate_candidate_for_governance(session, seed[1].id)
    first = queue_transformation_governance(session, candidate, resolved_set)
    job = session.get(ProcessingJob, first.job_id)
    assert job is not None
    _expire_running_job(session, job, stale=True)

    newer = build_transformation_governance_executor(session, settings)
    newer.set_active_job(job.id)
    newer.execute(first.governance_set_id)
    session.refresh(job)
    assert job.claim_version >= 4

    stale_worker = build_transformation_governance_executor(session, settings)
    stale_worker.set_active_job(job.id)
    stale_worker._job_owner = True  # type: ignore[attr-defined]
    stale_worker._claim_version = 3  # type: ignore[attr-defined]
    assert stale_worker._claim_guard(require_running=True) is False  # type: ignore[attr-defined]
    assert stale_worker.claim_lost is True


def test_stage4_3_handoff_reports_stale_governance(session: Session) -> None:
    import json

    from app.models import TransformationPlan
    from app.transformation.governance.handoff import build_stage4_3_handoff

    settings, seed, plan_set = _setup(session)
    _run(session, settings, seed, plan_set)

    fresh = build_stage4_3_handoff(session, seed[1].id)
    assert fresh is not None
    assert fresh["governance_set"]["freshness"] == "VERIFIED_CURRENT"
    assert fresh["governance_set"]["current"] is True
    assert fresh["governance_set"]["stale"] is False
    assert fresh["plans"]
    assert all(plan["governance"]["eligible_for_stage4_3"] is True for plan in fresh["plans"])

    row = session.scalars(
        select(TransformationPlan).where(TransformationPlan.plan_set_id == plan_set.id)
    ).first()
    assert row is not None
    row.blocks = [dict(block) for block in (row.blocks or [])] + [
        {"index": 99, "block_type": "TRANSITION", "estimated_duration": 1.0}
    ]
    session.commit()

    stale = build_stage4_3_handoff(session, seed[1].id)
    assert stale is not None
    assert stale["governance_set"]["freshness"] == "STALE"
    assert stale["governance_set"]["stale"] is True
    assert stale["governance_set"]["current"] is False
    assert stale["plans"]
    assert all(plan["governance"]["eligible_for_stage4_3"] is False for plan in stale["plans"])
    serialized = json.dumps(stale)
    assert "selected_plan_id" not in serialized
    assert stale["stage4_3_implemented"] is True


def test_stage4_3_handoff_missing_fingerprint_is_unverifiable(session: Session) -> None:
    from app.transformation.governance.handoff import build_stage4_3_handoff

    settings, seed, plan_set = _setup(session)
    governance_set = _run(session, settings, seed, plan_set)
    governance_set.input_fingerprint = ""
    session.commit()

    handoff = build_stage4_3_handoff(session, seed[1].id)
    assert handoff is not None
    assert handoff["governance_set"]["freshness"] == "UNVERIFIABLE"
    assert handoff["governance_set"]["current"] is False
    assert handoff["plans"]
    assert all(plan["governance"]["eligible_for_stage4_3"] is False for plan in handoff["plans"])


def test_stage4_3_handoff_fingerprint_failure_is_unverifiable(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.transformation.governance import executor as executor_module
    from app.transformation.governance.handoff import build_stage4_3_handoff

    settings, seed, plan_set = _setup(session)
    _run(session, settings, seed, plan_set)

    class _Boom:
        def input_fingerprint(self, governance_set: object) -> str:
            raise RuntimeError("fingerprint recomputation failed")

    monkeypatch.setattr(
        executor_module,
        "build_transformation_governance_executor",
        lambda session, settings: _Boom(),
    )
    handoff = build_stage4_3_handoff(session, seed[1].id)
    assert handoff is not None
    assert handoff["governance_set"]["freshness"] == "UNVERIFIABLE"
    assert handoff["plans"]
    assert all(plan["governance"]["eligible_for_stage4_3"] is False for plan in handoff["plans"])


def test_stage4_3_handoff_not_current_is_ineligible(session: Session) -> None:
    from app.core.enums import GovernanceExecutionStatus
    from app.transformation.governance.handoff import build_stage4_3_handoff

    settings, seed, plan_set = _setup(session)
    governance_set = _run(session, settings, seed, plan_set)
    governance_set.execution_status = GovernanceExecutionStatus.FAILED
    session.commit()

    handoff = build_stage4_3_handoff(session, seed[1].id)
    assert handoff is not None
    assert handoff["governance_set"]["freshness"] == "NOT_CURRENT"
    assert handoff["plans"]
    assert all(plan["governance"]["eligible_for_stage4_3"] is False for plan in handoff["plans"])
