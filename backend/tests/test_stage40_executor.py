"""Focused Stage 4.0 executor, queue, persistence, and idempotency tests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session
from stage40_support import (
    FakeStage40Settings,
    install_stage40_settings,
)

from app.core.enums import (
    CandidateDisposition,
    ContentType,
    ExternalFactRequirement,
    JobKind,
    JobStatus,
    OriginalityRisk,
    RefinementPriority,
    RefinementStatus,
    RightsRisk,
    SemanticProviderMode,
    StrategyDisposition,
    SubstantiveValueKind,
    TransformationEligibilityOutcome,
    TransformationExecutionStatus,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.db.base import Base
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    ProcessingJob,
    SourceVideo,
    Transcript,
    TransformationEligibilityAnalysis,
    TransformationStrategyCandidate,
)
from app.transformation.executor import (
    TransformationCancelled,
    TransformationEligibilityExecutor,
    build_transformation_executor,
)
from app.transformation.providers import (
    TransformationProviderError,
    TransformationStrategyRequest,
)
from app.transformation.queue import (
    TransformationQueueError,
    TransformationQueueOutcome,
    queue_transformation_analysis,
    validate_candidate_for_transformation,
)
from app.transformation.types import (
    TransformationProviderResult,
    TransformationProviderStrategy,
)

TRANSCRIPT = (
    "The guest argues that remote work collapsed productivity because managers lost "
    "the ability to mentor junior staff and the data shows promotion rates fell sharply."
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def seed(session: Session) -> tuple[SourceVideo, ClipCandidate, CandidateRefinement]:
    source = SourceVideo(source_uri="/tmp/stage40.mp4", content_hash="stage40-hash")
    session.add(source)
    session.flush()
    transcript = Transcript(
        source_video_id=source.id,
        whisper_model="small",
        transcription_options={},
        input_fingerprint="t" * 64,
        raw_text=TRANSCRIPT,
        normalized_text=TRANSCRIPT,
        corrected_text=TRANSCRIPT,
        final_text=TRANSCRIPT,
        segments=[
            {"start": 0.0, "end": 20.0, "text": "The host introduces the guest and the topic."},
            {"start": 20.0, "end": 45.0, "text": TRANSCRIPT, "corrected_text": TRANSCRIPT},
            {"start": 45.0, "end": 90.0, "text": "The guest continues with more detail."},
            {"start": 90.0, "end": 120.0, "text": "The interview concludes."},
        ],
        duration=120.0,
        language="en",
    )
    session.add(transcript)
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key="stage40-candidate",
        disposition=CandidateDisposition.CANDIDATE,
        start_time=20.0,
        end_time=45.0,
        start_segment_index=1,
        end_segment_index=1,
        segment_indexes=[1],
        primary_content_type=ContentType.INTERVIEW_INSIGHT,
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
        refined_start=20.5,
        refined_end=44.5,
        automatic_transcript=TRANSCRIPT,
        final_transcript=TRANSCRIPT,
        word_timestamps=[
            {"text": "remote", "start": 21.0, "end": 21.3, "probability": 0.9},
            {"text": "work", "start": 21.3, "end": 21.6, "probability": 0.9},
        ],
        confidence=0.9,
        quality_level="CANDIDATE",
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.8,
        output_fingerprint="refinement-output-fp",
    )
    session.add(refinement)
    session.commit()
    return source, candidate, refinement


class FakeProvider:
    provider_name = "fake"

    def __init__(self, behavior: str = "ok") -> None:
        self.model = "fake-model"
        self.calls = 0
        self.behavior = behavior

    def select_tier(self, requests: Sequence[TransformationStrategyRequest]) -> str:
        return "ROUTINE"

    def discover(
        self, requests: Sequence[TransformationStrategyRequest]
    ) -> dict[str, TransformationProviderResult]:
        self.calls += 1
        if self.behavior == "rate_limited":
            raise TransformationProviderError("RATE_LIMITED")
        if self.behavior == "outage":
            raise TransformationProviderError("PROVIDER_ERROR")
        strategy = TransformationProviderStrategy(
            strategy_type=TransformationStrategyType.CONTEXT_HOOK,
            disposition=StrategyDisposition.RECOMMENDED,
            intensity=TransformationIntensity.MINIMAL,
            direction_summary="Add the missing economic context behind the mentorship claim",
            added_value_focus="Supply context on why promotion rates fell",
            substantive_value_kind=SubstantiveValueKind.MISSING_CONTEXT,
            preservation_requirements=("Keep the source moment as the hero.",),
            external_verification_requirement=ExternalFactRequirement.NOT_REQUIRED,
            verification_requirements=(),
            rejection_reasons=(),
            confidence=0.7,
            retention_preservation=0.8,
            source_moment_damage_risk=0.2,
            added_value_density=0.6,
            originality_potential=0.7,
            source_dominance_risk=0.42,
            generic_filler_risk=0.2,
            redundant_commentary_risk=0.2,
            template_staleness_risk=0.2,
        )
        return {
            requests[0].candidate_id: TransformationProviderResult(
                candidate_id=requests[0].candidate_id,
                strategies=(strategy,),
                confidence=0.7,
            )
        }

    def release(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "fake", "model": self.model}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


def _executor(
    session: Session,
    provider: FakeProvider | None = None,
    mode: SemanticProviderMode = SemanticProviderMode.DETERMINISTIC,
) -> TransformationEligibilityExecutor:
    return TransformationEligibilityExecutor(session=session, provider=provider, mode=mode)


def _analysis(session: Session, candidate: ClipCandidate) -> TransformationEligibilityAnalysis:
    from app.transformation.queue import get_or_create_analysis

    return get_or_create_analysis(session, candidate)


def test_executor_persists_eligible_analysis_and_strategies(session: Session) -> None:
    _, candidate, _ = seed(session)
    analysis = _analysis(session, candidate)
    _executor(session).execute(analysis.id)
    session.refresh(analysis)
    assert analysis.execution_status is TransformationExecutionStatus.COMPLETE
    assert (
        analysis.eligibility_outcome is TransformationEligibilityOutcome.ELIGIBLE_FOR_TRANSFORMATION
    )
    assert analysis.refinement_priority == "CANDIDATE"
    rows = session.scalars(
        select(TransformationStrategyCandidate).where(
            TransformationStrategyCandidate.analysis_id == analysis.id,
            TransformationStrategyCandidate.is_current.is_(True),
        )
    ).all()
    assert rows
    assert all(row.strategy_fingerprint for row in rows)


def test_candidate_grade_input_is_sufficient_and_final_not_required(session: Session) -> None:
    _, candidate, _ = seed(session)
    analysis = _analysis(session, candidate)
    _executor(session).execute(analysis.id)
    session.refresh(analysis)
    assert analysis.refinement_quality_level == "CANDIDATE"
    assert analysis.eligibility_outcome is not None


def test_queue_rejects_candidate_without_refinement(session: Session) -> None:
    source, candidate, refinement = seed(session)
    session.delete(refinement)
    session.commit()
    with pytest.raises(TransformationQueueError):
        validate_candidate_for_transformation(session, candidate.id)


def test_queue_rejects_stale_candidate(session: Session) -> None:
    _, candidate, _ = seed(session)
    candidate.is_current = False
    session.commit()
    with pytest.raises(TransformationQueueError):
        validate_candidate_for_transformation(session, candidate.id)


def test_repeated_execution_is_idempotent_and_stable(session: Session) -> None:
    _, candidate, _ = seed(session)
    analysis = _analysis(session, candidate)
    executor = _executor(session)
    executor.execute(analysis.id)
    session.refresh(analysis)
    first_ids = {
        row.strategy_type: row.id
        for row in session.scalars(
            select(TransformationStrategyCandidate).where(
                TransformationStrategyCandidate.analysis_id == analysis.id
            )
        )
    }
    output_fp = analysis.output_fingerprint
    executor.execute(analysis.id)
    session.refresh(analysis)
    second_ids = {
        row.strategy_type: row.id
        for row in session.scalars(
            select(TransformationStrategyCandidate).where(
                TransformationStrategyCandidate.analysis_id == analysis.id
            )
        )
    }
    assert first_ids == second_ids
    assert analysis.output_fingerprint == output_fp
    assert session.scalar(select(func.count()).select_from(TransformationEligibilityAnalysis)) == 1


def test_force_reuses_accepted_hosted_result_without_recalling(session: Session) -> None:
    _, candidate, _ = seed(session)
    analysis = _analysis(session, candidate)
    provider = FakeProvider()
    executor = _executor(session, provider, SemanticProviderMode.ADAPTIVE)
    executor.execute(analysis.id)
    session.refresh(analysis)
    assert provider.calls == 1
    assert analysis.provider_status in {"PROVIDER_OK", "REUSED"}
    executor.execute(analysis.id, force=True)
    session.refresh(analysis)
    assert provider.calls == 1


def test_degraded_provider_keeps_deterministic_and_is_not_cache_eligible(
    session: Session,
) -> None:
    _, candidate, _ = seed(session)
    analysis = _analysis(session, candidate)
    provider = FakeProvider(behavior="outage")
    executor = _executor(session, provider, SemanticProviderMode.ADAPTIVE)
    executor.execute(analysis.id)
    session.refresh(analysis)
    assert analysis.execution_status is TransformationExecutionStatus.PROVIDER_DEGRADED
    assert analysis.cache_eligible is False
    assert analysis.provider_status == "PROVIDER_DEGRADED"
    # Deterministic recommendations survive.
    assert any(
        row.is_current and row.disposition is StrategyDisposition.RECOMMENDED
        for row in session.scalars(
            select(TransformationStrategyCandidate).where(
                TransformationStrategyCandidate.analysis_id == analysis.id
            )
        )
    )


def test_rate_limited_provider_is_safe(session: Session) -> None:
    _, candidate, _ = seed(session)
    analysis = _analysis(session, candidate)
    provider = FakeProvider(behavior="rate_limited")
    _executor(session, provider, SemanticProviderMode.ADAPTIVE).execute(analysis.id)
    session.refresh(analysis)
    assert analysis.provider_status == "RATE_LIMITED"
    assert analysis.execution_status is TransformationExecutionStatus.PROVIDER_DEGRADED


def test_missing_provider_key_still_deterministic(session: Session) -> None:
    _, candidate, _ = seed(session)
    analysis = _analysis(session, candidate)
    # adaptive with provider None == key missing
    _executor(session, None, SemanticProviderMode.ADAPTIVE).execute(analysis.id)
    session.refresh(analysis)
    assert analysis.eligibility_outcome is not None
    assert analysis.provider_status in {"DETERMINISTIC", "NO_PROVIDER"}


def test_cancellation_stays_cancelled_and_schedules_nothing(session: Session) -> None:
    _, candidate, _ = seed(session)
    analysis = _analysis(session, candidate)
    job = ProcessingJob(
        source_video_id=candidate.source_video_id,
        kind=JobKind.TRANSFORMATION_ELIGIBILITY,
        transformation_analysis_id=analysis.id,
        status=JobStatus.CANCELLED,
    )
    session.add(job)
    session.commit()
    executor = _executor(session)
    executor.set_active_job(job.id)
    with pytest.raises(TransformationCancelled):
        executor.execute(analysis.id)
    session.refresh(analysis)
    assert analysis.execution_status is TransformationExecutionStatus.CANCELLED
    assert analysis.eligibility_outcome is None


def test_unusable_final_does_not_hide_usable_candidate(session: Session) -> None:
    source, candidate, candidate_ref = seed(session)
    queued_final = CandidateRefinement(
        source_video_id=source.id,
        clip_candidate_id=candidate.id,
        priority=RefinementPriority.FINAL_CLIP,
        status=RefinementStatus.QUEUED,
        coarse_start=20.0,
        coarse_end=45.0,
        context_start=12.0,
        context_end=53.0,
        refined_start=None,
        refined_end=None,
        automatic_transcript="",
        final_transcript="",
        confidence=0.0,
    )
    session.add(queued_final)
    session.commit()
    from app.transformation.inputs import resolve_effective_refinement

    resolved = resolve_effective_refinement(session, candidate)
    assert resolved is not None
    assert resolved.id == candidate_ref.id
    assert resolved.priority is RefinementPriority.CANDIDATE


def test_fingerprint_invalidates_on_transcript_and_boundary_change(session: Session) -> None:
    _, candidate, refinement = seed(session)
    executor = _executor(session)
    base = executor.input_fingerprint(candidate)
    refinement.final_transcript = TRANSCRIPT + " An extra measured sentence."
    session.commit()
    changed_text = executor.input_fingerprint(candidate)
    refinement.refined_end = 40.0
    session.commit()
    changed_boundary = executor.input_fingerprint(candidate)
    assert len({base, changed_text, changed_boundary}) == 3


def test_policy_change_invalidates_fingerprint(session: Session) -> None:
    _, candidate, _ = seed(session)
    from app.transformation.policy import DEFAULT_CONFIG

    base = _executor(session).input_fingerprint(candidate)
    changed = TransformationEligibilityExecutor(
        session=session, config=DEFAULT_CONFIG.with_overrides(min_added_value_density=0.9)
    ).input_fingerprint(candidate)
    assert base != changed


def test_unrelated_rendering_settings_are_not_in_fingerprint(session: Session) -> None:
    _, candidate, refinement = seed(session)
    from app.transformation.eligibility import derive_source_moment, transformation_necessity
    from app.transformation.fingerprints import build_input_fingerprint_payload
    from app.transformation.inputs import build_transformation_inputs
    from app.transformation.policy import DEFAULT_CONFIG
    from app.transformation.service import compute_provider_route

    inputs = build_transformation_inputs(session, candidate, refinement, DEFAULT_CONFIG)
    structure = derive_source_moment(inputs, DEFAULT_CONFIG)
    route = compute_provider_route(inputs, structure, transformation_necessity(inputs))
    payload = build_input_fingerprint_payload(
        inputs=inputs,
        config=DEFAULT_CONFIG,
        provider_identity={},
        provider_mode="deterministic",
        provider_route=route,
    )
    serialized = str(payload).casefold()
    assert "render" not in serialized
    assert "publish" not in serialized
    assert "frontend" not in serialized


def test_concurrent_active_queue_does_not_duplicate(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[object, ...]] = []

    def _delay(*args: object, **_kwargs: object) -> None:
        calls.append(args)

    monkeypatch.setattr("app.workers.tasks.run_transformation_analysis.delay", _delay)
    install_stage40_settings(
        monkeypatch, FakeStage40Settings(mode=SemanticProviderMode.DETERMINISTIC)
    )
    _, candidate, _ = seed(session)
    first = queue_transformation_analysis(session, candidate)
    second = queue_transformation_analysis(session, candidate)
    assert first.queued is True
    assert second.active is True
    assert second.analysis_id == first.analysis_id
    assert len(calls) == 1
    assert (
        session.scalar(
            select(func.count())
            .select_from(ProcessingJob)
            .where(ProcessingJob.kind == JobKind.TRANSFORMATION_ELIGIBILITY)
        )
        == 1
    )


def test_queue_cache_hit_after_execution(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.workers.tasks.run_transformation_analysis.delay", lambda *a, **k: None)
    settings = FakeStage40Settings(mode=SemanticProviderMode.DETERMINISTIC)
    install_stage40_settings(monkeypatch, settings)
    _, candidate, _ = seed(session)
    outcome = queue_transformation_analysis(session, candidate)
    queued_job = session.get(ProcessingJob, outcome.job_id)
    queued_job.status = JobStatus.SUCCEEDED
    session.commit()
    analysis = _analysis(session, candidate)
    build_transformation_executor(session, settings).execute(outcome.analysis_id)
    again = queue_transformation_analysis(session, candidate)
    assert again.cached is True
    assert again.job_id is None
    session.refresh(analysis)
    assert analysis.cache_eligible is True


def _run_via_settings(
    session: Session, settings: FakeStage40Settings, candidate_id: object
) -> None:
    from app.transformation.queue import get_or_create_analysis

    candidate = session.get(ClipCandidate, candidate_id)
    analysis = get_or_create_analysis(session, candidate)
    build_transformation_executor(session, settings).execute(analysis.id)


def test_adaptive_cache_hit_reuses_output_without_second_provider_call(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.workers.tasks.run_transformation_analysis.delay", lambda *a, **k: None)
    provider = FakeProvider()
    settings = FakeStage40Settings(provider=provider, mode=SemanticProviderMode.ADAPTIVE)
    install_stage40_settings(monkeypatch, settings)
    _, candidate, _ = seed(session)
    _run_via_settings(session, settings, candidate.id)
    analysis = _analysis(session, candidate)
    session.refresh(analysis)
    assert analysis.cache_eligible is True
    assert provider.calls == 1
    again = queue_transformation_analysis(session, candidate)
    assert again.cached is True
    assert again.job_id is None
    assert provider.calls == 1
    assert (
        session.scalar(
            select(func.count())
            .select_from(ProcessingJob)
            .where(ProcessingJob.kind == JobKind.TRANSFORMATION_ELIGIBILITY)
        )
        == 0
    )


def test_local_only_cache_hit_reuses_output_without_second_provider_call(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.workers.tasks.run_transformation_analysis.delay", lambda *a, **k: None)
    provider = FakeProvider()
    settings = FakeStage40Settings(provider=provider, mode=SemanticProviderMode.LOCAL_ONLY)
    install_stage40_settings(monkeypatch, settings)
    _, candidate, _ = seed(session)
    _run_via_settings(session, settings, candidate.id)
    analysis = _analysis(session, candidate)
    session.refresh(analysis)
    assert analysis.cache_eligible is True
    assert provider.calls == 1
    again = queue_transformation_analysis(session, candidate)
    assert again.cached is True
    assert provider.calls == 1


def test_deterministic_cache_hit_reuses_output(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.workers.tasks.run_transformation_analysis.delay", lambda *a, **k: None)
    settings = FakeStage40Settings(mode=SemanticProviderMode.DETERMINISTIC)
    install_stage40_settings(monkeypatch, settings)
    _, candidate, _ = seed(session)
    _run_via_settings(session, settings, candidate.id)
    analysis = _analysis(session, candidate)
    session.refresh(analysis)
    assert analysis.cache_eligible is True
    again = queue_transformation_analysis(session, candidate)
    assert again.cached is True
    assert again.job_id is None


def test_cache_misses_when_provider_identity_changes(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.workers.tasks.run_transformation_analysis.delay", lambda *a, **k: None)
    provider = FakeProvider()
    settings = FakeStage40Settings(provider=provider, mode=SemanticProviderMode.ADAPTIVE)
    install_stage40_settings(monkeypatch, settings)
    _, candidate, _ = seed(session)
    _run_via_settings(session, settings, candidate.id)
    # A changed provider runtime identity must invalidate the cache.
    changed = FakeProvider()
    changed.model = "fake-model-v2"
    install_stage40_settings(
        monkeypatch, FakeStage40Settings(provider=changed, mode=SemanticProviderMode.ADAPTIVE)
    )
    again = queue_transformation_analysis(session, candidate)
    assert again.cached is False
    assert again.queued is True


def test_concurrent_two_session_queue_creates_one_analysis_and_one_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'concurrent40.sqlite3'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    with factory() as setup:
        _, candidate, _ = seed(setup)
        candidate_id = candidate.id

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "app.workers.tasks.run_transformation_analysis.delay",
        lambda *a, **k: calls.append(a),
    )
    install_stage40_settings(
        monkeypatch, FakeStage40Settings(mode=SemanticProviderMode.DETERMINISTIC)
    )
    barrier = threading.Barrier(2)
    outcomes: list[TransformationQueueOutcome] = []
    errors: list[BaseException] = []

    def worker() -> None:
        with factory() as worker_session:
            candidate = worker_session.get(ClipCandidate, candidate_id)
            try:
                barrier.wait(timeout=30)
                outcomes.append(queue_transformation_analysis(worker_session, candidate))
            except BaseException as error:  # noqa: BLE001 - captured for assertion
                errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    assert len(outcomes) == 2
    analysis_ids = {outcome.analysis_id for outcome in outcomes}
    assert len(analysis_ids) == 1
    with factory() as verify:
        from app.models import TransformationEligibilityAnalysis

        analyses = verify.scalars(select(TransformationEligibilityAnalysis)).all()
        assert len(analyses) == 1
        active_jobs = verify.scalars(
            select(ProcessingJob).where(
                ProcessingJob.kind == JobKind.TRANSFORMATION_ELIGIBILITY,
                ProcessingJob.status.in_({JobStatus.QUEUED, JobStatus.RUNNING}),
            )
        ).all()
        assert len(active_jobs) == 1
        assert len(calls) == 1
    engine.dispose()


def test_accepted_hosted_output_survives_later_outage(session: Session) -> None:
    _, candidate, _ = seed(session)
    analysis = _analysis(session, candidate)
    provider = FakeProvider()
    executor = _executor(session, provider, SemanticProviderMode.ADAPTIVE)
    executor.execute(analysis.id)
    session.refresh(analysis)
    assert provider.calls == 1
    # Simulate a later outage: the force rerun must reuse accepted output.
    provider.behavior = "outage"
    executor.execute(analysis.id, force=True)
    session.refresh(analysis)
    assert provider.calls == 1
    assert analysis.provider_status == "REUSED"
    assert analysis.cache_eligible is True


def test_deterministic_results_survive_bad_provider_direction(session: Session) -> None:
    from app.transformation.types import (
        TransformationProviderResult,
        TransformationProviderStrategy,
    )

    class BadProvider(FakeProvider):
        def discover(
            self, requests: Sequence[TransformationStrategyRequest]
        ) -> dict[str, TransformationProviderResult]:
            self.calls += 1
            strategy = TransformationProviderStrategy(
                strategy_type=TransformationStrategyType.SUMMARY,
                disposition=StrategyDisposition.RECOMMENDED,
                intensity=TransformationIntensity.STRONG,
                direction_summary="You won't believe how this ends",
                added_value_focus="Fake drama over the same clip",
                substantive_value_kind=SubstantiveValueKind.SYNTHESIS,
                preservation_requirements=(),
                external_verification_requirement=ExternalFactRequirement.NOT_REQUIRED,
                verification_requirements=(),
                rejection_reasons=(),
                confidence=0.9,
                retention_preservation=0.5,
                source_moment_damage_risk=0.9,
                added_value_density=0.5,
                originality_potential=0.5,
                source_dominance_risk=0.5,
                generic_filler_risk=0.5,
                redundant_commentary_risk=0.5,
                template_staleness_risk=0.5,
            )
            return {
                requests[0].candidate_id: TransformationProviderResult(
                    candidate_id=requests[0].candidate_id, strategies=(strategy,), confidence=0.9
                )
            }

    _, candidate, _ = seed(session)
    analysis = _analysis(session, candidate)
    _executor(session, BadProvider(), SemanticProviderMode.ADAPTIVE).execute(analysis.id)
    session.refresh(analysis)
    assert analysis.execution_status is TransformationExecutionStatus.COMPLETE
    rows = session.scalars(
        select(TransformationStrategyCandidate).where(
            TransformationStrategyCandidate.analysis_id == analysis.id,
        )
    ).all()
    provider_rows = [row for row in rows if row.origin.value == "PROVIDER"]
    assert provider_rows
    assert all(row.disposition is StrategyDisposition.REJECTED for row in provider_rows)
