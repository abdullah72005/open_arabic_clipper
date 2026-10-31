"""Focused tests for the one-shot guarded recovery continuation.

These tests use the same isolated admitted checkpoint as the recovery suite and
exercise the operator runner/CLI, not a test-only executor path. No model or
provider is constructed; the deny-only ASR collaborator proves the runtime gate.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from test_stage35_recovery import (
    _APPROVED_TOKEN_COUNT,
    _CLIP_END,
    _CLIP_START,
    _OPTIONS,
    _seed_recovery,
    _service,
)

from app.core.enums import JobKind, JobStatus, RefinementStatus
from app.db.base import Base
from app.models import CandidateRefinement, ProcessingJob
from app.refinement.continuation import (
    ContinuationConfigurationError,
    execute_guarded_job,
    prepare_continuation_job,
    run_guarded_continuation,
)
from app.refinement.policy import DEFAULT_CONFIG
from app.runtime.heavy_model_lease import NoopHeavyModelLeaseFactory
from app.services.storage import StorageService


class _ContinuationSettings:
    """Minimal production-settings surface used by the real factory."""

    def __init__(
        self,
        storage_root: Path,
        *,
        options: object = _OPTIONS,
        routing_mode: str = "local_only",
        local_qwen_enabled: bool = False,
        hosted_key_present: bool = False,
    ) -> None:
        self.storage_root = storage_root
        self.whisper_model = "large-v3-turbo"
        self.whisper_device = "cpu"
        self.refinement_candidate_beam_size = 5
        self.refinement_final_beam_size = 8
        self.refinement_routing_mode = routing_mode
        self.local_qwen_enabled = local_qwen_enabled
        self.gemini_api_key_present = hosted_key_present
        self._options = options

    def targeted_transcription_options(self, _priority: str) -> object:
        return self._options

    def stage35_config(self) -> object:
        return DEFAULT_CONFIG

    def heavy_model_lease_factory(self) -> object:
        return NoopHeavyModelLeaseFactory()

    def hosted_transcription_provider(self) -> None:
        return None

    def adjudication_provider(self) -> None:
        return None

    def gemini_admission_controller(self) -> object:
        return object()

    def contextual_reconstructor(self) -> object:
        raise AssertionError("Qwen must not be constructed for this continuation")


@pytest.fixture
def session(sqlite_engine) -> Session:  # type: ignore[no-untyped-def]
    Base.metadata.create_all(sqlite_engine)
    factory = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    return factory()


@pytest.fixture
def storage(tmp_path: Path) -> StorageService:
    root = tmp_path / "storage"
    root.mkdir(exist_ok=True)
    return StorageService(root)


def _wire_task(
    monkeypatch: pytest.MonkeyPatch,
    session: Session,
    settings: _ContinuationSettings,
) -> sessionmaker:  # type: ignore[type-arg]
    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr("app.workers.tasks.get_settings", lambda: settings)
    monkeypatch.setattr("app.workers.tasks.create_session_factory", lambda: factory)
    return factory


def test_guarded_continuation_reuses_checkpoint_without_model_call(
    session: Session,
    storage: StorageService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    _service(session, storage).apply(expectation)
    session.commit()
    session.refresh(refinement)
    settings = _ContinuationSettings(storage.storage_root)
    factory = _wire_task(monkeypatch, session, settings)

    result = run_guarded_continuation(session, storage, settings, expectation)

    assert result.succeeded is True
    assert result.cancelled is False
    assert result.denial_calls == 0
    assert result.job_status == JobStatus.SUCCEEDED.value
    assert result.refinement_status == RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW.value
    assert result.metrics.get("component_reuse") == 1
    assert result.metrics.get("local_asr_runs") is None
    with factory() as verify:
        row = verify.get(CandidateRefinement, refinement.id)
        job = verify.get(ProcessingJob, result.job_id)
        assert job is not None and job.status is JobStatus.SUCCEEDED
        assert row is not None
        assert row.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
        assert row.manual_transcript is None
        assert row.cache_eligible is False
        assert len(row.final_transcript.split()) == _APPROVED_TOKEN_COUNT
        assert row.refined_start == pytest.approx(_CLIP_START)
        assert row.refined_end == pytest.approx(_CLIP_END)
        local = [
            item for item in row.transcript_evidence if item.get("kind") == "TARGETED_LOCAL_ASR"
        ]
        assert len(local) == 1
        assert local[0]["state"] == "REJECTED"
        assert local[0]["entity_metadata"]["raw_word_record_count"] == 107
        assert "الجو" not in row.final_transcript


def test_guarded_continuation_refuses_reuse_miss_before_model_work(
    session: Session,
    storage: StorageService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    _service(session, storage).apply(expectation)
    session.commit()
    session.refresh(refinement)
    before_final = refinement.final_transcript
    before_manual = refinement.manual_transcript

    # The read-only preflight passes under the admitted identity.
    preflight = _service(session, storage).preview(expectation)
    assert preflight.reuse_ready is True

    settings = _ContinuationSettings(storage.storage_root, options=replace(_OPTIONS, beam_size=7))
    factory = _wire_task(monkeypatch, session, settings)
    retry_calls: list[object] = []
    from app.workers.tasks import run_candidate_refinement

    monkeypatch.setattr(
        run_candidate_refinement,
        "retry",
        lambda *args, **kwargs: retry_calls.append((args, kwargs)),
    )

    result = run_guarded_continuation(session, storage, settings, expectation)

    assert result.succeeded is False
    assert result.cancelled is False
    assert result.denial_calls == 1
    assert result.job_status == JobStatus.FAILED.value
    assert result.refinement_status == RefinementStatus.REFINEMENT_FAILED.value
    assert result.error is not None
    assert "ContinuationModelCallDenied" in result.error
    assert retry_calls == []
    with factory() as verify:
        row = verify.get(CandidateRefinement, refinement.id)
        job = verify.get(ProcessingJob, result.job_id)
        assert job is not None and job.status is JobStatus.FAILED
        assert job.error_message == "RefinementError"
        assert row is not None
        assert row.status is RefinementStatus.REFINEMENT_FAILED
        assert row.active_job_id is None
        assert row.manual_transcript == before_manual
        assert row.final_transcript == before_final
        assert row.cache_eligible is False
        jobs = verify.scalars(
            select(ProcessingJob).where(ProcessingJob.candidate_refinement_id == refinement.id)
        ).all()
        assert len(jobs) == 2
        assert all(item.kind is JobKind.CANDIDATE_REFINEMENT for item in jobs)


def test_guarded_continuation_preserves_cancellation(
    session: Session,
    storage: StorageService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    _service(session, storage).apply(expectation)
    session.commit()
    session.refresh(refinement)
    settings = _ContinuationSettings(storage.storage_root)
    factory = _wire_task(monkeypatch, session, settings)

    prepared = prepare_continuation_job(session, storage, settings, expectation)
    job = session.get(ProcessingJob, prepared.job_id)
    assert job is not None
    job.status = JobStatus.CANCELLED
    session.commit()

    execution = execute_guarded_job(
        prepared,
        expected_candidate_id=expectation.candidate_id,
        expected_refinement_id=expectation.refinement_id,
    )

    assert execution.cancelled is True
    assert execution.error is None
    assert execution.denial_calls == 0
    with factory() as verify:
        row = verify.get(CandidateRefinement, refinement.id)
        cancelled_job = verify.get(ProcessingJob, prepared.job_id)
        assert cancelled_job is not None and cancelled_job.status is JobStatus.CANCELLED
        assert row is not None
        assert row.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW


@pytest.mark.parametrize(
    ("routing_mode", "local_qwen_enabled", "hosted_key_present", "code"),
    [
        ("adaptive", False, False, "routing_mode_not_local_only"),
        ("local_only", True, False, "local_qwen_enabled"),
        ("local_only", False, True, "hosted_key_present"),
    ],
)
def test_guarded_continuation_fails_closed_on_incompatible_settings(
    session: Session,
    storage: StorageService,
    tmp_path: Path,
    routing_mode: str,
    local_qwen_enabled: bool,
    hosted_key_present: bool,
    code: str,
) -> None:
    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    settings = _ContinuationSettings(
        storage.storage_root,
        routing_mode=routing_mode,
        local_qwen_enabled=local_qwen_enabled,
        hosted_key_present=hosted_key_present,
    )

    with pytest.raises(ContinuationConfigurationError) as error:
        run_guarded_continuation(session, storage, settings, expectation)
    assert error.value.code == code
    assert session.query(ProcessingJob).count() == 1
    session.refresh(refinement)
    assert refinement.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW


def test_cli_continuation_uses_guarded_runner(
    session: Session,
    storage: StorageService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from app import cli

    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    _service(session, storage).apply(expectation)
    session.commit()
    manifest = tmp_path / "expectation.json"
    manifest.write_text(json.dumps(expectation.as_dict(), ensure_ascii=False), encoding="utf-8")
    settings = _ContinuationSettings(storage.storage_root)
    factory = _wire_task(monkeypatch, session, settings)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "create_session_factory", lambda: factory)

    result = CliRunner().invoke(
        cli.app, ["candidate-recovery-continue", "--expectation", str(manifest)]
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["succeeded"] is True
    assert payload["denial_calls"] == 0
    assert payload["job_status"] == JobStatus.SUCCEEDED.value
    with factory() as verify:
        row = verify.get(CandidateRefinement, refinement.id)
        assert row is not None
        assert row.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
