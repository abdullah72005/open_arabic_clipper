from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import JobKind, JobStatus, PipelineRunStatus, PipelineStage, RightsStatus
from app.db.base import Base
from app.models import PipelineRun, ProcessingJob, SourceVideo
from app.pipeline.authorization import AutopilotAuthorizationError, require_autopilot_authorization
from app.pipeline.runner import PipelineRunner


class RecordingExecutor:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    def execute(self, source: SourceVideo) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


def _source(session: Session, rights_status: RightsStatus = RightsStatus.OWNED) -> SourceVideo:
    source = SourceVideo(source_uri=f"file:///tmp/{uuid.uuid4()}.mp4", rights_status=rights_status)
    session.add(source)
    session.commit()
    return source


def test_completed_stage_is_skipped(sqlite_engine: object) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = _source(session)
        session.add(
            PipelineRun(
                source_video_id=source.id,
                stage=PipelineStage.INGEST,
                status=PipelineRunStatus.SUCCEEDED,
            )
        )
        session.commit()
        executor = RecordingExecutor()

        result = PipelineRunner(session, {PipelineStage.INGEST: executor}).run(
            source.id, PipelineStage.INGEST
        )

        assert result.skipped is True
        assert executor.calls == 0


def test_failure_persists_stage_and_job_error(sqlite_engine: object) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = _source(session)
        executor = RecordingExecutor(RuntimeError("probe unavailable"))
        runner = PipelineRunner(session, {PipelineStage.INGEST: executor})

        with pytest.raises(RuntimeError, match="probe unavailable"):
            runner.run(source.id, PipelineStage.INGEST)

        run = session.scalar(select(PipelineRun))
        job = session.scalar(select(ProcessingJob))
        assert run is not None
        assert job is not None
        assert run.status is PipelineRunStatus.FAILED
        assert run.error_message == "probe unavailable"
        assert job.status is JobStatus.FAILED
        assert job.error_message == "probe unavailable"


def test_retry_increments_job_count_without_replacing_source(sqlite_engine: object) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = _source(session)
        source_id = source.id
        job = ProcessingJob(source_video_id=source.id, kind=JobKind.INGEST)
        session.add(job)
        session.commit()
        runner = PipelineRunner(session, {PipelineStage.INGEST: RecordingExecutor()})

        runner.retry(job.id)

        session.refresh(job)
        assert job.retry_count == 1
        assert job.source_video_id == source_id
        assert session.get(SourceVideo, source_id) is source


def test_autopilot_rejects_unknown_rights() -> None:
    with pytest.raises(AutopilotAuthorizationError, match="UNKNOWN"):
        require_autopilot_authorization(RightsStatus.UNKNOWN)


def test_transcription_stage_advances_to_normalization_and_uses_transcription_job(
    sqlite_engine: object,
) -> None:
    """Durable transcription work has its own retryable job and lifecycle transition."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = _source(session)
        source.lifecycle_state = PipelineStage.TRANSCRIPTION
        session.commit()

        result = PipelineRunner(session, {PipelineStage.TRANSCRIPTION: RecordingExecutor()}).run(
            source.id, PipelineStage.TRANSCRIPTION
        )

        job = session.get(ProcessingJob, result.job_id)
        session.refresh(source)
        assert job is not None
        assert job.kind is JobKind.TRANSCRIPTION
        assert source.lifecycle_state is PipelineStage.TRANSCRIPT_NORMALIZATION


def test_audio_analysis_is_terminal_worker_stage() -> None:
    """The runner advances it to READY_FOR_ANALYSIS without another executor task."""
    from app.workers.tasks import _NEXT_STAGE

    assert PipelineStage.AUDIO_ANALYSIS not in _NEXT_STAGE
