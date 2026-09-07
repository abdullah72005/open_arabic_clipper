from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import JobKind, JobStatus, PipelineRunStatus, PipelineStage, RightsStatus
from app.db.base import Base
from app.models import PipelineRun, ProcessingJob, SourceVideo, Transcript
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

    def input_fingerprint(self, source: SourceVideo) -> str:
        return "recording-input-v1"


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
                input_fingerprint="recording-input-v1",
            )
        )
        session.commit()
        executor = RecordingExecutor()

        result = PipelineRunner(session, {PipelineStage.INGEST: executor}).run(
            source.id, PipelineStage.INGEST
        )

        assert result.skipped is True
        assert executor.calls == 0


def test_force_reexecutes_a_completed_stage(sqlite_engine: object) -> None:
    """An operator-requested rerun must not be hidden by a historical success record."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = _source(session)
        session.add(
            PipelineRun(
                source_video_id=source.id,
                stage=PipelineStage.CONTEXTUAL_RECONSTRUCTION,
                status=PipelineRunStatus.SUCCEEDED,
            )
        )
        session.commit()
        executor = RecordingExecutor()

        result = PipelineRunner(session, {PipelineStage.CONTEXTUAL_RECONSTRUCTION: executor}).run(
            source.id, PipelineStage.CONTEXTUAL_RECONSTRUCTION, force=True
        )

        assert result.skipped is False
        assert executor.calls == 1


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


def test_probe_stage_uses_probe_job_kind_and_retries_to_probe(
    sqlite_engine: object,
) -> None:
    """PROBE work maps to its own job kind so listings and retries stay accurate."""

    from app.pipeline.runner import _job_kind_for_stage, _stage_for_job_kind

    assert _job_kind_for_stage(PipelineStage.PROBE) is JobKind.PROBE
    assert _stage_for_job_kind(JobKind.PROBE) is PipelineStage.PROBE

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = _source(session)
        job = ProcessingJob(source_video_id=source.id, kind=JobKind.PROBE)
        session.add(job)
        session.commit()

        runner = PipelineRunner(session, {PipelineStage.PROBE: RecordingExecutor()})

        result = runner.retry(job.id)

        assert result.run_id is not None
        assert source.lifecycle_state is PipelineStage.AUDIO_EXTRACTION


def test_reconstruction_stage_uses_its_own_job_and_advances_to_audio_analysis(
    sqlite_engine: object,
) -> None:
    """Stage 2.7 is independently retryable and sits before audio analysis."""

    from app.pipeline.runner import _job_kind_for_stage, _stage_for_job_kind

    assert _job_kind_for_stage(PipelineStage.CONTEXTUAL_RECONSTRUCTION) is JobKind.RECONSTRUCTION
    assert _stage_for_job_kind(JobKind.RECONSTRUCTION) is PipelineStage.CONTEXTUAL_RECONSTRUCTION

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = _source(session)
        source.lifecycle_state = PipelineStage.CONTEXTUAL_RECONSTRUCTION
        session.commit()

        result = PipelineRunner(
            session, {PipelineStage.CONTEXTUAL_RECONSTRUCTION: RecordingExecutor()}
        ).run(source.id, PipelineStage.CONTEXTUAL_RECONSTRUCTION)

        job = session.get(ProcessingJob, result.job_id)
        session.refresh(source)
        assert job is not None
        assert job.kind is JobKind.RECONSTRUCTION
        assert source.lifecycle_state is PipelineStage.AUDIO_ANALYSIS


class _RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe(self, _path: object, _options: object) -> object:
        self.calls.append("transcribe")
        from app.transcription.engine import TranscriptionResult

        return TranscriptionResult(
            language="ar",
            language_probability=0.9,
            raw_text="raw",
            segments=[],
            word_segments=[],
            duration=1.0,
        )


def test_transcription_stage_holds_heavy_lease_around_whisper(
    sqlite_engine: object,
) -> None:
    """The transcription stage holds the heavy-model lease around Whisper work."""

    from app.models import AudioArtifact
    from app.pipeline.stages import TranscriptionExecutor
    from app.runtime.heavy_model_lease import NoopHeavyModelLeaseFactory
    from app.transcription.service import TranscriptionOptions

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri=f"file:///tmp/{uuid.uuid4()}.mp4",
            content_hash="h",
            rights_status=RightsStatus.OWNED,
        )
        session.add(source)
        session.commit()
        session.add(
            AudioArtifact(
                source_video_id=source.id,
                output_path="/tmp/audio.wav",
                content_hash="h",
                sample_rate=16000,
                duration=1.0,
            )
        )
        session.commit()

        engine = _RecordingEngine()
        lease_factory = NoopHeavyModelLeaseFactory()
        executor = TranscriptionExecutor(
            session=session,
            engine=engine,  # type: ignore[arg-type]
            options=TranscriptionOptions("small", "cpu", "int8", 5),
            lease_factory=lease_factory,
        )

        executor.execute(source)

        events = lease_factory.events
        assert [event["event"] for event in events] == [
            "heavy_model_acquired",
            "heavy_model_released",
        ]
        assert events[0]["purpose"] == "whisper"
        assert engine.calls == ["transcribe"]


def test_transcription_stage_emits_labeled_memory_snapshots(
    sqlite_engine: object, caplog: pytest.LogCaptureFixture
) -> None:
    """The stage records before-load, after-transcribe, and after-cleanup snapshots."""

    import logging

    from app.models import AudioArtifact
    from app.pipeline.stages import TranscriptionExecutor
    from app.runtime.memory import MemorySnapshot
    from app.transcription.service import TranscriptionOptions

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri=f"file:///tmp/{uuid.uuid4()}.mp4",
            content_hash="h",
            rights_status=RightsStatus.OWNED,
        )
        session.add(source)
        session.commit()
        session.add(
            AudioArtifact(
                source_video_id=source.id,
                output_path="/tmp/audio.wav",
                content_hash="h",
                sample_rate=16000,
                duration=1.0,
            )
        )
        session.commit()
        snapshot = MemorySnapshot(0.0, 8 * 1024**3, 1, 1, 1, None, None, None, 1)
        executor = TranscriptionExecutor(
            session=session,
            engine=_RecordingEngine(),  # type: ignore[arg-type]
            options=TranscriptionOptions("small", "cpu", "int8", 5),
            snapshotter=lambda: snapshot,
        )

        with caplog.at_level(logging.INFO, logger="clipfactory.stages"):
            executor.execute(source)

        labels = [
            record.__dict__.get("snapshot")
            for record in caplog.records
            if record.msg == "transcription_memory_snapshot"
        ]
        assert labels == ["before_load", "after_transcribe", "after_cleanup"]


def test_reconstruction_stage_holds_heavy_lease_around_ollama(
    sqlite_engine: object,
) -> None:
    """The reconstruction stage holds the heavy-model lease around the model call."""

    from app.pipeline.stages import ContextualReconstructionExecutor
    from app.runtime.heavy_model_lease import NoopHeavyModelLeaseFactory
    from app.transcription.reconstruction.service import ContextualReconstructor

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri=f"file:///tmp/{uuid.uuid4()}.mp4",
            content_hash="h",
            rights_status=RightsStatus.OWNED,
        )
        session.add(source)
        session.commit()
        session.add(
            Transcript(
                source_video_id=source.id,
                whisper_model="large-v3-turbo",
                input_fingerprint="fp",
                normalization_fingerprint="nf",
                transcription_revision=1,
                correction_version="v1",
                segments=[],
            )
        )
        session.commit()

        lease_factory = NoopHeavyModelLeaseFactory()
        executor = ContextualReconstructionExecutor(
            session=session,
            reconstructor=ContextualReconstructor(None),
            lease_factory=lease_factory,
        )

        executor.execute(source, force=True)

        events = lease_factory.events
        assert [event["event"] for event in events] == [
            "heavy_model_acquired",
            "heavy_model_released",
        ]
        assert events[0]["purpose"] == "ollama"


def test_heavy_lease_contention_raises_retryable_busy(sqlite_engine: object) -> None:
    """A contended lease surfaces as a retryable error before any model starts."""

    from app.pipeline.stages import ContextualReconstructionExecutor
    from app.runtime.heavy_model_lease import HeavyModelLeaseBusy
    from app.transcription.reconstruction.service import ContextualReconstructor

    class BusyLeaseFactory:
        def acquire(self, *, purpose: str) -> object:
            raise HeavyModelLeaseBusy("heavy-model lease busy")

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri=f"file:///tmp/{uuid.uuid4()}.mp4",
            content_hash="h",
            rights_status=RightsStatus.OWNED,
        )
        session.add(source)
        session.commit()
        session.add(
            Transcript(
                source_video_id=source.id,
                whisper_model="large-v3-turbo",
                input_fingerprint="fp",
                normalization_fingerprint="nf",
                transcription_revision=1,
                correction_version="v1",
                segments=[],
            )
        )
        session.commit()
        executor = ContextualReconstructionExecutor(
            session=session,
            reconstructor=ContextualReconstructor(None),
            lease_factory=BusyLeaseFactory(),  # type: ignore[arg-type]
        )

        with pytest.raises(HeavyModelLeaseBusy, match="busy") as excinfo:
            executor.execute(source, force=True)

        assert getattr(excinfo.value, "retryable", False) is True
