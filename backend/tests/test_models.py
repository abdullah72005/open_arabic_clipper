from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import (
    JobKind,
    JobStatus,
    PipelineRunStatus,
    PipelineStage,
    ReconstructionStatus,
    RightsStatus,
)
from app.db.base import Base
from app.models import (
    AudioAnalysis,
    PipelineRun,
    ProcessingJob,
    SourceQualityAssessment,
    SourceVideo,
    Transcript,
)


def test_stage_2_7_truth_columns_exist() -> None:
    assert {"input_fingerprint", "output_fingerprint"} <= set(PipelineRun.__table__.columns.keys())
    assert {
        "transcription_revision",
        "normalization_fingerprint",
        "reconstruction_status",
    } <= set(Transcript.__table__.columns.keys())
    assert "input_fingerprint" in AudioAnalysis.__table__.columns
    assert {
        "transcript_quality_score",
        "low_confidence_word_ratio",
        "unresolved_segment_ratio",
        "manual_review_required",
        "input_fingerprint",
    } <= set(SourceQualityAssessment.__table__.columns.keys())

    reconstruction_status = Transcript.__table__.columns["reconstruction_status"]
    assert reconstruction_status.type.native_enum is False
    assert reconstruction_status.default.arg is ReconstructionStatus.NOT_REQUIRED

    constraint_sql = {
        str(constraint.sqltext)
        for table in (Transcript.__table__, SourceQualityAssessment.__table__)
        for constraint in table.constraints
        if hasattr(constraint, "sqltext")
    }
    assert "transcription_revision >= 0" in constraint_sql
    assert "transcript_quality_score >= 0 AND transcript_quality_score <= 1" in constraint_sql
    assert "low_confidence_word_ratio >= 0 AND low_confidence_word_ratio <= 1" in constraint_sql
    assert "unresolved_segment_ratio >= 0 AND unresolved_segment_ratio <= 1" in constraint_sql


def test_source_video_defaults_and_content_hash_is_unique(sqlite_engine: object) -> None:
    """A source records safe defaults and cannot duplicate known content."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="/imports/episode.mp4", content_hash="abc123")
        session.add(source)
        session.commit()

        assert source.id is not None
        assert source.rights_status is RightsStatus.UNKNOWN
        assert source.lifecycle_state is PipelineStage.INGEST
        assert source.created_at is not None
        assert source.updated_at is not None

        session.add(SourceVideo(source_uri="/imports/copy.mp4", content_hash="abc123"))
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
        else:
            raise AssertionError("content hashes must be unique")


def test_processing_job_defaults_to_queued_ingest(sqlite_engine: object) -> None:
    """New ingest work is represented by a durable queued job."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="/imports/episode.mp4")
        session.add(source)
        session.flush()
        job = ProcessingJob(source_video_id=source.id)
        session.add(job)
        session.commit()

        assert job.id is not None
        assert job.kind is JobKind.INGEST
        assert job.status is JobStatus.QUEUED
        assert job.retry_count == 0


def test_pipeline_run_can_transition_from_queued_to_running(sqlite_engine: object) -> None:
    """Pipeline-run status changes persist for resumable processing."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="/imports/episode.mp4")
        session.add(source)
        session.flush()
        run = PipelineRun(source_video_id=source.id)
        session.add(run)
        session.commit()

        assert run.stage is PipelineStage.INGEST
        assert run.status is PipelineRunStatus.QUEUED
        run.status = PipelineRunStatus.RUNNING
        session.commit()
        session.refresh(run)

        assert run.status is PipelineRunStatus.RUNNING


def test_source_persists_timestamped_transcript_data(sqlite_engine: object) -> None:
    """A reusable transcript retains timestamp and confidence data for later analysis."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="/imports/episode.mp4", content_hash="audio-source")
        session.add(source)
        session.flush()
        transcript = Transcript(
            source_video_id=source.id,
            language="ar",
            whisper_model="small",
            transcription_options={"beam_size": 5},
            input_fingerprint="f" * 64,
            raw_text="أهلا",
            normalized_text="أهلا",
            segments=[
                {
                    "start": 0.0,
                    "end": 0.8,
                    "text": "أهلا",
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.01,
                    "words": [],
                }
            ],
            word_segments=[],
            duration=0.8,
        )
        session.add(transcript)
        session.commit()

        assert source.transcript is not None
        assert source.transcript.segments[0]["start"] == 0.0
        assert source.transcript.segments[0]["no_speech_prob"] == 0.01


def test_transcript_persists_timestamped_semantic_chunks(sqlite_engine: object) -> None:
    from app.models import TranscriptChunk

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="/imports/episode.mp4")
        session.add(source)
        session.flush()
        transcript = Transcript(
            source_video_id=source.id,
            whisper_model="small",
            input_fingerprint="c" * 64,
            raw_text="أهلا hello",
            normalized_text="أهلا hello",
            segments=[],
            word_segments=[],
        )
        session.add(transcript)
        session.flush()
        session.add(
            TranscriptChunk(
                transcript_id=transcript.id,
                sequence=0,
                start_time=0.0,
                end_time=2.0,
                text="أهلا hello",
                segment_indexes=[0, 1],
                preceding_context="",
                following_context="Next sentence.",
            )
        )
        session.commit()

        assert transcript.chunks[0].segment_indexes == [0, 1]
