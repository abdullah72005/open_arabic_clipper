from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models import AudioAnalysis, AudioArtifact, SourceVideo, Transcript
from app.pipeline.stages import AudioAnalysisExecutor
from app.services.storage import StorageService


def transcript_with(
    *,
    probabilities: list[float],
    statuses: list[str] | None = None,
    provider_availability: str = "AVAILABLE",
) -> Transcript:
    statuses = statuses or ["UNCHANGED_HIGH_CONFIDENCE"]
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": " كلام",
            "corrected_text": "كلام",
            "reconstruction_status": statuses[min(index, len(statuses) - 1)],
            "reconstruction_confidence": 0.0,
            "routing_priority": "RECONSTRUCT",
            "words": [
                {"word": f" w{word}", "probability": probability}
                for word, probability in enumerate(probabilities)
            ],
        }
        for index in range(len(statuses))
    ]
    return Transcript(
        whisper_model="large-v3-turbo",
        transcription_options={},
        input_fingerprint="a" * 64,
        raw_text="كلام",
        normalized_text="كلام",
        corrected_text="كلام",
        final_text="كلام",
        language="ar",
        detected_language_probability=0.99,
        duration=3.0,
        segments=segments,
        word_segments=[],
        reconstruction_metadata={"provider_availability": provider_availability},
    )


def perfect_audio(source_id: UUID) -> AudioAnalysis:
    return AudioAnalysis(
        source_video_id=source_id,
        audio_hash="b" * 64,
        silence_intervals=[],
        features=[],
        silence_ratio=0.015,
        speech_density=0.985,
        speech_rate=120.0,
    )


def test_source_quality_assessment_is_persisted_and_advisory(sqlite_engine: object) -> None:
    """Stage 2 records quality evidence without rejecting a source."""

    from app.services.source_quality import assess_source

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="/imports/episode.mp4")
        session.add(source)
        session.flush()
        transcript = Transcript(
            source_video_id=source.id,
            whisper_model="small",
            transcription_options={},
            input_fingerprint="a" * 64,
            raw_text="أهلا hello",
            normalized_text="أهلا hello",
            raw_transcript_confidence=0.8,
            correction_confidence=0.95,
            corrected_segment_ratio=0.2,
            uncertain_segment_ratio=0.4,
            duration=10,
            segments=[{"start": 0.0, "end": 2.0, "text": "أهلا hello", "avg_logprob": -0.2}],
            word_segments=[],
            language="ar",
            detected_language_probability=0.9,
        )
        analysis = AudioAnalysis(
            source_video_id=source.id,
            audio_hash="b" * 64,
            silence_intervals=[],
            features=[],
            silence_ratio=0.1,
            speech_density=0.6,
            speech_rate=2.5,
        )
        session.add_all([transcript, analysis])
        session.commit()

        assessment = assess_source(session, source, transcript, analysis)

        assert assessment.overall_source_quality_score == assessment.transcript_quality_score
        assert assessment.reasons
        assert "transcript:correction_confidence=0.95" in assessment.reasons
        assert "transcript:corrected_segment_ratio=0.20" in assessment.reasons
        assert "transcript:uncertain_segment_ratio=0.40" in assessment.reasons
        assert "audio:speech_density=0.60" in assessment.reasons
        assert "audio:silence_ratio=0.10" in assessment.reasons
        assert assessment.input_fingerprint
        assert assessment.version == 2
        assert source.lifecycle_state.value != "FAILED"


def test_good_audio_cannot_mask_unavailable_uncertain_transcript(
    sqlite_engine: object,
) -> None:
    from app.services.source_quality import assess_source

    transcript = transcript_with(
        probabilities=[0.99, 0.53, 0.38],
        statuses=["PROVIDER_UNAVAILABLE"],
        provider_availability="UNAVAILABLE",
    )
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="/imports/bad.mp4")
        session.add(source)
        session.flush()
        transcript.source_video_id = source.id
        analysis = perfect_audio(source.id)
        session.add_all([transcript, analysis])
        session.commit()

        assessment = assess_source(session, source, transcript, analysis)

        assert assessment.audio_quality_score >= 0.95
        assert assessment.transcript_quality_score <= 0.40
        assert assessment.overall_source_quality_score == assessment.transcript_quality_score
        assert assessment.manual_review_required is True


def test_quality_fingerprint_changes_with_reconstruction_output(sqlite_engine: object) -> None:
    from app.services.source_quality import assess_source

    transcript = transcript_with(probabilities=[0.9])
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="/imports/reconstructed.mp4")
        session.add(source)
        session.flush()
        transcript.source_video_id = source.id
        analysis = perfect_audio(source.id)
        session.add_all([transcript, analysis])
        session.commit()

        first = assess_source(session, source, transcript, analysis)
        first_fingerprint = first.input_fingerprint
        transcript.segments = [
            {**transcript.segments[0], "reconstruction_status": "FAILED"}
        ]
        transcript.reconstruction_fingerprint = "c" * 64
        second = assess_source(session, source, transcript, analysis)

        assert second.input_fingerprint != first_fingerprint
        assert second.transcript_quality_score == 0.25


def test_cached_audio_recomputes_transcript_derived_quality(
    sqlite_engine: object, tmp_path: Path
) -> None:
    from app.services.source_quality import assess_source

    transcript = transcript_with(probabilities=[0.9])
    transcript.word_segments = [{"word": "كلام"}]
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="/imports/cached.mp4")
        session.add(source)
        session.flush()
        transcript.source_video_id = source.id
        artifact = AudioArtifact(
            source_video_id=source.id,
            output_path="cached.wav",
            content_hash="b" * 64,
            sample_rate=16_000,
            duration=3.0,
        )
        analysis = perfect_audio(source.id)
        analysis.features = [{"start": 0.0, "end": 3.0, "rms": 0.5}]
        session.add_all([transcript, artifact, analysis])
        session.flush()
        executor = AudioAnalysisExecutor(
            session=session,
            storage=StorageService(tmp_path),
            command_runner=lambda _args: (_ for _ in ()).throw(
                AssertionError("cached audio must not rerun FFmpeg")
            ),
        )
        analysis.input_fingerprint = executor.input_fingerprint(source)
        session.commit()
        assessment = assess_source(session, source, transcript, analysis)
        original_quality_fingerprint = assessment.input_fingerprint

        transcript.segments = [
            {**transcript.segments[0], "reconstruction_status": "FAILED"}
        ]
        transcript.reconstruction_fingerprint = "d" * 64
        result = executor.execute(source)

        assert result.value is analysis
        assert analysis.features == [{"start": 0.0, "end": 3.0, "rms": 0.5}]
        assert analysis.speech_rate == pytest.approx(20.0)
        assert source.quality_assessment.input_fingerprint != original_quality_fingerprint
        assert source.quality_assessment.transcript_quality_score == 0.25
