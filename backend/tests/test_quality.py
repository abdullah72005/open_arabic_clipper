from sqlalchemy.orm import Session

from app.db.base import Base
from app.models import AudioAnalysis, SourceVideo, Transcript


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

        assert assessment.overall_source_quality_score > 0
        assert assessment.reasons
        assert "raw_transcript_confidence=0.80" in assessment.reasons
        assert "correction_confidence=0.95" in assessment.reasons
        assert "corrected_segment_ratio=0.20" in assessment.reasons
        assert "uncertain_segment_ratio=0.40" in assessment.reasons
        assert source.lifecycle_state.value != "FAILED"
