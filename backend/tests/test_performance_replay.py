from app.candidates.service import CandidateAnalysisService
from app.core.enums import CandidateDisposition, MediaOriginType, RightsStatus
from app.transcription.correction import ContextualCorrector
from app.transcription.engine import TranscriptionResult
from app.transcription.performance_replay import CandidateSnapshot, replay_index_candidates
from app.transcription.reconstruction.service import ContextualReconstructor


def test_index_replay_is_read_only_and_reports_retained_candidate_overlap() -> None:
    result = TranscriptionResult(
        language="ar",
        language_probability=0.9,
        raw_text="ليه التطبيق ده سريع؟",
        duration=40.0,
        segments=[
            {
                "start": 0.0,
                "end": 40.0,
                "text": "ليه التطبيق ده سريع؟",
                "words": [{"start": 0.0, "end": 1.0, "word": "ليه"}],
                "avg_logprob": -0.1,
            }
        ],
        word_segments=[{"start": 0.0, "end": 1.0, "word": "ليه"}],
    )
    report = replay_index_candidates(
        source_id="source-1",
        result=result,
        duration=40.0,
        silence_intervals=[],
        audio_features=[],
        rights_status=RightsStatus.UNKNOWN,
        media_origin=MediaOriginType.OTHER,
        provenance_metadata={},
        dialect_override=None,
        baseline_candidates=[
            CandidateSnapshot("missing", CandidateDisposition.CANDIDATE),
        ],
        service=CandidateAnalysisService(),
        corrector=ContextualCorrector.from_default_lexicon(),
        historical_corpus=[],
        reconstructor=ContextualReconstructor(None),
        transcription_fingerprint="test",
        correction_version="test",
    )
    assert report.word_timestamp_count == 1
    assert report.raw_segment_count == 1
    assert report.baseline_retained_candidate_count == 1
    assert report.missing_baseline_candidate_keys == ("missing",)
