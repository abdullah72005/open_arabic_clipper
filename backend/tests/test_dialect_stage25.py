"""Stage 2.5 Egyptian-dialect isolation and inherited metadata tests."""

from copy import deepcopy
from typing import Any

import pytest

from app.models import SourceVideo, Transcript
from app.pipeline.fingerprints import canonical_fingerprint
from app.pipeline.stages import TranscriptNormalizationExecutor
from app.transcription.correction import ContextualCorrector
from app.transcription.dialect import (
    DIALECT_POLICY_VERSION,
    ArabicDialectProfile,
    DialectDetector,
)
from app.transcription.providers import CorrectionRequest, ProviderCorrection


def _segments(*texts: str) -> list[dict[str, object]]:
    return [
        {"start": float(index), "end": float(index + 1), "text": text, "raw_text": text}
        for index, text in enumerate(texts)
    ]


def test_egyptian_confusion_applies_only_for_egyptian_profile() -> None:
    raw = "خطي بالك"

    egyptian = ContextualCorrector.from_default_lexicon().correct(
        [{"text": raw}], profile=ArabicDialectProfile.EGYPTIAN
    )

    assert egyptian[0].corrected_text == "خلي بالك"
    assert egyptian[0].applied is True


@pytest.mark.parametrize(
    "profile",
    [
        ArabicDialectProfile.SAUDI,
        ArabicDialectProfile.GULF,
        ArabicDialectProfile.LEVANTINE,
        ArabicDialectProfile.MSA,
        ArabicDialectProfile.UNKNOWN_ARABIC,
        None,
    ],
)
def test_egyptian_confusion_is_not_replaced_outside_egyptian(profile: Any) -> None:
    raw = "خطي بالك"

    correction = ContextualCorrector.from_default_lexicon().correct(
        [{"text": raw}], profile=profile
    )[0]

    assert correction.corrected_text == raw
    assert correction.applied is False
    assert correction.method == "unchanged"


def test_non_egyptian_profile_makes_zero_optional_provider_calls() -> None:
    class RecordingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def correct_batch(self, requests: list[CorrectionRequest]) -> list[ProviderCorrection]:
            self.calls += 1
            return []

    provider = RecordingProvider()
    corrector = ContextualCorrector.from_default_lexicon(provider=provider)

    corrector.correct(_segments("وش رايك نخلص الشغل الحين"), profile=ArabicDialectProfile.SAUDI)

    assert provider.calls == 0


def test_egyptian_profile_can_use_the_optional_provider() -> None:
    class RecordingProvider:
        def __init__(self) -> None:
            self.requests: list[CorrectionRequest] = []

        def correct_batch(self, requests: list[CorrectionRequest]) -> list[ProviderCorrection]:
            self.requests = requests
            return [
                ProviderCorrection(
                    segment_index=request.segment_index,
                    corrected_text="خلي بالك",
                    changed=True,
                    confidence=0.96,
                    changes=[],
                )
                for request in requests
            ]

    provider = RecordingProvider()
    corrector = ContextualCorrector.from_default_lexicon(provider=provider)

    corrector.correct(
        [{"text": "خطي بالك"}, {"text": "الموضوع مش سهل"}],
        profile=ArabicDialectProfile.EGYPTIAN,
    )

    assert [request.segment_index for request in provider.requests] == [0]


def test_candidate_null_segments_are_never_sent_to_provider() -> None:
    class RecordingProvider:
        def __init__(self) -> None:
            self.requests: list[CorrectionRequest] = []

        def correct_batch(self, requests: list[CorrectionRequest]) -> list[ProviderCorrection]:
            self.requests = requests
            return [
                ProviderCorrection(
                    segment_index=request.segment_index,
                    corrected_text="خلي بالك",
                    changed=True,
                    confidence=0.96,
                    changes=[],
                )
                for request in requests
            ]

    provider = RecordingProvider()
    corrector = ContextualCorrector.from_default_lexicon(provider=provider)
    segments = _segments("الموضوع مش سهل", "خطي بالك", "الحمد لله")

    corrector.correct(segments, profile=ArabicDialectProfile.EGYPTIAN)

    assert [request.segment_index for request in provider.requests] == [1]
    assert all(request.candidate_text is not None for request in provider.requests)


def test_valid_saudi_wording_is_not_egyptianized_or_msa_forced() -> None:
    raw = "وش رايك نخلص الشغل الحين"

    correction = ContextualCorrector.from_default_lexicon().correct(
        [{"text": raw}], profile=ArabicDialectProfile.SAUDI
    )[0]

    assert correction.corrected_text == raw
    assert correction.applied is False


def test_valid_gulf_wording_is_not_egyptianized() -> None:
    raw = "شلونك اليوم، الشغل وايد زين"

    correction = ContextualCorrector.from_default_lexicon().correct(
        [{"text": raw}], profile=ArabicDialectProfile.GULF
    )[0]

    assert correction.corrected_text == raw
    assert correction.applied is False


def test_valid_levantine_wording_is_not_egyptianized() -> None:
    raw = "شو رأيك نبلش هلأ، الشغل كتير منيح"

    correction = ContextualCorrector.from_default_lexicon().correct(
        [{"text": raw}], profile=ArabicDialectProfile.LEVANTINE
    )[0]

    assert correction.corrected_text == raw
    assert correction.applied is False


def test_formal_msa_remains_formal_and_not_colloquialized() -> None:
    raw = "سوف نبدأ العمل الآن، وليس من الضروري التأخير"

    correction = ContextualCorrector.from_default_lexicon().correct(
        [{"text": raw}], profile=ArabicDialectProfile.MSA
    )[0]

    assert correction.corrected_text == raw
    assert correction.applied is False


def test_unknown_arabic_receives_conservative_no_change() -> None:
    raw = "هذا هو الحديث في هذا الشأن"

    correction = ContextualCorrector.from_default_lexicon().correct(
        [{"text": raw}], profile=ArabicDialectProfile.UNKNOWN_ARABIC
    )[0]

    assert correction.corrected_text == raw
    assert correction.applied is False


def test_latin_names_abbreviations_and_numbers_preserved_under_egyptian() -> None:
    raw = "أنا عملت deploy للـ backend امبارح الساعة 71"

    correction = ContextualCorrector.from_default_lexicon().correct(
        [{"text": raw}], profile=ArabicDialectProfile.EGYPTIAN
    )[0]

    assert correction.corrected_text == raw
    assert correction.applied is False


def test_required_profile_argument_is_explicit_in_production_path() -> None:
    with pytest.raises(TypeError):
        ContextualCorrector.from_default_lexicon().correct([{"text": "خطي بالك"}])


def test_stage25_safety_gate_rejects_destructive_technical_rewrites() -> None:
    """Stage 2.5 protected-token preservation rejects fragmented technical tokens."""

    corrector = ContextualCorrector.from_default_lexicon()
    cases = [
        ("C++", "C#"),
        ("foo/bar", "foo bar"),
        ("api/v2", "api v2"),
        ("#build", "build"),
        ("+icon", "icon"),
        ("https://example.com/page", "https example.com page"),
    ]
    for raw, destructive in cases:
        result = ProviderCorrection(0, destructive, changed=True, confidence=0.99, changes=[])
        assert corrector._provider_result_is_safe(raw, result) is False, (
            f"{raw!r} -> {destructive!r} must be rejected"
        )


def test_stage25_safety_gate_rejects_destructive_url_rewrites() -> None:
    """Stage 2.5 provider safety rejects URL fragmentation and query mutation."""

    corrector = ContextualCorrector.from_default_lexicon()
    cases = [
        ("https://example.com/page?foo=bar", "https://example.com/page foo bar"),
        ("https://example.com/page?foo=bar", "https://example.com/page?foo=baz"),
        (
            "https://example.com/page?foo=bar&mode=full",
            "https://example.com/page?mode=full&foo=bar",
        ),
        ("https://example.com/a%20path?q=x%2Fy", "https://example.com/a path?q=x/y"),
        ("https://example.com/page?foo=bar#section", "https://example.com/page?foo=bar"),
        ("https://example.com/page?foo=bar", "http://example.com/page?foo=bar"),
    ]
    for raw, destructive in cases:
        result = ProviderCorrection(0, destructive, changed=True, confidence=0.99, changes=[])
        assert corrector._provider_result_is_safe(raw, result) is False, (
            f"{raw!r} -> {destructive!r} must be rejected"
        )


def test_stage25_safety_gate_rejects_balanced_parenthesis_url_mutation() -> None:
    """Stage 2.5 provider safety rejects removing a URL's balanced closing parenthesis."""

    corrector = ContextualCorrector.from_default_lexicon()
    raw = "https://en.wikipedia.org/wiki/Function_(mathematics)"
    destructive = "https://en.wikipedia.org/wiki/Function_(mathematics"

    result = ProviderCorrection(0, destructive, changed=True, confidence=0.99, changes=[])

    assert corrector._provider_result_is_safe(raw, result) is False


def _setup_transcript(session: Any, segments: list[dict[str, object]], language: str = "ar") -> Any:
    source = SourceVideo(
        source_uri="file:///tmp/dialect.mp4", content_hash="h", rights_status="OWNED"
    )
    session.add(source)
    session.commit()
    transcript = Transcript(
        source_video_id=source.id,
        whisper_model="large-v3-turbo",
        input_fingerprint="asr-fp",
        normalization_fingerprint="norm-fp",
        transcription_revision=1,
        correction_version="egyptian-ar-v1",
        language=language,
        segments=segments,
        word_segments=[],
        raw_text=" ".join(str(segment["text"]) for segment in segments),
        corrected_text=" ".join(str(segment["text"]) for segment in segments),
        final_text=" ".join(str(segment["text"]) for segment in segments),
    )
    session.add(transcript)
    session.commit()
    return source, transcript


def test_normalization_inherits_source_dialect_into_segments(sqlite_engine: Any) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source, transcript = _setup_transcript(
            session,
            _segments("أنا عايز أعمل deploy للـ backend دلوقتي"),
        )
        source.dialect_profile_override = None
        session.commit()

        executor = TranscriptNormalizationExecutor(session=session)
        executor.execute(source)

        session.refresh(transcript)
        segment = transcript.segments[0]
        assert segment["dialect_profile"] == ArabicDialectProfile.EGYPTIAN.value
        assert segment["dialect_confidence"] > 0.8
        assert segment["dialect_selection"] == "detected"
        assert segment["dialect_policy_version"] == DIALECT_POLICY_VERSION
        assert transcript.dialect_profile == ArabicDialectProfile.EGYPTIAN.value
        assert transcript.dialect_confidence > 0.8
        assert transcript.dialect_evidence["selection_method"] == "detected"
        assert transcript.dialect_evidence["dialect_policy_version"] == DIALECT_POLICY_VERSION


def test_normalization_stores_code_switch_evidence(sqlite_engine: Any) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source, transcript = _setup_transcript(
            session,
            _segments("أنا عملت deploy للـ backend امبارح", "الوضع تمام"),
        )

        executor = TranscriptNormalizationExecutor(session=session)
        executor.execute(source)

        session.refresh(transcript)
        assert transcript.code_switch_suspected is True
        assert transcript.segments[0]["code_switch_suspected"] is True
        assert transcript.segments[0]["code_switch_tokens"] == ["deploy", "backend"]
        assert transcript.segments[1]["code_switch_suspected"] is False
        assert transcript.segments[1]["code_switch_tokens"] == []


def test_english_only_source_is_not_arabic_code_switching(sqlite_engine: Any) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source, transcript = _setup_transcript(
            session,
            _segments("Deploy the backend application now"),
            language="en",
        )

        executor = TranscriptNormalizationExecutor(session=session)
        executor.execute(source)

        session.refresh(transcript)
        assert transcript.dialect_profile is None
        assert transcript.code_switch_suspected is False
        assert transcript.segments[0]["code_switch_suspected"] is False
        assert transcript.segments[0]["dialect_profile"] is None


def test_numbers_alone_do_not_flag_code_switching(sqlite_engine: Any) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source, transcript = _setup_transcript(session, _segments("فيه 71 شخص امبارح"))

        executor = TranscriptNormalizationExecutor(session=session)
        executor.execute(source)

        session.refresh(transcript)
        assert transcript.code_switch_suspected is False
        assert transcript.segments[0]["code_switch_tokens"] == []


def test_long_source_arabic_outside_sample_preserves_code_switch_evidence(
    sqlite_engine: Any,
) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base
    from app.transcription.dialect import sample_segment_indexes

    segments = [
        {
            "start": float(index),
            "end": float(index + 1),
            "text": f"ordinary segment {index}",
            "raw_text": f"ordinary segment {index}",
        }
        for index in range(60)
    ]
    segments[7] = {
        "start": 7.0,
        "end": 8.0,
        "text": "أنا عملت deploy للـ backend امبارح",
        "raw_text": "أنا عملت deploy للـ backend امبارح",
    }
    assert 7 not in sample_segment_indexes(segments, max_samples=48)

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source, transcript = _setup_transcript(session, segments, language="en")

        executor = TranscriptNormalizationExecutor(session=session)
        executor.execute(source)

        session.refresh(transcript)
        assert transcript.dialect_profile == ArabicDialectProfile.UNKNOWN_ARABIC.value
        assert transcript.dialect_confidence == 0.0
        assert transcript.code_switch_suspected is True
        assert transcript.segments[7]["code_switch_suspected"] is True
        assert transcript.segments[7]["code_switch_tokens"] == ["deploy", "backend"]
        assert transcript.segments[0]["code_switch_suspected"] is False


def test_segment_code_switch_requires_arabic_script_within_the_segment(
    sqlite_engine: Any,
) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base

    segments = _segments(
        "أنا عملت deploy امبارح",
        "deploy backend",
        "فيه 71 شخص امبارح",
        "الوضع تمام",
    )
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source, transcript = _setup_transcript(session, segments)

        executor = TranscriptNormalizationExecutor(session=session)
        executor.execute(source)

        session.refresh(transcript)
        assert transcript.code_switch_suspected is True
        assert transcript.segments[0]["code_switch_suspected"] is True
        assert transcript.segments[0]["code_switch_tokens"] == ["deploy"]
        assert transcript.segments[1]["code_switch_suspected"] is False
        assert transcript.segments[2]["code_switch_suspected"] is False
        assert transcript.segments[3]["code_switch_suspected"] is False


def test_normalization_override_propagates_and_sets_confidence_one(sqlite_engine: Any) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source, transcript = _setup_transcript(session, _segments("وش رايك نخلص الشغل الحين"))
        source.dialect_profile_override = ArabicDialectProfile.EGYPTIAN
        session.commit()

        executor = TranscriptNormalizationExecutor(session=session)
        executor.execute(source)

        session.refresh(transcript)
        assert transcript.dialect_profile == ArabicDialectProfile.EGYPTIAN.value
        assert transcript.dialect_confidence == 1.0
        assert transcript.dialect_evidence["selection_method"] == "operator_override"
        assert transcript.segments[0]["dialect_selection"] == "operator_override"


def test_normalization_fingerprint_includes_override_and_dialect_identity(
    sqlite_engine: Any,
) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source, transcript = _setup_transcript(session, _segments("وش رايك نخلص الشغل الحين"))
        executor = TranscriptNormalizationExecutor(session=session)

        executor.input_fingerprint(source)
        executor.execute(source)
        baseline = transcript.normalization_fingerprint
        session.commit()

        source.dialect_profile_override = ArabicDialectProfile.SAUDI
        session.commit()
        forced = executor.execute(source, force=True)

        assert forced.output_fingerprint != baseline


def test_normalization_does_not_rewrite_raw_asr(sqlite_engine: Any) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        segments = _segments("أنا عايز أعمل deploy للـ backend دلوقتي", "الوضع تمام")
        original = deepcopy(segments)
        source, transcript = _setup_transcript(session, segments)

        executor = TranscriptNormalizationExecutor(session=session)
        executor.execute(source)

        session.refresh(transcript)
        assert transcript.segments[0]["raw_text"] == original[0]["text"]
        assert transcript.segments[1]["raw_text"] == original[1]["text"]
        assert transcript.segments[0]["start"] == original[0]["start"]
        assert transcript.segments[0]["end"] == original[0]["end"]


def test_detector_is_pure_lightweight_and_never_calls_providers(sqlite_engine: Any) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source, transcript = _setup_transcript(session, _segments("أنا عايز أعمل deploy دلوقتي"))

        result = DialectDetector().detect(transcript.segments, language=transcript.language)

        assert result.profile is ArabicDialectProfile.EGYPTIAN
        assert result.evidence["sampled_segment_count"] <= 48
        assert transcript.segments == transcript.segments


def test_normalization_input_fingerprint_is_canonical(sqlite_engine: Any) -> None:
    from sqlalchemy.orm import Session

    from app.db.base import Base

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source, transcript = _setup_transcript(session, _segments("أنا عايز أعمل deploy دلوقتي"))
        executor = TranscriptNormalizationExecutor(session=session)

        fingerprint = executor.input_fingerprint(source)

        assert fingerprint
        assert canonical_fingerprint("normalization-input", "2", {"unused": True}) != fingerprint
