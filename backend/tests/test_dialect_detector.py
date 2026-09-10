"""Pure deterministic dialect detection and protected-token preservation tests."""

from copy import deepcopy
from typing import Any

import pytest

from app.transcription.correction import normalize_for_comparison
from app.transcription.dialect import (
    DIALECT_POLICY_VERSION,
    ArabicDialectProfile,
    DialectDetector,
    DialectSelectionMethod,
    code_switch_evidence,
    extract_protected_tokens,
    sample_segment_indexes,
)

_EGYPTIAN = ArabicDialectProfile.EGYPTIAN
_SAUDI = ArabicDialectProfile.SAUDI
_GULF = ArabicDialectProfile.GULF
_LEVANTINE = ArabicDialectProfile.LEVANTINE
_MSA = ArabicDialectProfile.MSA
_UNKNOWN = ArabicDialectProfile.UNKNOWN_ARABIC


def _segments(*texts: str) -> list[dict[str, object]]:
    return [
        {"start": float(index), "end": float(index + 1), "text": text, "raw_text": text}
        for index, text in enumerate(texts)
    ]


def _detect(segments: list[dict[str, object]], **kwargs: Any) -> Any:
    return DialectDetector().detect(segments, **kwargs)


def test_detects_clear_egyptian_evidence() -> None:
    result = _detect(_segments("أنا عايز أعمل deploy للـ backend دلوقتي"))

    assert result.profile is _EGYPTIAN
    assert result.selection is DialectSelectionMethod.DETECTED
    assert 0.80 <= result.confidence <= 0.99
    assert result.reason_code == "detected_egyptian"


def test_detects_clear_saudi_evidence() -> None:
    result = _detect(_segments("وش رايك نخلص الشغل الحين"))

    assert result.profile is _SAUDI
    assert result.selection is DialectSelectionMethod.DETECTED
    assert 0.80 <= result.confidence <= 0.99
    assert result.reason_code == "detected_saudi"


def test_detects_clear_gulf_evidence() -> None:
    result = _detect(_segments("شلونك اليوم، الشغل وايد زين"))

    assert result.profile is _GULF
    assert result.selection is DialectSelectionMethod.DETECTED
    assert 0.80 <= result.confidence <= 0.99


def test_detects_clear_levantine_evidence() -> None:
    result = _detect(_segments("شو رأيك نبلش هلأ، الشغل كتير منيح"))

    assert result.profile is _LEVANTINE
    assert result.selection is DialectSelectionMethod.DETECTED
    assert 0.80 <= result.confidence <= 0.99


def test_detects_clear_msa_fusha_evidence() -> None:
    result = _detect(_segments("سوف نبدأ العمل الآن، وليس من الضروري التأخير"))

    assert result.profile is _MSA
    assert result.selection is DialectSelectionMethod.DETECTED
    assert 0.80 <= result.confidence <= 0.99
    assert result.reason_code == "detected_msa"


def test_weak_arabic_evidence_resolves_to_unknown() -> None:
    result = _detect(_segments("هذا هو الحديث في هذا الشأن من الناحية العامة"))

    assert result.profile is _UNKNOWN
    assert result.selection is DialectSelectionMethod.UNKNOWN
    assert result.confidence == 0.0


def test_conflicting_mixed_dialect_evidence_resolves_to_unknown() -> None:
    result = _detect(_segments("عايز دلوقتي وش الحين شو بدي"))

    assert result.profile is _UNKNOWN
    assert result.selection is DialectSelectionMethod.UNKNOWN
    assert result.confidence == 0.0


def test_english_only_material_resolves_to_none() -> None:
    result = _detect(_segments("Deploy the backend application now"))

    assert result.profile is None
    assert result.selection is DialectSelectionMethod.NOT_APPLICABLE
    assert result.confidence == 0.0


def test_english_only_with_reported_arabic_language_stays_applicable() -> None:
    result = _detect(_segments("Deploy the backend now"), language="ar")

    assert result.profile is not None
    assert result.selection is DialectSelectionMethod.UNKNOWN


def test_arabic_english_material_remains_arabic_applicable() -> None:
    result = _detect(_segments("أنا عملت deploy للـ backend امبارح"))

    assert result.profile is not None
    assert result.profile is not ArabicDialectProfile.MSA


def test_operator_override_wins_over_detection() -> None:
    result = _detect(_segments("وش رايك نخلص الشغل الحين"), override=_SAUDI)

    assert result.profile is _SAUDI
    assert result.selection is DialectSelectionMethod.OPERATOR_OVERRIDE
    assert result.confidence == 1.0
    assert result.reason_code == "operator_override"


def test_operator_override_precedes_detected_evidence() -> None:
    result = _detect(_segments("أنا عايز أعمل deploy دلوقتي"), override=_GULF)

    assert result.profile is _GULF
    assert result.selection is DialectSelectionMethod.OPERATOR_OVERRIDE
    assert result.confidence == 1.0


def test_explicit_unknown_override_is_valid() -> None:
    result = _detect(_segments("أنا عايز أعمل deploy دلوقتي"), override=_UNKNOWN)

    assert result.profile is _UNKNOWN
    assert result.selection is DialectSelectionMethod.OPERATOR_OVERRIDE
    assert result.confidence == 1.0


def test_representative_sampling_is_capped_at_48_and_includes_edges() -> None:
    segments = _segments(*[f"جملة رقم {index}" for index in range(200)])

    indexes = sample_segment_indexes(segments, max_samples=48)

    assert len(indexes) == 48
    assert indexes[0] == 0
    assert indexes[-1] == 199
    assert indexes == tuple(sorted(indexes))


def test_sampling_skips_empty_segments() -> None:
    segments = [{"text": "", "start": 0.0, "end": 1.0}, {"text": "كلام", "start": 1.0, "end": 2.0}]

    indexes = sample_segment_indexes(segments, max_samples=48)

    assert indexes == (1,)


def test_detector_never_mutates_input_segments() -> None:
    segments = _segments("أنا عايز أعمل deploy دلوقتي", "وش رايك الحين")
    original = deepcopy(segments)

    _detect(segments)

    assert segments == original


def test_detected_evidence_is_bounded_and_contains_no_transcript_bodies() -> None:
    result = _detect(_segments("أنا عايز أعمل deploy للـ backend دلوقتي"))
    serialized = repr(result.evidence)

    assert result.evidence["dialect_policy_version"] == DIALECT_POLICY_VERSION
    assert result.evidence["selection_method"] == "detected"
    assert result.evidence["sampled_segment_count"] == 1
    assert result.evidence["sampled_segment_indexes"] == [0]
    assert result.evidence["marker_scores"][_EGYPTIAN.value] >= 4
    assert result.evidence["distinct_marker_counts"][_EGYPTIAN.value] >= 2
    assert result.evidence["winner_score"] >= 4
    assert "عايز" not in serialized
    assert "دلوقتي" not in serialized
    assert "deploy" not in serialized


def test_marker_scoring_is_deterministic() -> None:
    segments = _segments("أنا عايز أعمل deploy دلوقتي")

    first = _detect(segments)
    second = _detect(segments)

    assert first.evidence == second.evidence
    assert first.confidence == second.confidence


def test_normalized_marker_matching_uses_comparison_copy() -> None:
    assert "رايك" in normalize_for_comparison("وش رأيك الحين")
    assert normalize_for_comparison("أنا عايز دلوقتي") != normalize_for_comparison("أنا أريد الآن")


def test_deploy_and_backend_survive_protected_token_extraction() -> None:
    tokens = extract_protected_tokens("أنا عملت deploy للـ backend امبارح")

    assert tokens == ("deploy", "backend")


def test_technical_forms_names_abbreviations_and_numbers_extract_exactly() -> None:
    tokens = extract_protected_tokens(
        "الـ api.example.com و v2.0 و backend-api و OpenAI و v3 و 25-11-2026 و 71 و ٠٧١"
    )

    assert "api.example.com" in tokens
    assert "v2.0" in tokens
    assert "backend-api" in tokens
    assert "OpenAI" in tokens
    assert "v3" in tokens
    assert "25-11-2026" in tokens
    assert "71" in tokens
    assert "٠٧١" in tokens
    assert tokens[0] == "api.example.com"


def test_case_change_is_detected_as_preservation_failure() -> None:
    raw = "أنا عملت deploy للـ backend امبارح"
    case_changed = "أنا عملت Deploy للـ Backend امبارح"

    assert extract_protected_tokens(raw) != extract_protected_tokens(case_changed)


def test_token_removal_is_detected_as_preservation_failure() -> None:
    raw = "أنا عملت deploy للـ backend امبارح"
    removed = "أنا عملت للـ امبارح"

    assert extract_protected_tokens(raw) != extract_protected_tokens(removed)


def test_token_reordering_is_detected_as_preservation_failure() -> None:
    raw = "أنا عملت deploy للـ backend امبارح"
    reordered = "أنا عملت backend للـ deploy امبارح"

    assert extract_protected_tokens(raw) != extract_protected_tokens(reordered)


def test_arabicized_token_is_detected_as_preservation_failure() -> None:
    raw = "أنا عملت deploy امبارح"
    arabicized = "أنا عملت دبلوي امبارح"

    assert extract_protected_tokens(raw) != extract_protected_tokens(arabicized)


def test_invented_latin_token_is_detected_as_preservation_failure() -> None:
    raw = "أنا عملت امبارح"
    invented = "أنا عملت deploy امبارح"

    assert extract_protected_tokens(raw) != extract_protected_tokens(invented)


def test_numbers_alone_do_not_imply_code_switching() -> None:
    evidence = code_switch_evidence("فيه 71 شخص امبارح")

    assert evidence.suspected is False
    assert evidence.tokens == ()


def test_latin_technical_token_in_arabic_text_implies_code_switching() -> None:
    evidence = code_switch_evidence("أنا عملت deploy للـ backend امبارح")

    assert evidence.suspected is True
    assert evidence.tokens == ("deploy", "backend")


def test_english_words_alone_are_latin_evidence_but_not_arabic_code_switching() -> None:
    evidence = code_switch_evidence("Deploy the backend now")

    assert evidence.suspected is True
    assert evidence.tokens == ("Deploy", "the", "backend", "now")


def test_arabic_indic_number_is_not_latin_evidence() -> None:
    evidence = code_switch_evidence("التقرير صدر يوم ٠٧١")

    assert evidence.suspected is False
    assert evidence.tokens == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ملف data_v2.xlsx جاهز", ("data_v2.xlsx",)),
        ("الإصدار 1.5.0 صدر", ("1.5.0",)),
        ("أرسلنا عبر #build و +icon", ("build", "icon")),
        ("الرابط https://example.com/page يعمل", ("https", "example.com", "page")),
    ],
)
def test_ordinary_technical_forms_are_preserved(text: str, expected: tuple[str, ...]) -> None:
    tokens = extract_protected_tokens(text)

    for token in expected:
        assert token in tokens
