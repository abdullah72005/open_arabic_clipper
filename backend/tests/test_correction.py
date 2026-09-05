from copy import deepcopy

from app.transcription.correction import ContextualCorrector, context_window


def test_corrects_known_egyptian_confusions_without_mutating_raw_segments() -> None:
    """Declared, high-confidence ASR confusions improve while timestamps stay raw evidence."""

    segments = [
        {"start": 0.0, "end": 1.0, "text": "عامل ايه يا جماعة"},
        {"start": 1.0, "end": 2.0, "text": "لاخبر سالفون"},
        {"start": 2.0, "end": 3.0, "text": "النهاردة هنتكلم عن"},
        {"start": 3.0, "end": 4.0, "text": "خطي بالك"},
        {"start": 4.0, "end": 6.0, "text": "بس فيه نس كتير بتقول إن الوضع صاب"},
    ]
    original = deepcopy(segments)

    corrections = ContextualCorrector.from_default_lexicon().correct(segments)

    assert [correction.corrected_text for correction in corrections] == [
        "عامل ايه يا جماعة",
        "الأخبار زي الفل",
        "النهاردة هنتكلم عن",
        "خلي بالك",
        "بس في ناس كتير بتقول إن الوضع صعب",
    ]
    assert [correction.applied for correction in corrections] == [False, True, False, True, True]
    assert [correction.segment_index for correction in corrections] == [0, 1, 2, 3, 4]
    assert segments == original


def test_keeps_low_confidence_and_protected_code_switching_raw() -> None:
    """Ambiguous speech, English words, names, and numbers never receive lexical guesses."""

    corrections = ContextualCorrector.from_default_lexicon().correct(
        [
            {"start": 0.0, "end": 1.0, "text": "بص الـ backend كان فيه issue في الـ database"},
            {"start": 1.0, "end": 2.0, "text": "Ahmed fixed backend issue 2026"},
            {"start": 2.0, "end": 3.0, "text": "مفيش حد جه"},
        ]
    )

    assert [correction.corrected_text for correction in corrections] == [
        "بص الـ backend كان فيه issue في الـ database",
        "Ahmed fixed backend issue 2026",
        "مفيش حد جه",
    ]
    assert all(correction.applied is False for correction in corrections)
    assert all(correction.uncertain is True for correction in corrections)
    assert all(correction.method == "unchanged" for correction in corrections)


def test_context_window_is_bounded_and_targets_only_current_segment() -> None:
    """Later correction providers can use local evidence without changing segment identity."""

    segments = [
        {"start": float(index), "end": float(index + 1), "text": f"segment {index}"}
        for index in range(5)
    ]

    window = context_window(segments, target_index=2, context_segments=2)

    assert window.previous == ("segment 0", "segment 1")
    assert window.current == "segment 2"
    assert window.following == ("segment 3", "segment 4")
