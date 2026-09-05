import pytest

from app.transcription.reconstruction.phonetics import phonetic_similarity


@pytest.mark.parametrize(
    ("left", "right", "minimum"),
    [
        ("دخم", "ضخمة", 0.72),
        ("صاب", "صعب", 0.72),
        ("مش لعين ياكل", "مش لاقيين ياكلوا", 0.72),
        ("في الوقت", "فيالوقت", 0.95),
    ],
)
def test_egyptian_near_matches_score_high(left: str, right: str, minimum: float) -> None:
    """Likely phonetic ASR forms remain distinguishable from unrelated text."""

    assert phonetic_similarity(left, right) >= minimum


def test_unrelated_story_insertion_scores_low() -> None:
    """Fluent but unrelated clauses cannot pass the phonetic plausibility gate."""

    assert phonetic_similarity("الراجل وصل", "الرئيس شرح تاريخ الشركة") < 0.55


def test_latin_and_numbers_require_exact_preservation() -> None:
    """Arabic normalization never converts protected code-switched evidence."""

    assert phonetic_similarity("United Fruit Company 1954", "United Fruit Company 1954") == 1.0
    assert phonetic_similarity("United Fruit Company 1954", "United Fruit Company 1955") < 0.95
