"""F1 bounded display projection: authoritative text over immutable timed words."""

from __future__ import annotations

import math

from app.refinement.display import (
    REASON_AMBIGUOUS_TOKENIZATION,
    REASON_INVALID_TIMING,
    REASON_MISSING_WORDS,
    REASON_MULTIPLE_CHANGES,
    REASON_NONMONOTONIC_TIMING,
    REASON_TOKEN_COUNT,
    REASON_UNANCHORED,
    REPLACEMENT,
    UNCHANGED,
    project_display_text,
)
from app.render.fingerprints import build_caption_source_payload, caption_source_fingerprint


def _words(tokens: list[str], *, start: float = 10.0) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "text": token,
            "start": round(start + index, 3),
            "end": round(start + index + 0.4, 3),
            "probability": 0.9,
        }
        for index, token in enumerate(tokens)
    ]


def test_unchanged_text_preserves_words_and_whitespace_joining() -> None:
    words = _words(["النتيجة:", "OpenAI", "API", "v2", "شكرا"])
    projection = project_display_text("  النتيجة:   OpenAI API v2 شكرا  ", words)

    assert projection.supported is True
    assert projection.status == UNCHANGED
    assert projection.final_text == "النتيجة:   OpenAI API v2 شكرا"
    assert projection.words == tuple(words)
    assert projection.corrected_token_index is None


def test_interior_replacement_retains_original_word_identity_and_timing() -> None:
    words = _words(["promotion", "rates", "fell", "sharply.", "today"])
    projection = project_display_text("promotion rates fell slightly. today", words)

    assert projection.supported is True
    assert projection.status == REPLACEMENT
    assert projection.corrected_token_index == 3
    assert [word["text"] for word in projection.words] == [
        "promotion",
        "rates",
        "fell",
        "slightly.",
        "today",
    ]
    assert projection.words[3]["index"] == words[3]["index"]
    assert projection.words[3]["start"] == words[3]["start"]
    assert projection.words[3]["end"] == words[3]["end"]
    assert [word["text"] for word in words][3] == "sharply."


def test_edge_replacement_requires_two_unchanged_anchors() -> None:
    first = _words(["sharply.", "rates", "fell", "today"])
    first_projection = project_display_text("slightly. rates fell today", first)
    assert first_projection.supported is True
    assert first_projection.words[0]["text"] == "slightly."

    last = _words(["rates", "fell", "sharply."])
    last_projection = project_display_text("rates fell slightly.", last)
    assert last_projection.supported is True
    assert last_projection.words[2]["text"] == "slightly."


def test_edge_replacement_without_two_anchors_is_refused() -> None:
    projection = project_display_text("slightly. fell", _words(["sharply.", "fell"]))

    assert projection.supported is False
    assert projection.reason == REASON_UNANCHORED
    assert projection.final_text == "slightly. fell"
    assert projection.words == ()


def test_arabic_replacement_preserves_logical_order_and_unchanged_tokens() -> None:
    words = _words(["أنا", "عملت", "امبارح", "في", "البيت"])
    projection = project_display_text("أنا عملت النهاردة في البيت", words)

    assert projection.supported is True
    assert projection.corrected_token_index == 2
    assert [word["text"] for word in projection.words] == [
        "أنا",
        "عملت",
        "النهاردة",
        "في",
        "البيت",
    ]


def test_mixed_arabic_latin_protected_tokens_survive_exactly() -> None:
    tokens = [
        "النتيجة:",
        "OpenAI",
        "API",
        "v2",
        "https://example.com/a?b=1&c=%20",
        "C++",
        "#build",
        "foo/bar",
        "30",
    ]
    words = _words(tokens)
    projection = project_display_text(
        "النتيجة: OpenAI API v3 https://example.com/a?b=1&c=%20 C++ #build foo/bar 30",
        words,
    )

    assert projection.supported is True
    assert projection.corrected_token_index == 3
    assert [word["text"] for word in projection.words] == [
        "النتيجة:",
        "OpenAI",
        "API",
        "v3",
        "https://example.com/a?b=1&c=%20",
        "C++",
        "#build",
        "foo/bar",
        "30",
    ]


def test_insertion_deletion_and_reordering_are_refused() -> None:
    words = _words(["one", "two", "three"])

    insertion = project_display_text("one two inserted three", words)
    assert insertion.supported is False
    assert insertion.reason == REASON_TOKEN_COUNT
    assert insertion.final_text == "one two inserted three"

    deletion = project_display_text("one three", words)
    assert deletion.supported is False
    assert deletion.reason == REASON_TOKEN_COUNT

    reordered = project_display_text("three two one", words)
    assert reordered.supported is False
    assert reordered.reason == REASON_MULTIPLE_CHANGES


def test_multiple_changes_are_refused_even_with_equal_token_count() -> None:
    projection = project_display_text(
        "promotion rates rose slightly. today",
        _words(["promotion", "rates", "fell", "sharply.", "today"]),
    )

    assert projection.supported is False
    assert projection.reason == REASON_MULTIPLE_CHANGES
    assert projection.final_text == "promotion rates rose slightly. today"


def test_missing_and_invalid_timing_are_refused() -> None:
    assert project_display_text("hello", []).reason == REASON_MISSING_WORDS

    missing_end = [{"index": 0, "text": "hello", "start": 1.0}]
    assert project_display_text("hello", missing_end).reason == REASON_INVALID_TIMING  # type: ignore[arg-type]

    reversed_bounds = [{"index": 0, "text": "hello", "start": 2.0, "end": 1.0}]
    assert project_display_text("hello", reversed_bounds).reason == REASON_INVALID_TIMING  # type: ignore[arg-type]

    non_finite = [{"index": 0, "text": "hello", "start": math.nan, "end": 1.0}]
    assert project_display_text("hello", non_finite).reason == REASON_INVALID_TIMING  # type: ignore[arg-type]

    nonmonotonic = [
        {"index": 0, "text": "hello", "start": 2.0, "end": 2.4},
        {"index": 1, "text": "world", "start": 1.0, "end": 1.4},
    ]
    assert project_display_text("hello world", nonmonotonic).reason == REASON_NONMONOTONIC_TIMING  # type: ignore[arg-type]


def test_ambiguous_word_tokenization_is_refused() -> None:
    words = [{"index": 0, "text": "New York", "start": 1.0, "end": 1.5}]

    projection = project_display_text("New York", words)  # type: ignore[arg-type]

    assert projection.supported is False
    assert projection.reason == REASON_AMBIGUOUS_TOKENIZATION


def test_display_projection_policy_and_text_change_caption_fingerprint() -> None:
    words = _words(["promotion", "rates", "fell", "sharply."])
    base = build_caption_source_payload(
        final_transcript="promotion rates fell slightly.",
        word_timestamps=[
            {**words[index], "text": text}
            for index, text in enumerate(["promotion", "rates", "fell", "slightly."])
        ],
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
        code_switch_evidence={"suspected": False, "tokens": []},
        final_refinement_output_fingerprint="final-fp",
    )
    changed_policy = build_caption_source_payload(
        final_transcript="promotion rates fell slightly.",
        word_timestamps=base["word_timestamps"],  # type: ignore[arg-type]
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
        code_switch_evidence={"suspected": False, "tokens": []},
        final_refinement_output_fingerprint="final-fp",
        display_projection_policy_version="manual-display-projection-v2",
    )
    changed_text = build_caption_source_payload(
        final_transcript="promotion rates fell sharply.",
        word_timestamps=base["word_timestamps"],  # type: ignore[arg-type]
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
        code_switch_evidence={"suspected": False, "tokens": []},
        final_refinement_output_fingerprint="final-fp",
    )

    assert caption_source_fingerprint(base) != caption_source_fingerprint(changed_policy)
    assert caption_source_fingerprint(base) != caption_source_fingerprint(changed_text)
