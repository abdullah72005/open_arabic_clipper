"""Stage 5.0 source-span rebinding primitives."""

from __future__ import annotations

from stage50_support import clip_words, source_block

from app.render.binding import align_excerpt, build_bound_spans, word_text
from app.render.types import ClipWord

TEXT = "alpha beta gamma delta epsilon zeta"


def test_full_window_sentinel_rebinds_to_refined_bounds() -> None:
    words = clip_words(TEXT, start=21.0, end=30.0)
    block = source_block(TEXT, word_start_index=None, word_end_index=None)
    alignment = align_excerpt(block, words, planning_refined_start=20.5, planning_refined_end=44.5)
    assert alignment.full_window is True
    assert alignment.rebound_start == 20.5
    assert alignment.rebound_end == 44.5


def test_bounded_alignment_selects_matching_words() -> None:
    words = clip_words(TEXT, start=21.0, end=30.0)
    block = source_block(
        "beta gamma delta",
        source_start=words[1].start,
        source_end=words[3].end,
        word_start_index=1,
        word_end_index=3,
    )
    alignment = align_excerpt(block, words, planning_refined_start=20.5, planning_refined_end=44.5)
    assert alignment.matched is True
    assert alignment.rebound_start == words[1].start
    assert alignment.rebound_end == words[3].end


def test_out_of_bounds_span_is_structurally_invalid() -> None:
    words = clip_words(TEXT, start=21.0, end=30.0)
    block = source_block(TEXT, source_start=1.0, source_end=2.0)
    alignment = align_excerpt(block, words, planning_refined_start=20.5, planning_refined_end=44.5)
    assert alignment.structural_valid is False


def test_bound_span_text_uses_final_clip_words_not_plan_text() -> None:
    plan_text = "alpha beta gamma"
    final_text = "alpha beta gamma changed"
    words = clip_words(final_text, start=21.0, end=30.0)
    block = source_block(plan_text, word_start_index=0, word_end_index=2)
    alignment = align_excerpt(block, words, planning_refined_start=20.5, planning_refined_end=44.5)
    assert alignment.matched is True

    class _Verdict:
        outcome = "COMPATIBLE_NON_MATERIAL_CHANGE"
        rebound_start = words[0].start
        rebound_end = words[2].end
        rebound_word_start = 0
        rebound_word_end = 2

    spans = build_bound_spans([block], {0: _Verdict()}, hero_block_index=0, words=words)
    assert spans[0].planning_text == plan_text
    assert spans[0].final_clip_text == "alpha beta gamma"


def test_word_text_rebuilds_from_indexes() -> None:
    words = (ClipWord(0, "a", 0.0, 1.0), ClipWord(1, "b", 1.0, 2.0))
    assert word_text(words, 0, 1) == "a b"
    assert word_text(words, None, None) == "a b"


_SPACED_TEXT = "one two three four five six seven eight nine ten eleven twelve"


def _spaced_clip(indices: list[int] | None = None, *, step: float = 5.0) -> tuple[ClipWord, ...]:
    tokens = _SPACED_TEXT.split()
    resolved = indices if indices is not None else list(range(len(tokens)))
    return tuple(
        ClipWord(
            index=resolved[i],
            text=token,
            start=round(i * step, 3),
            end=round((i + 1) * step, 3),
            probability=0.9,
        )
        for i, token in enumerate(tokens)
    )


def test_dense_global_ids_with_late_filtered_window_returns_authoritative_ids() -> None:
    # The time window drops the leading words, so the surviving window's
    # positions (0..6) no longer equal the surviving words' dense global IDs
    # (5..11). Indexing the filtered window with the global ID used to raise
    # IndexError or rebind to the wrong words.
    words = _spaced_clip()
    block = source_block(
        "eight nine ten",
        source_start=words[7].start,
        source_end=words[9].end,
        word_start_index=7,
        word_end_index=9,
    )
    alignment = align_excerpt(block, words, planning_refined_start=0.0, planning_refined_end=60.0)
    assert alignment.matched is True
    assert alignment.ambiguous is False
    assert alignment.word_start == 7
    assert alignment.word_end == 9
    assert alignment.rebound_start == words[7].start
    assert alignment.rebound_end == words[9].end


def test_sparse_nonzero_ids_return_authoritative_ids_and_times() -> None:
    indices = [100 + offset for offset in range(12)]
    words = _spaced_clip(indices)
    block = source_block(
        "eight nine ten",
        source_start=words[7].start,
        source_end=words[9].end,
        word_start_index=107,
        word_end_index=109,
    )
    alignment = align_excerpt(block, words, planning_refined_start=0.0, planning_refined_end=60.0)
    assert alignment.matched is True
    assert alignment.word_start == 107
    assert alignment.word_end == 109
    assert alignment.rebound_start == words[7].start
    assert alignment.rebound_end == words[9].end
