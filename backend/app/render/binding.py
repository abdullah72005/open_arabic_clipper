"""Pure FINAL_CLIP source-span rebinding.

Rebinds Stage 4.1 SOURCE_EXCERPT blocks (copied analysis text/timings) onto the
current FINAL_CLIP word evidence without ever mutating the frozen plan. All
alignment is deterministic stdlib work: no provider, model, network, or audio
decoding.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.candidates.text import matching_text
from app.render.policy import (
    ALIGNMENT_TIME_WINDOW_SECONDS,
    CONTRACTION_EXPANSIONS,
    FILLER_TOKENS,
    MAX_ALIGNMENT_WORDS,
    MIN_ALIGNMENT_TOKEN_COVERAGE,
    PLANNING_BOUNDS_TOLERANCE_SECONDS,
)
from app.render.types import BoundSourceSpan, ClipWord

_TOKEN = re.compile(r"[\u0600-\u06FF]+|[A-Za-z]+|\d+(?:[.,:/-]\d+)*")
_PUNCTUATION = re.compile(r"[.!؟?…,;:\"'`\-–—()\[\]{}]+")
_SPACE = re.compile(r"\s+")

# NFKC does not fold typographic/alternate apostrophes to ASCII, so ``can’t``
# and ``can't`` must be unified explicitly before contraction expansion. The
# frozen Stage 4.2 vocabulary is mirrored here; Stage 4.2 stays untouched.
_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u02bc": "'",  # modifier letter apostrophe
        "\uff07": "'",  # fullwidth apostrophe
        "\u0060": "'",  # grave accent
        "\u00b4": "'",  # acute accent
    }
)


def normalize_apostrophes(text: str) -> str:
    """Fold typographic/alternate apostrophes to ASCII before comparison."""

    return text.translate(_APOSTROPHE_TRANSLATION)


def expand_contractions(text: str) -> str:
    """Expand conservative English contractions before tokenizing."""

    expanded = text
    for contraction, replacement in CONTRACTION_EXPANSIONS.items():
        pattern = re.compile(
            rf"(?<![A-Za-z]){re.escape(contraction)}(?![A-Za-z])",
            re.IGNORECASE,
        )
        expanded = pattern.sub(replacement, expanded)
    return expanded


def analysis_tokens(text: str, *, keep_filler: bool = True) -> list[str]:
    """Deterministic comparison token sequence (never a display replacement).

    NFKC, Arabic diacritics/tatweel stripped, alif/ya unified, Latin casefolded,
    punctuation removed, contractions expanded before tokenizing.
    """

    if not text:
        return []
    expanded = expand_contractions(normalize_apostrophes(text))
    normalized = matching_text(expanded)
    normalized = _PUNCTUATION.sub(" ", normalized)
    tokens = _TOKEN.findall(normalized)
    if keep_filler:
        return tokens
    return [token for token in tokens if token not in FILLER_TOKENS]


def content_tokens(text: str) -> list[str]:
    """Content tokens with filler removed, used for change-ratio accounting."""

    return analysis_tokens(text, keep_filler=False)


def has_digit(token: str) -> bool:
    return any(character.isdigit() for character in token)


def word_text(words: Sequence[ClipWord], start_index: int | None, end_index: int | None) -> str:
    """Rebuild display text from FINAL_CLIP word evidence only."""

    if not words:
        return ""
    low = 0 if start_index is None else max(0, start_index)
    high = len(words) - 1 if end_index is None else min(len(words) - 1, end_index)
    if high < low:
        return ""
    return _SPACE.sub(" ", " ".join(word.text for word in words[low : high + 1])).strip()


@dataclass(frozen=True)
class Alignment:
    """Result of one deterministic excerpt-to-word alignment."""

    structural_valid: bool
    full_window: bool = False
    matched: bool = False
    ambiguous: bool = False
    coverage: float = 0.0
    rebound_start: float | None = None
    rebound_end: float | None = None
    word_start: int | None = None
    word_end: int | None = None
    drift: float | None = None
    stats: Mapping[str, object] = field(default_factory=dict)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_full_window_sentinel(block: Mapping[str, object]) -> bool:
    return block.get("word_start_index") is None and block.get("word_end_index") is None


def align_excerpt(
    block: Mapping[str, object],
    words: Sequence[ClipWord],
    *,
    planning_refined_start: float | None,
    planning_refined_end: float | None,
    alignment_window: float = ALIGNMENT_TIME_WINDOW_SECONDS,
    max_words: int = MAX_ALIGNMENT_WORDS,
    min_coverage: float = MIN_ALIGNMENT_TOKEN_COVERAGE,
    bounds_tolerance: float = PLANNING_BOUNDS_TOLERANCE_SECONDS,
) -> Alignment:
    """Bind one plan SOURCE_EXCERPT to FINAL_CLIP word evidence.

    Structural validation is always applied. A full-window sentinel binds
    directly to the FINAL_CLIP refined bounds. Otherwise bounded token alignment
    inside a time window around the planning bounds determines the rebound span.
    """

    plan_start = block.get("source_start")
    plan_end = block.get("source_end")
    if not _finite(plan_start) or not _finite(plan_end):
        return Alignment(structural_valid=False)
    start = float(plan_start)  # type: ignore[arg-type]
    end = float(plan_end)  # type: ignore[arg-type]
    if start < 0 or end <= start:
        return Alignment(structural_valid=False)
    if planning_refined_start is not None and start < planning_refined_start - bounds_tolerance:
        return Alignment(structural_valid=False)
    if planning_refined_end is not None and end > planning_refined_end + bounds_tolerance:
        return Alignment(structural_valid=False)
    if is_full_window_sentinel(block):
        return Alignment(
            structural_valid=True,
            full_window=True,
            matched=True,
            coverage=1.0,
            rebound_start=planning_refined_start,
            rebound_end=planning_refined_end,
            stats={"mode": "full_window_sentinel"},
        )
    if not words:
        return Alignment(structural_valid=True, matched=False, stats={"mode": "no_words"})

    window = [
        word
        for word in words
        if word.end >= start - alignment_window and word.start <= end + alignment_window
    ]
    if len(window) > max_words:
        window = window[:max_words]
    if not window:
        return Alignment(structural_valid=True, matched=False, stats={"mode": "empty_window"})

    plan_tokens = content_tokens(str(block.get("source_text") or ""))
    if not plan_tokens:
        plan_tokens = analysis_tokens(str(block.get("source_text") or ""))
    if not plan_tokens:
        return Alignment(structural_valid=True, matched=False, stats={"mode": "no_plan_tokens"})

    window_tokens: list[str] = []
    token_word_index: list[int] = []
    for word in window:
        for token in analysis_tokens(word.text):
            window_tokens.append(token)
            token_word_index.append(word.index)

    matcher = difflib.SequenceMatcher(None, plan_tokens, window_tokens, autojunk=False)
    matched_plan = 0
    matched_positions: list[int] = []
    blocks = 0
    for block_match in matcher.get_matching_blocks():
        if block_match.size <= 0:
            continue
        blocks += 1
        matched_plan += block_match.size
        matched_positions.extend(range(block_match.b, block_match.b + block_match.size))

    if not matched_positions:
        return Alignment(
            structural_valid=True,
            matched=False,
            coverage=0.0,
            stats={"mode": "no_shared_token", "plan_tokens": len(plan_tokens)},
        )

    coverage = matched_plan / max(1, len(plan_tokens))
    if coverage < min_coverage:
        return Alignment(
            structural_valid=True,
            matched=False,
            coverage=coverage,
            stats={"mode": "low_coverage", "plan_tokens": len(plan_tokens)},
        )

    if _ambiguous(plan_tokens, window_tokens, min_coverage):
        return Alignment(
            structural_valid=True,
            matched=True,
            ambiguous=True,
            coverage=coverage,
            stats={"mode": "ambiguous", "matching_blocks": blocks},
        )

    matched_positions.sort()
    first_token = matched_positions[0]
    last_token = matched_positions[-1]
    first_word = token_word_index[first_token]
    last_word = token_word_index[last_token]
    rebound_start = window[first_word].start
    rebound_end = window[last_word].end
    drift = max(abs(rebound_start - start), abs(rebound_end - end))
    return Alignment(
        structural_valid=True,
        matched=True,
        coverage=coverage,
        rebound_start=rebound_start,
        rebound_end=rebound_end,
        word_start=window[first_word].index,
        word_end=window[last_word].index,
        drift=drift,
        stats={
            "mode": "aligned",
            "coverage": round(coverage, 6),
            "matching_blocks": blocks,
            "plan_tokens": len(plan_tokens),
        },
    )


def _ambiguous(
    plan_tokens: Sequence[str], window_tokens: Sequence[str], min_coverage: float
) -> bool:
    """True when the excerpt head matches at multiple competing window origins."""

    if len(plan_tokens) < 3:
        return False
    head = list(plan_tokens[:4])
    origins: list[int] = []
    for origin in range(len(window_tokens) - len(head) + 1):
        if list(window_tokens[origin : origin + len(head)]) == head:
            origins.append(origin)
    if len(origins) < 2:
        return False
    window = window_tokens
    strong = 0
    for origin in origins:
        matcher = difflib.SequenceMatcher(None, plan_tokens, window[origin:], autojunk=False)
        matched = sum(match.size for match in matcher.get_matching_blocks())
        if matched / max(1, len(plan_tokens)) >= min_coverage:
            strong += 1
    return strong >= 2


def build_bound_spans(
    blocks: Sequence[Mapping[str, object]],
    verdicts: Mapping[int, object],
    *,
    hero_block_index: int | None,
    words: Sequence[ClipWord],
) -> list[BoundSourceSpan]:
    """Build bound spans for every SOURCE_EXCERPT block from its verdict."""

    spans: list[BoundSourceSpan] = []
    for block in blocks:
        if str(block.get("block_type")) != "SOURCE_EXCERPT":
            continue
        index = _as_int(block.get("index"))
        verdict = verdicts.get(index)
        outcome = getattr(verdict, "outcome", None)
        rebind_valid = outcome in {"EXACT_MATCH", "COMPATIBLE_NON_MATERIAL_CHANGE"}
        rebound_start = getattr(verdict, "rebound_start", None)
        rebound_end = getattr(verdict, "rebound_end", None)
        word_start = getattr(verdict, "rebound_word_start", None)
        word_end = getattr(verdict, "rebound_word_end", None)
        role = block.get("source_role")
        spans.append(
            BoundSourceSpan(
                block_index=index,
                planning_source_start=_opt_float(block.get("source_start")),
                planning_source_end=_opt_float(block.get("source_end")),
                planning_word_start=_opt_int(block.get("word_start_index")),
                planning_word_end=_opt_int(block.get("word_end_index")),
                planning_text=str(block.get("source_text") or ""),
                final_clip_start=rebound_start,
                final_clip_end=rebound_end,
                final_clip_word_start=word_start,
                final_clip_word_end=word_end,
                final_clip_text=word_text(words, word_start, word_end),
                source_role=str(role) if role is not None else None,
                is_hero=hero_block_index is not None and index == hero_block_index,
                compatibility_outcome=str(outcome or ""),
                preservation_constraints=tuple(
                    str(item) for item in _as_list(block.get("preservation_constraints"))
                ),
                rebind_valid=rebind_valid,
            )
        )
    return spans


def _opt_float(value: object) -> float | None:
    if _finite(value):
        return float(value)  # type: ignore[arg-type]
    return None


def _opt_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _as_list(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def edit_ratio(left: Sequence[str], right: Sequence[str]) -> float:
    """Change ratio over content tokens, bounded to ``[0, 1]``."""

    if not left and not right:
        return 0.0
    matcher = difflib.SequenceMatcher(None, list(left), list(right), autojunk=False)
    matched = sum(match.size for match in matcher.get_matching_blocks())
    return max(0.0, 1.0 - matched / max(len(left), len(right), 1))


__all__ = [
    "Alignment",
    "align_excerpt",
    "analysis_tokens",
    "build_bound_spans",
    "content_tokens",
    "edit_ratio",
    "expand_contractions",
    "has_digit",
    "is_full_window_sentinel",
    "normalize_apostrophes",
    "word_text",
]
