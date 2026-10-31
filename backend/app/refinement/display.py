"""Deterministic display projection from authoritative text to timed words.

The audio-backed word evidence is immutable: Stage 2.5/2.7/3.5 text and every
word's identity, source-time bounds, and order are never rewritten here. This
module only derives *display* text for the existing ordered timed words from an
authoritative manual/final transcript.

The bounded policy is deliberately conservative and pure: no network, no model,
no audio decoding, no database access, and no normalization that could hide a
real difference.

- Unchanged text is always supported.
- Exactly one isolated one-token replacement may be projected onto the matching
  timed word, retaining that word's original index/start/end.
- An interior replacement requires both immediate neighbours unchanged.
- An edge replacement requires two unchanged adjacent tokens on its available
  side.
- Insertions, deletions, reordering, multiple replacements, unanchored changes,
  ambiguous tokenization, and missing/invalid/nonmonotonic timing are refused.

Refusal never discards the authoritative text and never falls back to ASR copy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

DISPLAY_PROJECTION_POLICY_VERSION = "manual-display-projection-v1"

UNCHANGED = "UNCHANGED"
REPLACEMENT = "REPLACEMENT"

REASON_MISSING_WORDS = "manual_transcript_without_word_timestamps"
REASON_EMPTY_TEXT = "manual_transcript_empty"
REASON_INVALID_WORD = "manual_transcript_invalid_word_evidence"
REASON_INVALID_TIMING = "manual_transcript_invalid_word_timing"
REASON_NONMONOTONIC_TIMING = "manual_transcript_nonmonotonic_word_timing"
REASON_AMBIGUOUS_TOKENIZATION = "manual_transcript_ambiguous_tokenization"
REASON_TOKEN_COUNT = "manual_transcript_token_count_mismatch"
REASON_MULTIPLE_CHANGES = "manual_transcript_multiple_token_changes"
REASON_UNANCHORED = "manual_transcript_unanchored_token_change"


@dataclass(frozen=True)
class DisplayProjection:
    """One deterministic projection result over ordered timed words."""

    supported: bool
    status: str
    final_text: str
    words: tuple[dict[str, object], ...]
    reason: str | None = None
    corrected_token_index: int | None = None
    policy_version: str = DISPLAY_PROJECTION_POLICY_VERSION


def _refuse(final_text: str, reason: str) -> DisplayProjection:
    return DisplayProjection(
        supported=False,
        status="UNSUPPORTED",
        final_text=final_text,
        words=(),
        reason=reason,
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _normalize_words(
    words: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str], str | None]:
    normalized: list[dict[str, object]] = []
    tokens: list[str] = []
    previous_start: float | None = None
    previous_index: int | None = None
    indexed_flags = [raw.get("index") is not None for raw in words if isinstance(raw, Mapping)]
    if any(indexed_flags) and not all(indexed_flags):
        return [], [], REASON_INVALID_WORD
    for position, raw in enumerate(words):
        if not isinstance(raw, Mapping):
            return [], [], REASON_INVALID_WORD
        item = dict(raw)
        text = item.get("text", item.get("word"))
        if not isinstance(text, str):
            return [], [], REASON_INVALID_WORD
        parts = text.split()
        if len(parts) != 1:
            return [], [], REASON_AMBIGUOUS_TOKENIZATION
        index = _integer(item.get("index"))
        if item.get("index") is not None and index is None:
            return [], [], REASON_INVALID_WORD
        if index is not None:
            if previous_index is not None and index <= previous_index:
                return [], [], REASON_INVALID_WORD
            previous_index = index
        start = _finite_number(item.get("start"))
        end = _finite_number(item.get("end"))
        if start is None or end is None:
            return [], [], REASON_INVALID_TIMING
        if start < 0.0 or end < start:
            return [], [], REASON_INVALID_TIMING
        if previous_start is not None and start < previous_start:
            return [], [], REASON_NONMONOTONIC_TIMING
        previous_start = start
        item["index"] = position if index is None else index
        item["text"] = parts[0]
        item["start"] = start
        item["end"] = end
        normalized.append(item)
        tokens.append(parts[0])
    return normalized, tokens, None


def project_display_text(
    final_text: str,
    words: Sequence[Mapping[str, object]],
) -> DisplayProjection:
    """Project authoritative text onto existing ordered timed words.

    The caller must preserve ``final_text`` verbatim when ``supported`` is
    ``False``; the returned ``words`` are empty in that case so no consumer can
    accidentally fall back to the unprojected ASR spelling.
    """

    cleaned = (final_text or "").strip()
    if not cleaned:
        return _refuse(cleaned, REASON_EMPTY_TEXT)
    if not words:
        return _refuse(cleaned, REASON_MISSING_WORDS)

    normalized, source_tokens, reason = _normalize_words(words)
    if reason is not None:
        return _refuse(cleaned, reason)

    final_tokens = cleaned.split()
    if len(final_tokens) != len(source_tokens):
        return _refuse(cleaned, REASON_TOKEN_COUNT)

    changed = [
        index
        for index, (display, source) in enumerate(zip(final_tokens, source_tokens, strict=True))
        if display != source
    ]
    if not changed:
        return DisplayProjection(
            supported=True,
            status=UNCHANGED,
            final_text=cleaned,
            words=tuple(normalized),
        )
    if len(changed) > 1:
        return _refuse(cleaned, REASON_MULTIPLE_CHANGES)

    index = changed[0]
    if not _anchored(final_tokens, source_tokens, index):
        return _refuse(cleaned, REASON_UNANCHORED)

    projected = [dict(item) for item in normalized]
    projected[index]["text"] = final_tokens[index]
    return DisplayProjection(
        supported=True,
        status=REPLACEMENT,
        final_text=cleaned,
        words=tuple(projected),
        corrected_token_index=index,
    )


def _anchored(display: Sequence[str], source: Sequence[str], index: int) -> bool:
    count = len(source)
    if index == 0:
        return count >= 3 and display[1] == source[1] and display[2] == source[2]
    if index == count - 1:
        return count >= 3 and display[-2] == source[-2] and display[-3] == source[-3]
    return display[index - 1] == source[index - 1] and display[index + 1] == source[index + 1]


def word_payloads(projection: DisplayProjection) -> list[dict[str, object]]:
    """Return the projected word payloads (empty when unsupported)."""

    return [dict(word) for word in projection.words]


def resolve_effective_display(
    final_text: str, words: Sequence[Mapping[str, object]]
) -> DisplayProjection:
    """Shared read-only effective word view for final-clip consumers.

    Stage 4 input assembly and Stage 5 caption composition both derive their
    display words through this same pure projection. Unsupported projection
    returns ``supported=False``; callers must refuse rather than fall back to the
    superseded word spelling.
    """

    return project_display_text(final_text, words)


__all__ = [
    "DISPLAY_PROJECTION_POLICY_VERSION",
    "DisplayProjection",
    "REASON_AMBIGUOUS_TOKENIZATION",
    "REASON_EMPTY_TEXT",
    "REASON_INVALID_TIMING",
    "REASON_INVALID_WORD",
    "REASON_MISSING_WORDS",
    "REASON_MULTIPLE_CHANGES",
    "REASON_NONMONOTONIC_TIMING",
    "REASON_TOKEN_COUNT",
    "REASON_UNANCHORED",
    "REPLACEMENT",
    "UNCHANGED",
    "project_display_text",
    "resolve_effective_display",
    "word_payloads",
]
