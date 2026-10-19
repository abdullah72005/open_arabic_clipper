"""Strict shared grammar for Stage 4 grounding references.

Stage 4 providers cite evidence with bounded strings such as ``block:2``,
``words:3-7``, ``source:142.5-150.66``, or an exact source quote. Object forms
with an explicit ``kind`` are also supported. This module only parses the
declaration; it never resolves it against source evidence.

Parsing is intentionally strict: the complete string must match a supported
grammar. A malformed value is never reduced to a convenient numeric substring,
and a malformed container becomes an explicit invalid declaration rather than an
empty reference list.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

SOURCE_EXCERPT = "SOURCE_EXCERPT"
FINAL_CLIP = "FINAL_CLIP"
WORD_RANGE = "WORD_RANGE"
SOURCE_SPAN = "SOURCE_SPAN"
QUOTE = "QUOTE"
INVALID = "INVALID"

_BLOCK_RE = re.compile(r"(?i)^(block|blocks|excerpt|source_block|source-block):\s*(-?\d+)\s*$")
_WORD_RE = re.compile(r"(?i)^(word|words):\s*(.*?)\s*$")
_WORD_SINGLE_RE = re.compile(r"^(-?\d+)$")
_WORD_RANGE_RE = re.compile(r"^(-?\d+)\s*[-:]\s*(-?\d+)$")
_SPAN_RE = re.compile(r"(?i)^(source|span|time):\s*(.*?)\s*$")
_FLOAT_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_FLOAT_RANGE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*[-:]\s*(-?\d+(?:\.\d+)?)$")
_DIGITS_RE = re.compile(r"^\d+$")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]*:")

_KIND_ALIASES = {
    "SOURCE_EXCERPT": SOURCE_EXCERPT,
    "SOURCE_BLOCK": SOURCE_EXCERPT,
    "SOURCE-BLOCK": SOURCE_EXCERPT,
    "BLOCK": SOURCE_EXCERPT,
    "BLOCKS": SOURCE_EXCERPT,
    "EXCERPT": SOURCE_EXCERPT,
    "FINAL_CLIP": FINAL_CLIP,
    "FINALCLIP": FINAL_CLIP,
    "WORD": WORD_RANGE,
    "WORDS": WORD_RANGE,
    "WORD_RANGE": WORD_RANGE,
    "SOURCE": SOURCE_SPAN,
    "SPAN": SOURCE_SPAN,
    "TIME": SOURCE_SPAN,
    "SOURCE_SPAN": SOURCE_SPAN,
    "QUOTE": QUOTE,
    "TEXT": QUOTE,
    "INVALID": INVALID,
}


@dataclass(frozen=True)
class ParsedReference:
    kind: str
    raw: str
    block_index: int | None = None
    word_start_index: int | None = None
    word_end_index: int | None = None
    source_start: float | None = None
    source_end: float | None = None
    quote: str | None = None
    claim_id: str | None = None
    error: str | None = None


def _invalid(raw: object, reason: str) -> ParsedReference:
    return ParsedReference(kind=INVALID, raw=str(raw)[:400], error=reason)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_claim_id(value: object) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, "claim_id_not_string"
    text = value.strip()
    return (text or None), None


def _parse_mapping(value: Mapping[str, object]) -> ParsedReference:
    raw_value = value.get("raw")
    raw = raw_value[:400] if isinstance(raw_value, str) and raw_value else repr(dict(value))[:400]
    kind_value = value.get("kind")
    if not isinstance(kind_value, str) or not kind_value.strip():
        return _invalid(raw, "missing_or_invalid_kind")
    kind = _KIND_ALIASES.get(kind_value.strip().upper())
    if kind is None or kind is INVALID:
        return _invalid(raw, "unsupported_reference_kind")
    claim_id, claim_error = _optional_claim_id(value.get("claim_id"))
    if claim_error is not None:
        return _invalid(raw, claim_error)

    if kind is FINAL_CLIP:
        return ParsedReference(kind=kind, raw=raw, claim_id=claim_id)

    if kind is SOURCE_EXCERPT:
        index = value.get("block_index")
        if isinstance(index, bool) or not isinstance(index, int):
            return _invalid(raw, "missing_or_invalid_block_index")
        if index < 0:
            return _invalid(raw, "negative_block_index")
        return ParsedReference(kind=kind, raw=raw, block_index=index, claim_id=claim_id)

    if kind is WORD_RANGE:
        start = value.get("word_start_index")
        end = value.get("word_end_index")
        if isinstance(start, bool) or not isinstance(start, int):
            return _invalid(raw, "missing_or_invalid_word_start_index")
        if isinstance(end, bool) or not isinstance(end, int):
            return _invalid(raw, "missing_or_invalid_word_end_index")
        if start < 0 or end < 0:
            return _invalid(raw, "negative_word_index")
        if end < start:
            return _invalid(raw, "reversed_word_range")
        return ParsedReference(
            kind=kind,
            raw=raw,
            word_start_index=start,
            word_end_index=end,
            claim_id=claim_id,
        )

    if kind is SOURCE_SPAN:
        start_value = _finite(value.get("source_start"))
        end_value = _finite(value.get("source_end"))
        if start_value is None:
            return _invalid(raw, "missing_or_non_finite_source_start")
        if end_value is None:
            return _invalid(raw, "missing_or_non_finite_source_end")
        if start_value < 0 or end_value < 0:
            return _invalid(raw, "negative_source_time")
        if end_value < start_value:
            return _invalid(raw, "reversed_source_span")
        return ParsedReference(
            kind=kind,
            raw=raw,
            source_start=start_value,
            source_end=end_value,
            claim_id=claim_id,
        )

    if kind is QUOTE:
        quote = value.get("quote", value.get("text"))
        if not isinstance(quote, str) or not quote.strip():
            return _invalid(raw, "missing_or_empty_quote")
        return ParsedReference(kind=kind, raw=raw, quote=quote.strip(), claim_id=claim_id)

    return _invalid(raw, "unsupported_reference_kind")


def parse_reference(value: object) -> ParsedReference:
    """Parse one string/object declaration without resolving it."""

    if isinstance(value, Mapping):
        return _parse_mapping(value)
    if not isinstance(value, str):
        return _invalid(value, "unsupported_reference_type")

    raw = value[:400]
    text = value.strip()
    if not text:
        return _invalid(raw, "empty_reference")
    if text.casefold() == "final_clip":
        return ParsedReference(kind=FINAL_CLIP, raw=raw)

    match = _BLOCK_RE.fullmatch(text)
    if match is not None:
        index = int(match.group(2))
        if index < 0:
            return _invalid(raw, "negative_block_index")
        return ParsedReference(kind=SOURCE_EXCERPT, raw=raw, block_index=index)

    match = _WORD_RE.fullmatch(text)
    if match is not None:
        body = match.group(2)
        single = _WORD_SINGLE_RE.fullmatch(body)
        if single is not None:
            index = int(single.group(1))
            if index < 0:
                return _invalid(raw, "negative_word_index")
            return ParsedReference(
                kind=WORD_RANGE,
                raw=raw,
                word_start_index=index,
                word_end_index=index,
            )
        ranged = _WORD_RANGE_RE.fullmatch(body)
        if ranged is None:
            return _invalid(raw, "malformed_word_range")
        start = int(ranged.group(1))
        end = int(ranged.group(2))
        if start < 0 or end < 0:
            return _invalid(raw, "negative_word_index")
        if end < start:
            return _invalid(raw, "reversed_word_range")
        return ParsedReference(
            kind=WORD_RANGE,
            raw=raw,
            word_start_index=start,
            word_end_index=end,
        )

    match = _SPAN_RE.fullmatch(text)
    if match is not None:
        body = match.group(2)
        single = _FLOAT_RE.fullmatch(body)
        if single is not None:
            start = float(body)
            if not math.isfinite(start):
                return _invalid(raw, "non_finite_source_time")
            if start < 0:
                return _invalid(raw, "negative_source_time")
            return ParsedReference(kind=SOURCE_SPAN, raw=raw, source_start=start, source_end=start)
        ranged = _FLOAT_RANGE_RE.fullmatch(body)
        if ranged is None:
            return _invalid(raw, "malformed_source_span")
        start = float(ranged.group(1))
        end = float(ranged.group(2))
        if not math.isfinite(start) or not math.isfinite(end):
            return _invalid(raw, "non_finite_source_time")
        if start < 0 or end < 0:
            return _invalid(raw, "negative_source_time")
        if end < start:
            return _invalid(raw, "reversed_source_span")
        return ParsedReference(kind=SOURCE_SPAN, raw=raw, source_start=start, source_end=end)

    if _DIGITS_RE.fullmatch(text):
        return ParsedReference(kind=SOURCE_EXCERPT, raw=raw, block_index=int(text))

    if _LABEL_RE.match(text):
        return _invalid(raw, "unsupported_reference_label")

    # A plain declaration is an exact-source quote declaration. It is still
    # resolved strictly against approved source-excerpt evidence; no fallback is
    # ever substituted.
    return ParsedReference(kind=QUOTE, raw=raw, quote=text)


def parse_reference_list(value: object) -> tuple[ParsedReference, ...]:
    """Parse a declared reference container.

    ``None`` means absent optional evidence. Any other non-list container is an
    explicit invalid declaration, never an empty list.
    """

    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        return (_invalid(value, "invalid_reference_container"),)
    parsed: list[ParsedReference] = []
    for item in value:
        parsed.append(parse_reference(item))
        if len(parsed) >= 64:
            break
    return tuple(parsed)


def canonical_reference(parsed: ParsedReference) -> str:
    """Stable canonical string for claim-ledger persistence."""

    if parsed.kind is INVALID:
        return parsed.raw
    if parsed.kind is FINAL_CLIP:
        return "final_clip"
    if parsed.kind is SOURCE_EXCERPT:
        return f"block:{parsed.block_index}"
    if parsed.kind is WORD_RANGE:
        if parsed.word_start_index == parsed.word_end_index:
            return f"word:{parsed.word_start_index}"
        return f"words:{parsed.word_start_index}-{parsed.word_end_index}"
    if parsed.kind is SOURCE_SPAN:
        return f"source:{parsed.source_start}-{parsed.source_end}"
    if parsed.kind is QUOTE:
        return parsed.quote or ""
    return parsed.raw


__all__ = [
    "FINAL_CLIP",
    "INVALID",
    "QUOTE",
    "SOURCE_EXCERPT",
    "SOURCE_SPAN",
    "WORD_RANGE",
    "ParsedReference",
    "canonical_reference",
    "parse_reference",
    "parse_reference_list",
]
