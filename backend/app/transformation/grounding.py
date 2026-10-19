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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SOURCE_EXCERPT = "SOURCE_EXCERPT"
FINAL_CLIP = "FINAL_CLIP"
WORD_RANGE = "WORD_RANGE"
SOURCE_SPAN = "SOURCE_SPAN"
QUOTE = "QUOTE"
INVALID = "INVALID"

GROUNDING_REQUIRED_MISSING = "GROUNDING_REQUIRED_MISSING"
GROUNDING_DECLARATION_MALFORMED = "GROUNDING_DECLARATION_MALFORMED"
GROUNDING_EVIDENCE_UNRESOLVED = "GROUNDING_EVIDENCE_UNRESOLVED"

_SOURCE_GROUNDED_BLOCK_TYPES = frozenset({"ORIGINAL_VALUE", "TEXTUAL_ANNOTATION"})
_RANGE_TOLERANCE_SECONDS = 1e-6
_MAX_DECLARATIONS = 64

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


@dataclass(frozen=True)
class SourceGroundingFailure:
    code: str
    block_index: int
    declaration: str | None
    detail: str

    def reason(self) -> str:
        if self.declaration is None:
            return f"{self.code}[block={self.block_index}]: {self.detail}"
        return (
            f"{self.code}[block={self.block_index},"
            f"declaration={self.declaration[:200]!r}]: {self.detail}"
        )


def _block_index(block: Mapping[str, object]) -> int:
    value = block.get("index")
    if isinstance(value, bool) or not isinstance(value, int):
        return -1
    return int(value)


def _block_text(block: Mapping[str, object]) -> str:
    for key in ("source_text", "planning_source_text"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _block_span(block: Mapping[str, object]) -> tuple[float, float] | None:
    start = block.get("source_start")
    end = block.get("source_end")
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return None
    if isinstance(end, bool) or not isinstance(end, (int, float)):
        return None
    start_f = float(start)
    end_f = float(end)
    if not math.isfinite(start_f) or not math.isfinite(end_f) or end_f < start_f:
        return None
    return start_f, end_f


def _word_range(block: Mapping[str, object]) -> tuple[int, int] | None:
    start = block.get("word_start_index", block.get("planning_word_start_index"))
    end = block.get("word_end_index", block.get("planning_word_end_index"))
    if isinstance(start, bool) or not isinstance(start, int):
        return None
    if isinstance(end, bool) or not isinstance(end, int):
        return None
    if start < 0 or end < start:
        return None
    return int(start), int(end)


def _word_evidence(
    block: Mapping[str, object],
) -> tuple[tuple[int, float | None, float | None, str], ...]:
    bounds = _word_range(block)
    raw = block.get("planning_words")
    if bounds is None or not isinstance(raw, (list, tuple)):
        return ()
    start, end = bounds
    words = [item for item in raw if isinstance(item, Mapping)]
    if len(words) != end - start + 1:
        return ()
    result: list[tuple[int, float | None, float | None, str]] = []
    for offset, word in enumerate(words):
        text = word.get("text", word.get("word"))
        if not isinstance(text, str) or not text.strip():
            return ()
        index = word.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            index = start + offset
        word_start = word.get("start")
        word_end = word.get("end")
        start_value = (
            float(word_start)
            if isinstance(word_start, (int, float)) and not isinstance(word_start, bool)
            else None
        )
        end_value = (
            float(word_end)
            if isinstance(word_end, (int, float)) and not isinstance(word_end, bool)
            else None
        )
        result.append((int(index), start_value, end_value, text.strip()))
    return tuple(result)


def _word_map(block: Mapping[str, object]) -> dict[int, str]:
    return {index: text for index, _start, _end, text in _word_evidence(block)}


def _source_blocks(
    blocks: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return [block for block in blocks if str(block.get("block_type", "")) == SOURCE_EXCERPT]


def _resolve_block(
    index: int, source_blocks: Sequence[Mapping[str, object]]
) -> tuple[str | None, str | None]:
    block = next(
        (item for item in source_blocks if _block_index(item) == index),
        None,
    )
    if block is None:
        return None, "non_source_or_unknown_block"
    text = _block_text(block)
    if not text:
        return None, "empty_source_block"
    return text, None


def _resolve_word_range(
    start: int, end: int, source_blocks: Sequence[Mapping[str, object]]
) -> tuple[str | None, str | None]:
    words: dict[int, str] = {}
    for block in source_blocks:
        words.update(_word_map(block))
    selected = [words.get(index) for index in range(start, end + 1)]
    if any(text is None for text in selected):
        return None, "word_range_not_fully_covered_by_selected_evidence"
    return " ".join(text for text in selected if text), None


def _range_covered(spans: Sequence[tuple[float, float]], start: float, end: float) -> bool:
    cursor = start
    for left, right in sorted(spans):
        if right + _RANGE_TOLERANCE_SECONDS < cursor:
            continue
        if left > cursor + _RANGE_TOLERANCE_SECONDS:
            return False
        cursor = max(cursor, right)
        if cursor + _RANGE_TOLERANCE_SECONDS >= end:
            return True
    return cursor + _RANGE_TOLERANCE_SECONDS >= end


def _resolve_source_span(
    start: float, end: float, source_blocks: Sequence[Mapping[str, object]]
) -> tuple[str | None, str | None]:
    spans = [(span[0], span[1]) for block in source_blocks if (span := _block_span(block))]
    if not spans or not _range_covered(spans, start, end):
        return None, "source_span_out_of_selected_scope"
    words: list[tuple[float, float, str]] = []
    for block in source_blocks:
        for _index, word_start, word_end, text in _word_evidence(block):
            if word_start is None or word_end is None:
                continue
            if not math.isfinite(word_start) or not math.isfinite(word_end):
                continue
            words.append((word_start, word_end, text))
    inside = [
        text
        for word_start, word_end, text in sorted(words)
        if word_start >= start - _RANGE_TOLERANCE_SECONDS
        and word_end <= end + _RANGE_TOLERANCE_SECONDS
    ]
    if inside:
        return " ".join(inside), None
    chunks = [
        _block_text(block)
        for block in source_blocks
        if (span := _block_span(block))
        and span[0] < end - _RANGE_TOLERANCE_SECONDS
        and span[1] > start + _RANGE_TOLERANCE_SECONDS
    ]
    text = " ".join(chunk for chunk in chunks if chunk)
    return (text, None) if text else (None, "selected_source_text_unavailable")


def _resolve_quote(
    quote: str, source_blocks: Sequence[Mapping[str, object]]
) -> tuple[str | None, str | None]:
    folded = quote.strip().casefold()
    if not folded:
        return None, "empty_quote"
    for block in source_blocks:
        text = _block_text(block)
        if text and text.casefold() == folded:
            return text, None
    return None, "quote_not_in_selected_source_excerpt"


def resolve_reference(
    ref: object, source_blocks: Sequence[Mapping[str, object]]
) -> tuple[str | None, str | None]:
    """Resolve one declaration against authoritative selected source blocks."""

    parsed = parse_reference(ref)
    if parsed.kind is INVALID:
        return None, f"malformed_declaration:{parsed.error or 'invalid_reference'}"
    if parsed.kind is SOURCE_EXCERPT:
        if parsed.block_index is None:
            return None, "missing_block_index"
        return _resolve_block(parsed.block_index, source_blocks)
    if parsed.kind is WORD_RANGE:
        start = parsed.word_start_index
        end = parsed.word_end_index
        if start is None or end is None:
            return None, "missing_word_range"
        return _resolve_word_range(start, end, source_blocks)
    if parsed.kind is SOURCE_SPAN:
        start = parsed.source_start
        end = parsed.source_end
        if start is None or end is None:
            return None, "missing_source_span"
        return _resolve_source_span(float(start), float(end), source_blocks)
    if parsed.kind is QUOTE:
        return _resolve_quote(parsed.quote or "", source_blocks)
    if parsed.kind is FINAL_CLIP:
        return None, "final_clip_is_not_stage4_source_evidence"
    return None, "unsupported_reference_kind"


def _declared_references(
    value: object,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    if value is None:
        return (), ()
    if not isinstance(value, (list, tuple)):
        return (), ((repr(value)[:200], "invalid_reference_container"),)
    declarations: list[str] = []
    errors: list[tuple[str, str]] = []
    items = list(value)
    if len(items) > _MAX_DECLARATIONS:
        errors.append(
            (
                repr(items[_MAX_DECLARATIONS])[:200],
                "too_many_reference_declarations",
            )
        )
    for item in items[:_MAX_DECLARATIONS]:
        if isinstance(item, str):
            declarations.append(item)
            continue
        errors.append(
            (
                repr(item)[:200],
                f"unsupported_reference_type:{type(item).__name__}",
            )
        )
    return tuple(declarations), tuple(errors)


def external_verification_linked_indexes(
    blocks: Sequence[Mapping[str, object]],
) -> frozenset[int]:
    """Substantive blocks whose external claim is genuinely verification-linked."""

    linked: set[int] = set()
    by_index = {_block_index(block): block for block in blocks}
    for placeholder in blocks:
        if str(placeholder.get("block_type", "")) != "FACT_VERIFICATION_PLACEHOLDER":
            continue
        if not bool(placeholder.get("must_verify_before_execution")):
            continue
        claim = placeholder.get("claim_dependency")
        dependents = placeholder.get("dependent_block_ids")
        if not isinstance(claim, str) or not claim.strip() or not isinstance(dependents, list):
            continue
        for dependent in dependents:
            if isinstance(dependent, bool) or not isinstance(dependent, int):
                continue
            target = by_index.get(int(dependent))
            if target is None:
                continue
            if str(target.get("block_type", "")) not in _SOURCE_GROUNDED_BLOCK_TYPES:
                continue
            dependencies = target.get("dependency_ids")
            if isinstance(dependencies, list) and claim in dependencies:
                linked.add(dependent)
    return frozenset(linked)


def validate_source_grounding(
    blocks: Sequence[Mapping[str, object]],
    *,
    external_linked_indexes: frozenset[int] | None = None,
) -> tuple[SourceGroundingFailure, ...]:
    """Validate every declared declaration and required substantive grounding."""

    block_list = [block for block in blocks if isinstance(block, Mapping)]
    source_blocks = _source_blocks(block_list)
    linked = (
        external_linked_indexes
        if external_linked_indexes is not None
        else external_verification_linked_indexes(block_list)
    )
    failures: list[SourceGroundingFailure] = []
    for block in block_list:
        index = _block_index(block)
        block_type = str(block.get("block_type", ""))
        required = block_type in _SOURCE_GROUNDED_BLOCK_TYPES and index not in linked
        declarations, errors = _declared_references(block.get("grounding_refs"))
        for declaration, detail in errors:
            failures.append(
                SourceGroundingFailure(
                    GROUNDING_DECLARATION_MALFORMED,
                    index,
                    declaration,
                    detail,
                )
            )
        if required and not declarations and not errors:
            failures.append(
                SourceGroundingFailure(
                    GROUNDING_REQUIRED_MISSING,
                    index,
                    None,
                    "substantive block requires at least one source grounding declaration",
                )
            )
        for declaration in declarations:
            _text, reason = resolve_reference(declaration, source_blocks)
            if reason is None:
                continue
            code = (
                GROUNDING_DECLARATION_MALFORMED
                if reason.startswith("malformed_declaration:")
                else GROUNDING_EVIDENCE_UNRESOLVED
            )
            detail = reason.split(":", 1)[1] if ":" in reason else reason
            failures.append(SourceGroundingFailure(code, index, declaration, detail))
    return tuple(failures)


__all__ = [
    "FINAL_CLIP",
    "GROUNDING_DECLARATION_MALFORMED",
    "GROUNDING_EVIDENCE_UNRESOLVED",
    "GROUNDING_REQUIRED_MISSING",
    "INVALID",
    "QUOTE",
    "SOURCE_EXCERPT",
    "SOURCE_SPAN",
    "WORD_RANGE",
    "ParsedReference",
    "SourceGroundingFailure",
    "canonical_reference",
    "external_verification_linked_indexes",
    "parse_reference",
    "parse_reference_list",
    "resolve_reference",
    "validate_source_grounding",
]
