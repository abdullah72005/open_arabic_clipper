"""Strict Stage 6 grounding-reference parsing and authoritative resolution.

Stage 4.1 word references use the bounded *planning* word sequence. Stage 6
merged blocks separately expose FINAL_CLIP binding indexes. This module never
reinterprets a planning range in the FINAL_CLIP index space: it resolves the
cited planning range to source time, then reads the exact current evidence from
the authoritative Stage 5 binding / FINAL_CLIP words.

Only the cited range is resolved. There is no whole-transcript or neighboring
excerpt fallback. Invalid, unresolved, out-of-scope, and partially-overlapping
declarations are reported explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.materialization.inputs import MaterializationInputs
from app.materialization.types import GroundingRef
from app.transformation.grounding import (
    FINAL_CLIP,
    INVALID,
    QUOTE,
    SOURCE_EXCERPT,
    SOURCE_SPAN,
    WORD_RANGE,
    ParsedReference,
    parse_reference_list,
)

_RANGE_TOLERANCE_SECONDS = 1e-6


@dataclass(frozen=True)
class GroundingResolution:
    """Resolved grounding evidence for one authored block.

    ``declared`` records whether the block declared any grounding reference;
    ``unresolved`` records every declaration that could not be resolved to real
    source text, including its concrete reason.
    """

    text: str
    declared: bool
    unresolved: tuple[str, ...]
    errors: tuple[Mapping[str, object], ...] = ()


def parse_grounding_refs(value: object) -> tuple[GroundingRef, ...]:
    """Preserve every declared reference; invalid declarations stay explicit."""

    refs: list[GroundingRef] = []
    if isinstance(value, (list, tuple)):
        for item in value[:64]:
            if isinstance(item, GroundingRef):
                if item.kind == "INVALID":
                    refs.append(item)
                else:
                    refs.append(item)
                continue
            refs.append(_from_parsed(_parse_item(item)))
        return tuple(refs)
    if value is None:
        return ()
    return (
        GroundingRef(
            kind="INVALID",
            raw=str(value)[:400],
            invalid_reason="invalid_reference_container",
        ),
    )


def _parse_item(item: object) -> ParsedReference:
    if isinstance(item, GroundingRef):
        if item.kind == "INVALID":
            return ParsedReference(
                kind=INVALID,
                raw=item.raw or "",
                error=item.invalid_reason or "invalid_reference",
            )
        return ParsedReference(
            kind=item.kind.upper(),
            raw=item.raw or "",
            block_index=item.block_index,
            word_start_index=item.word_start_index,
            word_end_index=item.word_end_index,
            source_start=item.source_start,
            source_end=item.source_end,
            quote=item.quote,
            claim_id=item.claim_id,
            error=item.invalid_reason,
        )
    parsed = parse_reference_list([item])[0]
    if isinstance(item, Mapping) and item.get("invalid_reason"):
        return ParsedReference(
            kind=INVALID,
            raw=str(item.get("raw") or repr(dict(item))),
            error=str(item.get("invalid_reason")),
        )
    return parsed


def _from_parsed(parsed: ParsedReference) -> GroundingRef:
    if parsed.kind is INVALID:
        return GroundingRef(
            kind="INVALID",
            raw=parsed.raw,
            invalid_reason=parsed.error or "invalid_reference",
        )
    return GroundingRef(
        kind=parsed.kind,
        block_index=parsed.block_index,
        source_start=parsed.source_start,
        source_end=parsed.source_end,
        claim_id=parsed.claim_id,
        word_start_index=parsed.word_start_index,
        word_end_index=parsed.word_end_index,
        quote=parsed.quote,
        raw=parsed.raw,
    )


def resolve_grounding(
    inputs: MaterializationInputs, refs: Sequence[GroundingRef]
) -> GroundingResolution:
    """Resolve the exact source text each declaration cites."""

    chunks: list[str] = []
    errors: list[dict[str, object]] = []
    source_blocks = _source_blocks(inputs)
    bound_spans = [_bound_span(block) for block in source_blocks]
    approved_spans = [span for span in bound_spans if span is not None]

    for ref in refs:
        text, reason = _resolve_one(inputs, source_blocks, approved_spans, ref)
        if reason is not None:
            errors.append(
                {
                    "reference": ref.raw or _canonical(ref),
                    "kind": ref.kind,
                    "reason": reason,
                }
            )
            continue
        if text:
            chunks.append(text)

    unresolved = tuple(f"{item['reference']}: {item['reason']}" for item in errors)
    return GroundingResolution(
        text=" ".join(chunks).strip(),
        declared=bool(refs),
        unresolved=unresolved,
        errors=tuple(errors),
    )


def grounding_source_text(inputs: MaterializationInputs, refs: Sequence[GroundingRef]) -> str:
    return resolve_grounding(inputs, refs).text


def _canonical(ref: GroundingRef) -> str:
    if ref.kind == SOURCE_EXCERPT:
        return f"block:{ref.block_index}"
    if ref.kind == WORD_RANGE:
        return f"words:{ref.word_start_index}-{ref.word_end_index}"
    if ref.kind == SOURCE_SPAN:
        return f"source:{ref.source_start}-{ref.source_end}"
    if ref.kind == FINAL_CLIP:
        return "final_clip"
    return ref.quote or ref.kind


def _resolve_one(
    inputs: MaterializationInputs,
    source_blocks: Sequence[Mapping[str, object]],
    approved_spans: Sequence[tuple[float, float]],
    ref: GroundingRef,
) -> tuple[str | None, str | None]:
    kind = (ref.kind or "").upper()
    if kind in {INVALID, "INVALID_REFERENCE"} or ref.invalid_reason:
        return None, ref.invalid_reason or "invalid_reference"
    if kind == FINAL_CLIP:
        text = inputs.final_clip.text.strip()
        return (text, None) if text else (None, "empty_final_clip")
    if kind in {"SOURCE_EXCERPT", "SOURCE_BLOCK", "BLOCK"}:
        return _resolve_block(inputs, source_blocks, ref)
    if kind == WORD_RANGE:
        return _resolve_word_range(inputs, source_blocks, ref)
    if kind == SOURCE_SPAN:
        return _resolve_source_span(inputs, approved_spans, ref)
    if kind == QUOTE:
        return _resolve_quote(source_blocks, ref)
    return None, "unsupported_reference_kind"


def _source_blocks(inputs: MaterializationInputs) -> list[Mapping[str, object]]:
    return [
        block
        for block in inputs.blocks
        if isinstance(block, Mapping) and str(block.get("block_type", "")) == "SOURCE_EXCERPT"
    ]


def _planning_range(block: Mapping[str, object]) -> tuple[int, int] | None:
    start = block.get("planning_word_start_index", block.get("word_start_index"))
    end = block.get("planning_word_end_index", block.get("word_end_index"))
    if isinstance(start, bool) or not isinstance(start, int):
        return None
    if isinstance(end, bool) or not isinstance(end, int):
        return None
    if start < 0 or end < start:
        return None
    return start, end


def _bound_span(block: Mapping[str, object]) -> tuple[float, float] | None:
    start = block.get("source_start", block.get("planning_source_start"))
    end = block.get("source_end", block.get("planning_source_end"))
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return None
    if isinstance(end, bool) or not isinstance(end, (int, float)):
        return None
    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        return None
    if float(end) < float(start):
        return None
    return float(start), float(end)


def _bound_text(block: Mapping[str, object]) -> str:
    for key in ("source_text", "planning_source_text"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _planning_words(block: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    value = block.get("planning_words")
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _resolve_block(
    inputs: MaterializationInputs,
    source_blocks: Sequence[Mapping[str, object]],
    ref: GroundingRef,
) -> tuple[str | None, str | None]:
    if ref.block_index is None:
        return None, "missing_block_index"
    index = ref.block_index
    block = next(
        (item for item in inputs.blocks if item.get("index") == index),
        None,
    )
    if block is None:
        return None, f"missing_block:{index}"
    if str(block.get("block_type", "")) != "SOURCE_EXCERPT":
        return None, f"non_source_block:{index}"
    text = _bound_text(block)
    if not text:
        return None, f"empty_source_block:{index}"
    span = _bound_span(block)
    if span is not None:
        clip_start = inputs.final_clip.start
        clip_end = inputs.final_clip.end
        if (
            span[0] < clip_start - _RANGE_TOLERANCE_SECONDS
            or span[1] > clip_end + _RANGE_TOLERANCE_SECONDS
        ):
            return None, f"out_of_scope_block:{index}"
    return text, None


def _resolve_word_range(
    inputs: MaterializationInputs,
    source_blocks: Sequence[Mapping[str, object]],
    ref: GroundingRef,
) -> tuple[str | None, str | None]:
    start = ref.word_start_index
    end = ref.word_end_index
    if start is None or end is None:
        return None, "missing_word_range"
    if start < 0 or end < start:
        return None, "invalid_word_range"
    candidates: list[tuple[int, int, Mapping[str, object]]] = []
    for block in source_blocks:
        span = _planning_range(block)
        if span is None:
            continue
        if span[1] < start or span[0] > end:
            continue
        candidates.append((span[0], span[1], block))
    if not candidates:
        return None, "word_range_out_of_scope"
    candidates.sort(key=lambda item: item[0])

    cursor = start
    chunks: list[str] = []
    for planning_start, planning_end, block in candidates:
        if planning_end < cursor:
            continue
        if planning_start > cursor:
            return None, "partial_overlap"
        low = max(cursor, planning_start)
        high = min(end, planning_end)
        text, reason = _exact_word_text(inputs, block, low, high, planning_start, planning_end)
        if reason is not None:
            return None, reason
        if text:
            chunks.append(text)
        cursor = high + 1
        if cursor > end:
            break
    if cursor <= end:
        return None, "partial_overlap"
    if not chunks:
        return None, "empty_word_range"
    return " ".join(chunks), None


def _exact_word_text(
    inputs: MaterializationInputs,
    block: Mapping[str, object],
    low: int,
    high: int,
    planning_start: int,
    planning_end: int,
) -> tuple[str | None, str | None]:
    full_range = low == planning_start and high == planning_end
    if full_range:
        text = _bound_text(block)
        return (text, None) if text else (None, "empty_source_block")
    words = _planning_words(block)
    expected = planning_end - planning_start + 1
    if len(words) != expected:
        return None, "exact_word_evidence_unavailable"
    selected = words[low - planning_start : high - planning_start + 1]
    if not selected:
        return None, "empty_word_range"
    start = selected[0].get("start")
    end = selected[-1].get("end")
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return None, "planning_word_time_invalid"
    if isinstance(end, bool) or not isinstance(end, (int, float)):
        return None, "planning_word_time_invalid"
    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        return None, "planning_word_time_invalid"
    final_words = [
        word
        for word in inputs.final_clip.words
        if word.start >= float(start) - _RANGE_TOLERANCE_SECONDS
        and word.end <= float(end) + _RANGE_TOLERANCE_SECONDS
    ]
    if not final_words:
        return None, "final_clip_evidence_unavailable"
    if (
        final_words[0].start > float(start) + _RANGE_TOLERANCE_SECONDS
        or final_words[-1].end < float(end) - _RANGE_TOLERANCE_SECONDS
    ):
        return None, "final_clip_range_not_covered"
    return " ".join(word.text for word in final_words), None


def _resolve_source_span(
    inputs: MaterializationInputs,
    approved_spans: Sequence[tuple[float, float]],
    ref: GroundingRef,
) -> tuple[str | None, str | None]:
    start = ref.source_start
    end = ref.source_end
    if start is None or end is None:
        return None, "missing_source_span"
    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        return None, "non_finite_source_span"
    if float(start) < 0 or float(end) < float(start):
        return None, "invalid_source_span"
    start_f = float(start)
    end_f = float(end)
    if not _range_covered(approved_spans, start_f, end_f):
        return None, "source_span_out_of_scope"
    if start_f < inputs.final_clip.start - _RANGE_TOLERANCE_SECONDS:
        return None, "source_span_outside_final_clip"
    if end_f > inputs.final_clip.end + _RANGE_TOLERANCE_SECONDS:
        return None, "source_span_outside_final_clip"
    if start_f == end_f:
        words = [
            word
            for word in inputs.final_clip.words
            if word.start - _RANGE_TOLERANCE_SECONDS
            <= start_f
            <= word.end + _RANGE_TOLERANCE_SECONDS
        ]
        if not words:
            return None, "final_clip_evidence_unavailable"
        return " ".join(word.text for word in words), None
    words = [
        word
        for word in inputs.final_clip.words
        if word.start >= start_f - _RANGE_TOLERANCE_SECONDS
        and word.end <= end_f + _RANGE_TOLERANCE_SECONDS
    ]
    if not words:
        return None, "final_clip_evidence_unavailable"
    if (
        words[0].start > start_f + _RANGE_TOLERANCE_SECONDS
        or words[-1].end < end_f - _RANGE_TOLERANCE_SECONDS
    ):
        return None, "final_clip_range_not_covered"
    return " ".join(word.text for word in words), None


def _range_covered(spans: Sequence[tuple[float, float]], start: float, end: float) -> bool:
    intervals = sorted(spans)
    cursor = start
    for left, right in intervals:
        if right + _RANGE_TOLERANCE_SECONDS < cursor:
            continue
        if left > cursor + _RANGE_TOLERANCE_SECONDS:
            return False
        cursor = max(cursor, right)
        if cursor + _RANGE_TOLERANCE_SECONDS >= end:
            return True
    return cursor + _RANGE_TOLERANCE_SECONDS >= end


def _resolve_quote(
    source_blocks: Sequence[Mapping[str, object]], ref: GroundingRef
) -> tuple[str | None, str | None]:
    quote = (ref.quote or "").strip()
    if not quote:
        return None, "empty_quote"
    folded = quote.casefold()
    for block in source_blocks:
        text = _bound_text(block)
        if text and text.casefold() == folded:
            return text, None
    return None, "quote_not_in_source_excerpt"


__all__ = [
    "GroundingResolution",
    "grounding_source_text",
    "parse_grounding_refs",
    "resolve_grounding",
]
