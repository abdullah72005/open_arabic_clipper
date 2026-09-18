"""Pure deterministic FINAL_CLIP compatibility evaluation.

Uses only persisted evidence. It never trusts provider timestamps (it uses
indexed FINAL_CLIP word timings), never calls a provider, and never rewrites a
plan. The overall outcome is the most blocking per-block result under a fixed
precedence, with fail-closed treatment of unresolved or ambiguous evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from app.candidates.text import is_sentence_end
from app.core.enums import FinalClipCompatibilityOutcome
from app.refinement.entities import extract_entities
from app.render.binding import (
    align_excerpt,
    analysis_tokens,
    content_tokens,
    edit_ratio,
    has_digit,
    word_text,
)
from app.render.policy import (
    ALIGNMENT_AMBIGUOUS,
    ALIGNMENT_TIME_WINDOW_SECONDS,
    BOUNDARY_ADJUSTED,
    COMPATIBILITY_POLICY_VERSION,
    COMPATIBLE_EDIT_RATIO,
    COMPATIBLE_NON_MATERIAL_CHANGE,
    ENTITY_CHANGED,
    EXACT_MATCH,
    EXCERPT_CLIPPED_BY_WINDOW,
    EXCERPT_CUTS_THOUGHT,
    EXCERPT_OUT_OF_BOUNDS,
    FINAL_CLIP_MEANING_CRITICAL_UNRESOLVED,
    GROUNDING_QUOTE_LOST,
    MATERIAL_DRIFT_SECONDS,
    MATERIAL_SEMANTIC_CHANGE,
    MATERIAL_TIMING_CHANGE,
    MINOR_WORDING_CHANGE,
    NON_MATERIAL_DRIFT_SECONDS,
    NUMBER_OR_DATE_CHANGED,
    PAYOFF_COVERAGE_TOLERANCE_SECONDS,
    PAYOFF_NOT_COVERED,
    PLANNING_BOUNDS_TOLERANCE_SECONDS,
    PROTECTED_SEMANTIC_OPERATORS,
    RECOVERED_CODE_SWITCH_TOKEN,
    SEMANTIC_OPERATOR_CHANGED,
    SOURCE_SPAN_NO_LONGER_VALID,
    TIMING_DRIFT_BOUNDED,
    UNRESOLVED_COMPATIBILITY,
    UNRESOLVED_EDIT_RATIO,
    WORD_EVIDENCE_INSUFFICIENT,
    most_blocking_outcome,
)
from app.render.types import BlockCompatibility, ClipWord, CompatibilityResult, FinalClipEvidence

_SUBSTANTIVE_TYPES = {"ORIGINAL_VALUE", "TEXTUAL_ANNOTATION"}
_GROUNDING_ENTITY_TYPES = {"PERSON", "PRODUCT", "PLACE", "ABBREVIATION", "TECHNICAL"}
_LATIN = re.compile(r"[A-Za-z]")
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def _block_index(block: Mapping[str, object]) -> int:
    value = block.get("index")
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _as_sequence(value: object) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def evaluate_compatibility(
    *,
    blocks: Sequence[Mapping[str, object]],
    hero_block_index: int | None,
    hook_payoff_evidence: Mapping[str, object],
    final: FinalClipEvidence,
    source_segments: Sequence[Mapping[str, object]],
    planning_refined_start: float | None,
    planning_refined_end: float | None,
    exact_identity_match: bool,
    caption_source_fingerprint: str,
    planning_refinement_id: str,
    planning_refinement_priority: str,
    planning_refinement_quality_level: str,
    planning_output_fingerprint: str,
) -> CompatibilityResult:
    """Evaluate the current FINAL_CLIP against the selected plan's excerpts."""

    base: dict[str, Any] = dict(
        planning_refinement_id=planning_refinement_id,
        planning_refinement_priority=planning_refinement_priority,
        planning_refinement_quality_level=planning_refinement_quality_level,
        planning_output_fingerprint=planning_output_fingerprint,
        final_refinement_id=final.refinement_id,
        final_output_fingerprint=final.output_fingerprint,
        caption_source_fingerprint=caption_source_fingerprint,
        compatibility_policy_version=COMPATIBILITY_POLICY_VERSION,
    )

    if not final.word_coverage_sufficient or not final.words:
        return CompatibilityResult(
            outcome=UNRESOLVED_COMPATIBILITY,
            exact_match=False,
            reason_codes=(WORD_EVIDENCE_INSUFFICIENT,),
            **base,
        )

    verdicts: list[BlockCompatibility] = []
    verdict_by_index: dict[int, BlockCompatibility] = {}
    for block in blocks:
        if str(block.get("block_type")) != "SOURCE_EXCERPT":
            continue
        verdict = _evaluate_block(
            block,
            final=final,
            planning_refined_start=planning_refined_start,
            planning_refined_end=planning_refined_end,
            exact_identity_match=exact_identity_match,
        )
        verdicts.append(verdict)
        verdict_by_index[_block_index(block)] = verdict

    overall_reasons: list[str] = []
    per_block_outcomes = [verdict.outcome for verdict in verdicts]
    outcome = most_blocking_outcome(per_block_outcomes) or EXACT_MATCH

    rebound_spans = [
        (verdict.rebound_start, verdict.rebound_end)
        for verdict in verdicts
        if verdict.rebound_start is not None and verdict.rebound_end is not None
    ]

    if _payoff_not_covered(
        hook_payoff_evidence,
        source_segments=source_segments,
        spans=rebound_spans,
    ):
        outcome = _escalate(outcome, MATERIAL_TIMING_CHANGE)
        overall_reasons.append(PAYOFF_NOT_COVERED)

    if _meaning_critical_unresolved(final, spans=rebound_spans):
        outcome = _escalate(outcome, UNRESOLVED_COMPATIBILITY)
        overall_reasons.append(FINAL_CLIP_MEANING_CRITICAL_UNRESOLVED)

    if _grounding_quote_lost(
        blocks,
        final_words=final.words,
        rebound_spans=rebound_spans,
    ):
        outcome = _escalate(outcome, MATERIAL_SEMANTIC_CHANGE)
        overall_reasons.append(GROUNDING_QUOTE_LOST)

    exact_match = exact_identity_match and outcome == EXACT_MATCH

    recovered: list[str] = []
    for verdict in verdicts:
        recovered.extend(verdict.recovered_code_switch_tokens)

    return CompatibilityResult(
        outcome=outcome,
        exact_match=exact_match,
        per_block=tuple(verdicts),
        reason_codes=tuple(dict.fromkeys(overall_reasons)),
        unresolved_spans=final.unresolved_spans,
        recovered_code_switch_tokens=tuple(dict.fromkeys(recovered)),
        **base,
    )


def _evaluate_block(
    block: Mapping[str, object],
    *,
    final: FinalClipEvidence,
    planning_refined_start: float | None,
    planning_refined_end: float | None,
    exact_identity_match: bool,
) -> BlockCompatibility:
    index = _block_index(block)
    alignment = align_excerpt(
        block,
        final.words,
        planning_refined_start=planning_refined_start,
        planning_refined_end=planning_refined_end,
    )
    if not alignment.structural_valid:
        return BlockCompatibility(
            block_index=index,
            outcome=SOURCE_SPAN_NO_LONGER_VALID,
            reason_codes=(EXCERPT_OUT_OF_BOUNDS,),
            structural_valid=False,
            alignment=dict(alignment.stats),
        )
    if alignment.full_window:
        rebound_start = final.refined_start
        rebound_end = final.refined_end
        word_start = None
        word_end = None
        rebound_text = final.final_transcript
        drift = _drift(block, rebound_start, rebound_end)
    else:
        if not alignment.matched:
            if float(alignment.coverage) <= 0.0:
                return BlockCompatibility(
                    block_index=index,
                    outcome=SOURCE_SPAN_NO_LONGER_VALID,
                    reason_codes=(),
                    alignment=dict(alignment.stats),
                )
            return BlockCompatibility(
                block_index=index,
                outcome=UNRESOLVED_COMPATIBILITY,
                reason_codes=(ALIGNMENT_AMBIGUOUS,),
                coverage=alignment.coverage,
                alignment=dict(alignment.stats),
            )
        if alignment.ambiguous:
            return BlockCompatibility(
                block_index=index,
                outcome=UNRESOLVED_COMPATIBILITY,
                reason_codes=(ALIGNMENT_AMBIGUOUS,),
                coverage=alignment.coverage,
                alignment=dict(alignment.stats),
            )
        rebound_start = alignment.rebound_start
        rebound_end = alignment.rebound_end
        word_start = alignment.word_start
        word_end = alignment.word_end
        rebound_text = word_text(final.words, word_start, word_end)
        drift = alignment.drift

    plan_text = str(block.get("source_text") or "")

    if exact_identity_match:
        return BlockCompatibility(
            block_index=index,
            outcome=EXACT_MATCH,
            rebound_start=rebound_start,
            rebound_end=rebound_end,
            rebound_word_start=word_start,
            rebound_word_end=word_end,
            rebound_text=rebound_text,
            timing_drift_seconds=drift,
            alignment=dict(alignment.stats),
        )

    plan_tokens = content_tokens(plan_text)
    final_tokens = content_tokens(rebound_text)
    added, removed = _token_delta(plan_tokens, final_tokens)

    verdict = _semantic_verdict(
        block_index=index,
        plan_text=plan_text,
        rebound_text=rebound_text,
        plan_tokens=plan_tokens,
        final_tokens=final_tokens,
        added=added,
        removed=removed,
        final=final,
    )
    if verdict is not None:
        return replace(
            verdict,
            rebound_start=rebound_start,
            rebound_end=rebound_end,
            rebound_word_start=word_start,
            rebound_word_end=word_end,
            rebound_text=rebound_text,
            timing_drift_seconds=drift,
            alignment=dict(alignment.stats),
        )

    timing = _boundary_and_timing(
        block,
        final=final,
        alignment_end=rebound_end,
        plan_end=block.get("source_end"),
        planning_refined_end=planning_refined_end,
        drift=drift,
    )
    return replace(
        timing,
        block_index=index,
        rebound_start=rebound_start,
        rebound_end=rebound_end,
        rebound_word_start=word_start,
        rebound_word_end=word_end,
        rebound_text=rebound_text,
        added_tokens=tuple(added),
        removed_tokens=tuple(removed),
        alignment=dict(alignment.stats),
    )


def _drift(
    block: Mapping[str, object], rebound_start: float | None, rebound_end: float | None
) -> float | None:
    plan_start = block.get("source_start")
    plan_end = block.get("source_end")
    if not isinstance(plan_start, (int, float)) or not isinstance(plan_end, (int, float)):
        return None
    if rebound_start is None or rebound_end is None:
        return None
    return max(abs(rebound_start - float(plan_start)), abs(rebound_end - float(plan_end)))


def _semantic_verdict(
    *,
    block_index: int,
    plan_text: str,
    rebound_text: str,
    plan_tokens: Sequence[str],
    final_tokens: Sequence[str],
    added: Sequence[str],
    removed: Sequence[str],
    final: FinalClipEvidence,
) -> BlockCompatibility | None:
    changed = [token for token in [*added, *removed] if token]

    # No wording/operator/number/entity change at all: wording comparison is a
    # no-op, so the deterministic boundary/timing classification owns the
    # verdict (bounded drift, boundary adjustment, cut/clipped thought, or
    # material timing change). Returning a MINOR_WORDING_CHANGE here would make
    # every timing outcome permanently unreachable.
    if not changed:
        return None

    if any(token in PROTECTED_SEMANTIC_OPERATORS for token in changed):
        return BlockCompatibility(
            block_index=block_index,
            outcome=MATERIAL_SEMANTIC_CHANGE,
            reason_codes=(SEMANTIC_OPERATOR_CHANGED,),
            added_tokens=tuple(added),
            removed_tokens=tuple(removed),
        )
    if any(has_digit(token) for token in changed) or _numeric_entity_changed(
        plan_text, rebound_text
    ):
        return BlockCompatibility(
            block_index=block_index,
            outcome=MATERIAL_SEMANTIC_CHANGE,
            reason_codes=(NUMBER_OR_DATE_CHANGED,),
            added_tokens=tuple(added),
            removed_tokens=tuple(removed),
        )
    if _entity_changed(plan_text, rebound_text):
        return BlockCompatibility(
            block_index=block_index,
            outcome=MATERIAL_SEMANTIC_CHANGE,
            reason_codes=(ENTITY_CHANGED,),
            added_tokens=tuple(added),
            removed_tokens=tuple(removed),
        )

    recovered = _recovered_code_switch(added, final=final, plan_text=plan_text)
    if recovered:
        return BlockCompatibility(
            block_index=block_index,
            outcome=COMPATIBLE_NON_MATERIAL_CHANGE,
            reason_codes=(RECOVERED_CODE_SWITCH_TOKEN,),
            added_tokens=tuple(added),
            removed_tokens=tuple(removed),
            recovered_code_switch_tokens=tuple(recovered),
        )

    ratio = edit_ratio(plan_tokens, final_tokens)
    if ratio <= COMPATIBLE_EDIT_RATIO:
        return BlockCompatibility(
            block_index=block_index,
            outcome=COMPATIBLE_NON_MATERIAL_CHANGE,
            reason_codes=(MINOR_WORDING_CHANGE,),
            added_tokens=tuple(added),
            removed_tokens=tuple(removed),
        )
    if ratio <= UNRESOLVED_EDIT_RATIO:
        return BlockCompatibility(
            block_index=block_index,
            outcome=UNRESOLVED_COMPATIBILITY,
            reason_codes=(),
            added_tokens=tuple(added),
            removed_tokens=tuple(removed),
        )
    return BlockCompatibility(
        block_index=block_index,
        outcome=MATERIAL_SEMANTIC_CHANGE,
        reason_codes=(),
        added_tokens=tuple(added),
        removed_tokens=tuple(removed),
    )


def _boundary_and_timing(
    block: Mapping[str, object],
    *,
    final: FinalClipEvidence,
    alignment_end: float | None,
    plan_end: object,
    planning_refined_end: float | None,
    drift: float | None,
) -> BlockCompatibility:
    block_index = _block_index(block)
    plan_text = str(block.get("source_text") or "")
    sentence_final = is_sentence_end(plan_text) or (
        planning_refined_end is not None
        and isinstance(plan_end, (int, float))
        and abs(float(plan_end) - planning_refined_end) <= PLANNING_BOUNDS_TOLERANCE_SECONDS
    )
    if (
        sentence_final
        and alignment_end is not None
        and final.refined_end is not None
        and alignment_end < final.refined_end - PLANNING_BOUNDS_TOLERANCE_SECONDS
        and _followed_by_unterminated_words(final, alignment_end)
    ):
        return BlockCompatibility(
            block_index=block_index,
            outcome=MATERIAL_TIMING_CHANGE,
            reason_codes=(EXCERPT_CUTS_THOUGHT,),
            timing_drift_seconds=drift,
        )
    if (
        final.refined_end is not None
        and alignment_end is not None
        and isinstance(plan_end, (int, float))
        and alignment_end >= final.refined_end - PLANNING_BOUNDS_TOLERANCE_SECONDS
        and float(plan_end) > final.refined_end + PLANNING_BOUNDS_TOLERANCE_SECONDS
    ):
        return BlockCompatibility(
            block_index=block_index,
            outcome=MATERIAL_TIMING_CHANGE,
            reason_codes=(EXCERPT_CLIPPED_BY_WINDOW,),
            timing_drift_seconds=drift,
        )

    if drift is not None and drift > MATERIAL_DRIFT_SECONDS:
        return BlockCompatibility(
            block_index=block_index,
            outcome=MATERIAL_TIMING_CHANGE,
            reason_codes=(),
            timing_drift_seconds=drift,
        )
    if drift is not None and drift > NON_MATERIAL_DRIFT_SECONDS:
        return BlockCompatibility(
            block_index=block_index,
            outcome=COMPATIBLE_NON_MATERIAL_CHANGE,
            reason_codes=(BOUNDARY_ADJUSTED,),
            timing_drift_seconds=drift,
        )
    return BlockCompatibility(
        block_index=block_index,
        outcome=COMPATIBLE_NON_MATERIAL_CHANGE,
        reason_codes=(TIMING_DRIFT_BOUNDED,),
        timing_drift_seconds=drift,
    )


def _followed_by_unterminated_words(final: FinalClipEvidence, after: float) -> bool:
    following = [word for word in final.words if word.start >= after - 0.01]
    if not following:
        return False
    buffer = ""
    for word in following[:8]:
        buffer = f"{buffer} {word.text}".strip()
        if is_sentence_end(word.text):
            return False
        if word.start - after > ALIGNMENT_TIME_WINDOW_SECONDS:
            break
    return bool(buffer)


def _token_delta(
    plan_tokens: Sequence[str], final_tokens: Sequence[str]
) -> tuple[list[str], list[str]]:
    import difflib

    matcher = difflib.SequenceMatcher(None, list(plan_tokens), list(final_tokens), autojunk=False)
    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            removed.extend(plan_tokens[i1:i2])
        if tag in {"replace", "insert"}:
            added.extend(final_tokens[j1:j2])
    return added, removed


def _numeric_entity_changed(plan_text: str, final_text: str) -> bool:
    numeric_types = {"NUMBER", "DATE", "TIME", "MONEY", "PERCENTAGE", "SCORE"}
    plan_numeric = {
        m.normalized for m in extract_entities(plan_text) if m.entity_type in numeric_types
    }
    final_numeric = {
        m.normalized for m in extract_entities(final_text) if m.entity_type in numeric_types
    }
    return plan_numeric != final_numeric


def _entity_changed(plan_text: str, final_text: str) -> bool:
    plan_entities = extract_entities(plan_text)
    final_entities = extract_entities(final_text)
    for mention in plan_entities:
        if mention.entity_type not in _GROUNDING_ENTITY_TYPES:
            continue
        present = any(
            other.entity_type == mention.entity_type and other.normalized == mention.normalized
            for other in final_entities
        )
        if not present:
            return True
    return False


def _recovered_code_switch(
    added: Sequence[str], *, final: FinalClipEvidence, plan_text: str
) -> list[str]:
    """Admit an added Latin token as recovered code-switch only with evidence.

    A token is recovered only when it is listed in FINAL_CLIP code-switch
    evidence (case-folded membership) or when the planning excerpt itself
    contains Arabic-script tokens (the genuine omitted-English-in-Arabic case).
    An arbitrary inserted English intensifier/hedge in an English excerpt is
    never silently recovered; it flows through the normal change path.
    """

    code_switch = {
        str(token).casefold()
        for token in _as_sequence(final.code_switch_evidence.get("tokens"))
        if isinstance(token, str)
    }
    plan_has_arabic = bool(_ARABIC_SCRIPT.search(plan_text))
    recovered: list[str] = []
    for token in added:
        if token in PROTECTED_SEMANTIC_OPERATORS or has_digit(token):
            continue
        if not _LATIN.search(token):
            continue
        if token in code_switch or plan_has_arabic:
            recovered.append(token)
    return recovered


def _escalate(current: str, candidate: str) -> str:
    winner = most_blocking_outcome([current, candidate])
    return winner if winner is not None else current


def _payoff_not_covered(
    hook_payoff_evidence: Mapping[str, object],
    *,
    source_segments: Sequence[Mapping[str, object]],
    spans: Sequence[tuple[float, float]],
) -> bool:
    for key in ("payoff_index", "hook_index"):
        index = hook_payoff_evidence.get(key)
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if index < 0 or index >= len(source_segments):
            # Does not resolve: NOT_EVALUABLE, never a failure.
            continue
        segment = source_segments[index]
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if not _spans_cover(spans, float(start), float(end)):
            return True
    return False


def _spans_cover(spans: Sequence[tuple[float, float]], start: float, end: float) -> bool:
    tolerance = PAYOFF_COVERAGE_TOLERANCE_SECONDS
    return _point_covered(spans, start, tolerance) and _point_covered(spans, end, tolerance)


def _point_covered(spans: Sequence[tuple[float, float]], point: float, tolerance: float) -> bool:
    return any(min(a, b) - tolerance <= point <= max(a, b) + tolerance for a, b in spans)


def _meaning_critical_unresolved(
    final: FinalClipEvidence, *, spans: Sequence[tuple[float, float]]
) -> bool:
    for span in final.unresolved_spans:
        if not bool(span.get("meaning_critical")):
            continue
        if str(span.get("resolution_state") or "UNRESOLVED") == "RESOLVED":
            continue
        start = span.get("start")
        end = span.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return True
        for a, b in spans:
            if (
                float(start) <= b + PAYOFF_COVERAGE_TOLERANCE_SECONDS
                and float(end) >= a - PAYOFF_COVERAGE_TOLERANCE_SECONDS
            ):
                return True
    return False


def _grounding_quote_lost(
    blocks: Sequence[Mapping[str, object]],
    *,
    final_words: Sequence[ClipWord],
    rebound_spans: Sequence[tuple[float, float]],
) -> bool:
    if not final_words or not rebound_spans:
        return False
    final_text = " ".join(
        word.text
        for word in final_words
        if any(a - 1e-6 <= word.start <= b + 1e-6 for a, b in rebound_spans)
    )
    final_normalized = set(analysis_tokens(final_text))
    for block in blocks:
        if str(block.get("block_type")) not in _SUBSTANTIVE_TYPES:
            continue
        plan_text = str(block.get("source_text") or "")
        for reference in _as_sequence(block.get("grounding_refs")):
            reference_text = str(reference)
            reference_tokens = content_tokens(reference_text)
            if not reference_tokens:
                continue
            # Only references that quoted the planning excerpt are checked.
            plan_tokens = set(analysis_tokens(plan_text))
            if not set(reference_tokens) <= plan_tokens:
                continue
            if not set(reference_tokens) <= final_normalized:
                return True
    return False


def outcome_is_compatible(outcome: str) -> bool:
    return outcome in {
        FinalClipCompatibilityOutcome.EXACT_MATCH.value,
        FinalClipCompatibilityOutcome.COMPATIBLE_NON_MATERIAL_CHANGE.value,
    }


__all__ = ["evaluate_compatibility", "outcome_is_compatible"]
