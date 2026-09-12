"""Deterministic hook analysis and strict provider-hook validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from app.candidates.cues import (
    CONTENT_CUES,
    CONTRAST_CUES,
    HOOK_DIRECTION_CUES,
    PAYOFF_CUES,
    SEARCH_LED_CUES,
)
from app.candidates.policy import DEFAULT_CONFIG, Stage3Config
from app.candidates.text import contains_any_cue, contains_cue, is_question, matching_text
from app.candidates.types import HookRecord, Proposal
from app.core.enums import ContentType, HookOrigin, HookType
from app.transcription.dialect import extract_protected_tokens

_SENTENCE_SPLIT = re.compile(r"(?<=[.!؟?…۔])\s+")


def generate_hooks(
    proposal: Proposal,
    segments: Sequence[Mapping[str, object]],
    *,
    config: Stage3Config = DEFAULT_CONFIG,
) -> list[HookRecord]:
    """Deterministic, source-faithful hooks; never invents marketing claims."""

    text = proposal.text
    matching = matching_text(text)
    preferred: list[HookType] = []
    if is_question(text) or "?" in matching or "؟" in matching:
        preferred.append(HookType.QUESTION)
    if contains_any_cue(matching, CONTRAST_CUES):
        preferred.append(HookType.CONTRADICTION)
    if contains_any_cue(matching, CONTENT_CUES[ContentType.EMOTIONAL]):
        preferred.append(HookType.EMOTIONAL)
    if contains_any_cue(matching, PAYOFF_CUES):
        preferred.append(HookType.PAYOFF_FIRST)
    if contains_any_cue(matching, SEARCH_LED_CUES):
        preferred.append(HookType.SEARCH_LED)
    if contains_any_cue(matching, HOOK_DIRECTION_CUES[HookType.CURIOSITY]):
        preferred.append(HookType.CURIOSITY)
    if contains_any_cue(matching, HOOK_DIRECTION_CUES[HookType.DIRECT_CLAIM]):
        preferred.append(HookType.DIRECT_CLAIM)
    preferred.append(HookType.CONTEXTUAL)

    hooks: list[HookRecord] = []
    seen: set[str] = set()
    for hook_type in preferred:
        if len(hooks) >= config.max_hooks_per_candidate:
            break
        record = _build_hook(hook_type, proposal, segments, config)
        if record is None:
            continue
        marker = (record.text or "").casefold()
        if marker and marker in seen:
            continue
        seen.add(marker)
        hooks.append(record)
    return hooks


def _build_hook(
    hook_type: HookType,
    proposal: Proposal,
    segments: Sequence[Mapping[str, object]],
    config: Stage3Config,
) -> HookRecord | None:
    evidence_text, indexes = _hook_evidence(hook_type, proposal)
    if not evidence_text:
        return None
    bounded = evidence_text.strip()[: config.max_hook_characters]
    if not bounded:
        return None
    source_faithful = hook_type is not HookType.CONTEXTUAL
    faithfulness = 0.95 if source_faithful else 0.7
    naturalness = 0.9 if 8 <= len(bounded) <= 160 else 0.6
    missing_context = (
        0.25
        if hook_type
        in {
            HookType.CONTEXTUAL,
            HookType.CURIOSITY,
            HookType.SEARCH_LED,
        }
        else 0.1
    )
    return HookRecord(
        type=hook_type,
        text=bounded,
        source_segment_indexes=indexes,
        source_evidence=(bounded,),
        strength=_hook_strength(hook_type, bounded),
        faithfulness=faithfulness,
        naturalness=naturalness,
        audience_suitability=0.7,
        policy_risk=0.1,
        missing_context_dependency=missing_context,
        origin=HookOrigin.DETERMINISTIC,
    )


def _hook_evidence(hook_type: HookType, proposal: Proposal) -> tuple[str, tuple[int, ...]]:
    sentences = [
        sentence.strip() for sentence in _SENTENCE_SPLIT.split(proposal.text) if sentence.strip()
    ]
    if not sentences:
        return proposal.text.strip(), proposal.segment_indexes
    if hook_type is HookType.QUESTION:
        for sentence in sentences:
            if is_question(sentence):
                return sentence, proposal.segment_indexes
    if hook_type is HookType.CONTRADICTION:
        for sentence in sentences:
            if contains_any_cue(matching_text(sentence), CONTRAST_CUES):
                return sentence, proposal.segment_indexes
    if hook_type is HookType.PAYOFF_FIRST:
        for sentence in sentences:
            if contains_any_cue(matching_text(sentence), PAYOFF_CUES):
                return sentence, proposal.segment_indexes
    if hook_type is HookType.SEARCH_LED:
        for sentence in sentences:
            if contains_any_cue(matching_text(sentence), SEARCH_LED_CUES):
                return sentence, proposal.segment_indexes
    if hook_type is HookType.EMOTIONAL:
        for sentence in sentences:
            if contains_any_cue(matching_text(sentence), CONTENT_CUES[ContentType.EMOTIONAL]):
                return sentence, proposal.segment_indexes
    for cue in HOOK_DIRECTION_CUES.get(hook_type, ()):  # CURIOSITY/DIRECT_CLAIM
        for sentence in sentences:
            if contains_cue(matching_text(sentence), cue):
                return sentence, proposal.segment_indexes
    return sentences[0], proposal.segment_indexes


def _hook_strength(hook_type: HookType, text: str) -> float:
    strength = 0.6
    if hook_type in {HookType.QUESTION, HookType.CONTRADICTION, HookType.PAYOFF_FIRST}:
        strength += 0.2
    if len(text) <= 140:
        strength += 0.1
    return max(0.0, min(1.0, strength))


def validate_provider_hooks(
    raw_hooks: Sequence[object],
    proposal: Proposal,
    *,
    context_text: str = "",
    config: Stage3Config = DEFAULT_CONFIG,
) -> list[HookRecord]:
    """Strictly validate provider hooks against bounded source evidence.

    Rejects unknown types, invented/changed protected tokens or numbers,
    evidence that is not present in the bounded candidate/context, excessive
    length, duplicates, and low-faithfulness output.
    """

    source = proposal.text
    context = f"{context_text} {source}"
    source_protected = {token.casefold() for token in extract_protected_tokens(source)}
    accepted: list[HookRecord] = []
    seen: set[str] = set()
    numeric = re.compile(r"\d+(?:[.,:/-]\d+)*")
    for raw in raw_hooks:
        if not isinstance(raw, Mapping):
            continue
        hook_type = _hook_type(raw.get("type"))
        if hook_type is None:
            continue
        text = raw.get("text")
        if text is not None and not isinstance(text, str):
            continue
        bounded = (text or "").strip()[: config.max_hook_characters]
        if text is not None and len(text.strip()) > config.max_hook_characters:
            continue
        evidence = _evidence_list(raw.get("source_evidence"))
        if evidence and not all(item in context for item in evidence):
            continue
        if bounded:
            if not _protected_supported(bounded, source_protected):
                continue
            if not _numbers_supported(bounded, source, numeric):
                continue
            marker = bounded.casefold()
            if marker in seen:
                continue
            seen.add(marker)
        faithfulness = _bounded_float(raw.get("faithfulness"), default=0.6)
        if text is not None and faithfulness < 0.3:
            continue
        accepted.append(
            HookRecord(
                type=hook_type,
                text=bounded or None,
                source_segment_indexes=_segment_indexes(raw.get("source_segment_indexes")),
                source_evidence=tuple(evidence) or ((bounded,) if bounded else ()),
                strength=_bounded_float(raw.get("strength"), default=0.5),
                faithfulness=faithfulness,
                naturalness=_bounded_float(raw.get("naturalness"), default=0.5),
                audience_suitability=_bounded_float(raw.get("audience_suitability"), default=0.5),
                policy_risk=_bounded_float(raw.get("policy_risk"), default=0.1),
                missing_context_dependency=_bounded_float(
                    raw.get("missing_context_dependency"), default=0.2
                ),
                origin=HookOrigin.PROVIDER,
            )
        )
        if len(accepted) >= config.max_hooks_per_candidate:
            break
    return accepted


def _protected_supported(text: str, source_protected: set[str]) -> bool:
    """Every protected token in the hook must already exist in source evidence."""

    for token in extract_protected_tokens(text):
        if token.casefold() not in source_protected:
            return False
    return True


def _numbers_supported(text: str, source: str, numeric: re.Pattern[str]) -> bool:
    source_numbers = {match.casefold() for match in numeric.findall(source)}
    return all(match.casefold() in source_numbers for match in numeric.findall(text))


def _hook_type(value: object) -> HookType | None:
    if isinstance(value, HookType):
        return value
    if isinstance(value, str):
        try:
            return HookType(value)
        except ValueError:
            return None
    return None


def _evidence_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _segment_indexes(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        int(item) for item in value if isinstance(item, int) and not isinstance(item, bool)
    )


def _bounded_float(value: object, *, default: float) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if number == number and number not in (float("inf"), float("-inf")):
            return max(0.0, min(1.0, number))
    return default
