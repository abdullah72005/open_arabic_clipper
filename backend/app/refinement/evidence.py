"""Bounded evidence accumulation, consensus, validation, and final-text choice.

All functions are pure and deterministic. They never call a provider, never
decode audio, and never fabricate confidence. ``choose_final_transcript`` makes
automated work strictly subordinate to manual and operator text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from app.core.enums import EvidenceKind, EvidenceState
from app.refinement.types import EvidenceRecord, WordTimestamp
from app.transcription.arabic import normalize_for_comparison
from app.transcription.dialect import extract_protected_tokens
from app.transcription.reconstruction.phonetics import phonetic_similarity

CONSENSUS_POLICY_VERSION = "stage3.5-consensus-v1"
_DISAGREEMENT_CONFIDENCE_FACTOR = 0.6
_MAX_AGREEMENT_CONFIDENCE = 0.99
_AUDIO_BACKED_KINDS = frozenset({EvidenceKind.TARGETED_LOCAL_ASR, EvidenceKind.HOSTED_ASR})
_COLLOQUIAL_PROFILES = frozenset({"EGYPTIAN", "SAUDI", "GULF", "LEVANTINE"})
_DIALECT_MARKERS: dict[str, frozenset[str]] = {
    "EGYPTIAN": frozenset({"عايز", "دلوقتي", "ايه", "ازيك", "اوي", "امبارح", "كده", "مش"}),
    "SAUDI": frozenset({"وش", "الحين", "ابغي", "رايك"}),
    "GULF": frozenset({"شلون", "شلونك", "وايد", "اشكثر", "زين"}),
    "LEVANTINE": frozenset({"شو", "هلا", "منيح", "نبلش", "بدي", "رايك"}),
}
_MSA_MARKERS = frozenset(
    {"سوف", "الان", "التي", "الذين", "ليس", "وليست", "وليس", "من اجل", "من الضروري"}
)
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def dedupe_evidence(records: Sequence[EvidenceRecord], *, limit: int) -> tuple[EvidenceRecord, ...]:
    """Deduplicate by fingerprint in first-seen order and bound to ``limit``."""

    if limit <= 0:
        return ()
    seen: set[str] = set()
    unique: list[EvidenceRecord] = []
    for record in records:
        if record.fingerprint in seen:
            continue
        seen.add(record.fingerprint)
        unique.append(record)
        if len(unique) >= limit:
            break
    return tuple(unique)


def audio_backed_evidence(
    records: Sequence[EvidenceRecord],
) -> tuple[EvidenceRecord, ...]:
    """Accepted records that came from an actual candidate-window ASR pass."""

    return tuple(
        record
        for record in records
        if record.kind in _AUDIO_BACKED_KINDS and record.state is EvidenceState.ACCEPTED
    )


def select_consensus_text(
    records: Sequence[EvidenceRecord],
) -> tuple[str, float, tuple[EvidenceRecord, ...]]:
    """Select agreement-first consensus text among audio-backed accepted records.

    Normalized exact matches form an agreement group. The largest group wins
    (ties by highest member confidence); its highest-confidence display form is
    returned with a slightly boosted confidence. When no two records agree the
    single highest-confidence record is returned with reduced confidence and no
    false agreement. With no audio-backed accepted record, returns the empty
    result.
    """

    supported = audio_backed_evidence(records)
    if not supported:
        return ("", 0.0, ())

    groups: dict[str, list[EvidenceRecord]] = {}
    order: list[str] = []
    for record in supported:
        key = normalize_for_comparison(record.transcript)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)

    best_key = max(
        order,
        key=lambda key: (
            len(groups[key]),
            max(record.confidence for record in groups[key]),
        ),
    )
    group = groups[best_key]
    if len(group) >= 2:
        representative = max(group, key=lambda record: (record.confidence, record.fingerprint))
        confidence = min(
            _MAX_AGREEMENT_CONFIDENCE,
            max(record.confidence for record in group) + 0.05 * (len(group) - 1),
        )
        return (representative.transcript, confidence, tuple(group))

    representative = max(supported, key=lambda record: (record.confidence, record.fingerprint))
    return (
        representative.transcript,
        round(representative.confidence * _DISAGREEMENT_CONFIDENCE_FACTOR, 6),
        (representative,),
    )


def repeated_text_ratio(text: str) -> float:
    """Fraction of normalized tokens that are duplicates (hallucination signal)."""

    words = normalize_for_comparison(text).split()
    if not words:
        return 0.0
    return max(0.0, 1.0 - len(set(words)) / len(words))


def looks_translated_or_normalized(text: str, *, source_dialect: str | None) -> bool:
    """Whether text dropped the source dialect entirely or turned formal MSA.

    A colloquial source rendered without Arabic script is treated as a
    translation. A colloquial source rewritten with strong MSA markers and no
    surviving source-dialect marker is treated as unexpected MSA conversion.
    ``source_dialect=None`` (no Arabic evidence) never triggers this check.
    """

    if not source_dialect or not text.strip():
        return False
    profile = str(source_dialect).upper()
    if _ARABIC_SCRIPT.search(text) is None:
        return True
    if profile not in _COLLOQUIAL_PROFILES:
        return False
    words = set(normalize_for_comparison(text).split())
    source_markers = _DIALECT_MARKERS.get(profile, frozenset())
    return bool(words & _MSA_MARKERS) and not bool(words & source_markers)


def _latin_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in extract_protected_tokens(text) if _LATIN_LETTER.search(token))


_NUMERIC_LIKE = re.compile(r"^[0-9٠-٩]+(?:[.,:/-][0-9٠-٩]+)*$")


def _is_numeric_like(token: str) -> bool:
    return _NUMERIC_LIKE.fullmatch(token) is not None


def _edit_ratio(left: str, right: str) -> float:
    longest = max(len(left), len(right), 1)
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / longest


def _unexplained_inserted_content(reference: str, candidate: str, max_edit_ratio: float) -> bool:
    reference_words = normalize_for_comparison(reference).split()
    candidate_words = normalize_for_comparison(candidate).split()
    if not candidate_words:
        return False
    remaining = Counter(reference_words)
    inserted = 0
    for word in candidate_words:
        if remaining[word] > 0:
            remaining[word] -= 1
        else:
            inserted += 1
    return inserted / len(candidate_words) > max(0.5, max_edit_ratio)


def _timestamps_invalid(words: Sequence[WordTimestamp]) -> bool:
    previous_start: float | None = None
    for word in words:
        if not math.isfinite(word.start) or not math.isfinite(word.end):
            return True
        if word.start < 0.0 or word.end < word.start:
            return True
        if previous_start is not None and word.start < previous_start:
            return True
        previous_start = word.start
    return False


def validate_transcript_candidate(
    *,
    candidate_text: str,
    reference_text: str,
    word_timestamps: Sequence[WordTimestamp],
    protected_tokens: Sequence[str],
    source_dialect: str | None,
    candidate_dialect: str | None,
    max_edit_ratio: float,
    min_phonetic_similarity: float,
    repeats_max_ratio: float,
    allow_unsupported_latin: bool = False,
) -> tuple[bool, str | None]:
    """Deterministically accept or reject one transcript candidate.

    The reference for text-only logic is the immutable INDEX text, so this
    naturally prevents a text-only provider from inventing Latin/code-switch
    content. An audio-backed path is represented by passing the ASR record text
    as the reference, which makes its Latin tokens supported.
    """

    candidate = candidate_text or ""
    reference = reference_text or ""
    if not candidate.strip():
        return (False, "empty_text")

    candidate_latin = {token.casefold() for token in _latin_tokens(candidate)}
    reference_latin = {token.casefold() for token in _latin_tokens(reference)}
    if candidate_latin - reference_latin and not allow_unsupported_latin:
        return (False, "unsupported_omitted_english")

    expected_protected = tuple(protected_tokens) or extract_protected_tokens(reference)
    actual_protected = extract_protected_tokens(candidate)
    if allow_unsupported_latin:
        # Audio-backed ASR may legitimately restore omitted Latin speech or read
        # a different number/date that must be adjudicated. Non-numeric protected
        # tokens (URLs, technical forms, abbreviations) must never be dropped or
        # changed.
        actual_set = set(actual_protected)
        if any(
            token not in actual_set for token in expected_protected if not _is_numeric_like(token)
        ):
            return (False, "protected_tokens_changed")
    elif actual_protected != expected_protected:
        return (False, "protected_tokens_changed")

    if _edit_ratio(reference, candidate) > max_edit_ratio:
        return (False, "extreme_edit_ratio")

    source_profile = str(source_dialect).upper() if source_dialect else None
    candidate_profile = str(candidate_dialect).upper() if candidate_dialect else None
    if looks_translated_or_normalized(candidate, source_dialect=source_dialect):
        return (False, "unexpected_translation_or_msa_conversion")
    if source_profile in _COLLOQUIAL_PROFILES and candidate_profile == "MSA":
        return (False, "unexpected_translation_or_msa_conversion")

    if repeated_text_ratio(candidate) > repeats_max_ratio:
        return (False, "repeated_or_hallucinated_text")

    if _unexplained_inserted_content(reference, candidate, max_edit_ratio):
        return (False, "unexplained_inserted_content")

    if _timestamps_invalid(word_timestamps):
        return (False, "timestamp_out_of_window")

    if (
        not allow_unsupported_latin
        and phonetic_similarity(reference, candidate) < min_phonetic_similarity
    ):
        return (False, "low_phonetic_similarity")

    return (True, None)


def choose_final_transcript(
    *,
    manual: str | None,
    adjudicated: str | None,
    operator_segment_text: str | None,
    consensus: str | None,
    automatic: str | None,
    stage27_text: str | None,
    stage25_text: str | None,
    raw_text: str | None,
) -> str:
    """Apply the immutable final-text priority ladder.

    Priority: candidate-refinement manual > source-segment operator > validated
    adjudication > ASR agreement > best audio-backed ASR > Stage 2.7 accepted >
    Stage 2.5 corrected > raw INDEX. Manual text is returned verbatim and is
    never replaced or removed by automated work.
    """

    ladder = (
        manual,
        operator_segment_text,
        adjudicated,
        consensus,
        automatic,
        stage27_text,
        stage25_text,
        raw_text,
    )
    for value in ladder:
        if value is not None and value.strip():
            return value
    return ""
