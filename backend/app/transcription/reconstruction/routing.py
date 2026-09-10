"""Deterministic routing of transcript spans to reconstruction passes."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .types import ReconstructionWindow, WordEvidence


class RoutingPriority(str, Enum):
    CONTEXT_CHECK = "context_check"
    RECONSTRUCT = "reconstruct"
    LEAVE = "leave"


class RoutingMode(str, Enum):
    """Provider selection policy for Stage 2.7 reconstruction."""

    LOCAL_ONLY = "local_only"
    ADAPTIVE = "adaptive"
    GEMINI_ONLY = "gemini_only"


class ReconstructionRoute(str, Enum):
    """Final provider path chosen for one reconstruction target."""

    NO_LLM = "NO_LLM"
    LOCAL = "LOCAL"
    GEMINI_DIRECT = "GEMINI_DIRECT"


# Bumped for the clean-unchanged trust policy and residual-evidence semantics:
# NO_LLM now rests on affirmative clean evidence (or a trusted Stage 2.5 repair
# that removed the relevant uncertainty) instead of on the correction method
# alone. Routing thresholds, evidence coverage, and clean-average thresholds all
# participate in runtime identity and fingerprints through ``as_dict``.
ADAPTIVE_ROUTING_VERSION = "adaptive-routing-v3"


@dataclass(frozen=True)
class RoutingEvidence:
    score: float
    low_probability_ratio: float
    focus_spans: tuple[WordEvidence, ...]
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    priority: RoutingPriority
    evidence: RoutingEvidence
    focus_spans: tuple[WordEvidence, ...]
    reason: str

    @property
    def reasons(self) -> tuple[str, ...]:
        return (self.reason,)


@dataclass(frozen=True)
class RoutingConfig:
    low_probability_threshold: float = 0.72
    very_low_probability_threshold: float = 0.50
    low_ratio_threshold: float = 0.78
    high_ratio_threshold: float = 0.25
    score_threshold: float = 0.45


@dataclass(frozen=True)
class AdaptiveRoutingConfig:
    """Centralized conservative thresholds for the adaptive provider router.

    Every constant here is chosen once and documented; the router never
    calibrates itself. Thresholds influence only which provider path a target
    takes and never relax the shared validation/acceptance gates.

    NO_LLM trusts either (a) a clean, well-covered unchanged Stage 2.5 result or
    (b) a high-confidence Stage 2.5 repair that resolved the relevant raw-ASR
    uncertainty, and only when no independent residual suspicion remains.
    Missing evidence is never affirmative trust: a segment without adequate word
    coverage stays eligible for a conservative local check.
    """

    mode: RoutingMode = RoutingMode.ADAPTIVE
    # Stage 2.5 confidence at or above which a repair is trusted to have resolved
    # the uncertainty it covered.
    stage25_trust_confidence: float = 0.90
    # NO_LLM for an unchanged result requires this fraction of words to carry a
    # usable probability. Missing word evidence is not affirmative trust.
    evidence_coverage_min: float = 0.80
    # NO_LLM for an unchanged result requires the average word probability to be
    # at or above this value on top of full coverage and zero low-probability
    # words.
    clean_average_probability: float = 0.85
    # GEMINI_DIRECT: at least this many consecutive very-low-probability words.
    gemini_direct_contiguous_very_low: int = 3
    # GEMINI_DIRECT: at least this fraction of words below the low threshold,
    # and at least ``gemini_direct_min_low_count`` low words.
    gemini_direct_low_ratio: float = 0.55
    gemini_direct_min_low_count: int = 4
    # GEMINI_DIRECT: existing routing score at or above this value.
    gemini_direct_score: float = 0.70
    # GEMINI_DIRECT: at least this many very-low words overlapping a protected
    # Latin/number token.
    gemini_direct_protected_overlap_min_very_low: int = 2
    # LOCAL_THEN_GEMINI: escalate a local candidate whose computed score and
    # phonetic similarity are both within this margin of the HIGH gate.
    escalation_near_threshold_margin: float = 0.05

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "stage25_trust_confidence": self.stage25_trust_confidence,
            "evidence_coverage_min": self.evidence_coverage_min,
            "clean_average_probability": self.clean_average_probability,
            "gemini_direct_contiguous_very_low": self.gemini_direct_contiguous_very_low,
            "gemini_direct_low_ratio": self.gemini_direct_low_ratio,
            "gemini_direct_min_low_count": self.gemini_direct_min_low_count,
            "gemini_direct_score": self.gemini_direct_score,
            "gemini_direct_protected_overlap_min_very_low": (
                self.gemini_direct_protected_overlap_min_very_low
            ),
            "escalation_near_threshold_margin": self.escalation_near_threshold_margin,
        }


@dataclass(frozen=True)
class AdaptiveRoutingDecision:
    route: ReconstructionRoute
    reason: str
    evidence: tuple[str, ...]
    priority: RoutingPriority
    score: float | None = None
    focus_spans: tuple[WordEvidence, ...] = ()
    severity: float = 0.0


def route_segment(
    segment: ReconstructionWindow | Mapping[str, object],
    config: RoutingConfig = RoutingConfig(),
    language: str | None = None,
) -> RoutingDecision:
    words, avg = _segment_evidence(segment)
    return _route_decision(words, avg, language, config)


def _route_decision(
    words: tuple[WordEvidence, ...],
    avg: float | None,
    language: str | None,
    config: RoutingConfig,
) -> RoutingDecision:
    probs = [w.probability for w in words if w.probability is not None]
    low = [
        w
        for w in words
        if w.probability is not None and w.probability < config.low_probability_threshold
    ]
    ratio = len(low) / len(probs) if probs else 0.0
    focus = tuple(
        w
        for w in low
        if w.probability is not None and w.probability < config.very_low_probability_threshold
    )
    score = (
        0.50 * (1 - (avg if avg is not None else config.low_probability_threshold))
        + 0.25 * ratio
        + 0.25 * bool(focus)
    )
    if (len(low) >= 2 and ratio >= config.high_ratio_threshold) or score >= config.score_threshold:
        priority, reason = RoutingPriority.RECONSTRUCT, "multiple_low_probability_words"
    elif language == "ar" and (avg is None or avg >= config.low_probability_threshold) and not low:
        priority, reason = RoutingPriority.CONTEXT_CHECK, "high_confidence_arabic_context_check"
    else:
        priority, reason = RoutingPriority.LEAVE, "insufficient_uncertainty_evidence"
    evidence = RoutingEvidence(score, ratio, focus or tuple(low), reason)
    return RoutingDecision(priority, evidence, evidence.focus_spans, reason)


@dataclass(frozen=True)
class _Stage25State:
    """Parsed Stage 2.5 outcome used for residual-evidence routing."""

    method_normal: bool
    trusted_repair: bool
    resolved_indexes: frozenset[int]


def route_adaptive(
    segment: ReconstructionWindow | Mapping[str, object],
    config: AdaptiveRoutingConfig | None = None,
    language: str | None = None,
    decision: RoutingDecision | None = None,
) -> AdaptiveRoutingDecision:
    """Choose the provider path for one target before any provider is invoked.

    Evidence is evaluated on the raw-ASR words that Stage 2.5 did *not* resolve:

    - A high-confidence accepted Stage 2.5 repair removes the low-probability
      words its ``changes`` cover from the residual evidence. A clean remainder
      routes to ``NO_LLM`` instead of reprocessing the repair.
    - An unchanged Stage 2.5 result routes to ``NO_LLM`` only when it is backed
      by affirmative clean evidence: adequate word coverage, a high average
      probability, no low-probability span, no protected-token ambiguity, no
      contiguous uncertain span, and no hard-corruption indicator.
    - Missing word evidence is never affirmative trust and stays eligible for a
      conservative local check, never Gemini-direct merely because it is missing.
    """

    config = config or AdaptiveRoutingConfig()
    words, avg = _segment_evidence(segment)
    state = _stage25_state(segment, config)
    residual = _residual_words(words, state)
    residual_avg = _average_probability(residual) if residual else avg
    thresholds = RoutingConfig()
    base = _route_decision(residual, residual_avg, language, thresholds)
    low = tuple(
        w
        for w in residual
        if w.probability is not None and w.probability < thresholds.low_probability_threshold
    )
    protected_ambiguity = _protected_overlap(low)
    hard, evidence = _hard_corruption_evidence(residual, base, config)
    severity = _severity(residual, base, protected_ambiguity)
    if hard:
        return AdaptiveRoutingDecision(
            ReconstructionRoute.GEMINI_DIRECT,
            "hard_corruption_evidence",
            evidence,
            base.priority,
            base.evidence.score,
            base.focus_spans,
            severity,
        )
    no_residual_uncertainty = (
        not low
        and not protected_ambiguity
        and base.priority in {RoutingPriority.LEAVE, RoutingPriority.CONTEXT_CHECK}
    )
    if no_residual_uncertainty and state.method_normal and not _unresolved_warning(segment):
        clean_evidence = state.trusted_repair or (
            bool(words)
            and _evidence_coverage(words) >= config.evidence_coverage_min
            and avg is not None
            and avg >= config.clean_average_probability
        )
        if clean_evidence:
            reasons = _no_llm_reasons(state, words, low, protected_ambiguity)
            return AdaptiveRoutingDecision(
                ReconstructionRoute.NO_LLM,
                "clean_no_llm",
                reasons,
                base.priority,
                base.evidence.score,
                base.focus_spans,
                severity,
            )
        return AdaptiveRoutingDecision(
            ReconstructionRoute.LOCAL,
            "missing_affirmative_trust_evidence",
            ("missing_trust_evidence",),
            base.priority,
            base.evidence.score,
            base.focus_spans,
            severity,
        )
    evidence_label = _local_evidence_label(state, words, low, protected_ambiguity)
    if base.priority is RoutingPriority.RECONSTRUCT:
        return AdaptiveRoutingDecision(
            ReconstructionRoute.LOCAL,
            "mild_localized_uncertainty",
            evidence_label,
            base.priority,
            base.evidence.score,
            base.focus_spans,
            severity,
        )
    return AdaptiveRoutingDecision(
        ReconstructionRoute.LOCAL,
        "stage25_untrusted_or_isolated_uncertainty",
        evidence_label,
        base.priority,
        base.evidence.score,
        base.focus_spans,
        severity,
    )


def _no_llm_reasons(
    state: _Stage25State,
    words: tuple[WordEvidence, ...],
    low: tuple[WordEvidence, ...],
    protected_ambiguity: bool,
) -> tuple[str, ...]:
    if state.trusted_repair and not low:
        reasons = ["stage25_trusted_repair_resolved"]
    else:
        reasons = ["clean_high_probability_evidence"]
    if not low:
        reasons.append("no_low_probability_words")
    if not protected_ambiguity:
        reasons.append("no_protected_token_ambiguity")
    if words:
        reasons.append(f"evidence_coverage={_evidence_coverage(words):.2f}")
    return tuple(reasons)


def _local_evidence_label(
    state: _Stage25State,
    words: tuple[WordEvidence, ...],
    low: tuple[WordEvidence, ...],
    protected_ambiguity: bool,
) -> tuple[str, ...]:
    """Stable evidence label for a LOCAL decision."""

    if low:
        if len(low) == 1 and not protected_ambiguity:
            return ("isolated_low_probability_words",)
        return ("low_probability_words",)
    if state.method_normal:
        return ("missing_trust_evidence",)
    return ("stage25_untrusted",)


def _stage25_state(
    segment: ReconstructionWindow | Mapping[str, object], config: AdaptiveRoutingConfig
) -> _Stage25State:
    """Normalize the Stage 2.5 outcome and which raw words its repair resolved."""

    if isinstance(segment, ReconstructionWindow):
        return _Stage25State(False, False, frozenset())
    method = str(segment.get("correction_method") or "").strip().casefold()
    method_normal = method not in {"", "pending", "failed", "provider_unavailable"}
    applied = bool(segment.get("correction_applied"))
    confidence = segment.get("correction_confidence")
    trusted = (
        method_normal
        and applied
        and isinstance(confidence, int | float)
        and float(confidence) >= config.stage25_trust_confidence
    )
    resolved: set[int] = set()
    if trusted:
        resolved = _resolved_indexes(segment)
    return _Stage25State(method_normal, trusted, frozenset(resolved))


def _resolved_indexes(segment: Mapping[str, object]) -> set[int]:
    """Map a trusted repair's changes back to the raw words they resolve.

    A change whose normalized ``from`` equals the normalized raw segment text
    resolves every word index; otherwise each raw word whose normalized text
    equals a change's normalized ``from`` is resolved. Raw word order is
    preserved end-to-end, so positional resolution is exact.
    """

    changes = segment.get("correction_changes")
    if not isinstance(changes, list) or not changes:
        return set()
    raw_text = str(segment.get("raw_text", segment.get("text", "")))
    raw_words = segment.get("words")
    if not isinstance(raw_words, list):
        return set()
    normalized_raw = _normalize_token(raw_text)
    normalized_from = {
        _normalize_token(str(change.get("from", "")))
        for change in changes
        if isinstance(change, Mapping) and change.get("from")
    }
    if normalized_raw in normalized_from:
        return set(range(len(raw_words)))
    resolved: set[int] = set()
    for index, word in enumerate(raw_words):
        if (
            isinstance(word, Mapping)
            and _normalize_token(str(word.get("word", ""))) in normalized_from
        ):
            resolved.add(index)
    return resolved


def _residual_words(
    words: tuple[WordEvidence, ...], state: _Stage25State
) -> tuple[WordEvidence, ...]:
    """Raw words that still need scrutiny after a trusted repair."""

    if not state.trusted_repair or not state.resolved_indexes:
        return words
    return tuple(word for index, word in enumerate(words) if index not in state.resolved_indexes)


def _evidence_coverage(words: tuple[WordEvidence, ...]) -> float:
    if not words:
        return 0.0
    present = sum(1 for word in words if word.probability is not None)
    return present / len(words)


def _average_probability(words: tuple[WordEvidence, ...]) -> float | None:
    values = [word.probability for word in words if word.probability is not None]
    return sum(values) / len(values) if values else None


def _unresolved_warning(segment: ReconstructionWindow | Mapping[str, object]) -> bool:
    """True when the segment carries an explicit unresolved/reconstruction warning."""

    if isinstance(segment, ReconstructionWindow):
        return False
    flags = segment.get("quality_flags")
    if isinstance(flags, list):
        for flag in flags:
            if str(flag) in {"LOW_CONFIDENCE_UNRESOLVED", "RECONSTRUCTION_PROVIDER_ERROR"}:
                return True
    status = segment.get("reconstruction_status")
    return status in {"LOW_CONFIDENCE_UNRESOLVED", "PROVIDER_UNAVAILABLE"}


def _normalize_token(text: str) -> str:
    """Normalize Arabic spelling/layout so change mapping is spell tolerant."""

    normalized = unicodedata.normalize("NFC", text)
    without_diacritics = _ARABIC_DIACRITICS.sub("", normalized)
    return without_diacritics.translate(_ARABIC_COMPARISON).casefold().strip()


_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_ARABIC_COMPARISON = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})


def _segment_evidence(
    segment: ReconstructionWindow | Mapping[str, object],
) -> tuple[tuple[WordEvidence, ...], float | None]:
    if isinstance(segment, ReconstructionWindow):
        current = next(
            item for item in segment.segments if item.segment_index == segment.target_segment_index
        )
        return current.word_evidence, current.acoustic.average_word_probability
    raw_words = segment.get("words", [])
    words = (
        tuple(
            WordEvidence(
                str(w.get("word", "")),
                probability=float(w["probability"])
                if isinstance(w, Mapping) and isinstance(w.get("probability"), (int, float))
                else None,
            )
            for w in raw_words
            if isinstance(w, Mapping)
        )
        if isinstance(raw_words, list)
        else ()
    )
    vals = [w.probability for w in words if w.probability is not None]
    return words, (sum(vals) / len(vals) if vals else None)


def _hard_corruption_evidence(
    words: tuple[WordEvidence, ...],
    decision: RoutingDecision,
    config: AdaptiveRoutingConfig,
) -> tuple[bool, tuple[str, ...]]:
    """Return True plus named evidence when a segment clearly defeats a 4B pass."""

    thresholds = RoutingConfig()
    very_low = tuple(
        w
        for w in words
        if w.probability is not None and w.probability < thresholds.very_low_probability_threshold
    )
    low = tuple(
        w
        for w in words
        if w.probability is not None and w.probability < thresholds.low_probability_threshold
    )
    if not very_low and not low:
        return False, ()
    contiguous = _max_contiguous_very_low(words, thresholds.very_low_probability_threshold)
    low_ratio = decision.evidence.low_probability_ratio
    indicators: list[str] = []
    if contiguous >= config.gemini_direct_contiguous_very_low:
        indicators.append(f"contiguous_very_low_words={contiguous}")
    if (
        low_ratio >= config.gemini_direct_low_ratio
        and len(low) >= config.gemini_direct_min_low_count
    ):
        indicators.append(f"low_probability_ratio={low_ratio:.2f}")
    if decision.evidence.score >= config.gemini_direct_score:
        indicators.append(f"routing_score={decision.evidence.score:.2f}")
    if (
        _protected_overlap(very_low)
        and len(very_low) >= config.gemini_direct_protected_overlap_min_very_low
    ):
        indicators.append("uncertain_protected_entity_overlap")
    if len(very_low) >= 2 and low_ratio >= 0.40:
        indicators.append("multiple_severe_uncertainty_indicators")
    return bool(indicators), tuple(indicators)


def _severity(
    words: tuple[WordEvidence, ...],
    decision: RoutingDecision,
    protected_ambiguity: bool,
) -> float:
    """Deterministic difficulty score used to spend scarce provider budgets.

    Components: the existing routing score, a bounded per-contiguous-very-low
    word bonus, a bonus for multiple very-low words, a bonus for localized
    uncertainty over protected tokens, and a bonus for multiple severe
    uncertainty indicators. Only ordering and deterministic tie-breaking matter.
    """

    thresholds = RoutingConfig()
    very_low = tuple(
        w
        for w in words
        if w.probability is not None and w.probability < thresholds.very_low_probability_threshold
    )
    contiguous = _max_contiguous_very_low(words, thresholds.very_low_probability_threshold)
    score = decision.evidence.score
    score += 0.10 * min(contiguous, 5)
    if len(very_low) >= 2:
        score += 0.05
    if protected_ambiguity:
        score += 0.15
    if len(very_low) >= 2 and decision.evidence.low_probability_ratio >= 0.40:
        score += 0.10
    return score


def _max_contiguous_very_low(words: tuple[WordEvidence, ...], threshold: float) -> int:
    best = 0
    current = 0
    for word in words:
        if word.probability is not None and word.probability < threshold:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _protected_overlap(words: tuple[WordEvidence, ...]) -> bool:
    """Any uncertain word carrying a Latin or digit run (name/number/date)."""

    return any(
        character.isdigit() or (character.isascii() and character.isalpha())
        for w in words
        for character in w.text
    )
