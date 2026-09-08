"""Deterministic routing of transcript spans to reconstruction passes."""

from __future__ import annotations

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


ADAPTIVE_ROUTING_VERSION = "adaptive-routing-v2"


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
    """

    mode: RoutingMode = RoutingMode.ADAPTIVE
    # NO_LLM requires affirmative Stage 2.5 evidence: a non-unchanged correction
    # method carrying at least this confidence.
    stage25_trust_confidence: float = 0.90
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
    # Explicit OR routing rule: multiple weak words are sufficient evidence even
    # when the aggregate score is diluted by one confident neighboring word.
    if (len(low) >= 2 and ratio >= config.high_ratio_threshold) or score >= config.score_threshold:
        priority, reason = RoutingPriority.RECONSTRUCT, "multiple_low_probability_words"
    elif language == "ar" and (avg is None or avg >= config.low_probability_threshold) and not low:
        priority, reason = RoutingPriority.CONTEXT_CHECK, "high_confidence_arabic_context_check"
    else:
        priority, reason = RoutingPriority.LEAVE, "insufficient_uncertainty_evidence"
    evidence = RoutingEvidence(score, ratio, focus or tuple(low), reason)
    return RoutingDecision(priority, evidence, evidence.focus_spans, reason)


def route_adaptive(
    segment: ReconstructionWindow | Mapping[str, object],
    config: AdaptiveRoutingConfig | None = None,
    language: str | None = None,
    decision: RoutingDecision | None = None,
) -> AdaptiveRoutingDecision:
    """Choose the provider path for one target before any provider is invoked.

    ``NO_LLM`` requires affirmative Stage 2.5 trust (a non-unchanged correction
    at or above ``stage25_trust_confidence``) together with clean acoustic
    evidence and no localized protected-token ambiguity. High Whisper word
    probabilities alone never suppress contextual checking.
    """

    config = config or AdaptiveRoutingConfig()
    decision = decision or route_segment(segment, language=language)
    words, _avg = _segment_evidence(segment)
    thresholds = RoutingConfig()
    low = tuple(
        w
        for w in words
        if w.probability is not None and w.probability < thresholds.low_probability_threshold
    )
    protected_ambiguity = _protected_overlap(low)
    trust = _stage25_trustworthy(segment, config)
    severity = _severity(words, decision, protected_ambiguity)
    hard, evidence = _hard_corruption_evidence(words, decision, config)
    if hard:
        return AdaptiveRoutingDecision(
            ReconstructionRoute.GEMINI_DIRECT,
            "hard_corruption_evidence",
            evidence,
            decision.priority,
            decision.evidence.score,
            decision.focus_spans,
            severity,
        )
    if decision.priority is RoutingPriority.LEAVE:
        if trust and not low and not protected_ambiguity:
            return AdaptiveRoutingDecision(
                ReconstructionRoute.NO_LLM,
                "insufficient_uncertainty_evidence",
                ("no_low_probability_evidence", "stage25_trusted"),
                decision.priority,
                decision.evidence.score,
                decision.focus_spans,
                severity,
            )
        evidence_label = ("isolated_low_probability_words",) if low else ("stage25_untrusted",)
        return AdaptiveRoutingDecision(
            ReconstructionRoute.LOCAL,
            "stage25_untrusted_or_isolated_uncertainty",
            evidence_label,
            decision.priority,
            decision.evidence.score,
            decision.focus_spans,
            severity,
        )
    if decision.priority is RoutingPriority.CONTEXT_CHECK:
        if trust and not low and not protected_ambiguity and _has_positive_evidence(words):
            return AdaptiveRoutingDecision(
                ReconstructionRoute.NO_LLM,
                "high_confidence_trustworthy",
                ("high_confidence_arabic", "no_low_probability_words", "stage25_trusted"),
                decision.priority,
                decision.evidence.score,
                decision.focus_spans,
                severity,
            )
        evidence_label = (
            ("stage25_untrusted",)
            if not trust
            else ("missing_trust_evidence",)
            if not _has_positive_evidence(words)
            else ("low_probability_words",)
        )
        return AdaptiveRoutingDecision(
            ReconstructionRoute.LOCAL,
            "high_confidence_arabic_context_check",
            evidence_label,
            decision.priority,
            decision.evidence.score,
            decision.focus_spans,
            severity,
        )
    return AdaptiveRoutingDecision(
        ReconstructionRoute.LOCAL,
        "mild_localized_uncertainty",
        ("reconstruct_priority",),
        decision.priority,
        decision.evidence.score,
        decision.focus_spans,
        severity,
    )


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


def _stage25_trustworthy(
    segment: ReconstructionWindow | Mapping[str, object], config: AdaptiveRoutingConfig
) -> bool:
    """Affirmative Stage 2.5 trust: a real correction decision at high confidence.

    The unchanged path writes ``correction_confidence=0.0`` and
    ``correction_method="unchanged"``; that is exactly the untrusted case that
    must remain eligible for contextual checking.
    """

    if isinstance(segment, ReconstructionWindow):
        return False
    method = str(segment.get("correction_method") or "")
    if method in {"", "pending", "unchanged"}:
        return False
    confidence = segment.get("correction_confidence")
    if not isinstance(confidence, int | float):
        return False
    return float(confidence) >= config.stage25_trust_confidence


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
    """Deterministic difficulty score used to spend a scarce Gemini budget.

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


def _has_positive_evidence(words: tuple[WordEvidence, ...]) -> bool:
    return any(word.probability is not None for word in words)
