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
@dataclass(frozen=True)
class RoutingConfig:
    low_probability_threshold: float = 0.72
    very_low_probability_threshold: float = 0.50
    low_ratio_threshold: float = 0.78
    high_ratio_threshold: float = 0.25
    score_threshold: float = 0.45
def route_segment(segment: ReconstructionWindow | Mapping[str, object], config: RoutingConfig = RoutingConfig(), language: str | None = None) -> RoutingDecision:
    if isinstance(segment, ReconstructionWindow):
        current = next(item for item in segment.segments if item.segment_index == segment.target_segment_index)
        words, avg = current.word_evidence, current.acoustic.average_word_probability
    else:
        raw_words = segment.get("words", [])
        words = tuple(WordEvidence(str(w.get("word", "")), probability=float(w["probability"]) if isinstance(w, Mapping) and isinstance(w.get("probability"), (int, float)) else None) for w in raw_words if isinstance(w, Mapping)) if isinstance(raw_words, list) else ()
        vals = [w.probability for w in words if w.probability is not None]
        avg = sum(vals) / len(vals) if vals else None
    probs = [w.probability for w in words if w.probability is not None]
    low = [w for w in words if w.probability is not None and w.probability < config.low_probability_threshold]
    ratio = len(low) / len(probs) if probs else 0.0
    focus = tuple(w for w in low if w.probability is not None and w.probability < config.very_low_probability_threshold)
    score = 0.50 * (1 - (avg if avg is not None else config.low_probability_threshold)) + 0.25 * ratio + 0.25 * bool(focus)
    if len(low) >= 2 and ratio >= config.high_ratio_threshold and score >= config.score_threshold:
        priority, reason = RoutingPriority.RECONSTRUCT, "multiple_low_probability_words"
    elif language == "ar" and (avg is None or avg >= config.low_probability_threshold) and not low:
        priority, reason = RoutingPriority.CONTEXT_CHECK, "high_confidence_arabic_context_check"
    else:
        priority, reason = RoutingPriority.LEAVE, "insufficient_uncertainty_evidence"
    evidence = RoutingEvidence(score, ratio, focus or tuple(low), reason)
    return RoutingDecision(priority, evidence, evidence.focus_spans, reason)
