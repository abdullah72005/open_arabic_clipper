"""Deterministic Stage 3 candidate scoring and uncertainty derivation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from app.candidates.cues import CONTENT_CUES, FILLER_CUES, PAYOFF_CUES
from app.candidates.policy import DEFAULT_CONFIG, Stage3Config
from app.candidates.text import (
    analysis_segment_text,
    contains_any_cue,
    contains_cue,
    is_question,
    matching_text,
    tokenize,
)
from app.candidates.types import (
    CandidateScores,
    ContentClassification,
    Proposal,
    UncertaintyEvidence,
)
from app.core.enums import ReconstructionStatus, RefinementReason
from app.transcription.dialect import code_switch_evidence, extract_protected_tokens
from app.transcription.reconstruction.windows import acoustic_evidence

_UNRESOLVED_STATUSES = {
    ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED.value,
    ReconstructionStatus.PROVIDER_UNAVAILABLE.value,
    ReconstructionStatus.FAILED.value,
}
_BOUNDARY_QUALITY = {
    "speaker_change": 0.88,
    "pause": 0.90,
    "silence": 0.92,
    "sentence_end": 0.86,
    "question_answer": 0.85,
    "energy_change": 0.72,
    "contrast": 0.70,
    "max_window": 0.65,
    "end_of_source": 0.80,
    "tail": 0.55,
    "merged": 0.62,
    "fallback_window": 0.42,
    "single_segment": 0.35,
}


def compute_uncertainty(
    proposal: Proposal,
    segments: Sequence[Mapping[str, object]],
    *,
    config: Stage3Config = DEFAULT_CONFIG,
) -> UncertaintyEvidence:
    """Derive transcript/uncertainty evidence from the candidate's own bounds."""

    words: list[tuple[float, bool]] = []
    logprobs: list[float] = []
    no_speech: list[float] = []
    low_spans: list[dict[str, object]] = []
    unresolved: list[int] = []
    overrides = 0
    code_switch_tokens: list[str] = []
    protected_tokens: list[str] = []
    index_deferred = False

    for index in proposal.segment_indexes:
        segment = segments[index]
        text = analysis_segment_text(segment)
        if segment.get("operator_text"):
            overrides += 1
        for token in extract_protected_tokens(text):
            protected_tokens.append(token)
        if segment.get("code_switch_suspected"):
            stored_tokens = _as_str_list(segment.get("code_switch_tokens"))
            if stored_tokens:
                code_switch_tokens.extend(stored_tokens)
            else:
                code_switch_tokens.extend(code_switch_evidence(text).tokens)
        status = str(segment.get("reconstruction_status") or "")
        if status in _UNRESOLVED_STATUSES or segment.get("needs_refinement") is True:
            unresolved.append(index)
        if segment.get("reconstruction_method") == "index_deferred":
            index_deferred = True
        for word in _as_mapping_list(segment.get("words")):
            probability = _number(word.get("probability"))
            low = probability < config.low_word_probability_threshold
            words.append((probability, low))
            if low:
                low_spans.append(
                    {
                        "segment_index": index,
                        "word": str(word.get("word", "")),
                        "start": _number(word.get("start")),
                        "end": _number(word.get("end")),
                        "probability": probability,
                    }
                )
        if _number(segment.get("avg_logprob")):
            logprobs.append(_number(segment.get("avg_logprob")))
        if isinstance(segment.get("no_speech_prob"), int | float):
            no_speech.append(_number(segment.get("no_speech_prob")))
    if overrides and not words:
        transcript_confidence = 0.9
    elif words:
        mean_probability = sum(probability for probability, _ in words) / len(words)
        low_ratio = sum(1 for _, low in words if low) / len(words)
        transcript_confidence = _clamp(mean_probability * (1.0 - 0.35 * low_ratio))
    elif logprobs:
        transcript_confidence = _clamp(1.0 + sum(logprobs) / len(logprobs))
    elif overrides:
        transcript_confidence = 0.85
    else:
        transcript_confidence = 0.5
    if no_speech:
        transcript_confidence = _clamp(
            transcript_confidence * (1.0 - 0.3 * (sum(no_speech) / len(no_speech)))
        )
    if overrides:
        transcript_confidence = max(transcript_confidence, 0.9)

    low_word_span_ratio = sum(1 for _, low in words if low) / len(words) if words else 0.0
    unresolved_ratio = (
        len(unresolved) / len(proposal.segment_indexes) if proposal.segment_indexes else 0.0
    )

    reasons: list[RefinementReason] = []
    if transcript_confidence < config.low_transcript_confidence_threshold:
        reasons.append(RefinementReason.LOW_TRANSCRIPT_CONFIDENCE)
    if index_deferred or unresolved:
        reasons.append(RefinementReason.UNRESOLVED_INDEX_TEXT)
    if low_word_span_ratio > 0:
        reasons.append(RefinementReason.LOW_CONFIDENCE_WORD_SPAN)
    protected_uncertainty = bool(protected_tokens) and (low_word_span_ratio > 0 or bool(unresolved))
    if protected_uncertainty:
        reasons.append(RefinementReason.PROTECTED_ENTITY_UNCERTAINTY)
    code_switch_uncertainty = bool(code_switch_tokens) and (
        low_word_span_ratio > 0 or bool(unresolved) or transcript_confidence < 0.7
    )
    if code_switch_uncertainty:
        reasons.append(RefinementReason.CODE_SWITCH_UNCERTAINTY)

    return UncertaintyEvidence(
        transcript_confidence=transcript_confidence,
        low_confidence_spans=tuple(low_spans[:50]),
        unresolved_segment_indexes=tuple(unresolved),
        low_confidence_word_span_ratio=low_word_span_ratio,
        unresolved_ratio=unresolved_ratio,
        protected_tokens=tuple(dict.fromkeys(protected_tokens)),
        code_switch_tokens=tuple(dict.fromkeys(code_switch_tokens)),
        code_switch_uncertainty=code_switch_uncertainty,
        reasons=tuple(reasons),
    )


def compute_scores(
    proposal: Proposal,
    segments: Sequence[Mapping[str, object]],
    classification: ContentClassification,
    uncertainty: UncertaintyEvidence,
    *,
    audio_confidence: float | None = None,
    config: Stage3Config = DEFAULT_CONFIG,
) -> CandidateScores:
    """Compute independent normalized scores; content quality excludes transcript noise."""

    tokens = tokenize(proposal.text)
    word_count = len(tokens)
    duration = max(1.0, proposal.duration)
    words_per_second = word_count / duration
    density = _clamp(words_per_second / 3.0)
    matching = matching_text(proposal.text)
    cue_hits = sum(
        sum(1 for cue in cues if contains_cue(matching, cue)) for cues in CONTENT_CUES.values()
    )
    cue_strength = _clamp(cue_hits / 4.0)
    moment_density_score = _clamp(0.7 * density + 0.3 * cue_strength)

    short_form_score = _length_suitability(proposal.duration)
    ending_quality_score = _ending_quality(proposal, matching)
    loopability_score = _loopability(proposal, matching)

    filler = sum(1 for cue in FILLER_CUES if contains_cue(matching, cue))
    filler_ratio = filler / max(1, word_count)
    # Content-quality only: transcript uncertainty never lowers apparent moment
    # quality. The uncertainty term that previously inflated boredom is dropped;
    # the filler/density terms keep their original weights.
    boredom_risk_score = _clamp(0.5 * min(1.0, filler_ratio * 6.0) + 0.3 * (1.0 - density))

    boundary_confidence = _clamp(
        _BOUNDARY_QUALITY.get(proposal.boundary_reason, 0.55) - 0.25 * uncertainty.unresolved_ratio
    )
    reasons: list[RefinementReason] = list(uncertainty.reasons)
    if boundary_confidence < config.low_boundary_confidence_threshold:
        reasons.append(RefinementReason.LOW_BOUNDARY_CONFIDENCE)

    uncertainty_severity = _clamp(
        0.4 * (1.0 - uncertainty.transcript_confidence)
        + 0.25 * uncertainty.low_confidence_word_span_ratio
        + 0.2 * uncertainty.unresolved_ratio
        + 0.15 * (1.0 - boundary_confidence)
    )
    # Code switching alone is not uncertainty; only contributes when overlapping.
    if uncertainty.code_switch_uncertainty:
        uncertainty_severity = _clamp(uncertainty_severity + 0.1)

    engagement_confidence = _evidence_coverage(segments, proposal)
    resolved_audio = (
        audio_confidence if audio_confidence is not None else _audio_confidence(segments, proposal)
    )

    clip_score = aggregate_clip_score(
        moment_density_score=moment_density_score,
        short_form_score=short_form_score,
        word_density=density,
        ending_quality_score=ending_quality_score,
        boredom_risk_score=boredom_risk_score,
        loopability_score=loopability_score,
    )

    return CandidateScores(
        clip_score=clip_score,
        short_form_score=short_form_score,
        moment_density_score=moment_density_score,
        boredom_risk_score=boredom_risk_score,
        ending_quality_score=ending_quality_score,
        loopability_score=loopability_score,
        word_density=density,
        engagement_confidence=engagement_confidence,
        transcript_confidence=uncertainty.transcript_confidence,
        audio_confidence=resolved_audio,
        boundary_confidence=boundary_confidence,
        uncertainty_severity=uncertainty_severity,
        idea_novelty_score=0.0,
        topic_novelty_score=0.0,
        recent_semantic_similarity_risk=0.0,
        reasons=tuple(reason.value for reason in dict.fromkeys(reasons)),
    )


def material_uncertainty(
    scores: CandidateScores, uncertainty: UncertaintyEvidence, config: Stage3Config
) -> bool:
    return (
        bool(uncertainty.reasons)
        or scores.uncertainty_severity >= config.uncertainty_severity_threshold
        or scores.transcript_confidence < config.low_transcript_confidence_threshold
    )


def aggregate_clip_score(
    *,
    moment_density_score: float,
    short_form_score: float,
    word_density: float,
    ending_quality_score: float,
    boredom_risk_score: float,
    loopability_score: float,
) -> float:
    """Single content-quality aggregate shared by deterministic and provider paths.

    Only apparent content/moment quality participates. Transcript confidence,
    unresolved/deferred INDEX status, word confidence, code-switch uncertainty,
    audio confidence, boundary confidence, and uncertainty severity are separate
    fields and never enter this aggregate.
    """

    return _clamp(
        0.30 * moment_density_score
        + 0.20 * short_form_score
        + 0.15 * word_density
        + 0.12 * ending_quality_score
        + 0.13 * (1.0 - boredom_risk_score)
        + 0.10 * loopability_score
    )


def recompute_clip_score(scores: CandidateScores) -> float:
    """Recompute the content-quality aggregate after validated provider adjustments.

    Uses exactly the same aggregate formula as deterministic scoring via
    :func:`aggregate_clip_score`; a provider response with no accepted score
    adjustments therefore preserves the deterministic ClipScore.
    """

    return aggregate_clip_score(
        moment_density_score=scores.moment_density_score,
        short_form_score=scores.short_form_score,
        word_density=scores.word_density,
        ending_quality_score=scores.ending_quality_score,
        boredom_risk_score=scores.boredom_risk_score,
        loopability_score=scores.loopability_score,
    )


def _length_suitability(duration: float) -> float:
    if duration <= 0:
        return 0.0
    if 35.0 <= duration <= 75.0:
        return 1.0
    if duration < 35.0:
        return _clamp(duration / 35.0)
    return _clamp(1.0 - (duration - 75.0) / 75.0)


def _ending_quality(proposal: Proposal, matching: str) -> float:
    tail = matching[-120:]
    score = 0.5
    if contains_any_cue(tail, PAYOFF_CUES):
        score += 0.3
    if proposal.text.strip().endswith((".", "!", "؟", "?", "…")):
        score += 0.15
    if proposal.boundary_reason in {"sentence_end", "question_answer", "silence", "pause"}:
        score += 0.05
    return _clamp(score)


def _loopability(proposal: Proposal, matching: str) -> float:
    score = 0.4
    if is_question(proposal.text) or "?" in matching or "؟" in matching:
        score += 0.3
    if contains_any_cue(matching[:120], PAYOFF_CUES):
        score += 0.2
    if proposal.duration <= 60:
        score += 0.1
    return _clamp(score)


def _evidence_coverage(segments: Sequence[Mapping[str, object]], proposal: Proposal) -> float:
    if not proposal.segment_indexes:
        return 0.0
    text = 0
    words = 0
    acoustic = 0
    for index in proposal.segment_indexes:
        segment = segments[index]
        if analysis_segment_text(segment).strip():
            text += 1
        if _as_mapping_list(segment.get("words")):
            words += 1
        if acoustic_evidence(segment).confidence is not None:
            acoustic += 1
    total = len(proposal.segment_indexes)
    return _clamp((text / total + words / total + acoustic / total) / 3.0)


def _audio_confidence(segments: Sequence[Mapping[str, object]], proposal: Proposal) -> float:
    values = [
        _clamp(acoustic_evidence(segments[index]).confidence or 0.0)
        for index in proposal.segment_indexes
        if acoustic_evidence(segments[index]).confidence is not None
    ]
    if not values:
        return 0.4
    return _clamp(sum(values) / len(values))


def _as_mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _number(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _clamp(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))
