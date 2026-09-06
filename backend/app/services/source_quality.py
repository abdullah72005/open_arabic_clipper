"""Independent transcript and audio quality evidence for Stage 2 triage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import ReconstructionStatus
from app.models import AudioAnalysis, SourceQualityAssessment, SourceVideo, Transcript
from app.pipeline.fingerprints import canonical_fingerprint
from app.transcription.reconstruction.windows import acoustic_evidence

QUALITY_VERSION = 2
LOW_CONFIDENCE_WORD_THRESHOLD = 0.72
_UNRESOLVED_STATUSES = {
    ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED.value,
    ReconstructionStatus.PROVIDER_UNAVAILABLE.value,
    ReconstructionStatus.FAILED.value,
}


@dataclass(frozen=True)
class TranscriptQualityEvidence:
    """Bounded transcript-quality result derived from persisted public evidence."""

    score: float
    low_confidence_word_ratio: float
    unresolved_segment_ratio: float
    manual_review_required: bool
    reasons: tuple[str, ...]


def assess_transcript_quality(transcript: Transcript) -> TranscriptQualityEvidence:
    """Score speech segments without allowing waveform quality to inflate the result."""

    included: list[tuple[float, float]] = []
    raw_included: list[tuple[float, float]] = []
    speech_segments = 0
    unresolved_segments = 0
    word_probabilities: list[float] = []
    statuses: list[str] = []
    for segment in _segments(transcript):
        word_probabilities.extend(_word_probabilities(segment))
        raw_acoustic = _clamp(acoustic_evidence(segment).confidence or 0.0)
        status = _status(segment.get("reconstruction_status"))
        if status == ReconstructionStatus.NOT_REQUIRED.value:
            continue
        speech_segments += 1
        if status in _UNRESOLVED_STATUSES:
            unresolved_segments += 1
        if status:
            statuses.append(status)
        weight = _segment_weight(segment)
        included.append((weight, _segment_quality(segment, status, raw_acoustic)))
        raw_included.append((weight, raw_acoustic))

    score = _weighted_average(included)
    raw_acoustic = _weighted_average(raw_included)
    low_ratio = (
        sum(value < LOW_CONFIDENCE_WORD_THRESHOLD for value in word_probabilities)
        / len(word_probabilities)
        if word_probabilities
        else 0.0
    )
    unresolved_ratio = unresolved_segments / speech_segments if speech_segments else 0.0
    metadata = transcript.reconstruction_metadata or {}
    provider_availability = str(metadata.get("provider_availability", "AVAILABLE"))
    manual_review = unresolved_segments > 0 or (
        transcript.language == "ar" and provider_availability != "AVAILABLE"
    )
    status_summary = ",".join(sorted(set(statuses))) or ReconstructionStatus.NOT_REQUIRED.value
    reasons = (
        f"transcript:raw_acoustic_confidence={raw_acoustic:.2f}",
        f"transcript:low_confidence_word_ratio={low_ratio:.2f}",
        f"transcript:unresolved_segment_ratio={unresolved_ratio:.2f}",
        f"transcript:correction_confidence={_clamp(_number(transcript.correction_confidence)):.2f}",
        f"transcript:corrected_segment_ratio={_clamp(_number(transcript.corrected_segment_ratio)):.2f}",
        f"transcript:uncertain_segment_ratio={_clamp(_number(transcript.uncertain_segment_ratio)):.2f}",
        f"transcript:reconstruction_confidence={_clamp(_number(transcript.reconstruction_confidence)):.2f}",
        f"transcript:provider_availability={provider_availability}",
        f"transcript:reconstruction_status={status_summary}",
        f"transcript:manual_review_required={str(manual_review).lower()}",
    )
    return TranscriptQualityEvidence(
        score=_clamp(score),
        low_confidence_word_ratio=_clamp(low_ratio),
        unresolved_segment_ratio=_clamp(unresolved_ratio),
        manual_review_required=manual_review,
        reasons=reasons,
    )


def assess_source(
    session: Session,
    source: SourceVideo,
    transcript: Transcript,
    analysis: AudioAnalysis,
) -> SourceQualityAssessment:
    """Persist independent quality signals without changing source eligibility."""

    evidence = assess_transcript_quality(transcript)
    audio_quality = _clamp(1.0 - _number(analysis.silence_ratio))
    repetition = _repetition_score(transcript.normalized_text)
    assessment = session.scalar(
        select(SourceQualityAssessment).where(SourceQualityAssessment.source_video_id == source.id)
    )
    if assessment is None:
        assessment = SourceQualityAssessment(source_video=source)
        session.add(assessment)
    assessment.transcript_confidence = _weighted_raw_acoustic(transcript)
    assessment.transcript_quality_score = evidence.score
    assessment.low_confidence_word_ratio = evidence.low_confidence_word_ratio
    assessment.unresolved_segment_ratio = evidence.unresolved_segment_ratio
    assessment.manual_review_required = evidence.manual_review_required
    assessment.input_fingerprint = quality_input_fingerprint(transcript, analysis)
    assessment.speech_density = _clamp(_number(analysis.speech_density))
    assessment.silence_ratio = _clamp(_number(analysis.silence_ratio))
    assessment.audio_quality_score = audio_quality
    assessment.preliminary_visual_quality_score = None
    assessment.repetition_score = _clamp(repetition)
    assessment.estimated_candidate_density = None
    assessment.language_confidence = _clamp(_number(transcript.detected_language_probability))
    assessment.overall_source_quality_score = min(audio_quality, evidence.score)
    assessment.reasons = [
        f"audio:speech_density={assessment.speech_density:.2f}",
        f"audio:silence_ratio={assessment.silence_ratio:.2f}",
        *evidence.reasons,
    ]
    assessment.version = QUALITY_VERSION
    session.commit()
    session.refresh(assessment)
    return assessment


def quality_input_fingerprint(transcript: Transcript, analysis: AudioAnalysis) -> str:
    """Fingerprint every persisted audio/reconstruction input used by quality scoring."""

    reconstruction_segments = [
        {
            "words": segment.get("words", []),
            "avg_logprob": segment.get("avg_logprob"),
            "no_speech_prob": segment.get("no_speech_prob"),
            "correction_applied": segment.get("correction_applied"),
            "correction_confidence": segment.get("correction_confidence"),
            "reconstruction_status": segment.get("reconstruction_status"),
            "reconstruction_confidence": segment.get("reconstruction_confidence"),
            "routing_score": segment.get("routing_score"),
            "routing_priority": segment.get("routing_priority"),
            "start": segment.get("start"),
            "end": segment.get("end"),
        }
        for segment in _segments(transcript)
    ]
    return canonical_fingerprint(
        "source-quality-input",
        str(QUALITY_VERSION),
        {
            "quality_version": QUALITY_VERSION,
            "audio_analysis": {
                "audio_hash": analysis.audio_hash,
                "features": analysis.features,
                "silence_intervals": analysis.silence_intervals,
                "silence_ratio": analysis.silence_ratio,
                "speech_density": analysis.speech_density,
                "speech_rate": analysis.speech_rate,
            },
            "transcript": {
                "language": transcript.language,
                "detected_language_probability": transcript.detected_language_probability,
                "correction_confidence": transcript.correction_confidence,
                "corrected_segment_ratio": transcript.corrected_segment_ratio,
                "uncertain_segment_ratio": transcript.uncertain_segment_ratio,
                "reconstruction_fingerprint": transcript.reconstruction_fingerprint,
                "reconstruction_confidence": transcript.reconstruction_confidence,
                "reconstruction_status": _status(transcript.reconstruction_status),
                "reconstruction_metadata": transcript.reconstruction_metadata,
                "segments": reconstruction_segments,
            },
        },
    )


def _segment_quality(
    segment: Mapping[str, object], status: str | None, raw_acoustic: float
) -> float:
    reconstruction_confidence = _clamp(_number(segment.get("reconstruction_confidence")))
    if status == ReconstructionStatus.APPLIED.value:
        return _clamp(0.35 * raw_acoustic + 0.65 * reconstruction_confidence)
    if status == ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED.value:
        return min(raw_acoustic, 0.45)
    if status == ReconstructionStatus.PROVIDER_UNAVAILABLE.value:
        return min(raw_acoustic, 0.40) if _is_routed(segment) else raw_acoustic
    if status == ReconstructionStatus.FAILED.value:
        return min(raw_acoustic, 0.25)
    if status == ReconstructionStatus.MANUAL_OVERRIDE.value:
        return 0.95
    if status is None and segment.get("correction_applied") is True:
        correction_confidence = _clamp(_number(segment.get("correction_confidence")))
        return _clamp(0.50 * raw_acoustic + 0.50 * correction_confidence)
    return raw_acoustic


def _weighted_raw_acoustic(transcript: Transcript) -> float:
    values = [
        (_segment_weight(segment), _clamp(acoustic_evidence(segment).confidence or 0.0))
        for segment in _segments(transcript)
        if _status(segment.get("reconstruction_status"))
        != ReconstructionStatus.NOT_REQUIRED.value
    ]
    return _clamp(_weighted_average(values))


def _weighted_average(values: Sequence[tuple[float, float]]) -> float:
    if len(values) == 1:
        return values[0][1]
    total_weight = sum(weight for weight, _ in values)
    return (
        sum(weight * value for weight, value in values) / total_weight
        if total_weight
        else 0.0
    )


def _segment_weight(segment: Mapping[str, object]) -> float:
    words = segment.get("words")
    if isinstance(words, list) and words:
        return float(len(words))
    duration = max(0.0, _number(segment.get("end")) - _number(segment.get("start")))
    return duration or 1.0


def _word_probabilities(segment: Mapping[str, object]) -> list[float]:
    words = segment.get("words")
    if not isinstance(words, list):
        return []
    return [
        _clamp(float(word["probability"]))
        for word in words
        if isinstance(word, Mapping) and _is_number(word.get("probability"))
    ]


def _segments(transcript: Transcript) -> list[Mapping[str, object]]:
    return [segment for segment in (transcript.segments or []) if isinstance(segment, Mapping)]


def _is_routed(segment: Mapping[str, object]) -> bool:
    if _is_number(segment.get("routing_score")):
        return True
    return str(segment.get("routing_priority", "")).upper() == "RECONSTRUCT"


def _status(value: object) -> str | None:
    if isinstance(value, ReconstructionStatus):
        return value.value
    return str(value) if isinstance(value, str) and value else None


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _number(value: object) -> float:
    return float(value) if _is_number(value) else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _repetition_score(text: str) -> float:
    words = text.casefold().split()
    if len(words) < 2:
        return 0.0
    return 1.0 - len(set(words)) / len(words)
