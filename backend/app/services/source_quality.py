"""Inexpensive advisory quality signals for Stage 2 source triage."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AudioAnalysis, SourceQualityAssessment, SourceVideo, Transcript

QUALITY_VERSION = 1


def assess_source(
    session: Session,
    source: SourceVideo,
    transcript: Transcript,
    analysis: AudioAnalysis,
) -> SourceQualityAssessment:
    """Persist evidence-based quality signals without changing source eligibility."""

    confidence = _transcript_confidence(transcript)
    repetition = _repetition_score(transcript.normalized_text)
    audio_quality = max(0.0, min(1.0, 1.0 - analysis.silence_ratio))
    overall = (
        confidence * 0.35
        + analysis.speech_density * 0.25
        + audio_quality * 0.25
        + (1.0 - repetition) * 0.15
    )
    reasons = [
        f"transcript_confidence={confidence:.2f}",
        f"speech_density={analysis.speech_density:.2f}",
        f"silence_ratio={analysis.silence_ratio:.2f}",
    ]
    assessment = session.scalar(
        select(SourceQualityAssessment).where(SourceQualityAssessment.source_video_id == source.id)
    )
    if assessment is None:
        assessment = SourceQualityAssessment(source_video=source)
        session.add(assessment)
    assessment.transcript_confidence = confidence
    assessment.speech_density = analysis.speech_density
    assessment.silence_ratio = analysis.silence_ratio
    assessment.audio_quality_score = audio_quality
    assessment.preliminary_visual_quality_score = None
    assessment.repetition_score = repetition
    assessment.estimated_candidate_density = None
    assessment.language_confidence = transcript.detected_language_probability or 0.0
    assessment.overall_source_quality_score = max(0.0, min(1.0, overall))
    assessment.reasons = reasons
    assessment.version = QUALITY_VERSION
    session.commit()
    session.refresh(assessment)
    return assessment


def _transcript_confidence(transcript: Transcript) -> float:
    values = [segment.get("avg_logprob") for segment in transcript.segments]
    logprobs = [float(value) for value in values if isinstance(value, int | float)]
    if not logprobs:
        return 0.0
    return max(0.0, min(1.0, 1.0 + sum(logprobs) / len(logprobs)))


def _repetition_score(text: str) -> float:
    words = text.casefold().split()
    if len(words) < 2:
        return 0.0
    return 1.0 - len(set(words)) / len(words)
