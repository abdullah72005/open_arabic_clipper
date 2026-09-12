"""Read-only Stage 3 replay helpers for INDEX execution experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from app.candidates.service import CandidateAnalysisService
from app.candidates.types import CandidateAnalysisOutcome
from app.candidates.novelty import NoveltyItem
from app.core.enums import CandidateDisposition, MediaOriginType, RightsStatus
from app.transcription.correction import ContextualCorrector
from app.transcription.dialect import (
    ArabicDialectProfile,
    DialectDetector,
    code_switch_evidence,
    segment_code_switch_suspected,
)
from app.transcription.engine import TranscriptionResult
from app.transcription.normalization import normalize_transcript

_RETAINED_DISPOSITIONS = {
    CandidateDisposition.CANDIDATE,
    CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
}


@dataclass(frozen=True)
class CandidateSnapshot:
    """Minimal persisted-candidate evidence required for a read-only comparison."""

    candidate_key: str
    disposition: CandidateDisposition


@dataclass(frozen=True)
class IndexReplayReport:
    """Transcript and Stage 3 comparison for an unpersisted ASR experiment."""

    language: str | None
    word_timestamp_count: int
    raw_segment_count: int
    dialect_profile: str | None
    dialect_confidence: float
    code_switch_segment_count: int
    baseline_retained_candidate_count: int
    replay_retained_candidate_count: int
    retained_candidate_overlap_count: int
    missing_baseline_candidate_keys: tuple[str, ...]
    new_replay_candidate_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def replay_index_candidates(
    *,
    source_id: str,
    result: TranscriptionResult,
    duration: float,
    silence_intervals: Sequence[Mapping[str, object]],
    audio_features: Sequence[Mapping[str, object]],
    rights_status: RightsStatus,
    media_origin: MediaOriginType,
    provenance_metadata: Mapping[str, object],
    dialect_override: ArabicDialectProfile | None,
    baseline_candidates: Sequence[CandidateSnapshot],
    service: CandidateAnalysisService,
    corrector: ContextualCorrector,
    historical_corpus: Sequence[NoveltyItem],
) -> IndexReplayReport:
    """Replay normalization and deterministic Stage 3 without mutating durable rows."""

    segments, profile, confidence = _normalized_segments(
        result.segments,
        language=result.language,
        dialect_override=dialect_override,
        corrector=corrector,
    )
    outcome = service.analyze(
        source_id=source_id,
        segments=segments,
        duration=duration,
        language=result.language,
        dialect_profile=profile,
        dialect_confidence=confidence,
        silence_intervals=silence_intervals,
        audio_features=audio_features,
        rights_status=rights_status,
        media_origin=media_origin,
        provenance_metadata=provenance_metadata,
        historical_corpus=historical_corpus,
    )
    return _report(result, profile, confidence, segments, baseline_candidates, outcome)


def _normalized_segments(
    raw_segments: Sequence[Mapping[str, object]],
    *,
    language: str | None,
    dialect_override: ArabicDialectProfile | None,
    corrector: ContextualCorrector,
) -> tuple[list[dict[str, object]], str | None, float]:
    detector = DialectDetector()
    detection = detector.detect(raw_segments, language=language, override=dialect_override)
    corrections = corrector.correct(raw_segments, profile=detection.profile)
    segments: list[dict[str, object]] = []
    for segment, correction in zip(raw_segments, corrections, strict=True):
        switch = code_switch_evidence(correction.raw_text)
        segments.append(
            {
                **segment,
                "raw_text": correction.raw_text,
                "corrected_text": correction.corrected_text,
                "correction_applied": correction.applied,
                "correction_confidence": correction.confidence,
                "correction_method": correction.method,
                "correction_version": correction.version,
                "correction_changes": correction.changes,
                "operator_text": None,
                "final_text": correction.corrected_text,
                "normalized_text": normalize_transcript(correction.corrected_text),
                "dialect_profile": (
                    detection.profile.value if detection.profile is not None else None
                ),
                "dialect_confidence": detection.confidence,
                "dialect_selection": detection.selection.value,
                "code_switch_suspected": segment_code_switch_suspected(correction.raw_text),
                "code_switch_tokens": list(switch.tokens),
            }
        )
    return (
        segments,
        detection.profile.value if detection.profile is not None else None,
        detection.confidence,
    )


def _report(
    result: TranscriptionResult,
    profile: str | None,
    confidence: float,
    segments: Sequence[Mapping[str, object]],
    baseline_candidates: Sequence[CandidateSnapshot],
    outcome: CandidateAnalysisOutcome,
) -> IndexReplayReport:
    baseline = {
        candidate.candidate_key
        for candidate in baseline_candidates
        if candidate.disposition in _RETAINED_DISPOSITIONS
    }
    replay = {
        candidate.candidate_key
        for candidate in outcome.candidates
        if candidate.disposition in _RETAINED_DISPOSITIONS
    }
    return IndexReplayReport(
        language=result.language,
        word_timestamp_count=len(result.word_segments),
        raw_segment_count=len(result.segments),
        dialect_profile=profile,
        dialect_confidence=confidence,
        code_switch_segment_count=sum(
            bool(segment.get("code_switch_suspected")) for segment in segments
        ),
        baseline_retained_candidate_count=len(baseline),
        replay_retained_candidate_count=len(replay),
        retained_candidate_overlap_count=len(baseline & replay),
        missing_baseline_candidate_keys=tuple(sorted(baseline - replay)),
        new_replay_candidate_keys=tuple(sorted(replay - baseline)),
    )
