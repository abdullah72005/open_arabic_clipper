"""Immutable values shared by contextual reconstruction services."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class AcousticEvidence:
    """Public faster-whisper confidence indicators for one raw segment."""

    confidence: float | None
    average_word_probability: float | None
    average_log_probability: float | None
    no_speech_probability: float | None


@dataclass(frozen=True)
class WindowSegment:
    """One immutable transcript segment included in a reconstruction window."""

    segment_index: int
    start: float
    end: float
    raw_text: str
    corrected_text: str
    acoustic: AcousticEvidence


@dataclass(frozen=True)
class ReconstructionWindow:
    """Bounded local transcript context for exactly one target segment."""

    target_segment_index: int
    segments: tuple[WindowSegment, ...]


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class QualityFlag(str, Enum):
    HIGH_ASR_UNCERTAINTY = "HIGH_ASR_UNCERTAINTY"
    MULTIWORD_RECONSTRUCTION = "MULTIWORD_RECONSTRUCTION"
    POSSIBLE_ENTITY_ERROR = "POSSIBLE_ENTITY_ERROR"
    CONTEXT_DEPENDENT_CORRECTION = "CONTEXT_DEPENDENT_CORRECTION"
    LOW_CONFIDENCE_UNRESOLVED = "LOW_CONFIDENCE_UNRESOLVED"
    RECONSTRUCTION_PROVIDER_ERROR = "RECONSTRUCTION_PROVIDER_ERROR"


@dataclass(frozen=True)
class ReconstructionCandidate:
    candidate_id: str
    text: str
    changes: tuple[dict[str, object], ...] = ()
    evidence_segment_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ResolutionScores:
    semantic_coherence: float
    egyptian_naturalness: float
    discourse_continuity: float
    entity_consistency: float
    selection_confidence: float


@dataclass(frozen=True)
class SegmentReconstruction:
    segment_index: int
    raw_text: str
    corrected_text: str
    contextual_reconstructed_text: str
    candidate_text: str | None
    applied: bool
    confidence: float
    confidence_level: ConfidenceLevel
    quality_flags: tuple[QualityFlag, ...]


@dataclass(frozen=True)
class ReconstructionResult:
    segments: tuple[SegmentReconstruction, ...]
    contextual_reconstructed_text: str
    fingerprint: str
