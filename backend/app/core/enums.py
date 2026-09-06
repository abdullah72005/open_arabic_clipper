from enum import Enum


class RightsStatus(str, Enum):
    """The operator's declared authorization for a source."""

    UNKNOWN = "UNKNOWN"
    OWNED = "OWNED"
    LICENSED = "LICENSED"
    PERMISSION = "PERMISSION"
    PUBLIC_DOMAIN = "PUBLIC_DOMAIN"
    OTHER_ALLOWED = "OTHER_ALLOWED"
    THIRD_PARTY_UNKNOWN = "THIRD_PARTY_UNKNOWN"
    THIRD_PARTY_REUSE = "THIRD_PARTY_REUSE"


class PipelineStage(str, Enum):
    """The persisted source lifecycle."""

    INGEST = "INGEST"
    PROBE = "PROBE"
    READY_FOR_TRANSCRIPTION = "READY_FOR_TRANSCRIPTION"
    AUDIO_EXTRACTION = "AUDIO_EXTRACTION"
    TRANSCRIPTION = "TRANSCRIPTION"
    TRANSCRIPT_NORMALIZATION = "TRANSCRIPT_NORMALIZATION"
    CONTEXTUAL_RECONSTRUCTION = "CONTEXTUAL_RECONSTRUCTION"
    AUDIO_ANALYSIS = "AUDIO_ANALYSIS"
    READY_FOR_ANALYSIS = "READY_FOR_ANALYSIS"


class JobKind(str, Enum):
    """Background operations supported by the pipeline."""

    INGEST = "INGEST"
    TRANSCRIPTION = "TRANSCRIPTION"
    RECONSTRUCTION = "RECONSTRUCTION"
    PROBE = "PROBE"


class JobStatus(str, Enum):
    """Durable worker-job state."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PipelineRunStatus(str, Enum):
    """Durable state for one execution of a pipeline stage."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProviderAvailability(str, Enum):
    """Whether an optional reconstruction provider can serve requests."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    MISCONFIGURED = "MISCONFIGURED"


class ReconstructionStatus(str, Enum):
    """Truthful outcome of contextual transcript reconstruction."""

    NOT_REQUIRED = "NOT_REQUIRED"
    APPLIED = "APPLIED"
    UNCHANGED_HIGH_CONFIDENCE = "UNCHANGED_HIGH_CONFIDENCE"
    LOW_CONFIDENCE_UNRESOLVED = "LOW_CONFIDENCE_UNRESOLVED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    FAILED = "FAILED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
