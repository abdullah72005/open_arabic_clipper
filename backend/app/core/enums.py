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
    AUDIO_ANALYSIS = "AUDIO_ANALYSIS"
    READY_FOR_ANALYSIS = "READY_FOR_ANALYSIS"


class JobKind(str, Enum):
    """Background operations supported by the pipeline."""

    INGEST = "INGEST"
    TRANSCRIPTION = "TRANSCRIPTION"
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
