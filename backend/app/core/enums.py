from enum import Enum


class RightsStatus(str, Enum):
    """The operator's declared authorization for a source."""

    UNKNOWN = "UNKNOWN"
    OWNED = "OWNED"
    LICENSED = "LICENSED"
    PERMISSION = "PERMISSION"
    PUBLIC_DOMAIN = "PUBLIC_DOMAIN"
    OTHER_ALLOWED = "OTHER_ALLOWED"


class PipelineStage(str, Enum):
    """The persisted Stage 1 source lifecycle."""

    INGEST = "INGEST"
    PROBE = "PROBE"
    READY_FOR_TRANSCRIPTION = "READY_FOR_TRANSCRIPTION"


class JobKind(str, Enum):
    """Background operations supported by the Stage 1 pipeline."""

    INGEST = "INGEST"
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
