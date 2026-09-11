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
    CANDIDATE_ANALYSIS = "CANDIDATE_ANALYSIS"
    READY_FOR_REFINEMENT = "READY_FOR_REFINEMENT"


class JobKind(str, Enum):
    """Background operations supported by the pipeline."""

    INGEST = "INGEST"
    TRANSCRIPTION = "TRANSCRIPTION"
    RECONSTRUCTION = "RECONSTRUCTION"
    PROBE = "PROBE"
    CANDIDATE_ANALYSIS = "CANDIDATE_ANALYSIS"


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


class RefinementPriority(str, Enum):
    """Transcript refinement quality tier.

    The quality ladder is: whole-source transcripts are INDEX (indexing quality),
    a shortlisted window is CANDIDATE (semantic quality), and a selected final
    clip is FINAL_CLIP (publication/caption quality). Expensive provider
    reconstruction is deferred until a short region is close to publication.
    """

    INDEX = "INDEX"
    CANDIDATE = "CANDIDATE"
    FINAL_CLIP = "FINAL_CLIP"


class MediaOriginType(str, Enum):
    """How the source material was originally produced.

    ``OTHER`` is the truthful default: it means unclassified, never an
    operator-declared ownership statement. Media origin is independent of
    ``RightsStatus`` and never blocks local analysis.
    """

    YOUTUBE_CREATOR_VIDEO = "YOUTUBE_CREATOR_VIDEO"
    PODCAST_INTERVIEW = "PODCAST_INTERVIEW"
    MOVIE_TV = "MOVIE_TV"
    NEWS_CLIP = "NEWS_CLIP"
    SPORTS_BROADCAST = "SPORTS_BROADCAST"
    OTHER = "OTHER"


class ContentType(str, Enum):
    """Small closed content ontology for coarse candidate classification."""

    EDUCATIONAL = "EDUCATIONAL"
    CONTROVERSIAL_OPINION = "CONTROVERSIAL_OPINION"
    FUNNY = "FUNNY"
    STORY = "STORY"
    SURPRISING_FACT = "SURPRISING_FACT"
    EMOTIONAL = "EMOTIONAL"
    NEWS_CURRENT_EVENT = "NEWS_CURRENT_EVENT"
    INTERVIEW_INSIGHT = "INTERVIEW_INSIGHT"
    DEBATE = "DEBATE"
    MOTIVATIONAL = "MOTIVATIONAL"
    TUTORIAL = "TUTORIAL"
    ANALYSIS = "ANALYSIS"
    REACTION_WORTHY = "REACTION_WORTHY"
    OTHER = "OTHER"


class CandidateDisposition(str, Enum):
    """What Stage 3 decided about a proposal."""

    CANDIDATE = "CANDIDATE"
    CANDIDATE_NEEDS_REFINEMENT = "CANDIDATE_NEEDS_REFINEMENT"
    DO_NOT_CLIP = "DO_NOT_CLIP"
    DO_NOT_CLIP_RECENTLY_REDUNDANT = "DO_NOT_CLIP_RECENTLY_REDUNDANT"


class OriginalityRisk(str, Enum):
    """Originality/transformation risk, orthogonal to rights risk."""

    NOT_INDICATED = "NOT_INDICATED"
    UNDETERMINED = "UNDETERMINED"
    TRANSFORMATION_REQUIRED = "TRANSFORMATION_REQUIRED"


class RightsRisk(str, Enum):
    """Rights/provenance risk, orthogonal to originality risk."""

    LOW = "LOW"
    UNDETERMINED = "UNDETERMINED"
    ELEVATED = "ELEVATED"


class HookType(str, Enum):
    """Supported short-form hook directions."""

    CURIOSITY = "CURIOSITY"
    CONTRADICTION = "CONTRADICTION"
    QUESTION = "QUESTION"
    DIRECT_CLAIM = "DIRECT_CLAIM"
    EMOTIONAL = "EMOTIONAL"
    PAYOFF_FIRST = "PAYOFF_FIRST"
    CONTEXTUAL = "CONTEXTUAL"
    SEARCH_LED = "SEARCH_LED"


class HookOrigin(str, Enum):
    """Whether a hook was derived deterministically or by a provider."""

    DETERMINISTIC = "DETERMINISTIC"
    PROVIDER = "PROVIDER"


class SemanticProviderMode(str, Enum):
    """Stage 3 semantic evaluation strategy."""

    DETERMINISTIC = "deterministic"
    ADAPTIVE = "adaptive"
    LOCAL_ONLY = "local_only"


class RefinementReason(str, Enum):
    """Bounded reasons a promising candidate still needs Stage 3.5 work."""

    LOW_TRANSCRIPT_CONFIDENCE = "LOW_TRANSCRIPT_CONFIDENCE"
    UNRESOLVED_INDEX_TEXT = "UNRESOLVED_INDEX_TEXT"
    LOW_CONFIDENCE_WORD_SPAN = "LOW_CONFIDENCE_WORD_SPAN"
    CODE_SWITCH_UNCERTAINTY = "CODE_SWITCH_UNCERTAINTY"
    PROTECTED_ENTITY_UNCERTAINTY = "PROTECTED_ENTITY_UNCERTAINTY"
    LOW_BOUNDARY_CONFIDENCE = "LOW_BOUNDARY_CONFIDENCE"
