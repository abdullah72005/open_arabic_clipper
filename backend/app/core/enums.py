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
    CANDIDATE_REFINEMENT = "CANDIDATE_REFINEMENT"
    TRANSFORMATION_ELIGIBILITY = "TRANSFORMATION_ELIGIBILITY"
    TRANSFORMATION_PLANNING = "TRANSFORMATION_PLANNING"


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


class RefinementStatus(str, Enum):
    """Persisted Stage 3.5 candidate-refinement lifecycle.

    Readiness (``CANDIDATE_REFINED``/``FINAL_TRANSCRIPT_READY``) and provider
    availability (``PROVIDER_DEGRADED``) are separate concepts: optional hosted
    provider failure never makes a usable local refinement unusable.
    """

    QUEUED = "QUEUED"
    REFINING = "REFINING"
    CANDIDATE_REFINED = "CANDIDATE_REFINED"
    FINAL_TRANSCRIPT_READY = "FINAL_TRANSCRIPT_READY"
    NEEDS_MANUAL_TRANSCRIPT_REVIEW = "NEEDS_MANUAL_TRANSCRIPT_REVIEW"
    PROVIDER_DEGRADED = "PROVIDER_DEGRADED"
    REFINEMENT_FAILED = "REFINEMENT_FAILED"
    CANCELLED = "CANCELLED"


class EvidenceKind(str, Enum):
    """Where a bounded transcript-evidence record came from."""

    INDEX_RAW = "INDEX_RAW"
    STAGE25 = "STAGE25"
    STAGE27 = "STAGE27"
    TARGETED_LOCAL_ASR = "TARGETED_LOCAL_ASR"
    HOSTED_ASR = "HOSTED_ASR"
    ADJUDICATION = "ADJUDICATION"
    OPERATOR = "OPERATOR"


class EvidenceState(str, Enum):
    """Acceptance state of one bounded evidence record."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    CHECKPOINT = "CHECKPOINT"


class AdmissionPriority(str, Enum):
    """Shared Gemini admission priority classes."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    AVOID = "AVOID"


class TransformationEligibilityOutcome(str, Enum):
    """Stage 4.0 result: whether a credible transformation path exists.

    ``NO_TRANSFORMATION_STRATEGY_WORTH_USING`` is a normal successful outcome,
    not an error. Processing lifecycle is tracked separately by
    ``TransformationExecutionStatus``.
    """

    ELIGIBLE_FOR_TRANSFORMATION = "ELIGIBLE_FOR_TRANSFORMATION"
    ELIGIBLE_WITH_CAUTION = "ELIGIBLE_WITH_CAUTION"
    TRANSFORMATION_REQUIRED = "TRANSFORMATION_REQUIRED"
    NO_TRANSFORMATION_STRATEGY_WORTH_USING = "NO_TRANSFORMATION_STRATEGY_WORTH_USING"
    INSUFFICIENT_TRANSCRIPT_CONFIDENCE = "INSUFFICIENT_TRANSCRIPT_CONFIDENCE"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    UNRESOLVED_POLICY_OR_PROVENANCE_RISK = "UNRESOLVED_POLICY_OR_PROVENANCE_RISK"


class TransformationExecutionStatus(str, Enum):
    """Stage 4.0 processing lifecycle, separate from eligibility outcome."""

    QUEUED = "QUEUED"
    ANALYZING = "ANALYZING"
    COMPLETE = "COMPLETE"
    PROVIDER_DEGRADED = "PROVIDER_DEGRADED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TransformationStrategyType(str, Enum):
    """Bounded closed set of Stage 4.0 strategy directions."""

    CONTEXT_HOOK = "CONTEXT_HOOK"
    HOOK_PLUS_TAKEAWAY = "HOOK_PLUS_TAKEAWAY"
    EXPLANATORY = "EXPLANATORY"
    COMMENTARY = "COMMENTARY"
    ANALYSIS = "ANALYSIS"
    SUMMARY = "SUMMARY"
    COMPARISON = "COMPARISON"
    COUNTERPOINT = "COUNTERPOINT"
    REACTION_FRAMING = "REACTION_FRAMING"
    QUESTION_EXPLANATION_TAKEAWAY = "QUESTION_EXPLANATION_TAKEAWAY"
    CLAIM_CONTEXT_CONCLUSION = "CLAIM_CONTEXT_CONCLUSION"
    DEBATE_CONTEXT = "DEBATE_CONTEXT"
    NEWS_CONTEXT = "NEWS_CONTEXT"
    SOURCE_AS_EVIDENCE = "SOURCE_AS_EVIDENCE"
    SOURCE_LED_MINIMAL = "SOURCE_LED_MINIMAL"


class TransformationIntensity(str, Enum):
    """Least-intrusive-sufficient transformation intensity."""

    MINIMAL = "MINIMAL"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class StrategyDisposition(str, Enum):
    """Whether a discovered strategy direction is recommended or rejected."""

    RECOMMENDED = "RECOMMENDED"
    REJECTED = "REJECTED"


class StrategyOrigin(str, Enum):
    """Whether a strategy direction was discovered deterministically or by a provider."""

    DETERMINISTIC = "DETERMINISTIC"
    PROVIDER = "PROVIDER"


class SubstantiveValueKind(str, Enum):
    """Substantive value a strategy contributes beyond the source moment.

    Presentation-only changes (captions, crop, zoom, borders, music, B-roll,
    speed, filters) are never a substantive value kind and receive zero credit.
    """

    MISSING_CONTEXT = "MISSING_CONTEXT"
    INFERENCE = "INFERENCE"
    EXPLANATION = "EXPLANATION"
    COMPARISON = "COMPARISON"
    COUNTERPOINT = "COUNTERPOINT"
    VERIFICATION_CORRECTION = "VERIFICATION_CORRECTION"
    SYNTHESIS = "SYNTHESIS"
    AUTHORED_THESIS = "AUTHORED_THESIS"
    USEFUL_TAKEAWAY = "USEFUL_TAKEAWAY"
    SOURCE_AS_EVIDENCE = "SOURCE_AS_EVIDENCE"


class SourceMomentStructure(str, Enum):
    """Small deterministic model of the source moment's rhetorical shape."""

    CLAIM = "CLAIM"
    PAYOFF = "PAYOFF"
    QUESTION_ANSWER = "QUESTION_ANSWER"
    EXPLANATION = "EXPLANATION"
    DEBATE = "DEBATE"
    STORY = "STORY"
    JOKE = "JOKE"
    NEWS = "NEWS"
    UNKNOWN = "UNKNOWN"


class PlatformRiskKind(str, Enum):
    """Decision-support platform-risk dimensions (not legal conclusions)."""

    YOUTUBE_REUSED_CONTENT = "YOUTUBE_REUSED_CONTENT"
    YOUTUBE_INAUTHENTIC_REPETITIVE = "YOUTUBE_INAUTHENTIC_REPETITIVE"
    FACEBOOK_UNORIGINAL_CONTENT = "FACEBOOK_UNORIGINAL_CONTENT"
    SPAM_TEMPLATE_HEAVY = "SPAM_TEMPLATE_HEAVY"
    SOURCE_DOMINANCE = "SOURCE_DOMINANCE"


class PlatformRiskLevel(str, Enum):
    """Bounded platform-risk level for one risk dimension."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNDETERMINED = "UNDETERMINED"


class ExternalFactRequirement(str, Enum):
    """Whether a strategy depends on an unverified outside fact."""

    NOT_REQUIRED = "NOT_REQUIRED"
    REQUIRES_EXTERNAL_FACT_VERIFICATION = "REQUIRES_EXTERNAL_FACT_VERIFICATION"


class PlanExecutionStatus(str, Enum):
    """Stage 4.1 processing lifecycle, separate from the planning outcome.

    A truthfully deferred or provider-unavailable plan set is a successful
    semantic result; it is never a source/pipeline failure.
    """

    QUEUED = "QUEUED"
    PLANNING = "PLANNING"
    COMPLETE = "COMPLETE"
    PROVIDER_DEGRADED = "PROVIDER_DEGRADED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PlanSemanticOutcome(str, Enum):
    """Stage 4.1 semantic result of one candidate-scoped planning run."""

    PLANS_GENERATED = "PLANS_GENERATED"
    PLANS_GENERATED_WITH_VERIFICATION_REQUIRED = "PLANS_GENERATED_WITH_VERIFICATION_REQUIRED"
    PLANNING_DEFERRED = "PLANNING_DEFERRED"
    NO_VALID_PLAN_FROM_STRATEGY = "NO_VALID_PLAN_FROM_STRATEGY"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


class PlanStatus(str, Enum):
    """Persisted status of one validated transformation plan."""

    PLAN_GENERATED = "PLAN_GENERATED"
    PLAN_GENERATED_WITH_VERIFICATION_REQUIRED = "PLAN_GENERATED_WITH_VERIFICATION_REQUIRED"


class PlanBlockType(str, Enum):
    """Smallest useful closed Stage 4.1 block vocabulary."""

    SOURCE_EXCERPT = "SOURCE_EXCERPT"
    ORIGINAL_VALUE = "ORIGINAL_VALUE"
    TRANSITION = "TRANSITION"
    TEXTUAL_ANNOTATION = "TEXTUAL_ANNOTATION"
    FACT_VERIFICATION_PLACEHOLDER = "FACT_VERIFICATION_PLACEHOLDER"


class SourceExcerptRole(str, Enum):
    """Structural role of one source excerpt inside a plan."""

    HERO = "HERO"
    HOOK = "HOOK"
    PAYOFF = "PAYOFF"
    SUPPORT = "SUPPORT"


class NarrationNeed(str, Enum):
    """Abstract Stage 4.1 narration requirement level (never auto-required)."""

    NONE = "NONE"
    OPTIONAL = "OPTIONAL"
    RECOMMENDED = "RECOMMENDED"
    REQUIRED = "REQUIRED"


class NarrationPurpose(str, Enum):
    """What a future narration communicates; never how it sounds."""

    CONTEXT = "CONTEXT"
    ANALYSIS = "ANALYSIS"
    COUNTERPOINT = "COUNTERPOINT"
    EXPLANATION = "EXPLANATION"
    TAKEAWAY = "TAKEAWAY"
    HOOK = "HOOK"
    TRANSITION = "TRANSITION"


class DeliveryIntent(str, Enum):
    """How an original-value block is intended to be delivered (abstract only)."""

    ON_SCREEN_TEXT = "ON_SCREEN_TEXT"
    NARRATION = "NARRATION"
    FLEXIBLE = "FLEXIBLE"
