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
    TRANSFORMATION_GOVERNANCE = "TRANSFORMATION_GOVERNANCE"
    VISUAL_COMPOSITION = "VISUAL_COMPOSITION"


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


class GovernanceExecutionStatus(str, Enum):
    """Stage 4.2 processing lifecycle, separate from the semantic outcome.

    A semantic rejection, revision, verification block, all-plans-ineligible
    result, or provider deferral is a successful semantic result, never a
    pipeline/server failure.
    """

    QUEUED = "QUEUED"
    GOVERNING = "GOVERNING"
    COMPLETE = "COMPLETE"
    PROVIDER_DEGRADED = "PROVIDER_DEGRADED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class GovernancePlanStatus(str, Enum):
    """Independent Stage 4.2 result for one current Stage 4.1 plan."""

    APPROVED_FOR_SELECTION = "APPROVED_FOR_SELECTION"
    APPROVED_WITH_CAUTION = "APPROVED_WITH_CAUTION"
    BLOCKED_PENDING_VERIFICATION = "BLOCKED_PENDING_VERIFICATION"
    REVISION_REQUIRED = "REVISION_REQUIRED"
    REJECTED_BY_GOVERNOR = "REJECTED_BY_GOVERNOR"
    GOVERNANCE_DEFERRED = "GOVERNANCE_DEFERRED"


class GovernanceSemanticOutcome(str, Enum):
    """Stage 4.2 candidate-level semantic summary."""

    PLANS_ELIGIBLE_FOR_SELECTION = "PLANS_ELIGIBLE_FOR_SELECTION"
    NO_GOVERNOR_APPROVED_PLAN = "NO_GOVERNOR_APPROVED_PLAN"
    GOVERNANCE_DEFERRED = "GOVERNANCE_DEFERRED"


class GovernanceSeverityClass(str, Enum):
    """Explicit severity classes used by deterministic status precedence."""

    HARD_FAIL = "HARD_FAIL"
    BLOCKING_CONDITION = "BLOCKING_CONDITION"
    REVISION = "REVISION"
    WARNING = "WARNING"
    ADVISORY = "ADVISORY"


class GovernanceLevel(str, Enum):
    """Generic bounded LOW/MODERATE/HIGH/UNKNOWN dimension level."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class GovernanceEvidenceStrength(str, Enum):
    """Generic bounded strength for a value/source-dimension finding."""

    STRONG = "STRONG"
    ADEQUATE = "ADEQUATE"
    WEAK = "WEAK"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class ClaimGroundingState(str, Enum):
    """Deterministic state of a claim's grounding."""

    GROUNDED_IN_SOURCE = "GROUNDED_IN_SOURCE"
    EXTERNAL_REQUIRED_UNRESOLVED = "EXTERNAL_REQUIRED_UNRESOLVED"
    UNSUPPORTED_OR_FABRICATED = "UNSUPPORTED_OR_FABRICATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    # The claim cites real source evidence but the deterministic lexical check
    # cannot confirm support; requires selective semantic review (never a silent
    # approval).
    SUPPORT_UNVERIFIED = "SUPPORT_UNVERIFIED"


class NarrationBurdenFinding(str, Enum):
    """Deterministic narration-burden finding (never a TTS decision)."""

    APPROPRIATE = "APPROPRIATE"
    EXCESSIVE = "EXCESSIVE"
    REDUNDANT = "REDUNDANT"
    POSITION_DAMAGING = "POSITION_DAMAGING"
    NOT_NEEDED = "NOT_NEEDED"
    UNKNOWN = "UNKNOWN"


class SemanticFidelityFinding(str, Enum):
    """Provider semantic-fidelity finding (closed)."""

    PASS = "PASS"
    CONCERN = "CONCERN"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class SubstantiveValueFinding(str, Enum):
    """Provider substantive-value distinctness finding (closed)."""

    DISTINCT = "DISTINCT"
    ADEQUATE = "ADEQUATE"
    REDUNDANT = "REDUNDANT"
    GENERIC = "GENERIC"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class RetentionEffectFinding(str, Enum):
    """Provider source-moment retention-effect finding (closed)."""

    PRESERVED = "PRESERVED"
    MIXED = "MIXED"
    DAMAGED = "DAMAGED"
    UNKNOWN = "UNKNOWN"


class CoherenceFinding(str, Enum):
    """Provider plan coherence/watchability finding (closed)."""

    COHERENT = "COHERENT"
    MIXED = "MIXED"
    INCOHERENT = "INCOHERENT"
    UNKNOWN = "UNKNOWN"


class UnsupportedClaimFinding(str, Enum):
    """Provider unsupported-claim signal (closed)."""

    NONE = "NONE"
    POSSIBLE = "POSSIBLE"
    CLEAR = "CLEAR"
    UNKNOWN = "UNKNOWN"


class GovernanceRemediationPriority(str, Enum):
    """Bounded remediation priority for a revision-required plan."""

    REQUIRED = "REQUIRED"
    RECOMMENDED = "RECOMMENDED"
    ADVISORY = "ADVISORY"


class GovernanceRemediationAction(str, Enum):
    """Closed remediation actions; never a rewritten block or new script."""

    REMOVE_REDUNDANT_INTRO = "REMOVE_REDUNDANT_INTRO"
    SHORTEN_PREAMBLE = "SHORTEN_PREAMBLE"
    MOVE_EXPLANATION_AFTER_HERO = "MOVE_EXPLANATION_AFTER_HERO"
    REMOVE_PARAPHRASE = "REMOVE_PARAPHRASE"
    REDUCE_NARRATION = "REDUCE_NARRATION"
    PRESERVE_PAYOFF = "PRESERVE_PAYOFF"
    RESOLVE_VERIFICATION = "RESOLVE_VERIFICATION"
    REPLACE_GENERIC_TAKEAWAY = "REPLACE_GENERIC_TAKEAWAY"


class GovernanceReasonCode(str, Enum):
    """Closed Stage 4.2 reason codes."""

    PLAN_INTEGRITY_INVALID = "PLAN_INTEGRITY_INVALID"
    NO_SUBSTANTIVE_VALUE = "NO_SUBSTANTIVE_VALUE"
    PRESENTATION_ONLY_TRANSFORMATION = "PRESENTATION_ONLY_TRANSFORMATION"
    REDUNDANT_PARAPHRASE_ONLY = "REDUNDANT_PARAPHRASE_ONLY"
    GENERIC_FILLER_ONLY = "GENERIC_FILLER_ONLY"
    SEMANTIC_DISTORTION = "SEMANTIC_DISTORTION"
    CONTEXT_REVERSAL = "CONTEXT_REVERSAL"
    FALSE_ATTRIBUTION = "FALSE_ATTRIBUTION"
    SARCASM_LITERALIZED = "SARCASM_LITERALIZED"
    SPECULATION_PRESENTED_AS_FACT = "SPECULATION_PRESENTED_AS_FACT"
    UNRELATED_SOURCE_EVIDENCE = "UNRELATED_SOURCE_EVIDENCE"
    FAKE_OR_MISLEADING_HOOK = "FAKE_OR_MISLEADING_HOOK"
    UNSUPPORTED_CRITICAL_CLAIM = "UNSUPPORTED_CRITICAL_CLAIM"
    EXTERNAL_VERIFICATION_REQUIRED = "EXTERNAL_VERIFICATION_REQUIRED"
    SOURCE_MOMENT_SEVERELY_DAMAGED = "SOURCE_MOMENT_SEVERELY_DAMAGED"
    PAYOFF_INTERRUPTED = "PAYOFF_INTERRUPTED"
    EXCESSIVE_PREAMBLE = "EXCESSIVE_PREAMBLE"
    OVER_FRAGMENTED = "OVER_FRAGMENTED"
    NARRATION_EXCESSIVE = "NARRATION_EXCESSIVE"
    NARRATION_REDUNDANT = "NARRATION_REDUNDANT"
    NARRATION_POSITION_DAMAGING = "NARRATION_POSITION_DAMAGING"
    SOURCE_DOMINANCE_CONCERN = "SOURCE_DOMINANCE_CONCERN"
    TEMPLATE_MASS_PRODUCED_FEEL = "TEMPLATE_MASS_PRODUCED_FEEL"
    PLATFORM_REUSE_RISK = "PLATFORM_REUSE_RISK"
    PROVIDER_SEMANTIC_REVIEW_UNAVAILABLE = "PROVIDER_SEMANTIC_REVIEW_UNAVAILABLE"
    PROVIDER_OUTPUT_INVALID = "PROVIDER_OUTPUT_INVALID"
    PLATFORM_EVASION_TACTIC = "PLATFORM_EVASION_TACTIC"
    TTS_IDENTITY_FORBIDDEN = "TTS_IDENTITY_FORBIDDEN"
    SEMANTIC_FIDELITY_CONCERN = "SEMANTIC_FIDELITY_CONCERN"
    PLAN_INCOHERENT = "PLAN_INCOHERENT"
    TRANSFORMATION_OVER_EDIT = "TRANSFORMATION_OVER_EDIT"


class TransformationSelectionStatus(str, Enum):
    """Stage 4.3 deterministic final-plan selection outcome.

    Every outcome is a successful semantic result once the candidate exists and
    the input can be represented truthfully. ``NO_SELECTABLE_PLAN`` and
    ``SELECTION_DEFERRED`` are answers, never pipeline/server failures.
    """

    PLAN_SELECTED = "PLAN_SELECTED"
    PLAN_SELECTED_WITH_CAUTION = "PLAN_SELECTED_WITH_CAUTION"
    NO_SELECTABLE_PLAN = "NO_SELECTABLE_PLAN"
    SELECTION_DEFERRED = "SELECTION_DEFERRED"
    STALE_SELECTION_INPUT = "STALE_SELECTION_INPUT"


class TransformationExecutionReadiness(str, Enum):
    """Live Stage 4.3 execution-readiness state, separate from selection.

    Selection is not render readiness: a plan authored and governed from
    CANDIDATE-grade Stage 3.5 evidence may be selected while still requiring
    final-clip refinement before any Stage 5/6 execution.
    """

    READY_FOR_FINAL_REFINEMENT = "READY_FOR_FINAL_REFINEMENT"
    REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK = "REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK"
    READY_FOR_EXECUTION_PREP = "READY_FOR_EXECUTION_PREP"
    BLOCKED = "BLOCKED"


class RenderContractStatus(str, Enum):
    """Stage 5.0 deterministic execution-preflight outcome.

    Only ``READY_FOR_RENDER_PLANNING`` and ``MATERIALIZATION_REQUIRED`` are
    executable contracts; every other status is a truthful, persisted,
    non-executable preflight result.
    """

    READY_FOR_RENDER_PLANNING = "READY_FOR_RENDER_PLANNING"
    MATERIALIZATION_REQUIRED = "MATERIALIZATION_REQUIRED"
    FINAL_CLIP_REFINEMENT_REQUIRED = "FINAL_CLIP_REFINEMENT_REQUIRED"
    UPSTREAM_REVALIDATION_REQUIRED = "UPSTREAM_REVALIDATION_REQUIRED"
    SOURCE_MEDIA_UNAVAILABLE = "SOURCE_MEDIA_UNAVAILABLE"
    INVALID_SOURCE_BINDING = "INVALID_SOURCE_BINDING"
    BLOCKED = "BLOCKED"


class FinalClipCompatibilityOutcome(str, Enum):
    """Deterministic Stage 5.0 FINAL_CLIP compatibility result."""

    EXACT_MATCH = "EXACT_MATCH"
    COMPATIBLE_NON_MATERIAL_CHANGE = "COMPATIBLE_NON_MATERIAL_CHANGE"
    MATERIAL_SEMANTIC_CHANGE = "MATERIAL_SEMANTIC_CHANGE"
    MATERIAL_TIMING_CHANGE = "MATERIAL_TIMING_CHANGE"
    SOURCE_SPAN_NO_LONGER_VALID = "SOURCE_SPAN_NO_LONGER_VALID"
    UNRESOLVED_COMPATIBILITY = "UNRESOLVED_COMPATIBILITY"


class ExecutionSlotKind(str, Enum):
    """Closed Stage 5.0 execution-slot vocabulary.

    Describes what a future renderer must materialize; it is never a rendered
    artifact, provider decision, voice, or caption file.
    """

    SOURCE_MEDIA = "SOURCE_MEDIA"
    AUTHORED_NARRATION = "AUTHORED_NARRATION"
    AUTHORED_TEXT = "AUTHORED_TEXT"
    TRANSITION = "TRANSITION"
    VERIFICATION_EVIDENCE = "VERIFICATION_EVIDENCE"
