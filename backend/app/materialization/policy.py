"""Stage 6 deterministic materialization policy, versions, and bounds.

Stage 6 owns the semantic decisions (what is authored, whether narration is
needed, the timeline and audio recipe). The Stage 5.2 renderer executes a frozen
recipe; it never chooses strategies, writes scripts, or decides originality.

Everything here is code-defined and versioned. No value is a runtime secret, an
absolute path, or a transient quota.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STAGE6_POLICY_VERSION = "stage6-v1"
STAGE6_SCHEMA_VERSION = "stage6-schema-v1"
STAGE6_VALIDATION_VERSION = "stage6-validation-v2"
STAGE6_TIMELINE_SCHEMA_VERSION = "stage6-timeline-v2"
STAGE6_QUALITY_POLICY_VERSION = "stage6-quality-v1"
STAGE6_NATURALNESS_POLICY_VERSION = "stage6-naturalness-v1"
FINGERPRINT_VERSION = "1"

# Hard Stage 6 budgets (ceilings, never default call counts).
MAX_RAW_SEMANTIC_CALLS = 4
MAX_STRONG_TIER_SEMANTIC_CALLS = 2
SEMANTIC_ROUTINE_TIER = "ROUTINE"
SEMANTIC_STRONG_TIER = "STRONG"

# Narration bounds.
MAX_NARRATION_CHUNKS = 3
MAX_NARRATION_SECONDS = 60.0
MAX_RAW_SYNTHESIS_CALLS = 6
NARRATION_DURATION_TOLERANCE_SECONDS = 0.35

# Bounded narration mix gain. A channel profile may request a nonzero narration
# gain (via ``tts_settings['narration_gain_db']``); it is clamped to this
# conservative range so narration can never be boosted toward clipping or muted
# out. The applied gain participates in the audio recipe and output fingerprint.
MIN_NARRATION_GAIN_DB = -12.0
MAX_NARRATION_GAIN_DB = 6.0
DEFAULT_NARRATION_GAIN_DB = 0.0
NARRATION_GAIN_SETTING_KEY = "narration_gain_db"

# Delivery bounds.
MAX_CLIP_SECONDS = 180.0

# One repair per materialization generation, allocated by CAS.
MAX_REPAIR_COUNT = 1

# Mastering target (versioned). Conservative dynamics preservation.
TARGET_INTEGRATED_LUFS = -16.0
TRUE_PEAK_CEILING_DBTP = -1.5
TARGET_SAMPLE_RATE = 48_000
TARGET_CHANNELS = 2
MASTERING_LOUDNESS_TOLERANCE_LU = 1.0

# Conservative optional non-hero boundary-silence trimming.
MAX_BOUNDARY_TRIM_SECONDS = 0.6
MIN_RETAINED_BREATHING_ROOM_SECONDS = 0.2

# Supporting-visual bounds.
MAX_SUPPORTING_ASSETS_PER_MATERIALIZATION = 1

# Default neutral profile semantic context.
DEFAULT_TARGET_MARKET = "UNSPECIFIED"
DEFAULT_OUTPUT_LANGUAGE_POLICY = "SOURCE_LANGUAGE"
DEFAULT_REGISTER_INTENT = "SOURCE_COMPATIBLE"

# Reason codes surfaced by Stage 6 assessment and lifecycle.
REASON_UPSTREAM_REVALIDATION_REQUIRED = "UPSTREAM_REVALIDATION_REQUIRED"
REASON_PROFILE_CONTEXT_MISMATCH = "PROFILE_CONTEXT_MISMATCH"
REASON_NO_USABLE_PLAN = "NO_USABLE_PLAN"
REASON_NO_USABLE_FINAL_CLIP = "NO_USABLE_FINAL_CLIP"
REASON_CONTRACT_NOT_EXECUTABLE = "CONTRACT_NOT_EXECUTABLE"
REASON_VISUAL_PLAN_NOT_READY = "VISUAL_PLAN_NOT_READY"
REASON_MISSING_REQUIRED_VALUE = "MISSING_REQUIRED_VALUE"
REASON_MISSING_REQUIRED_NARRATION = "MISSING_REQUIRED_NARRATION"
REASON_TTS_UNAVAILABLE = "TTS_UNAVAILABLE"
REASON_UNSUPPORTED_ESSENTIAL_CLAIM = "UNSUPPORTED_ESSENTIAL_CLAIM"
REASON_PARAPHRASE_ONLY = "PARAPHRASE_ONLY"
REASON_GENERIC_FILLER = "GENERIC_FILLER"
REASON_PRESENTATION_ONLY = "PRESENTATION_ONLY"
REASON_INVENTED_NUMBER = "INVENTED_NUMBER"
REASON_PROTECTED_TOKEN_MUTATED = "PROTECTED_TOKEN_MUTATED"
REASON_SCRIPT_AUDIO_MISMATCH = "SCRIPT_AUDIO_MISMATCH"
REASON_HERO_DAMAGED = "HERO_DAMAGED"
REASON_UNREADABLE_REQUIRED_COPY = "UNREADABLE_REQUIRED_COPY"
REASON_UNSUITABLE_ASSET_RIGHTS = "UNSUITABLE_ASSET_RIGHTS"
REASON_OVEREDIT_RISK = "OVEREDIT_RISK"
REASON_STALE_CRITICAL_EVIDENCE = "STALE_CRITICAL_EVIDENCE"
REASON_NONFINITE_MEASUREMENT = "NONFINITE_MEASUREMENT"
REASON_MISSING_RENDER_EVIDENCE = "MISSING_RENDER_EVIDENCE"
REASON_MISSING_MEASUREMENT = "MISSING_MEASUREMENT"
REASON_LOUDNESS_OUT_OF_POLICY = "LOUDNESS_OUT_OF_POLICY"
REASON_TRUE_PEAK_EXCEEDED = "TRUE_PEAK_EXCEEDED"
REASON_GROUNDED_INFERENCE_REQUIRES_REVIEW = "GROUNDED_INFERENCE_REQUIRES_REVIEW"
REASON_INVALID_GROUNDING_REFERENCE = "INVALID_GROUNDING_REFERENCE"
REASON_UNRELATED_COPY = "UNRELATED_COPY"
REASON_MISSING_PROVENANCE = "MISSING_PROVENANCE"
REASON_MISSING_ATTRIBUTION = "MISSING_ATTRIBUTION"
REASON_INVALID_ASSET_HASH = "INVALID_ASSET_HASH"
REASON_MANUAL_TRANSCRIPT_REVIEW = "MANUAL_TRANSCRIPT_REVIEW"
REASON_REPAIR_SPENT = "REPAIR_SPENT"

# Protected-token preservation is delegated to the shared Stage 2.5 / Stage 2.7
# primitives; Stage 6 only asserts the accepted copy preserves them verbatim.
PROTECTED_TOKEN_POLICY_VERSION = "stage2.7.1-protected-tokens-v1"


@dataclass(frozen=True)
class Stage6Config:
    """Versioned deterministic Stage 6 bounds (never mutates frozen stages)."""

    policy_version: str = STAGE6_POLICY_VERSION
    schema_version: str = STAGE6_SCHEMA_VERSION
    validation_version: str = STAGE6_VALIDATION_VERSION
    timeline_schema_version: str = STAGE6_TIMELINE_SCHEMA_VERSION
    quality_policy_version: str = STAGE6_QUALITY_POLICY_VERSION
    naturalness_policy_version: str = STAGE6_NATURALNESS_POLICY_VERSION

    max_raw_semantic_calls: int = MAX_RAW_SEMANTIC_CALLS
    max_strong_tier_semantic_calls: int = MAX_STRONG_TIER_SEMANTIC_CALLS
    max_narration_chunks: int = MAX_NARRATION_CHUNKS
    max_narration_seconds: float = MAX_NARRATION_SECONDS
    max_raw_synthesis_calls: int = MAX_RAW_SYNTHESIS_CALLS
    min_narration_gain_db: float = MIN_NARRATION_GAIN_DB
    max_narration_gain_db: float = MAX_NARRATION_GAIN_DB
    default_narration_gain_db: float = DEFAULT_NARRATION_GAIN_DB
    max_clip_seconds: float = MAX_CLIP_SECONDS
    max_repair_count: int = MAX_REPAIR_COUNT
    max_boundary_trim_seconds: float = MAX_BOUNDARY_TRIM_SECONDS
    min_retained_breathing_room_seconds: float = MIN_RETAINED_BREATHING_ROOM_SECONDS
    narration_duration_tolerance_seconds: float = NARRATION_DURATION_TOLERANCE_SECONDS
    paraphrase_overlap_threshold: float = 0.7
    max_evidence_chars_per_target: int = 600

    target_integrated_lufs: float = TARGET_INTEGRATED_LUFS
    true_peak_ceiling_dbtp: float = TRUE_PEAK_CEILING_DBTP
    target_sample_rate: int = TARGET_SAMPLE_RATE
    target_channels: int = TARGET_CHANNELS
    mastering_loudness_tolerance_lu: float = MASTERING_LOUDNESS_TOLERANCE_LU

    # Ordering: the least-intrusive sufficient transformation wins ties.
    strategy_intensity_preference: tuple[str, ...] = ("MINIMAL", "MODERATE", "STRONG")


DEFAULT_CONFIG = Stage6Config()


def narration_gain_db(settings: object, config: Stage6Config) -> float:
    """Resolve the bounded narration mix gain from a profile's TTS settings.

    Non-numeric or missing values fall back to the neutral default; a numeric
    value is clamped to the configured conservative range, so the applied gain is
    always finite and bounded.
    """

    value = settings.get(NARRATION_GAIN_SETTING_KEY) if isinstance(settings, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return config.default_narration_gain_db
    gain = float(value)
    return max(config.min_narration_gain_db, min(config.max_narration_gain_db, gain))


@dataclass
class CallLedger:
    """Durable, replay-safe raw-call budget ledger for one generation.

    Every raw provider attempt carries a unique ``attempt_id`` and is reserved
    *before* invocation, so a crash mid-request can never under-count the budget
    and a replayed checkpoint can never double-count it. ``semantic``/``strong``
    track the semantic ceilings and ``synthesis`` the narration TTS ceiling; the
    counters are always derived from the distinct reserved attempt identities, so
    merging two ledgers unions attempts rather than adding or maximising partial
    counters.
    """

    semantic: int = 0
    strong: int = 0
    synthesis: int = 0
    ledger_entries: list[dict[str, object]] = field(default_factory=list)

    def attempt_ids(self) -> set[str]:
        return {
            str(entry["attempt_id"])
            for entry in self.ledger_entries
            if isinstance(entry.get("attempt_id"), str)
        }

    def reserve(self, attempt_id: str, *, kind: str, tier: str | None = None) -> bool:
        """Reserve one raw attempt under its unique identity.

        Returns ``True`` when the identity is new (the caller must invoke the
        provider exactly once). Returns ``False`` when the identity is already
        recorded — a replayed checkpoint or an uncertain/crashed invocation — in
        which case the budget is already consumed and the provider must never be
        called again.
        """

        if attempt_id in self.attempt_ids():
            return False
        entry: dict[str, object] = {"attempt_id": attempt_id, "kind": kind}
        if tier is not None:
            entry["tier"] = tier
        self.ledger_entries.append(entry)
        self._recount()
        return True

    def merge(self, other: "CallLedger") -> "CallLedger":
        """Union two ledgers by attempt identity and re-derive the counters.

        Legacy entries without an ``attempt_id`` (from before identity accounting)
        are retained under a positional key so no historical attempt is lost.
        """

        merged = CallLedger()
        seen: dict[str, dict[str, object]] = {}
        for index, entry in enumerate([*self.ledger_entries, *other.ledger_entries]):
            identity = entry.get("attempt_id")
            if not isinstance(identity, str):
                identity = f"legacy:{index}:{entry.get('kind')}:{entry.get('tier')}"
            seen[identity] = dict(entry)
        merged.ledger_entries = list(seen.values())
        merged._recount()
        return merged

    def _recount(self) -> None:
        self.semantic = sum(
            1 for entry in self.ledger_entries if entry.get("kind") in {"semantic", "critique"}
        )
        self.strong = sum(
            1 for entry in self.ledger_entries if entry.get("tier") == SEMANTIC_STRONG_TIER
        )
        self.synthesis = sum(1 for entry in self.ledger_entries if entry.get("kind") == "synthesis")

    def as_dict(self) -> dict[str, object]:
        return {
            "semantic": self.semantic,
            "strong": self.strong,
            "synthesis": self.synthesis,
            "entries": [dict(entry) for entry in self.ledger_entries],
        }

    @staticmethod
    def from_dict(data: object) -> "CallLedger":
        if not isinstance(data, dict):
            return CallLedger()
        entries = data.get("entries")
        ledger = CallLedger(
            ledger_entries=[dict(item) for item in entries] if isinstance(entries, list) else [],
        )
        ledger._recount()
        return ledger


__all__ = [
    "DEFAULT_CONFIG",
    "CallLedger",
    "Stage6Config",
    "STAGE6_POLICY_VERSION",
    "STAGE6_SCHEMA_VERSION",
    "STAGE6_VALIDATION_VERSION",
    "narration_gain_db",
]
