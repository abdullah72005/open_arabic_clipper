"""Immutable Stage 6 materialization value objects.

Everything here is immutable and JSON-serializable. The semantic layers consume
these types only: they never discover candidates, query providers outside the
declared adapters, or mutate database rows. Source times are source-local
seconds; output times are authored-timeline seconds at unit source speed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction

# --- Authored content ------------------------------------------------------


@dataclass(frozen=True)
class GroundingRef:
    """One resolvable grounding reference for authored copy.

    ``raw`` preserves the exact declaration a Stage 4 provider emitted even when
    it is invalid. Index fields for ``WORD_RANGE`` are Stage 4.1 *planning* word
    indexes; they are never FINAL_CLIP word indexes.
    """

    kind: str
    block_index: int | None = None
    source_start: float | None = None
    source_end: float | None = None
    claim_id: str | None = None
    word_start_index: int | None = None
    word_end_index: int | None = None
    quote: str | None = None
    raw: str | None = None
    invalid_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "block_index": self.block_index,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "claim_id": self.claim_id,
            "word_start_index": self.word_start_index,
            "word_end_index": self.word_end_index,
            "quote": self.quote,
            "raw": self.raw,
            "invalid_reason": self.invalid_reason,
        }


@dataclass(frozen=True)
class AuthoredBlock:
    """One materialized authored block (final copy plus evidence)."""

    block_index: int
    block_type: str
    slot_id: str
    substantive_value_kind: str
    copy: str
    adds_beyond_source: str
    claim_class: str
    grounding_refs: tuple[GroundingRef, ...] = ()
    delivery: str = "TEXT"
    layout_constraints: Mapping[str, object] = field(default_factory=dict)
    duration_constraints: Mapping[str, object] = field(default_factory=dict)
    provenance: str = "AUTOMATED"
    validation_evidence: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "block_type": self.block_type,
            "slot_id": self.slot_id,
            "substantive_value_kind": self.substantive_value_kind,
            "copy": self.copy,
            "adds_beyond_source": self.adds_beyond_source,
            "claim_class": self.claim_class,
            "grounding_refs": [ref.as_dict() for ref in self.grounding_refs],
            "delivery": self.delivery,
            "layout_constraints": dict(self.layout_constraints),
            "duration_constraints": dict(self.duration_constraints),
            "provenance": self.provenance,
            "validation_evidence": dict(self.validation_evidence),
        }


@dataclass(frozen=True)
class ClaimLedgerEntry:
    """One claim-ledger record for an authored block."""

    claim_id: str
    block_index: int
    claim_class: str
    text: str
    grounding_refs: tuple[GroundingRef, ...] = ()
    verification_dependency_id: str | None = None
    resolved: bool = True
    essential: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "block_index": self.block_index,
            "claim_class": self.claim_class,
            "text": self.text,
            "grounding_refs": [ref.as_dict() for ref in self.grounding_refs],
            "verification_dependency_id": self.verification_dependency_id,
            "resolved": self.resolved,
            "essential": self.essential,
        }


@dataclass(frozen=True)
class SlotResolution:
    """Explicit decision for one slot/block requirement."""

    slot_id: str
    block_index: int | None
    slot_kind: str
    resolution: str
    reason_code: str | None = None
    required: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "block_index": self.block_index,
            "slot_kind": self.slot_kind,
            "resolution": self.resolution,
            "reason_code": self.reason_code,
            "required": self.required,
        }


# --- Narration -------------------------------------------------------------


@dataclass(frozen=True)
class NarrationSynthesisRequest:
    """Provider-neutral synthesis request (spoken wording + delivery metadata)."""

    chunk_id: str
    block_index: int
    text: str
    language_hint: str | None = None
    target_duration_seconds: float | None = None
    provider: str = ""
    model: str = ""
    api_version: str = ""
    voice: str = ""
    style: str = ""
    settings: Mapping[str, object] = field(default_factory=dict)

    def identity(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_version": self.api_version,
            "voice": self.voice,
            "style": self.style,
            "settings": dict(self.settings),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "block_index": self.block_index,
            "text": self.text,
            "language_hint": self.language_hint,
            "target_duration_seconds": self.target_duration_seconds,
            "identity": self.identity(),
        }


@dataclass(frozen=True)
class NarrationAudio:
    """Managed lossless narration audio with actual format/duration/hash."""

    chunk_id: str
    block_index: int
    text: str
    relative_path: str
    sha256: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    container: str
    provider_identity: Mapping[str, object] = field(default_factory=dict)
    validation: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "block_index": self.block_index,
            "text": self.text,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "duration_seconds": round(self.duration_seconds, 6),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "container": self.container,
            "provider_identity": dict(self.provider_identity),
            "validation": dict(self.validation),
        }


# --- Supporting visuals ----------------------------------------------------


@dataclass(frozen=True)
class SupportingAssetSelection:
    """One supporting-visual selection decision (or explicit decline)."""

    asset_id: str | None
    block_index: int | None
    style: str
    resolution: str
    reason_code: str | None = None
    relative_path: str | None = None
    sha256: str | None = None
    duration_seconds: float | None = None
    rights: str = ""
    required_attribution: str = ""
    placement: str = "AUTHORED_SECTION_BACKGROUND"

    def as_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "block_index": self.block_index,
            "style": self.style,
            "resolution": self.resolution,
            "reason_code": self.reason_code,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "duration_seconds": self.duration_seconds,
            "rights": self.rights,
            "required_attribution": self.required_attribution,
            "placement": self.placement,
        }


# --- Timeline and audio recipe ---------------------------------------------


@dataclass(frozen=True)
class TimelineSegment:
    """One ordered authored-timeline segment bound to a plan block."""

    segment_id: str
    plan_block_index: int
    kind: str
    output_start: float
    output_end: float
    source_start: float | None = None
    source_end: float | None = None
    word_start_index: int | None = None
    word_end_index: int | None = None
    text: str = ""
    narration_chunk_id: str | None = None
    asset_id: str | None = None
    transition: str = "CUT"
    is_hero: bool = False
    delivery: str = ""
    delivery_rationale: str = ""

    @property
    def output_duration(self) -> float:
        return self.output_end - self.output_start

    def as_dict(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "plan_block_index": self.plan_block_index,
            "kind": self.kind,
            "output_start": round(self.output_start, 6),
            "output_end": round(self.output_end, 6),
            "source_start": self.source_start,
            "source_end": self.source_end,
            "word_start_index": self.word_start_index,
            "word_end_index": self.word_end_index,
            "text": self.text,
            "narration_chunk_id": self.narration_chunk_id,
            "asset_id": self.asset_id,
            "transition": self.transition,
            "is_hero": self.is_hero,
            "delivery": self.delivery,
            "delivery_rationale": self.delivery_rationale,
        }


@dataclass(frozen=True)
class AuthoredTimeline:
    """The immutable versioned final authored timeline."""

    segments: tuple[TimelineSegment, ...]
    frame_rate: Fraction
    sample_rate: int
    output_duration: float
    output_frame_count: int
    output_sample_count: int
    source_caption_asset: str = ""
    authored_caption_asset: str = ""
    slot_fulfillment: tuple[SlotResolution, ...] = ()
    schema_version: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frame_rate": {
                "numerator": self.frame_rate.numerator,
                "denominator": self.frame_rate.denominator,
                "value": float(self.frame_rate),
            },
            "sample_rate": self.sample_rate,
            "output_duration": round(self.output_duration, 6),
            "output_frame_count": self.output_frame_count,
            "output_sample_count": self.output_sample_count,
            "source_caption_asset": self.source_caption_asset,
            "authored_caption_asset": self.authored_caption_asset,
            "segments": [segment.as_dict() for segment in self.segments],
            "slot_fulfillment": [item.as_dict() for item in self.slot_fulfillment],
        }


@dataclass(frozen=True)
class AudioEntry:
    """One authored-audio placement in the final mix recipe."""

    entry_id: str
    kind: str
    output_start: float
    output_end: float
    gain_db: float = 0.0
    source_start: float | None = None
    source_end: float | None = None
    narration_chunk_id: str | None = None
    ducking_db: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "output_start": round(self.output_start, 6),
            "output_end": round(self.output_end, 6),
            "gain_db": round(self.gain_db, 6),
            "source_start": self.source_start,
            "source_end": self.source_end,
            "narration_chunk_id": self.narration_chunk_id,
            "ducking_db": self.ducking_db,
        }


@dataclass(frozen=True)
class AudioRecipe:
    """The immutable final audio/mix/mastering recipe (executed by Stage 5.2)."""

    entries: tuple[AudioEntry, ...]
    ducking_regions: tuple[Mapping[str, object], ...] = ()
    target_integrated_lufs: float = -16.0
    true_peak_ceiling_dbtp: float = -1.5
    sample_rate: int = 48_000
    channels: int = 2
    supporting_audio_muted: bool = True
    recipe_version: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "recipe_version": self.recipe_version,
            "entries": [entry.as_dict() for entry in self.entries],
            "ducking_regions": [dict(item) for item in self.ducking_regions],
            "target_integrated_lufs": self.target_integrated_lufs,
            "true_peak_ceiling_dbtp": self.true_peak_ceiling_dbtp,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "supporting_audio_muted": self.supporting_audio_muted,
        }


# --- Quality and assessment ------------------------------------------------


@dataclass(frozen=True)
class QualityFinding:
    """One categorical Stage 6 quality finding (never an aggregate score)."""

    dimension: str
    level: str
    reason_codes: tuple[str, ...] = ()
    evidence: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "level": self.level,
            "reason_codes": list(self.reason_codes),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class MaterializationAssessment:
    """Deterministic Stage 6 assessment of one materialization generation."""

    outcome: str
    human_review_required: bool
    publication_ready: bool
    findings: tuple[QualityFinding, ...]
    reason_codes: tuple[str, ...]
    hard_gates: tuple[Mapping[str, object], ...] = ()
    policy_version: str = ""
    metrics: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "human_review_required": self.human_review_required,
            "publication_ready": self.publication_ready,
            "policy_version": self.policy_version,
            "reason_codes": list(self.reason_codes),
            "findings": [finding.as_dict() for finding in self.findings],
            "hard_gates": [dict(item) for item in self.hard_gates],
            "metrics": dict(self.metrics),
        }


__all__ = [
    "AudioEntry",
    "AudioRecipe",
    "AuthoredBlock",
    "AuthoredTimeline",
    "ClaimLedgerEntry",
    "GroundingRef",
    "MaterializationAssessment",
    "NarrationAudio",
    "NarrationSynthesisRequest",
    "QualityFinding",
    "SlotResolution",
    "SupportingAssetSelection",
    "TimelineSegment",
]
