"""Frozen value objects for Stage 5.0 execution preflight and render contract.

Everything here is immutable and JSON-serializable. These types never carry a
rendered artifact, TTS voice/provider/model, caption file, crop path, or
publishing metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClipWord:
    """One FINAL_CLIP word with source-time bounds and an immutable index."""

    index: int
    text: str
    start: float
    end: float
    probability: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "probability": self.probability,
        }


@dataclass(frozen=True)
class FinalClipEvidence:
    """Typed FINAL_CLIP transcript evidence used for compatibility binding."""

    refinement_id: str
    priority: str
    quality_level: str
    status: str
    final_transcript: str
    refined_start: float | None
    refined_end: float | None
    words: tuple[ClipWord, ...] = ()
    unresolved_spans: tuple[Mapping[str, object], ...] = ()
    entity_evidence: tuple[Mapping[str, object], ...] = ()
    code_switch_evidence: Mapping[str, object] = field(default_factory=dict)
    dialect_profile: str | None = None
    dialect_confidence: float = 0.0
    output_fingerprint: str = ""

    @property
    def word_coverage_sufficient(self) -> bool:
        return bool(self.words) and self.refined_start is not None and self.refined_end is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "refinement_id": self.refinement_id,
            "priority": self.priority,
            "quality_level": self.quality_level,
            "status": self.status,
            "output_fingerprint": self.output_fingerprint,
            "word_coverage_sufficient": self.word_coverage_sufficient,
        }


@dataclass(frozen=True)
class BlockCompatibility:
    """Deterministic per-SOURCE_EXCERPT compatibility verdict."""

    block_index: int
    outcome: str
    reason_codes: tuple[str, ...] = ()
    timing_drift_seconds: float | None = None
    coverage: float | None = None
    added_tokens: tuple[str, ...] = ()
    removed_tokens: tuple[str, ...] = ()
    recovered_code_switch_tokens: tuple[str, ...] = ()
    alignment: Mapping[str, object] = field(default_factory=dict)
    structural_valid: bool = True
    rebound_start: float | None = None
    rebound_end: float | None = None
    rebound_word_start: int | None = None
    rebound_word_end: int | None = None
    rebound_text: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "outcome": self.outcome,
            "reason_codes": list(self.reason_codes),
            "timing_drift_seconds": self.timing_drift_seconds,
            "coverage": self.coverage,
            "added_tokens": list(self.added_tokens),
            "removed_tokens": list(self.removed_tokens),
            "recovered_code_switch_tokens": list(self.recovered_code_switch_tokens),
            "alignment": dict(self.alignment),
            "structural_valid": self.structural_valid,
            "rebound_start": self.rebound_start,
            "rebound_end": self.rebound_end,
            "rebound_word_start": self.rebound_word_start,
            "rebound_word_end": self.rebound_word_end,
            "rebound_text": self.rebound_text,
        }


@dataclass(frozen=True)
class CompatibilityResult:
    """Complete deterministic FINAL_CLIP compatibility evaluation."""

    outcome: str
    exact_match: bool
    planning_refinement_id: str
    planning_refinement_priority: str
    planning_refinement_quality_level: str
    planning_output_fingerprint: str
    final_refinement_id: str
    final_output_fingerprint: str
    caption_source_fingerprint: str
    per_block: tuple[BlockCompatibility, ...] = ()
    reason_codes: tuple[str, ...] = ()
    unresolved_spans: tuple[Mapping[str, object], ...] = ()
    recovered_code_switch_tokens: tuple[str, ...] = ()
    compatibility_policy_version: str = ""

    def verdicts(self) -> list[dict[str, object]]:
        return [verdict.as_dict() for verdict in self.per_block]


@dataclass(frozen=True)
class BoundSourceSpan:
    """A plan SOURCE_EXCERPT safely bound to current FINAL_CLIP timings."""

    block_index: int
    planning_source_start: float | None
    planning_source_end: float | None
    planning_word_start: int | None
    planning_word_end: int | None
    planning_text: str
    final_clip_start: float | None
    final_clip_end: float | None
    final_clip_word_start: int | None
    final_clip_word_end: int | None
    final_clip_text: str
    source_role: str | None
    is_hero: bool
    compatibility_outcome: str
    preservation_constraints: tuple[str, ...] = ()
    rebind_valid: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "planning_source_start": self.planning_source_start,
            "planning_source_end": self.planning_source_end,
            "planning_word_start": self.planning_word_start,
            "planning_word_end": self.planning_word_end,
            "planning_text": self.planning_text,
            "final_clip_start": self.final_clip_start,
            "final_clip_end": self.final_clip_end,
            "final_clip_word_start": self.final_clip_word_start,
            "final_clip_word_end": self.final_clip_word_end,
            "final_clip_text": self.final_clip_text,
            "source_role": self.source_role,
            "is_hero": self.is_hero,
            "compatibility_outcome": self.compatibility_outcome,
            "preservation_constraints": list(self.preservation_constraints),
            "rebind_valid": self.rebind_valid,
        }


@dataclass(frozen=True)
class ContractBlock:
    """One ordered execution-contract block preserving Stage 4.1 order."""

    block_index: int
    block_type: str
    purpose: str
    placement: str
    interrupts_source: bool
    estimated_duration_seconds: float
    timeline_start: float
    timeline_end: float
    timeline_authoritative: bool
    preservation_constraints: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    slot_kind: str
    source_binding: Mapping[str, object] | None = None
    materialization: Mapping[str, object] | None = None
    verification: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "block_index": self.block_index,
            "block_type": self.block_type,
            "purpose": self.purpose,
            "placement": self.placement,
            "interrupts_source": self.interrupts_source,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "timeline": {
                "start": self.timeline_start,
                "end": self.timeline_end,
                "authoritative": self.timeline_authoritative,
            },
            "preservation_constraints": list(self.preservation_constraints),
            "dependency_ids": list(self.dependency_ids),
            "slot_kind": self.slot_kind,
        }
        if self.source_binding is not None:
            payload["source_binding"] = dict(self.source_binding)
        if self.materialization is not None:
            payload["materialization"] = dict(self.materialization)
        if self.verification is not None:
            payload["verification"] = dict(self.verification)
        return payload


@dataclass(frozen=True)
class MaterializationSlot:
    """One ordered slot a future materialization stage must satisfy."""

    slot_id: str
    slot_kind: str
    block_index: int | None
    block_type: str
    required: bool
    reason_code: str
    payload: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "slot_kind": self.slot_kind,
            "block_index": self.block_index,
            "block_type": self.block_type,
            "required": self.required,
            "reason_code": self.reason_code,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class SourceMediaFacts:
    """Bounded read-only ffprobe facts for a managed source artifact."""

    duration_seconds: float
    video_codec: str
    width: int
    height: int
    frames_per_second: float
    audio_codec: str | None
    audio_sample_rate: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "duration_seconds": self.duration_seconds,
            "video_codec": self.video_codec,
            "width": self.width,
            "height": self.height,
            "frames_per_second": self.frames_per_second,
            "audio_codec": self.audio_codec,
            "audio_sample_rate": self.audio_sample_rate,
            "video_streams": 1,
            "audio_streams": 1 if self.audio_codec is not None else 0,
        }


@dataclass(frozen=True)
class SourceMediaIdentity:
    """Cheap identity for a managed source file (never a full-file hash)."""

    source_id: str
    content_hash: str
    relative_path: str
    size_bytes: int
    mtime_ns: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "content_hash": self.content_hash,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True)
class ContractDraft:
    """Fully assembled contract ready for persistence (or reuse)."""

    status: str
    contract_ready: bool
    reason_codes: tuple[str, ...]
    compatibility_outcome: str | None
    compatibility_evidence: Mapping[str, object]
    selection_id: str | None
    selected_plan_id: str | None
    final_refinement_id: str | None
    source_media_identity: Mapping[str, object]
    source_probe: Mapping[str, object]
    readiness: Mapping[str, object]
    contract_payload: Mapping[str, object]
    selected_plan_fingerprint: str
    planning_refinement_output_fingerprint: str
    final_refinement_output_fingerprint: str
    caption_source_fingerprint: str
    source_media_fingerprint: str
    probe_fingerprint: str
    input_fingerprint: str
    output_fingerprint: str
    profile_key: str
    profile_version: str
    metrics: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "contract_ready": self.contract_ready,
            "reason_codes": list(self.reason_codes),
            "compatibility_outcome": self.compatibility_outcome,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
        }


@dataclass(frozen=True)
class RenderContractPreflight:
    """The deterministic preflight result before/without persistence."""

    status: str
    reason_codes: tuple[str, ...]
    draft: ContractDraft
    source_probe_reused: bool = False

    @property
    def executable(self) -> bool:
        return self.status in {"READY_FOR_RENDER_PLANNING", "MATERIALIZATION_REQUIRED"}


__all__ = [
    "BlockCompatibility",
    "BoundSourceSpan",
    "ClipWord",
    "CompatibilityResult",
    "ContractBlock",
    "ContractDraft",
    "FinalClipEvidence",
    "MaterializationSlot",
    "RenderContractPreflight",
    "SourceMediaFacts",
    "SourceMediaIdentity",
]
