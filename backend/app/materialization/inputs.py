"""DB-independent Stage 6 materialization inputs.

The service assembles these from persisted Stage 4/5 evidence; every pure
deterministic module consumes only these immutable value objects, so the
semantic layers never query the database or a provider directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction


@dataclass(frozen=True)
class WordEvidence:
    """One FINAL_CLIP word timestamp used as authoritative source identity."""

    index: int
    text: str
    start: float
    end: float
    confidence: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class FinalClipEvidence:
    """Accepted FINAL_CLIP refinement evidence (text/words, never rewritten)."""

    refinement_id: str
    priority: str
    quality_level: str
    status: str
    output_fingerprint: str
    text: str
    words: tuple[WordEvidence, ...] = ()
    start: float = 0.0
    end: float = 0.0
    usable: bool = True
    manual_override: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "refinement_id": self.refinement_id,
            "priority": self.priority,
            "quality_level": self.quality_level,
            "status": self.status,
            "output_fingerprint": self.output_fingerprint,
            "text": self.text,
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "usable": self.usable,
            "manual_override": self.manual_override,
            "words": [word.as_dict() for word in self.words],
        }


@dataclass(frozen=True)
class VisualPlanEvidence:
    """Accepted Stage 5.1 composition evidence (canonical ASS, framing)."""

    visual_plan_id: str
    output_fingerprint: str
    plan_ready: bool
    ass_relative_path: str
    ass_sha256: str
    ass_policy_version: str
    caption_source_fingerprint: str
    framing_fingerprint: str
    source_media_fingerprint: str
    safe_zone_key: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "visual_plan_id": self.visual_plan_id,
            "output_fingerprint": self.output_fingerprint,
            "plan_ready": self.plan_ready,
            "ass_relative_path": self.ass_relative_path,
            "ass_sha256": self.ass_sha256,
            "ass_policy_version": self.ass_policy_version,
            "caption_source_fingerprint": self.caption_source_fingerprint,
            "framing_fingerprint": self.framing_fingerprint,
            "source_media_fingerprint": self.source_media_fingerprint,
            "safe_zone_key": self.safe_zone_key,
        }


@dataclass(frozen=True)
class SourceMediaEvidence:
    """Executable source media identity (never an absolute deployment path)."""

    relative_path: str
    content_hash: str
    size_bytes: int
    mtime_ns: int
    duration: float
    frame_rate: Fraction
    display_width: int
    display_height: int
    encoded_width: int
    encoded_height: int
    sample_rate: int
    channels: int

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "duration": round(self.duration, 6),
            "frame_rate": {
                "numerator": self.frame_rate.numerator,
                "denominator": self.frame_rate.denominator,
            },
            "display_width": self.display_width,
            "display_height": self.display_height,
            "encoded_width": self.encoded_width,
            "encoded_height": self.encoded_height,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


@dataclass(frozen=True)
class ProfileContext:
    """Channel production-profile semantic context and narrator identity."""

    profile_id: str | None
    profile_key: str
    target_market: str
    output_language_policy: str
    register_intent: str
    narration_allowed: bool
    tts_provider: str = ""
    tts_model: str = ""
    tts_api_version: str = ""
    tts_voice: str = ""
    tts_style: str = ""
    tts_settings: Mapping[str, object] = field(default_factory=dict)
    supporting_visual_mode: str = "OFF"
    supporting_visual_style: str = "AUTO"
    supporting_visual_defaults: Mapping[str, object] = field(default_factory=dict)

    def tts_identity(self) -> dict[str, object]:
        return {
            "provider": self.tts_provider,
            "model": self.tts_model,
            "api_version": self.tts_api_version,
            "voice": self.tts_voice,
            "style": self.tts_style,
            "settings": dict(self.tts_settings),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "profile_key": self.profile_key,
            "target_market": self.target_market,
            "output_language_policy": self.output_language_policy,
            "register_intent": self.register_intent,
            "narration_allowed": self.narration_allowed,
            "tts": self.tts_identity(),
            "supporting_visual_mode": self.supporting_visual_mode,
            "supporting_visual_style": self.supporting_visual_style,
            "supporting_visual_defaults": dict(self.supporting_visual_defaults),
        }


@dataclass(frozen=True)
class MaterializationInputs:
    """Complete immutable Stage 6 input evidence for one candidate."""

    candidate_id: str
    candidate_key: str
    source_id: str
    priority: str

    selection_id: str | None
    selection_output_fingerprint: str
    selected_plan_fingerprint: str

    plan_id: str
    plan_output_fingerprint: str
    strategy_type: str
    intensity: str
    blocks: tuple[Mapping[str, object], ...]
    hero_block_index: int
    narration: Mapping[str, object]
    verification_dependencies: tuple[Mapping[str, object], ...]
    original_value_kinds: tuple[str, ...]
    required_context: tuple[str, ...]
    preservation_constraints: tuple[str, ...]
    plan_target_context: Mapping[str, object]

    contract_id: str
    contract_status: str
    contract_input_fingerprint: str
    contract_output_fingerprint: str
    execution_slots: tuple[Mapping[str, object], ...]
    contract_source_bindings: tuple[Mapping[str, object], ...]

    final_clip: FinalClipEvidence
    visual_plan: VisualPlanEvidence
    source_media: SourceMediaEvidence
    profile: ProfileContext

    source_dialect: Mapping[str, object] = field(default_factory=dict)
    source_language: str | None = None
    operator_authored_copy: tuple[Mapping[str, object], ...] = ()
    supporting_requested: str = "OFF"

    def plan_block(self, index: int) -> Mapping[str, object] | None:
        for block in self.blocks:
            if block.get("index") == index:
                return block
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_key": self.candidate_key,
            "source_id": self.source_id,
            "priority": self.priority,
            "selection_id": self.selection_id,
            "selection_output_fingerprint": self.selection_output_fingerprint,
            "selected_plan_fingerprint": self.selected_plan_fingerprint,
            "plan_id": self.plan_id,
            "plan_output_fingerprint": self.plan_output_fingerprint,
            "strategy_type": self.strategy_type,
            "intensity": self.intensity,
            "blocks": [dict(block) for block in self.blocks],
            "hero_block_index": self.hero_block_index,
            "narration": dict(self.narration),
            "verification_dependencies": [dict(item) for item in self.verification_dependencies],
            "original_value_kinds": list(self.original_value_kinds),
            "required_context": list(self.required_context),
            "preservation_constraints": list(self.preservation_constraints),
            "plan_target_context": dict(self.plan_target_context),
            "contract_id": self.contract_id,
            "contract_status": self.contract_status,
            "contract_input_fingerprint": self.contract_input_fingerprint,
            "contract_output_fingerprint": self.contract_output_fingerprint,
            "execution_slots": [dict(item) for item in self.execution_slots],
            "contract_source_bindings": [dict(item) for item in self.contract_source_bindings],
            "final_clip": self.final_clip.as_dict(),
            "visual_plan": self.visual_plan.as_dict(),
            "source_media": self.source_media.as_dict(),
            "profile": self.profile.as_dict(),
            "source_dialect": dict(self.source_dialect),
            "source_language": self.source_language,
            "operator_authored_copy": [dict(item) for item in self.operator_authored_copy],
            "supporting_requested": self.supporting_requested,
        }


def sequence_of(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


__all__ = [
    "FinalClipEvidence",
    "MaterializationInputs",
    "ProfileContext",
    "SourceMediaEvidence",
    "VisualPlanEvidence",
    "WordEvidence",
    "mapping_sequence",
    "sequence_of",
]
