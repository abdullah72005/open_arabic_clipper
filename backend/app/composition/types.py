"""Frozen Stage 5.1 value objects for the visual-composition plan.

Everything here is immutable and JSON-serializable. These types never carry a
rendered artifact, TTS voice/provider/model, final codec setting, or publishing
metadata. All source/crop/caption times are source-local seconds; block order and
placement hints are carried, but no global timeline is ever frozen.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DisplayGeometry:
    """Display-oriented geometry for one source after rotation is applied."""

    encoded_width: int
    encoded_height: int
    rotation_degrees: int
    display_width: int
    display_height: int
    pixel_aspect_ratio: float = 1.0
    square_pixels_applied: bool = True
    exotic_pixel_aspect: bool = False

    @property
    def display_aspect(self) -> float:
        return self.display_width / self.display_height

    def as_dict(self) -> dict[str, object]:
        return {
            "encoded_width": self.encoded_width,
            "encoded_height": self.encoded_height,
            "rotation_degrees": self.rotation_degrees,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "display_aspect": round(self.display_aspect, 6),
            "pixel_aspect_ratio": self.pixel_aspect_ratio,
            "square_pixels_applied": self.square_pixels_applied,
            "exotic_pixel_aspect": self.exotic_pixel_aspect,
        }


@dataclass(frozen=True)
class FaceDetection:
    """One anonymous face box in display-normalized coordinates.

    Boxes are stored normalized by display width/height with origin top-left,
    x right, y down. No identity, embedding, or biometric attribute exists.
    """

    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def as_dict(self) -> dict[str, object]:
        return {
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "w": round(self.w, 6),
            "h": round(self.h, 6),
            "score": round(self.score, 6),
        }


@dataclass(frozen=True)
class TrackSample:
    """One anonymous face sample in a track (box geometry only)."""

    source_time: float
    cx: float
    cy: float
    w: float
    h: float
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "t": round(self.source_time, 4),
            "cx": round(self.cx, 6),
            "cy": round(self.cy, 6),
            "w": round(self.w, 6),
            "h": round(self.h, 6),
            "s": round(self.score, 4),
        }


@dataclass(frozen=True)
class FaceTrack:
    """An anonymous per-scene face track. Never crosses a hard cut."""

    track_id: int
    scene_index: int
    first_time: float
    last_time: float
    samples: tuple[TrackSample, ...]
    persistence: float
    mean_score: float
    mean_face_height_fraction: float
    stable: bool
    decimated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "scene_index": self.scene_index,
            "first_time": round(self.first_time, 4),
            "last_time": round(self.last_time, 4),
            "sample_count": len(self.samples),
            "persistence": round(self.persistence, 4),
            "mean_score": round(self.mean_score, 6),
            "mean_face_height_fraction": round(self.mean_face_height_fraction, 6),
            "stable": self.stable,
            "decimated": self.decimated,
            "samples": [sample.as_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class CropKeyframe:
    """One compact crop keyframe in display-normalized coordinates."""

    source_time: float
    mode: str
    center_x: float
    center_y: float
    height_fraction: float
    confidence: float
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "t": round(self.source_time, 4),
            "mode": self.mode,
            "cx": round(self.center_x, 6),
            "cy": round(self.center_y, 6),
            "height_fraction": round(self.height_fraction, 6),
            "confidence": round(self.confidence, 6),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class FramingDecision:
    """Deterministic per-scene framing mode plus its evidence."""

    mode: str
    evidence: tuple[str, ...] = ()
    fallback_parameters: Mapping[str, object] = field(default_factory=dict)
    protected: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "evidence": list(self.evidence),
            "fallback_parameters": dict(self.fallback_parameters),
            "protected": self.protected,
        }


@dataclass(frozen=True)
class Scene:
    """One analyzed source scene spanning a bounded selected region."""

    scene_index: int
    block_index: int
    source_start: float
    source_end: float
    cut_start: bool
    framing: FramingDecision
    tracks: tuple[FaceTrack, ...] = ()
    crop_keyframes: tuple[CropKeyframe, ...] = ()
    interpolation_policy: str = "smoothstep-ease"

    def as_dict(self) -> dict[str, object]:
        return {
            "scene_index": self.scene_index,
            "block_index": self.block_index,
            "source_start": round(self.source_start, 4),
            "source_end": round(self.source_end, 4),
            "cut_start": self.cut_start,
            "framing_mode": self.framing.mode,
            "evidence": list(self.framing.evidence),
            "tracks": [track.as_dict() for track in self.tracks],
            "crop_keyframes": [keyframe.as_dict() for keyframe in self.crop_keyframes],
            "interpolation_policy": self.interpolation_policy,
            "fallback_parameters": dict(self.framing.fallback_parameters),
            "protection": self.framing.protected,
        }


@dataclass(frozen=True)
class CaptionWordTiming:
    """One canonical FINAL_CLIP word and its exact spoken timing.

    ``text`` is the unchanged source token (logical Unicode order). Timing is
    never synthesized: it is copied from the FINAL_CLIP word timestamps.
    """

    index: int
    text: str
    start: float
    end: float

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
        }


@dataclass(frozen=True)
class CaptionEvent:
    """One FINAL_CLIP caption event referencing exact word indexes."""

    event_id: str
    block_index: int
    word_start_index: int
    word_end_index: int
    start: float
    end: float
    text: str
    lines: tuple[str, ...]
    placement_zone: str
    placement_reason: str
    collision_evidence: Mapping[str, object] = field(default_factory=dict)
    word_timings: tuple[CaptionWordTiming, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "block_index": self.block_index,
            "word_start_index": self.word_start_index,
            "word_end_index": self.word_end_index,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "text": self.text,
            "lines": list(self.lines),
            "placement_zone": self.placement_zone,
            "placement_reason": self.placement_reason,
            "collision_evidence": dict(self.collision_evidence),
            "word_timings": [word.as_dict() for word in self.word_timings],
        }


@dataclass(frozen=True)
class OverlayRequirement:
    """One materialization-required overlay placement requirement.

    ``text`` is always None: Stage 5.1 never invents or carries publication text.
    """

    requirement_id: str
    block_index: int | None
    block_type: str
    slot_kind: str
    status: str
    desired_zone: str
    text: str | None = None
    authoring_reference: Mapping[str, object] | None = None
    timeline_hint: Mapping[str, object] | None = None
    collision_constraints: tuple[str, ...] = ()
    placement_reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "block_index": self.block_index,
            "block_type": self.block_type,
            "slot_kind": self.slot_kind,
            "status": self.status,
            "desired_zone": self.desired_zone,
            "text": self.text,
            "authoring_reference": (
                dict(self.authoring_reference) if self.authoring_reference else None
            ),
            "timeline_hint": dict(self.timeline_hint) if self.timeline_hint else None,
            "collision_constraints": list(self.collision_constraints),
            "placement_reason": self.placement_reason,
        }


@dataclass(frozen=True)
class CompositionMetrics:
    """Bounded deterministic counters persisted with a plan."""

    analyzed_source_seconds: float = 0.0
    sampled_frames: int = 0
    scene_count: int = 0
    track_count: int = 0
    face_detections: int = 0
    tracked_crop_seconds: float = 0.0
    static_crop_seconds: float = 0.0
    multi_subject_seconds: float = 0.0
    background_fill_seconds: float = 0.0
    source_as_is_seconds: float = 0.0
    no_face_seconds: float = 0.0
    center_fallback_seconds: float = 0.0
    caption_events: int = 0
    caption_lines: int = 0
    analysis_wall_seconds: float = 0.0
    decode_wall_seconds: float = 0.0
    detection_wall_seconds: float = 0.0
    cache_reuse: bool = False
    fallback_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "analyzed_source_seconds": self.analyzed_source_seconds,
            "sampled_frames": self.sampled_frames,
            "scene_count": self.scene_count,
            "track_count": self.track_count,
            "face_detections": self.face_detections,
            "tracked_crop_seconds": self.tracked_crop_seconds,
            "static_crop_seconds": self.static_crop_seconds,
            "multi_subject_seconds": self.multi_subject_seconds,
            "background_fill_seconds": self.background_fill_seconds,
            "source_as_is_seconds": self.source_as_is_seconds,
            "no_face_seconds": self.no_face_seconds,
            "center_fallback_seconds": self.center_fallback_seconds,
            "caption_events": self.caption_events,
            "caption_lines": self.caption_lines,
            "analysis_wall_seconds": self.analysis_wall_seconds,
            "decode_wall_seconds": self.decode_wall_seconds,
            "detection_wall_seconds": self.detection_wall_seconds,
            "cache_reuse": self.cache_reuse,
            "fallback_reasons": list(self.fallback_reasons),
        }


@dataclass(frozen=True)
class BoundSpan:
    """A selected bound source span resolved from the Stage 5.0 contract."""

    block_index: int
    start: float
    end: float
    word_start_index: int | None
    word_end_index: int | None
    source_role: str | None
    is_hero: bool
    caption_text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "word_start_index": self.word_start_index,
            "word_end_index": self.word_end_index,
            "source_role": self.source_role,
            "is_hero": self.is_hero,
        }


@dataclass(frozen=True)
class PlannerInputs:
    """Everything the pure planner consumes from the Stage 5.0 contract."""

    candidate_id: str
    source_id: str
    contract_id: str
    contract_input_fingerprint: str
    contract_output_fingerprint: str
    contract_status: str
    contract_ready: bool
    selected_plan_id: str | None
    selection_id: str | None
    final_refinement_id: str | None
    source_media_relative_path: str
    source_media_identity: Mapping[str, object]
    display_geometry: DisplayGeometry
    frames_per_second: float
    spans: tuple[BoundSpan, ...]
    blocks: tuple[Mapping[str, object], ...]
    hero_block_index: int | None
    preservation_constraints: tuple[str, ...]
    retention: Mapping[str, object]
    governance: Mapping[str, object]
    narration: Mapping[str, object]
    materialization_slots: tuple[Mapping[str, object], ...]
    caption_input: Mapping[str, object]
    caption_source_fingerprint: str
    output_profile: Mapping[str, object]
    safe_zone_key: str
    language: str | None
    dialect_profile: str | None
    dialect_confidence: float
    code_switch_evidence: Mapping[str, object]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisualCompositionPlan:
    """Fully assembled plan payload (before persistence)."""

    status: str
    plan_ready: bool
    reason_codes: tuple[str, ...]
    payload: Mapping[str, object]
    metrics: Mapping[str, object]
    input_fingerprint: str
    output_fingerprint: str
    analysis_fingerprint: str
    framing_fingerprint: str
    ass_fingerprint: str
    caption_source_fingerprint: str
    contract_input_fingerprint: str
    contract_output_fingerprint: str
    source_media_fingerprint: str
    source_media_identity: Mapping[str, object]
    cache_eligible: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "plan_ready": self.plan_ready,
            "reason_codes": list(self.reason_codes),
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
        }


__all__ = [
    "BoundSpan",
    "CaptionEvent",
    "CaptionWordTiming",
    "CompositionMetrics",
    "CropKeyframe",
    "DisplayGeometry",
    "FaceDetection",
    "FaceTrack",
    "FramingDecision",
    "OverlayRequirement",
    "PlannerInputs",
    "Scene",
    "TrackSample",
    "VisualCompositionPlan",
]
