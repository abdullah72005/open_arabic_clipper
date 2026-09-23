"""Versioned Stage 5.1 visual-composition policy and closed constants.

Every output-affecting constant lives here so the plan is auditable and
fingerprintable. The module is pure: no network, no provider, no model loading,
no audio decoding, no rendering. Bumping any version here changes the plan input
fingerprint and invalidates prior Stage 5.1 plans at their correct boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Vendored, license-cleared detector asset (OpenCV Zoo YuNet 2023mar).
DEFAULT_DETECTOR_MODEL_PATH = str(
    Path(__file__).resolve().parent / "assets" / "face_detection_yunet_2023mar.onnx"
)

VISUAL_COMPOSITION_POLICY_VERSION = "stage5.1-v1"
SCHEMA_VERSION = "stage5.1-schema-v1"
FINGERPRINT_VERSION = "1"
FRAMING_POLICY_VERSION = "stage5.1-framing-v1"
CAPTION_LAYOUT_POLICY_VERSION = "stage5.1-caption-layout-v3"
ASS_POLICY_VERSION = "stage5.1-ass-v3"
BACKGROUND_FILL_POLICY_VERSION = "stage5.1-background-fill-v1"
SAFE_ZONE_PROFILE_VERSION = "shorts-reels-safe-zone-v1"

# Coarse deterministic output dimensions for the target Shorts/Reels profile.

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_ASPECT = OUTPUT_WIDTH / OUTPUT_HEIGHT

# Detector identity is part of the fingerprint. The vendored OpenCV Zoo YuNet
# 2023mar model is compiled with a fixed 640x640 input (empirically verified:
# a 320x320 tensor raises INVALID_ARGUMENT), so the default input size is 640,
# not the 320 the upstream Python sample can use with a dynamic export.
DETECTOR_IDENTITY: Mapping[str, object] = {
    "name": "opencv_zoo_face_detection_yunet",
    "model": "face_detection_yunet_2023mar.onnx",
    "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "input_size": 640,
    "score_threshold": 0.6,
    "nms_iou": 0.3,
    "execution_provider": "CPUExecutionProvider",
    "intra_op_threads": 2,
}

# --- Closed enums -----------------------------------------------------------


class FramingMode(str, Enum):
    """Deterministic per-scene framing decision."""

    SOURCE_AS_IS = "SOURCE_AS_IS"
    STATIC_CROP = "STATIC_CROP"
    TRACKED_CROP = "TRACKED_CROP"
    MULTI_SUBJECT_FIT = "MULTI_SUBJECT_FIT"
    BACKGROUND_FILL = "BACKGROUND_FILL"
    CENTER_FALLBACK = "CENTER_FALLBACK"


class FramingEvidence(str, Enum):
    """Closed reason codes recorded with every framing decision."""

    SINGLE_PERSISTENT_FACE = "SINGLE_PERSISTENT_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    NO_FACE = "NO_FACE"
    FACE_TRACK_UNSTABLE = "FACE_TRACK_UNSTABLE"
    SUBJECT_TOO_WIDE = "SUBJECT_TOO_WIDE"
    SUBJECT_TOO_SMALL = "SUBJECT_TOO_SMALL"
    IMPORTANT_CONTENT_WOULD_BE_CROPPED = "IMPORTANT_CONTENT_WOULD_BE_CROPPED"
    SOURCE_ALREADY_VERTICAL = "SOURCE_ALREADY_VERTICAL"
    SCREEN_CONTENT = "SCREEN_CONTENT"
    FACE_NEAR_SOURCE_EDGE = "FACE_NEAR_SOURCE_EDGE"
    DETECTION_LOST = "DETECTION_LOST"
    DETECTOR_UNAVAILABLE = "DETECTOR_UNAVAILABLE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    HERO_PROTECTION_APPLIED = "HERO_PROTECTION_APPLIED"
    CAPTION_ZONE_AFFECTED = "CAPTION_ZONE_AFFECTED"


class VisualCompositionStatus(str, Enum):
    """Semantic readiness of one persisted visual-composition plan."""

    READY_FOR_VISUAL_EXECUTION = "READY_FOR_VISUAL_EXECUTION"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class VisualCompositionExecutionStatus(str, Enum):
    """Processing lifecycle, separate from semantic readiness."""

    QUEUED = "QUEUED"
    ANALYZING = "ANALYZING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class CaptionPlacementZone(str, Enum):
    """Closed caption vertical band vocabulary."""

    LOWER = "LOWER"
    UPPER = "UPPER"


class OverlayZone(str, Enum):
    """Closed overlay vertical band vocabulary."""

    TOP_HOOK = "TOP_HOOK"
    UPPER_THIRD = "UPPER_THIRD"
    LOWER_THIRD = "LOWER_THIRD"
    CENTER = "CENTER"


class OverlayStatus(str, Enum):
    """Overlay readiness; currently-materialized text never exists in Stage 5.1."""

    MATERIALIZATION_REQUIRED = "MATERIALIZATION_REQUIRED"
    READY = "READY"


class PlanReasonCode(str, Enum):
    """Closed Stage 5.1 plan-level reason codes."""

    CONTRACT_NOT_CURRENT = "CONTRACT_NOT_CURRENT"
    CONTRACT_NOT_EXECUTABLE = "CONTRACT_NOT_EXECUTABLE"
    NO_BOUND_SOURCE_SPANS = "NO_BOUND_SOURCE_SPANS"
    SOURCE_MEDIA_UNAVAILABLE = "SOURCE_MEDIA_UNAVAILABLE"
    ANALYSIS_SCOPE_EXCEEDED = "ANALYSIS_SCOPE_EXCEEDED"
    DETECTOR_UNAVAILABLE = "DETECTOR_UNAVAILABLE"
    CAPTION_EVIDENCE_MISSING = "CAPTION_EVIDENCE_MISSING"
    CAPTION_COLLISION_UNRESOLVED = "CAPTION_COLLISION_UNRESOLVED"
    BIDI_CONTROL_NEUTRALIZED = "BIDI_CONTROL_NEUTRALIZED"
    TRACKS_RESET_AT_CUT = "TRACKS_RESET_AT_CUT"
    FALLBACK_APPLIED = "FALLBACK_APPLIED"


# --- Safe zones -------------------------------------------------------------


@dataclass(frozen=True)
class SafeZoneProfile:
    """Conservative platform-safe insets as fractions of the output frame.

    Fractions are applied to 1080x1920. Pixel getters return rounded integers so
    FFmpeg and ASS consume the same numbers.
    """

    key: str
    semantic_version: str
    top: float
    bottom: float
    left: float
    right: float
    caption_bottom_gap_px: int
    caption_top_gap_px: int

    def top_px(self) -> int:
        return round(self.top * OUTPUT_HEIGHT)

    def bottom_px(self) -> int:
        return round(self.bottom * OUTPUT_HEIGHT)

    def left_px(self) -> int:
        return round(self.left * OUTPUT_WIDTH)

    def right_px(self) -> int:
        return round(self.right * OUTPUT_WIDTH)

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "semantic_version": self.semantic_version,
            "fractions": {
                "top": self.top,
                "bottom": self.bottom,
                "left": self.left,
                "right": self.right,
            },
            "pixels": {
                "top": self.top_px(),
                "bottom": self.bottom_px(),
                "left": self.left_px(),
                "right": self.right_px(),
            },
            "caption_bottom_gap_px": self.caption_bottom_gap_px,
            "caption_top_gap_px": self.caption_top_gap_px,
        }


SHORTS_REELS_SAFE_ZONE = SafeZoneProfile(
    key="SHORTS_VERTICAL_SAFE_ZONE_V1",
    semantic_version=SAFE_ZONE_PROFILE_VERSION,
    top=0.12,
    bottom=0.25,
    left=0.05,
    right=0.15,
    caption_bottom_gap_px=24,
    caption_top_gap_px=24,
)

GENERIC_SAFE_ZONE = SafeZoneProfile(
    key="GENERIC_CONSERVATIVE_SAFE_ZONE_V1",
    semantic_version=SAFE_ZONE_PROFILE_VERSION,
    top=0.12,
    bottom=0.25,
    left=0.05,
    right=0.15,
    caption_bottom_gap_px=24,
    caption_top_gap_px=24,
)

SAFE_ZONE_PROFILES: Mapping[str, SafeZoneProfile] = {
    SHORTS_REELS_SAFE_ZONE.key: SHORTS_REELS_SAFE_ZONE,
    GENERIC_SAFE_ZONE.key: GENERIC_SAFE_ZONE,
}


def safe_zone_for(key: str | None) -> SafeZoneProfile:
    """Return the Stage 5.0-keyed safe zone or the conservative fallback."""

    if key is not None and key in SAFE_ZONE_PROFILES:
        return SAFE_ZONE_PROFILES[key]
    return GENERIC_SAFE_ZONE


# --- Caption style ----------------------------------------------------------


@dataclass(frozen=True)
class CaptionStyle:
    """Config-driven caption style with conservative Shorts/Reels defaults."""

    font_family: str = "Noto Sans Arabic"
    font_size: int = 88
    primary_color: str = "&H00FFFFFF"
    secondary_color: str = "&H000000FF"
    outline_color: str = "&H00000000"
    outline_width: int = 7
    shadow: int = 3
    active_color: str = "&H0000FFFF"
    active_emphasis: bool = True
    max_lines: int = 2
    max_line_width_fraction: float = 0.86
    spacing: int = 0
    alignment_lower: int = 2
    alignment_upper: int = 8
    max_event_duration: float = 3.6
    min_event_duration: float = 0.7
    pause_split_seconds: float = 0.45
    max_words_per_event: int = 7
    tail_after_last_word: float = 0.30

    def as_dict(self) -> dict[str, object]:
        return {
            "font_family": self.font_family,
            "font_size": self.font_size,
            "primary_color": self.primary_color,
            "secondary_color": self.secondary_color,
            "outline_color": self.outline_color,
            "outline_width": self.outline_width,
            "shadow": self.shadow,
            "active_color": self.active_color,
            "active_emphasis": self.active_emphasis,
            "max_lines": self.max_lines,
            "max_line_width_fraction": self.max_line_width_fraction,
            "spacing": self.spacing,
            "alignment_lower": self.alignment_lower,
            "alignment_upper": self.alignment_upper,
            "max_event_duration": self.max_event_duration,
            "min_event_duration": self.min_event_duration,
            "pause_split_seconds": self.pause_split_seconds,
            "max_words_per_event": self.max_words_per_event,
            "tail_after_last_word": self.tail_after_last_word,
        }


# --- Config -----------------------------------------------------------------


@dataclass(frozen=True)
class Stage51Config:
    """Bounded Stage 5.1 configuration derived from runtime Settings."""

    analysis_fps: float = 2.0
    max_analysis_frames: int = 1500
    max_analysis_seconds: float = 600.0
    analysis_frame_max_dimension: int = 640
    scene_cut_threshold: float = 0.35
    min_scene_seconds: float = 0.4
    scene_context_seconds: float = 0.5
    detector_enabled: bool = True
    detector_model_path: str = DEFAULT_DETECTOR_MODEL_PATH
    detector_input_size: int = 640
    detector_score_threshold: float = 0.6
    detector_nms_iou: float = 0.3
    track_max_gap_samples: int = 2
    track_min_persistence_samples: int = 3
    track_min_persistence_seconds: float = 1.0
    face_min_height_fraction: float = 0.06
    target_face_height_fraction: float = 0.38
    min_crop_height_fraction: float = 0.35
    headroom_fraction: float = 0.12
    chin_margin_fraction: float = 0.10
    dead_zone_fraction: float = 0.18
    max_pan_velocity_per_second: float = 0.55
    max_zoom_rate_per_second: float = 0.15
    min_hold_seconds: float = 0.5
    max_keyframes_per_scene: int = 40
    caption_font_family: str = "Noto Sans Arabic"
    caption_max_lines: int = 2
    caption_active_color: str = "&H0000FFFF"
    caption_dynamic_emphasis: bool = True
    caption_max_words_per_event: int = 7
    safe_zone_profile_key: str = "SHORTS_VERTICAL_SAFE_ZONE_V1"
    preview_enabled: bool = True
    preview_max_frames: int = 6

    def dedup_epsilon(self) -> float:
        return 1e-4

    def detection_hold_seconds(self) -> float:
        return max(self.min_hold_seconds, 1.0)

    def as_dict(self) -> dict[str, object]:
        payload = dict(self.__dict__)
        return payload


def config_fingerprint_dict(config: Stage51Config) -> dict[str, object]:
    """Config values that participate in fingerprints.

    A machine-specific deployment path is not identity (the detector sha256 is),
    so ``detector_model_path`` is always excluded.
    """

    payload = config.as_dict()
    payload.pop("detector_model_path", None)
    return payload


def stage51_config_payload(config: Stage51Config) -> dict[str, object]:
    """Deterministic fingerprint payload for all output-affecting policy.

    Deliberately excludes TTS provider/model/voice, narration audio/text,
    publishing title/schedule/metadata, codec/encoder settings, final render
    artifacts, and analytics config: none of those are Stage 5.1 inputs.
    """

    config_payload = config_fingerprint_dict(config)
    return {
        "policy_version": VISUAL_COMPOSITION_POLICY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "framing_policy_version": FRAMING_POLICY_VERSION,
        "caption_layout_policy_version": CAPTION_LAYOUT_POLICY_VERSION,
        "ass_policy_version": ASS_POLICY_VERSION,
        "background_fill_policy_version": BACKGROUND_FILL_POLICY_VERSION,
        "safe_zone_profile_version": SAFE_ZONE_PROFILE_VERSION,
        "output": {
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "aspect": round(OUTPUT_ASPECT, 6),
        },
        "config": config_payload,
        "caption_style": CaptionStyle(
            font_family=config.caption_font_family,
            max_lines=config.caption_max_lines,
            active_color=config.caption_active_color,
            active_emphasis=config.caption_dynamic_emphasis,
            max_words_per_event=config.caption_max_words_per_event,
        ).as_dict(),
        "safe_zone": safe_zone_for(config.safe_zone_profile_key).as_dict(),
        "detector_identity": dict(DETECTOR_IDENTITY),
    }


def framing_for_bounded_distance(distance: float) -> float:
    """Pure helper: deterministic smoothstep easing on a [0, 1] distance."""

    if distance <= 0.0:
        return 0.0
    if distance >= 1.0:
        return 1.0
    return distance * distance * (3.0 - 2.0 * distance)


__all__ = [
    "ASS_POLICY_VERSION",
    "BACKGROUND_FILL_POLICY_VERSION",
    "CAPTION_LAYOUT_POLICY_VERSION",
    "CaptionPlacementZone",
    "CaptionStyle",
    "DEFAULT_DETECTOR_MODEL_PATH",
    "DETECTOR_IDENTITY",
    "FINGERPRINT_VERSION",
    "FRAMING_POLICY_VERSION",
    "FramingEvidence",
    "FramingMode",
    "GENERIC_SAFE_ZONE",
    "OUTPUT_ASPECT",
    "OUTPUT_HEIGHT",
    "OUTPUT_WIDTH",
    "OverlayStatus",
    "OverlayZone",
    "PlanReasonCode",
    "SAFE_ZONE_PROFILES",
    "SAFE_ZONE_PROFILE_VERSION",
    "SCHEMA_VERSION",
    "SHORTS_REELS_SAFE_ZONE",
    "SafeZoneProfile",
    "Stage51Config",
    "VISUAL_COMPOSITION_POLICY_VERSION",
    "VisualCompositionExecutionStatus",
    "VisualCompositionStatus",
    "config_fingerprint_dict",
    "framing_for_bounded_distance",
    "safe_zone_for",
    "stage51_config_payload",
]
