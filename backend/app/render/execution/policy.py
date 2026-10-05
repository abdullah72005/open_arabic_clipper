"""Versioned Stage 5.2 render-execution policy, delivery profile, and bounds.

Every output-affecting constant lives here so a render is auditable and
fingerprintable. The module is pure: no network, no provider, no model loading,
no audio decoding, and no rendering. Stage 5.0/5.1 modules are untouched.

The Stage 5.0 contract owns output geometry, target frame rate, and
``SOURCE_AUDIO_PRESERVED``. Stage 5.2 adds a separate versioned delivery profile
(container/codecs/bitrate/faststart) that the Stage 5.0 contract deliberately
does not prescribe. The delivery profile only affects Stage 5.2 fingerprints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Versions

EXECUTION_POLICY_VERSION = "stage5.2-v2"
EXECUTION_SCHEMA_VERSION = "stage5.2-schema-v2"
EXECUTION_FINGERPRINT_VERSION = "2"
COMPILER_VERSION = "stage5.2-compiler-v2"
TIMELINE_POLICY_VERSION = "stage5.2-timeline-v2"
QC_POLICY_VERSION = "stage5.2-qc-v2"
DELIVERY_PROFILE_VERSION = "stage5.2-delivery-v1"
CONCURRENCY_POLICY_VERSION = "stage5.2-concurrency-v2"

# Artifact purpose. ``CORE_SOURCE_VALIDATION`` executes the ordered
# SOURCE_MEDIA occurrences from the Stage 5.0 contract. It is a terminal
# validation output, never a publication-final artifact and never a mandatory
# pipeline intermediate.

CORE_SOURCE_VALIDATION = "CORE_SOURCE_VALIDATION"
SUPPORTED_ARTIFACT_PURPOSES = frozenset({CORE_SOURCE_VALIDATION})

# Lifecycle (persisted)

QUEUED = "QUEUED"
RENDERING = "RENDERING"
QC_RUNNING = "QC_RUNNING"
COMPLETE = "COMPLETE"
BLOCKED = "BLOCKED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

# QC severity (separate from lifecycle)

QC_PASS = "PASS"
QC_WARN = "WARN"
QC_FAIL = "FAIL"

# Reason codes (closed vocabulary)

CANDIDATE_NOT_CURRENT = "CANDIDATE_NOT_CURRENT"
CANDIDATE_NOT_RETAINED = "CANDIDATE_NOT_RETAINED"
CONTRACT_NOT_FOUND = "CONTRACT_NOT_FOUND"
CONTRACT_NOT_CURRENT = "CONTRACT_NOT_CURRENT"
CONTRACT_NOT_EXECUTABLE = "CONTRACT_NOT_EXECUTABLE"
VISUAL_PLAN_NOT_FOUND = "VISUAL_PLAN_NOT_FOUND"
VISUAL_PLAN_NOT_CURRENT = "VISUAL_PLAN_NOT_CURRENT"
VISUAL_PLAN_NOT_READY = "VISUAL_PLAN_NOT_READY"
VISUAL_PLAN_FINGERPRINT_MISMATCH = "VISUAL_PLAN_FINGERPRINT_MISMATCH"
SOURCE_MEDIA_MISSING = "SOURCE_MEDIA_MISSING"
SOURCE_MEDIA_UNMANAGED = "SOURCE_MEDIA_UNMANAGED"
SOURCE_MEDIA_ZERO_BYTES = "SOURCE_MEDIA_ZERO_BYTES"
SOURCE_MEDIA_CHANGED = "SOURCE_MEDIA_CHANGED"
SOURCE_MEDIA_UNEXECUTABLE = "SOURCE_MEDIA_UNEXECUTABLE"
SOURCE_IDENTITY_MISMATCH = "SOURCE_IDENTITY_MISMATCH"
ASS_MISSING = "ASS_MISSING"
ASS_HASH_MISMATCH = "ASS_HASH_MISMATCH"
ASS_MALFORMED = "ASS_MALFORMED"
ASS_EMPTY = "ASS_EMPTY"
ASS_METADATA_MISMATCH = "ASS_METADATA_MISMATCH"
UNSUPPORTED_ARTIFACT_PURPOSE = "UNSUPPORTED_ARTIFACT_PURPOSE"
UNSUPPORTED_DELIVERY_PROFILE = "UNSUPPORTED_DELIVERY_PROFILE"
UNSUPPORTED_OUTPUT_GEOMETRY = "UNSUPPORTED_OUTPUT_GEOMETRY"
UNSUPPORTED_FRAMING_MODE = "UNSUPPORTED_FRAMING_MODE"
UNSUPPORTED_INTERPOLATION = "UNSUPPORTED_INTERPOLATION"
CONTRADICTORY_FRAMING_EVIDENCE = "CONTRADICTORY_FRAMING_EVIDENCE"
SCENE_COVERAGE_GAP = "SCENE_COVERAGE_GAP"
SCENE_OVERLAP = "SCENE_OVERLAP"
SCENE_BLOCK_MISMATCH = "SCENE_BLOCK_MISMATCH"
SCENE_BOUNDS_INVALID = "SCENE_BOUNDS_INVALID"
KEYFRAME_OUT_OF_SCOPE = "KEYFRAME_OUT_OF_SCOPE"
KEYFRAME_NON_MONOTONIC = "KEYFRAME_NON_MONOTONIC"
KEYFRAME_GEOMETRY_INVALID = "KEYFRAME_GEOMETRY_INVALID"
OCCURRENCE_BOUNDS_INVALID = "OCCURRENCE_BOUNDS_INVALID"
OCCURRENCE_BEYOND_MEDIA = "OCCURRENCE_BEYOND_MEDIA"
NO_RENDERABLE_OCCURRENCES = "NO_RENDERABLE_OCCURRENCES"
EXOTIC_PIXEL_ASPECT = "EXOTIC_PIXEL_ASPECT"
UNSUPPORTED_ROTATION = "UNSUPPORTED_ROTATION"
RENDER_TIMEOUT = "RENDER_TIMEOUT"
RENDER_CANCELLED = "RENDER_CANCELLED"
RENDER_ADMISSION_UNAVAILABLE = "RENDER_ADMISSION_UNAVAILABLE"
RENDER_PROCESS_FAILED = "RENDER_PROCESS_FAILED"
RENDER_OWNERSHIP_LOST = "RENDER_OWNERSHIP_LOST"
ARTIFACT_MISSING = "ARTIFACT_MISSING"
ARTIFACT_ZERO_BYTES = "ARTIFACT_ZERO_BYTES"
ARTIFACT_HASH_MISMATCH = "ARTIFACT_HASH_MISMATCH"
QC_PROBE_FAILED = "QC_PROBE_FAILED"
QC_STREAMS_MISMATCH = "QC_STREAMS_MISMATCH"
QC_CODEC_MISMATCH = "QC_CODEC_MISMATCH"
QC_PIXEL_FORMAT_MISMATCH = "QC_PIXEL_FORMAT_MISMATCH"
QC_GEOMETRY_MISMATCH = "QC_GEOMETRY_MISMATCH"
QC_ROTATION_MISMATCH = "QC_ROTATION_MISMATCH"
QC_DURATION_MISMATCH = "QC_DURATION_MISMATCH"
QC_FRAME_COUNT_MISMATCH = "QC_FRAME_COUNT_MISMATCH"
QC_SAMPLE_RATE_MISMATCH = "QC_SAMPLE_RATE_MISMATCH"
QC_CHANNEL_LAYOUT_MISMATCH = "QC_CHANNEL_LAYOUT_MISMATCH"
QC_DECODE_FAILED = "QC_DECODE_FAILED"
QC_CAPTION_OUT_OF_RANGE = "QC_CAPTION_OUT_OF_RANGE"
QC_BLANK_RENDER = "QC_BLANK_RENDER"
QC_FROZEN_RENDER = "QC_FROZEN_RENDER"
QC_TOTAL_SILENCE = "QC_TOTAL_SILENCE"
QC_PEAK_CLIPPING_RISK = "QC_PEAK_CLIPPING_RISK"
QC_DURATION_DRIFT = "QC_DURATION_DRIFT"
QC_CUMULATIVE_DRIFT = "QC_CUMULATIVE_DRIFT"
RENDER_DISABLED = "RENDER_DISABLED"

# Remediation reason codes (close the previously silent failure paths).
SOURCE_STREAMS_UNSUPPORTED = "SOURCE_STREAMS_UNSUPPORTED"
SOURCE_START_UNSUPPORTED = "SOURCE_START_UNSUPPORTED"
SOURCE_DURATION_EXCEEDS_LIMIT = "SOURCE_DURATION_EXCEEDS_LIMIT"
OUTPUT_DURATION_EXCEEDS_LIMIT = "OUTPUT_DURATION_EXCEEDS_LIMIT"
ATTEMPT_OWNERSHIP_LOST = "ATTEMPT_OWNERSHIP_LOST"
RENDER_DISPATCH_FAILED = "RENDER_DISPATCH_FAILED"
CACHE_ARTIFACT_MISSING = "CACHE_ARTIFACT_MISSING"
CACHE_ARTIFACT_CORRUPT = "CACHE_ARTIFACT_CORRUPT"
CACHE_QC_POLICY_CHANGED = "CACHE_QC_POLICY_CHANGED"
QC_TIMING_MISSING = "QC_TIMING_MISSING"
QC_AV_TIMING_MISMATCH = "QC_AV_TIMING_MISMATCH"
QC_STREAM_TIMING_MISMATCH = "QC_STREAM_TIMING_MISMATCH"
QC_SOURCE_AUDIO_MISMATCH = "QC_SOURCE_AUDIO_MISMATCH"
QC_SOURCE_AUDIO_UNAVAILABLE = "QC_SOURCE_AUDIO_UNAVAILABLE"
EXECUTION_ROW_NOT_CURRENT = "EXECUTION_ROW_NOT_CURRENT"
SEGMENT_LIMIT_EXCEEDED = "SEGMENT_LIMIT_EXCEEDED"

# Resource bounds and QC thresholds. Versioned so a policy change invalidates
# prior QC verdicts through the QC fingerprint.

MAX_SOURCE_DURATION_SECONDS = 3600.0
MAX_OUTPUT_DURATION_SECONDS = 900.0
MAX_FILTERGRAPH_CHARACTERS = 200_000
MAX_SEGMENTS = 64
MAX_TRACKED_SCENE_KEYFRAMES = 128
DEFAULT_MAX_RENDER_SECONDS = 900.0
DEFAULT_CANCEL_POLL_SECONDS = 0.5
DEFAULT_FFMPEG_THREADS = 2
DEFAULT_FILTER_THREADS = 1
DEFAULT_FILTER_COMPLEX_THREADS = 1
DEFAULT_GLOBAL_CONCURRENT_RENDERS = 1

# Source stream-timing tolerance. Frame/sample granularity makes a genuine
# stream start offset below this indistinguishable from jitter; anything larger
# is a supported explicit gap the compiler must preserve (never silently shift).
SOURCE_START_TOLERANCE_SECONDS = 0.02
QC_AV_DURATION_TOLERANCE_SECONDS = 0.10

# Technical QC thresholds

QC_BLACK_LUMA_THRESHOLD = 18.0
QC_BLACK_FRAME_FRACTION_FAIL = 0.98
QC_BLACK_FRAME_FRACTION_WARN = 0.85
QC_FROZEN_DIFF_THRESHOLD = 0.5
QC_FROZEN_FRAME_FRACTION_FAIL = 0.95
QC_FROZEN_FRAME_FRACTION_WARN = 0.80
QC_SILENCE_DBFS = -60.0
QC_PEAK_DBFS = -1.0
QC_MAX_SAMPLED_FRAMES = 24
QC_SAMPLE_LUMA_DIMENSION = 160
QC_TIMING_BASE_TOLERANCE_SECONDS = 0.10
QC_TIMING_AUDIO_TOLERANCE_SECONDS = 1024.0 / 48000.0
QC_CUMULATIVE_DRIFT_MS_PER_JOIN = 12.0

# The contract output profile key when no explicit profile is supplied.

OUTPUT_PROFILE_WIDTH = 1080
OUTPUT_PROFILE_HEIGHT = 1920
OUTPUT_ASPECT = OUTPUT_PROFILE_WIDTH / OUTPUT_PROFILE_HEIGHT


@dataclass(frozen=True)
class DeliveryProfile:
    """One code-defined, versioned Stage 5.2 delivery profile.

    Deliberately independent of the Stage 5.0 output profile: the Stage 5.0
    contract owns geometry/frame-rate/audio-preservation, Stage 5.2 owns
    container and codec settings. Only output-affecting values participate in
    Stage 5.2 fingerprints.
    """

    key: str
    semantic_version: str
    container: str
    video_codec: str
    video_preset: str
    video_crf: int
    pixel_format: str
    audio_codec: str
    audio_bitrate: str
    audio_sample_rate: int
    preserve_mono_and_stereo: bool
    downmix_wider_to_stereo: bool
    faststart: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "semantic_version": self.semantic_version,
            "container": self.container,
            "video_codec": self.video_codec,
            "video_preset": self.video_preset,
            "video_crf": self.video_crf,
            "pixel_format": self.pixel_format,
            "audio_codec": self.audio_codec,
            "audio_bitrate": self.audio_bitrate,
            "audio_sample_rate": self.audio_sample_rate,
            "preserve_mono_and_stereo": self.preserve_mono_and_stereo,
            "downmix_wider_to_stereo": self.downmix_wider_to_stereo,
            "faststart": self.faststart,
            "cpu_only": True,
        }


MP4_H264_AAC_1080X1920 = DeliveryProfile(
    key="MP4_H264_AAC_1080X1920_V1",
    semantic_version=DELIVERY_PROFILE_VERSION,
    container="mp4",
    video_codec="libx264",
    video_preset="veryfast",
    video_crf=20,
    pixel_format="yuv420p",
    audio_codec="aac",
    audio_bitrate="192k",
    audio_sample_rate=48000,
    preserve_mono_and_stereo=True,
    downmix_wider_to_stereo=True,
    faststart=True,
)

DELIVERY_PROFILES: Mapping[str, DeliveryProfile] = {
    MP4_H264_AAC_1080X1920.key: MP4_H264_AAC_1080X1920,
}

DEFAULT_DELIVERY_PROFILE_KEY = MP4_H264_AAC_1080X1920.key


def delivery_profile_for(key: str | None) -> DeliveryProfile | None:
    """Return the requested profile or ``None`` for an unsupported key."""

    if key is None:
        return MP4_H264_AAC_1080X1920
    return DELIVERY_PROFILES.get(key)


@dataclass(frozen=True)
class Stage52Config:
    """Bounded Stage 5.2 render-execution configuration."""

    encoder_threads: int = DEFAULT_FFMPEG_THREADS
    filter_threads: int = DEFAULT_FILTER_THREADS
    filter_complex_threads: int = DEFAULT_FILTER_COMPLEX_THREADS
    global_concurrent_renders: int = DEFAULT_GLOBAL_CONCURRENT_RENDERS
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    cancel_poll_seconds: float = DEFAULT_CANCEL_POLL_SECONDS
    admission_wait_seconds: float = 0.0
    max_render_seconds: float = DEFAULT_MAX_RENDER_SECONDS
    max_source_duration_seconds: float = MAX_SOURCE_DURATION_SECONDS
    max_output_duration_seconds: float = MAX_OUTPUT_DURATION_SECONDS
    max_segments: int = MAX_SEGMENTS
    qc_max_sampled_frames: int = QC_MAX_SAMPLED_FRAMES
    qc_sample_luma_dimension: int = QC_SAMPLE_LUMA_DIMENSION

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def stage52_config_payload(config: Stage52Config) -> dict[str, object]:
    """Deterministic fingerprint payload for output-affecting Stage 5.2 policy."""

    return {
        "policy_version": EXECUTION_POLICY_VERSION,
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "fingerprint_version": EXECUTION_FINGERPRINT_VERSION,
        "compiler_version": COMPILER_VERSION,
        "timeline_policy_version": TIMELINE_POLICY_VERSION,
        "delivery_profile_version": DELIVERY_PROFILE_VERSION,
        "concurrency_policy_version": CONCURRENCY_POLICY_VERSION,
        "config": {
            "encoder_threads": config.encoder_threads,
            "filter_threads": config.filter_threads,
            "filter_complex_threads": config.filter_complex_threads,
            "global_concurrent_renders": config.global_concurrent_renders,
            "stereo_downmix": MP4_H264_AAC_1080X1920.downmix_wider_to_stereo,
        },
        "delivery_profiles": {
            key: profile.as_dict() for key, profile in sorted(DELIVERY_PROFILES.items())
        },
        "output": {
            "width": OUTPUT_PROFILE_WIDTH,
            "height": OUTPUT_PROFILE_HEIGHT,
            "aspect": round(OUTPUT_ASPECT, 6),
        },
        "resource_bounds": {
            "max_source_duration_seconds": MAX_SOURCE_DURATION_SECONDS,
            "max_output_duration_seconds": MAX_OUTPUT_DURATION_SECONDS,
            "max_filtergraph_characters": MAX_FILTERGRAPH_CHARACTERS,
            "max_segments": MAX_SEGMENTS,
            "max_tracked_scene_keyframes": MAX_TRACKED_SCENE_KEYFRAMES,
        },
    }


def qc_payload(config: Stage52Config) -> dict[str, object]:
    """Deterministic fingerprint payload for the QC policy and thresholds."""

    return {
        "qc_policy_version": QC_POLICY_VERSION,
        "thresholds": {
            "black_luma_threshold": QC_BLACK_LUMA_THRESHOLD,
            "black_frame_fraction_fail": QC_BLACK_FRAME_FRACTION_FAIL,
            "black_frame_fraction_warn": QC_BLACK_FRAME_FRACTION_WARN,
            "frozen_diff_threshold": QC_FROZEN_DIFF_THRESHOLD,
            "frozen_frame_fraction_fail": QC_FROZEN_FRAME_FRACTION_FAIL,
            "frozen_frame_fraction_warn": QC_FROZEN_FRAME_FRACTION_WARN,
            "silence_dbfs": QC_SILENCE_DBFS,
            "peak_dbfs": QC_PEAK_DBFS,
            "max_sampled_frames": config.qc_max_sampled_frames,
            "sample_luma_dimension": config.qc_sample_luma_dimension,
            "timing_base_tolerance_seconds": QC_TIMING_BASE_TOLERANCE_SECONDS,
            "timing_audio_tolerance_seconds": QC_TIMING_AUDIO_TOLERANCE_SECONDS,
            "cumulative_drift_ms_per_join": QC_CUMULATIVE_DRIFT_MS_PER_JOIN,
        },
    }


__all__ = [
    "BLOCKED",
    "CANCELLED",
    "COMPLETE",
    "COMPILER_VERSION",
    "CONCURRENCY_POLICY_VERSION",
    "CORE_SOURCE_VALIDATION",
    "DEFAULT_CANCEL_POLL_SECONDS",
    "DEFAULT_DELIVERY_PROFILE_KEY",
    "DEFAULT_FFMPEG_THREADS",
    "DEFAULT_FILTER_THREADS",
    "DEFAULT_FILTER_COMPLEX_THREADS",
    "DEFAULT_GLOBAL_CONCURRENT_RENDERS",
    "DEFAULT_MAX_RENDER_SECONDS",
    "DELIVERY_PROFILES",
    "DELIVERY_PROFILE_VERSION",
    "DeliveryProfile",
    "EXECUTION_FINGERPRINT_VERSION",
    "EXECUTION_POLICY_VERSION",
    "EXECUTION_SCHEMA_VERSION",
    "FAILED",
    "MP4_H264_AAC_1080X1920",
    "OUTPUT_ASPECT",
    "OUTPUT_PROFILE_HEIGHT",
    "OUTPUT_PROFILE_WIDTH",
    "QC_PASS",
    "QC_POLICY_VERSION",
    "QC_WARN",
    "QC_FAIL",
    "QUEUED",
    "RENDERING",
    "QC_RUNNING",
    "SUPPORTED_ARTIFACT_PURPOSES",
    "Stage52Config",
    "TIMELINE_POLICY_VERSION",
    "delivery_profile_for",
    "qc_payload",
    "stage52_config_payload",
]
