"""Immutable Stage 5.2 render-execution value objects.

Everything here is immutable and JSON-serializable. The media components consume
these types only: they never discover candidates, query providers, or mutate
database rows. All source times are source-local seconds; output times are
unit-speed mapped seconds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from app.render.execution.policy import CORE_SOURCE_VALIDATION


@dataclass(frozen=True)
class CropKeyframeSpec:
    """One accepted crop keyframe in the source-local timeline."""

    t: float
    cx: float
    cy: float
    height_fraction: float
    mode: str = ""
    confidence: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "t": round(self.t, 6),
            "cx": round(self.cx, 6),
            "cy": round(self.cy, 6),
            "height_fraction": round(self.height_fraction, 6),
            "mode": self.mode,
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class SceneSpec:
    """One visual execution partition bound to an occurrence."""

    scene_index: int
    block_index: int
    source_start: float
    source_end: float
    framing_mode: str
    interpolation_policy: str
    crop_keyframes: tuple[CropKeyframeSpec, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "scene_index": self.scene_index,
            "block_index": self.block_index,
            "source_start": round(self.source_start, 6),
            "source_end": round(self.source_end, 6),
            "framing_mode": self.framing_mode,
            "interpolation_policy": self.interpolation_policy,
            "crop_keyframes": [frame.as_dict() for frame in self.crop_keyframes],
        }


@dataclass(frozen=True)
class AudioInstruction:
    """Explicit source-audio instruction for one occurrence.

    Stage 5.2 only supports preserved source audio with an optional explicit
    constant gain. Authored/narration audio is a future mixed-timeline seam and
    is never produced here.
    """

    mode: str = "SOURCE_AUDIO"
    gain_db: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "gain_db": round(self.gain_db, 6)}


@dataclass(frozen=True)
class TimelineOccurrence:
    """One ordered source occurrence mapped onto the output timeline."""

    occurrence_id: str
    block_index: int
    source_start: float
    source_end: float
    output_start: float
    output_end: float
    source_role: str | None
    is_hero: bool
    scenes: tuple[SceneSpec, ...]
    audio: AudioInstruction = field(default_factory=AudioInstruction)

    @property
    def source_duration(self) -> float:
        return self.source_end - self.source_start

    @property
    def output_duration(self) -> float:
        return self.output_end - self.output_start

    def as_dict(self) -> dict[str, object]:
        return {
            "occurrence_id": self.occurrence_id,
            "block_index": self.block_index,
            "source_start": round(self.source_start, 6),
            "source_end": round(self.source_end, 6),
            "output_start": round(self.output_start, 6),
            "output_end": round(self.output_end, 6),
            "source_role": self.source_role,
            "is_hero": self.is_hero,
            "scenes": [scene.as_dict() for scene in self.scenes],
            "audio": self.audio.as_dict(),
        }


@dataclass(frozen=True)
class TimelineManifest:
    """Deterministic source -> output mapping with quantized boundaries."""

    occurrences: tuple[TimelineOccurrence, ...]
    frame_rate: Fraction
    sample_rate: int
    source_duration: float
    output_duration: float
    output_frame_count: int
    output_sample_count: int
    quantization: str = "cumulative-boundaries"

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_rate": {
                "numerator": self.frame_rate.numerator,
                "denominator": self.frame_rate.denominator,
                "value": float(self.frame_rate),
            },
            "sample_rate": self.sample_rate,
            "source_duration": round(self.source_duration, 6),
            "output_duration": round(self.output_duration, 6),
            "output_frame_count": self.output_frame_count,
            "output_sample_count": self.output_sample_count,
            "quantization": self.quantization,
            "occurrences": [occurrence.as_dict() for occurrence in self.occurrences],
        }


@dataclass(frozen=True)
class AssAsset:
    """The canonical Stage 5.1 ASS asset referenced by a render request."""

    relative_path: str
    sha256: str
    event_count: int
    line_count: int
    policy_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "event_count": self.event_count,
            "line_count": self.line_count,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class CaptionEventSpec:
    """One canonical caption event used only for timeline-range validation."""

    event_id: str
    block_index: int
    start: float
    end: float

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "block_index": self.block_index,
            "start": round(self.start, 6),
            "end": round(self.end, 6),
        }


@dataclass(frozen=True)
class OmittedRequirement:
    """One authored block/slot deliberately omitted by source-core validation."""

    block_index: int | None
    block_type: str
    slot_kind: str
    reason_code: str

    def as_dict(self) -> dict[str, object]:
        return {
            "block_index": self.block_index,
            "block_type": self.block_type,
            "slot_kind": self.slot_kind,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class RuntimeIdentity:
    """Exact execution runtime identity.

    ``as_dict`` returns only the stable fingerprint fields. The concrete binary
    path, attempt directory, source path, and thread counts are execution-only
    and are deliberately excluded from every fingerprint (absolute deployment
    paths and job-scoped paths must never invalidate a render).
    """

    ffmpeg_version: str
    ffprobe_version: str
    libavformat_version: str
    libass_version: str
    font_family: str
    font_match: str
    compiler_version: str
    policy_version: str
    libavcodec_version: str = ""
    build_config_sha256: str = ""
    font_sha256: str = ""
    libass_sha256: str = ""
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    encoder_threads: int = 2
    filter_threads: int = 1
    filter_complex_threads: int = 1
    source_absolute_path: str = ""
    attempt_directory: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "ffmpeg_version": self.ffmpeg_version,
            "ffprobe_version": self.ffprobe_version,
            "libavformat_version": self.libavformat_version,
            "libavcodec_version": self.libavcodec_version,
            "libass_version": self.libass_version,
            "libass_sha256": self.libass_sha256,
            "font_family": self.font_family,
            "font_match": self.font_match,
            "font_sha256": self.font_sha256,
            "build_config_sha256": self.build_config_sha256,
            "compiler_version": self.compiler_version,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class RenderSpec:
    """The complete, validated Stage 5.2 render request (DB-independent)."""

    render_execution_id: str
    candidate_id: str
    source_id: str
    render_contract_id: str
    visual_plan_id: str
    artifact_purpose: str
    source_media_relative_path: str
    source_content_hash: str
    source_size_bytes: int
    source_mtime_ns: int
    source_duration: float
    source_frame_rate: Fraction
    display_width: int
    display_height: int
    encoded_width: int
    encoded_height: int
    rotation_degrees: int
    pixel_aspect_ratio: float
    output_width: int
    output_height: int
    output_frame_rate: Fraction
    output_profile: Mapping[str, object]
    delivery_profile_key: str
    plan_input_fingerprint: str
    plan_output_fingerprint: str
    ass: AssAsset
    caption_events: tuple[CaptionEventSpec, ...]
    occurrences: tuple[TimelineOccurrence, ...]
    omitted: tuple[OmittedRequirement, ...] = ()
    audio_channels: int = 2
    source_video_start_seconds: float = 0.0
    source_audio_start_seconds: float = 0.0

    @property
    def output_duration(self) -> float:
        if not self.occurrences:
            return 0.0
        return self.occurrences[-1].output_end

    def as_dict(self) -> dict[str, object]:
        return {
            "render_execution_id": self.render_execution_id,
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "render_contract_id": self.render_contract_id,
            "visual_plan_id": self.visual_plan_id,
            "artifact_purpose": self.artifact_purpose,
            "source_media_relative_path": self.source_media_relative_path,
            "source_content_hash": self.source_content_hash,
            "source_size_bytes": self.source_size_bytes,
            "source_mtime_ns": self.source_mtime_ns,
            "source_duration": round(self.source_duration, 6),
            "source_frame_rate": {
                "numerator": self.source_frame_rate.numerator,
                "denominator": self.source_frame_rate.denominator,
            },
            "output_geometry": {
                "width": self.output_width,
                "height": self.output_height,
                "rotation_degrees": self.rotation_degrees,
                "pixel_aspect_ratio": self.pixel_aspect_ratio,
            },
            "output_frame_rate": {
                "numerator": self.output_frame_rate.numerator,
                "denominator": self.output_frame_rate.denominator,
            },
            "source_stream_origin": {
                "video_start_seconds": round(self.source_video_start_seconds, 6),
                "audio_start_seconds": round(self.source_audio_start_seconds, 6),
                "audio_channels": self.audio_channels,
            },
            "delivery_profile_key": self.delivery_profile_key,
            "plan_input_fingerprint": self.plan_input_fingerprint,
            "plan_output_fingerprint": self.plan_output_fingerprint,
            "ass": self.ass.as_dict(),
            "caption_events": [event.as_dict() for event in self.caption_events],
            "occurrences": [occurrence.as_dict() for occurrence in self.occurrences],
            "omitted": [item.as_dict() for item in self.omitted],
        }


@dataclass(frozen=True)
class FilterNode:
    """One typed, allow-listed filter operation in a compiled graph segment."""

    kind: str
    inputs: tuple[str, ...]
    output: str
    params: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "inputs": list(self.inputs),
            "output": self.output,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class CompiledRender:
    """A deterministic compiled render ready to execute."""

    compiler_version: str
    fingerprint: str
    spec_fingerprint: str
    runtime_identity: RuntimeIdentity
    manifest: TimelineManifest
    output_relative_path: str
    ass_localized_name: str
    argv: tuple[str, ...]
    filtergraph: str
    filtergraph_relative_path: str
    nodes: tuple[FilterNode, ...]
    expected_output_duration: float
    expected_frame_count: int
    expected_sample_count: int
    expected: Mapping[str, object] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "compiler_version": self.compiler_version,
            "fingerprint": self.fingerprint,
            "spec_fingerprint": self.spec_fingerprint,
            "output_relative_path": self.output_relative_path,
            "ass_localized_name": self.ass_localized_name,
            "expected_output_duration": round(self.expected_output_duration, 6),
            "expected_frame_count": self.expected_frame_count,
            "expected_sample_count": self.expected_sample_count,
            "filtergraph_relative_path": self.filtergraph_relative_path,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class RenderArtifacts:
    """The produced artifact plus its normalized execution manifest."""

    output_path: Path
    output_relative_path: str
    sha256: str
    size_bytes: int
    probe: Mapping[str, object]
    manifest: Mapping[str, object]
    duration_seconds: float
    frame_count: int
    sample_count: int
    sample_rate: int
    channels: int
    stderr_tail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "output_relative_path": self.output_relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "duration_seconds": round(self.duration_seconds, 6),
            "frame_count": self.frame_count,
            "sample_count": self.sample_count,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


@dataclass(frozen=True)
class QCCheck:
    """One deterministic QC check outcome."""

    name: str
    status: str
    reason_code: str | None = None
    measured: Mapping[str, object] = field(default_factory=dict)
    expected: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"name": self.name, "status": self.status}
        if self.reason_code is not None:
            payload["reason_code"] = self.reason_code
        if self.measured:
            payload["measured"] = dict(self.measured)
        if self.expected:
            payload["expected"] = dict(self.expected)
        return payload


@dataclass(frozen=True)
class TechnicalQCResult:
    """Deterministic technical QC result (separate from lifecycle)."""

    status: str
    checks: tuple[QCCheck, ...]
    reason_codes: tuple[str, ...]
    measured: Mapping[str, object] = field(default_factory=dict)
    policy_version: str = ""

    @property
    def hard_failure(self) -> bool:
        return self.status == "FAIL"

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.status == "WARN")

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "policy_version": self.policy_version,
            "reason_codes": list(self.reason_codes),
            "checks": [check.as_dict() for check in self.checks],
            "measured": dict(self.measured),
        }


@dataclass
class AttemptContext:
    """Execution context for one render attempt (not a persisted type).

    ``ass_bytes`` carries the exact canonical Stage 5.1 ASS bytes so the runner
    can localize them under a controlled filename without re-serializing or
    re-planning captions.
    """

    attempt_directory: Path
    cancel_check: Callable[[], bool] = lambda: False
    progress_callback: Callable[[float], None] | None = None
    timeout_seconds: float = 0.0
    poll_seconds: float = 0.5
    ass_bytes: bytes = b""
    #: Absolute monotonic deadline shared by the whole expensive attempt
    #: (encode, post-encode probing, and QC). ``None`` falls back to a per-phase
    #: ``timeout_seconds`` budget when set.
    deadline: float | None = None

    def cancelled(self) -> bool:
        try:
            return bool(self.cancel_check())
        except Exception:
            return True

    def effective_deadline(self, started: float) -> float | None:
        """The absolute deadline for this attempt, if any."""

        if self.deadline is not None:
            return self.deadline
        if self.timeout_seconds > 0:
            return started + self.timeout_seconds
        return None


@dataclass(frozen=True)
class RenderRequestDraft:
    """The assembled request before fingerprinting/persistence."""

    spec: RenderSpec
    input_fingerprint: str
    artifact_purpose: str = CORE_SOURCE_VALIDATION


def default_audio_instruction() -> AudioInstruction:
    return AudioInstruction()


def sequence_of(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


__all__ = [
    "AssAsset",
    "AttemptContext",
    "AudioInstruction",
    "CaptionEventSpec",
    "CompiledRender",
    "CropKeyframeSpec",
    "FilterNode",
    "OmittedRequirement",
    "QCCheck",
    "RenderArtifacts",
    "RenderRequestDraft",
    "RenderSpec",
    "RuntimeIdentity",
    "SceneSpec",
    "TechnicalQCResult",
    "TimelineManifest",
    "TimelineOccurrence",
    "default_audio_instruction",
    "sequence_of",
]
