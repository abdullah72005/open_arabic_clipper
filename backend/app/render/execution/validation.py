"""Strict frozen-plan/ASS/media/profile validation for Stage 5.2.

Every check fails closed with a reason-coded :class:`RenderValidationError`.
Contradictory or unsupported scene/keyframe evidence is never guessed into
another framing mode. Preview helpers are intentionally not reused: they contain
lenient fallbacks and optional ASS handling that production execution must not
have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.composition.policy import FramingMode
from app.render.execution.policy import (
    ASS_EMPTY,
    ASS_HASH_MISMATCH,
    ASS_MALFORMED,
    ASS_METADATA_MISMATCH,
    ASS_MISSING,
    CONTRADICTORY_FRAMING_EVIDENCE,
    EXOTIC_PIXEL_ASPECT,
    KEYFRAME_GEOMETRY_INVALID,
    KEYFRAME_NON_MONOTONIC,
    KEYFRAME_OUT_OF_SCOPE,
    NO_RENDERABLE_OCCURRENCES,
    OCCURRENCE_BEYOND_MEDIA,
    OCCURRENCE_BOUNDS_INVALID,
    OUTPUT_PROFILE_HEIGHT,
    OUTPUT_PROFILE_WIDTH,
    QC_CAPTION_OUT_OF_RANGE,
    SCENE_BLOCK_MISMATCH,
    SCENE_BOUNDS_INVALID,
    SCENE_COVERAGE_GAP,
    SCENE_OVERLAP,
    SUPPORTED_ARTIFACT_PURPOSES,
    UNSUPPORTED_ARTIFACT_PURPOSE,
    UNSUPPORTED_DELIVERY_PROFILE,
    UNSUPPORTED_FRAMING_MODE,
    UNSUPPORTED_INTERPOLATION,
    UNSUPPORTED_OUTPUT_GEOMETRY,
    UNSUPPORTED_ROTATION,
    delivery_profile_for,
)
from app.render.execution.types import (
    AssAsset,
    CaptionEventSpec,
    RenderSpec,
    SceneSpec,
)

_COVERAGE_TOLERANCE = 0.05
_FORBIDDEN_INTERPOLATION = "smoothstep-ease"
_CROP_MODES = frozenset(
    {
        FramingMode.STATIC_CROP.value,
        FramingMode.TRACKED_CROP.value,
        FramingMode.MULTI_SUBJECT_FIT.value,
        FramingMode.CENTER_FALLBACK.value,
    }
)
_SUPPORTED_MODES = frozenset(mode.value for mode in FramingMode)
_SUPPORTED_ROTATIONS = frozenset({0, 90, 180, 270})


class RenderValidationError(ValueError):
    """A queue-time or pre-execution validation failure (reason-coded)."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class AssFileFacts:
    """File-derived ASS facts injected by the service or tests."""

    exists: bool
    sha256: str
    size_bytes: int
    dialogue_count: int
    has_canonical_header: bool


def _fail(reason_code: str) -> None:
    raise RenderValidationError(reason_code)


def _is_finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_spec(spec: RenderSpec) -> None:
    """Run every strict structural validation over a render spec."""

    if spec.artifact_purpose not in SUPPORTED_ARTIFACT_PURPOSES:
        _fail(UNSUPPORTED_ARTIFACT_PURPOSE)
    if delivery_profile_for(spec.delivery_profile_key) is None:
        _fail(UNSUPPORTED_DELIVERY_PROFILE)
    if spec.output_width != OUTPUT_PROFILE_WIDTH or spec.output_height != OUTPUT_PROFILE_HEIGHT:
        _fail(UNSUPPORTED_OUTPUT_GEOMETRY)
    if spec.rotation_degrees not in _SUPPORTED_ROTATIONS:
        _fail(UNSUPPORTED_ROTATION)
    if abs(spec.pixel_aspect_ratio - 1.0) > 0.01:
        _fail(EXOTIC_PIXEL_ASPECT)
    if spec.source_frame_rate <= 0 or spec.output_frame_rate <= 0:
        _fail(UNSUPPORTED_OUTPUT_GEOMETRY)
    if not spec.occurrences:
        _fail(NO_RENDERABLE_OCCURRENCES)

    cursor = 0.0
    for occurrence in spec.occurrences:
        if occurrence.source_end <= occurrence.source_start:
            _fail(OCCURRENCE_BOUNDS_INVALID)
        if occurrence.source_start < -1e-6:
            _fail(OCCURRENCE_BOUNDS_INVALID)
        if occurrence.source_end > spec.source_duration + 0.05:
            _fail(OCCURRENCE_BEYOND_MEDIA)
        if abs(occurrence.output_start - cursor) > 1e-4:
            _fail(OCCURRENCE_BOUNDS_INVALID)
        if occurrence.output_end <= occurrence.output_start:
            _fail(OCCURRENCE_BOUNDS_INVALID)
        cursor = occurrence.output_end
        _validate_occurrence_scenes(spec, occurrence)

    validate_captions(spec)


def _validate_occurrence_scenes(spec: RenderSpec, occurrence: object) -> None:
    from app.render.execution.types import TimelineOccurrence

    assert isinstance(occurrence, TimelineOccurrence)
    scenes = sorted(occurrence.scenes, key=lambda scene: scene.source_start)
    if not scenes:
        _fail(SCENE_COVERAGE_GAP)
    if scenes[0].source_start > occurrence.source_start + _COVERAGE_TOLERANCE:
        _fail(SCENE_COVERAGE_GAP)
    if scenes[-1].source_end < occurrence.source_end - _COVERAGE_TOLERANCE:
        _fail(SCENE_COVERAGE_GAP)
    previous_end: float | None = None
    for scene in scenes:
        if scene.block_index != occurrence.block_index:
            _fail(SCENE_BLOCK_MISMATCH)
        if scene.source_end <= scene.source_start:
            _fail(SCENE_BOUNDS_INVALID)
        if (
            scene.source_start < occurrence.source_start - _COVERAGE_TOLERANCE
            or scene.source_end > occurrence.source_end + _COVERAGE_TOLERANCE
        ):
            _fail(SCENE_BOUNDS_INVALID)
        if previous_end is not None:
            gap = scene.source_start - previous_end
            if gap > _COVERAGE_TOLERANCE:
                _fail(SCENE_COVERAGE_GAP)
            if gap < -_COVERAGE_TOLERANCE:
                _fail(SCENE_OVERLAP)
        previous_end = scene.source_end
        _validate_scene_keyframes(scene)


def _validate_scene_keyframes(scene: SceneSpec) -> None:
    mode = scene.framing_mode
    if mode not in _SUPPORTED_MODES:
        _fail(UNSUPPORTED_FRAMING_MODE)
    if mode in (FramingMode.SOURCE_AS_IS.value, FramingMode.BACKGROUND_FILL.value):
        return
    if scene.interpolation_policy not in {_FORBIDDEN_INTERPOLATION, "smoothstep-ease", ""}:
        _fail(UNSUPPORTED_INTERPOLATION)
    if mode not in _CROP_MODES:
        _fail(UNSUPPORTED_FRAMING_MODE)
    if not scene.crop_keyframes:
        _fail(CONTRADICTORY_FRAMING_EVIDENCE)
    previous_t: float | None = None
    for keyframe in scene.crop_keyframes:
        if not (
            _is_finite(keyframe.t)
            and _is_finite(keyframe.cx)
            and _is_finite(keyframe.cy)
            and _is_finite(keyframe.height_fraction)
        ):
            _fail(KEYFRAME_GEOMETRY_INVALID)
        if (
            keyframe.cx < -1e-6
            or keyframe.cx > 1.0 + 1e-6
            or keyframe.cy < -1e-6
            or keyframe.cy > 1.0 + 1e-6
            or keyframe.height_fraction <= 0.0
            or keyframe.height_fraction > 1.0 + 1e-6
        ):
            _fail(KEYFRAME_GEOMETRY_INVALID)
        if (
            keyframe.t < scene.source_start - _COVERAGE_TOLERANCE
            or keyframe.t > scene.source_end + _COVERAGE_TOLERANCE
        ):
            _fail(KEYFRAME_OUT_OF_SCOPE)
        if previous_t is not None and keyframe.t < previous_t - 1e-6:
            _fail(KEYFRAME_NON_MONOTONIC)
        previous_t = keyframe.t


def validate_captions(spec: RenderSpec) -> None:
    """Validate that every caption event stays inside a selected occurrence."""

    spans = [(occ.source_start, occ.source_end) for occ in spec.occurrences]
    for event in spec.caption_events:
        if not (_is_finite(event.start) and _is_finite(event.end)) or event.end < event.start:
            _fail(ASS_METADATA_MISMATCH)
        if not _inside_any_span(event.start, spans) or not _inside_any_span(event.end, spans):
            _fail(QC_CAPTION_OUT_OF_RANGE)


def _inside_any_span(t: float, spans: list[tuple[float, float]]) -> bool:
    for start, end in spans:
        if start - _COVERAGE_TOLERANCE <= t <= end + _COVERAGE_TOLERANCE:
            return True
    return False


def validate_ass_asset(asset: AssAsset, facts: AssFileFacts, *, event_count: int) -> None:
    """Strictly validate the referenced canonical ASS asset."""

    if not facts.exists:
        _fail(ASS_MISSING)
    if facts.size_bytes <= 0:
        _fail(ASS_EMPTY)
    if facts.sha256.lower() != (asset.sha256 or "").lower():
        _fail(ASS_HASH_MISMATCH)
    if not facts.has_canonical_header:
        _fail(ASS_MALFORMED)
    if asset.event_count != event_count:
        _fail(ASS_METADATA_MISMATCH)
    # A caption event may expand into multiple Dialogue states; require at least
    # one Dialogue row per caption event (never equality).
    if facts.dialogue_count < asset.event_count:
        _fail(ASS_METADATA_MISMATCH)
    if asset.line_count < 0:
        _fail(ASS_METADATA_MISMATCH)
    if asset.event_count == 0 and facts.dialogue_count != 0:
        _fail(ASS_METADATA_MISMATCH)


def ass_file_facts(data: bytes, *, exists: bool = True) -> AssFileFacts:
    """Derive strict ASS file facts from canonical bytes (no re-serialization)."""

    import hashlib

    if not exists:
        return AssFileFacts(False, "", 0, 0, False)
    text = data.decode("utf-8", errors="strict")
    lines = text.splitlines()
    canonical = (
        "[Script Info]" in lines
        and "[V4+ Styles]" in lines
        and "[Events]" in lines
        and any(line.strip() == "PlayResX: 1080" for line in lines)
        and any(line.strip() == "PlayResY: 1920" for line in lines)
    )
    dialogue_count = sum(1 for line in lines if line.startswith("Dialogue:"))
    return AssFileFacts(
        exists=True,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        dialogue_count=dialogue_count,
        has_canonical_header=canonical,
    )


def validate_caption_event_specs(events: tuple[CaptionEventSpec, ...]) -> None:
    for event in events:
        if not (_is_finite(event.start) and _is_finite(event.end)):
            _fail(ASS_METADATA_MISMATCH)


__all__ = [
    "AssFileFacts",
    "RenderValidationError",
    "ass_file_facts",
    "validate_ass_asset",
    "validate_caption_event_specs",
    "validate_captions",
    "validate_spec",
]
