"""Deterministic Stage 5.1 per-scene framing selection and crop keyframes.

This module is pure, CPU-local, and provider-free: it never decodes audio or
video, never loads a model, and never touches the network. It consumes anonymous
per-scene face tracks plus display geometry and produces:

- one :class:`FramingDecision` per scene through :func:`select_framing_mode`,
  using a short-circuiting precedence order that always records a mode and
  closed evidence codes; and
- a compact, clamped crop-keyframe path through :func:`build_crop_keyframes`
  (never one keyframe per decoded frame) with deterministic exponential target
  smoothing, a dead zone with hysteresis, bounded pan/zoom rates, detection-loss
  fallback, a hard per-scene keyframe cap, and hero protection.

Coordinates are display-normalized with origin top-left, x to the right, and y
down. Every crop this module emits is clamped through
:func:`app.composition.geometry.clamp_crop`: an oversized crop is reduced and
pinned, never silently rescaled.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from app.composition.geometry import aspect_within_tolerance, clamp_crop
from app.composition.policy import (
    FramingEvidence,
    FramingMode,
    Stage51Config,
    framing_for_bounded_distance,
)
from app.composition.tracking import persistent_tracks
from app.composition.types import (
    CropKeyframe,
    DisplayGeometry,
    FaceTrack,
    FramingDecision,
    Scene,
)

_TARGET_ASPECT = 9.0 / 16.0
_STATIC_MOTION_THRESHOLD = 0.02
_SMOOTHING_RATE = 6.0
_HYSTERESIS_RELEASE = 0.5
_MIN_DT = 1e-3
_HERO_PAN_SCALE = 0.5
_HERO_ZOOM_TOLERANCE = 0.02
_HORIZONTAL_MARGIN_FRACTION = 0.02
_MAX_FACE_TRACKS = 3
_HEIGHT_GRID_STEPS = 32
_EDGE_EPSILON = 1e-6
_FALLBACK_TARGET_FACE_HEIGHT = 0.38
_KEYFRAMES_SIMPLIFIED = "KEYFRAMES_SIMPLIFIED"

_STATIC_MODES = frozenset(
    {
        FramingMode.SOURCE_AS_IS.value,
        FramingMode.STATIC_CROP.value,
        FramingMode.MULTI_SUBJECT_FIT.value,
        FramingMode.CENTER_FALLBACK.value,
        FramingMode.BACKGROUND_FILL.value,
    }
)

_VALID_MODES = frozenset(mode.value for mode in FramingMode)


@dataclass(frozen=True)
class _CropTarget:
    """One clamped 9:16 crop window as normalized center plus height fraction."""

    center_x: float
    center_y: float
    height_fraction: float


@dataclass(frozen=True)
class _MultiFitResult:
    """Outcome of a bounded multi-subject fit attempt."""

    target: _CropTarget | None
    subject_too_wide: bool


def _as_mode(mode: FramingMode | str) -> str:
    return mode.value if isinstance(mode, FramingMode) else mode


def _valid_geometry(geometry: DisplayGeometry) -> bool:
    if geometry.display_width <= 0 or geometry.display_height <= 0:
        return False
    aspect = geometry.display_aspect
    return math.isfinite(aspect) and aspect > 0.0


def _crop_rect(
    center_x: float,
    center_y: float,
    height_fraction: float,
    geometry: DisplayGeometry,
) -> tuple[float, float, float, float]:
    """Return the clamped 9:16 display-pixel crop rectangle for a target."""

    height_px = height_fraction * geometry.display_height
    width_px = height_px * _TARGET_ASPECT
    x = center_x * geometry.display_width - width_px / 2.0
    y = center_y * geometry.display_height - height_px / 2.0
    clamped_x, clamped_y, clamped_width, clamped_height = clamp_crop(
        x,
        y,
        width_px,
        height_px,
        float(geometry.display_width),
        float(geometry.display_height),
    )
    return (
        float(clamped_x),
        float(clamped_y),
        float(clamped_width),
        float(clamped_height),
    )


def _clamp_target(
    center_x: float,
    center_y: float,
    height_fraction: float,
    geometry: DisplayGeometry,
) -> _CropTarget:
    x, y, width, height = _crop_rect(center_x, center_y, height_fraction, geometry)
    return _CropTarget(
        center_x=(x + width / 2.0) / geometry.display_width,
        center_y=(y + height / 2.0) / geometry.display_height,
        height_fraction=height / geometry.display_height,
    )


def _center_target(geometry: DisplayGeometry) -> _CropTarget:
    return _clamp_target(0.5, 0.5, 1.0, geometry)


def _box_contained(
    target: _CropTarget,
    box: tuple[float, float, float, float],
    geometry: DisplayGeometry,
) -> bool:
    x, y, width, height = _crop_rect(
        target.center_x, target.center_y, target.height_fraction, geometry
    )
    box_x, box_y, box_w, box_h = _to_pixels(box, geometry)
    return (
        box_x >= x - _EDGE_EPSILON
        and box_y >= y - _EDGE_EPSILON
        and box_x + box_w <= x + width + _EDGE_EPSILON
        and box_y + box_h <= y + height + _EDGE_EPSILON
    )


def _to_pixels(
    box: tuple[float, float, float, float],
    geometry: DisplayGeometry,
) -> tuple[float, float, float, float]:
    """Convert a normalized (x, y, w, h) box into display pixels."""

    box_x, box_y, box_w, box_h = box
    return (
        box_x * geometry.display_width,
        box_y * geometry.display_height,
        box_w * geometry.display_width,
        box_h * geometry.display_height,
    )


def _margins_satisfied(
    target: _CropTarget,
    box: tuple[float, float, float, float],
    config: Stage51Config,
    geometry: DisplayGeometry,
) -> bool:
    x, y, width, height = _crop_rect(
        target.center_x, target.center_y, target.height_fraction, geometry
    )
    box_x, box_y, box_w, box_h = _to_pixels(box, geometry)
    horizontal = _HORIZONTAL_MARGIN_FRACTION * width
    headroom = config.headroom_fraction * height
    chin = config.chin_margin_fraction * height
    return (
        box_x >= x + horizontal - _EDGE_EPSILON
        and box_x + box_w <= x + width - horizontal + _EDGE_EPSILON
        and box_y >= y + headroom - _EDGE_EPSILON
        and box_y + box_h <= y + height - chin + _EDGE_EPSILON
    )


def _height_candidates(base: float, config: Stage51Config) -> tuple[float, ...]:
    """Deterministic ascending candidate crop heights around a base target."""

    lower = max(_EDGE_EPSILON, min(1.0, config.min_crop_height_fraction))
    values = {min(1.0, max(lower, base))}
    for step in range(_HEIGHT_GRID_STEPS + 1):
        value = lower + (1.0 - lower) * step / _HEIGHT_GRID_STEPS
        values.add(min(1.0, max(lower, value)))
    return tuple(sorted(values))


def _target_from_box(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    config: Stage51Config,
    geometry: DisplayGeometry,
) -> _CropTarget | None:
    """Fit one face box into a clamped 9:16 crop, preferring margin compliance."""

    if (
        not math.isfinite(center_x)
        or not math.isfinite(center_y)
        or not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0.0
        or height <= 0.0
    ):
        return None

    target_fraction = config.target_face_height_fraction
    if not math.isfinite(target_fraction) or target_fraction <= 0.0:
        target_fraction = _FALLBACK_TARGET_FACE_HEIGHT
    base = height / target_fraction
    vertical = 1.0 - config.headroom_fraction - config.chin_margin_fraction
    if vertical > _EDGE_EPSILON:
        base = max(base, height / vertical)
    horizontal = _TARGET_ASPECT * (1.0 - 2.0 * _HORIZONTAL_MARGIN_FRACTION)
    if horizontal > _EDGE_EPSILON:
        base = max(base, width / horizontal)
    base = min(1.0, max(config.min_crop_height_fraction, base))

    face_top = center_y - height / 2.0
    box = (center_x - width / 2.0, face_top, width, height)

    best_margin: _CropTarget | None = None
    best_margin_distance = math.inf
    best_contained: _CropTarget | None = None
    best_contained_distance = math.inf
    for candidate in _height_candidates(base, config):
        crop_center_y = face_top - config.headroom_fraction * candidate + candidate / 2.0
        clamped = _clamp_target(center_x, crop_center_y, candidate, geometry)
        if not _box_contained(clamped, box, geometry):
            continue
        distance = abs(candidate - base)
        if _margins_satisfied(clamped, box, config, geometry):
            if distance < best_margin_distance:
                best_margin = clamped
                best_margin_distance = distance
        elif distance < best_contained_distance:
            best_contained = clamped
            best_contained_distance = distance
    if best_margin is not None:
        return best_margin
    return best_contained


def _mean_box(track: FaceTrack) -> tuple[float, float, float, float] | None:
    samples = track.samples
    if not samples:
        return None
    count = float(len(samples))
    center_x = sum(sample.cx for sample in samples) / count
    center_y = sum(sample.cy for sample in samples) / count
    width = sum(sample.w for sample in samples) / count
    height = sum(sample.h for sample in samples) / count
    return (center_x, center_y, width, height)


def _union_box(tracks: Sequence[FaceTrack]) -> tuple[float, float, float, float] | None:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    found = False
    for track in tracks:
        for sample in track.samples:
            found = True
            min_x = min(min_x, sample.cx - sample.w / 2.0)
            min_y = min(min_y, sample.cy - sample.h / 2.0)
            max_x = max(max_x, sample.cx + sample.w / 2.0)
            max_y = max(max_y, sample.cy + sample.h / 2.0)
    if not found:
        return None
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def _center_motion(track: FaceTrack) -> float:
    samples = track.samples
    if len(samples) < 2:
        return 0.0
    xs = [sample.cx for sample in samples]
    ys = [sample.cy for sample in samples]
    return float(max(max(xs) - min(xs), max(ys) - min(ys)))


def _confidence(track: FaceTrack | None) -> float:
    if track is None:
        return 0.0
    return float(min(1.0, max(0.0, track.mean_score)))


def _screen_content_requested(
    evidence: bool | Mapping[str, object] | None,
) -> bool:
    """Interpret the optional caller-supplied screen-content flag without decoding."""

    if evidence is None:
        return False
    if isinstance(evidence, bool):
        return evidence
    for key in ("screen_content", "is_screen_content", "screen", "information_dense", "dense"):
        value = evidence.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return float(value) > 0.0
    density = evidence.get("density")
    if isinstance(density, int | float) and not isinstance(density, bool):
        return float(density) >= 0.5
    return False


def _multi_subject_fit(
    tracks: Sequence[FaceTrack],
    config: Stage51Config,
    geometry: DisplayGeometry,
) -> _MultiFitResult:
    union = _union_box(tracks)
    if union is None:
        return _MultiFitResult(target=None, subject_too_wide=False)

    union_x, union_y, union_w, union_h = union
    margin = config.chin_margin_fraction * union_h
    expanded = (
        union_x - margin,
        union_y - config.headroom_fraction * union_h,
        union_w + 2.0 * margin,
        union_h * (1.0 + config.headroom_fraction + config.chin_margin_fraction),
    )
    box = (
        min(max(expanded[0], 0.0), 1.0),
        min(max(expanded[1], 0.0), 1.0),
        expanded[2],
        expanded[3],
    )
    box_x, box_y, box_w, box_h = box
    display_width = float(geometry.display_width)
    display_height = float(geometry.display_height)
    required_height_px = max(box_h * display_height, box_w * display_width * 16.0 / 9.0)
    required_height = required_height_px / display_height
    if required_height > 1.0 + _EDGE_EPSILON:
        return _MultiFitResult(target=None, subject_too_wide=True)

    center_x = box_x + box_w / 2.0
    center_y = box_y + box_h / 2.0
    lower = max(config.min_crop_height_fraction, required_height)
    for candidate in _height_candidates(min(1.0, lower), config):
        if candidate < lower - _EDGE_EPSILON:
            continue
        target = _clamp_target(center_x, center_y, candidate, geometry)
        if _box_contained(target, box, geometry):
            return _MultiFitResult(target=target, subject_too_wide=False)
    return _MultiFitResult(target=None, subject_too_wide=False)


def select_framing_mode(
    *,
    block_index: int,
    scene_start: float,
    scene_end: float,
    tracks: Sequence[FaceTrack],
    config: Stage51Config,
    geometry: DisplayGeometry,
    screen_content_evidence: bool | Mapping[str, object] | None = None,
    is_hero: bool = False,
) -> FramingDecision:
    """Choose one deterministic framing mode for a scene.

    Precedence is short-circuiting: vertical/no-crop, taller source, no-face
    screen content, no-face low density, too-small subject, single subject
    (tracked or static), multi-subject fit, then a conservative fallback.
    Geometry or detector failure degrades to ``CENTER_FALLBACK``.
    """

    hero_evidence = (FramingEvidence.HERO_PROTECTION_APPLIED.value,) if is_hero else ()
    try:
        return _select_framing_mode(
            block_index=block_index,
            scene_start=scene_start,
            scene_end=scene_end,
            tracks=tracks,
            config=config,
            geometry=geometry,
            screen_content_evidence=screen_content_evidence,
            is_hero=is_hero,
            hero_evidence=hero_evidence,
        )
    except (ArithmeticError, TypeError, ValueError):
        return FramingDecision(
            mode=FramingMode.CENTER_FALLBACK.value,
            evidence=(*hero_evidence, FramingEvidence.NO_FACE.value),
            fallback_parameters={"reason": "FRAMING_FAILURE", "block_index": block_index},
            protected=is_hero,
        )


def _select_framing_mode(
    *,
    block_index: int,
    scene_start: float,
    scene_end: float,
    tracks: Sequence[FaceTrack],
    config: Stage51Config,
    geometry: DisplayGeometry,
    screen_content_evidence: bool | Mapping[str, object] | None,
    is_hero: bool,
    hero_evidence: tuple[str, ...],
) -> FramingDecision:
    if not _valid_geometry(geometry):
        return FramingDecision(
            mode=FramingMode.CENTER_FALLBACK.value,
            evidence=(*hero_evidence, FramingEvidence.NO_FACE.value),
            fallback_parameters={"reason": "INVALID_GEOMETRY", "block_index": block_index},
            protected=is_hero,
        )

    aspect = geometry.display_aspect
    if aspect_within_tolerance(aspect, _TARGET_ASPECT):
        return FramingDecision(
            mode=FramingMode.SOURCE_AS_IS.value,
            evidence=(*hero_evidence, FramingEvidence.SOURCE_ALREADY_VERTICAL.value),
            fallback_parameters={"display_aspect": round(aspect, 6)},
            protected=is_hero,
        )
    if aspect < _TARGET_ASPECT:
        return FramingDecision(
            mode=FramingMode.BACKGROUND_FILL.value,
            evidence=(*hero_evidence, FramingEvidence.SOURCE_ALREADY_VERTICAL.value),
            fallback_parameters={
                "display_aspect": round(aspect, 6),
                "reason": "SOURCE_TALLER_THAN_TARGET",
            },
            protected=is_hero,
        )

    persistent = persistent_tracks(tracks, config)
    if not persistent:
        evidence = list(hero_evidence)
        if not config.detector_enabled:
            evidence.append(FramingEvidence.DETECTOR_UNAVAILABLE.value)
        evidence.append(
            FramingEvidence.FACE_TRACK_UNSTABLE.value if tracks else FramingEvidence.NO_FACE.value
        )
        if _screen_content_requested(screen_content_evidence):
            return FramingDecision(
                mode=FramingMode.BACKGROUND_FILL.value,
                evidence=(*evidence, FramingEvidence.SCREEN_CONTENT.value),
                fallback_parameters={
                    "persistent_face_count": 0,
                    "track_count": len(tracks),
                },
                protected=is_hero,
            )
        return FramingDecision(
            mode=FramingMode.CENTER_FALLBACK.value,
            evidence=tuple(evidence),
            fallback_parameters={
                "persistent_face_count": 0,
                "track_count": len(tracks),
            },
            protected=is_hero,
        )

    largest = persistent[0]
    face_height = largest.mean_face_height_fraction
    if face_height < config.face_min_height_fraction:
        return FramingDecision(
            mode=FramingMode.BACKGROUND_FILL.value,
            evidence=(*hero_evidence, FramingEvidence.SUBJECT_TOO_SMALL.value),
            fallback_parameters={
                "face_height_fraction": round(face_height, 6),
                "face_min_height_fraction": config.face_min_height_fraction,
            },
            protected=is_hero,
        )

    if len(persistent) >= 2:
        considered = persistent[:_MAX_FACE_TRACKS]
        result = _multi_subject_fit(considered, config, geometry)
        if result.target is not None:
            return FramingDecision(
                mode=FramingMode.MULTI_SUBJECT_FIT.value,
                evidence=(*hero_evidence, FramingEvidence.MULTIPLE_FACES.value),
                fallback_parameters={"persistent_face_count": len(persistent)},
                protected=is_hero,
            )
        evidence = [
            *hero_evidence,
            FramingEvidence.MULTIPLE_FACES.value,
            FramingEvidence.IMPORTANT_CONTENT_WOULD_BE_CROPPED.value,
        ]
        if result.subject_too_wide:
            evidence.append(FramingEvidence.SUBJECT_TOO_WIDE.value)
        return FramingDecision(
            mode=FramingMode.BACKGROUND_FILL.value,
            evidence=tuple(evidence),
            fallback_parameters={
                "persistent_face_count": len(persistent),
                "reason": "MULTI_SUBJECT_DOES_NOT_FIT",
            },
            protected=is_hero,
        )

    motion = _center_motion(largest)
    mode = (
        FramingMode.TRACKED_CROP.value
        if motion > _STATIC_MOTION_THRESHOLD
        else FramingMode.STATIC_CROP.value
    )
    return FramingDecision(
        mode=mode,
        evidence=(*hero_evidence, FramingEvidence.SINGLE_PERSISTENT_FACE.value),
        fallback_parameters={
            "track_id": largest.track_id,
            "motion": round(motion, 6),
            "face_height_fraction": round(face_height, 6),
            "scene_start": round(scene_start, 4),
            "scene_end": round(scene_end, 4),
        },
        protected=is_hero,
    )


def _make_keyframe(
    source_time: float,
    mode: str,
    target: _CropTarget,
    evidence: tuple[str, ...],
    confidence: float,
) -> CropKeyframe:
    return CropKeyframe(
        source_time=source_time,
        mode=mode,
        center_x=target.center_x,
        center_y=target.center_y,
        height_fraction=target.height_fraction,
        confidence=min(1.0, max(0.0, confidence)),
        evidence=evidence,
    )


def _static_target(
    mode: str,
    tracks: Sequence[FaceTrack],
    config: Stage51Config,
    geometry: DisplayGeometry,
) -> tuple[str, _CropTarget, tuple[str, ...], float]:
    """Resolve one static-mode keyframe target (mode, crop, evidence, confidence)."""

    if mode == FramingMode.SOURCE_AS_IS.value:
        return (
            mode,
            _center_target(geometry),
            (FramingEvidence.SOURCE_ALREADY_VERTICAL.value,),
            1.0,
        )

    persistent = persistent_tracks(tracks, config)

    if mode == FramingMode.STATIC_CROP.value:
        subject = persistent[0] if persistent else None
        if subject is None:
            return (
                FramingMode.CENTER_FALLBACK.value,
                _center_target(geometry),
                (FramingEvidence.FACE_TRACK_UNSTABLE.value,),
                0.0,
            )
        box = _mean_box(subject)
        target = (
            None
            if box is None
            else _target_from_box(box[0], box[1], box[2], box[3], config, geometry)
        )
        if target is None:
            return (
                FramingMode.BACKGROUND_FILL.value,
                _center_target(geometry),
                (FramingEvidence.FACE_NEAR_SOURCE_EDGE.value,),
                0.0,
            )
        return (
            mode,
            target,
            (FramingEvidence.SINGLE_PERSISTENT_FACE.value,),
            _confidence(subject),
        )

    if mode == FramingMode.MULTI_SUBJECT_FIT.value:
        result = _multi_subject_fit(persistent[:_MAX_FACE_TRACKS], config, geometry)
        if result.target is not None:
            return (
                mode,
                result.target,
                (FramingEvidence.MULTIPLE_FACES.value,),
                _confidence(persistent[0] if persistent else None),
            )
        evidence = [
            FramingEvidence.MULTIPLE_FACES.value,
            FramingEvidence.IMPORTANT_CONTENT_WOULD_BE_CROPPED.value,
        ]
        if result.subject_too_wide:
            evidence.append(FramingEvidence.SUBJECT_TOO_WIDE.value)
        return (
            FramingMode.BACKGROUND_FILL.value,
            _center_target(geometry),
            tuple(evidence),
            0.0,
        )

    if mode == FramingMode.CENTER_FALLBACK.value:
        return (
            mode,
            _center_target(geometry),
            (FramingEvidence.NO_FACE.value,),
            _confidence(persistent[0] if persistent else None),
        )

    return (FramingMode.BACKGROUND_FILL.value, _center_target(geometry), (), 0.0)


def _differs(first: _CropTarget, second: _CropTarget, epsilon: float) -> bool:
    return (
        abs(first.center_x - second.center_x) > epsilon
        or abs(first.center_y - second.center_y) > epsilon
        or abs(first.height_fraction - second.height_fraction) > epsilon
    )


def _path_delta(first: CropKeyframe, second: CropKeyframe, third: CropKeyframe) -> float:
    value = (
        abs(first.center_x - second.center_x)
        + abs(first.center_y - second.center_y)
        + abs(second.center_x - third.center_x)
        + abs(second.center_y - third.center_y)
        + abs(first.height_fraction - second.height_fraction)
        + abs(second.height_fraction - third.height_fraction)
    )
    return float(value)


def _finalize(keyframes: Sequence[CropKeyframe]) -> tuple[CropKeyframe, ...]:
    ordered: list[CropKeyframe] = []
    for keyframe in keyframes:
        if ordered and abs(keyframe.source_time - ordered[-1].source_time) <= _EDGE_EPSILON:
            ordered[-1] = keyframe
        else:
            ordered.append(keyframe)
    ordered.sort(key=lambda keyframe: keyframe.source_time)
    return tuple(ordered)


def _cap_keyframes(
    keyframes: tuple[CropKeyframe, ...],
    config: Stage51Config,
) -> tuple[CropKeyframe, ...]:
    cap = config.max_keyframes_per_scene
    if cap < 2 or len(keyframes) <= cap:
        return keyframes
    working = list(keyframes)
    while len(working) > cap:
        best_index = -1
        best_delta = math.inf
        for index in range(1, len(working) - 1):
            delta = _path_delta(working[index - 1], working[index], working[index + 1])
            if delta < best_delta - _EDGE_EPSILON:
                best_delta = delta
                best_index = index
        if best_index < 0:
            break
        del working[best_index]
    if working:
        first = working[0]
        evidence = tuple(first.evidence)
        if not any(item == _KEYFRAMES_SIMPLIFIED for item in evidence):
            evidence = (*evidence, _KEYFRAMES_SIMPLIFIED)
        working[0] = replace(first, evidence=evidence)
    return tuple(working)


def _tracked_keyframes(
    *,
    scene_start: float,
    scene_end: float,
    tracks: Sequence[FaceTrack],
    config: Stage51Config,
    geometry: DisplayGeometry,
    is_hero: bool,
) -> tuple[CropKeyframe, ...]:
    hero_evidence = (FramingEvidence.HERO_PROTECTION_APPLIED.value,) if is_hero else ()
    base_evidence = (FramingEvidence.SINGLE_PERSISTENT_FACE.value,)
    persistent = persistent_tracks(tracks, config)
    subject = persistent[0] if persistent else None
    if subject is None or not subject.samples:
        target = _center_target(geometry)
        evidence = (*base_evidence, FramingEvidence.FACE_TRACK_UNSTABLE.value, *hero_evidence)
        return (
            _make_keyframe(scene_start, FramingMode.CENTER_FALLBACK.value, target, evidence, 0.0),
            _make_keyframe(scene_end, FramingMode.CENTER_FALLBACK.value, target, evidence, 0.0),
        )

    samples = sorted(
        subject.samples,
        key=lambda sample: (sample.source_time, sample.cx, sample.cy, sample.h, sample.w),
    )
    first_sample = samples[0]
    first = _target_from_box(
        first_sample.cx, first_sample.cy, first_sample.w, first_sample.h, config, geometry
    )
    if first is None:
        target = _center_target(geometry)
        evidence = (*base_evidence, FramingEvidence.FACE_NEAR_SOURCE_EDGE.value, *hero_evidence)
        return (
            _make_keyframe(scene_start, FramingMode.BACKGROUND_FILL.value, target, evidence, 0.0),
            _make_keyframe(scene_end, FramingMode.BACKGROUND_FILL.value, target, evidence, 0.0),
        )

    confidence = _confidence(subject)
    hold = config.detection_hold_seconds()
    min_hold = config.min_hold_seconds
    current = first
    current_evidence = (*base_evidence, *hero_evidence)
    current_confidence = confidence
    panning = False
    last_emit_time = scene_start
    last_emitted = first
    keyframes: list[CropKeyframe] = [
        _make_keyframe(
            scene_start, FramingMode.TRACKED_CROP.value, first, current_evidence, confidence
        )
    ]

    previous_time = first_sample.source_time
    for sample in samples[1:]:
        gap = sample.source_time - previous_time
        if gap > hold:
            hold_end = previous_time + hold
            conservative = _center_target(geometry)
            loss_evidence = (
                *base_evidence,
                FramingEvidence.DETECTION_LOST.value,
                *hero_evidence,
            )
            if hold_end > last_emit_time + _EDGE_EPSILON:
                keyframes.append(
                    _make_keyframe(
                        hold_end,
                        FramingMode.TRACKED_CROP.value,
                        conservative,
                        loss_evidence,
                        0.0,
                    )
                )
                last_emit_time = hold_end
                last_emitted = conservative
                current = conservative
                current_evidence = loss_evidence
                current_confidence = 0.0
                panning = False

        delta_time = sample.source_time - previous_time
        previous_time = sample.source_time
        if delta_time <= 0.0:
            delta_time = _MIN_DT

        subject_target = _target_from_box(
            sample.cx, sample.cy, sample.w, sample.h, config, geometry
        )
        fallback = subject_target is None
        if subject_target is None:
            target = _center_target(geometry)
            step_evidence = (
                *base_evidence,
                FramingEvidence.DETECTION_LOST.value,
                *hero_evidence,
            )
            step_confidence = 0.0
        else:
            target = subject_target
            step_evidence = (*base_evidence, *hero_evidence)
            step_confidence = confidence

        alpha = 1.0 if fallback else 1.0 - math.exp(-_SMOOTHING_RATE * delta_time)
        smooth_x = current.center_x + (target.center_x - current.center_x) * alpha
        smooth_y = current.center_y + (target.center_y - current.center_y) * alpha
        smooth_height = (
            current.height_fraction + (target.height_fraction - current.height_fraction) * alpha
        )

        if not fallback:
            displacement = math.hypot(smooth_x - current.center_x, smooth_y - current.center_y)
            dead_zone = config.dead_zone_fraction * current.height_fraction
            if panning:
                if displacement < dead_zone * _HYSTERESIS_RELEASE:
                    panning = False
            elif displacement > dead_zone:
                panning = True
            if not panning:
                smooth_x = current.center_x
                smooth_y = current.center_y

        distance = math.hypot(smooth_x - current.center_x, smooth_y - current.center_y)
        max_pan = config.max_pan_velocity_per_second * current.height_fraction * delta_time
        if is_hero:
            max_pan *= _HERO_PAN_SCALE
        if distance > max_pan and max_pan > 0.0 and distance > 0.0:
            eased = framing_for_bounded_distance(max_pan / distance)
            new_x = current.center_x + (smooth_x - current.center_x) * eased
            new_y = current.center_y + (smooth_y - current.center_y) * eased
        else:
            new_x, new_y = smooth_x, smooth_y

        delta_height = smooth_height - current.height_fraction
        max_zoom = config.max_zoom_rate_per_second * delta_time
        if is_hero and abs(delta_height) > _HERO_ZOOM_TOLERANCE:
            delta_height = 0.0
        elif abs(delta_height) > max_zoom:
            delta_height = max_zoom if delta_height > 0.0 else -max_zoom
        new_height = min(
            1.0,
            max(config.min_crop_height_fraction, current.height_fraction + delta_height),
        )

        clamped = _clamp_target(new_x, new_y, new_height, geometry)
        current = clamped
        current_evidence = step_evidence
        current_confidence = step_confidence

        if sample.source_time - last_emit_time >= min_hold - _EDGE_EPSILON:
            if _differs(clamped, last_emitted, config.dedup_epsilon()):
                keyframes.append(
                    _make_keyframe(
                        sample.source_time,
                        FramingMode.TRACKED_CROP.value,
                        clamped,
                        step_evidence,
                        step_confidence,
                    )
                )
                last_emit_time = sample.source_time
                last_emitted = clamped

    last_time = samples[-1].source_time
    if scene_end - last_time > hold:
        hold_end = last_time + hold
        conservative = _center_target(geometry)
        loss_evidence = (
            *base_evidence,
            FramingEvidence.DETECTION_LOST.value,
            *hero_evidence,
        )
        if hold_end > last_emit_time + _EDGE_EPSILON:
            keyframes.append(
                _make_keyframe(
                    hold_end,
                    FramingMode.TRACKED_CROP.value,
                    conservative,
                    loss_evidence,
                    0.0,
                )
            )
            last_emit_time = hold_end
            last_emitted = conservative
            current = conservative
            current_evidence = loss_evidence
            current_confidence = 0.0

    if scene_end > last_emit_time + _EDGE_EPSILON:
        keyframes.append(
            _make_keyframe(
                scene_end,
                FramingMode.TRACKED_CROP.value,
                current,
                current_evidence,
                current_confidence,
            )
        )

    return _cap_keyframes(_finalize(keyframes), config)


def build_crop_keyframes(
    *,
    mode: FramingMode | str,
    scene_start: float,
    scene_end: float,
    tracks: Sequence[FaceTrack],
    config: Stage51Config,
    geometry: DisplayGeometry,
    is_hero: bool = False,
) -> tuple[CropKeyframe, ...]:
    """Build a compact, clamped crop-keyframe path for one scene.

    Static modes emit exactly two identical keyframes (start hold + end hold).
    ``TRACKED_CROP`` emits a smoothed, rate-limited, detection-loss-aware path.
    """

    mode_value = _as_mode(mode)
    start = min(scene_start, scene_end)
    end = max(scene_start, scene_end)
    hero_evidence = (FramingEvidence.HERO_PROTECTION_APPLIED.value,) if is_hero else ()

    if not _valid_geometry(geometry):
        target = _CropTarget(0.5, 0.5, 1.0)
        evidence = (*hero_evidence, FramingEvidence.NO_FACE.value)
        return (
            _make_keyframe(start, FramingMode.CENTER_FALLBACK.value, target, evidence, 0.0),
            _make_keyframe(end, FramingMode.CENTER_FALLBACK.value, target, evidence, 0.0),
        )

    if mode_value in _STATIC_MODES:
        resolved_mode, target, evidence, confidence = _static_target(
            mode_value, tracks, config, geometry
        )
        resolved_evidence = (*evidence, *hero_evidence)
        return (
            _make_keyframe(start, resolved_mode, target, resolved_evidence, confidence),
            _make_keyframe(end, resolved_mode, target, resolved_evidence, confidence),
        )

    return _tracked_keyframes(
        scene_start=start,
        scene_end=end,
        tracks=tracks,
        config=config,
        geometry=geometry,
        is_hero=is_hero,
    )


def build_scene(
    *,
    scene_index: int,
    block_index: int,
    scene_start: float,
    scene_end: float,
    cut_start: bool,
    mode: FramingDecision | FramingMode | str,
    tracks: Sequence[FaceTrack],
    config: Stage51Config,
    geometry: DisplayGeometry,
    is_hero: bool = False,
) -> Scene:
    """Assemble one :class:`Scene` from a framing decision or a bare mode."""

    hero_evidence = (FramingEvidence.HERO_PROTECTION_APPLIED.value,) if is_hero else ()
    if isinstance(mode, FramingDecision):
        decision = mode
        mode_value = decision.mode
    else:
        mode_value = mode.value if isinstance(mode, FramingMode) else mode
        if mode_value not in _VALID_MODES:
            mode_value = FramingMode.CENTER_FALLBACK.value
        decision = FramingDecision(mode=mode_value, evidence=hero_evidence, protected=is_hero)

    keyframes = build_crop_keyframes(
        mode=mode_value,
        scene_start=scene_start,
        scene_end=scene_end,
        tracks=tracks,
        config=config,
        geometry=geometry,
        is_hero=is_hero,
    )
    return Scene(
        scene_index=scene_index,
        block_index=block_index,
        source_start=scene_start,
        source_end=scene_end,
        cut_start=cut_start,
        framing=decision,
        tracks=tuple(tracks),
        crop_keyframes=keyframes,
        interpolation_policy="smoothstep-ease",
    )


__all__ = [
    "build_crop_keyframes",
    "build_scene",
    "select_framing_mode",
]
