"""Pure Stage 5.1 deterministic framing selection and crop-keyframe tests."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from app.composition.framing import (
    build_crop_keyframes,
    build_scene,
    select_framing_mode,
)
from app.composition.geometry import clamp_crop
from app.composition.policy import FramingEvidence, FramingMode, Stage51Config
from app.composition.types import (
    CropKeyframe,
    DisplayGeometry,
    FaceTrack,
    TrackSample,
)

_NINE_SIXTEENTH = 9.0 / 16.0


def _geometry(width: int, height: int, rotation: int = 0) -> DisplayGeometry:
    if rotation in (90, 270):
        display_width, display_height = height, width
    else:
        display_width, display_height = width, height
    return DisplayGeometry(
        encoded_width=width,
        encoded_height=height,
        rotation_degrees=rotation,
        display_width=display_width,
        display_height=display_height,
    )


def _config(
    *,
    track_max_gap_samples: int = 8,
    track_min_persistence_samples: int = 1,
    track_min_persistence_seconds: float = 0.0,
    detector_enabled: bool = True,
    face_min_height_fraction: float = 0.06,
    target_face_height_fraction: float = 0.38,
    min_crop_height_fraction: float = 0.35,
    headroom_fraction: float = 0.12,
    chin_margin_fraction: float = 0.10,
    dead_zone_fraction: float = 0.18,
    max_pan_velocity_per_second: float = 0.55,
    max_zoom_rate_per_second: float = 0.15,
    min_hold_seconds: float = 0.5,
    max_keyframes_per_scene: int = 40,
) -> Stage51Config:
    return Stage51Config(
        track_max_gap_samples=track_max_gap_samples,
        track_min_persistence_samples=track_min_persistence_samples,
        track_min_persistence_seconds=track_min_persistence_seconds,
        detector_enabled=detector_enabled,
        face_min_height_fraction=face_min_height_fraction,
        target_face_height_fraction=target_face_height_fraction,
        min_crop_height_fraction=min_crop_height_fraction,
        headroom_fraction=headroom_fraction,
        chin_margin_fraction=chin_margin_fraction,
        dead_zone_fraction=dead_zone_fraction,
        max_pan_velocity_per_second=max_pan_velocity_per_second,
        max_zoom_rate_per_second=max_zoom_rate_per_second,
        min_hold_seconds=min_hold_seconds,
        max_keyframes_per_scene=max_keyframes_per_scene,
    )


def _sample(
    source_time: float,
    cx: float,
    cy: float,
    w: float = 0.15,
    h: float = 0.2,
    score: float = 0.9,
) -> TrackSample:
    return TrackSample(source_time=source_time, cx=cx, cy=cy, w=w, h=h, score=score)


def _track(track_id: int, samples: list[TrackSample], stable: bool = True) -> FaceTrack:
    ordered = tuple(sorted(samples, key=lambda sample: sample.source_time))
    count = float(len(ordered))
    return FaceTrack(
        track_id=track_id,
        scene_index=0,
        first_time=ordered[0].source_time,
        last_time=ordered[-1].source_time,
        samples=ordered,
        persistence=count,
        mean_score=sum(sample.score for sample in ordered) / count,
        mean_face_height_fraction=sum(sample.h for sample in ordered) / count,
        stable=stable,
    )


def _keyframe_rect(
    keyframe: CropKeyframe, geometry: DisplayGeometry
) -> tuple[float, float, float, float]:
    center_x = keyframe.center_x
    center_y = keyframe.center_y
    height_fraction = keyframe.height_fraction
    height_px = height_fraction * geometry.display_height
    width_px = height_px * _NINE_SIXTEENTH
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


def _assert_same_crop(first: CropKeyframe, second: CropKeyframe) -> None:
    assert first.center_x == pytest.approx(second.center_x)
    assert first.center_y == pytest.approx(second.center_y)
    assert first.height_fraction == pytest.approx(second.height_fraction)


def _assert_inside_bounds(keyframes: Sequence[CropKeyframe], geometry: DisplayGeometry) -> None:
    for keyframe in keyframes:
        x, y, width, height = _keyframe_rect(keyframe, geometry)
        assert x >= -1e-6
        assert y >= -1e-6
        assert x + width <= geometry.display_width + 1e-6
        assert y + height <= geometry.display_height + 1e-6


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


def test_wide_source_is_not_source_as_is_and_uses_a_valid_vertical_strategy() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    track = _track(1, [_sample(index * 0.5, 0.5, 0.5) for index in range(5)])

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )

    assert decision.mode != FramingMode.SOURCE_AS_IS.value
    assert decision.mode in {mode.value for mode in FramingMode}
    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) >= 2
    for keyframe in keyframes:
        _, _, width, height = _keyframe_rect(keyframe, geometry)
        assert width == pytest.approx(height * _NINE_SIXTEENTH)


def test_already_vertical_source_is_source_as_is_with_no_crop() -> None:
    geometry = _geometry(1080, 1920)
    config = _config()
    track = _track(1, [_sample(0.0, 0.5, 0.5, w=0.2, h=0.3)])

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=1.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )

    assert decision.mode == FramingMode.SOURCE_AS_IS.value
    assert FramingEvidence.SOURCE_ALREADY_VERTICAL.value in decision.evidence

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=1.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) == 2
    _assert_same_crop(keyframes[0], keyframes[1])
    assert keyframes[0].height_fraction == pytest.approx(1.0)
    assert keyframes[0].center_x == pytest.approx(0.5)
    assert keyframes[0].center_y == pytest.approx(0.5)


def test_square_source_yields_a_valid_composition() -> None:
    geometry = _geometry(1080, 1080)
    config = _config()
    track = _track(1, [_sample(0.0, 0.5, 0.5, w=0.2, h=0.3)])

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=1.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert decision.mode in {mode.value for mode in FramingMode}
    assert decision.mode != FramingMode.SOURCE_AS_IS.value

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=1.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) >= 2
    _assert_inside_bounds(keyframes, geometry)


# ---------------------------------------------------------------------------
# Clamping and margin policy
# ---------------------------------------------------------------------------


def test_crop_is_always_inside_display_bounds() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    positions = [
        (0.5, 0.5),
        (0.05, 0.5),
        (0.95, 0.5),
        (0.5, 0.1),
        (0.5, 0.9),
        (0.08, 0.12),
        (0.92, 0.88),
    ]
    for cx, cy in positions:
        track = _track(1, [_sample(0.0, cx, cy, w=0.15, h=0.2)])
        decision = select_framing_mode(
            block_index=0,
            scene_start=0.0,
            scene_end=1.0,
            tracks=(track,),
            config=config,
            geometry=geometry,
        )
        keyframes = build_crop_keyframes(
            mode=decision.mode,
            scene_start=0.0,
            scene_end=1.0,
            tracks=(track,),
            config=config,
            geometry=geometry,
        )
        assert keyframes
        _assert_inside_bounds(keyframes, geometry)


def test_headroom_and_chin_margin_for_centered_subject() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    track = _track(1, [_sample(0.6, 0.5, 0.5, w=0.15, h=0.25)])

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=1.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.STATIC_CROP.value

    keyframe = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=1.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )[0]
    _, y, _, height = _keyframe_rect(keyframe, geometry)
    face_top = (0.5 - 0.25 / 2.0) * geometry.display_height
    face_bottom = (0.5 + 0.25 / 2.0) * geometry.display_height

    assert y <= face_top - config.headroom_fraction * height + 1.0
    assert y + height >= face_bottom + config.chin_margin_fraction * height - 1.0


def test_face_near_edge_falls_back_to_background_fill() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    track = _track(1, [_sample(0.6, 0.99, 0.5, w=0.2, h=0.2)])

    keyframes = build_crop_keyframes(
        mode=FramingMode.STATIC_CROP.value,
        scene_start=0.0,
        scene_end=1.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert all(
        FramingEvidence.FACE_NEAR_SOURCE_EDGE.value in keyframe.evidence for keyframe in keyframes
    )
    _assert_inside_bounds(keyframes, geometry)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_identical_serialized_keyframes() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    samples = [
        _sample(index * 0.4, 0.45 + 0.02 * math.sin(index), 0.5, h=0.22) for index in range(8)
    ]
    track = _track(1, samples)

    first = build_crop_keyframes(
        mode=FramingMode.TRACKED_CROP.value,
        scene_start=0.0,
        scene_end=3.2,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    second = build_crop_keyframes(
        mode=FramingMode.TRACKED_CROP.value,
        scene_start=0.0,
        scene_end=3.2,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )

    assert [keyframe.as_dict() for keyframe in first] == [keyframe.as_dict() for keyframe in second]


# ---------------------------------------------------------------------------
# Single-subject behavior
# ---------------------------------------------------------------------------


def test_centered_stable_subject_is_static() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    track = _track(1, [_sample(index * 0.5, 0.5, 0.5) for index in range(5)])

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.STATIC_CROP.value

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) == 2
    _assert_same_crop(keyframes[0], keyframes[1])


def test_slow_movement_is_smooth_and_tracked() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    samples = [_sample(index * 0.5, 0.45 + 0.02 * index, 0.5) for index in range(6)]
    track = _track(1, samples)

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=3.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.TRACKED_CROP.value

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=3.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) > 2
    _assert_inside_bounds(keyframes, geometry)


def test_jitter_is_effectively_static() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    samples = [
        _sample(index * 0.5, 0.5 + (0.005 if index % 2 else -0.005), 0.5) for index in range(6)
    ]
    track = _track(1, samples)

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.STATIC_CROP.value

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) == 2
    _assert_same_crop(keyframes[0], keyframes[1])


def test_large_deliberate_movement_is_followed_within_velocity_bounds() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    samples = [_sample(index * 0.5, 0.2 + 0.1 * index, 0.5) for index in range(7)]
    track = _track(1, samples)

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=3.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.TRACKED_CROP.value

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=3.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) > 2
    assert keyframes[-1].center_x > keyframes[0].center_x + 0.05

    for first, second in zip(keyframes, keyframes[1:], strict=False):
        elapsed = second.source_time - first.source_time
        if elapsed <= 0.0:
            continue
        limit = config.max_pan_velocity_per_second * first.height_fraction * elapsed + 1e-6
        assert abs(second.center_x - first.center_x) <= limit
        assert abs(second.center_y - first.center_y) <= limit


def test_brief_detection_loss_holds_without_detection_lost_evidence() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    samples = [
        _sample(0.0, 0.40, 0.5),
        _sample(0.5, 0.42, 0.5),
        _sample(1.2, 0.44, 0.5),
        _sample(1.7, 0.46, 0.5),
    ]
    track = _track(1, samples)

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.2,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.TRACKED_CROP.value

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=2.2,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert not any(
        FramingEvidence.DETECTION_LOST.value in keyframe.evidence for keyframe in keyframes
    )


def test_long_detection_loss_falls_back_conservatively() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    samples = [
        _sample(0.0, 0.40, 0.5),
        _sample(0.5, 0.42, 0.5),
        _sample(5.0, 0.44, 0.5),
        _sample(5.5, 0.46, 0.5),
    ]
    track = _track(1, samples)

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=6.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.TRACKED_CROP.value

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=6.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert any(FramingEvidence.DETECTION_LOST.value in keyframe.evidence for keyframe in keyframes)
    _assert_inside_bounds(keyframes, geometry)


def test_keyframe_cap_never_exceeds_config() -> None:
    geometry = _geometry(1920, 1080)
    config = _config(max_keyframes_per_scene=3)
    samples = [
        _sample(index * 0.5, 0.2 + 0.03 * index, 0.5, h=0.2 + 0.005 * index) for index in range(20)
    ]
    track = _track(1, samples)

    keyframes = build_crop_keyframes(
        mode=FramingMode.TRACKED_CROP.value,
        scene_start=0.0,
        scene_end=10.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert 2 <= len(keyframes) <= 3


# ---------------------------------------------------------------------------
# Multi-subject behavior
# ---------------------------------------------------------------------------


def test_two_close_faces_fit_one_crop_window() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    times = (0.0, 0.5, 1.0, 1.5)
    left = _track(1, [_sample(time, 0.45, 0.5, w=0.15, h=0.25) for time in times])
    right = _track(2, [_sample(time, 0.55, 0.5, w=0.15, h=0.25) for time in times])

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(left, right),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.MULTI_SUBJECT_FIT.value
    assert FramingEvidence.MULTIPLE_FACES.value in decision.evidence

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(left, right),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) == 2
    _assert_same_crop(keyframes[0], keyframes[1])
    x, _, width, _ = _keyframe_rect(keyframes[0], geometry)
    assert x <= 0.45 * geometry.display_width
    assert x + width >= 0.55 * geometry.display_width


def test_two_distant_faces_use_background_fill() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    times = (0.0, 0.5, 1.0, 1.5)
    left = _track(1, [_sample(time, 0.1, 0.5, w=0.15, h=0.25) for time in times])
    right = _track(2, [_sample(time, 0.9, 0.5, w=0.15, h=0.25) for time in times])

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(left, right),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.BACKGROUND_FILL.value
    assert FramingEvidence.IMPORTANT_CONTENT_WOULD_BE_CROPPED.value in decision.evidence

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(left, right),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) == 2
    _assert_inside_bounds(keyframes, geometry)


def test_unstable_track_is_never_tracked_crop() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    samples = [_sample(index * 0.5, 0.1 + 0.15 * index, 0.5) for index in range(5)]
    track = _track(1, samples, stable=False)

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert decision.mode != FramingMode.TRACKED_CROP.value
    # Detections exist but no track met the persistence threshold: keep every
    # subject visible with background-fill rather than a center crop.
    assert decision.mode == FramingMode.BACKGROUND_FILL.value
    assert FramingEvidence.FACE_TRACK_UNSTABLE.value in decision.evidence

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) == 2
    assert keyframes[0].height_fraction == pytest.approx(1.0)
    _assert_inside_bounds(keyframes, geometry)


def test_detections_without_persistent_track_use_background_fill_keeping_all_subjects() -> None:
    """Two short-lived detections at a cut must keep both subjects (finding fix)."""

    geometry = _geometry(1920, 1080)
    config = _config(track_min_persistence_samples=3, track_min_persistence_seconds=1.0)
    short = [
        _track(1, [_sample(0.0, 0.25, 0.5, w=0.12, h=0.18)], stable=False),
        _track(2, [_sample(0.1, 0.75, 0.5, w=0.12, h=0.18)], stable=False),
    ]

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=0.44,
        tracks=tuple(short),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.BACKGROUND_FILL.value
    assert decision.fallback_parameters.get("reason") == "NO_PERSISTENT_TRACK"

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=0.44,
        tracks=tuple(short),
        config=config,
        geometry=geometry,
    )
    # Background-fill keeps the full frame, so neither subject is cropped out.
    assert keyframes[0].height_fraction == pytest.approx(1.0)
    assert keyframes[1].height_fraction == pytest.approx(1.0)
    assert keyframes[0].center_x == pytest.approx(0.5)


def test_zero_detections_use_center_fallback() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.CENTER_FALLBACK.value
    assert decision.fallback_parameters.get("reason") == "NO_DETECTIONS"
    assert FramingEvidence.FACE_TRACK_UNSTABLE.value not in decision.evidence


def test_invalid_geometry_uses_center_fallback() -> None:
    config = _config()
    broken = DisplayGeometry(
        encoded_width=0,
        encoded_height=0,
        rotation_degrees=0,
        display_width=0,
        display_height=0,
    )
    samples = [_sample(index * 0.5, 0.5, 0.5) for index in range(4)]

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(_track(1, samples),),
        config=config,
        geometry=broken,
    )
    assert decision.mode == FramingMode.CENTER_FALLBACK.value
    assert decision.fallback_parameters.get("reason") == "INVALID_GEOMETRY"


# ---------------------------------------------------------------------------
# No-face and detector behavior
# ---------------------------------------------------------------------------


def test_no_face_yields_a_valid_conservative_fallback() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.CENTER_FALLBACK.value
    assert FramingEvidence.NO_FACE.value in decision.evidence

    keyframes = build_crop_keyframes(
        mode=decision.mode,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(),
        config=config,
        geometry=geometry,
    )
    assert len(keyframes) == 2
    _assert_inside_bounds(keyframes, geometry)


def test_no_face_with_screen_content_uses_background_fill() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(),
        config=config,
        geometry=geometry,
        screen_content_evidence=True,
    )
    assert decision.mode == FramingMode.BACKGROUND_FILL.value
    assert FramingEvidence.SCREEN_CONTENT.value in decision.evidence


def test_no_face_screen_content_mapping_is_honored() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(),
        config=config,
        geometry=geometry,
        screen_content_evidence={"screen_content": True},
    )
    assert decision.mode == FramingMode.BACKGROUND_FILL.value

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(),
        config=config,
        geometry=geometry,
        screen_content_evidence={"density": 0.9},
    )
    assert decision.mode == FramingMode.BACKGROUND_FILL.value


def test_detector_unavailable_records_conservative_evidence() -> None:
    geometry = _geometry(1920, 1080)
    config = _config(detector_enabled=False)

    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(),
        config=config,
        geometry=geometry,
    )
    assert decision.mode == FramingMode.CENTER_FALLBACK.value
    assert FramingEvidence.DETECTOR_UNAVAILABLE.value in decision.evidence
    assert FramingEvidence.NO_FACE.value in decision.evidence


# ---------------------------------------------------------------------------
# Hero protection
# ---------------------------------------------------------------------------


def test_hero_protection_freezes_mode_and_zoom() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    samples = [
        _sample(index * 0.5, 0.4 + 0.03 * index, 0.5, w=0.15, h=0.15 + 0.03 * index)
        for index in range(5)
    ]
    track = _track(1, samples)

    plain = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    hero = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
        is_hero=True,
    )

    assert hero.mode == plain.mode
    assert hero.protected is True
    assert FramingEvidence.HERO_PROTECTION_APPLIED.value in hero.evidence

    hero_keyframes = build_crop_keyframes(
        mode=hero.mode,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
        is_hero=True,
    )
    assert all(
        FramingEvidence.HERO_PROTECTION_APPLIED.value in keyframe.evidence
        for keyframe in hero_keyframes
    )
    assert len({round(keyframe.height_fraction, 6) for keyframe in hero_keyframes}) == 1

    plain_keyframes = build_crop_keyframes(
        mode=plain.mode,
        scene_start=0.0,
        scene_end=2.5,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )
    assert len({round(keyframe.height_fraction, 6) for keyframe in plain_keyframes}) > 1


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------


def test_build_scene_assembles_framing_keyframes_and_policy() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    track = _track(1, [_sample(index * 0.5, 0.5, 0.5) for index in range(4)])

    scene = build_scene(
        scene_index=0,
        block_index=3,
        scene_start=0.0,
        scene_end=2.0,
        cut_start=True,
        mode=FramingMode.STATIC_CROP,
        tracks=(track,),
        config=config,
        geometry=geometry,
    )

    assert scene.interpolation_policy == "smoothstep-ease"
    assert scene.framing.mode == FramingMode.STATIC_CROP.value
    assert scene.block_index == 3
    assert scene.cut_start is True
    assert len(scene.crop_keyframes) == 2
    payload = scene.as_dict()
    assert payload["interpolation_policy"] == "smoothstep-ease"
    assert payload["crop_keyframes"]


def test_build_scene_with_decision_uses_the_decision() -> None:
    geometry = _geometry(1920, 1080)
    config = _config()
    track = _track(1, [_sample(index * 0.5, 0.5, 0.5) for index in range(4)])
    decision = select_framing_mode(
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        tracks=(track,),
        config=config,
        geometry=geometry,
        is_hero=True,
    )

    scene = build_scene(
        scene_index=0,
        block_index=0,
        scene_start=0.0,
        scene_end=2.0,
        cut_start=False,
        mode=decision,
        tracks=(track,),
        config=config,
        geometry=geometry,
        is_hero=True,
    )

    assert scene.framing == decision
    assert all(
        FramingEvidence.HERO_PROTECTION_APPLIED.value in keyframe.evidence
        for keyframe in scene.crop_keyframes
    )
