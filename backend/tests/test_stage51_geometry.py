"""Stage 5.1 display-geometry, rotation, and crop-coordinate math."""

from __future__ import annotations

import pytest

from app.composition.geometry import (
    aspect_within_tolerance,
    clamp_crop,
    display_to_encoded,
    encoded_to_display,
    normalize_rotation,
    normalized_crop_to_display,
    parse_display_geometry,
)
from app.composition.types import DisplayGeometry
from app.media.ffprobe import ProbeParseError


def _geometry(width: int, height: int, rotation: int) -> DisplayGeometry:
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


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_encoded_display_round_trip_for_every_rotation(rotation: int) -> None:
    geometry = _geometry(1920, 1080, rotation)
    for point in [(0.0, 0.0), (1920.0, 1080.0), (100.0, 200.0), (960.0, 540.0)]:
        display = encoded_to_display(*point, geometry)
        assert 0.0 <= display[0] <= geometry.display_width
        assert 0.0 <= display[1] <= geometry.display_height
        assert display_to_encoded(*display, geometry) == pytest.approx(point)


def test_encoded_to_display_identity_at_zero_rotation() -> None:
    geometry = _geometry(1920, 1080, 0)
    assert encoded_to_display(100.0, 200.0, geometry) == (100.0, 200.0)


def test_encoded_to_display_swaps_axes_at_90() -> None:
    geometry = _geometry(1920, 1080, 90)
    assert (geometry.display_width, geometry.display_height) == (1080, 1920)
    assert encoded_to_display(0.0, 0.0, geometry) == (1080.0, 0.0)
    assert encoded_to_display(1920.0, 1080.0, geometry) == (0.0, 1920.0)


def test_encoded_to_display_mirrors_at_180() -> None:
    geometry = _geometry(1920, 1080, 180)
    assert (geometry.display_width, geometry.display_height) == (1920, 1080)
    assert encoded_to_display(0.0, 0.0, geometry) == (1920.0, 1080.0)


def test_encoded_to_display_swaps_axes_at_270() -> None:
    geometry = _geometry(1920, 1080, 270)
    assert (geometry.display_width, geometry.display_height) == (1080, 1920)
    assert encoded_to_display(0.0, 0.0, geometry) == (0.0, 1920.0)
    assert encoded_to_display(1920.0, 1080.0, geometry) == (1080.0, 0.0)


def test_normalize_rotation_snaps_and_wraps() -> None:
    assert normalize_rotation(0.0) == 0
    assert normalize_rotation(95.0) == 90
    assert normalize_rotation(-5.0) == 0
    assert normalize_rotation(450.0) == 90
    assert normalize_rotation(-90.0) == 270


def test_parse_display_geometry_reads_side_data_rotation() -> None:
    payload = {"streams": [{"width": 1920, "height": 1080, "side_data_list": [{"rotation": -90}]}]}
    geometry = parse_display_geometry(payload)
    assert geometry.rotation_degrees == 90
    assert (geometry.display_width, geometry.display_height) == (1080, 1920)


def test_parse_display_geometry_reads_legacy_rotate_tag() -> None:
    payload = {"streams": [{"width": 1920, "height": 1080, "tags": {"rotate": "270"}}]}
    geometry = parse_display_geometry(payload)
    assert geometry.rotation_degrees == 270
    assert (geometry.display_width, geometry.display_height) == (1080, 1920)


def test_parse_display_geometry_accepts_json_text_and_defaults() -> None:
    geometry = parse_display_geometry('{"streams": [{"width": 10, "height": 20}]}')
    assert geometry.rotation_degrees == 0
    assert (geometry.display_width, geometry.display_height) == (10, 20)


def test_parse_display_geometry_records_exotic_pixel_aspect() -> None:
    geometry = parse_display_geometry(
        {"streams": [{"width": 720, "height": 576, "sample_aspect_ratio": "16:15"}]}
    )
    assert geometry.exotic_pixel_aspect is True
    assert geometry.square_pixels_applied is True
    assert geometry.pixel_aspect_ratio == pytest.approx(16.0 / 15.0)
    assert (geometry.display_width, geometry.display_height) == (720, 576)


def test_parse_display_geometry_square_pixels_are_not_exotic() -> None:
    geometry = parse_display_geometry(
        {"streams": [{"width": 720, "height": 576, "sample_aspect_ratio": "1:1"}]}
    )
    assert geometry.exotic_pixel_aspect is False
    assert geometry.pixel_aspect_ratio == 1.0


def test_parse_display_geometry_rejects_missing_video_stream() -> None:
    with pytest.raises(ProbeParseError):
        parse_display_geometry({"streams": []})


def test_normalized_crop_is_centered_and_nine_by_sixteen() -> None:
    geometry = _geometry(1080, 1920, 0)
    crop = normalized_crop_to_display(0.5, 0.5, 0.5, geometry)
    assert crop == {"x": 270.0, "y": 480.0, "width": 540.0, "height": 960.0}


def test_normalized_crop_clamps_at_edges() -> None:
    geometry = _geometry(1080, 1920, 0)
    crop = normalized_crop_to_display(0.0, 1.0, 0.5, geometry)
    assert crop["x"] == 0.0
    assert crop["y"] == 960.0
    assert crop["width"] == 540.0
    assert crop["height"] == 960.0


def test_normalized_crop_oversized_reduces_to_frame() -> None:
    geometry = _geometry(1080, 1920, 0)
    crop = normalized_crop_to_display(0.5, 0.5, 4.0, geometry)
    assert crop == {"x": 0.0, "y": 0.0, "width": 1080.0, "height": 1920.0}


def test_clamp_crop_shifts_inside_frame() -> None:
    assert clamp_crop(-50.0, -10.0, 100.0, 200.0, 1080.0, 1920.0) == (
        0.0,
        0.0,
        100.0,
        200.0,
    )
    assert clamp_crop(1000.0, 1900.0, 100.0, 200.0, 1080.0, 1920.0) == (
        980.0,
        1720.0,
        100.0,
        200.0,
    )
    assert clamp_crop(100.0, 100.0, 100.0, 200.0, 1080.0, 1920.0) == (
        100.0,
        100.0,
        100.0,
        200.0,
    )


def test_clamp_crop_oversized_reduces_and_pins_to_origin() -> None:
    assert clamp_crop(-100.0, -100.0, 5000.0, 5000.0, 1080.0, 1920.0) == (
        0.0,
        0.0,
        1080.0,
        1920.0,
    )


def test_aspect_within_tolerance() -> None:
    assert aspect_within_tolerance(16.0 / 9.0, 16.0 / 9.0) is True
    assert aspect_within_tolerance(1.0, 1.04) is True
    assert aspect_within_tolerance(1.0, 1.06) is False
    assert aspect_within_tolerance(16.0 / 9.0, 16.0 / 9.0 + 0.1) is False


def test_display_geometry_as_dict_rounds_display_aspect() -> None:
    geometry = _geometry(1920, 1080, 90)
    payload = geometry.as_dict()
    assert payload["encoded_width"] == 1920
    assert payload["encoded_height"] == 1080
    assert payload["rotation_degrees"] == 90
    assert payload["display_width"] == 1080
    assert payload["display_height"] == 1920
    assert payload["display_aspect"] == pytest.approx(1080 / 1920)
