"""Deterministic Stage 5.1 display geometry and crop coordinate math.

Pure geometry only: no rendering, no provider, no model loading, no audio
decoding. The single external-process seam is :class:`FFprobeDisplayProbe`, one
bounded read-only ffprobe metadata call used to learn encoded dimensions,
rotation, and pixel (sample) aspect ratio.

Display pixels are treated as square after rotation: ``display_width`` and
``display_height`` are the encoded dimensions, swapped for 90/270 degree
rotations and never rescaled by pixel aspect. A non-square pixel aspect is
recorded as exotic evidence but does not distort the display coordinate space.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from app.composition.types import DisplayGeometry
from app.media.ffprobe import ProbeExecutionError, ProbeParseError

_DISPLAY_ENTRIES = "stream=width,height,sample_aspect_ratio,side_data_list:stream_tags=rotate"
_SQUARE_PIXEL_TOLERANCE = 0.01


class DisplayProbe(Protocol):
    """Injectable seam over one bounded display-geometry metadata read."""

    def probe(self, path: Path) -> DisplayGeometry: ...


class FFprobeDisplayProbe:
    """Bounded read-only ffprobe probe for encoded/display geometry."""

    def __init__(self, *, binary: str = "ffprobe") -> None:
        self._binary = binary

    def probe(self, path: Path) -> DisplayGeometry:
        command = [
            self._binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            _DISPLAY_ENTRIES,
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise ProbeExecutionError("ffprobe failed to inspect display geometry") from error
        return parse_display_geometry(result.stdout)


def parse_display_geometry(payload: str | Mapping[str, object]) -> DisplayGeometry:
    """Parse ffprobe JSON into normalized display geometry.

    Rotation is snapped to the nearest 90 degrees and stored modulo 360 as the
    clockwise rotation applied to the encoded frame to reach the display frame.
    FFmpeg reports the display-matrix angle counterclockwise, so the side-data
    value is negated; the legacy ``tags.rotate`` value is already clockwise.
    """

    parsed = _load_payload(payload)
    streams = parsed.get("streams")
    if not isinstance(streams, list):
        raise ProbeParseError("ffprobe streams must be a list")
    stream = _first_video_stream(streams)
    if stream is None:
        raise ProbeParseError("ffprobe output has no video stream")

    width = _positive_int(stream.get("width"), "encoded width")
    height = _positive_int(stream.get("height"), "encoded height")
    rotation = _rotation_from_stream(stream)
    pixel_aspect_ratio = _pixel_aspect_ratio(stream.get("sample_aspect_ratio"))
    exotic = abs(pixel_aspect_ratio - 1.0) > _SQUARE_PIXEL_TOLERANCE
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
        pixel_aspect_ratio=pixel_aspect_ratio,
        square_pixels_applied=True,
        exotic_pixel_aspect=exotic,
    )


def normalize_rotation(degrees: float) -> int:
    """Snap any angle to the nearest 90 degrees, modulo 360."""

    if not math.isfinite(degrees):
        return 0
    return int(round(degrees / 90.0)) * 90 % 360


def encoded_to_display(x: float, y: float, geometry: DisplayGeometry) -> tuple[float, float]:
    """Map encoded-frame pixel coordinates to display-frame pixel coordinates."""

    width = geometry.encoded_width
    height = geometry.encoded_height
    rotation = geometry.rotation_degrees
    if rotation == 90:
        return (height - y, x)
    if rotation == 180:
        return (width - x, height - y)
    if rotation == 270:
        return (y, width - x)
    return (x, y)


def display_to_encoded(x: float, y: float, geometry: DisplayGeometry) -> tuple[float, float]:
    """Inverse of :func:`encoded_to_display`."""

    width = geometry.encoded_width
    height = geometry.encoded_height
    rotation = geometry.rotation_degrees
    if rotation == 90:
        return (y, height - x)
    if rotation == 180:
        return (width - x, height - y)
    if rotation == 270:
        return (width - y, x)
    return (x, y)


def normalized_crop_to_display(
    cx: float,
    cy: float,
    height_fraction: float,
    geometry: DisplayGeometry,
) -> dict[str, float]:
    """Return a clamped 9:16 display-pixel crop centered at normalized (cx, cy)."""

    height = height_fraction * geometry.display_height
    width = height * 9.0 / 16.0
    center_x = cx * geometry.display_width
    center_y = cy * geometry.display_height
    x, y, clamped_width, clamped_height = clamp_crop(
        center_x - width / 2.0,
        center_y - height / 2.0,
        width,
        height,
        float(geometry.display_width),
        float(geometry.display_height),
    )
    return {"x": x, "y": y, "width": clamped_width, "height": clamped_height}


def clamp_crop(
    x: float,
    y: float,
    w: float,
    h: float,
    display_width: float,
    display_height: float,
) -> tuple[float, float, float, float]:
    """Shift (never scale) a crop fully inside the frame.

    An oversized crop is reduced to the frame size and pinned to the origin.
    This is the single clamp used for every crop this package produces.
    """

    if w > display_width:
        w = display_width
    if h > display_height:
        h = display_height
    max_x = display_width - w
    max_y = display_height - h
    x = min(max(x, 0.0), max_x)
    y = min(max(y, 0.0), max_y)
    return (x, y, w, h)


def aspect_within_tolerance(a: float, b: float, tolerance: float = 0.05) -> bool:
    """Return True when two aspect ratios differ by no more than ``tolerance``."""

    return abs(a - b) <= tolerance


def _rotation_from_stream(stream: Mapping[str, object]) -> int:
    side_rotation = _side_data_rotation(stream.get("side_data_list"))
    if side_rotation is not None:
        return normalize_rotation(-side_rotation)
    tags = stream.get("tags")
    if isinstance(tags, Mapping):
        legacy = _as_float(tags.get("rotate"))
        if legacy is not None:
            return normalize_rotation(legacy)
    return 0


def _side_data_rotation(value: object) -> float | None:
    if not isinstance(value, list):
        return None
    for entry in value:
        if isinstance(entry, Mapping):
            rotation = _as_float(entry.get("rotation"))
            if rotation is not None:
                return rotation
    return None


def _pixel_aspect_ratio(value: object) -> float:
    if value is None:
        return 1.0
    if isinstance(value, str):
        numerator_text, separator, denominator_text = value.partition(":")
        if separator:
            numerator = _as_float(numerator_text)
            denominator = _as_float(denominator_text)
            if numerator is None or denominator is None or denominator == 0.0:
                return 1.0
            return numerator / denominator
    aspect = _as_float(value)
    return aspect if aspect is not None else 1.0


def _load_payload(payload: str | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(payload, str):
        try:
            decoded: Any = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProbeParseError("ffprobe output is not JSON") from error
    else:
        decoded = payload
    if not isinstance(decoded, Mapping):
        raise ProbeParseError("ffprobe output must be an object")
    return decoded


def _first_video_stream(streams: list[object]) -> Mapping[str, object] | None:
    for stream in streams:
        if isinstance(stream, Mapping):
            codec_type = stream.get("codec_type")
            if codec_type is None or codec_type == "video":
                return stream
    return None


def _positive_int(value: object, field: str) -> int:
    try:
        parsed = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ProbeParseError(f"{field} must be a positive integer") from error
    if parsed <= 0:
        raise ProbeParseError(f"{field} must be a positive integer")
    return parsed


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


__all__ = [
    "DisplayProbe",
    "FFprobeDisplayProbe",
    "aspect_within_tolerance",
    "clamp_crop",
    "display_to_encoded",
    "encoded_to_display",
    "normalize_rotation",
    "normalized_crop_to_display",
    "parse_display_geometry",
]
