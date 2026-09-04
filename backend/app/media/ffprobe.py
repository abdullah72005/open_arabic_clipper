"""Typed parsing and safe invocation of ffprobe JSON output."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProbeError(RuntimeError):
    """Base error raised while inspecting media metadata."""


class ProbeParseError(ProbeError, ValueError):
    """Raised when ffprobe output lacks valid required media fields."""


class ProbeExecutionError(ProbeError):
    """Raised when ffprobe cannot inspect a media file."""


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    """The Stage 1 media fields extracted from ffprobe JSON."""

    duration_seconds: float
    video_codec: str
    width: int
    height: int
    frames_per_second: float
    audio_codec: str | None
    audio_sample_rate: int | None


class FFprobe:
    """Invoke a configured ffprobe binary without shell interpolation."""

    def __init__(self, *, binary: str = "ffprobe") -> None:
        self._binary = binary

    def probe(self, path: Path) -> MediaMetadata:
        command = [
            self._binary,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise ProbeExecutionError("ffprobe failed to inspect media") from error
        return parse_ffprobe_json(result.stdout)


def parse_ffprobe_json(payload: str | Mapping[str, object]) -> MediaMetadata:
    """Convert ffprobe JSON into required video and optional audio metadata."""

    parsed = _load_payload(payload)
    format_data = _mapping(parsed.get("format"), "format")
    streams = parsed.get("streams")
    if not isinstance(streams, list):
        raise ProbeParseError("ffprobe streams must be a list")

    video_stream = _first_stream(streams, "video")
    if video_stream is None:
        raise ProbeParseError("ffprobe output has no video stream")
    audio_stream = _first_stream(streams, "audio")

    return MediaMetadata(
        duration_seconds=_positive_float(format_data.get("duration"), "format duration"),
        video_codec=_nonempty_string(video_stream.get("codec_name"), "video codec"),
        width=_positive_int(video_stream.get("width"), "video width"),
        height=_positive_int(video_stream.get("height"), "video height"),
        frames_per_second=_frame_rate(video_stream.get("avg_frame_rate")),
        audio_codec=(
            _nonempty_string(audio_stream.get("codec_name"), "audio codec")
            if audio_stream is not None
            else None
        ),
        audio_sample_rate=(
            _positive_int(audio_stream.get("sample_rate"), "audio sample rate")
            if audio_stream is not None
            else None
        ),
    )


def _load_payload(payload: str | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(payload, str):
        try:
            decoded: Any = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProbeParseError("ffprobe output is not JSON") from error
    else:
        decoded = payload
    return _mapping(decoded, "ffprobe output")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProbeParseError(f"{field} must be an object")
    return value


def _first_stream(streams: list[object], stream_type: str) -> Mapping[str, object] | None:
    for stream in streams:
        if isinstance(stream, Mapping) and stream.get("codec_type") == stream_type:
            return stream
    return None


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProbeParseError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ProbeParseError(f"{field} must be a positive integer") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ProbeParseError(f"{field} must be a positive integer")
    return parsed


def _positive_float(value: object, field: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ProbeParseError(f"{field} must be a positive number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ProbeParseError(f"{field} must be a positive number")
    return parsed


def _frame_rate(value: object) -> float:
    rate = _nonempty_string(value, "video frame rate")
    numerator_text, separator, denominator_text = rate.partition("/")
    if not separator:
        return _positive_float(rate, "video frame rate")
    numerator = _positive_float(numerator_text, "video frame-rate numerator")
    denominator = _positive_float(denominator_text, "video frame-rate denominator")
    return numerator / denominator
