"""Lightweight parsing and calculations for audio analysis."""

from __future__ import annotations

import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path

_START = re.compile(r"silence_start:\s*(?P<start>[0-9.]+)")
_END = re.compile(
    r"silence_end:\s*(?P<end>[0-9.]+)\s*\|\s*silence_duration:\s*(?P<duration>[0-9.]+)"
)


@dataclass(frozen=True)
class SilenceInterval:
    """One silent time interval emitted by FFmpeg silencedetect."""

    start: float
    end: float
    duration: float


def parse_silencedetect(output: str) -> list[SilenceInterval]:
    """Pair FFmpeg silence starts and ends while ignoring incomplete trailing logs."""

    intervals: list[SilenceInterval] = []
    start: float | None = None
    for line in output.splitlines():
        if match := _START.search(line):
            start = float(match["start"])
        if match := _END.search(line):
            end = float(match["end"])
            duration = float(match["duration"])
            intervals.append(
                SilenceInterval(
                    start=start if start is not None else end - duration, end=end, duration=duration
                )
            )
            start = None
    return intervals


def silence_ratio(intervals: list[SilenceInterval], duration: float) -> float:
    """Return bounded silent-time ratio for a source duration."""

    if duration <= 0:
        return 0.0
    return min(1.0, sum(interval.duration for interval in intervals) / duration)


def windowed_rms(path: Path, window_seconds: float = 1.0) -> list[dict[str, float]]:
    """Return lightweight RMS amplitude windows from the cached 16-bit WAV."""
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getsampwidth() != 2 or audio.getframerate() <= 0:
                return []
            frames_per_window = max(1, int(audio.getframerate() * window_seconds))
            features: list[dict[str, float]] = []
            start = 0.0
            while frames := audio.readframes(frames_per_window):
                samples = memoryview(frames).cast("h")
                rms = (
                    math.sqrt(sum(sample * sample for sample in samples) / len(samples))
                    if samples
                    else 0.0
                )
                end = start + len(samples) / audio.getnchannels() / audio.getframerate()
                features.append({"start": start, "end": end, "rms": rms})
                start = end
            return features
    except (EOFError, wave.Error):
        return []
