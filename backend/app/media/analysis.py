"""Lightweight parsing and calculations for audio analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
