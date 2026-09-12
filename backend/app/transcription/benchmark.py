"""Repeatable local faster-whisper benchmark reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
import threading

from app.transcription.engine import TranscriptionResult, WhisperEngine
from app.transcription.service import TranscriptionOptions


@dataclass(frozen=True)
class BenchmarkReport:
    source_audio_seconds: float
    wall_clock_seconds: float
    real_time_factor: float
    audio_minutes_per_wall_minute: float
    model: str
    device: str
    compute_type: str

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def benchmark_transcription(
    audio_path: Path, engine: WhisperEngine, options: TranscriptionOptions
) -> BenchmarkReport:
    """Run the configured engine once and report throughput from actual timings."""

    return transcribe_for_benchmark(audio_path, engine, options)[0]


def transcribe_for_benchmark(
    audio_path: Path,
    engine: WhisperEngine,
    options: TranscriptionOptions,
    cancel_event: threading.Event | None = None,
) -> tuple[BenchmarkReport, TranscriptionResult]:
    """Run one isolated ASR call and retain its ephemeral result for a read-only replay."""
    device, compute_type = engine.resolved_hardware(options)
    started = monotonic()
    result = engine.transcribe(audio_path, options, cancel_event=cancel_event)
    wall_clock = monotonic() - started
    duration = result.duration
    return BenchmarkReport(
        source_audio_seconds=duration,
        wall_clock_seconds=wall_clock,
        real_time_factor=wall_clock / duration if duration else 0.0,
        audio_minutes_per_wall_minute=duration / wall_clock if wall_clock else 0.0,
        model=options.model,
        device=device,
        compute_type=compute_type,
    ), result
