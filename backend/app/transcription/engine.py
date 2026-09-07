"""Worker-side faster-whisper adapter with safe hardware selection."""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from app.runtime.model_process import (
    ModelProcessRunner,
    ProcessOutcome,
    SpawnedProcessError,
)
from app.transcription.service import TranscriptionOptions


class WhisperModel(Protocol):
    """Small subset of faster-whisper used by the application boundary."""

    def transcribe(self, path: str, **kwargs: object) -> tuple[Iterable[object], object]: ...


ModelFactory = Callable[[str, str, str], WhisperModel]
CudaAvailability = Callable[[], bool]


@dataclass(frozen=True)
class TranscriptionResult:
    """Raw timestamp evidence returned by the local Whisper backend."""

    language: str | None
    language_probability: float | None
    raw_text: str
    duration: float
    segments: list[dict[str, object]]
    word_segments: list[dict[str, object]]


class WhisperEngine:
    """Load faster-whisper only in a spawned child and convert its public result shape."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory | None = None,
        cuda_available: CudaAvailability | None = None,
        collect_garbage: Callable[[], int] = gc.collect,
        runner: Any | None = None,
    ) -> None:
        self._model_factory = model_factory or _default_model_factory
        self._cuda_available = cuda_available or _cuda_available
        self._collect_garbage = collect_garbage
        self._runner = runner or ModelProcessRunner(timeout_seconds=7_200.0)
        self._last_outcome: ProcessOutcome | None = None

    def last_child_peak_rss(self) -> int | None:
        """Return the peak RSS measured inside the last spawned child, in bytes."""

        return self._last_outcome.child_peak_rss_bytes if self._last_outcome is not None else None

    def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> TranscriptionResult:
        """Transcribe a WAV path without changing Whisper text or timestamps.

        The native model is loaded and run inside a spawned child so process exit,
        not Python garbage collection, is the hard reclamation boundary.
        """

        device, compute_type = self._resolve_hardware(options)
        outcome = self._runner.run(
            target=_run_transcription_child,
            args=(
                self._model_factory,
                options.model,
                device,
                compute_type,
                str(audio_path),
                options,
                self._collect_garbage,
            ),
        )
        self._last_outcome = outcome
        if not outcome.ok:
            raise SpawnedProcessError(outcome.error or "transcription child failed")
        return cast(TranscriptionResult, outcome.result)

    def resolved_hardware(self, options: TranscriptionOptions) -> tuple[str, str]:
        """Expose the effective device policy for operational reporting."""
        return self._resolve_hardware(options)

    def _resolve_hardware(self, options: TranscriptionOptions) -> tuple[str, str]:
        cuda = self._cuda_available()
        use_cuda = options.device == "cuda" and cuda or options.device == "auto" and cuda
        if use_cuda:
            return (
                "cuda",
                options.compute_type
                if options.compute_type != "auto"
                else options.cuda_compute_type,
            )
        return (
            "cpu",
            options.compute_type if options.compute_type != "auto" else options.cpu_compute_type,
        )


def _default_model_factory(model: str, device: str, compute_type: str) -> WhisperModel:
    try:
        import faster_whisper  # type: ignore[import-untyped]
    except ImportError as err:
        raise RuntimeError("faster-whisper is not installed") from err
    return cast(
        WhisperModel,
        faster_whisper.WhisperModel(model, device=device, compute_type=compute_type),
    )


def _run_transcription_child(
    model_factory: ModelFactory,
    model: str,
    device: str,
    compute_type: str,
    audio_path_str: str,
    options: TranscriptionOptions,
    collect_garbage: Callable[[], int],
) -> TranscriptionResult:
    """Load the model and transcribe entirely inside the spawned child.

    Generators are fully consumed here so the picklable result sent to the parent
    is already materialized. The child drops local references and collects
    garbage on both success and exceptions before it exits.
    """

    model_obj = model_factory(model, device, compute_type)
    try:
        segments, info = model_obj.transcribe(
            audio_path_str,
            beam_size=options.beam_size,
            language=options.language,
            word_timestamps=options.word_timestamps,
            temperature=options.temperature,
            condition_on_previous_text=options.condition_on_previous_text,
            vad_filter=options.vad_filter,
            initial_prompt=options.initial_prompt,
            hotwords=options.hotwords,
        )
        serialized_segments = [_serialize_segment(segment) for segment in segments]
        words = [
            word
            for segment in serialized_segments
            for word in cast(list[dict[str, object]], segment["words"])
        ]
        return TranscriptionResult(
            language=_optional_str(getattr(info, "language", None)),
            language_probability=_optional_float(getattr(info, "language_probability", None)),
            raw_text="".join(str(segment["text"]) for segment in serialized_segments).strip(),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            segments=serialized_segments,
            word_segments=words,
        )
    finally:
        del model_obj
        collect_garbage()


def _cuda_available() -> bool:
    try:
        import ctranslate2  # type: ignore[import-untyped]
    except ImportError:
        return False
    return bool(ctranslate2.get_cuda_device_count() > 0)


def _serialize_segment(segment: object) -> dict[str, object]:
    words = [_serialize_word(word) for word in getattr(segment, "words", None) or []]
    return {
        "start": float(getattr(segment, "start")),
        "end": float(getattr(segment, "end")),
        "text": str(getattr(segment, "text")),
        "tokens": [int(token) for token in getattr(segment, "tokens", None) or []],
        "avg_logprob": _optional_float(getattr(segment, "avg_logprob", None)),
        "compression_ratio": _optional_float(getattr(segment, "compression_ratio", None)),
        "no_speech_prob": _optional_float(getattr(segment, "no_speech_prob", None)),
        "temperature": _optional_float(getattr(segment, "temperature", None)),
        "words": words,
    }


def _serialize_word(word: object) -> dict[str, object]:
    return {
        "start": float(getattr(word, "start")),
        "end": float(getattr(word, "end")),
        "word": str(getattr(word, "word")),
        "probability": _optional_float(getattr(word, "probability", None)),
    }


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None
