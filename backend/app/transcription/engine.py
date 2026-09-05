"""Worker-side faster-whisper adapter with safe hardware selection."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
    """Load faster-whisper only in workers and convert its public result shape."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory | None = None,
        cuda_available: CudaAvailability | None = None,
    ) -> None:
        self._model_factory = model_factory or _default_model_factory
        self._cuda_available = cuda_available or _cuda_available

    def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> TranscriptionResult:
        """Transcribe a WAV path without changing Whisper text or timestamps."""

        device, compute_type = self._resolve_hardware(options)
        model = self._model_factory(options.model, device, compute_type)
        segments, info = model.transcribe(
            str(audio_path),
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
        words = [word for segment in serialized_segments for word in segment["words"]]
        return TranscriptionResult(
            language=_optional_str(getattr(info, "language", None)),
            language_probability=_optional_float(getattr(info, "language_probability", None)),
            raw_text="".join(str(segment["text"]) for segment in serialized_segments).strip(),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            segments=serialized_segments,
            word_segments=words,
        )

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
        from faster_whisper import WhisperModel as FasterWhisperModel
    except ImportError as err:
        raise RuntimeError("faster-whisper is not installed") from err
    return FasterWhisperModel(model, device=device, compute_type=compute_type)


def _cuda_available() -> bool:
    try:
        import ctranslate2
    except ImportError:
        return False
    return ctranslate2.get_cuda_device_count() > 0


def _serialize_segment(segment: object) -> dict[str, object]:
    words = [_serialize_word(word) for word in getattr(segment, "words", None) or []]
    return {
        "start": float(getattr(segment, "start")),
        "end": float(getattr(segment, "end")),
        "text": str(getattr(segment, "text")),
        "avg_logprob": _optional_float(getattr(segment, "avg_logprob", None)),
        "no_speech_prob": _optional_float(getattr(segment, "no_speech_prob", None)),
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
