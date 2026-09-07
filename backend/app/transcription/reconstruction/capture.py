"""Immutable ASR capture contract for reconstruction model A/B replay.

A capture stores the raw decoder output once, with the complete decoder identity
and clip/source provenance. Reconstruction models replay the exact same raw
segments; Whisper is never rerun for a model comparison.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

from app.pipeline.fingerprints import canonical_fingerprint
from app.services.storage import StorageCategory, StorageService
from app.transcription.engine import TranscriptionResult
from app.transcription.service import TranscriptionOptions

_SCHEMA_VERSION = "asr-capture-v1"


class CaptureValidationError(ValueError):
    """A capture violates the immutable ASR contract."""


@dataclass(frozen=True)
class DecoderIdentity:
    """Every output-affecting decoder setting captured with the raw text."""

    whisper_model: str
    faster_whisper_version: str
    ctranslate2_version: str
    device: str
    compute_type: str
    language: str | None
    beam_size: int
    word_timestamps: bool
    temperature: tuple[float, ...]
    condition_on_previous_text: bool
    vad_filter: bool
    initial_prompt: str | None
    hotwords: str | None


@dataclass(frozen=True)
class CaptureClip:
    """One immutable decoded clip inside an ASR capture."""

    clip_id: str
    source_id: str
    source_hash: str
    start_seconds: float
    end_seconds: float
    language: str | None
    language_probability: float | None
    duration: float
    segments: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ASRCapture:
    """A versioned, hashed, immutable ASR capture."""

    schema_version: str
    capture_id: str
    decoder: DecoderIdentity
    clips: tuple[CaptureClip, ...]
    wall_clock_seconds: float
    memory_snapshots: dict[str, object]


def decoder_versions() -> dict[str, str]:
    """Return installed faster-whisper and CTranslate2 versions for the identity."""

    def package_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "unknown"

    return {
        "faster_whisper": package_version("faster-whisper"),
        "ctranslate2": package_version("ctranslate2"),
    }


def build_decoder_identity(options: TranscriptionOptions) -> DecoderIdentity:
    """Capture the full decoder identity from the output-affecting options."""

    versions = decoder_versions()
    return DecoderIdentity(
        whisper_model=options.model,
        faster_whisper_version=versions["faster_whisper"],
        ctranslate2_version=versions["ctranslate2"],
        device=options.device,
        compute_type=options.compute_type or "auto",
        language=options.language,
        beam_size=options.beam_size,
        word_timestamps=options.word_timestamps,
        temperature=tuple(options.temperature),
        condition_on_previous_text=options.condition_on_previous_text,
        vad_filter=options.vad_filter,
        initial_prompt=options.initial_prompt,
        hotwords=options.hotwords,
    )


def build_capture(
    *,
    capture_id: str,
    clips: Sequence[tuple[str, str, float, float]],
    source_hashes: Mapping[str, str],
    results: Mapping[str, TranscriptionResult],
    options: TranscriptionOptions,
    wall_clock_seconds: float,
    memory_snapshots: dict[str, object] | None = None,
) -> ASRCapture:
    """Build an immutable capture from one Whisper pass over each clip."""

    captured_clips = tuple(
        CaptureClip(
            clip_id=clip_id,
            source_id=source_id,
            source_hash=source_hashes.get(source_id, ""),
            start_seconds=start,
            end_seconds=end,
            language=results[clip_id].language,
            language_probability=results[clip_id].language_probability,
            duration=results[clip_id].duration,
            segments=tuple(dict(segment) for segment in results[clip_id].segments),
        )
        for clip_id, source_id, start, end in clips
    )
    return ASRCapture(
        schema_version=_SCHEMA_VERSION,
        capture_id=capture_id,
        decoder=build_decoder_identity(options),
        clips=captured_clips,
        wall_clock_seconds=wall_clock_seconds,
        memory_snapshots=memory_snapshots or {},
    )


def capture_hash(capture: ASRCapture) -> str:
    """Hash the canonical schema content so any change invalidates the capture."""

    return cast(str, canonical_fingerprint("asr-capture", "1", _capture_payload(capture)))


def _capture_payload(capture: ASRCapture) -> dict[str, object]:
    return {
        "schema_version": capture.schema_version,
        "capture_id": capture.capture_id,
        "decoder": asdict(capture.decoder),
        "clips": [
            {
                "clip_id": clip.clip_id,
                "source_id": clip.source_id,
                "source_hash": clip.source_hash,
                "start_seconds": clip.start_seconds,
                "end_seconds": clip.end_seconds,
                "language": clip.language,
                "language_probability": clip.language_probability,
                "duration": clip.duration,
                "segments": clip.segments,
            }
            for clip in capture.clips
        ],
    }


def serialize_capture(capture: ASRCapture) -> str:
    """Serialize a capture to stable canonical JSON."""

    return json.dumps(_capture_payload(capture), ensure_ascii=False, sort_keys=True, indent=2)


def parse_capture(text: str) -> ASRCapture:
    """Parse and validate a capture from its canonical JSON form."""

    payload = json.loads(text)
    decoder = DecoderIdentity(**payload["decoder"])
    clips = tuple(
        CaptureClip(
            clip_id=item["clip_id"],
            source_id=item["source_id"],
            source_hash=item["source_hash"],
            start_seconds=item["start_seconds"],
            end_seconds=item["end_seconds"],
            language=item["language"],
            language_probability=item["language_probability"],
            duration=item["duration"],
            segments=tuple(item["segments"]),
        )
        for item in payload["clips"]
    )
    capture = ASRCapture(
        schema_version=payload["schema_version"],
        capture_id=payload["capture_id"],
        decoder=decoder,
        clips=clips,
        wall_clock_seconds=float(payload.get("wall_clock_seconds", 0.0)),
        memory_snapshots=payload.get("memory_snapshots", {}),
    )
    validate_capture(capture)
    return capture


def validate_capture(capture: ASRCapture) -> None:
    """Enforce the immutable contract: version, timestamps, and provenance."""

    if capture.schema_version != _SCHEMA_VERSION:
        raise CaptureValidationError(
            f"unsupported capture schema version: {capture.schema_version}"
        )
    if not capture.capture_id:
        raise CaptureValidationError("capture_id is required")
    seen: set[str] = set()
    for clip in capture.clips:
        if clip.clip_id in seen:
            raise CaptureValidationError(f"duplicate clip id: {clip.clip_id}")
        seen.add(clip.clip_id)
        if not clip.source_id:
            raise CaptureValidationError(f"clip {clip.clip_id} has no source id")
        if not clip.source_hash:
            raise CaptureValidationError(f"clip {clip.clip_id} has no source hash")
        if not clip.segments:
            raise CaptureValidationError(f"clip {clip.clip_id} has no raw segments")
        previous_end = None
        for index, segment in enumerate(clip.segments):
            start = float(cast(float, segment.get("start", 0.0)))
            end = float(cast(float, segment.get("end", start)))
            if start > end:
                raise CaptureValidationError(
                    f"clip {clip.clip_id} segment {index} has inverted timestamps"
                )
            if previous_end is not None and start < previous_end:
                raise CaptureValidationError(
                    f"clip {clip.clip_id} segment {index} overlaps the previous segment"
                )
            previous_end = end


def save_capture(storage: StorageService, capture: ASRCapture, *, name: str) -> Path:
    """Write a raw private capture through the storage service."""

    validate_capture(capture)
    directory = storage.resolve(StorageCategory.BENCHMARKS, "stage-2-7/captures")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}.json"
    storage.atomic_write(destination, [serialize_capture(capture).encode("utf-8")])
    return cast(Path, destination)


def load_capture(storage: StorageService, name: str) -> ASRCapture:
    """Load and validate a stored raw private capture."""

    path = storage.resolve(StorageCategory.BENCHMARKS, f"stage-2-7/captures/{name}.json")
    return parse_capture(path.read_text(encoding="utf-8"))
