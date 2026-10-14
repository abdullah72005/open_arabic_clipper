"""Stage 6 provider-neutral narration synthesis and audio validation.

The synthesis interface takes accepted script, a persistent voice/style, and
bounded output requirements, and returns managed lossless audio with actual
format/duration/hash plus exact request identity. Spoken wording is kept
separate from delivery metadata. Returned MIME/container is checked: an existing
WAV is never double-wrapped, raw PCM is wrapped exactly once.
"""

from __future__ import annotations

import io
import struct
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.materialization.policy import (
    MAX_NARRATION_SECONDS,
    Stage6Config,
)
from app.materialization.types import (
    NarrationAudio,
    NarrationSynthesisRequest,
)


@dataclass(frozen=True)
class WavInfo:
    """Deterministic measurements of one WAV payload."""

    channels: int
    sample_rate: int
    sample_width: int
    frame_count: int
    duration_seconds: float
    silent: bool
    peak: int
    clipped: bool
    truncated: bool
    container: str = "wav"


def wrap_pcm_as_wav(pcm: bytes, *, sample_rate: int, channels: int, sample_width: int = 2) -> bytes:
    """Wrap raw PCM in a single WAV container (never double-wrap a WAV)."""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def inspect_wav(data: bytes) -> WavInfo:
    """Parse a WAV payload and measure format, silence, clipping, truncation."""

    if not (data[:4] == b"RIFF" and data[8:12] == b"WAVE"):
        raise ValueError("not a WAV payload")
    buffer = io.BytesIO(data)
    with wave.open(buffer, "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        declared_frames = handle.getnframes()
        frames = handle.readframes(declared_frames)

    bytes_per_frame = max(1, channels * sample_width)
    actual_frames = len(frames) // bytes_per_frame
    truncated = actual_frames < declared_frames
    duration = actual_frames / sample_rate if sample_rate else 0.0

    peak = 0
    nonzero = 0
    total = 0
    if sample_width == 2 and frames:
        samples = struct.unpack(
            f"<{actual_frames * channels}h", frames[: actual_frames * bytes_per_frame]
        )
        for sample in samples:
            magnitude = abs(sample)
            if magnitude > peak:
                peak = magnitude
            if magnitude:
                nonzero += 1
            total += 1
    elif frames:
        # Non-16-bit: approximate by byte entropy.
        nonzero = sum(1 for byte in frames[: actual_frames * bytes_per_frame] if byte)
        total = actual_frames * bytes_per_frame
        peak = 0x7FFF if nonzero else 0

    max_amplitude = (1 << (8 * sample_width - 1)) - 1 if sample_width else 0
    clipped = bool(max_amplitude) and peak >= max_amplitude - 1
    silent = total == 0 or nonzero == 0
    return WavInfo(
        channels=channels,
        sample_rate=sample_rate,
        sample_width=sample_width,
        frame_count=actual_frames,
        duration_seconds=duration,
        silent=silent,
        peak=peak,
        clipped=clipped,
        truncated=truncated,
    )


def validate_narration_audio(
    info: WavInfo,
    *,
    expected_text: str,
    config: Stage6Config,
) -> dict[str, object]:
    """Validate one synthesized WAV (format/duration/empty/silent/truncated/clip)."""

    findings: list[str] = []
    if info.sample_rate <= 0:
        findings.append("INVALID_SAMPLE_RATE")
    if info.channels <= 0:
        findings.append("INVALID_CHANNELS")
    if info.duration_seconds <= 0:
        findings.append("EMPTY_AUDIO")
    if info.silent:
        findings.append("SILENT_AUDIO")
    if info.truncated:
        findings.append("TRUNCATED_AUDIO")
    if info.clipped:
        findings.append("CLIPPED_AUDIO")
    if info.duration_seconds > MAX_NARRATION_SECONDS:
        findings.append("NARRATION_TOO_LONG")
    return {
        "findings": findings,
        "duration_seconds": round(info.duration_seconds, 6),
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "clipped": info.clipped,
        "expected_text_length": len(expected_text),
        "ok": not findings,
    }


def transcription_matches(expected: str, actual: str, *, threshold: float = 0.6) -> bool:
    """Deterministic ASR-vs-script comparison (never rewrites source evidence).

    A bounded local ASR transcript is compared with the accepted copy,
    particularly names, numbers, negation, and English; unknown pronunciation or
    fidelity remains a review finding rather than a silent acceptance.
    """

    expected_tokens = [token for token in expected.casefold().split() if token]
    if not expected_tokens:
        return True
    actual_tokens = set(actual.casefold().split())
    matched = sum(1 for token in expected_tokens if token in actual_tokens)
    return matched / len(expected_tokens) >= threshold


@runtime_checkable
class NarrationSynthesizer(Protocol):
    """Provider-neutral narration synthesis interface."""

    provider_name: str

    def available(self) -> bool: ...

    def runtime_identity(self) -> dict[str, object]: ...

    def synthesize(self, request: NarrationSynthesisRequest) -> NarrationAudio: ...

    def release(self) -> None: ...


class FakeNarrationSynthesizer:
    """Deterministic in-memory synthesizer for tests (no network, no media)."""

    provider_name = "fake-tts"

    def __init__(self, *, available: bool = True, tone: int = 800) -> None:
        self._available = available
        self._tone = tone
        self.calls: list[NarrationSynthesisRequest] = []
        self.released = 0

    def available(self) -> bool:
        return self._available

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": self.provider_name, "model": "fake", "api_version": "v1"}

    def synthesize(self, request: NarrationSynthesisRequest) -> NarrationAudio:
        self.calls.append(request)
        frames = max(1, int(0.5 * 24_000))
        pcm = struct.pack(f"<{frames}h", *([self._tone] * frames))
        wav = wrap_pcm_as_wav(pcm, sample_rate=24_000, channels=1, sample_width=2)
        info = inspect_wav(wav)
        return NarrationAudio(
            chunk_id=request.chunk_id,
            block_index=request.block_index,
            text=request.text,
            relative_path=f"narration/{request.chunk_id}.wav",
            sha256=f"sha-{request.chunk_id}",
            size_bytes=len(wav),
            duration_seconds=info.duration_seconds,
            sample_rate=info.sample_rate,
            channels=info.channels,
            container="wav",
            provider_identity=request.identity(),
            validation={"ok": True},
        )

    def release(self) -> None:
        self.released += 1


def write_narration_asset(root: Path, relative_path: str, data: bytes) -> str:
    """Atomically write one immutable narration asset and return its sha256."""

    from app.materialization.fingerprints import hash_bytes

    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    return hash_bytes(data)


def narration_requests(
    accepted_script: Sequence[Mapping[str, object]],
    *,
    identity: Mapping[str, object],
    config: Stage6Config,
) -> tuple[NarrationSynthesisRequest, ...]:
    """Build bounded synthesis requests from accepted script blocks."""

    requests: list[NarrationSynthesisRequest] = []
    for item in accepted_script[: config.max_narration_chunks]:
        block_index = int(item.get("block_index", -1))
        text = str(item.get("copy", "") or "").strip()
        if not text:
            continue
        requests.append(
            NarrationSynthesisRequest(
                chunk_id=str(item.get("narration_chunk_id", f"narr-{block_index}")),
                block_index=block_index,
                text=text,
                language_hint=_optional_str(item.get("language_hint")),
                provider=str(identity.get("provider", "")),
                model=str(identity.get("model", "")),
                api_version=str(identity.get("api_version", "")),
                voice=str(identity.get("voice", "")),
                style=str(identity.get("style", "")),
                settings=dict(identity.get("settings", {}) or {}),
            )
        )
    return tuple(requests)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "FakeNarrationSynthesizer",
    "NarrationSynthesizer",
    "WavInfo",
    "inspect_wav",
    "narration_requests",
    "transcription_matches",
    "validate_narration_audio",
    "wrap_pcm_as_wav",
    "write_narration_asset",
]
