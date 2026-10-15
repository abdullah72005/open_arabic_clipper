"""Bounded Stage 6 provisional-voice audition (exactly 3 voices × 3 scripts).

The audition synthesizes each of the three fixed scripts with each of the three
prebuilt candidate voices — at most nine raw synthesis requests, no automatic
retries, and no extra model/voice comparison. Samples are cached by exact request
identity so a repeated audition never re-spends quota. Each sample records the
request identity, the actual managed audio, and an optional local audio-backed
fidelity check.

The result proposes ONE technically viable PROVISIONAL voice. Human quality
approval is still required before the voice is considered frozen; if no sample is
technically viable the result reports that honestly rather than selecting a bad
voice to force output.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from app.materialization.narration import inspect_wav
from app.materialization.types import NarrationAudio, NarrationSynthesisRequest
from app.services.storage import StorageService

AUDITION_VOICES = ("Charon", "Kore", "Achird")


@dataclass(frozen=True)
class AuditionScript:
    script_id: str
    text: str
    language: str


AUDITION_SCRIPTS = (
    AuditionScript(
        script_id="egyptian-conversational",
        text="بص يا صاحبي، الموضوع مش بالبساطة اللي الناس فاكراها.",
        language="ar-EG",
    ),
    AuditionScript(
        script_id="arabic-english-code-switch",
        text="الـ rollout الجديد أثّر على الـ conversion rate بشكل واضح.",
        language="ar-EG",
    ),
    AuditionScript(
        script_id="english",
        text="The new rollout clearly shifted the conversion rate.",
        language="en",
    ),
)


@dataclass
class AuditionSample:
    voice: str
    script_id: str
    request_identity: dict[str, object]
    relative_path: str = ""
    sha256: str = ""
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    cached: bool = False
    fidelity: dict[str, object] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "voice": self.voice,
            "script_id": self.script_id,
            "request_identity": dict(self.request_identity),
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "duration_seconds": round(self.duration_seconds, 6),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "cached": self.cached,
            "fidelity": dict(self.fidelity),
            "error": self.error,
        }


@dataclass
class AuditionResult:
    samples: tuple[AuditionSample, ...]
    provisional_voice: str | None
    raw_calls: int
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": [sample.as_dict() for sample in self.samples],
            "provisional_voice": self.provisional_voice,
            "raw_calls": self.raw_calls,
            "notes": self.notes,
        }


def audition_request_identity(voice: str, script: AuditionScript, model: str) -> dict[str, object]:
    return {
        "provider": "gemini-tts",
        "model": model,
        "voice": voice,
        "script_id": script.script_id,
        "language": script.language,
        "text_sha256": hashlib.sha256(script.text.encode("utf-8")).hexdigest(),
    }


def _audition_persist(
    storage: StorageService,
) -> Callable[[str, str, bytes], tuple[str, str]]:
    """Persist an audition WAV under the audition benchmark root (managed)."""

    def persist(voice: str, script_id: str, data: bytes) -> tuple[str, str]:
        relative = Path("benchmarks") / "stage-6" / "tts-audition" / f"{voice}-{script_id}.wav"
        target = storage.resolve_relative(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        storage.atomic_write(target, [data])
        return storage.storage_relative(target), hashlib.sha256(data).hexdigest()

    return persist


def run_audition(
    *,
    settings: object,
    storage: StorageService,
    model: str,
    voice_factory: Callable[[str], object | None],
    fidelity: Callable[[NarrationAudio], Mapping[str, object]] | None = None,
) -> AuditionResult:
    """Synthesize the bounded audition set and propose one provisional voice."""

    persist = _audition_persist(storage)
    samples: list[AuditionSample] = []
    raw_calls = 0
    per_voice_ok: dict[str, bool] = {voice: True for voice in AUDITION_VOICES}
    for voice in AUDITION_VOICES:
        provider = voice_factory(voice)
        for script in AUDITION_SCRIPTS:
            identity = audition_request_identity(voice, script, model)
            sample = AuditionSample(
                voice=voice, script_id=script.script_id, request_identity=identity
            )
            if provider is None or not getattr(provider, "available", lambda: False)():
                sample.error = "VOICE_UNAVAILABLE"
                per_voice_ok[voice] = False
                samples.append(sample)
                continue
            cached_path = _cached_path(storage, voice, script.script_id)
            if cached_path is not None:
                sample.cached = True
                _fill_from_path(sample, cached_path, storage)
                samples.append(sample)
                continue
            # Bind a per-voice persist closure so the shared synthesizer writes the
            # audition artifact under the audition benchmark root.
            request = NarrationSynthesisRequest(
                chunk_id=f"{voice}-{script.script_id}",
                block_index=0,
                text=script.text,
                language_hint=script.language,
                provider="gemini-tts",
                model=model,
                api_version="v1beta",
                voice=voice,
            )
            try:
                _bind_persist(provider, persist, voice, script.script_id)
                audio = provider.synthesize(request)  # type: ignore[attr-defined]
                raw_calls += 1
                sample.relative_path = audio.relative_path
                sample.sha256 = audio.sha256
                sample.duration_seconds = audio.duration_seconds
                sample.sample_rate = audio.sample_rate
                sample.channels = audio.channels
                if fidelity is not None:
                    sample.fidelity = dict(fidelity(audio))
            except Exception as error:  # noqa: BLE001 - report honestly per sample
                sample.error = type(error).__name__
                per_voice_ok[voice] = False
            samples.append(sample)

    viable = [voice for voice in AUDITION_VOICES if per_voice_ok[voice]]
    provisional = viable[0] if viable else None
    notes = (
        "PROVISIONAL only; human quality approval is required before freezing."
        if provisional
        else "No technically viable voice; TTS quality/provider unavailable."
    )
    return AuditionResult(
        samples=tuple(samples),
        provisional_voice=provisional,
        raw_calls=raw_calls,
        notes=notes,
    )


def _bind_persist(
    provider: object,
    persist: Callable[[str, str, bytes], tuple[str, str]],
    voice: str,
    script_id: str,
) -> None:
    def bound(chunk_id: str, data: bytes) -> tuple[str, str]:
        return persist(voice, script_id, data)

    setattr(provider, "_persist", bound)


def _cached_path(storage: StorageService, voice: str, script_id: str) -> Path | None:
    relative = Path("benchmarks") / "stage-6" / "tts-audition" / f"{voice}-{script_id}.wav"
    try:
        path = storage.resolve_relative(relative)
    except Exception:
        return None
    return path if path.is_file() else None


def _fill_from_path(sample: AuditionSample, path: Path, storage: StorageService) -> None:
    data = path.read_bytes()
    sample.relative_path = storage.storage_relative(path)
    sample.sha256 = hashlib.sha256(data).hexdigest()
    try:
        info = inspect_wav(data)
    except ValueError:
        sample.error = "INVALID_WAV"
        return
    sample.duration_seconds = info.duration_seconds
    sample.sample_rate = info.sample_rate
    sample.channels = info.channels


__all__ = [
    "AUDITION_SCRIPTS",
    "AUDITION_VOICES",
    "AuditionResult",
    "AuditionSample",
    "AuditionScript",
    "audition_request_identity",
    "run_audition",
]
