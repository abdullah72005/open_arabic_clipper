"""Bounded Stage 6 provisional-voice audition (exactly 3 voices × 3 scripts).

The audition synthesizes each of the three fixed scripts with each of the three
prebuilt candidate voices — at most nine raw synthesis requests, no automatic
retries, and no extra model/voice comparison. One audition is a *fixed*
model/API/configuration identity: the request identity (provider, model, voice,
script text hash) is persisted, so a changed model is a genuinely new audition
and can never masquerade as a cache hit or silently authorize another
model-comparison run.

Attempt budget is reserved *before* invocation and persisted durably: failures
and uncertain attempts consume the nine-attempt ceiling and are never
automatically repeated. A rerun reuses only verified matching artifacts. A
fidelity rejection/deferral or invalid audio makes a voice technically
unproven/nonviable, so a rejected sample can never propose a provisional voice.

The result proposes ONE technically viable PROVISIONAL voice; human approval is
still required.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from app.materialization.narration import inspect_wav
from app.materialization.types import NarrationAudio, NarrationSynthesisRequest
from app.services.storage import StorageService

AUDITION_VOICES = ("Charon", "Kore", "Achird")
MAX_AUDITION_ATTEMPTS = 9

_STATUS_RESPONDED = "RESPONDED"
_STATUS_UNCERTAIN = "UNCERTAIN"
_STATUS_UNAVAILABLE = "UNAVAILABLE"


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

    @property
    def viable(self) -> bool:
        fidelity_status = str(self.fidelity.get("status") or "")
        if self.error is not None:
            return False
        if not self.relative_path or not self.sha256:
            return False
        if self.duration_seconds <= 0 or self.sample_rate <= 0 or self.channels <= 0:
            return False
        if fidelity_status and fidelity_status != "ACCEPTED":
            return False
        return True

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
            "viable": self.viable,
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


def _identity_digest(identity: Mapping[str, object]) -> str:
    canonical = json.dumps(dict(identity), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _audition_dir(storage: StorageService) -> Path:
    return storage.resolve_relative(Path("benchmarks") / "stage-6" / "tts-audition")


def _load_ledger(storage: StorageService) -> dict[str, dict[str, object]]:
    path = _audition_dir(storage) / "ledger.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        return {}
    return {
        str(key): dict(value) for key, value in entries.items() if isinstance(value, Mapping)
    }


def _save_ledger(storage: StorageService, ledger: Mapping[str, dict[str, object]]) -> None:
    root = _audition_dir(storage)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "ledger.json"
    storage.atomic_write(
        path,
        [
            json.dumps(
                {"version": "stage6-tts-audition-v2", "entries": dict(ledger)},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ],
    )


def _persist_audition(storage: StorageService) -> Callable[[str, str, str, bytes], tuple[str, str]]:
    """Persist an audition WAV under the audition benchmark root (managed)."""

    def persist(voice: str, script_id: str, digest: str, data: bytes) -> tuple[str, str]:
        name = f"{voice}-{script_id}-{digest[:12]}.wav"
        relative = Path("benchmarks") / "stage-6" / "tts-audition" / name
        target = storage.resolve_relative(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        storage.atomic_write(target, [data])
        return storage.storage_relative(target), hashlib.sha256(data).hexdigest()

    return persist


def _verify_file(storage: StorageService, relative_path: str, sha256: str) -> bool:
    if not relative_path or not sha256:
        return False
    try:
        path = storage.resolve_relative(relative_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return False
    return digest == sha256


def run_audition(
    *,
    settings: object,
    storage: StorageService,
    model: str,
    voice_factory: Callable[[str], object | None],
    fidelity: Callable[[NarrationAudio], Mapping[str, object]] | None = None,
) -> AuditionResult:
    """Synthesize the bounded audition set and propose one provisional voice."""

    persist = _persist_audition(storage)
    ledger = _load_ledger(storage)
    samples: list[AuditionSample] = []
    attempts = 0
    for voice in AUDITION_VOICES:
        provider = voice_factory(voice)
        for script in AUDITION_SCRIPTS:
            identity = audition_request_identity(voice, script, model)
            digest = _identity_digest(identity)
            sample = AuditionSample(
                voice=voice, script_id=script.script_id, request_identity=identity
            )
            entry = ledger.get(digest)
            if entry is not None and entry.get("status") == _STATUS_RESPONDED:
                relative = str(entry.get("relative_path") or "")
                sha256 = str(entry.get("sha256") or "")
                if _verify_file(storage, relative, sha256):
                    sample.cached = True
                    sample.relative_path = relative
                    sample.sha256 = sha256
                    stored_fidelity = entry.get("fidelity")
                    if isinstance(stored_fidelity, Mapping):
                        sample.fidelity = dict(stored_fidelity)
                    _fill_measurements(sample, storage)
                    if fidelity is not None and not sample.fidelity:
                        sample.fidelity = dict(fidelity(_audio_ref(sample, script)))
                    samples.append(sample)
                    continue
            status = entry.get("status") if entry is not None else None
            if status in {_STATUS_UNCERTAIN, _STATUS_UNAVAILABLE}:
                # Never automatically repeat an uncertain/unavailable request.
                sample.error = str(status)
                samples.append(sample)
                continue
            if attempts >= MAX_AUDITION_ATTEMPTS:
                sample.error = "ATTEMPT_CEILING"
                samples.append(sample)
                continue
            if provider is None or not getattr(provider, "available", lambda: False)():
                sample.error = _STATUS_UNAVAILABLE
                ledger[digest] = {"status": _STATUS_UNAVAILABLE, "identity": dict(identity)}
                samples.append(sample)
                continue
            attempts += 1
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
                _bind_persist(provider, persist, voice, script.script_id, digest)
                audio = provider.synthesize(request)  # type: ignore[attr-defined]
            except Exception as error:  # noqa: BLE001 - report honestly per sample
                sample.error = f"UNCERTAIN:{type(error).__name__}"
                ledger[digest] = {"status": _STATUS_UNCERTAIN, "identity": dict(identity)}
                samples.append(sample)
                continue
            sample.relative_path = audio.relative_path
            sample.sha256 = audio.sha256
            sample.duration_seconds = audio.duration_seconds
            sample.sample_rate = audio.sample_rate
            sample.channels = audio.channels
            if fidelity is not None:
                sample.fidelity = dict(fidelity(audio))
            ledger[digest] = {
                "status": _STATUS_RESPONDED,
                "identity": dict(identity),
                "relative_path": sample.relative_path,
                "sha256": sample.sha256,
                "fidelity": dict(sample.fidelity),
            }
            samples.append(sample)
        _release(provider)

    _save_ledger(storage, ledger)
    viable = [voice for voice in AUDITION_VOICES if _voice_viable(samples, voice)]
    provisional = viable[0] if viable else None
    notes = (
        "PROVISIONAL only; human quality approval is required before freezing."
        if provisional
        else "No technically viable voice; TTS quality/provider unavailable."
    )
    return AuditionResult(
        samples=tuple(samples),
        provisional_voice=provisional,
        raw_calls=attempts,
        notes=notes,
    )


def _release(provider: object | None) -> None:
    if provider is None:
        return
    for name in ("release", "close"):
        closer = getattr(provider, name, None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
            return


def _voice_viable(samples: list[AuditionSample], voice: str) -> bool:
    voice_samples = [sample for sample in samples if sample.voice == voice]
    return bool(voice_samples) and all(sample.viable for sample in voice_samples)


def _audio_ref(sample: AuditionSample, script: AuditionScript) -> NarrationAudio:
    return NarrationAudio(
        chunk_id=f"{sample.voice}-{sample.script_id}",
        block_index=0,
        text=script.text,
        relative_path=sample.relative_path,
        sha256=sample.sha256,
        size_bytes=0,
        duration_seconds=sample.duration_seconds,
        sample_rate=sample.sample_rate,
        channels=sample.channels,
        container="wav",
    )


def _bind_persist(
    provider: object,
    persist: Callable[[str, str, str, bytes], tuple[str, str]],
    voice: str,
    script_id: str,
    digest: str,
) -> None:
    def bound(chunk_id: str, data: bytes) -> tuple[str, str]:
        return persist(voice, script_id, digest, data)

    setattr(provider, "_persist", bound)


def _fill_measurements(sample: AuditionSample, storage: StorageService) -> None:
    try:
        data = storage.resolve_relative(sample.relative_path).read_bytes()
        info = inspect_wav(data)
    except (ValueError, OSError, Exception):
        sample.error = "INVALID_WAV"
        return
    sample.duration_seconds = info.duration_seconds
    sample.sample_rate = info.sample_rate
    sample.channels = info.channels


__all__ = [
    "AUDITION_SCRIPTS",
    "AUDITION_VOICES",
    "MAX_AUDITION_ATTEMPTS",
    "AuditionResult",
    "AuditionSample",
    "AuditionScript",
    "audition_request_identity",
    "run_audition",
]
