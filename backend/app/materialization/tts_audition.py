"""Bounded Stage 6 provisional-voice audition (exactly 3 voices × 3 scripts).

The audition synthesizes each of the three fixed scripts with each of the three
prebuilt candidate voices — at most nine raw synthesis requests, no automatic
retries, and no extra model/voice comparison.

An audition is one *fixed* identity: the model/API/configuration plus the
prescribed voices and scripts. The identity and every attempt live in one
versioned, StorageService-resolved durable manifest guarded by an exclusive
process-safe lock:

* a concurrent invocation reports ``AUDITION_ACTIVE`` and makes zero calls;
* a changed identity (e.g. another model) reports ``AUDITION_IDENTITY_MISMATCH``
  and makes zero calls — it never silently authorizes a second nine-call run;
* each attempt is persisted as ``RESERVED`` *before* the HTTP operation, its
  response is persisted *before* local validation, and an unresolved reservation
  is recovered as ``UNCERTAIN``;
* the nine-attempt ceiling is derived from durable history (including failures
  and uncertainty), so a crash, a deleted/missing artifact, or a changed model
  can never reset the spent budget;
* a malformed existing manifest fails closed rather than resetting to an empty
  budget.

A fidelity rejection/deferral or invalid audio makes a voice technically
unproven/nonviable; a *missing* fidelity result is also nonviable. The result
proposes ONE technically viable PROVISIONAL voice; human approval is still
required.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from app.materialization.narration import inspect_wav
from app.materialization.types import NarrationAudio, NarrationSynthesisRequest
from app.services.storage import StorageService

AUDITION_VOICES = ("Charon", "Kore", "Achird")
MAX_AUDITION_ATTEMPTS = 9
MANIFEST_VERSION = "stage6-tts-audition-v3"

#: Result status codes.
AUDITION_OK = "OK"
AUDITION_ACTIVE = "AUDITION_ACTIVE"
AUDITION_IDENTITY_MISMATCH = "AUDITION_IDENTITY_MISMATCH"
AUDITION_HISTORY_INVALID = "AUDITION_HISTORY_INVALID"

_STATUS_RESERVED = "RESERVED"
_STATUS_RESPONDED = "RESPONDED"
_STATUS_UNCERTAIN = "UNCERTAIN"
_STATUS_UNAVAILABLE = "UNAVAILABLE"


class AuditionError(RuntimeError):
    """Raised when a durable audition history cannot be trusted."""


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
        if self.error is not None:
            return False
        if not self.relative_path or not self.sha256:
            return False
        if self.duration_seconds <= 0 or self.sample_rate <= 0 or self.channels <= 0:
            return False
        # Missing fidelity evidence never implies viability; only a known ACCEPTED
        # local audio-backed validation is acceptable.
        if str(self.fidelity.get("status") or "") != "ACCEPTED":
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
    status: str = AUDITION_OK
    identity: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "samples": [sample.as_dict() for sample in self.samples],
            "provisional_voice": self.provisional_voice,
            "raw_calls": self.raw_calls,
            "identity": dict(self.identity),
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


def audition_identity(model: str, api_version: str = "v1beta") -> dict[str, object]:
    """The ONE fixed audition identity (model/API/configuration + fixed set)."""

    return {
        "provider": "gemini-tts",
        "model": model,
        "api_version": api_version,
        "voices": list(AUDITION_VOICES),
        "scripts": [
            {
                "script_id": script.script_id,
                "language": script.language,
                "text_sha256": hashlib.sha256(script.text.encode("utf-8")).hexdigest(),
            }
            for script in AUDITION_SCRIPTS
        ],
    }


def _identity_digest(identity: Mapping[str, object]) -> str:
    canonical = json.dumps(dict(identity), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _audition_dir(storage: StorageService) -> Path:
    return storage.resolve_relative(Path("benchmarks") / "stage-6" / "tts-audition")


def _manifest_path(storage: StorageService) -> Path:
    return _audition_dir(storage) / "ledger.json"


def _new_manifest(identity: Mapping[str, object]) -> dict[str, object]:
    return {
        "version": MANIFEST_VERSION,
        "identity": dict(identity),
        "identity_digest": _identity_digest(identity),
        "attempts": [],
    }


def _load_manifest(storage: StorageService) -> dict[str, object] | None:
    """Load the durable manifest, failing closed on malformed history.

    A missing manifest is a fresh audition. A present-but-malformed manifest is
    never treated as an empty budget: it raises :class:`AuditionError` so the
    spent budget cannot be silently reset.
    """

    path = _manifest_path(storage)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise AuditionError(f"unreadable audition history: {error}") from None
    if not isinstance(payload, dict) or payload.get("version") != MANIFEST_VERSION:
        raise AuditionError("unrecognized audition history version")
    identity = payload.get("identity")
    digest = payload.get("identity_digest")
    attempts = payload.get("attempts")
    if not isinstance(identity, Mapping) or not isinstance(digest, str) or not isinstance(
        attempts, list
    ):
        raise AuditionError("malformed audition history")
    if digest != _identity_digest(identity):
        raise AuditionError("audition history identity digest mismatch")
    normalized: list[dict[str, object]] = []
    for item in attempts:
        if not isinstance(item, Mapping) or not isinstance(item.get("attempt_id"), str):
            raise AuditionError("malformed audition attempt entry")
        entry = dict(item)
        # A reservation left by a crash is unresolved uncertainty: it is budget
        # consumed and must never be silently retried.
        if entry.get("status") == _STATUS_RESERVED:
            entry["status"] = _STATUS_UNCERTAIN
        normalized.append(entry)
    return {
        "version": MANIFEST_VERSION,
        "identity": dict(identity),
        "identity_digest": digest,
        "attempts": normalized,
    }


def _save_manifest(storage: StorageService, manifest: Mapping[str, object]) -> None:
    root = _audition_dir(storage)
    root.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(storage)
    storage.atomic_write(
        path,
        [json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True).encode("utf-8")],
    )


@contextmanager
def _audition_lock(storage: StorageService) -> Iterator[bool]:
    """Exclusive process-safe audition lock; yields False when already held."""

    root = _audition_dir(storage)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "audition.lock"
    handle = open(lock_path, "a+", encoding="utf-8")
    acquired = False
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-posix fallback
            acquired = True
            yield True
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            acquired = False
        yield acquired
    finally:
        if acquired:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        handle.close()


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


def _attempts_by_fingerprint(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    attempts = manifest.get("attempts")
    if isinstance(attempts, list):
        for entry in attempts:
            if isinstance(entry, Mapping):
                fingerprint = entry.get("request_fingerprint")
                if isinstance(fingerprint, str):
                    index[fingerprint] = dict(entry)
    return index


def run_audition(
    *,
    settings: object,
    storage: StorageService,
    model: str,
    voice_factory: Callable[[str], object | None],
    fidelity: Callable[[NarrationAudio], Mapping[str, object]] | None = None,
    api_version: str = "v1beta",
    admit: Callable[[], bool] | None = None,
) -> AuditionResult:
    """Synthesize the bounded audition set and propose one provisional voice.

    Provider availability and shared admission are checked *before* any durable
    attempt is reserved: a missing key or denied admission spends zero synthesis
    attempts and never poisons the durable manifest, so configuring a provider
    afterwards can still run the audition.
    """

    identity = audition_identity(model, api_version)
    identity_digest = _identity_digest(identity)
    persist = _persist_audition(storage)
    forced_unavailable = admit is not None and not _admit(admit)

    with _audition_lock(storage) as acquired:
        if not acquired:
            return AuditionResult(
                samples=(),
                provisional_voice=None,
                raw_calls=0,
                status=AUDITION_ACTIVE,
                identity=identity,
                notes="Another audition is already running (audition lock held).",
            )
        try:
            manifest = _load_manifest(storage)
        except AuditionError as error:
            return AuditionResult(
                samples=(),
                provisional_voice=None,
                raw_calls=0,
                status=AUDITION_HISTORY_INVALID,
                identity=identity,
                notes=f"Malformed audition history; failing closed: {error}",
            )
        if manifest is not None and manifest.get("identity_digest") != identity_digest:
            return AuditionResult(
                samples=(),
                provisional_voice=None,
                raw_calls=0,
                status=AUDITION_IDENTITY_MISMATCH,
                identity=identity,
                notes=(
                    "Configured model/API differs from the existing audition identity; "
                    "a new model comparison is not automatically authorized."
                ),
            )
        if manifest is None:
            manifest = _new_manifest(identity)
            _save_manifest(storage, manifest)
        else:
            # Persist the normalized reservations (crash-left RESERVED -> UNCERTAIN).
            _save_manifest(storage, manifest)

        samples: list[AuditionSample] = []
        raw_calls = 0
        for voice in AUDITION_VOICES:
            provider = voice_factory(voice)
            try:
                for script in AUDITION_SCRIPTS:
                    identity_entry = audition_request_identity(voice, script, model)
                    fingerprint = _identity_digest(identity_entry)
                    sample = AuditionSample(
                        voice=voice, script_id=script.script_id, request_identity=identity_entry
                    )
                    existing = _attempts_by_fingerprint(manifest).get(fingerprint)
                    if existing is not None:
                        _reuse_attempt(sample, existing, storage, manifest, fidelity, settings)
                        samples.append(sample)
                        continue
                    # Provider availability and shared admission are checked BEFORE
                    # reserving a durable attempt: an unavailable provider spends no
                    # attempt and never poisons the manifest.
                    provider_ready = provider is not None and bool(
                        getattr(provider, "available", lambda: False)()
                    )
                    if not provider_ready or forced_unavailable:
                        sample.error = _STATUS_UNAVAILABLE
                        samples.append(sample)
                        continue
                    attempts = manifest.get("attempts")
                    assert isinstance(attempts, list)
                    if len(attempts) >= MAX_AUDITION_ATTEMPTS:
                        sample.error = "ATTEMPT_CEILING"
                        samples.append(sample)
                        continue
                    # Reserve durably BEFORE the HTTP operation.
                    attempt: dict[str, object] = {
                        "attempt_id": _attempt_id(),
                        "voice": voice,
                        "script_id": script.script_id,
                        "request_fingerprint": fingerprint,
                        "ordinal": len(attempts) + 1,
                        "status": _STATUS_RESERVED,
                        "relative_path": "",
                        "sha256": "",
                        "fidelity": {},
                    }
                    attempts.append(attempt)
                    _save_manifest(storage, manifest)
                    request = NarrationSynthesisRequest(
                        chunk_id=f"{voice}-{script.script_id}",
                        block_index=0,
                        text=script.text,
                        language_hint=script.language,
                        provider="gemini-tts",
                        model=model,
                        api_version=api_version,
                        voice=voice,
                    )
                    raw_calls += 1
                    try:
                        _bind_persist(provider, persist, voice, script.script_id, fingerprint)
                        audio = provider.synthesize(request)  # type: ignore[attr-defined]
                    except BaseException as error:  # noqa: BLE001 - report honestly
                        # A crash/exception may or may not have reached the
                        # provider: record uncertainty, never a retry.
                        attempt["status"] = _STATUS_UNCERTAIN
                        _save_manifest(storage, manifest)
                        if isinstance(error, Exception):
                            sample.error = f"UNCERTAIN:{type(error).__name__}"
                            samples.append(sample)
                            continue
                        raise
                    # Persist the accepted response BEFORE local validation.
                    attempt["status"] = _STATUS_RESPONDED
                    attempt["relative_path"] = getattr(audio, "relative_path", "")
                    attempt["sha256"] = getattr(audio, "sha256", "")
                    _save_manifest(storage, manifest)
                    sample.relative_path = str(getattr(audio, "relative_path", "") or "")
                    sample.sha256 = str(getattr(audio, "sha256", "") or "")
                    sample.duration_seconds = float(getattr(audio, "duration_seconds", 0.0) or 0.0)
                    sample.sample_rate = int(getattr(audio, "sample_rate", 0) or 0)
                    sample.channels = int(getattr(audio, "channels", 0) or 0)
                    if fidelity is not None:
                        sample.fidelity = dict(fidelity(audio))
                        sample.fidelity.setdefault(
                            "validator_identity", _validator_identity(fidelity)
                        )
                        attempt["fidelity"] = dict(sample.fidelity)
                        _save_manifest(storage, manifest)
                    samples.append(sample)
            finally:
                _release(provider)

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
            raw_calls=raw_calls,
            notes=notes,
            identity=identity,
        )


def _admit(admit: Callable[[], bool]) -> bool:
    try:
        return bool(admit())
    except Exception:
        return False


def _validator_identity(
    fidelity: Callable[[NarrationAudio], Mapping[str, object]],
) -> dict[str, object]:
    """The effective local-validation runtime identity (or an explicit sentinel)."""

    identity_fn = getattr(fidelity, "runtime_identity", None)
    if callable(identity_fn):
        try:
            return dict(identity_fn())
        except Exception:
            return {"runtime_identity": "ERROR"}
    return {"runtime_identity": "ABSENT"}


def _reused_audio(sample: AuditionSample) -> NarrationAudio:
    script = next(
        (item for item in AUDITION_SCRIPTS if item.script_id == sample.script_id), None
    )
    return NarrationAudio(
        chunk_id=f"{sample.voice}-{sample.script_id}",
        block_index=0,
        text=script.text if script is not None else "",
        relative_path=sample.relative_path,
        sha256=sample.sha256,
        size_bytes=0,
        duration_seconds=sample.duration_seconds,
        sample_rate=sample.sample_rate,
        channels=sample.channels,
        container="wav",
        provider_identity=dict(sample.request_identity),
        validation={},
    )


def _reuse_attempt(
    sample: AuditionSample,
    entry: Mapping[str, object],
    storage: StorageService,
    manifest: dict[str, object],
    fidelity: Callable[[NarrationAudio], Mapping[str, object]] | None = None,
    settings: object | None = None,
) -> None:
    """Reuse one persisted attempt; re-run missing/deferred local validation.

    The accepted synthesis is reused without any new raw call. Local validation is
    a *separate* checkpoint: a missing/deferred result, or a changed validation
    dependency (effective local-validator runtime identity), reruns only the local
    validator over the verified audio; a known ACCEPTED/REJECTED result is reused
    unchanged when its dependency identity still matches.
    """

    status = str(entry.get("status") or "")
    if status == _STATUS_RESPONDED:
        relative = str(entry.get("relative_path") or "")
        sha256 = str(entry.get("sha256") or "")
        if _verify_file(storage, relative, sha256):
            sample.cached = True
            sample.relative_path = relative
            sample.sha256 = sha256
            stored_fidelity = entry.get("fidelity")
            stored = dict(stored_fidelity) if isinstance(stored_fidelity, Mapping) else {}
            _fill_measurements(sample, storage)
            if fidelity is None:
                sample.fidelity = stored
                return
            current_identity = _validator_identity(fidelity)
            stored_identity = stored.get("validator_identity")
            stored_status = str(stored.get("status") or "")
            identity_changed = stored_identity != current_identity
            if stored_status in {"ACCEPTED", "REJECTED"} and not identity_changed:
                sample.fidelity = stored
                return
            # Rerun the separate local validation over the verified audio (no TTS).
            validation = dict(fidelity(_reused_audio(sample)))
            validation.setdefault("validator_identity", current_identity)
            sample.fidelity = validation
            entry["fidelity"] = dict(validation)
            _save_manifest(storage, manifest)
            return
        # A spent attempt whose artifact is missing/corrupted is not re-synthesized
        # (that would silently exceed the ceiling).
        sample.error = "MISSING_ARTIFACT"
        return
    if status == _STATUS_UNAVAILABLE:
        sample.error = _STATUS_UNAVAILABLE
        return
    # RESERVED/UNCERTAIN: unresolved uncertainty, never silently retried.
    sample.error = "UNCERTAIN"


def _attempt_id() -> str:
    import uuid

    return str(uuid.uuid4())


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


def _fill_measurements(sample: AuditionSample, storage: StorageService) -> None:
    try:
        data = storage.resolve_relative(sample.relative_path).read_bytes()
        info = inspect_wav(data)
    except Exception:
        sample.error = "INVALID_WAV"
        return
    sample.duration_seconds = info.duration_seconds
    sample.sample_rate = info.sample_rate
    sample.channels = info.channels


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


__all__ = [
    "AUDITION_ACTIVE",
    "AUDITION_HISTORY_INVALID",
    "AUDITION_IDENTITY_MISMATCH",
    "AUDITION_OK",
    "AUDITION_SCRIPTS",
    "AUDITION_VOICES",
    "MAX_AUDITION_ATTEMPTS",
    "AuditionError",
    "AuditionResult",
    "AuditionSample",
    "AuditionScript",
    "audition_identity",
    "audition_request_identity",
    "run_audition",
]
