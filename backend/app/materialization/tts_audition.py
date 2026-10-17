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
from datetime import UTC, datetime
from pathlib import Path

from app.materialization.delivery import RATE_LIMITED_REASON, raised_attempt_status
from app.materialization.narration import inspect_wav
from app.materialization.repository import (
    ATTEMPT_RESERVED,
    ATTEMPT_RESPONDED,
    ATTEMPT_UNCERTAIN,
)
from app.materialization.types import NarrationAudio, NarrationSynthesisRequest
from app.services.storage import StorageService

AUDITION_VOICES = ("Charon", "Kore", "Achird")
MAX_AUDITION_ATTEMPTS = 9
MANIFEST_VERSION = "stage6-tts-audition-v4"
CONSUMED_BUDGET_VERSION = 1
RECOVERY_VERSION = 1

#: Result status codes.
AUDITION_OK = "OK"
AUDITION_ACTIVE = "AUDITION_ACTIVE"
AUDITION_IDENTITY_MISMATCH = "AUDITION_IDENTITY_MISMATCH"
AUDITION_HISTORY_INVALID = "AUDITION_HISTORY_INVALID"
AUDITION_BUDGET_EXHAUSTED = "AUDITION_BUDGET_EXHAUSTED"
AUDITION_HISTORY_INCOMPLETE = "AUDITION_HISTORY_INCOMPLETE"

_STATUS_RESERVED = ATTEMPT_RESERVED
_STATUS_RESPONDED = ATTEMPT_RESPONDED
_STATUS_UNCERTAIN = ATTEMPT_UNCERTAIN
_STATUS_UNAVAILABLE = "UNAVAILABLE"
_STATUS_ADMISSION_DENIED = "ADMISSION_DENIED"


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
    #: Typed provider failure category (never inferred for historical uncertainty)
    #: and its sanitized reason, kept separate from the delivery outcome.
    category: str | None = None
    reason: str | None = None
    delivery_status: str | None = None

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
            "category": self.category,
            "reason": self.reason,
            "delivery_status": self.delivery_status,
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


def _manifest_digest(manifest: Mapping[str, object]) -> str:
    canonical = json.dumps(dict(manifest), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _floor_history_complete(
    floor: Mapping[str, object] | None, manifest: Mapping[str, object] | None
) -> bool:
    """Whether a positive consumed-budget floor is backed by complete history.

    A bare count can never prove which fixed requests remain safe to invoke, so a
    positive floor is treated as incomplete unless it is linked (by digest and
    exact attempt count) to the complete retained per-request manifest.
    """

    if not floor:
        return True
    consumed = int(floor.get("consumed_attempts") or 0)
    if consumed <= 0:
        return True
    if manifest is None:
        return False
    digest = floor.get("manifest_digest")
    if not isinstance(digest, str) or not digest:
        return False
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or consumed != len(attempts):
        return False
    return digest == _manifest_digest(manifest)


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
        # consumed and must never be silently retried. Its original provider
        # category is unknown, so none is inferred.
        if entry.get("status") == _STATUS_RESERVED:
            entry["status"] = _STATUS_UNCERTAIN
            entry.setdefault("uncertainty", "CRASH_LEFT_RESERVATION")
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


def _consumed_budget_path(storage: StorageService) -> Path:
    return _audition_dir(storage) / "consumed-budget.json"


def _recovery_path(storage: StorageService) -> Path:
    return _audition_dir(storage) / "recovery.json"


def _load_consumed_budget(storage: StorageService) -> dict[str, object] | None:
    """Read the durable consumed-budget floor, failing closed on malformed history.

    The floor survives a missing ledger, a changed storage root, or a container
    restart so a lost manifest can never authorize another nine-call batch.
    """

    path = _consumed_budget_path(storage)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AuditionError(f"unreadable consumed-budget record: {error}") from None
    if not isinstance(data, dict) or data.get("version") != CONSUMED_BUDGET_VERSION:
        raise AuditionError("unrecognized consumed-budget record")
    consumed = data.get("consumed_attempts")
    probe = data.get("probe_calls", 0)
    if isinstance(consumed, bool) or not isinstance(consumed, int) or consumed < 0:
        raise AuditionError("malformed consumed-budget attempt count")
    if isinstance(probe, bool) or not isinstance(probe, int) or probe < 0:
        raise AuditionError("malformed consumed-budget probe count")
    digest = data.get("manifest_digest")
    if digest is not None and (not isinstance(digest, str) or not digest):
        raise AuditionError("malformed consumed-budget manifest digest")
    return data


def _write_consumed_budget(
    storage: StorageService,
    *,
    consumed_attempts: int,
    probe_calls: int,
    basis: str,
    captured_report_sha256: str | None = None,
    manifest_digest: str | None = None,
) -> dict[str, object]:
    """Monotonic floor write. Caller must hold the audition lock.

    An unreadable/malformed prior record fails closed (``AuditionError``) and its
    bytes are preserved — a zero-based record is never silently substituted. A
    ``manifest_digest`` links the floor to the complete retained per-request
    history; it is retained only while the consumed count is unchanged.
    """

    prior = _load_consumed_budget(storage)
    prior_consumed = int(prior["consumed_attempts"]) if prior else 0
    prior_probe = int(prior.get("probe_calls", 0)) if prior else 0
    prior_digest = prior.get("manifest_digest") if prior else None
    record: dict[str, object] = {
        "version": CONSUMED_BUDGET_VERSION,
        "consumed_attempts": max(prior_consumed, int(consumed_attempts)),
        "probe_calls": max(prior_probe, int(probe_calls)),
        "basis": basis,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    if captured_report_sha256:
        record["captured_report_sha256"] = captured_report_sha256
    if isinstance(manifest_digest, str) and manifest_digest:
        record["manifest_digest"] = manifest_digest
    elif (
        isinstance(prior_digest, str)
        and prior_digest
        and record["consumed_attempts"] == prior_consumed
    ):
        record["manifest_digest"] = prior_digest
    storage.atomic_write(
        _consumed_budget_path(storage),
        [json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")],
    )
    return record


def record_consumed_budget(
    storage: StorageService,
    *,
    consumed_attempts: int,
    probe_calls: int,
    basis: str,
    captured_report_sha256: str | None = None,
    manifest_digest: str | None = None,
) -> dict[str, object]:
    """Durably record consumed audition attempts (a monotonic floor, never reset).

    The read/modify/write is serialized under the existing audition ownership lock,
    and a malformed prior record fails closed instead of being replaced.
    """

    with _audition_lock(storage, blocking=True) as acquired:
        if not acquired:  # pragma: no cover - a blocking lock always acquires
            raise AuditionError("audition lock unavailable for a budget update")
        return _write_consumed_budget(
            storage,
            consumed_attempts=consumed_attempts,
            probe_calls=probe_calls,
            basis=basis,
            captured_report_sha256=captured_report_sha256,
            manifest_digest=manifest_digest,
        )


def record_recovery(
    storage: StorageService,
    report: Mapping[str, object],
    *,
    probe_calls: int,
    captured_report_path: Path | None = None,
) -> dict[str, object]:
    """Export recovered audition evidence and preserve the consumed budget.

    Surviving audio referenced by the captured report is verified against the
    captured SHA-256 and left in managed storage; anything missing or mismatched is
    recorded as lost evidence. Attempts are counted from the captured samples (the
    authoritative consumed set) and persisted as a durable floor. No provider
    request ID or response metadata is fabricated.
    """

    raw_samples = report.get("samples")
    samples = (
        [item for item in raw_samples if isinstance(item, Mapping)]
        if isinstance(raw_samples, list)
        else []
    )
    consumed_attempts = int(report.get("consumed_attempts") or len(samples))
    recovered: list[dict[str, object]] = []
    lost: list[dict[str, object]] = []
    for sample in samples:
        relative = str(sample.get("relative_path") or "")
        expected = str(sample.get("sha256") or "")
        entry = {
            "voice": sample.get("voice"),
            "script_id": sample.get("script_id"),
            "relative_path": relative or None,
            "expected_sha256": expected or None,
        }
        if not relative:
            lost.append({**entry, "reason": "NO_ARTIFACT_IN_CAPTURE"})
            continue
        try:
            path = storage.resolve_relative(relative)
        except Exception:
            path = None
        if path is None or not path.is_file():
            lost.append({**entry, "reason": "MISSING_ARTIFACT"})
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and actual != expected:
            lost.append({**entry, "actual_sha256": actual, "reason": "HASH_MISMATCH"})
            continue
        recovered.append({**entry, "actual_sha256": actual, "verified": True})

    report_digest = None
    if captured_report_path is not None and captured_report_path.is_file():
        report_digest = hashlib.sha256(captured_report_path.read_bytes()).hexdigest()

    record: dict[str, object] = {
        "version": RECOVERY_VERSION,
        "recorded_at": datetime.now(UTC).isoformat(),
        "consumed_attempts": consumed_attempts,
        "probe_calls": int(probe_calls),
        "captured_report_sha256": report_digest,
        "recovered": recovered,
        "lost": lost,
        "notes": (
            "Recovered from the original audition capture. The probe call remains "
            "part of call accounting. Lost audio/history is reported, never "
            "reconstructed. A positive floor without complete per-request history "
            "blocks new synthesis."
        ),
    }
    # Serialize recovery and the budget read/modify/write under the existing
    # audition ownership lock; call the inner writer to avoid a nested acquisition.
    with _audition_lock(storage, blocking=True) as acquired:
        if not acquired:  # pragma: no cover - a blocking lock always acquires
            raise AuditionError("audition lock unavailable for recovery")
        root = _audition_dir(storage)
        root.mkdir(parents=True, exist_ok=True)
        storage.atomic_write(
            _recovery_path(storage),
            [json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")],
        )
        floor = _write_consumed_budget(
            storage,
            consumed_attempts=consumed_attempts,
            probe_calls=int(probe_calls),
            basis="stage6-tts-audition recovery",
            captured_report_sha256=report_digest,
        )
    return {"recovery": record, "consumed_budget": floor}


@contextmanager
def _audition_lock(storage: StorageService, *, blocking: bool = False) -> Iterator[bool]:
    """Exclusive process-safe audition lock.

    A non-blocking acquisition (the default) yields ``False`` when the lock is
    already held (a concurrent audition reports ``AUDITION_ACTIVE``). Budget
    read/modify/write uses ``blocking=True`` so recovery and execution serialize
    instead of racing.
    """

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
            flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), flags)
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
            if isinstance(entry, dict):
                fingerprint = entry.get("request_fingerprint")
                if isinstance(fingerprint, str):
                    # Keep the live entry so a reuse/revalidation mutation is
                    # persisted by the next manifest save.
                    index[fingerprint] = entry
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
    on_rate_limited: Callable[[], None] | None = None,
) -> AuditionResult:
    """Synthesize the bounded audition set and propose one provisional voice.

    Shared HIGH admission is checked before *each* new attempt reservation, so a
    denial (including a cooldown applied by a previous 429) spends no new attempt.
    Provider availability is likewise checked before reserving. A durable
    consumed-budget floor survives a lost ledger or changed storage root and can
    never authorize another nine-call batch.
    """

    identity = audition_identity(model, api_version)
    identity_digest = _identity_digest(identity)
    persist = _persist_audition(storage)

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
            floor = _load_consumed_budget(storage)
        except AuditionError as error:
            return AuditionResult(
                samples=(),
                provisional_voice=None,
                raw_calls=0,
                status=AUDITION_HISTORY_INVALID,
                identity=identity,
                notes=f"Malformed audition history; failing closed: {error}",
            )
        floor_consumed = int(floor["consumed_attempts"]) if floor else 0
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
        if floor_consumed > 0:
            # Normalize any retained manifest (crash-left RESERVED -> UNCERTAIN)
            # BEFORE judging completeness, so the digest reflects what is retained.
            if manifest is not None:
                _save_manifest(storage, manifest)
            if not _floor_history_complete(floor, manifest):
                return AuditionResult(
                    samples=(),
                    provisional_voice=None,
                    raw_calls=0,
                    status=(
                        AUDITION_BUDGET_EXHAUSTED
                        if floor_consumed >= MAX_AUDITION_ATTEMPTS
                        else AUDITION_HISTORY_INCOMPLETE
                    ),
                    identity=identity,
                    notes=(
                        "A positive consumed-budget floor without complete retained "
                        "per-request history cannot prove which fixed requests remain "
                        "safe to invoke; new synthesis is blocked. A count alone is not "
                        "authorization, and an empty ledger is never treated as complete."
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
                    provider_ready = provider is not None and bool(
                        getattr(provider, "available", lambda: False)()
                    )
                    if not provider_ready:
                        sample.error = _STATUS_UNAVAILABLE
                        samples.append(sample)
                        continue
                    attempts = manifest.get("attempts")
                    assert isinstance(attempts, list)
                    if len(attempts) >= MAX_AUDITION_ATTEMPTS:
                        sample.error = "ATTEMPT_CEILING"
                        samples.append(sample)
                        continue
                    # Shared HIGH admission is checked before EACH new attempt
                    # reservation: a denial spends no attempt.
                    if admit is not None and not _admit(admit):
                        sample.error = _STATUS_ADMISSION_DENIED
                        sample.reason = "shared HIGH admission denied"
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
                        if not isinstance(error, Exception):
                            attempt["status"] = _STATUS_UNCERTAIN
                            attempt.setdefault("uncertainty", "CALL_INTERRUPTED")
                            _save_manifest(storage, manifest)
                            raise
                        _record_failed_attempt(attempt, sample, error)
                        if attempt.get("category") == RATE_LIMITED_REASON:
                            _notify(on_rate_limited)
                        _save_manifest(storage, manifest)
                        samples.append(sample)
                        continue
                    # Persist the accepted response BEFORE local validation.
                    attempt["status"] = _STATUS_RESPONDED
                    attempt["relative_path"] = getattr(audio, "relative_path", "")
                    attempt["sha256"] = getattr(audio, "sha256", "")
                    _save_manifest(storage, manifest)
                    sample.delivery_status = _STATUS_RESPONDED
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


def _record_failed_attempt(
    attempt: dict[str, object], sample: AuditionSample, error: Exception
) -> None:
    """Classify a raised provider failure truthfully (delivery vs content).

    Uses the same pure classification as production materialization: an ambiguous
    transport loss is UNCERTAIN, a known HTTP rejection is FAILED_KNOWN, and a
    received-but-unusable response is RESPONDED with its content validation
    failure. The typed category and a sanitized reason are persisted; an untyped
    exception is conservatively UNCERTAIN and never invents a category.
    """

    status = raised_attempt_status(error)
    category = str(getattr(error, "category", "") or "") or None
    reason = _sanitize_reason(
        getattr(error, "reason_code", None) or category or type(error).__name__,
        type(error).__name__,
    )
    attempt["status"] = status
    if category:
        attempt["category"] = category
    attempt["reason"] = reason
    if status == _STATUS_UNCERTAIN:
        attempt.setdefault(
            "uncertainty", "TRANSPORT_AMBIGUOUS" if category else "UNTYPED_EXCEPTION"
        )
    if status == _STATUS_RESPONDED:
        # A response arrived but its content was unusable: the delivery succeeded
        # and the content validation failed.
        validation = {
            "status": "DEFERRED",
            "findings": [category or reason],
            "source": "provider_response",
        }
        attempt["content_failure"] = category or reason
        attempt["validation"] = dict(validation)
        sample.fidelity = dict(validation)
    sample.delivery_status = status
    sample.category = category
    sample.reason = reason
    sample.error = f"{status}:{reason}"


def _sanitize_reason(value: object, fallback: str) -> str:
    text = str(value or "").strip() or fallback
    return text[:200]


def _notify(callback: Callable[[], None] | None) -> None:
    if callback is None:
        return
    try:
        callback()
    except Exception:
        return


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
    dependency (effective local-validator runtime identity **including the
    validation policy version**), reruns only the local validator over the
    verified audio. The revalidated result is stored under ``revalidated_fidelity``;
    the original ``fidelity`` (the historical raw findings) is never rewritten.

    A received-but-unusable response is reused with its recorded content failure.
    Historical uncertainty keeps whatever category/reason was recorded; a missing
    historical category is never inferred and the attempt is never retried.
    """

    status = str(entry.get("status") or "")
    if status == _STATUS_RESPONDED:
        relative = str(entry.get("relative_path") or "")
        sha256 = str(entry.get("sha256") or "")
        if _verify_file(storage, relative, sha256):
            sample.cached = True
            sample.delivery_status = _STATUS_RESPONDED
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
            # Rerun the separate local validation over the verified audio (no TTS)
            # and preserve the historical raw findings instead of rewriting them.
            validation = dict(fidelity(_reused_audio(sample)))
            validation.setdefault("validator_identity", current_identity)
            sample.fidelity = validation
            entry["revalidated_fidelity"] = dict(validation)
            _save_manifest(storage, manifest)
            return
        content_failure = entry.get("content_failure")
        if content_failure:
            # The response arrived but its content was unusable: reuse the recorded
            # classification rather than calling it a missing artifact.
            stored_validation = entry.get("validation")
            sample.delivery_status = _STATUS_RESPONDED
            sample.category = str(entry.get("category") or "") or None
            sample.reason = str(entry.get("reason") or content_failure)
            sample.error = f"RESPONDED:{content_failure}"
            if isinstance(stored_validation, Mapping):
                sample.fidelity = dict(stored_validation)
            return
        # A spent attempt whose artifact is missing/corrupted is not re-synthesized
        # (that would silently exceed the ceiling).
        sample.error = "MISSING_ARTIFACT"
        return
    if status == _STATUS_UNAVAILABLE:
        sample.error = _STATUS_UNAVAILABLE
        return
    # RESERVED/UNCERTAIN: unresolved uncertainty, never silently retried. Preserve
    # the recorded category/reason; never infer a missing historical category.
    sample.delivery_status = _STATUS_UNCERTAIN
    sample.category = str(entry.get("category") or "") or None
    reason = str(entry.get("reason") or "") or None
    sample.reason = reason
    sample.error = f"UNCERTAIN:{reason}" if reason else "UNCERTAIN"


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
    "AUDITION_BUDGET_EXHAUSTED",
    "AUDITION_HISTORY_INCOMPLETE",
    "AUDITION_HISTORY_INVALID",
    "AUDITION_IDENTITY_MISMATCH",
    "AUDITION_OK",
    "AUDITION_SCRIPTS",
    "AUDITION_VOICES",
    "CONSUMED_BUDGET_VERSION",
    "MAX_AUDITION_ATTEMPTS",
    "RECOVERY_VERSION",
    "AuditionError",
    "AuditionResult",
    "AuditionSample",
    "AuditionScript",
    "audition_identity",
    "audition_request_identity",
    "record_consumed_budget",
    "record_recovery",
    "run_audition",
]
