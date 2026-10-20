"""Stage 6 SOL p5 regressions: export integrity, audition accounting, recovery.

These convert the SOL diagnostic probes (which asserted the observed defects) into
regressions that assert the corrected behavior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.core.settings import get_settings
from app.materialization.acceptance_export import (
    CatalogIntegrityError,
    _publish_case_package,
    _read_catalog,
)
from app.materialization.narration import inspect_wav, wrap_pcm_as_wav
from app.materialization.providers import MaterializationProviderError
from app.materialization.tts_audition import (
    AUDITION_BUDGET_EXHAUSTED,
    AUDITION_SCRIPTS,
    AUDITION_VOICES,
    MANIFEST_VERSION,
    MAX_AUDITION_ATTEMPTS,
    _identity_digest,
    audition_identity,
    audition_request_identity,
    record_consumed_budget,
    record_recovery,
    run_audition,
)
from app.materialization.types import NarrationAudio
from app.services.storage import StorageService


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_package(base: Path) -> str:
    """Write a self-consistent immutable package and return its identity."""

    base.mkdir(parents=True, exist_ok=True)
    artifact = base / "final.mp4"
    artifact.write_bytes(b"genuine-bytes")
    artifact_sha = _sha256(artifact)
    (base / "index.json").write_text(
        json.dumps(
            {
                "genuine_acceptance": True,
                "artifact_file": "final.mp4",
                "render_artifact_sha256": artifact_sha,
                "authored_ass_expected_sha256": "",
                "authored_ass_sha256": None,
                # A genuinely valid package carries an explicitly empty
                # authoritative inventory consistent with no bound declarations.
                "identity": {"assets": []},
                "readiness": {"assets": []},
                "assets": [],
            }
        ),
        encoding="utf-8",
    )
    lines = [
        f"{_sha256(entry)}  {entry.relative_to(base).as_posix()}"
        for entry in sorted(base.rglob("*"))
        if entry.is_file() and entry.name != "SHA256SUMS"
    ]
    (base / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256(base / "SHA256SUMS")


# --- Export integrity -------------------------------------------------------


def _install_package(root: Path, case: str) -> tuple[Path, str]:
    """Build a package in a temporary dir and install it under its identity."""

    staging = root / f".install-{case}-{len(list(root.iterdir()))}"
    identity = _write_package(staging)
    target = root / "packages" / case / identity
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(target)
    return target, identity


def test_corrupt_existing_package_fails_closed_and_preserves_staging(tmp_path: Path) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    target, identity = _install_package(root, "case-a")
    (target / "final.mp4").write_bytes(b"corrupted retained output")
    staging = root / ".staging-review"
    built_identity = _write_package(staging)
    assert built_identity == identity

    result = _publish_case_package(
        root,
        {
            "case": "case-a",
            "package_identity": identity,
            "staging_dir": str(staging),
            "genuine": True,
        },
    )

    assert result["genuine_acceptance"] is False
    assert result["published_reference"] is None
    assert any(reason.startswith("PACKAGE_FILE_MISMATCH") for reason in result["integrity_failure"])
    # The valid staging evidence is preserved and the existing package untouched.
    assert staging.exists()
    assert (target / "final.mp4").read_bytes() == b"corrupted retained output"


def test_valid_existing_package_is_reused(tmp_path: Path) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    target, identity = _install_package(root, "case-a")
    staging = root / ".staging-review"
    assert _write_package(staging) == identity

    result = _publish_case_package(
        root,
        {
            "case": "case-a",
            "package_identity": identity,
            "staging_dir": str(staging),
            "genuine": True,
        },
    )

    assert result["genuine_acceptance"] is True
    assert result["integrity_failure"] is None
    assert result["directory"] == str(target)
    assert not staging.exists()


def test_malformed_and_unsupported_catalog_fail_closed(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{invalid", encoding="utf-8")
    with pytest.raises(CatalogIntegrityError):
        _read_catalog(catalog)
    assert catalog.read_text(encoding="utf-8") == "{invalid"

    catalog.write_text(json.dumps({"version": 99, "cases": {"x": {}}}), encoding="utf-8")
    with pytest.raises(CatalogIntegrityError):
        _read_catalog(catalog)


def test_legacy_published_without_genuine_is_preserved_as_diagnostic(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "diagnostic-first",
                        "directory": "diagnostic-first",
                        "published": True,
                        "genuine_acceptance": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    migrated = _read_catalog(catalog)
    entry = migrated["cases"]["diagnostic-first"]
    assert entry["published"] is None
    assert entry["diagnostic_attempts"][0]["genuine_acceptance"] is False


def test_legacy_established_published_requires_verified_evidence(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "missing-evidence",
                        "directory": "packages/missing-evidence/does-not-exist",
                        "published": True,
                        "genuine_acceptance": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    migrated = _read_catalog(catalog)
    entry = migrated["cases"]["missing-evidence"]
    assert entry["published"] is None


# --- Audition admission and delivery accounting -----------------------------


class _CountingProvider:
    """Provider that records calls and either returns audio or raises."""

    def __init__(self, *, persist: bool = True, error: Exception | None = None) -> None:
        self.calls: list[object] = []
        self._persist = None
        self._error = error
        self._persist_audio = persist

    def available(self) -> bool:
        return True

    def release(self) -> None:
        return None

    def synthesize(self, request: object) -> NarrationAudio:  # type: ignore[override]
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        text = str(getattr(request, "text", ""))
        chunk_id = str(getattr(request, "chunk_id", "x"))
        if self._persist_audio and callable(self._persist):
            wav = wrap_pcm_as_wav(b"\x10\x20" * 4800, sample_rate=24000, channels=1)
            relative, sha = self._persist(chunk_id, wav)  # type: ignore[misc]
            info = inspect_wav(wav)
            return NarrationAudio(
                chunk_id=chunk_id,
                block_index=0,
                text=text,
                relative_path=relative,
                sha256=sha,
                size_bytes=len(wav),
                duration_seconds=info.duration_seconds,
                sample_rate=24000,
                channels=1,
                container="wav",
                provider_identity={},
                validation={"ok": True},
            )
        return NarrationAudio(
            chunk_id=chunk_id,
            block_index=0,
            text=text,
            relative_path="",
            sha256="",
            size_bytes=0,
            duration_seconds=1.0,
            sample_rate=24000,
            channels=1,
            container="wav",
            provider_identity={},
            validation={"ok": True},
        )


def _read_ledger(storage: StorageService) -> dict:
    path = storage.resolve_relative(Path("benchmarks") / "stage-6" / "tts-audition" / "ledger.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_admission_is_checked_before_each_attempt(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    storage = StorageService(get_settings().storage_root)
    provider = _CountingProvider()
    admissions: list[int] = []

    def admit() -> bool:
        admissions.append(1)
        return len(admissions) <= 2

    run_audition(
        settings=get_settings(),
        storage=storage,
        model="review-fake",
        voice_factory=lambda voice: provider,
        admit=admit,
    )
    ledger = _read_ledger(storage)
    # Nine attempts are considered, but only the two admitted invoke the provider.
    assert len(provider.calls) == 2
    assert len(admissions) == MAX_AUDITION_ATTEMPTS
    assert len(ledger["attempts"]) == 2


def test_rate_limit_notifies_cooldown_and_blocks_later_samples(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    storage = StorageService(get_settings().storage_root)
    provider = _CountingProvider(error=MaterializationProviderError("RATE_LIMITED", "KNOWN_429"))
    state = {"cooldown": False}

    def admit() -> bool:
        return not state["cooldown"]

    def on_rate_limited() -> None:
        state["cooldown"] = True

    run_audition(
        settings=get_settings(),
        storage=storage,
        model="review-fake",
        voice_factory=lambda voice: provider,
        admit=admit,
        on_rate_limited=on_rate_limited,
    )
    # The first attempt is admitted and 429s; the cooldown then denies the rest.
    assert len(provider.calls) == 1
    ledger = _read_ledger(storage)
    assert len(ledger["attempts"]) == 1
    entry = ledger["attempts"][0]
    assert entry["status"] == "FAILED_KNOWN"
    assert entry["category"] == "RATE_LIMITED"


def test_typed_timeout_is_uncertain(configured_test_environment: None, tmp_path: Path) -> None:  # noqa: ARG001
    storage = StorageService(get_settings().storage_root)
    provider = _CountingProvider(error=MaterializationProviderError("TIMEOUT"))

    result = run_audition(
        settings=get_settings(),
        storage=storage,
        model="review-fake",
        voice_factory=lambda voice: provider,
    )
    assert result.raw_calls == MAX_AUDITION_ATTEMPTS
    ledger = _read_ledger(storage)
    assert all(entry["status"] == "UNCERTAIN" for entry in ledger["attempts"])
    assert all(entry["category"] == "TIMEOUT" for entry in ledger["attempts"])
    assert result.samples[0].delivery_status == "UNCERTAIN"


def test_malformed_received_response_is_responded_with_content_failure(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    storage = StorageService(get_settings().storage_root)
    provider = _CountingProvider(error=MaterializationProviderError("MALFORMED_OUTPUT"))

    result = run_audition(
        settings=get_settings(),
        storage=storage,
        model="review-fake",
        voice_factory=lambda voice: provider,
    )
    ledger = _read_ledger(storage)
    assert all(entry["status"] == "RESPONDED" for entry in ledger["attempts"])
    assert all(entry["content_failure"] == "MALFORMED_OUTPUT" for entry in ledger["attempts"])
    assert result.samples[0].delivery_status == "RESPONDED"
    assert result.samples[0].error and result.samples[0].error.startswith("RESPONDED:")


def test_ordinary_replay_makes_zero_additional_provider_calls(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    storage = StorageService(get_settings().storage_root)

    def fidelity(_audio: object) -> dict[str, object]:
        return {"status": "ACCEPTED", "findings": []}

    fidelity.runtime_identity = lambda: {"provider": "f", "policy_version": "v"}  # type: ignore[attr-defined]

    first = _CountingProvider()
    run_audition(
        settings=get_settings(),
        storage=storage,
        model="review-fake",
        voice_factory=lambda voice: first,
        fidelity=fidelity,
    )
    assert len(first.calls) == MAX_AUDITION_ATTEMPTS

    second = _CountingProvider(error=AssertionError("provider must not be called on replay"))
    result = run_audition(
        settings=get_settings(),
        storage=storage,
        model="review-fake",
        voice_factory=lambda voice: second,
        fidelity=fidelity,
    )
    assert result.raw_calls == 0
    assert second.calls == []
    assert all(sample.cached for sample in result.samples)


def test_revalidation_preserves_historical_findings(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    storage = StorageService(get_settings().storage_root)

    def old_fidelity(_audio: object) -> dict[str, object]:
        return {"status": "REJECTED", "findings": ["MEANING_MISMATCH"]}

    old_fidelity.runtime_identity = lambda: {"provider": "f"}  # type: ignore[attr-defined]
    provider = _CountingProvider()
    run_audition(
        settings=get_settings(),
        storage=storage,
        model="review-fake",
        voice_factory=lambda voice: provider,
        fidelity=old_fidelity,
    )

    def new_fidelity(_audio: object) -> dict[str, object]:
        return {"status": "DEFERRED", "findings": ["ASR_TRANSCRIPT_DISAGREEMENT"]}

    new_fidelity.runtime_identity = lambda: {"provider": "f", "policy_version": "v2"}  # type: ignore[attr-defined]
    replay = _CountingProvider(error=AssertionError("no synthesis on revalidation"))
    run_audition(
        settings=get_settings(),
        storage=storage,
        model="review-fake",
        voice_factory=lambda voice: replay,
        fidelity=new_fidelity,
    )
    ledger = _read_ledger(storage)
    entry = ledger["attempts"][0]
    # Historical raw findings are preserved; the revalidation is stored separately.
    assert entry["fidelity"]["status"] == "REJECTED"
    assert entry["revalidated_fidelity"]["status"] == "DEFERRED"
    assert replay.calls == []


def test_crash_left_reservation_is_uncertain_without_a_call(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    storage = StorageService(get_settings().storage_root)
    model = "review-fake"
    identity = audition_identity(model, "v1beta")
    attempts = []
    for voice in AUDITION_VOICES:
        for script in AUDITION_SCRIPTS:
            request_identity = audition_request_identity(voice, script, model)
            attempts.append(
                {
                    "attempt_id": f"{voice}-{script.script_id}",
                    "voice": voice,
                    "script_id": script.script_id,
                    "request_fingerprint": _identity_digest(request_identity),
                    "ordinal": len(attempts) + 1,
                    "status": "RESERVED",
                    "relative_path": "",
                    "sha256": "",
                    "fidelity": {},
                }
            )
    root = storage.resolve_relative(Path("benchmarks") / "stage-6" / "tts-audition")
    root.mkdir(parents=True, exist_ok=True)
    (root / "ledger.json").write_text(
        json.dumps(
            {
                "version": MANIFEST_VERSION,
                "identity": identity,
                "identity_digest": _identity_digest(identity),
                "attempts": attempts,
            }
        ),
        encoding="utf-8",
    )
    provider = _CountingProvider(error=AssertionError("reserved attempts must not retry"))
    result = run_audition(
        settings=get_settings(),
        storage=storage,
        model=model,
        voice_factory=lambda voice: provider,
    )
    assert result.raw_calls == 0
    assert provider.calls == []
    assert all(sample.error and sample.error.startswith("UNCERTAIN") for sample in result.samples)


def test_consumed_budget_floor_blocks_a_missing_ledger(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    storage = StorageService(get_settings().storage_root)
    record_consumed_budget(storage, consumed_attempts=9, probe_calls=1, basis="test")
    provider = _CountingProvider(error=AssertionError("budget exhausted: no calls"))
    result = run_audition(
        settings=get_settings(),
        storage=storage,
        model="review-fake",
        voice_factory=lambda voice: provider,
    )
    assert result.status == AUDITION_BUDGET_EXHAUSTED
    assert result.raw_calls == 0
    assert provider.calls == []


def test_recovery_reports_lost_evidence_and_persists_the_floor(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    storage = StorageService(get_settings().storage_root)
    report = {
        "samples": [
            {
                "voice": "Charon",
                "script_id": "english",
                "relative_path": "benchmarks/stage-6/tts-audition/Charon-english-missing.wav",
                "sha256": "deadbeef",
            },
            {
                "voice": "Kore",
                "script_id": "egyptian-conversational",
                "relative_path": "",
                "sha256": "",
            },
        ]
    }
    result = record_recovery(storage, report, probe_calls=1)
    recovery = result["recovery"]
    assert recovery["consumed_attempts"] == 2
    assert recovery["probe_calls"] == 1
    reasons = {entry["reason"] for entry in recovery["lost"]}
    assert reasons == {"MISSING_ARTIFACT", "NO_ARTIFACT_IN_CAPTURE"}
    floor = result["consumed_budget"]
    assert floor["consumed_attempts"] == 2
    assert floor["probe_calls"] == 1
    assert (
        storage.resolve_relative(
            Path("benchmarks") / "stage-6" / "tts-audition" / "consumed-budget.json"
        )
    ).is_file()
