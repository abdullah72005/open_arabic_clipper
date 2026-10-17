"""Stage 6 SOL p451 regressions: budget sealing, package media, fidelity v3.

These convert the SOL diagnostic probes at /tmp/test_sol_451_review.py into
regressions asserting the corrected behavior.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.core.settings import get_settings
from app.materialization.acceptance_export import (
    _package_integrity_reasons,
    _read_catalog,
)
from app.materialization.narration import compare_transcripts, transcription_matches
from app.materialization.providers import MaterializationProviderError
from app.materialization.tts_audition import (
    AUDITION_BUDGET_EXHAUSTED,
    AUDITION_HISTORY_INCOMPLETE,
    AuditionError,
    record_consumed_budget,
    run_audition,
)
from app.materialization.tts_gemini import validate_narration_fidelity
from app.services.storage import StorageService


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 1. Seal incomplete-history budget recovery -----------------------------


class _TimeoutProvider:
    def __init__(self, calls: list[object]) -> None:
        self._calls = calls

    def available(self) -> bool:
        return True

    def release(self) -> None:
        return None

    def synthesize(self, request: object) -> object:
        self._calls.append(request)
        raise MaterializationProviderError("TIMEOUT")


def test_partial_lost_history_blocks_synthesis_across_restart(
    tmp_path: Path,
) -> None:
    storage = StorageService(tmp_path / "storage")
    settings = get_settings()
    calls: list[object] = []
    record_consumed_budget(storage, consumed_attempts=4, probe_calls=0, basis="lost history")

    first = run_audition(
        settings=settings, storage=storage, model="review-fake",
        voice_factory=lambda voice: _TimeoutProvider(calls),
    )
    second = run_audition(
        settings=settings, storage=storage, model="review-fake",
        voice_factory=lambda voice: _TimeoutProvider(calls),
    )

    # A positive floor without complete per-request history blocks new synthesis
    # entirely: no empty ledger is recreated and treated as complete.
    assert first.status == AUDITION_HISTORY_INCOMPLETE
    assert second.status == AUDITION_HISTORY_INCOMPLETE
    assert first.raw_calls == 0
    assert second.raw_calls == 0
    assert calls == []


def test_fully_consumed_floor_reports_exhausted(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    record_consumed_budget(storage, consumed_attempts=9, probe_calls=1, basis="recovery")
    result = run_audition(
        settings=get_settings(), storage=storage, model="review-fake",
        voice_factory=lambda voice: _TimeoutProvider([]),
    )
    assert result.status == AUDITION_BUDGET_EXHAUSTED
    assert result.raw_calls == 0


def test_malformed_floor_fails_closed_and_preserves_bytes(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    record_consumed_budget(storage, consumed_attempts=4, probe_calls=0, basis="initial")
    path = storage.resolve_relative(
        Path("benchmarks") / "stage-6" / "tts-audition" / "consumed-budget.json"
    )
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(AuditionError):
        record_consumed_budget(storage, consumed_attempts=0, probe_calls=0, basis="bad")

    assert path.read_text(encoding="utf-8") == "{broken"
    # The malformed floor also fails run_audition closed rather than resetting.
    result = run_audition(
        settings=get_settings(), storage=storage, model="review-fake",
        voice_factory=lambda voice: _TimeoutProvider([]),
    )
    assert result.status == "AUDITION_HISTORY_INVALID"


def test_concurrent_budget_updates_remain_monotonic(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    counts = [5, 3, 7, 1, 9, 2, 8, 6]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda value: record_consumed_budget(
                    storage, consumed_attempts=value, probe_calls=value, basis=f"t{value}"
                ),
                counts,
            )
        )
    path = storage.resolve_relative(
        Path("benchmarks") / "stage-6" / "tts-audition" / "consumed-budget.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["consumed_attempts"] == max(counts)
    assert record["probe_calls"] == max(counts)


# --- 2. Require actual media before legacy acceptance publication -----------


def _valid_package(base: Path) -> str:
    base.mkdir(parents=True, exist_ok=True)
    artifact = base / "final.mp4"
    artifact.write_bytes(b"real-media-bytes")
    (base / "index.json").write_text(
        json.dumps(
            {
                "genuine_acceptance": True,
                "artifact_file": "final.mp4",
                "render_artifact_sha256": _sha(artifact),
                "authored_ass_expected_sha256": "",
                "authored_ass_sha256": None,
                "assets": [],
            }
        ),
        encoding="utf-8",
    )
    lines = [
        f"{_sha(entry)}  {entry.relative_to(base).as_posix()}"
        for entry in sorted(base.rglob("*"))
        if entry.is_file() and entry.name != "SHA256SUMS"
    ]
    (base / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha(base / "SHA256SUMS")


def _write_manifest(base: Path) -> None:
    lines = [
        f"{_sha(entry)}  {entry.relative_to(base).as_posix()}"
        for entry in sorted(base.rglob("*"))
        if entry.is_file() and entry.name != "SHA256SUMS"
    ]
    (base / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_index_only_package_is_not_genuine(tmp_path: Path) -> None:
    package = tmp_path / "index-only"
    package.mkdir()
    index = package / "index.json"
    index.write_text(json.dumps({"genuine_acceptance": True}), encoding="utf-8")
    manifest = package / "SHA256SUMS"
    manifest.write_text(_sha(index) + "  index.json\n", encoding="utf-8")
    identity = _sha(manifest)
    reasons = _package_integrity_reasons(package, identity)
    assert "PACKAGE_ARTIFACT_MISSING" in reasons
    assert not (package / "final.mp4").exists()


def test_missing_artifact_metadata_is_not_genuine(tmp_path: Path) -> None:
    package = tmp_path / "no-metadata"
    package.mkdir()
    (package / "final.mp4").write_bytes(b"real")
    (package / "index.json").write_text(
        json.dumps({"genuine_acceptance": True, "artifact_file": "final.mp4"}), encoding="utf-8"
    )
    _write_manifest(package)
    reasons = _package_integrity_reasons(package, _sha(package / "SHA256SUMS"))
    assert "PACKAGE_ARTIFACT_SHA_MISSING" in reasons


def test_absent_media_is_not_genuine(tmp_path: Path) -> None:
    package = tmp_path / "absent-media"
    package.mkdir()
    (package / "index.json").write_text(
        json.dumps(
            {
                "genuine_acceptance": True,
                "artifact_file": "final.mp4",
                "render_artifact_sha256": "a" * 64,
                "assets": [],
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(package)
    reasons = _package_integrity_reasons(package, _sha(package / "SHA256SUMS"))
    assert any(reason.startswith("PACKAGE_ARTIFACT_FILE_MISSING") for reason in reasons)


def test_mismatched_artifact_identity_is_not_genuine(tmp_path: Path) -> None:
    package = tmp_path / "mismatch"
    package.mkdir()
    (package / "final.mp4").write_bytes(b"real")
    (package / "index.json").write_text(
        json.dumps(
            {
                "genuine_acceptance": True,
                "artifact_file": "final.mp4",
                "render_artifact_sha256": "0" * 64,
                "assets": [],
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(package)
    reasons = _package_integrity_reasons(package, _sha(package / "SHA256SUMS"))
    assert any(reason.startswith("PACKAGE_ARTIFACT_SHA_MISMATCH") for reason in reasons)


def test_manifest_path_escape_is_not_genuine(tmp_path: Path) -> None:
    package = tmp_path / "escape"
    package.mkdir()
    (package / "index.json").write_text(
        json.dumps(
            {
                "genuine_acceptance": True,
                "artifact_file": "final.mp4",
                "render_artifact_sha256": "a" * 64,
                "assets": [],
            }
        ),
        encoding="utf-8",
    )
    (package / "SHA256SUMS").write_text(
        _sha(package / "index.json") + "  index.json\n"
        + hashlib.sha256(b"x").hexdigest() + "  ../outside.mp4\n",
        encoding="utf-8",
    )
    reasons = _package_integrity_reasons(package, _sha(package / "SHA256SUMS"))
    assert any(reason.startswith("PACKAGE_PATH_ESCAPES") for reason in reasons)


def test_legacy_index_only_package_is_not_promoted(tmp_path: Path) -> None:
    package = tmp_path / "empty"
    package.mkdir()
    index = package / "index.json"
    index.write_text(json.dumps({"genuine_acceptance": True}), encoding="utf-8")
    manifest = package / "SHA256SUMS"
    manifest.write_text(_sha(index) + "  index.json\n", encoding="utf-8")
    identity = _sha(manifest)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "empty",
                        "directory": "empty",
                        "package_identity": identity,
                        "published": True,
                        "genuine_acceptance": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    migrated = _read_catalog(catalog)
    entry = migrated["cases"]["empty"]
    assert entry["published"] is None
    assert entry["diagnostic_attempts"][0]["genuine_acceptance"] is False


def test_legacy_valid_package_is_promoted(tmp_path: Path) -> None:
    package = tmp_path / "valid"
    identity = _valid_package(package)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "v",
                        "directory": "valid",
                        "package_identity": identity,
                        "published": True,
                        "genuine_acceptance": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    migrated = _read_catalog(catalog)
    published = migrated["cases"]["v"]["published"]
    assert published is not None and published["genuine_acceptance"] is True


# --- 3. Restore documented normalization without weakening fidelity ---------


def test_documented_speech_normalizations_are_equivalent() -> None:
    assert transcription_matches("I don't know", "I do not know")
    assert transcription_matches("We can start", "We um can start")


def test_technical_changes_still_are_refused() -> None:
    assert not transcription_matches("Use C++ now", "Use C now")
    assert not transcription_matches("Use C# now", "Use C now")
    assert not transcription_matches("Pay 1,000 now", "Pay 1000 now")
    assert not transcription_matches("visit https://x.com/a?b=1", "visit https://x.com/a?b=2")


class _FakeASREngine:
    def __init__(self, transcript: str) -> None:
        self._transcript = transcript

    def transcribe(self, path: object, options: object, cancel_event: object = None) -> object:
        return type("_Result", (), {"raw_text": self._transcript, "language": "ar"})()


class _Opts:
    model = "small"
    device = "cpu"
    compute_type = "int8"
    beam_size = 5


def test_arabic_spelling_disagreement_remains_deferred() -> None:
    engine = _FakeASREngine("الموضوع مش بالبساطة اللي الناس فكراها")
    result = validate_narration_fidelity(
        Path("not-decoded.wav"),
        expected_text="الموضوع مش بالبساطة اللي الناس فاكراها",
        engine=engine,
        options=_Opts(),
    )
    assert result.status == "DEFERRED"
    assert result.as_dict()["policy_version"] == "stage6-fidelity-v4"
    assert not compare_transcripts(
        "الموضوع مش بالبساطة اللي الناس فاكراها",
        "الموضوع مش بالبساطة اللي الناس فكراها",
    ).matches
