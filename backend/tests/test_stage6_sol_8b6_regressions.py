"""Stage 6 SOL 8b6 regressions: asset bytes, case-sensitive atoms, prefix proof.

These convert the SOL diagnostics at ``/tmp/test_sol_8b6_review.py`` (which
asserted the observed defects) into regressions asserting the corrected behavior:

1. a package's required exported assets must resolve to actual verified files;
2. protected technical/numeric/URL atoms are compared verbatim (case-sensitive);
3. the consumed-budget proof is a versioned history prefix stable across a
   legitimate append / local fidelity update.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.core.settings import get_settings
from app.materialization.acceptance_export import _package_integrity_reasons
from app.materialization.narration import transcription_matches
from app.materialization.providers import MaterializationProviderError
from app.materialization.tts_audition import AUDITION_SCRIPTS, AUDITION_VOICES
from app.services.storage import StorageService


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _write_manifest(base: Path, *, omit: tuple[str, ...] = ()) -> str:
    lines = []
    for entry in sorted(base.rglob("*")):
        if entry.is_file() and entry.name != "SHA256SUMS":
            relative = entry.relative_to(base).as_posix()
            if relative in omit:
                continue
            lines.append(f"{_file_sha(entry)}  {relative}")
    (base / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _file_sha(base / "SHA256SUMS")


def _asset_package(
    base: Path,
    *,
    exported: str = "authored.ass",
    written_bytes: bytes = b"ASS body",
    record_expected: str | None = None,
    declared_expected: str | None = None,
    copied: bool = True,
    write_file: bool = True,
    symlink: bool = False,
    extra_records: list[dict[str, object]] | None = None,
    omit_from_manifest: tuple[str, ...] = (),
    include_assets_key: bool = True,
) -> tuple[Path, str, str]:
    base.mkdir(parents=True, exist_ok=True)
    artifact = base / "final.mp4"
    artifact.write_bytes(b"media")
    actual_ass = _sha(written_bytes)
    expected = record_expected if record_expected is not None else actual_ass
    declared = declared_expected if declared_expected is not None else actual_ass
    record = {
        "kind": "authored_ass",
        "relative_path": "sources/review/authored.ass",
        "expected_sha256": expected,
        "actual_sha256": actual_ass,
        "present": True,
        "verified": expected == actual_ass,
        "copied": copied,
        "destination_sha256": actual_ass,
        "destination_verified": True,
        "exported_relative_path": exported,
    }
    records = [record, *(extra_records or [])]
    index: dict[str, object] = {
        "genuine_acceptance": True,
        "artifact_file": "final.mp4",
        "render_artifact_sha256": _file_sha(artifact),
        "authored_ass_expected_sha256": declared,
        "authored_ass_sha256": actual_ass,
        "identity": {
            "assets": [[r["kind"], r["relative_path"], r["expected_sha256"]] for r in records]
        },
        "readiness": {
            "assets": [
                {
                    "kind": r["kind"],
                    "relative_path": r["relative_path"],
                    "expected_sha256": r["expected_sha256"],
                }
                for r in records
            ]
        },
        "narration_references": [
            {"relative_path": r["relative_path"], "sha256": r["expected_sha256"]}
            for r in records
            if r["kind"] == "narration"
        ],
        "asset_references": [
            {"relative_path": r["relative_path"], "sha256": r["expected_sha256"]}
            for r in records
            if r["kind"] == "supporting"
        ],
    }
    if include_assets_key:
        index["assets"] = records
    (base / "index.json").write_text(json.dumps(index), encoding="utf-8")
    exported_path = base / exported
    safe = exported and not exported.startswith("..") and not Path(exported).is_absolute()
    if write_file and safe:
        exported_path.parent.mkdir(parents=True, exist_ok=True)
        if symlink:
            target = base.parent / "outside-ass.ass"
            target.write_bytes(written_bytes)
            exported_path.symlink_to(target)
        else:
            exported_path.write_bytes(written_bytes)
    identity = _write_manifest(base, omit=omit_from_manifest)
    return base, identity, actual_ass


# --- 1. Required exported assets must resolve to verified bytes --------------


def test_required_ass_bytes_cannot_be_replaced_by_index_claims(tmp_path: Path) -> None:
    # The SOL probe: matching authored_ass claims but no actual ASS file.
    package = tmp_path / "package"
    package.mkdir()
    (package / "final.mp4").write_bytes(b"review fixture media")
    absent_sha = _sha(b"missing required ASS")
    index = {
        "genuine_acceptance": True,
        "artifact_file": "final.mp4",
        "render_artifact_sha256": _file_sha(package / "final.mp4"),
        "authored_ass_expected_sha256": absent_sha,
        "authored_ass_sha256": absent_sha,
        "assets": [
            {
                "kind": "authored_ass",
                "relative_path": "sources/review/authored.ass",
                "expected_sha256": absent_sha,
                "copied": False,
                "destination_verified": False,
            }
        ],
    }
    (package / "index.json").write_text(json.dumps(index), encoding="utf-8")
    identity = _write_manifest(package)
    reasons = _package_integrity_reasons(package, identity)
    assert reasons != []
    assert not (package / "authored.ass").exists()


def test_valid_exported_asset_package_is_genuine(tmp_path: Path) -> None:
    package, identity, _ = _asset_package(tmp_path / "ok")
    assert _package_integrity_reasons(package, identity) == []


def test_exported_asset_hash_must_match_expected(tmp_path: Path) -> None:
    package, identity, actual = _asset_package(
        tmp_path / "bashash", record_expected=_sha(b"different expected bytes")
    )
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_HASH_MISMATCH") for reason in reasons)


def test_exported_asset_must_be_listed_in_manifest(tmp_path: Path) -> None:
    package, identity, actual = _asset_package(
        tmp_path / "unlisted", omit_from_manifest=("authored.ass",)
    )
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_UNLISTED") for reason in reasons)


def test_exported_asset_symlink_is_rejected(tmp_path: Path) -> None:
    package, identity, _ = _asset_package(tmp_path / "link", symlink=True)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_SYMLINK") for reason in reasons)


def test_escaping_exported_asset_path_is_rejected(tmp_path: Path) -> None:
    package, identity, _ = _asset_package(
        tmp_path / "escape", exported="../outside.ass", write_file=False
    )
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_PATH_ESCAPES") for reason in reasons)


def test_asset_destination_collision_is_rejected(tmp_path: Path) -> None:
    collision = {
        "kind": "narration",
        "relative_path": "sources/review/narration.wav",
        "expected_sha256": _sha(b"narration"),
        "copied": True,
        "destination_sha256": _sha(b"narration"),
        "destination_verified": True,
        "exported_relative_path": "authored.ass",
    }
    package, identity, _ = _asset_package(
        tmp_path / "collision",
        written_bytes=b"narration",
        extra_records=[collision],
    )
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_PATH_COLLISION") for reason in reasons)


def test_missing_asset_metadata_cannot_imply_empty_requirement(tmp_path: Path) -> None:
    package, identity, _ = _asset_package(
        tmp_path / "nomap", include_assets_key=False
    )
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_METADATA_MISSING") for reason in reasons)


def test_copied_false_asset_without_a_file_is_rejected(tmp_path: Path) -> None:
    package, identity, _ = _asset_package(
        tmp_path / "notcopied", copied=False, write_file=False
    )
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_FILE_MISSING") for reason in reasons)


def test_declared_authored_ass_must_match_record(tmp_path: Path) -> None:
    package, identity, _ = _asset_package(
        tmp_path / "assmatch", declared_expected=_sha(b"authoritative expected")
    )
    reasons = _package_integrity_reasons(package, identity)
    assert any(
        reason.startswith("PACKAGE_AUTHORED_ASS_EXPECTED_MISMATCH") for reason in reasons
    )


# --- 2. Case-sensitive protected atoms --------------------------------------


def test_case_sensitive_url_is_not_rewritten() -> None:
    expected = "Open https://example.test/Account?token=AbC now"
    actual = "Open https://example.test/account?token=abc now"
    assert not transcription_matches(expected, actual)


def test_technical_form_case_is_preserved() -> None:
    assert not transcription_matches("Use C++ now", "Use c++ now")
    assert not transcription_matches("See foo/Bar now", "See foo/bar now")


def test_documented_normalizations_still_hold() -> None:
    assert transcription_matches("I don't know", "I do not know")
    assert transcription_matches("We can start", "We um can start")


def test_fidelity_policy_is_v4() -> None:
    from app.materialization.tts_gemini import FIDELITY_POLICY_VERSION

    assert FIDELITY_POLICY_VERSION == "stage6-fidelity-v4"


# --- 3. Versioned history-prefix proof --------------------------------------


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


def _audition_kwargs(storage: StorageService) -> dict[str, object]:
    calls: list[object] = []
    return {
        "settings": get_settings(),
        "storage": storage,
        "model": "review-fake",
        "voice_factory": lambda voice: _TimeoutProvider(calls),
        "_calls": calls,
    }


def _read_ledger(storage: StorageService) -> dict:
    path = storage.resolve_relative(
        Path("benchmarks") / "stage-6" / "tts-audition" / "ledger.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _read_floor(storage: StorageService) -> dict:
    path = storage.resolve_relative(
        Path("benchmarks") / "stage-6" / "tts-audition" / "consumed-budget.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _once_admit(state: dict[str, int]) -> "callable":
    def admit() -> bool:
        state["checks"] = state.get("checks", 0) + 1
        return state["checks"] == 1

    return admit


def test_linked_history_prefix_survives_a_legitimate_append(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    from app.materialization.tts_audition import (
        AUDITION_OK,
        _history_prefix_proof,
        record_consumed_budget,
        run_audition,
    )

    storage = StorageService(get_settings().storage_root)
    kwargs = _audition_kwargs(storage)
    calls = kwargs.pop("_calls")

    first = run_audition(**kwargs, admit=_once_admit({}))
    assert first.raw_calls == 1
    manifest = _read_ledger(storage)
    proof = _history_prefix_proof(manifest, 1)
    assert proof is not None and proof["count"] == 1
    record_consumed_budget(
        storage, consumed_attempts=1, probe_calls=0, basis="linked prefix", history_proof=proof
    )

    second = run_audition(**kwargs, admit=_once_admit({}))
    assert second.raw_calls == 1
    third = run_audition(**kwargs, admit=_once_admit({}))

    # The append did not invalidate the prefix: the third run continues with the
    # next genuinely unspent request instead of being falsely blocked.
    assert third.status == AUDITION_OK
    assert third.raw_calls == 1
    assert len(_read_ledger(storage)["attempts"]) == 3
    assert len(calls) == 3


def test_legacy_full_manifest_digest_is_upgraded_to_a_prefix_proof(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    from app.materialization.tts_audition import (
        AUDITION_OK,
        HISTORY_PROOF_VERSION,
        _manifest_digest,
        record_consumed_budget,
        run_audition,
    )

    storage = StorageService(get_settings().storage_root)
    kwargs = _audition_kwargs(storage)
    kwargs.pop("_calls")

    run_audition(**kwargs, admit=_once_admit({}))
    manifest = _read_ledger(storage)
    record_consumed_budget(
        storage,
        consumed_attempts=1,
        probe_calls=0,
        basis="legacy",
        manifest_digest=_manifest_digest(manifest),
    )
    floor = _read_floor(storage)
    assert floor["history_proof"]["version"] == HISTORY_PROOF_VERSION
    assert "manifest_digest" not in floor

    block_second = run_audition(
        **kwargs, admit=_once_admit({})
    )
    assert block_second.status == AUDITION_OK
    assert block_second.raw_calls == 1


def test_linked_prefix_survives_a_local_fidelity_update(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    from app.materialization.tts_audition import (
        AUDITION_OK,
        _history_prefix_proof,
        record_consumed_budget,
        run_audition,
    )
    from app.materialization.types import NarrationAudio

    storage = StorageService(get_settings().storage_root)
    calls: list[object] = []

    class _Provider:
        def available(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def synthesize(self, request: object) -> NarrationAudio:
            calls.append(request)
            return NarrationAudio(
                chunk_id="c",
                block_index=0,
                text="t",
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

    def fidelity(policy: str) -> "callable":
        def validate(_audio: object) -> dict[str, object]:
            return {"status": "ACCEPTED", "findings": []}

        validate.runtime_identity = lambda: {"provider": "f", "policy_version": policy}  # type: ignore[attr-defined]
        return validate

    base = {"settings": get_settings(), "storage": storage, "model": "review-fake",
            "voice_factory": lambda voice: _Provider()}
    first = run_audition(**base, fidelity=fidelity("v1"))
    assert first.raw_calls == 9
    manifest = _read_ledger(storage)
    proof = _history_prefix_proof(manifest, 9)
    record_consumed_budget(
        storage, consumed_attempts=9, probe_calls=0, basis="linked prefix", history_proof=proof
    )

    # A changed local-validator identity reruns only the local validation (no TTS)
    # and mutates the manifest, but the immutable prefix proof stays valid.
    second = run_audition(**base, fidelity=fidelity("v2"))
    assert second.status == AUDITION_OK
    assert second.raw_calls == 0
    third = run_audition(**base, fidelity=fidelity("v2"))
    assert third.status == AUDITION_OK
    assert third.raw_calls == 0
    assert len(calls) == 9


def test_crash_left_reservation_with_a_valid_prefix_is_not_recalled(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    from app.materialization.tts_audition import (
        MANIFEST_VERSION,
        _history_prefix_proof,
        _identity_digest,
        audition_identity,
        audition_request_identity,
        record_consumed_budget,
        run_audition,
    )

    storage = StorageService(get_settings().storage_root)
    model = "review-fake"
    identity = audition_identity(model, "v1beta")
    attempts = []
    for voice in AUDITION_VOICES:
        for script in AUDITION_SCRIPTS:
            attempts.append(
                {
                    "attempt_id": f"{voice}-{script.script_id}",
                    "voice": voice,
                    "script_id": script.script_id,
                    "request_fingerprint": _identity_digest(
                        audition_request_identity(voice, script, model)
                    ),
                    "ordinal": len(attempts) + 1,
                    "status": "RESERVED",
                    "relative_path": "",
                    "sha256": "",
                    "fidelity": {},
                }
            )
    root = storage.resolve_relative(Path("benchmarks") / "stage-6" / "tts-audition")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": MANIFEST_VERSION,
        "identity": identity,
        "identity_digest": _identity_digest(identity),
        "attempts": attempts,
    }
    (root / "ledger.json").write_text(json.dumps(manifest), encoding="utf-8")
    proof = _history_prefix_proof(manifest, 9)
    assert proof is not None
    record_consumed_budget(
        storage, consumed_attempts=9, probe_calls=0, basis="linked prefix", history_proof=proof
    )
    calls: list[object] = []
    result = run_audition(
        settings=get_settings(),
        storage=storage,
        model=model,
        voice_factory=lambda voice: _TimeoutProvider(calls),
    )
    assert result.raw_calls == 0
    assert calls == []
    assert all(sample.error and sample.error.startswith("UNCERTAIN") for sample in result.samples)


def test_legacy_proof_that_no_longer_validates_blocks(
    configured_test_environment: None, tmp_path: Path
) -> None:  # noqa: ARG001
    from app.materialization.tts_audition import (
        AUDITION_HISTORY_INCOMPLETE,
        _manifest_digest,
        record_consumed_budget,
        run_audition,
    )

    storage = StorageService(get_settings().storage_root)
    kwargs = _audition_kwargs(storage)
    kwargs.pop("_calls")
    run_audition(**kwargs, admit=_once_admit({}))
    stale_digest = _manifest_digest(_read_ledger(storage))
    # Append another attempt, then record a floor whose digest was taken BEFORE the
    # append: the retained evidence no longer matches, so it must block.
    run_audition(**kwargs, admit=_once_admit({}))
    record_consumed_budget(
        storage, consumed_attempts=1, probe_calls=0, basis="stale", manifest_digest=stale_digest
    )
    assert "history_proof" not in _read_floor(storage)
    blocked = run_audition(**kwargs, admit=_once_admit({}))
    assert blocked.status == AUDITION_HISTORY_INCOMPLETE
    assert blocked.raw_calls == 0
