"""Stage 6 SOL b141 regressions: complete, self-contained export verification.

These convert the SOL diagnostics at ``/tmp/test_sol_b141_review.py`` (which
asserted the observed defects) into regressions:

* defect A — a required asset could be omitted from ``index.assets``/the package
  while still declared by the bound inventory/references, and the package was
  accepted and published;
* defect B — the transformed MP4 (and any asset) could be an external symlink with
  matching bytes and was accepted.

The corrected verifier derives the complete required-asset inventory from the
package's bound evidence (``identity.assets`` / ``readiness.assets`` / the bound
narration/supporting/ASS references) and requires an exact, byte-verified mapping
from every required identity to its exported copy, using one shared package-file
validation primitive for every package path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine

from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.acceptance_export import (
    _package_integrity_reasons,
    _publish_case_package,
    _read_catalog,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _write_manifest(base: Path) -> str:
    lines = []
    for entry in sorted(base.rglob("*")):
        if entry.is_file() and entry.name != "SHA256SUMS":
            relative = entry.relative_to(base).as_posix()
            lines.append(f"{_file_sha(entry)}  {relative}")
    (base / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _file_sha(base / "SHA256SUMS")


def _write_index(base: Path, index: dict) -> None:
    (base / "index.json").write_text(json.dumps(index), encoding="utf-8")


def _record(
    kind: str,
    source: str,
    expected: str,
    exported: str | None,
    *,
    destination: str | None = None,
) -> dict:
    return {
        "kind": kind,
        "relative_path": source,
        "expected_sha256": expected,
        "actual_sha256": expected,
        "present": True,
        "verified": True,
        "copied": exported is not None,
        "destination_sha256": destination if destination is not None else expected,
        "destination_verified": True,
        "exported_relative_path": exported,
    }


_ASS = b"authored-ass-bytes"
_NARR = b"narration-wav-bytes"
_SUPPORT = b"supporting-asset-bytes"


def _narrated_package(base: Path) -> tuple[Path, str, dict, dict[str, str]]:
    """Build a package mirroring the real persisted narrated export structure.

    The bound inventory (``identity.assets`` + ``readiness.assets`` + the narration
    reference) declares an authored ASS, one narration, and one supporting asset;
    ``index.assets`` carries their exported-copy evidence.
    """

    base.mkdir(parents=True, exist_ok=True)
    (base / "final.mp4").write_bytes(b"transformed-media-bytes")
    (base / "authored.ass").write_bytes(_ASS)
    (base / "assets").mkdir(exist_ok=True)
    (base / "assets" / "narration-narr.wav").write_bytes(_NARR)
    (base / "assets" / "supporting-logo.png").write_bytes(_SUPPORT)
    ass_sha, narr_sha, support_sha = _sha(_ASS), _sha(_NARR), _sha(_SUPPORT)
    ass_id = ["authored_ass", "sources/review/authored.ass", ass_sha]
    narr_id = ["narration", "sources/review/narr.wav", narr_sha]
    support_id = ["supporting", "sources/review/logo.png", support_sha]
    exported = {
        "authored_ass": "authored.ass",
        "narration": "assets/narration-narr.wav",
        "supporting": "assets/supporting-logo.png",
    }
    index: dict[str, object] = {
        "genuine_acceptance": True,
        "artifact_file": "final.mp4",
        "render_artifact_sha256": _file_sha(base / "final.mp4"),
        "authored_ass_expected_sha256": ass_sha,
        "authored_ass_sha256": ass_sha,
        "narration_references": [{"relative_path": narr_id[1], "sha256": narr_sha}],
        "asset_references": [{"relative_path": support_id[1], "sha256": support_sha}],
        "identity": {"assets": [ass_id, narr_id, support_id]},
        "readiness": {
            "assets": [
                {"kind": "authored_ass", "relative_path": ass_id[1], "expected_sha256": ass_sha},
                {"kind": "narration", "relative_path": narr_id[1], "expected_sha256": narr_sha},
                {
                    "kind": "supporting",
                    "relative_path": support_id[1],
                    "expected_sha256": support_sha,
                },
            ]
        },
        "assets": [
            _record("authored_ass", ass_id[1], ass_sha, exported["authored_ass"]),
            _record("narration", narr_id[1], narr_sha, exported["narration"]),
            _record("supporting", support_id[1], support_sha, exported["supporting"]),
        ],
    }
    _write_index(base, index)
    identity = _write_manifest(base)
    return base, identity, index, exported


def _rewrite(base: Path, index: dict) -> str:
    _write_index(base, index)
    return _write_manifest(base)


# --- Positive case ----------------------------------------------------------


def test_complete_narrated_package_is_genuine(tmp_path: Path) -> None:
    package, identity, _index, _exported = _narrated_package(tmp_path / "complete")
    assert _package_integrity_reasons(package, identity) == []


# --- Defect A: omission of a required mapping/file --------------------------


def test_omitted_narration_record_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, exported = _narrated_package(tmp_path / "omit-record")
    index["assets"] = [r for r in index["assets"] if r["kind"] != "narration"]
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_RECORD_MISSING:narration") for reason in reasons)
    assert (package / exported["narration"]).is_file()  # the file still exists; record removed


def test_omitted_narration_file_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, exported = _narrated_package(tmp_path / "omit-file")
    (package / exported["narration"]).unlink()
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_FILE_MISSING") for reason in reasons)


def test_omitted_supporting_asset_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, exported = _narrated_package(tmp_path / "omit-support")
    index["assets"] = [r for r in index["assets"] if r["kind"] != "supporting"]
    (package / exported["supporting"]).unlink()
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_RECORD_MISSING:supporting") for reason in reasons)


def test_removing_a_copy_record_never_removes_the_requirement(tmp_path: Path) -> None:
    # Identity/reference declaration is the authority; deleting every exported
    # record must not shrink the requirement set to empty.
    package, _identity, index, _exported = _narrated_package(tmp_path / "remove-all")
    index["assets"] = []
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_RECORD_MISSING") for reason in reasons)


def test_inventory_conflict_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, _exported = _narrated_package(tmp_path / "conflict")
    index["readiness"]["assets"] = index["readiness"]["assets"][:1]
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert "PACKAGE_INVENTORY_CONFLICT" in reasons


def test_reference_not_in_inventory_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, _exported = _narrated_package(tmp_path / "ref-mismatch")
    index["narration_references"].append(
        {"relative_path": "sources/review/unlisted.wav", "sha256": _sha(b"unlisted")}
    )
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert "PACKAGE_INVENTORY_REFERENCE_MISMATCH" in reasons


def test_duplicate_record_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, _exported = _narrated_package(tmp_path / "duplicate")
    index["assets"].append(dict(index["assets"][1], exported_relative_path="assets/dup.wav"))
    (package / "assets" / "dup.wav").write_bytes(_NARR)
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_RECORD_DUPLICATE") for reason in reasons)


def test_unexpected_record_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, _exported = _narrated_package(tmp_path / "unexpected")
    index["assets"].append(
        _record("supporting", "sources/review/extra.png", _sha(b"extra"), "assets/extra.png")
    )
    (package / "assets" / "extra.png").write_bytes(b"extra")
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_RECORD_UNEXPECTED") for reason in reasons)


# --- Defect B: external symlinks and escaping paths -------------------------


def test_symlinked_artifact_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, _exported = _narrated_package(tmp_path / "linked-artifact")
    artifact = package / "final.mp4"
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(outside)
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ARTIFACT_SYMLINK") for reason in reasons)
    assert any(reason.startswith("PACKAGE_FILE_SYMLINK") for reason in reasons)


def test_symlinked_asset_file_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, exported = _narrated_package(tmp_path / "linked-asset")
    target = package / exported["narration"]
    outside = tmp_path / "outside-narr.wav"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_SYMLINK") for reason in reasons)


def test_symlinked_parent_directory_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, _exported = _narrated_package(tmp_path / "linked-parent")
    # Move the exported asset out and replace the `assets` directory with a symlink.
    outside_dir = tmp_path / "outside-assets"
    (package / "assets").rename(outside_dir)
    (package / "assets").symlink_to(outside_dir, target_is_directory=True)
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_SYMLINK") for reason in reasons)


def test_escaping_exported_path_is_rejected(tmp_path: Path) -> None:
    package, _identity, index, _exported = _narrated_package(tmp_path / "escape")
    index["assets"][1]["exported_relative_path"] = "../outside-narr.wav"
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_PATH_ESCAPES") for reason in reasons)


# --- Publication / reuse / legacy migration ---------------------------------


def _publish_result(root: Path, package: Path, identity: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".staging-{len(list(root.iterdir()))}"
    staging.mkdir(parents=True)
    return _publish_case_package(
        root,
        {
            "case": "case-a",
            "package_identity": identity,
            "staging_dir": str(staging),
            "genuine": True,
        },
    )


def test_publication_rejects_omitted_narration(tmp_path: Path) -> None:
    root = tmp_path / "published"
    root.mkdir()
    package, _identity, index, exported = _narrated_package(root / ".staging-build")
    index["assets"] = [r for r in index["assets"] if r["kind"] != "narration"]
    (package / exported["narration"]).unlink()
    identity = _rewrite(package, index)
    result = _publish_result(root, package, identity)
    assert result["published"] is False
    assert result["genuine_acceptance"] is False
    assert result["integrity_failure"]


def test_reuse_rejects_corrupt_published_package(tmp_path: Path) -> None:
    root = tmp_path / "reuse"
    (root / "packages" / "case-a").mkdir(parents=True)
    package, identity, _index, _exported = _narrated_package(tmp_path / "reuse-src")
    target = root / "packages" / "case-a" / identity
    package.rename(target)
    # Corrupt the retained package's narration bytes after publication.
    (target / "assets" / "narration-narr.wav").write_bytes(b"tampered")
    result = _publish_result(root, target, identity)
    assert result["published"] is False
    assert result["integrity_failure"]


def test_legacy_migration_rejects_partial_package(tmp_path: Path) -> None:
    package, _identity, index, exported = _narrated_package(tmp_path / "legacy")
    index["assets"] = [r for r in index["assets"] if r["kind"] != "narration"]
    (package / exported["narration"]).unlink()
    identity = _rewrite(package, index)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "legacy",
                        "directory": "legacy",
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
    entry = migrated["cases"]["legacy"]
    assert entry["published"] is None
    assert entry["diagnostic_attempts"][0]["genuine_acceptance"] is False


def test_legacy_migration_promotes_complete_package(tmp_path: Path) -> None:
    package, identity, _index, _exported = _narrated_package(tmp_path / "legacy-ok")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "legacy-ok",
                        "directory": "legacy-ok",
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
    published = migrated["cases"]["legacy-ok"]["published"]
    assert published is not None and published["genuine_acceptance"] is True


# --- Real persisted narrated export -----------------------------------------


def test_persisted_narrated_export_refuses_omitted_narration_mapping(
    engine, monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    """From a real persisted narrated export: removing a required narration
    mapping/file must be refused, while the ASS is retained."""

    from test_stage6_cache_validation import _complete_materialization, _session
    from test_stage6_real_executor import _write_source

    from app.materialization.acceptance_export import export_materializations
    from app.models import ClipMaterialization

    _candidate, materialization_id, _render, storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    settings = get_settings()

    narration_bytes = b"synthetic-narration-bytes-for-export"
    narration_sha = _sha(narration_bytes)
    session = _session()
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        directory = storage.source_directory(envelope.source_video_id) / "narration-export"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "narr-0.wav"
        storage.atomic_write(destination, [narration_bytes])
        envelope.narration_references = [
            {
                "chunk_id": "narr-0",
                "relative_path": storage.storage_relative(destination),
                "sha256": narration_sha,
                "container": "wav",
            }
        ]
        session.commit()
    finally:
        session.close()

    session = _session()
    try:
        result = export_materializations(session, settings, [materialization_id])
    finally:
        session.close()
    case = result["cases"][0]
    assert case["genuine_acceptance"] is True, case
    package = Path(case["directory"])
    index = json.loads((package / "index.json").read_text(encoding="utf-8"))
    narration_records = [r for r in index["assets"] if r["kind"] == "narration"]
    assert narration_records and narration_records[0]["exported_relative_path"]
    # The ASS is retained and exported.
    assert [r for r in index["assets"] if r["kind"] == "authored_ass"]

    # Remove ONLY the required narration mapping + exported file, then recompute the
    # manifest. The bound inventory/reference still requires it, so this must fail.
    index["assets"] = [r for r in index["assets"] if r["kind"] != "narration"]
    _write_index(package, index)
    (package / narration_records[0]["exported_relative_path"]).unlink()
    identity = _write_manifest(package)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_RECORD_MISSING:narration") for reason in reasons)
    assert _publish_result(tmp_path / "real-published", package, identity)["published"] is False
