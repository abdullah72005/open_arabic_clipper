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
    """Publish the ACTUAL prepared package (never an empty surrogate directory).

    Submitting an empty directory would only ever prove ``PACKAGE_MANIFEST_MISSING``;
    every caller must exercise the intended bytes so the refusal reason is the one
    under test.
    """

    root.mkdir(parents=True, exist_ok=True)
    return _publish_case_package(
        root,
        {
            "case": "case-a",
            "package_identity": identity,
            "staging_dir": str(package),
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
    # The intended refusal is the missing required narration mapping, never a
    # generic missing manifest from an empty surrogate directory.
    assert any(
        reason.startswith("PACKAGE_ASSET_RECORD_MISSING:narration")
        for reason in result["integrity_failure"]
    )


def test_publication_publishes_complete_valid_package(tmp_path: Path) -> None:
    # Positive control: the same helper that refuses the partial package publishes
    # a complete, valid package.
    root = tmp_path / "publish-ok"
    root.mkdir()
    package, identity, _index, _exported = _narrated_package(root / ".staging-build")
    result = _publish_result(root, package, identity)
    assert result["published"] is True
    assert result["genuine_acceptance"] is True
    assert result["integrity_failure"] is None
    published = Path(result["directory"])
    assert published.parts[-3:] == ("packages", "case-a", identity)
    assert (published / "final.mp4").is_file()
    assert (published / "authored.ass").is_file()


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
    assert result["genuine_acceptance"] is False
    assert any(
        reason.startswith("PACKAGE_ASSET_HASH_MISMATCH") for reason in result["integrity_failure"]
    )


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


# --- Absent / malformed / explicitly-empty authoritative inventory -----------


def _inventory_package(
    base: Path,
    *,
    identity: object = None,
    readiness: object = None,
    declared_ass: str = "",
    narration_reference: dict | None = None,
) -> tuple[Path, str]:
    """A valid media/index/manifest package with a controlled inventory shape.

    ``identity``/``readiness`` are written only when supplied (``None`` omits the
    key); ``index.assets`` is always present as exported-copy evidence.
    """

    base.mkdir(parents=True, exist_ok=True)
    (base / "final.mp4").write_bytes(b"transformed-media-bytes")
    index: dict[str, object] = {
        "genuine_acceptance": True,
        "artifact_file": "final.mp4",
        "render_artifact_sha256": _file_sha(base / "final.mp4"),
        "authored_ass_expected_sha256": declared_ass,
        "assets": [],
    }
    if identity is not None:
        index["identity"] = identity
    if readiness is not None:
        index["readiness"] = readiness
    if narration_reference is not None:
        index["narration_references"] = [narration_reference]
    _write_index(base, index)
    return base, _write_manifest(base)


def test_absent_authoritative_inventory_is_refused(tmp_path: Path) -> None:
    # The reproduction: a declared required ASS, no identity.assets/readiness.assets,
    # index.assets=[] and no actual ASS. Empty copy evidence never proves zero
    # requirements, so this must fail closed (and never be promoted by migration).
    base, identity = _inventory_package(
        tmp_path / "absent", declared_ass=_sha(b"required-but-absent-ass")
    )
    reasons = _package_integrity_reasons(base, identity)
    assert "PACKAGE_ASSET_INVENTORY_MISSING" in reasons
    assert _publish_result(tmp_path / "out", base, identity)["published"] is False
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "absent",
                        "directory": "absent",
                        "package_identity": identity,
                        "published": True,
                        "genuine_acceptance": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    entry = _read_catalog(catalog)["cases"]["absent"]
    assert entry["published"] is None
    assert entry["diagnostic_attempts"][0]["genuine_acceptance"] is False


def test_malformed_authoritative_inventory_is_refused(tmp_path: Path) -> None:
    # A present-but-malformed inventory is never silently treated as absence.
    base, identity = _inventory_package(tmp_path / "malformed", identity={"assets": "not-a-list"})
    reasons = _package_integrity_reasons(base, identity)
    assert "PACKAGE_INVENTORY_INVALID:identity" in reasons
    assert _publish_result(tmp_path / "out", base, identity)["published"] is False


def test_non_mapping_inventory_container_is_invalid_not_absent(tmp_path: Path) -> None:
    base, identity = _inventory_package(tmp_path / "non-mapping", identity="oops")
    reasons = _package_integrity_reasons(base, identity)
    assert "PACKAGE_INVENTORY_INVALID:identity" in reasons
    assert _publish_result(tmp_path / "out", base, identity)["published"] is False


def test_explicitly_empty_inventory_without_declarations_is_genuine(tmp_path: Path) -> None:
    # An explicitly empty authoritative inventory is valid only when structurally
    # valid and consistent with every bound reference/ASS declaration (none here).
    base, identity = _inventory_package(
        tmp_path / "empty-ok",
        identity={"assets": []},
        readiness={"assets": []},
    )
    assert _package_integrity_reasons(base, identity) == []
    assert _publish_result(tmp_path / "out", base, identity)["published"] is True


def test_explicitly_empty_inventory_with_declared_ass_is_refused(tmp_path: Path) -> None:
    base, identity = _inventory_package(
        tmp_path / "empty-declared",
        identity={"assets": []},
        declared_ass=_sha(b"declared-ass"),
    )
    reasons = _package_integrity_reasons(base, identity)
    assert "PACKAGE_INVENTORY_ASS_MISSING" in reasons
    assert _publish_result(tmp_path / "out", base, identity)["published"] is False


def test_explicitly_empty_inventory_with_bound_reference_is_refused(tmp_path: Path) -> None:
    base, identity = _inventory_package(
        tmp_path / "empty-ref",
        identity={"assets": []},
        narration_reference={"relative_path": "sources/review/n.wav", "sha256": _sha(b"n")},
    )
    reasons = _package_integrity_reasons(base, identity)
    assert "PACKAGE_INVENTORY_REFERENCE_MISMATCH" in reasons
    assert _publish_result(tmp_path / "out", base, identity)["published"] is False


def test_legacy_migration_promotes_explicitly_empty_inventory(tmp_path: Path) -> None:
    base, identity = _inventory_package(
        tmp_path / "legacy-empty",
        identity={"assets": []},
        readiness={"assets": []},
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "legacy-empty",
                        "directory": "legacy-empty",
                        "package_identity": identity,
                        "published": True,
                        "genuine_acceptance": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    published = _read_catalog(catalog)["cases"]["legacy-empty"]["published"]
    assert published is not None and published["genuine_acceptance"] is True


# --- ABSENT / INVALID / VALID metadata parsing matrix -----------------------


def _mutation_package(base: Path, mutation: dict[str, object]) -> tuple[Path, str]:
    """A valid explicit-empty package with ``mutation`` applied to its index."""

    base.mkdir(parents=True, exist_ok=True)
    (base / "final.mp4").write_bytes(b"transformed-media-bytes")
    index: dict[str, object] = {
        "genuine_acceptance": True,
        "artifact_file": "final.mp4",
        "render_artifact_sha256": _file_sha(base / "final.mp4"),
        "authored_ass_expected_sha256": "",
        "identity": {"assets": []},
        "readiness": {"assets": []},
        "assets": [],
    }
    index.update(mutation)
    _write_index(base, index)
    return base, _write_manifest(base)


_MALFORMED_METADATA: list[tuple[dict[str, object], str]] = [
    # Null / wrong-typed / malformed authoritative inventory containers and assets.
    ({"identity": {"assets": None}}, "PACKAGE_INVENTORY_INVALID:identity"),
    ({"identity": {"assets": "not-a-list"}}, "PACKAGE_INVENTORY_INVALID:identity"),
    ({"identity": {"assets": {}}}, "PACKAGE_INVENTORY_INVALID:identity"),
    ({"identity": {"assets": [None]}}, "PACKAGE_INVENTORY_INVALID:identity"),
    ({"identity": "oops"}, "PACKAGE_INVENTORY_INVALID:identity"),
    ({"identity": {"assets": [{"kind": "authored_ass"}]}}, "PACKAGE_INVENTORY_INVALID:identity"),
    (
        {"identity": {"assets": [{"kind": "", "relative_path": "p", "expected_sha256": "h"}]}},
        "PACKAGE_INVENTORY_INVALID:identity",
    ),
    (
        {
            "identity": {
                "assets": [{"kind": "authored_ass", "relative_path": "", "expected_sha256": "h"}]
            }
        },
        "PACKAGE_INVENTORY_INVALID:identity",
    ),
    (
        {
            "identity": {
                "assets": [{"kind": "authored_ass", "relative_path": "p", "expected_sha256": ""}]
            }
        },
        "PACKAGE_INVENTORY_INVALID:identity",
    ),
    (
        {"identity": {"assets": [{"kind": 5, "relative_path": "p", "expected_sha256": "h"}]}},
        "PACKAGE_INVENTORY_INVALID:identity",
    ),
    ({"identity": {"assets": [["authored_ass", "p"]]}}, "PACKAGE_INVENTORY_INVALID:identity"),
    ({"readiness": [1, 2]}, "PACKAGE_INVENTORY_INVALID:readiness"),
    ({"readiness": {"assets": None}}, "PACKAGE_INVENTORY_INVALID:readiness"),
    ({"readiness": {"assets": "x"}}, "PACKAGE_INVENTORY_INVALID:readiness"),
    ({"readiness": {"assets": [["narration", "", "h"]]}}, "PACKAGE_INVENTORY_INVALID:readiness"),
    # Null / wrong-typed / malformed narration and supporting references.
    ({"narration_references": None}, "PACKAGE_INVENTORY_INVALID:references"),
    ({"narration_references": {}}, "PACKAGE_INVENTORY_INVALID:references"),
    ({"narration_references": [None]}, "PACKAGE_INVENTORY_INVALID:references"),
    ({"narration_references": [{"sha256": "a" * 64}]}, "PACKAGE_INVENTORY_INVALID:references"),
    ({"narration_references": [{"relative_path": "p"}]}, "PACKAGE_INVENTORY_INVALID:references"),
    (
        {"narration_references": [{"relative_path": "", "sha256": "a" * 64}]},
        "PACKAGE_INVENTORY_INVALID:references",
    ),
    (
        {"narration_references": [{"relative_path": "p", "sha256": ""}]},
        "PACKAGE_INVENTORY_INVALID:references",
    ),
    (
        {"narration_references": [{"relative_path": 5, "sha256": "a" * 64}]},
        "PACKAGE_INVENTORY_INVALID:references",
    ),
    ({"asset_references": [{"sha256": "b" * 64}]}, "PACKAGE_INVENTORY_INVALID:references"),
    ({"asset_references": "not-a-list"}, "PACKAGE_INVENTORY_INVALID:references"),
    # Null / wrong-typed exported-copy evidence.
    ({"assets": None}, "PACKAGE_ASSET_METADATA_INVALID"),
    ({"assets": {"kind": "authored_ass"}}, "PACKAGE_ASSET_METADATA_INVALID"),
]


@pytest.mark.parametrize(("mutation", "expected"), _MALFORMED_METADATA)
def test_malformed_metadata_is_refused_and_not_promoted(
    tmp_path: Path, mutation: dict[str, object], expected: str
) -> None:
    base, identity = _mutation_package(tmp_path / "malformed", mutation)
    reasons = _package_integrity_reasons(base, identity)
    assert expected in reasons, reasons
    result = _publish_result(tmp_path / "out", base, identity)
    assert result["published"] is False
    assert expected in result["integrity_failure"]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case": "malformed",
                        "directory": "malformed",
                        "package_identity": identity,
                        "published": True,
                        "genuine_acceptance": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    entry = _read_catalog(catalog)["cases"]["malformed"]
    assert entry["published"] is None
    assert entry["diagnostic_attempts"][0]["genuine_acceptance"] is False


def test_malformed_references_do_not_become_no_requirements(tmp_path: Path) -> None:
    # Reproduction A: references with a valid hash but no relative_path must be
    # INVALID (never silently skipped into a favorable empty requirement set).
    base, _identity = _inventory_package(
        tmp_path / "malformed-refs", identity={"assets": []}, readiness={"assets": []}
    )
    index = json.loads((base / "index.json").read_text(encoding="utf-8"))
    index["narration_references"] = [{"sha256": "a" * 64}]
    index["asset_references"] = [{"sha256": "b" * 64}]
    identity = _rewrite(base, index)
    reasons = _package_integrity_reasons(base, identity)
    assert "PACKAGE_INVENTORY_INVALID:references" in reasons
    assert _publish_result(tmp_path / "out", base, identity)["published"] is False


def test_null_identity_authority_is_invalid_not_absent(tmp_path: Path) -> None:
    # Reproduction B: a present null authority is INVALID; it must never be
    # classified as absent and silently fall back to readiness.
    package, _identity, index, _exported = _narrated_package(tmp_path / "null-authority")
    index["identity"]["assets"] = None
    identity = _rewrite(package, index)
    reasons = _package_integrity_reasons(package, identity)
    assert "PACKAGE_INVENTORY_INVALID:identity" in reasons
    result = _publish_result(tmp_path / "out", package, identity)
    assert result["published"] is False
    assert "PACKAGE_INVENTORY_INVALID:identity" in result["integrity_failure"]


def test_readiness_fallback_when_identity_absent_is_genuine(tmp_path: Path) -> None:
    # A genuinely absent identity.assets may still fall back to readiness.assets.
    base, identity = _inventory_package(tmp_path / "fallback", readiness={"assets": []})
    assert _package_integrity_reasons(base, identity) == []
    assert _publish_result(tmp_path / "out", base, identity)["published"] is True


def test_readiness_fallback_with_nonempty_inventory_is_genuine(tmp_path: Path) -> None:
    package, _identity, index, _exported = _narrated_package(tmp_path / "fallback-full")
    index.pop("identity")
    identity = _rewrite(package, index)
    assert _package_integrity_reasons(package, identity) == []
    assert _publish_result(tmp_path / "out", package, identity)["published"] is True


# --- Real persisted narrated export -----------------------------------------


def _complete_narrated_materialization(engine, monkeypatch):  # type: ignore[no-untyped-def]
    """Drive the milestone authored-card + synthetic-narration setup to COMPLETE.

    Reuses the existing ``test_stage6_milestone`` components: a real FFmpeg source,
    locally synthesized lossless narration persisted through ``StorageService``,
    the real render executor, the completion callback and the finalizer. Returns
    the persisted identifiers and storage; the generation is unchanged and valid.
    """

    from sqlalchemy import select
    from stage51_support import seed_stage51
    from test_stage6_milestone import (
        _NARRATION_PROFILE_KEY,
        _ensure_narration_profile,
        _inject_deterministic_providers,
        _no_broker,
        _ready_composition,
        _session,
        _SyntheticNarrationSynthesizer,
    )
    from test_stage6_real_executor import _write_source

    from app.core.enums import JobKind, JobStatus
    from app.materialization.queue import queue_materialization
    from app.models import ClipMaterialization, ProcessingJob
    from app.services.storage import StorageService
    from app.workers.tasks import (
        _request_finalization_after_render,
        run_materialization_finalization,
        run_materialization_job,
        run_render_execution,
    )

    _no_broker(monkeypatch)
    settings = get_settings()
    storage = StorageService(settings.storage_root)
    _ensure_narration_profile(engine)

    session = _session()
    try:
        fixture = seed_stage51(
            session, monkeypatch, narration="RECOMMENDED", media_writer=_write_source
        )
        candidate = fixture.stage50.selection.candidate
        candidate_id = candidate.id
        source_video_id = candidate.source_video_id
        _ready_composition(session, fixture)
        session.commit()
    finally:
        session.close()

    synthesizer = _SyntheticNarrationSynthesizer(storage, source_video_id)
    _inject_deterministic_providers(monkeypatch, narration=synthesizer)

    session = _session()
    try:
        outcome = queue_materialization(
            session, candidate_id, priority="CANDIDATE", profile_key=_NARRATION_PROFILE_KEY
        )
        session.commit()
        prepare_job_id = outcome.job_id
        materialization_id = outcome.materialization_id
    finally:
        session.close()

    run_materialization_job(
        str(candidate_id), str(prepare_job_id), profile_key=_NARRATION_PROFILE_KEY
    )

    session = _session()
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None and envelope.lifecycle.value == "AWAITING_RENDER", (
            envelope.reason_codes if envelope else None
        )
        render_id = envelope.render_execution_id
        assert render_id is not None
        render_job = session.scalars(
            select(ProcessingJob).where(ProcessingJob.render_execution_id == render_id)
        ).first()
        assert render_job is not None
        render_job_id = render_job.id
    finally:
        session.close()

    run_render_execution(str(render_id), str(render_job_id))
    _request_finalization_after_render(render_id)

    session = _session()
    try:
        finalizer = session.scalars(
            select(ProcessingJob)
            .where(
                ProcessingJob.materialization_id == materialization_id,
                ProcessingJob.kind == JobKind.MATERIALIZATION,
                ProcessingJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
            )
            .order_by(ProcessingJob.created_at.desc())
        ).first()
        assert finalizer is not None
        finalizer_id = finalizer.id
    finally:
        session.close()

    run_materialization_finalization(str(candidate_id), str(finalizer_id))
    return materialization_id, render_id, storage


def test_persisted_narrated_export_refuses_omitted_narration_mapping(
    engine, monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    """A real narrated generation exports successfully; removing ONLY its exported
    narration mapping/file is then refused while the ASS and authoritative
    inventory are retained."""

    from test_stage6_milestone import _session

    from app.materialization.acceptance_export import export_materializations
    from app.models import ClipMaterialization

    materialization_id, _render_id, storage = _complete_narrated_materialization(
        engine, monkeypatch
    )
    settings = get_settings()

    # The lossless narration asset actually bound into the render input is real.
    session = _session()
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        references = list(envelope.narration_references or [])
        assert references and references[0].get("container") == "wav"
        narration_sha = str(references[0]["sha256"])
        managed = storage.storage_root / str(references[0]["relative_path"])
        assert managed.is_file() and _file_sha(managed) == narration_sha
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
    # The ASS is retained and exported alongside the authoritative inventory.
    assert [r for r in index["assets"] if r["kind"] == "authored_ass"]
    assert [a for a in index["identity"]["assets"] if a[0] == "authored_ass"]
    assert [a for a in index["identity"]["assets"] if a[0] == "narration"]
    # The exported narration bytes match the asset actually rendered.
    exported_narration = package / narration_records[0]["exported_relative_path"]
    assert _file_sha(exported_narration) == narration_sha

    # Remove ONLY the required narration mapping + exported file, then recompute the
    # manifest. The bound inventory/reference still requires it, so this must fail
    # with the intended reason (not a generic missing manifest).
    index["assets"] = [r for r in index["assets"] if r["kind"] != "narration"]
    _write_index(package, index)
    exported_narration.unlink()
    identity = _write_manifest(package)
    reasons = _package_integrity_reasons(package, identity)
    assert any(reason.startswith("PACKAGE_ASSET_RECORD_MISSING:narration") for reason in reasons)
    result = _publish_result(tmp_path / "real-published", package, identity)
    assert result["published"] is False
    assert any(
        reason.startswith("PACKAGE_ASSET_RECORD_MISSING:narration")
        for reason in result["integrity_failure"]
    )
