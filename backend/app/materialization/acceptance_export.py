"""Read-only Stage 6 genuine acceptance-package export.

Assembles the exact persisted lineage for one or more reviewed materializations
into a self-contained directory: the transformed MP4, a readable index, the
authored ASS, timeline, audio recipe, QC/final-assessment evidence, provider
attempt and repair usage, and a SHA-256 manifest. It is strictly read-only over
the database and managed storage: it never fabricates rights, governance, speech,
or acceptance eligibility, and it never mutates a materialization.

The default output root is ``storage/benchmarks/stage-6/manual-acceptance/`` for
the genuine set and ``storage/benchmarks/stage-6/tts-audition/`` for auditions.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.materialization.readiness import ReferencedAsset
from app.models import ClipMaterialization, ProcessingJob, SourceVideo
from app.models.render_execution import RenderExecution
from app.services.storage import StorageService


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CatalogIntegrityError(RuntimeError):
    """Raised when an existing catalog cannot be trusted (fail closed)."""


def _package_integrity_reasons(package_dir: Path, package_identity: str | None) -> list[str]:
    """Concrete reasons a package is not a genuine acceptance package (empty == ok).

    Directory naming is never integrity evidence, and neither is a self-consistent
    manifest alone: the manifest must hash to the declared package identity, the
    index must be the supported acceptance structure and mark the package genuine,
    a required nonempty artifact filename and its recorded SHA-256 must be present,
    the actual nonempty artifact must match the index/manifest identity, every
    required referenced asset must be present and verified, and every manifest path
    must stay inside the package and hash exactly. The same verifier is applied to
    package reuse and to legacy-catalog migration.
    """

    reasons: list[str] = []
    if not package_dir.is_dir():
        return ["PACKAGE_MISSING"]
    manifest = package_dir / "SHA256SUMS"
    if not manifest.is_file():
        return ["PACKAGE_MANIFEST_MISSING"]
    try:
        manifest_lines = manifest.read_text(encoding="utf-8").splitlines()
        actual_identity = _sha256_file(manifest)
    except OSError:
        return ["PACKAGE_MANIFEST_UNREADABLE"]
    if package_identity and actual_identity != package_identity:
        return ["PACKAGE_IDENTITY_MISMATCH"]
    index_path = package_dir / "index.json"
    if not index_path.is_file():
        return ["PACKAGE_INDEX_MISSING"]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["PACKAGE_INDEX_INVALID"]
    if not isinstance(index, dict):
        return ["PACKAGE_INDEX_INVALID"]
    if index.get("genuine_acceptance") is not True:
        return ["PACKAGE_INDEX_NOT_GENUINE"]
    listed: dict[str, str] = {}
    for line in manifest_lines:
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or not digest or not relative:
            reasons.append("PACKAGE_MANIFEST_INVALID")
            continue
        # Manifest paths must stay inside the package: an absolute or escaping
        # path is not a valid acceptance reference.
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            reasons.append(f"PACKAGE_PATH_ESCAPES:{relative}")
            continue
        listed[relative] = digest.lower()
        target = package_dir / relative
        if not target.is_file():
            reasons.append(f"PACKAGE_FILE_MISSING:{relative}")
            continue
        try:
            if _sha256_file(target) != digest.lower():
                reasons.append(f"PACKAGE_FILE_MISMATCH:{relative}")
        except OSError:
            reasons.append(f"PACKAGE_FILE_UNREADABLE:{relative}")
    if not listed:
        reasons.append("PACKAGE_MANIFEST_EMPTY")
    if "index.json" not in listed:
        reasons.append("PACKAGE_INDEX_UNLISTED")
    _artifact_reasons(package_dir, index, listed, reasons)
    _referenced_asset_reasons(index, reasons)
    return reasons


def _artifact_reasons(
    package_dir: Path, index: Mapping[str, object], listed: Mapping[str, str], reasons: list[str]
) -> None:
    """Require actual, nonempty, identity-matching transformed media.

    Explicit favorable booleans and a self-consistent manifest are insufficient: a
    genuine package must declare a required artifact filename and its recorded
    SHA-256, that file must exist nonempty inside the package, and its actual hash
    must match both the index record and the manifest entry.
    """

    artifact_file = index.get("artifact_file")
    if not isinstance(artifact_file, str) or not artifact_file:
        reasons.append("PACKAGE_ARTIFACT_MISSING")
        return
    recorded_sha = index.get("render_artifact_sha256")
    if not isinstance(recorded_sha, str) or not recorded_sha:
        reasons.append("PACKAGE_ARTIFACT_SHA_MISSING")
        return
    recorded_sha = recorded_sha.lower()
    artifact_path = package_dir / artifact_file
    if not artifact_path.is_file():
        reasons.append(f"PACKAGE_ARTIFACT_FILE_MISSING:{artifact_file}")
        return
    try:
        if artifact_path.stat().st_size <= 0:
            reasons.append(f"PACKAGE_ARTIFACT_EMPTY:{artifact_file}")
            return
        actual_sha = _sha256_file(artifact_path)
    except OSError:
        reasons.append(f"PACKAGE_ARTIFACT_UNREADABLE:{artifact_file}")
        return
    if actual_sha != recorded_sha:
        reasons.append(f"PACKAGE_ARTIFACT_SHA_MISMATCH:{artifact_file}")
    manifest_sha = listed.get(artifact_file)
    if manifest_sha is None:
        reasons.append(f"PACKAGE_ARTIFACT_UNLISTED:{artifact_file}")
    elif manifest_sha != actual_sha:
        reasons.append(f"PACKAGE_ARTIFACT_MANIFEST_MISMATCH:{artifact_file}")


def _referenced_asset_reasons(index: Mapping[str, object], reasons: list[str]) -> None:
    """Require every required referenced asset to be present and verified.

    The index records, for each referenced asset, whether it was copied and
    verified against its authoritative expected hash. A genuine package must have
    every copied asset verified (index ↔ asset identity), and an expected authored
    ASS must be present and match its expected hash.
    """

    expected_ass = index.get("authored_ass_expected_sha256")
    if isinstance(expected_ass, str) and expected_ass:
        actual_ass = index.get("authored_ass_sha256")
        if not isinstance(actual_ass, str) or actual_ass.lower() != expected_ass.lower():
            reasons.append("PACKAGE_AUTHORED_ASS_UNVERIFIED")
    assets = index.get("assets")
    if isinstance(assets, list):
        for record in assets:
            if not isinstance(record, Mapping):
                reasons.append("PACKAGE_ASSET_RECORD_INVALID")
                continue
            if record.get("copied") is not True:
                continue
            expected = record.get("expected_sha256")
            actual = record.get("destination_sha256")
            if (
                record.get("destination_verified") is not True
                or not isinstance(expected, str)
                or not expected
                or not isinstance(actual, str)
                or actual.lower() != expected.lower()
            ):
                reasons.append(f"PACKAGE_ASSET_UNVERIFIED:{record.get('kind')}")


def _integrity_failure(
    case: str, package_dir: Path, package_identity: str, reasons: Sequence[str]
) -> dict[str, object]:
    """A fail-closed publication result: no valid reference, staging preserved."""

    return {
        "case": case,
        "directory": str(package_dir),
        "diagnostic_directory": None,
        "published": False,
        "genuine_acceptance": False,
        "package_identity": package_identity,
        "published_reference": None,
        "diagnostic_attempts": [],
        "integrity_failure": list(reasons),
    }


def _ledger_counts(ledger: object) -> dict[str, object]:
    if not isinstance(ledger, Mapping):
        return {"reserved": 0, "responded": 0, "uncertain": 0, "entries": []}
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        return {"reserved": 0, "responded": 0, "uncertain": 0, "entries": []}
    statuses = [str(entry.get("status", "")) for entry in entries if isinstance(entry, Mapping)]
    return {
        "reserved": len(entries),
        "responded": statuses.count("RESPONDED"),
        "uncertain": statuses.count("UNCERTAIN"),
        "failed_known": statuses.count("FAILED_KNOWN"),
        "entries": [dict(entry) for entry in entries if isinstance(entry, Mapping)],
    }


def _resolved_artifact_path(storage: StorageService, render: RenderExecution) -> Path | None:
    reference = render.artifact_reference if isinstance(render.artifact_reference, dict) else {}
    relative = reference.get("relative_path")
    if not isinstance(relative, str) or not relative:
        return None
    try:
        path = storage.resolve_relative(relative)
    except Exception:
        return None
    return path if path.is_file() else None


def genuine_acceptance_reasons(
    session: Session, storage: StorageService, envelope: ClipMaterialization
) -> list[str]:
    """Concrete reasons a materialization is NOT genuine acceptance (empty == genuine).

    Delegates to the single authoritative read-only readiness view
    (:func:`app.materialization.readiness.evaluate_readiness`) shared by the API
    and the exporter. It reuses the finalizer's own live-prerequisite, render
    binding, render identity/QC, assessment, artifact, and referenced-asset checks;
    it never relies on ``is_current``/``COMPLETE``/``READY_FOR_HUMAN_REVIEW`` labels
    alone, never treats finite-but-out-of-policy QC as satisfied, and never
    substitutes a newer file at a deterministic path for the bound accepted asset.
    """

    from app.materialization.readiness import evaluate_readiness

    return list(evaluate_readiness(session, storage, envelope).reasons)


def _identity_snapshot(
    envelope: ClipMaterialization,
    render: RenderExecution | None,
    assets: Sequence[ReferencedAsset] = (),
) -> dict[str, object]:
    """The exact accepted materialization/input/output/repair/render/QC/asset identity.

    The referenced-asset identities (authored ASS, narration, supporting) are part
    of the snapshot so a recheck can never publish a package whose copied assets no
    longer match the accepted set.
    """

    metrics = envelope.metrics if isinstance(envelope.metrics, Mapping) else {}
    reference = (
        render.artifact_reference
        if render is not None and isinstance(render.artifact_reference, Mapping)
        else {}
    )
    return {
        "materialization_id": str(envelope.id),
        "input_fingerprint": envelope.input_fingerprint,
        "output_fingerprint": envelope.output_fingerprint,
        "repair_count": int(envelope.repair_count or 0),
        "render_execution_id": (
            str(envelope.render_execution_id) if envelope.render_execution_id else None
        ),
        "render_output_fingerprint": render.output_fingerprint if render is not None else None,
        "render_qc_fingerprint": render.qc_fingerprint if render is not None else None,
        "render_qc_status": (
            getattr(render.qc_status, "value", render.qc_status)
            if render is not None and render.qc_status is not None
            else None
        ),
        "artifact_sha256": str(reference.get("sha256") or ""),
        "assessment_fingerprint": metrics.get("assessment_fingerprint"),
        "assessment_policy_version": metrics.get("assessment_policy_version"),
        "assets": sorted(
            (asset.kind, asset.relative_path, asset.expected_sha256) for asset in assets
        ),
    }


def _build_package(
    session: Session,
    storage: StorageService,
    materialization_id: uuid.UUID | str,
    output_root: Path,
    *,
    case_label: str | None = None,
) -> dict[str, object]:
    """Assemble one materialization's exact lineage in an export-owned staging dir.

    The package is staged (never published here) only after the copied artifact
    bytes are verified against the accepted reference hash and the authoritative
    rows are refreshed and confirmed to represent the SAME identity. Publication
    (into an immutable versioned directory) is a separate step so a failure while
    building can never disturb the currently published package or the catalog.
    """

    parsed = materialization_id if isinstance(materialization_id, uuid.UUID) else uuid.UUID(
        str(materialization_id)
    )
    envelope = session.get(ClipMaterialization, parsed, populate_existing=True)
    if envelope is None:
        raise ValueError("materialization does not exist")
    case = case_label or (
        f"{envelope.clip_candidate_id}-{envelope.profile_key}-{str(envelope.id)[:8]}"
    )
    staging_dir = output_root / f".staging-{case}-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    render_id = envelope.render_execution_id
    render = (
        session.get(RenderExecution, render_id, populate_existing=True)
        if render_id is not None
        else None
    )
    from app.materialization.readiness import evaluate_readiness

    readiness = evaluate_readiness(session, storage, envelope)
    initial_identity = _identity_snapshot(envelope, render, readiness.assets)
    expected_artifact_sha = str(initial_identity.get("artifact_sha256") or "")

    destination_reasons: list[str] = []

    artifact_name: str | None = None
    artifact_sha: str | None = None
    artifact_verified = True
    if render is not None:
        source = _resolved_artifact_path(storage, render)
        if source is not None:
            artifact_name = "final.mp4"
            shutil.copyfile(source, staging_dir / "final.mp4")
            artifact_sha = _sha256_file(staging_dir / "final.mp4")
            if expected_artifact_sha and artifact_sha != expected_artifact_sha.lower():
                artifact_verified = False
        elif expected_artifact_sha:
            artifact_verified = False
    elif expected_artifact_sha:
        artifact_verified = False

    # Copy the exact bound authored ASS and every referenced narration/supporting
    # asset from the persisted render request, then verify the COPIED destination
    # bytes against each authoritative expected hash. A missing or mismatched copy
    # invalidates acceptance: rechecking an unchanged source file never verifies the
    # destination bytes actually published.
    authored_ass_sha: str | None = None
    ass_expected_sha = ""
    copied_assets: list[dict[str, object]] = []
    asset_dir = staging_dir / "assets"
    for asset in readiness.assets:
        record = asset.as_dict()
        record["copied"] = False
        record["destination_sha256"] = None
        record["destination_verified"] = False
        expected = asset.expected_sha256
        if asset.kind == "authored_ass":
            ass_expected_sha = expected
            if asset.present and asset.verified:
                source_path = storage.resolve_relative(asset.relative_path)
                destination = staging_dir / "authored.ass"
                shutil.copyfile(source_path, destination)
                authored_ass_sha = _sha256_file(destination)
                record["copied"] = True
                record["destination_sha256"] = authored_ass_sha
                record["destination_verified"] = bool(expected) and (
                    authored_ass_sha == expected.lower()
                )
                if not record["destination_verified"]:
                    destination_reasons.append(f"EXPORT_ASSET_MISMATCH:{asset.kind}")
            elif not asset.present:
                destination_reasons.append(f"EXPORT_ASSET_MISSING:{asset.kind}")
            # present-but-unverified-at-source is already a shared-readiness reason
            copied_assets.append(record)
            continue
        if asset.present:
            source_path = storage.resolve_relative(asset.relative_path)
            asset_dir.mkdir(parents=True, exist_ok=True)
            target = asset_dir / f"{asset.kind}-{source_path.name}"
            if source_path.resolve() != target.resolve():
                shutil.copyfile(source_path, target)
            destination_sha = _sha256_file(target)
            record["copied"] = True
            record["destination_sha256"] = destination_sha
            record["destination_verified"] = bool(expected) and (
                destination_sha == expected.lower()
            )
            if not record["destination_verified"]:
                destination_reasons.append(f"EXPORT_ASSET_MISMATCH:{asset.kind}")
        elif not asset.present:
            destination_reasons.append(f"EXPORT_ASSET_MISSING:{asset.kind}")
        copied_assets.append(record)

    # Refresh the authoritative rows and re-verify the SAME identity before
    # publishing: a changed generation (even one that would independently be
    # review-ready) can never be exported under the originally captured identity.
    session.expire_all()
    fresh = session.get(ClipMaterialization, parsed, populate_existing=True)
    fresh_render = (
        session.get(RenderExecution, fresh.render_execution_id, populate_existing=True)
        if fresh is not None and fresh.render_execution_id is not None
        else None
    )
    recheck = evaluate_readiness(session, storage, fresh) if fresh is not None else None
    fresh_identity = (
        _identity_snapshot(fresh, fresh_render, recheck.assets)
        if fresh is not None and recheck is not None
        else ({} if fresh is None else _identity_snapshot(fresh, fresh_render))
    )
    reasons = list(recheck.reasons) if recheck is not None else ["MATERIALIZATION_MISSING"]
    if fresh_identity != initial_identity:
        reasons.append("EXPORT_IDENTITY_CHANGED")
    if not artifact_verified:
        reasons.append("EXPORT_ARTIFACT_MISMATCH")
    reasons.extend(destination_reasons)
    if readiness.review_ready and recheck is not None and not recheck.review_ready:
        reasons.append("EVIDENCE_CHANGED_DURING_EXPORT")
    genuine = not reasons

    metrics = dict(envelope.metrics or {})
    index: dict[str, object] = {
        "materialization_id": str(envelope.id),
        "clip_candidate_id": str(envelope.clip_candidate_id),
        "source_video_id": str(envelope.source_video_id),
        "profile_key": envelope.profile_key,
        "priority": envelope.priority.value,
        "genuine_acceptance": genuine,
        "genuine_acceptance_reasons": reasons,
        "diagnostic": not genuine,
        "readiness": recheck.as_dict() if recheck is not None else {},
        "identity": fresh_identity,
        "lifecycle": envelope.lifecycle.value,
        "semantic_outcome": envelope.semantic_outcome.value if envelope.semantic_outcome else None,
        "human_review_required": bool(envelope.human_review_required),
        "publication_ready": False,
        "is_current": bool(envelope.is_current),
        "repair_count": int(envelope.repair_count or 0),
        "repair_actions": [dict(item) for item in (envelope.repair_actions or [])],
        "reason_codes": list(envelope.reason_codes or []),
        "input_fingerprint": envelope.input_fingerprint,
        "output_fingerprint": envelope.output_fingerprint,
        "assessment_fingerprint": metrics.get("assessment_fingerprint"),
        "final_integrated_lufs": metrics.get("final_integrated_lufs"),
        "final_true_peak_dbtp": metrics.get("final_true_peak_dbtp"),
        "render_execution_id": str(render_id) if render_id else None,
        "render_output_fingerprint": render.output_fingerprint if render is not None else None,
        "render_lifecycle": render.lifecycle.value if render is not None else None,
        "render_artifact_purpose": (
            render.artifact_purpose.value if render is not None else None
        ),
        "render_artifact_sha256": artifact_sha,
        "authored_ass_sha256": authored_ass_sha,
        "authored_ass_expected_sha256": ass_expected_sha,
        "assets": copied_assets,
        "authored_blocks": [dict(item) for item in (envelope.authored_blocks or [])],
        "slot_resolutions": [dict(item) for item in (envelope.slot_resolutions or [])],
        "narration_references": [dict(item) for item in (envelope.narration_references or [])],
        "asset_references": [dict(item) for item in (envelope.asset_references or [])],
        "claim_ledger": [dict(item) for item in (envelope.claim_ledger or [])],
        "timeline": dict(envelope.timeline_payload or {}),
        "audio_recipe": dict(envelope.audio_recipe or {}),
        "quality_findings": [dict(item) for item in (envelope.quality_findings or [])],
        "qc_result": dict(render.qc_result or {}) if render is not None else {},
        "provider_attempts": _ledger_counts(envelope.call_ledger),
        "provider_identity": dict(envelope.provider_identity or {}),
        "component_checkpoints": dict(envelope.component_checkpoints or {}),
        "artifact_file": artifact_name,
    }
    (staging_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    manifest_lines = []
    for entry in sorted(staging_dir.rglob("*")):
        if entry.is_file() and entry.name != "SHA256SUMS":
            relative = entry.relative_to(staging_dir).as_posix()
            manifest_lines.append(f"{_sha256_file(entry)}  {relative}")
    (staging_dir / "SHA256SUMS").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )

    package_identity = _sha256_file(staging_dir / "SHA256SUMS")
    return {
        "case": case,
        "staging_dir": staging_dir,
        "package_identity": package_identity,
        "index": index,
        "genuine": genuine,
        "materialization_id": str(envelope.id),
        "clip_candidate_id": str(envelope.clip_candidate_id),
    }


def export_materializations(
    session: Session,
    settings: Settings,
    materialization_ids: Sequence[uuid.UUID | str],
    *,
    output_root: Path | None = None,
    case_labels: Sequence[str | None] | None = None,
) -> dict[str, object]:
    """Export a bounded set of materializations and a versioned catalog.

    Each verified package is published into an immutable versioned directory
    ``packages/<case>/<package_identity>/`` (a diagnostic/failed attempt into a
    separate versioned diagnostic directory) BEFORE the catalog pointer is
    switched. The versioned catalog is keyed by stable case identity, keeps the
    published valid reference separate from diagnostic attempt references, merges
    without dropping unrelated cases, and is read/merged/written under an
    export-owned process lock then atomically replaced. A failure before catalog
    replacement leaves the previous reference and bytes usable.
    """

    storage = StorageService(settings.storage_root)
    root = output_root or (settings.storage_root / "benchmarks" / "stage-6" / "manual-acceptance")
    root.mkdir(parents=True, exist_ok=True)
    labels = list(case_labels or [])
    results: list[dict[str, object]] = []
    for index, materialization_id in enumerate(materialization_ids):
        label = labels[index] if index < len(labels) else None
        build = _build_package(session, storage, materialization_id, root, case_label=label)
        published = _publish_case_package(root, build)
        public = {
            "case": published["case"],
            "directory": published["directory"],
            "genuine_acceptance": published["genuine_acceptance"],
            "published": published["published"],
            "diagnostic_directory": published["diagnostic_directory"],
            "package_identity": published["package_identity"],
        }
        failure = published.get("integrity_failure")
        if failure:
            public["integrity_failure"] = list(failure)
        results.append(
            {
                "case": build["case"],
                "materialization_id": build["materialization_id"],
                "clip_candidate_id": build["clip_candidate_id"],
                "published_reference": published["published_reference"],
                "diagnostic_attempts": published["diagnostic_attempts"],
                "public": public,
                "integrity_failure": list(failure) if failure else None,
            }
        )

    # Switch the catalog pointer only after every package is published; merge into
    # the existing catalog under the export-owned lock, then atomically replace.
    # An integrity failure is never merged as a success (its staging evidence is
    # preserved and the previously published reference is left untouched).
    with _catalog_lock(root):
        catalog = _read_catalog(root / "catalog.json")
        for result in results:
            if result.get("integrity_failure"):
                continue
            _merge_case_into_catalog(catalog, result)
        _write_catalog_atomic(root, catalog)
    return {
        "root": str(root),
        "cases": [result["public"] for result in results],
        "catalog": catalog,
    }


def _publish_case_package(root: Path, build: dict[str, object]) -> dict[str, object]:
    """Publish a verified staging package into an immutable versioned directory.

    Contents are published before the catalog pointer is switched (by the caller):
    a genuine package becomes ``packages/<case>/<package_identity>/`` and a
    diagnostic/failed attempt a separate versioned diagnostic directory. An
    *existing* target is never assumed intact from its name alone: its manifest,
    index, and every listed destination file are re-verified before it may be
    reused. On an integrity mismatch this fails closed with a specific reason — the
    valid staging evidence is preserved, the existing package is not overwritten,
    and no successful reference is produced. The last valid package is never
    renamed, deleted, or overwritten.
    """

    case = str(build["case"])
    package_identity = str(build["package_identity"])
    staging_dir = Path(build["staging_dir"])
    genuine = bool(build["genuine"])
    packages_dir = root / "packages" / case
    packages_dir.mkdir(parents=True, exist_ok=True)
    if genuine:
        target = packages_dir / package_identity
        if target.exists():
            reasons = _package_integrity_reasons(target, package_identity)
            if reasons:
                return _integrity_failure(case, target, package_identity, reasons)
            # Verified identical immutable package: the staging duplicate is safe
            # to drop.
            shutil.rmtree(staging_dir, ignore_errors=True)
        else:
            staging_reasons = _package_integrity_reasons(staging_dir, package_identity)
            if staging_reasons:
                return _integrity_failure(case, staging_dir, package_identity, staging_reasons)
            os.replace(staging_dir, target)
            published_reasons = _package_integrity_reasons(target, package_identity)
            if published_reasons:
                return _integrity_failure(case, target, package_identity, published_reasons)
        return {
            "case": case,
            "directory": str(target),
            "diagnostic_directory": None,
            "published": True,
            "genuine_acceptance": True,
            "package_identity": package_identity,
            "integrity_failure": None,
            "published_reference": {
                "identity": package_identity,
                "directory": target.relative_to(root).as_posix(),
                "genuine_acceptance": True,
            },
            "diagnostic_attempts": [],
        }
    target = packages_dir / f"diagnostic-{package_identity[:16]}-{uuid.uuid4().hex[:8]}"
    os.replace(staging_dir, target)
    return {
        "case": case,
        "directory": str(target),
        "diagnostic_directory": str(target),
        "published": False,
        "genuine_acceptance": False,
        "package_identity": package_identity,
        "integrity_failure": None,
        "published_reference": None,
        "diagnostic_attempts": [
            {
                "identity": package_identity,
                "directory": target.relative_to(root).as_posix(),
                "genuine_acceptance": False,
            }
        ],
    }


def export_materialization(
    session: Session,
    storage: StorageService,
    materialization_id: uuid.UUID | str,
    output_root: Path,
    *,
    case_label: str | None = None,
) -> dict[str, object]:
    """Build and publish one package into an immutable versioned directory.

    Retained for direct callers: the staging package is published immediately
    (catalog merging is performed by :func:`export_materializations`). The last
    valid package is never overwritten by a diagnostic or failed attempt.
    """

    build = _build_package(session, storage, materialization_id, output_root, case_label=case_label)
    published = _publish_case_package(output_root, build)
    return {**published, "index": build["index"], "genuine": build["genuine"]}


#: Versioned catalog schema keyed by stable case identity.
CATALOG_VERSION = 2


@contextmanager
def _catalog_lock(root: Path) -> Iterator[None]:
    """Serialize catalog read/merge/write with an export-owned process lock."""

    lock_path = root / ".catalog.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _read_catalog(catalog_path: Path) -> dict[str, object]:
    """Read the versioned catalog, migrating (and preserving) any legacy form.

    An unreadable, malformed, or unsupported catalog fails closed with
    :class:`CatalogIntegrityError` rather than being silently reset to an empty
    catalog: the original file and every existing reference are preserved so a
    corrupt catalog can never authorize dropping previously published packages.

    A legacy flat-list catalog is migrated conservatively. A published valid
    reference is established only from explicit favorable metadata
    (``published`` *and* ``genuine_acceptance`` both true) *and* verified package
    evidence; an ambiguous entry (including ``published=true`` with
    ``genuine_acceptance=false``) is preserved as a diagnostic reference and never
    promoted.
    """

    if not catalog_path.is_file():
        return {"version": CATALOG_VERSION, "cases": {}}
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CatalogIntegrityError(f"unreadable catalog: {error}") from None
    if not isinstance(data, dict):
        raise CatalogIntegrityError("catalog is not a JSON object")
    if "cases" not in data:
        raise CatalogIntegrityError("catalog has no cases key")
    raw_cases = data.get("cases")
    cases: dict[str, object] = {}
    if isinstance(raw_cases, dict):
        # A keyed catalog is only supported at the current version. A different
        # version is a future/unsupported schema: fail closed rather than
        # reinterpret it.
        if data.get("version") != CATALOG_VERSION:
            raise CatalogIntegrityError(
                f"unsupported catalog version: {data.get('version')!r}"
            )
        for key, value in raw_cases.items():
            cases[str(key)] = value
        return {"version": CATALOG_VERSION, "cases": cases}
    if not isinstance(raw_cases, list):
        raise CatalogIntegrityError("catalog cases is neither an object nor a list")
    root = catalog_path.parent
    for entry in raw_cases:
        if not isinstance(entry, dict) or not entry.get("case"):
            raise CatalogIntegrityError("malformed legacy catalog entry")
        case = str(entry["case"])
        directory = entry.get("directory")
        identity = entry.get("package_identity")
        published_flag = entry.get("published")
        genuine_flag = entry.get("genuine_acceptance")
        diagnostic_dir = entry.get("diagnostic_directory")
        published: dict[str, object] | None = None
        if (
            published_flag is True
            and genuine_flag is True
            and isinstance(directory, str)
            and directory
            and not _package_integrity_reasons(
                root / directory, identity if isinstance(identity, str) else None
            )
        ):
            published = {
                "identity": identity,
                "directory": directory,
                "genuine_acceptance": True,
            }
        diagnostics: list[dict[str, object]] = []
        if published is None:
            reference_dir = (
                diagnostic_dir
                if isinstance(diagnostic_dir, str) and diagnostic_dir
                else (directory if isinstance(directory, str) and directory else None)
            )
            if reference_dir:
                diagnostics.append(
                    {
                        "directory": reference_dir,
                        "identity": identity,
                        "genuine_acceptance": False,
                    }
                )
        cases[case] = {
            "case": case,
            "published": published,
            "diagnostic_attempts": diagnostics,
        }
    return {"version": CATALOG_VERSION, "cases": cases}


def _write_catalog_atomic(root: Path, catalog: dict[str, object]) -> None:
    """Atomically replace the catalog file after a full merge."""

    catalog_path = root / "catalog.json"
    temporary = root / f".catalog-{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, catalog_path)


def _merge_case_into_catalog(catalog: dict[str, object], entry: Mapping[str, object]) -> None:
    """Merge one case's references without dropping unrelated cases.

    A diagnostic/failed attempt never replaces the published valid reference: only
    a verified genuine package updates ``published``. Other case keys are left
    exactly as read from the existing catalog.
    """

    cases = catalog.get("cases")
    if not isinstance(cases, dict):
        cases = {}
        catalog["cases"] = cases
    case = str(entry["case"])
    existing = cases.get(case)
    if not isinstance(existing, dict):
        existing = {"case": case, "published": None, "diagnostic_attempts": []}
    published = entry.get("published_reference")
    if isinstance(published, Mapping):
        existing["published"] = dict(published)
    attempts = existing.get("diagnostic_attempts")
    if not isinstance(attempts, list):
        attempts = []
    for attempt in entry.get("diagnostic_attempts", []) or []:
        if not isinstance(attempt, Mapping):
            continue
        if not any(
            isinstance(item, Mapping) and item.get("identity") == attempt.get("identity")
            for item in attempts
        ):
            attempts.append(dict(attempt))
    existing["diagnostic_attempts"] = attempts
    if entry.get("materialization_id") is not None:
        existing["materialization_id"] = entry.get("materialization_id")
    if entry.get("clip_candidate_id") is not None:
        existing["clip_candidate_id"] = entry.get("clip_candidate_id")
    cases[case] = existing


def discover_eligible_materializations(
    session: Session, *, limit: int = 4, storage: StorageService | None = None
) -> list[uuid.UUID]:
    """Bounded inventory of currently-eligible, review-ready materializations.

    A candidate qualifies only when it passes the full genuine-acceptance check:
    live prerequisites, effective completed materialization, exact reverse-bound
    TRANSFORMED_SHORT render/output/repair identity, final assessment, QC, and the
    actual on-disk artifact hash. An absent or stale prerequisite simply yields
    fewer rows; eligibility is never fabricated from ``is_current``/``COMPLETE``.
    """

    if storage is None:
        from app.core.settings import get_settings

        storage = StorageService(get_settings().storage_root)
    rows = session.scalars(
        select(ClipMaterialization)
        .where(
            ClipMaterialization.is_current.is_(True),
            ClipMaterialization.human_review_required.is_(True),
        )
        .order_by(ClipMaterialization.created_at.desc())
        .limit(max(1, int(limit)))
    ).all()
    eligible: list[uuid.UUID] = []
    for row in rows:
        if not genuine_acceptance_reasons(session, storage, row):
            eligible.append(uuid.UUID(str(row.id)))
    return eligible


def inventory_source_readiness(
    session: Session, *, limit: int = 8, storage: StorageService | None = None
) -> dict[str, object]:
    """A bounded read-only inventory of existing source/candidate readiness.

    Reporting this separately prevents an *empty completed-materialization
    inventory* from being misread as "source media is unavailable": each entry
    reports the concrete prerequisite stage reached (candidate, usable
    refinement, materialization, review-ready) so a count alone never stands in
    for eligibility, and absence is explained by named reasons.
    """

    from app.models import CandidateRefinement, ClipCandidate

    if storage is None:
        from app.core.settings import get_settings

        storage = StorageService(get_settings().storage_root)
    sources = session.scalars(
        select(SourceVideo).order_by(SourceVideo.created_at.desc()).limit(max(1, int(limit)))
    ).all()
    usable_refinement_statuses = {
        "CANDIDATE_REFINED",
        "FINAL_TRANSCRIPT_READY",
        "NEEDS_MANUAL_TRANSCRIPT_REVIEW",
        "PROVIDER_DEGRADED",
    }
    entries: list[dict[str, object]] = []
    for source in sources:
        candidates = session.scalars(
            select(ClipCandidate.id).where(
                ClipCandidate.source_video_id == source.id,
                ClipCandidate.is_current.is_(True),
            )
        ).all()
        refinements = session.scalars(
            select(CandidateRefinement.status).where(
                CandidateRefinement.source_video_id == source.id
            )
        ).all()
        materializations = session.scalars(
            select(ClipMaterialization).where(
                ClipMaterialization.source_video_id == source.id,
                ClipMaterialization.is_current.is_(True),
            )
        ).all()
        has_usable_refinement = any(
            getattr(status, "value", str(status)) in usable_refinement_statuses
            for status in refinements
        )
        review_ready = sum(
            1 for row in materializations if not genuine_acceptance_reasons(session, storage, row)
        )
        reasons: list[str] = []
        if not candidates:
            reasons.append("NO_CURRENT_CANDIDATE")
        if not has_usable_refinement:
            reasons.append("NO_USABLE_REFINEMENT")
        if not materializations:
            reasons.append("NO_MATERIALIZATION")
        elif review_ready == 0:
            reasons.append("NO_REVIEW_READY_MATERIALIZATION")
        entries.append(
            {
                "source_video_id": str(source.id),
                "rights_status": getattr(source.rights_status, "value", str(source.rights_status)),
                "media_origin": getattr(source.media_origin, "value", str(source.media_origin)),
                "lifecycle": getattr(
                    source.lifecycle_state, "value", str(source.lifecycle_state)
                ),
                "candidate_count": len(candidates),
                "refinement_count": len(refinements),
                "has_usable_refinement": has_usable_refinement,
                "current_materialization_count": len(materializations),
                "review_ready_materializations": review_ready,
                "prerequisite_reasons": reasons,
                "review_ready": review_ready > 0,
            }
        )
    return {
        "source_count": len(entries),
        "review_ready_source_count": sum(1 for entry in entries if entry["review_ready"]),
        "sources": entries,
    }


def export_tts_audition(
    settings: Settings,
    samples: Sequence[Mapping[str, object]],
    *,
    output_root: Path | None = None,
) -> dict[str, object]:
    """Export audition audio/identity/checks and an index under the audition root."""

    root = output_root or (settings.storage_root / "benchmarks" / "stage-6" / "tts-audition")
    root.mkdir(parents=True, exist_ok=True)
    storage = StorageService(settings.storage_root)
    written: list[dict[str, object]] = []
    for sample in samples:
        relative = str(sample.get("relative_path") or "")
        name = str(sample.get("name") or Path(relative).stem or "sample")
        record = {key: sample[key] for key in sample if key != "bytes"}
        record["name"] = name
        if relative:
            try:
                source = storage.resolve_relative(relative)
            except Exception:
                source = None
            if source is not None and source.is_file():
                target = root / f"{name}.wav"
                # A default export writes the audition asset into the very root it
                # already lives in; never copy a file onto itself.
                if source.resolve() != target.resolve():
                    shutil.copyfile(source, target)
                record["sha256"] = _sha256_file(target)
        written.append(record)
    (root / "index.json").write_text(
        json.dumps({"samples": written}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {"root": str(root), "samples": written}


def materialization_job_summary(session: Session, materialization_id: uuid.UUID) -> list[dict]:
    """Read-only disposition summary of every job bound to a materialization."""

    return [
        {
            "id": str(job.id),
            "kind": job.kind.value,
            "status": job.status.value,
        }
        for job in session.scalars(
            select(ProcessingJob)
            .where(ProcessingJob.materialization_id == materialization_id)
            .order_by(ProcessingJob.created_at.asc())
        ).all()
    ]


__all__ = [
    "discover_eligible_materializations",
    "export_materialization",
    "export_materializations",
    "export_tts_audition",
    "genuine_acceptance_reasons",
    "inventory_source_readiness",
    "materialization_job_summary",
]
