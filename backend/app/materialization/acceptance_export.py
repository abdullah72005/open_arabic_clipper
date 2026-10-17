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
        results.append(
            {
                "case": build["case"],
                "materialization_id": build["materialization_id"],
                "clip_candidate_id": build["clip_candidate_id"],
                "published_reference": published["published_reference"],
                "diagnostic_attempts": published["diagnostic_attempts"],
                "public": {
                    "case": published["case"],
                    "directory": published["directory"],
                    "genuine_acceptance": published["genuine_acceptance"],
                    "published": published["published"],
                    "diagnostic_directory": published["diagnostic_directory"],
                    "package_identity": published["package_identity"],
                },
            }
        )

    # Switch the catalog pointer only after every package is published; merge into
    # the existing catalog under the export-owned lock, then atomically replace.
    with _catalog_lock(root):
        catalog = _read_catalog(root / "catalog.json")
        for result in results:
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
    diagnostic/failed attempt a separate versioned diagnostic directory. The last
    valid package is never renamed, deleted, or overwritten.
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
            # Idempotent republish: an identical immutable package already exists.
            shutil.rmtree(staging_dir, ignore_errors=True)
        else:
            os.replace(staging_dir, target)
        return {
            "case": case,
            "directory": str(target),
            "diagnostic_directory": None,
            "published": True,
            "genuine_acceptance": True,
            "package_identity": package_identity,
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
    """Read the versioned catalog, migrating (and preserving) any legacy form."""

    if not catalog_path.is_file():
        return {"version": CATALOG_VERSION, "cases": {}}
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": CATALOG_VERSION, "cases": {}}
    if not isinstance(data, dict):
        return {"version": CATALOG_VERSION, "cases": {}}
    raw_cases = data.get("cases")
    cases: dict[str, object] = {}
    if isinstance(raw_cases, dict):
        for key, value in raw_cases.items():
            cases[str(key)] = value
    elif isinstance(raw_cases, list):
        # Legacy flat-catalog list: preserve each entry under its case key.
        for entry in raw_cases:
            if not isinstance(entry, dict) or not entry.get("case"):
                continue
            published = (
                {
                    "identity": entry.get("package_identity"),
                    "directory": entry.get("directory"),
                    "genuine_acceptance": True,
                }
                if entry.get("published")
                else None
            )
            diagnostics = (
                [
                    {
                        "directory": entry.get("diagnostic_directory"),
                        "identity": entry.get("package_identity"),
                        "genuine_acceptance": False,
                    }
                ]
                if entry.get("diagnostic_directory")
                else []
            )
            cases[str(entry["case"])] = {
                "case": str(entry["case"]),
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
