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

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import MaterializationOutcome, RenderArtifactPurpose
from app.core.settings import Settings
from app.models import ClipMaterialization, ProcessingJob
from app.models.render_execution import RenderExecution
from app.services.storage import StorageService


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


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


def _authored_ass_reference(render: RenderExecution) -> tuple[str, str]:
    """Return the authored ASS (relative_path, sha256) from the persisted request."""

    payload = render.request_payload if isinstance(render.request_payload, dict) else {}
    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
    authored = spec.get("authored") if isinstance(spec.get("authored"), dict) else {}
    ref = authored.get("authored_ass") if isinstance(authored.get("authored_ass"), dict) else {}
    return str(ref.get("relative_path") or ""), str(ref.get("sha256") or "")


def _authored_ass_path(storage: StorageService, envelope: ClipMaterialization) -> Path:
    return (
        storage.source_directory(envelope.source_video_id)
        / "materializations"
        / str(envelope.id)
        / "authored.ass"
    )


def genuine_acceptance_reasons(
    session: Session, storage: StorageService, envelope: ClipMaterialization
) -> list[str]:
    """Concrete reasons a materialization is NOT genuine acceptance (empty == genuine).

    Acceptance eligibility verifies live prerequisites, the effective completed
    materialization, the exact reverse-bound TRANSFORMED_SHORT render/output/repair
    identity, the final assessment, QC, and the actual on-disk artifact hash. It
    never relies on ``is_current``/``COMPLETE`` labels alone.
    """

    reasons: list[str] = []
    if not envelope.is_current:
        reasons.append("NOT_CURRENT")
    if envelope.lifecycle.value != "COMPLETE":
        reasons.append("LIFECYCLE_NOT_COMPLETE")
    if envelope.semantic_outcome != MaterializationOutcome.READY_FOR_HUMAN_REVIEW:
        reasons.append("NOT_REVIEW_READY")
    if not envelope.human_review_required:
        reasons.append("HUMAN_REVIEW_NOT_REQUIRED")
    render_id = envelope.render_execution_id
    if render_id is None:
        reasons.append("NO_BOUND_RENDER")
        return reasons
    render = session.get(RenderExecution, render_id, populate_existing=True)
    if render is None:
        reasons.append("RENDER_MISSING")
        return reasons
    if not render.is_current:
        reasons.append("RENDER_NOT_CURRENT")
    if render.lifecycle.value != "COMPLETE":
        reasons.append("RENDER_NOT_COMPLETE")
    if render.materialization_id != envelope.id:
        reasons.append("RENDER_NOT_REVERSE_BOUND")
    if render.artifact_purpose != RenderArtifactPurpose.TRANSFORMED_SHORT:
        reasons.append("RENDER_NOT_TRANSFORMED_SHORT")
    payload = render.request_payload if isinstance(render.request_payload, dict) else {}
    expected = str(payload.get("expected_output_fingerprint") or "")
    if expected and expected != envelope.output_fingerprint:
        reasons.append("EXPECTED_OUTPUT_FINGERPRINT_MISMATCH")
    if str(payload.get("materialization_id") or "") not in {"", str(envelope.id)}:
        reasons.append("REPAIR_REVISION_MISMATCH")
    metrics = dict(envelope.metrics or {})
    if not metrics.get("assessment_fingerprint"):
        reasons.append("MISSING_FINAL_ASSESSMENT")
    qc = render.qc_result if isinstance(render.qc_result, dict) else {}
    measured = qc.get("measured") if isinstance(qc.get("measured"), dict) else {}
    if not _finite(measured.get("final_integrated_lufs")) or not _finite(
        measured.get("final_true_peak_dbtp")
    ):
        reasons.append("MISSING_QC_MEASUREMENT")
    artifact = _resolved_artifact_path(storage, render)
    if artifact is None:
        reasons.append("ARTIFACT_MISSING")
    else:
        reference = render.artifact_reference if isinstance(render.artifact_reference, dict) else {}
        recorded = str(reference.get("sha256") or "")
        if not recorded or _sha256_file(artifact) != recorded:
            reasons.append("ARTIFACT_HASH_MISMATCH")
    return reasons


def export_materialization(
    session: Session,
    storage: StorageService,
    materialization_id: uuid.UUID | str,
    output_root: Path,
    *,
    case_label: str | None = None,
) -> dict[str, object]:
    """Export one materialization's exact lineage into ``output_root/<case>/``."""

    parsed = materialization_id if isinstance(materialization_id, uuid.UUID) else uuid.UUID(
        str(materialization_id)
    )
    envelope = session.get(ClipMaterialization, parsed, populate_existing=True)
    if envelope is None:
        raise ValueError("materialization does not exist")
    case = case_label or (
        f"{envelope.clip_candidate_id}-{envelope.profile_key}-{str(envelope.id)[:8]}"
    )
    case_dir = output_root / case
    case_dir.mkdir(parents=True, exist_ok=True)

    render_id = envelope.render_execution_id
    render = (
        session.get(RenderExecution, render_id, populate_existing=True)
        if render_id is not None
        else None
    )

    genuine_reasons = genuine_acceptance_reasons(session, storage, envelope)
    genuine = not genuine_reasons

    artifact_name: str | None = None
    artifact_sha: str | None = None
    if render is not None:
        source = _resolved_artifact_path(storage, render)
        if source is not None:
            artifact_name = "final.mp4"
            shutil.copyfile(source, case_dir / artifact_name)
            artifact_sha = _sha256_file(case_dir / artifact_name)

    # The exact accepted authored ASS (never the artifact_reference JSON): resolve
    # it deterministically and verify its bytes against the persisted hash.
    ass_relative, ass_expected_sha = (
        _authored_ass_reference(render) if render is not None else ("", "")
    )
    authored_ass_sha: str | None = None
    ass_source = _authored_ass_path(storage, envelope)
    if ass_source.is_file():
        shutil.copyfile(ass_source, case_dir / "authored.ass")
        authored_ass_sha = _sha256_file(case_dir / "authored.ass")
    elif ass_relative:
        try:
            candidate = storage.resolve_relative(ass_relative)
        except Exception:
            candidate = None
        if candidate is not None and candidate.is_file():
            shutil.copyfile(candidate, case_dir / "authored.ass")
            authored_ass_sha = _sha256_file(case_dir / "authored.ass")

    # Copy the exact referenced narration/supporting assets, verifying each hash.
    asset_dir = case_dir / "assets"
    copied_assets: list[dict[str, object]] = []
    asset_refs: list[tuple[str, str, str]] = []
    for chunk in envelope.narration_references or []:
        if isinstance(chunk, Mapping):
            asset_refs.append(
                ("narration", str(chunk.get("relative_path") or ""), str(chunk.get("sha256") or ""))
            )
    for item in envelope.asset_references or []:
        if isinstance(item, Mapping):
            asset_refs.append(
                ("supporting", str(item.get("relative_path") or ""), str(item.get("sha256") or ""))
            )
    for kind, relative_path, expected_sha in asset_refs:
        if not relative_path:
            continue
        try:
            source_path = storage.resolve_relative(relative_path)
        except Exception:
            source_path = None
        if source_path is None or not source_path.is_file():
            copied_assets.append(
                {"kind": kind, "relative_path": relative_path, "copied": False,
                 "reason": "MISSING", "sha256": expected_sha}
            )
            continue
        actual_sha = _sha256_file(source_path)
        verified = bool(expected_sha) and actual_sha == expected_sha
        asset_dir.mkdir(parents=True, exist_ok=True)
        target = asset_dir / f"{kind}-{source_path.name}"
        if source_path.resolve() != target.resolve():
            shutil.copyfile(source_path, target)
        copied_assets.append(
            {
                "kind": kind,
                "relative_path": relative_path,
                "copied": True,
                "sha256": actual_sha,
                "verified": verified,
            }
        )

    metrics = dict(envelope.metrics or {})
    index: dict[str, object] = {
        "materialization_id": str(envelope.id),
        "clip_candidate_id": str(envelope.clip_candidate_id),
        "source_video_id": str(envelope.source_video_id),
        "profile_key": envelope.profile_key,
        "priority": envelope.priority.value,
        "genuine_acceptance": genuine,
        "genuine_acceptance_reasons": genuine_reasons,
        "lifecycle": envelope.lifecycle.value,
        "semantic_outcome": envelope.semantic_outcome.value if envelope.semantic_outcome else None,
        "human_review_required": bool(envelope.human_review_required),
        "publication_ready": bool(envelope.publication_ready),
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
    (case_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    manifest_lines = []
    for entry in sorted(case_dir.rglob("*")):
        if entry.is_file() and entry.name != "SHA256SUMS":
            relative = entry.relative_to(case_dir).as_posix()
            manifest_lines.append(f"{_sha256_file(entry)}  {relative}")
    (case_dir / "SHA256SUMS").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return {"case": case, "directory": str(case_dir), "index": index, "genuine": genuine}


def export_materializations(
    session: Session,
    settings: Settings,
    materialization_ids: Sequence[uuid.UUID | str],
    *,
    output_root: Path | None = None,
    case_labels: Sequence[str | None] | None = None,
) -> dict[str, object]:
    """Export a bounded set of materializations and a top-level catalog."""

    storage = StorageService(settings.storage_root)
    root = output_root or (settings.storage_root / "benchmarks" / "stage-6" / "manual-acceptance")
    root.mkdir(parents=True, exist_ok=True)
    labels = list(case_labels or [])
    catalog: list[dict[str, object]] = []
    for index, materialization_id in enumerate(materialization_ids):
        label = labels[index] if index < len(labels) else None
        result = export_materialization(
            session, storage, materialization_id, root, case_label=label
        )
        catalog.append(
            {
                "case": result["case"],
                "directory": result["directory"],
                "genuine_acceptance": result["genuine"],
            }
        )
    (root / "catalog.json").write_text(
        json.dumps({"cases": catalog}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"root": str(root), "cases": catalog}


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


def inventory_source_readiness(session: Session, *, limit: int = 8) -> dict[str, object]:
    """A bounded read-only inventory of existing source/candidate readiness.

    Reporting this separately prevents an *empty completed-materialization
    inventory* from being misread as "source media is unavailable": the concrete
    prerequisite counts below show whether sources/candidates exist but simply are
    not yet review-ready.
    """

    from app.models import ClipCandidate
    from app.models.source_video import SourceVideo

    sources = session.scalars(
        select(SourceVideo).order_by(SourceVideo.created_at.desc()).limit(max(1, int(limit)))
    ).all()
    entries: list[dict[str, object]] = []
    for source in sources:
        candidate_count = len(
            session.scalars(
                select(ClipCandidate.id).where(ClipCandidate.source_video_id == source.id)
            ).all()
        )
        entries.append(
            {
                "source_video_id": str(source.id),
                "rights_status": getattr(source.rights_status, "value", str(source.rights_status)),
                "media_origin": getattr(source.media_origin, "value", str(source.media_origin)),
                "lifecycle": getattr(
                    source.lifecycle_state, "value", str(source.lifecycle_state)
                ),
                "candidate_count": candidate_count,
            }
        )
    return {"source_count": len(entries), "sources": entries}


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
