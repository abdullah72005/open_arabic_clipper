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

    artifact_name: str | None = None
    artifact_sha: str | None = None
    if render is not None:
        source = _resolved_artifact_path(storage, render)
        if source is not None:
            artifact_name = "final.mp4"
            shutil.copyfile(source, case_dir / artifact_name)
            artifact_sha = _sha256_file(case_dir / artifact_name)

    (case_dir / "authored.ass").write_text(
        json.dumps(envelope.artifact_reference or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = dict(envelope.metrics or {})
    index: dict[str, object] = {
        "materialization_id": str(envelope.id),
        "clip_candidate_id": str(envelope.clip_candidate_id),
        "source_video_id": str(envelope.source_video_id),
        "profile_key": envelope.profile_key,
        "priority": envelope.priority.value,
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
        "render_artifact_sha256": artifact_sha,
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
    for entry in sorted(case_dir.iterdir()):
        if entry.is_file() and entry.name != "SHA256SUMS":
            manifest_lines.append(f"{_sha256_file(entry)}  {entry.name}")
    (case_dir / "SHA256SUMS").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return {"case": case, "directory": str(case_dir), "index": index}


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
        catalog.append({"case": result["case"], "directory": result["directory"]})
    (root / "catalog.json").write_text(
        json.dumps({"cases": catalog}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"root": str(root), "cases": catalog}


def discover_eligible_materializations(
    session: Session, *, limit: int = 4
) -> list[uuid.UUID]:
    """Bounded inventory of currently-eligible, review-ready materializations.

    A candidate qualifies only when the current Stage 6 generation is COMPLETE,
    requests human review, and has a completed transformed render. This never
    fabricates eligibility: an absent prerequisite simply yields fewer rows.
    """

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
        if row.render_execution_id is None:
            continue
        render = session.get(RenderExecution, row.render_execution_id, populate_existing=True)
        if render is None or render.lifecycle.value != "COMPLETE":
            continue
        eligible.append(uuid.UUID(str(row.id)))
    return eligible


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
    "materialization_job_summary",
]
