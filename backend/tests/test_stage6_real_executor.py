"""Stage 6 real persisted executor integration (real FFmpeg, not manual evidence).

Drives the *actual* production route through normal executor entry points with a
generated (but real) source: materialization POST/queue -> prepare executor ->
render handoff -> RenderExecutionExecutor -> real FFmpeg/compiler/runner/probe/QC
-> completion callback -> finalizer executor. No COMPLETE/PASS/artifact evidence
is inserted manually: the artifact, QC, and loudness come from the real render.

This is a synthetic integration fixture (generated media), not product
acceptance.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    seed_stage51,
)

from app.composition.executor import build_visual_composition_executor
from app.composition.queue import get_or_create_plan_row
from app.core.enums import JobKind, JobStatus
from app.core.settings import get_settings
from app.db.base import Base
from app.db.session import create_session_factory
from app.materialization import queue as queue_module
from app.materialization.executor import build_materialization_executor
from app.materialization.providers import FakeCritiqueProvider, FakeWritingProvider
from app.materialization.queue import queue_materialization
from app.models import ClipMaterialization, ProcessingJob
from app.models.render_execution import RenderExecution
from app.render.execution.executor import build_render_execution_executor

_ANNOTATION_COPY = (
    "That drop looks milder once you control for which teams already promoted heavily."
)

_TERMINAL = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


def _write_source(path: Path) -> None:
    """Narrow test-fixture media factory: real 1080p30/AAC source >= source bounds."""

    settings = get_settings()
    subprocess.run(  # noqa: S603
        [
            settings.ffmpeg_binary,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30:duration=50",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=50",
            "-vf",
            "hue=H=2*PI*t:s=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


class _RenderCapture:
    def __init__(self) -> None:
        self.render_id: str | None = None
        self.job_id: str | None = None

    def dispatch(self, render_id: object, job_id: object, force: bool = False) -> None:
        self.render_id = str(render_id)
        self.job_id = str(job_id)


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _ready_composition(session: Session, fixture: Any) -> Any:
    candidate = fixture.stage50.selection.candidate
    row = get_or_create_plan_row(session, candidate)
    job = ProcessingJob(
        source_video_id=candidate.source_video_id,
        kind=JobKind.VISUAL_COMPOSITION,
        visual_composition_plan_id=row.id,
    )
    session.add(job)
    session.commit()
    executor = build_visual_composition_executor(
        session,
        fixture.stage50.storage,
        fixture.settings,
        display_probe=fixture.display_probe,
        frame_sampler=FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=FakeDetector(),
    )
    executor.set_active_job(job.id)
    executor.execute(row.id)
    session.refresh(row)
    assert row.plan_ready is True
    return row


def _session() -> Session:
    return create_session_factory()()


def _export_synthetic_package(
    session: Session,
    storage: Any,
    envelope: ClipMaterialization,
    render: RenderExecution,
    artifact: Path,
    *,
    case_name: str = "happy-no-narration",
) -> Path:
    """Persist a SYNTHETIC INTEGRATION evidence package (not product acceptance).

    Each case is written to its own identifiable subdirectory and keeps the actual
    DB UUIDs, fingerprints, the final MP4, the authored ASS, the audio recipe
    (including narration gain), mastering measurements, QC, repair history, job
    dispositions, and a SHA256SUMS manifest, so a reviewer can inspect the real
    production-path result without re-running it.
    """

    import hashlib
    import json
    import shutil

    root = (
        Path(os.environ.get("CLIPFACTORY_BENCHMARK_ROOT", "/var/lib/clipfactory/benchmarks"))
        / "stage-6"
        / "synthetic-integration"
        / case_name
    )
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, root / "final.mp4")
    request = render.request_payload if isinstance(render.request_payload, dict) else {}
    spec = request.get("spec") if isinstance(request.get("spec"), dict) else {}
    authored = spec.get("authored") if isinstance(spec.get("authored"), dict) else {}
    ass = authored.get("authored_ass") if isinstance(authored.get("authored_ass"), dict) else {}
    ass_rel = ass.get("relative_path")
    if isinstance(ass_rel, str) and ass_rel:
        ass_path = storage.storage_root / ass_rel
        if ass_path.is_file():
            shutil.copy2(ass_path, root / "authored.ass")
    jobs = [
        {
            "job_id": str(job.id),
            "kind": job.kind.value,
            "status": job.status.value,
            "error_code": job.error_code,
        }
        for job in session.scalars(
            select(ProcessingJob).where(
                ProcessingJob.materialization_id == envelope.id
            )
        ).all()
    ]
    jobs.extend(
        {
            "job_id": str(job.id),
            "kind": job.kind.value,
            "status": job.status.value,
            "error_code": job.error_code,
        }
        for job in session.scalars(
            select(ProcessingJob).where(ProcessingJob.render_execution_id == render.id)
        ).all()
    )
    index = {
        "kind": "SYNTHETIC_INTEGRATION",
        "case": case_name,
        "note": "Generated test media. Test evidence, not product acceptance.",
        "materialization_id": str(envelope.id),
        "candidate_id": str(envelope.clip_candidate_id),
        "source_video_id": str(envelope.source_video_id),
        "selection_id": str(envelope.selection_id),
        "selected_plan_id": str(envelope.selected_plan_id),
        "refinement_id": str(envelope.refinement_id),
        "render_contract_id": str(envelope.render_contract_id),
        "visual_composition_plan_id": str(envelope.visual_composition_plan_id),
        "input_fingerprint": envelope.input_fingerprint,
        "output_fingerprint": envelope.output_fingerprint,
        "render_execution_id": str(render.id),
        "render_output_fingerprint": render.output_fingerprint,
        "render_qc_fingerprint": render.qc_fingerprint,
        "render_qc_status": render.qc_status.value if render.qc_status else None,
        "semantic_outcome": (
            envelope.semantic_outcome.value if envelope.semantic_outcome else None
        ),
        "lifecycle": envelope.lifecycle.value,
        "publication_ready": bool(envelope.publication_ready),
        "human_review_required": bool(envelope.human_review_required),
        "cache_eligible": bool(envelope.cache_eligible),
        "assessment_policy_version": (envelope.metrics or {}).get("assessment_policy_version"),
        "assessment_fingerprint": (envelope.metrics or {}).get("assessment_fingerprint"),
        "measured_final_integrated_lufs": (envelope.metrics or {}).get("final_integrated_lufs"),
        "measured_final_true_peak_dbtp": (envelope.metrics or {}).get("final_true_peak_dbtp"),
        "semantic_findings": envelope.quality_findings,
        "reason_codes": envelope.reason_codes,
        "repair_count": int(envelope.repair_count or 0),
        "repair_actions": envelope.repair_actions,
        "narration_references": envelope.narration_references,
        "audio_recipe": envelope.audio_recipe,
        "timeline_payload": envelope.timeline_payload,
        "qc_result": render.qc_result,
        "artifact_reference": render.artifact_reference,
        "artifact_sha256": (
            render.artifact_reference.get("sha256")
            if isinstance(render.artifact_reference, dict)
            else None
        ),
        "jobs": jobs,
    }
    (root / "index.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
    lines = []
    for name in ("index.json", "final.mp4", "authored.ass"):
        path = root / name
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {name}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    catalog: dict[str, object] = {}
    for case_dir in sorted(root.parent.iterdir()):
        case_index = case_dir / "index.json"
        if not case_index.is_file():
            continue
        data = json.loads(case_index.read_text(encoding="utf-8"))
        catalog[case_dir.name] = {
            "materialization_id": data.get("materialization_id"),
            "render_execution_id": data.get("render_execution_id"),
            "output_fingerprint": data.get("output_fingerprint"),
            "assessment_fingerprint": data.get("assessment_fingerprint"),
            "semantic_outcome": data.get("semantic_outcome"),
            "repair_count": data.get("repair_count"),
            "narration_count": len(data.get("narration_references") or []),
        }
    (root.parent / "catalog.json").write_text(
        json.dumps(catalog, indent=2, default=str), encoding="utf-8"
    )
    return root


def test_real_persisted_executor_end_to_end(engine: Engine, monkeypatch: Any) -> None:
    settings = get_settings()
    monkeypatch.setattr(queue_module, "_dispatch", lambda *args, **kwargs: None)
    capture = _RenderCapture()

    session = _session()
    try:
        fixture = seed_stage51(session, monkeypatch, narration="NONE", media_writer=_write_source)
        candidate = fixture.stage50.selection.candidate
        candidate_id = candidate.id
        storage = fixture.stage50.storage
        _ready_composition(session, fixture)
        session.commit()

        # 1. POST/queue the materialization (prepare job created + dispatched).
        outcome = queue_materialization(
            session, candidate_id, priority="CANDIDATE", render_dispatcher=capture
        )
        session.commit()
        assert outcome.job_id is not None
        prepare_job_id = outcome.job_id
        materialization_id = outcome.materialization_id
        assert materialization_id is not None
    finally:
        session.close()

    # 2. Prepare executor (fresh session): accepted components + render handoff.
    session = _session()
    try:
        executor = build_materialization_executor(
            session,
            storage,
            settings,
            writing=FakeWritingProvider({1: _ANNOTATION_COPY}),
            critique=FakeCritiqueProvider(accepted={1: True}),
        )
        executor.set_active_job(prepare_job_id)
        executor.set_render_dispatcher(capture)
        executor.execute(str(candidate_id))
        session.commit()
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        assert envelope.lifecycle.value == "AWAITING_RENDER", envelope.reason_codes
        assert envelope.render_execution_id is not None
        assert capture.render_id == str(envelope.render_execution_id)
        render_job_id = uuid.UUID(str(capture.job_id))
        assert render_job_id is not None
        prepare_job = session.get(ProcessingJob, prepare_job_id)
        assert prepare_job is not None and prepare_job.status is JobStatus.SUCCEEDED
    finally:
        session.close()

    # 3. Real RenderExecutionExecutor: FFmpeg compiler/runner/probe/QC.
    session = _session()
    try:
        render_id = envelope.render_execution_id
        renderer = build_render_execution_executor(session, storage, settings)
        renderer.set_active_job(render_job_id)
        renderer.execute(render_id)
        session.commit()
        render = session.get(RenderExecution, render_id)
        assert render is not None
        assert render.lifecycle.value == "COMPLETE", render.reason_codes
        assert render.qc_status is not None
        measured = (render.qc_result or {}).get("measured") or {}
        assert isinstance(measured.get("final_integrated_lufs"), (int, float))
        assert isinstance(measured.get("final_true_peak_dbtp"), (int, float))
        reference = render.artifact_reference or {}
        artifact = storage.storage_root / str(reference["relative_path"])
        assert artifact.is_file() and artifact.stat().st_size > 0
        # The artifact is real, playable H.264/AAC with a decoded duration.
        probe = subprocess.run(  # noqa: S603
            [
                settings.ffprobe_binary,
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name",
                "-of",
                "default=nw=1",
                str(artifact),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "duration=" in probe.stdout
        # Core and transformed artifacts coexist: the transformed render row is
        # stage6, and no core row is demoted by this flow.
        assert render.artifact_purpose.value == "TRANSFORMED_SHORT"
    finally:
        session.close()

    # 4. Completion callback -> finalizer job for the exact revision.
    session = _session()
    try:
        from app.workers.tasks import _request_finalization_after_render

        _request_finalization_after_render(render_id)
    finally:
        session.close()

    session = _session()
    try:
        finalizer_job = session.scalars(
            select(ProcessingJob)
            .where(
                ProcessingJob.materialization_id == materialization_id,
                ProcessingJob.kind == JobKind.MATERIALIZATION,
                ProcessingJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
            )
            .order_by(ProcessingJob.created_at.desc())
        ).first()
        assert finalizer_job is not None
        finalizer_job_id = finalizer_job.id
    finally:
        session.close()

    # 5. Finalizer executor.
    session = _session()
    try:
        finalizer = build_materialization_executor(session, storage, settings)
        finalizer.set_active_job(finalizer_job_id)
        finalizer.finalize(str(candidate_id))
        session.commit()
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        assert envelope.lifecycle.value == "COMPLETE"
        assert envelope.publication_ready is False
        assert envelope.semantic_outcome is not None
        assert envelope.semantic_outcome.value in {
            "READY_FOR_HUMAN_REVIEW",
            "MATERIALIZED_WITH_CAUTION",
        }
        assert envelope.human_review_required is True
        # Every job is terminal.
        jobs = session.scalars(
            select(ProcessingJob).where(
                ProcessingJob.materialization_id == materialization_id
            )
        ).all()
        for job in jobs:
            assert job.status in _TERMINAL
        render_jobs = session.scalars(
            select(ProcessingJob).where(ProcessingJob.render_execution_id == render_id)
        ).all()
        for job in render_jobs:
            assert job.status in _TERMINAL
        first_fingerprint = (envelope.metrics or {}).get("assessment_fingerprint")
        assert first_fingerprint
        _export_synthetic_package(session, storage, envelope, render, artifact)
    finally:
        session.close()

    # 6. Replay: identical assessment, zero provider/render calls, job terminal.
    session = _session()
    try:
        from datetime import datetime, timezone

        from app.materialization.repository import bind_active_job, claim_materialization

        envelope = session.get(ClipMaterialization, materialization_id)
        envelope.active_job_id = None
        replay_job = ProcessingJob(
            source_video_id=envelope.source_video_id,
            kind=JobKind.MATERIALIZATION,
            materialization_id=envelope.id,
            status=JobStatus.RUNNING,
            claim_version=1,
            started_at=datetime.now(timezone.utc),
            heartbeat_at=datetime.now(timezone.utc),
        )
        session.add(replay_job)
        session.commit()
        bind_active_job(session, envelope.id, replay_job.id)
        session.commit()
        owner = claim_materialization(
            session, envelope.id, replay_job.id, expected_job_claim_version=1
        )
        assert owner is not None
        from app.materialization.finalization import finalize_materialization

        result = finalize_materialization(
            session, owner, settings=settings, storage=storage
        )
        session.commit()
        assert result.reused is True
        assert result.assessment_fingerprint == first_fingerprint
        refreshed = session.get(ProcessingJob, replay_job.id)
        assert refreshed is not None and refreshed.status is JobStatus.SUCCEEDED
    finally:
        session.close()
