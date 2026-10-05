"""Stage 5.2 live-contract end-to-end test against a disposable PostgreSQL.

The only Stage 5.2 test that exercises the *real* seams end to end:

    alembic upgrade head
      -> seeded retained candidate + FINAL_CLIP refinement (real rows)
      -> real ``create_render_contract`` with the real FFprobe on real media
      -> real visual-composition plan with the canonical ASS asset
      -> ``queue_render_execution`` (one render row + one RENDER_EXECUTION job)
      -> real ``RenderExecutionExecutor`` with the real compiler/FFmpeg/QC seams
      -> persisted managed MP4 artifact + technical QC

It is gated on ``CLIPFACTORY_TEST_POSTGRES_URL`` (plus real media under the
mounted storage root) and skips otherwise rather than faking the run. Artifacts
are written to ``CLIPFACTORY_LIVE_STAGE52_ARTIFACTS`` (default
``/var/lib/clipfactory/benchmarks/stage-5-2/live-contract``).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import seed_stage50
from test_stage51_live_contract import (
    _alembic_config,
    _reset_public_schema,
    _select_real_media,
)

from alembic import command
from app.composition.analysis import (
    FFmpegFrameSampler,
    FFmpegSceneCutDetector,
    scaled_dimensions,
)
from app.composition.detector import detector_from_config
from app.composition.executor import build_visual_composition_executor
from app.composition.geometry import FFprobeDisplayProbe
from app.composition.queue import queue_visual_composition
from app.composition.service import get_current_visual_composition, resolve_planner_inputs
from app.core.enums import JobStatus, RenderExecutionLifecycle
from app.core.settings import get_settings
from app.media.ffprobe import FFprobe
from app.models import ProcessingJob
from app.models.render_execution import RenderExecution
from app.render.execution.executor import build_render_execution_executor
from app.render.execution.queue import queue_render_execution
from app.render.service import create_render_contract

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL,
    reason="CLIPFACTORY_TEST_POSTGRES_URL is required for the Stage 5.2 live run",
)


def _artifacts_dir() -> Path:
    configured = os.environ.get("CLIPFACTORY_LIVE_STAGE52_ARTIFACTS")
    if configured:
        return Path(configured)
    mounted = Path("/var/lib/clipfactory/benchmarks/stage-5-2/live-contract")
    if mounted.parent.parent.exists():
        return mounted
    return (
        Path(__file__).resolve().parents[2]
        / "storage"
        / "benchmarks"
        / "stage-5-2"
        / "live-contract"
    )


def _write_report(report: dict[str, object], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "report.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return target


@pytest.fixture(autouse=True)  # noqa: PT004 - name mirrors Stage 5.1 live test
def _no_dispatch(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.composition.queue._dispatch", lambda *args: None)
    monkeypatch.setattr("app.render.execution.queue._dispatch", lambda *args: None)


def test_stage52_live_contract_end_to_end(tmp_path: Path, monkeypatch: Any) -> None:
    assert _URL is not None
    report: dict[str, object] = {"assertions": {}, "artifacts": {}, "media": None}
    artifacts = _artifacts_dir()
    session: Session | None = None
    engine: Engine | None = None
    failure: str | None = None
    try:
        settings = get_settings()
        media, media_meta = _select_real_media(settings.ffprobe_binary)
        report["media"] = media_meta

        config: Config = _alembic_config()
        engine = create_engine(_URL)
        _reset_public_schema(engine)
        command.upgrade(config, "head")
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert "render_executions" in tables
        assert "visual_composition_plans" in tables

        factory = sessionmaker(bind=engine, expire_on_commit=False)
        session = factory()

        governance = FakeGovernanceSettings()
        install_selection_settings(monkeypatch, governance)
        stage50 = seed_stage50(session, settings=governance, planning_on_final=True)
        candidate = stage50.selection.candidate
        shutil.copyfile(media, stage50.source_path)
        session.flush()

        real_prober = FFprobe(binary=settings.ffprobe_binary)
        contract = create_render_contract(
            session,
            candidate.id,
            storage=stage50.storage,
            prober=real_prober,
            settings=settings,
        )
        session.commit()
        assert contract is not None and contract.row.contract_ready is True

        config51 = settings.stage51_config()
        outcome = queue_visual_composition(session, candidate, settings=settings)
        assert outcome.queued is True and outcome.job_id is not None
        plan_row = get_current_visual_composition(session, candidate.id)
        assert plan_row is not None

        probe = FFprobeDisplayProbe(binary=settings.ffprobe_binary)
        exec_inputs = resolve_planner_inputs(
            session, candidate.id, display_probe=probe, config=config51
        )
        assert exec_inputs is not None
        media_path = (
            Path(str(stage50.storage.storage_root)) / exec_inputs.source_media_relative_path
        )
        frame_size = scaled_dimensions(
            int(exec_inputs.display_geometry.display_width),
            int(exec_inputs.display_geometry.display_height),
            config51.analysis_frame_max_dimension,
        )
        frame_sampler = FFmpegFrameSampler(
            media_path,
            frame_size=frame_size,
            ffmpeg_binary=settings.ffmpeg_binary,
            storage=stage50.storage,
        )
        detector = detector_from_config(config51)
        assert detector.ready() is True
        composition_executor = build_visual_composition_executor(
            session,
            stage50.storage,
            settings,
            display_probe=probe,
            frame_sampler=frame_sampler,
            scene_cut_detector=FFmpegSceneCutDetector(ffmpeg_binary=settings.ffmpeg_binary),
            detector=detector,
        )
        composition_executor.set_active_job(outcome.job_id)
        composition_executor.execute(plan_row.id)
        session.refresh(plan_row)
        assert plan_row.status.value == "READY_FOR_VISUAL_EXECUTION"

        # ----- Real render execution through the actual task entry point ------
        render_outcome = queue_render_execution(session, candidate.id, settings=settings)
        assert render_outcome.queued is True and render_outcome.job_id is not None
        executor = build_render_execution_executor(session, stage50.storage, settings)
        executor.set_active_job(render_outcome.job_id)
        executor.execute(render_outcome.render_execution_id)

        session.expire_all()
        row = session.get(RenderExecution, render_outcome.render_execution_id)
        job = session.get(ProcessingJob, render_outcome.job_id)
        assert row is not None and job is not None
        report["assertions"]["render_execution"] = {
            "job_status": job.status.value,
            "lifecycle": row.lifecycle.value,
            "qc_status": row.qc_status.value if row.qc_status else None,
            "cache_eligible": bool(row.cache_eligible),
            "publication_ready": bool(row.publication_ready),
            "stage6_implemented": bool(row.stage6_implemented),
            "artifact": dict(row.artifact_reference or {}),
            "qc_reason_codes": list(row.reason_codes or []),
        }
        assert job.status is JobStatus.SUCCEEDED
        assert row.lifecycle is RenderExecutionLifecycle.COMPLETE
        assert row.cache_eligible is True
        assert row.publication_ready is False
        assert row.stage6_implemented is False
        assert row.qc_status is not None and row.qc_status.value in {"PASS", "WARN"}

        relative = row.artifact_reference.get("relative_path")
        assert isinstance(relative, str) and relative
        artifact_path = Path(str(stage50.storage.storage_root)) / relative
        assert artifact_path.is_file() and artifact_path.stat().st_size > 0
        probe_facts = (row.execution_manifest or {}).get("output") or {}
        report["assertions"]["artifact"] = {
            "relative_path": relative,
            "size_bytes": probe_facts.get("size_bytes"),
            "duration_seconds": probe_facts.get("duration_seconds"),
            "frame_count": probe_facts.get("frame_count"),
        }
        assert int(probe_facts.get("size_bytes") or 0) > 0

        measured = (row.qc_result or {}).get("measured") or {}
        report["assertions"]["qc_measured"] = measured
        assert measured.get("video_duration", 0) > 0
        assert measured.get("audio_duration", 0) > 0
        assert "QC_TOTAL_SILENCE" not in (row.reason_codes or [])

        cached = queue_render_execution(session, candidate.id, settings=settings)
        report["assertions"]["cache_hit"] = {
            "cached": bool(cached.cached),
            "queued": bool(cached.queued),
        }
        assert cached.cached is True and cached.queued is False

        artifacts.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifact_path, artifacts / "live-source-validation.mp4")
        report["artifacts"]["video"] = str(artifacts / "live-source-validation.mp4")
    except BaseException as error:  # noqa: BLE001 - recorded then re-raised
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["failure"] = failure
        report["artifacts"]["report"] = str(_write_report(report, artifacts))
        print("\nSTAGE52_LIVE_CONTRACT_REPORT=" + json.dumps(report, default=str))
        if session is not None:
            session.close()
        if engine is not None:
            engine.dispose()
