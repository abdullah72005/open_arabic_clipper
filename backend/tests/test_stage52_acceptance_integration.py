"""Stage 5.2 production-acceptance mechanics against a disposable PostgreSQL.

Gated on ``CLIPFACTORY_TEST_POSTGRES_URL``, real media under the mounted storage
root, and real FFmpeg. It seeds genuine persisted upstream rows (real contract,
real composition plan + canonical ASS, real audio-backed ``FINAL_CLIP``
refinement) and then exercises the production acceptance generator's mechanics:

* ``--no-queue`` with no valid execution is read-only and creates nothing;
* ``--no-queue`` with a valid execution reuses it without dispatch or DB writes;
* a stale/upstream-changed execution is rejected before export;
* a failed regeneration preserves the existing valid package.

The seeded transcript/caption content is synthetic (SEEDED_PLUMBING-like); this
file verifies *acceptance mechanics*, not caption correctness. Genuine product
media is generated separately via the ``production-acceptance`` CLI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import seed_stage50
from test_stage51_live_contract import _select_real_media

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
from app.core.enums import RefinementPriority
from app.core.settings import get_settings
from app.media.ffprobe import FFprobe
from app.models import AudioArtifact, ProcessingJob
from app.models.render_execution import RenderExecution
from app.refinement.audio_window import CandidateAudioWindowService, _run_command
from app.refinement.fingerprints import component_fingerprint
from app.render.execution.acceptance import (
    AcceptanceValidationError,
    build_production_acceptance,
)
from app.render.service import create_render_contract
from app.services.hashing import sha256_file
from app.services.storage import StorageCategory

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL,
    reason="CLIPFACTORY_TEST_POSTGRES_URL is required for acceptance integration",
)

_DATABASE = f"clipfactory_stage52_acc_{uuid.uuid4().hex[:12]}"


def _admin_url() -> str:
    assert _URL is not None
    return str(make_url(_URL).set(database="postgres").render_as_string(hide_password=False))


def _target_url() -> str:
    assert _URL is not None
    return str(make_url(_URL).set(database=_DATABASE).render_as_string(hide_password=False))


def _create_database() -> None:
    admin = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            from sqlalchemy import text

            connection.execute(text(f'DROP DATABASE IF EXISTS "{_DATABASE}" WITH (FORCE)'))
            connection.execute(text(f'CREATE DATABASE "{_DATABASE}"'))
    finally:
        admin.dispose()


def _drop_database() -> None:
    admin = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            from sqlalchemy import text

            connection.execute(text(f'DROP DATABASE IF EXISTS "{_DATABASE}" WITH (FORCE)'))
    finally:
        admin.dispose()


def _alembic_config(url: str) -> Config:
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    return config


@pytest.fixture(autouse=True)  # noqa: PT004
def _no_dispatch(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.composition.queue._dispatch", lambda *args: None)
    monkeypatch.setattr("app.render.execution.queue._dispatch", lambda *args: None)


def _snapshot(session: Session) -> dict[str, object]:
    executions = session.scalar(select(func.count()).select_from(RenderExecution))
    jobs = session.scalar(select(func.count()).select_from(ProcessingJob))
    claims = session.scalars(
        select(RenderExecution.id).where(RenderExecution.active_job_id.is_not(None))
    ).all()
    return {"executions": int(executions or 0), "jobs": int(jobs or 0), "claims": list(claims)}


def _build_fixture(session: Session, media: Path, duration: float, monkeypatch: Any) -> Any:
    settings = get_settings()
    governance = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, governance)
    stage50 = seed_stage50(session, settings=governance, planning_on_final=True)
    candidate = stage50.selection.candidate
    source = stage50.selection.source
    shutil.copyfile(media, stage50.source_path)
    session.flush()

    contract = create_render_contract(
        session,
        candidate.id,
        storage=stage50.storage,
        prober=FFprobe(binary=settings.ffprobe_binary),
        settings=settings,
    )
    session.commit()
    assert contract is not None and contract.row.contract_ready is True

    outcome = queue_visual_composition(session, candidate, settings=settings)
    assert outcome.queued is True and outcome.job_id is not None
    plan_row = get_current_visual_composition(session, candidate.id)
    assert plan_row is not None

    probe = FFprobeDisplayProbe(binary=settings.ffprobe_binary)
    config51 = settings.stage51_config()
    exec_inputs = resolve_planner_inputs(
        session, candidate.id, display_probe=probe, config=config51
    )
    assert exec_inputs is not None
    media_path = Path(str(stage50.storage.storage_root)) / exec_inputs.source_media_relative_path
    frame_size = scaled_dimensions(
        int(exec_inputs.display_geometry.display_width),
        int(exec_inputs.display_geometry.display_height),
        config51.analysis_frame_max_dimension,
    )
    executor = build_visual_composition_executor(
        session,
        stage50.storage,
        settings,
        display_probe=probe,
        frame_sampler=FFmpegFrameSampler(
            media_path,
            frame_size=frame_size,
            ffmpeg_binary=settings.ffmpeg_binary,
            storage=stage50.storage,
        ),
        scene_cut_detector=FFmpegSceneCutDetector(ffmpeg_binary=settings.ffmpeg_binary),
        detector=detector_from_config(config51),
    )
    executor.set_active_job(outcome.job_id)
    executor.execute(plan_row.id)
    session.refresh(plan_row)
    assert plan_row.status.value == "READY_FOR_VISUAL_EXECUTION"

    _upgrade_refinement_to_audio_backed(
        session, stage50, candidate, source, media_path, duration, settings
    )
    session.commit()
    return stage50


def _upgrade_refinement_to_audio_backed(
    session: Session,
    stage50: Any,
    candidate: Any,
    source: Any,
    media_path: Path,
    duration: float,
    settings: Any,
) -> None:
    artifact_relative = f"{source.id}/speech-analysis.wav"
    artifact_path = stage50.storage.resolve(StorageCategory.SOURCES, artifact_relative)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            settings.ffmpeg_binary,
            "-y",
            "-nostdin",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(artifact_path),
        ],
        check=True,
        capture_output=True,
    )
    session.add(
        AudioArtifact(
            source_video_id=source.id,
            output_path=artifact_relative,
            content_hash=sha256_file(artifact_path),
            source_content_hash=source.content_hash,
            sample_rate=16_000,
            duration=duration,
        )
    )
    session.flush()

    service = CandidateAudioWindowService(
        storage=stage50.storage,
        session=session,
        config=settings.stage35_config(),
        command_runner=_run_command,
    )
    window = service.extract(
        source=source, candidate=candidate, priority=RefinementPriority.FINAL_CLIP
    )
    refinement = stage50.selection.refinement
    refinement.audio_relative_path = window.relative_path
    refinement.audio_content_hash = window.content_hash
    refinement.audio_input_fingerprint = window.input_fingerprint
    refinement.component_fingerprints = {
        "audio_extraction": component_fingerprint(
            "audio-extraction",
            {
                "audio_input_fingerprint": window.input_fingerprint,
                "content_hash": window.content_hash,
            },
        ),
        "local_asr": component_fingerprint("local-asr", {"seed": "acceptance-integration"}),
    }
    session.flush()


def test_production_acceptance_mechanics(tmp_path: Path, monkeypatch: Any) -> None:
    assert _URL is not None
    target_url = _target_url()
    _create_database()
    session: Session | None = None
    engine: Engine | None = None
    output = tmp_path / "manual-acceptance"
    try:
        monkeypatch.setenv("CLIPFACTORY_DATABASE_URL", target_url)
        get_settings.cache_clear()
        settings = get_settings()
        media, media_meta = _select_real_media(settings.ffprobe_binary)
        duration = float(media_meta.get("duration_seconds") or 90.0)

        command.upgrade(_alembic_config(target_url), "head")
        engine = create_engine(target_url)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        session = factory()
        stage50 = _build_fixture(session, media, duration, monkeypatch)
        candidate_id = str(stage50.selection.candidate.id)

        # 1. --no-queue with no valid execution is read-only and creates nothing.
        before = _snapshot(session)
        with pytest.raises(AcceptanceValidationError) as error:
            build_production_acceptance(
                candidate_id,
                output,
                settings=settings,
                session_factory=lambda: Session(engine),
                queue=False,
            )
        assert error.value.reason_code == "NO_CACHE_ELIGIBLE_EXECUTION"
        assert _snapshot(session) == before
        assert not (output / "manifest.json").exists()

        # 2. Normal generation runs through the production queue/executor.
        manifest = build_production_acceptance(
            candidate_id,
            output,
            settings=settings,
            session_factory=lambda: Session(engine),
        )
        assert manifest["classification"] == "PRODUCTION_MANUAL_ACCEPTANCE"
        assert len(manifest["cases"]) == 1
        case = manifest["cases"][0]
        output_sha = case["output_sha256"]
        assert Path(case["path"]).is_file()
        assert (output / "manifest.json").is_file()

        # 3. --no-queue valid hit: no dispatch, no DB writes, stable package.
        hit_before = _snapshot(session)
        hit_manifest = build_production_acceptance(
            candidate_id,
            output,
            settings=settings,
            session_factory=lambda: Session(engine),
            queue=False,
        )
        assert _snapshot(session) == hit_before
        assert len(hit_manifest["cases"]) == 1
        assert hit_manifest["cases"][0]["output_sha256"] == output_sha

        # 4. Stale/upstream-changed execution is rejected before export.
        original_threads = settings.render_encoder_threads
        monkeypatch.setattr(settings, "render_encoder_threads", original_threads + 1)
        with pytest.raises(AcceptanceValidationError) as stale_error:
            build_production_acceptance(
                candidate_id,
                output,
                settings=settings,
                session_factory=lambda: Session(engine),
                queue=False,
            )
        assert stale_error.value.reason_code == "NO_CACHE_ELIGIBLE_EXECUTION"
        monkeypatch.setattr(settings, "render_encoder_threads", original_threads)
        assert (
            json.loads((output / "manifest.json").read_text())["cases"][0]["output_sha256"]
            == output_sha
        )

        # 5. A failed regeneration preserves the existing valid package.
        managed = Path(str(stage50.storage.storage_root)) / case["managed_artifact_relative_path"]
        original_bytes = managed.read_bytes()
        managed.write_bytes(b"corrupted")
        with pytest.raises(AcceptanceValidationError):
            build_production_acceptance(
                candidate_id,
                output,
                settings=settings,
                session_factory=lambda: Session(engine),
                queue=False,
            )
        managed.write_bytes(original_bytes)
        assert Path(case["path"]).is_file()
        assert sha256_file(Path(case["path"])) == output_sha
        assert (
            json.loads((output / "manifest.json").read_text())["cases"][0]["output_sha256"]
            == output_sha
        )
        assert not list(output.glob(".staging-*"))

        # 6. A failure at the manifest publication boundary during a valid
        # regeneration must leave the previous package (media + manifest) intact.
        import app.render.execution.acceptance as acceptance

        real_write_json = acceptance._write_json

        def flaky_manifest_write(path: Path, payload: object) -> None:
            if path.name == "manifest.json":
                raise RuntimeError("simulated manifest publication failure")
            real_write_json(path, payload)

        monkeypatch.setattr(acceptance, "_write_json", flaky_manifest_write)
        with pytest.raises(RuntimeError):
            build_production_acceptance(
                candidate_id,
                output,
                settings=settings,
                session_factory=lambda: Session(engine),
                queue=False,
            )
        monkeypatch.setattr(acceptance, "_write_json", real_write_json)
        assert Path(case["path"]).is_file()
        assert sha256_file(Path(case["path"])) == output_sha
        assert (
            json.loads((output / "manifest.json").read_text())["cases"][0]["output_sha256"]
            == output_sha
        )
        assert not list(output.glob(".staging-*"))
        assert not list(output.glob("*.previous-*"))
    finally:
        if session is not None:
            session.close()
        if engine is not None:
            engine.dispose()
        get_settings.cache_clear()
        _drop_database()
