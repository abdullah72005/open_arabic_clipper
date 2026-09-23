"""Stage 5.1 live-contract end-to-end test against a disposable PostgreSQL.

This is the only Stage 5.1 test that exercises the *real* seams end to end:

    alembic upgrade head
      -> seeded retained candidate + FINAL_CLIP refinement (real rows)
      -> real ``create_render_contract`` with the real FFprobe on real media
      -> ``queue_visual_composition`` (one plan row + one VISUAL_COMPOSITION job)
      -> ``VisualCompositionExecutor.execute`` with the real FFprobe/FFmpeg/YuNet
         seams (``FFprobeDisplayProbe``, ``FFmpegFrameSampler``,
         ``FFmpegSceneCutDetector``, ``detector_from_config``)
      -> ``build_stage5_2_handoff``
      -> real ``render_preview_pngs`` (1080x1920 PNGs, faithful plan crop,
         burned-in caption ink).

It is gated on ``CLIPFACTORY_TEST_POSTGRES_URL`` and skips otherwise, mirroring
``test_stage51_postgres.py``. Real media is discovered under the mounted storage
root; when no suitable clip is present the test skips with a clear reason rather
than faking the run.

Set ``CLIPFACTORY_LIVE_STAGE51_MEDIA`` to force one file, or
``CLIPFACTORY_LIVE_STAGE51_MEDIA_ROOT`` to point at a ``sources`` directory.
Artifacts (JSON report + preview PNGs) are written to
``CLIPFACTORY_LIVE_STAGE51_ARTIFACTS`` (default
``/var/lib/clipfactory/benchmarks/stage-5-1/live-contract``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import seed_stage50

from alembic import command
from app.composition.analysis import (
    FFmpegFrameSampler,
    FFmpegSceneCutDetector,
    scaled_dimensions,
)
from app.composition.detector import detector_from_config
from app.composition.executor import build_visual_composition_executor
from app.composition.geometry import FFprobeDisplayProbe
from app.composition.handoff import build_stage5_2_handoff
from app.composition.policy import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    FramingMode,
    VisualCompositionExecutionStatus,
    VisualCompositionStatus,
)
from app.composition.preview import (
    _frame_arguments,
    preview_filtergraph,
    render_preview_pngs,
    resolve_preview_crop,
)
from app.composition.queue import queue_visual_composition
from app.composition.service import (
    get_current_visual_composition,
    read_visual_composition,
    resolve_planner_inputs,
)
from app.core.enums import JobKind, JobStatus
from app.core.settings import get_settings
from app.media.ffprobe import FFprobe
from app.models import ProcessingJob
from app.models.visual_composition_plan import VisualCompositionPlan
from app.render.policy import EXECUTABLE_STATUSES
from app.render.service import create_render_contract

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL,
    reason="CLIPFACTORY_TEST_POSTGRES_URL is required for PostgreSQL Stage 5.1 tests",
)

_CROP_MODES = frozenset(
    {
        FramingMode.STATIC_CROP.value,
        FramingMode.TRACKED_CROP.value,
        FramingMode.MULTI_SUBJECT_FIT.value,
        FramingMode.CENTER_FALLBACK.value,
    }
)
_MEDIA_SUFFIXES = (".webm", ".mp4", ".mkv")
_MIN_MEDIA_DURATION_SECONDS = 60.0


class _RaisingProbe:
    """Fail loudly if a read-time freshness path ever probes display geometry."""

    def __init__(self) -> None:
        self.calls = 0

    def probe(self, path: Path) -> Any:
        self.calls += 1
        raise AssertionError("read-time freshness must never re-probe media")


# --------------------------------------------------------------------------- #
# Disposable-PostgreSQL lifecycle helpers (mirror test_stage51_postgres.py)
# --------------------------------------------------------------------------- #


def _alembic_config() -> Config:
    if _URL:
        os.environ["CLIPFACTORY_DATABASE_URL"] = _URL
        get_settings.cache_clear()
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    config.set_main_option("sqlalchemy.url", _URL or "")
    return config


def _reset_public_schema(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execution_options(isolation_level="AUTOCOMMIT")
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))


# --------------------------------------------------------------------------- #
# Real media discovery
# --------------------------------------------------------------------------- #


def _media_roots() -> list[Path]:
    single = os.environ.get("CLIPFACTORY_LIVE_STAGE51_MEDIA")
    if single:
        path = Path(single)
        if path.is_file():
            return [path]
    roots: list[Path] = []
    configured = os.environ.get("CLIPFACTORY_LIVE_STAGE51_MEDIA_ROOT")
    if configured:
        roots.append(Path(configured))
    roots.extend(
        [
            Path("/var/lib/clipfactory/sources"),
            Path(__file__).resolve().parents[2] / "storage" / "sources",
        ]
    )
    return roots


def _media_candidates() -> list[Path]:
    files: list[Path] = []
    for root in _media_roots():
        if root.is_file():
            files.append(root)
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in _MEDIA_SUFFIXES:
                files.append(path)
    return files


def _select_real_media(ffprobe_binary: str) -> tuple[Path, dict[str, object]]:
    prober = FFprobe(binary=ffprobe_binary)
    for path in sorted(_media_candidates(), key=lambda item: item.stat().st_size):
        try:
            metadata = prober.probe(path)
        except Exception:  # noqa: BLE001 - candidate discovery, not a product path
            continue
        if (
            metadata.duration_seconds >= _MIN_MEDIA_DURATION_SECONDS
            and metadata.width > 0
            and metadata.height > 0
            and metadata.frames_per_second > 0
            and metadata.audio_codec
        ):
            return path, {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "duration_seconds": metadata.duration_seconds,
                "width": metadata.width,
                "height": metadata.height,
                "frames_per_second": metadata.frames_per_second,
                "video_codec": metadata.video_codec,
                "audio_codec": metadata.audio_codec,
            }
    pytest.skip("no suitable real media (>=60s, video+audio) found for the Stage 5.1 live run")


# --------------------------------------------------------------------------- #
# Artifact helpers
# --------------------------------------------------------------------------- #


def _artifacts_dir() -> Path:
    configured = os.environ.get("CLIPFACTORY_LIVE_STAGE51_ARTIFACTS")
    if configured:
        return Path(configured)
    mounted = Path("/var/lib/clipfactory/benchmarks/stage-5-1/live-contract")
    if mounted.parent.parent.exists():
        return mounted
    return (
        Path(__file__).resolve().parents[2]
        / "storage"
        / "benchmarks"
        / "stage-5-1"
        / "live-contract"
    )


def _write_report(report: dict[str, object], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "report.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return target


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width, height


def _decode_rgb(ffmpeg_binary: str, path: Path) -> bytes:
    result = subprocess.run(
        [
            ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _pixel_diff(first: Path, second: Path, ffmpeg_binary: str) -> int:
    left = _decode_rgb(ffmpeg_binary, first)
    right = _decode_rgb(ffmpeg_binary, second)
    assert len(left) == len(right), "preview frames have different raw sizes"
    return sum(1 for a, b in zip(left, right) if a != b)


def _render_frame(
    ffmpeg_binary: str,
    source_path: Path,
    source_time: float,
    output_path: Path,
    filtergraph: str,
    cwd: Path,
) -> None:
    executable = shutil.which(ffmpeg_binary) or ffmpeg_binary
    arguments = _frame_arguments(
        executable=executable,
        source_path=source_path,
        source_time=source_time,
        output_path=output_path,
        filtergraph=filtergraph,
    )
    subprocess.run(arguments, check=True, capture_output=True, text=True, cwd=str(cwd))


def _scene_start_times(plan_payload: dict[str, object]) -> list[float]:
    scenes = plan_payload.get("scenes")
    if not isinstance(scenes, Sequence) or isinstance(scenes, (str, bytes)):
        return []
    times: list[float] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        start = scene.get("source_start")
        if isinstance(start, (int, float)) and not isinstance(start, bool):
            times.append(float(start))
    deduped: list[float] = []
    for value in sorted(times):
        if not deduped or abs(value - deduped[-1]) > 1e-4:
            deduped.append(value)
    return deduped


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: Any) -> None:
    """Never touch Celery/Redis: the executor is driven synchronously in-process."""

    monkeypatch.setattr("app.composition.queue._dispatch", lambda *args: None)


# --------------------------------------------------------------------------- #
# The live end-to-end test
# --------------------------------------------------------------------------- #


def test_stage51_live_contract_end_to_end(tmp_path: Path, monkeypatch: Any) -> None:
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

        # ----- Assertion 1: alembic upgrade head creates the Stage 5.x schema --
        config = _alembic_config()
        engine = create_engine(_URL)
        _reset_public_schema(engine)
        command.upgrade(config, "head")
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        job_columns = {column["name"] for column in inspector.get_columns("processing_jobs")}
        report["assertions"]["1_migration"] = {
            "render_contracts": "render_contracts" in tables,
            "visual_composition_plans": "visual_composition_plans" in tables,
            "processing_jobs.visual_composition_plan_id": (
                "visual_composition_plan_id" in job_columns
            ),
            "alembic_version": inspector.has_table("alembic_version"),
        }
        assert "render_contracts" in tables
        assert "visual_composition_plans" in tables
        assert "visual_composition_plan_id" in job_columns

        factory = sessionmaker(bind=engine, expire_on_commit=False)
        session = factory()

        # ----- Assertion 2: seeded FINAL_CLIP candidate + real contract -------
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
        assert contract is not None
        report["assertions"]["2_render_contract"] = {
            "contract_ready": bool(contract.row.contract_ready),
            "status": contract.row.status.value,
            "executable": contract.row.status.value in EXECUTABLE_STATUSES,
            "live_freshness": contract.live_freshness,
            "effective": bool(contract.effective),
            "probe_reused": bool(contract.row.source_probe.get("probe_reused")),
            "managed_relative_path": contract.row.contract_payload.get("source_media", {}).get(
                "managed_relative_path"
            ),
        }
        assert contract.row.contract_ready is True
        assert contract.row.status.value in EXECUTABLE_STATUSES
        assert contract.live_freshness == "CURRENT"
        assert contract.effective is True
        assert contract.row.contract_payload

        config51 = settings.stage51_config()

        # ----- Assertion 3: queue + real executor -----------------------------
        outcome = queue_visual_composition(session, candidate, settings=settings)
        assert outcome.queued is True
        assert outcome.job_id is not None
        queue_jobs = (
            session.query(ProcessingJob)
            .filter(ProcessingJob.kind == JobKind.VISUAL_COMPOSITION)
            .all()
        )
        plan_rows = (
            session.query(VisualCompositionPlan)
            .filter(VisualCompositionPlan.clip_candidate_id == candidate.id)
            .all()
        )
        assert len(queue_jobs) == 1
        assert len(plan_rows) == 1
        row = get_current_visual_composition(session, candidate.id)
        assert row is not None and row.id == outcome.plan_id

        detector = detector_from_config(config51)
        detector_ready = bool(detector.ready())
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
        scene_cut_detector = FFmpegSceneCutDetector(ffmpeg_binary=settings.ffmpeg_binary)
        executor = build_visual_composition_executor(
            session,
            stage50.storage,
            settings,
            display_probe=probe,
            frame_sampler=frame_sampler,
            scene_cut_detector=scene_cut_detector,
            detector=detector,
        )
        executor.set_active_job(outcome.job_id)
        result = executor.execute(row.id)
        assert result is not None
        session.refresh(row)
        job = session.get(ProcessingJob, outcome.job_id)
        assert job is not None
        session.refresh(job)
        report["assertions"]["3_executor"] = {
            "detector_ready": detector_ready,
            "seams": {
                "display_probe": type(probe).__name__,
                "frame_sampler": type(frame_sampler).__name__,
                "scene_cut_detector": type(scene_cut_detector).__name__,
                "detector": type(detector).__name__,
            },
            "job_status": job.status.value,
            "plan_status": row.status.value,
            "execution_status": row.execution_status.value,
            "is_current": bool(row.is_current),
            "plan_ready": bool(row.plan_ready),
            "cache_eligible": bool(row.cache_eligible),
            "has_input_fingerprint": bool(row.input_fingerprint),
            "has_output_fingerprint": bool(row.output_fingerprint),
            "has_ass_fingerprint": bool(row.ass_fingerprint),
            "scene_count": (row.metrics or {}).get("scene_count"),
            "caption_events": (row.metrics or {}).get("caption_events"),
            "sampled_frames": (row.metrics or {}).get("sampled_frames"),
            "face_detections": (row.metrics or {}).get("face_detections"),
        }
        assert job.status is JobStatus.SUCCEEDED
        assert row.status is VisualCompositionStatus.READY_FOR_VISUAL_EXECUTION
        assert row.execution_status is VisualCompositionExecutionStatus.COMPLETE
        assert row.is_current is True
        assert row.plan_ready is True
        assert row.input_fingerprint
        assert row.output_fingerprint
        assert row.ass_fingerprint
        assert detector_ready is True, "the real YuNet detector must be available"

        # ----- Assertion 4: read is CURRENT/effective without probing ---------
        no_probe = _RaisingProbe()
        view = read_visual_composition(
            session, candidate.id, display_probe=no_probe, config=config51
        )
        assert view is not None
        report["assertions"]["4_read_effective"] = {
            "live_freshness": view.live_freshness,
            "effective": bool(view.effective),
            "probe_calls": no_probe.calls,
        }
        assert view.live_freshness == "CURRENT"
        assert view.effective is True
        assert no_probe.calls == 0

        # ----- Assertion 5: Stage 5.2 handoff ---------------------------------
        display_probe = FFprobeDisplayProbe(binary=settings.ffprobe_binary)
        handoff = build_stage5_2_handoff(
            session, candidate.id, display_probe=display_probe, config=config51
        )
        assert handoff is not None
        report["assertions"]["5_handoff"] = {
            "stage5_2_implemented": handoff.get("stage5_2_implemented"),
            "stage6_implemented": handoff.get("stage6_implemented"),
            "final_timeline_frozen": handoff.get("final_timeline_frozen"),
            "readiness": handoff.get("readiness"),
            "scene_count": len(handoff.get("scenes") or []),
            "caption_event_count": len((handoff.get("captions") or {}).get("events") or []),
            "ass": handoff.get("ass"),
            "overlay_count": len(handoff.get("overlays") or []),
            "materialization": handoff.get("materialization"),
            "block_count": len(handoff.get("blocks") or []),
            "bound_source_spans": len(handoff.get("bound_source_spans") or []),
        }
        assert handoff["stage5_2_implemented"] is False
        assert handoff["stage6_implemented"] is False
        assert handoff["final_timeline_frozen"] is False
        readiness = handoff["readiness"]
        assert isinstance(readiness, dict)
        assert readiness.get("stage5_2_handoff_eligible") is True
        assert handoff["scenes"]
        assert handoff["captions"]
        assert (handoff["captions"] or {}).get("events")
        assert handoff["ass"] and handoff["ass"].get("asset_path")
        assert handoff["blocks"]
        assert handoff["bound_source_spans"]

        # ----- Assertion 6: faithful preview PNGs -----------------------------
        previews_dir = artifacts / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        plan_payload = dict(row.plan_payload)
        inputs = resolve_planner_inputs(
            session, candidate.id, display_probe=display_probe, config=config51
        )
        assert inputs is not None
        media_path = Path(str(stage50.storage.storage_root)) / inputs.source_media_relative_path
        outputs = render_preview_pngs(
            source_path=media_path,
            inputs=inputs,
            plan_payload=plan_payload,
            config=config51,
            storage=stage50.storage,
            output_directory=previews_dir,
            ffmpeg_binary=settings.ffmpeg_binary,
        )
        assert outputs
        sizes = {str(path.name): list(_png_size(path)) for path in outputs}
        for width, height in (_png_size(path) for path in outputs):
            assert (width, height) == (OUTPUT_WIDTH, OUTPUT_HEIGHT)

        times = _scene_start_times(plan_payload)[: config51.preview_max_frames]
        assert len(outputs) == len(times)
        scene_by_start = {
            float(scene["source_start"]): scene for scene in plan_payload.get("scenes") or []
        }
        crop_checks: list[dict[str, object]] = []
        preview_ink: list[int] = []
        for index, source_time in enumerate(times):
            crop = resolve_preview_crop(plan_payload, source_time)
            graph = preview_filtergraph(plan_payload, source_time)
            scene = scene_by_start.get(source_time)
            assert scene is not None
            keyframes = sorted(
                (frame for frame in scene.get("crop_keyframes") or []),
                key=lambda frame: float(frame.get("t", 0.0)),
            )
            assert keyframes, "the real plan must carry crop keyframes"
            lead = keyframes[0]
            # The preview composition is the plan's composition at that time.
            assert crop.center_x == pytest.approx(float(lead["cx"]), abs=1e-6)
            assert crop.center_y == pytest.approx(float(lead["cy"]), abs=1e-6)
            assert crop.height_fraction == pytest.approx(float(lead["height_fraction"]), abs=1e-6)
            if crop.mode in _CROP_MODES:
                assert (
                    f"crop={crop.crop_width}:{crop.crop_height}:{crop.crop_x}:{crop.crop_y}"
                    in graph
                )
            # Ink actually burned into this preview frame, measured against the
            # identical plan composition rendered without the ASS asset.
            reference = previews_dir / f"noref-{index:04d}.png"
            _render_frame(
                settings.ffmpeg_binary,
                media_path,
                source_time,
                reference,
                preview_filtergraph(plan_payload, source_time, ass_filename=None),
                previews_dir,
            )
            preview_ink.append(_pixel_diff(outputs[index], reference, settings.ffmpeg_binary))
            crop_checks.append(
                {
                    "source_time": source_time,
                    "mode": crop.mode,
                    "crop": [crop.crop_x, crop.crop_y, crop.crop_width, crop.crop_height],
                    "caption_ink_pixels": preview_ink[-1],
                }
            )

        # Caption ink: at least one real preview PNG must contain burned-in ink.
        assert max(preview_ink) > 0, "at least one preview PNG must contain caption ink"

        # Guarantee the ink assertion is independent of scene-start timing by also
        # rendering the plan composition at a mid-event time with and without ASS.
        events = (plan_payload.get("captions") or {}).get("events") or []
        chosen = None
        for event in events:
            start = float(event["start"])
            end = float(event["end"])
            if end - start > 0.05:
                chosen = (start + min(0.15, (end - start) / 2.0), event)
                break
        assert chosen is not None, "the real plan must emit at least one caption event"
        ink_time, ink_event = chosen
        ass_asset = previews_dir / "captions.ass"
        assert ass_asset.is_file(), "render_preview_pngs must localize the ASS asset"
        with_graph = preview_filtergraph(plan_payload, ink_time, ass_filename=ass_asset.name)
        without_graph = preview_filtergraph(plan_payload, ink_time, ass_filename=None)
        with_png = previews_dir / "captions-with-ass.png"
        without_png = previews_dir / "captions-without-ass.png"
        _render_frame(
            settings.ffmpeg_binary, media_path, ink_time, with_png, with_graph, previews_dir
        )
        _render_frame(
            settings.ffmpeg_binary, media_path, ink_time, without_png, without_graph, previews_dir
        )
        assert _png_size(with_png) == (OUTPUT_WIDTH, OUTPUT_HEIGHT)
        ink_pixels = _pixel_diff(with_png, without_png, settings.ffmpeg_binary)
        assert ink_pixels > 0
        ink_crop = resolve_preview_crop(plan_payload, ink_time)

        report["assertions"]["6_preview"] = {
            "frame_count": len(outputs),
            "sizes": sizes,
            "crop_checks": crop_checks,
            "preview_caption_ink_pixels": preview_ink,
            "ink_time": ink_time,
            "ink_event_id": ink_event.get("event_id"),
            "ink_event_text": ink_event.get("text"),
            "ink_crop": ink_crop.as_dict(),
            "ink_pixel_difference": ink_pixels,
        }
        report["artifacts"]["preview_pngs"] = [str(path) for path in outputs]
        report["artifacts"]["caption_ink_pngs"] = [str(with_png), str(without_png)]

        # ----- Assertion 7: re-queue without force is a cache hit -------------
        cached = queue_visual_composition(session, candidate, settings=settings)
        jobs_after_cache = (
            session.query(ProcessingJob)
            .filter(ProcessingJob.kind == JobKind.VISUAL_COMPOSITION)
            .count()
        )
        report["assertions"]["7_cache_hit"] = {
            "cached": bool(cached.cached),
            "queued": bool(cached.queued),
            "job_id": str(cached.job_id) if cached.job_id else None,
            "plan_id_matches": str(cached.plan_id) == str(row.id),
            "visual_composition_job_count": jobs_after_cache,
        }
        assert cached.cached is True
        assert cached.queued is False
        assert cached.job_id is None
        assert str(cached.plan_id) == str(row.id)
        assert jobs_after_cache == 1

        # ----- Assertion 8: stat change makes the plan STALE ------------------
        before = read_visual_composition(
            session, candidate.id, display_probe=_RaisingProbe(), config=config51
        )
        assert before is not None and before.live_freshness == "CURRENT"
        now = time.time()
        os.utime(media_path, (now + 30.0, now + 30.0))
        after = read_visual_composition(
            session, candidate.id, display_probe=_RaisingProbe(), config=config51
        )
        assert after is not None
        report["assertions"]["8_stale"] = {
            "before_freshness": before.live_freshness,
            "after_freshness": after.live_freshness,
            "after_effective": bool(after.effective),
        }
        assert after.live_freshness == "STALE"
        assert after.effective is False
    except BaseException as error:  # noqa: BLE001 - recorded then re-raised
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["failure"] = failure
        report["artifacts"]["report"] = str(_write_report(report, artifacts))
        print("\nSTAGE51_LIVE_CONTRACT_REPORT=" + json.dumps(report, default=str))
        if session is not None:
            session.close()
        if engine is not None:
            engine.dispose()
