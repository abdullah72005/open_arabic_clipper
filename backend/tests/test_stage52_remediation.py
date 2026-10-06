"""Stage 5.2 remediation regressions for the reviewed defect surface.

Each test exercises an observable outcome of a previously broken behavior:
force-after-success attempt reset, authoritative fencing, upstream invalidation
during execution, strict plan validation, cache-artifact validity, QC-only cache
invalidation, historical reactivation, dispatch recovery, offset-preserving A/V,
source-aware silence QC, and truthful runtime identity.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, update
from sqlalchemy.orm import Session
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    Stage51Fixture,
    seed_stage51,
)
from stage52_support import install_fake_stream_probe, make_spec, occurred

from app.composition.queue import queue_visual_composition
from app.composition.service import execute_visual_composition, get_current_visual_composition
from app.core.enums import JobStatus, RenderExecutionLifecycle
from app.core.settings import get_settings
from app.db.base import Base
from app.models import ProcessingJob, VisualCompositionPlan
from app.models.render_execution import RenderExecution
from app.pipeline.executor import StageCancelled
from app.render.execution.compiler import compile_render
from app.render.execution.executor import build_render_execution_executor
from app.render.execution.policy import Stage52Config
from app.render.execution.qc import check_render_artifact
from app.render.execution.queue import queue_render_execution
from app.render.execution.runner import RenderCancelled, RenderProcessError, run_compiled_render
from app.render.execution.service import (
    RenderExecutionError,
    build_render_spec,
    request_input_fingerprint,
    resolve_runtime_identity,
)
from app.render.execution.types import (
    AttemptContext,
    RenderArtifacts,
    RuntimeIdentity,
    SceneSpec,
    TechnicalQCResult,
)
from app.services.storage import StorageService

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
FFMPEG_BIN = FFMPEG or "ffmpeg"
_requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe unavailable"
)

_VALID_ASS = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1080\n"
    "PlayResY: 1920\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    "Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
    "0,0,0,0,100,100,0,0,1,1,1,2,10,10,10,1\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
).encode("utf-8")


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.render.execution.queue._dispatch", lambda *args: None)
    monkeypatch.setattr("app.composition.queue._dispatch", lambda *args: None)
    install_fake_stream_probe(monkeypatch)


def _plan_ready(session: Session, fixture: Stage51Fixture) -> None:
    queue_visual_composition(session, fixture.stage50.selection.candidate)
    row = get_current_visual_composition(session, fixture.stage50.selection.candidate.id)
    assert row is not None
    execute_visual_composition(
        session,
        row.id,
        storage=fixture.stage50.storage,
        settings=fixture.settings,
        display_probe=fixture.display_probe,
        frame_sampler=FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=FakeDetector(),
    )
    session.commit()


class _FakeRunner:
    def __init__(self, *, on_call: Any | None = None) -> None:
        self.calls = 0
        self._on_call = on_call

    def __call__(self, compiled: Any, context: Any) -> RenderArtifacts:
        self.calls += 1
        if self._on_call is not None:
            self._on_call()
        output = context.attempt_directory / "output.mp4"
        payload = f"fake-render-bytes-{self.calls}".encode()
        output.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        probe = {
            "width": 1080,
            "height": 1920,
            "video_codec": "h264",
            "pix_fmt": "yuv420p",
            "audio_codec": "aac",
            "audio_sample_rate": 48000,
            "audio_channels": 2,
            "duration_seconds": 2.0,
            "frame_count": 60,
            "streams": {"video": 1, "audio": 1, "subtitle": 0, "data": 0},
            "rotation_degrees": 0,
            "sample_aspect_ratio": "1:1",
        }
        return RenderArtifacts(
            output_path=output,
            output_relative_path="output.mp4",
            sha256=digest,
            size_bytes=len(payload),
            probe=probe,
            manifest=compiled.manifest.as_dict(),
            duration_seconds=2.0,
            frame_count=60,
            sample_count=96000,
            sample_rate=48000,
            channels=2,
        )


def _fake_qc(
    artifacts: Any,
    manifest: Any,
    config: Any,
    *,
    source_path: Any = None,
    cancel_check: Any = None,
    deadline: Any = None,
) -> TechnicalQCResult:
    return TechnicalQCResult(status="PASS", checks=(), reason_codes=(), policy_version="test")


def _runtime_factory() -> Any:
    from app.render.execution.service import resolve_runtime_identity as real

    settings = get_settings()

    def factory(
        _settings: Any, *, source_absolute_path: str = "", attempt_directory: str = ""
    ) -> RuntimeIdentity:
        return real(
            settings,
            source_absolute_path=source_absolute_path,
            attempt_directory=attempt_directory,
        )

    return factory


def _executor(
    session: Session,
    runner: _FakeRunner,
    *,
    qc_checker: Any | None = None,
    admission: Any | None = None,
) -> Any:
    settings = get_settings()
    return build_render_execution_executor(
        session,
        StorageService(settings.storage_root),
        settings,
        admission=admission,
        runner=runner,
        qc_checker=qc_checker or _fake_qc,
        runtime_factory=_runtime_factory(),
    )


def _cancel_job_in_other_session(engine: Engine, job_id: Any) -> None:
    """Flip the real job to CANCELLED on a separate committed connection."""

    other = Session(engine)
    try:
        job = other.get(ProcessingJob, job_id)
        assert job is not None
        job.status = JobStatus.CANCELLED
        other.commit()
    finally:
        other.close()


def _claimed_executor(session: Session, outcome: Any, *, admission: Any | None = None) -> Any:
    executor = _executor(session, _FakeRunner(), admission=admission)
    executor.set_active_job(outcome.job_id)
    assert executor._claim_job() is True
    executor._row_id = outcome.render_execution_id
    return executor


class _LosingAdmission:
    def acquire(self, *, wait_seconds: float, cancel_check: Any) -> bool:
        return True

    def release(self) -> None:  # pragma: no cover - not exercised
        return None

    def held(self) -> bool:
        return False


def _complete_once(session: Session, fixture: Stage51Fixture, runner: _FakeRunner) -> Any:
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    executor = _executor(session, runner)
    executor.set_active_job(outcome.job_id)
    executor.execute(outcome.render_execution_id)
    return outcome


def test_force_after_success_resets_attempt_state_and_fails_cleanly(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    _complete_once(session, fixture, _FakeRunner())
    session.expire_all()

    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id, force=True)
    assert outcome.queued is True

    def _boom() -> None:
        raise RenderProcessError("RENDER_PROCESS_FAILED", "forced rerender failure")

    failing = _FakeRunner(on_call=_boom)
    executor = _executor(session, failing)
    executor.set_active_job(outcome.job_id)
    with pytest.raises(RenderProcessError):
        executor.execute(outcome.render_execution_id, force=True)
    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None
    assert row.lifecycle is RenderExecutionLifecycle.FAILED
    assert row.cache_eligible is False
    assert row.artifact_reference == {}


def test_upstream_invalidation_during_render_blocks_publication(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    candidate_id = fixture.stage50.selection.candidate.id
    outcome = queue_render_execution(session, candidate_id)

    def _invalidate() -> None:
        from app.models import ClipCandidate

        candidate = session.get(ClipCandidate, candidate_id)
        assert candidate is not None
        candidate.is_current = False
        session.commit()

    runner = _FakeRunner(on_call=_invalidate)
    executor = _executor(session, runner)
    executor.set_active_job(outcome.job_id)
    executor.execute(outcome.render_execution_id)
    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None
    assert row.lifecycle is not RenderExecutionLifecycle.COMPLETE
    assert row.cache_eligible is False


def test_post_qc_upstream_invalidation_blocks_publication(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    candidate_id = fixture.stage50.selection.candidate.id
    outcome = queue_render_execution(session, candidate_id)

    def invalidating_qc(*args: Any, **kwargs: Any) -> TechnicalQCResult:
        # Invalidate *after* QC passes, in the window that used to sit between
        # revalidation and publication.
        from app.models import ClipCandidate

        candidate = session.get(ClipCandidate, candidate_id)
        assert candidate is not None
        candidate.is_current = False
        session.commit()
        return TechnicalQCResult(status="PASS", checks=(), reason_codes=(), policy_version="test")

    executor = _executor(session, _FakeRunner(), qc_checker=invalidating_qc)
    executor.set_active_job(outcome.job_id)
    executor.execute(outcome.render_execution_id)
    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None
    assert row.lifecycle is not RenderExecutionLifecycle.COMPLETE
    assert row.cache_eligible is False
    job = session.get(ProcessingJob, outcome.job_id)
    assert job is not None and job.status is not JobStatus.SUCCEEDED


def test_publication_failure_propagates_and_does_not_falsely_succeed(
    session: Session, monkeypatch: Any, sqlite_engine: Engine
) -> None:
    import app.render.execution.executor as executor_module
    import app.workers.tasks as tasks

    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert outcome.job_id is not None

    def boom(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("publication backend failure")

    monkeypatch.setattr(executor_module, "finalize_success", boom)
    monkeypatch.setattr(tasks, "create_session_factory", lambda: lambda: Session(sqlite_engine))

    with pytest.raises(RuntimeError):
        tasks.run_render_execution.apply(
            args=[str(outcome.render_execution_id), str(outcome.job_id), False]
        ).get()

    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    job = session.get(ProcessingJob, outcome.job_id)
    assert row is not None
    assert row.lifecycle is not RenderExecutionLifecycle.COMPLETE
    assert row.cache_eligible is False
    # The failure is recorded truthfully; the job is never left RUNNING.
    assert job is not None and job.status is not JobStatus.RUNNING
    assert job.status is JobStatus.FAILED


def test_source_audio_analysis_does_not_start_after_cancel(monkeypatch: Any) -> None:
    from app.render.execution import qc as qc_module
    from app.render.execution.qc import QCCancelled

    calls: list[str] = []
    state = {"cancel": False}

    def fake_luma(binary: str, path: Path, time_s: float, *, deadline: Any = None) -> bytes:
        return b"\x00" * (160 * 284)

    def fake_volume(
        binary: str,
        path: Path,
        start: float = 0.0,
        end: float | None = None,
        *,
        cancel_check: Any = None,
        deadline: Any = None,
    ) -> dict[str, float]:
        calls.append(path.name)
        if path.name == "output.mp4":
            # Cancellation is observed during output-audio analysis.
            state["cancel"] = True
            return {"mean": -math.inf, "max": -math.inf}
        return {"mean": -20.0, "max": -3.0}

    monkeypatch.setattr(qc_module, "_extract_luma", fake_luma)
    monkeypatch.setattr(qc_module, "_volume_metrics", fake_volume)
    manifest = {
        "timeline": {
            "occurrences": [
                {
                    "output_start": 0.0,
                    "output_end": 2.0,
                    "source_start": 10.0,
                    "source_end": 12.0,
                }
            ]
        }
    }
    artifacts = RenderArtifacts(
        output_path=Path("/tmp/output.mp4"),
        output_relative_path="output.mp4",
        sha256="x",
        size_bytes=1,
        probe={
            "streams": {"video": 1, "audio": 1, "subtitle": 0, "data": 0},
            "avg_frame_rate": "30/1",
        },
        manifest=manifest,
        duration_seconds=2.0,
        frame_count=60,
        sample_count=96000,
        sample_rate=48000,
        channels=2,
    )
    with pytest.raises(QCCancelled):
        qc_module.check_render_artifact(
            artifacts,
            manifest,
            Stage52Config(),
            source_path=Path("/tmp/source.mp4"),
            cancel_check=lambda: state["cancel"],
        )
    # No source-audio subprocess may start once cancellation is observed.
    assert "source.mp4" not in calls


def test_malformed_plan_fails_closed(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    candidate_id = fixture.stage50.selection.candidate.id
    from sqlalchemy import select

    plan = session.scalars(
        select(VisualCompositionPlan).where(VisualCompositionPlan.clip_candidate_id == candidate_id)
    ).first()
    assert plan is not None
    payload = dict(plan.plan_payload)
    scenes = [dict(scene) for scene in payload.get("scenes", [])]
    assert scenes
    # Simulate corrupted upstream data: drop framing_mode and invalidate cx.
    scenes[0].pop("framing_mode", None)
    if scenes[0].get("crop_keyframes"):
        keyframes = [dict(frame) for frame in scenes[0]["crop_keyframes"]]
        keyframes[0]["cx"] = "not-a-number"
        scenes[0]["crop_keyframes"] = keyframes
    payload["scenes"] = scenes
    plan.plan_payload = payload
    session.commit()

    from app.models import ClipCandidate

    candidate = session.get(ClipCandidate, candidate_id)
    assert candidate is not None
    with pytest.raises(RenderExecutionError):
        build_render_spec(
            session,
            candidate,
            artifact_purpose="CORE_SOURCE_VALIDATION",
            delivery_profile_key="MP4_H264_AAC_1080X1920_V1",
            settings=fixture.settings,
            storage=fixture.stage50.storage,
        )


def test_missing_artifact_is_not_reused_from_cache(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    runner = _FakeRunner()
    outcome = _complete_once(session, fixture, runner)
    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None
    relative = row.artifact_reference["relative_path"]
    (StorageService(fixture.settings.storage_root).storage_root / relative).unlink()

    cached = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert cached.cached is False
    assert cached.job_id is not None

    rerun = _executor(session, runner)
    rerun.set_active_job(cached.job_id)
    rerun.execute(cached.render_execution_id)
    assert runner.calls == 2
    session.expire_all()
    assert session.get(RenderExecution, cached.render_execution_id).cache_eligible is True


def test_qc_policy_change_invalidates_request_fingerprint() -> None:
    spec = make_spec()
    base = request_input_fingerprint(spec, Stage52Config())
    changed = request_input_fingerprint(
        spec, Stage52Config(qc_max_sampled_frames=Stage52Config().qc_max_sampled_frames + 8)
    )
    assert base != changed


def test_runtime_identity_is_truthful() -> None:
    from app.render.execution import service

    service._RUNTIME_VERSION_CACHE.clear()
    identity = resolve_runtime_identity(get_settings())
    assert identity.ffmpeg_version
    assert identity.libavcodec_version
    # Complete, whitespace-collapsed version (never a truncated fragment).
    assert identity.libavformat_version.count(".") >= 2
    assert "  " not in identity.libavformat_version
    assert identity.build_config_sha256
    # Truthful libass identity, never the configure capability flag.
    assert identity.libass_version
    assert identity.libass_version != "--enable-libass"
    assert identity.libass_sha256
    # Actual resolved font content is hashed, never an empty hash or a basename.
    assert identity.font_sha256
    assert identity.font_match.startswith("/")
    assert Path(identity.font_match).is_file()


def test_runtime_cache_detects_dependency_change(monkeypatch: Any) -> None:
    from app.render.execution import service

    service._RUNTIME_VERSION_CACHE.clear()
    calls = {"n": 0}

    def fake_binary_output(binary: str, flag: str) -> str:
        calls["n"] += 1
        return (
            "ffmpeg version test\n"
            "configuration: --enable-libass\n"
            "libavformat    61.  7.103 / 61.  7.103\n"
            "libavcodec     61. 19.101 / 61. 19.101\n"
        )

    monkeypatch.setattr(service, "_binary_output", fake_binary_output)
    service._runtime_version_fields("ffmpeg", "ffprobe", "Noto Sans Arabic")
    after_first = calls["n"]
    assert after_first > 0
    # Repeated identical dependency signature is a cache hit.
    service._runtime_version_fields("ffmpeg", "ffprobe", "Noto Sans Arabic")
    assert calls["n"] == after_first
    # A changed dependency signature under the same path must invalidate.
    monkeypatch.setattr(service, "_stat_signature", lambda path: (999999, 1) if path else (0, 0))
    service._runtime_version_fields("ffmpeg", "ffprobe", "Noto Sans Arabic")
    assert calls["n"] > after_first


def test_historical_request_reactivation(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    settings = get_settings()
    monkeypatch.setattr(settings, "render_encoder_threads", 2)
    first = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    first_runner = _FakeRunner()
    ex = _executor(session, first_runner)
    ex.set_active_job(first.job_id)
    ex.execute(first.render_execution_id)

    monkeypatch.setattr(settings, "render_encoder_threads", 3)
    second = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert second.render_execution_id != first.render_execution_id
    second_runner = _FakeRunner()
    ex2 = _executor(session, second_runner)
    ex2.set_active_job(second.job_id)
    ex2.execute(second.render_execution_id)

    monkeypatch.setattr(settings, "render_encoder_threads", 2)
    reactivated = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert reactivated.render_execution_id == first.render_execution_id
    session.expire_all()
    first_row = session.get(RenderExecution, first.render_execution_id)
    second_row = session.get(RenderExecution, second.render_execution_id)
    assert first_row is not None and first_row.is_current is True
    assert second_row is not None and second_row.is_current is False


def test_dispatch_failure_is_recovered_on_next_request(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    state = {"fail": True, "calls": 0}

    def _flaky(*args: Any) -> None:
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("broker unavailable")

    monkeypatch.setattr("app.render.execution.queue._dispatch", _flaky)
    first = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert first.queued is True and first.job_id is not None
    session.expire_all()
    job = session.get(ProcessingJob, first.job_id)
    assert job is not None and job.error_code == "RENDER_DISPATCH_FAILED"

    state["fail"] = False
    second = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert second.active is True and second.job_id == first.job_id
    assert state["calls"] == 2
    session.expire_all()
    job = session.get(ProcessingJob, first.job_id)
    assert job is not None and job.error_code is None


def _generate_offset_source(path: Path) -> None:
    subprocess.run(
        [
            FFMPEG_BIN,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30:duration=3",
            "-itsoffset",
            "0.25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2.6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@_requires_ffmpeg
def test_real_render_preserves_audio_start_offset(tmp_path: Path) -> None:
    source = tmp_path / "offset-source.mp4"
    _generate_offset_source(source)
    scene = SceneSpec(0, 0, 0.0, 1.0, "SOURCE_AS_IS", "smoothstep-ease")
    occ = occurred("block-0", 0, 0.0, 1.0, 0.0, (scene,))
    spec = make_spec(
        occurrences=(occ,),
        caption_events=(),
        source_duration=3.0,
        source_video_start=0.0,
        source_audio_start=0.25,
    )
    attempt = tmp_path / "attempt"
    from stage52_support import fake_runtime

    runtime = fake_runtime(
        ffmpeg_binary=FFMPEG_BIN, source_absolute_path=str(source), attempt_directory=str(attempt)
    )
    compiled = compile_render(spec, runtime)
    artifacts = run_compiled_render(
        compiled,
        AttemptContext(attempt_directory=attempt, ass_bytes=_VALID_ASS, timeout_seconds=300),
    )
    probe = artifacts.probe
    assert float(probe["video_duration"]) == pytest.approx(1.0, abs=0.15)
    assert float(probe["audio_duration"]) == pytest.approx(1.0, abs=0.2)


def _generate_absolute_offset_source(path: Path) -> None:
    subprocess.run(
        [
            FFMPEG_BIN,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30:duration=2",
            "-itsoffset",
            "0.25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-output_ts_offset",
            "5",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@_requires_ffmpeg
def test_absolute_stream_offsets_normalize_to_shared_origin(tmp_path: Path) -> None:
    from app.render.execution.qc import _volume_metrics
    from app.render.execution.service import _probe_source_stream_facts

    source = tmp_path / "absolute-offset.mp4"
    _generate_absolute_offset_source(source)
    facts = _probe_source_stream_facts(source, get_settings())
    assert facts is not None
    # Common 5 s container offset is removed; the relative 250 ms A/V offset stays.
    assert facts.video_start_seconds == pytest.approx(0.0, abs=0.02)
    assert facts.audio_start_seconds == pytest.approx(0.25, abs=0.05)
    assert facts.verified is True

    scene = SceneSpec(0, 0, 0.0, 1.0, "SOURCE_AS_IS", "smoothstep-ease")
    occ = occurred("block-0", 0, 0.0, 1.0, 0.0, (scene,))
    spec = make_spec(
        occurrences=(occ,),
        caption_events=(),
        source_duration=2.0,
        source_video_start=facts.video_start_seconds,
        source_audio_start=facts.audio_start_seconds,
    )
    attempt = tmp_path / "attempt"
    from stage52_support import fake_runtime

    runtime = fake_runtime(
        ffmpeg_binary=FFMPEG_BIN, source_absolute_path=str(source), attempt_directory=str(attempt)
    )
    compiled = compile_render(spec, runtime)
    artifacts = run_compiled_render(
        compiled,
        AttemptContext(attempt_directory=attempt, ass_bytes=_VALID_ASS, timeout_seconds=300),
    )
    # A 1 s source-local span must render ~1 s, never the ~6 s the raw offset caused.
    assert artifacts.duration_seconds == pytest.approx(1.0, abs=0.15)
    # Decoded audio: leading ~250 ms silence, then real signal.
    leading = _volume_metrics(FFMPEG_BIN, artifacts.output_path, 0.0, 0.2)
    signal = _volume_metrics(FFMPEG_BIN, artifacts.output_path, 0.4, 0.9)
    assert leading is not None and leading.get("mean", 0.0) <= -60.0
    assert signal is not None and signal.get("mean", -100.0) > -60.0


def _generate_partly_silent_source(path: Path) -> None:
    subprocess.run(
        [
            FFMPEG_BIN,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30:duration=3",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=if(lt(t\\,2)\\,0.6*sin(2*PI*440*t)\\,0):s=48000:d=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@_requires_ffmpeg
def test_legitimate_silent_selected_span_is_not_a_hard_failure(tmp_path: Path) -> None:
    source = tmp_path / "partly-silent.mp4"
    _generate_partly_silent_source(source)
    scene = SceneSpec(0, 0, 2.0, 2.9, "SOURCE_AS_IS", "smoothstep-ease")
    occ = occurred("block-0", 0, 2.0, 2.9, 0.0, (scene,))
    spec = make_spec(occurrences=(occ,), caption_events=(), source_duration=3.0)
    attempt = tmp_path / "attempt"
    from stage52_support import fake_runtime

    runtime = fake_runtime(
        ffmpeg_binary=FFMPEG_BIN, source_absolute_path=str(source), attempt_directory=str(attempt)
    )
    compiled = compile_render(spec, runtime)
    artifacts = run_compiled_render(
        compiled,
        AttemptContext(attempt_directory=attempt, ass_bytes=_VALID_ASS, timeout_seconds=300),
    )
    qc = check_render_artifact(artifacts, artifacts.manifest, Stage52Config(), source_path=source)
    assert qc.status != "FAIL"
    assert "QC_TOTAL_SILENCE" not in qc.reason_codes


def test_qc_failure_persists_result_and_is_not_cache_eligible(
    session: Session, monkeypatch: Any
) -> None:
    from app.render.execution.types import QCCheck

    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)

    def failing_qc(
        artifacts: Any,
        manifest: Any,
        config: Any,
        *,
        source_path: Any = None,
        cancel_check: Any = None,
        deadline: Any = None,
    ) -> TechnicalQCResult:
        return TechnicalQCResult(
            status="FAIL",
            checks=(QCCheck(name="not_blank", status="FAIL", reason_code="QC_BLANK_RENDER"),),
            reason_codes=("QC_BLANK_RENDER",),
            measured={"black_frame_fraction": 1.0},
            policy_version="test",
        )

    executor = _executor(session, _FakeRunner(), qc_checker=failing_qc)
    executor.set_active_job(outcome.job_id)
    executor.execute(outcome.render_execution_id)
    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None
    assert row.lifecycle is RenderExecutionLifecycle.FAILED
    assert row.qc_status is not None and row.qc_status.value == "FAIL"
    assert row.qc_result and row.qc_result.get("reason_codes") == ["QC_BLANK_RENDER"]
    assert row.cache_eligible is False
    assert row.artifact_reference == {}
    job = session.get(ProcessingJob, outcome.job_id)
    assert job is not None and job.status is JobStatus.FAILED


def test_running_cancellation_finalizes_execution(
    session: Session, monkeypatch: Any, sqlite_engine: Engine
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert outcome.job_id is not None
    job_id = outcome.job_id

    class _CancelRunner:
        def __call__(self, compiled: Any, context: Any) -> Any:
            # The API session cancels the real job on another connection while
            # the worker is rendering.
            _cancel_job_in_other_session(sqlite_engine, job_id)
            raise RenderCancelled("render cancelled")

    executor = _executor(session, _CancelRunner())
    executor.set_active_job(job_id)
    with pytest.raises(StageCancelled):
        executor.execute(outcome.render_execution_id)

    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None
    assert row.lifecycle is RenderExecutionLifecycle.CANCELLED
    assert row.active_job_id is None
    assert row.cache_eligible is False
    assert row.artifact_reference == {}
    job = session.get(ProcessingJob, job_id)
    assert job is not None and job.status is JobStatus.CANCELLED


def test_qc_cancellation_finalizes_execution(
    session: Session, monkeypatch: Any, sqlite_engine: Engine
) -> None:
    from app.render.execution.qc import QCCancelled

    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert outcome.job_id is not None
    job_id = outcome.job_id

    def cancelling_qc(*args: Any, **kwargs: Any) -> TechnicalQCResult:
        _cancel_job_in_other_session(sqlite_engine, job_id)
        raise QCCancelled("cancelled during QC")

    executor = _executor(session, _FakeRunner(), qc_checker=cancelling_qc)
    executor.set_active_job(job_id)
    with pytest.raises(StageCancelled):
        executor.execute(outcome.render_execution_id)

    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None
    assert row.lifecycle is RenderExecutionLifecycle.CANCELLED
    assert row.active_job_id is None
    job = session.get(ProcessingJob, job_id)
    assert job is not None and job.status is JobStatus.CANCELLED


def test_ownership_loss_latch_is_sticky_and_stops_work(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    executor = _claimed_executor(session, outcome)
    assert executor._should_stop() is False
    executor._mark_ownership_lost()
    assert executor._should_stop() is True
    # The latch is sticky even while the job is still RUNNING and owned.
    assert executor._should_stop() is True


def test_terminal_job_status_stops_work(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    executor = _claimed_executor(session, outcome)
    session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == outcome.job_id)
        .values(status=JobStatus.SUCCEEDED)
    )
    session.commit()
    assert executor._should_stop() is True


def test_claim_replacement_stops_work(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    executor = _claimed_executor(session, outcome)
    session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == outcome.job_id)
        .values(claim_version=executor._claim_version + 5)
    )
    session.commit()
    assert executor._should_stop() is True


def test_active_job_reassignment_stops_work(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    executor = _claimed_executor(session, outcome)
    session.execute(
        update(RenderExecution)
        .where(RenderExecution.id == outcome.render_execution_id)
        .values(active_job_id=uuid.uuid4())
    )
    session.commit()
    assert executor._should_stop() is True


def test_admission_loss_stops_work(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    executor = _claimed_executor(session, outcome, admission=_LosingAdmission())
    executor._admission_held = True
    assert executor._should_stop() is True


def test_queued_cancellation_finalizes_through_task_entry_point(
    session: Session, monkeypatch: Any, sqlite_engine: Engine
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    outcome = queue_render_execution(session, fixture.stage50.selection.candidate.id)
    assert outcome.job_id is not None
    job = session.get(ProcessingJob, outcome.job_id)
    assert job is not None
    job.status = JobStatus.CANCELLED
    session.commit()

    import app.workers.tasks as tasks

    monkeypatch.setattr(tasks, "create_session_factory", lambda: lambda: Session(sqlite_engine))
    result = tasks.run_render_execution.apply(
        args=[str(outcome.render_execution_id), str(outcome.job_id), False]
    ).get()
    assert result["cancelled"] is True
    session.expire_all()
    row = session.get(RenderExecution, outcome.render_execution_id)
    assert row is not None and row.lifecycle is RenderExecutionLifecycle.CANCELLED
    assert row.active_job_id is None
