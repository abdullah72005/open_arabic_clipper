"""Bounded, cancellable FFmpeg/ffprobe child execution for Stage 5.2.

Uses safe argument arrays with ``shell=False`` and stdin disabled. stderr is
drained and bounded; cancellation and ownership are polled at a fixed cadence.
On cancellation, timeout, or process failure the child process group is
terminated, escalated to ``SIGKILL`` after a bounded grace, and reaped. No
unbounded process output is ever retained.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.render.execution.types import (
    AttemptContext,
    CompiledRender,
    RenderArtifacts,
)

_STDERR_LIMIT = 32_768
_TERMINATE_GRACE_SECONDS = 5.0
_DEFAULT_POLL_SECONDS = 0.5
_PROBE_TIMEOUT_SECONDS = 120.0


class RenderCancelled(RuntimeError):
    """The render was cooperatively cancelled."""


class RenderTimeout(RuntimeError):
    """The render exceeded its bounded wall-clock ceiling."""


class RenderProcessError(RuntimeError):
    """The FFmpeg process failed or produced no usable artifact.

    ``manifest`` optionally carries a partial measurement manifest produced before
    an MP4 existed (for example a completed unmastered audio analysis), so the
    Stage 6 finalizer can justify a bounded repair without pretending a failed
    render completed QC.
    """

    def __init__(
        self,
        reason_code: str,
        detail: str = "",
        *,
        manifest: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.detail = detail
        self.manifest: dict[str, object] = dict(manifest) if manifest else {}


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        group = os.getpgid(process.pid)
        os.killpg(group, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.terminate()
        except OSError:
            return
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        try:
            group = os.getpgid(process.pid)
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        pass


def _localize_ass(
    attempt_directory: Path, compiled: CompiledRender, context: AttemptContext
) -> None:
    data = context.ass_bytes
    if not data:
        raise RenderProcessError("ASS_MISSING", "no canonical ASS bytes were supplied")
    (attempt_directory / compiled.ass_localized_name).write_bytes(data)
    if compiled.authored_ass_localized_name:
        authored = context.authored_ass_bytes
        if not authored:
            raise RenderProcessError("ASS_MISSING", "no authored ASS bytes were supplied")
        (attempt_directory / compiled.authored_ass_localized_name).write_bytes(authored)


def _run_process_group(
    argv: list[str] | tuple[str, ...],
    attempt_directory: Path,
    context: AttemptContext,
    deadline: float | None,
    poll: float,
) -> tuple[int, str]:
    """Run one allow-listed child under the shared deadline and cancellation.

    Returns ``(returncode, stderr_tail)``; cancellation and deadline exhaustion
    terminate the process group and raise so no further process starts. The whole
    authored attempt (mix, measure, master, encode) shares one deadline.
    """

    process = subprocess.Popen(  # noqa: S603 - allow-listed argv, shell=False
        list(argv),
        cwd=str(attempt_directory),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stderr is not None
    stderr_buffer = bytearray()

    def _drain() -> None:
        try:
            while True:
                chunk = process.stderr.read(4096) if process.stderr is not None else b""
                if not chunk:
                    break
                room = _STDERR_LIMIT - len(stderr_buffer)
                if room > 0:
                    stderr_buffer.extend(chunk[:room])
        except (OSError, ValueError):  # pragma: no cover - defensive
            return

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()
    started = time.monotonic()
    timed_out = False
    cancelled = False
    while True:
        if process.poll() is not None:
            break
        if context.cancelled():
            cancelled = True
            break
        if deadline is not None and time.monotonic() > deadline:
            timed_out = True
            break
        if context.progress_callback is not None:
            try:
                context.progress_callback(
                    min(
                        1.0,
                        (time.monotonic() - started) / max(1e-6, context.timeout_seconds or 1.0),
                    )
                )
            except Exception:
                pass
        time.sleep(poll)

    if cancelled or timed_out:
        _terminate_process_group(process)
    else:
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            _terminate_process_group(process)
    drain_thread.join(timeout=_TERMINATE_GRACE_SECONDS)
    stderr_tail = bytes(stderr_buffer)[-_STDERR_LIMIT:].decode("utf-8", errors="replace")

    if cancelled:
        raise RenderCancelled("render cancelled")
    if timed_out:
        raise RenderTimeout("render timed out")
    return process.returncode, stderr_tail


def _checkpoint(deadline: float | None, context: AttemptContext) -> None:
    if context.cancelled():
        raise RenderCancelled("render cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        raise RenderTimeout("render timed out")


def _prepare_authored_audio(
    compiled: CompiledRender,
    context: AttemptContext,
    attempt_directory: Path,
    deadline: float | None,
    poll: float,
) -> None:
    """Execute mix -> measure -> linear master before the final encode.

    Partial measurements are attached to any raised :class:`RenderProcessError`
    so the finalizer can justify a bounded peak repair from real evidence.
    """

    from app.render.execution.audio import (
        LIMITED_RELATIVE_NAME,
        AudioPreparationError,
        build_limit_argv,
        build_master_argv,
        build_measure_argv,
        measurement_or_fail,
        reject_decision,
        select_mastering,
        verify_linear_mastering,
    )

    plan = compiled.audio_preparation
    assert plan is not None
    binary = compiled.runtime_identity.ffmpeg_binary
    partial: dict[str, object] = {}

    (attempt_directory / plan.mix_filtergraph_name).write_text(
        plan.mix_filtergraph, encoding="utf-8"
    )
    _checkpoint(deadline, context)
    code, stderr = _run_process_group(plan.mix_argv, attempt_directory, context, deadline, poll)
    if code != 0:
        raise RenderProcessError(
            "RENDER_PROCESS_FAILED",
            stderr[-2048:],
            manifest={"phase": "MIX", **partial},
        )

    _checkpoint(deadline, context)
    unmastered = str(attempt_directory / plan.unmastered_name)
    code, stderr = _run_process_group(
        build_measure_argv(binary, unmastered, plan.mastering),
        attempt_directory,
        context,
        deadline,
        poll,
    )
    try:
        measurement = measurement_or_fail(stderr)
        partial["mix_measurement"] = measurement.as_dict()
    except AudioPreparationError as error:
        raise RenderProcessError(error.reason_code, error.detail, manifest=partial) from error

    # One authorized bounded peak-limiter repair may be applied before the final
    # linear mastering; it starts from the lossless unmastered mix and is
    # re-measured so the mastering decision uses the actual limited signal.
    if float(getattr(plan.mastering, "repair_peak_limit_db", 0.0) or 0.0) > 0.0:
        _checkpoint(deadline, context)
        limited = str(attempt_directory / LIMITED_RELATIVE_NAME)
        code, stderr = _run_process_group(
            build_limit_argv(binary, unmastered, limited, measurement, plan.mastering),
            attempt_directory,
            context,
            deadline,
            poll,
        )
        if code != 0:
            raise RenderProcessError(
                "RENDER_PROCESS_FAILED", stderr[-2048:], manifest={"phase": "LIMIT", **partial}
            )
        unmastered = limited
        partial["peak_limit_applied"] = True
        _checkpoint(deadline, context)
        code, stderr = _run_process_group(
            build_measure_argv(binary, unmastered, plan.mastering),
            attempt_directory,
            context,
            deadline,
            poll,
        )
        try:
            measurement = measurement_or_fail(stderr)
            partial["limited_measurement"] = measurement.as_dict()
        except AudioPreparationError as error:
            raise RenderProcessError(error.reason_code, error.detail, manifest=partial) from error

    try:
        decision = select_mastering(measurement, plan.mastering)
        partial["mastering_decision"] = decision.as_dict()
        reject_decision(decision)
    except AudioPreparationError as error:
        raise RenderProcessError(error.reason_code, error.detail, manifest=partial) from error

    _checkpoint(deadline, context)
    master = str(attempt_directory / plan.master_name)
    code, stderr = _run_process_group(
        build_master_argv(binary, unmastered, master, measurement, decision, plan.mastering),
        attempt_directory,
        context,
        deadline,
        poll,
    )
    if code != 0:
        raise RenderProcessError(
            "RENDER_PROCESS_FAILED", stderr[-2048:], manifest={"phase": "MASTER", **partial}
        )
    try:
        verify_linear_mastering(stderr)
    except AudioPreparationError as error:
        raise RenderProcessError(error.reason_code, error.detail, manifest=partial) from error
    partial["phase"] = "MASTERED"
    partial["master_file"] = plan.master_name


def run_compiled_render(compiled: CompiledRender, context: AttemptContext) -> RenderArtifacts:
    """Execute one compiled render, returning the artifact plus probe facts."""

    attempt_directory = context.attempt_directory
    attempt_directory.mkdir(parents=True, exist_ok=True)
    (attempt_directory / compiled.filtergraph_relative_path).write_text(
        compiled.filtergraph, encoding="utf-8"
    )
    _localize_ass(attempt_directory, compiled, context)

    poll = context.poll_seconds if context.poll_seconds > 0 else _DEFAULT_POLL_SECONDS
    started = time.monotonic()
    deadline = context.effective_deadline(started)

    if compiled.audio_preparation is not None:
        _prepare_authored_audio(compiled, context, attempt_directory, deadline, poll)

    returncode, stderr_tail = _run_process_group(
        compiled.argv, attempt_directory, context, deadline, poll
    )
    if returncode != 0:
        raise RenderProcessError("RENDER_PROCESS_FAILED", stderr_tail[-2048:])

    output_path = attempt_directory / compiled.output_relative_path
    if not output_path.is_file():
        raise RenderProcessError("ARTIFACT_MISSING", stderr_tail[-2048:])
    size_bytes = output_path.stat().st_size
    if size_bytes <= 0:
        raise RenderProcessError("ARTIFACT_ZERO_BYTES", stderr_tail[-2048:])

    # Post-encode work shares the attempt deadline and polls cancellation between
    # bounded steps; no further process starts once the deadline is exhausted.
    if context.cancelled():
        raise RenderCancelled("render cancelled before hashing")
    digest = _sha256_file(output_path)
    # Cancellation is checked before the deadline so a stop at the boundary is
    # reported as cancellation, and the probe budget is recomputed *after*
    # hashing (which may itself have consumed the remaining budget).
    if context.cancelled():
        raise RenderCancelled("render cancelled before output probing")
    probe_budget = _remaining_budget(deadline, _PROBE_TIMEOUT_SECONDS)
    try:
        probe = _probe_output(
            compiled.runtime_identity.ffprobe_binary,
            output_path,
            timeout_seconds=probe_budget,
            deadline=deadline,
        )
    except (RenderTimeout, RenderProcessError):
        # A cancellation/ownership stop observed during probing must enter the
        # cancellation path, not be reported as a timeout or probe failure. This
        # covers every probe exception boundary: shared-deadline exhaustion,
        # nonzero exit, launch failure, and malformed output.
        if context.cancelled():
            raise RenderCancelled("render cancelled during output probing") from None
        raise
    if context.cancelled():
        raise RenderCancelled("render cancelled after output probing")
    _ensure_deadline_not_expired(deadline)
    duration = _as_float(probe.get("duration_seconds"), 0.0)
    frame_count = _as_int(probe.get("frame_count"), 0)
    sample_rate = _as_int(probe.get("audio_sample_rate"), 0)
    channels = _as_int(probe.get("audio_channels"), 0)
    sample_count = int(round(duration * sample_rate)) if sample_rate else 0
    manifest = {
        "compiler_version": compiled.compiler_version,
        "timeline": compiled.manifest.as_dict(),
        "output": {
            "relative_path": compiled.output_relative_path,
            "sha256": digest,
            "size_bytes": size_bytes,
            "duration_seconds": round(duration, 6),
            "frame_count": frame_count,
            "sample_rate": sample_rate,
            "channels": channels,
        },
        "diagnostics": dict(compiled.diagnostics),
        "expected": dict(compiled.expected),
        "argv_digest": hashlib.sha256("\x00".join(compiled.argv).encode("utf-8")).hexdigest(),
    }
    return RenderArtifacts(
        output_path=output_path,
        output_relative_path=compiled.output_relative_path,
        sha256=digest,
        size_bytes=size_bytes,
        probe=probe,
        manifest=manifest,
        duration_seconds=duration,
        frame_count=frame_count,
        sample_count=sample_count,
        sample_rate=sample_rate,
        channels=channels,
        stderr_tail=stderr_tail[-2048:],
    )


def _remaining_budget(deadline: float | None, base: float) -> float:
    """Remaining shared budget for a bounded post-encode call.

    Raises ``RenderTimeout`` when the shared attempt deadline has expired so no
    further subprocess is started.
    """

    if deadline is None:
        return base
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RenderTimeout("render timed out before output probing")
    return min(base, remaining)


def _ensure_deadline_not_expired(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise RenderTimeout("render timed out during output probing")


def _probe_output(
    binary: str,
    path: Path,
    *,
    timeout_seconds: float = _PROBE_TIMEOUT_SECONDS,
    deadline: float | None = None,
) -> dict[str, object]:
    command = [
        binary,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - allow-listed argv, shell=False
            command, check=True, capture_output=True, text=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as error:
        # A probe that consumed its shared budget is deadline exhaustion, not an
        # ordinary probe failure.
        if deadline is not None and time.monotonic() >= deadline:
            raise RenderTimeout("render timed out during output probing") from error
        raise RenderProcessError("QC_PROBE_FAILED", type(error).__name__) from error
    except (OSError, subprocess.SubprocessError) as error:
        raise RenderProcessError("QC_PROBE_FAILED", type(error).__name__) from error
    try:
        payload: Any = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RenderProcessError("QC_PROBE_FAILED", "invalid ffprobe json") from error
    if not isinstance(payload, Mapping):
        raise RenderProcessError("QC_PROBE_FAILED", "invalid ffprobe payload")
    streams = payload.get("streams")
    video = (
        next(
            (s for s in streams if isinstance(s, Mapping) and s.get("codec_type") == "video"),
            None,
        )
        if isinstance(streams, list)
        else None
    )
    audio = (
        next(
            (s for s in streams if isinstance(s, Mapping) and s.get("codec_type") == "audio"),
            None,
        )
        if isinstance(streams, list)
        else None
    )
    format_section = payload.get("format")
    fmt: Mapping[str, Any] = format_section if isinstance(format_section, Mapping) else {}
    duration = _as_float(fmt.get("duration"), 0.0)
    if duration <= 0 and isinstance(video, Mapping):
        duration = _as_float(video.get("duration"), 0.0)
    result: dict[str, object] = {
        "duration_seconds": duration,
        "video_codec": video.get("codec_name") if isinstance(video, Mapping) else None,
        "pix_fmt": video.get("pix_fmt") if isinstance(video, Mapping) else None,
        "width": _as_int(video.get("width"), 0) if isinstance(video, Mapping) else 0,
        "height": _as_int(video.get("height"), 0) if isinstance(video, Mapping) else 0,
        "sample_aspect_ratio": video.get("sample_aspect_ratio")
        if isinstance(video, Mapping)
        else None,
        "display_aspect_ratio": video.get("display_aspect_ratio")
        if isinstance(video, Mapping)
        else None,
        "frame_count": _as_int(video.get("nb_frames"), 0) if isinstance(video, Mapping) else 0,
        "avg_frame_rate": video.get("avg_frame_rate") if isinstance(video, Mapping) else None,
        "video_start_time": _as_float(video.get("start_time"), 0.0)
        if isinstance(video, Mapping)
        else 0.0,
        "video_duration": _stream_duration(video, duration),
        "audio_codec": audio.get("codec_name") if isinstance(audio, Mapping) else None,
        "audio_sample_rate": _as_int(audio.get("sample_rate"), 0)
        if isinstance(audio, Mapping)
        else 0,
        "audio_channels": _as_int(audio.get("channels"), 0) if isinstance(audio, Mapping) else 0,
        "audio_start_time": _as_float(audio.get("start_time"), 0.0)
        if isinstance(audio, Mapping)
        else 0.0,
        "audio_duration": _stream_duration(audio, duration),
        "streams": {
            "video": sum(
                1 for s in streams if isinstance(s, Mapping) and s.get("codec_type") == "video"
            )
            if isinstance(streams, list)
            else 0,
            "audio": sum(
                1 for s in streams if isinstance(s, Mapping) and s.get("codec_type") == "audio"
            )
            if isinstance(streams, list)
            else 0,
            "subtitle": sum(
                1 for s in streams if isinstance(s, Mapping) and s.get("codec_type") == "subtitle"
            )
            if isinstance(streams, list)
            else 0,
            "data": sum(
                1 for s in streams if isinstance(s, Mapping) and s.get("codec_type") == "data"
            )
            if isinstance(streams, list)
            else 0,
        },
    }
    rotation = _rotation_from_stream(video) if isinstance(video, Mapping) else 0
    result["rotation_degrees"] = rotation
    return result


def _stream_duration(stream: object, fallback: float) -> float:
    if isinstance(stream, Mapping):
        explicit = _as_float(stream.get("duration"), 0.0)
        if explicit > 0:
            return explicit
    return fallback


def _rotation_from_stream(stream: Mapping[str, object]) -> int:
    side = stream.get("side_data_list")
    if isinstance(side, list):
        for entry in side:
            if isinstance(entry, Mapping) and entry.get("rotation") is not None:
                try:
                    return int(round(float(entry["rotation"]))) % 360
                except (TypeError, ValueError):
                    return 0
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


__all__ = [
    "RenderCancelled",
    "RenderProcessError",
    "RenderTimeout",
    "run_compiled_render",
]
