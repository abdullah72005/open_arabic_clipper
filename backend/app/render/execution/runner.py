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


class RenderCancelled(RuntimeError):
    """The render was cooperatively cancelled."""


class RenderTimeout(RuntimeError):
    """The render exceeded its bounded wall-clock ceiling."""


class RenderProcessError(RuntimeError):
    """The FFmpeg process failed or produced no usable artifact."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.detail = detail


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


def run_compiled_render(compiled: CompiledRender, context: AttemptContext) -> RenderArtifacts:
    """Execute one compiled render, returning the artifact plus probe facts."""

    attempt_directory = context.attempt_directory
    attempt_directory.mkdir(parents=True, exist_ok=True)
    (attempt_directory / compiled.filtergraph_relative_path).write_text(
        compiled.filtergraph, encoding="utf-8"
    )
    _localize_ass(attempt_directory, compiled, context)

    poll = _DEFAULT_POLL_SECONDS
    process = subprocess.Popen(  # noqa: S603 - allow-listed argv, shell=False
        list(compiled.argv),
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
    deadline = started + context.timeout_seconds if context.timeout_seconds > 0 else None
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
    if process.returncode != 0:
        raise RenderProcessError("RENDER_PROCESS_FAILED", stderr_tail[-2048:])

    output_path = attempt_directory / compiled.output_relative_path
    if not output_path.is_file():
        raise RenderProcessError("ARTIFACT_MISSING", stderr_tail[-2048:])
    size_bytes = output_path.stat().st_size
    if size_bytes <= 0:
        raise RenderProcessError("ARTIFACT_ZERO_BYTES", stderr_tail[-2048:])

    digest = _sha256_file(output_path)
    probe = _probe_output(compiled.runtime_identity.ffprobe_binary, output_path)
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


def _probe_output(binary: str, path: Path) -> dict[str, object]:
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
            command, check=True, capture_output=True, text=True, timeout=120
        )
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
        "audio_codec": audio.get("codec_name") if isinstance(audio, Mapping) else None,
        "audio_sample_rate": _as_int(audio.get("sample_rate"), 0)
        if isinstance(audio, Mapping)
        else 0,
        "audio_channels": _as_int(audio.get("channels"), 0) if isinstance(audio, Mapping) else 0,
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
