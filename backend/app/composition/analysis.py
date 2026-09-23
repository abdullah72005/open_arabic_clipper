"""Bounded, CPU-local, provider-free span sampling and scene-cut detection.

This is the only Stage 5.1 module that starts an external decoder. It is
deliberately narrow:

- it analyzes *only* the union of the selected bound source spans plus a bounded
  ``scene_context_seconds`` margin on each side. The margin exists solely to let
  scene-cut detection see a little lead-in/lead-out around a cut; it must never
  appear in scenes, keyframes, captions, or output ranges, and it is never
  sampled as an output frame;
- one FFmpeg child process is started *per span* and its raw RGB frames are read
  incrementally from a pipe. A whole source is never scanned;
- no PyAV, OpenCV, or PIL is ever imported; only FFmpeg CLI and ``numpy`` for the
  frame buffer are used;
- no network, no provider, no model loading.

Pinned frame time mapping (empirically verified against FFmpeg 7.x):

``source_time = span_start + sample_index / effective_fps``

Sample ``n`` of a span is produced at exactly ``span_start + n / effective_fps``:
the ``-ss`` input seek rebases FFmpeg's output timestamps to zero, and the
``fps=`` filter then emits samples at ``n / effective_fps``. The sampler does not
trust FFmpeg timestamps; it computes the mapping itself.

Post-scale frame dimensions: the scaled width is ``min(max_dimension, iw)`` and
the height uses FFmpeg's ``-2`` rounding. Because ``rawvideo`` carries no header,
the exact byte layout cannot be read from the stream. Following the documented
option, the caller passes the post-scale dimensions (normally derived once from
probed display geometry with :func:`scaled_dimensions`). A short read therefore
surfaces as an explicit error rather than a silent misalignment.
"""

from __future__ import annotations

import math
import os
import select
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from app.services.storage import StorageCategory, StorageService

Frame: TypeAlias = npt.NDArray[np.uint8]
Span: TypeAlias = tuple[int, float, float]
CancelCheck: TypeAlias = Callable[[], bool]

_EPSILON = 1e-9
_READ_CHUNK_BYTES = 1 << 20
_MAX_SCENE_OUTPUT_BYTES = 4 << 20
_REAP_TIMEOUT_SECONDS = 5.0
_MAX_REDUCTION_ITERATIONS = 64


class AnalysisScopeExceeded(RuntimeError):
    """The requested analysis footprint exceeds a hard bounded budget."""


class AnalysisCancelled(RuntimeError):
    """The caller cancelled analysis while a child process was running."""


class FrameSamplingError(RuntimeError):
    """A frame could not be produced or read from FFmpeg."""


class SceneCutDetectionError(RuntimeError):
    """Scene-cut detection could not run or could not be parsed."""


class FFmpegUnavailableError(RuntimeError):
    """The configured FFmpeg binary could not be started at all."""


@dataclass(frozen=True)
class AnalysisPlan:
    """A bounded, deterministic plan for which frames may be decoded.

    ``spans`` is the context-expanded analysis footprint: it is what the sampler
    and the scene-cut detector may read. ``total_selected_seconds`` counts only
    the selected union before context; ``total_context_seconds`` is the bounded
    extra lead-in/lead-out used solely for cut alignment.
    """

    effective_fps: float
    spans: tuple[Span, ...]
    total_selected_seconds: float
    total_context_seconds: float
    total_frames: int
    reduced_fps: bool
    reason: str


@dataclass(frozen=True)
class SampledFrame:
    """One decoded RGB frame with its exact pinned source timestamp."""

    span_block_index: int
    sample_index: int
    source_time: float
    rgb_frame: Frame


class FrameSampler(Protocol):
    """Incremental, bounded frame source used by the Stage 5.1 analyzer."""

    def samples(
        self,
        spans: Sequence[Span],
        effective_fps: float,
        max_dimension: int,
        cancel_check: CancelCheck | None = None,
    ) -> Iterator[SampledFrame]:
        """Yield one frame at a time for each span, never buffering a whole span."""


class SceneCutDetector(Protocol):
    """Bounded per-span hard-cut detector."""

    def cuts(
        self,
        path: Path | str,
        span_start: float,
        span_end: float,
        threshold: float,
        min_scene_seconds: float,
    ) -> tuple[float, ...]:
        """Return sorted source-time hard cuts inside the span."""


# --- Pure planning helpers --------------------------------------------------


def _normalize_span(span: Span) -> Span:
    block_index, start, end = span
    start_value = float(start)
    end_value = float(end)
    if not (math.isfinite(start_value) and math.isfinite(end_value)):
        raise ValueError("span bounds must be finite")
    if start_value < 0.0:
        raise ValueError("span start must be non-negative")
    if end_value <= start_value:
        raise ValueError("span end must be greater than span start")
    return (int(block_index), round(start_value, 6), round(end_value, 6))


def merge_spans(spans: Sequence[Span]) -> tuple[Span, ...]:
    """Return the union footprint, merging overlapping spans per block index."""

    normalized = sorted((_normalize_span(span) for span in spans), key=lambda s: (s[0], s[1], s[2]))
    merged: list[Span] = []
    for block_index, start, end in normalized:
        if merged and merged[-1][0] == block_index and start <= merged[-1][2] + _EPSILON:
            last_block, last_start, last_end = merged[-1]
            merged[-1] = (last_block, last_start, max(last_end, end))
            continue
        merged.append((block_index, start, end))
    return tuple(merged)


def expand_spans_with_context(
    spans: Sequence[Span],
    scene_context_seconds: float,
) -> tuple[Span, ...]:
    """Expand each span by a bounded context margin used only for cut alignment."""

    if not math.isfinite(scene_context_seconds) or scene_context_seconds < 0.0:
        raise ValueError("scene context seconds must be finite and non-negative")
    expanded = [
        (block_index, max(0.0, start - scene_context_seconds), end + scene_context_seconds)
        for block_index, start, end in spans
    ]
    return merge_spans(expanded)


def _span_frame_count(duration: float, fps: float) -> int:
    if duration <= 0.0 or fps <= 0.0:
        return 0
    return max(0, int(math.ceil(duration * fps - _EPSILON)))


def _frames_for_spans(spans: Sequence[Span], fps: float) -> int:
    return sum(_span_frame_count(end - start, fps) for _, start, end in spans)


def plan_analysis(
    spans: Sequence[Span],
    *,
    analysis_fps: float,
    max_analysis_seconds: float,
    max_analysis_frames: int,
    min_fps: float = 1.0,
    scene_context_seconds: float = 0.0,
) -> AnalysisPlan:
    """Compute a bounded decode plan for the union of the selected spans.

    The selected union is always checked against ``max_analysis_seconds``. If the
    context-expanded footprint would exceed ``max_analysis_frames`` at
    ``analysis_fps``, the effective fps is deterministically reduced (never below
    ``min_fps``); if even the floor cannot fit the budget,
    :class:`AnalysisScopeExceeded` is raised.
    """

    if not math.isfinite(analysis_fps) or analysis_fps <= 0.0:
        raise ValueError("analysis_fps must be finite and positive")
    if not math.isfinite(min_fps) or min_fps <= 0.0:
        raise ValueError("min_fps must be finite and positive")
    if not math.isfinite(max_analysis_seconds) or max_analysis_seconds < 0.0:
        raise ValueError("max_analysis_seconds must be finite and non-negative")
    if max_analysis_frames <= 0:
        raise ValueError("max_analysis_frames must be positive")

    selected = merge_spans(spans)
    total_selected = sum(end - start for _, start, end in selected)
    if total_selected > max_analysis_seconds + _EPSILON:
        raise AnalysisScopeExceeded(
            f"selected analysis seconds {total_selected} exceed "
            f"max_analysis_seconds={max_analysis_seconds}"
        )

    footprint = expand_spans_with_context(selected, scene_context_seconds)
    footprint_seconds = sum(end - start for _, start, end in footprint)
    total_context = max(0.0, footprint_seconds - total_selected)

    effective_fps = float(analysis_fps)
    total_frames = _frames_for_spans(footprint, effective_fps)
    reduced = False
    reason = ""

    if total_frames > max_analysis_frames:
        reduced = True
        iterations = 0
        while total_frames > max_analysis_frames and effective_fps > min_fps + _EPSILON:
            scalar = max_analysis_frames / max(1, total_frames)
            candidate = max(min_fps, effective_fps * scalar)
            candidate = math.floor(candidate * 1_000_000.0) / 1_000_000.0
            if candidate >= effective_fps - _EPSILON:
                candidate = min_fps
            effective_fps = candidate
            total_frames = _frames_for_spans(footprint, effective_fps)
            iterations += 1
            if iterations >= _MAX_REDUCTION_ITERATIONS:
                break
        if total_frames > max_analysis_frames:
            raise AnalysisScopeExceeded(
                f"analysis footprint of {footprint_seconds} s needs more than "
                f"max_analysis_frames={max_analysis_frames} even at min_fps={min_fps}"
            )
        reason = (
            f"reduced effective fps from {analysis_fps} to {effective_fps} "
            f"to respect max_analysis_frames={max_analysis_frames}"
        )

    return AnalysisPlan(
        effective_fps=effective_fps,
        spans=footprint,
        total_selected_seconds=round(total_selected, 6),
        total_context_seconds=round(total_context, 6),
        total_frames=total_frames,
        reduced_fps=reduced,
        reason=reason,
    )


def scaled_dimensions(source_width: int, source_height: int, max_dimension: int) -> tuple[int, int]:
    """Mirror FFmpeg's ``scale='min(maxdim,iw)':-2`` post-scale dimensions.

    ``-2`` rounds the derived height to the nearest even integer so the frame
    stays 4:2:0-friendly.
    """

    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive")
    width = min(max_dimension, source_width)
    raw_height = int(round(source_height * width / source_width))
    height = raw_height if raw_height % 2 == 0 else raw_height + 1
    return (width, max(height, 2))


def merge_cuts(
    cuts: Sequence[float],
    span_start: float,
    span_end: float,
    min_scene_seconds: float,
) -> tuple[float, ...]:
    """Clamp cuts to a span, drop out-of-range cuts, and merge close cuts."""

    if span_end <= span_start:
        raise ValueError("span end must be greater than span start")
    if not math.isfinite(min_scene_seconds) or min_scene_seconds < 0.0:
        raise ValueError("min_scene_seconds must be finite and non-negative")
    kept: list[float] = []
    for cut in sorted({float(value) for value in cuts}):
        if not math.isfinite(cut):
            continue
        if cut < span_start - _EPSILON or cut > span_end + _EPSILON:
            continue
        clamped = min(max(cut, span_start), span_end)
        if kept and clamped - kept[-1] < min_scene_seconds - _EPSILON:
            continue
        kept.append(clamped)
    return tuple(kept)


def segment_scenes(
    span_start: float,
    span_end: float,
    cuts: Sequence[float],
    *,
    min_scene_seconds: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """Turn interior cuts into contiguous scenes covering ``[span_start, span_end]``.

    Cuts outside the span are ignored. Scenes shorter than ``min_scene_seconds``
    are deterministically merged with their shortest-duration neighbour until
    every scene meets the minimum (or only one scene remains).
    """

    if not (math.isfinite(span_start) and math.isfinite(span_end)):
        raise ValueError("span bounds must be finite")
    if span_end <= span_start:
        raise ValueError("span end must be greater than span start")
    if not math.isfinite(min_scene_seconds) or min_scene_seconds < 0.0:
        raise ValueError("min_scene_seconds must be finite and non-negative")

    boundaries = [span_start]
    interior = sorted(
        {
            float(cut)
            for cut in cuts
            if math.isfinite(cut) and span_start + _EPSILON < float(cut) < span_end - _EPSILON
        }
    )
    boundaries.extend(interior)
    boundaries.append(span_end)

    scenes = [[boundaries[index], boundaries[index + 1]] for index in range(len(boundaries) - 1)]
    while len(scenes) > 1:
        shortest = min(range(len(scenes)), key=lambda i: (scenes[i][1] - scenes[i][0], i))
        if scenes[shortest][1] - scenes[shortest][0] >= min_scene_seconds - _EPSILON:
            break
        if shortest == 0:
            neighbour = 1
        elif shortest == len(scenes) - 1:
            neighbour = shortest - 1
        else:
            previous = scenes[shortest - 1][1] - scenes[shortest - 1][0]
            following = scenes[shortest + 1][1] - scenes[shortest + 1][0]
            neighbour = shortest - 1 if previous <= following else shortest + 1
        merge_index = min(shortest, neighbour)
        remove_index = max(shortest, neighbour)
        scenes[merge_index] = [scenes[merge_index][0], scenes[remove_index][1]]
        del scenes[remove_index]

    return tuple((round(start, 6), round(end, 6)) for start, end in scenes)


# --- FFmpeg frame sampler ---------------------------------------------------


class FFmpegFrameSampler:
    """One FFmpeg child process per span, reading raw RGB frames incrementally."""

    def __init__(
        self,
        path: Path | str,
        *,
        frame_size: tuple[int, int],
        ffmpeg_binary: str = "ffmpeg",
        storage: StorageService | None = None,
        temporary_root: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame_size dimensions must be positive")
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        self._path = Path(path)
        self._frame_size = (int(width), int(height))
        self._ffmpeg_binary = ffmpeg_binary
        self._timeout_seconds = timeout_seconds
        if temporary_root is not None:
            self._temporary_root: Path | None = temporary_root
        elif storage is not None:
            self._temporary_root = storage.category_root(StorageCategory.TEMPORARY)
        else:
            self._temporary_root = None

    def samples(
        self,
        spans: Sequence[Span],
        effective_fps: float,
        max_dimension: int,
        cancel_check: CancelCheck | None = None,
    ) -> Iterator[SampledFrame]:
        """Yield one frame at a time for every span in order."""

        if not math.isfinite(effective_fps) or effective_fps <= 0.0:
            raise ValueError("effective_fps must be finite and positive")
        for block_index, start, end in merge_spans(spans):
            yield from self._sample_span(
                block_index, start, end, effective_fps, max_dimension, cancel_check
            )

    def _sample_span(
        self,
        block_index: int,
        start: float,
        end: float,
        effective_fps: float,
        max_dimension: int,
        cancel_check: CancelCheck | None,
    ) -> Iterator[SampledFrame]:
        duration = end - start
        if duration <= 0.0:
            return
        width, height = self._frame_size
        frame_bytes = width * height * 3
        args = [
            self._ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            _format_seconds(start),
            "-i",
            str(self._path),
            "-t",
            _format_seconds(duration),
            "-vf",
            _video_filter(effective_fps, max_dimension),
            "-an",
            "-sn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]

        stderr_file: IO[bytes] | None = None
        stderr_path: Path | None = None
        stderr_target: int | IO[bytes]
        if self._temporary_root is not None:
            self._temporary_root.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix="stage51-frames-", suffix=".log", dir=str(self._temporary_root)
            )
            stderr_file = os.fdopen(descriptor, "wb")
            stderr_path = Path(name)
            stderr_target = stderr_file
        else:
            stderr_target = subprocess.DEVNULL

        process: subprocess.Popen[bytes] | None = None
        try:
            try:
                process = subprocess.Popen(
                    args,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=stderr_target,
                )
            except OSError as error:
                raise FFmpegUnavailableError("ffmpeg is unavailable for frame sampling") from error

            stdout = process.stdout
            if stdout is None:  # pragma: no cover - PIPE always yields a stream
                raise FrameSamplingError("ffmpeg frame pipe is unavailable")
            file_descriptor = stdout.fileno()
            deadline = (
                None if self._timeout_seconds is None else _monotonic() + self._timeout_seconds
            )
            sample_index = 0
            while True:
                if cancel_check is not None and cancel_check():
                    raise AnalysisCancelled("frame sampling cancelled")
                payload = _read_exact(file_descriptor, frame_bytes, deadline, cancel_check)
                if payload is None:
                    break
                frame: Frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
                yield SampledFrame(
                    span_block_index=block_index,
                    sample_index=sample_index,
                    source_time=start + sample_index / effective_fps,
                    rgb_frame=cast(Frame, np.ascontiguousarray(frame, dtype=np.uint8)),
                )
                sample_index += 1

            return_code = process.wait()
            if return_code != 0:
                raise FrameSamplingError(
                    f"ffmpeg frame sampling failed with code {return_code}: "
                    f"{_stderr_text(stderr_path)}"
                )
        finally:
            if process is not None:
                _terminate_process(process)
            if stderr_file is not None:
                try:
                    stderr_file.close()
                except OSError:  # pragma: no cover - defensive close
                    pass
            if stderr_path is not None:
                stderr_path.unlink(missing_ok=True)


def _format_seconds(value: float) -> str:
    return f"{value:.6f}"


def _video_filter(effective_fps: float, max_dimension: int) -> str:
    fps_text = f"{effective_fps:.6f}".rstrip("0").rstrip(".")
    return f"fps={fps_text},scale='min({int(max_dimension)},iw)':-2"


def _monotonic() -> float:
    import time

    return time.monotonic()


def _read_exact(
    file_descriptor: int,
    size: int,
    deadline: float | None,
    cancel_check: CancelCheck | None,
) -> bytes | None:
    """Read exactly ``size`` bytes, returning None only at a clean frame boundary."""

    buffer = bytearray()
    while len(buffer) < size:
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelled("frame sampling cancelled")
        timeout: float | None = None
        if deadline is not None:
            timeout = deadline - _monotonic()
            if timeout <= 0.0:
                raise FrameSamplingError("timed out reading ffmpeg frame data")
        try:
            ready, _, _ = select.select([file_descriptor], [], [], timeout)
        except InterruptedError:  # pragma: no cover - signal interruption
            continue
        if not ready:
            raise FrameSamplingError("timed out reading ffmpeg frame data")
        chunk = os.read(file_descriptor, min(size - len(buffer), _READ_CHUNK_BYTES))
        if not chunk:
            if not buffer:
                return None
            raise FrameSamplingError("ffmpeg produced a truncated frame")
        buffer.extend(chunk)
    return bytes(buffer)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            process.kill()
        except OSError:  # pragma: no cover - already reaped
            pass
    try:
        process.wait(timeout=_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive reap
        pass
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            try:
                stream.close()
            except OSError:  # pragma: no cover - defensive close
                pass


def _stderr_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()[-500:]
    except OSError:  # pragma: no cover - defensive read
        return ""


# --- FFmpeg scene-cut detector ----------------------------------------------


class FFmpegSceneCutDetector:
    """One bounded FFmpeg pass per span using ``select`` + ``metadata=print``."""

    def __init__(
        self, *, ffmpeg_binary: str = "ffmpeg", timeout_seconds: float | None = None
    ) -> None:
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        self._ffmpeg_binary = ffmpeg_binary
        self._timeout_seconds = timeout_seconds

    def cuts(
        self,
        path: Path | str,
        span_start: float,
        span_end: float,
        threshold: float = 0.35,
        min_scene_seconds: float = 0.4,
        cancel_check: CancelCheck | None = None,
    ) -> tuple[float, ...]:
        """Return sorted source-time cuts inside ``[span_start, span_end]``."""

        if not (math.isfinite(span_start) and math.isfinite(span_end)):
            raise ValueError("span bounds must be finite")
        if span_end <= span_start:
            raise ValueError("span end must be greater than span start")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")

        args = [
            self._ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            _format_seconds(span_start),
            "-i",
            str(path),
            "-t",
            _format_seconds(span_end - span_start),
            "-vf",
            f"select='gt(scene,{threshold})',metadata=mode=print:file=-",
            "-an",
            "-sn",
            "-f",
            "null",
            "-",
        ]
        process: subprocess.Popen[bytes] | None = None
        try:
            try:
                process = subprocess.Popen(
                    args,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as error:
                raise FFmpegUnavailableError(
                    "ffmpeg is unavailable for scene-cut detection"
                ) from error

            stdout = process.stdout
            if stdout is None:  # pragma: no cover - PIPE always yields a stream
                raise SceneCutDetectionError("ffmpeg scene-cut pipe is unavailable")
            deadline = (
                None if self._timeout_seconds is None else _monotonic() + self._timeout_seconds
            )
            output = _read_stream(stdout.fileno(), deadline, cancel_check)
            return_code = process.wait()
            if return_code != 0:
                raise SceneCutDetectionError(
                    f"ffmpeg scene-cut detection failed with code {return_code}"
                )
        finally:
            if process is not None:
                _terminate_process(process)

        relative_cuts = parse_scene_cut_times(output)
        source_cuts = tuple(span_start + value for value in relative_cuts)
        return merge_cuts(source_cuts, span_start, span_end, min_scene_seconds)


def _read_stream(
    file_descriptor: int,
    deadline: float | None,
    cancel_check: CancelCheck | None,
) -> str:
    buffer = bytearray()
    while True:
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelled("scene-cut detection cancelled")
        timeout: float | None = None
        if deadline is not None:
            timeout = deadline - _monotonic()
            if timeout <= 0.0:
                raise SceneCutDetectionError("timed out during scene-cut detection")
        try:
            ready, _, _ = select.select([file_descriptor], [], [], timeout)
        except InterruptedError:  # pragma: no cover - signal interruption
            continue
        if not ready:
            raise SceneCutDetectionError("timed out during scene-cut detection")
        chunk = os.read(file_descriptor, _READ_CHUNK_BYTES)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > _MAX_SCENE_OUTPUT_BYTES:
            raise SceneCutDetectionError("scene-cut metadata output exceeded its bound")
    return buffer.decode("utf-8", errors="replace")


def parse_scene_cut_times(output: str) -> tuple[float, ...]:
    """Parse ``pts_time`` values from ``metadata=print`` frame log lines."""

    times: set[float] = set()
    marker = "pts_time:"
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("frame:"):
            continue
        index = stripped.find(marker)
        if index == -1:
            continue
        token = stripped[index + len(marker) :].split(maxsplit=1)[0]
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value):
            times.add(value)
    return tuple(sorted(times))


__all__ = [
    "AnalysisCancelled",
    "AnalysisPlan",
    "AnalysisScopeExceeded",
    "FFmpegFrameSampler",
    "FFmpegSceneCutDetector",
    "FFmpegUnavailableError",
    "Frame",
    "FrameSampler",
    "FrameSamplingError",
    "SampledFrame",
    "SceneCutDetectionError",
    "SceneCutDetector",
    "Span",
    "expand_spans_with_context",
    "merge_cuts",
    "merge_spans",
    "parse_scene_cut_times",
    "plan_analysis",
    "scaled_dimensions",
    "segment_scenes",
]
