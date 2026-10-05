"""Deterministic technical QC for Stage 5.2 render artifacts.

QC is evidence, not subjective quality. It never claims lip-sync correctness,
readability, good cropping, or visual taste. Hard checks are structural
(streams/codecs/geometry/duration/counts); appearance checks are bounded
low-resolution samples and remain warnings unless source evidence clearly
disagrees.
"""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from app.render.execution.policy import (
    QC_BLACK_FRAME_FRACTION_FAIL,
    QC_BLACK_FRAME_FRACTION_WARN,
    QC_BLACK_LUMA_THRESHOLD,
    QC_FAIL,
    QC_FROZEN_DIFF_THRESHOLD,
    QC_FROZEN_FRAME_FRACTION_FAIL,
    QC_FROZEN_FRAME_FRACTION_WARN,
    QC_PASS,
    QC_PEAK_DBFS,
    QC_POLICY_VERSION,
    QC_SILENCE_DBFS,
    QC_TIMING_AUDIO_TOLERANCE_SECONDS,
    QC_TIMING_BASE_TOLERANCE_SECONDS,
    QC_WARN,
    Stage52Config,
)
from app.render.execution.types import QCCheck, RenderArtifacts, TechnicalQCResult

_SAMPLE_W = 160
_SAMPLE_H = 284
_VOLUME_RE = re.compile(r"(mean|max)_volume:\s*(-?[\d.]+|inf|-inf)\s*dB")


@dataclass(frozen=True)
class _SampledFrame:
    output_time: float
    source_time: float | None
    luma: bytes


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _timing_tolerance(frame_rate: Fraction) -> float:
    fps = float(frame_rate) if frame_rate > 0 else 30.0
    return max(
        QC_TIMING_BASE_TOLERANCE_SECONDS,
        2.0 / fps + QC_TIMING_AUDIO_TOLERANCE_SECONDS,
    )


def _extract_luma(binary: str, path: Path, time_s: float) -> bytes | None:
    command = [
        binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, time_s):.4f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={_SAMPLE_W}:{_SAMPLE_H}:flags=bilinear,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - allow-listed argv
            command, check=True, capture_output=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    data = completed.stdout
    if len(data) < _SAMPLE_W * _SAMPLE_H:
        return None
    return data[: _SAMPLE_W * _SAMPLE_H]


def _mean_luma(frame: bytes, start_row: int, end_row: int) -> float:
    start = start_row * _SAMPLE_W
    end = end_row * _SAMPLE_W
    band = frame[start:end]
    if not band:
        return 0.0
    return sum(band) / len(band)


def _center_band() -> tuple[int, int]:
    return int(_SAMPLE_H * 0.18), int(_SAMPLE_H * 0.55)


def _frame_diff(first: bytes, second: bytes) -> float:
    start, end = _center_band()
    a = first[start * _SAMPLE_W : end * _SAMPLE_W]
    b = second[start * _SAMPLE_W : end * _SAMPLE_W]
    if not a or len(a) != len(b):
        return 0.0
    total = sum(abs(x - y) for x, y in zip(a, b))
    return total / len(a)


def _volume_metrics(
    binary: str, path: Path, start: float = 0.0, end: float | None = None
) -> dict[str, float] | None:
    command = [binary, "-nostdin", "-hide_banner", "-loglevel", "info"]
    if start > 0 or end is not None:
        command += [
            "-ss",
            f"{max(0.0, start):.4f}",
            "-t",
            f"{max(0.05, (end or start) - start):.4f}",
        ]
    command += ["-i", str(path), "-af", "volumedetect", "-f", "null", "-"]
    try:
        completed = subprocess.run(  # noqa: S603 - allow-listed argv
            command, check=False, capture_output=True, text=True, timeout=180
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = completed.stderr or ""
    metrics: dict[str, float] = {}
    for name, value in _VOLUME_RE.findall(text):
        if value in {"inf", "+inf"}:
            metrics[name] = 0.0
        elif value == "-inf":
            metrics[name] = -math.inf
        else:
            try:
                metrics[name] = float(value)
            except ValueError:
                continue
    return metrics or None


def _sample_times(manifest: Mapping[str, object], limit: int) -> list[tuple[float, float | None]]:
    timeline = manifest.get("timeline")
    occurrences = []
    if isinstance(timeline, Mapping):
        raw = timeline.get("occurrences")
        if isinstance(raw, list):
            occurrences = [item for item in raw if isinstance(item, Mapping)]
    samples: list[tuple[float, float | None]] = []
    for occurrence in occurrences:
        out_start = _as_float(occurrence.get("output_start"))
        out_end = _as_float(occurrence.get("output_end"))
        src_start = _as_float(occurrence.get("source_start"))
        duration = max(0.0, out_end - out_start)
        if duration <= 0:
            continue
        points = [
            out_start + 0.2 * duration,
            out_start + 0.5 * duration,
            out_start + 0.8 * duration,
        ]
        for point in points:
            source_time = src_start + (point - out_start)
            samples.append((point, source_time))
    if not samples:
        return []
    samples.sort(key=lambda item: item[0])
    if len(samples) > limit:
        step = len(samples) / limit
        samples = [samples[int(index * step)] for index in range(limit)]
    return samples


def check_render_artifact(
    artifacts: RenderArtifacts,
    manifest: Mapping[str, object],
    qc_policy: Stage52Config,
    *,
    source_path: Path | None = None,
) -> TechnicalQCResult:
    """Run deterministic technical QC over a produced artifact."""

    checks: list[QCCheck] = []
    reasons: list[str] = []
    measured: dict[str, object] = {}

    def record(
        name: str, ok: bool, reason: str | None = None, *, warn_only: bool = False, **data: object
    ) -> None:
        if ok:
            checks.append(QCCheck(name=name, status="PASS"))
            return
        status = QC_WARN if warn_only else QC_FAIL
        checks.append(QCCheck(name=name, status=status, reason_code=reason, measured=dict(data)))
        if reason is not None:
            reasons.append(reason)

    expected = manifest.get("expected")
    expected = expected if isinstance(expected, Mapping) else {}
    probe = artifacts.probe

    record("artifact_present", artifacts.size_bytes > 0, "ARTIFACT_ZERO_BYTES")
    record("probe_succeeded", bool(probe), "QC_PROBE_FAILED")

    streams = probe.get("streams") if isinstance(probe.get("streams"), Mapping) else {}
    video_count = _as_int(streams.get("video"), 0) if isinstance(streams, Mapping) else 0
    audio_count = _as_int(streams.get("audio"), 0) if isinstance(streams, Mapping) else 0
    subtitle_count = _as_int(streams.get("subtitle"), 0) if isinstance(streams, Mapping) else 0
    data_count = _as_int(streams.get("data"), 0) if isinstance(streams, Mapping) else 0
    record("single_video_stream", video_count == 1, "QC_STREAMS_MISMATCH", video=video_count)
    record("single_audio_stream", audio_count == 1, "QC_STREAMS_MISMATCH", audio=audio_count)
    record(
        "no_subtitle_stream",
        subtitle_count == 0,
        "QC_STREAMS_MISMATCH",
        warn_only=True,
        subtitle=subtitle_count,
    )
    record(
        "no_data_stream", data_count == 0, "QC_STREAMS_MISMATCH", warn_only=True, data=data_count
    )

    record(
        "video_codec",
        str(probe.get("video_codec") or "")
        == str(expected.get("video_codec") or probe.get("video_codec")),
        "QC_CODEC_MISMATCH",
        measured_codec=probe.get("video_codec"),
        expected_codec=expected.get("video_codec"),
    )
    record(
        "pixel_format",
        str(probe.get("pix_fmt") or "")
        == str(expected.get("pixel_format") or probe.get("pix_fmt")),
        "QC_PIXEL_FORMAT_MISMATCH",
        measured_pix_fmt=probe.get("pix_fmt"),
    )
    record(
        "audio_codec",
        str(probe.get("audio_codec") or "")
        == str(expected.get("audio_codec") or probe.get("audio_codec")),
        "QC_CODEC_MISMATCH",
        measured_codec=probe.get("audio_codec"),
    )

    width = _as_int(probe.get("width"), 0)
    height = _as_int(probe.get("height"), 0)
    record(
        "output_geometry",
        width == _as_int(expected.get("width"), width)
        and height == _as_int(expected.get("height"), height),
        "QC_GEOMETRY_MISMATCH",
        measured_width=width,
        measured_height=height,
    )
    sar = str(probe.get("sample_aspect_ratio") or "1:1")
    record(
        "square_pixels",
        sar in {"1:1", "1", "N/A", ""},
        "QC_GEOMETRY_MISMATCH",
        sample_aspect_ratio=sar,
    )
    rotation = _as_int(probe.get("rotation_degrees"), 0) % 360
    record(
        "rotation_state",
        rotation == _as_int(expected.get("rotation_degrees"), 0),
        "QC_ROTATION_MISMATCH",
        measured_rotation=rotation,
    )

    frame_rate = Fraction(
        _as_int(expected.get("frame_rate", {}).get("numerator"), 30)
        if isinstance(expected.get("frame_rate"), Mapping)
        else 30,
        _as_int(expected.get("frame_rate", {}).get("denominator"), 1)
        if isinstance(expected.get("frame_rate"), Mapping)
        else 1,
    )
    tolerance = _timing_tolerance(frame_rate)
    duration = artifacts.duration_seconds
    expected_duration = _as_float(expected.get("output_duration"), duration)
    record(
        "duration_plausible",
        math.isfinite(duration) and duration > 0.0,
        "QC_DURATION_MISMATCH",
        duration=duration,
    )
    record(
        "duration_matches_timeline",
        abs(duration - expected_duration) <= tolerance,
        "QC_DURATION_MISMATCH",
        measured=duration,
        expected=expected_duration,
        tolerance=tolerance,
    )

    actual_frames = artifacts.frame_count
    expected_frames = _as_int(expected.get("frame_count"), actual_frames)
    frame_tolerance = max(2, int(round(0.10 * float(frame_rate))))
    if actual_frames > 0:
        record(
            "frame_count_matches",
            abs(actual_frames - expected_frames) <= frame_tolerance,
            "QC_FRAME_COUNT_MISMATCH",
            measured=actual_frames,
            expected=expected_frames,
            tolerance=frame_tolerance,
        )
    else:
        checks.append(
            QCCheck(
                name="frame_count_matches", status="WARN", reason_code="QC_FRAME_COUNT_MISMATCH"
            )
        )

    sample_rate = artifacts.sample_rate
    record(
        "audio_sample_rate",
        sample_rate == _as_int(expected.get("audio_sample_rate"), sample_rate),
        "QC_SAMPLE_RATE_MISMATCH",
        measured=sample_rate,
    )
    expected_channels = _as_int(expected.get("channels"), artifacts.channels)
    record(
        "audio_channels",
        artifacts.channels == expected_channels,
        "QC_CHANNEL_LAYOUT_MISMATCH",
        measured=artifacts.channels,
        expected=expected_channels,
    )

    measured["duration_seconds"] = duration
    measured["frame_count"] = actual_frames
    measured["sample_rate"] = sample_rate
    measured["channels"] = artifacts.channels

    # Bounded appearance/audio sampling.
    _appearance_checks(checks, reasons, measured, artifacts, manifest, qc_policy, source_path)

    status = QC_PASS
    if any(check.status == QC_FAIL for check in checks):
        status = QC_FAIL
    elif any(check.status == QC_WARN for check in checks):
        status = QC_WARN
    return TechnicalQCResult(
        status=status,
        checks=tuple(checks),
        reason_codes=tuple(dict.fromkeys(reasons)),
        measured=measured,
        policy_version=QC_POLICY_VERSION,
    )


def _appearance_checks(
    checks: list[QCCheck],
    reasons: list[str],
    measured: dict[str, object],
    artifacts: RenderArtifacts,
    manifest: Mapping[str, object],
    qc_policy: Stage52Config,
    source_path: Path | None,
) -> None:
    times = _sample_times(manifest, max(2, qc_policy.qc_max_sampled_frames))
    if not times:
        checks.append(QCCheck(name="decode_sampled_frames", status="PASS", measured={"samples": 0}))
        return
    frames: list[_SampledFrame] = []
    decode_ok = 0
    for output_time, source_time in times:
        luma = _extract_luma(qc_policy.ffmpeg_binary, artifacts.output_path, output_time)
        if luma is None:
            continue
        decode_ok += 1
        frames.append(_SampledFrame(output_time, source_time, luma))
    if decode_ok == 0:
        checks.append(
            QCCheck(name="decode_sampled_frames", status=QC_FAIL, reason_code="QC_DECODE_FAILED")
        )
        reasons.append("QC_DECODE_FAILED")
        return
    checks.append(
        QCCheck(name="decode_sampled_frames", status="PASS", measured={"decoded": decode_ok})
    )

    source_frames: list[bytes] = []
    if source_path is not None:
        for _output_time, source_time in times:
            if source_time is None:
                continue
            source_luma = _extract_luma(qc_policy.ffmpeg_binary, source_path, source_time)
            if source_luma is not None:
                source_frames.append(source_luma)

    means = [_mean_luma(frame.luma, 0, _SAMPLE_H) for frame in frames]
    black_fraction = sum(1 for value in means if value < QC_BLACK_LUMA_THRESHOLD) / len(means)
    source_means = (
        [_mean_luma(frame, 0, _SAMPLE_H) for frame in source_frames] if source_frames else []
    )
    source_black_fraction = (
        sum(1 for value in source_means if value < QC_BLACK_LUMA_THRESHOLD) / len(source_means)
        if source_means
        else None
    )
    measured["black_frame_fraction"] = round(black_fraction, 4)
    if source_black_fraction is not None:
        measured["source_black_frame_fraction"] = round(source_black_fraction, 4)
    if black_fraction >= QC_BLACK_FRAME_FRACTION_FAIL:
        if source_black_fraction is not None and source_black_fraction >= 0.5:
            checks.append(
                QCCheck(
                    name="not_blank",
                    status=QC_WARN,
                    reason_code="QC_BLANK_RENDER",
                    measured={"black_fraction": black_fraction},
                )
            )
            reasons.append("QC_BLANK_RENDER")
        else:
            checks.append(
                QCCheck(
                    name="not_blank",
                    status=QC_FAIL,
                    reason_code="QC_BLANK_RENDER",
                    measured={"black_fraction": black_fraction},
                )
            )
            reasons.append("QC_BLANK_RENDER")
    elif black_fraction >= QC_BLACK_FRAME_FRACTION_WARN:
        checks.append(
            QCCheck(
                name="not_blank",
                status=QC_WARN,
                reason_code="QC_BLANK_RENDER",
                measured={"black_fraction": black_fraction},
            )
        )
        reasons.append("QC_BLANK_RENDER")
    else:
        checks.append(QCCheck(name="not_blank", status="PASS"))

    diffs = [
        _frame_diff(frames[index].luma, frames[index + 1].luma) for index in range(len(frames) - 1)
    ]
    frozen_fraction = (
        sum(1 for value in diffs if value < QC_FROZEN_DIFF_THRESHOLD) / len(diffs) if diffs else 0.0
    )
    source_diffs = [
        _frame_diff(source_frames[index], source_frames[index + 1])
        for index in range(len(source_frames) - 1)
    ]
    source_changes = (
        any(value > QC_FROZEN_DIFF_THRESHOLD for value in source_diffs) if source_diffs else None
    )
    measured["frozen_frame_fraction"] = round(frozen_fraction, 4)
    if frozen_fraction >= QC_FROZEN_FRAME_FRACTION_FAIL and source_changes is True:
        checks.append(
            QCCheck(
                name="not_frozen",
                status=QC_FAIL,
                reason_code="QC_FROZEN_RENDER",
                measured={"frozen_fraction": frozen_fraction},
            )
        )
        reasons.append("QC_FROZEN_RENDER")
    elif frozen_fraction >= QC_FROZEN_FRAME_FRACTION_WARN:
        checks.append(
            QCCheck(
                name="not_frozen",
                status=QC_WARN,
                reason_code="QC_FROZEN_RENDER",
                measured={"frozen_fraction": frozen_fraction},
            )
        )
        reasons.append("QC_FROZEN_RENDER")
    else:
        checks.append(QCCheck(name="not_frozen", status="PASS"))

    output_volume = _volume_metrics(qc_policy.ffmpeg_binary, artifacts.output_path)
    if output_volume is not None:
        measured["output_mean_volume_db"] = output_volume.get("mean")
        measured["output_max_volume_db"] = output_volume.get("max")
        max_volume = output_volume.get("max", -math.inf)
        mean_volume = output_volume.get("mean", -math.inf)
        if mean_volume <= QC_SILENCE_DBFS:
            source_volume = (
                _volume_metrics(qc_policy.ffmpeg_binary, source_path)
                if source_path is not None
                else None
            )
            source_mean = source_volume.get("mean", -math.inf) if source_volume else None
            if source_mean is not None and source_mean > QC_SILENCE_DBFS:
                checks.append(
                    QCCheck(
                        name="audio_not_silent",
                        status=QC_FAIL,
                        reason_code="QC_TOTAL_SILENCE",
                        measured={"mean": mean_volume},
                    )
                )
                reasons.append("QC_TOTAL_SILENCE")
            else:
                checks.append(
                    QCCheck(
                        name="audio_not_silent",
                        status=QC_WARN,
                        reason_code="QC_TOTAL_SILENCE",
                        measured={"mean": mean_volume},
                    )
                )
                reasons.append("QC_TOTAL_SILENCE")
        else:
            checks.append(QCCheck(name="audio_not_silent", status="PASS"))
        if isinstance(max_volume, float) and max_volume >= QC_PEAK_DBFS:
            checks.append(
                QCCheck(
                    name="audio_peak",
                    status=QC_WARN,
                    reason_code="QC_PEAK_CLIPPING_RISK",
                    measured={"max": max_volume},
                )
            )
            reasons.append("QC_PEAK_CLIPPING_RISK")
        else:
            checks.append(QCCheck(name="audio_peak", status="PASS"))
    else:
        checks.append(
            QCCheck(name="audio_not_silent", status=QC_WARN, reason_code="QC_PROBE_FAILED")
        )


__all__ = ["check_render_artifact"]
