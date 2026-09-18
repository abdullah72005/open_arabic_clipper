"""PNG-only Stage 5.1 preview/validation renders through safe FFmpeg argument arrays.

This module never encodes a video and never selects a final codec: it composes a
scaled/cropped background and burns the already-serialized ASS captions onto a
bounded set of representative source frames, writing one PNG per frame. There is
no libx264/aac/loudnorm path and no MP4 output anywhere in this module.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.composition.policy import OUTPUT_HEIGHT, OUTPUT_WIDTH, Stage51Config
from app.composition.service import PreviewError
from app.composition.types import PlannerInputs
from app.services.storage import StorageCategory, StorageService

_FORBIDDEN_ARGUMENTS = (
    "libx264",
    "libx265",
    "libvpx",
    "aac",
    "loudnorm",
    "mpeg4",
)


def render_preview_pngs(
    *,
    source_path: Path,
    inputs: PlannerInputs,
    plan_payload: Mapping[str, object],
    config: Stage51Config,
    storage: StorageService,
    output_directory: Path | None = None,
    ffmpeg_binary: str = "ffmpeg",
    max_frames: int | None = None,
) -> list[Path]:
    """Render representative PNG previews (never video) for one plan.

    Raises :class:`PreviewError` when previews are disabled, FFmpeg is missing,
    or FFmpeg fails. The caller owns the output directory when supplied.
    """

    if not config.preview_enabled:
        raise PreviewError("preview rendering is disabled")
    executable = shutil.which(ffmpeg_binary)
    if executable is None:
        raise PreviewError("ffmpeg is unavailable for preview rendering")

    destination = output_directory or _default_output_directory(storage, inputs)
    destination.mkdir(parents=True, exist_ok=True)
    ass_path = _localize_ass(plan_payload, storage, destination)
    times = _representative_times(inputs, plan_payload, max_frames or config.preview_max_frames)
    if not times:
        raise PreviewError("no representative frames were available for preview")

    outputs: list[Path] = []
    for index, source_time in enumerate(times):
        output_path = destination / f"preview-{index:04d}.png"
        if output_path.suffix.lower() != ".png":
            raise PreviewError("preview rendering only produces PNG files")
        arguments = _frame_arguments(
            executable=executable,
            source_path=source_path,
            source_time=source_time,
            output_path=output_path,
            ass_filename=ass_path.name if ass_path is not None else None,
        )
        _assert_no_video_arguments(arguments)
        try:
            subprocess.run(
                arguments,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(destination),
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise PreviewError("ffmpeg preview render failed") from error
        outputs.append(output_path)
    return outputs


def _frame_arguments(
    *,
    executable: str,
    source_path: Path,
    source_time: float,
    output_path: Path,
    ass_filename: str | None,
) -> list[str]:
    filters = [
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase",
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}",
    ]
    if ass_filename is not None:
        filters.append(f"ass={ass_filename}")
    return [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        # Preserve the source timestamps so the ASS subtitle filter can match
        # events that start after 0. Without -copyts, input seeking rebases the
        # single output frame's PTS to ~0 and late captions never activate.
        "-copyts",
        "-ss",
        f"{source_time:.4f}",
        "-i",
        str(source_path),
        "-frames:v",
        "1",
        "-vf",
        ",".join(filters),
        "-an",
        "-sn",
        "-y",
        str(output_path),
    ]


def _assert_no_video_arguments(arguments: Sequence[str]) -> None:
    joined = " ".join(arguments).casefold()
    for forbidden in _FORBIDDEN_ARGUMENTS:
        if forbidden in joined:
            raise PreviewError(f"preview renderer must never emit video: {forbidden}")


def _localize_ass(
    plan_payload: Mapping[str, object], storage: StorageService, destination: Path
) -> Path | None:
    ass = plan_payload.get("ass")
    if not isinstance(ass, Mapping):
        return None
    relative = ass.get("asset_path")
    if not isinstance(relative, str) or not relative:
        return None
    source = storage.storage_root / relative
    if not source.is_file():
        return None
    target = destination / "captions.ass"
    target.write_bytes(source.read_bytes())
    return target


def _representative_times(
    inputs: PlannerInputs, plan_payload: Mapping[str, object], limit: int
) -> list[float]:
    limit = max(1, int(limit))
    times: list[float] = []
    scenes = plan_payload.get("scenes")
    if isinstance(scenes, Sequence) and not isinstance(scenes, (str, bytes)):
        for scene in scenes:
            if isinstance(scene, Mapping):
                start = scene.get("source_start")
                if isinstance(start, (int, float)) and not isinstance(start, bool):
                    times.append(float(start))
    if not times:
        times = [span.start for span in inputs.spans]
    deduped: list[float] = []
    for value in sorted(times):
        if not deduped or abs(value - deduped[-1]) > 1e-4:
            deduped.append(value)
    return deduped[:limit]


def _default_output_directory(storage: StorageService, inputs: PlannerInputs) -> Path:
    benchmark_root = Path(storage.category_root(StorageCategory.BENCHMARKS))
    return benchmark_root / "visual-composition" / inputs.candidate_id / inputs.contract_id


__all__ = ["render_preview_pngs"]
