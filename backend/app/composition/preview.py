"""PNG-only Stage 5.1 preview/validation renders through safe FFmpeg argument arrays.

This module never encodes a video and never selects a final codec: it composes
the plan's *real* framing at each representative timestamp (interpolated crop
keyframes for crop modes, planned contain-over-blurred-self for
``BACKGROUND_FILL``, bounded scale/pad for ``SOURCE_AS_IS``) and burns the
already-serialized ASS captions onto the result, writing one PNG per frame.
There is no libx264/aac/loudnorm path and no MP4 output anywhere in this module.

FFmpeg autorotates decoded frames from container/side-data metadata by default,
so the plan's display-oriented geometry maps directly onto the decoded frame.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.composition.geometry import clamp_crop, normalized_crop_to_display
from app.composition.policy import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    FramingMode,
    Stage51Config,
    framing_for_bounded_distance,
)
from app.composition.service import PreviewError
from app.composition.types import DisplayGeometry, PlannerInputs
from app.services.storage import StorageCategory, StorageService

_FORBIDDEN_ARGUMENTS = (
    "libx264",
    "libx265",
    "libvpx",
    "aac",
    "loudnorm",
    "mpeg4",
)

BACKGROUND_FILL_BLUR_FILTER = "gblur=sigma=36:steps=2"
BACKGROUND_FILL_TONE_FILTER = "eq=brightness=-0.18:saturation=0.70"
BACKGROUND_FILL_BACKGROUND_FILTER = f"{BACKGROUND_FILL_BLUR_FILTER},{BACKGROUND_FILL_TONE_FILTER}"

_CROP_MODES = frozenset(
    {
        FramingMode.STATIC_CROP.value,
        FramingMode.TRACKED_CROP.value,
        FramingMode.MULTI_SUBJECT_FIT.value,
        FramingMode.CENTER_FALLBACK.value,
    }
)


@dataclass(frozen=True)
class PreviewCrop:
    """The plan's exact composition at one source-local timestamp."""

    source_time: float
    mode: str
    center_x: float
    center_y: float
    height_fraction: float
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    display_width: int
    display_height: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source_time": round(self.source_time, 4),
            "mode": self.mode,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "height_fraction": self.height_fraction,
            "crop_x": self.crop_x,
            "crop_y": self.crop_y,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "display_width": self.display_width,
            "display_height": self.display_height,
        }


def _geometry_from_plan(plan_payload: Mapping[str, object]) -> DisplayGeometry | None:
    geometry = plan_payload.get("geometry")
    if not isinstance(geometry, Mapping):
        return None
    width = geometry.get("display_width")
    height = geometry.get("display_height")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return None
    if width <= 0 or height <= 0:
        return None
    return DisplayGeometry(
        encoded_width=int(geometry.get("encoded_width", width)),
        encoded_height=int(geometry.get("encoded_height", height)),
        rotation_degrees=int(geometry.get("rotation_degrees", 0)),
        display_width=int(width),
        display_height=int(height),
    )


def _scenes_from_plan(plan_payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    scenes = plan_payload.get("scenes")
    if not isinstance(scenes, Sequence) or isinstance(scenes, (str, bytes)):
        return []
    return [scene for scene in scenes if isinstance(scene, Mapping)]


def _scene_for_time(
    scenes: Sequence[Mapping[str, object]], source_time: float
) -> Mapping[str, object] | None:
    if not scenes:
        return None
    for scene in scenes:
        start = scene.get("source_start")
        end = scene.get("source_end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            if float(start) <= source_time <= float(end):
                return scene
    last_end = _as_float(scenes[-1].get("source_end"), 0.0)
    return scenes[-1] if source_time > last_end else scenes[0]


def _keyframes(scene: Mapping[str, object]) -> list[Mapping[str, object]]:
    frames = scene.get("crop_keyframes")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return []
    ordered = [frame for frame in frames if isinstance(frame, Mapping)]
    ordered.sort(key=lambda frame: _as_float(frame.get("t"), 0.0))
    return ordered


def _interpolate_keyframe(
    keyframes: Sequence[Mapping[str, object]], source_time: float
) -> tuple[float, float, float]:
    """Smoothstep-eased interpolation of the plan's crop keyframes."""

    if not keyframes:
        return (0.5, 0.5, 1.0)
    if len(keyframes) == 1 or source_time <= _as_float(keyframes[0].get("t"), 0.0):
        first = keyframes[0]
        return (
            _as_float(first.get("cx"), 0.5),
            _as_float(first.get("cy"), 0.5),
            _as_float(first.get("height_fraction"), 1.0),
        )
    last = keyframes[-1]
    if source_time >= _as_float(last.get("t"), 0.0):
        return (
            _as_float(last.get("cx"), 0.5),
            _as_float(last.get("cy"), 0.5),
            _as_float(last.get("height_fraction"), 1.0),
        )
    for left, right in zip(keyframes, keyframes[1:]):
        left_t = _as_float(left.get("t"), 0.0)
        right_t = _as_float(right.get("t"), 0.0)
        if left_t <= source_time <= right_t:
            span = right_t - left_t
            ratio = 0.0 if span <= 0.0 else (source_time - left_t) / span
            eased = framing_for_bounded_distance(ratio)
            return (
                _lerp(_as_float(left.get("cx"), 0.5), _as_float(right.get("cx"), 0.5), eased),
                _lerp(_as_float(left.get("cy"), 0.5), _as_float(right.get("cy"), 0.5), eased),
                _lerp(
                    _as_float(left.get("height_fraction"), 1.0),
                    _as_float(right.get("height_fraction"), 1.0),
                    eased,
                ),
            )
    return (0.5, 0.5, 1.0)


def _lerp(first: float, second: float, ratio: float) -> float:
    return first + (second - first) * ratio


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def resolve_preview_crop(plan_payload: Mapping[str, object], source_time: float) -> PreviewCrop:
    """Resolve the plan's real composition (mode + clamped crop) at one time."""

    geometry = _geometry_from_plan(plan_payload)
    if geometry is None:
        raise PreviewError("plan payload has no usable display geometry")
    scenes = _scenes_from_plan(plan_payload)
    scene = _scene_for_time(scenes, source_time)
    if scene is None:
        mode = FramingMode.CENTER_FALLBACK.value
        cx, cy, height_fraction = 0.5, 0.5, 1.0
    else:
        mode = str(scene.get("framing_mode") or FramingMode.CENTER_FALLBACK.value)
        cx, cy, height_fraction = _interpolate_keyframe(_keyframes(scene), source_time)
    crop = normalized_crop_to_display(cx, cy, height_fraction, geometry)
    clamped = clamp_crop(
        crop["x"],
        crop["y"],
        crop["width"],
        crop["height"],
        float(geometry.display_width),
        float(geometry.display_height),
    )
    return PreviewCrop(
        source_time=source_time,
        mode=mode,
        center_x=cx,
        center_y=cy,
        height_fraction=height_fraction,
        crop_x=int(round(clamped[0])),
        crop_y=int(round(clamped[1])),
        crop_width=max(2, int(round(clamped[2]))),
        crop_height=max(2, int(round(clamped[3]))),
        display_width=geometry.display_width,
        display_height=geometry.display_height,
    )


def preview_filtergraph(
    plan_payload: Mapping[str, object],
    source_time: float,
    *,
    ass_filename: str | None = None,
) -> str:
    """Build the deterministic FFmpeg filtergraph for one plan composition."""

    crop = resolve_preview_crop(plan_payload, source_time)
    mode = crop.mode
    tail = f",ass={ass_filename}" if ass_filename is not None else ""
    if mode == FramingMode.BACKGROUND_FILL.value:
        return (
            "split=2[bg][fg];"
            f"[bg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},{BACKGROUND_FILL_BACKGROUND_FILTER}[bgc];"
            f"[fg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgc][fgs]overlay=(W-w)/2:(H-h)/2{tail}"
        )
    if mode == FramingMode.SOURCE_AS_IS.value:
        return (
            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2{tail}"
        )
    # Crop modes (STATIC_CROP, TRACKED_CROP, MULTI_SUBJECT_FIT, CENTER_FALLBACK).
    return (
        f"crop={crop.crop_width}:{crop.crop_height}:{crop.crop_x}:{crop.crop_y},"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}{tail}"
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
        filtergraph = preview_filtergraph(
            plan_payload,
            source_time,
            ass_filename=ass_path.name if ass_path is not None else None,
        )
        arguments = _frame_arguments(
            executable=executable,
            source_path=source_path,
            source_time=source_time,
            output_path=output_path,
            filtergraph=filtergraph,
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
    filtergraph: str | None = None,
) -> list[str]:
    if filtergraph is None:
        filtergraph = (
            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
        )
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
        filtergraph,
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
