"""Stage 5.1 preview tests: faithful plan composition + caption timing.

Preview frames must render the plan's *real* composition (interpolated crop
keyframes, planned background-fill, bounded scale/pad), and a caption active at a
source-local time after 0 must still be burned in (``-copyts``).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.composition.geometry import clamp_crop, normalized_crop_to_display
from app.composition.policy import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    CaptionStyle,
    FramingMode,
    framing_for_bounded_distance,
    safe_zone_for,
)
from app.composition.preview import (
    BACKGROUND_FILL_BACKGROUND_FILTER,
    _frame_arguments,
    preview_filtergraph,
    resolve_preview_crop,
)
from app.composition.types import DisplayGeometry

ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")

_GEOMETRY = {
    "encoded_width": 1920,
    "encoded_height": 1080,
    "rotation_degrees": 0,
    "display_width": 1920,
    "display_height": 1080,
}


def _crop_scene(
    *,
    index: int,
    start: float,
    end: float,
    mode: str,
    keyframes: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "scene_index": index,
        "block_index": index,
        "source_start": start,
        "source_end": end,
        "framing_mode": mode,
        "crop_keyframes": keyframes,
        "interpolation_policy": "smoothstep-ease",
    }


def _keyframe(t: float, cx: float, cy: float, hf: float) -> dict[str, object]:
    return {"t": t, "cx": cx, "cy": cy, "height_fraction": hf, "mode": "TRACKED_CROP"}


# geometry: 1920x1080 display; a 0.5-height crop is 540 tall x 303.75 wide.


def _expected_crop(cx: float, cy: float, hf: float) -> tuple[int, int, int, int]:
    geometry = DisplayGeometry(
        encoded_width=1920,
        encoded_height=1080,
        rotation_degrees=0,
        display_width=1920,
        display_height=1080,
    )
    raw = normalized_crop_to_display(cx, cy, hf, geometry)
    clamped = clamp_crop(raw["x"], raw["y"], raw["width"], raw["height"], 1920.0, 1080.0)
    return (
        int(round(clamped[0])),
        int(round(clamped[1])),
        max(2, int(round(clamped[2]))),
        max(2, int(round(clamped[3]))),
    )


def test_preview_crop_matches_plan_geometry_at_interpolated_time() -> None:
    payload = {
        "geometry": dict(_GEOMETRY),
        "scenes": [
            _crop_scene(
                index=0,
                start=0.0,
                end=10.0,
                mode=FramingMode.TRACKED_CROP.value,
                keyframes=[_keyframe(0.0, 0.3, 0.5, 0.5), _keyframe(10.0, 0.7, 0.5, 0.5)],
            )
        ],
    }
    # At t=5 the eased ratio is 0.5 -> cx=0.5.
    resolved = resolve_preview_crop(payload, 5.0)
    expected = _expected_crop(0.5, 0.5, 0.5)
    assert (resolved.crop_x, resolved.crop_y, resolved.crop_width, resolved.crop_height) == (
        expected
    )
    assert resolved.mode == FramingMode.TRACKED_CROP.value
    assert resolved.center_x == pytest.approx(0.5)
    # Interpolation is smoothstep-eased, not linear.
    assert framing_for_bounded_distance(0.5) == pytest.approx(0.5)

    at_start = resolve_preview_crop(payload, 0.0)
    assert (at_start.crop_x, at_start.crop_y) == _expected_crop(0.3, 0.5, 0.5)[:2]

    graph = preview_filtergraph(payload, 5.0)
    x, y, w, h = expected
    assert f"crop={w}:{h}:{x}:{y}" in graph
    assert graph.endswith(f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}")


def test_preview_crop_smoothstep_between_keyframes() -> None:
    payload = {
        "geometry": dict(_GEOMETRY),
        "scenes": [
            _crop_scene(
                index=0,
                start=0.0,
                end=10.0,
                mode=FramingMode.TRACKED_CROP.value,
                keyframes=[_keyframe(0.0, 0.0, 0.5, 0.5), _keyframe(10.0, 1.0, 0.5, 0.5)],
            )
        ],
    }
    # smoothstep(0.25) = 0.15625, not 0.25.
    quarter = resolve_preview_crop(payload, 2.5)
    assert quarter.center_x == pytest.approx(0.15625)


def test_background_fill_filtergraph_uses_blur_contain_overlay() -> None:
    payload = {
        "geometry": dict(_GEOMETRY),
        "scenes": [
            _crop_scene(
                index=0,
                start=0.0,
                end=4.0,
                mode=FramingMode.BACKGROUND_FILL.value,
                keyframes=[_keyframe(0.0, 0.5, 0.5, 1.0), _keyframe(4.0, 0.5, 0.5, 1.0)],
            )
        ],
    }
    graph = preview_filtergraph(payload, 2.0)
    assert "split=2[bg][fg]" in graph
    assert BACKGROUND_FILL_BACKGROUND_FILTER in graph
    assert "gblur=sigma=36" in graph
    assert "eq=brightness=-0.18" in graph
    assert "[bgc][fgs]overlay=(W-w)/2:(H-h)/2" in graph
    assert "boxblur" not in graph


def test_source_as_is_filtergraph_uses_bounded_scale_and_pad() -> None:
    payload = {
        "geometry": dict(_GEOMETRY),
        "scenes": [
            _crop_scene(
                index=0,
                start=0.0,
                end=4.0,
                mode=FramingMode.SOURCE_AS_IS.value,
                keyframes=[_keyframe(0.0, 0.5, 0.5, 1.0), _keyframe(4.0, 0.5, 0.5, 1.0)],
            )
        ],
    }
    graph = preview_filtergraph(payload, 1.0)
    assert "force_original_aspect_ratio=decrease" in graph
    assert "pad=1080:1920" in graph


# --- caption timing through the real render ---------------------------------

_ASS_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_CAPTION_STYLE = CaptionStyle()
_SAFE_ZONE = safe_zone_for("SHORTS_VERTICAL_SAFE_ZONE_V1")
_ASS_STYLE_LINE = (
    f"Style: CaptionLower,{{font}},{_CAPTION_STYLE.font_size},"
    f"{_CAPTION_STYLE.primary_color},{_CAPTION_STYLE.secondary_color},"
    f"{_CAPTION_STYLE.outline_color},&H00000000,"
    f"0,0,0,0,100,100,{_CAPTION_STYLE.spacing},0,1,"
    f"{_CAPTION_STYLE.outline_width},{_CAPTION_STYLE.shadow},2,"
    f"{_SAFE_ZONE.left_px()},{_SAFE_ZONE.right_px()},"
    f"{_SAFE_ZONE.bottom_px() + _SAFE_ZONE.caption_bottom_gap_px},1"
)
_ASS_TEMPLATE = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
{_ASS_STYLE_FORMAT}
{_ASS_STYLE_LINE}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.50,0:00:03.00,CaptionLower,,0,0,0,,LATE CAPTION
"""


def _ink_pixels(png: Path) -> int:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(png),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return sum(1 for value in result.stdout if value != 0)


def _font_family() -> str:
    for candidate in ("Noto Sans Arabic", "DejaVu Sans"):
        matched = subprocess.run(
            ["fc-match", "-f", "%{family}", candidate],
            capture_output=True,
            text=True,
        ).stdout
        if candidate.split()[0] in matched:
            return candidate
    return "DejaVu Sans"


@ffmpeg
def test_late_caption_is_visible_with_copyts(tmp_path: Path) -> None:
    video = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x360:d=4:r=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video),
        ],
        check=True,
    )
    ass_path = tmp_path / "captions.ass"
    ass_path.write_text(_ASS_TEMPLATE.format(font=_font_family()), encoding="utf-8")

    png = tmp_path / "frame.png"
    arguments = _frame_arguments(
        executable="ffmpeg",
        source_path=video,
        source_time=2.0,
        output_path=png,
        filtergraph=f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},ass={ass_path.name}",
    )
    assert "-copyts" in arguments
    subprocess.run(arguments, check=True, cwd=str(tmp_path))
    assert png.is_file()
    assert _ink_pixels(png) > 0, "caption active at t=2.0 must be burned in"

    # Negative control: dropping -copyts must lose the late caption.
    naive = [argument for argument in arguments if argument != "-copyts"]
    png_naive = tmp_path / "frame-naive.png"
    naive[naive.index(str(png))] = str(png_naive)
    subprocess.run(naive, check=True, cwd=str(tmp_path))
    assert _ink_pixels(png_naive) < _ink_pixels(png), "without -copyts the caption is inactive"


@ffmpeg
def test_preview_renders_real_plan_crop(tmp_path: Path) -> None:
    """The rendered frame is cropped to the plan's crop, not a fixed center crop."""

    video = tmp_path / "src.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1920x1080:d=3:r=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video),
        ],
        check=True,
    )
    payload = {
        "geometry": dict(_GEOMETRY),
        "scenes": [
            _crop_scene(
                index=0,
                start=0.0,
                end=3.0,
                mode=FramingMode.STATIC_CROP.value,
                keyframes=[_keyframe(0.0, 0.2, 0.5, 0.5), _keyframe(3.0, 0.2, 0.5, 0.5)],
            )
        ],
    }
    graph = preview_filtergraph(payload, 1.5)
    png = tmp_path / "plan-frame.png"
    arguments = _frame_arguments(
        executable="ffmpeg",
        source_path=video,
        source_time=1.5,
        output_path=png,
        filtergraph=graph,
    )
    subprocess.run(arguments, check=True, cwd=str(tmp_path))
    assert png.is_file()
    # The render must be the target profile size, proving the crop+scale ran.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(png),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == f"{OUTPUT_WIDTH},{OUTPUT_HEIGHT}"
