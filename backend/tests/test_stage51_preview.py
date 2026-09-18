"""Stage 5.1 preview caption-timing regression (Defect 2 guard).

A preview frame rendered at a source-local time after 0 must show the caption
event active at that time. Input seeking without ``-copyts`` rebases the frame
PTS and silently drops every late caption, so this test proves the real render.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.composition.preview import _frame_arguments

ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")

_ASS_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_ASS_STYLE_LINE = (
    "Style: CaptionLower,{font},56,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
    "0,0,0,0,100,100,0,0,1,4,0,2,54,162,504,1"
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
        ass_filename=ass_path.name,
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
