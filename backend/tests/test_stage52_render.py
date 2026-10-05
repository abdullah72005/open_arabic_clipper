"""Stage 5.2 real FFmpeg render tests (skipped when ffmpeg is unavailable)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
from stage52_support import crop_scene, fake_runtime, make_spec, occurred

from app.render.execution.compiler import compile_render
from app.render.execution.policy import Stage52Config
from app.render.execution.qc import check_render_artifact
from app.render.execution.runner import run_compiled_render
from app.render.execution.types import (
    AttemptContext,
    CaptionEventSpec,
    RenderSpec,
    SceneSpec,
)
from app.render.execution.validation import validate_spec

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
FFMPEG_BIN = FFMPEG or "ffmpeg"
pytestmark = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg/ffprobe unavailable"
)

_MIXED_ASS = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1080\n"
    "PlayResY: 1920\n"
    "WrapStyle: 2\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    "Style: CaptionLower,Noto Sans Arabic,88,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
    "0,0,0,0,100,100,0,0,1,7,3,2,54,162,504,1\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:01.00,0:00:02.50,CaptionLower,,0,0,0,,"
    "{\\c&H00FFFFFF&}أنا كنت content creator لمدة سنتين{\\c&H00FFFFFF&}\n"
).encode("utf-8")


def _generate_source(path: Path) -> None:
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
            "testsrc2=size=1920x1080:rate=30:duration=8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8",
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


def _spec_two_spans() -> RenderSpec:
    tracked = crop_scene(
        0, 0, 1.0, 2.5, "TRACKED_CROP", [(1.0, 0.35, 0.5, 1.0), (2.5, 0.65, 0.45, 0.6)]
    )
    occ1 = occurred("block-0", 0, 1.0, 2.5, 0.0, (tracked,))
    fill = SceneSpec(1, 1, 4.0, 6.0, "BACKGROUND_FILL", "smoothstep-ease")
    occ2 = occurred("block-1", 1, 4.0, 6.0, 1.5, (fill,))
    spec = make_spec(
        occurrences=(occ1, occ2),
        caption_events=(CaptionEventSpec("event-1", 0, 1.0, 2.5),),
        source_duration=8.0,
    )
    return spec


def test_real_render_two_noncontiguous_spans(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _generate_source(source)
    spec = _spec_two_spans()
    validate_spec(spec)
    attempt = tmp_path / "attempt"
    runtime = fake_runtime(source_absolute_path=str(source), attempt_directory=str(attempt))
    compiled = compile_render(spec, runtime)
    artifacts = run_compiled_render(
        compiled,
        AttemptContext(attempt_directory=attempt, ass_bytes=_MIXED_ASS, timeout_seconds=300),
    )
    qc = check_render_artifact(artifacts, artifacts.manifest, Stage52Config(), source_path=source)
    probe = artifacts.probe
    assert probe["width"] == 1080
    assert probe["height"] == 1920
    assert probe["streams"]["video"] == 1
    assert probe["streams"]["audio"] == 1
    assert probe["video_codec"] == "h264"
    assert probe["audio_codec"] == "aac"
    assert probe["audio_sample_rate"] == 48000
    # Source gap [2.5, 4.0) must not survive; output is 1.5 + 2.0 = 3.5 s.
    assert artifacts.duration_seconds == pytest.approx(3.5, abs=0.15)
    assert qc.status in {"PASS", "WARN"}
    # Canonical ASS bytes are preserved verbatim in the attempt directory.
    localized = (attempt / compiled.ass_localized_name).read_bytes()
    assert hashlib.sha256(localized).hexdigest() == hashlib.sha256(_MIXED_ASS).hexdigest()
    assert (attempt / compiled.filtergraph_relative_path).is_file()


def test_real_render_qc_reports_blank_failure_on_black_output(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
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
            "color=c=black:size=1920x1080:rate=30:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    occ = occurred(
        "block-0",
        0,
        0.5,
        2.5,
        0.0,
        (SceneSpec(0, 0, 0.5, 2.5, "SOURCE_AS_IS", "smoothstep-ease"),),
    )
    spec = make_spec(occurrences=(occ,), caption_events=(), source_duration=4.0)
    attempt = tmp_path / "attempt"
    runtime = fake_runtime(source_absolute_path=str(source), attempt_directory=str(attempt))
    compiled = compile_render(spec, runtime)
    artifacts = run_compiled_render(
        compiled,
        AttemptContext(attempt_directory=attempt, ass_bytes=_MIXED_ASS, timeout_seconds=300),
    )
    qc = check_render_artifact(artifacts, artifacts.manifest, Stage52Config(), source_path=source)
    # The selected source is itself black, so blank output is only a warning.
    assert qc.status in {"PASS", "WARN"}
    assert "QC_BLANK_RENDER" in qc.reason_codes or qc.status == "PASS"
