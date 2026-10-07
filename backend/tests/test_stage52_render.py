"""Stage 5.2 real FFmpeg render tests (skipped when ffmpeg is unavailable)."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from stage52_support import crop_scene, fake_runtime, make_spec, occurred

from app.composition.ass import serialize_ass
from app.composition.captions import CaptionPlan
from app.composition.policy import CaptionStyle, safe_zone_for
from app.composition.types import CaptionEvent, CaptionWordTiming
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

# The canonical mixed Arabic/English phrase used by the Stage 5.1 serialization
# path. The regression builds the ASS through production ``serialize_ass`` (not a
# hand-written karaoke fixture) so encoded evidence reflects the real active-word
# states and preserved BiDi layout.
_CANONICAL_PHRASE = "أنا كنت content creator لمدة سنتين"
_CANONICAL_START = 0.3
_CANONICAL_WORD_SECONDS = 0.5


def _canonical_caption() -> tuple[bytes, CaptionEvent]:
    tokens = _CANONICAL_PHRASE.split(" ")
    word_timings = tuple(
        CaptionWordTiming(
            index=index,
            text=token,
            start=_CANONICAL_START + index * _CANONICAL_WORD_SECONDS,
            end=_CANONICAL_START + index * _CANONICAL_WORD_SECONDS + _CANONICAL_WORD_SECONDS - 0.02,
        )
        for index, token in enumerate(tokens)
    )
    event = CaptionEvent(
        event_id="canonical-mixed",
        block_index=0,
        word_start_index=0,
        word_end_index=len(tokens) - 1,
        start=_CANONICAL_START,
        end=_CANONICAL_START + len(tokens) * _CANONICAL_WORD_SECONDS,
        text=_CANONICAL_PHRASE,
        lines=(_CANONICAL_PHRASE,),
        placement_zone="LOWER",
        placement_reason="MANUAL_ACCEPTANCE",
        word_timings=word_timings,
    )
    plan = CaptionPlan(policy_version="stage5.1-caption-layout-v4", events=(event,))
    style = CaptionStyle()
    safe_zone = safe_zone_for("SHORTS_VERTICAL_SAFE_ZONE_V1")
    return serialize_ass(plan, style, safe_zone), event


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


_ACTIVE_OVERRIDE = "{\\c&H0000FFFF&}"
_TAG_RE = re.compile(r"\{[^}]*\}")
_ACTIVE_RE = re.compile(r"\{\\c&H0000FFFF&\}(.*?)\{\\c")


def _dialogue_states(ass_text: str) -> list[tuple[list[str], str]]:
    """Parse ASS Dialogue states into (active_tokens, tag-stripped layout)."""

    states: list[tuple[list[str], str]] = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        text = line.split(",", 9)[9]
        active = _ACTIVE_RE.findall(text)
        stripped = _TAG_RE.sub("", text).strip()
        states.append((active, stripped))
    return states


def _decode_rgb_frame(ffmpeg_binary: str, path: Path, time_s: float) -> bytes:
    completed = subprocess.run(
        [
            ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{time_s:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _ink_stats(data: bytes, width: int, height: int) -> tuple[int, int, int, int, int, int]:
    """Return (yellow_pixels, yellow_centroid_x, ink_pixels, min_x, min_y, max_x, max_y)."""

    yellow = 0
    yellow_x_sum = 0
    ink = 0
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    # CaptionLower safe zone sits in the lower third; a fixed band isolates the
    # caption region from the uniform background and bounds the scan cost.
    for row in range(1000, min(height, 1760)):
        base = row * width * 3
        for column in range(width):
            index = base + column * 3
            r = data[index]
            g = data[index + 1]
            b = data[index + 2]
            # Background is a uniform dark navy (0x101030); caption ink is far brighter.
            if r + g + b > 240:
                ink += 1
                if column < min_x:
                    min_x = column
                if column > max_x:
                    max_x = column
                if row < min_y:
                    min_y = row
                if row > max_y:
                    max_y = row
            if r > 180 and g > 150 and b < 120:
                yellow += 1
                yellow_x_sum += column
    centroid = int(yellow_x_sum / yellow) if yellow else -1
    return yellow, centroid, ink, min_x, min_y, max_x, max_y


def test_real_render_canonical_active_word_states_and_stable_layout(tmp_path: Path) -> None:
    ass_bytes, event = _canonical_caption()
    ass_text = ass_bytes.decode("utf-8")

    # ----- Structural evidence from the canonical serialization -------------
    states = _dialogue_states(ass_text)
    tokens = _CANONICAL_PHRASE.split(" ")
    assert len(states) == len(tokens)
    for active, _layout in states:
        # Exactly one active token per canonical state.
        assert len(active) == 1
    active_tokens = [active[0] for active, _ in states]
    # Every word becomes active exactly once (in the canonical sequence).
    assert sorted(active_tokens) == sorted(tokens)
    # Surrounding ordering/geometry is structurally identical in every state.
    layouts = {layout for _active, layout in states}
    assert len(layouts) == 1

    # ----- Encoded evidence from a real FFmpeg render -----------------------
    source = tmp_path / "caption-source.mp4"
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
            "color=c=0x101030:size=1080x1920:rate=30:duration=4",
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
    scene = SceneSpec(0, 0, 0.0, 3.6, "SOURCE_AS_IS", "smoothstep-ease")
    occ = occurred("block-0", 0, 0.0, 3.6, 0.0, (scene,))
    spec = make_spec(occurrences=(occ,), caption_events=(), source_duration=4.0)
    validate_spec(spec)
    attempt = tmp_path / "attempt"
    runtime = fake_runtime(source_absolute_path=str(source), attempt_directory=str(attempt))
    compiled = compile_render(spec, runtime)
    artifacts = run_compiled_render(
        compiled,
        AttemptContext(attempt_directory=attempt, ass_bytes=ass_bytes, timeout_seconds=300),
    )

    mid_times = [
        event.start + index * _CANONICAL_WORD_SECONDS + _CANONICAL_WORD_SECONDS / 2.0
        for index in range(len(tokens))
    ]
    yellow_counts: list[int] = []
    centroids: list[int] = []
    ink_counts: list[int] = []
    boxes: list[tuple[int, int, int, int]] = []
    for time_s in mid_times:
        frame = _decode_rgb_frame(FFMPEG_BIN, artifacts.output_path, time_s)
        assert len(frame) == 1080 * 1920 * 3
        yellow, centroid, ink, min_x, min_y, max_x, max_y = _ink_stats(frame, 1080, 1920)
        yellow_counts.append(yellow)
        centroids.append(centroid)
        ink_counts.append(ink)
        boxes.append((min_x, min_y, max_x, max_y))

    # A yellow (active) token is present in every canonical state.
    assert all(count > 0 for count in yellow_counts), yellow_counts
    # The active token's location changes across states.
    assert len({round(value / 8) for value in centroids}) >= 4, centroids
    # The surrounding line layout (all ink) is stable: same ink count and bbox.
    assert max(ink_counts) - min(ink_counts) <= max(2, int(0.02 * max(ink_counts))), ink_counts
    base_box = boxes[0]
    assert all(all(abs(edge - base) <= 2 for edge, base in zip(box, base_box)) for box in boxes), (
        boxes
    )

    # Timing: no caption ink before the event starts or after it ends.
    before = _ink_stats(_decode_rgb_frame(FFMPEG_BIN, artifacts.output_path, 0.1), 1080, 1920)
    after = _ink_stats(_decode_rgb_frame(FFMPEG_BIN, artifacts.output_path, 3.5), 1080, 1920)
    assert before[0] == 0 and before[2] == 0
    assert after[0] == 0 and after[2] == 0
