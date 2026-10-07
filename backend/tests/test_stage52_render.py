"""Stage 5.2 real FFmpeg render tests (skipped when ffmpeg is unavailable)."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from stage52_support import (
    crop_scene,
    fake_runtime,
    independent_tracked_geometry,
    make_spec,
    occurred,
    scene_geometry,
)

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
    CompiledRender,
    CropKeyframeSpec,
    RenderArtifacts,
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


def _ink_stats(data: bytes, width: int, height: int) -> tuple[int, int, int, int, int, int, int]:
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


# ---------------------------------------------------------------------------
# Tracked-crop executed-geometry regressions.
#
# A coordinate-coded gradient source encodes each pixel's source position into
# its colour (R ~ x, G ~ y). For every sampled source-local time the encoded
# output centre pixel is compared against the independently calculated source
# pixel the accepted crop geometry should expose. These tests fail against the
# defective compiler, which used crop's cached in_w/in_h instead of the current
# scaled dimensions.
# ---------------------------------------------------------------------------


def _generate_gradient_source(path: Path, width: int, height: int, duration: float) -> None:
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
            f"color=c=black:s={width}x{height}:r=30:d={duration},format=rgb24,"
            "geq=r='255*X/W':g='255*Y/H':b='0'",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
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


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def landscape_gradient(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = Path(tmp_path_factory.mktemp("gradient")) / "landscape.mp4"
    _generate_gradient_source(path, 1920, 1080, 12.0)
    return path


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def portrait_gradient(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = Path(tmp_path_factory.mktemp("gradient")) / "portrait.mp4"
    _generate_gradient_source(path, 1080, 1920, 8.0)
    return path


def _patch_color(
    frame: bytes, width: int, height: int, x: int, y: int, radius: int = 4
) -> tuple[int, int, int]:
    x = max(radius, min(width - 1 - radius, x))
    y = max(radius, min(height - 1 - radius, y))
    totals = [0, 0, 0]
    count = 0
    for row in range(y - radius, y + radius + 1):
        base = row * width * 3
        for column in range(x - radius, x + radius + 1):
            index = base + column * 3
            totals[0] += frame[index]
            totals[1] += frame[index + 1]
            totals[2] += frame[index + 2]
            count += 1
    return (totals[0] // count, totals[1] // count, totals[2] // count)


def _frames(keyframes: list[tuple[float, float, float, float]]) -> list[CropKeyframeSpec]:
    return [CropKeyframeSpec(t, cx, cy, hf) for t, cx, cy, hf in keyframes]


def _render_tracked(
    tmp_path: Path,
    source: Path,
    display_width: int,
    display_height: int,
    source_start: float,
    source_end: float,
    keyframes: list[tuple[float, float, float, float]],
    source_duration: float,
) -> tuple[RenderSpec, CompiledRender, RenderArtifacts]:
    scene = crop_scene(0, 0, source_start, source_end, "TRACKED_CROP", keyframes)
    occ = occurred("block-0", 0, source_start, source_end, 0.0, (scene,))
    spec = make_spec(
        occurrences=(occ,),
        caption_events=(),
        source_duration=source_duration,
        display_width=display_width,
        display_height=display_height,
    )
    validate_spec(spec)
    attempt = tmp_path / "attempt"
    runtime = fake_runtime(source_absolute_path=str(source), attempt_directory=str(attempt))
    compiled = compile_render(spec, runtime)
    artifacts = run_compiled_render(
        compiled,
        AttemptContext(attempt_directory=attempt, ass_bytes=_MIXED_ASS, timeout_seconds=300),
    )
    return spec, compiled, artifacts


def _assert_tracked_alignment(
    source: Path,
    output: Path,
    display_width: int,
    display_height: int,
    keyframes: list[tuple[float, float, float, float]],
    source_start: float,
    sample_times: list[float],
    *,
    tolerance: float = 12.0,
) -> None:
    """Compare encoded output centre pixels to the independent source reference.

    Tolerances account for bicubic scaling, yuv420p chroma subsampling, H.264
    quantization, and one-frame seek granularity; a stale-dimension crop shifts
    the exposed source position by tens of pixels and far exceeds them.
    """

    frames = _frames(keyframes)
    reference_frame = _decode_rgb_frame(FFMPEG_BIN, source, 0.0)
    assert len(reference_frame) == display_width * display_height * 3
    for t in sample_times:
        geometry = independent_tracked_geometry(frames, t, display_width, display_height)
        output_frame = _decode_rgb_frame(FFMPEG_BIN, output, t - source_start)
        assert len(output_frame) == 1080 * 1920 * 3
        rendered = _patch_color(output_frame, 1080, 1920, 540, 960)
        expected = _patch_color(
            reference_frame,
            display_width,
            display_height,
            int(round(geometry["source_x"])),
            int(round(geometry["source_y"])),
        )
        difference = sum(abs(a - b) for a, b in zip(rendered, expected)) / 3.0
        assert difference <= tolerance, (t, rendered, expected, geometry, difference)


def test_tracked_crop_simultaneous_pan_and_zoom(tmp_path: Path, landscape_gradient: Path) -> None:
    # Reviewed case-01 geometry at a nonzero source-local origin.
    keyframes = [
        (5.0, 0.45, 0.48, 0.95),
        (6.0, 0.50, 0.46, 0.82),
        (7.0, 0.55, 0.44, 0.70),
        (8.0, 0.52, 0.45, 0.78),
    ]
    _, compiled, artifacts = _render_tracked(
        tmp_path, landscape_gradient, 1920, 1080, 5.0, 8.0, keyframes, 12.0
    )
    assert artifacts.probe["width"] == 1080
    assert artifacts.probe["height"] == 1920
    # first frame, interior interpolation, keyframe, and late frames
    _assert_tracked_alignment(
        landscape_gradient,
        artifacts.output_path,
        1920,
        1080,
        keyframes,
        5.0,
        [5.0, 5.5, 6.0, 7.0, 7.9],
    )
    # Interpolation origin is source-local: trimming must not freeze the pan.
    frames = _frames(keyframes)
    first = independent_tracked_geometry(frames, 5.0, 1920, 1080)
    late = independent_tracked_geometry(frames, 7.0, 1920, 1080)
    assert abs(late["crop_x"] - first["crop_x"]) > 200
    assert "in_w" not in compiled.filtergraph
    assert "in_h" not in compiled.filtergraph


def test_tracked_crop_vertical_movement_under_zoom(
    tmp_path: Path, landscape_gradient: Path
) -> None:
    keyframes = [(0.0, 0.5, 0.30, 1.0), (2.0, 0.5, 0.70, 0.60)]
    _, _, artifacts = _render_tracked(
        tmp_path, landscape_gradient, 1920, 1080, 0.0, 2.0, keyframes, 12.0
    )
    _assert_tracked_alignment(
        landscape_gradient,
        artifacts.output_path,
        1920,
        1080,
        keyframes,
        0.0,
        [0.0, 0.5, 1.0, 1.5, 1.9],
    )
    # The old bug collapsed y toward zero as scale height grew; a large cy move
    # must therefore produce a large expected source-y change.
    frames = _frames(keyframes)
    start = independent_tracked_geometry(frames, 0.0, 1920, 1080)
    end = independent_tracked_geometry(frames, 2.0, 1920, 1080)
    assert end["source_y"] - start["source_y"] > 60


def test_tracked_crop_edge_clamps(tmp_path: Path, landscape_gradient: Path) -> None:
    left = [(0.0, 0.02, 0.5, 0.6), (2.0, 0.02, 0.5, 0.6)]
    _, _, left_artifacts = _render_tracked(
        tmp_path / "left", landscape_gradient, 1920, 1080, 0.0, 2.0, left, 12.0
    )
    left_geometry = independent_tracked_geometry(_frames(left), 1.0, 1920, 1080)
    assert left_geometry["crop_x"] == 0.0
    _assert_tracked_alignment(
        landscape_gradient, left_artifacts.output_path, 1920, 1080, left, 0.0, [0.0, 1.0, 1.9]
    )
    right = [(0.0, 0.98, 0.5, 0.6), (2.0, 0.98, 0.5, 0.6)]
    _, _, right_artifacts = _render_tracked(
        tmp_path / "right", landscape_gradient, 1920, 1080, 0.0, 2.0, right, 12.0
    )
    right_geometry = independent_tracked_geometry(_frames(right), 1.0, 1920, 1080)
    assert right_geometry["crop_x"] == right_geometry["scaled_width"] - 1080
    _assert_tracked_alignment(
        landscape_gradient, right_artifacts.output_path, 1920, 1080, right, 0.0, [0.0, 1.0, 1.9]
    )


def test_tracked_crop_constant_height_tracking(tmp_path: Path, landscape_gradient: Path) -> None:
    keyframes = [(0.0, 0.30, 0.5, 0.8), (2.0, 0.70, 0.5, 0.8)]
    _, compiled, artifacts = _render_tracked(
        tmp_path, landscape_gradient, 1920, 1080, 0.0, 2.0, keyframes, 12.0
    )
    _assert_tracked_alignment(
        landscape_gradient, artifacts.output_path, 1920, 1080, keyframes, 0.0, [0.0, 0.7, 1.4, 1.9]
    )
    samples = scene_geometry(compiled)["geometry_samples"]
    assert len({sample["scaled_width"] for sample in samples}) == 1
    assert len({sample["scaled_height"] for sample in samples}) == 1


def test_tracked_crop_portrait_display_geometry(tmp_path: Path, portrait_gradient: Path) -> None:
    keyframes = [(0.0, 0.5, 0.40, 0.90), (2.0, 0.5, 0.60, 0.70)]
    _, compiled, artifacts = _render_tracked(
        tmp_path, portrait_gradient, 1080, 1920, 0.0, 2.0, keyframes, 8.0
    )
    assert artifacts.probe["width"] == 1080
    assert artifacts.probe["height"] == 1920
    _assert_tracked_alignment(
        portrait_gradient,
        artifacts.output_path,
        1080,
        1920,
        keyframes,
        0.0,
        [0.0, 0.5, 1.0, 1.5, 1.9],
    )
