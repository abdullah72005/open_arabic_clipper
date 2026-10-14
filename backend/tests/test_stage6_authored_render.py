"""Stage 6 authored render: real FFmpeg SOURCE -> CARD -> SOURCE to one MP4.

This is a synthetic integration fixture for the authored renderer milestone. It
is NOT genuine product acceptance: the source is generated test media. It proves
the authored compiler and runner produce one real H.264/AAC output with cards,
both caption time domains, mixed audio, and a measured probe.
"""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from app.core.settings import get_settings
from app.render.execution.authored import (
    ASSET_ROLE_NARRATION,
    ASSET_ROLE_SOURCE_VIDEO,
    AUTHORED_SCHEMA_VERSION,
    AuthoredRenderPayload,
)
from app.render.execution.compiler import compile_render
from app.render.execution.policy import TRANSFORMED_SHORT, delivery_profile_for
from app.render.execution.runner import run_compiled_render
from app.render.execution.types import (
    AssAsset,
    AttemptContext,
    RenderSpec,
    RuntimeIdentity,
    SceneSpec,
    TimelineOccurrence,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64

_ASS_HEADER = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 360\n"
    "PlayResY: 640\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
)

_CANONICAL_ASS = (
    _ASS_HEADER + "Style: Default,DejaVu Sans,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
    "0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:00.00,0:00:02.50,Default,,0,0,0,,source caption\n"
    "Dialogue: 0,0:00:03.00,0:00:05.50,Default,,0,0,0,,second caption\n"
)

_AUTHORED_ASS = (
    _ASS_HEADER + "Style: Card,DejaVu Sans,28,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
    "0,0,0,0,100,100,0,0,1,2,0,5,20,20,20,1\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:03.00,0:00:04.00,Card,,0,0,0,,authored card text\n"
)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603


@pytest.fixture  # type: ignore[untyped-decorator]
def workspace(tmp_path: Path) -> Path:
    settings = get_settings()
    ffmpeg = settings.ffmpeg_binary
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    source = attempt / "source.mp4"
    _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=360x640:rate=30:duration=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300:sample_rate=48000:duration=6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ]
    )
    narration = attempt / "narr.wav"
    _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=700:sample_rate=48000:duration=1.0",
            "-c:a",
            "pcm_s16le",
            str(narration),
        ]
    )
    (attempt / "captions.ass").write_text(_CANONICAL_ASS, encoding="utf-8")
    return attempt


def _spec(source: Path, narration: Path) -> RenderSpec:
    profile = delivery_profile_for("MP4_H264_AAC_1080X1920_V1")
    assert profile is not None
    from app.render.execution.types import CropKeyframeSpec

    crop = (CropKeyframeSpec(t=0.0, cx=0.5, cy=0.5, height_fraction=1.0, mode="STATIC"),)
    scenes = (
        SceneSpec(0, 0, 0.0, 3.0, "STATIC", "NONE", crop),
        SceneSpec(0, 2, 3.0, 6.0, "STATIC", "NONE", crop),
    )
    occurrences = (
        TimelineOccurrence("occ-0", 0, 0.0, 3.0, 0.0, 3.0, "HERO", True, (scenes[0],)),
        TimelineOccurrence("occ-2", 2, 3.0, 6.0, 4.0, 7.0, "CONTEXT", False, (scenes[1],)),
    )
    assets = (
        {
            "asset_id": "src",
            "role": ASSET_ROLE_SOURCE_VIDEO,
            "storage_relative_path": "source.mp4",
            "sha256": _HASH_A,
            "size_bytes": source.stat().st_size,
            "duration_seconds": 6.0,
            "sample_rate": 48000,
            "channels": 2,
            "width": 360,
            "height": 640,
        },
        {
            "asset_id": "narr",
            "role": ASSET_ROLE_NARRATION,
            "storage_relative_path": "narr.wav",
            "sha256": _HASH_B,
            "size_bytes": narration.stat().st_size,
            "duration_seconds": 1.0,
            "sample_rate": 48000,
            "channels": 1,
        },
    )
    payload_dict = {
        "schema_version": AUTHORED_SCHEMA_VERSION,
        "materialization_id": "mat-1",
        "materialization_output_fingerprint": _HASH_A,
        "repair_revision": 0,
        "units": [
            {
                "unit_id": "u0",
                "kind": "SOURCE",
                "block_index": 0,
                "start_frame": 0,
                "end_frame": 90,
                "occurrence_id": "occ-0",
                "background_asset_id": None,
                "background_start_seconds": None,
            },
            {
                "unit_id": "u1",
                "kind": "CARD",
                "block_index": 1,
                "start_frame": 90,
                "end_frame": 120,
                "occurrence_id": None,
                "background_asset_id": None,
                "background_start_seconds": None,
            },
            {
                "unit_id": "u2",
                "kind": "SOURCE",
                "block_index": 2,
                "start_frame": 120,
                "end_frame": 210,
                "occurrence_id": "occ-2",
                "background_asset_id": None,
                "background_start_seconds": None,
            },
        ],
        "assets": list(assets),
        "tracks": [
            {
                "track_id": "t-src",
                "asset_id": "src",
                "source_start_seconds": 0.0,
                "source_end_seconds": 6.0,
                "output_start_sample": 0,
                "output_end_sample": 336000,
                "gain_points": [{"sample_index": 0, "gain_db": 0.0}],
            },
            {
                "track_id": "t-narr",
                "asset_id": "narr",
                "source_start_seconds": 0.0,
                "source_end_seconds": 1.0,
                "output_start_sample": 168000,
                "output_end_sample": 216000,
                "gain_points": [{"sample_index": 168000, "gain_db": 0.0}],
            },
        ],
        "authored_ass": {
            "relative_path": "authored.ass",
            "sha256": _HASH_B,
            "policy_version": "stage6-ass-v1",
        },
        "mastering": {
            "target_lufs": -16.0,
            "ceiling_dbtp": -1.5,
            "render_guard_dbtp": -1.8,
            "sample_rate": 48000,
            "channels": 2,
            "mode": "LINEAR_PREFERRED",
            "repair_peak_limit_db": 0.0,
        },
        "frame_count": 210,
        "sample_count": 336000,
        "source_protection_evidence": {"hero_preserved": True},
    }
    payload = AuthoredRenderPayload.from_dict(payload_dict)
    return RenderSpec(
        render_execution_id="re-1",
        candidate_id="cand-1",
        source_id="src-1",
        render_contract_id="contract-1",
        visual_plan_id="vis-1",
        artifact_purpose=TRANSFORMED_SHORT,
        source_media_relative_path="source.mp4",
        source_content_hash=_HASH_A,
        source_size_bytes=source.stat().st_size,
        source_mtime_ns=0,
        source_duration=6.0,
        source_frame_rate=Fraction(30, 1),
        display_width=360,
        display_height=640,
        encoded_width=360,
        encoded_height=640,
        rotation_degrees=0,
        pixel_aspect_ratio=1.0,
        output_width=1080,
        output_height=1920,
        output_frame_rate=Fraction(30, 1),
        output_profile={"width": 1080, "height": 1920},
        delivery_profile_key="MP4_H264_AAC_1080X1920_V1",
        plan_input_fingerprint="in",
        plan_output_fingerprint="out",
        ass=AssAsset(
            relative_path="captions.ass",
            sha256=_HASH_A,
            event_count=2,
            line_count=2,
            policy_version="stage5.1",
        ),
        caption_events=(),
        occurrences=occurrences,
        authored=payload,
    )


def _runtime(source: Path, attempt: Path) -> RuntimeIdentity:
    settings = get_settings()
    return RuntimeIdentity(
        ffmpeg_version="test",
        ffprobe_version="test",
        libavformat_version="test",
        libass_version="test",
        font_family="DejaVu Sans",
        font_match="DejaVu Sans",
        compiler_version="test",
        policy_version="test",
        ffmpeg_binary=settings.ffmpeg_binary,
        ffprobe_binary=settings.ffprobe_binary,
        source_absolute_path=str(source),
        attempt_directory=str(attempt),
        asset_localized_paths={"narr": "narr.wav", "src": "source.mp4"},
    )


def _mean_luma(video: Path, timestamp: float) -> float:
    settings = get_settings()
    completed = subprocess.run(  # noqa: S603
        [
            settings.ffmpeg_binary,
            "-v",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    data = completed.stdout
    return (sum(data) / len(data)) if data else 0.0


def test_authored_render_produces_one_real_mp4(workspace: Path) -> None:
    source = workspace / "source.mp4"
    narration = workspace / "narr.wav"
    spec = _spec(source, narration)
    runtime = _runtime(source, workspace)

    compiled = compile_render(spec, runtime)
    assert compiled.compiler_version.startswith("stage6-authored")
    assert compiled.authored_ass_localized_name == "authored.ass"
    assert compiled.input_relative_paths == ("narr.wav",)

    context = AttemptContext(
        attempt_directory=workspace,
        ass_bytes=_CANONICAL_ASS.encode("utf-8"),
        authored_ass_bytes=_AUTHORED_ASS.encode("utf-8"),
        timeout_seconds=120.0,
    )
    artifacts = run_compiled_render(compiled, context)

    output = workspace / "output.mp4"
    assert output.is_file() and output.stat().st_size > 0
    assert artifacts.probe["video_codec"] == "h264"
    assert artifacts.probe["audio_codec"] == "aac"
    assert artifacts.probe["audio_sample_rate"] == 48000
    assert artifacts.probe["audio_channels"] == 2
    assert artifacts.frame_count > 0
    assert abs(artifacts.duration_seconds - 7.0) < 0.4

    # Card frames actually appear: the card interval is a near-black authored
    # card with legible text, so its mean luma is far below the animated source.
    card_luma = _mean_luma(output, 3.5)
    source_luma = _mean_luma(output, 1.5)
    assert card_luma < source_luma
    assert card_luma < 60.0
