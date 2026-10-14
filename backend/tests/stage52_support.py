"""Shared Stage 5.2 test helpers (engine-level, DB-free fixtures)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Any, cast

from app.render.execution.types import (
    AssAsset,
    CaptionEventSpec,
    CompiledRender,
    CropKeyframeSpec,
    OmittedRequirement,
    RenderSpec,
    RuntimeIdentity,
    SceneSpec,
    TimelineOccurrence,
)


def fake_runtime(
    *,
    ffmpeg_binary: str = "ffmpeg",
    source_absolute_path: str = "/tmp/source.mp4",
    attempt_directory: str = "/tmp/attempt",
    encoder_threads: int = 2,
    filter_threads: int = 1,
    filter_complex_threads: int = 1,
) -> RuntimeIdentity:
    return RuntimeIdentity(
        ffmpeg_version="ffmpeg version test",
        ffprobe_version="ffprobe version test",
        libavformat_version="libavformat-60",
        libass_version="--enable-libass",
        font_family="Noto Sans Arabic",
        font_match="NotoSansArabic-Regular.ttf",
        compiler_version="stage5.2-compiler-v1",
        policy_version="stage5.2-v1",
        ffmpeg_binary=ffmpeg_binary,
        ffprobe_binary="ffprobe",
        encoder_threads=encoder_threads,
        filter_threads=filter_threads,
        filter_complex_threads=filter_complex_threads,
        source_absolute_path=source_absolute_path,
        attempt_directory=attempt_directory,
    )


def crop_scene(
    scene_index: int,
    block_index: int,
    start: float,
    end: float,
    mode: str,
    keyframes: Sequence[tuple[float, float, float, float]],
) -> SceneSpec:
    return SceneSpec(
        scene_index=scene_index,
        block_index=block_index,
        source_start=start,
        source_end=end,
        framing_mode=mode,
        interpolation_policy="smoothstep-ease",
        crop_keyframes=tuple(
            CropKeyframeSpec(t=t, cx=cx, cy=cy, height_fraction=hf) for t, cx, cy, hf in keyframes
        ),
    )


def occurred(
    occurrence_id: str,
    block_index: int,
    source_start: float,
    source_end: float,
    output_start: float,
    scenes: Sequence[SceneSpec],
) -> TimelineOccurrence:
    return TimelineOccurrence(
        occurrence_id=occurrence_id,
        block_index=block_index,
        source_start=source_start,
        source_end=source_end,
        output_start=output_start,
        output_end=output_start + (source_end - source_start),
        source_role="HERO" if block_index == 0 else "SUPPORT",
        is_hero=block_index == 0,
        scenes=tuple(scenes),
    )


def make_spec(
    *,
    occurrences: Sequence[TimelineOccurrence] | None = None,
    caption_events: Sequence[CaptionEventSpec] | None = None,
    omitted: Sequence[OmittedRequirement] = (),
    delivery_profile_key: str = "MP4_H264_AAC_1080X1920_V1",
    artifact_purpose: str = "CORE_SOURCE_VALIDATION",
    source_frame_rate: Fraction = Fraction(30, 1),
    output_frame_rate: Fraction = Fraction(30, 1),
    source_duration: float = 90.0,
    rotation_degrees: int = 0,
    pixel_aspect_ratio: float = 1.0,
    audio_channels: int = 2,
    source_video_start: float = 0.0,
    source_audio_start: float = 0.0,
    display_width: int = 1920,
    display_height: int = 1080,
) -> RenderSpec:
    if occurrences is None:
        occurrences = (
            occurred(
                "block-0",
                0,
                10.0,
                12.0,
                0.0,
                (
                    crop_scene(
                        0,
                        0,
                        10.0,
                        11.0,
                        "STATIC_CROP",
                        [(10.0, 0.5, 0.5, 0.8), (11.0, 0.5, 0.5, 0.8)],
                    ),
                    SceneSpec(
                        scene_index=1,
                        block_index=0,
                        source_start=11.0,
                        source_end=12.0,
                        framing_mode="SOURCE_AS_IS",
                        interpolation_policy="smoothstep-ease",
                    ),
                ),
            ),
            occurred(
                "block-1",
                1,
                30.0,
                33.0,
                2.0,
                (
                    SceneSpec(
                        scene_index=2,
                        block_index=1,
                        source_start=30.0,
                        source_end=33.0,
                        framing_mode="BACKGROUND_FILL",
                        interpolation_policy="smoothstep-ease",
                    ),
                ),
            ),
        )
    if caption_events is None:
        caption_events = (CaptionEventSpec("event-1", 0, 10.5, 11.5),)
    return RenderSpec(
        render_execution_id="00000000-0000-0000-0000-000000000001",
        candidate_id="candidate",
        source_id="source",
        render_contract_id="contract",
        visual_plan_id="plan",
        artifact_purpose=artifact_purpose,
        source_media_relative_path="sources/source/source.mp4",
        source_content_hash="abc",
        source_size_bytes=2048,
        source_mtime_ns=1,
        source_duration=source_duration,
        source_frame_rate=source_frame_rate,
        display_width=display_width,
        display_height=display_height,
        encoded_width=display_width,
        encoded_height=display_height,
        rotation_degrees=rotation_degrees,
        pixel_aspect_ratio=pixel_aspect_ratio,
        output_width=1080,
        output_height=1920,
        output_frame_rate=output_frame_rate,
        output_profile={
            "profile_key": "SHORTS_1080X1920",
            "width": 1080,
            "height": 1920,
            "target_frame_rate": float(output_frame_rate),
        },
        delivery_profile_key=delivery_profile_key,
        plan_input_fingerprint="plan-in",
        plan_output_fingerprint="plan-out",
        ass=AssAsset(
            relative_path="sources/source/visual-composition/fp.ass",
            sha256="",
            event_count=len(caption_events),
            line_count=1,
            policy_version="stage5.1-ass-v5",
        ),
        caption_events=tuple(caption_events),
        occurrences=tuple(occurrences),
        omitted=tuple(omitted),
        audio_channels=audio_channels,
        source_video_start_seconds=source_video_start,
        source_audio_start_seconds=source_audio_start,
    )


def install_fake_stream_probe(monkeypatch: Any, *, channels: int = 2) -> None:
    """Inject deterministic stream facts for hermetic DB tests (no real probe)."""

    from app.render.execution import service

    monkeypatch.setattr(
        service,
        "_SOURCE_STREAM_PROBE_OVERRIDE",
        lambda path, settings: channels,
    )


def scene_geometry(compiled: CompiledRender, index: int = 0) -> dict[str, Any]:
    """Typed access to one scene's compiler diagnostics (crop geometry/samples)."""

    scenes = cast(list[dict[str, Any]], compiled.diagnostics.get("scenes", []))
    geometry = scenes[index].get("geometry")
    assert isinstance(geometry, dict)
    return cast(dict[str, Any], geometry)


def independent_tracked_geometry(
    frames: Sequence[CropKeyframeSpec],
    t: float,
    display_width: int,
    display_height: int,
) -> dict[str, float]:
    """Independent reference geometry for one accepted tracked-crop keyframe set.

    Deliberately re-derives the accepted smoothstep interpolation, the even
    scale rounding, the fixed-output crop position, and the output-center ->
    source-pixel mapping without importing or calling the compiler helper. Used
    by regressions to check encoded pixels against an external calculation.
    """

    def interp(getter: str) -> float:
        values = [getattr(frame, getter) for frame in frames]
        if len(frames) == 1 or all(abs(value - values[0]) < 1e-9 for value in values):
            return float(values[0])
        if t < frames[0].t:
            return float(values[0])
        for index in range(len(frames) - 1):
            left = frames[index]
            right = frames[index + 1]
            if t < right.t:
                span = right.t - left.t
                if span < 1e-9:
                    return float(getattr(left, getter))
                u = (t - left.t) / span
                ease = u * u * (3.0 - 2.0 * u)
                return float(
                    getattr(left, getter) + (getattr(right, getter) - getattr(left, getter)) * ease
                )
        return float(values[-1])

    height_fraction = min(1.0, max(0.20, interp("height_fraction")))
    scaled_height = math.trunc((1920.0 / height_fraction) / 2.0) * 2
    scaled_width = (
        math.trunc((display_width * 1920.0 / (height_fraction * display_height)) / 2.0) * 2
    )
    crop_x = min(max(interp("cx") * scaled_width - 540.0, 0.0), scaled_width - 1080)
    crop_y = min(max(interp("cy") * scaled_height - 960.0, 0.0), scaled_height - 1920)
    return {
        "scaled_width": float(scaled_width),
        "scaled_height": float(scaled_height),
        "crop_x": crop_x,
        "crop_y": crop_y,
        "source_x": (crop_x + 540.0) * display_width / scaled_width,
        "source_y": (crop_y + 960.0) * display_height / scaled_height,
    }


__all__ = [
    "crop_scene",
    "fake_runtime",
    "independent_tracked_geometry",
    "install_fake_stream_probe",
    "make_spec",
    "occurred",
    "scene_geometry",
]
