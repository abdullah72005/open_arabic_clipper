"""Shared Stage 5.2 test helpers (engine-level, DB-free fixtures)."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from app.render.execution.types import (
    AssAsset,
    CaptionEventSpec,
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
        display_width=1920,
        display_height=1080,
        encoded_width=1920,
        encoded_height=1080,
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


__all__ = [
    "crop_scene",
    "fake_runtime",
    "install_fake_stream_probe",
    "make_spec",
    "occurred",
]
