"""Deterministic Stage 5.2 FFmpeg compiler.

Produces one final video encode and one final audio encode from the accepted
Stage 5.1 visual plan and canonical ASS. The graph:

- decodes the single managed source once;
- trims each selected source occurrence/scene with explicit ``trim``/``atrim``;
- burns the canonical ASS while timestamps are source-local (preserving the
  original ASS bytes and dynamic highlight states), then rebases with
  ``setpts``;
- joins video per scene inside each occurrence and joins audio only at real
  occurrence boundaries (never per visual scene), so visual splitting cannot
  repeatedly round or pad audio;
- concatenates occurrences and encodes once.

The installed FFmpeg's ``crop`` filter does not expose runtime width/height
commands, so a genuinely changing ``height_fraction`` is executed with a
per-frame dynamic ``scale`` (``eval=frame``) plus a per-frame ``crop`` position.
This is the smallest proven equivalent; the accepted normalized crop path is
not redesigned.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from app.composition.geometry import clamp_crop
from app.render.execution.fingerprints import render_compiled_fingerprint
from app.render.execution.policy import (
    COMPILER_VERSION,
    MAX_FILTERGRAPH_CHARACTERS,
    MAX_OUTPUT_DURATION_SECONDS,
    MAX_SEGMENTS,
    MAX_SOURCE_DURATION_SECONDS,
    MAX_TRACKED_SCENE_KEYFRAMES,
    OUTPUT_PROFILE_HEIGHT,
    OUTPUT_PROFILE_WIDTH,
    SOURCE_START_TOLERANCE_SECONDS,
    DeliveryProfile,
    delivery_profile_for,
)
from app.render.execution.timeline import build_timeline
from app.render.execution.types import (
    CompiledRender,
    CropKeyframeSpec,
    FilterNode,
    RenderSpec,
    RuntimeIdentity,
    SceneSpec,
)

_ASS_LOCAL_NAME = "captions.ass"
_OUTPUT_NAME = "output.mp4"
_FILTERGRAPH_NAME = "filtergraph.txt"
_CROP_MIN_HEIGHT_FRACTION = 0.20
_SOURCE_START_EPSILON = SOURCE_START_TOLERANCE_SECONDS


class CompileError(ValueError):
    """A render could not be compiled deterministically."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _num(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _esc(expression: str) -> str:
    return expression.replace(",", "\\,").replace(":", "\\:")


def _lerp_expression(left: CropKeyframeSpec, right: CropKeyframeSpec, getter: str) -> str:
    left_value = _num(getattr(left, getter))
    right_value = _num(getattr(right, getter))
    if abs(right.t - left.t) < 1e-9:
        return left_value
    t0 = _num(left.t)
    span = _num(right.t - left.t)
    u = f"((t-({t0}))/({span}))"
    ease = f"(({u})*({u})*(3-2*({u})))"
    return f"({left_value}+(({right_value})-({left_value}))*{ease})"


def _piecewise(keyframes: Sequence[CropKeyframeSpec], getter: str) -> str:
    """Build a smoothstep-eased piecewise expression over ordered keyframes."""

    values = [getattr(frame, getter) for frame in keyframes]
    if len(keyframes) == 1 or all(abs(value - values[0]) < 1e-9 for value in values):
        return _num(values[0])
    expression = _num(values[-1])
    for index in range(len(keyframes) - 2, -1, -1):
        segment = _lerp_expression(keyframes[index], keyframes[index + 1], getter)
        boundary = _num(keyframes[index + 1].t)
        expression = f"if(lt(t,({boundary})),{segment},{expression})"
    first_t = _num(keyframes[0].t)
    expression = f"if(lt(t,({first_t})),{_num(values[0])},{expression})"
    return _esc(expression)


def _has_varying(keyframes: Sequence[CropKeyframeSpec], getter: str) -> bool:
    values = [getattr(frame, getter) for frame in keyframes]
    return any(abs(value - values[0]) > 1e-9 for value in values)


def _static_crop_chain(spec: RenderSpec, scene: SceneSpec) -> tuple[str, dict[str, object]]:
    keyframe = scene.crop_keyframes[0]
    height = keyframe.height_fraction * spec.display_height
    width = height * 9.0 / 16.0
    center_x = keyframe.cx * spec.display_width
    center_y = keyframe.cy * spec.display_height
    x, y, crop_width, crop_height = clamp_crop(
        center_x - width / 2.0,
        center_y - height / 2.0,
        width,
        height,
        float(spec.display_width),
        float(spec.display_height),
    )
    even_width = max(2, int(round(crop_width)) // 2 * 2)
    even_height = max(2, int(round(crop_height)) // 2 * 2)
    even_x = min(max(int(round(x)), 0), spec.display_width - even_width)
    even_y = min(max(int(round(y)), 0), spec.display_height - even_height)
    chain = (
        f"crop=w={even_width}:h={even_height}:x={even_x}:y={even_y},"
        f"scale={OUTPUT_PROFILE_WIDTH}:{OUTPUT_PROFILE_HEIGHT}:flags=bicubic"
    )
    return chain, {
        "crop_width": even_width,
        "crop_height": even_height,
        "crop_x": even_x,
        "crop_y": even_y,
    }


def _tracked_crop_chain(scene: SceneSpec) -> tuple[str, dict[str, object]]:
    frames = scene.crop_keyframes
    if len(frames) > MAX_TRACKED_SCENE_KEYFRAMES:
        raise CompileError("KEYFRAME_OUT_OF_SCOPE")
    if _has_varying(frames, "height_fraction"):
        height_expr = _piecewise(frames, "height_fraction")
    else:
        height_expr = _num(frames[0].height_fraction)
    center_x_expr = _piecewise(frames, "cx") if _has_varying(frames, "cx") else _num(frames[0].cx)
    center_y_expr = _piecewise(frames, "cy") if _has_varying(frames, "cy") else _num(frames[0].cy)
    height_expr = f"min(1\\,max({_num(_CROP_MIN_HEIGHT_FRACTION)}\\,{height_expr}))"
    scale_w = f"trunc(iw*(1920/(({height_expr})*ih))/2)*2"
    scale_h = f"trunc((1920/({height_expr}))/2)*2"
    half_w = OUTPUT_PROFILE_WIDTH // 2
    half_h = OUTPUT_PROFILE_HEIGHT // 2
    crop_x = f"max(0\\,min(in_w-{OUTPUT_PROFILE_WIDTH}\\,({center_x_expr})*in_w-{half_w}))"
    crop_y = f"max(0\\,min(in_h-{OUTPUT_PROFILE_HEIGHT}\\,({center_y_expr})*in_h-{half_h}))"
    chain = (
        f"scale=w='{scale_w}':h='{scale_h}':eval=frame:flags=bicubic,"
        f"crop=w={OUTPUT_PROFILE_WIDTH}:h={OUTPUT_PROFILE_HEIGHT}:x='{crop_x}':y='{crop_y}'"
    )
    return chain, {
        "dynamic": True,
        "height_expression": height_expr,
        "center_x_expression": center_x_expr,
        "center_y_expression": center_y_expr,
    }


def _framing_segment(
    spec: RenderSpec, scene: SceneSpec, input_label: str, output_label: str
) -> tuple[list[str], dict[str, object]]:
    mode = scene.framing_mode
    if mode == "SOURCE_AS_IS":
        chain = (
            f"scale={OUTPUT_PROFILE_WIDTH}:{OUTPUT_PROFILE_HEIGHT}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={OUTPUT_PROFILE_WIDTH}:{OUTPUT_PROFILE_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        return [f"[{input_label}]{chain}[{output_label}]"], {"mode": mode}
    if mode == "BACKGROUND_FILL":
        lines = [
            f"[{input_label}]split=2[bgr{output_label}][fgr{output_label}]",
            f"[bgr{output_label}]scale={OUTPUT_PROFILE_WIDTH}:{OUTPUT_PROFILE_HEIGHT}:"
            "force_original_aspect_ratio=increase,"
            f"crop={OUTPUT_PROFILE_WIDTH}:{OUTPUT_PROFILE_HEIGHT},"
            f"gblur=sigma=36:steps=2,eq=brightness=-0.18:saturation=0.70[bgc{output_label}]",
            f"[fgr{output_label}]scale={OUTPUT_PROFILE_WIDTH}:{OUTPUT_PROFILE_HEIGHT}:"
            f"force_original_aspect_ratio=decrease[fgs{output_label}]",
            f"[bgc{output_label}][fgs{output_label}]overlay=(W-w)/2:(H-h)/2[{output_label}]",
        ]
        return lines, {"mode": mode}
    if mode == "TRACKED_CROP":
        chain, geometry = _tracked_crop_chain(scene)
        return [f"[{input_label}]{chain}[{output_label}]"], {"mode": mode, **geometry}
    chain, geometry = _static_crop_chain(spec, scene)
    return [f"[{input_label}]{chain}[{output_label}]"], {"mode": mode, **geometry}


def _container_codec(encoder: str) -> str:
    return {
        "libx264": "h264",
        "libx265": "hevc",
        "libvpx-vp9": "vp9",
        "libvpx": "vp8",
    }.get(encoder, encoder)


def _occurrence_leading_gap(stream_start_seconds: float, occurrence_start: float) -> float:
    """Genuine leading offset of a stream relative to the shared occurrence origin.

    A stream that begins after the selected occurrence start has real leading
    silence/freeze that must be preserved (never silently shifted). A stream that
    begins before the occurrence is simply trimmed from the origin.
    """

    return max(0.0, stream_start_seconds - occurrence_start)


def _audio_layout(channels: int) -> str:
    return "mono" if channels <= 1 else "stereo"


def compile_render(spec: RenderSpec, runtime_identity: RuntimeIdentity) -> CompiledRender:
    """Compile a validated render spec into a deterministic FFmpeg command."""

    profile = delivery_profile_for(spec.delivery_profile_key)
    if profile is None:
        raise CompileError("UNSUPPORTED_DELIVERY_PROFILE")
    if spec.source_duration > MAX_SOURCE_DURATION_SECONDS + 1e-6:
        raise CompileError("SOURCE_DURATION_EXCEEDS_LIMIT")
    manifest = build_timeline(spec)
    if manifest.output_duration > MAX_OUTPUT_DURATION_SECONDS + 1e-6:
        raise CompileError("OUTPUT_DURATION_EXCEEDS_LIMIT")

    total_scenes = sum(len(occurrence.scenes) for occurrence in spec.occurrences)
    if total_scenes > MAX_SEGMENTS:
        raise CompileError("SEGMENT_LIMIT_EXCEEDED")
    lines: list[str] = []
    nodes: list[FilterNode] = []
    scene_diagnostics: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {"scenes": scene_diagnostics}

    if total_scenes == 1:
        video_sources = ["0:v"]
    else:
        labels = "".join(f"[vin{index}]" for index in range(total_scenes))
        lines.append(f"[0:v]split={total_scenes}{labels}")
        video_sources = [f"vin{index}" for index in range(total_scenes)]

    if len(spec.occurrences) == 1:
        audio_sources = ["0:a"]
    else:
        labels = "".join(f"[ain{index}]" for index in range(len(spec.occurrences)))
        lines.append(f"[0:a]asplit={len(spec.occurrences)}{labels}")
        audio_sources = [f"ain{index}" for index in range(len(spec.occurrences))]

    scene_cursor = 0
    occurrence_video: list[str] = []
    occurrence_audio: list[str] = []
    for occurrence_index, occurrence in enumerate(spec.occurrences):
        scene_outputs: list[str] = []
        for scene in occurrence.scenes:
            source_label = video_sources[scene_cursor]
            scene_cursor += 1
            trimmed = f"vtr{occurrence_index}_{scene.scene_index}"
            framed = f"vfr{occurrence_index}_{scene.scene_index}"
            scene_out = f"vsc{occurrence_index}_{scene.scene_index}"
            lines.append(
                f"[{source_label}]trim=start={_num(scene.source_start)}:"
                f"end={_num(scene.source_end)}[{trimmed}]"
            )
            segment_lines, geometry = _framing_segment(spec, scene, trimmed, framed)
            lines.extend(segment_lines)
            leading_video_gap = _occurrence_leading_gap(
                spec.source_video_start_seconds, scene.source_start
            )
            tpad = (
                f",tpad=start_duration={_num(leading_video_gap)}:start_mode=clone"
                if leading_video_gap > _SOURCE_START_EPSILON
                else ""
            )
            lines.append(
                f"[{framed}]ass={_ASS_LOCAL_NAME},setsar=1,"
                f"format={profile.pixel_format}{tpad},setpts=PTS-STARTPTS[{scene_out}]"
            )
            scene_outputs.append(scene_out)
            nodes.append(
                FilterNode(
                    kind="video_scene",
                    inputs=(source_label,),
                    output=scene_out,
                    params={
                        "scene_index": scene.scene_index,
                        "framing_mode": scene.framing_mode,
                        **geometry,
                    },
                )
            )
            scene_diagnostics.append(
                {
                    "scene_index": scene.scene_index,
                    "block_index": scene.block_index,
                    "framing_mode": scene.framing_mode,
                    "keyframes": len(scene.crop_keyframes),
                    "dynamic": bool(geometry.get("dynamic", False)),
                }
            )
        if len(scene_outputs) == 1:
            occurrence_video.append(scene_outputs[0])
        else:
            occurrence_label = f"voc{occurrence_index}"
            inputs = "".join(f"[{label}]" for label in scene_outputs)
            lines.append(f"{inputs}concat=n={len(scene_outputs)}:v=1:a=0[{occurrence_label}]")
            occurrence_video.append(occurrence_label)

        layout = _audio_layout(spec.audio_channels)
        audio_label = f"aoc{occurrence_index}"
        gain = (
            ""
            if abs(occurrence.audio.gain_db) < 1e-9
            else f",volume={_num(occurrence.audio.gain_db)}dB"
        )
        duration = max(0.0, occurrence.source_end - occurrence.source_start)
        leading_audio_gap = _occurrence_leading_gap(
            spec.source_audio_start_seconds, occurrence.source_start
        )
        audio_origin = ""
        if leading_audio_gap > _SOURCE_START_EPSILON:
            delay_ms = max(0, int(round(leading_audio_gap * 1000.0)))
            audio_origin = f",adelay={delay_ms}:all=1,apad=whole_dur={_num(duration)}"
        lines.append(
            f"[{audio_sources[occurrence_index]}]atrim=start={_num(occurrence.source_start)}:"
            f"end={_num(occurrence.source_end)}{audio_origin},asetpts=PTS-STARTPTS{gain},"
            f"aresample={profile.audio_sample_rate},"
            f"aformat=sample_fmts=fltp:sample_rates={profile.audio_sample_rate}:"
            f"channel_layouts={layout}[{audio_label}]"
        )
        occurrence_audio.append(audio_label)

    if len(spec.occurrences) == 1:
        lines.append(f"[{occurrence_video[0]}]null[outv]")
        lines.append(f"[{occurrence_audio[0]}]anull[outa]")
    else:
        pairs = "".join(
            f"[{video}][{audio}]" for video, audio in zip(occurrence_video, occurrence_audio)
        )
        lines.append(f"{pairs}concat=n={len(spec.occurrences)}:v=1:a=1[outv][outa]")

    filtergraph = ";".join(lines)
    if len(filtergraph) > MAX_FILTERGRAPH_CHARACTERS:
        raise CompileError("UNSUPPORTED_FRAMING_MODE")

    argv = _build_argv(spec, profile, runtime_identity)
    spec_fingerprint = render_compiled_fingerprint(spec.as_dict())
    compiled_fingerprint = render_compiled_fingerprint(
        {
            "spec": spec.as_dict(),
            "runtime": runtime_identity.as_dict(),
            "filtergraph": filtergraph,
            "compiler_version": COMPILER_VERSION,
        }
    )
    return CompiledRender(
        compiler_version=COMPILER_VERSION,
        fingerprint=compiled_fingerprint,
        spec_fingerprint=spec_fingerprint,
        runtime_identity=runtime_identity,
        manifest=manifest,
        output_relative_path=_OUTPUT_NAME,
        ass_localized_name=_ASS_LOCAL_NAME,
        argv=tuple(argv),
        filtergraph=filtergraph,
        filtergraph_relative_path=_FILTERGRAPH_NAME,
        nodes=tuple(nodes),
        expected_output_duration=manifest.output_duration,
        expected_frame_count=manifest.output_frame_count,
        expected_sample_count=manifest.output_sample_count,
        expected={
            "width": spec.output_width,
            "height": spec.output_height,
            "video_codec": _container_codec(profile.video_codec),
            "pixel_format": profile.pixel_format,
            "audio_codec": profile.audio_codec,
            "audio_sample_rate": profile.audio_sample_rate,
            "channels": 1 if spec.audio_channels <= 1 else 2,
            "output_duration": manifest.output_duration,
            "frame_count": manifest.output_frame_count,
            "sample_count": manifest.output_sample_count,
            "frame_rate": {
                "numerator": spec.output_frame_rate.numerator,
                "denominator": spec.output_frame_rate.denominator,
            },
            "rotation_degrees": 0,
            "artifact_purpose": spec.artifact_purpose,
        },
        diagnostics=diagnostics,
    )


def _build_argv(
    spec: RenderSpec,
    profile: DeliveryProfile,
    runtime_identity: RuntimeIdentity,
) -> list[str]:
    attempt = runtime_identity.attempt_directory
    filtergraph_path = os.path.join(attempt, _FILTERGRAPH_NAME)
    output_path = os.path.join(attempt, _OUTPUT_NAME)
    source_path = runtime_identity.source_absolute_path
    argv: list[str] = [
        runtime_identity.ffmpeg_binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-filter_threads",
        str(max(1, runtime_identity.filter_threads)),
        "-filter_complex_threads",
        str(max(1, runtime_identity.filter_complex_threads)),
        "-i",
        source_path,
        "-filter_complex_script",
        filtergraph_path,
        "-map",
        "[outv]",
        "-map",
        "[outa]",
        "-c:v",
        profile.video_codec,
        "-preset",
        profile.video_preset,
        "-crf",
        str(profile.video_crf),
        "-pix_fmt",
        profile.pixel_format,
        "-threads",
        str(max(1, runtime_identity.encoder_threads)),
        "-c:a",
        profile.audio_codec,
        "-b:a",
        profile.audio_bitrate,
        "-ar",
        str(profile.audio_sample_rate),
        "-sn",
        "-dn",
        "-metadata:s:v:0",
        "rotate=0",
        "-r",
        f"{spec.output_frame_rate.numerator}/{spec.output_frame_rate.denominator}",
    ]
    if profile.faststart:
        argv.extend(["-movflags", "+faststart"])
    argv.extend(["-y", output_path])
    return argv


__all__ = ["CompileError", "compile_render"]
