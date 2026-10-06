"""Bounded Stage 5.2 manual-acceptance package builder.

Produces a small set of short playable MP4s plus a manifest, QC results, and
inspection README under ``storage/benchmarks/stage-5-2/manual-acceptance``.

This is an operational script, not a pipeline stage. It uses the production
compiler/runner/QC directly and never touches database rows. Real source A/V is
preserved for human lip-sync and timing inspection; where a crop path or caption
string is demonstration evidence (not an accepted frozen plan), it is labeled as
such in the manifest.

Usage::

    python -m app.render.execution.acceptance --source <media> --output <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path

from app.composition.ass import serialize_ass
from app.composition.captions import CaptionPlan
from app.composition.policy import CaptionStyle, safe_zone_for
from app.composition.types import CaptionEvent, CaptionWordTiming
from app.render.execution.compiler import compile_render
from app.render.execution.policy import Stage52Config
from app.render.execution.qc import check_render_artifact
from app.render.execution.runner import run_compiled_render
from app.render.execution.types import (
    AssAsset,
    AttemptContext,
    CaptionEventSpec,
    CropKeyframeSpec,
    RenderSpec,
    RuntimeIdentity,
    SceneSpec,
    TimelineOccurrence,
)


def _quote_spec(
    *,
    source_path: Path,
    execution_id: str,
    occurrences: tuple[TimelineOccurrence, ...],
    caption_events: tuple[CaptionEventSpec, ...],
    source_duration: float,
    ass_bytes: bytes,
    ass_event_count: int,
    display_width: int,
    display_height: int,
    frame_rate: Fraction,
) -> RenderSpec:
    return RenderSpec(
        render_execution_id=execution_id,
        candidate_id="manual-acceptance",
        source_id="manual-acceptance",
        render_contract_id="manual-acceptance",
        visual_plan_id="manual-acceptance",
        artifact_purpose="CORE_SOURCE_VALIDATION",
        source_media_relative_path=str(source_path),
        source_content_hash="",
        source_size_bytes=source_path.stat().st_size,
        source_mtime_ns=source_path.stat().st_mtime_ns,
        source_duration=source_duration,
        source_frame_rate=frame_rate,
        display_width=display_width,
        display_height=display_height,
        encoded_width=display_width,
        encoded_height=display_height,
        rotation_degrees=0,
        pixel_aspect_ratio=1.0,
        output_width=1080,
        output_height=1920,
        output_frame_rate=frame_rate,
        output_profile={"profile_key": "SHORTS_1080X1920", "width": 1080, "height": 1920},
        delivery_profile_key="MP4_H264_AAC_1080X1920_V1",
        plan_input_fingerprint="manual",
        plan_output_fingerprint="manual",
        ass=AssAsset(
            relative_path="manual/captions.ass",
            sha256="",
            event_count=ass_event_count,
            line_count=ass_event_count,
            policy_version="stage5.1-ass-v5",
        ),
        caption_events=caption_events,
        occurrences=occurrences,
        audio_channels=2,
    )


def _scene(
    index: int,
    block: int,
    start: float,
    end: float,
    mode: str,
    keyframes: tuple[CropKeyframeSpec, ...] = (),
) -> SceneSpec:
    return SceneSpec(
        scene_index=index,
        block_index=block,
        source_start=start,
        source_end=end,
        framing_mode=mode,
        interpolation_policy="smoothstep-ease",
        crop_keyframes=keyframes,
    )


def _occurrence(
    block: int, start: float, end: float, output_start: float, scenes: tuple[SceneSpec, ...]
) -> TimelineOccurrence:
    return TimelineOccurrence(
        occurrence_id=f"block-{block}",
        block_index=block,
        source_start=start,
        source_end=end,
        output_start=output_start,
        output_end=output_start + (end - start),
        source_role="HERO" if block == 0 else "SUPPORT",
        is_hero=block == 0,
        scenes=scenes,
    )


def _mixed_caption_ass() -> tuple[bytes, CaptionEvent, int]:
    """Build the mixed Arabic/English caption via production ASS serialization."""

    text = "أنا كنت content creator لمدة سنتين"
    tokens = text.split(" ")
    start = 85.5
    word_timings = tuple(
        CaptionWordTiming(
            index=index, text=token, start=start + index * 0.42, end=start + index * 0.42 + 0.40
        )
        for index, token in enumerate(tokens)
    )
    event = CaptionEvent(
        event_id="manual-mixed",
        block_index=0,
        word_start_index=0,
        word_end_index=len(tokens) - 1,
        start=start,
        end=start + len(tokens) * 0.42,
        text=text,
        lines=(text,),
        placement_zone="LOWER",
        placement_reason="MANUAL_ACCEPTANCE",
        word_timings=word_timings,
    )
    plan = CaptionPlan(policy_version="stage5.1-caption-layout-v4", events=(event,))
    style = CaptionStyle()
    safe_zone = safe_zone_for("SHORTS_VERTICAL_SAFE_ZONE_V1")
    return serialize_ass(plan, style, safe_zone), event, len(tokens)


def _runtime(source: Path, attempt: Path) -> RuntimeIdentity:
    return RuntimeIdentity(
        ffmpeg_version="ffmpeg",
        ffprobe_version="ffprobe",
        libavformat_version="",
        libass_version="",
        font_family="Noto Sans Arabic",
        font_match="",
        compiler_version="stage5.2-compiler-v3",
        policy_version="stage5.2-v3",
        source_absolute_path=str(source),
        attempt_directory=str(attempt),
    )


def build(source: Path, output: Path, config: Stage52Config) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    display_width, display_height, frame_rate, source_duration = _probe_media(source)
    cases: list[dict[str, object]] = []

    # 01 tracked speaker (real source, synthetic smooth tracked path)
    tracked = _scene(
        0,
        0,
        85.2,
        88.2,
        "TRACKED_CROP",
        (
            CropKeyframeSpec(85.2, 0.45, 0.48, 0.95),
            CropKeyframeSpec(86.2, 0.50, 0.46, 0.82),
            CropKeyframeSpec(87.2, 0.55, 0.44, 0.70),
            CropKeyframeSpec(88.2, 0.52, 0.45, 0.78),
        ),
    )
    cases.append(
        _render_case(
            name="01-tracked-speaker.mp4",
            output=output,
            source=source,
            config=config,
            spec=_quote_spec(
                source_path=source,
                execution_id="manual-01",
                occurrences=(_occurrence(0, 85.2, 88.2, 0.0, (tracked,)),),
                caption_events=(),
                source_duration=source_duration,
                ass_bytes=b"",
                ass_event_count=0,
                display_width=display_width,
                display_height=display_height,
                frame_rate=frame_rate,
            ),
            ass_bytes=_empty_ass(),
            label=(
                "real source A/V; synthetic smoothstep tracked crop path "
                "(not the frozen Stage 5.1 plan)"
            ),
        )
    )

    # 02 multi-person background fill
    fill = _scene(0, 0, 71.78, 74.28, "BACKGROUND_FILL")
    cases.append(
        _render_case(
            name="02-multi-person-background-fill.mp4",
            output=output,
            source=source,
            config=config,
            spec=_quote_spec(
                source_path=source,
                execution_id="manual-02",
                occurrences=(_occurrence(0, 71.78, 74.28, 0.0, (fill,)),),
                caption_events=(),
                source_duration=source_duration,
                ass_bytes=b"",
                ass_event_count=0,
                display_width=display_width,
                display_height=display_height,
                frame_rate=frame_rate,
            ),
            ass_bytes=_empty_ass(),
            label="real source A/V; accepted BACKGROUND_FILL background treatment",
        )
    )

    # 03 mixed bidi highlight
    ass_bytes, event, token_count = _mixed_caption_ass()
    static = _scene(0, 0, 84.8, 88.4, "STATIC_CROP", (CropKeyframeSpec(84.8, 0.5, 0.5, 0.95),))
    cases.append(
        _render_case(
            name="03-mixed-bidi-highlight.mp4",
            output=output,
            source=source,
            config=config,
            spec=_quote_spec(
                source_path=source,
                execution_id="manual-03",
                occurrences=(_occurrence(0, 84.8, 88.4, 0.0, (static,)),),
                caption_events=(CaptionEventSpec("manual-mixed", 0, event.start, event.end),),
                source_duration=source_duration,
                ass_bytes=ass_bytes,
                ass_event_count=1,
                display_width=display_width,
                display_height=display_height,
                frame_rate=frame_rate,
            ),
            ass_bytes=ass_bytes,
            label=(
                "real source A/V; DEMO caption text (not the speaker's words) "
                "with advancing active-word color"
            ),
        )
    )

    # 04 multi-span A/V sync (noncontiguous source spans)
    scene_a = _scene(0, 0, 30.0, 32.0, "SOURCE_AS_IS")
    scene_b = _scene(1, 1, 85.2, 88.2, "SOURCE_AS_IS")
    cases.append(
        _render_case(
            name="04-multi-span-av-sync.mp4",
            output=output,
            source=source,
            config=config,
            spec=_quote_spec(
                source_path=source,
                execution_id="manual-04",
                occurrences=(
                    _occurrence(0, 30.0, 32.0, 0.0, (scene_a,)),
                    _occurrence(1, 85.2, 88.2, 2.0, (scene_b,)),
                ),
                caption_events=(),
                source_duration=source_duration,
                ass_bytes=b"",
                ass_event_count=0,
                display_width=display_width,
                display_height=display_height,
                frame_rate=frame_rate,
            ),
            ass_bytes=_empty_ass(),
            label=(
                "real source A/V across two noncontiguous spans; inspect A/V sync across the join"
            ),
        )
    )

    manifest = {
        "stage": "5.2",
        "purpose": "CORE_SOURCE_VALIDATION",
        "source": str(source),
        "cases": cases,
        "publication_ready": False,
        "stage6_implemented": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(_readme(manifest), encoding="utf-8")
    return manifest


def _render_case(
    *,
    name: str,
    output: Path,
    source: Path,
    config: Stage52Config,
    spec: RenderSpec,
    ass_bytes: bytes,
    label: str,
) -> dict[str, object]:
    case_dir = output / name.replace(".mp4", "")
    case_dir.mkdir(parents=True, exist_ok=True)
    compiled = compile_render(spec, _runtime(source, case_dir))
    artifacts = run_compiled_render(
        compiled,
        AttemptContext(
            attempt_directory=case_dir,
            timeout_seconds=600,
            ass_bytes=ass_bytes if ass_bytes else _empty_ass(),
        ),
    )
    qc = check_render_artifact(artifacts, artifacts.manifest, config, source_path=source)
    (case_dir / "qc.json").write_text(json.dumps(qc.as_dict(), indent=2), encoding="utf-8")
    produced = case_dir / "output.mp4"
    target = output / name
    produced.replace(target)
    return {
        "name": name,
        "path": str(target),
        "evidence": label,
        "probe": artifacts.probe,
        "duration_seconds": artifacts.duration_seconds,
        "wc": qc.status,
        "qc_reason_codes": list(qc.reason_codes),
        "source_intervals": [
            {
                "source_start": occ.source_start,
                "source_end": occ.source_end,
                "output_start": occ.output_start,
                "output_end": occ.output_end,
            }
            for occ in spec.occurrences
        ],
    }


def _probe_media(source: Path) -> tuple[int, int, Fraction, float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    duration = float(payload.get("format", {}).get("duration") or 300.0)
    return (
        int(stream["width"]),
        int(stream["height"]),
        _fraction(stream.get("r_frame_rate")),
        duration,
    )


def _fraction(text: object) -> Fraction:
    if isinstance(text, str) and "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return Fraction(int(numerator), int(denominator))
        except (ValueError, ZeroDivisionError):
            pass
    return Fraction(30, 1)


def _empty_ass() -> bytes:
    return (
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
    ).encode("utf-8")


def _readme(manifest: dict[str, object]) -> str:
    lines = [
        "# Stage 5.2 manual acceptance",
        "",
        "Short playable source-core validation MP4s for human inspection.",
        "These are terminal validation outputs, not publication-final artifacts.",
        "",
        "## Cases",
        "",
    ]
    for case in _case_list(manifest):
        lines.append(f"- `{case.get('name')}` — {case.get('evidence')} (QC {case.get('wc')})")
    lines += [
        "",
        "## Inspect",
        "",
        "1. Lip sync and source-cut timing.",
        "2. Crop correctness/smoothness (case 01 pan/zoom).",
        "3. Black flashes and caption timing/highlighting (case 03).",
        "4. Mixed Arabic/English BiDi order and stable surrounding layout.",
        "5. Sharpness and background fill (case 02).",
        "6. Source-audio level and multi-span synchronization (case 04).",
        "",
        "Automatic QC proves structural/timing correctness only; human visual",
        "acceptance is required.",
        "",
    ]
    return "\n".join(lines)


def _case_list(manifest: dict[str, object]) -> list[dict[str, object]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Stage 5.2 manual-acceptance package.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(args.source, args.output, Stage52Config())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": [case.get("name") for case in _case_list(manifest)],
            }
        )
    )


if __name__ == "__main__":
    main()
