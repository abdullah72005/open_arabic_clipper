"""Stage 6 authored timeline and audio-recipe construction.

The authored timeline is an immutable, versioned map from plan blocks to output
segments at unit source speed. Actual measured audio duration and readable text
layout determine authored durations; Stage 4/5 authored times are only estimates.
The hero interval is protected from trimming, narration interruption, source
replacement, and obscuring overlays. Repeated occurrences remain distinct and
cumulative renderer quantization is reused for boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from app.materialization.inputs import MaterializationInputs
from app.materialization.policy import (
    MAX_BOUNDARY_TRIM_SECONDS,
    MIN_RETAINED_BREATHING_ROOM_SECONDS,
    STAGE6_TIMELINE_SCHEMA_VERSION,
    Stage6Config,
)
from app.materialization.types import (
    AudioEntry,
    AudioRecipe,
    AuthoredBlock,
    AuthoredTimeline,
    NarrationAudio,
    SlotResolution,
    TimelineSegment,
)

READING_CHARS_PER_SECOND = 14.0
MIN_AUTHORED_CARD_SECONDS = 1.2
MAX_AUTHORED_CARD_SECONDS = 8.0
_TRANSITION_BRIDGE_SECONDS = 0.4


@dataclass(frozen=True)
class BoundaryTrimEvidence:
    """Agreeing acoustic + word evidence for an optional boundary-silence trim."""

    block_index: int
    leading_silence_seconds: float = 0.0
    trailing_silence_seconds: float = 0.0
    word_evidence_agrees: bool = False


@dataclass(frozen=True)
class TimelineBuild:
    """Timeline build result plus any recorded boundary removals."""

    timeline: AuthoredTimeline
    removals: tuple[Mapping[str, object], ...] = ()
    conflicts: tuple[Mapping[str, object], ...] = ()


def _estimate_authored_seconds(
    block: Mapping[str, object], copy: str, narration_seconds: float
) -> float:
    if narration_seconds > 0:
        return max(MIN_AUTHORED_CARD_SECONDS, min(MAX_AUTHORED_CARD_SECONDS, narration_seconds))
    if copy:
        estimate = len(copy) / READING_CHARS_PER_SECOND
        return max(MIN_AUTHORED_CARD_SECONDS, min(MAX_AUTHORED_CARD_SECONDS, estimate))
    duration = block.get("estimated_duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
        return float(duration)
    return MIN_AUTHORED_CARD_SECONDS


def _quantize(value: float, frame_rate: Fraction) -> float:
    if frame_rate <= 0:
        return value
    frames = round(value * frame_rate)
    return float(Fraction(frames, 1) / frame_rate)


def build_timeline(
    inputs: MaterializationInputs,
    authored_blocks: Sequence[AuthoredBlock],
    narration: Sequence[NarrationAudio],
    *,
    supporting_by_block: Mapping[int, str] | None = None,
    slot_resolutions: Sequence[SlotResolution] = (),
    frame_rate: Fraction | None = None,
    trim_evidence: Sequence[BoundaryTrimEvidence] = (),
    config: Stage6Config,
) -> TimelineBuild:
    """Build the immutable authored timeline (source + authored cards)."""

    resolved_frame_rate = frame_rate or inputs.source_media.frame_rate or Fraction(30, 1)
    authored_by_index = {block.block_index: block for block in authored_blocks}
    narration_by_block = {chunk.block_index: chunk for chunk in narration}
    supporting_by_block = dict(supporting_by_block or {})
    trims = {item.block_index: item for item in trim_evidence}
    hero_index = inputs.hero_block_index

    segments: list[TimelineSegment] = []
    removals: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    cursor = 0.0
    occurrence = 0

    for block in inputs.blocks:
        index = _int(block.get("index", -1))
        block_type = str(block.get("block_type", ""))
        occurrence += 1
        segment_id = f"seg-{occurrence}-{index}"

        if block_type == "SOURCE_EXCERPT":
            start = _float(block.get("source_start"))
            end = _float(block.get("source_end"))
            if end <= start:
                conflicts.append({"block_index": index, "reason": "INVALID_SOURCE_SPAN"})
                continue
            is_hero = index == hero_index
            trim = trims.get(index)
            trim_removed = 0.0
            if trim is not None and not is_hero and trim.word_evidence_agrees:
                leading = min(trim.leading_silence_seconds, MAX_BOUNDARY_TRIM_SECONDS)
                trailing = min(trim.trailing_silence_seconds, MAX_BOUNDARY_TRIM_SECONDS)
                # Retain breathing room; never trim into speech/thought.
                leading = max(0.0, leading - MIN_RETAINED_BREATHING_ROOM_SECONDS)
                trailing = max(0.0, trailing - MIN_RETAINED_BREATHING_ROOM_SECONDS)
                start += leading
                end -= trailing
                trim_removed = leading + trailing
                if trim_removed > 0:
                    removals.append(
                        {
                            "block_index": index,
                            "removed_seconds": round(trim_removed, 6),
                            "evidence": "acoustic_silence+word_agreement",
                        }
                    )
            duration = end - start
            output_start = _quantize(cursor, resolved_frame_rate)
            output_end = _quantize(cursor + duration, resolved_frame_rate)
            segments.append(
                TimelineSegment(
                    segment_id=segment_id,
                    plan_block_index=index,
                    kind="SOURCE",
                    output_start=output_start,
                    output_end=output_end,
                    source_start=round(start, 6),
                    source_end=round(end, 6),
                    word_start_index=_optional_int(block.get("word_start_index")),
                    word_end_index=_optional_int(block.get("word_end_index")),
                    is_hero=is_hero,
                )
            )
            cursor = output_end

        elif block_type in {"ORIGINAL_VALUE", "TEXTUAL_ANNOTATION"}:
            authored = authored_by_index.get(index)
            chunk = narration_by_block.get(index)
            copy = authored.copy if authored is not None else ""
            narration_seconds = chunk.duration_seconds if chunk is not None else 0.0
            duration = _estimate_authored_seconds(block, copy, narration_seconds)
            output_start = _quantize(cursor, resolved_frame_rate)
            output_end = _quantize(cursor + duration, resolved_frame_rate)
            segments.append(
                TimelineSegment(
                    segment_id=segment_id,
                    plan_block_index=index,
                    kind="AUTHORED_CARD",
                    output_start=output_start,
                    output_end=output_end,
                    text=copy,
                    narration_chunk_id=chunk.chunk_id if chunk is not None else None,
                    asset_id=supporting_by_block.get(index),
                )
            )
            cursor = output_end

        elif block_type == "TRANSITION":
            duration = _TRANSITION_BRIDGE_SECONDS
            output_start = _quantize(cursor, resolved_frame_rate)
            output_end = _quantize(cursor + duration, resolved_frame_rate)
            segments.append(
                TimelineSegment(
                    segment_id=segment_id,
                    plan_block_index=index,
                    kind="TRANSITION",
                    output_start=output_start,
                    output_end=output_end,
                    transition="CUT",
                )
            )
            cursor = output_end

        elif block_type == "FACT_VERIFICATION_PLACEHOLDER":
            # A dependency, never blank video: a zero-duration explicit marker.
            segments.append(
                TimelineSegment(
                    segment_id=segment_id,
                    plan_block_index=index,
                    kind="VERIFICATION_DEPENDENCY",
                    output_start=_quantize(cursor, resolved_frame_rate),
                    output_end=_quantize(cursor, resolved_frame_rate),
                    text=str(block.get("claim_dependency", "") or ""),
                )
            )

    output_duration = segments[-1].output_end if segments else 0.0
    frame_count = int(round(output_duration * resolved_frame_rate))
    sample_rate = inputs.source_media.sample_rate or config.target_sample_rate
    sample_count = int(round(output_duration * sample_rate))

    timeline = AuthoredTimeline(
        segments=tuple(segments),
        frame_rate=resolved_frame_rate,
        sample_rate=sample_rate,
        output_duration=output_duration,
        output_frame_count=frame_count,
        output_sample_count=sample_count,
        source_caption_asset=inputs.visual_plan.ass_relative_path,
        slot_fulfillment=tuple(slot_resolutions),
        schema_version=STAGE6_TIMELINE_SCHEMA_VERSION,
    )
    return TimelineBuild(timeline=timeline, removals=tuple(removals), conflicts=tuple(conflicts))


def hero_is_protected(timeline: AuthoredTimeline, inputs: MaterializationInputs) -> bool:
    """Whether the hero interval is a source segment at unit speed, uninterrupted."""

    hero_segments = [seg for seg in timeline.segments if seg.is_hero]
    if not hero_segments:
        return False
    hero = hero_segments[0]
    if hero.kind != "SOURCE" or hero.source_start is None or hero.source_end is None:
        return False
    source_duration = hero.source_end - hero.source_start
    if abs(hero.output_duration - source_duration) > 1e-6:
        return False
    hero_start, hero_end = hero.output_start, hero.output_end
    for segment in timeline.segments:
        if segment.segment_id == hero.segment_id:
            continue
        # Any authored/narration segment strictly inside the hero interval
        # interrupts the protected source moment.
        if (
            segment.kind == "AUTHORED_CARD"
            and segment.output_start < hero_end
            and segment.output_end > hero_start
        ):
            return False
        if (
            segment.narration_chunk_id
            and segment.output_start < hero_end
            and segment.output_end > hero_start
        ):
            return False
    return True


def build_audio_recipe(
    inputs: MaterializationInputs,
    timeline: AuthoredTimeline,
    narration: Sequence[NarrationAudio],
    *,
    allow_overlap: bool = False,
    config: Stage6Config,
) -> tuple[AudioRecipe, tuple[Mapping[str, object], ...]]:
    """Build the final mix recipe and report any competing-speech conflicts."""

    entries: list[AudioEntry] = []
    conflicts: list[dict[str, object]] = []
    narration_by_block = {chunk.block_index: chunk for chunk in narration}
    narration_intervals: list[tuple[float, float, int]] = []

    for index, segment in enumerate(timeline.segments):
        if segment.kind == "SOURCE" and segment.source_start is not None:
            entries.append(
                AudioEntry(
                    entry_id=f"src-{index}",
                    kind="SOURCE",
                    output_start=segment.output_start,
                    output_end=segment.output_end,
                    source_start=segment.source_start,
                    source_end=segment.source_end,
                )
            )
        elif segment.kind == "AUTHORED_CARD" and segment.narration_chunk_id:
            chunk = narration_by_block.get(segment.plan_block_index)
            if chunk is not None:
                entries.append(
                    AudioEntry(
                        entry_id=f"narr-{index}",
                        kind="NARRATION",
                        output_start=segment.output_start,
                        output_end=segment.output_end,
                        narration_chunk_id=chunk.chunk_id,
                    )
                )
                narration_intervals.append(
                    (segment.output_start, segment.output_end, segment.plan_block_index)
                )

    ducking: list[dict[str, object]] = []
    for source in [entry for entry in entries if entry.kind == "SOURCE"]:
        for start, end, block_index in narration_intervals:
            overlap = min(source.output_end, end) - max(source.output_start, start)
            if overlap <= 0:
                continue
            if allow_overlap:
                ducking.append(
                    {
                        "source_entry_id": source.entry_id,
                        "narration_block_index": block_index,
                        "start": round(max(source.output_start, start), 6),
                        "end": round(min(source.output_end, end), 6),
                        "ducking_db": -12.0,
                    }
                )
            else:
                conflicts.append(
                    {
                        "reason": "COMPETING_SPEECH_OVERLAP",
                        "source_entry_id": source.entry_id,
                        "narration_block_index": block_index,
                        "overlap_seconds": round(overlap, 6),
                    }
                )

    recipe = AudioRecipe(
        entries=tuple(entries),
        ducking_regions=tuple(ducking),
        target_integrated_lufs=config.target_integrated_lufs,
        true_peak_ceiling_dbtp=config.true_peak_ceiling_dbtp,
        sample_rate=config.target_sample_rate,
        channels=config.target_channels,
        supporting_audio_muted=True,
        recipe_version="stage6-audio-v1",
    )
    return recipe, tuple(conflicts)


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return -1
    return value


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


__all__ = [
    "BoundaryTrimEvidence",
    "TimelineBuild",
    "build_audio_recipe",
    "build_timeline",
    "hero_is_protected",
]
