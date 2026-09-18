"""Stage 5.1 bounded span sampling and scene-cut detection tests.

FFmpeg-backed tests generate synthetic fixtures at test time under the storage
TEMPORARY root and are skipped with a clear reason when the host has no
``ffmpeg`` binary. Pure planning/segmentation tests always run.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from app.composition.analysis import (
    AnalysisCancelled,
    AnalysisScopeExceeded,
    FFmpegFrameSampler,
    FFmpegSceneCutDetector,
    FFmpegUnavailableError,
    SampledFrame,
    Span,
    expand_spans_with_context,
    merge_cuts,
    merge_spans,
    parse_scene_cut_times,
    plan_analysis,
    scaled_dimensions,
    segment_scenes,
)
from app.core.settings import get_settings
from app.services.storage import StorageCategory, StorageService

_FFMPEG = shutil.which("ffmpeg")
_VIDEO_WIDTH = 320
_VIDEO_HEIGHT = 240
_MISSING_FFMPEG = "/nonexistent/ffmpeg-stage51"


def _ffmpeg_or_skip() -> str:
    ffmpeg = _FFMPEG
    if ffmpeg is None:
        pytest.skip("ffmpeg binary not available for synthetic media fixtures")
    assert ffmpeg is not None
    return ffmpeg


def _temporary_dir() -> Path:
    storage = StorageService(get_settings().storage_root)
    directory: Path = storage.category_root(StorageCategory.TEMPORARY)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _build_concat_video(ffmpeg: str, destination: Path, sources: Sequence[str]) -> None:
    args = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    for source in sources:
        args.extend(["-f", "lavfi", "-i", source])
    filter_graph = (
        "".join(f"[{index}:v]" for index in range(len(sources)))
        + f"concat=n={len(sources)}:v=1:a=0[out]"
    )
    args.extend(
        [
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(destination),
        ]
    )
    subprocess.run(args, check=True, capture_output=True)


def _solid_color(ffmpeg: str, destination: Path, colors: Sequence[str]) -> None:
    sources = [
        f"color={color}:s={_VIDEO_WIDTH}x{_VIDEO_HEIGHT}:d=1,format=yuv420p" for color in colors
    ]
    _build_concat_video(ffmpeg, destination, sources)


class _RecordingSampler:
    """Fake sampler that records exactly which spans were requested."""

    def __init__(self) -> None:
        self.requested: list[Span] = []
        self.effective_fps: float = 0.0
        self.max_dimension: int = 0

    def samples(
        self,
        spans: Sequence[Span],
        effective_fps: float,
        max_dimension: int,
        cancel_check: object = None,
    ) -> Iterator[SampledFrame]:
        self.requested = list(spans)
        self.effective_fps = effective_fps
        self.max_dimension = max_dimension
        return iter(())


class _FakeSceneCutDetector:
    """Fake detector whose raw cuts are clamped and merged by the shared helper."""

    def __init__(self, raw_cuts: Sequence[float]) -> None:
        self._raw_cuts = list(raw_cuts)

    def cuts(
        self,
        path: Path | str,
        span_start: float,
        span_end: float,
        threshold: float,
        min_scene_seconds: float,
    ) -> tuple[float, ...]:
        return merge_cuts(self._raw_cuts, span_start, span_end, min_scene_seconds)


# --- Pure planning -----------------------------------------------------------


def test_plan_analysis_computes_union_context_and_frame_totals() -> None:
    plan = plan_analysis(
        [(0, 10.0, 20.0), (1, 30.0, 35.0)],
        analysis_fps=2.0,
        max_analysis_seconds=600.0,
        max_analysis_frames=1500,
        scene_context_seconds=0.5,
    )

    assert plan.spans == ((0, 9.5, 20.5), (1, 29.5, 35.5))
    assert plan.total_selected_seconds == 15.0
    assert plan.total_context_seconds == 2.0
    assert plan.effective_fps == 2.0
    assert plan.reduced_fps is False
    assert plan.reason == ""
    assert plan.total_frames == 22 + 12


def test_plan_analysis_merges_overlapping_spans_in_one_block() -> None:
    plan = plan_analysis(
        [(0, 10.0, 20.0), (0, 15.0, 25.0)],
        analysis_fps=2.0,
        max_analysis_seconds=600.0,
        max_analysis_frames=1500,
    )

    assert plan.spans == ((0, 10.0, 25.0),)
    assert plan.total_selected_seconds == 15.0


def test_plan_analysis_raises_when_selected_seconds_exceed_budget() -> None:
    with pytest.raises(AnalysisScopeExceeded):
        plan_analysis(
            [(0, 0.0, 700.0)],
            analysis_fps=2.0,
            max_analysis_seconds=600.0,
            max_analysis_frames=100_000,
        )


def test_plan_analysis_reduces_fps_deterministically() -> None:
    plan = plan_analysis(
        [(0, 0.0, 100.0)],
        analysis_fps=2.0,
        max_analysis_seconds=1000.0,
        max_analysis_frames=100,
        min_fps=1.0,
    )

    assert plan.reduced_fps is True
    assert plan.effective_fps == 1.0
    assert plan.total_frames <= 100
    assert "reduced" in plan.reason


def test_plan_analysis_raises_when_min_fps_cannot_fit() -> None:
    with pytest.raises(AnalysisScopeExceeded):
        plan_analysis(
            [(0, 0.0, 100.0)],
            analysis_fps=2.0,
            max_analysis_seconds=1000.0,
            max_analysis_frames=50,
            min_fps=1.0,
        )


def test_expand_spans_with_context_clamps_at_zero() -> None:
    assert expand_spans_with_context([(0, 0.0, 1.0)], 0.5) == ((0, 0.0, 1.5),)


def test_merge_spans_unions_per_block() -> None:
    assert merge_spans([(0, 5.0, 8.0), (0, 1.0, 2.0), (1, 9.0, 10.0)]) == (
        (0, 1.0, 2.0),
        (0, 5.0, 8.0),
        (1, 9.0, 10.0),
    )


def test_only_selected_spans_with_bounded_context_are_requested() -> None:
    source_duration = 100.0
    selected: list[Span] = [(0, 10.0, 20.0), (1, 30.0, 35.0)]
    plan = plan_analysis(
        selected,
        analysis_fps=2.0,
        max_analysis_seconds=600.0,
        max_analysis_frames=1500,
        scene_context_seconds=0.5,
    )
    sampler = _RecordingSampler()
    list(sampler.samples(plan.spans, plan.effective_fps, 640))

    assert sampler.requested == list(plan.spans)
    assert sampler.effective_fps == plan.effective_fps
    assert sampler.max_dimension == 640
    assert sampler.requested[0][1] >= 10.0 - 0.5 - 1e-9
    assert sampler.requested[0][2] <= 20.0 + 0.5 + 1e-9
    assert sampler.requested[1][1] >= 30.0 - 0.5 - 1e-9
    assert sampler.requested[1][2] <= 35.0 + 0.5 + 1e-9
    assert max(end - start for _, start, end in sampler.requested) < source_duration


def test_scaled_dimensions_matches_ffmpeg_even_rounding() -> None:
    assert scaled_dimensions(320, 240, 640) == (320, 240)
    assert scaled_dimensions(320, 240, 160) == (160, 120)
    assert scaled_dimensions(320, 242, 160) == (160, 122)


def test_segment_scenes_covers_span_with_interior_cuts() -> None:
    assert segment_scenes(0.0, 10.0, [3.0, 7.0]) == ((0.0, 3.0), (3.0, 7.0), (7.0, 10.0))


def test_segment_scenes_ignores_out_of_range_and_duplicate_cuts() -> None:
    assert segment_scenes(0.0, 10.0, [-1.0, 5.0, 11.0, 5.0]) == ((0.0, 5.0), (5.0, 10.0))


def test_segment_scenes_merges_short_scenes() -> None:
    assert segment_scenes(0.0, 10.0, [2.5, 3.0], min_scene_seconds=1.0) == (
        (0.0, 3.0),
        (3.0, 10.0),
    )


def test_merge_cuts_clamps_merges_and_drops() -> None:
    assert merge_cuts([-1.0, 1.0, 1.1, 5.0, 11.0], 0.0, 10.0, 0.5) == (1.0, 5.0)


def test_parse_scene_cut_times_reads_metadata_frame_lines() -> None:
    output = (
        "frame:0    pts:6400   pts_time:0.5\n"
        "lavfi.scene_score=0.400000\n"
        "frame:1    pts:12800   pts_time:1.0\n"
        "lavfi.scene_score=1.000000\n"
    )

    assert parse_scene_cut_times(output) == (0.5, 1.0)


def test_fake_scene_cut_detector_is_clamped_and_merged() -> None:
    detector = _FakeSceneCutDetector([0.9, 1.0, 1.05, 5.0, 99.0])

    assert detector.cuts(Path("unused.mp4"), 1.0, 6.0, 0.35, 0.2) == (1.0, 5.0)


# --- FFmpeg-backed ----------------------------------------------------------


def test_ffmpeg_frame_sampler_pins_sample_time_mapping() -> None:
    ffmpeg = _ffmpeg_or_skip()
    directory = _temporary_dir()
    video = directory / "black-white.mp4"
    _solid_color(ffmpeg, video, ["black", "white"])
    sampler = FFmpegFrameSampler(
        video,
        frame_size=(_VIDEO_WIDTH, _VIDEO_HEIGHT),
        ffmpeg_binary=ffmpeg,
        temporary_root=directory,
    )

    frames = list(sampler.samples([(0, 0.0, 2.0)], 2.0, 640))

    assert [frame.sample_index for frame in frames] == [0, 1, 2, 3]
    assert [frame.source_time for frame in frames] == [0.0, 0.5, 1.0, 1.5]
    assert all(frame.span_block_index == 0 for frame in frames)
    means = [float(frame.rgb_frame.mean()) for frame in frames]
    assert means[0] < 10.0 and means[1] < 10.0
    assert means[2] > 245.0 and means[3] > 245.0
    assert next(index for index, mean in enumerate(means) if mean > 128.0) == 2


def test_ffmpeg_frame_sampler_offsets_nonzero_span_start() -> None:
    ffmpeg = _ffmpeg_or_skip()
    directory = _temporary_dir()
    video = directory / "black-white-offset.mp4"
    _solid_color(ffmpeg, video, ["black", "white"])
    sampler = FFmpegFrameSampler(
        video,
        frame_size=(_VIDEO_WIDTH, _VIDEO_HEIGHT),
        ffmpeg_binary=ffmpeg,
        temporary_root=directory,
    )

    frames = list(sampler.samples([(3, 0.25, 1.25)], 2.0, 640))

    assert [frame.source_time for frame in frames] == [0.25, 0.75]
    assert all(frame.span_block_index == 3 for frame in frames)


def test_ffmpeg_frame_sampler_stops_on_cancellation() -> None:
    ffmpeg = _ffmpeg_or_skip()
    directory = _temporary_dir()
    video = directory / "black-white-cancel.mp4"
    _solid_color(ffmpeg, video, ["black", "white"])
    sampler = FFmpegFrameSampler(
        video,
        frame_size=(_VIDEO_WIDTH, _VIDEO_HEIGHT),
        ffmpeg_binary=ffmpeg,
        temporary_root=directory,
    )

    with pytest.raises(AnalysisCancelled):
        list(sampler.samples([(0, 0.0, 2.0)], 1.0, 640, cancel_check=lambda: True))


def test_ffmpeg_scene_cut_detector_finds_single_cut() -> None:
    ffmpeg = _ffmpeg_or_skip()
    directory = _temporary_dir()
    video = directory / "red-blue.mp4"
    _solid_color(ffmpeg, video, ["red", "blue"])
    detector = FFmpegSceneCutDetector(ffmpeg_binary=ffmpeg)

    cuts = detector.cuts(video, 0.0, 2.0, 0.35, 0.4)

    assert len(cuts) == 1
    assert abs(cuts[0] - 1.0) < 0.1


def test_ffmpeg_scene_cut_detector_reports_source_time_for_offset_span() -> None:
    ffmpeg = _ffmpeg_or_skip()
    directory = _temporary_dir()
    video = directory / "red-blue-offset.mp4"
    _solid_color(ffmpeg, video, ["red", "blue"])
    detector = FFmpegSceneCutDetector(ffmpeg_binary=ffmpeg)

    cuts = detector.cuts(video, 0.5, 1.5, 0.35, 0.4)

    assert len(cuts) == 1
    assert abs(cuts[0] - 1.0) < 0.1


# --- Availability -----------------------------------------------------------


def test_scene_cut_detector_reports_missing_ffmpeg() -> None:
    detector = FFmpegSceneCutDetector(ffmpeg_binary=_MISSING_FFMPEG)

    with pytest.raises(FFmpegUnavailableError):
        detector.cuts(Path("unused.mp4"), 0.0, 1.0, 0.35, 0.4)


def test_frame_sampler_reports_missing_ffmpeg() -> None:
    sampler = FFmpegFrameSampler(
        Path("unused.mp4"),
        frame_size=(16, 16),
        ffmpeg_binary=_MISSING_FFMPEG,
    )

    with pytest.raises(FFmpegUnavailableError):
        list(sampler.samples([(0, 0.0, 1.0)], 1.0, 640))
