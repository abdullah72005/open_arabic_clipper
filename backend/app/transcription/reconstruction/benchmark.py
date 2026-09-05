"""Private unseen-audio reconstruction benchmark acceptance gate."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class BenchmarkClip(BaseModel):
    """One private authorized interval from a source recording."""

    id: str = Field(min_length=1)
    source_recording_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    authorized: bool
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    categories: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def _validate_interval(self) -> "BenchmarkClip":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("clip end must follow its start")
        return self


class BenchmarkManifest(BaseModel):
    """Private test-split topology; never serialize its transcript rows to output."""

    split: str = "test"
    clips: list[BenchmarkClip]
    tuning_splits: set[str] = Field(default_factory=set)
    report: "BenchmarkReport"

    @model_validator(mode="after")
    def _validate_test_split(self) -> "BenchmarkManifest":
        if self.split != "test":
            raise ValueError("benchmark manifest must be an unseen test split")
        if self.split in self.tuning_splits:
            raise ValueError("test split must not be used in tuning metadata")
        if len(self.clips) < 5:
            raise ValueError("benchmark requires at least five clips")
        if not all(clip.authorized for clip in self.clips):
            raise ValueError("every benchmark clip requires operator authorization")
        if len({clip.topic for clip in self.clips}) < 3:
            raise ValueError("benchmark requires at least three topics")
        if len({clip.source_recording_id for clip in self.clips}) < 2:
            raise ValueError("benchmark requires at least two source recordings")
        total_seconds = sum(clip.end_seconds - clip.start_seconds for clip in self.clips)
        if not 120 <= total_seconds <= 300:
            raise ValueError("benchmark requires 2 to 5 minutes of evaluated speech")
        required = {"slang", "fast_speech", "code_switching", "entities", "narrative"}
        present = set().union(*(clip.categories for clip in self.clips))
        if missing := required - present:
            raise ValueError(f"benchmark missing required categories: {sorted(missing)}")
        for source_id in {clip.source_recording_id for clip in self.clips}:
            intervals = sorted(
                (clip.start_seconds, clip.end_seconds)
                for clip in self.clips
                if clip.source_recording_id == source_id
            )
            if any(
                left[1] > right[0] for left, right in zip(intervals, intervals[1:], strict=False)
            ):
                raise ValueError("benchmark clip intervals cannot overlap")
        return self


class BenchmarkReport(BaseModel):
    """Privacy-safe aggregate metrics from a frozen, human-reviewed test split."""

    model_identifier: str = Field(min_length=1)
    model_digest: str = Field(min_length=1)
    semantic_correct_stage25: float = Field(ge=0, le=1)
    semantic_correct_stage27: float = Field(ge=0, le=1)
    stage25_wrong: int = Field(ge=0)
    improved: int = Field(ge=0)
    stage25_correct: int = Field(ge=0)
    regressed: int = Field(ge=0)
    preserved: int = Field(ge=0)
    hallucinated: int = Field(ge=0)
    comprehensibility_stage25: dict[str, float]
    comprehensibility_stage27: dict[str, float]
    source_audio_seconds: float = Field(gt=0)
    wall_clock_seconds: float = Field(gt=0)
    peak_ram_bytes: int = Field(ge=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_denominators(self) -> "BenchmarkReport":
        if self.improved > self.stage25_wrong:
            raise ValueError("improved cannot exceed Stage 2.5 wrong segments")
        if self.regressed + self.preserved > self.stage25_correct:
            raise ValueError("regressed and preserved exceed Stage 2.5 correct segments")
        return self

    @property
    def throughput_audio_minutes_per_wall_minute(self) -> float:
        return self.source_audio_seconds / self.wall_clock_seconds


def evaluate_completion_gate(report: BenchmarkReport) -> tuple[bool, list[str]]:
    """Return every failed immutable Stage 2.7 acceptance criterion."""

    reasons: list[str] = []
    if report.semantic_correct_stage27 - report.semantic_correct_stage25 < 0.10:
        reasons.append("semantic-correct rate must improve by at least 10 percentage points")
    if report.stage25_wrong == 0 or report.improved / report.stage25_wrong < 0.25:
        reasons.append("at least 25% of Stage 2.5-wrong segments must improve")
    if report.stage25_correct == 0 or report.regressed / report.stage25_correct > 0.02:
        reasons.append("regressions must be at most 2% of Stage 2.5-correct segments")
    if report.hallucinated:
        reasons.append("hallucinated facts/names/numbers/clauses must be zero")
    if report.stage25_correct == 0 or report.preserved / report.stage25_correct < 0.98:
        reasons.append("at least 98% of Stage 2.5-correct segments must remain correct")
    categories = set(report.comprehensibility_stage25) | set(report.comprehensibility_stage27)
    if not categories or any(
        report.comprehensibility_stage27.get(category, 0)
        < report.comprehensibility_stage25.get(category, 0)
        for category in categories
    ):
        reasons.append("comprehensibility must not fall in any required category")
    if sum(report.comprehensibility_stage27.values()) / max(
        len(report.comprehensibility_stage27), 1
    ) <= sum(report.comprehensibility_stage25.values()) / max(
        len(report.comprehensibility_stage25), 1
    ):
        reasons.append("mean comprehensibility must improve")
    return not reasons, reasons


def load_benchmark_manifest(path: Path) -> BenchmarkManifest:
    """Read a private JSON manifest from a path already owned by StorageService."""

    try:
        return BenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid private reconstruction benchmark manifest") from error


def run_reconstruction_benchmark(manifest: BenchmarkManifest) -> BenchmarkReport:
    """Return frozen human-reviewed metrics without printing private transcript content."""

    return manifest.report
