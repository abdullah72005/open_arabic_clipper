"""Private unseen-audio reconstruction benchmark acceptance gate."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


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
