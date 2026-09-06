"""Deterministic, storage-owned private reconstruction benchmark runner."""

from __future__ import annotations

import json
import resource
import subprocess
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel, Field, model_validator

from app.core.enums import ReconstructionStatus
from app.pipeline.fingerprints import canonical_fingerprint
from app.services.storage import StorageCategory, StorageService, StorageValidationError
from app.transcription.correction import SegmentCorrection
from app.transcription.engine import TranscriptionResult
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
    ReconstructionResult,
)
from app.transcription.service import TranscriptionOptions


class Transcriber(Protocol):
    """Minimal ASR boundary used by the benchmark runner."""

    def transcribe(self, path: Path, options: TranscriptionOptions) -> TranscriptionResult: ...


class Corrector(Protocol):
    """Minimal Stage 2.5 correction boundary used by the benchmark runner."""

    def correct(self, segments: list[dict[str, object]]) -> Sequence[SegmentCorrection]: ...


class Reconstructor(Protocol):
    """Minimal Stage 2.7 reconstruction boundary used by the benchmark runner."""

    def reconstruct(
        self,
        segments: list[dict[str, object]],
        *,
        language: str | None,
        transcription_fingerprint: str,
        correction_version: str,
    ) -> ReconstructionResult: ...


_MANIFEST_VERSION = "stage-2-7-private-v1"
_REQUIRED_CATEGORIES = {"slang", "fast_speech", "code_switching", "entities", "narrative"}
_SWAP_INFEASIBLE_BYTES = 1024**3


class BenchmarkSource(BaseModel):
    """One authorized, storage-owned recording used by the benchmark."""

    id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    authorized: bool = False

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _validate(self) -> "BenchmarkSource":
        candidate = Path(self.path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("benchmark source path must be storage-relative")
        if not self.authorized:
            raise ValueError("every benchmark source requires operator authorization")
        return self


class ReferenceSegment(BaseModel):
    """One human-reviewed reference row for a transcribed segment."""

    segment_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    reviewed: bool = False
    human_label: str | None = None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _validate(self) -> "ReferenceSegment":
        if not self.reviewed:
            raise ValueError("every benchmark reference must be reviewed")
        return self


class BenchmarkClip(BaseModel):
    """One private authorized interval with its reviewed reference segments."""

    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    categories: set[str] = Field(default_factory=set)
    reference_segments: list[ReferenceSegment] = Field(default_factory=list)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _validate(self) -> "BenchmarkClip":
        if self.id in {".", ".."} or "/" in self.id or "\\" in self.id:
            raise ValueError("benchmark clip id must be a safe filename component")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("clip end must follow its start")
        indexes = [segment.segment_index for segment in self.reference_segments]
        if indexes != list(range(len(indexes))):
            raise ValueError("reference segment indexes must be exact and contiguous")
        return self


class BenchmarkManifest(BaseModel):
    """Private test-split topology; never serializes transcript rows to output."""

    version: str = _MANIFEST_VERSION
    split: str = "test"
    sources: list[BenchmarkSource]
    clips: list[BenchmarkClip]
    known_regression_set: bool = False

    @property
    def readiness_eligible(self) -> bool:
        """A known regression diagnostic can never satisfy the unseen gate."""

        return not self.known_regression_set

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _validate(self) -> "BenchmarkManifest":
        if self.version != _MANIFEST_VERSION:
            raise ValueError("unsupported benchmark manifest version")
        if self.split != "test":
            raise ValueError("benchmark manifest must be an unseen test split")
        source_ids = {source.id for source in self.sources}
        if any(clip.source_id not in source_ids for clip in self.clips):
            raise ValueError("every benchmark clip must reference a declared source")
        if not self.clips:
            raise ValueError("benchmark requires at least one clip")
        if self.known_regression_set:
            return self
        if len(self.clips) < 5:
            raise ValueError("benchmark requires at least five clips")
        if len({clip.topic for clip in self.clips}) < 3:
            raise ValueError("benchmark requires at least three topics")
        if len({clip.source_id for clip in self.clips}) < 2:
            raise ValueError("benchmark requires at least two source recordings")
        total_seconds = sum(clip.end_seconds - clip.start_seconds for clip in self.clips)
        if not 120 <= total_seconds <= 300:
            raise ValueError("benchmark requires 2 to 5 minutes of evaluated speech")
        present = set().union(*(clip.categories for clip in self.clips))
        if missing := _REQUIRED_CATEGORIES - present:
            raise ValueError(f"benchmark missing required categories: {sorted(missing)}")
        for source_id in source_ids:
            intervals = sorted(
                (clip.start_seconds, clip.end_seconds)
                for clip in self.clips
                if clip.source_id == source_id
            )
            if any(
                left[1] > right[0] for left, right in zip(intervals, intervals[1:], strict=False)
            ):
                raise ValueError("benchmark clip intervals cannot overlap")
        return self


class BenchmarkReport(BaseModel):
    """Privacy-safe aggregate metrics from a frozen, human-reviewed test split."""

    model_identifier: str = Field(min_length=1)
    model_digest: str = Field(default="")
    prompt_settings_fingerprint: str = Field(min_length=1)
    provider_available: bool = False
    human_labels_complete: bool = False
    readiness_eligible: bool = True
    model_feasible: bool = True
    semantic_correct_stage25: float = Field(ge=0, le=1)
    semantic_correct_stage27: float = Field(ge=0, le=1)
    stage25_wrong: int = Field(ge=0)
    improved: int = Field(ge=0)
    unchanged_correct: int = Field(ge=0)
    unchanged_wrong: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    stage25_correct: int = Field(ge=0)
    regressed: int = Field(ge=0)
    preserved: int = Field(ge=0)
    hallucinated: int = Field(ge=0)
    comprehensibility_stage25: dict[str, float] = Field(default_factory=dict)
    comprehensibility_stage27: dict[str, float] = Field(default_factory=dict)
    source_audio_seconds: float = Field(gt=0)
    wall_clock_seconds: float = Field(gt=0)
    peak_ram_bytes: int = Field(ge=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    swap_before_bytes: int = Field(default=0, ge=0)
    swap_after_bytes: int = Field(default=0, ge=0)
    comparison_path: Path | None = None
    report_path: Path | None = None
    worksheet_path: Path | None = None

    @property
    def throughput_audio_minutes_per_wall_minute(self) -> float:
        return self.source_audio_seconds / self.wall_clock_seconds


class BenchmarkRunner:
    """Execute raw ASR, Stage 2.5, then Stage 2.7 for one private manifest."""

    def __init__(
        self,
        storage: StorageService,
        *,
        whisper_engine: Transcriber,
        corrector: Corrector,
        reconstructor: Reconstructor,
        transcription_options: TranscriptionOptions,
        provider_health: ProviderHealth | None = None,
        ffmpeg_binary: str = "ffmpeg",
        command_runner: Callable[[list[str]], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        peak_rss_bytes: Callable[[], int] = lambda: (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        swap_bytes: Callable[[], int] | None = None,
        peak_vram_bytes: Callable[[], int | None] = lambda: None,
        prompt_settings_fingerprint: str = "",
    ) -> None:
        self._storage = storage
        self._whisper_engine = whisper_engine
        self._corrector = corrector
        self._reconstructor = reconstructor
        self._transcription_options = transcription_options
        self._provider_health = provider_health
        self._ffmpeg_binary = ffmpeg_binary
        self._command_runner = command_runner or _run_command
        self._monotonic = monotonic
        self._peak_rss_bytes = peak_rss_bytes
        self._swap_bytes = swap_bytes or _swap_bytes
        self._peak_vram_bytes = peak_vram_bytes
        self._prompt_settings_fingerprint = prompt_settings_fingerprint

    def run(self, manifest: BenchmarkManifest) -> BenchmarkReport:
        """Run the private benchmark and write deterministic artifacts via storage."""

        started = self._monotonic()
        swap_before = self._swap_bytes()
        run_directory = self._new_run_directory()
        clip_directory = run_directory / "clips"
        clip_directory.mkdir(parents=True, exist_ok=True)
        comparison_path = run_directory / "comparison.jsonl"
        worksheet_path = run_directory / "review-worksheet.jsonl"
        report_path = run_directory / "report.json"

        health = self._provider_health
        provider_available = (
            health is not None and health.availability is ProviderAvailability.AVAILABLE
        )
        model_identifier = health.model if health is not None and health.model else "unavailable"
        model_digest = health.model_digest or "" if health is not None else ""

        source_map = {source.id: source for source in manifest.sources}
        rows: list[dict[str, object]] = []
        counts: Counter[str] = Counter()
        provider_died = False

        for clip in manifest.clips:
            source_path = self._storage.resolve(
                StorageCategory.SOURCES, source_map[clip.source_id].path
            )
            if not source_path.is_file():
                raise StorageValidationError(
                    f"benchmark source media is missing from storage: {clip.source_id}"
                )
            audio_path = clip_directory / f"{clip.id}.wav"
            self._extract_clip(source_path, clip.start_seconds, clip.end_seconds, audio_path)
            raw_result = self._whisper_engine.transcribe(audio_path, self._transcription_options)
            raw_segments = [dict(segment) for segment in raw_result.segments]
            identity = [
                (index, segment.get("start"), segment.get("end"))
                for index, segment in enumerate(raw_segments)
            ]
            corrections = self._corrector.correct(raw_segments)
            stage25 = [
                dict(
                    segment, raw_text=correction.raw_text, corrected_text=correction.corrected_text
                )
                for segment, correction in zip(raw_segments, corrections, strict=True)
            ]
            if identity != [
                (index, segment.get("start"), segment.get("end"))
                for index, segment in enumerate(stage25)
            ]:
                raise ValueError("Stage 2.5 changed raw segment IDs or timestamps")
            reconstruction = self._reconstructor.reconstruct(
                stage25,
                language=raw_result.language,
                transcription_fingerprint="benchmark",
                correction_version="benchmark",
            )
            results = {item.segment_index: item for item in reconstruction.segments}
            references = {segment.segment_index: segment for segment in clip.reference_segments}
            for index, segment in enumerate(stage25):
                item = results.get(index)
                corrected = str(segment["corrected_text"])
                final = str(item.contextual_reconstructed_text) if item is not None else corrected
                reference_segment = references.get(index)
                reference = reference_segment.text if reference_segment is not None else ""
                human_label = (
                    reference_segment.human_label or "" if reference_segment is not None else ""
                )
                status = comparison_status(
                    item.status if item is not None else None, corrected, final, reference
                )
                if item is not None and item.status is ReconstructionStatus.PROVIDER_UNAVAILABLE:
                    provider_died = True
                counts[human_label or status] += 1
                rows.append(
                    {
                        "clip_id": clip.id,
                        "segment_index": index,
                        "raw": str(segment.get("raw_text", segment.get("text", ""))),
                        "stage25": corrected,
                        "stage27": final,
                        "reference": reference,
                        "status": status,
                        "confidence": item.confidence if item is not None else None,
                        "wer": _word_error_rate(reference, final),
                        "cer": _character_error_rate(reference, final),
                        "model": model_identifier,
                        "model_digest": model_digest,
                        "human_label": human_label,
                    }
                )

        swap_after = self._swap_bytes()
        model_feasible = (swap_after - swap_before) <= _SWAP_INFEASIBLE_BYTES and not provider_died
        human_labels_complete = bool(rows) and all(bool(row["human_label"]) for row in rows)

        improved = counts["improved"]
        unchanged_correct = counts["unchanged_correct"]
        unchanged_wrong = counts["unchanged_wrong"]
        regressed = counts["regressed"]
        hallucinated = counts["hallucinated"]
        unresolved = counts["unresolved"]
        stage25_correct = unchanged_correct + regressed
        stage25_wrong = improved + unchanged_wrong + hallucinated
        reviewed_total = stage25_correct + stage25_wrong

        report = BenchmarkReport(
            model_identifier=model_identifier,
            model_digest=model_digest,
            prompt_settings_fingerprint=self._prompt_settings_fingerprint,
            provider_available=provider_available and not provider_died,
            human_labels_complete=human_labels_complete,
            readiness_eligible=manifest.readiness_eligible,
            model_feasible=model_feasible,
            semantic_correct_stage25=stage25_correct / reviewed_total if reviewed_total else 0.0,
            semantic_correct_stage27=(unchanged_correct + improved) / reviewed_total
            if reviewed_total
            else 0.0,
            stage25_wrong=stage25_wrong,
            improved=improved,
            unchanged_correct=unchanged_correct,
            unchanged_wrong=unchanged_wrong,
            unresolved=unresolved,
            stage25_correct=stage25_correct,
            regressed=regressed,
            preserved=unchanged_correct,
            hallucinated=hallucinated,
            source_audio_seconds=sum(
                clip.end_seconds - clip.start_seconds for clip in manifest.clips
            ),
            wall_clock_seconds=max(self._monotonic() - started, 1e-9),
            peak_ram_bytes=self._peak_rss_bytes(),
            peak_vram_bytes=self._peak_vram_bytes(),
            swap_before_bytes=swap_before,
            swap_after_bytes=swap_after,
            comparison_path=comparison_path,
            report_path=report_path,
            worksheet_path=worksheet_path,
        )

        self._storage.atomic_write(
            comparison_path,
            [(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rows],
        )
        self._storage.atomic_write(
            worksheet_path,
            [
                (
                    json.dumps(
                        {
                            "clip_id": row["clip_id"],
                            "segment_index": row["segment_index"],
                            "status": row["status"],
                            "human_label": row["human_label"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                for row in rows
            ],
        )
        self._storage.atomic_write(report_path, [report.model_dump_json(indent=2).encode()])
        return report

    def _new_run_directory(self) -> Path:
        run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        directory = self._storage.resolve(StorageCategory.BENCHMARKS, f"stage-2-7/results/{run_id}")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _extract_clip(self, source: Path, start: float, end: float, destination: Path) -> None:
        self._command_runner(
            [
                self._ffmpeg_binary,
                "-y",
                "-ss",
                str(start),
                "-i",
                str(source),
                "-t",
                str(end - start),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(destination),
            ]
        )


def classify_comparison(stage25: str, stage27: str, reference: str) -> str:
    """Classify one segment by deterministic string comparison to its reference."""

    stage25_correct = _normalize(stage25) == _normalize(reference)
    stage27_correct = _normalize(stage27) == _normalize(reference)
    if stage25_correct and stage27_correct:
        return "unchanged_correct"
    if stage25_correct and not stage27_correct:
        return "regressed"
    if not stage25_correct and stage27_correct:
        return "improved"
    if _normalize(stage27) == _normalize(stage25):
        return "unchanged_wrong"
    return "hallucinated"


def comparison_status(
    reconstruction_status: object | None, stage25: str, stage27: str, reference: str
) -> str:
    """Return the six-way comparison outcome, honoring unresolved state."""

    if not reference.strip():
        return "unresolved"
    if reconstruction_status is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED:
        return "unresolved"
    return classify_comparison(stage25, stage27, reference)


def evaluate_completion_gate(
    report: BenchmarkReport, *, expected_prompt_settings_fingerprint: str | None = None
) -> tuple[bool, list[str]]:
    """Return every failed immutable Stage 2.7 acceptance criterion."""

    reasons: list[str] = []
    if not report.readiness_eligible:
        reasons.append("known regression sets cannot pass the unseen readiness gate")
    if not report.provider_available:
        reasons.append("reconstruction provider must be available")
    if not report.model_digest:
        reasons.append("provider model digest is required")
    if not report.human_labels_complete:
        reasons.append("every comparison row requires a human label")
    if not report.prompt_settings_fingerprint or (
        expected_prompt_settings_fingerprint is not None
        and report.prompt_settings_fingerprint != expected_prompt_settings_fingerprint
    ):
        reasons.append("benchmark must use the exact production prompt/settings fingerprint")
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


def load_benchmark_manifest(
    path: Path,
    *,
    allow_known_regression_set: bool = False,
    known_regression_manifest_path: Path | None = None,
) -> BenchmarkManifest:
    """Read a private JSON manifest from a path already owned by StorageService."""

    if allow_known_regression_set and path != known_regression_manifest_path:
        raise ValueError(
            "known regression override is reserved for the Chernobyl diagnostic manifest"
        )
    try:
        payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid private reconstruction benchmark manifest") from error
    payload["known_regression_set"] = allow_known_regression_set
    try:
        return cast(BenchmarkManifest, BenchmarkManifest.model_validate(payload))
    except ValueError as error:
        raise ValueError("invalid private reconstruction benchmark manifest") from error


def prompt_settings_fingerprint(
    *,
    provider: str,
    model: str | None,
    digest: str | None,
    whisper_options: dict[str, object],
) -> str:
    """Fingerprint the exact production prompt and settings used for a run."""

    return canonical_fingerprint(
        "reconstruction-benchmark-settings",
        "1",
        {
            "provider": provider,
            "model": model,
            "digest": digest,
            "prompt": "stage-2-7-v1",
            "whisper": whisper_options,
        },
    )


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _run_command(args: list[str]) -> None:
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _swap_bytes() -> int:
    try:
        content = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return 0
    total = 0
    free = 0
    for line in content.splitlines():
        if line.startswith("SwapTotal:"):
            total = int(line.split()[1]) * 1024
        elif line.startswith("SwapFree:"):
            free = int(line.split()[1]) * 1024
    return max(0, total - free)


def _word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _levenshtein(reference_words, hypothesis_words) / len(reference_words)


def _character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(list(reference), list(hypothesis)) / len(reference)


def _levenshtein(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, reference_item in enumerate(reference, start=1):
        current = [index]
        for position, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[position] + 1,
                    previous[position - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]
