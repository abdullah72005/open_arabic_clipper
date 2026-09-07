"""Deterministic tests for the private reconstruction benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from app.core.enums import ReconstructionStatus
from app.services.storage import StorageCategory, StorageService, StorageValidationError
from app.transcription.correction import SegmentCorrection
from app.transcription.engine import TranscriptionResult
from app.transcription.reconstruction.benchmark import (
    BenchmarkManifest,
    BenchmarkReport,
    BenchmarkRunner,
    classify_comparison,
    comparison_status,
    evaluate_completion_gate,
    exact_classify_comparison,
    load_benchmark_manifest,
    load_review_worksheet,
)
from app.transcription.reconstruction.types import (
    ConfidenceLevel,
    ProviderAvailability,
    ProviderHealth,
    ReconstructionResult,
    SegmentReconstruction,
)
from app.transcription.service import TranscriptionOptions

STATUSES = [
    "improved",
    "unchanged_correct",
    "unchanged_wrong",
    "regressed",
    "hallucinated",
    "changed_wrong",
]


def _options() -> TranscriptionOptions:
    return TranscriptionOptions(
        model="large-v3-turbo", device="cpu", compute_type="int8", beam_size=5
    )


def _source(source_id: str) -> dict[str, object]:
    return {"id": source_id, "path": f"{source_id}/video.webm", "authorized": True}


def _reference(
    segment_index: int, label: str | None = None, text: str | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "segment_index": segment_index,
        "text": text or f"reference-{segment_index}",
        "reviewed": True,
    }
    if label is not None:
        payload["human_label"] = label
    return payload


def _manifest(**changes: object) -> BenchmarkManifest:
    clips = []
    categories = ["slang", "fast_speech", "code_switching", "entities", "narrative"]
    for index, category in enumerate(categories):
        labels = STATUSES if index == 0 else ["unchanged_correct"]
        clips.append(
            {
                "id": f"clip-{index}",
                "source_id": "source-a" if index < 3 else "source-b",
                "topic": ["history", "technology", "sport"][index % 3],
                "start_seconds": index * 30 if index < 3 else (index - 3) * 30,
                "end_seconds": index * 30 + 30 if index < 3 else (index - 3) * 30 + 30,
                "categories": [category],
                "reference_segments": [
                    _reference(segment_index, label) for segment_index, label in enumerate(labels)
                ],
            }
        )
    payload: dict[str, object] = {
        "version": "stage-2-7-private-v1",
        "split": "test",
        "sources": [_source("source-a"), _source("source-b")],
        "clips": clips,
    }
    payload.update(changes)
    return cast(BenchmarkManifest, BenchmarkManifest.model_validate(payload))


def _report(**changes: object) -> BenchmarkReport:
    values: dict[str, object] = {
        "model_identifier": "qwen3:8b",
        "model_digest": "sha256:test",
        "prompt_settings_fingerprint": "a" * 64,
        "provider_available": True,
        "human_labels_complete": True,
        "readiness_eligible": True,
        "semantic_correct_stage25": 0.70,
        "semantic_correct_stage27": 0.82,
        "stage25_wrong": 20,
        "improved": 5,
        "unchanged_correct": 79,
        "unchanged_wrong": 15,
        "stage25_correct": 80,
        "regressed": 1,
        "preserved": 79,
        "hallucinated": 0,
        "changed_wrong": 0,
        "unresolved": 0,
        "comprehensibility_stage25": {"slang": 3.0, "fast_speech": 3.0},
        "comprehensibility_stage27": {"slang": 3.5, "fast_speech": 3.0},
        "source_audio_seconds": 240.0,
        "wall_clock_seconds": 120.0,
        "peak_ram_bytes": 1,
        "peak_vram_bytes": None,
    }
    values.update(changes)
    return cast(BenchmarkReport, BenchmarkReport.model_validate(values))


def test_classify_comparison_assigns_all_five_string_statuses() -> None:
    assert classify_comparison("wrong", "right", "right") == "improved"
    assert classify_comparison("right", "right", "right") == "unchanged_correct"
    assert classify_comparison("wrong", "wrong", "right") == "unchanged_wrong"
    assert classify_comparison("right", "worse", "right") == "regressed"
    assert classify_comparison("wrong", "invented 999", "right") == "changed_wrong"


def test_changed_but_wrong_output_is_never_automatically_hallucinated() -> None:
    """Hallucination is a human safety label; automated comparison emits changed_wrong."""

    assert classify_comparison("wrong", "other", "right") == "changed_wrong"
    assert exact_classify_comparison("wrong", "other", "right") == "changed_wrong"


def test_classify_comparison_accepts_reviewed_dialect_orthography_pairs() -> None:
    stage25 = " ثلاثة يام بس لتطهير النطقة"
    candidate = "تلاتة يام بس لتطهير المنطقة"
    reference = "تلات أيام بس لتطهير المنطقة"
    assert classify_comparison(stage25, candidate, reference) == "improved"
    assert classify_comparison(stage25, stage25, reference) == "unchanged_wrong"
    assert exact_classify_comparison(stage25, candidate, reference) == "changed_wrong"


def test_classify_comparison_keeps_real_meaning_changes_as_errors() -> None:
    stage25 = "بعد كارثة شرنوبل، الحكومة هتمر السكان بعمل عملية إخلاق مؤقت"
    reference = "وقت كارثة شرنوبل الحكومة هتمر السكان بعمل عملية إخلاء مؤقت"
    assert classify_comparison(stage25, stage25, reference) == "unchanged_wrong"
    changed = stage25.replace("بعد", "خلاف")
    assert classify_comparison(stage25, changed, reference) == "changed_wrong"


def test_exact_comparison_rejects_spelling_layout_and_diacritic_variants() -> None:
    """Exact comparison is literal NFC equality: no alef, ة/ه, ى/ي, or diacritic folding."""

    assert exact_classify_comparison("آخر", "اخر", "آخر") == "regressed"
    assert exact_classify_comparison("تلاته", "تلاتة", "تلاته") == "regressed"
    assert exact_classify_comparison("حتى", "حتي", "حتى") == "regressed"
    assert exact_classify_comparison("جيد،", "جيد", "جيد،") == "regressed"
    assert exact_classify_comparison("مهم", "مُهم", "مهم") == "regressed"


def test_exact_comparison_permits_nfc_and_documented_whitespace_collapse() -> None:
    """Only NFC normalization and outer/duplicate-whitespace collapse are permitted."""

    assert exact_classify_comparison("كلام   كتير", "كلام كتير", "كلام كتير") == "unchanged_correct"


def test_deterministic_equivalence_rejects_unreviewed_word_variants() -> None:
    """Possessives, dental-shift names, verb changes, numbers, and Latin tokens stay distinct."""

    assert classify_comparison("كتاب", "كتاب", "كتابي") == "unchanged_wrong"
    assert classify_comparison("حسن", "حسن", "حسني") == "unchanged_wrong"
    assert classify_comparison("تامر", "تامر", "ثامر") == "unchanged_wrong"
    assert classify_comparison("عمل", "يعمل", "عمل") == "regressed"
    assert classify_comparison("سنة 1950", "سنة 1951", "سنة 1950") == "regressed"
    assert classify_comparison("United Fruit", "United Fruits", "United Fruit") == "regressed"


def test_comparison_status_separates_runtime_status_from_text_comparison() -> None:
    """A referenced row is compared even when reconstruction runtime status is unresolved."""

    assert comparison_status(ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED, "a", "b", "c") == (
        "changed_wrong",
        "changed_wrong",
    )
    assert comparison_status(ReconstructionStatus.APPLIED, "a", "b", "") == (
        "unreviewed",
        "unreviewed",
    )
    assert comparison_status(ReconstructionStatus.APPLIED, "right", "right", "right") == (
        "unchanged_correct",
        "unchanged_correct",
    )


def test_completion_gate_accepts_measured_quality_and_preservation_thresholds() -> None:
    passed, reasons = evaluate_completion_gate(_report())
    assert passed is True
    assert reasons == []


def test_completion_gate_rejects_missing_live_evidence_and_any_hallucination() -> None:
    passed, reasons = evaluate_completion_gate(
        _report(
            hallucinated=1,
            provider_available=False,
            model_digest="",
            human_labels_complete=False,
            readiness_eligible=False,
            prompt_settings_fingerprint="wrong",
        ),
        expected_prompt_settings_fingerprint="expected",
    )
    assert passed is False
    assert "hallucinated facts/names/numbers/clauses must be zero" in reasons
    assert "reconstruction provider must be available" in reasons
    assert "provider model digest is required" in reasons
    assert "every comparison row requires a human label" in reasons
    assert "known regression sets cannot pass the unseen readiness gate" in reasons
    assert "benchmark must use the exact production prompt/settings fingerprint" in reasons


def test_manifest_refuses_unreviewed_unauthorized_and_out_of_storage_inputs() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["sources"][0]["authorized"] = False
    with pytest.raises(ValueError, match="authorization"):
        BenchmarkManifest.model_validate(payload)

    payload = _manifest().model_dump(mode="json")
    payload["clips"][0]["reference_segments"][0]["reviewed"] = False
    with pytest.raises(ValueError, match="reviewed"):
        BenchmarkManifest.model_validate(payload)

    payload = _manifest().model_dump(mode="json")
    payload["sources"][0]["path"] = "../outside.webm"
    with pytest.raises(ValueError, match="storage-relative"):
        BenchmarkManifest.model_validate(payload)

    payload = _manifest().model_dump(mode="json")
    payload["sources"][0]["path"] = "/absolute/outside.webm"
    with pytest.raises(ValueError, match="storage-relative"):
        BenchmarkManifest.model_validate(payload)

    for unsafe_clip_id in ("/tmp/escape", "../../escape"):
        payload = _manifest().model_dump(mode="json")
        payload["clips"][0]["id"] = unsafe_clip_id
        with pytest.raises(ValueError, match="safe filename"):
            BenchmarkManifest.model_validate(payload)


def test_manifest_rejects_unknown_human_label() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["clips"][0]["reference_segments"][0]["human_label"] = "unknown-label"
    with pytest.raises(ValueError, match="human label"):
        BenchmarkManifest.model_validate(payload)


def test_manifest_rejects_unresolved_as_a_human_label() -> None:
    """A referenced row cannot be removed from correctness via a human label."""

    payload = _manifest().model_dump(mode="json")
    payload["clips"][0]["reference_segments"][0]["human_label"] = "unresolved"
    with pytest.raises(ValueError, match="human label"):
        BenchmarkManifest.model_validate(payload)


def test_load_review_worksheet_rejects_unknown_human_label(tmp_path: Path) -> None:
    worksheet = tmp_path / "review-worksheet.jsonl"
    worksheet.write_text(
        json.dumps({"clip_id": "clip-0", "segment_index": 0, "human_label": "wrong-label"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="human label"):
        load_review_worksheet(worksheet)


def test_load_review_worksheet_rejects_unresolved_human_label(tmp_path: Path) -> None:
    worksheet = tmp_path / "review-worksheet.jsonl"
    worksheet.write_text(
        json.dumps({"clip_id": "clip-0", "segment_index": 0, "human_label": "unresolved"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="human label"):
        load_review_worksheet(worksheet)


def test_manifest_loader_refuses_non_chernobyl_diagnostic_override(tmp_path: Path) -> None:
    payload = _manifest().model_dump(mode="json")
    payload["sources"] = payload["sources"][:1]
    payload["clips"] = payload["clips"][:1]
    manifest_path = tmp_path / "arbitrary-diagnostic.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Chernobyl diagnostic"):
        load_benchmark_manifest(manifest_path, allow_known_regression_set=True)


def test_manifest_requires_exact_reference_indexes_and_readiness_topology() -> None:
    payload = _manifest().model_dump(mode="json")
    payload["clips"][0]["reference_segments"][1]["segment_index"] = 20
    with pytest.raises(ValueError, match="contiguous"):
        BenchmarkManifest.model_validate(payload)

    payload = _manifest().model_dump(mode="json")
    payload["clips"] = payload["clips"][:1]
    with pytest.raises(ValueError, match="five clips"):
        BenchmarkManifest.model_validate(payload)


class _RecordingStorage(StorageService):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.writes: list[Path] = []

    def atomic_write(self, destination: Path, chunks: object) -> Path:
        self.writes.append(destination)
        return super().atomic_write(destination, chunks)  # type: ignore[arg-type]


class _Engine:
    def __init__(self, events: list[str], counts: dict[int, int] | None = None) -> None:
        self.events = events
        self.counts = counts or {}

    def transcribe(self, path: Path, options: TranscriptionOptions) -> TranscriptionResult:
        clip_id = Path(path).stem
        self.events.append(f"asr:{clip_id}:{options.model}")
        index = int(clip_id.split("-")[1])
        count = self.counts.get(index, 1 if index else 6)
        segments = [
            {
                "start": float(position),
                "end": float(position + 1),
                "text": f"raw-{index}-{position}",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.0,
                "words": [],
            }
            for position in range(count)
        ]
        return TranscriptionResult("ar", 0.99, "", float(count), segments, [])


class _Corrector:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def correct(self, segments: list[dict[str, object]]) -> list[SegmentCorrection]:
        self.events.append("stage25")
        clip_index = int(str(segments[0]["text"]).split("-")[1])
        return [
            SegmentCorrection(
                segment_index=index,
                raw_text=str(segment["text"]),
                corrected_text=f"corrected-{clip_index}-{index}",
                applied=True,
                confidence=0.95,
                method="fake-stage25",
                version="egyptian-ar-v1",
                changes=[],
                uncertain=False,
            )
            for index, segment in enumerate(segments)
        ]


class _Reconstructor:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.received: list[list[dict[str, object]]] = []

    def reconstruct(
        self,
        segments: list[dict[str, object]],
        *,
        language: str | None,
        transcription_fingerprint: str,
        correction_version: str,
    ) -> ReconstructionResult:
        self.events.append("stage27")
        self.received.append([dict(segment) for segment in segments])
        clip_index = int(str(segments[0]["raw_text"]).split("-")[1])
        rows = []
        for index, segment in enumerate(segments):
            final = f"final-{clip_index}-{index}"
            rows.append(
                SegmentReconstruction(
                    index,
                    str(segment["raw_text"]),
                    str(segment["corrected_text"]),
                    final,
                    final,
                    True,
                    0.91,
                    ConfidenceLevel.HIGH,
                    (),
                    ReconstructionStatus.APPLIED,
                    reconstruction_method="ollama:qwen3:8b",
                )
            )
        return ReconstructionResult(tuple(rows), "joined", "fingerprint")


def _available_health() -> ProviderHealth:
    return ProviderHealth(
        ProviderAvailability.AVAILABLE, "ollama", "qwen3:8b", "sha256:test", "model available"
    )


def _write_clip(args: list[str]) -> None:
    Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
    Path(args[-1]).write_bytes(b"RIFF")


def _seed_sources(storage: StorageService, manifest: BenchmarkManifest) -> None:
    for source in manifest.sources:
        path = storage.resolve(StorageCategory.SOURCES, source.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"authorized media")


def test_runner_executes_pipeline_in_order_and_writes_deterministic_artifacts(
    tmp_path: Path,
) -> None:
    storage = _RecordingStorage(tmp_path / "storage")
    _seed_sources(storage, _manifest())
    events: list[str] = []

    def ffmpeg(args: list[str]) -> None:
        events.append("ffmpeg")
        assert args[:3] == ["ffmpeg", "-y", "-ss"]
        assert "-t" in args and "-vn" in args and "-ac" in args and "-ar" in args
        Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(args[-1]).write_bytes(b"RIFF")

    reconstructor = _Reconstructor(events)
    runner = BenchmarkRunner(
        storage=storage,
        whisper_engine=_Engine(events),
        corrector=_Corrector(events),
        reconstructor=reconstructor,
        provider_health=_available_health(),
        transcription_options=_options(),
        ffmpeg_binary="ffmpeg",
        command_runner=ffmpeg,
        monotonic=iter([10.0, 20.0]).__next__,
        peak_rss_bytes=lambda: 1234,
        swap_bytes=lambda: 0,
        peak_vram_bytes=lambda: None,
        prompt_settings_fingerprint="production-fingerprint",
    )

    result = runner.run(_manifest())

    assert events[:4] == ["ffmpeg", "asr:clip-0:large-v3-turbo", "stage25", "stage27"]
    assert result.model_identifier == "qwen3:8b"
    assert result.model_digest == "sha256:test"
    assert result.provider_available is True
    assert result.prompt_settings_fingerprint == "production-fingerprint"
    assert result.wall_clock_seconds == pytest.approx(10.0)
    assert result.peak_ram_bytes == 1234
    assert result.human_labels_complete is True
    assert [
        result.improved,
        result.unchanged_correct,
        result.unchanged_wrong,
        result.regressed,
        result.changed_wrong,
        result.hallucinated,
        result.unresolved,
        result.unreviewed,
    ] == [1, 5, 1, 1, 1, 1, 0, 0]
    assert result.semantic_correct_stage25 == pytest.approx(0.6)
    assert result.semantic_correct_stage27 == pytest.approx(0.6)
    assert result.exact_changed_wrong == 10
    assert result.exact_improved == 0
    assert [path.name for path in storage.writes] == [
        "comparison.jsonl",
        "review-worksheet.jsonl",
        "report.json",
    ]
    comparison = [json.loads(line) for line in result.comparison_path.read_text().splitlines()]  # type: ignore[union-attr]
    assert {row["human_label"] for row in comparison} == set(STATUSES)
    assert all(row["status"] in STATUSES for row in comparison)
    required = {
        "clip_id",
        "segment_index",
        "raw",
        "stage25",
        "stage27",
        "reference",
        "status",
        "exact_status",
        "confidence",
        "wer",
        "cer",
        "model",
        "model_digest",
        "human_label",
    }
    assert all(required <= row.keys() for row in comparison)
    assert result.report_path is not None
    assert result.report_path.is_relative_to(storage.category_root(StorageCategory.BENCHMARKS))


def test_runner_preserves_raw_segment_identity_through_derivation(tmp_path: Path) -> None:
    storage = _RecordingStorage(tmp_path / "storage")
    _seed_sources(storage, _manifest())
    events: list[str] = []
    reconstructor = _Reconstructor(events)
    runner = BenchmarkRunner(
        storage=storage,
        whisper_engine=_Engine(events),
        corrector=_Corrector(events),
        reconstructor=reconstructor,
        provider_health=_available_health(),
        transcription_options=_options(),
        command_runner=lambda args: _write_clip(args),
        prompt_settings_fingerprint="production-fingerprint",
    )
    runner.run(_manifest())
    for segments in reconstructor.received:
        assert [segment["start"] for segment in segments] == [
            float(i) for i in range(len(segments))
        ]
        assert [segment["end"] for segment in segments] == [
            float(i + 1) for i in range(len(segments))
        ]
        assert all("raw_text" in segment and "corrected_text" in segment for segment in segments)


def test_referenced_unresolved_row_stays_in_correctness_denominators(tmp_path: Path) -> None:
    """Runtime-unresolved referenced rows stay in Stage 2.5/2.7 denominators."""

    storage = _RecordingStorage(tmp_path / "storage")
    manifest = _small_known_manifest([_reference(0)])
    _seed_sources(storage, manifest)
    events: list[str] = []

    class UnresolvedReconstructor(_Reconstructor):
        def reconstruct(self, segments, **kwargs) -> ReconstructionResult:
            segment = segments[0]
            corrected = str(segment["corrected_text"])
            row = SegmentReconstruction(
                0,
                str(segment["raw_text"]),
                corrected,
                corrected,
                None,
                False,
                0.0,
                ConfidenceLevel.LOW,
                (),
                ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
                reconstruction_method="ollama:qwen3:8b",
            )
            return ReconstructionResult((row,), corrected, "fingerprint")

    runner = BenchmarkRunner(
        storage=storage,
        whisper_engine=_Engine(events, counts={0: 1}),
        corrector=_Corrector(events),
        reconstructor=UnresolvedReconstructor(events),
        provider_health=_available_health(),
        transcription_options=_options(),
        command_runner=lambda args: _write_clip(args),
        prompt_settings_fingerprint="production-fingerprint",
    )

    result = runner.run(manifest)

    assert result.unresolved == 1
    assert result.unchanged_wrong == 1
    assert result.stage25_wrong == 1
    assert result.stage25_correct == 0
    assert result.semantic_correct_stage25 == 0.0
    assert result.semantic_correct_stage27 == 0.0


def test_referenced_unresolved_correct_row_stays_in_correct_denominator(
    tmp_path: Path,
) -> None:
    """A runtime-unresolved row whose Stage 2.5 text is correct stays in stage25_correct."""

    storage = _RecordingStorage(tmp_path / "storage")
    manifest = _small_known_manifest([_reference(0, "unchanged_correct", text="corrected-0-0")])
    _seed_sources(storage, manifest)
    events: list[str] = []

    class UnresolvedReconstructor(_Reconstructor):
        def reconstruct(self, segments, **kwargs) -> ReconstructionResult:
            segment = segments[0]
            corrected = str(segment["corrected_text"])
            row = SegmentReconstruction(
                0,
                str(segment["raw_text"]),
                corrected,
                corrected,
                None,
                False,
                0.0,
                ConfidenceLevel.LOW,
                (),
                ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
                reconstruction_method="ollama:qwen3:8b",
            )
            return ReconstructionResult((row,), corrected, "fingerprint")

    runner = BenchmarkRunner(
        storage=storage,
        whisper_engine=_Engine(events, counts={0: 1}),
        corrector=_Corrector(events),
        reconstructor=UnresolvedReconstructor(events),
        provider_health=_available_health(),
        transcription_options=_options(),
        command_runner=lambda args: _write_clip(args),
        prompt_settings_fingerprint="production-fingerprint",
    )

    result = runner.run(manifest)

    assert result.unresolved == 1
    assert result.unchanged_correct == 1
    assert result.stage25_correct == 1
    assert result.reviewed_denominator == 1


def test_unreferenced_row_is_reported_unreviewed_not_implicitly_safe(tmp_path: Path) -> None:
    """Segments without a human reference are unreviewed and excluded from denominators."""

    storage = _RecordingStorage(tmp_path / "storage")
    manifest = _small_known_manifest([_reference(0, "unchanged_correct", text="corrected-0-0")])
    _seed_sources(storage, manifest)
    events: list[str] = []

    class TwoSegmentReconstructor(_Reconstructor):
        def reconstruct(self, segments, **kwargs) -> ReconstructionResult:
            rows = []
            for index, segment in enumerate(segments):
                corrected = str(segment["corrected_text"])
                rows.append(
                    SegmentReconstruction(
                        index,
                        str(segment["raw_text"]),
                        corrected,
                        corrected,
                        corrected,
                        True,
                        0.95,
                        ConfidenceLevel.HIGH,
                        (),
                        ReconstructionStatus.APPLIED,
                        reconstruction_method="ollama:qwen3:8b",
                    )
                )
            return ReconstructionResult(tuple(rows), "joined", "fingerprint")

    runner = BenchmarkRunner(
        storage=storage,
        whisper_engine=_Engine(events, counts={0: 2}),
        corrector=_Corrector(events),
        reconstructor=TwoSegmentReconstructor(events),
        provider_health=_available_health(),
        transcription_options=_options(),
        command_runner=lambda args: _write_clip(args),
        prompt_settings_fingerprint="production-fingerprint",
    )

    result = runner.run(manifest)

    assert result.unreviewed == 1
    assert result.unchanged_correct == 1
    assert result.stage25_correct == 1
    assert result.reviewed_denominator == 1
    comparison = [json.loads(line) for line in result.comparison_path.read_text().splitlines()]  # type: ignore[union-attr]
    assert comparison[1]["status"] == "unreviewed"
    assert comparison[1]["exact_status"] == "unreviewed"


def _small_known_manifest(reference_segments: list[dict[str, object]]) -> BenchmarkManifest:
    payload: dict[str, object] = {
        "version": "stage-2-7-private-v1",
        "split": "test",
        "sources": [_source("source-a")],
        "clips": [
            {
                "id": "mini-0000-0030",
                "source_id": "source-a",
                "topic": "history",
                "start_seconds": 0,
                "end_seconds": 30,
                "categories": ["narrative"],
                "reference_segments": reference_segments,
            }
        ],
        "known_regression_set": True,
    }
    return cast(BenchmarkManifest, BenchmarkManifest.model_validate(payload))


def test_runner_rejects_missing_source_media(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    runner = BenchmarkRunner(
        storage=storage,
        whisper_engine=_Engine([]),
        corrector=_Corrector([]),
        reconstructor=_Reconstructor([]),
        provider_health=_available_health(),
        transcription_options=_options(),
        command_runner=lambda args: None,
        prompt_settings_fingerprint="production-fingerprint",
    )
    with pytest.raises(StorageValidationError, match="source media"):
        runner.run(_manifest())


def test_known_regression_set_can_run_but_never_passes_readiness(tmp_path: Path) -> None:
    payload = _manifest().model_dump(mode="json")
    payload["sources"] = payload["sources"][:1]
    payload["clips"] = [
        {
            "id": "chernobyl-0000-0030",
            "source_id": "source-a",
            "topic": "history",
            "start_seconds": 0,
            "end_seconds": 30,
            "categories": ["narrative"],
            "reference_segments": [_reference(0, "unchanged_correct")],
        }
    ]
    payload["known_regression_set"] = True

    manifest = BenchmarkManifest.model_validate(payload)
    assert manifest.readiness_eligible is False

    passed, reasons = evaluate_completion_gate(_report(readiness_eligible=False))
    assert passed is False
    assert "known regression sets cannot pass the unseen readiness gate" in reasons


def test_runner_marks_provider_unavailable_and_model_infeasible(tmp_path: Path) -> None:
    storage = _RecordingStorage(tmp_path / "storage")
    _seed_sources(storage, _manifest())
    runner = BenchmarkRunner(
        storage=storage,
        whisper_engine=_Engine([]),
        corrector=_Corrector([]),
        reconstructor=_Reconstructor([]),
        provider_health=ProviderHealth(
            ProviderAvailability.UNAVAILABLE, "ollama", "qwen3:8b", None, "not installed"
        ),
        transcription_options=_options(),
        command_runner=lambda args: _write_clip(args),
        swap_bytes=iter([0, 2 * 1024**3]).__next__,
        prompt_settings_fingerprint="production-fingerprint",
    )
    result = runner.run(_manifest())
    assert result.provider_available is False
    assert result.model_digest == ""
    assert result.swap_after_bytes - result.swap_before_bytes > 1024**3
    assert result.model_feasible is False
