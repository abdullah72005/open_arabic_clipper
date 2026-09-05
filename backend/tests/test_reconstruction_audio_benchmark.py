from app.transcription.reconstruction.benchmark import BenchmarkReport, evaluate_completion_gate


def _report(**changes: object) -> BenchmarkReport:
    values: dict[str, object] = {
        "model_identifier": "qwen3:8b",
        "model_digest": "sha256:test",
        "semantic_correct_stage25": 0.70,
        "semantic_correct_stage27": 0.82,
        "stage25_wrong": 20,
        "improved": 5,
        "stage25_correct": 80,
        "regressed": 1,
        "preserved": 79,
        "hallucinated": 0,
        "comprehensibility_stage25": {"slang": 3.0, "fast_speech": 3.0},
        "comprehensibility_stage27": {"slang": 3.5, "fast_speech": 3.0},
        "source_audio_seconds": 720.0,
        "wall_clock_seconds": 120.0,
        "peak_ram_bytes": 1,
        "peak_vram_bytes": None,
    }
    values.update(changes)
    return BenchmarkReport.model_validate(values)


def test_completion_gate_accepts_measured_quality_and_preservation_thresholds() -> None:
    passed, reasons = evaluate_completion_gate(_report())

    assert passed is True
    assert reasons == []


def test_completion_gate_rejects_any_hallucination() -> None:
    passed, reasons = evaluate_completion_gate(_report(hallucinated=1))

    assert passed is False
    assert "hallucinated facts/names/numbers/clauses must be zero" in reasons
