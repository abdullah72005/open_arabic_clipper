from app.transcription.correction import ContextualCorrector
from app.transcription.correction_benchmark import (
    default_fixture_path,
    run_correction_fixture_benchmark,
)


def test_fixture_benchmark_improves_egyptian_examples_without_english_regression() -> None:
    """The regression corpus measures useful corrections and catches category regressions."""

    report = run_correction_fixture_benchmark(
        default_fixture_path(), ContextualCorrector.from_default_lexicon()
    )

    assert report.improved == 3
    assert report.worsened == 0
    assert report.unchanged == 11
    assert report.automatic_correction_rate == 3 / 14
    assert report.uncertain_rate == 11 / 14
    assert report.by_category["english_only"].worsened == 0
    assert report.corrected_exact_match_rate > report.baseline_exact_match_rate
    assert report.wall_clock_seconds >= 0
    assert report.peak_memory_bytes > 0
