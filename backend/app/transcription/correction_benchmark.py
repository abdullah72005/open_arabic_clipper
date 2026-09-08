"""Deterministic fixture benchmark for conservative transcript correction."""

from __future__ import annotations

import argparse
import json
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from app.transcription.correction import ContextualCorrector, normalize_for_comparison
from app.transcription.normalization import normalize_transcript


@dataclass(frozen=True)
class CategoryBenchmark:
    total: int = 0
    improved: int = 0
    unchanged: int = 0
    worsened: int = 0


@dataclass(frozen=True)
class CorrectionBenchmarkReport:
    total: int
    improved: int
    unchanged: int
    worsened: int
    automatic_correction_rate: float
    uncertain_rate: float
    baseline_exact_match_rate: float
    corrected_exact_match_rate: float
    baseline_normalized_token_error_rate: float
    normalized_token_error_rate: float
    wall_clock_seconds: float
    peak_memory_bytes: int
    by_category: dict[str, CategoryBenchmark]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def default_fixture_path() -> Path:
    """Return the repository-shipped manually reviewed fixture corpus."""

    return Path(__file__).with_name("fixtures") / "egyptian_ar_correction.json"


def run_correction_fixture_benchmark(
    path: Path, corrector: ContextualCorrector | None
) -> CorrectionBenchmarkReport:
    """Compare current baseline normalization with optional correction on the same cases."""

    fixtures = json.loads(path.read_text(encoding="utf-8"))["fixtures"]
    started = perf_counter()
    tracemalloc.start()
    improved = unchanged = worsened = automatic = uncertain = 0
    baseline_matches = corrected_matches = 0
    baseline_distance = corrected_distance = expected_token_count = 0
    categories: dict[str, CategoryBenchmark] = {}

    for fixture in fixtures:
        raw = str(fixture["raw"])
        expected = str(fixture["expected"])
        baseline = normalize_transcript(raw)
        if corrector is None:
            corrected = baseline
            applied = False
            is_uncertain = True
        else:
            correction = corrector.correct(
                [
                    {"text": str(fixture["previous"])},
                    {"text": raw},
                    {"text": str(fixture["next"])},
                ]
            )[1]
            corrected = correction.corrected_text
            applied = correction.applied
            is_uncertain = correction.uncertain

        baseline_is_correct = baseline == expected
        corrected_is_correct = corrected == expected
        if not baseline_is_correct and corrected_is_correct:
            outcome = "improved"
            improved += 1
        elif baseline_is_correct and not corrected_is_correct:
            outcome = "worsened"
            worsened += 1
        else:
            outcome = "unchanged"
            unchanged += 1
        automatic += int(applied)
        uncertain += int(is_uncertain)
        baseline_matches += int(baseline_is_correct)
        corrected_matches += int(corrected_is_correct)
        expected_tokens = _tokens(expected)
        expected_token_count += max(len(expected_tokens), 1)
        baseline_distance += _token_distance(_tokens(baseline), expected_tokens)
        corrected_distance += _token_distance(_tokens(corrected), expected_tokens)

        category = str(fixture["category"])
        current = categories.get(category, CategoryBenchmark())
        categories[category] = CategoryBenchmark(
            total=current.total + 1,
            improved=current.improved + int(outcome == "improved"),
            unchanged=current.unchanged + int(outcome == "unchanged"),
            worsened=current.worsened + int(outcome == "worsened"),
        )

    _current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total = len(fixtures)
    return CorrectionBenchmarkReport(
        total=total,
        improved=improved,
        unchanged=unchanged,
        worsened=worsened,
        automatic_correction_rate=automatic / total if total else 0.0,
        uncertain_rate=uncertain / total if total else 0.0,
        baseline_exact_match_rate=baseline_matches / total if total else 0.0,
        corrected_exact_match_rate=corrected_matches / total if total else 0.0,
        baseline_normalized_token_error_rate=baseline_distance / expected_token_count,
        normalized_token_error_rate=corrected_distance / expected_token_count,
        wall_clock_seconds=perf_counter() - started,
        peak_memory_bytes=peak_memory,
        by_category=categories,
    )


def main(argv: list[str] | None = None) -> None:
    """Print machine-readable baseline or corrected benchmark evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=default_fixture_path())
    parser.add_argument("--baseline", action="store_true")
    arguments = parser.parse_args(argv)
    report = run_correction_fixture_benchmark(
        arguments.fixture,
        None if arguments.baseline else ContextualCorrector.from_default_lexicon(),
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, sort_keys=True))


def _tokens(text: str) -> list[str]:
    return normalize_for_comparison(text).split()


def _token_distance(actual: list[str], expected: list[str]) -> int:
    """Levenshtein distance for a readable fixture signal, not dialect ground truth."""

    previous = list(range(len(expected) + 1))
    for actual_index, actual_token in enumerate(actual, start=1):
        current = [actual_index]
        for expected_index, expected_token in enumerate(expected, start=1):
            current.append(
                min(
                    previous[expected_index] + 1,
                    current[expected_index - 1] + 1,
                    previous[expected_index - 1] + (actual_token != expected_token),
                )
            )
        previous = current
    return previous[-1]


if __name__ == "__main__":
    main()
