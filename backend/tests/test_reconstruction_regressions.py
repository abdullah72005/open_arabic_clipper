from __future__ import annotations

import json
from pathlib import Path

FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "app"
    / "transcription"
    / "fixtures"
    / "egyptian_ar_reconstruction.json"
)


def test_reconstruction_fixture_has_required_families_and_unchanged_cases() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]

    assert {"narrative", "entities", "slang", "filler", "code_switching"} <= {
        fixture["category"] for fixture in fixtures
    }
    assert sum(fixture["must_change"] for fixture in fixtures) >= 3
    required = {"previous", "raw", "next", "expected", "protected_tokens"}
    assert all(required <= fixture.keys() for fixture in fixtures)


def test_fixture_phrase_pairs_are_not_embedded_in_reconstruction_production_code() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]
    production_root = FIXTURE_PATH.parents[1] / "reconstruction"
    production = "\n".join(
        path.read_text(encoding="utf-8") for path in production_root.glob("*.py")
    )

    for fixture in fixtures:
        assert not (fixture["raw"] in production and fixture["expected"] in production)
