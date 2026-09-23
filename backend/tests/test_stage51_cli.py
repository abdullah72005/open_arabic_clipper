"""Stage 5.1 CLI smoke tests for visual-composition and Stage 5.2 handoff."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from stage51_support import FakeDisplayProbe, seed_stage51
from typer.testing import CliRunner

from app.cli import app
from app.core.settings import get_settings
from app.db.base import Base


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _hermetic_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.composition.queue._dispatch", lambda *args: None)
    monkeypatch.setattr("app.cli.FFprobeDisplayProbe", lambda **kwargs: FakeDisplayProbe())


@pytest.fixture  # type: ignore[untyped-decorator]
def cli_candidate(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    engine = create_engine(get_settings().database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        fixture = seed_stage51(session, monkeypatch)
        candidate_id = str(fixture.stage50.selection.candidate.id)
        session.commit()
    try:
        yield candidate_id
    finally:
        engine.dispose()


def _invoke(args: list[str]) -> dict[str, object]:
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def test_visual_composition_cli_queues_json(cli_candidate: str) -> None:
    payload = _invoke(["visual-composition", cli_candidate])
    assert payload["plan_id"]
    assert payload["job_id"]
    assert payload["status"] == "QUEUED"
    assert payload["queued"] is True
    assert payload["cached"] is False
    assert payload["active"] is False


def test_visual_composition_status_cli_emits_json(cli_candidate: str) -> None:
    created = _invoke(["visual-composition", cli_candidate])
    status = _invoke(["visual-composition-status", cli_candidate])
    assert status["id"] == created["plan_id"]
    assert status["is_current"] is True
    assert "live_freshness" in status
    assert "effective" in status


def test_stage5_2_handoff_cli_flags_false(cli_candidate: str) -> None:
    _invoke(["visual-composition", cli_candidate])
    handoff = _invoke(["stage5-2-handoff", cli_candidate])
    assert handoff["final_timeline_frozen"] is False
    assert handoff["publication_ready"] is False
    assert handoff["render_ready"] is False
    assert handoff["stage5_2_implemented"] is False
    assert handoff["stage6_implemented"] is False


def test_visual_composition_cli_missing_candidate_fails() -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    assert CliRunner().invoke(app, ["visual-composition", missing]).exit_code != 0
    assert CliRunner().invoke(app, ["visual-composition-status", missing]).exit_code != 0
    assert CliRunner().invoke(app, ["stage5-2-handoff", missing]).exit_code != 0
