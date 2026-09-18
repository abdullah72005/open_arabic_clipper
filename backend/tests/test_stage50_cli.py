"""Stage 5.0 CLI smoke tests for render-contract and Stage 5.1 handoff."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import FakeProber, seed_stage50
from typer.testing import CliRunner

from app.cli import app
from app.core.settings import get_settings
from app.db.base import Base
from app.models import PipelineRun, ProcessingJob


@pytest.fixture  # type: ignore[untyped-decorator]
def cli_candidate(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    monkeypatch.setattr("app.render.service.FFprobe", lambda **kwargs: FakeProber())
    engine = create_engine(get_settings().database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        fixture = seed_stage50(session, settings=settings)
        candidate_id = str(fixture.selection.candidate.id)
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


def test_render_contract_cli_creates_executable_contract(cli_candidate: str) -> None:
    payload = _invoke(["render-contract", cli_candidate])
    assert payload["contract_ready"] is True
    assert payload["status"] in {"READY_FOR_RENDER_PLANNING", "MATERIALIZATION_REQUIRED"}
    assert payload["live_freshness"] == "CURRENT"
    assert payload["effective"] is True
    assert payload["contract_payload"]["caption_input"]["logical_order_preserved"] is True


def test_render_contract_status_cli_is_read_only(cli_candidate: str) -> None:
    created = _invoke(["render-contract", cli_candidate])
    status = _invoke(["render-contract-status", cli_candidate])
    assert status["id"] == created["id"]
    assert status["status"] == created["status"]
    assert status["live_freshness"] == "CURRENT"


def test_stage5_1_handoff_cli_flags_false(cli_candidate: str) -> None:
    _invoke(["render-contract", cli_candidate])
    handoff = _invoke(["stage5-1-handoff", cli_candidate])
    assert handoff["stage5_1_implemented"] is True
    assert handoff["stage5_2_implemented"] is False
    assert handoff["stage6_implemented"] is False
    assert handoff["contract"]["effective"] is True
    assert handoff["contract"]["live_freshness"] == "CURRENT"


def test_render_contract_cli_creates_no_jobs_or_runs(cli_candidate: str) -> None:
    _invoke(["render-contract", cli_candidate])
    engine = create_engine(get_settings().database_url)
    try:
        with Session(engine) as session:
            assert session.query(ProcessingJob).count() == 0
            assert session.query(PipelineRun).count() == 0
    finally:
        engine.dispose()


def test_render_contract_cli_missing_candidate_fails() -> None:
    result = CliRunner().invoke(app, ["render-contract", "00000000-0000-0000-0000-000000000000"])
    assert result.exit_code != 0
