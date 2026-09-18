"""Stage 4.3 API, CLI, execution handoff, and scope-boundary tests."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from stage43_support import (
    FakeGovernanceSettings,
    install_selection_settings,
    make_result_spec,
    seed_selection_fixture,
)

from app.api.app import create_app
from app.core.enums import GovernancePlanStatus
from app.db.base import Base
from app.services.storage import StorageService

ApiFixture = tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings]


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def dispatch(self, source_id: uuid.UUID, job_id: uuid.UUID) -> None:
        self.calls.append((source_id, job_id))


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.workers.tasks.run_transformation_planning.delay", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.workers.tasks.run_transformation_governance.delay", lambda *a, **k: None
    )


@pytest.fixture  # type: ignore[untyped-decorator]
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ApiFixture]:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api43.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = StorageService(tmp_path / "storage")
    app = create_app(session_factory=factory, storage=storage, dispatcher=RecordingDispatcher())
    app.state.session_factory = factory
    app.state.storage = storage
    with TestClient(app) as client:
        yield client, factory, settings


def _seed(factory: sessionmaker[Session], settings: FakeGovernanceSettings) -> uuid.UUID:
    with factory() as session:
        fixture = seed_selection_fixture(
            session,
            settings=settings,
            result_specs=[
                make_result_spec(status=GovernancePlanStatus.APPROVED_FOR_SELECTION.value)
            ],
        )
        candidate_id = uuid.UUID(str(fixture.candidate.id))
        session.commit()
    return candidate_id


def test_create_and_read_selection(api: ApiFixture) -> None:
    client, factory, settings = api
    candidate_id = _seed(factory, settings)
    response = client.post(f"/api/candidates/{candidate_id}/transformation-selection")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "PLAN_SELECTED"
    assert payload["selected_plan_id"]
    assert payload["live_freshness"] == "VERIFIED_CURRENT"

    read = client.get(f"/api/candidates/{candidate_id}/transformation-selection")
    assert read.status_code == 200
    assert read.json()["id"] == payload["id"]

    direct = client.get(f"/api/transformation-selections/{payload['id']}")
    assert direct.status_code == 200
    assert direct.json()["id"] == payload["id"]
    assert direct.json()["effective"] is True


def test_historical_selection_get_returns_requested_row(api: ApiFixture) -> None:
    client, factory, settings = api
    candidate_id = _seed(factory, settings)
    first = client.post(f"/api/candidates/{candidate_id}/transformation-selection").json()
    assert first["status"] == "PLAN_SELECTED"
    original_plan_id = first["selected_plan_id"]

    from app.models import TransformationPlan
    from app.transformation.planning.queue import get_plan_set_for_candidate

    with factory() as session:
        plan_set = get_plan_set_for_candidate(session, candidate_id)
        assert plan_set is not None
        plan = session.query(TransformationPlan).filter_by(plan_set_id=plan_set.id).first()
        assert plan is not None
        plan.blocks = [dict(block) for block in (plan.blocks or [])] + [
            {"index": 99, "block_type": "TRANSITION", "estimated_duration": 1.0}
        ]
        session.commit()

    second = client.post(f"/api/candidates/{candidate_id}/transformation-selection").json()
    assert second["id"] != first["id"]

    response = client.get(f"/api/transformation-selections/{first['id']}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == first["id"]
    assert payload["status"] == "PLAN_SELECTED"
    assert payload["selected_plan_id"] == original_plan_id
    assert payload["selected_governance_snapshot"] == first["selected_governance_snapshot"]
    assert payload["effective"] is False
    assert payload["is_current"] is False

    current = client.get(f"/api/transformation-selections/{second['id']}").json()
    assert current["id"] == second["id"]
    assert current["is_current"] is True


def test_create_selection_unknown_candidate_is_404(api: ApiFixture) -> None:
    client, _factory, _settings = api
    response = client.post(f"/api/candidates/{uuid.uuid4()}/transformation-selection")
    assert response.status_code == 404


def test_get_selection_before_selection_is_404(api: ApiFixture) -> None:
    client, factory, settings = api
    candidate_id = _seed(factory, settings)
    response = client.get(f"/api/candidates/{candidate_id}/transformation-selection")
    assert response.status_code == 404


def test_execution_handoff_exposes_one_plan_or_none(api: ApiFixture) -> None:
    client, factory, settings = api
    candidate_id = _seed(factory, settings)
    client.post(f"/api/candidates/{candidate_id}/transformation-selection")
    response = client.get(f"/api/candidates/{candidate_id}/execution-handoff")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["selected_plan"] is not None
    assert payload["blocks"]
    assert payload["execution_readiness"] == "READY_FOR_FINAL_REFINEMENT"
    assert payload["final_clip_refinement_required"] is True
    assert payload["stage5_implemented"] is False
    assert payload["stage6_tts_implemented"] is False
    serialized = json.dumps(payload).casefold()
    for forbidden in (
        "safe_for_youtube",
        "safe_for_facebook",
        "will_be_monetized",
        "will_not_be_flagged",
        "algorithm_safe",
        "voice_id",
        "tts_provider",
    ):
        assert forbidden not in serialized


def test_execution_handoff_unknown_candidate_is_404(api: ApiFixture) -> None:
    client, _factory, _settings = api
    response = client.get(f"/api/candidates/{uuid.uuid4()}/execution-handoff")
    assert response.status_code == 404


def test_execution_handoff_without_selection_reports_blocked(api: ApiFixture) -> None:
    client, factory, settings = api
    candidate_id = _seed(factory, settings)
    response = client.get(f"/api/candidates/{candidate_id}/execution-handoff")
    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_plan"] is None
    assert payload["execution_readiness"] == "BLOCKED"
    assert payload["selection"]["selection_id"] is None


def test_cli_select_and_handoff(api: ApiFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    _client, factory, settings = api
    candidate_id = _seed(factory, settings)
    import app.cli as cli_module

    monkeypatch.setattr(cli_module, "create_session_factory", lambda: factory)
    from typer.testing import CliRunner

    runner = CliRunner()
    selected = runner.invoke(cli_module.app, ["transformation-selection", str(candidate_id)])
    assert selected.exit_code == 0, selected.output
    assert "PLAN_SELECTED" in selected.output

    handoff = runner.invoke(cli_module.app, ["transformation-selection-handoff", str(candidate_id)])
    assert handoff.exit_code == 0, handoff.output
    assert "execution_readiness" in handoff.output


def test_governance_handoff_still_has_no_winner_but_reports_stage43_implemented(
    api: ApiFixture,
) -> None:
    client, factory, settings = api
    candidate_id = _seed(factory, settings)
    response = client.get(f"/api/candidates/{candidate_id}/stage4-3-handoff")
    assert response.status_code == 200
    payload = response.json()
    serialized = json.dumps(payload)
    assert payload["stage4_3_implemented"] is True
    assert "selected_plan_id" not in serialized
    assert "winner" not in serialized


def test_stage_4_3_adds_no_pipeline_stage_or_next_stage(api: ApiFixture) -> None:
    from app.core.enums import PipelineStage
    from app.workers.tasks import _NEXT_STAGE

    assert not hasattr(PipelineStage, "TRANSFORMATION_SELECTION")
    assert "stage4_3" not in json.dumps(_NEXT_STAGE).casefold()
