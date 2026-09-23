"""Stage 5.1 API: visual-composition queue/read and Stage 5.2 handoff endpoints."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage51_support import (
    FakeDetector,
    FakeDisplayProbe,
    FakeFrameSampler,
    FakeSceneCutDetector,
    FakeStage51Settings,
    Stage51Fixture,
    seed_stage51,
)

from app.api.app import create_app
from app.composition.policy import Stage51Config, VisualCompositionStatus
from app.composition.queue import get_or_create_plan_row
from app.composition.service import execute_visual_composition
from app.core.settings import get_settings
from app.db.base import Base
from app.models import ClipCandidate, PipelineRun, ProcessingJob, RenderContract
from app.services.storage import StorageService


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def dispatch(self, source_id: uuid.UUID, job_id: uuid.UUID) -> None:
        self.calls.append((source_id, job_id))


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _hermetic_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.composition.queue._dispatch", lambda *args: None)
    monkeypatch.setattr("app.api.app.FFprobeDisplayProbe", lambda **kwargs: FakeDisplayProbe())


@pytest.fixture  # type: ignore[untyped-decorator]
def api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings]]:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api51.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = StorageService(tmp_path / "storage")
    app = create_app(session_factory=factory, storage=storage, dispatcher=RecordingDispatcher())
    with TestClient(app) as client:
        yield client, factory, settings


def _seed(factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> Stage51Fixture:
    with factory() as session:
        fixture = seed_stage51(session, monkeypatch)
        session.commit()
        return fixture


def _candidate_id(fixture: Stage51Fixture) -> uuid.UUID:
    return uuid.UUID(str(fixture.stage50.selection.candidate.id))


def _execute(
    factory: sessionmaker[Session],
    fixture: Stage51Fixture,
    *,
    config: Stage51Config | None = None,
) -> uuid.UUID:
    resolved = FakeStage51Settings(
        config=config or get_settings().stage51_config(),
        storage_root=fixture.stage50.storage.storage_root,
    )
    with factory() as session:
        candidate = session.get(ClipCandidate, _candidate_id(fixture))
        assert candidate is not None
        row = get_or_create_plan_row(session, candidate)
        result = execute_visual_composition(
            session,
            row.id,
            storage=fixture.stage50.storage,
            settings=resolved,  # type: ignore[arg-type]
            display_probe=fixture.display_probe,
            frame_sampler=FakeFrameSampler(),
            scene_cut_detector=FakeSceneCutDetector(),
            detector=FakeDetector(),
        )
        assert result is not None and result.plan_ready is True
        session.commit()
        return uuid.UUID(str(result.id))


def test_queue_accepts_and_returns_queue_response(
    api: tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, factory, _ = api
    fixture = _seed(factory, monkeypatch)
    response = client.post(f"/api/candidates/{_candidate_id(fixture)}/visual-composition")
    assert response.status_code == 202
    body = response.json()
    assert body["plan_id"]
    assert body["job_id"]
    assert body["status"] == "QUEUED"
    assert body["queued"] is True
    assert body["cached"] is False
    assert body["active"] is False
    with factory() as session:
        assert session.query(ProcessingJob).count() == 1
        assert session.query(PipelineRun).count() == 0


def test_missing_candidate_returns_404(
    api: tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings],
) -> None:
    client, _, _ = api
    missing = uuid.uuid4()
    assert client.post(f"/api/candidates/{missing}/visual-composition").status_code == 404
    assert client.get(f"/api/candidates/{missing}/visual-composition").status_code == 404
    assert client.get(f"/api/candidates/{missing}/stage5-2-handoff").status_code == 404


def test_non_executable_contract_returns_409(
    api: tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, factory, _ = api
    fixture = _seed(factory, monkeypatch)
    with factory() as session:
        contract = session.get(RenderContract, fixture.contract.row.id)
        assert contract is not None
        contract.contract_ready = False
        contract.status = VisualCompositionStatus.BLOCKED
        session.commit()
    response = client.post(f"/api/candidates/{_candidate_id(fixture)}/visual-composition")
    assert response.status_code == 409


def test_get_current_returns_404_when_none(
    api: tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, factory, _ = api
    fixture = _seed(factory, monkeypatch)
    response = client.get(f"/api/candidates/{_candidate_id(fixture)}/visual-composition")
    assert response.status_code == 404


def test_get_by_id_returns_exact_historical_row_state(
    api: tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, factory, _ = api
    fixture = _seed(factory, monkeypatch)
    historical_id = _execute(
        factory,
        fixture,
        config=replace(get_settings().stage51_config(), analysis_fps=4.0),
    )
    current_id = _execute(factory, fixture, config=get_settings().stage51_config())
    assert historical_id != current_id

    current = client.get(f"/api/candidates/{_candidate_id(fixture)}/visual-composition")
    assert current.status_code == 200
    current_body = current.json()
    assert current_body["id"] == str(current_id)
    assert current_body["is_current"] is True
    assert current_body["live_freshness"] == "CURRENT"
    assert current_body["effective"] is True

    historical = client.get(f"/api/visual-compositions/{historical_id}")
    assert historical.status_code == 200
    historical_body = historical.json()
    assert historical_body["id"] == str(historical_id)
    assert historical_body["is_current"] is False
    assert historical_body["effective"] is False
    assert historical_body["live_freshness"] == "STALE"

    missing = client.get(f"/api/visual-compositions/{uuid.uuid4()}")
    assert missing.status_code == 404


def test_stage5_2_handoff_reports_flags_false(
    api: tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, factory, _ = api
    fixture = _seed(factory, monkeypatch)
    _execute(factory, fixture)
    response = client.get(f"/api/candidates/{_candidate_id(fixture)}/stage5-2-handoff")
    assert response.status_code == 200
    body = response.json()
    assert body["final_timeline_frozen"] is False
    assert body["publication_ready"] is False
    assert body["render_ready"] is False
    assert body["stage5_2_implemented"] is False
    assert body["stage6_implemented"] is False
    assert body["plan"]["effective"] is True
    assert body["readiness"]["stage5_2_handoff_eligible"] is True
    assert body["readiness"]["source_framing_ready"] is True
    assert body["readiness"]["source_captions_ready"] is True
