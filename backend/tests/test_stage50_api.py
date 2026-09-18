"""Stage 5.0 API and Stage 5.1 handoff endpoints."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import FakeProber, seed_stage50

from app.api.app import create_app
from app.db.base import Base
from app.models import PipelineRun, ProcessingJob, TransformationPlanSelection
from app.services.storage import StorageService


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def dispatch(self, source_id: uuid.UUID, job_id: uuid.UUID) -> None:
        self.calls.append((source_id, job_id))


@pytest.fixture  # type: ignore[untyped-decorator]
def api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings]]:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api50.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = StorageService(tmp_path / "storage")
    app = create_app(session_factory=factory, storage=storage, dispatcher=RecordingDispatcher())
    with TestClient(app) as client:
        yield client, factory, settings


def _seed(
    factory: sessionmaker[Session],
    settings: FakeGovernanceSettings,
    *,
    with_selection: bool = True,
) -> uuid.UUID:
    with factory() as session:
        fixture = seed_stage50(session, settings=settings)
        candidate_id = uuid.UUID(str(fixture.selection.candidate.id))
        if not with_selection:
            for row in session.scalars(select(TransformationPlanSelection)).all():
                session.delete(row)
        session.commit()
    return candidate_id


def test_missing_candidate_returns_404(
    api: tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings],
) -> None:
    client, _, _ = api
    missing = uuid.uuid4()
    assert client.post(f"/api/candidates/{missing}/render-contract").status_code == 404
    assert client.get(f"/api/candidates/{missing}/render-contract").status_code == 404
    assert client.get(f"/api/candidates/{missing}/stage5-1-handoff").status_code == 404


def test_non_executable_preflight_is_persisted_and_reused(
    api: tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings],
) -> None:
    client, factory, settings = api
    candidate_id = _seed(factory, settings, with_selection=False)
    first = client.post(f"/api/candidates/{candidate_id}/render-contract")
    assert first.status_code == 200
    body = first.json()
    assert body["contract_ready"] is False
    assert body["status"] == "BLOCKED"
    second = client.post(f"/api/candidates/{candidate_id}/render-contract")
    assert second.json()["id"] == body["id"]
    fetched = client.get(f"/api/candidates/{candidate_id}/render-contract")
    assert fetched.status_code == 200
    by_id = client.get(f"/api/render-contracts/{body['id']}")
    assert by_id.status_code == 200
    handoff = client.get(f"/api/candidates/{candidate_id}/stage5-1-handoff")
    assert handoff.status_code == 200
    payload = handoff.json()
    assert payload["stage5_1_implemented"] is True
    assert payload["stage5_2_implemented"] is False
    assert payload["stage6_implemented"] is False
    assert payload["contract"]["status"] == "BLOCKED"


def test_executable_contract_handoff_and_no_jobs(
    api: tuple[TestClient, sessionmaker[Session], FakeGovernanceSettings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, factory, settings = api
    candidate_id = _seed(factory, settings, with_selection=True)
    prober = FakeProber()
    monkeypatch.setattr("app.render.service.FFprobe", lambda **kwargs: prober)
    response = client.post(f"/api/candidates/{candidate_id}/render-contract")
    assert response.status_code == 200
    body = response.json()
    assert body["contract_ready"] is True
    assert body["status"] in {"READY_FOR_RENDER_PLANNING", "MATERIALIZATION_REQUIRED"}
    assert body["contract_payload"]["caption_input"]["logical_order_preserved"] is True

    handoff = client.get(f"/api/candidates/{candidate_id}/stage5-1-handoff").json()
    assert handoff["contract"]["id"] == body["id"]
    assert handoff["materialization"]["slots"] is not None
    assert handoff["compatibility"]["outcome"] is not None

    with factory() as session:
        assert session.query(ProcessingJob).count() == 0
        assert session.query(PipelineRun).count() == 0
