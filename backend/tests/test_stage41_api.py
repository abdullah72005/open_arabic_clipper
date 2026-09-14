"""Focused Stage 4.1 API, CLI, handoff, and lifecycle-boundary tests."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from stage41_support import (
    FakeStage41Settings,
    install_stage41_settings,
    run_planning,
    seed_stage41,
)

from app.api.app import create_app
from app.core.enums import PipelineStage, SemanticProviderMode, TransformationStrategyType
from app.db.base import Base
from app.services.storage import StorageService
from app.workers.tasks import _NEXT_STAGE

ApiFixture = tuple[TestClient, sessionmaker[Session], FakeStage41Settings]


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def dispatch(self, source_id: uuid.UUID, job_id: uuid.UUID) -> None:
        self.calls.append((source_id, job_id))


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.workers.tasks.run_transformation_planning.delay", lambda *a, **k: None)


@pytest.fixture  # type: ignore[untyped-decorator]
def api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ApiFixture]:
    monkeypatch.delenv("CLIPFACTORY_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = FakeStage41Settings(mode=SemanticProviderMode.ADAPTIVE)
    install_stage41_settings(monkeypatch, settings)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api41.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = StorageService(tmp_path / "storage")
    app = create_app(session_factory=factory, storage=storage, dispatcher=RecordingDispatcher())
    app.state.session_factory = factory
    app.state.storage = storage
    with TestClient(app) as client:
        yield client, factory, settings


def _seed_ready(
    factory: sessionmaker[Session],
    settings: FakeStage41Settings,
    *,
    second_strategy: tuple[TransformationStrategyType, str] | None = None,
) -> tuple[uuid.UUID, tuple[object, ...]]:
    with factory() as session:
        seed = seed_stage41(session, settings=settings, second_strategy=second_strategy)
        session.commit()
        return seed[1].id, seed


def test_queue_endpoint_requires_ready_stage40(api: ApiFixture) -> None:
    client, factory, _settings = api
    with factory() as session:
        from app.core.enums import (
            CandidateDisposition,
            ContentType,
            OriginalityRisk,
            RightsRisk,
        )
        from app.models import ClipCandidate, SourceVideo

        source = SourceVideo(source_uri="/tmp/no-ref.mp4", content_hash=f"no-ref-{uuid.uuid4()}")
        session.add(source)
        session.flush()
        candidate = ClipCandidate(
            source_video_id=source.id,
            candidate_key=f"no-ref-{uuid.uuid4()}",
            disposition=CandidateDisposition.CANDIDATE,
            start_time=0.0,
            end_time=10.0,
            start_segment_index=0,
            end_segment_index=0,
            segment_indexes=[0],
            primary_content_type=ContentType.OTHER,
            rights_risk=RightsRisk.UNDETERMINED,
            originality_risk=OriginalityRisk.UNDETERMINED,
        )
        session.add(candidate)
        session.commit()
        candidate_id = candidate.id
    response = client.post(f"/api/candidates/{candidate_id}/transformation-plans")
    assert response.status_code == 409


def test_queue_and_read_plan_set(api: ApiFixture) -> None:
    client, factory, settings = api
    candidate_id, seed = _seed_ready(factory, settings)
    with factory() as session:
        run_planning(session, settings, seed)
        session.commit()
    response = client.get(f"/api/candidates/{candidate_id}/transformation-plans")
    assert response.status_code == 200
    payload = response.json()
    assert payload["plans"]
    assert payload["plans"][0]["blocks"]

    plan_set_id = payload["id"]
    detail = client.get(f"/api/transformation-plan-sets/{plan_set_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == plan_set_id

    queue = client.post(f"/api/candidates/{candidate_id}/transformation-plans")
    assert queue.status_code == 202
    assert queue.json()["cached"] is True


def test_stage42_handoff_reports_not_implemented(api: ApiFixture) -> None:
    client, factory, settings = api
    candidate_id, seed = _seed_ready(factory, settings)
    with factory() as session:
        run_planning(session, settings, seed)
        session.commit()
    response = client.get(f"/api/candidates/{candidate_id}/stage4-2-handoff")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stage4_2_implemented"] is False
    assert payload["stage4_3_implemented"] is False
    assert payload["plans"]
    plan = payload["plans"][0]
    assert plan["blocks"]
    assert plan["hero"]["block_index"] in (0, 1)
    serialized = json.dumps(payload).casefold()
    assert "winner" not in serialized
    assert "approved" not in serialized


def test_cli_planning_commands(api: ApiFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    client, factory, settings = api
    candidate_id, seed = _seed_ready(factory, settings)
    with factory() as session:
        run_planning(session, settings, seed)
        session.commit()

    from typer.testing import CliRunner

    import app.cli as cli_module

    monkeypatch.setattr(cli_module, "create_session_factory", lambda: factory)
    runner = CliRunner()
    generate = runner.invoke(cli_module.app, ["transformation-plan-generate", str(candidate_id)])
    assert generate.exit_code == 0
    assert "plan_set_id" in generate.stdout

    plans = runner.invoke(cli_module.app, ["transformation-plans", str(candidate_id)])
    assert plans.exit_code == 0
    parsed = json.loads(plans.stdout)
    assert parsed["plans"]

    handoff = runner.invoke(cli_module.app, ["transformation-plan-handoff", str(candidate_id)])
    assert handoff.exit_code == 0
    assert json.loads(handoff.stdout)["stage4_2_implemented"] is False


def test_no_new_pipeline_stage_or_next_stage_entry() -> None:
    stage_values = {stage.value for stage in PipelineStage}
    assert all("PLANNING" not in value for value in stage_values)
    assert all("TRANSFORM" not in value for value in stage_values)
    assert "TRANSFORMATION_PLANNING" not in {k.value for k in _NEXT_STAGE}
