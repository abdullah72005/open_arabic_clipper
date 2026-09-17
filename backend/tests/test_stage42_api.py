"""Stage 4.2 API, CLI, Stage 4.3 handoff, and scope-boundary tests."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from stage41_support import run_planning, seed_stage41
from stage42_support import (
    FakeGovernanceSettings,
    install_stage42_settings,
)

from app.api.app import create_app
from app.core.enums import PipelineStage
from app.db.base import Base
from app.services.storage import StorageService
from app.workers.tasks import _NEXT_STAGE

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
    monkeypatch.delenv("CLIPFACTORY_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = FakeGovernanceSettings()
    install_stage42_settings(monkeypatch, settings)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api42.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = StorageService(tmp_path / "storage")
    app = create_app(session_factory=factory, storage=storage, dispatcher=RecordingDispatcher())
    app.state.session_factory = factory
    app.state.storage = storage
    with TestClient(app) as client:
        yield client, factory, settings


def _seed_planned(factory: sessionmaker[Session], settings: FakeGovernanceSettings) -> uuid.UUID:
    with factory() as session:
        seed = seed_stage41(session, settings=settings)
        session.commit()
        candidate_id = seed[1].id
    with factory() as session:
        run_planning(session, settings, seed)
        session.commit()
    return candidate_id


def test_queue_requires_complete_stage41_plan_set(api: ApiFixture, tmp_path: Path) -> None:
    client, factory, _settings = api
    from app.core.enums import CandidateDisposition, ContentType, OriginalityRisk, RightsRisk
    from app.models import ClipCandidate, SourceVideo

    with factory() as session:
        source = SourceVideo(
            source_uri="/tmp/no-plan-set.mp4", content_hash=f"no-plan-{uuid.uuid4()}"
        )
        session.add(source)
        session.flush()
        candidate = ClipCandidate(
            source_video_id=source.id,
            candidate_key=f"no-plan-{uuid.uuid4()}",
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
    response = client.post(f"/api/candidates/{candidate_id}/transformation-governance")
    assert response.status_code == 409


def test_queue_and_read_governance_set(api: ApiFixture) -> None:
    client, factory, settings = api
    candidate_id = _seed_planned(factory, settings)
    response = client.post(f"/api/candidates/{candidate_id}/transformation-governance")
    assert response.status_code == 202
    payload = response.json()
    assert payload["governance_set_id"]
    assert payload["queued"] is True

    from app.transformation.governance.executor import (
        build_transformation_governance_executor,
    )
    from app.transformation.governance.queue import get_governance_set_for_candidate

    with factory() as session:
        governance_set = get_governance_set_for_candidate(session, candidate_id)
        assert governance_set is not None
        build_transformation_governance_executor(session, settings).execute(governance_set.id)
        session.commit()

    read = client.get(f"/api/candidates/{candidate_id}/transformation-governance")
    assert read.status_code == 200
    body = read.json()
    assert body["execution_status"] == "COMPLETE"
    assert body["results"]
    assert body["platform_policy_profile_version"] == "stage4.2-platform-policy-2026-09-17-v1"

    direct = client.get(f"/api/transformation-governance-sets/{body['id']}")
    assert direct.status_code == 200


def test_stage4_3_handoff_has_no_winner_or_selection(api: ApiFixture) -> None:
    client, factory, settings = api
    candidate_id = _seed_planned(factory, settings)
    client.post(f"/api/candidates/{candidate_id}/transformation-governance")
    from app.transformation.governance.executor import (
        build_transformation_governance_executor,
    )
    from app.transformation.governance.queue import get_governance_set_for_candidate

    with factory() as session:
        governance_set = get_governance_set_for_candidate(session, candidate_id)
        assert governance_set is not None
        build_transformation_governance_executor(session, settings).execute(governance_set.id)
        session.commit()

    response = client.get(f"/api/candidates/{candidate_id}/stage4-3-handoff")
    assert response.status_code == 200
    payload = response.json()
    serialized = json.dumps(payload)
    assert payload["stage4_3_implemented"] is False
    assert "selected_plan_id" not in serialized
    assert "winner" not in serialized
    assert "render_ready" not in serialized
    assert "publication_approved" not in serialized
    assert payload["rights_and_provenance"]["rights_risk"] in {"LOW", "UNDETERMINED", "ELEVATED"}
    assert payload["governance_set"] is not None
    assert payload["plans"]


def test_narration_none_requires_no_tts_identity(api: ApiFixture) -> None:
    client, factory, settings = api
    candidate_id = _seed_planned(factory, settings)
    client.post(f"/api/candidates/{candidate_id}/transformation-governance")
    from app.transformation.governance.executor import (
        build_transformation_governance_executor,
    )
    from app.transformation.governance.queue import get_governance_set_for_candidate

    with factory() as session:
        governance_set = get_governance_set_for_candidate(session, candidate_id)
        build_transformation_governance_executor(session, settings).execute(governance_set.id)
        session.commit()
    response = client.get(f"/api/candidates/{candidate_id}/stage4-3-handoff")
    serialized = json.dumps(response.json()).casefold()
    for token in ("tts_provider", "voice_id", "render_resolution", "publish_schedule"):
        assert token not in serialized


def test_stage_4_3_remains_unimplemented_and_no_lifecycle(api: ApiFixture) -> None:
    assert not hasattr(PipelineStage, "TRANSFORMATION_GOVERNANCE")
    assert all("GOVERNANCE" not in str(value) for value in _NEXT_STAGE.values())
    assert "stage4_3" not in json.dumps(_NEXT_STAGE).casefold()


def test_cli_govern_and_read(api: ApiFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    _client, factory, settings = api
    candidate_id = _seed_planned(factory, settings)
    import app.cli as cli_module

    monkeypatch.setattr(cli_module, "create_session_factory", lambda: factory)
    from typer.testing import CliRunner

    runner = CliRunner()
    queued = runner.invoke(cli_module.app, ["transformation-govern", str(candidate_id)])
    assert queued.exit_code == 0, queued.output
    assert "governance_set_id" in queued.output

    from app.transformation.governance.executor import (
        build_transformation_governance_executor,
    )
    from app.transformation.governance.queue import get_governance_set_for_candidate

    with factory() as session:
        governance_set = get_governance_set_for_candidate(session, candidate_id)
        assert governance_set is not None
        build_transformation_governance_executor(session, settings).execute(governance_set.id)
        session.commit()

    read = runner.invoke(cli_module.app, ["transformation-governance", str(candidate_id)])
    assert read.exit_code == 0, read.output
    assert "execution_status" in read.output

    handoff = runner.invoke(
        cli_module.app, ["transformation-governance-handoff", str(candidate_id)]
    )
    assert handoff.exit_code == 0, handoff.output
    assert "stage4_3_implemented" in handoff.output
