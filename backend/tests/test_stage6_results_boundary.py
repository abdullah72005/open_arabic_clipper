"""Stage 6 Results boundary: transformed entries consult the Stage 6 assessment.

The existing ``/api/render-executions`` Results surface must consult the shared
read-only Stage 6 readiness evaluator for ``TRANSFORMED_SHORT`` entries rather
than inferring review readiness from a completed renderer row + a purpose label.
Review-ready outputs are distinguished from pending/deferred/historical/diagnostic
media, and historical inspectability never implies current review readiness.

Real FFmpeg media; SYNTHETIC INTEGRATION evidence, not product acceptance.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.api.app import create_app
from app.core.enums import RenderArtifactPurpose, RenderExecutionLifecycle
from app.core.settings import get_settings
from app.db.base import Base
from app.db.session import create_session_factory
from app.models import ClipMaterialization
from app.models.render_execution import RenderExecution
from app.services.storage import StorageService


class _NoopDispatcher:
    def dispatch(self, *_args: object, **_kwargs: object) -> None:
        return None

    def dispatch_finalization(self, *_args: object, **_kwargs: object) -> None:
        return None


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _client() -> TestClient:
    factory = create_session_factory()
    app = create_app(
        session_factory=factory,
        storage=StorageService(get_settings().storage_root),
        dispatcher=_NoopDispatcher(),
    )
    app.state.session_factory = factory
    return TestClient(app)


def _entry(payload: list[dict[str, Any]], render_id: uuid.UUID) -> dict[str, Any]:
    return next(item for item in payload if item["id"] == str(render_id))


def test_transformed_results_entry_consults_committed_readiness(
    engine: Engine, monkeypatch: Any
) -> None:
    from test_stage6_cache_validation import _complete_materialization
    from test_stage6_real_executor import _write_source

    _candidate_id, materialization_id, render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    with _client() as client:
        response = client.get("/api/render-executions?limit=50")
    assert response.status_code == 200
    entry = _entry(response.json(), render_id)
    assert entry["kind"] == "TRANSFORMED_SHORT"
    assert entry["materialization_id"] == str(materialization_id)
    assert entry["review_ready"] is True
    assert entry["materialization_state"] == "REVIEW_READY"
    assert entry["materialization_lifecycle"] == "COMPLETE"
    assert entry["semantic_outcome"] in {"READY_FOR_HUMAN_REVIEW", "MATERIALIZED_WITH_CAUTION"}
    assert entry["artifact_endpoint"] == "materialization"
    assert entry["artifact_available"] is True
    assert entry["historical"] is False
    assert entry["publication_ready"] is False
    assert entry["render_contract_id"]
    assert entry["selection_id"]


def test_transformed_results_historical_is_not_review_ready(
    engine: Engine, monkeypatch: Any
) -> None:
    from test_stage6_cache_validation import _complete_materialization
    from test_stage6_real_executor import _write_source

    _candidate_id, materialization_id, render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    with Session(engine) as session:
        row = session.get(ClipMaterialization, materialization_id)
        assert row is not None
        row.is_current = False
        session.commit()
    with _client() as client:
        payload = client.get("/api/render-executions?limit=50").json()
    entry = _entry(payload, render_id)
    assert entry["materialization_state"] == "HISTORICAL"
    assert entry["review_ready"] is False
    # Historical verified media stays inspectable through the materialization
    # endpoint, but that never implies current review readiness.
    assert entry["historical"] is True
    assert entry["artifact_endpoint"] == "materialization"
    assert any("NOT_CURRENT" in reason for reason in entry["readiness_reasons"])


def test_produce_results_http_flow_is_review_ready(
    engine: Engine, monkeypatch: Any
) -> None:
    """A real review-ready generation is exercised through the HTTP produce→results boundary."""

    from test_stage6_cache_validation import _complete_materialization
    from test_stage6_real_executor import _write_source

    candidate_id, materialization_id, render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    with _client() as client:
        queued = client.post(f"/api/candidates/{candidate_id}/materialization")
        assert queued.status_code == 202, queued.text
        body = queued.json()
        # A genuinely complete generation is a verified cache hit through POST.
        assert body["cached"] is True
        assert body["review_ready"] is True
        detail = client.get(f"/api/materializations/{materialization_id}")
        assert detail.status_code == 200
        assert detail.json()["readiness"]["review_ready"] is True
        artifact = client.get(f"/api/materializations/{materialization_id}/artifact")
        assert artifact.status_code == 200
        results = client.get("/api/render-executions?limit=50").json()
        entry = _entry(results, render_id)
        assert entry["review_ready"] is True
        assert entry["materialization_state"] == "REVIEW_READY"


def test_core_render_entry_is_preserved(engine: Engine) -> None:
    storage = StorageService(get_settings().storage_root)
    relative = "render-executions/core/results.mp4"
    artifact = storage.storage_root / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"\x00\x00\x00\x18ftypmp42core")
    render_id = uuid.uuid4()
    with Session(engine) as session:
        session.add(
            RenderExecution(
                id=render_id,
                source_video_id=uuid.uuid4(),
                clip_candidate_id=uuid.uuid4(),
                artifact_purpose=RenderArtifactPurpose.CORE_SOURCE_VALIDATION,
                lifecycle=RenderExecutionLifecycle.COMPLETE,
                cache_eligible=True,
                is_current=True,
                artifact_reference={
                    "relative_path": relative,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                },
            )
        )
        session.commit()
    with _client() as client:
        payload = client.get("/api/render-executions?limit=50").json()
    entry = _entry(payload, render_id)
    assert entry["kind"] == "CORE_SOURCE_VALIDATION"
    assert entry["materialization_state"] == "CORE"
    assert entry["artifact_endpoint"] == "render_execution"
    assert entry["review_ready"] is False
