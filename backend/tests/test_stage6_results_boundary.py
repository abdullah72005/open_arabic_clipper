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
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from app.api.app import create_app
from app.core.enums import JobKind, JobStatus, RenderArtifactPurpose, RenderExecutionLifecycle
from app.core.settings import get_settings
from app.db.base import Base
from app.db.session import create_session_factory
from app.models import ClipMaterialization, ProcessingJob
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
    """A completed generation is served through a *cached* produce POST → results.

    This case starts from an already-completed generation (``_complete_materialization``)
    and therefore exercises the cached-POST path: the POST does no work and returns the
    verified cache hit. The fresh POST → pending → executor/callback/finalizer → ready
    path is covered separately by
    ``test_fresh_produce_post_through_executor_to_results_ready``.
    """

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
        assert entry["bound_render"] is True


def test_fresh_produce_post_through_executor_to_results_ready(
    engine: Engine, monkeypatch: Any
) -> None:
    """A fresh POST → pending → executor/callback/finalizer → ready HTTP flow.

    Unlike the cached-POST case above, this starts from a prepared generation
    (``AWAITING_RENDER``), POSTs through the real HTTP boundary (which must report
    pending, not review-ready), then drives the exact render → callback → finalizer
    worker callables before asserting Results review readiness.
    """

    from test_stage6_cache_validation import _prepare_to_awaiting_render
    from test_stage6_real_executor import _write_source

    from app.workers.tasks import (
        _request_finalization_after_render,
        run_materialization_finalization,
        run_render_execution,
    )

    candidate_id, materialization_id, render_id, render_job_id, _storage = (
        _prepare_to_awaiting_render(engine, monkeypatch, media_writer=_write_source)
    )

    # The POST sees a durable, not-yet-complete generation: it is pending, never a
    # cache hit and never review-ready.
    with _client() as client:
        queued = client.post(f"/api/candidates/{candidate_id}/materialization")
        assert queued.status_code == 202, queued.text
        body = queued.json()
        assert body["cached"] is False
        assert body["review_ready"] is False
        assert body["materialization_id"] == str(materialization_id)

    # Drive render → callback → finalizer exactly as the workers would.
    run_render_execution(str(render_id), str(render_job_id))
    _request_finalization_after_render(render_id)
    with Session(engine) as session:
        finalizer = session.scalars(
            select(ProcessingJob)
            .where(
                ProcessingJob.materialization_id == materialization_id,
                ProcessingJob.kind == JobKind.MATERIALIZATION,
                ProcessingJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
            )
            .order_by(ProcessingJob.created_at.desc())
        ).first()
        assert finalizer is not None
        finalizer_id = finalizer.id
    run_materialization_finalization(str(candidate_id), str(finalizer_id))

    with _client() as client:
        results = client.get("/api/render-executions?limit=50").json()
    entry = _entry(results, render_id)
    assert entry["review_ready"] is True
    assert entry["materialization_state"] == "REVIEW_READY"
    assert entry["bound_render"] is True
    assert entry["artifact_endpoint"] == "materialization"
    assert entry["artifact_available"] is True


def test_retained_historical_render_does_not_inherit_current_readiness(
    engine: Engine, monkeypatch: Any
) -> None:
    """A retained historical render for the same envelope is never review-ready.

    Both renders share the current materialization's ``materialization_id``. Only the
    exact reverse-bound render may inherit its committed readiness; the older render
    is HISTORICAL, keeps its own output/repair lineage, and is served only through
    its own verified render-by-ID artifact route — never the current artifact URL.
    """

    from test_stage6_cache_validation import _complete_materialization
    from test_stage6_real_executor import _write_source

    _candidate_id, materialization_id, render_id, storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    historical_relative = "render-executions/historical/persisted.mp4"
    historical_path = storage.storage_root / historical_relative
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    historical_path.write_bytes(b"\x00\x00\x00\x18ftypmp42historical")
    historical_id = uuid.uuid4()
    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        session.add(
            RenderExecution(
                id=historical_id,
                source_video_id=envelope.source_video_id,
                clip_candidate_id=envelope.clip_candidate_id,
                materialization_id=envelope.id,
                artifact_purpose=RenderArtifactPurpose.TRANSFORMED_SHORT,
                lifecycle=RenderExecutionLifecycle.COMPLETE,
                cache_eligible=True,
                is_current=False,
                input_fingerprint="historical-input-fingerprint",
                output_fingerprint="historical-output-fingerprint",
                artifact_reference={
                    "relative_path": historical_relative,
                    "sha256": hashlib.sha256(historical_path.read_bytes()).hexdigest(),
                },
            )
        )
        session.commit()

    with _client() as client:
        payload = client.get("/api/render-executions?limit=50").json()
        bound_direct = client.get(f"/api/render-executions/{render_id}/artifact")
        historical_direct = client.get(f"/api/render-executions/{historical_id}/artifact")

    bound = _entry(payload, render_id)
    assert bound["review_ready"] is True
    assert bound["bound_render"] is True
    assert bound["materialization_state"] == "REVIEW_READY"
    assert bound["artifact_endpoint"] == "materialization"
    assert bound_direct.status_code == 200

    historical = _entry(payload, historical_id)
    assert historical["is_current"] is False
    assert historical["review_ready"] is False
    assert historical["historical"] is True
    assert historical["bound_render"] is False
    assert historical["materialization_state"] == "HISTORICAL"
    assert historical["artifact_endpoint"] == "render_execution"
    assert historical["materialization_id"] == str(materialization_id)
    assert historical["render_output_fingerprint"] == "historical-output-fingerprint"
    assert historical["artifact_available"] is True
    # Historical inspection uses the render's OWN verified artifact route.
    assert historical_direct.status_code == 200


def test_historical_render_artifact_without_valid_bytes_is_suppressed(
    engine: Engine, monkeypatch: Any
) -> None:
    """A historical transformed render with corrupt bytes offers no artifact."""

    from test_stage6_cache_validation import _complete_materialization
    from test_stage6_real_executor import _write_source

    _candidate_id, materialization_id, render_id, storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    historical_relative = "render-executions/historical/corrupt.mp4"
    historical_path = storage.storage_root / historical_relative
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    historical_path.write_bytes(b"\x00\x00\x00\x18ftypmp42historical")
    recorded_sha = hashlib.sha256(historical_path.read_bytes()).hexdigest()
    # Corrupt after recording the accepted hash.
    historical_path.write_bytes(historical_path.read_bytes() + b"corruption")
    historical_id = uuid.uuid4()
    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        session.add(
            RenderExecution(
                id=historical_id,
                source_video_id=envelope.source_video_id,
                clip_candidate_id=envelope.clip_candidate_id,
                materialization_id=envelope.id,
                artifact_purpose=RenderArtifactPurpose.TRANSFORMED_SHORT,
                lifecycle=RenderExecutionLifecycle.COMPLETE,
                cache_eligible=True,
                is_current=False,
                input_fingerprint="historical-corrupt-input",
                output_fingerprint="historical-corrupt-output",
                artifact_reference={"relative_path": historical_relative, "sha256": recorded_sha},
            )
        )
        session.commit()

    with _client() as client:
        payload = client.get("/api/render-executions?limit=50").json()
        historical_direct = client.get(f"/api/render-executions/{historical_id}/artifact")
    historical = _entry(payload, historical_id)
    assert historical["review_ready"] is False
    assert historical["artifact_available"] is False
    assert historical["artifact_endpoint"] == "render_execution"
    assert historical["artifact_sha256"] is None
    assert historical_direct.status_code == 404
    # The bound current render is unaffected.
    assert _entry(payload, render_id)["review_ready"] is True


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
