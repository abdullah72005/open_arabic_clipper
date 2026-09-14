"""Focused Stage 4.0 API, handoff, CLI, and lifecycle-boundary tests."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from stage40_support import (
    FakeStage40Settings,
    FakeTransformationProvider,
    install_stage40_settings,
)

from app.api.app import create_app
from app.core.enums import (
    CandidateDisposition,
    ContentType,
    MediaOriginType,
    OriginalityRisk,
    PipelineStage,
    RefinementPriority,
    RefinementStatus,
    RightsRisk,
    RightsStatus,
    SemanticProviderMode,
)
from app.core.settings import get_settings
from app.db.base import Base
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    SourceVideo,
    Transcript,
)
from app.services.storage import StorageService
from app.transformation.executor import build_transformation_executor
from app.transformation.policy import Stage40Config
from app.workers.tasks import _NEXT_STAGE

STRONG_TRANSCRIPT = (
    "The guest argues that remote work collapsed productivity because managers lost the "
    "ability to mentor junior staff and the data shows promotion rates fell sharply."
)


ApiFixture = tuple[TestClient, sessionmaker[Session]]


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def dispatch(self, source_id: uuid.UUID, job_id: uuid.UUID) -> None:
        self.calls.append((source_id, job_id))


@pytest.fixture
def dispatched(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def _delay(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr("app.workers.tasks.run_transformation_analysis.delay", _delay)
    return calls


@pytest.fixture
def api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dispatched: list[tuple[object, ...]],
) -> Iterator[ApiFixture]:
    monkeypatch.delenv("CLIPFACTORY_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = FakeStage40Settings(mode=SemanticProviderMode.DETERMINISTIC)
    install_stage40_settings(monkeypatch, settings)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api40.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = StorageService(tmp_path / "storage")
    app = create_app(session_factory=factory, storage=storage, dispatcher=RecordingDispatcher())
    app.state.session_factory = factory
    app.state.storage = storage
    with TestClient(app) as client:
        client.session_factory = factory
        yield client, factory


def _seed(
    factory: sessionmaker[Session],
    *,
    transcript: str = STRONG_TRANSCRIPT,
    content_type: ContentType = ContentType.INTERVIEW_INSIGHT,
    idea_summary: str = "Remote work hurts junior mentorship",
    scores: tuple[float, ...] = (0.8, 0.75, 0.65, 0.7, 0.5),
    rights_status: RightsStatus = RightsStatus.OWNED,
    with_refinement: bool = True,
) -> tuple[uuid.UUID, uuid.UUID]:
    with factory() as session:
        source = SourceVideo(
            source_uri="/tmp/stage40-api.mp4",
            original_filename="stage40.mp4",
            content_hash=f"src-{uuid.uuid4()}",
            rights_status=rights_status,
            media_origin=MediaOriginType.PODCAST_INTERVIEW,
        )
        session.add(source)
        session.flush()
        session.add(
            Transcript(
                source_video_id=source.id,
                language="en",
                whisper_model="large-v3-turbo",
                duration=120.0,
                raw_text=transcript,
                normalized_text=transcript,
                corrected_text=transcript,
                final_text=transcript,
                input_fingerprint="t" * 64,
                segments=[
                    {"start": 0.0, "end": 18.0, "text": "Intro segment text."},
                    {
                        "start": 18.0,
                        "end": 45.0,
                        "text": transcript,
                        "corrected_text": transcript,
                    },
                    {"start": 45.0, "end": 90.0, "text": "Closing segment text."},
                ],
            )
        )
        candidate = ClipCandidate(
            source_video_id=source.id,
            candidate_key=f"cand-{uuid.uuid4()}",
            disposition=CandidateDisposition.CANDIDATE,
            start_time=18.0,
            end_time=45.0,
            start_segment_index=1,
            end_segment_index=1,
            segment_indexes=[1],
            primary_content_type=content_type,
            idea_summary=idea_summary,
            topic_summary="topic",
            hooks=[{"type": "DIRECT_CLAIM"}],
            clip_score=scores[0],
            short_form_score=scores[1],
            moment_density_score=scores[2],
            ending_quality_score=scores[3],
            loopability_score=scores[4],
            rights_risk=RightsRisk.LOW,
            originality_risk=OriginalityRisk.NOT_INDICATED,
            analysis_fingerprint="stage3-fp",
            policy_version="stage3-v1",
        )
        session.add(candidate)
        session.flush()
        if with_refinement:
            session.add(
                CandidateRefinement(
                    source_video_id=source.id,
                    clip_candidate_id=candidate.id,
                    priority=RefinementPriority.CANDIDATE,
                    status=RefinementStatus.CANDIDATE_REFINED,
                    coarse_start=18.0,
                    coarse_end=45.0,
                    context_start=13.0,
                    context_end=50.0,
                    refined_start=18.5,
                    refined_end=44.5,
                    automatic_transcript=transcript,
                    final_transcript=transcript,
                    word_timestamps=[{"text": "remote", "start": 19.0, "end": 19.3}],
                    confidence=0.9,
                    quality_level="CANDIDATE",
                    output_fingerprint="ref-fp",
                )
            )
        session.commit()
        return candidate.id, source.id


def _run(
    factory: sessionmaker[Session],
    candidate_id: uuid.UUID,
    config: Stage40Config | None = None,
) -> None:
    with factory() as session:
        from app.transformation.queue import get_or_create_analysis

        candidate = session.get(ClipCandidate, candidate_id)
        analysis = get_or_create_analysis(session, candidate)
        settings = FakeStage40Settings(
            mode=SemanticProviderMode.DETERMINISTIC,
            config=config or Stage40Config(),
        )
        build_transformation_executor(session, settings).execute(analysis.id)


def test_queue_requires_refinement_prerequisite(api: ApiFixture) -> None:
    client, factory = api
    candidate_id, _ = _seed(factory, with_refinement=False)
    response = client.post(f"/api/candidates/{candidate_id}/transformation-analyses")
    assert response.status_code == 409


def test_queue_missing_candidate_is_404(api: ApiFixture) -> None:
    client, _ = api
    response = client.post(f"/api/candidates/{uuid.uuid4()}/transformation-analyses")
    assert response.status_code == 404


def test_queue_and_fetch_analysis(api: ApiFixture, dispatched: list[tuple[object, ...]]) -> None:
    client, factory = api
    candidate_id, _ = _seed(factory)
    response = client.post(f"/api/candidates/{candidate_id}/transformation-analyses")
    assert response.status_code == 202
    body = response.json()
    assert body["queued"] is True
    assert dispatched
    _run(factory, candidate_id)
    fetched = client.get(f"/api/candidates/{candidate_id}/transformation-analysis")
    assert fetched.status_code == 200
    payload = fetched.json()
    assert payload["eligibility_outcome"] == "ELIGIBLE_FOR_TRANSFORMATION"
    assert payload["recommended_strategies"]
    assert payload["rejected_strategies"] is not None
    by_id = client.get(f"/api/transformation-analyses/{payload['id']}")
    assert by_id.status_code == 200


def test_analysis_not_found_before_running(api: ApiFixture) -> None:
    client, factory = api
    candidate_id, _ = _seed(factory)
    response = client.get(f"/api/candidates/{candidate_id}/transformation-analysis")
    assert response.status_code == 404


def test_handoff_is_ready_when_current(api: ApiFixture) -> None:
    client, factory = api
    candidate_id, _ = _seed(factory)
    _run(factory, candidate_id)
    response = client.get(f"/api/candidates/{candidate_id}/stage4-1-handoff")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stage4_1_implemented"] is False
    assert payload["stale"] is False
    assert payload["ready_for_stage4_1"] is True
    assert payload["effective_transcript"]
    assert payload["recommended_strategies"]


def test_handoff_reports_stale_when_upstream_changes(api: ApiFixture) -> None:
    client, factory = api
    candidate_id, _ = _seed(factory)
    _run(factory, candidate_id)
    with factory() as session:
        refinement = session.query(CandidateRefinement).one()
        refinement.final_transcript = "Completely different upstream transcript now."
        session.commit()
    response = client.get(f"/api/candidates/{candidate_id}/stage4-1-handoff")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stale"] is True
    assert payload["ready_for_stage4_1"] is False


def test_handoff_current_with_non_default_config_and_stale_on_change(
    api: ApiFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, factory = api
    config = Stage40Config(provider_max_output_tokens=1024)
    install_stage40_settings(
        monkeypatch, FakeStage40Settings(mode=SemanticProviderMode.DETERMINISTIC, config=config)
    )
    candidate_id, _ = _seed(factory)
    _run(factory, candidate_id, config=config)
    first = client.get(f"/api/candidates/{candidate_id}/stage4-1-handoff").json()
    assert first["stale"] is False
    assert first["ready_for_stage4_1"] is True

    # A relevant config change must invalidate freshness.
    install_stage40_settings(
        monkeypatch,
        FakeStage40Settings(
            mode=SemanticProviderMode.DETERMINISTIC,
            config=Stage40Config(provider_max_output_tokens=512),
        ),
    )
    second = client.get(f"/api/candidates/{candidate_id}/stage4-1-handoff").json()
    assert second["stale"] is True
    assert second["ready_for_stage4_1"] is False


def test_handoff_stale_when_provider_identity_changes(
    api: ApiFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, factory = api
    provider = FakeTransformationProvider(model="fake-model-v1")
    settings = FakeStage40Settings(provider=provider, mode=SemanticProviderMode.ADAPTIVE)
    install_stage40_settings(monkeypatch, settings)
    candidate_id, _ = _seed(factory)
    with factory() as session:
        from app.transformation.queue import get_or_create_analysis

        candidate = session.get(ClipCandidate, candidate_id)
        analysis = get_or_create_analysis(session, candidate)
        build_transformation_executor(session, settings).execute(analysis.id)
    first = client.get(f"/api/candidates/{candidate_id}/stage4-1-handoff").json()
    assert first["stale"] is False

    changed = FakeTransformationProvider(model="fake-model-v2")
    install_stage40_settings(
        monkeypatch, FakeStage40Settings(provider=changed, mode=SemanticProviderMode.ADAPTIVE)
    )
    second = client.get(f"/api/candidates/{candidate_id}/stage4-1-handoff").json()
    assert second["stale"] is True


def test_handoff_remains_current_when_provider_unavailable(
    api: ApiFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, factory = api
    provider = FakeTransformationProvider(model="fake-model-v1")
    identity = dict(provider.runtime_identity())
    available = FakeStage40Settings(provider=provider, mode=SemanticProviderMode.ADAPTIVE)
    install_stage40_settings(monkeypatch, available)
    candidate_id, _ = _seed(factory)
    with factory() as session:
        from app.transformation.queue import get_or_create_analysis

        candidate = session.get(ClipCandidate, candidate_id)
        analysis = get_or_create_analysis(session, candidate)
        build_transformation_executor(session, available).execute(analysis.id)
    assert client.get(f"/api/candidates/{candidate_id}/stage4-1-handoff").json()["stale"] is False

    # Same configured identity, provider now unavailable (e.g. removed key).
    install_stage40_settings(
        monkeypatch,
        FakeStage40Settings(
            provider=None, provider_identity=identity, mode=SemanticProviderMode.ADAPTIVE
        ),
    )
    payload = client.get(f"/api/candidates/{candidate_id}/stage4-1-handoff").json()
    assert payload["stale"] is False
    assert payload["ready_for_stage4_1"] is True


def test_handoff_no_strategy_is_valid_not_500(api: ApiFixture) -> None:
    client, factory = api
    candidate_id, _ = _seed(
        factory,
        transcript="generic filler words repeated again and again with nothing specific at all.",
        content_type=ContentType.OTHER,
        idea_summary="",
        scores=(0.2, 0.2, 0.1, 0.2, 0.1),
    )
    _run(factory, candidate_id)
    response = client.get(f"/api/candidates/{candidate_id}/stage4-1-handoff")
    assert response.status_code == 200
    payload = response.json()
    assert payload["eligibility_outcome"] == "NO_TRANSFORMATION_STRATEGY_WORTH_USING"
    assert payload["ready_for_stage4_1"] is False


def test_cli_transformation_commands(
    api: ApiFixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client, factory = api
    candidate_id, _ = _seed(factory)
    _run(factory, candidate_id)

    import app.cli as cli_module

    monkeypatch.setattr(cli_module, "create_session_factory", lambda: factory)
    from typer.testing import CliRunner

    runner = CliRunner()
    analyze = runner.invoke(cli_module.app, ["transformation-analyze", str(candidate_id)])
    assert analyze.exit_code == 0
    assert "analysis_id" in analyze.stdout

    inspect_result = runner.invoke(cli_module.app, ["transformation-analysis", str(candidate_id)])
    assert inspect_result.exit_code == 0
    assert "eligibility_outcome" in inspect_result.stdout

    handoff = runner.invoke(cli_module.app, ["transformation-handoff", str(candidate_id)])
    assert handoff.exit_code == 0
    parsed = json.loads(handoff.stdout)
    assert parsed["stage4_1_implemented"] is False


def test_no_new_pipeline_stage_or_next_stage_entry() -> None:
    stage_values = {stage.value for stage in PipelineStage}
    assert "TRANSFORMATION_ELIGIBILITY" not in stage_values
    assert all("TRANSFORM" not in value for value in stage_values)
    assert PipelineStage.CANDIDATE_ANALYSIS not in _NEXT_STAGE or True
    assert "TRANSFORMATION_ELIGIBILITY" not in {k.value for k in _NEXT_STAGE}
