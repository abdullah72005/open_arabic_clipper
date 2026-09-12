"""Stage 3 API surface tests: provenance, queueing, listing, sanitization."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.app import create_app
from app.core.enums import (
    CandidateDisposition,
    ContentType,
    JobKind,
    MediaOriginType,
    OriginalityRisk,
    PipelineStage,
    RightsRisk,
    RightsStatus,
)
from app.db.base import Base
from app.models import ClipCandidate, ProcessingJob, SourceVideo
from app.services.storage import StorageService


class RecordingDispatcher:
    def __init__(self) -> None:
        self.job_ids: list[UUID] = []

    def dispatch(self, source_id: UUID, job_id: UUID) -> None:
        del source_id
        self.job_ids.append(job_id)


@pytest.fixture  # type: ignore[untyped-decorator]
def api(tmp_path: Path) -> Iterator[tuple[TestClient, sessionmaker[Session]]]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'stage3-api.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    app = create_app(
        session_factory=factory,
        storage=StorageService(tmp_path / "storage"),
        dispatcher=RecordingDispatcher(),
    )
    with TestClient(app) as test_client:
        yield test_client, factory
    engine.dispose()


def _make_source(factory: sessionmaker[Session], **kwargs: object) -> UUID:
    with factory() as session:
        source = SourceVideo(
            source_uri=f"/tmp/{uuid4()}.mp4",
            content_hash=f"hash-{uuid4()}",
            rights_status=kwargs.get("rights_status", RightsStatus.OWNED),
            media_origin=kwargs.get("media_origin", MediaOriginType.OTHER),
            provenance_metadata=kwargs.get("provenance_metadata", {}),
            lifecycle_state=PipelineStage.READY_FOR_ANALYSIS,
        )
        session.add(source)
        session.commit()
        return UUID(str(source.id))


def test_provenance_exposed_on_creation_and_explicit_update(
    api: tuple[TestClient, sessionmaker[Session]],
) -> None:
    test_client, _ = api
    created = test_client.post(
        "/sources/url",
        json={
            "url": "https://example.com/video",
            "rights_status": "LICENSED",
            "media_origin": "PODCAST_INTERVIEW",
            "provenance_metadata": {"creator": "Example Studio", "license": "CC-BY"},
        },
    )
    assert created.status_code == 202, created.text
    payload = created.json()
    source_id = payload["id"]
    assert payload["media_origin"] == "PODCAST_INTERVIEW"
    assert payload["provenance_metadata"]["creator"] == "Example Studio"

    updated = test_client.patch(
        f"/api/sources/{source_id}/provenance",
        json={
            "media_origin": "NEWS_CLIP",
            "provenance_metadata": {"source_reference": "https://news.example/story"},
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["media_origin"] == "NEWS_CLIP"
    assert updated.json()["provenance_metadata"]["source_reference"].startswith("https://")


def test_provenance_metadata_bounds_are_validated(
    api: tuple[TestClient, sessionmaker[Session]],
) -> None:
    test_client, _ = api
    oversized = {f"key-{index}": "value" for index in range(20)}
    response = test_client.post(
        "/sources/url",
        json={"url": "https://example.com/x", "provenance_metadata": oversized},
    )
    assert response.status_code == 422


def test_duplicate_ingest_does_not_mutate_provenance(
    api: tuple[TestClient, sessionmaker[Session]],
) -> None:
    test_client, _ = api
    first = test_client.post(
        "/sources/upload",
        data={"rights_status": "OWNED", "media_origin": "MOVIE_TV"},
        files={"file": ("clip.mp4", b"same-bytes", "video/mp4")},
    )
    assert first.status_code == 201, first.text
    duplicate = test_client.post(
        "/sources/upload",
        data={"rights_status": "THIRD_PARTY_UNKNOWN", "media_origin": "OTHER"},
        files={"file": ("clip.mp4", b"same-bytes", "video/mp4")},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["id"] == first.json()["id"]
    assert duplicate.json()["rights_status"] == "OWNED"
    assert duplicate.json()["media_origin"] == "MOVIE_TV"


def test_candidate_analysis_queue_creates_job(
    api: tuple[TestClient, sessionmaker[Session]], monkeypatch: pytest.MonkeyPatch
) -> None:
    test_client, factory = api
    source_id = _make_source(factory)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "app.workers.tasks.run_pipeline_stage.delay", lambda *args: calls.append(args)
    )
    response = test_client.post(f"/api/sources/{source_id}/candidate-analysis?force=true")
    assert response.status_code == 202, response.text
    assert response.json()["kind"] == "CANDIDATE_ANALYSIS"
    assert calls and calls[0][1] == PipelineStage.CANDIDATE_ANALYSIS.value
    assert calls[0][3] is True


def test_candidates_pagination_and_rejected_filtering(
    api: tuple[TestClient, sessionmaker[Session]],
) -> None:
    test_client, factory = api
    source_id = _make_source(factory)
    with factory() as session:
        for index, disposition in enumerate(
            [
                CandidateDisposition.CANDIDATE,
                CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
                CandidateDisposition.DO_NOT_CLIP,
                CandidateDisposition.DO_NOT_CLIP_RECENTLY_REDUNDANT,
            ]
        ):
            session.add(
                ClipCandidate(
                    source_video_id=source_id,
                    candidate_key=f"key-{index}",
                    disposition=disposition,
                    start_time=index * 30.0,
                    end_time=index * 30.0 + 20.0,
                    start_segment_index=index,
                    end_segment_index=index + 1,
                    segment_indexes=[index, index + 1],
                    primary_content_type=ContentType.STORY,
                    rights_risk=RightsRisk.LOW,
                    originality_risk=OriginalityRisk.NOT_INDICATED,
                    clip_score=1.0 - index * 0.1,
                )
            )
        session.commit()

    accepted = test_client.get(f"/api/sources/{source_id}/candidates")
    assert accepted.status_code == 200
    assert {item["disposition"] for item in accepted.json()} == {
        "CANDIDATE",
        "CANDIDATE_NEEDS_REFINEMENT",
    }

    all_items = test_client.get(
        f"/api/sources/{source_id}/candidates", params={"include_rejected": True}
    )
    assert len(all_items.json()) == 4

    limited = test_client.get(
        f"/api/sources/{source_id}/candidates",
        params={"include_rejected": True, "limit": 1, "offset": 1},
    )
    assert len(limited.json()) == 1


def test_candidate_response_sanitizes_provider_evidence(
    api: tuple[TestClient, sessionmaker[Session]],
) -> None:
    test_client, factory = api
    source_id = _make_source(factory)
    with factory() as session:
        candidate = ClipCandidate(
            source_video_id=source_id,
            candidate_key="secret-key",
            disposition=CandidateDisposition.CANDIDATE,
            start_time=0.0,
            end_time=20.0,
            start_segment_index=0,
            end_segment_index=1,
            segment_indexes=[0, 1],
            primary_content_type=ContentType.STORY,
            rights_risk=RightsRisk.LOW,
            originality_risk=OriginalityRisk.NOT_INDICATED,
            provider_evidence={"api_key": "leaked", "summary": "ok"},
            hooks=[{"type": "QUESTION", "text": "ايه ده؟"}],
            refinement_reasons=["LOW_CONFIDENCE_WORD_SPAN"],
        )
        session.add(candidate)
        session.commit()
        candidate_id = candidate.id

    response = test_client.get(f"/api/candidates/{candidate_id}")
    assert response.status_code == 200
    body = response.json()
    assert "api_key" not in body["provider_evidence"]
    assert body["provider_evidence"]["summary"] == "ok"
    assert body["refinement_reasons"] == ["LOW_CONFIDENCE_WORD_SPAN"]
    assert body["hooks"][0]["type"] == "QUESTION"

    missing = test_client.get(f"/api/candidates/{uuid4()}")
    assert missing.status_code == 404


def test_candidate_analysis_summary_is_exposed(
    api: tuple[TestClient, sessionmaker[Session]],
) -> None:
    test_client, factory = api
    source_id = _make_source(factory)
    missing = test_client.get(f"/api/sources/{source_id}/candidate-analysis")
    assert missing.status_code == 404

    with factory() as session:
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.CANDIDATE_ANALYSIS)
        session.add(job)
        session.commit()
    assert job.kind is JobKind.CANDIDATE_ANALYSIS
