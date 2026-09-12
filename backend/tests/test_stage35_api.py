"""Stage 3.5 API, queueing, batch, manual, cancellation, and handoff tests."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.app import create_app
from app.core.enums import (
    CandidateDisposition,
    MediaOriginType,
    RefinementPriority,
    RefinementStatus,
    RightsStatus,
)
from app.core.settings import get_settings
from app.db.base import Base
from app.models import (
    AudioArtifact,
    CandidateRefinement,
    ClipCandidate,
    ProcessingJob,
    SourceVideo,
    Transcript,
)
from app.refinement.executor import build_candidate_refinement_executor
from app.services.storage import StorageService
from app.transcription.dialect import ArabicDialectProfile


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[uuid.UUID, uuid.UUID]] = []

    def dispatch(self, source_id: uuid.UUID, job_id: uuid.UUID) -> None:
        self.calls.append((source_id, job_id))


@pytest.fixture
def dispatched(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []

    def _delay(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("app.workers.tasks.run_candidate_refinement.delay", _delay)
    return calls


@pytest.fixture
def api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dispatched
) -> Iterator[tuple[TestClient, sessionmaker]]:
    monkeypatch.delenv("CLIPFACTORY_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    storage = StorageService(tmp_path / "storage")
    app = create_app(session_factory=factory, storage=storage, dispatcher=RecordingDispatcher())
    app.state.session_factory = factory
    app.state.storage = storage
    with TestClient(app) as client:
        client.session_factory = factory  # type: ignore[attr-defined]
        client.storage = storage  # type: ignore[attr-defined]
        yield client, factory


def _make_source(
    factory: sessionmaker,
    *,
    disposition: CandidateDisposition = CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
    clip_score: float = 0.8,
    is_current: bool = True,
    with_transcript: bool = True,
    source_id: uuid.UUID | None = None,
) -> tuple[uuid.UUID, uuid.UUID]:
    with factory() as session:
        if source_id is None:
            source = SourceVideo(
                source_uri="/tmp/source.mp4",
                original_filename="source.mp4",
                content_hash=f"src-{uuid.uuid4()}",
                rights_status=RightsStatus.OWNED,
                media_origin=MediaOriginType.OTHER,
            )
            session.add(source)
            session.flush()
            if with_transcript:
                session.add(
                    Transcript(
                        source_video_id=source.id,
                        language="ar",
                        whisper_model="large-v3-turbo",
                        duration=60.0,
                        raw_text="أنا عملت امبارح",
                        segments=[
                            {
                                "start": 10.0,
                                "end": 16.0,
                                "text": "أنا عملت امبارح",
                                "raw_text": "أنا عملت امبارح",
                                "words": [],
                            }
                        ],
                        dialect_profile=ArabicDialectProfile.EGYPTIAN,
                        dialect_confidence=0.9,
                        input_fingerprint="tf",
                        correction_version="v1",
                    )
                )
                session.add(
                    AudioArtifact(
                        source_video_id=source.id,
                        output_path=f"{source.id}/speech-analysis.wav",
                        content_hash="audiohash",
                        source_content_hash=source.content_hash,
                        sample_rate=16000,
                        duration=60.0,
                    )
                )
            resolved_source_id = source.id
        else:
            resolved_source_id = source_id
        candidate = ClipCandidate(
            source_video_id=resolved_source_id,
            candidate_key=f"ck-{uuid.uuid4()}",
            is_current=is_current,
            disposition=disposition,
            start_time=10.0,
            end_time=16.0,
            start_segment_index=0,
            end_segment_index=0,
            segment_indexes=[0],
            transcript_excerpt="أنا عملت امبارح",
            clip_score=clip_score,
            analysis_fingerprint="caf",
        )
        session.add(candidate)
        session.commit()
        return resolved_source_id, candidate.id


def test_queue_creates_refinement_job_and_dispatches(api, dispatched) -> None:
    client, factory = api
    _source_id, candidate_id = _make_source(factory)
    response = client.post(f"/api/candidates/{candidate_id}/refinements?priority=CANDIDATE")
    assert response.status_code == 202
    body = response.json()
    assert body["queued"] is True
    assert body["job_id"] is not None
    assert len(dispatched) == 1
    args = dispatched[0][0]
    assert args[1] == RefinementPriority.CANDIDATE.value
    with factory() as session:
        job = session.get(ProcessingJob, uuid.UUID(body["job_id"]))
        assert job.kind.value == "CANDIDATE_REFINEMENT"
        assert str(job.candidate_refinement_id) == body["refinement_id"]


def test_queue_rejects_stale_and_rejected_candidates(api, dispatched) -> None:
    client, factory = api
    _s, stale_id = _make_source(factory, is_current=False)
    _s2, rejected_id = _make_source(factory, disposition=CandidateDisposition.DO_NOT_CLIP)
    assert (
        client.post(f"/api/candidates/{stale_id}/refinements?priority=CANDIDATE").status_code == 409
    )
    assert (
        client.post(f"/api/candidates/{rejected_id}/refinements?priority=FINAL_CLIP").status_code
        == 409
    )
    assert dispatched == []


def test_queue_rejects_index_priority_and_unknown_candidate(api, dispatched) -> None:
    client, factory = api
    _s, candidate_id = _make_source(factory)
    assert (
        client.post(f"/api/candidates/{candidate_id}/refinements?priority=INDEX").status_code == 409
    )
    assert (
        client.post(f"/api/candidates/{uuid.uuid4()}/refinements?priority=CANDIDATE").status_code
        == 404
    )


def test_repeated_queue_reuses_active_job(api, dispatched) -> None:
    client, factory = api
    _s, candidate_id = _make_source(factory)
    first = client.post(f"/api/candidates/{candidate_id}/refinements?priority=CANDIDATE").json()
    second = client.post(f"/api/candidates/{candidate_id}/refinements?priority=CANDIDATE").json()
    assert first["job_id"] == second["job_id"]
    assert second["active"] is True
    assert len(dispatched) == 1


def test_matching_ready_result_returns_without_job(api, dispatched) -> None:
    client, factory = api
    _source_id, candidate_id = _make_source(factory)
    storage: StorageService = client.storage  # type: ignore[attr-defined]
    with factory() as session:
        candidate = session.get(ClipCandidate, candidate_id)
        assert candidate is not None
        fingerprint = build_candidate_refinement_executor(
            session, storage, get_settings()
        ).input_fingerprint(candidate, RefinementPriority.CANDIDATE)
        row = CandidateRefinement(
            source_video_id=candidate.source_video_id,
            clip_candidate_id=candidate.id,
            priority=RefinementPriority.CANDIDATE,
            status=RefinementStatus.CANDIDATE_REFINED,
            coarse_start=10.0,
            coarse_end=16.0,
            context_start=5.0,
            context_end=21.0,
            input_fingerprint=fingerprint,
            cache_eligible=True,
        )
        session.add(row)
        session.commit()
    response = client.post(f"/api/candidates/{candidate_id}/refinements?priority=CANDIDATE")
    body = response.json()
    assert body["cached"] is True
    assert body["job_id"] is None
    assert dispatched == []


def test_batch_is_score_ordered_and_capped(api, dispatched) -> None:
    client, factory = api
    source_id, candidate_id = _make_source(factory, clip_score=0.9)
    candidate_ids = [candidate_id]
    for score in (0.8, 0.7, 0.6):
        _sid, cid = _make_source(factory, clip_score=score, source_id=source_id)
        candidate_ids.append(cid)
    response = client.post(f"/api/sources/{source_id}/candidate-refinements/batch?limit=2")
    assert response.status_code == 202
    body = response.json()
    assert len(body) == 2
    assert len(dispatched) == 2
    with factory() as session:
        refined = [
            session.get(CandidateRefinement, uuid.UUID(item["refinement_id"])).clip_candidate_id
            for item in body
        ]
    assert refined == candidate_ids[:2]


def test_batch_default_limit_is_five_and_max_ten(api, dispatched) -> None:
    client, factory = api
    source_id, _cid = _make_source(factory, clip_score=0.9)
    for index in range(11):
        _make_source(factory, clip_score=0.9 - (index + 1) * 0.01, source_id=source_id)
    assert len(client.post(f"/api/sources/{source_id}/candidate-refinements/batch").json()) == 5
    assert (
        len(client.post(f"/api/sources/{source_id}/candidate-refinements/batch?limit=100").json())
        == 7
    )


def test_list_get_manual_and_handoff(api) -> None:
    client, factory = api
    _source_id, candidate_id = _make_source(factory)
    queued = client.post(f"/api/candidates/{candidate_id}/refinements?priority=FINAL_CLIP").json()
    refinement_id = queued["refinement_id"]

    listed = client.get(f"/api/candidates/{candidate_id}/refinements").json()
    assert [item["priority"] for item in listed] == ["FINAL_CLIP"]
    assert client.get(f"/api/refinements/{refinement_id}").status_code == 200

    manual = client.post(
        f"/api/refinements/{refinement_id}/manual",
        json={"text": "نص نهائي يدوي", "resolutions": {}},
    )
    assert manual.status_code == 200
    assert manual.json()["manual_transcript"] == "نص نهائي يدوي"
    assert manual.json()["final_transcript"] == "نص نهائي يدوي"

    handoff = client.get(f"/api/candidates/{candidate_id}/stage4-handoff").json()
    assert handoff["refinement"]["final_ready"] is False  # candidate/FINAL not yet run to readiness
    assert handoff["stage4_implemented"] is False
    assert handoff["stage3"]["clip_score"] == 0.8


def test_handoff_never_mislabels_candidate_as_final_ready(api) -> None:
    client, factory = api
    _source_id, candidate_id = _make_source(factory)
    with factory() as session:
        candidate = session.get(ClipCandidate, candidate_id)
        assert candidate is not None
        session.add(
            CandidateRefinement(
                source_video_id=candidate.source_video_id,
                clip_candidate_id=candidate.id,
                priority=RefinementPriority.CANDIDATE,
                status=RefinementStatus.CANDIDATE_REFINED,
                coarse_start=10.0,
                coarse_end=16.0,
                context_start=5.0,
                context_end=21.0,
                final_transcript="نص بجودة المرشح",
                quality_level="CANDIDATE",
                cache_eligible=True,
            )
        )
        session.commit()
    handoff = client.get(f"/api/candidates/{candidate_id}/stage4-handoff").json()
    assert handoff["refinement"]["quality_level"] == "CANDIDATE"
    assert handoff["refinement"]["final_ready"] is False


def test_job_cancellation_works_for_candidate_refinement(api) -> None:
    client, factory = api
    _source_id, candidate_id = _make_source(factory)
    queued = client.post(f"/api/candidates/{candidate_id}/refinements?priority=CANDIDATE").json()
    cancelled = client.post(f"/jobs/{queued['job_id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"


def test_manual_endpoint_rejects_unrelated_final_text(api) -> None:
    client, factory = api
    _source_id, candidate_id = _make_source(factory)
    with factory() as session:
        candidate = session.get(ClipCandidate, candidate_id)
        assert candidate is not None
        row = CandidateRefinement(
            source_video_id=candidate.source_video_id,
            clip_candidate_id=candidate.id,
            priority=RefinementPriority.FINAL_CLIP,
            status=RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW,
            coarse_start=10.0,
            coarse_end=16.0,
            context_start=5.0,
            context_end=21.0,
            automatic_transcript="أنا عملت امبارح",
            word_timestamps=[
                {"text": "أنا", "start": 10.1, "end": 10.3},
                {"text": "عملت", "start": 10.3, "end": 10.6},
                {"text": "امبارح", "start": 10.6, "end": 11.0},
            ],
        )
        session.add(row)
        session.commit()
        refinement_id = row.id

    unrelated = client.post(
        f"/api/refinements/{refinement_id}/manual",
        json={"text": "كلام مختلف تماما وغير مرتبط", "resolutions": {}},
    )
    assert unrelated.status_code == 200
    assert unrelated.json()["status"] == "NEEDS_MANUAL_TRANSCRIPT_REVIEW"

    aligned = client.post(
        f"/api/refinements/{refinement_id}/manual",
        json={"text": "أنا عملت امبارح", "resolutions": {}},
    )
    assert aligned.status_code == 200
    assert aligned.json()["status"] == "FINAL_TRANSCRIPT_READY"
