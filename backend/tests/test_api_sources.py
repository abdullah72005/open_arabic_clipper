from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.app import CeleryDispatcher, create_app
from app.core.enums import PipelineStage, RightsStatus
from app.db.base import Base
from app.models import SourceVideo, Transcript
from app.services.storage import StorageService


class RecordingDispatcher:
    def __init__(self) -> None:
        self.job_ids: list[UUID] = []

    def dispatch(self, source_id: UUID, job_id: UUID) -> None:
        del source_id
        self.job_ids.append(job_id)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, RecordingDispatcher]]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.sqlite3'}")
    Base.metadata.create_all(engine)
    dispatcher = RecordingDispatcher()
    factory = sessionmaker(engine, expire_on_commit=False)
    app = create_app(
        session_factory=factory,
        storage=StorageService(tmp_path / "storage"),
        dispatcher=dispatcher,
    )
    app.state.session_factory = factory
    with TestClient(app) as test_client:
        yield test_client, dispatcher
    engine.dispose()


def test_upload_rejects_empty_and_oversized_files(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    test_client, _ = client
    empty = test_client.post("/sources/upload", files={"file": ("empty.mp4", b"")})
    assert empty.status_code == 422

    large = test_client.post("/sources/upload", files={"file": ("large.mp4", b"x" * 32)})
    assert large.status_code == 201


def test_upload_rejects_content_over_configured_limit(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'limited.sqlite3'}")
    Base.metadata.create_all(engine)
    app = create_app(
        session_factory=sessionmaker(engine, expire_on_commit=False),
        storage=StorageService(tmp_path / "storage"),
        dispatcher=RecordingDispatcher(),
        max_upload_bytes=4,
    )
    with TestClient(app) as test_client:
        response = test_client.post("/sources/upload", files={"file": ("large.mp4", b"video")})
    engine.dispose()

    assert response.status_code == 413


def test_upload_creates_source_and_schedules_job_without_executing_it(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    test_client, dispatcher = client
    response = test_client.post("/sources/upload", files={"file": ("clip.mp4", b"video")})

    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "clip.mp4"
    assert body["lifecycle_state"] == "INGEST"
    assert len(dispatcher.job_ids) == 1


def test_celery_dispatcher_sends_the_source_id_to_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    source_id = UUID("0d9f0117-739f-4f34-b0cf-b3d0f1f5ebd1")
    job_id = UUID("5c2a9a34-4d2c-4e01-8ca1-5090cfb4906c")

    monkeypatch.setattr(
        "app.workers.tasks.run_pipeline_stage.delay", lambda *args: calls.append(args)
    )

    CeleryDispatcher().dispatch(source_id, job_id)

    assert calls == [(str(source_id), PipelineStage.INGEST.value, str(job_id))]


def test_upload_reads_the_request_file_in_bounded_chunks(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    test_client, _ = client
    route = next(
        route
        for route in test_client.app.routes
        if getattr(route, "path", None) == "/sources/upload"
    )
    reader = _BoundedReader(b"video-data")
    fake_upload = SimpleNamespace(filename="streamed.mp4", file=reader)
    session_factory = test_client.app.state.session_factory

    with session_factory() as database:
        response = route.endpoint(  # type: ignore[union-attr]
            Response(), fake_upload, RightsStatus.OWNED, database
        )

    assert response.rights_status is RightsStatus.OWNED
    assert reader.read_sizes
    assert all(size <= 1024 * 1024 for size in reader.read_sizes)


class _BoundedReader:
    def __init__(self, content: bytes) -> None:
        self._content = content
        self._offset = 0
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        if size <= 0 or size > 1024 * 1024:
            raise AssertionError("uploads must be read in bounded chunks")
        self.read_sizes.append(size)
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def test_duplicate_upload_returns_existing_source_without_scheduling_again(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    test_client, dispatcher = client
    first = test_client.post("/sources/upload", files={"file": ("clip.mp4", b"video")})
    duplicate = test_client.post("/sources/upload", files={"file": ("again.mp4", b"video")})

    assert first.status_code == 201
    assert duplicate.status_code == 200
    assert duplicate.json()["id"] == first.json()["id"]
    assert len(dispatcher.job_ids) == 1


def test_url_requires_valid_public_http_url(client: tuple[TestClient, RecordingDispatcher]) -> None:
    test_client, _ = client
    response = test_client.post("/sources/url", json={"url": "file:///private/video.mp4"})
    assert response.status_code == 422


def test_upload_persists_operator_selected_rights_status(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    test_client, _ = client

    response = test_client.post(
        "/sources/upload",
        data={"rights_status": "LICENSED"},
        files={"file": ("licensed.mp4", b"video")},
    )

    assert response.status_code == 201
    assert response.json()["rights_status"] == "LICENSED"


def test_url_accepts_all_declared_operator_rights_statuses(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    test_client, _ = client

    response = test_client.post(
        "/sources/url",
        json={"url": "https://example.com/licensed-video", "rights_status": "OTHER_ALLOWED"},
    )

    assert response.status_code == 202
    assert response.json()["rights_status"] == "OTHER_ALLOWED"


def test_delete_rejects_sources_with_active_jobs(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    test_client, _ = client
    source = test_client.post("/sources/upload", files={"file": ("clip.mp4", b"video")}).json()
    response = test_client.delete(f"/sources/{source['id']}")
    assert response.status_code == 409


def test_cancelled_job_is_not_dispatched_again(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    test_client, dispatcher = client
    source = test_client.post("/sources/upload", files={"file": ("clip.mp4", b"video")}).json()
    jobs = test_client.get("/jobs").json()
    response = test_client.post(f"/jobs/{jobs[0]['id']}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "CANCELLED"
    assert len(dispatcher.job_ids) == 1
    assert test_client.post(f"/sources/{source['id']}/process").status_code == 202
    assert len(dispatcher.job_ids) == 2


def test_transcript_search_returns_timestamped_mixed_language_segment(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    """API search keeps Arabic/English source text and its seek position."""

    test_client, _ = client
    factory = test_client.app.state.session_factory
    with factory() as session:
        source = SourceVideo(source_uri="/imports/episode.mp4")
        session.add(source)
        session.flush()
        session.add(
            Transcript(
                source_video_id=source.id,
                whisper_model="small",
                transcription_options={},
                input_fingerprint="a" * 64,
                raw_text="أهلا hello",
                normalized_text="أهلا hello",
                segments=[{"start": 12.4, "end": 14.0, "text": "أهلا hello", "words": []}],
                word_segments=[],
                duration=14.0,
                language="ar",
            )
        )
        session.commit()
        source_id = source.id

    response = test_client.get(f"/api/sources/{source_id}/transcript/search?q=hello")

    assert response.status_code == 200
    assert response.json()["segments"][0]["start"] == 12.4


def test_retranscribe_queues_a_transcription_job(
    client: tuple[TestClient, RecordingDispatcher], monkeypatch: pytest.MonkeyPatch
) -> None:
    test_client, _ = client
    source = test_client.post("/sources/upload", files={"file": ("clip.mp4", b"video")}).json()
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "app.workers.tasks.run_pipeline_stage.delay", lambda *args: calls.append(args)
    )

    response = test_client.post(f"/api/sources/{source['id']}/retranscribe")

    assert response.status_code == 202
    assert response.json()["kind"] == "TRANSCRIPTION"
    assert calls[0][0] == source["id"]
    assert calls[0][1] == "TRANSCRIPTION"
