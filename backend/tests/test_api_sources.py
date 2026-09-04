from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.app import create_app
from app.db.base import Base
from app.services.storage import StorageService


class RecordingDispatcher:
    def __init__(self) -> None:
        self.job_ids: list[UUID] = []

    def dispatch(self, job_id: UUID) -> None:
        self.job_ids.append(job_id)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, RecordingDispatcher]]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.sqlite3'}")
    Base.metadata.create_all(engine)
    dispatcher = RecordingDispatcher()
    app = create_app(
        session_factory=sessionmaker(engine, expire_on_commit=False),
        storage=StorageService(tmp_path / "storage"),
        dispatcher=dispatcher,
    )
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
        response = test_client.post(
            "/sources/upload", files={"file": ("large.mp4", b"video")}
        )
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
