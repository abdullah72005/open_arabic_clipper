from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
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


def test_source_media_returns_stored_local_upload(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    test_client, _ = client

    created = test_client.post(
        "/sources/upload",
        data={"rights_status": RightsStatus.OWNED.value},
        files={"file": ("owned-video.mp4", b"owned media", "video/mp4")},
    )

    assert created.status_code == 201
    source_id = created.json()["id"]
    media = test_client.get(f"/api/sources/{source_id}/media")

    assert media.status_code == 200, media.text
    assert media.content == b"owned media"
    assert media.headers["content-type"].startswith("video/mp4")


def test_source_media_uses_configured_storage_when_not_injected(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'default-storage.sqlite3'}")
    Base.metadata.create_all(engine)
    app = create_app(
        session_factory=sessionmaker(engine, expire_on_commit=False),
        dispatcher=RecordingDispatcher(),
    )

    with TestClient(app) as test_client:
        created = test_client.post(
            "/sources/upload",
            files={"file": ("owned-video.mp4", b"owned media", "video/mp4")},
        )
        media = test_client.get(f"/api/sources/{created.json()['id']}/media")

    engine.dispose()
    assert media.status_code == 200, media.text


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


def test_operator_override_preserves_raw_correction_and_timestamp(
    client: tuple[TestClient, RecordingDispatcher],
) -> None:
    """Manual feedback changes only final display text and remains available for evaluation."""

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
                input_fingerprint="o" * 64,
                raw_text="خطي بالك",
                normalized_text="خلي بالك",
                corrected_text="خلي بالك",
                final_text="خلي بالك",
                segments=[
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "خطي بالك",
                        "raw_text": "خطي بالك",
                        "corrected_text": "خلي بالك",
                        "final_text": "خلي بالك",
                        "correction_applied": True,
                        "correction_confidence": 0.97,
                        "correction_method": "lexicon",
                        "correction_version": "egyptian-ar-v1",
                        "contextual_reconstructed_text": "خلي بالك يا صاحبي",
                        "reconstruction_applied": True,
                        "reconstruction_confidence": 0.93,
                        "reconstruction_confidence_level": "HIGH",
                        "words": [],
                    }
                ],
                word_segments=[],
                duration=1.0,
            )
        )
        session.commit()
        source_id = source.id

    response = test_client.post(
        f"/api/sources/{source_id}/transcript/segments/0/override",
        json={"text": "خلي بالك يا أحمد"},
    )

    assert response.status_code == 200
    assert response.json()["raw_text"] == "خطي بالك"
    assert response.json()["corrected_text"] == "خلي بالك"
    assert response.json()["final_text"] == "خلي بالك يا أحمد"
    assert response.json()["start"] == 0.0
    assert response.json()["end"] == 1.0
    assert test_client.get(f"/api/sources/{source_id}/transcript/search?q=أحمد").json()["segments"]

    cleared = test_client.delete(f"/api/sources/{source_id}/transcript/segments/0/override")

    assert cleared.status_code == 200
    assert cleared.json()["final_text"] == "خلي بالك يا صاحبي"


def test_reconstruct_queues_independent_job_and_force_only_clears_its_cache(
    client: tuple[TestClient, RecordingDispatcher], monkeypatch: pytest.MonkeyPatch
) -> None:
    test_client, _ = client
    source = test_client.post("/sources/upload", files={"file": ("clip.mp4", b"video")}).json()
    factory = test_client.app.state.session_factory
    with factory() as session:
        session.add(
            Transcript(
                source_video_id=UUID(source["id"]),
                whisper_model="small",
                input_fingerprint="asr-fingerprint",
                reconstruction_fingerprint="reconstruction-fingerprint",
                raw_text="cached",
                normalized_text="cached",
                segments=[],
                word_segments=[],
            )
        )
        session.commit()
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "app.workers.tasks.run_pipeline_stage.delay", lambda *args: calls.append(args)
    )

    response = test_client.post(f"/api/sources/{source['id']}/reconstruct?force=true")

    assert response.status_code == 202
    assert response.json()["kind"] == "RECONSTRUCTION"
    assert calls[0][0:2] == (source["id"], "CONTEXTUAL_RECONSTRUCTION")
    assert calls[0][3] is True
    with factory() as session:
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == UUID(source["id"]))
        )
        assert transcript is not None
        assert transcript.input_fingerprint == "asr-fingerprint"
        assert transcript.reconstruction_fingerprint == ""

    transcript_response = test_client.get(f"/api/sources/{source['id']}/transcript")
    assert transcript_response.status_code == 200
    assert transcript_response.json()["reconstruction_method"] == "pending"
    assert transcript_response.json()["contextual_reconstructed_text"] == ""


def test_retranscribe_queues_a_transcription_job(
    client: tuple[TestClient, RecordingDispatcher], monkeypatch: pytest.MonkeyPatch
) -> None:
    test_client, _ = client
    source = test_client.post("/sources/upload", files={"file": ("clip.mp4", b"video")}).json()
    factory = test_client.app.state.session_factory
    with factory() as session:
        session.add(
            Transcript(
                source_video_id=UUID(source["id"]),
                whisper_model="small",
                input_fingerprint="cached",
                raw_text="cached",
                normalized_text="cached",
                segments=[],
                word_segments=[],
            )
        )
        session.commit()
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "app.workers.tasks.run_pipeline_stage.delay", lambda *args: calls.append(args)
    )

    response = test_client.post(f"/api/sources/{source['id']}/retranscribe?force=true")

    assert response.status_code == 202
    assert response.json()["kind"] == "TRANSCRIPTION"
    assert calls[0][0] == source["id"]
    assert calls[0][1] == "TRANSCRIPTION"
    with factory() as session:
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == UUID(source["id"]))
        )
        assert transcript is not None
        assert transcript.input_fingerprint == ""
