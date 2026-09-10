"""Stage 2.7.1 persistence, API, and migration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.enums import RightsStatus
from app.db.base import Base
from app.models import SourceVideo, Transcript
from app.transcription.dialect import ArabicDialectProfile

_EGYPTIAN = ArabicDialectProfile.EGYPTIAN
_SAUDI = ArabicDialectProfile.SAUDI
_UNKNOWN = ArabicDialectProfile.UNKNOWN_ARABIC


def _transcript_factory(source_id: Any) -> Transcript:
    return Transcript(
        source_video_id=source_id,
        whisper_model="large-v3-turbo",
        input_fingerprint="asr-fp",
        raw_text="كلام",
        corrected_text="كلام",
        final_text="كلام",
        segments=[],
        word_segments=[],
        duration=1.0,
    )


def test_models_declare_dialect_and_code_switch_columns() -> None:
    assert "dialect_profile_override" in SourceVideo.__table__.c
    assert "dialect_profile" in Transcript.__table__.c
    assert "dialect_confidence" in Transcript.__table__.c
    assert "dialect_evidence" in Transcript.__table__.c
    assert "code_switch_suspected" in Transcript.__table__.c


def test_transcript_confidence_check_constraint_exists() -> None:
    constraints = {
        constraint.name for constraint in Transcript.__table__.constraints if constraint.name
    }
    assert "ck_transcripts_dialect_confidence_bounds" in constraints


def test_dialect_enum_values_are_exact() -> None:
    assert [profile.value for profile in ArabicDialectProfile] == [
        "EGYPTIAN",
        "SAUDI",
        "GULF",
        "LEVANTINE",
        "MSA",
        "UNKNOWN_ARABIC",
    ]


def test_source_override_and_transcript_fields_persist(sqlite_engine: Any) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri="file:///tmp/d.mp4",
            content_hash="h",
            rights_status=RightsStatus.OWNED,
            dialect_profile_override=_SAUDI,
        )
        session.add(source)
        session.commit()
        transcript = _transcript_factory(source.id)
        transcript.dialect_profile = _UNKNOWN
        transcript.dialect_confidence = 0.4
        transcript.dialect_evidence = {"selection_method": "unknown"}
        transcript.code_switch_suspected = True
        session.add(transcript)
        session.commit()
        session.expire_all()

        loaded_source = session.get(SourceVideo, source.id)
        loaded = session.get(Transcript, transcript.id)
        assert loaded_source.dialect_profile_override is _SAUDI
        assert loaded.dialect_profile is _UNKNOWN
        assert loaded.dialect_confidence == 0.4
        assert loaded.dialect_evidence == {"selection_method": "unknown"}
        assert loaded.code_switch_suspected is True


def test_dialect_confidence_constraint_rejects_out_of_bounds(sqlite_engine: Any) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="file:///tmp/d.mp4", content_hash="h")
        session.add(source)
        session.commit()
        transcript = _transcript_factory(source.id)
        transcript.dialect_confidence = 1.4
        session.add(transcript)
        with pytest.raises(Exception):
            session.commit()


def test_migration_upgrades_and_downgrades_cleanly(tmp_path: Path) -> None:
    from alembic.config import Config

    from alembic import command

    config = Config()
    config.set_main_option("script_location", str(Path(__file__).resolve().parents[1] / "alembic"))

    command.upgrade(config, "20260906_0009")
    command.upgrade(config, "20260910_0010")
    command.downgrade(config, "20260906_0009")
    command.upgrade(config, "head")


def test_migration_backfills_arabic_rows_conservatively(tmp_path: Path) -> None:
    from alembic.config import Config
    from sqlalchemy import create_engine, text

    from alembic import command
    from app.core.settings import get_settings

    config = Config()
    config.set_main_option("script_location", str(Path(__file__).resolve().parents[1] / "alembic"))
    command.upgrade(config, "20260906_0009")

    engine = create_engine(get_settings().database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO source_videos (id, source_uri, content_hash, rights_status, "
                "lifecycle_state, created_at, updated_at) VALUES "
                "('00000000-0000-0000-0000-000000000001', 'file:///a.mp4', 'h1', 'UNKNOWN', "
                "'INGEST', datetime('now'), datetime('now'))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO transcripts (id, source_video_id, language, whisper_model, "
                "transcription_options, input_fingerprint, raw_text, normalized_text, "
                "segments, word_segments, duration, created_at, updated_at) VALUES "
                "('00000000-0000-0000-0000-000000000002', "
                "'00000000-0000-0000-0000-000000000001', 'ar', 'large-v3-turbo', '{}', 'fp', "
                "'كلام', 'كلام', '[]', '[]', 1.0, datetime('now'), datetime('now'))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO source_videos (id, source_uri, content_hash, rights_status, "
                "lifecycle_state, created_at, updated_at) VALUES "
                "('00000000-0000-0000-0000-000000000002', 'file:///b.mp4', 'h2', 'UNKNOWN', "
                "'INGEST', datetime('now'), datetime('now'))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO transcripts (id, source_video_id, language, whisper_model, "
                "transcription_options, input_fingerprint, raw_text, normalized_text, "
                "segments, word_segments, duration, created_at, updated_at) VALUES "
                "('00000000-0000-0000-0000-000000000003', "
                "'00000000-0000-0000-0000-000000000002', 'en', 'large-v3-turbo', '{}', 'fp', "
                "'hello', 'hello', '[]', '[]', 1.0, datetime('now'), datetime('now'))"
            )
        )
    engine.dispose()

    command.upgrade(config, "head")

    engine = create_engine(get_settings().database_url)
    with engine.connect() as connection:
        result = connection.execute(
            text(
                "SELECT dialect_profile, dialect_confidence, dialect_evidence, "
                "code_switch_suspected FROM transcripts ORDER BY id"
            )
        ).fetchall()
    engine.dispose()
    assert result[0][0] == "UNKNOWN_ARABIC"
    assert result[0][1] == 0
    assert result[0][2] in ({}, "{}")
    assert bool(result[0][3]) is False
    assert result[1][0] is None


def test_duplicate_source_creation_does_not_mutate_existing_override(tmp_path: Path) -> None:
    from app.api.app import create_app
    from app.services.storage import StorageService

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)

    class RecordingDispatcher:
        def __init__(self) -> None:
            self.jobs: list[Any] = []

        def dispatch(self, source_id: Any, job_id: Any) -> None:
            self.jobs.append((source_id, job_id))

    dispatcher = RecordingDispatcher()
    app = create_app(
        session_factory=factory,
        storage=StorageService(tmp_path / "storage"),
        dispatcher=dispatcher,
    )
    with TestClient(app) as client:
        first = client.post(
            "/sources/url",
            json={"url": "https://example.com/video.mp4", "dialect_profile_override": "SAUDI"},
        )
        assert first.status_code == 202
        assert first.json()["dialect_profile_override"] == "SAUDI"

        duplicate = client.post(
            "/sources/url",
            json={"url": "https://example.com/video.mp4", "dialect_profile_override": "EGYPTIAN"},
        )
        assert duplicate.status_code == 200
        assert duplicate.json()["dialect_profile_override"] == "SAUDI"

    engine.dispose()


def test_upload_accepts_optional_dialect_override(tmp_path: Path) -> None:
    from app.api.app import create_app
    from app.services.storage import StorageService

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'upload.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)

    class RecordingDispatcher:
        def __init__(self) -> None:
            self.jobs: list[Any] = []

        def dispatch(self, source_id: Any, job_id: Any) -> None:
            self.jobs.append((source_id, job_id))

    app = create_app(
        session_factory=factory,
        storage=StorageService(tmp_path / "storage"),
        dispatcher=RecordingDispatcher(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/sources/upload",
            files={"file": ("clip.mp4", b"video-bytes", "video/mp4")},
            data={"dialect_profile_override": "EGYPTIAN"},
        )
        assert response.status_code == 201
        assert response.json()["dialect_profile_override"] == "EGYPTIAN"


def test_transcript_response_exposes_effective_dialect_fields(tmp_path: Path) -> None:
    from app.api.app import create_app
    from app.services.storage import StorageService

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    app = create_app(
        session_factory=factory,
        storage=StorageService(tmp_path / "storage"),
        dispatcher=type("D", (), {"dispatch": lambda self, s, j: None})(),
    )
    with TestClient(app) as client:
        with factory() as session:
            source = SourceVideo(source_uri="file:///tmp/x.mp4", content_hash="x")
            session.add(source)
            session.commit()
            transcript = _transcript_factory(source.id)
            transcript.dialect_profile = _EGYPTIAN
            transcript.dialect_confidence = 0.9
            transcript.dialect_evidence = {"selection_method": "detected"}
            transcript.code_switch_suspected = True
            session.add(transcript)
            session.commit()
            source_id = source.id

        response = client.get(f"/api/sources/{source_id}/transcript")

        assert response.status_code == 200
        body = response.json()
        assert body["dialect_profile"] == "EGYPTIAN"
        assert body["dialect_confidence"] == 0.9
        assert body["dialect_evidence"] == {"selection_method": "detected"}
        assert body["code_switch_suspected"] is True
