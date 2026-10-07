"""Stage 5.2 migration wiring and reversibility (SQLite + gated PostgreSQL)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from alembic import command
from app.core.enums import (
    CandidateDisposition,
    ContentType,
    JobKind,
    JobStatus,
    OriginalityRisk,
    RightsRisk,
)
from app.core.settings import get_settings
from app.models import ClipCandidate, ProcessingJob, SourceVideo

_POSTGRES_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")


def _load_migration() -> Any:
    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "20260918_0022_stage_5_2_render_execution.py"
    )
    specification = importlib.util.spec_from_file_location("stage_5_2_migration", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _database_url(url: str, database: str) -> str:
    return str(make_url(url).set(database=database).render_as_string(hide_password=False))


def test_stage_5_2_migration_declares_revision_metadata() -> None:
    module = _load_migration()
    assert module.revision == "20260918_0022"
    assert module.down_revision == "20260918_0021"
    assert "RENDER_EXECUTION" in module._JOB_CURRENT
    assert "RENDER_EXECUTION" not in module._JOB_PREVIOUS


def test_stage_5_2_migration_is_reversible_and_preserves_prior_stages() -> None:
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    database_url = get_settings().database_url
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert "render_executions" in inspector.get_table_names()
        assert "visual_composition_plans" in inspector.get_table_names()
        columns = {column["name"] for column in inspector.get_columns("processing_jobs")}
        assert "render_execution_id" in columns
        indexes = {index["name"] for index in inspector.get_indexes("render_executions")}
        assert "uq_render_executions_current_scope" in indexes

        with Session(engine) as session:
            source = SourceVideo(
                source_uri="/tmp/stage52-migration.mp4",
                content_hash="stage52-migration-hash",
            )
            session.add(source)
            session.flush()
            session.add(
                ClipCandidate(
                    source_video_id=source.id,
                    candidate_key="stage52-migration-candidate",
                    disposition=CandidateDisposition.CANDIDATE,
                    start_time=0.0,
                    end_time=10.0,
                    start_segment_index=0,
                    end_segment_index=0,
                    primary_content_type=ContentType.STORY,
                    rights_risk=RightsRisk.LOW,
                    originality_risk=OriginalityRisk.NOT_INDICATED,
                )
            )
            session.add(
                ProcessingJob(
                    source_video_id=source.id,
                    kind=JobKind.INGEST,
                    status=JobStatus.SUCCEEDED,
                )
            )
            session.add(
                ProcessingJob(
                    source_video_id=source.id,
                    kind=JobKind.RENDER_EXECUTION,
                    status=JobStatus.QUEUED,
                )
            )
            session.commit()
            source_id = str(source.id)

        with engine.connect() as connection:
            accepted = connection.execute(
                text("SELECT count(*) FROM processing_jobs WHERE kind = 'RENDER_EXECUTION'")
            ).scalar_one()
        assert accepted == 1

        # Downgrade must remove Stage 5.2 jobs/rows itself on a populated
        # database (no test-side cleanup masking the defect).
        command.downgrade(config, "-1")
        inspector = inspect(engine)
        assert "render_executions" not in inspector.get_table_names()
        assert "visual_composition_plans" in inspector.get_table_names()
        columns = {column["name"] for column in inspector.get_columns("processing_jobs")}
        assert "render_execution_id" not in columns
        with engine.connect() as connection:
            render_jobs = connection.execute(
                text("SELECT count(*) FROM processing_jobs WHERE kind = 'RENDER_EXECUTION'")
            ).scalar_one()
            jobs = connection.execute(text("SELECT count(*) FROM processing_jobs")).scalar_one()
            sources = connection.execute(
                text("SELECT count(*) FROM source_videos WHERE source_uri = :uri"),
                {"uri": "/tmp/stage52-migration.mp4"},
            ).scalar_one()
            candidates = connection.execute(
                text("SELECT count(*) FROM clip_candidates WHERE source_video_id = :id"),
                {"id": source_id.replace("-", "")},
            ).scalar_one()
        assert render_jobs == 0
        assert jobs == 1
        assert sources == 1
        assert candidates == 1

        command.upgrade(config, "head")
        assert "render_executions" in inspect(engine).get_table_names()
    finally:
        engine.dispose()


@pytest.mark.skipif(  # type: ignore[untyped-decorator]
    not _POSTGRES_URL,
    reason="CLIPFACTORY_TEST_POSTGRES_URL is required for PostgreSQL migration validation",
)
def test_stage_5_2_postgresql_alembic_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _POSTGRES_URL is not None
    admin_url = _database_url(_POSTGRES_URL, "postgres")
    database = "clipfactory_stage52_migration_test"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parents[1] / "alembic"))
    engine = None
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
            connection.execute(text(f'CREATE DATABASE "{database}"'))
        target_url = _database_url(_POSTGRES_URL, database)
        monkeypatch.setenv("CLIPFACTORY_DATABASE_URL", target_url)
        get_settings.cache_clear()
        try:
            command.upgrade(config, "head")
            engine = create_engine(target_url)
            inspector = inspect(engine)
            assert "render_executions" in inspector.get_table_names()
            assert "render_execution_id" in {
                column["name"] for column in inspector.get_columns("processing_jobs")
            }
            indexes = {index["name"] for index in inspector.get_indexes("render_executions")}
            assert "uq_render_executions_current_scope" in indexes

            command.downgrade(config, "-1")
            inspector = inspect(engine)
            assert "render_executions" not in inspector.get_table_names()
            assert "render_execution_id" not in {
                column["name"] for column in inspector.get_columns("processing_jobs")
            }
            command.upgrade(config, "head")
            assert "render_executions" in inspect(engine).get_table_names()
        finally:
            get_settings.cache_clear()
    finally:
        if engine is not None:
            engine.dispose()
        admin.dispose()
        cleanup = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        try:
            with cleanup.connect() as connection:
                connection.execute(text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)'))
        finally:
            cleanup.dispose()
