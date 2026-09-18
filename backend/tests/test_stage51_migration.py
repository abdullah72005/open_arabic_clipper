"""Stage 5.1 visual-composition migration wiring and reversibility.

Runs against the repository's SQLite default and, when
``CLIPFACTORY_TEST_POSTGRES_URL`` is configured, against PostgreSQL too.
"""

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
        / "20260918_0021_stage_5_1_visual_composition.py"
    )
    specification = importlib.util.spec_from_file_location("stage_5_1_migration", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _database_url(url: str, database: str) -> str:
    return str(make_url(url).set(database=database).render_as_string(hide_password=False))


def test_stage_5_1_migration_declares_revision_metadata() -> None:
    module = _load_migration()
    assert module.revision == "20260918_0021"
    assert module.down_revision == "20260918_0020"
    assert "VISUAL_COMPOSITION" in module._JOB_CURRENT
    assert "VISUAL_COMPOSITION" not in module._JOB_PREVIOUS


def test_stage_5_1_migration_is_reversible_and_preserves_prior_stages() -> None:
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    database_url = get_settings().database_url

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert "visual_composition_plans" in inspector.get_table_names()
        assert "render_contracts" in inspector.get_table_names()
        columns = {column["name"] for column in inspector.get_columns("processing_jobs")}
        assert "visual_composition_plan_id" in columns
        indexes = {index["name"] for index in inspector.get_indexes("visual_composition_plans")}
        assert "uq_visual_composition_plans_current" in indexes

        with Session(engine) as session:
            source = SourceVideo(
                source_uri="/tmp/stage51-migration.mp4",
                content_hash="stage51-migration-hash",
            )
            session.add(source)
            session.flush()
            session.add(
                ClipCandidate(
                    source_video_id=source.id,
                    candidate_key="stage51-migration-candidate",
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
            job = ProcessingJob(
                source_video_id=source.id,
                kind=JobKind.VISUAL_COMPOSITION,
                status=JobStatus.QUEUED,
            )
            session.add(job)
            session.commit()
            source_id = str(source.id)

        # The migrated job_kind constraint accepts the Stage 5.1 value.
        with engine.connect() as connection:
            accepted = connection.execute(
                text("SELECT count(*) FROM processing_jobs WHERE kind = 'VISUAL_COMPOSITION'")
            ).scalar_one()
        assert accepted == 1

        # Remove the Stage 5.1 job so the downgrade can legally restore the
        # prior job_kind constraint over the preserved rows.
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM processing_jobs WHERE kind = 'VISUAL_COMPOSITION'")
            )

        command.downgrade(config, "-1")

        inspector = inspect(engine)
        assert "visual_composition_plans" not in inspector.get_table_names()
        assert "render_contracts" in inspector.get_table_names()
        columns = {column["name"] for column in inspector.get_columns("processing_jobs")}
        assert "visual_composition_plan_id" not in columns
        job_checks = {
            str(constraint["sqltext"])
            for constraint in inspector.get_check_constraints("processing_jobs")
        }
        assert all("VISUAL_COMPOSITION" not in sqltext for sqltext in job_checks)
        with engine.connect() as connection:
            sources = connection.execute(
                text("SELECT count(*) FROM source_videos WHERE source_uri = :uri"),
                {"uri": "/tmp/stage51-migration.mp4"},
            ).scalar_one()
            candidates = connection.execute(
                text("SELECT count(*) FROM clip_candidates WHERE source_video_id = :id"),
                {"id": source_id.replace("-", "")},
            ).scalar_one()
        assert sources == 1
        assert candidates == 1

        command.upgrade(config, "head")
        inspector = inspect(engine)
        assert "visual_composition_plans" in inspector.get_table_names()
        assert "visual_composition_plan_id" in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }
    finally:
        engine.dispose()


@pytest.mark.skipif(  # type: ignore[untyped-decorator]
    not _POSTGRES_URL,
    reason="CLIPFACTORY_TEST_POSTGRES_URL is required for PostgreSQL migration validation",
)
def test_stage_5_1_postgresql_alembic_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whole migration chain runs on PostgreSQL and Stage 5.1 reverses cleanly."""

    assert _POSTGRES_URL is not None
    admin_url = _database_url(_POSTGRES_URL, "postgres")
    database = "clipfactory_stage51_migration_test"
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
            assert "visual_composition_plans" in inspector.get_table_names()
            assert "visual_composition_plan_id" in {
                column["name"] for column in inspector.get_columns("processing_jobs")
            }

            command.downgrade(config, "-1")
            inspector = inspect(engine)
            assert "visual_composition_plans" not in inspector.get_table_names()
            assert "render_contracts" in inspector.get_table_names()
            command.upgrade(config, "head")
            assert "visual_composition_plans" in inspect(engine).get_table_names()
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
