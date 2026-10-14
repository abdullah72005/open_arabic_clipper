"""Stage 6 materialization persistence and migration reversibility."""

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
    MaterializationLifecycle,
    MaterializationPriority,
    OriginalityRisk,
    RightsRisk,
)
from app.core.settings import get_settings
from app.models import (
    ChannelProductionProfile,
    ClipCandidate,
    ClipMaterialization,
    SourceVideo,
)

_POSTGRES_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")


def _database_url(url: str, database: str) -> str:
    return str(make_url(url).set(database=database).render_as_string(hide_password=False))


def _load_migration() -> Any:
    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "20261014_0023_stage_6_materialization.py"
    )
    specification = importlib.util.spec_from_file_location("stage_6_migration", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_stage_6_migration_declares_revision_metadata() -> None:
    module = _load_migration()
    assert module.revision == "20261014_0023"
    assert module.down_revision == "20260918_0022"


def test_stage_6_migration_is_reversible_and_preserves_prior_stages() -> None:
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    database_url = get_settings().database_url

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            source = SourceVideo(
                source_uri="/tmp/stage6-migration.mp4",
                content_hash="stage6-migration-hash",
            )
            session.add(source)
            session.flush()
            candidate = ClipCandidate(
                source_video_id=source.id,
                candidate_key="stage6-migration-candidate",
                disposition=CandidateDisposition.CANDIDATE,
                start_time=0.0,
                end_time=10.0,
                start_segment_index=0,
                end_segment_index=0,
                primary_content_type=ContentType.STORY,
                rights_risk=RightsRisk.LOW,
                originality_risk=OriginalityRisk.NOT_INDICATED,
            )
            session.add(candidate)
            profile = ChannelProductionProfile(channel_key="stage6-migration-channel")
            session.add(profile)
            session.flush()
            session.add(
                ClipMaterialization(
                    source_video_id=source.id,
                    clip_candidate_id=candidate.id,
                    channel_production_profile_id=profile.id,
                    profile_key="stage6-migration-channel",
                    priority=MaterializationPriority.CANDIDATE,
                    lifecycle=MaterializationLifecycle.QUEUED,
                )
            )
            session.commit()
            source_id = str(source.id)

        inspector = inspect(engine)
        assert "clip_materializations" in inspector.get_table_names()
        assert "channel_production_profiles" in inspector.get_table_names()
        assert "materialization_id" in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }
        assert "materialization_id" in {
            column["name"] for column in inspector.get_columns("render_executions")
        }

        command.downgrade(config, "20260918_0022")

        inspector = inspect(engine)
        assert "clip_materializations" not in inspector.get_table_names()
        assert "channel_production_profiles" not in inspector.get_table_names()
        assert "materialization_id" not in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }
        assert "materialization_id" not in {
            column["name"] for column in inspector.get_columns("render_executions")
        }
        with engine.connect() as connection:
            hex_id = source_id.replace("-", "")
            sources = connection.execute(
                text("SELECT count(*) FROM source_videos WHERE source_uri = :uri"),
                {"uri": "/tmp/stage6-migration.mp4"},
            ).scalar_one()
            candidates = connection.execute(
                text("SELECT count(*) FROM clip_candidates WHERE source_video_id = :id"),
                {"id": hex_id},
            ).scalar_one()
        assert sources == 1
        assert candidates == 1

        command.upgrade(config, "head")
        inspector = inspect(engine)
        assert "clip_materializations" in inspector.get_table_names()
        indexes = {index["name"] for index in inspector.get_indexes("clip_materializations")}
        assert "uq_clip_materializations_current_scope" in indexes
    finally:
        engine.dispose()


def test_materialization_constraints_exist() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    from app.db.base import Base

    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    checks = {
        str(constraint["sqltext"])
        for constraint in inspector.get_check_constraints("clip_materializations")
    }
    assert any("repair_count" in sqltext for sqltext in checks)
    assert any("cache_eligible" in sqltext for sqltext in checks)
    profile_checks = {
        str(constraint["sqltext"])
        for constraint in inspector.get_check_constraints("channel_production_profiles")
    }
    assert any("revision" in sqltext for sqltext in profile_checks)


@pytest.mark.skipif(  # type: ignore[untyped-decorator]
    not _POSTGRES_URL,
    reason="CLIPFACTORY_TEST_POSTGRES_URL is required for PostgreSQL migration validation",
)
def test_stage_6_postgresql_alembic_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real Alembic upgrade/downgrade on PostgreSQL with no DDL accommodation."""

    assert _POSTGRES_URL is not None
    admin_url = _database_url(_POSTGRES_URL, "postgres")
    database = "clipfactory_stage6_migration_test"
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
            assert "clip_materializations" in inspector.get_table_names()
            assert "channel_production_profiles" in inspector.get_table_names()
            command.downgrade(config, "20260918_0022")
            assert "clip_materializations" not in inspect(engine).get_table_names()
            command.upgrade(config, "head")
            assert "clip_materializations" in inspect(engine).get_table_names()
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
