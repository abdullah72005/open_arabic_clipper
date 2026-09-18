"""Stage 4.3 selection persistence and migration reversibility."""

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
    OriginalityRisk,
    RightsRisk,
    TransformationSelectionStatus,
)
from app.core.settings import get_settings
from app.models import ClipCandidate, SourceVideo, TransformationPlanSelection

_POSTGRES_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")


def _database_url(url: str, database: str) -> str:
    return str(make_url(url).set(database=database).render_as_string(hide_password=False))


def _load_migration() -> Any:
    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "20260918_0019_stage_4_3_plan_selection.py"
    )
    specification = importlib.util.spec_from_file_location("stage_4_3_migration", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_stage_4_3_migration_declares_revision_metadata() -> None:
    module = _load_migration()
    assert module.revision == "20260918_0019"
    assert module.down_revision == "20260917_0018"


def test_stage_4_3_migration_is_reversible_and_preserves_prior_stages() -> None:
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    database_url = get_settings().database_url

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            source = SourceVideo(
                source_uri="/tmp/stage43-migration.mp4",
                content_hash="stage43-migration-hash",
            )
            session.add(source)
            session.flush()
            candidate = ClipCandidate(
                source_video_id=source.id,
                candidate_key="stage43-migration-candidate",
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
            session.flush()
            session.add(
                TransformationPlanSelection(
                    source_video_id=source.id,
                    clip_candidate_id=candidate.id,
                    status=TransformationSelectionStatus.NO_SELECTABLE_PLAN,
                    is_current=True,
                    input_fingerprint="migration-input-fp",
                )
            )
            session.commit()
            source_id = str(source.id)

        inspector = inspect(engine)
        assert "transformation_plan_selections" in inspector.get_table_names()

        command.downgrade(config, "20260917_0018")

        inspector = inspect(engine)
        assert "transformation_plan_selections" not in inspector.get_table_names()
        with engine.connect() as connection:
            hex_id = source_id.replace("-", "")
            sources = connection.execute(
                text("SELECT count(*) FROM source_videos WHERE source_uri = :uri"),
                {"uri": "/tmp/stage43-migration.mp4"},
            ).scalar_one()
            candidates = connection.execute(
                text("SELECT count(*) FROM clip_candidates WHERE source_video_id = :id"),
                {"id": hex_id},
            ).scalar_one()
        assert sources == 1
        assert candidates == 1

        command.upgrade(config, "head")
        inspector = inspect(engine)
        assert "transformation_plan_selections" in inspector.get_table_names()
        indexes = {
            index["name"] for index in inspector.get_indexes("transformation_plan_selections")
        }
        assert "uq_transformation_plan_selections_current" in indexes
    finally:
        engine.dispose()


def test_selection_constraints_exist() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    from app.db.base import Base

    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    checks = {
        str(constraint["sqltext"])
        for constraint in inspector.get_check_constraints("transformation_plan_selections")
    }
    assert any("selected_plan_id" in sqltext for sqltext in checks)
    assert any("PLAN_SELECTED" in sqltext for sqltext in checks)
    assert any("selected_with_caution" in sqltext for sqltext in checks)


@pytest.mark.skipif(  # type: ignore[untyped-decorator]
    not _POSTGRES_URL,
    reason="CLIPFACTORY_TEST_POSTGRES_URL is required for PostgreSQL migration validation",
)
def test_stage_4_3_postgresql_alembic_upgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real Alembic upgrade/downgrade on PostgreSQL with no DDL accommodation.

    The whole migration chain (including the frozen Stage 4.0/4.1/4.2 revisions)
    must execute against a real PostgreSQL database so the Stage 4.3 boolean
    predicates are proven valid. No production constraint is altered or removed.
    """

    assert _POSTGRES_URL is not None
    admin_url = _database_url(_POSTGRES_URL, "postgres")
    database = "clipfactory_stage43_migration_test"
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
            assert "transformation_plan_selections" in inspector.get_table_names()
            checks = {
                str(constraint["sqltext"]).upper()
                for constraint in inspector.get_check_constraints("transformation_plan_selections")
            }
            assert any("SELECTED_PLAN_ID" in sqltext for sqltext in checks)
            assert any("SELECTED_WITH_CAUTION" in sqltext for sqltext in checks)
            # No SQLite-style boolean-to-integer comparison may remain, because
            # PostgreSQL has no such operator.
            assert not any("IN (0, 1)" in sqltext or "IN (0,1)" in sqltext for sqltext in checks)
            assert not any("= 0" in sqltext or "= 1" in sqltext for sqltext in checks)

            command.downgrade(config, "20260917_0018")
            assert "transformation_plan_selections" not in inspect(engine).get_table_names()
            command.upgrade(config, "head")
            assert "transformation_plan_selections" in inspect(engine).get_table_names()
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
