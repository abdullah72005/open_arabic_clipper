"""Stage 5.1 PostgreSQL-gated migration reversibility and concurrency.

SQLite cannot meaningfully validate the partial unique index or row locking, so
this module is gated on ``CLIPFACTORY_TEST_POSTGRES_URL`` and skips otherwise.
Run it against the repository's compose PostgreSQL with that variable set.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import seed_stage50

from alembic import command
from app.composition.policy import (
    VisualCompositionExecutionStatus,
    VisualCompositionStatus,
)
from app.composition.queue import get_or_create_plan_row
from app.core.enums import JobKind, JobStatus
from app.core.settings import get_settings
from app.db.base import Base
from app.models import ProcessingJob, SourceVideo, VisualCompositionPlan

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL, reason="CLIPFACTORY_TEST_POSTGRES_URL is required for PostgreSQL Stage 5.1 tests"
)


def _alembic_config() -> Config:
    # The shared autouse test fixture forces a per-test SQLite URL and clears
    # the settings cache; alembic's env.py reads ``get_settings()``, so the
    # PostgreSQL target must be re-established for the migration run.
    if _URL:
        os.environ["CLIPFACTORY_DATABASE_URL"] = _URL
        get_settings.cache_clear()
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    config.set_main_option("sqlalchemy.url", _URL or "")
    return config


def _reset_public_schema(engine: Engine) -> None:
    """Drop and recreate the PostgreSQL ``public`` schema deterministically.

    ``Base.metadata.drop_all`` (the concurrency fixture teardown) removes ORM
    tables but never the Alembic ``alembic_version`` marker, so a second run
    against the same database would see the schema already at ``head`` and the
    migration ``upgrade`` would be a no-op. Dropping the whole schema also
    removes ``alembic_version`` and any other object outside ``Base.metadata``,
    so each test starts from a known-empty schema independent of prior runs.
    """

    with engine.connect() as connection:
        connection.execution_options(isolation_level="AUTOCOMMIT")
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))


def test_stage_5_1_migration_upgrade_and_downgrade_on_postgres() -> None:
    assert _URL is not None
    config = _alembic_config()
    engine = create_engine(_URL)
    try:
        # Idempotent start: clear any leftover schema/alembic_version from a
        # prior run before applying migrations from a known-empty state.
        _reset_public_schema(engine)
        command.upgrade(config, "head")

        inspector = inspect(engine)
        assert "visual_composition_plans" in inspector.get_table_names()
        # The Stage 5.1 upgrade preserves every Stage 5.0 table.
        assert "render_contracts" in inspector.get_table_names()
        assert "visual_composition_plan_id" in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }
        indexes = {index["name"] for index in inspector.get_indexes("visual_composition_plans")}
        assert "uq_visual_composition_plans_current" in indexes

        # The migrated job_kind constraint accepts the Stage 5.1 value.
        with Session(engine) as session:
            source = SourceVideo(
                source_uri="/tmp/stage51-postgres.mp4",
                content_hash="stage51-postgres-hash",
            )
            session.add(source)
            session.flush()
            session.add(
                ProcessingJob(
                    source_video_id=source.id,
                    kind=JobKind.VISUAL_COMPOSITION,
                    status=JobStatus.QUEUED,
                )
            )
            session.commit()

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
        # The downgrade removes only Stage 5.1 and preserves every prior stage.
        assert "render_contracts" in inspector.get_table_names()
        assert "visual_composition_plan_id" not in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }
        job_checks = {
            str(constraint["sqltext"])
            for constraint in inspector.get_check_constraints("processing_jobs")
        }
        assert all("VISUAL_COMPOSITION" not in sqltext for sqltext in job_checks)

        command.upgrade(config, "head")
        inspector = inspect(engine)
        assert "visual_composition_plans" in inspector.get_table_names()
        assert "visual_composition_plan_id" in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }
    finally:
        # Leave the disposable database at head for a later module/test.
        command.upgrade(config, "head")
        engine.dispose()


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    assert _URL is not None
    test_engine = create_engine(_URL)
    _reset_public_schema(test_engine)
    Base.metadata.create_all(test_engine)
    try:
        yield test_engine
    finally:
        # Reset (not just drop_all) so no alembic_version or other leftover
        # state can break a subsequent run against the same database.
        _reset_public_schema(test_engine)
        test_engine.dispose()


def test_concurrent_visual_composition_plans_converge_on_one_current_row(
    engine: Engine, monkeypatch: Any
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        fixture = seed_stage50(session, settings=settings)
        candidate = fixture.selection.candidate
        # A pre-existing historical (non-current) version must survive.
        session.add(
            VisualCompositionPlan(
                source_video_id=candidate.source_video_id,
                clip_candidate_id=candidate.id,
                status=VisualCompositionStatus.BLOCKED,
                execution_status=VisualCompositionExecutionStatus.COMPLETE,
                plan_ready=False,
                is_current=False,
                input_fingerprint="h" * 64,
            )
        )
        session.commit()

    barrier = threading.Barrier(2)
    results: list[object] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with factory() as session:
                barrier.wait()
                row = get_or_create_plan_row(session, candidate)
                results.append(row.id)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert len(results) == 2
    assert results[0] == results[1]
    with factory() as session:
        rows = session.scalars(
            select(VisualCompositionPlan).where(
                VisualCompositionPlan.clip_candidate_id == candidate.id
            )
        ).all()
        current = [row for row in rows if row.is_current]
        # The partial unique index enforces exactly one current row per candidate.
        assert len(current) == 1
        # The losing racer leaves no duplicate and the historical row is intact.
        assert len(rows) == 2
        history = [row for row in rows if row.input_fingerprint == "h" * 64]
        assert len(history) == 1
        assert history[0].is_current is False
