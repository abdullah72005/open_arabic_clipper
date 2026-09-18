"""Stage 5.0 PostgreSQL-gated migration reversibility and concurrency.

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
from sqlalchemy import Engine, create_engine, inspect, select
from sqlalchemy.orm import sessionmaker
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import FakeProber, seed_stage50

from alembic import command
from app.core.settings import get_settings
from app.db.base import Base
from app.models import RenderContract
from app.render.service import create_render_contract

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL, reason="CLIPFACTORY_TEST_POSTGRES_URL is required for PostgreSQL Stage 5.0 tests"
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


def test_stage_5_0_migration_upgrade_and_downgrade_on_postgres() -> None:
    assert _URL is not None
    config = _alembic_config()
    command.upgrade(config, "20260918_0020")
    engine = create_engine(_URL)
    try:
        tables_at_0020 = set(inspect(engine).get_table_names())
        assert "render_contracts" in tables_at_0020
        assert "transformation_plan_selections" in tables_at_0020

        command.downgrade(config, "20260918_0019")
        tables_at_0019 = set(inspect(engine).get_table_names())
        assert "render_contracts" not in tables_at_0019
        # The downgrade removes only Stage 5.0 and preserves Stage 4.3.
        assert "transformation_plan_selections" in tables_at_0019
    finally:
        command.upgrade(config, "head")
        engine.dispose()


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    assert _URL is not None
    test_engine = create_engine(_URL)
    Base.metadata.drop_all(test_engine)
    Base.metadata.create_all(test_engine)
    try:
        yield test_engine
    finally:
        Base.metadata.drop_all(test_engine)
        test_engine.dispose()


def test_concurrent_render_contracts_converge_on_one_current_row(
    engine: Engine, monkeypatch: Any
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        fixture = seed_stage50(session, settings=settings)
        candidate_id = fixture.selection.candidate.id
        storage = fixture.storage
        session.commit()

    results: list[object] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with factory() as session:
                view = create_render_contract(
                    session, candidate_id, storage=storage, prober=FakeProber()
                )
                assert view is not None
                session.commit()
                results.append(view.row.id)
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
        current = session.scalars(
            select(RenderContract)
            .where(RenderContract.clip_candidate_id == candidate_id)
            .where(RenderContract.is_current.is_(True))
        ).all()
        assert len(current) == 1
