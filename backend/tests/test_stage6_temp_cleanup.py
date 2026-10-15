"""Stage 6 worker cleanup owns only its own resources.

The materialization (and Stage 5.1 composition) executor release path must never
run a global age-zero ``.tmp`` sweep: that deleted another worker's fresh
in-progress upload. This regression runs the actual release method against the
real StorageService and asserts an unrelated in-progress temporary file survives.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.executor import build_materialization_executor
from app.services.storage import StorageCategory, StorageService


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def test_release_resource_does_not_sweep_other_workers_temp_files(
    session: Session,
) -> None:
    settings = get_settings()
    storage = StorageService(settings.storage_root)
    temporary_root = storage.category_root(StorageCategory.TEMPORARY)
    temporary_root.mkdir(parents=True, exist_ok=True)
    # A fresh in-progress temporary file owned by a *different* worker.
    other_tmp = temporary_root / "another-worker-upload.tmp"
    other_tmp.write_bytes(b"in progress")

    executor = build_materialization_executor(session, storage, settings)
    executor._release_resources()  # the defect was exactly this release method

    assert other_tmp.is_file()
    assert other_tmp.read_bytes() == b"in progress"
