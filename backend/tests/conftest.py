from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def configured_test_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Provide isolated database and media storage locations for each test."""

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    database_path = tmp_path / "clipfactory.sqlite3"
    monkeypatch.setenv("CLIPFACTORY_DATABASE_URL", f"sqlite+pysqlite:///{database_path}")
    monkeypatch.setenv("CLIPFACTORY_STORAGE_ROOT", str(storage_root))
    yield
