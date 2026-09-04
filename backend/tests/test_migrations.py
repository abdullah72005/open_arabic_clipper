import importlib.util
from pathlib import Path


def _load_migration() -> object:
    path = Path(__file__).parents[1] / "alembic" / "versions" / "20260904_0002_expand_rights_statuses.py"
    specification = importlib.util.spec_from_file_location("rights_status_migration", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_rights_status_migration_normalizes_legacy_authorized_rows(monkeypatch: object) -> None:
    migration = _load_migration()
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)

    migration._normalize_legacy_authorized_rows()

    assert statements == [
        "UPDATE source_videos SET rights_status = 'PERMISSION' WHERE rights_status = 'AUTHORIZED'"
    ]
