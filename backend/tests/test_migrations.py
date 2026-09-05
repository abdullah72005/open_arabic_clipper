import importlib.util
from pathlib import Path


def _load_migration() -> object:
    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "20260904_0002_expand_rights_statuses.py"
    )
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


def test_stage_2_migration_declares_transcript_tables() -> None:
    """Schema migration introduces durable transcript and analysis records."""

    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "20260904_0003_stage_2_transcription.py"
    )
    specification = importlib.util.spec_from_file_location("stage_2_migration", path)

    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    assert module.revision == "20260904_0003"
    assert module.down_revision == "20260904_0002"
    assert "INGEST" in module._PIPELINE_BEFORE


def test_pipeline_stage_forward_migration_restores_ingest() -> None:
    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "20260904_0005_add_ingest_pipeline_stage.py"
    )
    specification = importlib.util.spec_from_file_location("pipeline_ingest_migration", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    assert module.revision == "20260904_0005"
    assert module.down_revision == "20260904_0004"
    assert "INGEST" in module._AFTER


def test_stage_2_5_migration_preserves_raw_and_adds_derived_correction_columns() -> None:
    path = (
        Path(__file__).parents[1] / "alembic" / "versions" / "20260905_0006_stage_2_5_correction.py"
    )
    specification = importlib.util.spec_from_file_location("stage_2_5_correction_migration", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    assert module.revision == "20260905_0006"
    assert module.down_revision == "20260904_0005"
