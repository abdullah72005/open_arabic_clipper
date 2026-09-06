import importlib.util
from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command
from app.core.settings import get_settings


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


def test_stage_2_7_truth_migration_is_reversible() -> None:
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    database_url = get_settings().database_url

    command.upgrade(config, "20260905_0008")
    engine = create_engine(database_url)
    try:
        tables = (
            "pipeline_runs",
            "transcripts",
            "audio_analyses",
            "source_quality_assessments",
        )
        before = {
            table: {column["name"] for column in inspect(engine).get_columns(table)}
            for table in tables
        }

        command.upgrade(config, "20260906_0009")
        after = {
            table: {column["name"] for column in inspect(engine).get_columns(table)}
            for table in tables
        }
        expected = {
            "pipeline_runs": {"input_fingerprint", "output_fingerprint"},
            "transcripts": {
                "transcription_revision",
                "normalization_fingerprint",
                "reconstruction_status",
            },
            "audio_analyses": {"input_fingerprint"},
            "source_quality_assessments": {
                "transcript_quality_score",
                "low_confidence_word_ratio",
                "unresolved_segment_ratio",
                "manual_review_required",
                "input_fingerprint",
            },
        }
        for table, columns in expected.items():
            assert after[table] - before[table] == columns

        transcript_checks = {
            constraint["sqltext"]
            for constraint in inspect(engine).get_check_constraints("transcripts")
        }
        quality_checks = {
            constraint["sqltext"]
            for constraint in inspect(engine).get_check_constraints("source_quality_assessments")
        }
        assert "transcription_revision >= 0" in transcript_checks
        assert {
            "transcript_quality_score >= 0 AND transcript_quality_score <= 1",
            "low_confidence_word_ratio >= 0 AND low_confidence_word_ratio <= 1",
            "unresolved_segment_ratio >= 0 AND unresolved_segment_ratio <= 1",
        } <= quality_checks

        command.downgrade(config, "20260905_0008")
        downgraded = {
            table: {column["name"] for column in inspect(engine).get_columns(table)}
            for table in tables
        }
        assert downgraded == before
    finally:
        engine.dispose()
