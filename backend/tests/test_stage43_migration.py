"""Stage 4.3 selection persistence and migration reversibility."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
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
