"""Stage 3.5 candidate-refinement migration reversibility tests.

Uses the disposable per-test SQLite database configured by ``conftest``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from alembic import command
from app.core.enums import (
    CandidateDisposition,
    ContentType,
    JobKind,
    JobStatus,
    OriginalityRisk,
    PipelineRunStatus,
    PipelineStage,
    RefinementPriority,
    RefinementStatus,
    RightsRisk,
)
from app.core.settings import get_settings
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    PipelineRun,
    ProcessingJob,
    SourceVideo,
)


def _load_migration() -> object:
    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "20260913_0012_stage_3_5_candidate_refinement.py"
    )
    specification = importlib.util.spec_from_file_location("stage_3_5_migration", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_stage_3_5_migration_declares_revision_metadata() -> None:
    module = _load_migration()

    assert module.revision == "20260913_0012"
    assert module.down_revision == "20260912_0011"


def test_stage_3_5_migration_is_reversible_and_preserves_stage_1_to_3_data() -> None:
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    database_url = get_settings().database_url

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            source = SourceVideo(
                source_uri="/tmp/stage35-migration.mp4",
                content_hash="stage35-migration-hash",
            )
            session.add(source)
            session.flush()
            session.add(
                ProcessingJob(
                    source_video_id=source.id,
                    kind=JobKind.CANDIDATE_ANALYSIS,
                    status=JobStatus.SUCCEEDED,
                )
            )
            session.add(
                PipelineRun(
                    source_video_id=source.id,
                    stage=PipelineStage.CANDIDATE_ANALYSIS,
                    status=PipelineRunStatus.SUCCEEDED,
                )
            )
            candidate = ClipCandidate(
                source_video_id=source.id,
                candidate_key="stage35-migration-candidate",
                disposition=CandidateDisposition.CANDIDATE,
                start_time=0.0,
                end_time=10.0,
                start_segment_index=0,
                end_segment_index=1,
                primary_content_type=ContentType.STORY,
                rights_risk=RightsRisk.LOW,
                originality_risk=OriginalityRisk.NOT_INDICATED,
            )
            session.add(candidate)
            session.flush()
            refinement = CandidateRefinement(
                source_video_id=source.id,
                clip_candidate_id=candidate.id,
                priority=RefinementPriority.CANDIDATE,
                status=RefinementStatus.QUEUED,
                coarse_start=0.0,
                coarse_end=10.0,
                context_start=0.0,
                context_end=20.0,
                refined_start=0.0,
                refined_end=10.0,
            )
            session.add(refinement)
            session.flush()
            session.add(
                ProcessingJob(
                    source_video_id=source.id,
                    candidate_refinement_id=refinement.id,
                    kind=JobKind.CANDIDATE_REFINEMENT,
                    status=JobStatus.QUEUED,
                )
            )
            session.commit()
            source_id = str(source.id)
            candidate_key = candidate.candidate_key

        inspector = inspect(engine)
        assert "candidate_refinements" in inspector.get_table_names()
        assert "candidate_refinement_id" in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }

        command.downgrade(config, "20260912_0011")

        inspector = inspect(engine)
        assert "candidate_refinements" not in inspector.get_table_names()
        assert "candidate_refinement_id" not in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }
        job_checks = {
            str(constraint["sqltext"])
            for constraint in inspector.get_check_constraints("processing_jobs")
        }
        assert all("CANDIDATE_REFINEMENT" not in sqltext for sqltext in job_checks)
        assert any("CANDIDATE_ANALYSIS" in sqltext for sqltext in job_checks)

        with engine.connect() as connection:
            hex_id = source_id.replace("-", "")
            sources = connection.execute(
                text("SELECT count(*) FROM source_videos WHERE source_uri = :uri"),
                {"uri": "/tmp/stage35-migration.mp4"},
            ).scalar_one()
            kinds = (
                connection.execute(
                    text("SELECT kind FROM processing_jobs WHERE source_video_id = :id"),
                    {"id": hex_id},
                )
                .scalars()
                .all()
            )
            runs = connection.execute(
                text("SELECT count(*) FROM pipeline_runs WHERE source_video_id = :id"),
                {"id": hex_id},
            ).scalar_one()
            candidates = connection.execute(
                text("SELECT count(*) FROM clip_candidates WHERE candidate_key = :key"),
                {"key": candidate_key},
            ).scalar_one()
        assert sources == 1
        assert kinds == ["CANDIDATE_ANALYSIS"]
        assert runs == 1
        assert candidates == 1

        command.upgrade(config, "head")
        inspector = inspect(engine)
        assert "candidate_refinements" in inspector.get_table_names()
        assert "candidate_refinement_id" in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }
    finally:
        engine.dispose()
