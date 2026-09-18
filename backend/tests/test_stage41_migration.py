"""Stage 4.1 transformation-planning migration metadata and reversibility tests."""

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
    TransformationEligibilityOutcome,
    TransformationExecutionStatus,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.core.settings import get_settings
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    PipelineRun,
    ProcessingJob,
    SourceVideo,
    Transcript,
    TransformationEligibilityAnalysis,
    TransformationStrategyCandidate,
)


def _load_migration() -> object:
    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "20260914_0015_stage_4_1_transformation_planning.py"
    )
    specification = importlib.util.spec_from_file_location("stage_4_1_migration", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_stage_4_1_migration_declares_revision_metadata() -> None:
    module = _load_migration()
    assert module.revision == "20260914_0015"
    assert module.down_revision == "20260914_0014"


def test_stage_4_1_migration_is_reversible_and_preserves_stage_1_to_4_0_data() -> None:
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    database_url = get_settings().database_url

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            source = SourceVideo(
                source_uri="/tmp/stage41-migration.mp4",
                content_hash="stage41-migration-hash",
            )
            session.add(source)
            session.flush()
            session.add(
                Transcript(
                    source_video_id=source.id,
                    whisper_model="large-v3-turbo",
                    input_fingerprint="m" * 64,
                    raw_text="raw",
                    segments=[{"start": 0.0, "end": 10.0, "text": "raw"}],
                    duration=10.0,
                    language="ar",
                )
            )
            candidate = ClipCandidate(
                source_video_id=source.id,
                candidate_key="stage41-migration-candidate",
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
            refinement = CandidateRefinement(
                source_video_id=source.id,
                clip_candidate_id=candidate.id,
                priority=RefinementPriority.CANDIDATE,
                status=RefinementStatus.CANDIDATE_REFINED,
                coarse_start=0.0,
                coarse_end=10.0,
                context_start=0.0,
                context_end=20.0,
                refined_start=0.0,
                refined_end=10.0,
                final_transcript="refined",
                confidence=0.9,
            )
            session.add(refinement)
            session.flush()
            analysis = TransformationEligibilityAnalysis(
                source_video_id=source.id,
                clip_candidate_id=candidate.id,
                refinement_id=refinement.id,
                execution_status=TransformationExecutionStatus.COMPLETE,
                eligibility_outcome=(TransformationEligibilityOutcome.ELIGIBLE_FOR_TRANSFORMATION),
                output_fingerprint="stage40-out",
            )
            session.add(analysis)
            session.flush()
            session.add(
                TransformationStrategyCandidate(
                    analysis_id=analysis.id,
                    strategy_key="k",
                    strategy_type=TransformationStrategyType.SOURCE_AS_EVIDENCE,
                    disposition="RECOMMENDED",
                    intensity=TransformationIntensity.MODERATE,
                    rank=1,
                )
            )
            session.add(
                PipelineRun(
                    source_video_id=source.id,
                    stage=PipelineStage.CANDIDATE_ANALYSIS,
                    status=PipelineRunStatus.SUCCEEDED,
                )
            )
            session.add(
                ProcessingJob(
                    source_video_id=source.id,
                    candidate_refinement_id=refinement.id,
                    kind=JobKind.CANDIDATE_REFINEMENT,
                    status=JobStatus.SUCCEEDED,
                )
            )
            session.commit()
            source_id = str(source.id)
            analysis_id = str(analysis.id)
            candidate_key = candidate.candidate_key

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "transformation_plan_sets" in tables
        assert "transformation_plans" in tables
        assert "transformation_plan_set_id" in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }

        command.downgrade(config, "20260914_0014")

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "transformation_plan_sets" not in tables
        assert "transformation_plans" not in tables
        assert "transformation_plan_set_id" not in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }
        job_checks = {
            str(constraint["sqltext"])
            for constraint in inspector.get_check_constraints("processing_jobs")
        }
        assert all("TRANSFORMATION_PLANNING" not in sqltext for sqltext in job_checks)
        assert any("TRANSFORMATION_ELIGIBILITY" in sqltext for sqltext in job_checks)

        with engine.connect() as connection:
            hex_id = source_id.replace("-", "")
            sources = connection.execute(
                text("SELECT count(*) FROM source_videos WHERE source_uri = :uri"),
                {"uri": "/tmp/stage41-migration.mp4"},
            ).scalar_one()
            refinements = connection.execute(
                text("SELECT count(*) FROM candidate_refinements WHERE source_video_id = :id"),
                {"id": hex_id},
            ).scalar_one()
            runs = connection.execute(
                text("SELECT count(*) FROM pipeline_runs WHERE source_video_id = :id"),
                {"id": hex_id},
            ).scalar_one()
            analyses = connection.execute(
                text(
                    "SELECT count(*) FROM transformation_eligibility_analyses "
                    "WHERE source_video_id = :id"
                ),
                {"id": hex_id},
            ).scalar_one()
            strategies = connection.execute(
                text(
                    "SELECT count(*) FROM transformation_strategy_candidates "
                    "WHERE analysis_id = :id"
                ),
                {"id": analysis_id.replace("-", "")},
            ).scalar_one()
            jobs = (
                connection.execute(
                    text("SELECT kind FROM processing_jobs WHERE source_video_id = :id"),
                    {"id": hex_id},
                )
                .scalars()
                .all()
            )
            candidates = connection.execute(
                text("SELECT count(*) FROM clip_candidates WHERE candidate_key = :key"),
                {"key": candidate_key},
            ).scalar_one()
            lifecycle = connection.execute(
                text("SELECT lifecycle_state FROM source_videos WHERE source_uri = :uri"),
                {"uri": "/tmp/stage41-migration.mp4"},
            ).scalar_one()
        assert sources == 1
        assert refinements == 1
        assert runs == 1
        assert analyses == 1
        assert strategies == 1
        assert jobs == ["CANDIDATE_REFINEMENT"]
        assert candidates == 1
        assert lifecycle == PipelineStage.INGEST.value

        command.upgrade(config, "head")
        inspector = inspect(engine)
        assert "transformation_plan_sets" in inspector.get_table_names()
    finally:
        engine.dispose()
