"""Stage 4.2 governance persistence, constraints, and migration reversibility."""

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
        / "20260917_0018_stage_4_2_transformation_governance.py"
    )
    specification = importlib.util.spec_from_file_location("stage_4_2_migration", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_stage_4_2_migration_declares_revision_metadata() -> None:
    module = _load_migration()
    assert module.revision == "20260917_0018"
    assert module.down_revision == "20260915_0017"


def test_stage_4_2_migration_is_reversible_and_preserves_stage_1_to_4_1_data() -> None:
    backend_root = Path(__file__).parents[1]
    config = Config()
    config.set_main_option("script_location", str(backend_root / "alembic"))
    database_url = get_settings().database_url

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            source = SourceVideo(
                source_uri="/tmp/stage42-migration.mp4",
                content_hash="stage42-migration-hash",
            )
            session.add(source)
            session.flush()
            session.add(
                Transcript(
                    source_video_id=source.id,
                    whisper_model="large-v3-turbo",
                    input_fingerprint="n" * 64,
                    raw_text="raw",
                    segments=[{"start": 0.0, "end": 10.0, "text": "raw"}],
                    duration=10.0,
                    language="ar",
                )
            )
            candidate = ClipCandidate(
                source_video_id=source.id,
                candidate_key="stage42-migration-candidate",
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

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "transformation_governance_sets" in tables
        assert "transformation_governance_results" in tables
        assert "transformation_governance_set_id" in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }

        command.downgrade(config, "20260915_0017")

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "transformation_governance_sets" not in tables
        assert "transformation_governance_results" not in tables
        assert "transformation_governance_set_id" not in {
            column["name"] for column in inspector.get_columns("processing_jobs")
        }
        job_checks = {
            str(constraint["sqltext"])
            for constraint in inspector.get_check_constraints("processing_jobs")
        }
        assert all("TRANSFORMATION_GOVERNANCE" not in sqltext for sqltext in job_checks)
        assert any("TRANSFORMATION_PLANNING" in sqltext for sqltext in job_checks)

        with engine.connect() as connection:
            hex_id = source_id.replace("-", "")
            sources = connection.execute(
                text("SELECT count(*) FROM source_videos WHERE source_uri = :uri"),
                {"uri": "/tmp/stage42-migration.mp4"},
            ).scalar_one()
            refinements = connection.execute(
                text("SELECT count(*) FROM candidate_refinements WHERE source_video_id = :id"),
                {"id": hex_id},
            ).scalar_one()
            runs = connection.execute(
                text("SELECT count(*) FROM pipeline_runs WHERE source_video_id = :id"),
                {"id": hex_id},
            ).scalar_one()
            jobs = (
                connection.execute(
                    text("SELECT kind FROM processing_jobs WHERE source_video_id = :id"),
                    {"id": hex_id},
                )
                .scalars()
                .all()
            )
        assert sources == 1
        assert refinements == 1
        assert runs == 1
        assert jobs == ["CANDIDATE_REFINEMENT"]

        command.upgrade(config, "head")
        inspector = inspect(engine)
        assert "transformation_governance_sets" in inspector.get_table_names()
    finally:
        engine.dispose()


def test_governance_result_constraints_exist() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    from app.db.base import Base

    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    result_checks = {
        str(constraint["sqltext"])
        for constraint in inspector.get_check_constraints("transformation_governance_results")
    }
    assert any("eligible_for_stage4_3" in sqltext for sqltext in result_checks)
    assert any("APPROVED_FOR_SELECTION" in sqltext for sqltext in result_checks)
    job_kind_values = {
        str(constraint["sqltext"])
        for constraint in inspector.get_check_constraints("processing_jobs")
    }
    assert any("TRANSFORMATION_GOVERNANCE" in sqltext for sqltext in job_kind_values)
