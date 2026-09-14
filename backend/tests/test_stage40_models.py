"""Focused Stage 4.0 persistence-model constraint tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import (
    CandidateDisposition,
    ContentType,
    JobKind,
    JobStatus,
    OriginalityRisk,
    RightsRisk,
    StrategyDisposition,
    SubstantiveValueKind,
    TransformationExecutionStatus,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.db.base import Base
from app.models import (
    ClipCandidate,
    ProcessingJob,
    SourceVideo,
    TransformationEligibilityAnalysis,
    TransformationStrategyCandidate,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _analysis(session: Session) -> TransformationEligibilityAnalysis:
    source = SourceVideo(source_uri="/tmp/s40-models.mp4", content_hash="h")
    session.add(source)
    session.flush()
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key="k",
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
    analysis = TransformationEligibilityAnalysis(
        source_video_id=source.id,
        clip_candidate_id=candidate.id,
    )
    session.add(analysis)
    session.flush()
    return analysis


def _strategy(
    analysis: TransformationEligibilityAnalysis,
    strategy_type: TransformationStrategyType = TransformationStrategyType.ANALYSIS,
) -> TransformationStrategyCandidate:
    return TransformationStrategyCandidate(
        analysis_id=analysis.id,
        strategy_type=strategy_type,
        strategy_key=f"{analysis.id}:{strategy_type.value}",
        disposition=StrategyDisposition.RECOMMENDED,
        intensity=TransformationIntensity.MODERATE,
        substantive_value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )


def test_one_analysis_per_candidate(session: Session) -> None:
    analysis = _analysis(session)
    session.add(
        TransformationEligibilityAnalysis(
            source_video_id=analysis.source_video_id,
            clip_candidate_id=analysis.clip_candidate_id,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_strategy_type_unique_per_analysis(session: Session) -> None:
    analysis = _analysis(session)
    session.add(_strategy(analysis))
    session.commit()
    session.add(_strategy(analysis))
    with pytest.raises(IntegrityError):
        session.commit()


def test_strategy_assessment_bounds_enforced(session: Session) -> None:
    analysis = _analysis(session)
    row = _strategy(analysis)
    row.retention_preservation = 1.5
    session.add(row)
    with pytest.raises(IntegrityError):
        session.commit()


def test_processing_job_supports_stage_4_0_kind_and_fk(session: Session) -> None:
    analysis = _analysis(session)
    job = ProcessingJob(
        source_video_id=analysis.source_video_id,
        kind=JobKind.TRANSFORMATION_ELIGIBILITY,
        transformation_analysis_id=analysis.id,
        status=JobStatus.QUEUED,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    assert job.transformation_analysis_id == analysis.id
    assert analysis.execution_status is TransformationExecutionStatus.QUEUED
