"""Focused Stage 3.5 candidate-refinement persistence tests.

Deterministic SQLite-only fixtures; no live providers or network access.
"""

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
    RefinementPriority,
    RefinementStatus,
    RightsRisk,
)
from app.db.base import Base
from app.models import CandidateRefinement, ClipCandidate, ProcessingJob, SourceVideo


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _source(session: Session, *, uri: str = "/tmp/stage35-models.mp4") -> SourceVideo:
    source = SourceVideo(source_uri=uri, content_hash=f"hash-{uri}")
    session.add(source)
    session.flush()
    return source


def _candidate(session: Session, source: SourceVideo, *, key: str = "candidate-1") -> ClipCandidate:
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key=key,
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
    return candidate


def _refinement(
    source: SourceVideo,
    candidate: ClipCandidate,
    *,
    priority: RefinementPriority = RefinementPriority.CANDIDATE,
    **overrides: object,
) -> CandidateRefinement:
    values: dict[str, object] = {
        "source_video_id": source.id,
        "clip_candidate_id": candidate.id,
        "priority": priority,
        "coarse_start": 0.0,
        "coarse_end": 10.0,
        "context_start": 0.0,
        "context_end": 20.0,
        "refined_start": 0.0,
        "refined_end": 10.0,
    }
    values.update(overrides)
    return CandidateRefinement(**values)


def test_candidate_priority_pair_is_unique(session: Session) -> None:
    source = _source(session)
    candidate = _candidate(session, source)
    session.add(_refinement(source, candidate))
    session.commit()

    session.add(_refinement(source, candidate))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_check_constraints_reject_out_of_range_and_invalid_bounds(session: Session) -> None:
    source = _source(session)
    candidate = _candidate(session, source)
    session.commit()
    candidate_id = candidate.id

    session.add(_refinement(source, candidate, confidence=1.5))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(_refinement(source, candidate, coarse_start=5.0, coarse_end=5.0))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(_refinement(source, candidate, context_start=5.0, context_end=1.0))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    refreshed = session.get(ClipCandidate, candidate_id)
    assert refreshed is not None
    assert refreshed.id == candidate_id


def test_processing_job_accepts_refinement_kind_and_reference(session: Session) -> None:
    source = _source(session)
    candidate = _candidate(session, source)
    refinement = _refinement(source, candidate)
    session.add(refinement)
    session.commit()

    job = ProcessingJob(
        source_video_id=source.id,
        candidate_refinement_id=refinement.id,
        kind=JobKind.CANDIDATE_REFINEMENT,
        status=JobStatus.QUEUED,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    assert job.kind is JobKind.CANDIDATE_REFINEMENT
    assert job.candidate_refinement_id == refinement.id


def test_enum_columns_and_quality_defaults_are_contractual(session: Session) -> None:
    columns = CandidateRefinement.__table__.columns
    assert columns["priority"].type.native_enum is False
    assert columns["status"].type.native_enum is False
    assert ProcessingJob.__table__.columns["kind"].type.native_enum is False

    source = _source(session)
    candidate = _candidate(session, source)
    refinement = _refinement(source, candidate)
    session.add(refinement)
    session.commit()
    session.refresh(refinement)

    assert refinement.quality_level == "CANDIDATE"
    assert refinement.cache_eligible is False
    assert refinement.status is RefinementStatus.QUEUED


def test_json_defaults_are_independent_per_instance(session: Session) -> None:
    source = _source(session)
    first = _candidate(session, source, key="candidate-a")
    second = _candidate(session, source, key="candidate-b")
    first_refinement = _refinement(source, first)
    second_refinement = _refinement(source, second)
    session.add_all([first_refinement, second_refinement])
    session.flush()

    assert first_refinement.word_timestamps == []
    assert second_refinement.word_timestamps == []
    assert first_refinement.metrics == {}
    assert second_refinement.metrics == {}
    assert first_refinement.word_timestamps is not second_refinement.word_timestamps
    assert first_refinement.metrics is not second_refinement.metrics

    first_refinement.word_timestamps.append({"start": 0.0, "end": 1.0, "word": "مرحبا"})
    first_refinement.metrics["provider_calls"] = 1

    assert second_refinement.word_timestamps == []
    assert second_refinement.metrics == {}
