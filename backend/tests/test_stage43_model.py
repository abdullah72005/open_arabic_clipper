"""Stage 4.3 selection model: constraints and one-current uniqueness."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import (
    CandidateDisposition,
    ContentType,
    OriginalityRisk,
    RightsRisk,
    TransformationSelectionStatus,
)
from app.db.base import Base
from app.models import ClipCandidate, SourceVideo, TransformationPlanSelection


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _candidate(session: Session) -> ClipCandidate:
    source = SourceVideo(source_uri="/tmp/stage43-model.mp4", content_hash="stage43-model-hash")
    session.add(source)
    session.flush()
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key="stage43-model-candidate",
        disposition=CandidateDisposition.CANDIDATE,
        start_time=0.0,
        end_time=10.0,
        start_segment_index=0,
        end_segment_index=0,
        primary_content_type=ContentType.OTHER,
        rights_risk=RightsRisk.LOW,
        originality_risk=OriginalityRisk.NOT_INDICATED,
    )
    session.add(candidate)
    session.flush()
    return candidate


def _selection(
    candidate: ClipCandidate,
    *,
    status: TransformationSelectionStatus = TransformationSelectionStatus.NO_SELECTABLE_PLAN,
    input_fingerprint: str = "fp",
    is_current: bool = True,
    selected_plan_id: object = None,
    selected_governance_result_id: object = None,
    selected_with_caution: bool = False,
) -> TransformationPlanSelection:
    return TransformationPlanSelection(
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        status=status,
        is_current=is_current,
        selected_with_caution=selected_with_caution,
        selected_plan_id=selected_plan_id,
        selected_governance_result_id=selected_governance_result_id,
        input_fingerprint=input_fingerprint,
    )


def test_non_selected_selection_persists(session: Session) -> None:
    candidate = _candidate(session)
    session.add(_selection(candidate))
    session.commit()
    row = session.scalars(select(TransformationPlanSelection)).one()
    assert row.selected_plan_id is None
    assert row.is_current is True


def test_selected_status_requires_selected_plan(session: Session) -> None:
    candidate = _candidate(session)
    session.add(_selection(candidate, status=TransformationSelectionStatus.PLAN_SELECTED))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_non_selected_status_rejects_selected_plan(session: Session) -> None:
    candidate = _candidate(session)
    session.add(_selection(candidate, selected_plan_id=candidate.id))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_selected_pair_requires_both_ids(session: Session) -> None:
    candidate = _candidate(session)
    session.add(
        _selection(
            candidate,
            status=TransformationSelectionStatus.PLAN_SELECTED,
            selected_plan_id=candidate.id,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_caution_flag_only_for_caution_status(session: Session) -> None:
    candidate = _candidate(session)
    session.add(
        _selection(
            candidate,
            status=TransformationSelectionStatus.NO_SELECTABLE_PLAN,
            selected_with_caution=True,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_at_most_one_current_selection_per_candidate(session: Session) -> None:
    candidate = _candidate(session)
    session.add(_selection(candidate, input_fingerprint="fp-one"))
    session.commit()
    session.add(_selection(candidate, input_fingerprint="fp-two"))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_historical_and_current_rows_coexist(session: Session) -> None:
    candidate = _candidate(session)
    session.add(_selection(candidate, input_fingerprint="fp-old", is_current=False))
    session.add(_selection(candidate, input_fingerprint="fp-new", is_current=True))
    session.commit()
    rows = session.scalars(
        select(TransformationPlanSelection).order_by(TransformationPlanSelection.input_fingerprint)
    ).all()
    assert [row.input_fingerprint for row in rows] == ["fp-new", "fp-old"]
    assert sum(1 for row in rows if row.is_current) == 1


def test_same_input_fingerprint_is_unique_per_candidate(session: Session) -> None:
    candidate = _candidate(session)
    session.add(_selection(candidate, input_fingerprint="fp", is_current=False))
    session.commit()
    session.add(_selection(candidate, input_fingerprint="fp", is_current=True))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()
