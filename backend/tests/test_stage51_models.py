"""Stage 5.1 visual-composition-plan model constraints and enum round-trips."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import seed_stage50

from app.composition.policy import (
    VisualCompositionExecutionStatus,
    VisualCompositionStatus,
)
from app.db.base import Base
from app.models import VisualCompositionPlan
from app.render.service import create_render_contract


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as seeded:
        yield seeded


def _seed_graph(session: Session, monkeypatch: Any) -> Any:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_stage50(session, settings=settings)
    view = create_render_contract(
        session,
        fixture.selection.candidate.id,
        storage=fixture.storage,
        prober=fixture.prober,
    )
    assert view is not None
    return fixture, view.row


def _row(candidate: Any, contract: Any, **overrides: object) -> VisualCompositionPlan:
    values: dict[str, object] = {
        "source_video_id": candidate.source_video_id,
        "clip_candidate_id": candidate.id,
        "render_contract_id": contract.id,
        "transformation_selection_id": contract.transformation_selection_id,
        "selected_plan_id": contract.selected_plan_id,
        "final_refinement_id": contract.final_refinement_id,
        "status": VisualCompositionStatus.READY_FOR_VISUAL_EXECUTION,
        "execution_status": VisualCompositionExecutionStatus.QUEUED,
        "plan_ready": True,
        "is_current": True,
        "reason_codes": [],
        "input_fingerprint": uuid.uuid4().hex,
        "output_fingerprint": uuid.uuid4().hex,
        "plan_payload": {"scenes": []},
        "readiness": {"ready": True},
        "metrics": {"sampled_frames": 0},
        "cache_eligible": False,
    }
    values.update(overrides)
    return VisualCompositionPlan(**values)  # type: ignore[arg-type]


def test_row_with_bound_foreign_keys_persists(session: Session, monkeypatch: Any) -> None:
    fixture, contract = _seed_graph(session, monkeypatch)
    row = _row(fixture.selection.candidate, contract)
    session.add(row)
    session.flush()

    stored = session.get(VisualCompositionPlan, row.id)
    assert stored is not None
    assert stored.clip_candidate_id == fixture.selection.candidate.id
    assert stored.render_contract_id == contract.id
    assert stored.transformation_selection_id == contract.transformation_selection_id
    assert stored.selected_plan_id == contract.selected_plan_id


def test_row_accepts_nullable_final_refinement(session: Session, monkeypatch: Any) -> None:
    fixture, contract = _seed_graph(session, monkeypatch)
    candidate = fixture.selection.candidate
    session.add(
        _row(
            candidate,
            contract,
            final_refinement_id=fixture.selection.refinement.id,
        )
    )
    session.flush()


def test_unique_candidate_input_fingerprint(session: Session, monkeypatch: Any) -> None:
    fixture, contract = _seed_graph(session, monkeypatch)
    candidate = fixture.selection.candidate
    session.add(_row(candidate, contract, input_fingerprint="a" * 64, is_current=True))
    session.flush()
    session.add(_row(candidate, contract, input_fingerprint="a" * 64, is_current=False))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_partial_unique_current_plan(session: Session, monkeypatch: Any) -> None:
    fixture, contract = _seed_graph(session, monkeypatch)
    candidate = fixture.selection.candidate
    session.add(_row(candidate, contract, input_fingerprint="b" * 64))
    session.flush()
    session.add(_row(candidate, contract, input_fingerprint="c" * 64))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_non_current_plan_history_is_preserved(session: Session, monkeypatch: Any) -> None:
    fixture, contract = _seed_graph(session, monkeypatch)
    candidate = fixture.selection.candidate
    session.add(_row(candidate, contract, input_fingerprint="d" * 64, is_current=False))
    session.add(_row(candidate, contract, input_fingerprint="e" * 64, is_current=True))
    session.flush()

    current = session.scalars(
        select(VisualCompositionPlan)
        .where(VisualCompositionPlan.clip_candidate_id == candidate.id)
        .where(VisualCompositionPlan.is_current.is_(True))
    ).all()
    assert len(current) == 1
    assert current[0].input_fingerprint == "e" * 64


def test_enum_round_trip(session: Session, monkeypatch: Any) -> None:
    fixture, contract = _seed_graph(session, monkeypatch)
    row = _row(
        fixture.selection.candidate,
        contract,
        status=VisualCompositionStatus.BLOCKED,
        execution_status=VisualCompositionExecutionStatus.COMPLETE,
    )
    session.add(row)
    session.commit()
    row_id = row.id
    session.expunge_all()

    reloaded = session.get(VisualCompositionPlan, row_id)
    assert reloaded is not None
    assert reloaded.status is VisualCompositionStatus.BLOCKED
    assert reloaded.execution_status is VisualCompositionExecutionStatus.COMPLETE
