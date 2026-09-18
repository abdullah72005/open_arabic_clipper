"""Stage 5.0 render-contract model constraints and migration wiring."""

from __future__ import annotations

import importlib.util
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import CandidateDisposition, RenderContractStatus
from app.db.base import Base
from app.models import ClipCandidate, RenderContract, SourceVideo


def test_migration_revision_chain() -> None:
    path = (
        Path(__file__).parents[1]
        / "alembic"
        / "versions"
        / "20260918_0020_stage_5_0_render_contracts.py"
    )
    specification = importlib.util.spec_from_file_location("stage_5_0_migration", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert module.revision == "20260918_0020"
    assert module.down_revision == "20260918_0019"


def _seed_candidate(session: Session) -> ClipCandidate:
    source = SourceVideo(source_uri="/tmp/source.mp4", content_hash=uuid.uuid4().hex)
    session.add(source)
    session.flush()
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key=f"c-{uuid.uuid4().hex}",
        start_time=0.0,
        end_time=1.0,
        start_segment_index=0,
        end_segment_index=0,
        segment_indexes=[0],
        disposition=CandidateDisposition.CANDIDATE,
    )
    session.add(candidate)
    session.flush()
    return candidate


def _row(candidate: ClipCandidate, **overrides: object) -> RenderContract:
    values: dict[str, object] = {
        "source_video_id": candidate.source_video_id,
        "clip_candidate_id": candidate.id,
        "status": RenderContractStatus.READY_FOR_RENDER_PLANNING,
        "is_current": True,
        "contract_ready": True,
        "input_fingerprint": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return RenderContract(**values)  # type: ignore[arg-type]


def test_contract_ready_coupling_constraint(sqlite_engine: Engine) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        candidate = _seed_candidate(session)
        # contract_ready True requires an executable status.
        session.add(_row(candidate, status=RenderContractStatus.BLOCKED, contract_ready=True))
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()


def test_non_ready_contract_persists(sqlite_engine: Engine) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        candidate = _seed_candidate(session)
        session.add(
            _row(
                candidate,
                status=RenderContractStatus.SOURCE_MEDIA_UNAVAILABLE,
                contract_ready=False,
            )
        )
        session.flush()


def test_partial_unique_current_contract(sqlite_engine: Engine) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        candidate = _seed_candidate(session)
        session.add(_row(candidate, input_fingerprint="a" * 64))
        session.flush()
        session.add(_row(candidate, input_fingerprint="b" * 64))
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()


def test_unique_candidate_input_fingerprint(sqlite_engine: Engine) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        candidate = _seed_candidate(session)
        session.add(_row(candidate, input_fingerprint="c" * 64, is_current=True))
        session.flush()
        session.add(_row(candidate, input_fingerprint="c" * 64, is_current=False))
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()
