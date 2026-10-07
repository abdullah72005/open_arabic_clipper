"""Stage 5.2 handoff regression: authoritative Stage 5.0 output profile."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    Stage51Fixture,
    seed_stage51,
)

from app.composition.handoff import build_stage5_2_handoff
from app.composition.queue import queue_visual_composition
from app.composition.service import execute_visual_composition, get_current_visual_composition
from app.db.base import Base
from app.render.handoff import build_stage5_1_handoff


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.composition.queue._dispatch", lambda *args: None)


def _plan_ready(session: Session, fixture: Stage51Fixture) -> None:
    queue_visual_composition(session, fixture.stage50.selection.candidate)
    row = get_current_visual_composition(session, fixture.stage50.selection.candidate.id)
    assert row is not None
    execute_visual_composition(
        session,
        row.id,
        storage=fixture.stage50.storage,
        settings=fixture.settings,
        display_probe=fixture.display_probe,
        frame_sampler=FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=FakeDetector(),
    )
    session.commit()


def test_stage5_2_handoff_exposes_stage5_0_output_profile(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _plan_ready(session, fixture)
    contract_handoff = build_stage5_1_handoff(session, fixture.stage50.selection.candidate.id)
    assert contract_handoff is not None
    contract_profile = dict(contract_handoff.get("output_profile") or {})
    assert contract_profile, "Stage 5.0 contract must expose an output profile"

    handoff = build_stage5_2_handoff(
        session,
        fixture.stage50.selection.candidate.id,
        display_probe=fixture.display_probe,
        config=fixture.settings.stage51_config(),
    )
    assert handoff is not None
    profile = dict(handoff.get("output_profile") or {})
    assert profile, "Stage 5.2 handoff must not report an empty output profile"
    assert profile.get("width") == 1080
    assert profile.get("height") == 1920
    assert profile.get("profile_key") == contract_profile.get("profile_key")
    assert profile.get("target_frame_rate") == contract_profile.get("target_frame_rate")
    assert handoff.get("stage5_2_implemented") is False
    assert handoff.get("publication_ready") is False
