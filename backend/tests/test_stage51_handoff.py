"""Stage 5.1 -> Stage 5.2 handoff truthfulness and readiness flags."""

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
from app.composition.queue import get_or_create_plan_row
from app.composition.service import execute_visual_composition
from app.db.base import Base


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _run(session: Session, fixture: Stage51Fixture) -> None:
    row = get_or_create_plan_row(session, fixture.stage50.selection.candidate)
    result = execute_visual_composition(
        session,
        row.id,
        storage=fixture.stage50.storage,
        settings=fixture.settings,  # type: ignore[arg-type]
        display_probe=fixture.display_probe,
        frame_sampler=FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=FakeDetector(),
    )
    assert result is not None and result.plan_ready is True


def test_handoff_exposes_plan_without_render_or_publication_readiness(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _run(session, fixture)
    handoff = build_stage5_2_handoff(
        session,
        fixture.stage50.selection.candidate.id,
        display_probe=fixture.display_probe,
        config=fixture.settings.stage51_config(),
    )
    assert handoff is not None
    assert handoff["plan"]["effective"] is True
    assert handoff["plan"]["live_freshness"] == "CURRENT"
    assert handoff["final_timeline_frozen"] is False
    assert handoff["publication_ready"] is False
    assert handoff["render_ready"] is False
    assert handoff["stage5_2_implemented"] is False
    assert handoff["stage6_implemented"] is False
    assert handoff["blocks"]
    assert handoff["bound_source_spans"]
    assert handoff["scenes"]
    assert handoff["captions"]["events"]
    assert handoff["ass"]["sha256"]
    assert handoff["materialization"]["slots"]
    assert handoff["safe_zone"]
    assert handoff["protection_markers"]
    assert handoff["source_media"]["managed_relative_path"]
    assert handoff["display_geometry"]["display_width"] == 1920
    assert handoff["readiness"]["stage5_2_handoff_eligible"] is True
    assert handoff["readiness"]["source_framing_ready"] is True
    assert handoff["readiness"]["source_captions_ready"] is True


def test_handoff_fails_closed_when_plan_is_stale(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    _run(session, fixture)
    fixture.stage50.source_path.write_bytes(b"\x00" * 4096)
    session.flush()
    handoff = build_stage5_2_handoff(
        session,
        fixture.stage50.selection.candidate.id,
        display_probe=fixture.display_probe,
        config=fixture.settings.stage51_config(),
    )
    assert handoff is not None
    assert handoff["plan"]["effective"] is False
    assert handoff["plan"]["live_freshness"] == "STALE"
    assert handoff["readiness"]["stage5_2_handoff_eligible"] is False
    assert handoff["readiness"]["source_framing_ready"] is False
    assert handoff["readiness"]["source_captions_ready"] is False
    assert handoff["render_ready"] is False
    assert handoff["publication_ready"] is False


def test_handoff_without_plan_reports_no_plan(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    handoff = build_stage5_2_handoff(
        session,
        fixture.stage50.selection.candidate.id,
        display_probe=fixture.display_probe,
        config=fixture.settings.stage51_config(),
    )
    assert handoff is not None
    assert handoff["plan"] is None
    assert handoff["reason"] == "NO_VISUAL_COMPOSITION_PLAN"
    assert handoff["readiness"]["stage5_2_handoff_eligible"] is False
    assert handoff["stage5_2_implemented"] is False


def test_handoff_returns_none_for_unknown_candidate(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    assert (
        build_stage5_2_handoff(
            session,
            "00000000-0000-0000-0000-000000000000",
            display_probe=fixture.display_probe,
            config=fixture.settings.stage51_config(),
        )
        is None
    )
