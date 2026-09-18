"""Stage 5.1 service: input resolution, persistence, reads, and freshness."""

from __future__ import annotations

import json
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

from app.composition.policy import Stage51Config, stage51_config_payload
from app.composition.preview import (
    _assert_no_video_arguments,
    _frame_arguments,
    render_preview_pngs,
)
from app.composition.service import (
    PreviewError,
    composition_input_fingerprint,
    execute_visual_composition,
    get_current_visual_composition,
    read_visual_composition,
    read_visual_composition_by_id,
    resolve_planner_inputs,
)
from app.db.base import Base
from app.models.visual_composition_plan import VisualCompositionPlan


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _run(
    session: Session,
    fixture: Stage51Fixture,
    *,
    config: Stage51Config | None = None,
    sampler: FakeFrameSampler | None = None,
) -> VisualCompositionPlan:
    if config is not None:
        fixture.settings.config = config
    candidate = fixture.stage50.selection.candidate
    row = get_current_visual_composition(session, candidate.id)
    if row is None:
        from app.composition.queue import get_or_create_plan_row

        row = get_or_create_plan_row(session, candidate)
    result = execute_visual_composition(
        session,
        row.id,
        storage=fixture.stage50.storage,
        settings=fixture.settings,  # type: ignore[arg-type]
        display_probe=fixture.display_probe,
        frame_sampler=sampler or FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=FakeDetector(),
    )
    assert result is not None
    return result


def test_resolve_planner_inputs_from_current_contract(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    inputs = resolve_planner_inputs(
        session,
        fixture.stage50.selection.candidate.id,
        display_probe=fixture.display_probe,
        config=fixture.settings.stage51_config(),
    )
    assert inputs is not None
    assert inputs.contract_id == str(fixture.contract.row.id)
    assert inputs.spans
    assert all(span.end > span.start for span in inputs.spans)
    assert inputs.caption_input["refinement_id"]
    assert inputs.display_geometry is fixture.display_probe.geometry


def test_resolve_returns_none_without_contract(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    session.delete(fixture.contract.row)
    session.flush()
    assert (
        resolve_planner_inputs(
            session,
            fixture.stage50.selection.candidate.id,
            display_probe=fixture.display_probe,
            config=fixture.settings.stage51_config(),
        )
        is None
    )


def test_execute_persists_ready_current_plan(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    result = _run(session, fixture)
    assert result.plan_ready is True
    assert result.cache_eligible is True
    current = get_current_visual_composition(session, fixture.stage50.selection.candidate.id)
    assert current is not None
    assert current.id == result.id
    assert current.plan_payload["scenes"]
    assert current.plan_payload["ass"]["sha256"]


def test_new_input_fingerprint_versions_row_without_deleting_history(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    first = _run(session, fixture)
    second = _run(session, fixture, config=Stage51Config(analysis_fps=4.0))
    session.expire_all()
    rows = (
        session.query(VisualCompositionPlan)
        .filter(VisualCompositionPlan.clip_candidate_id == first.clip_candidate_id)
        .all()
    )
    assert len(rows) == 2
    assert first.id != second.id
    assert second.is_current is True
    assert first.is_current is False
    current = get_current_visual_composition(session, first.clip_candidate_id)
    assert current is not None and current.id == second.id


def test_read_current_stale_and_unverifiable(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    config = fixture.settings.stage51_config()
    result = _run(session, fixture)

    current = read_visual_composition(
        session,
        result.clip_candidate_id,
        display_probe=fixture.display_probe,
        config=config,
    )
    assert current is not None
    assert current.live_freshness == "CURRENT"
    assert current.effective is True

    # A policy change invalidates the persisted input fingerprint.
    stale = read_visual_composition(
        session,
        result.clip_candidate_id,
        display_probe=fixture.display_probe,
        config=Stage51Config(analysis_fps=4.0),
    )
    assert stale is not None and stale.live_freshness == "STALE" and stale.effective is False

    # A changed source stat is detected without probing geometry.
    fixture.stage50.source_path.write_bytes(b"\x00" * 4096)
    session.flush()
    stat_stale = read_visual_composition(
        session,
        result.clip_candidate_id,
        display_probe=fixture.display_probe,
        config=config,
    )
    assert stat_stale is not None and stat_stale.live_freshness == "STALE"


def test_read_is_unverifiable_when_geometry_is_missing(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    config = fixture.settings.stage51_config()
    result = _run(session, fixture)
    result.plan_payload = {
        key: value for key, value in result.plan_payload.items() if key != "geometry"
    }
    session.flush()
    view = read_visual_composition(
        session,
        result.clip_candidate_id,
        display_probe=fixture.display_probe,
        config=config,
    )
    assert view is not None
    assert view.live_freshness == "UNVERIFIABLE"
    assert view.effective is False


def test_read_is_stale_when_contract_fingerprint_changes(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch)
    config = fixture.settings.stage51_config()
    result = _run(session, fixture)
    fixture.contract.row.input_fingerprint = "changed-contract-fingerprint"
    session.flush()
    view = read_visual_composition(
        session,
        result.clip_candidate_id,
        display_probe=fixture.display_probe,
        config=config,
    )
    assert view is not None and view.live_freshness == "STALE"


def test_read_by_id_returns_historical_row_state(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    old_config = Stage51Config()
    first = _run(session, fixture, config=old_config)
    second = _run(session, fixture, config=Stage51Config(analysis_fps=4.0))

    historical = read_visual_composition_by_id(
        session,
        first.id,
        display_probe=fixture.display_probe,
        config=old_config,
    )
    assert historical is not None
    assert historical.row.id == first.id
    assert historical.row.is_current is False
    assert historical.effective is False
    assert historical.live_freshness == "CURRENT"

    current_view = read_visual_composition_by_id(
        session,
        second.id,
        display_probe=fixture.display_probe,
        config=Stage51Config(analysis_fps=4.0),
    )
    assert current_view is not None and current_view.effective is True
    assert (
        read_visual_composition_by_id(
            session,
            "00000000-0000-0000-0000-000000000000",
            display_probe=fixture.display_probe,
            config=old_config,
        )
        is None
    )


def test_tts_and_publishing_changes_do_not_invalidate(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    config = fixture.settings.stage51_config()
    inputs = resolve_planner_inputs(
        session,
        fixture.stage50.selection.candidate.id,
        display_probe=fixture.display_probe,
        config=config,
    )
    assert inputs is not None
    before = composition_input_fingerprint(inputs, config)

    fixture.settings.tts_provider = "elevenlabs"  # type: ignore[attr-defined]
    fixture.settings.publishing_schedule = "nightly"  # type: ignore[attr-defined]
    after = composition_input_fingerprint(inputs, config)
    assert before == after

    serialized = json.dumps(stage51_config_payload(config)).casefold()
    assert "tts" not in serialized
    assert "publish" not in serialized


def test_preview_is_gated_and_never_produces_video(session: Session, monkeypatch: Any) -> None:
    fixture = seed_stage51(session, monkeypatch)
    config = fixture.settings.stage51_config()
    inputs = resolve_planner_inputs(
        session,
        fixture.stage50.selection.candidate.id,
        display_probe=fixture.display_probe,
        config=config,
    )
    assert inputs is not None
    with pytest.raises(PreviewError):
        render_preview_pngs(
            source_path=fixture.stage50.source_path,
            inputs=inputs,
            plan_payload={},
            config=Stage51Config(preview_enabled=False),
            storage=fixture.stage50.storage,
        )
    with pytest.raises(PreviewError):
        render_preview_pngs(
            source_path=fixture.stage50.source_path,
            inputs=inputs,
            plan_payload={},
            config=Stage51Config(preview_enabled=True),
            storage=fixture.stage50.storage,
            ffmpeg_binary="definitely-not-ffmpeg-xyz",
        )
    arguments = _frame_arguments(
        executable="ffmpeg",
        source_path=fixture.stage50.source_path,
        source_time=21.0,
        output_path=fixture.stage50.source_path.parent / "preview-0000.png",
        ass_filename=None,
    )
    _assert_no_video_arguments(arguments)
    assert arguments[-1].endswith(".png")
    joined = " ".join(arguments).casefold()
    for forbidden in ("libx264", "aac", "loudnorm"):
        assert forbidden not in joined
