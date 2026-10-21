"""Stage 6 finalization/readiness critique-binding regressions (F3).

Offline SQLite fixture with fake writer/critic and fake render/QC through the
real persisted materialization executor, finalizer and shared readiness view.
These are application-contract checks only: no production, hosted, TTS, or
genuine acceptance evidence is produced here.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from test_stage6_finalization_gates import _attach_render, _finalize
from test_stage6_vertical_slice import _PROFILE, _persisted_materialization, _stream_probe

from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.components import critique_request_fingerprint
from app.materialization.finalization import (
    FINALIZATION_COMPLETE,
    _accepted_script_blocks,
)
from app.materialization.inputs import MaterializationInputs
from app.materialization.policy import DEFAULT_CONFIG
from app.materialization.providers import (
    CritiqueTarget,
    critique_source_context,
)
from app.materialization.queue import CACHE_VALID, _completed_cache_state
from app.materialization.readiness import evaluate_readiness
from app.materialization.rendering import build_transformed_render_spec
from app.models import ClipMaterialization

_REVIEWABLE = {"READY_FOR_HUMAN_REVIEW", "MATERIALIZED_WITH_CAUTION"}


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _inputs(session: Session, fixture: Any, row: ClipMaterialization) -> MaterializationInputs:
    from app.materialization.service import assemble_inputs

    return assemble_inputs(
        session,
        fixture.stage50.selection.candidate.id,
        profile_key=row.profile_key,
    )


def _bind_render(session: Session, fixture: Any, row: ClipMaterialization) -> Any:
    candidate = fixture.stage50.selection.candidate
    spec = build_transformed_render_spec(
        session,
        candidate,
        materialization_id=row.id,
        expected_output_fingerprint=row.output_fingerprint,
        delivery_profile_key=_PROFILE,
        settings=get_settings(),
        storage=fixture.stage50.storage,
        stream_probe=_stream_probe,
    )
    render = _attach_render(session, fixture, row)
    payload = dict(render.request_payload)
    payload["spec"] = spec.as_dict()
    render.request_payload = payload
    session.flush()
    return render


def _script(row: ClipMaterialization) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoints = copy.deepcopy(dict(row.component_checkpoints or {}))
    script = dict(checkpoints["SCRIPT"])
    return checkpoints, script


def _mutate_critique(row: ClipMaterialization, mutate: Callable[[dict[str, Any]], None]) -> None:
    checkpoints, script = _script(row)
    critique = dict(script["critique"])
    mutate(critique)
    script["critique"] = critique
    checkpoints["SCRIPT"] = script
    row.component_checkpoints = checkpoints


def _remove_critique(row: ClipMaterialization) -> None:
    checkpoints, script = _script(row)
    script.pop("critique", None)
    checkpoints["SCRIPT"] = script
    row.component_checkpoints = checkpoints


def _remove_binding(row: ClipMaterialization) -> None:
    def mutate(critique: dict[str, Any]) -> None:
        critique.pop("request_fingerprint", None)
        critique.pop("reviewed_targets", None)

    _mutate_critique(row, mutate)


def _rewrite_fingerprint(row: ClipMaterialization, inputs: MaterializationInputs) -> None:
    """Recompute the fingerprint over tampered targets (a stale-fact simulation)."""

    def mutate(critique: dict[str, Any]) -> None:
        raw_targets = critique.get("reviewed_targets")
        assert isinstance(raw_targets, list)
        parsed = [CritiqueTarget.from_dict(item) for item in raw_targets]
        assert all(target is not None for target in parsed)
        fingerprint = critique_request_fingerprint(
            [target.as_dict() for target in parsed if target is not None],
            critique_source_context(inputs),
        )
        assert fingerprint is not None
        critique["request_fingerprint"] = fingerprint

    _mutate_critique(row, mutate)


def _assert_not_reviewable(session: Session, fixture: Any, row: ClipMaterialization) -> None:
    session.refresh(row)
    assert row.cache_eligible is False
    assert row.human_review_required is False
    assert row.publication_ready is False
    view = evaluate_readiness(session, fixture.stage50.storage, row)
    assert view.review_ready is False
    assert view.reasons
    assert _completed_cache_state(session, row, settings=get_settings()) != CACHE_VALID


def test_current_bound_critique_finalizes_and_is_review_ready(
    session: Session, monkeypatch: Any
) -> None:
    fixture, row = _persisted_materialization(session, monkeypatch)
    inputs = _inputs(session, fixture, row)
    accepted, _blocks = _accepted_script_blocks(row, inputs, DEFAULT_CONFIG)
    assert accepted is True
    _bind_render(session, fixture, row)

    result = _finalize(session, fixture, row)

    assert result.outcome == FINALIZATION_COMPLETE
    session.refresh(row)
    assert row.cache_eligible is True
    assert row.human_review_required is True
    assert row.semantic_outcome is not None
    assert row.semantic_outcome.value in _REVIEWABLE
    view = evaluate_readiness(session, fixture.stage50.storage, row)
    assert view.review_ready is True, view.reasons


@pytest.mark.parametrize("tamper", ["remove_critique", "remove_binding"])
def test_missing_required_critique_binding_defers(
    session: Session, monkeypatch: Any, tamper: str
) -> None:
    fixture, row = _persisted_materialization(session, monkeypatch)
    if tamper == "remove_critique":
        _remove_critique(row)
    else:
        _remove_binding(row)
    session.flush()
    inputs = _inputs(session, fixture, row)

    accepted, _blocks = _accepted_script_blocks(row, inputs, DEFAULT_CONFIG)
    assert accepted is False
    _bind_render(session, fixture, row)

    result = _finalize(session, fixture, row)

    assert result.outcome != FINALIZATION_COMPLETE
    _assert_not_reviewable(session, fixture, row)


@pytest.mark.parametrize("tamper", ["tampered_copy", "tampered_delivery", "rewritten_binding"])
def test_inconsistent_reviewed_facts_are_not_trusted(
    session: Session, monkeypatch: Any, tamper: str
) -> None:
    fixture, row = _persisted_materialization(session, monkeypatch)
    inputs = _inputs(session, fixture, row)

    def mutate(critique: dict[str, Any]) -> None:
        targets = critique["reviewed_targets"]
        target = dict(targets[0])
        if tamper in {"tampered_copy", "rewritten_binding"}:
            target["copy"] = "Copy that the reviewer never saw."
        elif tamper == "tampered_delivery":
            proposed = dict(target["proposed_delivery"])
            proposed["delivery"] = "NARRATION"
            target["proposed_delivery"] = proposed
        targets[0] = target
        critique["reviewed_targets"] = targets

    _mutate_critique(row, mutate)
    if tamper == "rewritten_binding":
        _rewrite_fingerprint(row, inputs)
    session.flush()

    accepted, _blocks = _accepted_script_blocks(row, inputs, DEFAULT_CONFIG)
    assert accepted is False
    _bind_render(session, fixture, row)

    result = _finalize(session, fixture, row)

    assert result.outcome != FINALIZATION_COMPLETE
    _assert_not_reviewable(session, fixture, row)


def test_committed_assessment_stops_being_cache_after_binding_loss(
    session: Session, monkeypatch: Any
) -> None:
    fixture, row = _persisted_materialization(session, monkeypatch)
    _bind_render(session, fixture, row)
    result = _finalize(session, fixture, row)
    assert result.outcome == FINALIZATION_COMPLETE
    session.refresh(row)
    assert row.cache_eligible is True
    before_ledger = copy.deepcopy(dict(row.call_ledger or {}))
    before_checkpoints = copy.deepcopy(dict(row.component_checkpoints or {}))

    _remove_binding(row)
    session.flush()

    view = evaluate_readiness(session, fixture.stage50.storage, row)
    assert view.review_ready is False
    assert view.reasons
    assert _completed_cache_state(session, row, settings=get_settings()) != CACHE_VALID
    session.refresh(row)
    after_checkpoints = copy.deepcopy(dict(row.component_checkpoints or {}))
    assert dict(row.call_ledger or {}) == before_ledger
    # Reusable writing/accepted copy survives; only the obsolete critique binding
    # was removed by this diagnostic.
    assert after_checkpoints["SCRIPT"]["writing"] == before_checkpoints["SCRIPT"]["writing"]
    assert after_checkpoints["SCRIPT"]["result"] == before_checkpoints["SCRIPT"]["result"]
    assert after_checkpoints["NARRATION"] == before_checkpoints["NARRATION"]
    assert "request_fingerprint" not in after_checkpoints["SCRIPT"]["critique"]
    assert row.lifecycle.value == "COMPLETE"
