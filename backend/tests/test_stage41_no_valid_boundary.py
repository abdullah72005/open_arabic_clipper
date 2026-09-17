"""Stage 4.1 provider no-valid boundary-bypass regressions.

Integration-level tests drive parsed provider output through the real
``PlanningService.plan()``/executor path and prove that forbidden provider text
in an explicit ``no_valid_plan`` payload never reaches persistence, checkpoints,
outcome reasons, cache reuse, or the Stage 4.2 handoff.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage41_support import (
    FakePlanningProvider,
    FakeStage41Settings,
    install_stage41_settings,
    make_source_value_plan,
    run_planning,
    seed_stage41,
)

from app.core.enums import (
    PlanSemanticOutcome,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.db.base import Base
from app.transformation.planning.executor import TransformationPlanningExecutor
from app.transformation.planning.handoff import build_stage4_2_handoff
from app.transformation.planning.policy import DEFAULT_CONFIG
from app.transformation.planning.queue import list_plans
from app.transformation.planning.types import PlanProviderPlan

_FORBIDDEN_REASONS = {
    "tts_provider_voice": "Use Gemini voice Charon to read it",
    "rendering": "Add an ffmpeg timeline and a shot list",
    "platform_evasion": "Mirror the video to evade detection by the platform",
    "speaker_imitation": "Make it sound like Morgan Freeman",
}

_LEAK_TOKENS = ("Gemini", "Charon", "ffmpeg", "Mirror", "Freeman")


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _seed(session: Session, settings: FakeStage41Settings, **kwargs: Any) -> tuple[Any, ...]:
    from _pytest.monkeypatch import MonkeyPatch

    install_stage41_settings(MonkeyPatch(), settings)
    return seed_stage41(session, settings=settings, **kwargs)


def _no_valid_plan(strategy: Any, reason: str, **overrides: Any) -> PlanProviderPlan:
    return PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        confidence=0.0,
        no_valid_plan=True,
        no_valid_reason=reason,
        **overrides,
    )


def _leaks(text: str) -> bool:
    return any(token in text for token in _LEAK_TOKENS)


# --- Forbidden no-valid payloads are rejected and never persisted -------------


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "reason", list(_FORBIDDEN_REASONS.values()), ids=list(_FORBIDDEN_REASONS)
)
def test_forbidden_no_valid_reason_rejected_and_never_persisted(
    session: Session, reason: str
) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider([_no_valid_plan(strategy, reason)])
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)

    attempts = plan_set.strategy_attempts or []
    assert attempts
    # Not an explicit decline: a safe invalid/malformed provider result.
    assert attempts[0]["status"] == "INVALID"
    assert attempts[0]["checkpoint"] is None
    assert plan_set.metrics["explicit_no_valid_plan_strategies"] == 0
    assert plan_set.metrics["invalid_malformed_strategies"] == 1
    assert plan_set.cache_eligible is False
    assert list_plans(session, plan_set.id) == []

    persisted = json.dumps(
        {
            "attempts": attempts,
            "outcome_reasons": plan_set.outcome_reasons,
            "provider_evidence": plan_set.provider_evidence,
        }
    )
    assert not _leaks(persisted)

    handoff = build_stage4_2_handoff(session, seed[1].id)
    assert handoff is not None
    assert not _leaks(json.dumps(handoff))


def test_forbidden_planner_notes_in_no_valid_payload_never_persisted(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [
            _no_valid_plan(
                strategy,
                "No grounded substantive value",
                planner_notes="Then use Gemini voice Charon for the read",
            )
        ]
    )
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)
    attempts = plan_set.strategy_attempts or []
    assert attempts[0]["status"] == "INVALID"
    assert not _leaks(json.dumps(plan_set.strategy_attempts))
    assert not _leaks(json.dumps(plan_set.provider_evidence))


# --- Clean no-valid remains a legitimate no-plan outcome ----------------------


def test_clean_no_valid_plan_remains_legitimate(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    reason = "No source-grounded substantive value is available for this strategy"
    provider = FakePlanningProvider([_no_valid_plan(strategy, reason)])
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)

    attempts = plan_set.strategy_attempts or []
    assert attempts[0]["status"] == "NO_VALID_PLAN"
    assert attempts[0]["checkpoint"] is not None
    assert plan_set.planning_outcome is PlanSemanticOutcome.NO_VALID_PLAN_FROM_STRATEGY
    assert plan_set.cache_eligible is True
    assert plan_set.metrics["explicit_no_valid_plan_strategies"] == 1
    assert list_plans(session, plan_set.id) == []


# --- Stale/forbidden existing checkpoints are not reused ----------------------


def test_forbidden_checkpoint_is_not_reused(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [_no_valid_plan(strategy, "No source-grounded substantive value")]
    )
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)
    attempts = copy.deepcopy(plan_set.strategy_attempts or [])
    assert attempts[0]["checkpoint"] is not None

    # Simulate a stale checkpoint persisted by the pre-fix version that carried
    # forbidden provider text.
    attempts[0]["checkpoint"]["result"]["plans"][0]["no_valid_reason"] = (
        "Use Gemini voice Charon to read it"
    )
    plan_set.strategy_attempts = attempts
    session.commit()

    reuse_provider = FakePlanningProvider([])
    executor = TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=reuse_provider,  # type: ignore[arg-type]
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=settings.transformation_planning_semantic_mode(),
        config=DEFAULT_CONFIG,
    )
    executor.execute(plan_set.id, force=True)
    session.refresh(plan_set)

    assert reuse_provider.calls == 0  # checkpoint reuse path, no provider call
    refreshed = plan_set.strategy_attempts or []
    assert refreshed[0]["status"] == "INVALID"
    assert refreshed[0]["checkpoint"] is None
    assert not _leaks(json.dumps(refreshed))
    assert plan_set.cache_eligible is False


def test_valid_accepted_checkpoint_still_reused(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [
            make_source_value_plan(
                str(strategy.id),
                strategy.strategy_key,
                kind=SubstantiveValueKind.AUTHORED_THESIS,
            )
        ]
    )
    plan_set = run_planning(session, settings, seed, provider)
    assert provider.calls == 1

    reuse_provider = FakePlanningProvider([])
    executor = TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=reuse_provider,  # type: ignore[arg-type]
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=settings.transformation_planning_semantic_mode(),
        config=DEFAULT_CONFIG,
    )
    executor.execute(plan_set.id, force=True)
    session.refresh(plan_set)
    assert reuse_provider.calls == 0
    assert len(list_plans(session, plan_set.id)) == 1
