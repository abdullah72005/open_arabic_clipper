"""Stage 4.1 final truthfulness + single-token speaker boundary regressions.

Covers two P1 defects: forbidden/malformed no-valid output must produce a
truthful deferred outcome (never a false ``NO_VALID_PLAN_FROM_STRATEGY``), and
explicit narrator/voice/speaker selection naming a single-token identity must be
rejected. Integration cases drive the real service/executor persistence path.
"""

from __future__ import annotations

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
    NarrationNeed,
    PlanBlockType,
    PlanSemanticOutcome,
    SourceExcerptRole,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.db.base import Base
from app.transformation.planning.inputs import build_planning_inputs
from app.transformation.planning.policy import DEFAULT_CONFIG
from app.transformation.planning.queue import list_plans
from app.transformation.planning.types import (
    NarrationRequirement,
    PlanProviderBlock,
    PlanProviderPlan,
)
from app.transformation.planning.validation import (
    REJECT_SPEAKER_SELECTION,
    validate_provider_plan,
)

_LEAK_TOKENS = ("Charon", "Morgan", "Freeman", "ffmpeg", "Mirror")


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _seed(session: Session, settings: FakeStage41Settings, **kwargs: Any) -> tuple[Any, ...]:
    from _pytest.monkeypatch import MonkeyPatch

    install_stage41_settings(MonkeyPatch(), settings)
    return seed_stage41(session, settings=settings, **kwargs)


def _no_valid(strategy: Any, reason: str) -> PlanProviderPlan:
    return PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        no_valid_plan=True,
        no_valid_reason=reason,
    )


def _leaks(text: str) -> bool:
    return any(token in text for token in _LEAK_TOKENS)


def _valid_plan(strategy: Any) -> PlanProviderPlan:
    return make_source_value_plan(
        str(strategy.id), strategy.strategy_key, kind=SubstantiveValueKind.AUTHORED_THESIS
    )


# --- Finding 1: truthful invalid outcomes ------------------------------------


def test_forbidden_no_valid_only_is_deferred_not_no_valid(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider([_no_valid(strategy, "Use Gemini voice Charon to read it")])
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)

    attempts = plan_set.strategy_attempts or []
    assert attempts[0]["status"] == "INVALID"
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANNING_DEFERRED
    assert plan_set.cache_eligible is False
    assert list_plans(session, plan_set.id) == []
    assert not _leaks(json.dumps(attempts))


def test_clean_no_valid_only_is_no_valid_outcome(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [_no_valid(strategy, "No source-grounded substantive value is available")]
    )
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)

    attempts = plan_set.strategy_attempts or []
    assert attempts[0]["status"] == "NO_VALID_PLAN"
    assert plan_set.planning_outcome is PlanSemanticOutcome.NO_VALID_PLAN_FROM_STRATEGY
    assert plan_set.cache_eligible is True


def test_mixed_clean_and_forbidden_no_valid_is_deferred(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
        second_strategy=(
            TransformationStrategyType.EXPLANATORY,
            "Explain the causal mechanism behind the drop",
        ),
    )
    first, second = seed[4][0], seed[4][1]
    provider = FakePlanningProvider(
        [
            _no_valid(first, "No source-grounded substantive value"),
            _no_valid(second, "Use narrator Charon to explain the claim"),
        ]
    )
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)

    statuses = {attempt["status"] for attempt in (plan_set.strategy_attempts or [])}
    assert statuses == {"NO_VALID_PLAN", "INVALID"}
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANNING_DEFERRED
    assert plan_set.cache_eligible is False
    assert not _leaks(json.dumps(plan_set.strategy_attempts))


def test_valid_plan_plus_invalid_sibling_preserves_valid_plan(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
        second_strategy=(
            TransformationStrategyType.EXPLANATORY,
            "Explain the causal mechanism behind the drop",
        ),
    )
    first, second = seed[4][0], seed[4][1]
    provider = FakePlanningProvider(
        [_valid_plan(first), _no_valid(second, "Use narrator Charon to explain the claim")]
    )
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)

    rows = list_plans(session, plan_set.id)
    assert len(rows) == 1
    assert str(rows[0].strategy_candidate_id) == str(first.id)
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANS_GENERATED
    # An unrelated invalid sibling still blocks a cacheable "all clean" result.
    assert plan_set.cache_eligible is False
    assert not _leaks(json.dumps(plan_set.strategy_attempts))


def test_retry_after_invalid_provider_output_can_succeed(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    bad = FakePlanningProvider([_no_valid(strategy, "Use Gemini voice Charon to read it")])
    plan_set = run_planning(session, settings, seed, bad)
    session.refresh(plan_set)
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANNING_DEFERRED
    assert list_plans(session, plan_set.id) == []

    good = FakePlanningProvider([_valid_plan(strategy)])
    plan_set = run_planning(session, settings, seed, good)
    session.refresh(plan_set)
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANS_GENERATED
    assert len(list_plans(session, plan_set.id)) == 1
    assert plan_set.cache_eligible is True


# --- Finding 2: single-token speaker identity --------------------------------


def _plan_inputs(session: Session, settings: FakeStage41Settings):
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    inputs = build_planning_inputs(
        session,
        seed[1],
        seed[3],
        seed[2],
        settings,
        DEFAULT_CONFIG,  # type: ignore[arg-type]
    )
    return seed, inputs


def _validate_intent(session: Session, intent: str):
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    strategy = inputs.stage40_strategies[0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy["id"]),
        strategy_key=str(strategy["strategy_key"]),
        confidence=0.7,
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                use_full_window=True,
                source_role=SourceExcerptRole.HERO,
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.ORIGINAL_VALUE,
                estimated_duration=4.0,
                substantive_value_kind=SubstantiveValueKind.AUTHORED_THESIS,
                semantic_intent=intent,
                why_unavailable="The excerpt alone does not provide this added dimension",
            ),
        ),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )
    return validate_provider_plan(
        plan,
        strategy,
        inputs,
        DEFAULT_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="fp",
    )


def test_use_narrator_single_token_rejected(session: Session) -> None:
    result = _validate_intent(session, "Use narrator Charon to explain the claim")
    assert result.plan is None
    assert result.reasons == (REJECT_SPEAKER_SELECTION,)


def test_use_voice_called_single_token_rejected(session: Session) -> None:
    result = _validate_intent(session, "Use a voice called Charon")
    assert result.plan is None
    assert result.reasons == (REJECT_SPEAKER_SELECTION,)


def test_single_token_selection_in_planner_notes_rejected(session: Session) -> None:
    seed, inputs = _plan_inputs(session, FakeStage41Settings())
    strategy = inputs.stage40_strategies[0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy["id"]),
        strategy_key=str(strategy["strategy_key"]),
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                use_full_window=True,
                source_role=SourceExcerptRole.HERO,
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.ORIGINAL_VALUE,
                estimated_duration=4.0,
                substantive_value_kind=SubstantiveValueKind.AUTHORED_THESIS,
                semantic_intent="Explain the causal mechanism behind the drop",
                why_unavailable="The excerpt alone does not provide this added dimension",
            ),
        ),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
        planner_notes="Use a voice called Charon for the read",
    )
    result = validate_provider_plan(
        plan,
        strategy,
        inputs,
        DEFAULT_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="fp",
    )
    assert result.plan is None
    assert result.reasons == (REJECT_SPEAKER_SELECTION,)


def test_single_token_selection_in_no_valid_payload_rejected_and_not_persisted(
    session: Session,
) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [_no_valid(strategy, "Use narrator Charon to explain the claim")]
    )
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)
    attempts = plan_set.strategy_attempts or []
    assert attempts[0]["status"] == "INVALID"
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANNING_DEFERRED
    assert not _leaks(json.dumps(attempts))
    assert not _leaks(json.dumps(plan_set.outcome_reasons))


def test_ordinary_narrator_and_model_wording_allowed(session: Session) -> None:
    for intent in (
        "Explain the economic model behind the productivity drop",
        "Clarify how the narrator frames the argument for viewers",
    ):
        result = _validate_intent(session, intent)
        assert result.plan is not None, intent


def test_existing_multi_word_protections_remain(session: Session) -> None:
    assert _validate_intent(session, "Have Morgan Freeman narrate the analysis").reasons == (
        REJECT_SPEAKER_SELECTION,
    )
    assert _validate_intent(session, "In the style of Morgan Freeman").reasons == (
        REJECT_SPEAKER_SELECTION,
    )
