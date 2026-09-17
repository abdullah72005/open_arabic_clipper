"""Stage 4.1 bilingual speaker-boundary closure regressions.

Covers English and Arabic explicit per-video narrator/voice/speaker identity
selection (single- and multi-token), the tightened English grammar that no
longer rejects ordinary prose ("The narrator named several causes"), and the
real service/executor persistence path so forbidden text never reaches
attempts, checkpoints, cache reuse, or the Stage 4.2 handoff.
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
from app.transformation.planning.handoff import build_stage4_2_handoff
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
    REJECT_TTS_SELECTION,
    validate_provider_plan,
)

_REJECTED_EN = (
    "Narrated by Charon",
    "Use Charon as narrator",
    "Set Charon as the speaker",
    "Assign Charon as the voice",
    "Use narrator Charon to explain the claim",
    "Use a voice called Charon",
    "Charon as the narrator",
)
_REJECTED_AR = (
    "استخدم صوت شيرون للسرد",
    "اختر شيرون راوياً",
    "شيرون كراوٍ",
    "عيّن شيرون متحدثاً",
)
_ALLOWED = (
    "The narrator named several causes",
    "Clarify how the narrator frames the argument",
    "Explain the economic model behind the productivity drop",
    "Reference Morgan Freeman's career as context for the claim",
    "The narrator explains the model to viewers",
)
_LEAK_TOKENS = ("Charon", "شيرون", "Morgan", "Freeman")


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _seed(session: Session, settings: FakeStage41Settings, **kwargs: Any) -> tuple[Any, ...]:
    from _pytest.monkeypatch import MonkeyPatch

    install_stage41_settings(MonkeyPatch(), settings)
    return seed_stage41(session, settings=settings, **kwargs)


def _leaks(text: str) -> bool:
    return any(token in text for token in _LEAK_TOKENS)


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


@pytest.mark.parametrize("intent", _REJECTED_EN)  # type: ignore[untyped-decorator]
def test_english_speaker_selection_rejected(session: Session, intent: str) -> None:
    result = _validate_intent(session, intent)
    assert result.plan is None
    assert result.reasons == (REJECT_SPEAKER_SELECTION,)


@pytest.mark.parametrize("intent", _REJECTED_AR)  # type: ignore[untyped-decorator]
def test_arabic_speaker_selection_rejected(session: Session, intent: str) -> None:
    result = _validate_intent(session, intent)
    assert result.plan is None
    assert result.reasons == (REJECT_SPEAKER_SELECTION,)


@pytest.mark.parametrize("intent", _ALLOWED)  # type: ignore[untyped-decorator]
def test_ordinary_prose_allowed(session: Session, intent: str) -> None:
    result = _validate_intent(session, intent)
    assert result.plan is not None


def test_multiword_identity_and_imitation_still_rejected(session: Session) -> None:
    for intent in (
        "Have Morgan Freeman narrate the analysis",
        "In the style of Morgan Freeman",
    ):
        result = _validate_intent(session, intent)
        assert result.plan is None, intent
        assert result.reasons == (REJECT_SPEAKER_SELECTION,), intent
    # Provider-plus-voice remains a TTS-selection rejection.
    tts = _validate_intent(session, "Use Gemini voice Charon")
    assert tts.plan is None
    assert tts.reasons == (REJECT_TTS_SELECTION,)


def _forbidden_no_valid(strategy: Any, reason: str) -> PlanProviderPlan:
    return PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        no_valid_plan=True,
        no_valid_reason=reason,
    )


def test_forbidden_speaker_in_no_valid_payload_never_persisted(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider([_forbidden_no_valid(strategy, "استخدم صوت شيرون للسرد")])
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)

    attempts = plan_set.strategy_attempts or []
    assert attempts[0]["status"] == "INVALID"
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANNING_DEFERRED
    assert plan_set.cache_eligible is False
    assert not _leaks(json.dumps(attempts))
    assert not _leaks(json.dumps(plan_set.outcome_reasons))
    handoff = build_stage4_2_handoff(session, seed[1].id)
    assert handoff is not None
    assert not _leaks(json.dumps(handoff))


def test_forbidden_speaker_in_original_block_never_persisted(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    plan = make_source_value_plan(
        str(strategy.id),
        strategy.strategy_key,
        kind=SubstantiveValueKind.AUTHORED_THESIS,
        intent="Narrated by Charon to explain the claim",
    )
    provider = FakePlanningProvider([plan])
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)

    attempts = plan_set.strategy_attempts or []
    assert attempts[0]["status"] == "INVALID"
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANNING_DEFERRED
    assert plan_set.cache_eligible is False
    assert list_plans(session, plan_set.id) == []
    assert not _leaks(json.dumps(attempts))
    assert not _leaks(json.dumps(plan_set.outcome_reasons))


def test_clean_no_valid_remains_valid_and_cacheable(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [_forbidden_no_valid(strategy, "No source-grounded substantive value is available")]
    )
    plan_set = run_planning(session, settings, seed, provider)
    session.refresh(plan_set)
    attempts = plan_set.strategy_attempts or []
    assert attempts[0]["status"] == "NO_VALID_PLAN"
    assert plan_set.planning_outcome is PlanSemanticOutcome.NO_VALID_PLAN_FROM_STRATEGY
    assert plan_set.cache_eligible is True
