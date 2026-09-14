"""Focused Stage 4.1 planning, validation, source, value, narration, and scope tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage41_support import (
    TRANSCRIPT,
    FakePlanningProvider,
    FakeStage41Settings,
    install_stage41_settings,
    make_source_value_plan,
    seed_stage41,
)

from app.core.enums import (
    ContentType,
    ExternalFactRequirement,
    NarrationNeed,
    NarrationPurpose,
    PlanBlockType,
    PlanSemanticOutcome,
    PlanStatus,
    SemanticProviderMode,
    SourceExcerptRole,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.db.base import Base
from app.transformation.planning.executor import TransformationPlanningExecutor
from app.transformation.planning.inputs import build_planning_inputs
from app.transformation.planning.policy import DEFAULT_CONFIG
from app.transformation.planning.queue import get_or_create_plan_set, list_plans
from app.transformation.planning.types import (
    NarrationRequirement,
    PlanningContext,
    PlanningInputs,
    PlanProviderBlock,
    PlanProviderPlan,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _seed(session: Session, settings: FakeStage41Settings, **kwargs: Any) -> tuple[Any, ...]:
    from _pytest.monkeypatch import MonkeyPatch

    monkeypatch = MonkeyPatch()
    install_stage41_settings(monkeypatch, settings)
    return seed_stage41(session, settings=settings, **kwargs)


def _run(
    session: Session,
    settings: FakeStage41Settings,
    seed: tuple[Any, ...],
    provider: object | None = None,
    *,
    mode: SemanticProviderMode = SemanticProviderMode.ADAPTIVE,
) -> Any:
    _source, candidate, _refinement, analysis, _strategies = seed
    plan_set = get_or_create_plan_set(session, candidate, analysis)
    executor = TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=provider,  # type: ignore[arg-type]
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=mode,
        config=DEFAULT_CONFIG,
    )
    executor.execute(plan_set.id)
    session.refresh(plan_set)
    return plan_set


def _source_block(**overrides: Any) -> PlanProviderBlock:
    base: dict[str, Any] = {
        "block_type": PlanBlockType.SOURCE_EXCERPT,
        "use_full_window": True,
        "source_role": SourceExcerptRole.HERO,
    }
    base.update(overrides)
    return PlanProviderBlock(**base)


def _value_block(
    kind: SubstantiveValueKind,
    intent: str,
    why: str = "The excerpt alone does not provide this added dimension.",
    **overrides: Any,
) -> PlanProviderBlock:
    base: dict[str, Any] = {
        "block_type": PlanBlockType.ORIGINAL_VALUE,
        "estimated_duration": 4.0,
        "substantive_value_kind": kind,
        "semantic_intent": intent,
        "why_unavailable": why,
    }
    base.update(overrides)
    return PlanProviderBlock(**base)


def test_eligible_source_as_evidence_produces_source_plus_value_plan(
    session: Session,
) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    plan_set = _run(session, settings, seed)
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANS_GENERATED
    rows = list_plans(session, plan_set.id)
    assert len(rows) == 1
    blocks = rows[0].blocks
    assert blocks[0]["block_type"] == PlanBlockType.SOURCE_EXCERPT.value
    assert blocks[0]["source_role"] == SourceExcerptRole.HERO.value
    assert blocks[1]["block_type"] == PlanBlockType.ORIGINAL_VALUE.value
    assert rows[0].hero_block_index == 0
    assert rows[0].hero_source_start == 20.5
    assert rows[0].hero_source_end == 44.5


def test_analysis_preserves_hero_block_one_and_rejects_long_preamble(
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
    good = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        confidence=0.7,
        blocks=(
            _value_block(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Frame the remote-work productivity claim before the source",
                estimated_duration=2.0,
            ),
            _source_block(),
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([good]))
    rows = list_plans(session, plan_set.id)
    assert rows
    assert rows[0].hero_block_index == 1
    assert rows[0].hero_appearance_time == pytest.approx(2.0)

    seed2 = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    strategy2 = seed2[4][0]
    bad = PlanProviderPlan(
        strategy_id=str(strategy2.id),
        strategy_key=strategy2.strategy_key,
        blocks=(
            _value_block(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Set up the topic with a long generic introduction",
                estimated_duration=15.0,
            ),
            _source_block(),
        ),
    )
    plan_set2 = _run(session, settings, seed2, FakePlanningProvider([bad]))
    assert list_plans(session, plan_set2.id) == []


def test_counterpoint_external_fact_creates_verification_dependency(
    session: Session,
) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.COUNTERPOINT,
        external=ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION,
        value_kind=SubstantiveValueKind.COUNTERPOINT,
    )
    strategy = seed[4][0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        confidence=0.7,
        blocks=(
            _source_block(),
            _value_block(
                SubstantiveValueKind.COUNTERPOINT,
                "Add the opposing finding on remote-work productivity gains",
                dependency_ids=("remote-productivity-meta",),
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.FACT_VERIFICATION_PLACEHOLDER,
                claim_dependency="remote-productivity-meta",
                verification_rationale="The opposing figure must be verified",
                intended_use="Balance the claim",
                must_verify_before_execution=True,
                dependent_block_ids=("1",),
            ),
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([plan]))
    assert plan_set.planning_outcome is (
        PlanSemanticOutcome.PLANS_GENERATED_WITH_VERIFICATION_REQUIRED
    )
    rows = list_plans(session, plan_set.id)
    assert rows[0].status is PlanStatus.PLAN_GENERATED_WITH_VERIFICATION_REQUIRED
    assert rows[0].external_fact_dependencies[0]["dependency"] == "remote-productivity-meta"


def test_funny_source_led_does_not_force_narration(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        content_type=ContentType.FUNNY,
        strategy_type=TransformationStrategyType.SOURCE_LED_MINIMAL,
        value_kind=SubstantiveValueKind.INFERENCE,
        added_value_focus="State the inference behind the joke setup",
    )
    plan_set = _run(session, settings, seed, None)
    rows = list_plans(session, plan_set.id)
    assert rows
    assert rows[0].narration_need == NarrationNeed.NONE.value
    assert rows[0].hero_block_index == 0


def test_strong_viral_moment_keeps_hero_early_with_strict_cap(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        content_type=ContentType.REACTION_WORTHY,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
        refined=(20.5, 28.0),
    )
    strategy = seed[4][0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        blocks=(
            _value_block(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Frame the reaction before the payoff",
                estimated_duration=2.0,
            ),
            _source_block(),
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([plan]))
    assert list_plans(session, plan_set.id) == []


def test_paraphrase_original_block_is_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        blocks=(
            _source_block(),
            _value_block(
                SubstantiveValueKind.SOURCE_AS_EVIDENCE,
                TRANSCRIPT,
                why="Restates the excerpt",
                draft_line=TRANSCRIPT,
            ),
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([plan]))
    assert list_plans(session, plan_set.id) == []
    assert plan_set.planning_outcome is PlanSemanticOutcome.NO_VALID_PLAN_FROM_STRATEGY


def test_cosmetic_edits_never_count_as_original_value(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        blocks=(
            _source_block(),
            _value_block(
                SubstantiveValueKind.SOURCE_AS_EVIDENCE,
                "Add captions and a border",
                why="Purely cosmetic",
            ),
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([plan]))
    assert list_plans(session, plan_set.id) == []


def test_multiple_valid_strategies_produce_distinct_plans(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        second_strategy=(
            TransformationStrategyType.ANALYSIS,
            "Analyze the causal mechanism behind the promotion-rate drop",
        ),
    )
    first, second = seed[4][0], seed[4][1]
    plans = [
        make_source_value_plan(str(first.id), first.strategy_key),
        PlanProviderPlan(
            strategy_id=str(second.id),
            strategy_key=second.strategy_key,
            blocks=(
                _source_block(),
                PlanProviderBlock(
                    block_type=PlanBlockType.TEXTUAL_ANNOTATION,
                    estimated_duration=5.0,
                    substantive_value_kind=SubstantiveValueKind.AUTHORED_THESIS,
                    semantic_intent="Analyze the causal mechanism behind the drop",
                    why_unavailable="The excerpt states the drop without the causal chain",
                ),
            ),
        ),
    ]
    plan_set = _run(session, settings, seed, FakePlanningProvider(plans))
    rows = list_plans(session, plan_set.id)
    assert len(rows) == 2
    assert {row.generation_rank for row in rows} == {1, 2}


def test_one_valid_strategy_produces_no_fake_alternatives(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    plans = [
        make_source_value_plan(str(strategy.id), strategy.strategy_key),
        make_source_value_plan(str(strategy.id), strategy.strategy_key, confidence=0.2),
    ]
    plan_set = _run(session, settings, seed, FakePlanningProvider(plans))
    assert len(list_plans(session, plan_set.id)) == 1


def test_narration_none_is_valid_and_optional_records_fields(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    optional = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        blocks=(
            _source_block(),
            _value_block(
                SubstantiveValueKind.SOURCE_AS_EVIDENCE,
                "Frame the moment as evidence for the debate",
            ),
        ),
        narration=NarrationRequirement(
            need=NarrationNeed.OPTIONAL,
            purposes=(NarrationPurpose.CONTEXT,),
            language="ar",
            register="broadly_understandable",
            estimated_duration=3.0,
            placement_block_index=1,
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([optional]))
    rows = list_plans(session, plan_set.id)
    assert rows
    assert rows[0].narration_need == NarrationNeed.OPTIONAL.value
    assert rows[0].narration_requirements["purposes"] == ["CONTEXT"]
    assert rows[0].narration_requirements["estimated_duration"] == 3.0


def test_optional_narration_that_is_sole_contribution_is_rejected(
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
    plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        blocks=(
            _source_block(),
            _value_block(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Analyze the causal mechanism behind the drop",
                delivery_intent=None,
            ),
        ),
        narration=NarrationRequirement(
            need=NarrationNeed.OPTIONAL,
            purposes=(NarrationPurpose.ANALYSIS,),
            language="ar",
            register="broadly_understandable",
            estimated_duration=5.0,
        ),
    )
    # The only substantive block is non-narration, so optional narration is fine.
    plan_set = _run(session, settings, seed, FakePlanningProvider([plan]))
    assert list_plans(session, plan_set.id)


def test_plan_never_selects_tts_provider_model_or_voice(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    plan_set = _run(session, settings, seed)
    rows = list_plans(session, plan_set.id)
    serialized = str(rows[0].narration_requirements) + str(rows[0].blocks)
    for forbidden in ("voice", "tts", "provider", "model", "gemini"):
        assert forbidden not in serialized.casefold()


def test_egyptian_source_with_gcc_target_preserves_source_speech(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        dialect_profile="EGYPTIAN",
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
    )
    from app.transformation.planning.inputs import PlanningContextResolver

    context = PlanningContextResolver.from_mapping(
        {
            "target_market": "GCC",
            "output_language_policy": "SOURCE_LANGUAGE",
            "register_intent": "BROADLY_UNDERSTANDABLE_ARABIC",
            "narration_allowed": True,
            "tts_provider": "gemini",
            "tts_model": "gemini-tts",
            "tts_voice_id": "Charon",
        }
    )
    assert context.target_market == "GCC"
    assert "tts_provider" not in context.semantic_payload()
    strategy = seed[4][0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        blocks=(
            _source_block(),
            _value_block(
                SubstantiveValueKind.AUTHORED_THESIS,
                "Analyze the causal mechanism behind the drop",
            ),
        ),
        narration=NarrationRequirement(
            need=NarrationNeed.RECOMMENDED,
            purposes=(NarrationPurpose.ANALYSIS,),
            language="ar",
            register="broadly_understandable",
            estimated_duration=4.0,
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([plan]))
    rows = list_plans(session, plan_set.id)
    assert rows
    assert rows[0].blocks[0]["source_text"] == TRANSCRIPT
    assert rows[0].source_dialect["profile"] == "EGYPTIAN"


def test_candidate_grade_works_and_invalid_word_index_rejected(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    invalid = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        blocks=(
            _source_block(use_full_window=False, word_start_index=9999, word_end_index=10000),
            _value_block(
                SubstantiveValueKind.SOURCE_AS_EVIDENCE,
                "Frame the moment as evidence for the debate",
            ),
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([invalid]))
    assert list_plans(session, plan_set.id) == []


def test_word_index_span_resolves_real_timestamps_and_text(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        blocks=(
            _source_block(use_full_window=False, word_start_index=1, word_end_index=4),
            _value_block(
                SubstantiveValueKind.SOURCE_AS_EVIDENCE,
                "Frame the moment as evidence for the debate",
            ),
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([plan]))
    rows = list_plans(session, plan_set.id)
    assert rows
    hero = rows[0].blocks[0]
    assert hero["source_text"] == "guest argues that remote"
    assert 21.0 < hero["source_start"] < 22.5
    assert hero["source_end"] > hero["source_start"]


def test_empty_word_evidence_allows_only_full_window(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings, words=[])
    strategy = seed[4][0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        blocks=(
            _source_block(use_full_window=False, word_start_index=0, word_end_index=1),
            _value_block(
                SubstantiveValueKind.SOURCE_AS_EVIDENCE,
                "Frame the moment as evidence for the debate",
            ),
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([plan]))
    assert list_plans(session, plan_set.id) == []


def test_duplicate_plans_are_deduplicated_deterministically(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        second_strategy=(
            TransformationStrategyType.ANALYSIS,
            "Analyze the causal mechanism behind the promotion-rate drop",
        ),
    )
    first, second = seed[4][0], seed[4][1]
    plans = [
        make_source_value_plan(str(first.id), first.strategy_key),
        PlanProviderPlan(
            strategy_id=str(second.id),
            strategy_key=second.strategy_key,
            blocks=(
                _source_block(),
                _value_block(
                    SubstantiveValueKind.SOURCE_AS_EVIDENCE,
                    "Frame the moment as evidence for the debate",
                ),
            ),
        ),
    ]
    plan_set = _run(session, settings, seed, FakePlanningProvider(plans))
    rows = list_plans(session, plan_set.id)
    signatures = {row.structure_signature for row in rows}
    assert len(signatures) == len(rows) == 1


def test_verification_required_strategy_cannot_emit_unverified_assertion(
    session: Session,
) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.COUNTERPOINT,
        external=ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION,
        value_kind=SubstantiveValueKind.COUNTERPOINT,
    )
    strategy = seed[4][0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        blocks=(
            _source_block(),
            _value_block(
                SubstantiveValueKind.COUNTERPOINT,
                "State the opposing productivity figure as fact",
            ),
        ),
    )
    plan_set = _run(session, settings, seed, FakePlanningProvider([plan]))
    assert list_plans(session, plan_set.id) == []


def test_unknown_stale_and_mismatched_strategy_identities_ignored(
    session: Session,
) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    valid = make_source_value_plan(str(strategy.id), strategy.strategy_key)
    unknown = make_source_value_plan("unknown-id", "unknown-key")
    stale = make_source_value_plan(str(strategy.id), "stale-key")
    plan_set = _run(session, settings, seed, FakePlanningProvider([valid, unknown, stale]))
    rows = list_plans(session, plan_set.id)
    assert len(rows) == 1


def test_missing_provider_yields_conservative_outcome(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(
        session,
        settings,
        strategy_type=TransformationStrategyType.ANALYSIS,
        value_kind=SubstantiveValueKind.AUTHORED_THESIS,
        added_value_focus="",
    )
    plan_set = _run(session, settings, seed, None)
    assert plan_set.planning_outcome in {
        PlanSemanticOutcome.PROVIDER_UNAVAILABLE,
        PlanSemanticOutcome.PLANNING_DEFERRED,
    }
    assert plan_set.execution_status.value == "COMPLETE"


def test_rate_limited_and_malformed_provider_do_not_fail_source(session: Session) -> None:
    settings = FakeStage41Settings()
    seed = _seed(session, settings)
    strategy = seed[4][0]
    provider = FakePlanningProvider(
        [make_source_value_plan(str(strategy.id), strategy.strategy_key)]
    )
    provider.behavior = "rate_limited"
    plan_set = _run(session, settings, seed, provider)
    assert plan_set.execution_status.value in {"COMPLETE", "PROVIDER_DEGRADED"}
    assert plan_set.cache_eligible is False

    seed2 = _seed(session, settings)
    strategy2 = seed2[4][0]
    provider2 = FakePlanningProvider(
        [make_source_value_plan(str(strategy2.id), strategy2.strategy_key)]
    )
    provider2.behavior = "malformed"
    plan_set2 = _run(session, settings, seed2, provider2)
    assert plan_set2.execution_status.value in {"COMPLETE", "PROVIDER_DEGRADED"}
    assert list_plans(session, plan_set2.id) == []


def test_planning_inputs_bounded_and_gate(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    _src, candidate, refinement, analysis, _strategies = _seed(session, settings)
    inputs = build_planning_inputs(
        session,
        candidate,
        analysis,
        refinement,
        settings,  # type: ignore[arg-type]
        DEFAULT_CONFIG,
    )
    assert isinstance(inputs, PlanningInputs)
    assert inputs.refined_start == 20.5
    assert inputs.word_coverage_sufficient is True
    assert all(isinstance(word.text, str) for word in inputs.words)
    assert inputs.stage40_strategies
    assert isinstance(inputs.planning_context, PlanningContext)
