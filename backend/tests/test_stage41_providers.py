"""Focused Stage 4.1 provider parsing, Gemini adapter, and local adapter tests."""

from __future__ import annotations

from collections.abc import Iterator

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

from app.core.enums import TransformationStrategyType
from app.db.base import Base
from app.transformation.planning.gemini import GeminiPlanningProvider
from app.transformation.planning.local import LocalPlanningProvider
from app.transformation.planning.policy import (
    GEMINI_ROUTINE_MODEL,
    GEMINI_STRONG_MODEL,
    is_complex_strategy,
    strategy_value_kinds,
)
from app.transformation.planning.providers import (
    DeterministicPlanningProvider,
    PlanningProviderError,
    PlanningRequest,
    deserialize_provider_result,
    parse_plan_results,
    planning_prompt_hash,
    serialize_provider_result,
)
from app.transformation.planning.queue import list_plans
from app.transformation.planning.types import PlanProviderResult


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _request(key: str = "k", strategy_id: str = "id") -> PlanningRequest:
    return PlanningRequest(
        strategy_id=strategy_id,
        strategy_key=key,
        strategy_type="ANALYSIS",
        intensity="MODERATE",
        direction_summary="d",
        added_value_focus="f",
        substantive_value_kind="AUTHORED_THESIS",
        preservation_requirements=(),
        external_verification_requirement="NOT_REQUIRED",
        verification_requirements=(),
        content_type="ANALYSIS",
        source_moment_structure="CLAIM",
        dialect_profile="EGYPTIAN",
        code_switch_tokens=(),
        target_market="UNSPECIFIED",
        output_language_policy="SOURCE_LANGUAGE",
        register_intent="SOURCE_COMPATIBLE",
        narration_allowed=True,
        max_blocks=8,
        strict_hero_cap_seconds=1.5,
        refined_transcript="text",
        refined_start=0.0,
        refined_end=10.0,
        context_text="",
        idea_summary="",
        topic_summary="",
        hooks=(),
        words=(),
    )


def test_parser_isolates_malformed_and_unknown_identities() -> None:
    content = {
        "plans": [
            {
                "strategy_id": "id",
                "strategy_key": "k",
                "blocks": [
                    {
                        "block_type": "SOURCE_EXCERPT",
                        "source_role": "HERO",
                        "use_full_window": True,
                    },
                    {"block_type": "NOT_A_BLOCK"},
                    {
                        "block_type": "ORIGINAL_VALUE",
                        "substantive_value_kind": "AUTHORED_THESIS",
                        "semantic_intent": "Analyze the mechanism",
                        "why_unavailable": "Not in source",
                        "estimated_duration": 3,
                    },
                ],
            },
            {"strategy_id": "other", "strategy_key": "unknown"},
            {"strategy_id": "wrong", "strategy_key": "k"},
        ]
    }
    results = parse_plan_results(content, [_request()])
    assert set(results) == {"k"}
    plan = results["k"].plans[0]
    assert len(plan.blocks) == 2
    assert plan.blocks[0].block_type.value == "SOURCE_EXCERPT"


def test_parser_raises_on_non_list_plans() -> None:
    with pytest.raises(PlanningProviderError):
        parse_plan_results({"plans": "nope"}, [_request()])


def test_serialize_deserialize_round_trip() -> None:
    plan = make_source_value_plan("id", "k")
    result = PlanProviderResult(plans=(plan,), confidence=0.7)
    payload = serialize_provider_result(result)
    restored = deserialize_provider_result(payload, _request())
    assert restored is not None
    assert restored.plans[0].strategy_key == "k"
    assert restored.plans[0].blocks[0].block_type.value == "SOURCE_EXCERPT"


def test_deterministic_provider_makes_no_results() -> None:
    provider = DeterministicPlanningProvider()
    assert provider.plan([_request()]) == {}
    assert "prompt_hash" in provider.runtime_identity()


def test_gemini_provider_identity_excludes_key_and_scrubs_on_release() -> None:
    provider = GeminiPlanningProvider(
        api_key="super-secret-key",
        routine_model=GEMINI_ROUTINE_MODEL,
        strong_model=GEMINI_STRONG_MODEL,
    )
    identity = provider.runtime_identity()
    assert "super-secret-key" not in str(identity)
    assert identity["routine_model"] == GEMINI_ROUTINE_MODEL
    assert identity["strong_model"] == GEMINI_STRONG_MODEL
    provider.release()
    assert provider._api_key is None


def test_gemini_provider_missing_key_fails_sanitized() -> None:
    provider = GeminiPlanningProvider(api_key=None)
    with pytest.raises(PlanningProviderError) as error:
        provider.plan([_request()], "ROUTINE")
    assert "MISSING_KEY" in error.value.category or error.value.category == "MISSING_KEY"


def test_gemini_provider_uses_client_factory_and_parses_response() -> None:
    class _Models:
        def generate_content(self, **kwargs: object) -> object:
            return type(
                "_Response",
                (),
                {
                    "parsed": {"plans": [{"strategy_id": "id", "strategy_key": "k", "blocks": []}]},
                    "usage_metadata": None,
                    "prompt_feedback": None,
                    "candidates": [type("_C", (), {"finish_reason": None})()],
                },
            )()

    class _Client:
        def __init__(self) -> None:
            self.models = _Models()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    client = _Client()
    provider = GeminiPlanningProvider(api_key="key", client_factory=lambda: client)
    results = provider.plan([_request()], "ROUTINE")
    assert results["k"].plans[0].strategy_id == "id"
    provider.release()
    assert client.closed is True


def test_local_provider_builds_planning_prompt_and_parses() -> None:
    captured: dict[str, bytes] = {}

    def fake_request(
        method: str, url: str, body: bytes | None, headers: dict[str, str], timeout: float
    ) -> bytes:
        captured["body"] = body or b""
        import json

        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "plans": [
                                        {"strategy_id": "id", "strategy_key": "k", "blocks": []}
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
        ).encode()

    provider = LocalPlanningProvider(
        base_url="http://ollama:11434", model="qwen3.5:4b", request=fake_request
    )
    results = provider.plan([_request()], "ROUTINE")
    assert results["k"].plans[0].strategy_key == "k"
    assert b"plans_requested" in captured["body"]
    assert planning_prompt_hash()


def test_strategy_value_kind_and_complex_routing_tables() -> None:
    from app.core.enums import TransformationIntensity

    assert strategy_value_kinds(TransformationStrategyType.COUNTERPOINT)
    assert is_complex_strategy(
        TransformationStrategyType.ANALYSIS, TransformationIntensity.MODERATE, False
    )
    assert not is_complex_strategy(
        TransformationStrategyType.SOURCE_AS_EVIDENCE, TransformationIntensity.MINIMAL, False
    )


def test_batching_uses_at_most_two_calls_and_one_per_tier(session: Session) -> None:
    settings = FakeStage41Settings()
    from _pytest.monkeypatch import MonkeyPatch

    install_stage41_settings(MonkeyPatch(), settings)
    seed = seed_stage41(
        session,
        settings=settings,
        strategy_type=TransformationStrategyType.SOURCE_AS_EVIDENCE,
        second_strategy=(
            TransformationStrategyType.ANALYSIS,
            "Analyze the causal mechanism behind the promotion-rate drop",
        ),
    )
    first, second = seed[4][0], seed[4][1]
    plans = [
        make_source_value_plan(str(first.id), first.strategy_key),
        make_source_value_plan(str(second.id), second.strategy_key),
    ]
    provider = FakePlanningProvider(plans)
    plan_set = run_planning(session, settings, seed, provider)
    assert provider.calls <= 2
    assert len(provider.tiers) == len(set(provider.tiers))
    assert list_plans(session, plan_set.id)
