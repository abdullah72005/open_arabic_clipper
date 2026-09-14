"""Focused Stage 4.0 provider schema, Gemini, and Qwen tests.

Every network path is mocked; no test makes a live call.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.candidates.providers import ProviderErrorCategory
from app.core.enums import (
    ExternalFactRequirement,
    StrategyDisposition,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.core.settings import Settings, get_settings
from app.transformation.gemini import GeminiTransformationProvider
from app.transformation.local import LocalTransformationProvider
from app.transformation.providers import (
    TransformationProviderError,
    TransformationStrategyRequest,
    deserialize_provider_result,
    parse_strategy_results,
    serialize_provider_result,
)


def _request(candidate_id: str = "c1") -> TransformationStrategyRequest:
    return TransformationStrategyRequest(
        candidate_id=candidate_id,
        content_type="INTERVIEW_INSIGHT",
        source_moment_structure="CLAIM",
        refined_transcript="remote work collapsed mentorship",
    )


def _entry(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "candidate_id": "c1",
        "confidence": 0.6,
        "notes": "n",
        "strategies": [
            {
                "strategy_type": "ANALYSIS",
                "disposition": "RECOMMENDED",
                "intensity": "MODERATE",
                "direction_summary": "analyze",
                "added_value_focus": "explain the mechanism",
                "substantive_value_kind": "AUTHORED_THESIS",
                "confidence": 0.6,
            }
        ],
    }
    base.update(overrides)
    return base


def test_parser_ignores_unknown_and_duplicate_candidates() -> None:
    content = {"candidates": [_entry(candidate_id="unknown"), _entry(), _entry()]}
    results = parse_strategy_results(content, [_request()])
    assert set(results) == {"c1"}
    assert len(results["c1"].strategies) == 1


def test_parser_isolates_malformed_and_unknown_strategy_items() -> None:
    content = {
        "candidates": [
            _entry(
                strategies=[
                    {"strategy_type": "NOT_A_TYPE", "disposition": "RECOMMENDED"},
                    {"disposition": "RECOMMENDED"},
                    {
                        "strategy_type": "ANALYSIS",
                        "disposition": "RECOMMENDED",
                        "added_value_focus": "a",
                    },
                    {
                        "strategy_type": "ANALYSIS",
                        "disposition": "RECOMMENDED",
                        "added_value_focus": "dup",
                    },
                ]
            )
        ]
    }
    results = parse_strategy_results(content, [_request()])
    assert len(results["c1"].strategies) == 1


def test_parser_clamps_and_drops_non_finite_assessments() -> None:
    content = {
        "candidates": [
            _entry(
                strategies=[
                    {
                        "strategy_type": "ANALYSIS",
                        "disposition": "RECOMMENDED",
                        "added_value_focus": "a",
                        "added_value_density": 5.0,
                        "originality_potential": float("inf"),
                    }
                ]
            )
        ]
    }
    parsed = parse_strategy_results(content, [_request()])["c1"].strategies[0]
    assert parsed.added_value_density == 1.0
    assert parsed.originality_potential is None


def test_provider_result_round_trip() -> None:
    results = parse_strategy_results({"candidates": [_entry()]}, [_request()])
    serialized = serialize_provider_result(results["c1"])
    restored = deserialize_provider_result(serialized, "c1")
    assert restored is not None
    assert restored.strategies[0].strategy_type is TransformationStrategyType.ANALYSIS
    assert restored.strategies[0].substantive_value_kind is SubstantiveValueKind.AUTHORED_THESIS


class _FakeModels:
    def __init__(self, outcome: object) -> None:
        self._outcome = outcome
        self.calls = 0

    def generate_content(self, **_kwargs: object) -> object:
        self.calls += 1
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _FakeClient:
    def __init__(self, outcome: object) -> None:
        self.models = _FakeModels(outcome)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _response(payload: dict[str, object]) -> object:
    return SimpleNamespace(
        parsed=payload,
        text=None,
        prompt_feedback=None,
        candidates=[SimpleNamespace(finish_reason=None)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
            thoughts_token_count=2,
        ),
    )


def test_gemini_provider_parses_structured_output_and_closes() -> None:
    client = _FakeClient(_response({"candidates": [_entry()]}))
    provider = GeminiTransformationProvider(api_key="secret", client_factory=lambda: client)
    try:
        results = provider.discover([_request()])
    finally:
        provider.release()
    assert results["c1"].strategies
    assert client.closed is True
    assert provider.usage_summary()["total_token_count"] == 15
    assert "secret" not in json.dumps(provider.runtime_identity())


def test_gemini_provider_rate_limit_is_sanitized() -> None:
    error = RuntimeError("rate limited")
    error.code = 429  # type: ignore[attr-defined]
    client = _FakeClient(error)
    provider = GeminiTransformationProvider(
        api_key="secret", client_factory=lambda: client, sleep=lambda _s: None
    )
    with pytest.raises(TransformationProviderError) as excinfo:
        provider.discover([_request()])
    assert excinfo.value.category == ProviderErrorCategory.RATE_LIMITED.value
    assert provider.rate_limited is True
    assert "secret" not in str(excinfo.value)


def test_gemini_provider_requires_key() -> None:
    provider = GeminiTransformationProvider(api_key=None)
    with pytest.raises(TransformationProviderError) as excinfo:
        provider.discover([_request()])
    assert excinfo.value.category == ProviderErrorCategory.MISSING_KEY.value


def test_gemini_selects_strong_tier_for_complex_case() -> None:
    provider = GeminiTransformationProvider(api_key="secret")
    routine = provider.select_tier([_request()])
    complex_request = TransformationStrategyRequest(
        candidate_id="c1",
        content_type="DEBATE",
        source_moment_structure="DEBATE",
        refined_transcript="a debate claim",
        complex_case=True,
    )
    assert routine == "ROUTINE"
    assert provider.select_tier([complex_request]) == "STRONG"


def test_local_provider_parses_without_network() -> None:
    body = json.dumps(
        {"choices": [{"message": {"content": json.dumps({"candidates": [_entry()]})}}]}
    )

    def request(*_args: object, **_kwargs: object) -> bytes:
        return body.encode()

    provider = LocalTransformationProvider(
        base_url="http://ollama:11434", model="qwen3.5:4b", request=request
    )
    results = provider.discover([_request()])
    assert results["c1"].strategies


def test_qwen_disabled_by_default_and_adaptive_never_uses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.transformation.local import LocalTransformationProvider

    def _settings() -> Settings:
        get_settings.cache_clear()
        return get_settings()

    def _clear_keys() -> None:
        monkeypatch.delenv("CLIPFACTORY_GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    monkeypatch.delenv("CLIPFACTORY_TRANSFORMATION_PROVIDER_MODE", raising=False)
    monkeypatch.delenv("CLIPFACTORY_LOCAL_QWEN_ENABLED", raising=False)
    monkeypatch.delenv("CLIPFACTORY_RECONSTRUCTION_PROVIDER", raising=False)
    _clear_keys()
    get_settings.cache_clear()
    default = _settings()
    assert default.local_qwen_enabled is False
    assert default.transformation_semantic_mode().value == "adaptive"
    # adaptive with no key builds no provider at all and never Qwen.
    assert default.transformation_provider() is None

    # adaptive with a Gemini key uses Gemini and never falls back to Qwen.
    monkeypatch.setenv("CLIPFACTORY_GEMINI_API_KEY", "hermetic-test-key")
    adaptive = _settings()
    provider = adaptive.transformation_provider()
    assert provider is not None
    assert getattr(provider, "provider_name", None) == "gemini"

    # local_only with Qwen disabled returns no provider even if a Gemini key exists.
    monkeypatch.setenv("CLIPFACTORY_TRANSFORMATION_PROVIDER_MODE", "local_only")
    local_only = _settings()
    assert local_only.transformation_provider() is None

    # explicit local_only with Qwen enabled preserves the supported local path.
    monkeypatch.setenv("CLIPFACTORY_LOCAL_QWEN_ENABLED", "true")
    local_enabled = _settings()
    local_provider = local_enabled.transformation_provider()
    assert isinstance(local_provider, LocalTransformationProvider)
    assert local_provider.provider_name == "ollama"
    get_settings.cache_clear()


def test_gemini_key_is_secret_and_presence_only() -> None:
    settings = Settings(CLIPFACTORY_GEMINI_API_KEY=SecretStr("sekret"))
    assert settings.gemini_api_key_present is True
    assert "sekret" not in repr(settings)


def test_external_verification_enum_is_parsed() -> None:
    content = {
        "candidates": [
            _entry(
                strategies=[
                    {
                        "strategy_type": "NEWS_CONTEXT",
                        "disposition": "RECOMMENDED",
                        "added_value_focus": "context",
                        "substantive_value_kind": "MISSING_CONTEXT",
                        "external_verification_requirement": "REQUIRES_EXTERNAL_FACT_VERIFICATION",
                        "verification_requirements": ["verify the budget vote count"],
                    }
                ]
            )
        ]
    }
    parsed = parse_strategy_results(content, [_request()])["c1"].strategies[0]
    assert (
        parsed.external_verification_requirement
        is ExternalFactRequirement.REQUIRES_EXTERNAL_FACT_VERIFICATION
    )
    assert parsed.disposition is StrategyDisposition.RECOMMENDED
