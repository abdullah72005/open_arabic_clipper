"""Stage 6 Gemini writing/critique boundary tests through the installed SDK.

These tests exercise the *real* ``google-genai`` request serialization via a
fake httpx transport: the model, API version, JSON body, structured-output
schema, temperature/thinking settings, and single-request (retries disabled)
behavior are all asserted against the actual SDK boundary, not a hand-built
dict. No live network call is made.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx
from google import genai
from google.genai import types
from test_stage6_materialization_core import make_inputs

from app.materialization.gemini import (
    GEMINI_ROUTINE_MODEL,
    GEMINI_STRONG_MODEL,
    GeminiCritiqueProvider,
    GeminiWritingProvider,
)
from app.materialization.policy import DEFAULT_CONFIG, CallLedger
from app.materialization.providers import (
    CRITIQUE_DIMENSIONS,
    BlockCritique,
    CritiqueDimensionFinding,
    CritiqueRequest,
    CritiqueResult,
    CritiqueTarget,
    DeterministicWritingProvider,
    MaterializationCritiqueProvider,
    MaterializationScript,
    WritingRequest,
    WritingTarget,
    materialize_script,
)

NOVEL_COPY = "The mechanism is that mentorship loss compounds"


class _Ok:
    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = dict(payload)

    def as_response(self) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": json.dumps(self.payload)}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 15,
                },
            },
        )


class _Transport:
    """Records every request and returns queued/handler-produced responses."""

    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses or [])
        self._handler: Any = None

    def set_handler(self, handler: Any) -> None:
        self._handler = handler

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._handler is not None:
            return self._handler(request)
        if self._responses:
            return self._responses.pop(0)
        return httpx.Response(500, json={"error": {"code": 500, "message": "unexpected"}})

    def client(self) -> genai.Client:
        return genai.Client(
            api_key="fake-key",
            http_options=types.HttpOptions(
                api_version="v1",
                retry_options=types.HttpRetryOptions(attempts=1),
                httpx_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            ),
        )


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content.decode())


def _writing_request(tier: str = "ROUTINE") -> WritingRequest:
    return WritingRequest(
        targets=(
            WritingTarget(
                block_index=1,
                block_type="ORIGINAL_VALUE",
                substantive_value_kind="INFERENCE",
                semantic_intent="Explain the drop",
                draft_line=None,
                grounding_refs=(),
                source_evidence="teams promoted heavily",
            ),
        ),
        priority="CANDIDATE",
        profile_semantic_context={"target_market": "EGYPT"},
        tier=tier,
    )


def _critique_request() -> CritiqueRequest:
    return CritiqueRequest(
        targets=(
            CritiqueTarget(
                block_index=1,
                copy=NOVEL_COPY,
                substantive_value_kind="INFERENCE",
                semantic_intent="Explain the drop",
                claim_class="SOURCE_GROUNDED_INFERENCE",
                source_evidence="teams promoted heavily",
            ),
        )
    )


def _writing_provider(transport: _Transport, **kwargs: object) -> GeminiWritingProvider:
    return GeminiWritingProvider(api_key="fake-key", client_factory=transport.client, **kwargs)


def _critique_provider(transport: _Transport, **kwargs: object) -> GeminiCritiqueProvider:
    return GeminiCritiqueProvider(api_key="fake-key", client_factory=transport.client, **kwargs)


def test_writing_adapter_v1_serialization_and_single_request() -> None:
    transport = _Transport()
    transport.set_handler(
        lambda request: _Ok({"copies": [{"block_index": 1, "text": "تفسير"}]}).as_response()
    )
    provider = _writing_provider(transport)
    result = provider.write(_writing_request())

    assert result.deferred is False
    assert result.copies == {1: "تفسير"}
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert str(request.url) == (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{GEMINI_ROUTINE_MODEL}:generateContent"
    )
    body = _body(request)
    config = body["generationConfig"]
    assert config["temperature"] == 0.0
    assert config["responseMimeType"] == "application/json"
    assert config["maxOutputTokens"] == provider._output_tokens
    assert "responseSchema" in config
    # The bounded actual evidence travels in the request contents.
    assert "teams promoted heavily" in body["contents"][0]["parts"][0]["text"]
    # No hidden SDK retry configuration beyond a single attempt.
    provider.release()
    assert provider._api_key is None


def test_writing_strong_tier_uses_strong_model_and_thinking() -> None:
    transport = _Transport()
    transport.set_handler(
        lambda request: _Ok({"copies": [{"block_index": 1, "text": "x"}]}).as_response()
    )
    provider = _writing_provider(transport, thinking_level="low")
    provider.write(_writing_request(tier="STRONG"))

    assert len(transport.requests) == 1
    assert GEMINI_STRONG_MODEL in str(transport.requests[0].url)
    config = _body(transport.requests[0])["generationConfig"]
    thinking = config.get("thinkingConfig", {})
    assert thinking and list(thinking.values())[0]
    provider.release()


def test_writing_adapter_non_retryable_failure_defers_with_one_request() -> None:
    transport = _Transport()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"code": 429, "message": "quota"}})

    transport.set_handler(handler)
    provider = _writing_provider(transport)
    result = provider.write(_writing_request())

    assert result.deferred is True
    assert result.reason_code == "RATE_LIMITED"
    assert len(transport.requests) == 1
    assert provider.rate_limited is True


def test_writing_adapter_malformed_output_defers() -> None:
    transport = _Transport()
    transport.set_handler(lambda request: httpx.Response(200, json={"candidates": []}))
    result = _writing_provider(transport).write(_writing_request())
    assert result.deferred is True
    assert result.reason_code == "MALFORMED_OUTPUT"


def test_writing_adapter_safety_refusal_defers() -> None:
    transport = _Transport()
    transport.set_handler(
        lambda request: httpx.Response(
            200,
            json={
                "promptFeedback": {"blockReason": "SAFETY"},
                "candidates": [],
            },
        )
    )
    result = _writing_provider(transport).write(_writing_request())
    assert result.deferred is True
    assert result.reason_code == "SAFETY_REFUSAL"


def test_writing_adapter_missing_key_is_unavailable_and_defers() -> None:
    provider = GeminiWritingProvider(api_key=None)
    assert provider.available() is False
    result = provider.write(_writing_request())
    assert result.deferred is True
    assert result.reason_code == "MISSING_KEY"


def _intent_payload() -> dict[str, object]:
    return {
        "copy_quote": NOVEL_COPY,
        "grounding_refs": ["block:0"],
        "added_relation": "The copy states the mechanism behind the observed drop.",
    }


def _complete_findings(**overrides: str) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for dimension in CRITIQUE_DIMENSIONS:
        item: dict[str, object] = {
            "dimension": dimension,
            "level": overrides.get(dimension, "PASS"),
            "evidence": "e",
        }
        if dimension == "INTENT_FULFILLMENT":
            item["intent_fulfillment"] = _intent_payload()
        findings.append(item)
    return findings


def _complete_block(index: int, **overrides: str) -> BlockCritique:
    findings: list[CritiqueDimensionFinding] = []
    for dimension in CRITIQUE_DIMENSIONS:
        if dimension == "INTENT_FULFILLMENT":
            findings.append(
                CritiqueDimensionFinding(
                    dimension,
                    overrides.get(dimension, "PASS"),
                    copy_quote=NOVEL_COPY,
                    grounding_refs=("block:0",),
                    added_relation="The copy states the mechanism behind the observed drop.",
                )
            )
        else:
            findings.append(CritiqueDimensionFinding(dimension, overrides.get(dimension, "PASS")))
    return BlockCritique(block_index=index, findings=tuple(findings))


def test_critique_adapter_parses_categorical_findings() -> None:
    transport = _Transport()
    payload = {"blocks": [{"block_index": 1, "findings": _complete_findings()}]}
    transport.set_handler(lambda request: _Ok(payload).as_response())
    result = _critique_provider(transport).critique(_critique_request())

    assert result.deferred is False
    assert result.decision(1) is True
    block = result.blocks[1]
    assert {finding.dimension for finding in block.findings} == set(CRITIQUE_DIMENSIONS)
    assert block.verdict() == "ACCEPTED"
    assert len(transport.requests) == 1


def test_critique_unknown_grounding_is_not_accepted() -> None:
    # A structurally complete response whose critical grounding dimension is
    # UNKNOWN defers (unknown is never approval). The whole response is valid; the
    # block verdict is DEFERRED, so the block is not accepted.
    transport = _Transport()
    findings = _complete_findings(GROUNDING_ENTAILMENT="UNKNOWN")
    payload = {"blocks": [{"block_index": 1, "findings": findings}]}
    transport.set_handler(lambda request: _Ok(payload).as_response())
    result = _critique_provider(transport).critique(_critique_request())
    assert result.deferred is False
    assert result.decision(1) is False
    assert result.verdict(1) == "DEFERRED"


def test_critique_incomplete_dimensions_invalidate_whole_response() -> None:
    # A structural violation anywhere invalidates the entire response; a favorable
    # remainder is never retained. Here block 1 omits most dimensions.
    transport = _Transport()
    payload = {
        "blocks": [
            {
                "block_index": 1,
                "findings": [
                    {"dimension": "GROUNDING_ENTAILMENT", "level": "PASS", "evidence": "e"},
                    {"dimension": "SUBSTANTIVE_ADDED_VALUE", "level": "PASS", "evidence": "e"},
                ],
            }
        ]
    }
    transport.set_handler(lambda request: _Ok(payload).as_response())
    result = _critique_provider(transport).critique(_critique_request())
    assert result.deferred is True
    assert result.reason_code == "MALFORMED_OUTPUT"
    assert result.blocks == {}


def test_critique_critical_concern_is_not_accepted() -> None:
    block = BlockCritique(
        block_index=1,
        findings=(
            CritiqueDimensionFinding("GROUNDING_ENTAILMENT", "PASS"),
            CritiqueDimensionFinding("SUBSTANTIVE_ADDED_VALUE", "PASS"),
            CritiqueDimensionFinding("UNSUPPORTED_FACTS", "CONCERN"),
        ),
    )
    assert block.accepted is False


def test_critique_dimension_and_level_literals_are_closed() -> None:
    from app.materialization.gemini import _DimensionLiteral, _LevelLiteral

    assert tuple(_DimensionLiteral.__args__) == tuple(CRITIQUE_DIMENSIONS)
    assert set(_LevelLiteral.__args__) == {"PASS", "CONCERN", "FAIL", "UNKNOWN"}


class _CategoricalCritiqueProvider:
    provider_name = "fake-categorical"

    def __init__(self, blocks: Mapping[int, BlockCritique]) -> None:
        self._blocks = dict(blocks)
        self.calls = 0

    def available(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": self.provider_name, "model": "categorical", "api_version": "v1"}

    def critique(self, request: CritiqueRequest) -> CritiqueResult:
        self.calls += 1
        return CritiqueResult(
            accepted={index: block.accepted for index, block in self._blocks.items()},
            blocks=dict(self._blocks),
            provider_identity=self.runtime_identity(),
        )

    def release(self) -> None:
        return None


class _CountingHostedWriting:
    provider_name = "fake-hosted-writing"

    def __init__(self) -> None:
        self.calls = 0

    def available(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": self.provider_name, "model": "m", "api_version": "v1"}

    def write(self, request: WritingRequest) -> Any:
        self.calls += 1
        from app.materialization.providers import WritingResult

        return WritingResult(
            copies={target.block_index: NOVEL_COPY for target in request.targets},
            provider_identity=self.runtime_identity(),
        )

    def release(self) -> None:
        return None


def _inputs_with_draft(draft: str | None) -> Any:
    inputs = make_inputs()
    blocks = list(inputs.blocks)
    blocks[1] = {**blocks[1], "draft_line": draft, "grounding_refs": ["block:0"]}
    return make_inputs(blocks=tuple(blocks))


def test_materialize_script_requires_known_positive_critique() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    failing = _CategoricalCritiqueProvider(
        {
            1: BlockCritique(
                block_index=1,
                findings=(
                    CritiqueDimensionFinding("GROUNDING_ENTAILMENT", "FAIL"),
                    CritiqueDimensionFinding("SUBSTANTIVE_ADDED_VALUE", "PASS"),
                ),
            )
        }
    )
    script = materialize_script(
        DeterministicWritingProvider(),
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        critique_provider=failing,
    )
    assert script.authored_blocks == ()
    assert failing.calls == 1

    passing = _CategoricalCritiqueProvider({1: _complete_block(1, NATURALNESS="CONCERN")})
    accepted = materialize_script(
        DeterministicWritingProvider(),
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        critique_provider=passing,
    )
    assert accepted.accepted is True
    assert accepted.authored_blocks[0].copy == NOVEL_COPY
    assert passing.calls == 1
    assert any(
        finding.dimension == "LANGUAGE_REGISTER_NATURALNESS" and finding.level == "CONCERN"
        for finding in accepted.findings
    )


def test_hosted_disabled_makes_zero_provider_calls() -> None:
    inputs = _inputs_with_draft(NOVEL_COPY)
    writing = _CountingHostedWriting()
    result = materialize_script(
        writing,
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=False,
    )
    assert result.deferred is True
    assert writing.calls == 0


def test_build_providers_does_not_infer_hosted_from_critique() -> None:
    from app.core.settings import get_settings
    from app.materialization.executor import build_materialization_providers
    from app.materialization.providers import DeterministicWritingProvider, FakeCritiqueProvider

    bundle = build_materialization_providers(
        get_settings(),
        writing=DeterministicWritingProvider(),
        critique=FakeCritiqueProvider(accepted={1: True}),
    )
    assert bundle.allow_hosted is False

    hosted = build_materialization_providers(
        get_settings(),
        writing=_CountingHostedWriting(),
        critique=FakeCritiqueProvider(accepted={1: True}),
    )
    assert hosted.allow_hosted is True


def test_deterministic_provider_identity_has_no_key() -> None:
    provider = GeminiWritingProvider(api_key="super-secret-key")
    identity = provider.runtime_identity()
    assert "super-secret-key" not in str(identity)
    assert identity["routine_model"] == GEMINI_ROUTINE_MODEL
    assert identity["strong_model"] == GEMINI_STRONG_MODEL


def test_critique_provider_implements_protocol() -> None:
    assert isinstance(GeminiCritiqueProvider(api_key=None), MaterializationCritiqueProvider)


def test_materialization_script_type_exposed() -> None:
    # Guard against accidental removal of the accepted-script public type.
    assert MaterializationScript is not None
