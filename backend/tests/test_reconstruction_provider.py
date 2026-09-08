import json
from dataclasses import replace
from urllib.parse import urlsplit

import pytest

from app.transcription.reconstruction.providers import (
    _SYSTEM_INSTRUCTION,
    OpenAICompatibleReconstructionProvider,
    ProviderResponseError,
    ReconstructionRequest,
    _parse_reconstructions,
    _shrink_request_to_budget,
)
from app.transcription.reconstruction.types import (
    AcousticEvidence,
    ProviderAvailability,
    ProviderHealth,
    WordEvidence,
    estimate_tokens,
)


def test_provider_uses_structured_one_pass_contract() -> None:
    """A single request returns the proposed target, scores, and explanation."""

    captured: list[dict[str, object]] = []

    def request(
        method: str,
        url: str,
        body: bytes | None,
        _headers: dict[str, str],
        _timeout: float,
    ) -> bytes:
        assert method == "POST"
        assert urlsplit(url).path == "/v1/chat/completions"
        assert body is not None
        payload = json.loads(body)
        captured.append(payload)
        return _response(
            {
                "reconstructions": [
                    {
                        "segment_id": 4,
                        "corrected_text": "كان بيقودها الرئيس",
                        "unchanged": False,
                        "confidence": 0.92,
                        "explanation": "restore likely elided hamza",
                        "changes": [{"from": "كان بيقودها الريس", "to": "كان بيقودها الرئيس"}],
                    }
                ]
            }
        )

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=12,
        request=request,
    )
    result = provider.reconstruct_segments(
        [
            ReconstructionRequest(
                segment_index=4,
                raw_text="كان بيقودها الريس",
                corrected_text="كان بيقودها الريس",
                previous=("قبل",),
                following=("بعد",),
                word_evidence=(
                    WordEvidence("كان", 0.0, 1.0, 0.95),
                    WordEvidence("بيقودها", 1.0, 2.0, 0.55),
                    WordEvidence("الريس", 2.0, 3.0, 0.42),
                ),
                acoustic=AcousticEvidence(0.64, 0.64, None, None),
                entities=("الريس",),
                routing_reasons=("low_probability_word",),
                focus_spans=(WordEvidence("الريس", 2.0, 3.0, 0.42),),
                language="ar",
            )
        ]
    )

    candidate = result[4]
    assert candidate.candidate_id == "provider-0"
    assert candidate.text == "كان بيقودها الرئيس"
    assert candidate.provider_confidence == 0.92
    assert getattr(candidate, "scores", None) is None
    assert captured[0]["temperature"] == 0
    assert captured[0]["max_tokens"] == 256
    assert "response_format" not in captured[0]
    assert "EGYPTIAN ARABIC" in captured[0]["messages"][0]["content"]
    assert "Output ONLY a JSON object" in captured[0]["messages"][0]["content"]
    payload = captured[0]["messages"][1]["content"]
    assert "كان بيقودها الريس" in payload
    assert "قبل" in payload
    assert "بعد" in payload


def test_provider_rejects_missing_target_response() -> None:
    """A provider cannot silently omit a persistent segment output slot."""

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=12,
        request=lambda *_args: _response({"reconstructions": []}),
    )

    with pytest.raises(ProviderResponseError, match="omitted"):
        provider.reconstruct_segments(
            [ReconstructionRequest(segment_index=4, raw_text="raw", corrected_text="raw")]
        )


def test_provider_shrinks_over_budget_context_deterministically() -> None:
    """A request that exceeds the model context budget shrinks context, never silently truncates."""

    captured: list[dict[str, object]] = []

    def request(
        _method: str,
        _url: str,
        body: bytes | None,
        _headers: dict[str, str],
        _timeout: float,
    ) -> bytes:
        assert body is not None
        captured.append(json.loads(body))
        return _response(
            {"reconstructions": [{"segment_id": 4, "corrected_text": "هدف", "unchanged": True}]}
        )

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=12,
        max_context_tokens=2048,
        request=request,
    )

    word = WordEvidence("كلمة", 0.0, 1.0, 0.5)
    result = provider.reconstruct_segments(
        [
            ReconstructionRequest(
                segment_index=4,
                raw_text="هدف",
                corrected_text="هدف",
                previous=("سياق",) * 60,
                following=("سياق",) * 60,
                entities=("جهة",) * 60,
                word_evidence=(word,) * 60,
            )
        ]
    )

    assert result[4].text == "هدف"
    payload = json.loads(captured[0]["messages"][1]["content"])
    target = payload["targets"][0]
    assert target["segment_id"] == 4
    assert target["raw_text"] == "هدف"
    assert target["corrected_text"] == "هدف"
    assert len(target["following"]) < 60
    assert len(target["previous"]) < 60
    assert len(target["entities"]) < 60
    assert len(target["words"]) < 60


def test_shrinking_drops_following_fully_before_previous_context() -> None:
    """Context is removed in documented order while target text always survives."""

    base = ReconstructionRequest(segment_index=4, raw_text="هدف", corrected_text="هدف")
    item = "سياق"

    def envelope(request: ReconstructionRequest) -> int:
        return request.estimated_tokens(
            system_instruction="stable system instruction",
            output_tokens=256,
            chat_framing_reserve=64,
            safety_reserve=128,
        )

    budget = envelope(replace(base, previous=(item,) * 20))
    request = replace(base, previous=(item,) * 50, following=(item,) * 50)

    shrunk = _shrink_request_to_budget(request, budget, envelope=envelope)

    assert shrunk.following == ()
    assert len(shrunk.previous) == 20
    assert shrunk.segment_index == 4
    assert shrunk.raw_text == "هدف"
    assert shrunk.corrected_text == "هدف"


def test_shrinking_drops_entities_fully_before_word_evidence() -> None:
    """Entity evidence is exhausted before word evidence shrinks."""

    base = ReconstructionRequest(segment_index=4, raw_text="هدف", corrected_text="هدف")
    entity = "جهة"
    word = WordEvidence("كلمة", 0.0, 1.0, 0.5)

    def envelope(request: ReconstructionRequest) -> int:
        return request.estimated_tokens(
            system_instruction="stable system instruction",
            output_tokens=256,
            chat_framing_reserve=64,
            safety_reserve=128,
        )

    budget = envelope(replace(base, word_evidence=(word,) * 8))
    request = replace(base, entities=(entity,) * 40, word_evidence=(word,) * 40)

    shrunk = _shrink_request_to_budget(request, budget, envelope=envelope)

    assert shrunk.entities == ()
    assert len(shrunk.word_evidence) == 8
    assert shrunk.segment_index == 4
    assert shrunk.raw_text == "هدف"
    assert shrunk.corrected_text == "هدف"


def test_estimated_tokens_budgets_the_complete_chat_envelope() -> None:
    """The estimate covers system instruction, user wrapper, framing, output, and reserve."""

    request = ReconstructionRequest(segment_index=4, raw_text="هدف", corrected_text="هدف")
    system = "stable system instruction"
    framing = 64
    output = 256
    safety = 128

    payload_only = request.estimated_tokens()
    total = request.estimated_tokens(
        system_instruction=system,
        output_tokens=output,
        chat_framing_reserve=framing,
        safety_reserve=safety,
    )

    assert total > payload_only
    assert total - payload_only == len(system.encode("utf-8")) // 2 + framing + output + safety


def test_irreducible_request_raises_before_http_dispatch() -> None:
    """A request that cannot fit after full shrink fails before any transport call."""

    calls = 0

    def request(*_args: object) -> bytes:
        nonlocal calls
        calls += 1
        return b"{}"

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=12,
        max_context_tokens=10,
        request=request,
    )

    with pytest.raises(ProviderResponseError, match="context budget"):
        provider.reconstruct_segments(
            [ReconstructionRequest(segment_index=4, raw_text="هدف", corrected_text="هدف")]
        )

    assert calls == 0


def test_wrapped_user_content_exceeds_budget_before_http_dispatch() -> None:
    """The targets wrapper counts toward the budget: bare payload fits, wrapped does not."""

    calls = 0

    def request(*_args: object) -> bytes:
        nonlocal calls
        calls += 1
        return b"{}"

    request_obj = ReconstructionRequest(segment_index=4, raw_text="هدف", corrected_text="هدف")
    framing = 64
    output = 256
    safety = 128
    bare_payload = estimate_tokens(json.dumps(request_obj.to_payload(), ensure_ascii=False))
    wrapped_payload = estimate_tokens(
        json.dumps({"targets": [request_obj.to_payload()]}, ensure_ascii=False)
    )
    assert wrapped_payload > bare_payload
    bare_full = estimate_tokens(_SYSTEM_INSTRUCTION) + bare_payload + framing + output + safety

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=12,
        max_context_tokens=bare_full,
        request=request,
    )

    with pytest.raises(ProviderResponseError, match="context budget"):
        provider.reconstruct_segments([request_obj])

    assert calls == 0


def test_provider_records_request_size_diagnostics() -> None:
    """The provider stores measured serialized bytes and estimated input tokens."""

    def request(
        _method: str,
        _url: str,
        body: bytes | None,
        _headers: dict[str, str],
        _timeout: float,
    ) -> bytes:
        assert body is not None
        return _response(
            {"reconstructions": [{"segment_id": 4, "corrected_text": "هدف", "unchanged": True}]}
        )

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=12,
        request=request,
    )

    sent = ReconstructionRequest(
        segment_index=4,
        raw_text="هدف",
        corrected_text="هدف",
        previous=("قبل",),
        following=("بعد",),
    )
    provider.reconstruct_segments([sent])

    diagnostics = provider.last_request_sizes()
    assert len(diagnostics) == 1
    assert diagnostics[0].segment_index == 4
    assert diagnostics[0].serialized_bytes == len(
        json.dumps({"targets": [sent.to_payload()]}, ensure_ascii=False).encode("utf-8")
    )
    assert diagnostics[0].estimated_input_tokens > 0


def test_openai_compatible_health_requires_exact_model_id() -> None:
    def request(
        method: str,
        url: str,
        body: bytes | None,
        _headers: dict[str, str],
        _timeout: float,
    ) -> bytes:
        assert method == "GET"
        assert urlsplit(url).path == "/v1/models"
        assert body is None
        return b'{"data":[{"id":"qwen3.5:4b"}]}'

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://provider:11434",
        model="qwen3.5:4b",
        timeout_seconds=3,
        request=request,
    )

    assert provider.health() == ProviderHealth(
        ProviderAvailability.AVAILABLE,
        "openai_compatible",
        "qwen3.5:4b",
        None,
        "model available",
    )


def test_openai_compatible_health_reports_missing_model_without_response_content() -> None:
    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://provider:11434",
        model="secret-model",
        timeout_seconds=3,
        request=lambda *_args: b'{"data":[{"id":"other-model"}],"secret":"do-not-leak"}',
    )

    result = provider.health()

    assert result.availability is ProviderAvailability.UNAVAILABLE
    assert result.detail == "configured model secret-model is not available"
    assert "do-not-leak" not in result.detail


def test_openai_compatible_release_is_a_no_op() -> None:
    calls = 0

    def request(*_args: object) -> bytes:
        nonlocal calls
        calls += 1
        return b"{}"

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://provider:11434",
        model="qwen3.5:4b",
        timeout_seconds=3,
        request=request,
    )

    provider.release()

    assert calls == 0


def test_parse_reconstructions_preserves_single_provider_confidence() -> None:
    request = ReconstructionRequest(segment_index=4, raw_text="raw", corrected_text="raw")
    result = _parse_reconstructions(
        {
            "reconstructions": [
                {
                    "segment_id": 4,
                    "corrected_text": "new",
                    "unchanged": False,
                    "confidence": 0.88,
                    "explanation": "fix",
                }
            ]
        },
        [request],
    )
    assert result[4].text == "new"
    assert result[4].provider_confidence == 0.88
    assert getattr(result[4], "scores", None) is None


@pytest.mark.parametrize(
    "message_content",
    [
        "no json object here",
        '{"reconstructions": [',
        json.dumps({"reconstructions": [{"segment_id": 99, "corrected_text": "x"}]}),
        json.dumps(
            {
                "reconstructions": [
                    {"segment_id": 4, "corrected_text": "a", "confidence": 0.9},
                    {"segment_id": 4, "corrected_text": "b", "confidence": 0.9},
                ]
            }
        ),
        json.dumps(
            {"reconstructions": [{"segment_id": 4, "corrected_text": "x", "confidence": "0.9"}]}
        ),
        json.dumps(
            {"reconstructions": [{"segment_id": 4, "corrected_text": "x", "confidence": True}]}
        ),
        json.dumps(
            {"reconstructions": [{"segment_id": 4, "corrected_text": "x", "confidence": -0.01}]}
        ),
        json.dumps(
            {"reconstructions": [{"segment_id": 4, "corrected_text": "x", "confidence": 1.01}]}
        ),
        json.dumps(
            {
                "reconstructions": [
                    {"segment_id": 4, "corrected_text": "x", "confidence": float("nan")}
                ]
            }
        ),
        json.dumps(
            {
                "reconstructions": [
                    {"segment_id": 4, "corrected_text": "x", "confidence": float("inf")}
                ]
            }
        ),
        '{"reconstructions": [{"segment_id": 4, "corrected_text": "x", "confidence": 1e309}]}',
    ],
)
def test_provider_rejects_malformed_responses_as_contained_provider_error(
    message_content: str,
) -> None:
    """Malformed provider output must be a contained ProviderResponseError, never a bare
    ValueError, TypeError, or KeyError."""
    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=12,
        request=lambda *_args: _raw_message(message_content),
    )
    with pytest.raises(ProviderResponseError):
        provider.reconstruct_segments(
            [ReconstructionRequest(segment_index=4, raw_text="raw", corrected_text="raw")]
        )


def _response(content: dict[str, object]) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]},
        ensure_ascii=False,
    ).encode()


def _raw_message(message_content: str) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"content": message_content}}]},
        ensure_ascii=False,
    ).encode()
