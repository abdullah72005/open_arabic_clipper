import json
from urllib.parse import urlsplit

import pytest

from app.transcription.reconstruction.providers import (
    GenerationRequest,
    OpenAICompatibleReconstructionProvider,
    ProviderResponseError,
    ResolutionRequest,
)
from app.transcription.reconstruction.types import ProviderAvailability, ProviderHealth


def test_provider_uses_structured_two_pass_contract() -> None:
    """Passes receive stable IDs and return only schema-validated candidate evidence."""

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
        if len(captured) == 1:
            return _response(
                {
                    "generations": [
                        {
                            "segment_id": 4,
                            "candidates": [
                                {
                                    "text": "كان بيقودها الرئيس",
                                    "changes": [],
                                    "evidence_segment_ids": [3, 4, 5],
                                }
                            ],
                        }
                    ]
                }
            )
        return _response(
            {
                "resolutions": [
                    {
                        "segment_id": 4,
                        "candidate_id": "provider-0",
                        "semantic_coherence": 0.9,
                        "egyptian_naturalness": 0.9,
                        "discourse_continuity": 0.9,
                        "entity_consistency": 1.0,
                        "selection_confidence": 0.9,
                    }
                ]
            }
        )

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434", model="qwen3:8b", timeout_seconds=12, request=request
    )
    generated = provider.generate_candidates([GenerationRequest(4, "raw", (), ())])
    resolved = provider.resolve_candidates(
        [ResolutionRequest(4, "raw", (), (), tuple(generated[4]))]
    )

    assert generated[4][0].candidate_id == "provider-0"
    assert resolved[4].candidate_id == "provider-0"
    assert captured[0]["temperature"] == 0
    assert captured[0]["response_format"]["type"] == "json_schema"
    assert captured[1]["messages"][1]["content"]


def test_provider_rejects_missing_target_response() -> None:
    """A provider cannot silently omit a persistent segment output slot."""

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3:8b",
        timeout_seconds=12,
        request=lambda *_args: _response({"generations": []}),
    )

    with pytest.raises(ProviderResponseError, match="omitted"):
        provider.generate_candidates([GenerationRequest(4, "raw", (), ())])


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
        return b'{"data":[{"id":"qwen3:8b"}]}'

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://provider:11434",
        model="qwen3:8b",
        timeout_seconds=3,
        request=request,
    )

    assert provider.health() == ProviderHealth(
        ProviderAvailability.AVAILABLE,
        "openai_compatible",
        "qwen3:8b",
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
        model="qwen3:8b",
        timeout_seconds=3,
        request=request,
    )

    provider.release()

    assert calls == 0


def _response(content: dict[str, object]) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]},
        ensure_ascii=False,
    ).encode()
