from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.transcription.reconstruction.gemini import (
    GeminiErrorCategory,
    GeminiProviderError,
    GeminiReconstructionProvider,
)
from app.transcription.reconstruction.providers import ReconstructionRequest
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
)


def _request(segment_index: int = 0, raw: str = "دخم") -> ReconstructionRequest:
    return ReconstructionRequest(segment_index=segment_index, raw_text=raw, corrected_text=raw)


def _ok_content() -> dict[str, object]:
    return {
        "reconstructions": [
            {
                "segment_id": 0,
                "corrected_text": "ضخمة",
                "unchanged": False,
                "confidence": 0.95,
                "explanation": "repair",
                "changes": [],
            }
        ]
    }


def _response(
    *,
    parsed: object | None = None,
    text: str | None = None,
    usage: bool = True,
    finish: str = "STOP",
    block: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        parsed=parsed,
        text=text,
        usage_metadata=(
            SimpleNamespace(prompt_token_count=11, candidates_token_count=7, total_token_count=18)
            if usage
            else None
        ),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        prompt_feedback=SimpleNamespace(block_reason=block) if block is not None else None,
    )


class FakeModels:
    def __init__(
        self,
        *,
        response: object | None = None,
        error: Exception | None = None,
        get_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.get_error = get_error
        self.generate_calls = 0
        self.contents: list[object] = []
        self.configs: list[object] = []

    def get(self, model: str) -> object:
        if self.get_error is not None:
            raise self.get_error
        return SimpleNamespace(name=f"models/{model}", version="3.6")

    def generate_content(self, model: str, contents: str, config: object) -> object:
        self.generate_calls += 1
        self.contents.append(contents)
        self.configs.append(config)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, **kwargs: object) -> None:
        self.models = FakeModels(**kwargs)


def _provider(
    api_key: str | None = "secret-key",
    model: str = "gemini-3.6-flash",
    **kwargs: object,
) -> GeminiReconstructionProvider:
    factory = kwargs.pop("client_factory", None)
    return GeminiReconstructionProvider(
        api_key=api_key,
        model=model,
        timeout_seconds=10.0,
        retry_attempts=kwargs.pop("retry_attempts", 1),
        retry_backoff_seconds=kwargs.pop("retry_backoff_seconds", 0.0),
        max_output_tokens=kwargs.pop("max_output_tokens", 256),
        sleep=kwargs.pop("sleep", lambda _seconds: None),
        client_factory=factory,
    )


def test_gemini_health_available_with_live_digest() -> None:
    client = FakeClient(response=_response(parsed=_ok_content()))
    provider = _provider(client_factory=lambda: client)

    health = provider.health()

    assert health == ProviderHealth(
        ProviderAvailability.AVAILABLE,
        "gemini",
        "gemini-3.6-flash",
        health.model_digest,
        "gemini model available",
    )
    assert health.model_digest is not None


def test_gemini_health_misconfigured_when_key_missing() -> None:
    provider = _provider(api_key=None)

    assert provider.health() == ProviderHealth(
        ProviderAvailability.MISCONFIGURED,
        "gemini",
        "gemini-3.6-flash",
        None,
        "gemini api key is not configured",
    )


def test_gemini_health_unavailable_on_auth_failure() -> None:
    class AuthError(Exception):
        code = 401

    provider = _provider(client_factory=lambda: FakeClient(get_error=AuthError("bad key")))

    health = provider.health()

    assert health.availability is ProviderAvailability.UNAVAILABLE
    assert health.detail == "gemini_AUTHENTICATION"


def test_gemini_reconstruct_parses_structured_output() -> None:
    client = FakeClient(response=_response(parsed=_ok_content()))
    provider = _provider(client_factory=lambda: client)

    result = provider.reconstruct_segments([_request()])

    assert result[0] == ReconstructionCandidate(
        "provider-0", "ضخمة", provider_confidence=0.95, explanation="repair"
    )
    assert client.models.generate_calls == 1
    assert provider.usage_summary() == {
        "prompt_token_count": 11,
        "candidates_token_count": 7,
        "total_token_count": 18,
    }


def test_gemini_reconstruct_rejects_missing_coverage() -> None:
    partial = {"reconstructions": [{"segment_id": 99, "corrected_text": "ضخمة", "confidence": 0.9}]}
    provider = _provider(client_factory=lambda: FakeClient(response=_response(parsed=partial)))

    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])

    assert raised.value.category is GeminiErrorCategory.MALFORMED_OUTPUT


def test_gemini_reconstruct_rejects_malformed_json() -> None:
    provider = _provider(client_factory=lambda: FakeClient(response=_response(text="not json")))

    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])

    assert raised.value.category is GeminiErrorCategory.MALFORMED_OUTPUT


def test_gemini_safety_refusal_is_not_retried() -> None:
    provider = _provider(
        retry_attempts=1,
        client_factory=lambda: FakeClient(response=_response(finish="SAFETY")),
    )

    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])

    assert raised.value.category is GeminiErrorCategory.SAFETY_REFUSAL


def test_gemini_rate_limit_is_not_retried() -> None:
    class RateLimitError(Exception):
        code = 429

    client = FakeClient(error=RateLimitError("quota"))
    provider = _provider(retry_attempts=1, client_factory=lambda: client)

    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])

    assert raised.value.category is GeminiErrorCategory.RATE_LIMITED
    assert client.models.generate_calls == 1


def test_gemini_timeout_is_retried_once_then_raised() -> None:
    class Timeout(Exception):
        code = 504

    client = FakeClient(error=Timeout("slow"))
    provider = _provider(retry_attempts=1, client_factory=lambda: client)

    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])

    assert raised.value.category is GeminiErrorCategory.TIMEOUT
    assert client.models.generate_calls == 2


def test_gemini_connection_error_is_retried_then_raised() -> None:
    client = FakeClient(error=ConnectionError("offline"))
    provider = _provider(retry_attempts=2, client_factory=lambda: client)

    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])

    assert raised.value.category is GeminiErrorCategory.CONNECTION
    assert client.models.generate_calls == 3


def test_gemini_missing_key_raises_before_any_call() -> None:
    provider = _provider(api_key=None, client_factory=lambda: FakeClient())

    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])

    assert raised.value.category is GeminiErrorCategory.MISSING_KEY


def test_gemini_exception_never_contains_secret() -> None:
    class FailingError(Exception):
        code = 401

    provider = _provider(
        api_key="AIzaSuperSecret",
        client_factory=lambda: FakeClient(
            error=FailingError("unauthorized request AIzaSuperSecret")
        ),
    )

    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])

    assert "AIzaSuperSecret" not in str(raised.value)
    assert "AIzaSuperSecret" not in repr(raised.value)
    assert str(raised.value) == "gemini_AUTHENTICATION"


def test_gemini_runtime_identity_never_contains_secret() -> None:
    provider = _provider(api_key="AIzaSuperSecret", client_factory=lambda: FakeClient())

    serialized = repr(provider.runtime_identity()) + repr(provider.refresh_runtime_identity())

    assert "AIzaSuperSecret" not in serialized
    assert "api_key" not in serialized.casefold()


def test_gemini_release_is_noop() -> None:
    provider = _provider(
        client_factory=lambda: FakeClient(response=_response(parsed=_ok_content()))
    )

    assert provider.release() is None
