"""Focused tests for Gemini transient-503 classification and retry bounds.

These use fake SDK clients/exceptions only: HTTP 503 is classified as
SERVICE_UNAVAILABLE and retried exactly once (initial + one bounded retry);
429, 401/403, malformed output, and safety/validation failures are never
retried; and a sentinel fake API key never appears in exceptions, tracebacks, or
metadata. No live Google calls are made.
"""

from __future__ import annotations

import traceback

import pytest

from app.transcription.reconstruction.confidence import CONFIDENCE_POLICY_VERSION
from app.transcription.reconstruction.gemini import (
    GeminiErrorCategory,
    GeminiProviderError,
    GeminiReconstructionProvider,
)
from app.transcription.reconstruction.providers import (
    ReconstructionRequest,
)
from app.transcription.reconstruction.routing import AdaptiveRoutingConfig, RoutingMode
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
)
from app.transcription.reconstruction.validation import VALIDATION_VERSION


class _CodeError(Exception):
    def __init__(self, code: int, message: str = "upstream failure") -> None:
        self.code = code
        super().__init__(message)


class _RaisingModels:
    def __init__(self, error: Exception, attempts: list[int]) -> None:
        self.error = error
        self.attempts = attempts

    def generate_content(self, model: str, contents: str, config: object) -> object:
        self.attempts.append(1)
        raise self.error


class _MalformedModels:
    def __init__(self, attempts: list[int]) -> None:
        self.attempts = attempts

    def generate_content(self, model: str, contents: str, config: object) -> object:
        self.attempts.append(1)
        return _GarbageResponse()


class _GarbageResponse:
    parsed = None
    text = "this is not json"
    usage_metadata = None
    candidates = [type("C", (), {"finish_reason": type("F", (), {"name": "STOP"})()})()]
    prompt_feedback = None


class _FakeSdkClient:
    def __init__(self, models: object) -> None:
        self.models = models
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _provider(
    error: Exception | None = None,
    *,
    malformed: bool = False,
    attempts: list[int] | None = None,
    api_key: str = "test-key",
) -> GeminiReconstructionProvider:
    attempts = [] if attempts is None else attempts

    def factory() -> object:
        if malformed:
            return _FakeSdkClient(_MalformedModels(attempts))
        return _FakeSdkClient(_RaisingModels(error, attempts))

    return GeminiReconstructionProvider(
        api_key=api_key,
        model="gemini-3.8-flash",
        retry_attempts=1,
        retry_backoff_seconds=0.0,
        client_factory=factory,
        owns_client=True,
    )


def _request() -> ReconstructionRequest:
    return ReconstructionRequest(segment_index=0, raw_text="دخم", corrected_text="دخم")


def test_http_503_is_retried_exactly_once_then_service_unavailable() -> None:
    attempts: list[int] = []
    provider = _provider(_CodeError(503), attempts=attempts)
    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])
    assert len(attempts) == 2  # initial + one bounded retry
    assert raised.value.category is GeminiErrorCategory.SERVICE_UNAVAILABLE
    assert str(raised.value) == "gemini_SERVICE_UNAVAILABLE"


def test_http_503_exhausted_falls_back_safely_with_sanitized_evidence() -> None:
    attempts: list[int] = []
    gemini = _provider(_CodeError(503), attempts=attempts)

    class FakeLocal:
        def health(self) -> ProviderHealth:
            return ProviderHealth(
                ProviderAvailability.AVAILABLE, "ollama", "qwen3.5:4b", "sha256:x", "ok"
            )

        def runtime_identity(self) -> dict[str, object]:
            return {
                "provider": "ollama",
                "model": "qwen3.5:4b",
                "digest": "sha256:x",
                "prompt_hash": "p",
                "schema_version": "s",
                "max_context_tokens": 4096,
                "output_tokens": 256,
                "chat_framing_reserve": 64,
                "safety_reserve": 128,
                "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
                "validation_version": VALIDATION_VERSION,
            }

        def refresh_runtime_identity(self) -> dict[str, object]:
            return self.runtime_identity()

        def release(self) -> None:
            pass

        def reconstruct_segments(
            self, requests: list[ReconstructionRequest]
        ) -> dict[int, ReconstructionCandidate]:
            return {
                request.segment_index: ReconstructionCandidate(
                    "provider-0", "ضخمة", provider_confidence=1.0
                )
                for request in requests
            }

    segment = {
        "start": 0.0,
        "end": 1.0,
        "text": "مش قادر",
        "raw_text": "مش قادر",
        "corrected_text": "مش قادر",
        "words": [
            {"word": "م", "probability": 0.30},
            {"word": "ش", "probability": 0.25},
            {"word": "قادر", "probability": 0.20},
            {"word": "يفهم", "probability": 0.90},
        ],
    }
    reconstructor = ContextualReconstructor(
        FakeLocal(),
        gemini_provider=gemini,
        routing=AdaptiveRoutingConfig(mode=RoutingMode.ADAPTIVE),
        gemini_budget=10,
    )
    result = reconstructor.reconstruct(
        [segment],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )
    assert len(attempts) == 2
    assert result.segments[0].gemini_attempted is True
    assert result.segments[0].gemini_result_state == "failure:SERVICE_UNAVAILABLE"
    assert result.segments[0].applied is False  # safe unresolved/fallback
    counts = result.metadata["routing_counts"]
    assert counts["gemini_failures"] == 1


def test_http_429_is_never_retried_and_classified_rate_limited() -> None:
    attempts: list[int] = []
    provider = _provider(_CodeError(429), attempts=attempts)
    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])
    assert len(attempts) == 1
    assert raised.value.category is GeminiErrorCategory.RATE_LIMITED


def test_http_401_and_403_are_never_retried_and_classified_authentication() -> None:
    for code in (401, 403):
        attempts: list[int] = []
        provider = _provider(_CodeError(code), attempts=attempts)
        with pytest.raises(GeminiProviderError) as raised:
            provider.reconstruct_segments([_request()])
        assert len(attempts) == 1
        assert raised.value.category is GeminiErrorCategory.AUTHENTICATION


def test_malformed_structured_output_is_never_retried() -> None:
    attempts: list[int] = []
    provider = _provider(malformed=True, attempts=attempts)
    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])
    assert len(attempts) == 1
    assert raised.value.category is GeminiErrorCategory.MALFORMED_OUTPUT


def test_sentinel_key_absent_from_503_exception_traceback_and_metadata() -> None:
    sentinel = "AIzaSENTINEL_503_FAKE_KEY_12345"
    attempts: list[int] = []
    provider = _provider(
        _CodeError(503, f"upstream failure {sentinel}"),
        attempts=attempts,
        api_key=sentinel,
    )
    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([_request()])
    tb = traceback.format_exception(type(raised.value), raised.value, raised.value.__traceback__)
    assert sentinel not in str(raised.value)
    assert sentinel not in repr(raised.value)
    assert sentinel not in "".join(tb)
    assert sentinel not in repr(provider.runtime_identity())
    assert sentinel not in repr(provider.usage_summary())
    provider.release()
