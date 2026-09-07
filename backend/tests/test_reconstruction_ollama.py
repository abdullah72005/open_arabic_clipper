from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest

from app.transcription.reconstruction.ollama import OllamaReconstructionProvider
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    body: bytes | None


class StubHttp:
    def __init__(self, responses: dict[tuple[str, str], bytes | BaseException]) -> None:
        self.responses = responses
        self.calls: list[RecordedRequest] = []

    def __call__(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> bytes:
        del headers, timeout
        path = urlsplit(url).path
        self.calls.append(RecordedRequest(method, path, body))
        response = self.responses[(method, path)]
        if isinstance(response, BaseException):
            raise response
        return response


def test_ollama_health_requires_exact_model_and_returns_digest() -> None:
    request = StubHttp(
        {("GET", "/api/tags"): b'{"models":[{"name":"qwen3:8b","digest":"sha256:abc"}]}'}
    )
    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3:8b",
        timeout_seconds=3,
        request=request,
    )

    assert provider.health() == ProviderHealth(
        ProviderAvailability.AVAILABLE,
        "ollama",
        "qwen3:8b",
        "sha256:abc",
        "model available",
    )


def test_ollama_health_reports_missing_exact_model() -> None:
    request = StubHttp({("GET", "/api/tags"): b'{"models":[{"name":"qwen3:8b-latest"}]}'})
    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3:8b",
        timeout_seconds=3,
        request=request,
    )

    assert provider.health() == ProviderHealth(
        ProviderAvailability.UNAVAILABLE,
        "ollama",
        "qwen3:8b",
        None,
        "configured model qwen3:8b is not installed",
    )


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(TimeoutError("timed out at secret endpoint"), id="timeout"),
        pytest.param(ConnectionRefusedError("secret host refused"), id="connection-refused"),
        pytest.param(
            HTTPError("http://secret", 500, "private response", None, None),
            id="non-2xx",
        ),
        pytest.param(b'{"models":', id="invalid-json"),
        pytest.param(b"{}", id="missing-models"),
    ],
)
def test_ollama_health_maps_probe_failures_without_leaking_response_details(
    response: bytes | BaseException,
) -> None:
    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3:8b",
        timeout_seconds=3,
        request=StubHttp({("GET", "/api/tags"): response}),
    )

    result = provider.health()

    assert result == ProviderHealth(
        ProviderAvailability.UNAVAILABLE,
        "ollama",
        "qwen3:8b",
        None,
        "provider health check failed",
    )


def test_ollama_release_unloads_configured_model() -> None:
    request = StubHttp(
        {
            ("POST", "/api/generate"): b'{"done":true}',
            ("GET", "/api/ps"): b'{"models":[]}',
        }
    )
    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3:8b",
        timeout_seconds=3,
        request=request,
    )

    outcome = provider.release()

    assert json.loads(request.calls[0].body or b"{}") == {
        "model": "qwen3:8b",
        "keep_alive": 0,
    }
    assert outcome.requested is True
    assert outcome.confirmed is True
    assert outcome.warning is None
    assert ("GET", "/api/ps") in [(call.method, call.path) for call in request.calls]


def test_ollama_release_confirms_after_delayed_disappearance() -> None:
    clock = FakeClock()

    def transport(
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> bytes:
        del body, headers, timeout
        if urlsplit(url).path == "/api/generate":
            return b'{"done":true}'
        if clock.now >= 1.0:
            return b'{"models":[]}'
        return b'{"models":[{"name":"qwen3:8b"}]}'

    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3:8b",
        timeout_seconds=3,
        unload_timeout_seconds=5,
        unload_poll_seconds=0.1,
        sleep=clock.advance,
        monotonic=clock,
        request=transport,
    )

    outcome = provider.release()

    assert outcome.requested is True
    assert outcome.confirmed is True
    assert outcome.warning is None
    assert outcome.elapsed_seconds >= 1.0


def test_ollama_release_times_out_while_model_still_resident() -> None:
    clock = FakeClock()

    def transport(
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> bytes:
        del body, headers, timeout
        if urlsplit(url).path == "/api/generate":
            return b'{"done":true}'
        return b'{"models":[{"name":"qwen3:8b"}]}'

    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3:8b",
        timeout_seconds=3,
        unload_timeout_seconds=2,
        unload_poll_seconds=0.1,
        sleep=clock.advance,
        monotonic=clock,
        request=transport,
    )

    outcome = provider.release()

    assert outcome.requested is True
    assert outcome.confirmed is False
    assert "still resident" in (outcome.warning or "")


def test_ollama_release_transport_failure_reports_request_failed() -> None:
    def transport(
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> bytes:
        del method, url, body, headers, timeout
        raise ConnectionRefusedError("secret host refused")

    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3:8b",
        timeout_seconds=3,
        request=transport,
    )

    outcome = provider.release()

    assert outcome.requested is False
    assert outcome.confirmed is False
    assert "failed" in (outcome.warning or "")


def test_ollama_release_can_leave_model_loaded_when_explicitly_configured() -> None:
    request = StubHttp({})
    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3:8b",
        timeout_seconds=3,
        release_after_run=False,
        request=request,
    )

    provider.release()

    assert request.calls == []
