"""Fake-transport tests for the hosted Gemini provider compatibility boundary.

These tests exercise the *effective* HTTP surface through the installed
``google-genai`` SDK (URL/version and serialized request body), not just the
Python constructor arguments. The Interactions transcription provider must hit
the ``v1beta`` surface with ``timestamp_granularities`` nested inside the
verbatim ``mode`` object; the adjudication provider must keep using the stable
``v1`` generateContent surface. No network call is ever made.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.refinement.hosted import (
    GeminiAdjudicationProvider,
    GeminiAudioTranscriptionProvider,
)
from app.refinement.types import AdjudicationRequest

genai = pytest.importorskip("google.genai")
types = pytest.importorskip("google.genai.types")

_BASE_URL = "https://gemini.example.test"


class _CapturingTransport:
    """Records the effective URL and decoded JSON body of every request."""

    def __init__(self, responder) -> None:
        self.requests: list[dict[str, object]] = []
        self._responder = responder

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload: object
        try:
            payload = json.loads(request.content.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = request.content.decode("utf-8", "replace")
        self.requests.append({"url": str(request.url), "json": payload})
        return self._responder(request)


def _real_client(capturing: _CapturingTransport, *, api_version: str):
    client = genai.Client(
        api_key="fake-key",
        http_options=types.HttpOptions(api_version=api_version, base_url=_BASE_URL),
    )
    httpx_client = client._api_client._httpx_client
    if not isinstance(httpx_client, httpx.Client):  # pragma: no cover - SDK drift
        pytest.skip("installed google-genai http client is not patchable")
    httpx_client._transport = httpx.MockTransport(capturing.handler)
    return client


def _transcription_responder(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "int_1",
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello world",
                            "annotations": [
                                {
                                    "type": "word_info",
                                    "text": "hello",
                                    "start_offset": "0.0s",
                                    "end_offset": "0.5s",
                                },
                                {
                                    "type": "word_info",
                                    "text": "world",
                                    "start_offset": "0.5s",
                                    "end_offset": "1.0s",
                                },
                            ],
                        }
                    ],
                }
            ],
        },
    )


class _FakeFiles:
    def upload(self, *, file: Path, config: object = None) -> object:
        return SimpleNamespace(name="files/abc", uri="https://remote.invalid/files/abc")

    def delete(self, *, name: str, config: object = None) -> object:
        return SimpleNamespace(name=name)


class _HybridClient:
    """Real interactions/models surfaces plus a fake Files API."""

    def __init__(self, real_client: object) -> None:
        self.interactions = real_client.interactions  # type: ignore[attr-defined]
        self.models = real_client.models  # type: ignore[attr-defined]
        self.files = _FakeFiles()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _transcription_provider(client: object) -> GeminiAudioTranscriptionProvider:
    return GeminiAudioTranscriptionProvider(
        api_key="fake-key",
        client_factory=lambda: _HybridClient(client),
        owns_client=False,
    )


def test_transcription_effective_url_and_serialized_body(tmp_path: Path) -> None:
    capturing = _CapturingTransport(_transcription_responder)
    client = _real_client(capturing, api_version="v1beta")
    provider = _transcription_provider(client)
    audio = tmp_path / "candidate.wav"
    audio.write_bytes(b"RIFF")

    result = provider.transcribe(audio, language_codes=["ar-EG"])

    assert capturing.requests, "no HTTP request was issued"
    request = capturing.requests[0]
    assert request["url"] == f"{_BASE_URL}/v1beta/interactions"
    body = request["json"]
    assert isinstance(body, dict)
    assert body["model"] == "gemini-3.5-transcribe"
    transcription_config = body["generation_config"]["transcription_config"]  # type: ignore[index]
    assert transcription_config["mode"] == {
        "type": "verbatim",
        "timestamp_granularities": ["word"],
    }
    assert "timestamp_granularities" not in transcription_config
    assert transcription_config["language_codes"] == ["ar-EG"]
    # Audio content is passed directly as input.
    serialized_input = json.dumps(body["input"])
    assert '"type": "audio"' in serialized_input
    assert "remote.invalid" in serialized_input

    # Documented word annotations parse into timed words.
    assert result.transcript == "hello world"
    assert [(word.text, word.start, word.end) for word in result.word_timestamps] == [
        ("hello", 0.0, 0.5),
        ("world", 0.5, 1.0),
    ]


def _adjudication_responder(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "adjudications": [
                                            {
                                                "ambiguity_id": "a1",
                                                "selected_reading": "خمسة وعشرين",
                                                "confidence": 0.9,
                                                "reason": "audio",
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        },
    )


def test_adjudication_uses_stable_v1_generate_content() -> None:
    capturing = _CapturingTransport(_adjudication_responder)
    client = _real_client(capturing, api_version="v1")
    provider = GeminiAdjudicationProvider(
        api_key="fake-key",
        client_factory=lambda: _HybridClient(client),
        owns_client=False,
    )
    requests = [
        AdjudicationRequest(
            ambiguity_id="a1",
            context="قال الرقم",
            candidate_readings=("خمسة وعشرين", "خمسة وتسعين"),
            evidence_summary=({"provider": "local", "text": "خمسة وعشرين"},),
            dialect_profile="EGYPTIAN",
            meaning_critical=True,
        )
    ]

    results = provider.adjudicate(requests)

    assert capturing.requests, "no HTTP request was issued"
    assert capturing.requests[0]["url"] == f"{_BASE_URL}/v1/models/gemini-3.8-flash:generateContent"
    assert results["a1"].selected_reading == "خمسة وعشرين"


def test_settings_factory_effective_transcription_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production Settings factory must produce an effective ``v1beta`` URL.

    This asserts the *effective* HTTP surface produced by the full settings →
    provider → SDK chain, so a regression in the settings default (previously
    ``v1``) fails here even though the provider constructor default is already
    ``v1beta``.
    """

    from app.core.settings import Settings

    monkeypatch.delenv("CLIPFACTORY_GEMINI_TRANSCRIPTION_API_VERSION", raising=False)
    monkeypatch.setenv("CLIPFACTORY_GEMINI_API_KEY", "fake-key")

    settings = Settings(_env_file=None)
    provider = settings.hosted_transcription_provider()
    assert isinstance(provider, GeminiAudioTranscriptionProvider)
    assert provider.runtime_identity()["api_version"] == "v1beta"

    capturing = _CapturingTransport(_transcription_responder)
    real_client = _real_client(capturing, api_version=str(provider._api_version))
    provider._client = _HybridClient(real_client)
    provider._owns_client = False

    audio = tmp_path / "candidate.wav"
    audio.write_bytes(b"RIFF")
    provider.transcribe(audio, language_codes=["ar-EG"])

    assert capturing.requests, "no HTTP request was issued"
    assert capturing.requests[0]["url"] == f"{_BASE_URL}/v1beta/interactions"
    body = capturing.requests[0]["json"]
    assert isinstance(body, dict)
    mode = body["generation_config"]["transcription_config"]["mode"]  # type: ignore[index]
    assert mode["type"] == "verbatim"
    assert mode["timestamp_granularities"] == ["word"]

    # Adjudication stays on the stable v1 generateContent surface.
    adjudication = settings.adjudication_provider()
    assert adjudication is not None
    assert adjudication.runtime_identity()["api_version"] == "v1"


def test_provider_clients_keep_isolated_api_versions() -> None:
    transcription = GeminiAudioTranscriptionProvider(api_key="fake-key")
    adjudication = GeminiAdjudicationProvider(api_key="fake-key")

    transcription_client = transcription._client_instance()
    adjudication_client = adjudication._client_instance()
    assert transcription_client is not None
    assert adjudication_client is not None
    assert transcription_client._api_client._http_options.api_version == "v1beta"
    assert adjudication_client._api_client._http_options.api_version == "v1"

    assert transcription.runtime_identity()["api_version"] == "v1beta"
    assert adjudication.runtime_identity()["api_version"] == "v1"
