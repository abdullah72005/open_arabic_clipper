"""Stage 6 Gemini TTS adapter boundary + local fidelity validation tests.

The synthesizer is exercised through the *real* installed google-genai SDK using
a fake httpx transport, so the Interactions v1beta request serialization (model,
verbatim input, speech metadata, response_format, speech_config voice) and the
single-call/no-retry behavior are asserted at the actual SDK boundary. No live
network call is made. Local fidelity validation uses a fake ASR engine.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
from google import genai
from google.genai import types

from app.materialization.narration import inspect_wav, wrap_pcm_as_wav
from app.materialization.providers import MaterializationProviderError
from app.materialization.tts_gemini import (
    FIDELITY_ACCEPTED,
    FIDELITY_DEFERRED,
    FIDELITY_REJECTED,
    GEMINI_TTS_MODEL,
    GeminiNarrationSynthesizer,
    validate_narration_fidelity,
)
from app.materialization.types import NarrationSynthesisRequest


def _request(text: str = "مرحبا بكم في القناة") -> NarrationSynthesisRequest:
    return NarrationSynthesisRequest(
        chunk_id="narr-1",
        block_index=1,
        text=text,
        language_hint="ar",
        provider="gemini-tts",
        model=GEMINI_TTS_MODEL,
        api_version="v1beta",
        voice="Kore",
        style="calm",
    )


def _source_wav(seconds: float = 0.1, rate: int = 24_000, tone: int = 800) -> bytes:
    import struct

    frames = max(1, int(seconds * rate))
    pcm = struct.pack(f"<{frames}h", *([tone] * frames))
    return wrap_pcm_as_wav(pcm, sample_rate=rate, channels=1)


class _Transport:
    def __init__(self, response: dict[str, object]) -> None:
        self.requests: list[httpx.Request] = []
        self._response = response
        self.status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self._response)

    def client(self) -> genai.Client:
        return genai.Client(
            api_key="fake-key",
            http_options=types.HttpOptions(
                api_version="v1beta",
                retry_options=types.HttpRetryOptions(attempts=1),
                httpx_client=httpx.Client(transport=httpx.MockTransport(self.handler)),
            ),
        )


def _interaction_response(
    mime: str, data: bytes, *, rate: int | None, channels: int | None
) -> dict:
    audio: dict[str, object] = {
        "type": "audio",
        "mime_type": mime,
        "data": base64.b64encode(data).decode(),
    }
    if rate is not None:
        audio["sample_rate"] = rate
    if channels is not None:
        audio["channels"] = channels
    return {
        "id": "int_1",
        "model": GEMINI_TTS_MODEL,
        "steps": [{"type": "model_output", "content": [audio]}],
    }


def test_tts_pcm_is_wrapped_once_and_request_is_v1beta() -> None:
    payload = _interaction_response("audio/l16", b"\x10\x20" * 50, rate=24_000, channels=1)
    transport = _Transport(payload)
    persisted: dict[str, bytes] = {}

    def persist(chunk_id: str, data: bytes) -> tuple[str, str]:
        persisted[chunk_id] = data
        return f"narration/{chunk_id}.wav", "a" * 64

    provider = GeminiNarrationSynthesizer(
        api_key="fake-key", client_factory=transport.client, persist=persist
    )
    audio = provider.synthesize(_request())

    assert audio.container == "wav"
    assert audio.sample_rate == 24_000
    assert audio.channels == 1
    assert audio.relative_path == "narration/narr-1.wav"
    assert "narr-1" in persisted
    # A headerless PCM payload is wrapped exactly once.
    info = inspect_wav(persisted["narr-1"])
    assert info.sample_rate == 24_000
    assert info.channels == 1
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/interactions"
    body = json.loads(request.content.decode())
    assert body["model"] == GEMINI_TTS_MODEL
    assert body["response_format"]["type"] == "audio"
    assert body["response_format"]["mime_type"] == "audio/l16"
    assert body["generation_config"]["speech_config"] == [{"voice": "Kore"}]
    content = body["input"][0]["content"][0]
    assert content["text"] == _request().text
    assert content["annotations"][0]["type"] == "speech_metadata"
    assert content["annotations"][0]["style"] == "calm"
    provider.release()
    assert provider._api_key is None


def test_tts_wav_is_not_double_wrapped() -> None:
    wav = _source_wav()
    transport = _Transport(_interaction_response("audio/wav", wav, rate=None, channels=None))
    persisted: dict[str, bytes] = {}

    def persist(chunk_id: str, data: bytes) -> tuple[str, str]:
        persisted[chunk_id] = data
        return f"narration/{chunk_id}.wav", "b" * 64

    provider = GeminiNarrationSynthesizer(
        api_key="fake-key", client_factory=transport.client, persist=persist
    )
    provider.synthesize(_request())
    assert persisted["narr-1"] == wav


def test_tts_parameterized_pcm_mime_is_wrapped_from_declared_params() -> None:
    # The live provider echoes a parameterized container rather than bare
    # ``audio/l16``; the declared params must drive the single PCM wrap.
    payload = _interaction_response(
        "audio/l16; rate=24000; channels=1", b"\x10\x20" * 50, rate=None, channels=None
    )
    transport = _Transport(payload)
    persisted: dict[str, bytes] = {}

    def persist(chunk_id: str, data: bytes) -> tuple[str, str]:
        persisted[chunk_id] = data
        return f"narration/{chunk_id}.wav", "c" * 64

    provider = GeminiNarrationSynthesizer(
        api_key="fake-key", client_factory=transport.client, persist=persist
    )
    audio = provider.synthesize(_request())

    assert audio.sample_rate == 24_000
    assert audio.channels == 1
    info = inspect_wav(persisted["narr-1"])
    assert info.sample_rate == 24_000
    assert info.channels == 1


def test_tts_conflicting_pcm_metadata_is_refused() -> None:
    # A container parameter that disagrees with the declared numeric field is
    # never reconciled by guessing.
    payload = _interaction_response(
        "audio/l16; rate=24000; channels=1", b"\x10\x20" * 50, rate=48_000, channels=1
    )
    transport = _Transport(payload)
    provider = GeminiNarrationSynthesizer(
        api_key="fake-key", client_factory=transport.client, persist=lambda c, d: ("p", "h")
    )
    try:
        provider.synthesize(_request())
    except MaterializationProviderError as error:
        assert error.category == "MALFORMED_OUTPUT"
    else:  # pragma: no cover
        raise AssertionError("expected MALFORMED_OUTPUT")


def test_tts_unknown_parameterized_container_is_refused() -> None:
    payload = _interaction_response(
        "audio/mpeg; rate=24000", b"\x10\x20" * 50, rate=None, channels=None
    )
    transport = _Transport(payload)
    provider = GeminiNarrationSynthesizer(
        api_key="fake-key", client_factory=transport.client, persist=lambda c, d: ("p", "h")
    )
    try:
        provider.synthesize(_request())
    except MaterializationProviderError as error:
        assert error.category == "MALFORMED_OUTPUT"
    else:  # pragma: no cover
        raise AssertionError("expected MALFORMED_OUTPUT")


def test_tts_missing_key_is_unavailable_and_raises() -> None:
    provider = GeminiNarrationSynthesizer(api_key=None)
    assert provider.available() is False
    try:
        provider.synthesize(_request())
    except MaterializationProviderError as error:
        assert error.category == "MISSING_KEY"
    else:  # pragma: no cover
        raise AssertionError("expected MISSING_KEY")


def test_tts_malformed_output_raises() -> None:
    transport = _Transport({"id": "int_1", "model": GEMINI_TTS_MODEL, "steps": []})
    provider = GeminiNarrationSynthesizer(
        api_key="fake-key", client_factory=transport.client, persist=lambda c, d: ("p", "h")
    )
    try:
        provider.synthesize(_request())
    except MaterializationProviderError as error:
        assert error.category == "MALFORMED_OUTPUT"
    else:  # pragma: no cover
        raise AssertionError("expected MALFORMED_OUTPUT")


def test_tts_unavailable_without_bound_storage_raises() -> None:
    payload = _interaction_response("audio/l16", b"\x00\x01" * 20, rate=24_000, channels=1)
    transport = _Transport(payload)
    provider = GeminiNarrationSynthesizer(api_key="fake-key", client_factory=transport.client)
    try:
        provider.synthesize(_request())
    except MaterializationProviderError as error:
        assert error.category == "PROVIDER_ERROR"
    else:  # pragma: no cover
        raise AssertionError("expected a storage-binding failure")


def test_tts_identity_excludes_key_and_reports_v1beta() -> None:
    provider = GeminiNarrationSynthesizer(api_key="super-secret-key")
    identity = provider.runtime_identity()
    assert "super-secret-key" not in str(identity)
    assert identity["api_version"] == "v1beta"
    assert identity["model"] == GEMINI_TTS_MODEL


class _FakeASREngine:
    def __init__(self, transcript: str) -> None:
        self._transcript = transcript

    def transcribe(self, path: Any, options: Any, cancel_event: Any = None) -> Any:
        return type(
            "_Result",
            (),
            {"raw_text": self._transcript, "language": "ar"},
        )()


class _Opts:
    model = "small"
    device = "cpu"
    compute_type = "int8"
    beam_size = 5


def test_fidelity_accepts_matching_transcript() -> None:
    engine = _FakeASREngine("مرحبا بكم في القناة")
    result = validate_narration_fidelity(
        __import__("pathlib").Path("x.wav"),
        expected_text="مرحبا بكم في القناة",
        engine=engine,
        options=_Opts(),
    )
    assert result.status == FIDELITY_ACCEPTED
    assert result.accepted is True


def test_fidelity_defers_on_empty_transcript() -> None:
    engine = _FakeASREngine("   ")
    result = validate_narration_fidelity(
        __import__("pathlib").Path("x.wav"),
        expected_text="مرحبا بكم في القناة",
        engine=engine,
        options=_Opts(),
    )
    assert result.status == FIDELITY_DEFERRED
    assert "EMPTY_TRANSCRIPT" in result.findings


def test_fidelity_rejects_negation_scope_reversal() -> None:
    engine = _FakeASREngine("this will increase the price")
    result = validate_narration_fidelity(
        __import__("pathlib").Path("x.wav"),
        expected_text="this will not increase the price",
        engine=engine,
        options=_Opts(),
    )
    assert result.status == FIDELITY_REJECTED
    assert "MEANING_MISMATCH" in result.findings
