"""Deterministic tests for hosted Gemini transcription/adjudication providers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.refinement.hosted import (
    GeminiAdjudicationProvider,
    GeminiAudioTranscriptionProvider,
    HostedErrorCategory,
    HostedProviderError,
)
from app.refinement.types import AdjudicationRequest


class FakeFiles:
    def __init__(self, *, fail_upload: bool = False) -> None:
        self.uploads: list[Path] = []
        self.deleted: list[str] = []
        self.fail_upload = fail_upload

    def upload(self, *, file: Path, config: object = None) -> object:
        if self.fail_upload:
            raise RuntimeError("upload failed")
        self.uploads.append(Path(file))
        return SimpleNamespace(name="files/abc", uri="https://example.invalid/files/abc")

    def delete(self, *, name: str, config: object = None) -> object:
        self.deleted.append(name)
        return SimpleNamespace(name=name)


class FakeInteractions:
    def __init__(self, response: object | None = None, *, error: Exception | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.response = response
        self.error = error

    def create(self, **body: object) -> object:
        self.calls.append(body)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: object | None = None, *, error: Exception | None = None) -> None:
        self.files = FakeFiles()
        self.interactions = FakeInteractions(response, error=error)
        self.closed = False
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **body: object) -> object:
        return getattr(self, "_model_response", None)

    def close(self) -> None:
        self.closed = True


def _transcription_response() -> object:
    return SimpleNamespace(
        output_text="أنا عملت deploy للbackend امبارح",
        steps=[
            {
                "content": [
                    {"text": "deploy", "start_offset": "1.0s", "end_offset": "1.4s"},
                    {"text": "backend", "start_offset": "1.5s", "end_offset": "1.9s"},
                ]
            }
        ],
        usage=SimpleNamespace(total_tokens=42),
    )


def test_transcription_without_key_never_constructs_client(tmp_path: Path) -> None:
    provider = GeminiAudioTranscriptionProvider(api_key=None)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    assert provider.available() is False
    with pytest.raises(HostedProviderError) as error:
        provider.transcribe(audio)
    assert error.value.category is HostedErrorCategory.MISSING_KEY


def test_transcription_is_verbatim_one_clip_and_deletes_upload(tmp_path: Path) -> None:
    response = _transcription_response()
    client = FakeClient(response)
    provider = GeminiAudioTranscriptionProvider(
        api_key="fake-secret", client_factory=lambda: client, owns_client=False
    )
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")

    result = provider.transcribe(audio)

    assert result.transcript.startswith("أنا")
    assert [word.text for word in result.word_timestamps] == ["deploy", "backend"]
    assert client.files.uploads == [audio]
    assert client.files.deleted == ["files/abc"]
    assert len(client.interactions.calls) == 1
    call = client.interactions.calls[0]
    assert call["model"] == "gemini-3.5-transcribe"
    config = call["generation_config"]["transcription_config"]  # type: ignore[index]
    assert config["mode"] == {"type": "verbatim"}
    assert config["timestamp_granularities"] == ["word"]
    assert "custom_vocabulary" not in config
    # The remote URI is never persisted on the result.
    assert "example.invalid" not in repr(result)


def test_transcription_deletes_upload_on_error(tmp_path: Path) -> None:
    client = FakeClient(error=RuntimeError("boom"))
    provider = GeminiAudioTranscriptionProvider(
        api_key="fake-secret", client_factory=lambda: client, owns_client=False
    )
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(HostedProviderError):
        provider.transcribe(audio)
    assert client.files.deleted == ["files/abc"]


def test_transcription_falls_back_only_on_retryable_errors(tmp_path: Path) -> None:
    class Forbidden(RuntimeError):
        status_code = 403

    client = FakeClient(error=Forbidden("forbidden"))
    provider = GeminiAudioTranscriptionProvider(
        api_key="fake-secret",
        client_factory=lambda: client,
        owns_client=False,
        retry_attempts=3,
        sleep=lambda _s: None,
    )
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(HostedProviderError) as error:
        provider.transcribe(audio)
    assert error.value.category is HostedErrorCategory.AUTHENTICATION
    assert len(client.interactions.calls) == 1


def test_adjudication_selects_only_supplied_reading() -> None:
    client = FakeClient()
    client._model_response = SimpleNamespace(
        parsed={
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
    provider = GeminiAdjudicationProvider(
        api_key="fake-secret", client_factory=lambda: client, owns_client=False
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
    assert results["a1"].selected_reading == "خمسة وعشرين"
    assert results["a1"].confidence == pytest.approx(0.9)


def test_adjudication_rejects_arbitrary_rewrite_and_missing_result() -> None:
    client = FakeClient()
    client._model_response = SimpleNamespace(
        parsed={
            "adjudications": [
                {"ambiguity_id": "a1", "selected_reading": "نص جديد", "confidence": 0.99}
            ]
        }
    )
    provider = GeminiAdjudicationProvider(
        api_key="fake-secret", client_factory=lambda: client, owns_client=False
    )
    requests = [
        AdjudicationRequest(
            ambiguity_id="a1",
            context="ctx",
            candidate_readings=("خمسة وعشرين",),
            evidence_summary=(),
            dialect_profile=None,
            meaning_critical=True,
        ),
        AdjudicationRequest(
            ambiguity_id="a2",
            context="ctx",
            candidate_readings=("A", "B"),
            evidence_summary=(),
            dialect_profile=None,
            meaning_critical=False,
        ),
    ]
    results = provider.adjudicate(requests)
    assert results["a1"].selected_reading is None
    assert results["a1"].rejected_reason == "invalid_reading"
    assert results["a2"].selected_reading is None
    assert results["a2"].rejected_reason == "missing_result"


def test_release_closes_owned_client_and_scrubs_key() -> None:
    client = FakeClient(_transcription_response())
    provider = GeminiAudioTranscriptionProvider(
        api_key="fake-secret", client_factory=lambda: client, owns_client=True
    )
    provider._client_instance()
    provider.release()
    assert client.closed is True
    assert provider._api_key is None


def test_identity_never_contains_key() -> None:
    provider = GeminiAudioTranscriptionProvider(api_key="super-secret-value")
    identity = provider.runtime_identity()
    assert "super-secret-value" not in repr(identity)
    adjudicator = GeminiAdjudicationProvider(api_key="super-secret-value")
    assert "super-secret-value" not in repr(adjudicator.runtime_identity())
