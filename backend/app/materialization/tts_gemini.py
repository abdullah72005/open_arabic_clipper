"""Stage 6 Gemini TTS narration adapter and local audio-backed fidelity check.

The synthesizer executes exactly one Interactions API request per chunk (the
durable gateway reserves the attempt first), requests verbatim audio, and
persists the returned bytes as immutable lossless audio through StorageService.
Returned MIME is honored: an ``audio/wav`` payload is saved as-is (never
double-wrapped), a headerless ``audio/l16`` payload is wrapped exactly once using
its declared sample rate/channels, and any other container is refused rather than
misinterpreted. The declared container may carry parameters (the live provider
echoes ``audio/l16; rate=24000; channels=1``); the base type and its parameters
are parsed together and must agree with any numeric fields on the payload. The
API key is never logged or serialized; the SDK client is built lazily, its
automatic retries are disabled, and the key is scrubbed on release.

Local fidelity validation is a *separate* step: it transcribes the synthesized
WAV with the existing leased faster-whisper engine and applies the strict
order/occurrence-, negation-, number- and code-switch-aware comparison. A loose
similarity score is never sufficient; unknown/empty transcript defers.
"""

from __future__ import annotations

import base64
import hashlib
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from app.candidates.providers import ProviderErrorCategory
from app.materialization.narration import (
    compare_transcripts,
    inspect_wav,
    validate_narration_audio,
    wrap_pcm_as_wav,
)
from app.materialization.providers import MaterializationProviderError
from app.materialization.types import NarrationAudio, NarrationSynthesisRequest
from app.services.storage import StorageService

GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
GEMINI_TTS_API_VERSION = "v1beta"
GEMINI_TTS_SAMPLE_RATE = 24_000
GEMINI_TTS_MIME_PCM = "audio/l16"
GEMINI_TTS_MIME_WAV = "audio/wav"
_SUPPORTED_MIME = frozenset({GEMINI_TTS_MIME_PCM, GEMINI_TTS_MIME_WAV})


def _parse_audio_mime(mime: str) -> tuple[str, dict[str, str]]:
    """Split a provider container string into its base type and parameters.

    The provider may echo a parameterized container (``audio/l16; rate=24000;
    channels=1``) rather than the bare requested type. Parameters are parsed,
    never dropped, so PCM metadata is honored from the response while unknown
    base containers stay refused. A duplicate parameter whose values disagree is
    contradictory metadata and is refused (``ValueError``) rather than resolved by
    position.
    """

    segments = [segment.strip() for segment in mime.split(";")]
    base = segments[0].lower()
    params: dict[str, str] = {}
    for segment in segments[1:]:
        if not segment:
            continue
        key, separator, value = segment.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if key in params and params[key] != value:
            raise ValueError(f"conflicting duplicate mime parameter: {key}")
        params[key] = value
    return base, params


def _resolve_int(declared: object, parametric: str | None) -> int | None:
    """Resolve an integer from the declared field and/or a container parameter.

    Returns ``None`` when the declared value is present but not a valid integer
    (invalid non-null metadata is never silently replaced by a parameter), when
    the two sources disagree (a response is never interpreted from conflicting
    metadata), or when neither yields an integer.
    """

    if declared is None:
        declared_int: int | None = None
    elif isinstance(declared, bool) or not isinstance(declared, int):
        return None
    else:
        declared_int = declared
    param_int: int | None = None
    if parametric is not None:
        try:
            param_int = int(parametric)
        except ValueError:
            return None
    if declared_int is not None and param_int is not None and declared_int != param_int:
        return None
    return declared_int if declared_int is not None else param_int


def _persist_to_storage(
    storage: StorageService, source_id: str
) -> Callable[[str, bytes], tuple[str, str]]:
    from app.materialization.narration import write_narration_asset

    def persist(chunk_id: str, data: bytes) -> tuple[str, str]:
        # Content-addressed: the chunk id is a logical identifier, not a globally
        # reusable filename. Changed content gets a distinct path; identical bytes
        # deterministically reuse the same verified path and never overwrite a
        # historical or another candidate's accepted audio.
        digest = hashlib.sha256(data).hexdigest()
        filename = f"{chunk_id}-{digest[:16]}.wav"
        return write_narration_asset(storage, source_id, filename, data)

    return persist


class GeminiNarrationSynthesizer:
    """Single-request hosted narration adapter implementing the synthesis protocol."""

    provider_name = "gemini-tts"
    hosted_provider = True

    def __init__(
        self,
        *,
        api_key: object | None,
        model: str = GEMINI_TTS_MODEL,
        api_version: str = GEMINI_TTS_API_VERSION,
        voice: str = "Kore",
        style: str = "",
        mime_type: str = GEMINI_TTS_MIME_PCM,
        sample_rate: int = GEMINI_TTS_SAMPLE_RATE,
        timeout_seconds: float = 60.0,
        storage: StorageService | None = None,
        source_id: str = "",
        persist: Callable[[str, bytes], tuple[str, str]] | None = None,
        owns_client: bool = True,
        client_factory: Callable[[], object] | None = None,
    ) -> None:
        self.model = model
        self._api_version = api_version or GEMINI_TTS_API_VERSION
        self._voice = voice or "Kore"
        self._style = style or ""
        if mime_type not in _SUPPORTED_MIME:
            # An unsupported requested container is refused rather than silently
            # reinterpreted; the caller must request WAV or documented PCM.
            raise MaterializationProviderError(
                ProviderErrorCategory.INVALID_REQUEST.value,
                f"unsupported tts mime type: {mime_type}",
            )
        self._mime_type = mime_type
        self._sample_rate = max(1, int(sample_rate))
        self._timeout = max(0.1, float(timeout_seconds))
        self._owns_client = owns_client
        self._client_factory = client_factory
        self._client: object | None = None
        self._key_present = bool(api_key)
        self._api_key = api_key if self._key_present else None
        self._raw_calls = 0
        if persist is not None:
            self._persist = persist
        elif storage is not None and source_id:
            self._persist = _persist_to_storage(storage, source_id)
        else:
            self._persist = None
        self._digest = hashlib.sha256(f"gemini-tts:{model}:{self._voice}".encode()).hexdigest()

    def available(self) -> bool:
        return self._key_present

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "gemini-tts",
            "model": self.model,
            "api_version": self._api_version,
            "voice": self._voice,
            "style": self._style,
            "mime_type": self._mime_type,
            "sample_rate": self._sample_rate,
            "digest": self._digest,
        }

    def raw_call_count(self) -> int:
        return self._raw_calls

    def release(self) -> None:
        client = self._client
        self._client = None
        self._api_key = None
        if client is None or not self._owns_client:
            return
        close = getattr(client, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass

    def _build_client(self) -> object:
        return genai.Client(
            api_key=str(self._api_key or ""),
            http_options=types.HttpOptions(
                api_version=self._api_version,
                timeout=int(self._timeout * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def _client_instance(self) -> object | None:
        if self._client is not None:
            return self._client
        if not self._key_present:
            return None
        try:
            client = self._client_factory() if self._client_factory else self._build_client()
        except MaterializationProviderError:
            raise
        except Exception:
            raise MaterializationProviderError(
                ProviderErrorCategory.PROVIDER_ERROR.value, "gemini tts client construction failed"
            ) from None
        self._disable_interactions_retries(client)
        self._client = client
        return client

    @staticmethod
    def _disable_interactions_retries(client: object) -> None:
        """Establish the verified no-retry boundary on the private Interactions client.

        The generated Interactions resource ignores the parent ``HttpRetryOptions``
        (an ``attempts=0`` parent is normalized to one before the Interactions
        bridge consumes it) and retries from its own
        ``sdk_configuration.retry_config``; the durable gateway must observe
        exactly one raw HTTP request per reserved attempt. Only this client-local
        Interactions configuration is changed — never the shared generateContent
        retry configuration, and never a global SDK monkeypatch. If the installed
        SDK cannot establish the verified no-retry behavior, fail closed before
        any invocation.
        """

        interactions = getattr(client, "interactions", None)
        sdk_configuration = getattr(interactions, "sdk_configuration", None)
        if sdk_configuration is None or not hasattr(sdk_configuration, "retry_config"):
            raise MaterializationProviderError(
                ProviderErrorCategory.PROVIDER_ERROR.value,
                "installed SDK cannot establish the no-retry interactions boundary",
            )
        sdk_configuration.retry_config = None

    def synthesize(self, request: NarrationSynthesisRequest) -> NarrationAudio:
        if not self._key_present:
            raise MaterializationProviderError(ProviderErrorCategory.MISSING_KEY.value)
        if not request.text.strip():
            raise MaterializationProviderError(ProviderErrorCategory.INVALID_REQUEST.value)
        client = self._client_instance()
        if client is None:
            raise MaterializationProviderError(ProviderErrorCategory.MISSING_KEY.value)
        self._raw_calls += 1
        voice = request.voice or self._voice
        style = request.style or self._style
        annotation: dict[str, object] = {"type": "speech_metadata"}
        if style:
            annotation["style"] = style
        try:
            interaction = client.interactions.create(  # type: ignore[attr-defined]
                model=self.model,
                input=[
                    {
                        "type": "user_input",
                        "content": [
                            {
                                "type": "text",
                                "text": request.text,
                                "annotations": [annotation],
                            }
                        ],
                    }
                ],
                response_format={
                    "type": "audio",
                    "mime_type": self._mime_type,
                    "sample_rate": self._sample_rate,
                },
                generation_config={"speech_config": [{"voice": voice}]},
            )
        except Exception as error:
            raise MaterializationProviderError(_classify_exception(error)) from None
        return self._narration_from_interaction(request, interaction, voice, style)

    def _narration_from_interaction(
        self,
        request: NarrationSynthesisRequest,
        interaction: Any,
        voice: str,
        style: str,
    ) -> NarrationAudio:
        audio = getattr(interaction, "output_audio", None)
        if audio is None and isinstance(interaction, dict):
            audio = interaction.get("output_audio")
        if audio is None:
            raise MaterializationProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        data_b64 = audio.get("data") if isinstance(audio, dict) else getattr(audio, "data", None)
        mime = (
            audio.get("mime_type")
            if isinstance(audio, dict)
            else getattr(audio, "mime_type", None)
        )
        channels = (
            audio.get("channels")
            if isinstance(audio, dict)
            else getattr(audio, "channels", None)
        )
        sample_rate = (
            audio.get("sample_rate")
            if isinstance(audio, dict)
            else getattr(audio, "sample_rate", None)
        )
        if not isinstance(data_b64, str) or not data_b64:
            raise MaterializationProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        try:
            raw = base64.b64decode(data_b64, validate=True)
        except Exception:
            raise MaterializationProviderError(
                ProviderErrorCategory.MALFORMED_OUTPUT.value
            ) from None
        if not isinstance(mime, str) or not mime:
            # A response without a declared container is refused: the format is
            # never guessed from the request.
            raise MaterializationProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        try:
            base_mime, mime_params = _parse_audio_mime(mime)
        except ValueError:
            # Contradictory duplicate parameters are refused, never resolved by
            # position.
            raise MaterializationProviderError(
                ProviderErrorCategory.MALFORMED_OUTPUT.value
            ) from None
        if base_mime == GEMINI_TTS_MIME_WAV:
            wav_bytes = raw
        elif base_mime == GEMINI_TTS_MIME_PCM:
            # The live provider echoes a parameterized container
            # (e.g. ``audio/l16; rate=24000; channels=1``); the declared numeric
            # fields and any container parameters must agree before raw PCM is
            # wrapped, and neither is ever guessed.
            resolved_rate = _resolve_int(sample_rate, mime_params.get("rate"))
            resolved_channels = _resolve_int(channels, mime_params.get("channels"))
            if (
                resolved_rate is None
                or resolved_rate <= 0
                or resolved_channels is None
                or resolved_channels <= 0
            ):
                raise MaterializationProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
            wav_bytes = wrap_pcm_as_wav(raw, sample_rate=resolved_rate, channels=resolved_channels)
        else:
            raise MaterializationProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        try:
            info = inspect_wav(wav_bytes)
        except ValueError:
            raise MaterializationProviderError(
                ProviderErrorCategory.MALFORMED_OUTPUT.value
            ) from None
        if self._persist is None:
            raise MaterializationProviderError(
                ProviderErrorCategory.PROVIDER_ERROR.value, "narration storage is not bound"
            )
        relative_path, sha256 = self._persist(request.chunk_id, wav_bytes)
        validation = validate_narration_audio(info, expected_text=request.text, config=_config())
        return NarrationAudio(
            chunk_id=request.chunk_id,
            block_index=request.block_index,
            text=request.text,
            relative_path=relative_path,
            sha256=sha256,
            size_bytes=len(wav_bytes),
            duration_seconds=info.duration_seconds,
            sample_rate=info.sample_rate,
            channels=info.channels,
            container="wav",
            provider_identity={
                **request.identity(),
                "voice": voice,
                "style": style,
                "model": self.model,
                "api_version": self._api_version,
            },
            validation=validation,
        )


def _config() -> Any:
    from app.materialization.policy import DEFAULT_CONFIG

    return DEFAULT_CONFIG


def _classify_exception(error: Exception) -> str:
    if isinstance(error, MaterializationProviderError):
        return error.category
    kind = type(error).__name__.casefold()
    if "timeout" in kind:
        return ProviderErrorCategory.TIMEOUT.value
    if "connect" in kind:
        return ProviderErrorCategory.CONNECTION.value
    if isinstance(error, TimeoutError):
        return ProviderErrorCategory.TIMEOUT.value
    if isinstance(error, (ConnectionError, OSError)):
        return ProviderErrorCategory.CONNECTION.value
    code = getattr(error, "code", None)
    if not isinstance(code, int):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
        code = status if isinstance(status, int) else None
    if code in {401, 403}:
        return ProviderErrorCategory.AUTHENTICATION.value
    if code == 429:
        return ProviderErrorCategory.RATE_LIMITED.value
    if code in {408, 504} or "timeout" in str(error).casefold():
        return ProviderErrorCategory.TIMEOUT.value
    if code == 503:
        return ProviderErrorCategory.SERVICE_UNAVAILABLE.value
    if code is not None and code >= 500:
        return ProviderErrorCategory.PROVIDER_ERROR.value
    return ProviderErrorCategory.PROVIDER_ERROR.value


# Local audio-backed fidelity validation


FIDELITY_ACCEPTED = "ACCEPTED"
FIDELITY_DEFERRED = "DEFERRED"
FIDELITY_REJECTED = "REJECTED"

#: Version of the local fidelity validation policy. A change to the comparison,
#: the deferred/rejected classification, or the audio-failure handling must bump
#: this so dependent validation fingerprints (which include it) invalidate and a
#: retained audio is revalidated locally without repeating synthesis.
FIDELITY_POLICY_VERSION = "stage6-fidelity-v3"

#: Findings from ``validate_narration_audio`` that are independently established
#: audio/content failures: they reject the chunk regardless of the transcript.
_FIDELITY_AUDIO_FAILURE_FINDINGS = frozenset(
    {
        "EMPTY_AUDIO",
        "SILENT_AUDIO",
        "TRUNCATED_AUDIO",
        "CLIPPED_AUDIO",
        "INVALID_SAMPLE_RATE",
        "INVALID_CHANNELS",
        "NARRATION_TOO_LONG",
    }
)


@dataclass(frozen=True)
class NarrationFidelityResult:
    """Outcome of one local audio-backed narration fidelity check."""

    status: str
    transcript: str
    findings: tuple[str, ...]
    asr_identity: dict[str, object]
    diff: dict[str, object] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == FIDELITY_ACCEPTED

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "policy_version": FIDELITY_POLICY_VERSION,
            "transcript": self.transcript,
            "findings": list(self.findings),
            "asr_identity": dict(self.asr_identity),
            "diff": dict(self.diff),
        }


def validate_narration_fidelity(
    audio_path: Path,
    *,
    expected_text: str,
    engine: Any,
    options: Any,
    lease_factory: Any | None = None,
    cancel_event: threading.Event | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    audio_validation: Mapping[str, object] | None = None,
) -> NarrationFidelityResult:
    """Transcribe the synthesized audio under the heavy-model lease and compare.

    Lease loss and (optional) executor cancellation are connected to the Whisper
    child's cancellation event: a lost lease or a cancelled/foreign job sets the
    event, which terminates and reaps the child before the lease is released.

    Accept/reject/defer classification (policy ``stage6-fidelity-v2``):

    * an independently established audio/content failure (silent/empty/truncated/
      clipped/invalid audio, reported by the synthesizer's own validation)
      rejects;
    * a lost lease or a missing/unusable transcript defers;
    * an exact match (ordered content sequences plus atomic protected technical/
      numeric forms) accepts;
    * a lone ASR transcript disagreement **defers** with a structured reason/diff
      — a single disagreeing transcript is evidence, not a confirmed meaning
      mismatch.

    Deferral never approves a voice or review readiness. The comparison is the
    strict check in ``narration.compare_transcripts``; a scalar similarity ratio is
    never accepted as proof of meaning.
    """

    from app.runtime.heavy_model_lease import NoopHeavyModelLeaseFactory

    factory = lease_factory if lease_factory is not None else NoopHeavyModelLeaseFactory()
    event = cancel_event if cancel_event is not None else threading.Event()

    audio_findings = ()
    if isinstance(audio_validation, Mapping):
        raw_findings = audio_validation.get("findings")
        if isinstance(raw_findings, (list, tuple)):
            audio_findings = tuple(str(item) for item in raw_findings)
    established = tuple(item for item in audio_findings if item in _FIDELITY_AUDIO_FAILURE_FINDINGS)
    if established:
        # The audio itself is unusable: an independently established failure does
        # not need (and must not wait on) a transcript.
        return NarrationFidelityResult(FIDELITY_REJECTED, "", established, {})

    def _on_ownership_lost() -> None:
        event.set()

    watcher_stop = threading.Event()
    watcher: threading.Thread | None = None
    if is_cancelled is not None:

        def _watch() -> None:
            while not watcher_stop.wait(0.1):
                if is_cancelled():
                    event.set()
                    return

        watcher = threading.Thread(target=_watch, name="narration-fidelity-cancel", daemon=True)
        watcher.start()

    ownership_lost = False
    try:
        with factory.acquire(
            purpose="narration-fidelity", on_ownership_lost=_on_ownership_lost
        ) as lease:
            result = engine.transcribe(audio_path, options, cancel_event=event)
            ownership_lost = bool(getattr(lease, "ownership_lost", False))
    finally:
        watcher_stop.set()
        if watcher is not None:
            watcher.join(timeout=1.0)
    asr_identity = {
        "provider": "faster-whisper",
        "model": getattr(options, "model", ""),
        "device": getattr(options, "device", ""),
        "compute_type": getattr(options, "compute_type", ""),
        "beam_size": getattr(options, "beam_size", None),
        "language": getattr(result, "language", None),
        "policy_version": FIDELITY_POLICY_VERSION,
    }
    if ownership_lost or event.is_set():
        return NarrationFidelityResult(
            FIDELITY_DEFERRED, "", ("LEASE_LOST",), asr_identity
        )
    transcript = str(getattr(result, "raw_text", "") or "").strip()
    if not transcript:
        return NarrationFidelityResult(
            FIDELITY_DEFERRED, "", ("EMPTY_TRANSCRIPT",), asr_identity
        )
    comparison = compare_transcripts(expected_text, transcript)
    if comparison.matches:
        return NarrationFidelityResult(FIDELITY_ACCEPTED, transcript, (), asr_identity)
    # A lone transcript disagreement is deferred evidence, never a confirmed
    # meaning mismatch. The structured diff preserves exactly what disagreed.
    return NarrationFidelityResult(
        FIDELITY_DEFERRED,
        transcript,
        ("ASR_TRANSCRIPT_DISAGREEMENT", *comparison.findings),
        asr_identity,
        dict(comparison.diff),
    )


__all__ = [
    "FIDELITY_ACCEPTED",
    "FIDELITY_DEFERRED",
    "FIDELITY_POLICY_VERSION",
    "FIDELITY_REJECTED",
    "GEMINI_TTS_API_VERSION",
    "GEMINI_TTS_MIME_PCM",
    "GEMINI_TTS_MIME_WAV",
    "GEMINI_TTS_MODEL",
    "GeminiNarrationSynthesizer",
    "NarrationFidelityResult",
    "validate_narration_fidelity",
]
