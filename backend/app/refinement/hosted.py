"""Optional hosted Gemini transcription and adjudication for Stage 3.5.

Two providers, deliberately separate from the Stage 2.7 text-reconstruction
provider:

* :class:`GeminiAudioTranscriptionProvider` uses the documented Interactions API
  with the dedicated ``gemini-3.5-transcribe`` model in **verbatim** mode. It
  uploads exactly one bounded candidate WAV, requests word timestamps, never
  enables smart transcription, and never combines custom vocabulary with
  timestamps in the same pass.
* :class:`GeminiAdjudicationProvider` uses ``gemini-3.8-flash`` with structured
  output to choose one supplied reading or ``UNRESOLVED``. It never rewrites a
  whole transcript and never recovers omitted English (that belongs to ASR).

Official documentation was checked on 2026-09-12 (Gemini API audio
transcription, Gemini 3.5 Transcribe, Gemini 3.8 Flash, Files API, Interactions
API). The installed SDK is ``google-genai`` 2.23.0; the Interactions API surface
used here (``client.interactions.create`` with
``generation_config.transcription_config``, ``VerbatimTranscriptionMode``,
``WordInfo`` and ``client.files.upload``/``delete``) was verified against that
installed package. Remote uploads are deleted in ``finally`` on every exit path
and their URIs are never persisted or logged. API keys are unwrapped only at
lazy client construction and never logged.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.refinement.policy import (
    ADJUDICATION_SCHEMA_VERSION,
    HOSTED_TRANSCRIPTION_SCHEMA_VERSION,
)
from app.refinement.types import AdjudicationRequest, AdjudicationResult, WordTimestamp

_GEMINI_API_VERSION = "v1"
_TRANSCRIPTION_MODEL = "gemini-3.5-transcribe"
_RETRYABLE_CATEGORIES = frozenset(
    {"CONNECTION", "TIMEOUT", "SERVICE_UNAVAILABLE", "PROVIDER_ERROR"}
)


class HostedErrorCategory(str, Enum):
    """Sanitized hosted-provider error categories."""

    MISSING_KEY = "MISSING_KEY"
    AUTHENTICATION = "AUTHENTICATION"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    INVALID_REQUEST = "INVALID_REQUEST"
    CONNECTION = "CONNECTION"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    SAFETY_REFUSAL = "SAFETY_REFUSAL"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"


class HostedProviderError(Exception):
    """A hosted-provider failure with a sanitized category and never a key."""

    def __init__(self, category: HostedErrorCategory, detail: str = "") -> None:
        self.category = category
        self.detail = detail
        super().__init__(f"hosted_{category.value}")


@dataclass(frozen=True)
class HostedTranscriptionResult:
    """Bounded audio-backed transcription of one candidate window."""

    transcript: str
    language: str | None
    language_probability: float | None
    word_timestamps: tuple[WordTimestamp, ...]
    confidence: float
    provider: str = "gemini"
    model: str = _TRANSCRIPTION_MODEL
    rejected_reason: str | None = None
    usage: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _UploadedFile:
    name: str
    uri: str


def _secret_value(api_key: object | None) -> str | None:
    if api_key is None:
        return None
    secret = getattr(api_key, "get_secret_value", None)
    value = secret() if callable(secret) else api_key
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _stable_digest(model: str) -> str:
    import hashlib

    return hashlib.sha256(f"gemini-model:{model}".encode("utf-8")).hexdigest()


def _coerce_offset(value: object) -> float | None:
    """Best-effort conversion of a provider offset to seconds.

    The SDK exposes word ``start_offset``/``end_offset`` as opaque values; this
    accepts plain numbers, numeric strings, and ``"<n>ms"``/``"<n>s"`` strings.
    Values that look like milliseconds are scaled. Malformed values return
    ``None`` and the caller drops the annotation instead of guessing.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        multiplier = 1.0
        if text.endswith("ms"):
            multiplier = 0.001
            text = text[:-2]
        elif text.endswith("s"):
            multiplier = 1.0
            text = text[:-1]
        try:
            numeric = float(text)
        except ValueError:
            return None
        numeric *= multiplier
    else:
        return None
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return numeric


def _iter_word_annotations(node: object) -> list[dict[str, object]]:
    """Recursively collect word annotations without trusting one shape."""

    found: list[dict[str, object]] = []
    stack: list[object] = [node]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, list):
            for item in current:
                stack.append(item)
            continue
        if isinstance(current, dict):
            if {"start_offset", "end_offset"} <= set(current.keys()) or (
                "start_offset" in current and "text" in current
            ):
                found.append(current)
            for value in current.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
            continue
        # pydantic models / SDK objects
        fields = getattr(current, "model_fields", None)
        if fields:
            data = {name: getattr(current, name, None) for name in fields}
            if "start_offset" in fields and "end_offset" in fields:
                found.append(data)
            for value in data.values():
                if isinstance(value, (dict, list)) or hasattr(value, "model_fields"):
                    stack.append(value)
            continue
        for attr in ("steps", "content", "annotations", "words"):
            value = getattr(current, attr, None)
            if isinstance(value, (dict, list, str)) or hasattr(value, "model_fields"):
                stack.append(value)
    return found


def _extract_transcript(response: object) -> str:
    for attr in ("output_text", "text"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    parsed = getattr(response, "parsed", None)
    if parsed is not None and not isinstance(parsed, (list, dict)):
        text = getattr(parsed, "text", None)
        if isinstance(text, str):
            return text
    return ""


def _parse_word_annotations(response: object) -> tuple[WordTimestamp, ...]:
    words: list[WordTimestamp] = []
    for annotation in _iter_word_annotations(response):
        start = _coerce_offset(annotation.get("start_offset"))
        end = _coerce_offset(annotation.get("end_offset"))
        text = annotation.get("text") or annotation.get("word")
        if start is None or end is None or not isinstance(text, str) or not text.strip():
            continue
        if end < start:
            continue
        probability = annotation.get("probability")
        words.append(
            WordTimestamp(
                text=text,
                start=start,
                end=end,
                probability=float(probability) if isinstance(probability, (int, float)) else None,
            )
        )
    words.sort(key=lambda word: (word.start, word.end))
    return tuple(words)


def _classify_exception(error: BaseException) -> HostedErrorCategory:
    if isinstance(error, TimeoutError):
        return HostedErrorCategory.TIMEOUT
    if isinstance(error, (ConnectionError, OSError)):
        return HostedErrorCategory.CONNECTION
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    code = getattr(error, "status", None)
    resolved = status if isinstance(status, int) else (code if isinstance(code, int) else None)
    if resolved == 401 or resolved == 403:
        return HostedErrorCategory.AUTHENTICATION
    if resolved == 404:
        return HostedErrorCategory.MODEL_NOT_FOUND
    if resolved == 429:
        return HostedErrorCategory.RATE_LIMITED
    if resolved in (408, 504):
        return HostedErrorCategory.TIMEOUT
    if resolved == 503:
        return HostedErrorCategory.SERVICE_UNAVAILABLE
    if resolved is not None and 500 <= resolved < 600:
        return HostedErrorCategory.PROVIDER_ERROR
    if resolved is not None and 400 <= resolved < 500:
        return HostedErrorCategory.INVALID_REQUEST
    message = str(error).lower()
    if "safety" in message or "blocked" in message or "prohibited" in message:
        return HostedErrorCategory.SAFETY_REFUSAL
    return HostedErrorCategory.PROVIDER_ERROR


class _LazyClientMixin:
    _api_key: str | None
    _client: object | None
    _owns_client: bool
    _client_factory: Callable[..., object] | None
    _timeout_seconds: float

    def _client_instance(self) -> object | None:
        if self._client is not None:
            return self._client
        if self._api_key is None:
            return None
        factory = self._client_factory or self._build_client
        self._client = factory()
        return self._client

    def _build_client(self) -> object:
        from google import genai
        from google.genai import types

        assert self._api_key is not None
        return genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                api_version=_GEMINI_API_VERSION,
                timeout=int(self._timeout_seconds * 1000),
            ),
        )

    def _release_client(self) -> None:
        client = self._client
        self._client = None
        self._api_key = None
        if client is None or not self._owns_client:
            return
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _run_with_retry(
    call: Callable[[], object],
    *,
    attempts: int,
    sleep: Callable[[float], None],
    backoff_seconds: float,
) -> object:
    remaining = max(0, attempts)
    while True:
        try:
            return call()
        except Exception as error:
            category = _classify_exception(error)
            if category.value not in _RETRYABLE_CATEGORIES or remaining <= 0:
                raise HostedProviderError(category) from None
            remaining -= 1
            sleep(backoff_seconds)


class GeminiAudioTranscriptionProvider(_LazyClientMixin):
    """Hosted verbatim transcription of one bounded candidate WAV at a time."""

    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: object | None,
        model: str = _TRANSCRIPTION_MODEL,
        api_version: str = _GEMINI_API_VERSION,
        timeout_seconds: float = 60.0,
        retry_attempts: int = 1,
        retry_backoff_seconds: float = 1.5,
        max_output_tokens: int = 2_048,
        owns_client: bool = True,
        client_factory: Callable[..., object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = _secret_value(api_key)
        self._key_present = self._api_key is not None
        self.model = model
        self._api_version = api_version
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._max_output_tokens = max_output_tokens
        self._owns_client = owns_client
        self._client_factory = client_factory
        self._client: object | None = None
        self._sleep = sleep

    def available(self) -> bool:
        return self._key_present

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "gemini",
            "model": self.model,
            "digest": _stable_digest(self.model),
            "schema_version": HOSTED_TRANSCRIPTION_SCHEMA_VERSION,
            "api": "interactions",
            "api_version": self._api_version,
            "mode": "verbatim",
            "word_timestamps": True,
            "max_output_tokens": self._max_output_tokens,
        }

    def transcribe(
        self,
        audio_path: Path,
        *,
        language_codes: Sequence[str] | None = None,
        dialect_profile: str | None = None,
        custom_vocabulary: Sequence[str] = (),
        include_word_timestamps: bool = True,
    ) -> HostedTranscriptionResult:
        client = self._client_instance()
        if client is None:
            raise HostedProviderError(HostedErrorCategory.MISSING_KEY)
        if not audio_path.is_file():
            raise HostedProviderError(HostedErrorCategory.INVALID_REQUEST, "audio missing")

        uploaded = self._upload(client, audio_path)
        try:
            response = _run_with_retry(
                lambda: self._create_transcription(
                    client,
                    uploaded,
                    language_codes=language_codes,
                    custom_vocabulary=custom_vocabulary,
                    include_word_timestamps=include_word_timestamps,
                ),
                attempts=self._retry_attempts,
                sleep=self._sleep,
                backoff_seconds=self._retry_backoff_seconds,
            )
        finally:
            self._delete(client, uploaded)

        transcript = _extract_transcript(response)
        words = _parse_word_annotations(response) if include_word_timestamps else ()
        rejected_reason: str | None = None
        if not transcript.strip():
            rejected_reason = "empty hosted transcript"
        probabilities = [word.probability for word in words if word.probability is not None]
        confidence = (
            sum(probabilities) / len(probabilities)
            if probabilities
            else (0.5 if transcript else 0.0)
        )
        language = None
        language_probability = None
        for candidate in _iter_language_fields(response):
            language, language_probability = candidate
            break
        return HostedTranscriptionResult(
            transcript=transcript,
            language=language,
            language_probability=language_probability,
            word_timestamps=words,
            confidence=confidence,
            rejected_reason=rejected_reason,
            usage=_usage_dict(response),
        )

    # internal

    def _upload(self, client: object, audio_path: Path) -> _UploadedFile:
        from google.genai import types

        result = _run_with_retry(
            lambda: client.files.upload(  # type: ignore[attr-defined]
                file=audio_path,
                config=types.UploadFileConfig(mime_type="audio/wav"),
            ),
            attempts=self._retry_attempts,
            sleep=self._sleep,
            backoff_seconds=self._retry_backoff_seconds,
        )
        name = getattr(result, "name", None)
        uri = getattr(result, "uri", None)
        if not isinstance(name, str) or not isinstance(uri, str):
            raise HostedProviderError(HostedErrorCategory.MALFORMED_OUTPUT, "upload response")
        return _UploadedFile(name=name, uri=uri)

    def _create_transcription(
        self,
        client: object,
        uploaded: _UploadedFile,
        *,
        language_codes: Sequence[str] | None,
        custom_vocabulary: Sequence[str],
        include_word_timestamps: bool,
    ) -> object:
        # Word timestamps and custom vocabulary are never combined in one pass.
        vocabulary = [] if include_word_timestamps else list(custom_vocabulary)
        transcription_config: dict[str, object] = {
            "mode": {"type": "verbatim"},
        }
        if language_codes:
            transcription_config["language_codes"] = list(language_codes)
        if include_word_timestamps:
            transcription_config["timestamp_granularities"] = ["word"]
        if vocabulary:
            transcription_config["custom_vocabulary"] = vocabulary
        body = {
            "model": self.model,
            "input": [
                {
                    "type": "user_input",
                    "content": [{"type": "audio", "uri": uploaded.uri, "mime_type": "audio/wav"}],
                }
            ],
            "generation_config": {"transcription_config": transcription_config},
        }
        return client.interactions.create(  # type: ignore[attr-defined]
            api_version=self._api_version,
            timeout=self._timeout_seconds,
            **body,
        )

    def _delete(self, client: object, uploaded: _UploadedFile) -> None:
        delete = getattr(getattr(client, "files", None), "delete", None)
        if not callable(delete):
            return
        try:
            delete(name=uploaded.name)
        except Exception:
            # Best-effort cleanup; never replaces a valid result or leaks a URI.
            pass

    def release(self) -> None:
        self._release_client()


def _iter_language_fields(response: object) -> list[tuple[str | None, float | None]]:
    values: list[tuple[str | None, float | None]] = []
    for attr in ("language_code", "detected_language", "language"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value:
            values.append((value, None))
    return values


def _usage_dict(response: object) -> dict[str, object]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    fields = getattr(usage, "model_fields", None)
    if fields:
        return {
            name: getattr(usage, name, None)
            for name in fields
            if isinstance(getattr(usage, name, None), (int, float, str, bool, type(None)))
        }
    return {}


class GeminiAdjudicationProvider(_LazyClientMixin):
    """Selective Flash reasoning adjudication over bounded ambiguity readings."""

    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: object | None,
        model: str = "gemini-3.8-flash",
        api_version: str = _GEMINI_API_VERSION,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 1,
        retry_backoff_seconds: float = 1.5,
        max_output_tokens: int = 1_024,
        thinking_level: str | None = "low",
        temperature: float = 0.0,
        owns_client: bool = True,
        client_factory: Callable[..., object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = _secret_value(api_key)
        self._key_present = self._api_key is not None
        self.model = model
        self._api_version = api_version
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._max_output_tokens = max_output_tokens
        self._thinking_level = thinking_level
        self._temperature = temperature
        self._owns_client = owns_client
        self._client_factory = client_factory
        self._client: object | None = None
        self._sleep = sleep

    def available(self) -> bool:
        return self._key_present

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "gemini",
            "model": self.model,
            "digest": _stable_digest(self.model),
            "schema_version": ADJUDICATION_SCHEMA_VERSION,
            "api": "generate_content",
            "api_version": self._api_version,
            "thinking_level": self._thinking_level,
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }

    def adjudicate(
        self,
        requests: Sequence[AdjudicationRequest],
        *,
        audio_path: Path | None = None,
    ) -> dict[str, AdjudicationResult]:
        if not requests:
            return {}
        client = self._client_instance()
        if client is None:
            raise HostedProviderError(HostedErrorCategory.MISSING_KEY)

        schema = _adjudication_schema()
        payload: dict[str, object] = {
            "ambiguities": [_request_payload(request) for request in requests]
        }
        try:
            response = _run_with_retry(
                lambda: self._call_once(client, payload, schema, audio_path),
                attempts=self._retry_attempts,
                sleep=self._sleep,
                backoff_seconds=self._retry_backoff_seconds,
            )
        except HostedProviderError:
            raise
        except Exception as error:
            raise HostedProviderError(_classify_exception(error)) from None
        return self._parse(response, requests)

    def _call_once(
        self,
        client: object,
        payload: dict[str, object],
        schema: dict[str, object],
        audio_path: Path | None,
    ) -> object:
        from google.genai import types

        contents: list[object] = [json.dumps(payload, ensure_ascii=False)]
        if audio_path is not None and audio_path.is_file():
            contents.append(
                types.Part.from_bytes(data=audio_path.read_bytes(), mime_type="audio/wav")
            )
        config = types.GenerateContentConfig(
            system_instruction=_ADJUDICATION_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )
        if self._thinking_level is not None:
            config.thinking_config = types.ThinkingConfig(thinking_level=self._thinking_level)
        return client.models.generate_content(  # type: ignore[attr-defined]
            model=self.model,
            contents=contents,
            config=config,
        )

    def _parse(
        self, response: object, requests: Sequence[AdjudicationRequest]
    ) -> dict[str, AdjudicationResult]:
        data = _response_json(response)
        if not isinstance(data, dict):
            return {
                request.ambiguity_id: AdjudicationResult(
                    ambiguity_id=request.ambiguity_id,
                    selected_reading=None,
                    confidence=0.0,
                    reason="malformed_output",
                    rejected_reason="malformed_output",
                )
                for request in requests
            }
        entries = data.get("adjudications")
        if not isinstance(entries, list):
            entries = []
        by_id = {
            str(entry.get("ambiguity_id")): entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("ambiguity_id") is not None
        }
        results: dict[str, AdjudicationResult] = {}
        for request in requests:
            entry = by_id.get(request.ambiguity_id)
            if entry is None:
                results[request.ambiguity_id] = AdjudicationResult(
                    ambiguity_id=request.ambiguity_id,
                    selected_reading=None,
                    confidence=0.0,
                    reason="missing_result",
                    rejected_reason="missing_result",
                )
                continue
            selected = entry.get("selected_reading")
            confidence = entry.get("confidence")
            reason = entry.get("reason")
            if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
                confidence = 0.0
            confidence = min(max(float(confidence), 0.0), 1.0)
            if selected is not None and selected not in request.candidate_readings:
                # Never allow an arbitrary new sentence.
                results[request.ambiguity_id] = AdjudicationResult(
                    ambiguity_id=request.ambiguity_id,
                    selected_reading=None,
                    confidence=0.0,
                    reason="invalid_reading",
                    rejected_reason="invalid_reading",
                )
                continue
            results[request.ambiguity_id] = AdjudicationResult(
                ambiguity_id=request.ambiguity_id,
                selected_reading=selected if isinstance(selected, str) else None,
                confidence=confidence,
                reason=str(reason) if isinstance(reason, str) else "",
            )
        return results

    def release(self) -> None:
        self._release_client()


_ADJUDICATION_INSTRUCTION = (
    "You adjudicate a small number of already-supplied transcript reading "
    "candidates using the attached bounded audio clip. Choose exactly one of "
    "the supplied readings verbatim, or return null (UNRESOLVED) when the audio "
    "does not clearly support one reading. Never invent, translate, reformat, "
    "or rewrite text. Return only the requested JSON."
)


def _adjudication_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "adjudications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ambiguity_id": {"type": "string"},
                        "selected_reading": {"type": "string", "nullable": True},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["ambiguity_id", "confidence"],
                },
            }
        },
        "required": ["adjudications"],
    }


def _request_payload(request: AdjudicationRequest) -> dict[str, object]:
    return {
        "ambiguity_id": request.ambiguity_id,
        "context": request.context,
        "candidate_readings": list(request.candidate_readings),
        "evidence": [dict(item) for item in request.evidence_summary],
        "dialect_profile": request.dialect_profile,
        "meaning_critical": request.meaning_critical,
    }


def _response_json(response: object) -> object:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, dict):
            return parsed
        dump = getattr(parsed, "model_dump", None)
        if callable(dump):
            return dump()
    text = getattr(response, "text", None)
    if isinstance(text, str):
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return None
    return None
