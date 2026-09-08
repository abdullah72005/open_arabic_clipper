"""Hosted Gemini reconstruction provider using the official Google Gen AI SDK.

The provider adapts the shared one-pass reconstruction request to the Gemini
structured-output API, then runs the shared strict provider-boundary parser and
validation so a Gemini candidate is never trusted more than a local one. All
provider-specific behavior lives here: client construction, request/output
adaptation, token-usage extraction, sanitized error classification, and the
Gemini runtime identity.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, cast

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.transcription.reconstruction.confidence import CONFIDENCE_POLICY_VERSION
from app.transcription.reconstruction.providers import (
    SYSTEM_INSTRUCTION,
    ProviderResponseError,
    ReconstructionRequest,
    _extract_json_object,
    _parse_reconstructions,
)
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
    RequestSizeDiagnostics,
)
from app.transcription.reconstruction.validation import VALIDATION_VERSION

_GEMINI_SCHEMA_VERSION = "gemini-reconstruction-v1"


class _GeminiReconstructionEntry(BaseModel):
    segment_id: int
    corrected_text: str
    unchanged: bool = False
    confidence: float = Field(default=0.0)
    explanation: str = ""
    changes: list[str] = Field(default_factory=list)


class _GeminiReconstructionOutput(BaseModel):
    reconstructions: list[_GeminiReconstructionEntry]


_GEMINI_PROMPT_HASH = hashlib.sha256(
    (
        SYSTEM_INSTRUCTION
        + json.dumps(_GeminiReconstructionOutput.model_json_schema(), sort_keys=True)
    ).encode("utf-8")
).hexdigest()


class GeminiErrorCategory(str, Enum):
    """Sanitized provider failure categories; never carry request details."""

    MISSING_KEY = "MISSING_KEY"
    AUTHENTICATION = "AUTHENTICATION"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    CONNECTION = "CONNECTION"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    SAFETY_REFUSAL = "SAFETY_REFUSAL"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"


class GeminiProviderError(Exception):
    """Bounded, sanitized Gemini failure that never exposes the API key."""

    def __init__(self, category: GeminiErrorCategory, detail: str = "") -> None:
        super().__init__(f"gemini_{category.value}")
        self.category = category
        self.detail = detail


class GeminiReconstructionProvider:
    """Hosted structured-output provider sharing the strict local boundary."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 1,
        retry_backoff_seconds: float = 1.5,
        max_output_tokens: int = 1024,
        client_factory: Callable[[], object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider_name = "gemini"
        self.model = model
        self._timeout = max(0.1, float(timeout_seconds))
        self._retry_attempts = max(0, int(retry_attempts))
        self._retry_backoff = max(0.0, float(retry_backoff_seconds))
        self._output_tokens = max(1, int(max_output_tokens))
        self._sleep = sleep
        self._key_present = bool(api_key)
        self._client: object | None = None
        if self._key_present:
            self._client = (
                client_factory() if client_factory else self._build_client(cast(str, api_key))
            )
        self._digest: str | None = None
        self._last_request_sizes: tuple[RequestSizeDiagnostics, ...] = ()
        self._usage: dict[str, int] = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
        }

    def _build_client(self, api_key: str) -> object:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(self._timeout * 1000)),
        )

    def health(self) -> ProviderHealth:
        """Check key presence and model availability without generating content."""

        if not self._key_present:
            return ProviderHealth(
                ProviderAvailability.MISCONFIGURED,
                "gemini",
                self.model,
                None,
                "gemini api key is not configured",
            )
        if self._client is None:
            return ProviderHealth(
                ProviderAvailability.MISCONFIGURED,
                "gemini",
                self.model,
                None,
                "gemini client is unavailable",
            )
        try:
            model = self._client.models.get(model=self.model)  # type: ignore[attr-defined]
        except Exception as error:
            category = _classify_exception(error)
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE,
                "gemini",
                self.model,
                None,
                f"gemini_{category.value}",
            )
        self._digest = _model_digest(model)
        return ProviderHealth(
            ProviderAvailability.AVAILABLE,
            "gemini",
            self.model,
            self._digest,
            "gemini model available",
        )

    def release(self) -> None:
        """Hosted Gemini has no local model to unload."""

    def runtime_identity(self) -> dict[str, object]:
        """Return every output-affecting Gemini dependency as stable data."""

        return {
            "provider": "gemini",
            "model": self.model,
            "digest": self._digest or "digest_unavailable",
            "prompt_hash": _GEMINI_PROMPT_HASH,
            "schema_version": _GEMINI_SCHEMA_VERSION,
            "timeout_seconds": self._timeout,
            "retry_attempts": self._retry_attempts,
            "retry_backoff_seconds": self._retry_backoff,
            "max_output_tokens": self._output_tokens,
            "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
            "validation_version": VALIDATION_VERSION,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        """Resolve the live Gemini model digest now and return the refreshed identity."""

        if self._key_present and self._client is not None:
            try:
                model = self._client.models.get(model=self.model)  # type: ignore[attr-defined]
            except Exception:
                self._digest = None
            else:
                self._digest = _model_digest(model)
        return self.runtime_identity()

    def last_request_sizes(self) -> tuple[RequestSizeDiagnostics, ...]:
        return self._last_request_sizes

    def usage_summary(self) -> dict[str, int]:
        return dict(self._usage)

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        if not self._key_present or self._client is None:
            raise GeminiProviderError(
                GeminiErrorCategory.MISSING_KEY, "gemini api key is not configured"
            )
        if not requests:
            return {}
        self._last_request_sizes = tuple(
            RequestSizeDiagnostics(
                segment_index=request.segment_index,
                serialized_bytes=len(
                    json.dumps({"targets": [request.to_payload()]}, ensure_ascii=False).encode(
                        "utf-8"
                    )
                ),
                estimated_input_tokens=_estimate_envelope(request, self._output_tokens),
            )
            for request in requests
        )
        attempts = 1 + self._retry_attempts
        last_error: GeminiProviderError | None = None
        for attempt in range(attempts):
            try:
                return self._call_once(requests)
            except GeminiProviderError as error:
                if (
                    error.category
                    in {
                        GeminiErrorCategory.CONNECTION,
                        GeminiErrorCategory.TIMEOUT,
                        GeminiErrorCategory.PROVIDER_ERROR,
                    }
                    and attempt < attempts - 1
                ):
                    last_error = error
                    self._sleep(self._retry_backoff)
                    continue
                raise
        raise last_error or GeminiProviderError(
            GeminiErrorCategory.PROVIDER_ERROR, "gemini request failed"
        )

    def _call_once(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        payload = {"targets": [item.to_payload() for item in requests]}
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_GeminiReconstructionOutput,
            max_output_tokens=self._output_tokens,
        )
        assert self._client is not None
        client = self._client
        try:
            response = client.models.generate_content(  # type: ignore[attr-defined]
                model=self.model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=config,
            )
        except Exception as error:
            raise GeminiProviderError(_classify_exception(error)) from error
        self._accumulate_usage(response)
        issue = _response_issue(response)
        if issue is not None:
            raise GeminiProviderError(issue)
        content = _response_content(response)
        if not isinstance(content, dict):
            raise GeminiProviderError(GeminiErrorCategory.MALFORMED_OUTPUT)
        try:
            return _parse_reconstructions(content, requests)
        except ProviderResponseError as error:
            raise GeminiProviderError(
                GeminiErrorCategory.MALFORMED_OUTPUT, "gemini structured output failed validation"
            ) from error

    def _accumulate_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        for key in ("prompt_token_count", "candidates_token_count", "total_token_count"):
            value = getattr(usage, key, None)
            if isinstance(value, int):
                self._usage[key] += value


def _model_digest(model: object) -> str:
    name = getattr(model, "name", None) or getattr(model, "model", None) or ""
    version = getattr(model, "version", None) or ""
    return hashlib.sha256(f"{name}:{version}".encode("utf-8")).hexdigest()


def _response_issue(response: Any) -> GeminiErrorCategory | None:
    """Classify blocked, truncated, or empty responses before parsing."""

    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None and getattr(prompt_feedback, "block_reason", None):
        return GeminiErrorCategory.SAFETY_REFUSAL
    candidates = getattr(response, "candidates", None) or ()
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        reason = getattr(finish_reason, "name", None) or finish_reason
        if reason in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
            return GeminiErrorCategory.SAFETY_REFUSAL
        if reason in {"MAX_TOKENS", "RECITATION"}:
            return GeminiErrorCategory.MALFORMED_OUTPUT
    else:
        return GeminiErrorCategory.MALFORMED_OUTPUT
    return None


def _response_content(response: Any) -> object:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, BaseModel):
        return parsed.model_dump()
    if isinstance(parsed, dict):
        return parsed
    text = getattr(response, "text", None)
    if isinstance(text, str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return _extract_json_object(text)
            except ProviderResponseError:
                raise GeminiProviderError(
                    GeminiErrorCategory.MALFORMED_OUTPUT, "gemini returned invalid JSON"
                ) from None
    raise GeminiProviderError(GeminiErrorCategory.MALFORMED_OUTPUT)


def _estimate_envelope(request: ReconstructionRequest, output_tokens: int) -> int:
    return request.estimated_tokens(
        system_instruction=SYSTEM_INSTRUCTION,
        output_tokens=output_tokens,
        chat_framing_reserve=64,
        safety_reserve=128,
    )


def _classify_exception(error: Exception) -> GeminiErrorCategory:
    if isinstance(error, GeminiProviderError):
        return error.category
    if isinstance(error, TimeoutError):
        return GeminiErrorCategory.TIMEOUT
    if isinstance(error, (ConnectionError, OSError)):
        return GeminiErrorCategory.CONNECTION
    code = _error_code(error)
    if code in {401, 403}:
        return GeminiErrorCategory.AUTHENTICATION
    if code == 404:
        return GeminiErrorCategory.MODEL_NOT_FOUND
    if code == 429:
        return GeminiErrorCategory.RATE_LIMITED
    if code in {408, 504} or "timeout" in str(error).casefold():
        return GeminiErrorCategory.TIMEOUT
    return GeminiErrorCategory.PROVIDER_ERROR


def _error_code(error: Exception) -> int | None:
    code = getattr(error, "code", None)
    if isinstance(code, int):
        return code
    response = getattr(error, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        status = getattr(response, "status", None)
        if isinstance(status, int):
            return status
    return None
