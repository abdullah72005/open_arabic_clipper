"""Stage 4.0 hosted Gemini strategy-discovery adapter.

Two stable tiers are configured but only one is used per call, chosen by a pure
deterministic router before any network call: ``gemini-3.5-flash-lite`` for
routine bounded discovery and ``gemini-3.8-flash`` with low thinking for a
genuinely complex/high-value claim, debate, news, or transformation-required
case. The SDK client is constructed lazily and always closed; the API key is
never logged, serialized, or included in runtime identity.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.candidates.providers import ProviderErrorCategory
from app.transformation.policy import (
    GEMINI_API_VERSION,
    GEMINI_ROUTINE_MODEL,
    GEMINI_STRONG_MODEL,
    SCHEMA_VERSION,
)
from app.transformation.providers import (
    TRANSFORMATION_SYSTEM_INSTRUCTION,
    TransformationProviderError,
    TransformationStrategyRequest,
    parse_strategy_results,
    transformation_prompt_hash,
)
from app.transformation.types import TransformationProviderResult

_ROUTINE = "ROUTINE"
_STRONG = "STRONG"

_RETRYABLE = frozenset(
    {
        ProviderErrorCategory.CONNECTION,
        ProviderErrorCategory.TIMEOUT,
        ProviderErrorCategory.SERVICE_UNAVAILABLE,
        ProviderErrorCategory.PROVIDER_ERROR,
    }
)


class _GeminiStrategyEntry(BaseModel):
    strategy_type: str
    disposition: str = "RECOMMENDED"
    intensity: str = "MODERATE"
    direction_summary: str = ""
    added_value_focus: str = ""
    substantive_value_kind: str | None = None
    preservation_requirements: list[str] = Field(default_factory=list)
    retention_preservation: float | None = None
    source_moment_damage_risk: float | None = None
    added_value_density: float | None = None
    originality_potential: float | None = None
    source_dominance_risk: float | None = None
    generic_filler_risk: float | None = None
    redundant_commentary_risk: float | None = None
    template_staleness_risk: float | None = None
    external_verification_requirement: str = "NOT_REQUIRED"
    verification_requirements: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class _GeminiCandidateEntry(BaseModel):
    candidate_id: str
    confidence: float = 0.0
    notes: str = ""
    strategies: list[_GeminiStrategyEntry] = Field(default_factory=list)


class _GeminiTransformationOutput(BaseModel):
    candidates: list[_GeminiCandidateEntry] = Field(default_factory=list)


class GeminiTransformationProvider:
    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: object | None,
        routine_model: str = GEMINI_ROUTINE_MODEL,
        strong_model: str = GEMINI_STRONG_MODEL,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 1,
        retry_backoff_seconds: float = 1.5,
        max_output_tokens: int = 2_048,
        thinking_level: str | None = "low",
        temperature: float = 0.0,
        api_version: str = GEMINI_API_VERSION,
        owns_client: bool = True,
        client_factory: Callable[[], object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.routine_model = routine_model
        self.strong_model = strong_model
        self.model = routine_model
        self._timeout = max(0.1, float(timeout_seconds))
        self._retry_attempts = max(0, int(retry_attempts))
        self._retry_backoff = max(0.0, float(retry_backoff_seconds))
        self._output_tokens = max(1, int(max_output_tokens))
        self._thinking_level = thinking_level
        self._temperature = max(0.0, float(temperature))
        self._api_version = api_version or GEMINI_API_VERSION
        self._owns_client = owns_client
        self._sleep = sleep
        self._key_present = bool(api_key)
        self._api_key = api_key if self._key_present else None
        self._client_factory = client_factory
        self._client: object | None = None
        self._digest = hashlib.sha256(
            f"gemini-transformation:{routine_model}:{strong_model}".encode()
        ).hexdigest()
        self._selected_tier = _ROUTINE
        self.rate_limited = False
        self._usage: dict[str, int] = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "thoughts_token_count": 0,
        }

    def select_tier(self, requests: Sequence[TransformationStrategyRequest]) -> str:
        """Pure deterministic tier selection before any network call."""

        if any(request.complex_case or request.transformation_required for request in requests):
            return _STRONG
        return _ROUTINE

    def _build_client(self) -> object:
        return genai.Client(
            api_key=self._api_key or "",
            http_options=types.HttpOptions(
                api_version=self._api_version,
                timeout=int(self._timeout * 1000),
            ),
        )

    def _client_instance(self) -> object | None:
        if self._client is not None:
            return self._client
        if not self._key_present:
            return None
        try:
            client = self._client_factory() if self._client_factory else self._build_client()
        except Exception:
            raise TransformationProviderError(
                ProviderErrorCategory.PROVIDER_ERROR.value, "gemini client construction failed"
            ) from None
        self._client = client
        return client

    def release(self) -> None:
        client = self._client
        self._client = None
        self._api_key = None
        if client is None or not self._owns_client:
            return
        close = getattr(client, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:
            raise TransformationProviderError(
                ProviderErrorCategory.PROVIDER_ERROR.value, "gemini client close failed"
            ) from None

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "gemini",
            "routine_model": self.routine_model,
            "strong_model": self.strong_model,
            "digest": self._digest,
            "prompt_hash": transformation_prompt_hash(),
            "schema_version": SCHEMA_VERSION,
            "api_version": self._api_version,
            "temperature": self._temperature,
            "max_output_tokens": self._output_tokens,
            "thinking_level": self._thinking_level,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def usage_summary(self) -> dict[str, int]:
        return dict(self._usage)

    def discover(
        self, requests: Sequence[TransformationStrategyRequest]
    ) -> dict[str, TransformationProviderResult]:
        if not self._key_present:
            raise TransformationProviderError(ProviderErrorCategory.MISSING_KEY.value)
        if not requests:
            return {}
        self._selected_tier = self.select_tier(requests)
        self.model = self.strong_model if self._selected_tier == _STRONG else self.routine_model
        client = self._client_instance()
        if client is None:
            raise TransformationProviderError(ProviderErrorCategory.MISSING_KEY.value)
        attempts = 1 + self._retry_attempts
        last_error: TransformationProviderError | None = None
        for attempt in range(attempts):
            try:
                return self._call_once(client, requests)
            except TransformationProviderError as error:
                if error.category == ProviderErrorCategory.RATE_LIMITED.value:
                    self.rate_limited = True
                    raise
                if ProviderErrorCategory(error.category) in _RETRYABLE and attempt < attempts - 1:
                    last_error = error
                    self._sleep(self._retry_backoff)
                    continue
                raise
        raise last_error or TransformationProviderError(ProviderErrorCategory.PROVIDER_ERROR.value)

    def _call_once(
        self, client: object, requests: Sequence[TransformationStrategyRequest]
    ) -> dict[str, TransformationProviderResult]:
        payload = {"candidates": [request.to_payload() for request in requests]}
        config = types.GenerateContentConfig(
            system_instruction=TRANSFORMATION_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_GeminiTransformationOutput,
            max_output_tokens=self._output_tokens,
            temperature=self._temperature,
        )
        if self._thinking_level is not None:
            config.thinking_config = types.ThinkingConfig(thinking_level=self._thinking_level)
        try:
            response = client.models.generate_content(  # type: ignore[attr-defined]
                model=self.model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=config,
            )
        except Exception as error:
            raise TransformationProviderError(_classify_exception(error)) from None
        self._accumulate_usage(response)
        issue = _response_issue(response)
        if issue is not None:
            raise TransformationProviderError(issue)
        content = _response_content(response)
        if not isinstance(content, dict):
            raise TransformationProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        return parse_strategy_results(content, requests)

    def _accumulate_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        for key in (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "thoughts_token_count",
        ):
            value = getattr(usage, key, None)
            if isinstance(value, int):
                self._usage[key] += value


def _response_issue(response: Any) -> str | None:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None and getattr(prompt_feedback, "block_reason", None):
        return ProviderErrorCategory.SAFETY_REFUSAL.value
    candidates = getattr(response, "candidates", None) or ()
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        reason = getattr(finish_reason, "name", None) or finish_reason
        if reason in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
            return ProviderErrorCategory.SAFETY_REFUSAL.value
        if reason in {"MAX_TOKENS", "RECITATION"}:
            return ProviderErrorCategory.MALFORMED_OUTPUT.value
    else:
        return ProviderErrorCategory.MALFORMED_OUTPUT.value
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
            return None
    return None


def _classify_exception(error: Exception) -> str:
    if isinstance(error, TransformationProviderError):
        return error.category
    if isinstance(error, TimeoutError):
        return ProviderErrorCategory.TIMEOUT.value
    if isinstance(error, (ConnectionError, OSError)):
        return ProviderErrorCategory.CONNECTION.value
    code = _error_code(error)
    if code in {401, 403}:
        return ProviderErrorCategory.AUTHENTICATION.value
    if code == 404:
        return ProviderErrorCategory.MODEL_NOT_FOUND.value
    if code == 429:
        return ProviderErrorCategory.RATE_LIMITED.value
    if code in {408, 504} or "timeout" in str(error).casefold():
        return ProviderErrorCategory.TIMEOUT.value
    if code == 503:
        return ProviderErrorCategory.SERVICE_UNAVAILABLE.value
    if code is not None and 500 <= code < 600:
        return ProviderErrorCategory.PROVIDER_ERROR.value
    if code is not None and 400 <= code < 500:
        return ProviderErrorCategory.INVALID_REQUEST.value
    return ProviderErrorCategory.PROVIDER_ERROR.value


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
