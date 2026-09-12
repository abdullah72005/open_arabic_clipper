"""Stage 3 hosted Gemini semantic-evaluation adapter.

Reuses the established Gemini conventions: configuration-level availability, a
lazily constructed SDK client, structured Pydantic output, deterministic
generation, sanitized error categories, at most one bounded retry for transient
failures, usage capture, and resource cleanup on every exit path. The API key is
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

from app.candidates.policy import SEMANTIC_SCHEMA_VERSION
from app.candidates.providers import (
    SEMANTIC_SYSTEM_INSTRUCTION,
    ProviderErrorCategory,
    SemanticEvaluationRequest,
    SemanticEvaluationResult,
    SemanticProviderError,
    parse_semantic_entries,
    semantic_prompt_hash,
)
from app.core.enums import ContentType

_GEMINI_API_VERSION = "v1"


class _GeminiHook(BaseModel):
    type: str
    text: str = ""
    source_evidence: list[str] = Field(default_factory=list)
    source_segment_indexes: list[int] = Field(default_factory=list)
    strength: float = Field(default=0.0)
    faithfulness: float = Field(default=0.0)
    naturalness: float = Field(default=0.0)
    audience_suitability: float = Field(default=0.0)
    policy_risk: float = Field(default=0.0)
    missing_context_dependency: float = Field(default=0.0)


class _GeminiSemanticEntry(BaseModel):
    candidate_id: str
    primary_content_type: str = ContentType.OTHER.value
    secondary_content_types: list[str] = Field(default_factory=list)
    score_adjustments: dict[str, float] = Field(default_factory=dict)
    idea_summary: str = ""
    topic_summary: str = ""
    hooks: list[_GeminiHook] = Field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""


class _GeminiSemanticOutput(BaseModel):
    evaluations: list[_GeminiSemanticEntry]


_RETRYABLE = frozenset(
    {
        ProviderErrorCategory.CONNECTION,
        ProviderErrorCategory.TIMEOUT,
        ProviderErrorCategory.SERVICE_UNAVAILABLE,
        ProviderErrorCategory.PROVIDER_ERROR,
    }
)


class GeminiSemanticProvider:
    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 1,
        retry_backoff_seconds: float = 1.5,
        max_output_tokens: int = 1024,
        thinking_level: str | None = None,
        temperature: float = 0.0,
        api_version: str = _GEMINI_API_VERSION,
        owns_client: bool = True,
        client_factory: Callable[[], object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._timeout = max(0.1, float(timeout_seconds))
        self._retry_attempts = max(0, int(retry_attempts))
        self._retry_backoff = max(0.0, float(retry_backoff_seconds))
        self._output_tokens = max(1, int(max_output_tokens))
        self._thinking_level = thinking_level
        self._temperature = max(0.0, float(temperature))
        self._api_version = api_version or _GEMINI_API_VERSION
        self._owns_client = owns_client
        self._sleep = sleep
        self._key_present = bool(api_key)
        self._api_key: str | None = api_key if self._key_present else None
        self._client_factory = client_factory
        self._client: object | None = None
        self._digest = hashlib.sha256(f"gemini-semantic:{model}".encode()).hexdigest()
        self.rate_limited = False
        self._usage: dict[str, int] = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "thoughts_token_count": 0,
        }

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
            raise SemanticProviderError(
                ProviderErrorCategory.PROVIDER_ERROR, "gemini client construction failed"
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
            raise SemanticProviderError(
                ProviderErrorCategory.PROVIDER_ERROR, "gemini client close failed"
            ) from None

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "gemini",
            "model": self.model,
            "digest": self._digest,
            "prompt_hash": semantic_prompt_hash(),
            "schema_version": SEMANTIC_SCHEMA_VERSION,
            "api_version": self._api_version,
            "temperature": self._temperature,
            "max_output_tokens": self._output_tokens,
            "thinking_level": self._thinking_level,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def usage_summary(self) -> dict[str, int]:
        return dict(self._usage)

    def evaluate(
        self, requests: Sequence[SemanticEvaluationRequest]
    ) -> dict[str, SemanticEvaluationResult]:
        if not self._key_present:
            raise SemanticProviderError(ProviderErrorCategory.MISSING_KEY)
        if not requests:
            return {}
        client = self._client_instance()
        if client is None:
            raise SemanticProviderError(ProviderErrorCategory.MISSING_KEY)
        attempts = 1 + self._retry_attempts
        last_error: SemanticProviderError | None = None
        for attempt in range(attempts):
            try:
                return self._call_once(client, requests)
            except SemanticProviderError as error:
                if error.category is ProviderErrorCategory.RATE_LIMITED:
                    self.rate_limited = True
                    raise
                if error.category in _RETRYABLE and attempt < attempts - 1:
                    last_error = error
                    self._sleep(self._retry_backoff)
                    continue
                raise
        raise last_error or SemanticProviderError(ProviderErrorCategory.PROVIDER_ERROR)

    def _call_once(
        self, client: object, requests: Sequence[SemanticEvaluationRequest]
    ) -> dict[str, SemanticEvaluationResult]:
        payload = {"candidates": [request.to_payload() for request in requests]}
        config = types.GenerateContentConfig(
            system_instruction=SEMANTIC_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_GeminiSemanticOutput,
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
            raise SemanticProviderError(_classify_exception(error)) from None
        self._accumulate_usage(response)
        issue = _response_issue(response)
        if issue is not None:
            raise SemanticProviderError(issue)
        content = _response_content(response)
        if not isinstance(content, dict):
            raise SemanticProviderError(ProviderErrorCategory.MALFORMED_OUTPUT)
        return parse_semantic_entries(content, requests)

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


def _response_issue(response: Any) -> ProviderErrorCategory | None:
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None and getattr(prompt_feedback, "block_reason", None):
        return ProviderErrorCategory.SAFETY_REFUSAL
    candidates = getattr(response, "candidates", None) or ()
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        reason = getattr(finish_reason, "name", None) or finish_reason
        if reason in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
            return ProviderErrorCategory.SAFETY_REFUSAL
        if reason in {"MAX_TOKENS", "RECITATION"}:
            return ProviderErrorCategory.MALFORMED_OUTPUT
    else:
        return ProviderErrorCategory.MALFORMED_OUTPUT
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


def _classify_exception(error: Exception) -> ProviderErrorCategory:
    if isinstance(error, SemanticProviderError):
        category = error.category
        try:
            return ProviderErrorCategory(category)
        except ValueError:
            return ProviderErrorCategory.PROVIDER_ERROR
    if isinstance(error, TimeoutError):
        return ProviderErrorCategory.TIMEOUT
    if isinstance(error, (ConnectionError, OSError)):
        return ProviderErrorCategory.CONNECTION
    code = _error_code(error)
    if code in {401, 403}:
        return ProviderErrorCategory.AUTHENTICATION
    if code == 404:
        return ProviderErrorCategory.MODEL_NOT_FOUND
    if code == 429:
        return ProviderErrorCategory.RATE_LIMITED
    if code in {408, 504} or "timeout" in str(error).casefold():
        return ProviderErrorCategory.TIMEOUT
    if code == 503:
        return ProviderErrorCategory.SERVICE_UNAVAILABLE
    if code is not None and 500 <= code < 600:
        return ProviderErrorCategory.PROVIDER_ERROR
    if code is not None and 400 <= code < 500:
        return ProviderErrorCategory.INVALID_REQUEST
    return ProviderErrorCategory.PROVIDER_ERROR


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
