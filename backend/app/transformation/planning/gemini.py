"""Stage 4.1 hosted Gemini planning adapter.

Two stable tiers are configured and the deterministic router chooses one before
any network call. The SDK client is constructed lazily, closed on every exit
path, and the API key is never logged, serialized, or included in runtime
identity. The adapter executes exactly one HTTP call per invocation (the service
batches strategies per tier, at most once per tier and twice per plan set).
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
from app.transformation.planning.policy import (
    GEMINI_API_VERSION,
    GEMINI_ROUTINE_MODEL,
    GEMINI_STRONG_MODEL,
    SCHEMA_VERSION,
)
from app.transformation.planning.providers import (
    PLANNING_SYSTEM_INSTRUCTION,
    PlanningProviderError,
    PlanningRequest,
    parse_plan_results,
    planning_prompt_hash,
)
from app.transformation.planning.types import PlanProviderResult

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


class _GeminiNarration(BaseModel):
    need: str = "NONE"
    purposes: list[str] = Field(default_factory=list)
    language: str | None = None
    register_intent: str | None = None
    estimated_duration: float = 0.0
    placement_block_index: int | None = None
    max_source_interruption_seconds: float = 0.0
    overlaps_source_audio: bool = False
    replaces_silence: bool = False
    essential: bool = False
    verification_dependency_ids: list[str] = Field(default_factory=list)


class _GeminiBlock(BaseModel):
    block_type: str
    purpose: str = ""
    estimated_duration: float = 0.0
    interrupts_source: bool = False
    preservation_constraints: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)
    source_role: str | None = None
    word_start_index: int | None = None
    word_end_index: int | None = None
    use_full_window: bool = False
    continuity_rationale: str | None = None
    substantive_value_kind: str | None = None
    semantic_intent: str = ""
    why_unavailable: str = ""
    grounding_refs: list[str] = Field(default_factory=list)
    delivery_intent: str | None = None
    draft_line: str | None = None
    claim_dependency: str | None = None
    verification_rationale: str | None = None
    intended_use: str | None = None
    must_verify_before_execution: bool = False
    dependent_block_ids: list[str] = Field(default_factory=list)


class _GeminiPlan(BaseModel):
    strategy_id: str
    strategy_key: str
    confidence: float = 0.0
    no_valid_plan: bool = False
    no_valid_reason: str = ""
    planner_notes: str = ""
    preservation_constraints: list[str] = Field(default_factory=list)
    blocks: list[_GeminiBlock] = Field(default_factory=list)
    narration: _GeminiNarration = Field(default_factory=_GeminiNarration)


class _GeminiPlanningOutput(BaseModel):
    plans: list[_GeminiPlan] = Field(default_factory=list)


class GeminiPlanningProvider:
    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: object | None,
        routine_model: str = GEMINI_ROUTINE_MODEL,
        strong_model: str = GEMINI_STRONG_MODEL,
        timeout_seconds: float = 45.0,
        retry_attempts: int = 1,
        retry_backoff_seconds: float = 1.5,
        max_output_tokens: int = 4_096,
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
            f"gemini-transformation-planning:{routine_model}:{strong_model}".encode()
        ).hexdigest()
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
            raise PlanningProviderError(
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
            raise PlanningProviderError(
                ProviderErrorCategory.PROVIDER_ERROR.value, "gemini client close failed"
            ) from None

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "gemini",
            "routine_model": self.routine_model,
            "strong_model": self.strong_model,
            "digest": self._digest,
            "prompt_hash": planning_prompt_hash(),
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

    def plan(
        self, requests: Sequence[PlanningRequest], tier: str = _ROUTINE
    ) -> dict[str, PlanProviderResult]:
        if not self._key_present:
            raise PlanningProviderError(ProviderErrorCategory.MISSING_KEY.value)
        if not requests:
            return {}
        self.model = self.strong_model if tier == _STRONG else self.routine_model
        thinking: str | None = self._thinking_level if tier == _STRONG else None
        client = self._client_instance()
        if client is None:
            raise PlanningProviderError(ProviderErrorCategory.MISSING_KEY.value)
        attempts = 1 + self._retry_attempts
        last_error: PlanningProviderError | None = None
        for attempt in range(attempts):
            try:
                return self._call_once(client, requests, thinking)
            except PlanningProviderError as error:
                if error.category == ProviderErrorCategory.RATE_LIMITED.value:
                    self.rate_limited = True
                    raise
                if ProviderErrorCategory(error.category) in _RETRYABLE and attempt < attempts - 1:
                    last_error = error
                    self._sleep(self._retry_backoff)
                    continue
                raise
        raise last_error or PlanningProviderError(ProviderErrorCategory.PROVIDER_ERROR.value)

    def _call_once(
        self,
        client: object,
        requests: Sequence[PlanningRequest],
        thinking: str | None,
    ) -> dict[str, PlanProviderResult]:
        payload = {"plans_requested": [request.to_payload() for request in requests]}
        config = types.GenerateContentConfig(
            system_instruction=PLANNING_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_GeminiPlanningOutput,
            max_output_tokens=self._output_tokens,
            temperature=self._temperature,
        )
        if thinking is not None:
            config.thinking_config = types.ThinkingConfig(thinking_level=thinking)
        try:
            response = client.models.generate_content(  # type: ignore[attr-defined]
                model=self.model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=config,
            )
        except Exception as error:
            raise PlanningProviderError(_classify_exception(error)) from None
        self._accumulate_usage(response)
        issue = _response_issue(response)
        if issue is not None:
            raise PlanningProviderError(issue)
        content = _response_content(response)
        if not isinstance(content, dict):
            raise PlanningProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        return parse_plan_results(content, requests)

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
    if isinstance(error, PlanningProviderError):
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
