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
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.candidates.providers import ProviderErrorCategory
from app.core.enums import TransformationStrategyType
from app.transformation.planning.policy import (
    GEMINI_API_VERSION,
    GEMINI_ROUTINE_MODEL,
    GEMINI_STRONG_MODEL,
    SCHEMA_VERSION,
    strategy_value_kinds,
)
from app.transformation.planning.providers import (
    PLANNING_SCHEMA_BINDING_VERSION,
    PLANNING_SYSTEM_INSTRUCTION,
    PlanningBlockSchema,
    PlanningNarrationSchema,
    PlanningPlanSchema,
    PlanningProviderError,
    PlanningRequest,
    build_planning_output_schema,
    parse_plan_results,
    planning_output_schema_digest,
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

# The authoritative base schema lives in ``providers``. Each call derives a
# strategy-bound variant so the object Google receives, the prompt hash family,
# and the offline schema tests share one contract.
_GeminiNarration = PlanningNarrationSchema
_GeminiBlock = PlanningBlockSchema
_GeminiPlan = PlanningPlanSchema


class GeminiPlanningProvider:
    provider_name = "gemini"
    hosted_provider = True

    def __init__(
        self,
        *,
        api_key: object | None,
        routine_model: str = GEMINI_ROUTINE_MODEL,
        strong_model: str = GEMINI_STRONG_MODEL,
        timeout_seconds: float = 45.0,
        retry_attempts: int = 0,
        retry_backoff_seconds: float = 1.5,
        max_output_tokens: int = 4_096,
        thinking_level: str | None = "low",
        temperature: float = 0.0,
        api_version: str = GEMINI_API_VERSION,
        max_raw_calls: int = 2,
        owns_client: bool = True,
        client_factory: Callable[[], object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.routine_model = routine_model
        self.strong_model = strong_model
        self.model = routine_model
        self._timeout = max(0.1, float(timeout_seconds))
        # Stage 4.1 keeps no per-tier retry inside its two-call ceiling.
        self._retry_attempts = max(0, int(retry_attempts))
        self._retry_backoff = max(0.0, float(retry_backoff_seconds))
        self._max_raw_calls = max(1, int(max_raw_calls))
        self._raw_calls = 0
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
            "schema_binding_version": PLANNING_SCHEMA_BINDING_VERSION,
            "api_version": self._api_version,
            "temperature": self._temperature,
            "max_output_tokens": self._output_tokens,
            "thinking_level": self._thinking_level,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def usage_summary(self) -> dict[str, int]:
        return dict(self._usage)

    def raw_call_count(self) -> int:
        """Actual raw hosted generate_content calls made by this provider."""

        return self._raw_calls

    def plan(
        self, requests: Sequence[PlanningRequest], tier: str = _ROUTINE
    ) -> dict[str, PlanProviderResult]:
        if not self._key_present:
            raise PlanningProviderError(ProviderErrorCategory.MISSING_KEY.value)
        if not requests:
            return {}
        if self._raw_calls >= self._max_raw_calls:
            raise PlanningProviderError(
                ProviderErrorCategory.PROVIDER_ERROR.value,
                "hosted planning raw-call budget exhausted",
            )
        self.model = self.strong_model if tier == _STRONG else self.routine_model
        thinking: str | None = self._thinking_level if tier == _STRONG else None
        client = self._client_instance()
        if client is None:
            raise PlanningProviderError(ProviderErrorCategory.MISSING_KEY.value)
        # Exactly one raw call per tier. Stage 4.1 deliberately performs no
        # per-tier retry so a retryable failure cannot silently exceed the
        # hard two-call hosted budget for one plan-set run.
        try:
            return self._call_once(client, requests, thinking)
        except PlanningProviderError as error:
            if error.category == ProviderErrorCategory.RATE_LIMITED.value:
                self.rate_limited = True
            raise

    def _call_once(
        self,
        client: object,
        requests: Sequence[PlanningRequest],
        thinking: str | None,
    ) -> dict[str, PlanProviderResult]:
        payload = {"plans_requested": [request.to_payload() for request in requests]}
        schema = build_planning_output_schema(requests)
        config = types.GenerateContentConfig(
            system_instruction=PLANNING_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=self._output_tokens,
            temperature=self._temperature,
        )
        if thinking is not None:
            config.thinking_config = types.ThinkingConfig(thinking_level=thinking)
        try:
            self._raw_calls += 1
            response = client.models.generate_content(  # type: ignore[attr-defined]
                model=self.model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=config,
            )
        except ValidationError as error:
            raise PlanningProviderError(
                ProviderErrorCategory.MALFORMED_OUTPUT.value,
                "response_schema_validation_failed",
                rejections=_schema_rejection_details(error, None, requests),
            ) from None
        except Exception as error:
            raise PlanningProviderError(_classify_exception(error)) from None
        self._accumulate_usage(response)
        issue = _response_issue(response)
        if issue is not None:
            raise PlanningProviderError(issue)
        content = _response_content(response)
        if not isinstance(content, dict):
            raise PlanningProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        evidence = _response_evidence(content)
        evidence["output_schema_digest"] = planning_output_schema_digest(requests)
        return parse_plan_results(
            content,
            requests,
            response_evidence=evidence,
            schema_rejections=_sdk_schema_rejections(response, schema, content, requests),
        )

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


def _response_evidence(content: Mapping[str, object]) -> dict[str, object]:
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    plans = content.get("plans")
    return {
        "content_digest": hashlib.sha256(encoded).hexdigest(),
        "content_bytes": len(encoded),
        "truncated": False,
        "returned_plan_count": len(plans) if isinstance(plans, list) else 0,
    }


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


def _bounded_rejected_value(value: object) -> object:
    """Bounded representation of one rejected value; never a full response."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:80]
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return {
        "type": type(value).__name__,
        "digest": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "characters": len(encoded),
    }


def _schema_rejection_details(
    error: ValidationError,
    content: Mapping[str, object] | None,
    requests: Sequence[PlanningRequest],
) -> tuple[dict[str, object], ...]:
    """Turn an SDK Pydantic rejection into bounded, non-secret diagnostics."""

    by_key = {request.strategy_key: request for request in requests}
    plans = content.get("plans") if isinstance(content, Mapping) else None
    details: list[dict[str, object]] = []
    for item in error.errors()[:8]:
        loc = list(item.get("loc") or ())
        plan_index: int | None = None
        if len(loc) >= 2 and loc[0] == "plans" and isinstance(loc[1], int):
            plan_index = int(loc[1])
        entry: Mapping[str, object] | None = None
        if isinstance(plans, list) and plan_index is not None and plan_index < len(plans):
            candidate = plans[plan_index]
            if isinstance(candidate, Mapping):
                entry = candidate
        strategy_key = entry.get("strategy_key") if entry is not None else None
        strategy_id = entry.get("strategy_id") if entry is not None else None
        request: PlanningRequest | None = None
        if isinstance(strategy_key, str):
            request = by_key.get(strategy_key)
        if request is None and len(requests) == 1:
            request = requests[0]
        if strategy_key is None and request is not None:
            strategy_key = request.strategy_key
        if strategy_id is None and request is not None:
            strategy_id = request.strategy_id
        block_index: int | None = None
        for position, part in enumerate(loc):
            if part == "blocks" and position + 1 < len(loc) and isinstance(loc[position + 1], int):
                block_index = int(loc[position + 1])
                break
        field = next((part for part in reversed(loc) if isinstance(part, str)), None)
        allowed: list[str] | None = None
        if field == "substantive_value_kind" and request is not None:
            try:
                strategy_type = TransformationStrategyType(str(request.strategy_type))
                allowed = [item.value for item in strategy_value_kinds(strategy_type)]
            except ValueError:
                allowed = []
        details.append(
            {
                "origin": "SDK_PARSING",
                "code": (
                    "SCHEMA_VALUE_NOT_ALLOWED" if allowed is not None else "SCHEMA_VALIDATION_ERROR"
                ),
                "strategy_id": str(strategy_id)[:200] if strategy_id is not None else None,
                "strategy_key": str(strategy_key)[:200] if strategy_key is not None else None,
                "plan_index": plan_index,
                "block_index": block_index,
                "field": str(field)[:80] if field is not None else None,
                "actual_value": _bounded_rejected_value(item.get("input")),
                "allowed_values": allowed,
                "location": [part if isinstance(part, int) else str(part)[:80] for part in loc][
                    :12
                ],
            }
        )
    return tuple(details)


def _sdk_schema_rejections(
    response: Any,
    schema: type[BaseModel],
    content: Mapping[str, object],
    requests: Sequence[PlanningRequest],
) -> tuple[dict[str, object], ...]:
    """Detect a silent SDK schema parse fallback and retain bounded evidence.

    The google-genai SDK currently swallows Pydantic validation errors and
    leaves ``response.parsed`` unset. When raw JSON content is still available,
    validate it against the exact batch-bound schema so the failure is recorded
    as SDK parsing evidence rather than being mistaken for a clean response.
    """

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, (BaseModel, dict)):
        return ()
    try:
        schema.model_validate(content)
    except ValidationError as error:
        return _schema_rejection_details(error, content, requests)
    except Exception:
        return (
            {
                "origin": "SDK_PARSING",
                "code": "SCHEMA_VALIDATION_FAILED",
                "actual_value": None,
                "allowed_values": None,
            },
        )
    return ()


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
