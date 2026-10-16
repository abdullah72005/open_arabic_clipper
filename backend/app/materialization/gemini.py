"""Stage 6 hosted Gemini writing and categorical-critique adapters.

Both adapters execute exactly one raw ``generateContent`` request per invocation
(the durable gateway reserves the attempt before the call and the caller batches
all targets into that single request). The SDK client is constructed lazily,
closed on every exit path, and the API key is never logged, serialized, or
included in runtime identity. Automatic SDK retries are disabled so a hosted
failure can never silently exceed the generation's raw-call ceiling.

A hosted failure (429/quota, refusal, malformed/incomplete output, timeout, or
connection loss) is returned as a truthful *deferred* result, never raised into
the pipeline: the source defers honestly instead of failing. The deterministic
validation and authoritative evidence retain control of acceptance.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.candidates.providers import ProviderErrorCategory
from app.materialization.policy import SEMANTIC_STRONG_TIER
from app.materialization.providers import (
    CRITIQUE_DIMENSIONS,
    CRITIQUE_LEVELS,
    BlockCritique,
    CritiqueDimensionFinding,
    CritiqueRequest,
    CritiqueResult,
    MaterializationProviderError,
    WritingRequest,
    WritingResult,
)

GEMINI_API_VERSION = "v1"
GEMINI_ROUTINE_MODEL = "gemini-3.5-flash-lite"
GEMINI_STRONG_MODEL = "gemini-3.8-flash"
#: Bumped whenever the prompt text or structured-output schema changes, so a
#: stale accepted checkpoint is never reused against a different instruction.
GEMINI_SCHEMA_VERSION = "stage6-gemini-schema-v1"

WRITING_SYSTEM_INSTRUCTION = (
    "You are a precise bilingual short-form video writer. For each target you "
    "receive bounded ACTUAL resolved source evidence, an optional source "
    "language/dialect (descriptive evidence about the material, not a target "
    "audience), and a semantic intent. Write only the final on-screen/narration "
    "prose that delivers the intent. Rules: (1) ground every statement in the "
    "provided evidence; never invent numbers, names, facts, quotes, or source "
    "timestamps. (2) produce genuinely substantive value, not a paraphrase, "
    "restatement, filler, or presentation-only rewording of the source. (3) "
    "preserve the source's meaning, negation scope, quantities and their "
    "associations, and any Latin/code-switch token or proper name verbatim. (4) "
    "write natural, conversational language compatible with the requested "
    "register and target context; do not force Modern Standard Arabic, a "
    "particular dialect, or an invented dialect. (5) be concise: one short "
    "sentence per target. Output only the requested structured JSON; you never "
    "choose source spans, strategy, rights, renderer settings, or whether a "
    "deterministic failure may be ignored."
)

CRITIQUE_SYSTEM_INSTRUCTION = (
    "You are a strict bilingual semantic reviewer. For each block you receive the "
    "authored copy, its intended value, an optional source language/dialect, and "
    "the bounded ACTUAL source evidence it claims to be grounded in. Return "
    "exactly one categorical finding for each dimension: GROUNDING_ENTAILMENT, "
    "SUBSTANTIVE_ADDED_VALUE, SEMANTIC_FIDELITY, UNSUPPORTED_FACTS, "
    "CODE_SWITCH_PRESERVATION, REDUNDANCY, DIALECT_REGISTER_FIT, NATURALNESS, "
    "LENGTH_NECESSITY. Each level is PASS, CONCERN, FAIL, or UNKNOWN with short "
    "evidence. Grounding, substantive added value, semantic fidelity, unsupported "
    "facts, and code-switch preservation must each be a known PASS to be approved; "
    "a missing or UNKNOWN critical dimension cannot be approved. Do not infer "
    "knowledge from outside the provided evidence: if the evidence is insufficient "
    "to judge a dimension, answer UNKNOWN rather than PASS."
)


class _CopyOutput(BaseModel):
    block_index: int
    text: str = ""


class _WritingOutput(BaseModel):
    copies: list[_CopyOutput] = Field(default_factory=list)


_DimensionLiteral = Literal[
    "GROUNDING_ENTAILMENT",
    "SUBSTANTIVE_ADDED_VALUE",
    "SEMANTIC_FIDELITY",
    "UNSUPPORTED_FACTS",
    "CODE_SWITCH_PRESERVATION",
    "REDUNDANCY",
    "DIALECT_REGISTER_FIT",
    "NATURALNESS",
    "LENGTH_NECESSITY",
]
_LevelLiteral = Literal["PASS", "CONCERN", "FAIL", "UNKNOWN"]


class _CritiqueFindingOutput(BaseModel):
    dimension: _DimensionLiteral
    level: _LevelLiteral
    evidence: str = ""


class _BlockCritiqueOutput(BaseModel):
    block_index: int
    findings: list[_CritiqueFindingOutput] = Field(default_factory=list)


class _CritiqueOutput(BaseModel):
    blocks: list[_BlockCritiqueOutput] = Field(default_factory=list)


# The declared enum vocabularies must equal the shared closed sets; a drifting
# Literal would let the model emit a dimension/level the parser silently drops.
assert tuple(_DimensionLiteral.__args__) == tuple(CRITIQUE_DIMENSIONS)
assert tuple(_LevelLiteral.__args__) == tuple(CRITIQUE_LEVELS)


_RETRYABLE = frozenset(
    {
        ProviderErrorCategory.CONNECTION.value,
        ProviderErrorCategory.TIMEOUT.value,
        ProviderErrorCategory.SERVICE_UNAVAILABLE.value,
        ProviderErrorCategory.PROVIDER_ERROR.value,
        ProviderErrorCategory.RATE_LIMITED.value,
    }
)


class _GeminiChatAdapter:
    """Shared lazy-client, identity, single-raw-call, and error-mapping boundary."""

    provider_name = "gemini"
    hosted_provider = True
    _system_instruction = ""
    _kind = "materialization"

    def __init__(
        self,
        *,
        api_key: object | None,
        routine_model: str = GEMINI_ROUTINE_MODEL,
        strong_model: str = GEMINI_STRONG_MODEL,
        timeout_seconds: float = 45.0,
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
        self._raw_calls = 0
        self._digest = hashlib.sha256(
            f"gemini-{self._kind}:{routine_model}:{strong_model}".encode()
        ).hexdigest()
        self._prompt_sha256 = hashlib.sha256(
            self._system_instruction.encode("utf-8")
        ).hexdigest()
        self.rate_limited = False
        self._usage: dict[str, int] = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "thoughts_token_count": 0,
        }

    def available(self) -> bool:
        return self._key_present

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "gemini",
            "kind": self._kind,
            "routine_model": self.routine_model,
            "strong_model": self.strong_model,
            "digest": self._digest,
            "api_version": self._api_version,
            "temperature": self._temperature,
            "max_output_tokens": self._output_tokens,
            "thinking_level": self._thinking_level,
            "prompt_sha256": self._prompt_sha256,
            "schema_version": GEMINI_SCHEMA_VERSION,
        }

    def usage_summary(self) -> dict[str, int]:
        return dict(self._usage)

    def raw_call_count(self) -> int:
        return self._raw_calls

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
        except Exception:
            raise MaterializationProviderError(
                ProviderErrorCategory.PROVIDER_ERROR.value, "gemini client construction failed"
            ) from None
        self._client = client
        return client

    def _generate_once(
        self,
        *,
        tier: str,
        schema: type[BaseModel],
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Execute exactly one raw generateContent call; return the parsed JSON dict.

        Raises ``MaterializationProviderError`` for any provider failure; callers
        translate that into a truthful deferred result.
        """

        if not self._key_present:
            raise MaterializationProviderError(ProviderErrorCategory.MISSING_KEY.value)
        client = self._client_instance()
        if client is None:
            raise MaterializationProviderError(ProviderErrorCategory.MISSING_KEY.value)
        self.model = self.strong_model if tier == SEMANTIC_STRONG_TIER else self.routine_model
        thinking = self._thinking_level if tier == SEMANTIC_STRONG_TIER else None
        config = types.GenerateContentConfig(
            system_instruction=self._system_instruction,
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
        except Exception as error:
            category = _classify_exception(error)
            if category == ProviderErrorCategory.RATE_LIMITED.value:
                self.rate_limited = True
            raise MaterializationProviderError(category) from None
        self._accumulate_usage(response)
        issue = _response_issue(response)
        if issue is not None:
            if issue == ProviderErrorCategory.RATE_LIMITED.value:
                self.rate_limited = True
            raise MaterializationProviderError(issue)
        content = _response_content(response)
        if not isinstance(content, dict):
            raise MaterializationProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        return content

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


class GeminiWritingProvider(_GeminiChatAdapter):
    """Single-request hosted writing adapter implementing the writing protocol."""

    _kind = "writing"
    _system_instruction = WRITING_SYSTEM_INSTRUCTION

    def write(self, request: WritingRequest) -> WritingResult:
        if not request.targets:
            return WritingResult(copies={}, provider_identity=self.runtime_identity())
        try:
            content = self._generate_once(
                tier=request.tier,
                schema=_WritingOutput,
                payload=request.as_dict(),
            )
        except MaterializationProviderError as error:
            return WritingResult(
                copies={},
                provider_identity=self.runtime_identity(),
                deferred=True,
                reason_code=error.reason_code,
            )
        copies = _parse_copies(content, [target.block_index for target in request.targets])
        if copies is None:
            return WritingResult(
                copies={},
                provider_identity=self.runtime_identity(),
                deferred=True,
                reason_code=ProviderErrorCategory.MALFORMED_OUTPUT.value,
            )
        return WritingResult(
            copies=copies,
            provider_identity=self.runtime_identity(),
            notes="gemini",
        )


class GeminiCritiqueProvider(_GeminiChatAdapter):
    """Single-request categorical-critique adapter implementing the critique protocol."""

    _kind = "critique"
    _system_instruction = CRITIQUE_SYSTEM_INSTRUCTION

    def critique(self, request: CritiqueRequest) -> CritiqueResult:
        if not request.targets:
            return CritiqueResult(
                accepted={}, blocks={}, provider_identity=self.runtime_identity()
            )
        try:
            content = self._generate_once(
                tier=request.tier,
                schema=_CritiqueOutput,
                payload=request.as_dict(),
            )
        except MaterializationProviderError as error:
            return CritiqueResult(
                accepted={},
                blocks={},
                provider_identity=self.runtime_identity(),
                deferred=True,
                reason_code=error.reason_code,
            )
        blocks = _parse_critique(content, [target.block_index for target in request.targets])
        if blocks is None:
            return CritiqueResult(
                accepted={},
                blocks={},
                provider_identity=self.runtime_identity(),
                deferred=True,
                reason_code=ProviderErrorCategory.MALFORMED_OUTPUT.value,
            )
        return CritiqueResult(
            accepted={index: block.accepted for index, block in blocks.items()},
            blocks=blocks,
            provider_identity=self.runtime_identity(),
            notes="gemini",
        )


def _parse_copies(content: Mapping[str, object], requested: Sequence[int]) -> dict[int, str] | None:
    """Strictly parse the writing response against the exact requested block set.

    The response is validated as a whole before any copy is accepted: every
    requested block index must appear exactly once, an unexpected/duplicate/
    malformed entry invalidates the entire response (``None``), and a favorable
    remainder is never retained. There is no partial acceptance.
    """

    raw = content.get("copies")
    if not isinstance(raw, list):
        return None
    requested_list = list(requested)
    requested_set = set(requested_list)
    if len(requested_set) != len(requested_list) or not requested_set:
        return None
    copies: dict[int, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        index = item.get("block_index")
        if isinstance(index, bool) or not isinstance(index, int):
            return None
        if index not in requested_set or index in copies:
            return None
        text = item.get("text", "")
        if not isinstance(text, str):
            return None
        copies[index] = text
    if set(copies) != requested_set:
        return None
    return copies


def _parse_critique(
    content: Mapping[str, object], requested: Sequence[int]
) -> dict[int, BlockCritique] | None:
    """Strictly parse the critique response against the exact requested block set.

    The response is validated as a whole before any decision is created: every
    requested block index must appear exactly once with exactly one valid finding
    for every requested dimension. An unexpected, duplicate, malformed, missing,
    or extra entry invalidates the entire response (``None``); a favorable
    remainder is never retained. There is no partial acceptance.
    """

    raw = content.get("blocks")
    if not isinstance(raw, list):
        return None
    requested_list = list(requested)
    requested_set = set(requested_list)
    if len(requested_set) != len(requested_list) or not requested_set:
        return None
    dimensions = set(CRITIQUE_DIMENSIONS)
    levels = set(CRITIQUE_LEVELS)
    blocks: dict[int, BlockCritique] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        index = item.get("block_index")
        if isinstance(index, bool) or not isinstance(index, int):
            return None
        if index not in requested_set or index in blocks:
            return None
        raw_findings = item.get("findings")
        if not isinstance(raw_findings, list):
            return None
        findings: list[CritiqueDimensionFinding] = []
        seen: set[str] = set()
        for finding in raw_findings:
            if not isinstance(finding, Mapping):
                return None
            dimension = finding.get("dimension")
            level = finding.get("level")
            if not isinstance(dimension, str) or dimension not in dimensions:
                return None
            if not isinstance(level, str) or level not in levels:
                return None
            if dimension in seen:
                return None
            seen.add(dimension)
            evidence = finding.get("evidence", "")
            findings.append(
                CritiqueDimensionFinding(
                    dimension=dimension,
                    level=level,
                    evidence="" if evidence is None else str(evidence),
                )
            )
        if seen != dimensions:
            return None
        blocks[index] = BlockCritique(block_index=index, findings=tuple(findings))
    if set(blocks) != requested_set:
        return None
    return blocks


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


__all__ = [
    "CRITIQUE_SYSTEM_INSTRUCTION",
    "GEMINI_API_VERSION",
    "GEMINI_ROUTINE_MODEL",
    "GEMINI_STRONG_MODEL",
    "WRITING_SYSTEM_INSTRUCTION",
    "GeminiCritiqueProvider",
    "GeminiWritingProvider",
]
