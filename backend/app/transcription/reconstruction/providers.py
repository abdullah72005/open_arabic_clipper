"""Strict OpenAI-compatible one-pass reconstruction provider boundary."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, cast
from urllib.request import Request, urlopen

from app.transcription.reconstruction.confidence import CONFIDENCE_POLICY_VERSION
from app.transcription.reconstruction.types import (
    AcousticEvidence,
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
    RequestSizeDiagnostics,
    WordEvidence,
    estimate_tokens,
)
from app.transcription.reconstruction.validation import VALIDATION_VERSION


class ProviderResponseError(ValueError):
    """Provider output cannot safely map to requested stable segment IDs."""


_BATCH_OUTPUT_TOKEN_CAP = 4_096


def _batch_output_tokens(base_output_tokens: int, request_count: int) -> int:
    """Scale the output budget to the bounded number of requested targets.

    A micro-batch of ``request_count`` targets may legitimately need up to
    ``base_output_tokens`` per target; the budget is capped so a pathological
    batch can never request unbounded output.
    """

    return min(_BATCH_OUTPUT_TOKEN_CAP, base_output_tokens * max(1, request_count))


class ModelNotFoundError(ProviderResponseError):
    """The configured model is absent from the provider's model listing."""


@dataclass(frozen=True)
class ReconstructionRequest:
    """Small local context for a single target segment reconstruction."""

    segment_index: int
    raw_text: str
    corrected_text: str = ""
    previous: tuple[str, ...] = ()
    following: tuple[str, ...] = ()
    word_evidence: tuple[WordEvidence, ...] = ()
    acoustic: AcousticEvidence | None = None
    entities: tuple[str, ...] = ()
    routing_reasons: tuple[str, ...] = ()
    focus_spans: tuple[WordEvidence, ...] = ()
    language: str | None = None
    dialect_profile: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_index,
            "raw_text": self.raw_text,
            "corrected_text": self.corrected_text,
            "previous": list(self.previous),
            "following": list(self.following),
            "words": [
                {
                    "text": word.text,
                    "start": word.start,
                    "end": word.end,
                    "probability": word.probability,
                }
                for word in self.word_evidence
            ],
            "acoustic": {
                "confidence": self.acoustic.confidence if self.acoustic else None,
                "average_word_probability": (
                    self.acoustic.average_word_probability if self.acoustic else None
                ),
            },
            "entities": list(self.entities),
            "routing_reasons": list(self.routing_reasons),
            "focus_spans": [
                {
                    "text": span.text,
                    "start": span.start,
                    "end": span.end,
                    "probability": span.probability,
                }
                for span in self.focus_spans
            ],
            "language": self.language,
            "dialect_profile": self.dialect_profile,
        }

    def estimated_tokens(
        self,
        *,
        system_instruction: str = "",
        output_tokens: int = 0,
        chat_framing_reserve: int = 0,
        safety_reserve: int = 0,
    ) -> int:
        """Conservative estimate of the complete chat envelope, not only the payload.

        The estimate counts the exact serialized ``{"targets": [...]}`` wrapper
        that is sent as the user message.
        """

        payload = json.dumps({"targets": [self.to_payload()]}, ensure_ascii=False)
        return (
            estimate_tokens(system_instruction)
            + estimate_tokens(payload)
            + chat_framing_reserve
            + output_tokens
            + safety_reserve
        )


class ReconstructionProvider(Protocol):
    def health(self) -> ProviderHealth: ...

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]: ...

    def release(self) -> None: ...

    def runtime_identity(self) -> dict[str, object]: ...

    def refresh_runtime_identity(self) -> dict[str, object]: ...


HttpRequest = Callable[[str, str, bytes | None, dict[str, str], float], bytes]


class OpenAICompatibleReconstructionProvider:
    """Use bounded JSON-only calls against explicitly configured local endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        max_context_tokens: int | None = None,
        output_tokens: int = 256,
        chat_framing_reserve: int = 64,
        safety_reserve: int = 128,
        request: HttpRequest | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout_seconds
        self._max_context_tokens = max_context_tokens
        self._output_tokens = output_tokens
        self._chat_framing_reserve = chat_framing_reserve
        self._safety_reserve = safety_reserve
        self._request = request or _request_bytes
        self.provider_name = "openai_compatible"
        self._last_request_sizes: tuple[RequestSizeDiagnostics, ...] = ()
        self._model_digest: str | None = None

    def last_request_sizes(self) -> tuple[RequestSizeDiagnostics, ...]:
        """Return measured serialized prompt sizes for the most recent call."""

        return self._last_request_sizes

    def runtime_identity(self) -> dict[str, object]:
        """Return every output-affecting reconstruction dependency as stable data."""

        return {
            "provider": self.provider_name,
            "model": self.model,
            "digest": self._model_digest or "digest_unavailable",
            "prompt_hash": _PROMPT_HASH,
            "schema_version": _PROMPT_SCHEMA_VERSION,
            "max_context_tokens": self._max_context_tokens,
            "output_tokens": self._output_tokens,
            "chat_framing_reserve": self._chat_framing_reserve,
            "safety_reserve": self._safety_reserve,
            "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
            "validation_version": VALIDATION_VERSION,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        """Resolve the live model digest now and return the refreshed identity.

        A failed lookup is represented as ``digest_unavailable``; a successful
        lookup always overwrites any previously cached digest.
        """

        try:
            self._model_digest = self._fetch_live_digest()
        except ProviderResponseError:
            self._model_digest = None
        return self.runtime_identity()

    def _fetch_live_digest(self) -> str | None:
        payload = self._json_request("GET", "/v1/models", None)
        models = payload.get("data")
        if not isinstance(models, list):
            raise ProviderResponseError("provider response is missing models")
        match = next(
            (item for item in models if isinstance(item, dict) and item.get("id") == self.model),
            None,
        )
        if match is None:
            raise ModelNotFoundError(f"configured model {self.model} is not available")
        return str(match.get("digest") or "") or None

    def health(self) -> ProviderHealth:
        try:
            digest = self._fetch_live_digest()
        except ModelNotFoundError as error:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE,
                "openai_compatible",
                self.model,
                None,
                str(error),
            )
        except ProviderResponseError:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE,
                "openai_compatible",
                self.model,
                None,
                "provider health check failed",
            )
        self._model_digest = digest
        return ProviderHealth(
            ProviderAvailability.AVAILABLE,
            "openai_compatible",
            self.model,
            digest,
            "model available",
        )

    def release(self) -> None:
        """Generic OpenAI-compatible endpoints have no portable unload operation."""

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        """Execute exactly one real provider request for the given targets.

        This method never silently splits its input into several HTTP calls. The
        orchestration layer plans context-safe actual request groups with
        ``plan_aggregate_batches`` and invokes this method once per group, so
        cancellation, local wall-time checks, and checkpoints can run between
        every real request. As a defensive boundary, an over-budget combined set
        that was not pre-planned raises before any HTTP dispatch instead of
        looping.
        """

        profile = next(
            (request.dialect_profile for request in requests if request.dialect_profile), None
        )
        system_instruction = self._system_instruction(profile)
        planned = self.plan_aggregate_batches(requests)
        if self._max_context_tokens is not None and len(planned) != 1:
            raise ProviderResponseError(
                "provider request exceeds context budget; orchestration must plan "
                "context-safe batches"
            )
        batch = planned[0] if planned else []
        self._last_request_sizes = tuple(
            RequestSizeDiagnostics(
                segment_index=request.segment_index,
                serialized_bytes=len(
                    json.dumps({"targets": [request.to_payload()]}, ensure_ascii=False).encode(
                        "utf-8"
                    )
                ),
                estimated_input_tokens=self._envelope_estimate(
                    system_instruction, self._output_tokens
                )(request),
            )
            for request in batch
        )
        output_tokens = _batch_output_tokens(self._output_tokens, len(batch))
        content = self._call(
            system_instruction,
            {"targets": [item.to_payload() for item in batch]},
            output_tokens=output_tokens,
        )
        return _parse_reconstructions(content, batch)

    def plan_aggregate_batches(
        self, requests: list[ReconstructionRequest]
    ) -> list[list[ReconstructionRequest]]:
        """Plan context-safe actual request groups without executing any HTTP call.

        The service-layer planner bounds the number of windows and characters;
        this is the correctness bound at the real request boundary. Every request
        is first shrunk so its own single-target envelope fits, then requests are
        greedily grouped (in stable order) while the exact combined envelope that
        will be sent — system instruction, full ``{"targets": [...]}`` payload,
        chat-framing reserve, safety reserve, and the scaled output budget for the
        group — stays within ``max_context_tokens``. Each returned group is one
        actual provider request that orchestration schedules, polls cancellation
        and wall-time around, and checkpoints after. A single target that still
        cannot fit raises before any HTTP dispatch (no recursive split/retry).
        """

        profile = next(
            (request.dialect_profile for request in requests if request.dialect_profile), None
        )
        system_instruction = self._system_instruction(profile)
        if self._max_context_tokens is None:
            return [list(requests)]
        single_envelope = self._envelope_estimate(system_instruction, self._output_tokens)
        requests = [
            _shrink_request_to_budget(
                request,
                self._max_context_tokens,
                envelope=single_envelope,
            )
            for request in requests
        ]
        for request in requests:
            estimated = single_envelope(request)
            if estimated > self._max_context_tokens:
                raise ProviderResponseError(
                    f"request for segment {request.segment_index} exceeds context budget "
                    f"({estimated} > {self._max_context_tokens} tokens)"
                )
        batches: list[list[ReconstructionRequest]] = []
        current: list[ReconstructionRequest] = []
        for request in requests:
            if current:
                trial = [*current, request]
                if self._aggregate_envelope(trial, system_instruction) > self._max_context_tokens:
                    batches.append(current)
                    current = []
            current.append(request)
        if current:
            batches.append(current)
        return batches

    def _aggregate_envelope(
        self, requests: list[ReconstructionRequest], system_instruction: str
    ) -> int:
        """Conservative token estimate of the exact combined chat envelope sent."""

        payload = json.dumps(
            {"targets": [item.to_payload() for item in requests]}, ensure_ascii=False
        )
        return (
            estimate_tokens(system_instruction)
            + estimate_tokens(payload)
            + self._chat_framing_reserve
            + _batch_output_tokens(self._output_tokens, len(requests))
            + self._safety_reserve
        )

    def _system_instruction(self, profile: str | None = None) -> str:
        instruction = SYSTEM_INSTRUCTION
        if self.provider_name == "ollama" and self.model.startswith("qwen3"):
            instruction = instruction + " /no_think"
        return instruction_for_profile(instruction, profile)

    def _envelope_estimate(
        self, system_instruction: str, output_tokens: int
    ) -> Callable[[ReconstructionRequest], int]:
        def estimate(request: ReconstructionRequest) -> int:
            return request.estimated_tokens(
                system_instruction=system_instruction,
                output_tokens=output_tokens,
                chat_framing_reserve=self._chat_framing_reserve,
                safety_reserve=self._safety_reserve,
            )

        return estimate

    def _call(
        self, instruction: str, payload: dict[str, object], output_tokens: int
    ) -> dict[str, object]:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": output_tokens,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        if self.provider_name == "ollama":
            body["reasoning_effort"] = "none"
        parsed = self._json_request("POST", "/v1/chat/completions", body)
        try:
            choices = parsed["choices"]
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise TypeError
            message = choices[0]["message"]
            if not isinstance(message, dict):
                raise TypeError
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError
            result = _extract_json_object(content)
        except (
            KeyError,
            IndexError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            OSError,
        ) as error:
            raise ProviderResponseError("provider returned invalid structured JSON") from error
        if not isinstance(result, dict):
            raise ProviderResponseError("provider response must be an object")
        return result

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        headers = {} if body is None else {"Content-Type": "application/json"}
        try:
            response = self._request(
                method,
                f"{self._base_url}/{path.lstrip('/')}",
                body,
                headers,
                self._timeout,
            )
            parsed = json.loads(response.decode())
        except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
            raise ProviderResponseError("provider request failed") from error
        if not isinstance(parsed, dict):
            raise ProviderResponseError("provider response must be an object")
        return parsed


SYSTEM_INSTRUCTION = (
    "You are a conservative Arabic ASR post-processor. "
    "Repair only probable speech-recognition errors. "
    "Preserve the dialect and register actually evidenced in the source and context. "
    "Preserve detected Arabic-English code switching. "
    "Preserve names, abbreviations, technical tokens, and numbers exactly. "
    "Leave already plausible text unchanged. "
    "Do not translate, summarize, paraphrase, formalize, colloquialize, standardize, "
    "or add missing clauses or facts. "
    "Do not invent words absent from the text evidence. "
    "Use only the small local context provided. "
    "If the raw text is already correct, return it unchanged and set unchanged=true. "
    "Output ONLY a JSON object with this exact shape: "
    '{"reconstructions": [{"segment_id": int, "corrected_text": string, '
    '"unchanged": bool, "confidence": number, "explanation": string, "changes": []}]}.'
)

_PROMPT_SCHEMA_VERSION = "stage-2-7-one-pass-v2"
_PROMPT_HASH = hashlib.sha256(SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest()

# Backward-compatible alias for existing consumers of the shared instruction.
_SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION

# Validated dialect-profile preservation addenda. Every provider receives the
# same dialect-neutral base instruction plus, when a profile is supplied, the
# narrow profile-specific addendum below. Nothing is interpolated from
# unvalidated input.
PROFILE_ADDENDA: dict[str, str] = {
    "EGYPTIAN": (
        "The source is Egyptian Arabic. Preserve Egyptian colloquial word choices "
        "and pronunciation-driven spelling; do not convert them to Modern Standard "
        "Arabic or any other dialect."
    ),
    "SAUDI": (
        "The source is Saudi Arabic. Preserve Saudi speech; do not Egyptianize, "
        "generic-Gulf-normalize, or formalize it."
    ),
    "GULF": (
        "The source is Gulf Arabic. Preserve Gulf speech; do not Egyptianize, "
        "Saudi-normalize, or formalize it."
    ),
    "LEVANTINE": (
        "The source is Levantine Arabic. Preserve Levantine speech; do not "
        "Egyptianize or formalize it."
    ),
    "MSA": (
        "The source is Modern Standard Arabic (Fusha). Preserve formal MSA wording; "
        "do not colloquialize it."
    ),
    "UNKNOWN_ARABIC": (
        "The Arabic dialect is unknown or mixed. Preserve the observed forms "
        "conservatively; do not force any regional dialect or Modern Standard Arabic."
    ),
}


def instruction_for_profile(base: str, profile: str | None) -> str:
    """Append a validated dialect-profile preservation addendum when supplied.

    ``profile`` may be any stable profile name string from
    ``ArabicDialectProfile``. Unknown or missing profiles append nothing, so a
    provider never receives an arbitrary interpolated dialect instruction.
    """

    if not profile:
        return base
    addendum = PROFILE_ADDENDA.get(profile)
    if addendum is None:
        return base
    return base + "\n" + addendum


def _shrink_request_to_budget(
    request: ReconstructionRequest,
    max_tokens: int,
    *,
    envelope: Callable[[ReconstructionRequest], int],
) -> ReconstructionRequest:
    """Drop surrounding context deterministically until the request fits the budget."""

    previous = list(request.previous)
    following = list(request.following)
    entities = list(request.entities)
    word_evidence = list(request.word_evidence)
    while envelope(request) > max_tokens and (previous or following):
        if following:
            following.pop()
        elif previous:
            previous.pop()
        request = replace(
            request, previous=tuple(previous), following=tuple(following), entities=tuple(entities)
        )
    while envelope(request) > max_tokens and entities:
        entities.pop()
        request = replace(request, entities=tuple(entities))
    while envelope(request) > max_tokens and len(word_evidence) > 1:
        word_evidence.pop()
        request = replace(request, word_evidence=tuple(word_evidence))
    return request


def _parse_reconstructions(
    content: dict[str, object], requests: list[ReconstructionRequest]
) -> dict[int, ReconstructionCandidate]:
    entries = content.get("reconstructions")
    requested = {item.segment_index for item in requests}
    if not isinstance(entries, list):
        raise ProviderResponseError("provider reconstructions must be a list")
    result: dict[int, ReconstructionCandidate] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("segment_id"), int):
            raise ProviderResponseError("provider reconstruction has invalid segment ID")
        index = entry["segment_id"]
        if index not in requested or index in result:
            raise ProviderResponseError("provider reconstruction has invalid target coverage")
        text = entry.get("corrected_text") or entry.get("text")
        if not isinstance(text, str):
            raise ProviderResponseError("provider reconstruction has invalid text")
        unchanged = bool(entry.get("unchanged"))
        confidence = _validated_confidence(entry)
        changes = tuple(entry.get("changes", [])) if isinstance(entry.get("changes"), list) else ()
        explanation = str(entry.get("explanation", ""))
        result[index] = ReconstructionCandidate(
            candidate_id="raw" if unchanged else "provider-0",
            text=text,
            changes=changes,
            provider_confidence=confidence,
            explanation=explanation,
        )
    if set(result) != requested:
        raise ProviderResponseError("provider omitted one or more target segments")
    return result


def _validated_confidence(entry: dict[str, object]) -> float:
    """Accept only a non-boolean finite real confidence inside the inclusive unit range."""

    value = entry.get("confidence", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderResponseError("provider reconstruction has invalid confidence")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ProviderResponseError("provider reconstruction has invalid confidence")
    return numeric


def _extract_json_object(text: str) -> dict[str, object]:
    """Extract the payload JSON object from a model response that may include
    reasoning or markdown.
    """

    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Some thinking models (e.g. qwen3) emit chain-of-thought before the JSON payload.
    # Try a clean parse first, then scan for the largest valid JSON object.
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    best: dict[str, object] | None = None
    best_len = 0
    for start in range(len(text)):
        if text[start] != "{":
            continue
        for end in range(len(text), start, -1):
            try:
                candidate = json.loads(text[start:end])
                if isinstance(candidate, dict) and end - start > best_len:
                    best = candidate
                    best_len = end - start
                    break
            except json.JSONDecodeError:
                continue
    if best is None:
        raise ProviderResponseError("no JSON object found in response")
    return best


def _request_bytes(
    method: str,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    timeout: float,
) -> bytes:
    with urlopen(
        Request(url, data=body, headers=headers, method=method), timeout=timeout
    ) as response:  # noqa: S310
        return cast(bytes, response.read())
