"""Strict OpenAI-compatible one-pass reconstruction provider boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, cast
from urllib.request import Request, urlopen

from app.transcription.reconstruction.types import (
    AcousticEvidence,
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
    ResolutionScores,
    WordEvidence,
)


class ProviderResponseError(ValueError):
    """Provider output cannot safely map to requested stable segment IDs."""


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
        }

    def estimated_tokens(self) -> int:
        """Rough token count for prompt budgeting; 1 token ~= 2 UTF-8 chars for Arabic."""
        payload = json.dumps(self.to_payload(), ensure_ascii=False)
        return len(payload) // 2


class ReconstructionProvider(Protocol):
    def health(self) -> ProviderHealth: ...

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]: ...

    def release(self) -> None: ...


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
        request: HttpRequest | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout_seconds
        self._max_context_tokens = max_context_tokens
        self._request = request or _request_bytes
        self.provider_name = "openai_compatible"

    def health(self) -> ProviderHealth:
        try:
            payload = self._json_request("GET", "/v1/models", None)
            models = payload.get("data")
            if not isinstance(models, list):
                raise ProviderResponseError("provider response is missing models")
        except ProviderResponseError:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE,
                "openai_compatible",
                self.model,
                None,
                "provider health check failed",
            )
        match = next(
            (item for item in models if isinstance(item, dict) and item.get("id") == self.model),
            None,
        )
        if match is None:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE,
                "openai_compatible",
                self.model,
                None,
                f"configured model {self.model} is not available",
            )
        return ProviderHealth(
            ProviderAvailability.AVAILABLE,
            "openai_compatible",
            self.model,
            None,
            "model available",
        )

    def release(self) -> None:
        """Generic OpenAI-compatible endpoints have no portable unload operation."""

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        if self._max_context_tokens is not None:
            requests = [
                _shrink_request_to_budget(request, self._max_context_tokens) for request in requests
            ]
            for request in requests:
                if request.estimated_tokens() > self._max_context_tokens:
                    raise ProviderResponseError(
                        f"request for segment {request.segment_index} exceeds context budget "
                        f"({request.estimated_tokens()} > {self._max_context_tokens} tokens)"
                    )
        content = self._call(
            "You are a conservative Arabic ASR post-processor for Egyptian Arabic speech. "
            "For the target segment, return the most plausible SPOKEN EGYPTIAN ARABIC text. "
            "Preserve Egyptian colloquial word choices, pronunciation-driven spelling, "
            "and dialect. "
            "Do NOT standardize into Modern Standard Arabic (MSA). "
            "Example: ASR 'ثلاثة يام' should become 'تلات أيام' (spoken Egyptian), "
            "not 'ثلاثة أيام' (MSA). "
            "Preserve all names, numbers, Latin tokens, and digits exactly as they appear. "
            "Do not add facts, clauses, or change entities. "
            "Use only the small local context provided. "
            "If the raw text is already correct, return it unchanged and set unchanged=true. "
            "Output ONLY a JSON object with this exact shape: "
            '{"reconstructions": [{"segment_id": int, "corrected_text": string, '
            '"unchanged": bool, "confidence": number, "explanation": string, "changes": []}]}.',
            {"targets": [item.to_payload() for item in requests]},
        )
        return _parse_reconstructions(content, requests)

    def _call(self, instruction: str, payload: dict[str, object]) -> dict[str, object]:
        system_content = instruction
        if self.provider_name == "ollama" and self.model.startswith("qwen3"):
            system_content = instruction + " /no_think"
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": system_content},
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


def _shrink_request_to_budget(
    request: ReconstructionRequest, max_tokens: int
) -> ReconstructionRequest:
    """Drop surrounding context deterministically until the request fits the budget."""

    previous = list(request.previous)
    following = list(request.following)
    entities = list(request.entities)
    word_evidence = list(request.word_evidence)
    while request.estimated_tokens() > max_tokens and (previous or following):
        if following:
            following.pop()
        elif previous:
            previous.pop()
        request = replace(
            request, previous=tuple(previous), following=tuple(following), entities=tuple(entities)
        )
    while request.estimated_tokens() > max_tokens and entities:
        entities.pop()
        request = replace(request, entities=tuple(entities))
    while request.estimated_tokens() > max_tokens and len(word_evidence) > 1:
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
        confidence = entry.get("confidence", 0.0)
        if not isinstance(confidence, int | float):
            confidence = 0.0
        confidence = float(confidence)
        scores = ResolutionScores(
            semantic_coherence=confidence,
            egyptian_naturalness=confidence,
            discourse_continuity=confidence,
            entity_consistency=confidence,
            selection_confidence=confidence,
        )
        changes = tuple(entry.get("changes", [])) if isinstance(entry.get("changes"), list) else ()
        explanation = str(entry.get("explanation", ""))
        result[index] = ReconstructionCandidate(
            candidate_id="raw" if unchanged else "provider-0",
            text=text,
            changes=changes,
            scores=scores,
            explanation=explanation,
        )
    if set(result) != requested:
        raise ProviderResponseError("provider omitted one or more target segments")
    return result


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
        raise ValueError("no JSON object found in response")
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
