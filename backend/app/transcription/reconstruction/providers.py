"""Strict OpenAI-compatible two-pass reconstruction provider boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.request import Request, urlopen

from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
    ResolutionScores,
)


class ProviderResponseError(ValueError):
    """Provider output cannot safely map to requested stable segment IDs."""


@dataclass(frozen=True)
class GenerationRequest:
    segment_index: int
    raw_text: str
    previous: tuple[str, ...]
    following: tuple[str, ...]


@dataclass(frozen=True)
class ResolutionRequest:
    segment_index: int
    raw_text: str
    previous: tuple[str, ...]
    following: tuple[str, ...]
    candidates: tuple[ReconstructionCandidate, ...]


@dataclass(frozen=True)
class ResolutionChoice:
    candidate_id: str
    scores: ResolutionScores


class ReconstructionProvider(Protocol):
    def health(self) -> ProviderHealth: ...

    def generate_candidates(
        self, requests: list[GenerationRequest]
    ) -> dict[int, list[ReconstructionCandidate]]: ...

    def resolve_candidates(
        self, requests: list[ResolutionRequest]
    ) -> dict[int, ResolutionChoice]: ...

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
        request: HttpRequest | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout_seconds
        self._request = request or _request_bytes

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

    def generate_candidates(
        self, requests: list[GenerationRequest]
    ) -> dict[int, list[ReconstructionCandidate]]:
        content = self._call(
            "Generate no more than two spoken Egyptian Arabic reconstructions per segment. "
            "Do not add facts, translate, formalize, or change numbers or Latin text.",
            {
                "targets": [
                    {
                        "segment_id": item.segment_index,
                        "raw_text": item.raw_text,
                        "previous": item.previous,
                        "following": item.following,
                    }
                    for item in requests
                ]
            },
            _generation_schema(),
        )
        return _parse_generations(content, {item.segment_index for item in requests})

    def resolve_candidates(self, requests: list[ResolutionRequest]) -> dict[int, ResolutionChoice]:
        content = self._call(
            "Select only one supplied candidate per segment using Egyptian naturalness, "
            "semantic coherence, and local discourse. Raw is always allowed.",
            {
                "targets": [
                    {
                        "segment_id": item.segment_index,
                        "raw_text": item.raw_text,
                        "previous": item.previous,
                        "following": item.following,
                        "candidates": [
                            {"candidate_id": candidate.candidate_id, "text": candidate.text}
                            for candidate in item.candidates
                        ],
                    }
                    for item in requests
                ]
            },
            _resolution_schema(),
        )
        return _parse_resolutions(content, requests)

    def _call(
        self, instruction: str, payload: dict[str, object], schema: dict[str, object]
    ) -> dict[str, object]:
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "reconstruction", "schema": schema},
            },
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
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
            result = json.loads(content)
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


def _parse_generations(
    content: dict[str, object], requested: set[int]
) -> dict[int, list[ReconstructionCandidate]]:
    entries = content.get("generations")
    if not isinstance(entries, list):
        raise ProviderResponseError("provider generations must be a list")
    result: dict[int, list[ReconstructionCandidate]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("segment_id"), int):
            raise ProviderResponseError("provider generation has invalid segment ID")
        index = entry["segment_id"]
        candidates = entry.get("candidates")
        if (
            index not in requested
            or index in result
            or not isinstance(candidates, list)
            or len(candidates) > 2
        ):
            raise ProviderResponseError("provider generation has invalid target coverage")
        result[index] = [
            ReconstructionCandidate(
                f"provider-{position}",
                str(candidate["text"]),
                tuple(candidate.get("changes", [])),
                tuple(candidate.get("evidence_segment_ids", [])),
            )
            for position, candidate in enumerate(candidates)
            if isinstance(candidate, dict) and isinstance(candidate.get("text"), str)
        ]
        if len(result[index]) != len(candidates):
            raise ProviderResponseError("provider candidate has invalid text")
    if set(result) != requested:
        raise ProviderResponseError("provider omitted one or more target segments")
    return result


def _parse_resolutions(
    content: dict[str, object], requests: list[ResolutionRequest]
) -> dict[int, ResolutionChoice]:
    entries = content.get("resolutions")
    allowed = {
        item.segment_index: {candidate.candidate_id for candidate in item.candidates}
        for item in requests
    }
    if not isinstance(entries, list):
        raise ProviderResponseError("provider resolutions must be a list")
    result: dict[int, ResolutionChoice] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("segment_id"), int):
            raise ProviderResponseError("provider resolution has invalid segment ID")
        index = entry["segment_id"]
        candidate_id = entry.get("candidate_id")
        values = [entry.get(name) for name in _SCORE_NAMES]
        if (
            index not in allowed
            or index in result
            or not isinstance(candidate_id, str)
            or candidate_id not in allowed[index]
            or any(not isinstance(value, int | float) or not 0 <= value <= 1 for value in values)
        ):
            raise ProviderResponseError("provider resolution has invalid candidate ID or scores")
        result[index] = ResolutionChoice(candidate_id, ResolutionScores(*map(float, values)))
    if set(result) != set(allowed):
        raise ProviderResponseError("provider omitted one or more target segments")
    return result


_SCORE_NAMES = (
    "semantic_coherence",
    "egyptian_naturalness",
    "discourse_continuity",
    "entity_consistency",
    "selection_confidence",
)


def _generation_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"generations": {"type": "array"}},
        "required": ["generations"],
    }


def _resolution_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"resolutions": {"type": "array"}},
        "required": ["resolutions"],
    }


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
