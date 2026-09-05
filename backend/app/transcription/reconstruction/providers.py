"""Strict OpenAI-compatible two-pass reconstruction provider boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.request import Request, urlopen

from app.transcription.reconstruction.types import ReconstructionCandidate, ResolutionScores


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
    def generate_candidates(
        self, requests: list[GenerationRequest]
    ) -> dict[int, list[ReconstructionCandidate]]: ...

    def resolve_candidates(
        self, requests: list[ResolutionRequest]
    ) -> dict[int, ResolutionChoice]: ...


HttpRequest = Callable[[str, bytes, dict[str, str], float], bytes]


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
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._model = model
        self._timeout = timeout_seconds
        self._request = request or _request_bytes

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
            "model": self._model,
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
        try:
            response = self._request(
                self._url,
                json.dumps(body, ensure_ascii=False).encode(),
                {"Content-Type": "application/json"},
                self._timeout,
            )
            parsed = json.loads(response.decode())
            content = parsed["choices"][0]["message"]["content"]
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


def _request_bytes(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
    with urlopen(
        Request(url, data=body, headers=headers, method="POST"), timeout=timeout
    ) as response:  # noqa: S310
        return response.read()
