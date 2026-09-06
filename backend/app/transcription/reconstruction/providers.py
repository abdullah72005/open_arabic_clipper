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
    ReconstructionWindow,
    ResolutionScores,
)


class ProviderResponseError(ValueError):
    """Provider output cannot safely map to requested stable segment IDs."""


@dataclass(frozen=True)
class GenerationRequest:
    segment_index: int | None = None
    raw_text: str = ""
    previous: tuple[str, ...] = ()
    following: tuple[str, ...] = ()
    window: ReconstructionWindow | None = None
    language: str | None = None
    entity_forms: tuple[str, ...] = ()
    routing_decision: object | None = None

    def __post_init__(self) -> None:
        if self.window is not None:
            current = next(
                item
                for item in self.window.segments
                if item.segment_index == self.window.target_segment_index
            )
            object.__setattr__(self, "segment_index", self.window.target_segment_index)
            object.__setattr__(self, "raw_text", current.raw_text)

    def to_payload(self) -> dict[str, object]:
        if self.window is None:
            items: list[dict[str, object]] = [
                {
                    "segment_id": self.segment_index,
                    "raw_text": self.raw_text,
                    "corrected_text": self.raw_text,
                    "previous_raw": list(self.previous),
                    "following_raw": list(self.following),
                }
            ]
        else:
            target = self.window.target_segment_index
            ordered = self.window.segments
            position = next(i for i, item in enumerate(ordered) if item.segment_index == target)
            items = [
                {
                    "segment_id": item.segment_index,
                    "start": item.start,
                    "end": item.end,
                    "raw_text": item.raw_text,
                    "corrected_text": item.corrected_text,
                    "previous_raw": [entry.raw_text for entry in ordered[:position]],
                    "previous_corrected": [entry.corrected_text for entry in ordered[:position]],
                    "following_raw": [entry.raw_text for entry in ordered[position + 1 :]],
                    "following_corrected": [
                        entry.corrected_text for entry in ordered[position + 1 :]
                    ],
                    "word_evidence": [
                        {
                            "text": word.text,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        }
                        for word in item.word_evidence
                    ],
                }
                for item in self.window.segments
            ]
        routing = self.routing_decision
        reason = getattr(routing, "reason", "") if routing is not None else ""
        reasons = list(getattr(routing, "reasons", (reason,))) if routing is not None else []
        focus = getattr(routing, "focus_spans", ()) if routing is not None else ()
        return {
            "segment_id": self.segment_index,
            "window": items,
            "language": self.language,
            "entities": list(self.entity_forms),
            "routing": {
                "priority": getattr(getattr(routing, "priority", None), "value", None),
                "reason": reason,
                "reasons": reasons,
                "focus_spans": [
                    {
                        "text": span.text,
                        "start": span.start,
                        "end": span.end,
                        "probability": span.probability,
                    }
                    for span in focus
                ],
            },
        }


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
    candidate_scores: dict[str, ResolutionScores] | None = None

    @property
    def margin(self) -> float:
        scores = self.candidate_scores or {self.candidate_id: self.scores}
        ranked = sorted(((value.score, key) for key, value in scores.items()), reverse=True)
        return ranked[0][0] - ranked[1][0] if len(ranked) > 1 else ranked[0][0]


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

    def generate_candidates(
        self, requests: list[GenerationRequest]
    ) -> dict[int, list[ReconstructionCandidate]]:
        content = self._call(
            "Reconstruct only what was most plausibly spoken in the target segment. "
            "Preserve Egyptian Arabic; do not rewrite into MSA. "
            "Do not summarize, paraphrase stylistically, translate, add facts, names, numbers, "
            "or clauses. Use focus_spans and surrounding raw/corrected evidence. Return zero "
            "candidates when unchanged is safer. "
            "Generate no more than two spoken Egyptian Arabic reconstructions per segment.",
            {"targets": [item.to_payload() for item in requests]},
            _generation_schema(),
        )
        return _parse_generations(content, {item.segment_index for item in requests})

    def resolve_candidates(self, requests: list[ResolutionRequest]) -> dict[int, ResolutionChoice]:
        content = self._call(
            "Score every supplied candidate, then select one supplied candidate per segment using Egyptian naturalness, "
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
        if self.provider_name == "ollama":
            body["reasoning_effort"] = "none"
            if self.model.startswith("qwen3"):
                body["messages"][0]["content"] = instruction + " /no_think"
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
        candidate_id = entry.get("selected_candidate_id", entry.get("candidate_id"))
        score_entries = entry.get("candidate_scores")
        if score_entries is None:
            # Accept the pre-Task-5 response shape during rolling upgrades.
            score_entries = [{"candidate_id": candidate_id, **dict(zip(_SCORE_NAMES, [entry.get(name) for name in _SCORE_NAMES]))}]
        score_map: dict[str, ResolutionScores] = {}
        if not isinstance(score_entries, list):
            raise ProviderResponseError("provider candidate scores must be a list")
        for scored in score_entries:
            if not isinstance(scored, dict) or not isinstance(scored.get("candidate_id"), str):
                raise ProviderResponseError("provider candidate score has invalid ID")
            sid = scored["candidate_id"]
            values = [scored.get(name) for name in _SCORE_NAMES]
            if sid in score_map or sid not in allowed.get(index, set()) or any(not isinstance(value, int | float) or not 0 <= value <= 1 for value in values):
                raise ProviderResponseError("provider candidate score has invalid ID or values")
            score_map[sid] = ResolutionScores(*map(float, values))
        if index in allowed and set(score_map) != allowed[index]:
            raise ProviderResponseError("provider candidate scores are incomplete")
        selected = score_map.get(candidate_id)
        values = ([getattr(selected, name) for name in _SCORE_NAMES] if selected is not None
                  else [entry.get(name) for name in _SCORE_NAMES])
        if (
            index not in allowed
            or index in result
            or not isinstance(candidate_id, str)
            or candidate_id not in allowed[index]
            or (score_map.get(candidate_id) is None and any(not isinstance(value, int | float) or not 0 <= value <= 1 for value in values))
        ):
            raise ProviderResponseError("provider resolution has invalid candidate ID or scores")
        selected_scores = score_map.get(candidate_id, ResolutionScores(*map(float, values)))
        result[index] = ResolutionChoice(candidate_id, selected_scores, score_map)
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
        "properties": {"resolutions": {"type": "array", "items": {"type": "object", "required": ["segment_id", "selected_candidate_id", "candidate_scores"]}}},
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


def batch_generation_requests(
    requests: list[GenerationRequest], *, max_windows: int = 8, max_characters: int = 24_000
) -> list[list[GenerationRequest]]:
    """Group deterministic UTF-8 payloads, prioritizing reconstruction work."""
    if not 1 <= max_windows <= 16:
        raise ValueError("max_windows exceeds hard maximum 16")
    if not 1 <= max_characters <= 48_000:
        raise ValueError("max_characters exceeds hard maximum 48000")
    ordered = sorted(
        enumerate(requests),
        key=lambda pair: (
            0
            if getattr(getattr(pair[1].routing_decision, "priority", None), "value", "")
            == "reconstruct"
            else 1,
            pair[0],
        ),
    )
    batches: list[list[GenerationRequest]] = []
    current: list[GenerationRequest] = []
    for _, request in ordered:
        candidate = [*current, request]
        encoded = json.dumps(
            [item.to_payload() for item in candidate], ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        if current and (len(candidate) > max_windows or len(encoded) > max_characters):
            batches.append(current)
            current = [request]
        elif not current and len(encoded) > max_characters:
            raise ValueError("single generation request exceeds character bound")
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches
