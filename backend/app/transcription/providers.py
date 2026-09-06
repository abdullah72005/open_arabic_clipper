"""Optional, strictly validated local LLM correction providers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.error import URLError
from urllib.request import Request, urlopen

CORRECTION_SYSTEM_PROMPT = """You are correcting speech-recognition errors in an Egyptian Arabic
transcript.

Preserve the speaker's exact meaning and dialect.
Make only changes strongly justified by phonetics and surrounding context.
Do not translate.
Do not formalize.
Do not summarize.
Do not add information.
Preserve English/code-switched words.
Only choose between raw_text and candidate_text. If candidate_text is null,
return raw_text unchanged.

If the raw text is already plausible, return it unchanged."""


class ProviderResponseError(ValueError):
    """A provider response cannot safely be associated with requested segments."""


@dataclass(frozen=True)
class CorrectionRequest:
    segment_index: int
    previous: tuple[str, ...]
    raw_text: str
    following: tuple[str, ...]
    candidate_text: str | None = None


@dataclass(frozen=True)
class ProviderCorrection:
    segment_index: int
    corrected_text: str
    changed: bool
    confidence: float
    changes: list[dict[str, str]]


class CorrectionProvider(Protocol):
    def correct_batch(self, requests: list[CorrectionRequest]) -> list[ProviderCorrection]: ...


HttpRequest = Callable[[str, bytes, dict[str, str], float], bytes]


class OpenAICompatibleCorrectionProvider:
    """Minimal synchronous adapter for locally hosted OpenAI-compatible models."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: float,
        request: HttpRequest | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._request = request or _request_bytes

    def correct_batch(self, requests: list[CorrectionRequest]) -> list[ProviderCorrection]:
        payload = {
            "model": self._model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "segments": [
                                {
                                    "segment_id": request.segment_index,
                                    "previous": list(request.previous),
                                    "raw_text": request.raw_text,
                                    "candidate_text": request.candidate_text,
                                    "next": list(request.following),
                                }
                                for request in requests
                            ],
                            "return_schema": {
                                "corrections": [
                                    {
                                        "segment_id": 0,
                                        "corrected_text": "string",
                                        "changed": True,
                                        "confidence": 0.94,
                                        "changes": [
                                            {"from": "string", "to": "string", "reason": "string"}
                                        ],
                                    }
                                ]
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = self._request(
                self._url,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers,
                self._timeout_seconds,
            )
        except (OSError, URLError) as error:
            raise ProviderResponseError("provider request failed") from error
        return _parse_openai_response(response)


def validate_provider_results(
    requested_ids: set[int], results: list[ProviderCorrection]
) -> dict[int, ProviderCorrection]:
    """Require exactly one valid provider annotation per requested segment identity."""

    validated: dict[int, ProviderCorrection] = {}
    for result in results:
        if result.segment_index not in requested_ids:
            raise ProviderResponseError("provider returned an unrequested segment ID")
        if result.segment_index in validated:
            raise ProviderResponseError("provider returned a duplicate segment ID")
        if not isinstance(result.corrected_text, str) or not result.corrected_text.strip():
            raise ProviderResponseError("provider corrected_text must be a non-empty string")
        if not isinstance(result.changed, bool):
            raise ProviderResponseError("provider changed must be a boolean")
        if not isinstance(result.confidence, float | int) or not 0 <= result.confidence <= 1:
            raise ProviderResponseError("provider confidence must be between zero and one")
        if not isinstance(result.changes, list) or any(
            not isinstance(change, dict)
            or set(change) != {"from", "to", "reason"}
            or any(not isinstance(value, str) for value in change.values())
            for change in result.changes
        ):
            raise ProviderResponseError(
                "provider changes must contain from, to, and reason strings"
            )
        validated[result.segment_index] = result
    if set(validated) != requested_ids:
        raise ProviderResponseError("provider response omitted one or more segment IDs")
    return validated


def _parse_openai_response(response: bytes) -> list[ProviderCorrection]:
    try:
        payload = json.loads(response.decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        corrections = json.loads(content)["corrections"]
    except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderResponseError("provider returned invalid structured JSON") from error
    if not isinstance(corrections, list):
        raise ProviderResponseError("provider corrections must be a list")
    parsed: list[ProviderCorrection] = []
    for correction in corrections:
        if not isinstance(correction, dict):
            raise ProviderResponseError("provider correction must be an object")
        try:
            parsed.append(
                ProviderCorrection(
                    segment_index=correction["segment_id"],
                    corrected_text=correction["corrected_text"],
                    changed=correction["changed"],
                    confidence=correction["confidence"],
                    changes=correction["changes"],
                )
            )
        except KeyError as error:
            raise ProviderResponseError(
                "provider correction is missing a required field"
            ) from error
    return parsed


def _request_bytes(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local endpoint
        return cast(bytes, response.read())
