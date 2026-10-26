"""Validated operator planning-request scope and editorial direction.

The request is operator input, never source evidence and never accepted copy.
It is persisted on the existing Stage 4.1 plan-set envelope so a separate worker
can recover the exact same constraints. Bounds are finite and paired; editorial
direction is bounded, stripped text.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

OPERATOR_REQUEST_VERSION = "stage4.1-operator-request-v1"
MAX_EDITORIAL_INTENT_CHARACTERS = 2_000


class OperatorRequestError(ValueError):
    """An operator planning request is malformed or internally inconsistent."""


@dataclass(frozen=True)
class OperatorPlanningRequest:
    """Optional operator-approved source window and editorial direction."""

    source_start: float | None = None
    source_end: float | None = None
    editorial_intent: str | None = None
    version: str = OPERATOR_REQUEST_VERSION

    @property
    def has_window(self) -> bool:
        return self.source_start is not None and self.source_end is not None

    @property
    def is_scoped(self) -> bool:
        return self.has_window or bool(self.editorial_intent)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "editorial_intent": self.editorial_intent,
        }

    def fingerprint_payload(self) -> dict[str, object]:
        """Only request fields that can affect planning semantics."""

        payload: dict[str, object] = {"version": self.version}
        if self.has_window:
            payload["source_start"] = self.source_start
            payload["source_end"] = self.source_end
        if self.editorial_intent:
            payload["editorial_intent"] = self.editorial_intent
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "OperatorPlanningRequest | None":
        if not isinstance(value, Mapping) or not value:
            return None
        version_value = value.get("version")
        if version_value is not None and not isinstance(version_value, str):
            raise OperatorRequestError("unsupported operator planning-request version")
        version = version_value or OPERATOR_REQUEST_VERSION
        if version != OPERATOR_REQUEST_VERSION:
            raise OperatorRequestError("unsupported operator planning-request version")
        start = value.get("source_start")
        end = value.get("source_end")
        intent = value.get("editorial_intent")
        if start is not None and (isinstance(start, bool) or not isinstance(start, (int, float))):
            raise OperatorRequestError("source_start must be a finite number")
        if end is not None and (isinstance(end, bool) or not isinstance(end, (int, float))):
            raise OperatorRequestError("source_end must be a finite number")
        if intent is not None and not isinstance(intent, str):
            raise OperatorRequestError("editorial_intent must be text")
        return parse_operator_request(
            source_start=start,
            source_end=end,
            editorial_intent=intent,
        )


def normalize_operator_request(
    value: OperatorPlanningRequest | Mapping[str, object] | None,
) -> OperatorPlanningRequest | None:
    if value is None or isinstance(value, OperatorPlanningRequest):
        return value
    if isinstance(value, Mapping):
        return OperatorPlanningRequest.from_mapping(value)
    raise OperatorRequestError("operator planning request must be a mapping or request object")


def parse_operator_request(
    *,
    source_start: float | None,
    source_end: float | None,
    editorial_intent: str | None,
) -> OperatorPlanningRequest | None:
    """Validate raw entry-point options, returning ``None`` for an empty request."""

    start = _finite_bound(source_start, "source_start")
    end = _finite_bound(source_end, "source_end")
    if (start is None) != (end is None):
        raise OperatorRequestError("source_start and source_end must be provided together")
    if start is not None and end is not None:
        if start < 0:
            raise OperatorRequestError("source_start must be non-negative")
        if end <= start:
            raise OperatorRequestError("source_end must be greater than source_start")

    intent = _clean_intent(editorial_intent)
    if start is None and intent is None:
        return None
    return OperatorPlanningRequest(source_start=start, source_end=end, editorial_intent=intent)


def _finite_bound(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperatorRequestError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise OperatorRequestError(f"{field} must be finite")
    return number


def _clean_intent(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OperatorRequestError("editorial_intent must be text")
    intent = value.strip()
    if not intent:
        raise OperatorRequestError("editorial_intent cannot be empty")
    if len(intent) > MAX_EDITORIAL_INTENT_CHARACTERS:
        raise OperatorRequestError(
            f"editorial_intent cannot exceed {MAX_EDITORIAL_INTENT_CHARACTERS} characters"
        )
    if any(ord(character) < 32 and character not in "\n\t\r" for character in intent):
        raise OperatorRequestError("editorial_intent contains control characters")
    return intent


__all__ = [
    "MAX_EDITORIAL_INTENT_CHARACTERS",
    "OPERATOR_REQUEST_VERSION",
    "OperatorPlanningRequest",
    "OperatorRequestError",
    "normalize_operator_request",
    "parse_operator_request",
]
