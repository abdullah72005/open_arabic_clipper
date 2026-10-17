"""Shared pure delivery classification for Stage 6 provider attempts.

Both the production materialization components and the bounded TTS audition must
interpret a provider outcome the same way, so the classification is defined once
here and imported by both. It is pure (no I/O, no persistence): it maps a raised
typed provider error or a returned provider response onto a truthful durable
attempt status.

Delivery outcome (did a response arrive?) is kept separate from content validity
(was the arrived content usable?):

* an ambiguous transport loss (timeout/connection, or an untyped exception from a
  hosted call) is ``UNCERTAIN`` — the request may or may not have reached the
  provider;
* a response that was received but whose content is unusable
  (``MALFORMED_OUTPUT``/``SAFETY_REFUSAL``) is ``RESPONDED``;
* a known HTTP rejection (auth, rate limit, service unavailable, provider error,
  invalid request) is ``FAILED_KNOWN``;
* a crash-left reservation (no response ever recorded) is ``UNCERTAIN``.
"""

from __future__ import annotations

from app.materialization.repository import (
    ATTEMPT_FAILED_KNOWN,
    ATTEMPT_RESPONDED,
    ATTEMPT_UNCERTAIN,
)

#: Categories that mean an HTTP response was received (even if its semantic
#: content was unusable). A malformed/safety-refused response is a truthful
#: RESPONDED attempt; only its *content* was not accepted.
RESPONDED_DELIVERY_REASONS = frozenset({"MALFORMED_OUTPUT", "SAFETY_REFUSAL"})
#: Categories that mean the request may have reached the provider but no response
#: was received (ambiguous transport loss).
UNCERTAIN_DELIVERY_REASONS = frozenset({"TIMEOUT", "CONNECTION"})
RATE_LIMITED_REASON = "RATE_LIMITED"


def raised_attempt_status(error: BaseException) -> str:
    """Classify a *raised* typed provider error into a truthful durable status.

    A typed ``MaterializationProviderError`` whose category is an ambiguous
    transport loss (timeout/connection) is ``UNCERTAIN``; one whose category means
    a response was received but unusable (malformed/safety) is ``RESPONDED``; any
    other known category is a known HTTP rejection (``FAILED_KNOWN``). Any untyped
    exception from a hosted call is ambiguous and therefore conservatively
    ``UNCERTAIN``.
    """

    category = str(getattr(error, "category", "") or "")
    if category in UNCERTAIN_DELIVERY_REASONS:
        return ATTEMPT_UNCERTAIN
    if category in RESPONDED_DELIVERY_REASONS:
        return ATTEMPT_RESPONDED
    if category:
        return ATTEMPT_FAILED_KNOWN
    return ATTEMPT_UNCERTAIN


def delivery_status(response: object) -> str:
    """Classify a returned provider response into a truthful attempt outcome.

    A non-deferred response (or a received response whose content is unusable) is
    RESPONDED; an ambiguous transport loss after possible submission is UNCERTAIN;
    a known provider rejection is FAILED_KNOWN. Only a RESPONDED *and validated*
    response may become an accepted checkpoint.
    """

    deferred = bool(getattr(response, "deferred", False))
    reason = getattr(response, "reason_code", None)
    reason_code = str(reason) if reason else ""
    if not deferred:
        return ATTEMPT_RESPONDED
    if reason_code in UNCERTAIN_DELIVERY_REASONS:
        return ATTEMPT_UNCERTAIN
    if reason_code in RESPONDED_DELIVERY_REASONS:
        return ATTEMPT_RESPONDED
    return ATTEMPT_FAILED_KNOWN


def is_rate_limited(reason: object) -> bool:
    """Whether a raised error or returned response reports a rate/quota rejection."""

    if isinstance(reason, BaseException):
        return str(getattr(reason, "category", "") or "") == RATE_LIMITED_REASON
    return str(getattr(reason, "reason_code", "") or "") == RATE_LIMITED_REASON


__all__ = [
    "RATE_LIMITED_REASON",
    "RESPONDED_DELIVERY_REASONS",
    "UNCERTAIN_DELIVERY_REASONS",
    "delivery_status",
    "is_rate_limited",
    "raised_attempt_status",
]
