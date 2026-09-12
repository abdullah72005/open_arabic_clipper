"""Shared Redis-backed Gemini admission control for Stage 3.5.

Admission is decided before any provider upload or generation. The controller is
network-free and delegates the atomic cooldown-plus-reserve check to a backend.
Transient counters and cooldowns are deliberately excluded from fingerprints:
only the static, versioned policy identity is stable identity.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from app.core.enums import AdmissionPriority
from app.refinement.policy import ADMISSION_POLICY_VERSION, AdmissionPolicy

ADMISSION_KEY_PREFIX = "clipfactory:gemini:admission:"
ADMISSION_COOLDOWN_KEY = f"{ADMISSION_KEY_PREFIX}cooldown"

ADMISSION_LUA = """
local cooldown_ttl = redis.call('pttl', KEYS[1])
if cooldown_ttl > 0 then
    local used = tonumber(redis.call('get', KEYS[2]) or '0')
    return {0, used, cooldown_ttl}
end
local total = tonumber(ARGV[1])
local critical_reserve = tonumber(ARGV[2])
local high_reserve = tonumber(ARGV[3])
local low_enabled = ARGV[4] == '1'
local priority = ARGV[5]
local window_ttl = tonumber(ARGV[6])
local ceiling = 0
if priority == 'CRITICAL' then
    ceiling = total
elseif priority == 'HIGH' then
    ceiling = total - critical_reserve
elseif priority == 'MEDIUM' then
    ceiling = total - critical_reserve - high_reserve
elseif priority == 'LOW' and low_enabled then
    ceiling = total - critical_reserve - high_reserve
end
local used = tonumber(redis.call('get', KEYS[2]) or '0')
if used < ceiling then
    used = redis.call('incr', KEYS[2])
    redis.call('expire', KEYS[2], window_ttl)
    return {1, used, 0}
end
return {0, used, 0}
""".strip()


class AdmissionDeniedReason(str, Enum):
    """Why an admission attempt was admitted or denied."""

    ADMITTED = "ADMITTED"
    PRIORITY = "PRIORITY"
    COOLDOWN = "COOLDOWN"
    POLICY = "POLICY"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class AdmissionDecision:
    """The outcome of one admission attempt."""

    admitted: bool
    priority: AdmissionPriority
    reason: AdmissionDeniedReason
    remaining: int
    retry_after_seconds: float | None = None


class AdmissionBackend(Protocol):
    """The small, clock-owning protocol the controller depends on.

    ``admit`` must apply the cooldown and reserve arithmetic atomically and
    return ``(admitted, used_after_attempt, cooldown_remaining_seconds)``.
    """

    def admit(
        self, window_index: int, priority: AdmissionPriority, policy: AdmissionPolicy
    ) -> tuple[bool, int, float | None]: ...

    def set_cooldown(self, seconds: float) -> None: ...

    def cooldown_remaining(self) -> float: ...


def _priority_ceiling(priority: AdmissionPriority, policy: AdmissionPolicy) -> int:
    if priority is AdmissionPriority.CRITICAL:
        return policy.total_calls
    if priority is AdmissionPriority.HIGH:
        return policy.total_calls - policy.critical_reserve
    if priority is AdmissionPriority.MEDIUM:
        return policy.total_calls - policy.critical_reserve - policy.high_reserve
    if priority is AdmissionPriority.LOW and policy.low_enabled:
        return policy.total_calls - policy.critical_reserve - policy.high_reserve
    return 0


def _as_int(value: object) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


class InMemoryAdmissionBackend:
    """Deterministic in-process backend that mirrors the Lua arithmetic exactly."""

    def __init__(self, *, now: Callable[[], float] = time.time) -> None:
        self._now = now
        self._counters: dict[int, int] = {}
        self._cooldown_until = 0.0

    def admit(
        self, window_index: int, priority: AdmissionPriority, policy: AdmissionPolicy
    ) -> tuple[bool, int, float | None]:
        used = self._counters.get(window_index, 0)
        cooldown = self.cooldown_remaining()
        if cooldown > 0:
            return (False, used, cooldown)
        ceiling = _priority_ceiling(priority, policy)
        if used < ceiling:
            used += 1
            self._counters[window_index] = used
            return (True, used, None)
        return (False, used, None)

    def set_cooldown(self, seconds: float) -> None:
        self._cooldown_until = self._now() + max(0.0, seconds)

    def cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - self._now())


class RedisAdmissionBackend:
    """Production backend: one atomic Lua script per admission attempt."""

    def __init__(self, redis: object) -> None:
        self._redis = redis

    def admit(
        self, window_index: int, priority: AdmissionPriority, policy: AdmissionPolicy
    ) -> tuple[bool, int, float | None]:
        result = self._redis.eval(  # type: ignore[attr-defined]
            ADMISSION_LUA,
            2,
            ADMISSION_COOLDOWN_KEY,
            self._window_key(window_index),
            policy.total_calls,
            policy.critical_reserve,
            policy.high_reserve,
            1 if policy.low_enabled else 0,
            priority.value,
            max(1, math.ceil(policy.window_seconds)),
        )
        admitted = _as_int(result[0]) == 1
        used = _as_int(result[1])
        cooldown_ms = _as_int(result[2])
        cooldown = cooldown_ms / 1000.0 if cooldown_ms > 0 else None
        return (admitted, used, cooldown)

    def set_cooldown(self, seconds: float) -> None:
        if seconds <= 0:
            self._redis.delete(ADMISSION_COOLDOWN_KEY)  # type: ignore[attr-defined]
            return
        milliseconds = max(1, math.ceil(seconds * 1000))
        self._redis.set(  # type: ignore[attr-defined]
            ADMISSION_COOLDOWN_KEY, "1", px=milliseconds
        )

    def cooldown_remaining(self) -> float:
        pttl = self._redis.pttl(ADMISSION_COOLDOWN_KEY)  # type: ignore[attr-defined]
        pttl_value = _as_int(pttl)
        if pttl_value <= 0:
            return 0.0
        return pttl_value / 1000.0

    @staticmethod
    def _window_key(window_index: int) -> str:
        return f"{ADMISSION_KEY_PREFIX}{window_index}"


class GeminiAdmissionController:
    """Network-free admission gate; backend unavailability fails closed."""

    def __init__(
        self,
        *,
        backend: AdmissionBackend,
        policy: AdmissionPolicy,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._backend = backend
        self._policy = policy
        self._now = now

    def acquire(self, priority: AdmissionPriority) -> AdmissionDecision:
        window_index = int(self._now() / self._policy.window_seconds)
        try:
            admitted, used, cooldown_remaining = self._backend.admit(
                window_index, priority, self._policy
            )
        except Exception:
            return AdmissionDecision(
                admitted=False,
                priority=priority,
                reason=AdmissionDeniedReason.UNAVAILABLE,
                remaining=0,
            )

        remaining = max(0, _priority_ceiling(priority, self._policy) - used)
        if cooldown_remaining is not None and cooldown_remaining > 0:
            return AdmissionDecision(
                admitted=False,
                priority=priority,
                reason=AdmissionDeniedReason.COOLDOWN,
                remaining=remaining,
                retry_after_seconds=cooldown_remaining,
            )
        if priority is AdmissionPriority.AVOID or (
            priority is AdmissionPriority.LOW and not self._policy.low_enabled
        ):
            return AdmissionDecision(
                admitted=False,
                priority=priority,
                reason=AdmissionDeniedReason.POLICY,
                remaining=remaining,
            )
        if admitted:
            return AdmissionDecision(
                admitted=True,
                priority=priority,
                reason=AdmissionDeniedReason.ADMITTED,
                remaining=remaining,
            )
        return AdmissionDecision(
            admitted=False,
            priority=priority,
            reason=AdmissionDeniedReason.PRIORITY,
            remaining=remaining,
        )

    def record_rate_limit(self, retry_after_seconds: float | None = None) -> None:
        base = retry_after_seconds or self._policy.provider_cooldown_seconds
        self._set_cooldown_safely(min(base, self._policy.max_retry_after_seconds))

    def record_exhaustion(self) -> None:
        self._set_cooldown_safely(self._policy.provider_cooldown_seconds)

    def runtime_identity(self) -> dict[str, object]:
        """Static policy identity for fingerprints; never transient counters."""

        return {
            "admission_policy_version": ADMISSION_POLICY_VERSION,
            "policy": self._policy.as_dict(),
        }

    def _set_cooldown_safely(self, seconds: float) -> None:
        try:
            self._backend.set_cooldown(max(0.0, seconds))
        except Exception:
            return
