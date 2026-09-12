"""Deterministic tests for the Stage 3.5 shared Gemini admission controller."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.enums import AdmissionPriority
from app.refinement.admission import (
    ADMISSION_COOLDOWN_KEY,
    ADMISSION_KEY_PREFIX,
    ADMISSION_LUA,
    AdmissionDeniedReason,
    GeminiAdmissionController,
    InMemoryAdmissionBackend,
    RedisAdmissionBackend,
)
from app.refinement.policy import ADMISSION_POLICY_VERSION, AdmissionPolicy


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class FakeAdmissionRedis:
    """Minimal redis double emulating the admission Lua contract."""

    def __init__(self, now: FakeClock) -> None:
        self._now = now
        self.data: dict[str, str] = {}
        self.expires_at: dict[str, float] = {}
        self.eval_scripts: list[str] = []
        self.eval_args: list[tuple[Any, ...]] = []

    def _purge(self, key: str) -> None:
        deadline = self.expires_at.get(key)
        if deadline is not None and self._now() > deadline:
            self.data.pop(key, None)
            self.expires_at.pop(key, None)

    def get(self, key: str) -> str | None:
        self._purge(key)
        return self.data.get(key)

    def set(self, key: str, value: str, *, px: int | None = None) -> bool:
        self.data[key] = value
        if px is None:
            self.expires_at.pop(key, None)
        else:
            self.expires_at[key] = self._now() + px / 1000
        return True

    def delete(self, key: str) -> int:
        existed = key in self.data
        self.data.pop(key, None)
        self.expires_at.pop(key, None)
        return 1 if existed else 0

    def pttl(self, key: str) -> int:
        if key not in self.data:
            return -2
        if key not in self.expires_at:
            return -1
        remaining = self.expires_at[key] - self._now()
        if remaining <= 0:
            self._purge(key)
            return -2
        return int(remaining * 1000)

    def eval(self, script: str, numkeys: int, *args: object) -> list[int]:
        self.eval_scripts.append(script)
        self.eval_args.append(args)
        keys = [str(arg) for arg in args[:numkeys]]
        argv = [str(arg) for arg in args[numkeys:]]
        cooldown_key, window_key = keys[0], keys[1]
        pttl = self.pttl(cooldown_key)
        used = int(self.get(window_key) or "0")
        if pttl > 0:
            return [0, used, pttl]
        total = int(argv[0])
        critical_reserve = int(argv[1])
        high_reserve = int(argv[2])
        low_enabled = argv[3] == "1"
        priority = argv[4]
        window_ttl = int(argv[5])
        ceiling = 0
        if priority == "CRITICAL":
            ceiling = total
        elif priority == "HIGH":
            ceiling = total - critical_reserve
        elif priority == "MEDIUM":
            ceiling = total - critical_reserve - high_reserve
        elif priority == "LOW" and low_enabled:
            ceiling = total - critical_reserve - high_reserve
        if used < ceiling:
            used += 1
            self.data[window_key] = str(used)
            self.expires_at[window_key] = self._now() + window_ttl
            return [1, used, 0]
        return [0, used, 0]


class BytesAdmissionRedis(FakeAdmissionRedis):
    """redis-like double returning byte replies for EVAL list elements."""

    def eval(self, script: str, numkeys: int, *args: object) -> list[bytes]:
        result = super().eval(script, numkeys, *args)
        return [str(value).encode() for value in result]


class ExplodingBackend:
    def admit(
        self, window_index: int, priority: AdmissionPriority, policy: AdmissionPolicy
    ) -> tuple[bool, int, float | None]:
        raise ConnectionError("redis unavailable")

    def set_cooldown(self, seconds: float) -> None:
        raise ConnectionError("redis unavailable")

    def cooldown_remaining(self) -> float:
        raise ConnectionError("redis unavailable")


def _controller(
    clock: FakeClock,
    policy: AdmissionPolicy,
    backend: InMemoryAdmissionBackend | None = None,
) -> GeminiAdmissionController:
    return GeminiAdmissionController(
        backend=backend or InMemoryAdmissionBackend(now=clock),
        policy=policy,
        now=clock,
    )


def test_critical_admitted_until_total_then_denied() -> None:
    clock = FakeClock()
    policy = AdmissionPolicy(total_calls=10, critical_reserve=3, high_reserve=2)
    controller = _controller(clock, policy)

    for _ in range(10):
        assert controller.acquire(AdmissionPriority.CRITICAL).admitted is True

    decision = controller.acquire(AdmissionPriority.CRITICAL)
    assert decision.admitted is False
    assert decision.reason is AdmissionDeniedReason.PRIORITY


def test_critical_capacity_remains_after_medium_consumes_its_allowance() -> None:
    clock = FakeClock()
    policy = AdmissionPolicy(total_calls=10, critical_reserve=3, high_reserve=2)
    controller = _controller(clock, policy)

    for _ in range(5):
        assert controller.acquire(AdmissionPriority.MEDIUM).admitted is True
    assert controller.acquire(AdmissionPriority.MEDIUM).admitted is False

    for _ in range(3):
        assert controller.acquire(AdmissionPriority.CRITICAL).admitted is True


def test_high_cannot_consume_critical_reserve() -> None:
    clock = FakeClock()
    policy = AdmissionPolicy(total_calls=10, critical_reserve=8, high_reserve=1)
    controller = _controller(clock, policy)

    for _ in range(8):
        assert controller.acquire(AdmissionPriority.CRITICAL).admitted is True

    high = controller.acquire(AdmissionPriority.HIGH)
    assert high.admitted is False
    assert high.reason is AdmissionDeniedReason.PRIORITY
    assert controller.acquire(AdmissionPriority.CRITICAL).admitted is True


def test_low_denied_when_disabled_and_admitted_within_capacity_when_enabled() -> None:
    disabled = AdmissionPolicy(
        total_calls=10, critical_reserve=3, high_reserve=2, low_enabled=False
    )
    controller = _controller(FakeClock(), disabled)
    denied = controller.acquire(AdmissionPriority.LOW)
    assert denied.admitted is False
    assert denied.reason is AdmissionDeniedReason.POLICY

    enabled = AdmissionPolicy(total_calls=10, critical_reserve=3, high_reserve=2, low_enabled=True)
    controller = _controller(FakeClock(), enabled)
    for _ in range(5):
        assert controller.acquire(AdmissionPriority.LOW).admitted is True
    assert controller.acquire(AdmissionPriority.LOW).admitted is False


def test_avoid_is_always_denied_by_policy() -> None:
    controller = _controller(FakeClock(), AdmissionPolicy())

    decision = controller.acquire(AdmissionPriority.AVOID)

    assert decision.admitted is False
    assert decision.reason is AdmissionDeniedReason.POLICY


def test_rate_limit_cooldown_denies_critical_then_expires_with_injected_clock() -> None:
    clock = FakeClock()
    policy = AdmissionPolicy(
        total_calls=10,
        critical_reserve=8,
        high_reserve=1,
        provider_cooldown_seconds=60.0,
        max_retry_after_seconds=3600.0,
    )
    controller = _controller(clock, policy)

    controller.record_rate_limit()

    decision = controller.acquire(AdmissionPriority.CRITICAL)
    assert decision.admitted is False
    assert decision.reason is AdmissionDeniedReason.COOLDOWN
    assert decision.retry_after_seconds == pytest.approx(60.0)

    clock.now += 59.0
    assert controller.acquire(AdmissionPriority.CRITICAL).reason is AdmissionDeniedReason.COOLDOWN

    clock.now += 2.0
    assert controller.acquire(AdmissionPriority.CRITICAL).admitted is True


def test_record_exhaustion_denies_critical_until_cooldown_expires() -> None:
    clock = FakeClock()
    policy = AdmissionPolicy(provider_cooldown_seconds=30.0, max_retry_after_seconds=3600.0)
    controller = _controller(clock, policy)

    controller.record_exhaustion()

    decision = controller.acquire(AdmissionPriority.CRITICAL)
    assert decision.admitted is False
    assert decision.reason is AdmissionDeniedReason.COOLDOWN
    assert decision.retry_after_seconds == pytest.approx(30.0)

    clock.now += 31.0
    assert controller.acquire(AdmissionPriority.CRITICAL).admitted is True


def test_retry_after_is_bounded_by_max_retry_after() -> None:
    policy = AdmissionPolicy(provider_cooldown_seconds=60.0, max_retry_after_seconds=120.0)
    controller = _controller(FakeClock(), policy)

    controller.record_rate_limit(retry_after_seconds=9999.0)

    decision = controller.acquire(AdmissionPriority.CRITICAL)
    assert decision.reason is AdmissionDeniedReason.COOLDOWN
    assert decision.retry_after_seconds == pytest.approx(120.0)


def test_retry_after_defaults_to_provider_cooldown_when_absent() -> None:
    policy = AdmissionPolicy(provider_cooldown_seconds=45.0, max_retry_after_seconds=120.0)
    controller = _controller(FakeClock(), policy)

    controller.record_rate_limit()

    decision = controller.acquire(AdmissionPriority.CRITICAL)
    assert decision.retry_after_seconds == pytest.approx(45.0)


def test_backend_exception_fails_closed_for_acquire_and_record() -> None:
    controller = GeminiAdmissionController(
        backend=ExplodingBackend(),
        policy=AdmissionPolicy(),
        now=FakeClock(),
    )

    decision = controller.acquire(AdmissionPriority.CRITICAL)

    assert decision.admitted is False
    assert decision.reason is AdmissionDeniedReason.UNAVAILABLE
    controller.record_rate_limit()
    controller.record_exhaustion()


def test_runtime_identity_is_static_policy_only() -> None:
    policy = AdmissionPolicy(total_calls=17)
    controller = _controller(FakeClock(), policy)
    controller.acquire(AdmissionPriority.CRITICAL)

    identity = controller.runtime_identity()

    assert identity["admission_policy_version"] == ADMISSION_POLICY_VERSION
    assert identity["policy"] == policy.as_dict()
    assert set(identity) == {"admission_policy_version", "policy"}
    rendered = repr(identity)
    assert "used" not in rendered
    assert "remaining" not in rendered
    assert "window_index" not in rendered


def test_redis_backend_calls_lua_and_maps_integer_list() -> None:
    clock = FakeClock()
    redis = FakeAdmissionRedis(clock)
    backend = RedisAdmissionBackend(redis)
    policy = AdmissionPolicy(total_calls=10, critical_reserve=3, high_reserve=2)

    admitted, used, cooldown = backend.admit(0, AdmissionPriority.CRITICAL, policy)

    assert redis.eval_scripts == [ADMISSION_LUA]
    assert (admitted, used, cooldown) == (True, 1, None)


def test_redis_backend_maps_byte_replies_and_window_keys() -> None:
    clock = FakeClock()
    redis = BytesAdmissionRedis(clock)
    backend = RedisAdmissionBackend(redis)
    policy = AdmissionPolicy(total_calls=10, critical_reserve=3, high_reserve=2)

    admitted, used, cooldown = backend.admit(7, AdmissionPriority.CRITICAL, policy)

    assert (admitted, used, cooldown) == (True, 1, None)
    args = redis.eval_args[-1]
    assert args[0] == ADMISSION_COOLDOWN_KEY
    assert args[1] == f"{ADMISSION_KEY_PREFIX}7"


def test_redis_backend_cooldown_round_trip() -> None:
    clock = FakeClock()
    redis = FakeAdmissionRedis(clock)
    backend = RedisAdmissionBackend(redis)
    policy = AdmissionPolicy()

    backend.set_cooldown(30.0)
    admitted, used, cooldown = backend.admit(0, AdmissionPriority.CRITICAL, policy)

    assert admitted is False
    assert cooldown == pytest.approx(30.0)

    clock.now += 30.5
    assert backend.cooldown_remaining() == 0.0
    assert backend.admit(0, AdmissionPriority.CRITICAL, policy)[0] is True


def test_controller_over_redis_backend_admits_and_denies() -> None:
    clock = FakeClock()
    redis = FakeAdmissionRedis(clock)
    policy = AdmissionPolicy(total_calls=2, critical_reserve=0, high_reserve=0)
    controller = GeminiAdmissionController(
        backend=RedisAdmissionBackend(redis), policy=policy, now=clock
    )

    assert controller.acquire(AdmissionPriority.MEDIUM).admitted is True
    assert controller.acquire(AdmissionPriority.MEDIUM).admitted is True
    assert controller.acquire(AdmissionPriority.MEDIUM).admitted is False
