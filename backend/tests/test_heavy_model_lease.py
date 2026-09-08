from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.core.settings import Settings
from app.runtime.heavy_model_lease import (
    HeavyModelLease,
    HeavyModelLeaseBusy,
    HeavyModelLeaseFactory,
    HeavyModelUnsafe,
)

_LEASE_KEY = "clipfactory:heavy-model"
_UNSAFE_KEY = "clipfactory:heavy-model:unsafe"


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class FakeRedis:
    def __init__(self, now: FakeClock) -> None:
        self._now = now
        self.data: dict[str, str] = {}
        self.expires_at: dict[str, float] = {}
        self.eval_scripts: list[str] = []

    def set(self, name: str, value: str, *, nx: bool = False, px: int | None = None) -> bool | None:
        if self.get(name) is not None:
            return None
        self.data[name] = value
        self.expires_at[name] = self._now() + (px / 1000 if px else float("inf"))
        return True

    def get(self, name: str) -> str | None:
        if name in self.data and self._now() > self.expires_at.get(name, float("inf")):
            del self.data[name]
            del self.expires_at[name]
        return self.data.get(name)

    def expire(self, name: str, ttl_seconds: float) -> bool:
        if name not in self.data:
            return False
        self.expires_at[name] = self._now() + ttl_seconds
        return True

    def delete(self, name: str) -> int:
        if name in self.data:
            del self.data[name]
            del self.expires_at[name]
            return 1
        return 0

    def eval(self, script: str, numkeys: int, *args: object) -> object:
        self.eval_scripts.append(script)
        if "pexpire" in script:
            key = str(args[0])
            token = str(args[1])
            if self.get(key) != token:
                return 0
            ttl_ms = int(str(args[2]))
            self.expires_at[key] = self._now() + ttl_ms / 1000
            return 1
        if "UNSAFE" in script:
            unsafe_key = str(args[0])
            lease_key = str(args[1])
            token = str(args[2])
            ttl_ms = int(str(args[3]))
            if self.get(unsafe_key) is not None:
                return "UNSAFE"
            if self.get(lease_key) is not None:
                return "BUSY"
            self.set(lease_key, token, px=ttl_ms)
            return "ACQUIRED"
        if "exists" in script:
            unsafe_key = str(args[0])
            lease_key = str(args[1])
            if self.get(unsafe_key) is None:
                return 0
            self.delete(lease_key)
            self.delete(unsafe_key)
            return 1
        key = str(args[0])
        token = str(args[1])
        if self.get(key) != token:
            return 0
        del self.data[key]
        del self.expires_at[key]
        return 1


def _lease(
    redis: FakeRedis,
    now: FakeClock,
    *,
    ttl: float = 300,
    renewal: float = 60,
    timeout: float = 5,
    token: str | None = None,
) -> HeavyModelLease:
    def advance(seconds: float) -> None:
        now.now += seconds

    return HeavyModelLease(
        redis=redis,
        ttl_seconds=ttl,
        renewal_interval_seconds=renewal,
        acquisition_timeout_seconds=timeout,
        purpose="test",
        monotonic=now,
        sleep=advance,
        token=token,
    )


def test_acquire_success_sets_unique_owner_token() -> None:
    now = FakeClock()
    redis = FakeRedis(now)
    lease = _lease(redis, now)

    lease.acquire()

    assert lease.token
    assert redis.get(_LEASE_KEY) == lease.token


def test_unique_owner_tokens_per_lease() -> None:
    now = FakeClock()
    redis = FakeRedis(now)

    assert _lease(redis, now).token != _lease(redis, now).token


def test_acquire_busy_raises_retryable_error() -> None:
    now = FakeClock()
    redis = FakeRedis(now)
    redis.set(_LEASE_KEY, "other-token", nx=True, px=300000)

    lease = _lease(redis, now, timeout=1)
    with pytest.raises(HeavyModelLeaseBusy):
        lease.acquire()


def test_renewal_extends_ttl_only_when_still_owner() -> None:
    now = FakeClock()
    redis = FakeRedis(now)
    lease = _lease(redis, now, ttl=300)
    lease.acquire()

    now.now += 100
    lease.renew()

    assert redis.get(_LEASE_KEY) == lease.token
    assert redis.expires_at[_LEASE_KEY] == pytest.approx(now.now + 300)


def test_release_never_deletes_a_lease_owned_by_another_token() -> None:
    now = FakeClock()
    redis = FakeRedis(now)
    owner = _lease(redis, now, token="owner-a")
    owner.acquire()
    intruder = _lease(redis, now, token="intruder")

    intruder.release()

    assert redis.get(_LEASE_KEY) == "owner-a"

    owner.release()

    assert redis.get(_LEASE_KEY) is None


def test_expired_lease_can_be_reacquired() -> None:
    now = FakeClock()
    redis = FakeRedis(now)
    first = _lease(redis, now, ttl=10)
    first.acquire()

    now.now += 15

    second = _lease(redis, now)
    second.acquire()

    assert redis.get(_LEASE_KEY) == second.token


def test_release_runs_after_exception_in_context() -> None:
    now = FakeClock()
    redis = FakeRedis(now)
    lease = _lease(redis, now)

    with pytest.raises(RuntimeError, match="boom"):
        with lease:
            raise RuntimeError("boom")

    assert redis.get(_LEASE_KEY) is None


def test_lease_settings_require_ttl_above_two_renewal_intervals() -> None:
    settings = Settings(_env_file=None)

    assert settings.heavy_model_lease_ttl_seconds > 0
    assert settings.heavy_model_lease_renewal_interval_seconds > 0
    assert settings.heavy_model_lease_acquisition_timeout_seconds > 0
    assert (
        settings.heavy_model_lease_ttl_seconds
        > 2 * settings.heavy_model_lease_renewal_interval_seconds
    )

    with pytest.raises(ValidationError, match="two renewal intervals"):
        Settings(
            _env_file=None,
            heavy_model_lease_ttl_seconds=10,
            heavy_model_lease_renewal_interval_seconds=60,
        )


def test_lease_context_marks_acquired_and_released() -> None:
    now = FakeClock()
    redis = FakeRedis(now)
    lease = _lease(redis, now)

    with lease as active:
        assert active.token
        assert redis.get(_LEASE_KEY) == active.token

    assert redis.get(_LEASE_KEY) is None


def test_renewal_ownership_loss_is_detected() -> None:
    now = FakeClock()
    redis = FakeRedis(now)
    lease = _lease(redis, now)
    lease.acquire()
    redis.data[_LEASE_KEY] = "intruder"
    redis.expires_at[_LEASE_KEY] = now.now + 300

    assert lease.renew() is False
    assert lease.ownership_lost is True
    assert lease.renew() is False


def test_renewal_redis_exception_marks_ownership_lost() -> None:
    class ExplodingRedis(FakeRedis):
        def eval(self, script: str, numkeys: int, *args: object) -> object:
            if "pexpire" in script:
                raise ConnectionError("redis unavailable")
            return super().eval(script, numkeys, *args)

    now = FakeClock()
    redis = ExplodingRedis(now)
    lease = _lease(redis, now)
    lease.acquire()

    assert lease.renew() is False
    assert lease.ownership_lost is True


def test_retained_lease_keeps_renewing_after_release() -> None:
    """An unsafe unload block survives the normal TTL; only operator recovery ends it."""

    now = FakeClock()
    redis = FakeRedis(now)
    lease = _lease(redis, now)
    lease.acquire()
    lease.retain()

    lease.release()

    assert redis.get(_LEASE_KEY) == lease.token
    assert lease.renewing is True

    lease.clear_retained()

    assert redis.get(_LEASE_KEY) is None
    assert lease.renewing is False


def test_factory_retain_persists_unsafe_marker_without_ttl() -> None:
    """Retaining records a persistent unsafe marker that survives lease expiry."""

    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    lease = factory.acquire(purpose="ollama")
    lease.acquire()
    lease.retain()
    lease.release()

    assert factory.unsafe_recorded() is True
    assert factory.unsafe_reason() == "lease retained for ollama"

    now.now += 10_000

    assert factory.unsafe_recorded() is True


def test_unsafe_marker_blocks_new_acquisition_until_recovered() -> None:
    """No heavy-model work may start while the unsafe marker is present."""

    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    factory.mark_unsafe(reason="unload failed")

    with pytest.raises(HeavyModelUnsafe):
        with factory.acquire(purpose="whisper"):
            pass


def test_unsafe_marker_persists_across_factory_and_process_boundary() -> None:
    """A fresh factory (new worker/CLI process) still sees the persisted marker."""

    now = FakeClock()
    redis = FakeRedis(now)
    first = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    first.mark_unsafe(reason="model still resident after unload timeout")

    second = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )

    assert second.unsafe_recorded() is True
    assert second.unsafe_reason() == "model still resident after unload timeout"
    with pytest.raises(HeavyModelUnsafe):
        second.acquire(purpose="whisper").acquire()


def test_recovery_only_clears_marker_and_lease_after_confirming_not_resident() -> None:
    """recover() clears the persistent marker and the stale lease key."""

    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    factory.mark_unsafe(reason="lease retained for ollama")
    redis.set(_LEASE_KEY, "stale-token", nx=True, px=300000)

    factory.recover()

    assert factory.unsafe_recorded() is False
    assert redis.get(_LEASE_KEY) is None


def test_ownership_loss_callback_fires_and_persists_unsafe() -> None:
    """Lease loss notifies the caller and records persistent unsafe state."""

    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    events: list[str] = []
    lease = factory.acquire(purpose="whisper", on_ownership_lost=lambda: events.append("lost"))
    lease.acquire()
    redis.data[_LEASE_KEY] = "intruder"
    redis.expires_at[_LEASE_KEY] = now.now + 300

    assert lease.renew() is False
    assert lease.ownership_lost is True
    assert events == ["lost"]
    assert factory.unsafe_recorded() is True


def test_unsafe_marker_blocks_acquisition_and_never_writes_lease() -> None:
    """A pre-existing unsafe marker blocks acquisition without touching the lease key."""

    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    factory.mark_unsafe(reason="unload failed")

    with pytest.raises(HeavyModelUnsafe):
        factory.acquire(purpose="whisper").acquire()

    assert redis.get(_LEASE_KEY) is None


def test_unsafe_marker_appearing_during_acquisition_retry_blocks_waiter() -> None:
    """An unsafe marker recorded while another waiter retries must stop that waiter."""

    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    redis.set(_LEASE_KEY, "other-token", nx=True, px=300000)
    real_eval = redis.eval

    def mark_unsafe_before_next_attempt(script: str, numkeys: int, *args: object) -> object:
        redis.set(_UNSAFE_KEY, '{"reason": "unload failed"}')
        return real_eval(script, numkeys, *args)

    redis.eval = mark_unsafe_before_next_attempt  # type: ignore[method-assign]
    lease = factory.acquire(purpose="whisper")

    with pytest.raises(HeavyModelUnsafe):
        lease.acquire()

    assert redis.get(_LEASE_KEY) == "other-token"
    assert factory.unsafe_recorded() is True


def test_recovery_clears_stale_lease_and_unsafe_marker_in_one_atomic_operation() -> None:
    """Recovery removes stale lease state and the unsafe marker as a single Redis script."""

    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    factory.mark_unsafe(reason="lease retained for ollama")
    redis.set(_LEASE_KEY, "stale-token", nx=True, px=300000)

    factory.recover()

    assert factory.unsafe_recorded() is False
    assert redis.get(_LEASE_KEY) is None
    assert len(redis.eval_scripts) == 1


def test_acquisition_and_recovery_are_single_atomic_redis_operations() -> None:
    """Acquire observes unsafe inside its own script; recovery clears both keys in one script.

    This makes it impossible for an acquisition to slip between the unsafe-marker
    clear and the lease delete that the old recovery performed as two operations.
    """

    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    factory.mark_unsafe(reason="lease retained for ollama")
    redis.set(_LEASE_KEY, "stale-token", nx=True, px=300000)

    waiter = factory.acquire(purpose="whisper")
    with pytest.raises(HeavyModelUnsafe):
        waiter.acquire()

    factory.recover()

    assert len(redis.eval_scripts) == 2
    assert "UNSAFE" in redis.eval_scripts[0]
    assert "exists" in redis.eval_scripts[1]


def test_recovery_without_unsafe_marker_never_deletes_a_valid_new_lease() -> None:
    """A recovery racing with a fresh acquisition must not delete the new lease."""

    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    holder = factory.acquire(purpose="whisper")
    holder.acquire()

    factory.recover()

    assert redis.get(_LEASE_KEY) == holder.token


def test_acquisition_works_normally_after_successful_recovery() -> None:
    """Once recovery completes, a fresh acquisition proceeds normally."""

    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    factory.mark_unsafe(reason="lease retained for ollama")
    redis.set(_LEASE_KEY, "stale-token", nx=True, px=300000)

    factory.recover()

    lease = factory.acquire(purpose="whisper")
    lease.acquire()

    assert redis.get(_LEASE_KEY) == lease.token
