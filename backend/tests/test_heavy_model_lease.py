import pytest
from pydantic import ValidationError

from app.core.settings import Settings
from app.runtime.heavy_model_lease import HeavyModelLease, HeavyModelLeaseBusy

_LEASE_KEY = "clipfactory:heavy-model"


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

    def eval(self, script: str, numkeys: int, *args: object) -> int:
        self.eval_scripts.append(script)
        key = str(args[0])
        token = str(args[1])
        if self.get(key) != token:
            return 0
        if "pexpire" in script:
            ttl_ms = int(str(args[2]))
            self.expires_at[key] = self._now() + ttl_ms / 1000
            return 1
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
