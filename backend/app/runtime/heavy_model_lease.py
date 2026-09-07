"""Redis-backed cross-process lease that serializes heavy-model usage.

Worker and CLI processes coordinate through a single key so Whisper and Ollama
can never be resident concurrently. Release is an atomic compare-and-delete so a
lease is never deleted by a token that does not own it.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Callable
from typing import Literal

from app.runtime.memory import MemorySnapshot, capture_memory

_LEASE_KEY = "clipfactory:heavy-model"

_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""

_RENEW_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""


class HeavyModelLeaseBusy(RuntimeError):
    """Another process owns the heavy-model lease; retryable by the caller."""

    retryable = True


class HeavyModelLease:
    """A bounded, renewing, ownership-checked lease for one heavy model slot."""

    def __init__(
        self,
        *,
        redis: object,
        ttl_seconds: float,
        renewal_interval_seconds: float,
        acquisition_timeout_seconds: float,
        purpose: str = "",
        owner_pid: int | None = None,
        token: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_acquire: Callable[[float, str], None] | None = None,
        on_release: Callable[[], None] | None = None,
    ) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds
        self._renewal_interval_seconds = renewal_interval_seconds
        self._acquisition_timeout_seconds = acquisition_timeout_seconds
        self._purpose = purpose
        self._owner_pid = owner_pid or os.getpid()
        self._token = token or uuid.uuid4().hex
        self._monotonic = monotonic
        self._sleep = sleep
        self._on_acquire = on_acquire
        self._on_release = on_release
        self._acquired = False
        self._renewer: threading.Thread | None = None
        self._retained = False

    @property
    def token(self) -> str:
        return self._token

    @property
    def purpose(self) -> str:
        return self._purpose

    @property
    def owner_pid(self) -> int:
        return self._owner_pid

    @property
    def acquired(self) -> bool:
        return self._acquired

    def __enter__(self) -> "HeavyModelLease":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        self.release()
        return False

    def acquire(self) -> "HeavyModelLease":
        """Acquire the lease within a bounded deadline or raise HeavyModelLeaseBusy."""

        started = self._monotonic()
        deadline = started + self._acquisition_timeout_seconds
        while True:
            if self._redis.set(  # type: ignore[attr-defined]
                _LEASE_KEY, self._token, nx=True, px=int(self._ttl_seconds * 1000)
            ):
                self._acquired = True
                if self._on_acquire is not None:
                    self._on_acquire(self._monotonic() - started, self._token)
                self._start_renewer()
                return self
            if self._monotonic() >= deadline:
                raise HeavyModelLeaseBusy(
                    "heavy-model lease is held by another process; retry later"
                )
            self._sleep(min(0.1, self._acquisition_timeout_seconds))

    def renew(self) -> None:
        """Extend the TTL only while this token still owns the lease."""

        if not self._acquired:
            return
        self._redis.eval(  # type: ignore[attr-defined]
            _RENEW_LUA, 1, _LEASE_KEY, self._token, int(self._ttl_seconds * 1000)
        )

    def release(self) -> None:
        """Atomically delete the lease only when this token still owns it.

        A retained lease is not deleted, so another heavy model cannot start
        until the TTL expires or an operator clears the unsafe unload state.
        """

        self._acquired = False
        if self._renewer is not None:
            self._renewer.join(timeout=0.5)
            self._renewer = None
        if self._retained:
            return
        self._redis.eval(  # type: ignore[attr-defined]
            _RELEASE_LUA, 1, _LEASE_KEY, self._token
        )
        if self._on_release is not None:
            self._on_release()

    def retain(self) -> None:
        """Hold the lease across release so another heavy model is blocked."""

        self._retained = True

    def _start_renewer(self) -> None:
        if self._renewer is not None:
            return

        def loop() -> None:
            while True:
                time.sleep(self._renewal_interval_seconds)
                if not self._acquired:
                    return
                self.renew()

        self._renewer = threading.Thread(target=loop, name="heavy-model-lease-renewer", daemon=True)
        self._renewer.start()


class HeavyModelLeaseFactory:
    """Builds configured leases and records structured acquire/release events."""

    def __init__(
        self,
        *,
        redis: object,
        ttl_seconds: float,
        renewal_interval_seconds: float,
        acquisition_timeout_seconds: float,
        snapshotter: Callable[[], MemorySnapshot] = capture_memory,
    ) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds
        self._renewal_interval_seconds = renewal_interval_seconds
        self._acquisition_timeout_seconds = acquisition_timeout_seconds
        self._snapshotter = snapshotter
        self.events: list[dict[str, object]] = []

    def acquire(self, *, purpose: str) -> HeavyModelLease:
        """Return a configured, not-yet-acquired lease for the caller to hold."""

        owner_pid = os.getpid()

        def record_acquired(wait_seconds: float, token: str) -> None:
            self.events.append(
                {
                    "event": "heavy_model_acquired",
                    "purpose": purpose,
                    "owner_pid": owner_pid,
                    "token": token,
                    "wait_seconds": round(wait_seconds, 3),
                    "effective_capacity": self._snapshotter().effective_capacity,
                }
            )

        def record_released() -> None:
            self.events.append(
                {
                    "event": "heavy_model_released",
                    "purpose": purpose,
                    "owner_pid": owner_pid,
                    "effective_capacity": self._snapshotter().effective_capacity,
                }
            )

        return HeavyModelLease(
            redis=self._redis,
            ttl_seconds=self._ttl_seconds,
            renewal_interval_seconds=self._renewal_interval_seconds,
            acquisition_timeout_seconds=self._acquisition_timeout_seconds,
            purpose=purpose,
            owner_pid=owner_pid,
            on_acquire=record_acquired,
            on_release=record_released,
        )


class NoopHeavyModelLease:
    """Acquires instantly, never contends; used by tests and disabled providers."""

    def __init__(self, factory: "NoopHeavyModelLeaseFactory", purpose: str) -> None:
        self._factory = factory
        self._purpose = purpose
        self.token = "noop"
        self.owner_pid = os.getpid()
        self.acquired = False
        self._retained = False

    def __enter__(self) -> "NoopHeavyModelLease":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        self.release()
        return False

    def acquire(self) -> "NoopHeavyModelLease":
        self.acquired = True
        self._factory._record("heavy_model_acquired", self._purpose)
        return self

    def renew(self) -> None:
        return None

    def retain(self) -> None:
        self._retained = True

    def release(self) -> None:
        self.acquired = False
        if self._retained:
            return
        self._factory._record("heavy_model_released", self._purpose)


class NoopHeavyModelLeaseFactory:
    """In-memory lease factory that never contends; the safe default."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def acquire(self, *, purpose: str) -> NoopHeavyModelLease:
        return NoopHeavyModelLease(self, purpose)

    def _record(self, event: str, purpose: str) -> None:
        self.events.append(
            {
                "event": event,
                "purpose": purpose,
                "owner_pid": os.getpid(),
                "wait_seconds": 0.0,
            }
        )
