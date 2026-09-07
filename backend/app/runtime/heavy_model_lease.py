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
        self._acquired = False
        self._renewer: threading.Thread | None = None

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

        deadline = self._monotonic() + self._acquisition_timeout_seconds
        while True:
            if self._redis.set(  # type: ignore[attr-defined]
                _LEASE_KEY, self._token, nx=True, px=int(self._ttl_seconds * 1000)
            ):
                self._acquired = True
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
        """Atomically delete the lease only when this token still owns it."""

        self._acquired = False
        if self._renewer is not None:
            self._renewer.join(timeout=0.5)
            self._renewer = None
        self._redis.eval(  # type: ignore[attr-defined]
            _RELEASE_LUA, 1, _LEASE_KEY, self._token
        )

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
