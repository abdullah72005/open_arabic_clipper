"""Redis-backed cross-process lease that serializes heavy-model usage.

Worker and CLI processes coordinate through a single key so Whisper and Ollama
can never be resident concurrently. Acquisition and operator recovery are each
one Redis-side Lua script, so no gap can occur between the unsafe-marker check
and lease acquisition or between clearing the marker and deleting a stale lease.
Release is an atomic compare-and-delete so a lease is never deleted by a token
that does not own it.

Unsafe model residency is a separate, persistent Redis marker. It is written
when unload fails or lease ownership is lost while a heavy model may still be
resident, and it survives worker restart, CLI exit, and the lease TTL. No new
heavy-model work may acquire the lease while the marker is present; an operator
must confirm the model is no longer resident and clear the marker explicitly.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from typing import Literal

from app.runtime.memory import MemorySnapshot, capture_memory

_LEASE_KEY = "clipfactory:heavy-model"
_UNSAFE_KEY = "clipfactory:heavy-model:unsafe"

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

_ACQUIRE_LUA = """
if redis.call('exists', KEYS[1]) == 1 then
    return 'UNSAFE'
end
if redis.call('exists', KEYS[2]) == 1 then
    return 'BUSY'
end
redis.call('set', KEYS[2], ARGV[1], 'PX', ARGV[2])
return 'ACQUIRED'
"""

_RECOVER_LUA = """
if redis.call('exists', KEYS[1]) == 0 then
    return 0
end
redis.call('del', KEYS[2])
redis.call('del', KEYS[1])
return 1
"""


class HeavyModelLeaseBusy(RuntimeError):
    """Another process owns the heavy-model lease; retryable by the caller."""

    retryable = True


class HeavyModelUnsafe(RuntimeError):
    """Unsafe heavy-model residency is recorded; heavy work is blocked until cleared."""

    retryable = False


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
        renewal_sleep: Callable[[float], None] = time.sleep,
        on_acquire: Callable[[float, str], None] | None = None,
        on_release: Callable[[], None] | None = None,
        on_ownership_lost: Callable[[], None] | None = None,
        on_unsafe: Callable[[str], None] | None = None,
        on_unsafe_clear: Callable[[], None] | None = None,
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
        self._renewal_sleep = renewal_sleep
        self._on_acquire = on_acquire
        self._on_release = on_release
        self._on_ownership_lost = on_ownership_lost
        self._on_unsafe = on_unsafe
        self._on_unsafe_clear = on_unsafe_clear
        self._acquired = False
        self._renewer: threading.Thread | None = None
        self._retained = False
        self._ownership_lost = False

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

    @property
    def ownership_lost(self) -> bool:
        """True once a renewal failed or the ownership check returned zero."""

        return self._ownership_lost

    @property
    def renewing(self) -> bool:
        """True while the background renewer keeps the lease alive."""

        return self._renewer is not None and self._renewer.is_alive()

    def __enter__(self) -> "HeavyModelLease":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        self.release()
        return False

    def acquire(self) -> "HeavyModelLease":
        """Acquire the lease within a bounded deadline or raise HeavyModelLeaseBusy.

        Acquisition is one Redis-side atomic operation that first checks the
        persistent unsafe marker: an active marker returns UNSAFE and never
        acquires. The retry loop re-runs the same atomic operation, so an unsafe
        marker written while this process waits still blocks it.
        """

        started = self._monotonic()
        deadline = started + self._acquisition_timeout_seconds
        while True:
            result = self._redis.eval(  # type: ignore[attr-defined]
                _ACQUIRE_LUA,
                2,
                _UNSAFE_KEY,
                _LEASE_KEY,
                self._token,
                int(self._ttl_seconds * 1000),
            )
            if isinstance(result, bytes):
                result = result.decode("utf-8", errors="replace")
            if result == "ACQUIRED":
                self._acquired = True
                if self._on_acquire is not None:
                    self._on_acquire(self._monotonic() - started, self._token)
                self._start_renewer()
                return self
            if result == "UNSAFE":
                raise HeavyModelUnsafe(
                    "unsafe model residency is recorded; run the recovery command"
                )
            if self._monotonic() >= deadline:
                raise HeavyModelLeaseBusy(
                    "heavy-model lease is held by another process; retry later"
                )
            self._sleep(min(0.1, self._acquisition_timeout_seconds))

    def renew(self) -> bool:
        """Extend the TTL and report ownership.

        A Redis exception or a Lua return value of zero is lease loss, never
        success: the caller must stop heavy work and fail closed.
        """

        if self._ownership_lost:
            return False
        if not (self._acquired or self._retained):
            return True
        try:
            renewed = self._redis.eval(  # type: ignore[attr-defined]
                _RENEW_LUA, 1, _LEASE_KEY, self._token, int(self._ttl_seconds * 1000)
            )
        except Exception:
            renewed = 0
        if isinstance(renewed, bytes):
            renewed = renewed.decode("utf-8", errors="replace")
        try:
            renewed = int(renewed)
        except (TypeError, ValueError):
            renewed = 0
        if not renewed:
            self._mark_ownership_lost()
        return bool(renewed)

    def release(self) -> None:
        """Atomically delete the lease only when this token still owns it.

        A retained lease is an unsafe-residency block: it keeps renewing so the
        key never expires and another heavy model cannot start until an operator
        explicitly clears the block with ``clear_retained``.
        """

        if self._retained:
            self._acquired = False
            if self._renewer is None:
                self._start_renewer()
            return
        self._acquired = False
        if self._renewer is not None:
            self._renewer.join(timeout=0.5)
            self._renewer = None
        self._redis.eval(  # type: ignore[attr-defined]
            _RELEASE_LUA, 1, _LEASE_KEY, self._token
        )
        if self._on_release is not None:
            self._on_release()

    def retain(self) -> None:
        """Hold the lease across release so another heavy model is blocked.

        Retaining also records persistent unsafe state so the block survives a
        worker restart, CLI exit, or lease TTL expiry until an operator clears it.
        """

        self._retained = True
        if self._renewer is None:
            self._start_renewer()
        if self._on_unsafe is not None:
            self._on_unsafe(self._purpose)

    def clear_retained(self) -> None:
        """Operator recovery: end the unsafe block and release the lease.

        The caller must confirm the model is no longer resident before invoking
        this; clearing the retained block also clears the persistent unsafe marker.
        """

        self._retained = False
        self._acquired = False
        if self._renewer is not None:
            self._renewer.join(timeout=0.5)
            self._renewer = None
        self._redis.eval(  # type: ignore[attr-defined]
            _RELEASE_LUA, 1, _LEASE_KEY, self._token
        )
        if self._on_unsafe_clear is not None:
            self._on_unsafe_clear()
        if self._on_release is not None:
            self._on_release()

    def _mark_ownership_lost(self) -> None:
        self._ownership_lost = True
        if self._on_ownership_lost is not None:
            self._on_ownership_lost()
        if self._on_unsafe is not None:
            self._on_unsafe(self._purpose)

    def _start_renewer(self) -> None:
        if self._renewer is not None:
            return

        def loop() -> None:
            while True:
                self._renewal_sleep(self._renewal_interval_seconds)
                if not (self._acquired or self._retained):
                    return
                if not self.renew():
                    return

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

    def mark_unsafe(self, *, reason: str, owner_pid: int | None = None) -> None:
        """Persist unsafe heavy-model residency with no TTL.

        The marker is intentionally written without an expiry so it survives
        worker restart, CLI exit, and the lease TTL. Only an operator recovery
        that first confirms the model is no longer resident clears it.
        """

        self._redis.set(  # type: ignore[attr-defined]
            _UNSAFE_KEY,
            json.dumps(
                {
                    "reason": reason,
                    "owner_pid": owner_pid or os.getpid(),
                    "recorded_at": time.time(),
                },
                sort_keys=True,
            ),
        )

    def unsafe_recorded(self) -> bool:
        """Return True when an unsafe marker is currently persisted."""

        return bool(self._redis.get(_UNSAFE_KEY))  # type: ignore[attr-defined]

    def unsafe_reason(self) -> str | None:
        """Return the persisted unsafe reason, or None when no marker exists."""

        try:
            raw = self._redis.get(_UNSAFE_KEY)  # type: ignore[attr-defined]
        except Exception:
            return None
        if not raw:
            return None
        try:
            payload = json.loads(str(raw))
        except ValueError:
            return str(raw)
        if isinstance(payload, dict):
            return str(payload.get("reason", "unsafe model residency recorded"))
        return str(raw)

    def clear_unsafe(self) -> None:
        """Clear the persistent unsafe marker; caller must confirm not-resident first."""

        self._redis.delete(_UNSAFE_KEY)  # type: ignore[attr-defined]

    def recover(self) -> None:
        """Operator recovery: clear unsafe state and any stale lease key atomically.

        One Redis-side script clears the stale lease and the unsafe marker with
        no gap in which a new acquisition can occur. The script refuses to run
        when no unsafe marker exists, so it never deletes a valid newly acquired
        lease. Callers must confirm the model is no longer resident first.
        """

        self._redis.eval(  # type: ignore[attr-defined]
            _RECOVER_LUA, 2, _UNSAFE_KEY, _LEASE_KEY
        )

    def acquire(
        self, *, purpose: str, on_ownership_lost: Callable[[], None] | None = None
    ) -> HeavyModelLease:
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
            on_ownership_lost=on_ownership_lost,
            on_unsafe=self._record_unsafe,
            on_unsafe_clear=self.clear_unsafe,
        )

    def _record_unsafe(self, purpose: str) -> None:
        self.mark_unsafe(reason=f"lease retained for {purpose}")


class NoopHeavyModelLease:
    """Acquires instantly, never contends; used by tests and disabled providers."""

    def __init__(self, factory: "NoopHeavyModelLeaseFactory", purpose: str) -> None:
        self._factory = factory
        self._purpose = purpose
        self.token = "noop"
        self.owner_pid = os.getpid()
        self.acquired = False
        self._retained = False
        self._ownership_lost = False

    def __enter__(self) -> "NoopHeavyModelLease":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        self.release()
        return False

    @property
    def ownership_lost(self) -> bool:
        return self._ownership_lost

    @property
    def renewing(self) -> bool:
        return self.acquired or self._retained

    def acquire(self) -> "NoopHeavyModelLease":
        if self._factory._unsafe:
            raise HeavyModelUnsafe("unsafe model residency is recorded; run the recovery command")
        self.acquired = True
        self._factory._record("heavy_model_acquired", self._purpose)
        return self

    def renew(self) -> bool:
        return not self._ownership_lost

    def retain(self) -> None:
        self._retained = True
        self._factory._unsafe = True

    def release(self) -> None:
        self.acquired = False
        if self._retained:
            return
        self._factory._record("heavy_model_released", self._purpose)

    def clear_retained(self) -> None:
        self._retained = False
        self.acquired = False
        self._factory._unsafe = False
        self._factory._record("heavy_model_released", self._purpose)


class NoopHeavyModelLeaseFactory:
    """In-memory lease factory that never contends; the safe default."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self._unsafe = False

    def acquire(
        self, *, purpose: str, on_ownership_lost: Callable[[], None] | None = None
    ) -> NoopHeavyModelLease:
        return NoopHeavyModelLease(self, purpose)

    def mark_unsafe(self, *, reason: str, owner_pid: int | None = None) -> None:
        self._unsafe = True

    def unsafe_recorded(self) -> bool:
        return self._unsafe

    def unsafe_reason(self) -> str | None:
        return "lease retained for heavy-model work" if self._unsafe else None

    def clear_unsafe(self) -> None:
        self._unsafe = False

    def recover(self) -> None:
        self._unsafe = False

    def _record(self, event: str, purpose: str) -> None:
        self.events.append(
            {
                "event": event,
                "purpose": purpose,
                "owner_pid": os.getpid(),
                "wait_seconds": 0.0,
            }
        )
