"""Narrow global render admission for Stage 5.2.

Caps simultaneous render attempts across workers with one PostgreSQL session
advisory lock held on a dedicated connection for the expensive render/QC
lifetime. It never holds candidate row locks or a long ORM transaction during
encoding, and it never transparently reconnects to pretend a lost lock is still
held. The heavy-model residency marker/lease is never reused for this.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

_NAMESPACE = b"clipfactory:stage5.2:render-execution:admission"


def global_render_admission_key() -> int:
    digest = hashlib.sha256(_NAMESPACE).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


class RenderAdmission(Protocol):
    def acquire(self, *, wait_seconds: float, cancel_check: Callable[[], bool]) -> bool: ...

    def release(self) -> None: ...

    def held(self) -> bool: ...


class NullRenderAdmission:
    """No-op admission for SQLite and single-process local execution."""

    def __init__(self) -> None:
        self._held = False

    def acquire(self, *, wait_seconds: float, cancel_check: Callable[[], bool]) -> bool:
        self._held = True
        return True

    def release(self) -> None:
        self._held = False

    def held(self) -> bool:
        return self._held


class PostgresRenderAdmission:
    """Dedicated-connection PostgreSQL advisory-lock admission."""

    def __init__(self, engine: Engine, *, key: int | None = None) -> None:
        self._engine = engine
        self._key = key if key is not None else global_render_admission_key()
        self._connection: Connection | None = None

    def acquire(self, *, wait_seconds: float, cancel_check: Callable[[], bool]) -> bool:
        if self.held():
            return True
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while True:
            if self._try_once():
                return True
            if cancel_check():
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.5)

    def _try_once(self) -> bool:
        try:
            connection = self._engine.connect()
        except Exception:
            return False
        try:
            acquired = bool(
                connection.execute(
                    text("SELECT pg_try_advisory_lock(:key)"), {"key": self._key}
                ).scalar()
            )
        except Exception:
            connection.close()
            return False
        if not acquired:
            connection.close()
            return False
        self._connection = connection
        return True

    def release(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is None:
            return
        try:
            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": self._key})
            connection.commit()
        except Exception:
            pass
        finally:
            connection.close()

    def held(self) -> bool:
        connection = self._connection
        if connection is None:
            return False
        try:
            if connection.closed:
                self._connection = None
                return False
        except Exception:
            return False
        return True


def render_admission_for(engine: Engine | None) -> RenderAdmission:
    """Return PostgreSQL admission when the engine is PostgreSQL, else no-op."""

    if engine is None:
        return NullRenderAdmission()
    dialect = getattr(engine, "dialect", None)
    name = getattr(dialect, "name", "")
    if name == "postgresql":
        return PostgresRenderAdmission(engine)
    return NullRenderAdmission()


__all__ = [
    "NullRenderAdmission",
    "PostgresRenderAdmission",
    "RenderAdmission",
    "global_render_admission_key",
    "render_admission_for",
]
