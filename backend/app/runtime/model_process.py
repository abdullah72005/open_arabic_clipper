"""Spawn a child process that owns a native model and return a bounded result.

The parent never imports or constructs the native model. The child loads it,
runs one call, fully consumes generators, and returns a picklable outcome. The
parent always joins/reaps the child and verifies a non-live exit state, which is
the hard reclamation boundary for native allocators like CTranslate2.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProcessOutcome:
    """Bounded typed result of one spawned child call."""

    ok: bool
    result: Any | None
    error: str | None
    child_pid: int | None
    exit_code: int | None
    elapsed_seconds: float


class SpawnedProcessError(RuntimeError):
    """A spawned model call failed or timed out."""


def _child_entrypoint(
    queue: Any,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Run in the spawned child; send a bounded envelope back to the parent."""

    try:
        result = target(*args, **kwargs)
        queue.put(("ok", result, __import__("os").getpid()))
    except BaseException as error:  # noqa: BLE001 - bounded envelope, no traceback
        queue.put(("error", f"{type(error).__name__}: {error}", __import__("os").getpid()))


class ModelProcessRunner:
    """Run one heavy call in a spawned child with a bounded timeout and reap."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        context: Any | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._context = context or multiprocessing.get_context("spawn")

    def run(
        self,
        *,
        target: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> ProcessOutcome:
        """Execute target in a child; the parent always reaps the child."""

        queue = self._context.Queue()
        process = self._context.Process(
            target=_child_entrypoint, args=(queue, target, args, kwargs or {})
        )
        started = time.monotonic()
        process.start()
        try:
            envelope = queue.get(timeout=self._timeout_seconds)
        except Exception:
            process.terminate()
            process.join()
            return ProcessOutcome(
                False,
                None,
                "child timed out",
                process.pid,
                process.exitcode,
                time.monotonic() - started,
            )
        process.join()
        if process.is_alive():
            process.terminate()
            process.join()
        elapsed = time.monotonic() - started
        if envelope is None:
            return ProcessOutcome(
                False, None, "child produced no result", process.pid, process.exitcode, elapsed
            )
        kind, payload, child_pid = envelope
        if kind == "ok":
            return ProcessOutcome(True, payload, None, child_pid, process.exitcode, elapsed)
        return ProcessOutcome(False, None, str(payload), child_pid, process.exitcode, elapsed)


class DirectRunner:
    """Run the target in-process; used by tests and lightweight callers."""

    def run(
        self,
        *,
        target: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> ProcessOutcome:
        started = time.monotonic()
        try:
            result = target(*args, **(kwargs or {}))
            return ProcessOutcome(
                True, result, None, __import__("os").getpid(), 0, time.monotonic() - started
            )
        except Exception as error:
            return ProcessOutcome(
                False,
                None,
                f"{type(error).__name__}: {error}",
                __import__("os").getpid(),
                0,
                time.monotonic() - started,
            )
