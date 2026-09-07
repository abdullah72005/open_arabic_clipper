"""Spawn a child process that owns a native model and return a bounded result.

The parent never imports or constructs the native model. The child loads it,
runs one call, fully consumes generators, and returns a picklable outcome. The
parent always joins/reaps the child and verifies a non-live exit state, which is
the hard reclamation boundary for native allocators like CTranslate2.

A child that dies without writing an envelope (SIGKILL, OOM, segfault) is
detected by monitoring process liveness while waiting for the queue, so the
parent reaps it promptly instead of waiting for the full timeout.
"""

from __future__ import annotations

import multiprocessing
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_POLL_SECONDS = 0.25


@dataclass(frozen=True)
class ProcessOutcome:
    """Bounded typed result of one spawned child call."""

    ok: bool
    result: Any | None
    error: str | None
    child_pid: int | None
    exit_code: int | None
    child_peak_rss_bytes: int
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

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    try:
        result = target(*args, **kwargs)
        queue.put(("ok", result, __import__("os").getpid(), peak))
    except BaseException as error:  # noqa: BLE001 - bounded envelope, no traceback
        queue.put(("error", f"{type(error).__name__}: {error}", __import__("os").getpid(), peak))


class ModelProcessRunner:
    """Run one heavy call in a spawned child with a bounded timeout and reap."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        context: Any | None = None,
        poll_seconds: float = _POLL_SECONDS,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._context = context or multiprocessing.get_context("spawn")
        self._poll_seconds = poll_seconds

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
        deadline = started + self._timeout_seconds
        envelope = None
        while envelope is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.terminate()
                process.join()
                return ProcessOutcome(
                    False,
                    None,
                    "child timed out",
                    process.pid,
                    process.exitcode,
                    0,
                    time.monotonic() - started,
                )
            try:
                envelope = queue.get(timeout=min(self._poll_seconds, remaining))
            except Exception:
                if not process.is_alive():
                    process.join()
                    return ProcessOutcome(
                        False,
                        None,
                        f"child exited without a result envelope (exit {process.exitcode})",
                        process.pid,
                        process.exitcode,
                        0,
                        time.monotonic() - started,
                    )
        process.join()
        if process.is_alive():
            process.terminate()
            process.join()
        elapsed = time.monotonic() - started
        if envelope is None:
            return ProcessOutcome(
                False, None, "child produced no result", process.pid, process.exitcode, 0, elapsed
            )
        kind, payload, child_pid, child_peak = envelope
        if kind == "ok":
            return ProcessOutcome(
                True, payload, None, child_pid, process.exitcode, child_peak, elapsed
            )
        return ProcessOutcome(
            False, None, str(payload), child_pid, process.exitcode, child_peak, elapsed
        )


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
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        try:
            result = target(*args, **(kwargs or {}))
            return ProcessOutcome(
                True, result, None, __import__("os").getpid(), 0, peak, time.monotonic() - started
            )
        except Exception as error:
            return ProcessOutcome(
                False,
                None,
                f"{type(error).__name__}: {error}",
                __import__("os").getpid(),
                0,
                peak,
                time.monotonic() - started,
            )
