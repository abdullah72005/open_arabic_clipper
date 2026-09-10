"""Typed boundary between durable orchestration and stage work."""

from dataclasses import dataclass
from typing import Protocol

from app.models import SourceVideo


class ReconstructionCancelled(RuntimeError):
    """Cooperative cancellation was requested while reconstruction was running.

    The job must stay CANCELLED, no later pipeline stage may be scheduled, and
    already-checkpointed results are preserved for a bounded retry.
    """

    retryable = False


@dataclass(frozen=True)
class StageExecutionResult:
    output_fingerprint: str
    value: object | None = None

    def __getattr__(self, name: str) -> object:
        if self.value is not None:
            return getattr(self.value, name)
        raise AttributeError(name)


class StageExecutor(Protocol):
    """Execute one pipeline stage for a source video."""

    def input_fingerprint(self, source: SourceVideo) -> str: ...

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        """Perform stage work or raise an exception."""
