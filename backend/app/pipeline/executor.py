"""Typed boundary between durable orchestration and stage work."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from app.models import SourceVideo


class StageCancelled(RuntimeError):
    """Cooperative cancellation was requested while a stage was running.

    The job must stay CANCELLED, no later pipeline stage may be scheduled, and
    already-checkpointed results are preserved for a bounded retry. Stage 2.7's
    ``ReconstructionCancelled`` is a compatibility subclass so all existing
    reconstruction behavior and callers are unchanged.
    """

    retryable = False


class ReconstructionCancelled(StageCancelled):
    """Deprecated alias retained for backward compatibility with Stage 2.7."""


@dataclass(frozen=True)
class StageExecutionResult:
    output_fingerprint: str
    value: object | None = None
    metrics: Mapping[str, object] = field(default_factory=dict)

    def __getattr__(self, name: str) -> object:
        if self.value is not None:
            return getattr(self.value, name)
        raise AttributeError(name)


class StageExecutor(Protocol):
    """Execute one pipeline stage for a source video."""

    def input_fingerprint(self, source: SourceVideo) -> str: ...

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        """Perform stage work or raise an exception."""
