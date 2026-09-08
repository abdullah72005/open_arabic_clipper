"""Typed boundary between durable orchestration and stage work."""

from dataclasses import dataclass
from typing import Protocol

from app.models import SourceVideo


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
