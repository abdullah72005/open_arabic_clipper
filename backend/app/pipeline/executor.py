"""Typed boundary between durable orchestration and stage work."""

from typing import Protocol

from app.models import SourceVideo


class StageExecutor(Protocol):
    """Execute one pipeline stage for a source video."""

    def execute(self, source: SourceVideo) -> None:
        """Perform stage work or raise an exception."""
