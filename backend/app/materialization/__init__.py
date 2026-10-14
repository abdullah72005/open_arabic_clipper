"""Stage 6 transformed-short materialization.

Stage 6 owns authored content, timeline, and editing decisions. The Stage 5.2
renderer deterministically executes a frozen recipe; it never writes scripts,
selects strategies, calls semantic providers, decides originality, or chooses
retention edits.
"""

from __future__ import annotations

__all__: list[str] = []
