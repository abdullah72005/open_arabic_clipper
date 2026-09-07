"""Read-only runtime memory snapshot for the heavy-model lifecycle.

Host RAM, the WSL VM limit, the container cgroup limit, process RSS, and swap
are deliberately kept as separate labeled fields. ``/proc`` and cgroup parsing
is pure and separated from filesystem access so tests never depend on the
developer machine.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

_KB = 1024
_PAGE_SIZE = os.sysconf("SC_PAGESIZE")


class MemoryReadError(RuntimeError):
    """Required Linux memory inputs cannot be read."""


@dataclass(frozen=True)
class MemorySnapshot:
    """One point-in-time labeled memory state."""

    timestamp: float
    linux_total: int
    linux_available: int
    swap_total: int
    swap_free: int
    cgroup_limit: int | None
    cgroup_current: int | None
    cgroup_peak: int | None
    process_rss: int

    @property
    def effective_capacity(self) -> int:
        """The narrowest active capacity available to this process."""

        if self.cgroup_limit is not None:
            return min(self.linux_total, self.cgroup_limit)
        return self.linux_total


def parse_meminfo(text: str) -> tuple[int, int, int, int]:
    """Return (total, available, swap_total, swap_free) bytes from meminfo text."""

    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            parts = rest.split()
            if parts:
                try:
                    values[key] = int(parts[0]) * _KB
                except ValueError:
                    pass
    if "MemTotal" not in values:
        raise ValueError("meminfo is missing required MemTotal")
    return (
        values["MemTotal"],
        values.get("MemAvailable", 0),
        values.get("SwapTotal", 0),
        values.get("SwapFree", 0),
    )


def parse_cgroup_value(text: str) -> int | None:
    """Parse a cgroup byte value; ``max``, empty, or malformed input is None."""

    stripped = text.strip()
    if not stripped or stripped == "max":
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def parse_statm(text: str) -> int:
    """Return the resident page count from /proc/self/statm."""

    parts = text.split()
    if len(parts) < 2:
        raise ValueError("statm has no resident field")
    return int(parts[1])


def capture_memory(
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    timestamp: float | None = None,
) -> MemorySnapshot:
    """Read the current labeled memory state; raises only for required inputs."""

    try:
        total, available, swap_total, swap_free = parse_meminfo(
            (proc_root / "meminfo").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise MemoryReadError("cannot read required /proc/meminfo") from error
    cgroup_limit, cgroup_current, cgroup_peak = _read_cgroup(cgroup_root)
    rss = _read_process_rss(proc_root)
    return MemorySnapshot(
        timestamp=timestamp if timestamp is not None else time.time(),
        linux_total=total,
        linux_available=available,
        swap_total=swap_total,
        swap_free=swap_free,
        cgroup_limit=cgroup_limit,
        cgroup_current=cgroup_current,
        cgroup_peak=cgroup_peak,
        process_rss=rss,
    )


def assert_8b_preflight(
    snapshot: MemorySnapshot, *, minimum_bytes: int = 10 * 1024**3
) -> str | None:
    """Return an actionable error if the snapshot lacks capacity for an 8B trial."""

    if snapshot.effective_capacity >= minimum_bytes:
        return None
    measured_gib = snapshot.effective_capacity / (1024**3)
    required_gib = minimum_bytes / (1024**3)
    return (
        f"8B benchmark requires at least {required_gib:.1f} GiB effective capacity; "
        f"measured {measured_gib:.2f} GiB. "
        "Raise the WSL/Docker VM limit per docs/ENVIRONMENT.md and rerun "
        "scripts/diagnose-memory.sh."
    )


def _read_cgroup(cgroup_root: Path) -> tuple[int | None, int | None, int | None]:
    v2_limit = cgroup_root / "memory.max"
    if v2_limit.is_file():
        return (
            _read_optional(v2_limit),
            _read_optional(cgroup_root / "memory.current"),
            _read_optional(cgroup_root / "memory.peak"),
        )
    v1_root = cgroup_root / "memory"
    v1_limit = v1_root / "memory.limit_in_bytes"
    if v1_limit.is_file():
        return (
            _read_optional(v1_limit),
            _read_optional(v1_root / "memory.usage_in_bytes"),
            _read_optional(v1_root / "memory.max_usage_in_bytes"),
        )
    return (None, None, None)


def _read_optional(path: Path) -> int | None:
    try:
        return parse_cgroup_value(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def _read_process_rss(proc_root: Path) -> int:
    try:
        pages = parse_statm((proc_root / "self" / "statm").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return pages * _PAGE_SIZE
