import os
from pathlib import Path

import pytest

from app.runtime.memory import (
    MemoryReadError,
    MemorySnapshot,
    assert_8b_preflight,
    capture_memory,
    parse_cgroup_value,
    parse_meminfo,
    parse_statm,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _meminfo(
    total: int = 7803048,
    available: int = 4553532,
    swap_total: int = 2097152,
    swap_free: int = 735320,
) -> str:
    return (
        f"MemTotal:        {total} kB\n"
        "MemFree:          1 kB\n"
        f"MemAvailable:    {available} kB\n"
        f"SwapTotal:       {swap_total} kB\n"
        f"SwapFree:        {swap_free} kB\n"
    )


def test_parse_meminfo_extracts_total_available_and_swap() -> None:
    total, available, swap_total, swap_free = parse_meminfo(_meminfo())

    assert total == 7803048 * 1024
    assert available == 4553532 * 1024
    assert swap_total == 2097152 * 1024
    assert swap_free == 735320 * 1024


def test_parse_meminfo_requires_mem_total() -> None:
    with pytest.raises(ValueError, match="MemTotal"):
        parse_meminfo("SwapTotal: 1 kB\n")


def test_parse_cgroup_value_max_and_malformed_are_unavailable() -> None:
    assert parse_cgroup_value("max") is None
    assert parse_cgroup_value("  383229952\n") == 383229952
    assert parse_cgroup_value("garbage") is None
    assert parse_cgroup_value("") is None


def test_parse_statm_returns_resident_pages() -> None:
    assert parse_statm("100 42 0 0 0 0 0\n") == 42


def test_snapshot_effective_capacity_is_min_with_finite_cgroup_limit(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    _write(proc / "meminfo", _meminfo())
    _write(proc / "self" / "statm", "10 5 0 0 0 0 0\n")
    cgroup = tmp_path / "cgroup"
    _write(cgroup / "memory.max", "4294967296\n")
    _write(cgroup / "memory.current", "1048576\n")
    _write(cgroup / "memory.peak", "2097152\n")

    snapshot = capture_memory(proc_root=proc, cgroup_root=cgroup)

    assert snapshot.linux_total == 7803048 * 1024
    assert snapshot.cgroup_limit == 4294967296
    assert snapshot.cgroup_current == 1048576
    assert snapshot.cgroup_peak == 2097152
    assert snapshot.effective_capacity == 4294967296
    assert snapshot.process_rss == 5 * os.sysconf("SC_PAGESIZE")


def test_cgroup_max_means_unlimited_and_missing_optional_files_are_none(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    _write(proc / "meminfo", _meminfo())
    _write(proc / "self" / "statm", "10 5 0 0 0 0 0\n")
    cgroup = tmp_path / "cgroup"
    _write(cgroup / "memory.max", "max\n")

    snapshot = capture_memory(proc_root=proc, cgroup_root=cgroup)

    assert snapshot.cgroup_limit is None
    assert snapshot.cgroup_current is None
    assert snapshot.cgroup_peak is None
    assert snapshot.effective_capacity == snapshot.linux_total


def test_missing_cgroup_tree_and_malformed_optional_values_are_tolerated(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    _write(proc / "meminfo", _meminfo())
    _write(proc / "self" / "statm", "10 5 0 0 0 0 0\n")

    snapshot = capture_memory(proc_root=proc, cgroup_root=tmp_path / "missing-cgroup")
    assert snapshot.cgroup_limit is None
    assert snapshot.cgroup_current is None
    assert snapshot.cgroup_peak is None

    cgroup = tmp_path / "cgroup"
    _write(cgroup / "memory.max", "max\n")
    _write(cgroup / "memory.current", "garbage\n")
    _write(cgroup / "memory.peak", "also-bad\n")

    snapshot2 = capture_memory(proc_root=proc, cgroup_root=cgroup)
    assert snapshot2.cgroup_current is None
    assert snapshot2.cgroup_peak is None


def test_cgroup_v1_fallback_when_supported(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _write(proc / "meminfo", _meminfo())
    _write(proc / "self" / "statm", "10 5 0 0 0 0 0\n")
    v1 = tmp_path / "cgroup" / "memory"
    _write(v1 / "memory.limit_in_bytes", "5368709120\n")
    _write(v1 / "memory.usage_in_bytes", "1048576\n")
    _write(v1 / "memory.max_usage_in_bytes", "2097152\n")

    snapshot = capture_memory(proc_root=proc, cgroup_root=v1.parent)

    assert snapshot.cgroup_limit == 5368709120
    assert snapshot.cgroup_current == 1048576
    assert snapshot.cgroup_peak == 2097152


def test_required_meminfo_missing_raises_read_error(tmp_path: Path) -> None:
    with pytest.raises(MemoryReadError):
        capture_memory(proc_root=tmp_path / "empty-proc", cgroup_root=tmp_path / "cgroup")


def test_assert_8b_preflight_reports_capacity_and_docs() -> None:
    low = MemorySnapshot(0.0, 7803048 * 1024, 1, 1, 1, None, None, None, 1)

    message = assert_8b_preflight(low)

    assert message is not None
    assert "7.44" in message
    assert "docs/ENVIRONMENT.md" in message

    enough = MemorySnapshot(0.0, 11 * 1024**3, 1, 1, 1, None, None, None, 1)
    assert assert_8b_preflight(enough) is None
