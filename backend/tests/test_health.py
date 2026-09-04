from __future__ import annotations

from pathlib import Path

from app.services.health import CheckStatus, HealthService
from app.services.storage import StorageService


def test_unhealthy_required_dependency_fails_aggregate(tmp_path: Path) -> None:
    service = HealthService(
        storage=StorageService(tmp_path),
        checks={"database": lambda: (CheckStatus.FAILED, "unreachable")},
    )
    report = service.report()
    assert report.status is CheckStatus.FAILED
    assert report.checks[0].detail == "unreachable"


def test_unhealthy_optional_dependency_degrades_aggregate(tmp_path: Path) -> None:
    service = HealthService(
        storage=StorageService(tmp_path),
        checks={"redis": lambda: (CheckStatus.DEGRADED, "unavailable")},
    )
    report = service.report()
    assert report.status is CheckStatus.DEGRADED


def test_storage_report_includes_capacity_values(tmp_path: Path) -> None:
    report = HealthService(storage=StorageService(tmp_path)).storage_report()
    assert report.total_bytes > 0
    assert report.free_bytes >= 0
    assert report.used_bytes >= 0
