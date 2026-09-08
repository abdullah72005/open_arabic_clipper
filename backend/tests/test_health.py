from __future__ import annotations

from pathlib import Path

from app.services.health import CheckStatus, HealthCheck, HealthService
from app.services.storage import StorageService
from app.transcription.reconstruction.types import ProviderAvailability, ProviderHealth


class MissingModelProvider:
    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.UNAVAILABLE,
            "ollama",
            "qwen3:8b",
            None,
            "configured model qwen3:8b is not installed",
        )

    def release(self) -> None:
        return None


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


def test_missing_reconstruction_model_degrades_named_health_check(tmp_path: Path) -> None:
    service = HealthService(
        storage=StorageService(tmp_path),
        reconstruction_provider=MissingModelProvider(),
    )

    report = service.report()

    assert report.status is CheckStatus.DEGRADED
    assert report.checks == [
        HealthCheck(
            name="reconstruction_provider",
            status=CheckStatus.DEGRADED,
            detail="configured model qwen3:8b is not installed",
        )
    ]
