"""Operational checks for local ClipFactory dependencies."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from app.services.storage import StorageService
from app.transcription.reconstruction.types import ProviderAvailability, ProviderHealth


class CheckStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class HealthReport:
    status: CheckStatus
    checks: list[HealthCheck]


@dataclass(frozen=True)
class StorageReport:
    total_bytes: int
    used_bytes: int
    free_bytes: int


Check = Callable[[], tuple[CheckStatus, str]]


class ProviderHealthProbe(Protocol):
    def health(self) -> ProviderHealth: ...


class HealthService:
    """Aggregate bounded dependency checks without raising to HTTP callers."""

    def __init__(
        self,
        storage: StorageService,
        checks: dict[str, Check] | None = None,
        reconstruction_provider: ProviderHealthProbe | None = None,
    ) -> None:
        self._storage = storage
        self._checks = dict(checks or {})
        if reconstruction_provider is not None:
            self._checks["reconstruction_provider"] = lambda: self._provider_status(
                reconstruction_provider
            )

    def report(self) -> HealthReport:
        checks = [self._run(name, check) for name, check in self._checks.items()]
        status = CheckStatus.HEALTHY
        if any(check.status is CheckStatus.FAILED for check in checks):
            status = CheckStatus.FAILED
        elif any(check.status is CheckStatus.DEGRADED for check in checks):
            status = CheckStatus.DEGRADED
        return HealthReport(status=status, checks=checks)

    def storage_report(self) -> StorageReport:
        usage = shutil.disk_usage(self._storage.storage_root)
        return StorageReport(total_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free)

    @staticmethod
    def _run(name: str, check: Check) -> HealthCheck:
        try:
            status, detail = check()
        except Exception as err:
            return HealthCheck(name=name, status=CheckStatus.FAILED, detail=str(err))
        return HealthCheck(name=name, status=status, detail=detail)

    @staticmethod
    def _provider_status(provider: ProviderHealthProbe) -> tuple[CheckStatus, str]:
        health = provider.health()
        status = (
            CheckStatus.HEALTHY
            if health.availability is ProviderAvailability.AVAILABLE
            else CheckStatus.DEGRADED
        )
        return status, health.detail
