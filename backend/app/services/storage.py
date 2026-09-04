"""Safe, centrally owned application storage paths and writes."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path


class StorageError(RuntimeError):
    """Base error raised by storage operations."""


class StorageValidationError(StorageError, ValueError):
    """Raised for an invalid category, source ID, or filesystem path."""


class StorageUnavailableError(StorageError):
    """A recoverable storage capacity or availability failure."""

    retryable = True


class StorageCategory(str, Enum):
    """Persistent path categories owned by :class:`StorageService`."""

    SOURCES = "sources"
    JOBS = "jobs"
    TEMPORARY = "temporary"


class StorageService:
    """Own all persistent filesystem paths below a configured root."""

    def __init__(
        self,
        storage_root: Path,
        *,
        category_roots: Mapping[StorageCategory, Path] | None = None,
    ) -> None:
        self._storage_root = storage_root.expanduser().resolve()
        configured_roots = category_roots or {}
        self._category_roots = {
            category: self._validate_category_root(
                configured_roots.get(category, self._storage_root / category.value)
            )
            for category in StorageCategory
        }
        for root in self._category_roots.values():
            root.mkdir(parents=True, exist_ok=True)

    @property
    def storage_root(self) -> Path:
        """Return the resolved root containing every category directory."""

        return self._storage_root

    def category_root(self, category: StorageCategory | str) -> Path:
        """Return the resolved root for a validated storage category."""

        return self._category_roots[self._coerce_category(category)]

    def source_directory(self, source_id: uuid.UUID | str) -> Path:
        """Create and return the stable directory for one source UUID."""

        validated_id = self._validate_uuid(source_id)
        directory = self.resolve(StorageCategory.SOURCES, str(validated_id))
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def job_directory(self, job_id: uuid.UUID | str) -> Path:
        """Create and return the stable directory for one job UUID."""

        validated_id = self._validate_uuid(job_id)
        directory = self.resolve(StorageCategory.JOBS, str(validated_id))
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def resolve(self, category: StorageCategory | str, relative_path: Path | str) -> Path:
        """Resolve a relative path safely within its configured category root."""

        root = self.category_root(category)
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise StorageValidationError("absolute paths are not allowed")
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise StorageValidationError("path traversal outside storage category") from error
        return resolved

    def ensure_capacity(self, required_bytes: int) -> None:
        """Raise a recoverable error when storage cannot hold ``required_bytes``."""

        if required_bytes < 0:
            raise StorageValidationError("required bytes must be non-negative")
        free_bytes = shutil.disk_usage(self._storage_root)[2]
        if free_bytes < required_bytes:
            raise StorageUnavailableError(
                f"Insufficient free storage: need {required_bytes} bytes, have {free_bytes}"
            )

    def atomic_write(self, destination: Path, chunks: Iterable[bytes]) -> Path:
        """Write bytes to a temporary sibling and atomically replace ``destination``."""

        destination = self._validate_owned_destination(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as temporary_file:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise StorageValidationError("atomic writes require byte chunks")
                    temporary_file.write(chunk)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return destination

    def cleanup_temporary_files(self, *, older_than_seconds: float, limit: int = 100) -> int:
        """Delete at most ``limit`` expired ``.tmp`` files from the temporary root."""

        if older_than_seconds < 0:
            raise StorageValidationError("temporary file age must be non-negative")
        if limit <= 0:
            raise StorageValidationError("cleanup limit must be positive")
        cutoff = time.time() - older_than_seconds
        removed = 0
        for candidate in self.category_root(StorageCategory.TEMPORARY).rglob("*.tmp"):
            if removed == limit:
                break
            if candidate.is_file() and candidate.stat().st_mtime <= cutoff:
                candidate.unlink()
                removed += 1
        return removed

    def _validate_category_root(self, category_root: Path) -> Path:
        resolved = category_root.expanduser().resolve()
        try:
            resolved.relative_to(self._storage_root)
        except ValueError as error:
            raise StorageValidationError("category roots must stay below storage root") from error
        return resolved

    @staticmethod
    def _coerce_category(category: StorageCategory | str) -> StorageCategory:
        try:
            return StorageCategory(category)
        except ValueError as error:
            raise StorageValidationError(f"unknown storage category: {category}") from error

    @staticmethod
    def _validate_uuid(value: uuid.UUID | str) -> uuid.UUID:
        try:
            return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise StorageValidationError("source and job identifiers must be UUIDs") from error

    def _validate_owned_destination(self, destination: Path) -> Path:
        resolved = destination.expanduser().resolve()
        if not any(
            self._is_within(resolved, category_root)
            for category_root in self._category_roots.values()
        ):
            raise StorageValidationError("destination is outside configured storage categories")
        return resolved

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        return True
