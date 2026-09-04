from __future__ import annotations

import hashlib
import os
import time
import uuid
from pathlib import Path

import pytest

from app.services.hashing import HASH_CHUNK_SIZE, sha256_file
from app.services.storage import (
    StorageCategory,
    StorageService,
    StorageUnavailableError,
    StorageValidationError,
)


def test_source_directory_is_stable_and_below_sources_category(tmp_path: Path) -> None:
    storage = StorageService(tmp_path)
    source_id = uuid.uuid4()

    first = storage.source_directory(source_id)
    second = storage.source_directory(str(source_id))

    assert first == second
    assert first.is_relative_to(storage.category_root(StorageCategory.SOURCES))
    assert first.name == str(source_id)


def test_resolve_rejects_traversal_outside_a_category(tmp_path: Path) -> None:
    storage = StorageService(tmp_path)

    with pytest.raises(StorageValidationError, match="traversal"):
        storage.resolve(StorageCategory.SOURCES, "../../outside.mp4")


def test_atomic_write_replaces_the_complete_final_file(tmp_path: Path) -> None:
    storage = StorageService(tmp_path)
    destination = storage.source_directory(uuid.uuid4()) / "original.mp4"
    destination.write_bytes(b"old")

    storage.atomic_write(destination, (part for part in (b"new-", b"content")))

    assert destination.read_bytes() == b"new-content"
    assert not list(destination.parent.glob("*.tmp"))


def test_cleanup_removes_only_expired_temporary_files(tmp_path: Path) -> None:
    storage = StorageService(tmp_path)
    temporary = storage.category_root(StorageCategory.TEMPORARY)
    old_temp = temporary / "old.tmp"
    fresh_temp = temporary / "fresh.tmp"
    regular = temporary / "keep.mp4"
    for path in (old_temp, fresh_temp, regular):
        path.write_bytes(b"data")
    old_timestamp = time.time() - 120
    os.utime(old_temp, (old_timestamp, old_timestamp))

    removed = storage.cleanup_temporary_files(older_than_seconds=60, limit=1)

    assert removed == 1
    assert not old_temp.exists()
    assert fresh_temp.exists()
    assert regular.exists()


def test_cleanup_does_not_recursively_traverse_temporary_directories(tmp_path: Path) -> None:
    storage = StorageService(tmp_path)
    nested_temporary_file = (
        storage.category_root(StorageCategory.TEMPORARY) / "nested" / "old.tmp"
    )
    nested_temporary_file.parent.mkdir()
    nested_temporary_file.write_bytes(b"data")
    old_timestamp = time.time() - 120
    os.utime(nested_temporary_file, (old_timestamp, old_timestamp))

    removed = storage.cleanup_temporary_files(older_than_seconds=60)

    assert removed == 0
    assert nested_temporary_file.exists()


def test_ensure_capacity_raises_recoverable_error_when_free_space_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = StorageService(tmp_path)
    monkeypatch.setattr("app.services.storage.shutil.disk_usage", lambda _: (0, 0, 9))

    with pytest.raises(StorageUnavailableError, match="Insufficient free storage"):
        storage.ensure_capacity(10)


def test_sha256_file_reads_the_file_in_fixed_size_chunks(tmp_path: Path) -> None:
    content = b"a" * (HASH_CHUNK_SIZE + 1)
    source = tmp_path / "video.mp4"
    source.write_bytes(content)

    digest = sha256_file(source)

    assert digest == hashlib.sha256(content).hexdigest()
