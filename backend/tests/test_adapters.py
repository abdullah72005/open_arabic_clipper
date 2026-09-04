from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

from app.services.source_adapters import (
    LocalFileAdapter,
    SourceValidationError,
    YtDlpAdapter,
    normalize_source_url,
)
from app.services.storage import StorageService


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/video.mp4",
        "ftp://example.com/video.mp4",
        "https://operator:secret@example.com/video.mp4",
        "https:///video.mp4",
    ],
)
def test_normalize_source_url_rejects_unsupported_or_credentialed_urls(url: str) -> None:
    with pytest.raises(SourceValidationError):
        normalize_source_url(url)


def test_local_file_adapter_copies_permitted_file_to_source_directory(tmp_path: Path) -> None:
    local_video = tmp_path / "my source video.mp4"
    local_video.write_bytes(b"permitted local media")
    storage = StorageService(tmp_path / "storage")
    adapter = LocalFileAdapter(storage)
    source_id = uuid.uuid4()

    acquired = adapter.acquire(source_id, local_video)

    assert acquired.path.parent == storage.source_directory(source_id)
    assert acquired.path.read_bytes() == b"permitted local media"
    assert acquired.original_filename == "my_source_video.mp4"
    assert "authorized" in adapter.permission_notice.lower()


def test_ytdlp_commands_are_argument_vectors_without_shell_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = StorageService(tmp_path / "storage")
    adapter = YtDlpAdapter(storage, binary="yt-dlp-test")
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args[0], kwargs))
        return subprocess.CompletedProcess(args[0], 0, stdout='{"title": "safe", "filesize": 1}')

    monkeypatch.setattr("app.services.source_adapters.subprocess.run", fake_run)

    metadata = adapter.inspect("HTTPS://Example.COM/video#fragment")

    assert metadata["title"] == "safe"
    command, keyword_arguments = calls[0]
    assert isinstance(command, list)
    assert command[0] == "yt-dlp-test"
    assert "shell" not in keyword_arguments
    assert "--no-simulate" not in command
    assert command[-1] == "https://example.com/video"
