from __future__ import annotations

import io
import socket
import subprocess
import uuid
from pathlib import Path

import pytest

from app.services.source_adapters import (
    MAX_DOWNLOAD_DIAGNOSTIC_BYTES,
    LocalFileAdapter,
    SourceAcquisitionError,
    SourceValidationError,
    YtDlpAdapter,
    _directory_size,
    _downloaded_path,
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
        "http://127.0.0.1/video.mp4",
        "http://169.254.1.1/video.mp4",
        "http://10.0.0.1/video.mp4",
    ],
)
def test_normalize_source_url_rejects_unsupported_or_credentialed_urls(url: str) -> None:
    with pytest.raises(SourceValidationError):
        normalize_source_url(url)


def test_ytdlp_adapter_rejects_hostname_resolving_to_private_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = StorageService(tmp_path / "storage")
    adapter = YtDlpAdapter(storage, egress_proxy="http://trusted-proxy:8080")

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    )

    with pytest.raises(SourceValidationError, match="non-public"):
        adapter.inspect("https://example.com/video")


def test_ytdlp_adapter_rejects_unknown_download_size_before_running_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = StorageService(tmp_path / "storage")
    adapter = YtDlpAdapter(storage, egress_proxy="http://trusted-proxy:8080")
    monkeypatch.setattr(adapter, "inspect", lambda url: {"title": "unknown size"})

    def fail_if_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("download must not run without a known size")

    monkeypatch.setattr("app.services.source_adapters.subprocess.run", fail_if_run)

    with pytest.raises(SourceAcquisitionError, match="known size"):
        adapter.acquire(uuid.uuid4(), "https://8.8.8.8/video")


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
    adapter = YtDlpAdapter(storage, binary="yt-dlp-test", egress_proxy="http://trusted-proxy:8080")
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args[0], kwargs))
        return subprocess.CompletedProcess(args[0], 0, stdout='{"title": "safe", "filesize": 1}')

    monkeypatch.setattr("app.services.source_adapters.subprocess.run", fake_run)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )

    metadata = adapter.inspect("HTTPS://Example.COM/video#fragment")

    assert metadata["title"] == "safe"
    command, keyword_arguments = calls[0]
    assert isinstance(command, list)
    assert command[0] == "yt-dlp-test"
    assert "--ignore-config" in command
    assert command[command.index("--proxy") + 1] == "http://trusted-proxy:8080"
    assert "shell" not in keyword_arguments
    assert "--no-simulate" not in command
    assert command[-1] == "https://example.com/video"


def test_ytdlp_download_command_ignores_ambient_configuration(tmp_path: Path) -> None:
    adapter = YtDlpAdapter(
        StorageService(tmp_path / "storage"), egress_proxy="http://trusted-proxy:8080"
    )

    command = adapter._download_command(tmp_path / "source", "https://example.com/video")

    assert "--ignore-config" in command
    assert command[command.index("--proxy") + 1] == "http://trusted-proxy:8080"
    assert command[command.index("--max-filesize") + 1] == str(adapter._max_download_bytes)
    assert "--no-progress" in command
    assert "--print" not in command


def test_ytdlp_direct_download_command_uses_no_proxy_or_credentials(tmp_path: Path) -> None:
    adapter = YtDlpAdapter(StorageService(tmp_path / "storage"))

    command = adapter._download_command(tmp_path / "source", "https://example.com/video")

    assert "--proxy" not in command
    assert "--cookies" not in command
    assert "--username" not in command


def test_directory_size_ignores_fragment_removed_during_download_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fragment = tmp_path / "video.part-Frag42"
    fragment.write_bytes(b"partial")
    original_stat = Path.stat
    calls = 0

    def disappearing_stat(path: Path, *args: object, **kwargs: object):
        nonlocal calls
        if path == fragment:
            calls += 1
            if calls == 2:
                raise FileNotFoundError
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", disappearing_stat)

    assert _directory_size(tmp_path) == 0


def test_downloaded_path_accepts_restricted_filename_with_leading_dots(tmp_path: Path) -> None:
    media = tmp_path / "..-video-id.webm"
    media.write_bytes(b"authorized video")

    assert _downloaded_path(tmp_path) == media


def test_ytdlp_download_monitor_terminates_when_directory_exceeds_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_directory = tmp_path / "source"
    output_directory.mkdir()
    (output_directory / "partial.mp4").write_bytes(b"12345")
    adapter = YtDlpAdapter(
        StorageService(tmp_path / "storage"),
        egress_proxy="http://trusted-proxy:8080",
        max_download_bytes=4,
    )

    class RunningProcess:
        terminated = False
        stderr = io.BytesIO()

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def communicate(self) -> tuple[str, str]:
            return "", ""

        def wait(self) -> int:
            return 0

    process = RunningProcess()
    monkeypatch.setattr(
        "app.services.source_adapters.subprocess.Popen", lambda *args, **kwargs: process
    )

    with pytest.raises(SourceAcquisitionError, match="configured download limit"):
        adapter._run_download(["yt-dlp"], "https://8.8.8.8/video", output_directory)

    assert process.terminated


def test_ytdlp_download_continuously_drains_noisy_stderr_into_bounded_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_directory = tmp_path / "source"
    output_directory.mkdir()
    adapter = YtDlpAdapter(
        StorageService(tmp_path / "storage"),
        egress_proxy="http://trusted-proxy:8080",
    )
    calls: list[dict[str, object]] = []

    class FinishedNoisyProcess:
        returncode = 0
        stderr = io.BytesIO(b"x" * (MAX_DOWNLOAD_DIAGNOSTIC_BYTES * 2))

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        "app.services.source_adapters.subprocess.Popen",
        lambda *args, **kwargs: (calls.append(kwargs), FinishedNoisyProcess())[1],
    )

    result = adapter._run_download(["yt-dlp"], "https://8.8.8.8/video", output_directory)

    assert calls[0]["stdout"] is subprocess.DEVNULL
    assert calls[0]["stderr"] is subprocess.PIPE
    assert len(result.stderr) == MAX_DOWNLOAD_DIAGNOSTIC_BYTES
