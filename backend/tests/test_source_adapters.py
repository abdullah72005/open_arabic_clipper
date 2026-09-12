from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

from app.services.source_adapters import YtDlpAdapter
from app.services.storage import StorageService


def test_remote_acquisition_records_metadata_and_download_boundaries(tmp_path: Path) -> None:
    """Remote timings bracket the two yt-dlp subprocess boundaries independently."""

    clock_values = iter((10.0, 11.5, 12.0, 16.0))

    class RecordingAdapter(YtDlpAdapter):
        def inspect(self, url: str) -> dict[str, object]:
            assert url == "https://example.com/owned"
            return {"filesize": 4}

        def _run_download(
            self, command: list[str], normalized_url: str, output_directory: Path
        ) -> subprocess.CompletedProcess[str]:
            assert normalized_url == "https://example.com/owned"
            assert command[-1] == normalized_url
            (output_directory / "owned.webm").write_bytes(b"data")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    adapter = RecordingAdapter(
        StorageService(tmp_path / "storage"),
        monotonic=lambda: next(clock_values),
    )

    acquired = adapter.acquire(uuid4(), "https://example.com/owned")

    assert acquired.path.read_bytes() == b"data"
    assert acquired.metrics == {
        "source_kind": "remote",
        "cache_reuse": "miss",
        "metadata_seconds": 1.5,
        "download_seconds": 4.0,
        "directory_scan_seconds": 0.0,
        "directory_scan_count": 0,
        "storage_write_seconds": 0.0,
        "source_hash_seconds": 0.0,
        "artifact_bytes": 4,
        "postprocess_state": "not_requested",
        "postprocess_seconds": None,
    }
