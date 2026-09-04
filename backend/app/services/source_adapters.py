"""Safe acquisition adapters for operator-authorized media sources."""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import SplitResult, urlsplit, urlunsplit

from app.services.storage import StorageService, StorageValidationError

_INVALID_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_REMOTE_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024


class SourceAdapterError(RuntimeError):
    """Base error raised while acquiring an authorized source."""


class SourceValidationError(SourceAdapterError, ValueError):
    """Raised when a source input cannot be acquired safely."""


class SourceConfigurationError(SourceAdapterError):
    """Raised when safe URL acquisition has not been configured."""


class SourcePermissionError(SourceAdapterError):
    """Raised when the operator has not affirmed permission to process a source."""


class SourceAcquisitionError(SourceAdapterError):
    """Raised when an external source acquisition command fails."""


@dataclass(frozen=True, slots=True)
class AcquiredSource:
    """A local, storage-owned copy of an acquired source."""

    path: Path
    original_filename: str
    source_url: str | None = None


class SourceAdapter(Protocol):
    """Acquire an operator-authorized source into its stable source directory."""

    permission_notice: str

    def acquire(self, source_id: uuid.UUID | str, source: Path | str) -> AcquiredSource:
        """Acquire ``source`` into storage for ``source_id``."""


def normalize_source_url(value: str) -> str:
    """Validate and normalize a public HTTP(S) URL without credentials or fragments."""

    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise SourceValidationError("only public http and https URLs are supported")
    if parsed.username is not None or parsed.password is not None:
        raise SourceValidationError("URLs with credentials are not supported")
    if not parsed.hostname:
        raise SourceValidationError("URL must include a hostname")
    _assert_public_ip_literal(parsed.hostname)

    normalized = SplitResult(
        scheme=parsed.scheme.lower(),
        netloc=_normalized_netloc(parsed),
        path=parsed.path or "/",
        query=parsed.query,
        fragment="",
    )
    return urlunsplit(normalized)


class LocalFileAdapter:
    """Copy a local file the operator is authorized to process into storage."""

    permission_notice = (
        "Only acquire local media you own or are authorized to process; this adapter "
        "does not bypass access controls."
    )

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    def acquire(self, source_id: uuid.UUID | str, source: Path | str) -> AcquiredSource:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise SourceValidationError("local source must be an existing regular file")

        try:
            self._storage.ensure_capacity(source_path.stat().st_size)
            filename = sanitize_filename(source_path.name)
            destination = self._storage.source_directory(source_id) / filename
            self._storage.atomic_write(destination, _file_chunks(source_path))
        except StorageValidationError as error:
            raise SourceValidationError(str(error)) from error
        return AcquiredSource(path=destination, original_filename=filename)


class YtDlpAdapter:
    """Acquire legally permitted public URLs through an uncredentialed yt-dlp invocation."""

    permission_notice = (
        "Use yt-dlp only for public media you own or are authorized to process. "
        "Authentication, DRM, paywall, CAPTCHA, and access-control bypass are unsupported."
    )

    def __init__(
        self,
        storage: StorageService,
        *,
        binary: str = "yt-dlp",
        max_download_bytes: int = DEFAULT_MAX_REMOTE_DOWNLOAD_BYTES,
        egress_proxy: str | None = None,
    ) -> None:
        if max_download_bytes <= 0:
            raise SourceValidationError("maximum remote download size must be positive")
        self._storage = storage
        self._binary = binary
        self._max_download_bytes = max_download_bytes
        self._egress_proxy = egress_proxy

    def inspect(self, url: str) -> Mapping[str, object]:
        """Read public metadata before downloading, using a safe argument vector."""

        self._require_egress_proxy()
        normalized_url = normalize_source_url(url)
        result = self._run(self._metadata_command(normalized_url), normalized_url)
        try:
            metadata = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise SourceAcquisitionError("yt-dlp returned invalid metadata JSON") from error
        if not isinstance(metadata, dict):
            raise SourceAcquisitionError("yt-dlp metadata must be an object")
        return metadata

    def acquire(self, source_id: uuid.UUID | str, source: Path | str) -> AcquiredSource:
        self._require_egress_proxy()
        if not isinstance(source, str):
            raise SourceValidationError("yt-dlp source must be a URL string")
        normalized_url = normalize_source_url(source)
        metadata = self.inspect(normalized_url)
        expected_bytes = _expected_bytes(metadata, self._max_download_bytes)
        self._storage.ensure_capacity(expected_bytes)
        source_directory = self._storage.source_directory(source_id)
        result = self._run_download(
            self._download_command(source_directory, normalized_url),
            normalized_url,
            source_directory,
        )
        acquired_path = _downloaded_path(result.stdout, source_directory)
        if not acquired_path.is_file():
            raise SourceAcquisitionError(
                "yt-dlp did not produce a media file in the source directory"
            )
        return AcquiredSource(
            path=acquired_path,
            original_filename=sanitize_filename(acquired_path.name),
            source_url=normalized_url,
        )

    def _metadata_command(self, normalized_url: str) -> list[str]:
        return [
            self._binary,
            "--ignore-config",
            "--proxy",
            self._require_egress_proxy(),
            "--no-playlist",
            "--dump-single-json",
            normalized_url,
        ]

    def _download_command(self, source_directory: Path, normalized_url: str) -> list[str]:
        output_template = str(source_directory / "%(title).100B-%(id)s.%(ext)s")
        return [
            self._binary,
            "--ignore-config",
            "--proxy",
            self._require_egress_proxy(),
            "--no-playlist",
            "--no-write-info-json",
            "--no-write-thumbnail",
            "--no-write-subs",
            "--restrict-filenames",
            "--max-filesize",
            str(self._max_download_bytes),
            "--output",
            output_template,
            "--print",
            "after_move:filepath",
            normalized_url,
        ]

    @staticmethod
    def _run(command: list[str], normalized_url: str) -> subprocess.CompletedProcess[str]:
        _assert_public_host_resolution(urlsplit(normalized_url).hostname)
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise SourceAcquisitionError("yt-dlp acquisition failed") from error

    def _run_download(
        self, command: list[str], normalized_url: str, output_directory: Path
    ) -> subprocess.CompletedProcess[str]:
        """Run yt-dlp while enforcing a bounded output directory size."""

        _assert_public_host_resolution(urlsplit(normalized_url).hostname)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as error:
            raise SourceAcquisitionError("yt-dlp acquisition failed") from error

        while process.poll() is None:
            if _directory_size(output_directory) > self._max_download_bytes:
                process.terminate()
                process.communicate()
                raise SourceAcquisitionError("yt-dlp exceeded the configured download limit")
            time.sleep(0.1)

        stdout, stderr = process.communicate()
        if _directory_size(output_directory) > self._max_download_bytes:
            raise SourceAcquisitionError("yt-dlp exceeded the configured download limit")
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, output=stdout, stderr=stderr
            )
        return subprocess.CompletedProcess(
            command, process.returncode, stdout=stdout, stderr=stderr
        )

    def _require_egress_proxy(self) -> str:
        if not self._egress_proxy:
            raise SourceConfigurationError(
                "URL acquisition requires a configured trusted egress proxy"
            )
        return self._egress_proxy


def sanitize_filename(value: str) -> str:
    """Return a non-empty portable filename without directory components."""

    name = Path(value).name.strip().replace(" ", "_")
    cleaned = _INVALID_FILENAME_CHARS.sub("_", name).strip("._")
    if not cleaned or cleaned in {".", ".."}:
        raise SourceValidationError("source filename is not usable")
    return cleaned[:200]


def _normalized_netloc(parsed: SplitResult) -> str:
    hostname = parsed.hostname
    if hostname is None:  # guarded by ``normalize_source_url`` for type narrowing
        raise SourceValidationError("URL must include a hostname")
    try:
        port = parsed.port
    except ValueError as error:
        raise SourceValidationError("URL has an invalid port") from error
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    return hostname.lower() if port is None or default_port else f"{hostname.lower()}:{port}"


def _file_chunks(path: Path) -> Iterator[bytes]:
    with path.open("rb") as source_file:
        while chunk := source_file.read(_CHUNK_SIZE):
            yield chunk


def _expected_bytes(metadata: Mapping[str, object], maximum_bytes: int) -> int:
    for key in ("filesize", "filesize_approx"):
        value = metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            if value > maximum_bytes:
                raise SourceAcquisitionError(
                    "yt-dlp metadata exceeds the configured download limit"
                )
            return value
    raise SourceAcquisitionError("yt-dlp metadata must provide a known size before download")


def _assert_public_ip_literal(hostname: str) -> None:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if not address.is_global:
        raise SourceValidationError("URL target must be a public address")


def _assert_public_host_resolution(hostname: str | None) -> None:
    if hostname is None:
        raise SourceValidationError("URL must include a hostname")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise SourceValidationError("URL target must be a public address")
        return
    try:
        resolved_addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError as error:
        raise SourceValidationError("URL hostname could not be resolved") from error
    if not resolved_addresses:
        raise SourceValidationError("URL hostname could not be resolved")
    for _, _, _, _, sockaddr in resolved_addresses:
        if not ipaddress.ip_address(sockaddr[0]).is_global:
            raise SourceValidationError("URL hostname resolves to a non-public address")


def _downloaded_path(stdout: str, source_directory: Path) -> Path:
    printed_paths = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not printed_paths:
        raise SourceAcquisitionError("yt-dlp did not report an output path")
    candidate = Path(printed_paths[-1]).resolve()
    try:
        candidate.relative_to(source_directory.resolve())
    except ValueError as error:
        raise SourceAcquisitionError(
            "yt-dlp reported a path outside the source directory"
        ) from error
    return candidate


def _directory_size(directory: Path) -> int:
    """Return regular-file bytes written below an adapter-owned output directory."""

    total = 0
    for candidate in directory.rglob("*"):
        if candidate.is_file():
            total += candidate.stat().st_size
    return total
