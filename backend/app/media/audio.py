"""Safe extraction and caching of speech-analysis WAV artifacts."""

from __future__ import annotations

import subprocess
import wave
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AudioArtifact, SourceVideo
from app.services.hashing import sha256_file
from app.services.storage import StorageCategory, StorageService


class AudioExtractionError(RuntimeError):
    """Base error raised while preparing audio for transcription."""


class MissingAudioStreamError(AudioExtractionError):
    """Raised when FFmpeg reports that no usable audio stream exists."""


CommandRunner = Callable[[list[str]], None]


def _run_command(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True, text=True)


class AudioExtractor:
    """Build an idempotent mono 16 kHz PCM WAV artifact for one source."""

    def __init__(
        self,
        *,
        storage: StorageService,
        session: Session,
        ffmpeg_binary: str = "ffmpeg",
        command_runner: CommandRunner = _run_command,
    ) -> None:
        self._storage = storage
        self._session = session
        self._ffmpeg_binary = ffmpeg_binary
        self._command_runner = command_runner

    def extract(self, source: SourceVideo) -> AudioArtifact:
        """Return a valid cached artifact or extract one through FFmpeg."""

        existing = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        if existing is not None and self._is_valid(existing, source):
            return existing

        source_path = Path(source.source_uri)
        if not source_path.is_file():
            raise AudioExtractionError("source media file is unavailable for audio extraction")

        output_path = self._storage.resolve(
            StorageCategory.SOURCES, f"{source.id}/speech-analysis.wav"
        )
        self._storage.ensure_capacity(max(source_path.stat().st_size, 1))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            self._ffmpeg_binary,
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        try:
            self._command_runner(args)
        except subprocess.CalledProcessError as err:
            details = (err.stderr or "").lower()
            if "does not contain any stream" in details or "no audio" in details:
                raise MissingAudioStreamError("source has no usable audio stream") from err
            raise AudioExtractionError("ffmpeg failed to extract analysis audio") from err
        except OSError as err:
            raise AudioExtractionError("ffmpeg is unavailable for audio extraction") from err

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AudioExtractionError("ffmpeg did not produce analysis audio")

        artifact = existing or AudioArtifact(source_video=source, source_video_id=source.id)
        artifact.output_path = str(
            output_path.relative_to(self._storage.category_root(StorageCategory.SOURCES))
        )
        artifact.content_hash = sha256_file(output_path)
        artifact.source_content_hash = source.content_hash
        artifact.sample_rate = 16_000
        artifact.duration = _wav_duration(output_path)
        if existing is None:
            self._session.add(artifact)
        self._session.commit()
        self._session.refresh(artifact)
        return artifact

    def _is_valid(self, artifact: AudioArtifact, source: SourceVideo) -> bool:
        if artifact.source_content_hash != source.content_hash:
            return False
        path = self._storage.resolve(StorageCategory.SOURCES, artifact.output_path)
        return path.is_file() and sha256_file(path) == artifact.content_hash


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getframerate() <= 0:
                return 0.0
            return wav.getnframes() / wav.getframerate()
    except wave.Error:
        return 0.0
