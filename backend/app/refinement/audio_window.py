"""Bounded candidate audio-window extraction for Stage 3.5.

The service owns the only path from a coarse candidate to a short mono 16 kHz
WAV that a local ASR provider may read. It never hands a whole source to a
provider, never persists a remote URI, and treats the cached Stage 2 audio
artifact plus the original source media as the authority for reuse.
"""

from __future__ import annotations

import math
import os
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import RefinementPriority
from app.models import AudioArtifact, ClipCandidate, SourceVideo
from app.refinement.fingerprints import component_fingerprint
from app.refinement.policy import DEFAULT_CONFIG, Stage35Config
from app.refinement.types import RefinementAudioWindow
from app.services.hashing import sha256_file
from app.services.storage import StorageCategory, StorageService


class RefinementAudioError(RuntimeError):
    """A bounded refinement window could not be produced or validated."""


CommandRunner = Callable[[list[str]], None]


def _run_command(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True, text=True)


class CandidateAudioWindowService:
    """Extract and cache the bounded audio interval for one candidate."""

    def __init__(
        self,
        *,
        storage: StorageService,
        session: Session,
        config: Stage35Config = DEFAULT_CONFIG,
        ffmpeg_binary: str = "ffmpeg",
        command_runner: CommandRunner = _run_command,
    ) -> None:
        self._storage = storage
        self._session = session
        self._config = config
        self._ffmpeg_binary = ffmpeg_binary
        self._command_runner = command_runner

    # context planning

    def context_bounds(
        self,
        *,
        coarse_start: float,
        coarse_end: float,
        source_duration: float,
        priority: RefinementPriority,
    ) -> tuple[float, float]:
        """Return the bounded context window around a coarse candidate interval.

        Pre/post padding is priority-scoped, the window is clamped to the source,
        and any window longer than ``max_refinement_window_seconds`` is shrunk
        symmetrically around the coarse midpoint so a single extraction stays
        bounded regardless of how coarse the candidate is.
        """

        if not (
            math.isfinite(coarse_start)
            and math.isfinite(coarse_end)
            and math.isfinite(source_duration)
        ):
            raise RefinementAudioError("coarse bounds and source duration must be finite")
        if coarse_start < 0 or coarse_end <= coarse_start or source_duration <= 0:
            raise RefinementAudioError("coarse bounds are invalid")

        pre, post = self._padding(priority)
        start = max(0.0, coarse_start - pre)
        end = min(source_duration, coarse_end + post)

        max_window = self._config.max_refinement_window_seconds
        if not math.isfinite(max_window) or max_window <= 0:
            raise RefinementAudioError("max refinement window must be finite and positive")
        if end - start > max_window:
            midpoint = (coarse_start + coarse_end) / 2.0
            start = midpoint - max_window / 2.0
            end = midpoint + max_window / 2.0
            if start < 0:
                start = 0.0
                end = min(source_duration, max_window)
            if end > source_duration:
                end = source_duration
                start = max(0.0, source_duration - max_window)

        if not (math.isfinite(start) and math.isfinite(end)) or not (0.0 <= start < end):
            raise RefinementAudioError("computed context window is invalid")
        return float(start), float(end)

    def _padding(self, priority: RefinementPriority) -> tuple[float, float]:
        if priority is RefinementPriority.FINAL_CLIP:
            return (
                self._config.final_pre_context_seconds,
                self._config.final_post_context_seconds,
            )
        if priority is RefinementPriority.CANDIDATE:
            return (
                self._config.candidate_pre_context_seconds,
                self._config.candidate_post_context_seconds,
            )
        raise RefinementAudioError(f"unsupported refinement priority: {priority}")

    # fingerprints

    def audio_input_fingerprint(
        self,
        *,
        source: SourceVideo,
        artifact: AudioArtifact,
        context_start: float,
        context_end: float,
        priority: RefinementPriority,
    ) -> str:
        """Return the stable checkpoint identity for one bounded window."""

        config = self._config
        return component_fingerprint(
            "audio-extraction",
            {
                "source_id": str(source.id),
                "source_content_hash": source.content_hash,
                "artifact_content_hash": artifact.content_hash,
                "artifact_source_content_hash": artifact.source_content_hash,
                "context_start": round(float(context_start), 6),
                "context_end": round(float(context_end), 6),
                "priority": priority.value,
                "candidate_pre_context_seconds": config.candidate_pre_context_seconds,
                "candidate_post_context_seconds": config.candidate_post_context_seconds,
                "final_pre_context_seconds": config.final_pre_context_seconds,
                "final_post_context_seconds": config.final_post_context_seconds,
                "max_refinement_window_seconds": config.max_refinement_window_seconds,
            },
        )

    # extraction

    def extract(
        self,
        *,
        source: SourceVideo,
        candidate: ClipCandidate,
        priority: RefinementPriority,
        force: bool = False,
    ) -> RefinementAudioWindow:
        """Return a validated bounded WAV for the candidate, extracting if needed."""

        artifact = self._validated_artifact(source)
        source_path = Path(source.source_uri)
        if not source_path.is_file():
            raise RefinementAudioError("source media file is unavailable for refinement audio")

        context_start, context_end = self.context_bounds(
            coarse_start=candidate.start_time,
            coarse_end=candidate.end_time,
            source_duration=artifact.duration,
            priority=priority,
        )
        relative = f"{source.id}/candidate-refinements/{candidate.id}/{priority.value}.wav"
        destination = self._storage.resolve(StorageCategory.SOURCES, relative)
        fingerprint = self.audio_input_fingerprint(
            source=source,
            artifact=artifact,
            context_start=context_start,
            context_end=context_end,
            priority=priority,
        )

        if not force and self._cache_matches(destination, fingerprint):
            return self._window(
                source=source,
                candidate=candidate,
                priority=priority,
                context_start=context_start,
                context_end=context_end,
                relative=relative,
                destination=destination,
                content_hash=sha256_file(destination),
                fingerprint=fingerprint,
            )

        content_hash = self._extract_window(
            source_path=source_path,
            destination=destination,
            context_start=context_start,
            context_end=context_end,
        )
        self._write_fingerprint(destination, fingerprint)
        return self._window(
            source=source,
            candidate=candidate,
            priority=priority,
            context_start=context_start,
            context_end=context_end,
            relative=relative,
            destination=destination,
            content_hash=content_hash,
            fingerprint=fingerprint,
        )

    def _validated_artifact(self, source: SourceVideo) -> AudioArtifact:
        artifact: AudioArtifact | None = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        if artifact is None:
            raise RefinementAudioError("cached audio artifact is unavailable")
        if artifact.source_content_hash != source.content_hash:
            raise RefinementAudioError("cached audio artifact is stale for this source")
        path = self._storage.resolve(StorageCategory.SOURCES, artifact.output_path)
        if not path.is_file() or sha256_file(path) != artifact.content_hash:
            raise RefinementAudioError("cached audio artifact content does not match its hash")
        return artifact

    def _extract_window(
        self,
        *,
        source_path: Path,
        destination: Path,
        context_start: float,
        context_end: float,
    ) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.tmp")
        requested = context_end - context_start
        args = [
            self._ffmpeg_binary,
            "-y",
            "-ss",
            str(context_start),
            "-i",
            str(source_path),
            "-t",
            str(requested),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
        try:
            self._command_runner(args)
            duration = _wav_duration(temporary)
            valid = (
                temporary.is_file()
                and temporary.stat().st_size > 0
                and math.isfinite(duration)
                and 0.05 <= duration <= requested + 1.0
            )
            if not valid:
                raise RefinementAudioError("ffmpeg did not produce a valid refinement window")
            content_hash = str(sha256_file(temporary))
            os.replace(temporary, destination)
            return content_hash
        except Exception as error:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            if isinstance(error, RefinementAudioError):
                raise
            if isinstance(error, subprocess.CalledProcessError):
                raise RefinementAudioError("ffmpeg failed to extract refinement audio") from error
            if isinstance(error, OSError):
                raise RefinementAudioError(
                    "ffmpeg is unavailable for refinement audio extraction"
                ) from error
            raise RefinementAudioError("refinement audio extraction failed") from error

    def _cache_matches(self, destination: Path, fingerprint: str) -> bool:
        if not destination.is_file():
            return False
        sidecar = _fingerprint_path(destination)
        if not sidecar.is_file():
            return False
        try:
            return sidecar.read_text(encoding="utf-8").strip() == fingerprint
        except OSError:
            return False

    def _write_fingerprint(self, destination: Path, fingerprint: str) -> None:
        try:
            _fingerprint_path(destination).write_text(fingerprint, encoding="utf-8")
        except OSError:
            pass

    def _window(
        self,
        *,
        source: SourceVideo,
        candidate: ClipCandidate,
        priority: RefinementPriority,
        context_start: float,
        context_end: float,
        relative: str,
        destination: Path,
        content_hash: str,
        fingerprint: str,
    ) -> RefinementAudioWindow:
        return RefinementAudioWindow(
            source_id=str(source.id),
            candidate_id=str(candidate.id),
            priority=priority,
            coarse_start=candidate.start_time,
            coarse_end=candidate.end_time,
            context_start=context_start,
            context_end=context_end,
            relative_path=relative,
            content_hash=content_hash,
            duration=_wav_duration(destination),
            input_fingerprint=fingerprint,
        )


def _fingerprint_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.fingerprint")


def _wav_duration(path: Path) -> float:
    """Return the duration of a PCM WAV, or 0.0 when it cannot be read."""

    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getframerate() <= 0:
                return 0.0
            return wav.getnframes() / wav.getframerate()
    except (EOFError, wave.Error, OSError):
        return 0.0
