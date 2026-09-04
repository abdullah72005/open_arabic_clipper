"""Concrete durable Stage 2 executors used only by worker processes."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.media.analysis import parse_silencedetect, silence_ratio, windowed_rms
from app.media.audio import AudioExtractor
from app.models import AudioAnalysis, AudioArtifact, SourceVideo, Transcript, TranscriptChunk
from app.pipeline.runner import StageExecutionError
from app.services.source_quality import assess_source
from app.services.storage import StorageCategory, StorageService
from app.transcription.chunking import ChunkConfig, build_chunks
from app.transcription.engine import TranscriptionResult, WhisperEngine
from app.transcription.normalization import normalize_transcript
from app.transcription.service import TranscriptionOptions


class TranscriptionExecutor:
    """Run local Whisper once per cache fingerprint and persist raw evidence."""

    def __init__(
        self,
        *,
        session: Session,
        engine: WhisperEngine,
        options: TranscriptionOptions | None = None,
        storage: StorageService | None = None,
    ) -> None:
        self._session = session
        self._engine = engine
        self._options = options or get_settings().transcription_options()
        self._storage = storage

    def execute(self, source: SourceVideo) -> Transcript:
        artifact = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        if artifact is None:
            raise StageExecutionError("speech-analysis audio is missing")
        fingerprint = self._options.fingerprint(artifact.content_hash)
        existing = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if existing is not None and existing.input_fingerprint == fingerprint:
            return existing
        audio_path = Path(artifact.output_path)
        if not audio_path.is_absolute():
            storage = self._storage or StorageService(get_settings().storage_root)
            audio_path = storage.resolve(StorageCategory.SOURCES, audio_path)
        started_at = monotonic()
        result = self._engine.transcribe(audio_path, self._options)
        transcript = existing or Transcript(source_video_id=source.id)
        self._apply(transcript, result, fingerprint, monotonic() - started_at)
        if existing is None:
            self._session.add(transcript)
        self._session.commit()
        self._session.refresh(transcript)
        return transcript

    def _apply(
        self,
        transcript: Transcript,
        result: TranscriptionResult,
        fingerprint: str,
        processing_duration: float,
    ) -> None:
        transcript.language = result.language
        transcript.detected_language_probability = result.language_probability
        transcript.whisper_model = self._options.model
        transcript.transcription_options = {
            "model": self._options.model,
            "device": self._options.device,
            "compute_type": self._options.compute_type,
            "beam_size": self._options.beam_size,
            "language": self._options.language,
            "word_timestamps": self._options.word_timestamps,
        }
        transcript.input_fingerprint = fingerprint
        transcript.raw_text = result.raw_text
        transcript.normalized_text = result.raw_text
        transcript.segments = result.segments
        transcript.word_segments = result.word_segments
        transcript.duration = result.duration
        transcript.processing_duration = processing_duration


class AudioExtractionExecutor:
    """Prepare the cached WAV before local transcription."""

    def __init__(self, extractor: AudioExtractor) -> None:
        self._extractor = extractor

    def execute(self, source: SourceVideo) -> AudioArtifact:
        return self._extractor.extract(source)


class TranscriptNormalizationExecutor:
    """Normalize a persisted transcript without rewriting its raw ASR evidence."""

    def __init__(self, *, session: Session) -> None:
        self._session = session

    def execute(self, source: SourceVideo) -> Transcript:
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if transcript is None:
            raise StageExecutionError("transcript is missing")
        transcript.normalized_text = normalize_transcript(transcript.raw_text)
        transcript.segments = [
            {**segment, "normalized_text": normalize_transcript(str(segment.get("text", "")))}
            for segment in transcript.segments
        ]
        self._session.execute(
            delete(TranscriptChunk).where(TranscriptChunk.transcript_id == transcript.id)
        )
        self._session.add_all(
            TranscriptChunk(
                transcript_id=transcript.id,
                sequence=sequence,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text=chunk.text,
                segment_indexes=chunk.segment_indexes,
                preceding_context=chunk.preceding_context,
                following_context=chunk.following_context,
            )
            for sequence, chunk in enumerate(build_chunks(transcript.segments, ChunkConfig()))
        )
        self._session.commit()
        self._session.refresh(transcript)
        return transcript


SilenceCommandRunner = Callable[[list[str]], str]


def _run_silence_command(args: list[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stderr


class AudioAnalysisExecutor:
    """Persist silence and speech-density signals, then refresh source quality."""

    def __init__(
        self,
        *,
        session: Session,
        storage: StorageService,
        ffmpeg_binary: str = "ffmpeg",
        command_runner: SilenceCommandRunner = _run_silence_command,
    ) -> None:
        self._session = session
        self._storage = storage
        self._ffmpeg_binary = ffmpeg_binary
        self._command_runner = command_runner

    def execute(self, source: SourceVideo) -> AudioAnalysis:
        artifact = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if artifact is None or transcript is None:
            raise StageExecutionError("audio artifact and normalized transcript are required")
        existing = self._session.scalar(
            select(AudioAnalysis).where(AudioAnalysis.source_video_id == source.id)
        )
        if existing is not None and existing.audio_hash == artifact.content_hash:
            return existing
        audio_path = self._storage.resolve(StorageCategory.SOURCES, artifact.output_path)
        args = [
            self._ffmpeg_binary,
            "-i",
            str(audio_path),
            "-af",
            "silencedetect=n=-35dB:d=0.4",
            "-f",
            "null",
            "-",
        ]
        try:
            intervals = parse_silencedetect(self._command_runner(args))
        except (OSError, subprocess.CalledProcessError) as err:
            raise StageExecutionError("ffmpeg audio analysis failed") from err
        duration = max(artifact.duration, transcript.duration)
        ratio = silence_ratio(intervals, duration)
        analysis = existing or AudioAnalysis(source_video_id=source.id)
        analysis.audio_hash = artifact.content_hash
        analysis.silence_intervals = [interval.__dict__ for interval in intervals]
        analysis.features = windowed_rms(audio_path)
        if not analysis.features:
            analysis.features = [{"start": 0.0, "end": duration, "rms": 0.0}]
        analysis.silence_ratio = ratio
        analysis.speech_density = 1.0 - ratio
        analysis.speech_rate = len(transcript.word_segments) * 60.0 / duration if duration else 0.0
        if existing is None:
            self._session.add(analysis)
        self._session.commit()
        self._session.refresh(analysis)
        assess_source(self._session, source, transcript, analysis)
        return analysis
