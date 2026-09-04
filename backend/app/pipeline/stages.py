"""Concrete durable Stage 2 executors used only by worker processes."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.models import AudioArtifact, SourceVideo, Transcript
from app.pipeline.runner import StageExecutionError
from app.transcription.engine import TranscriptionResult, WhisperEngine
from app.transcription.service import TranscriptionOptions


class TranscriptionExecutor:
    """Run local Whisper once per cache fingerprint and persist raw evidence."""

    def __init__(
        self,
        *,
        session: Session,
        engine: WhisperEngine,
        options: TranscriptionOptions | None = None,
    ) -> None:
        self._session = session
        self._engine = engine
        self._options = options or get_settings().transcription_options()

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
        result = self._engine.transcribe(Path(artifact.output_path), self._options)
        transcript = existing or Transcript(source_video_id=source.id)
        self._apply(transcript, result, fingerprint)
        if existing is None:
            self._session.add(transcript)
        self._session.commit()
        self._session.refresh(transcript)
        return transcript

    def _apply(self, transcript: Transcript, result: TranscriptionResult, fingerprint: str) -> None:
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
