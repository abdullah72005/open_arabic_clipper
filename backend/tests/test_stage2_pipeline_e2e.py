from __future__ import annotations

import subprocess
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.enums import PipelineStage
from app.db.base import Base
from app.media.audio import AudioExtractor
from app.media.ffprobe import FFprobe
from app.models import SourceVideo
from app.pipeline.runner import PipelineRunner
from app.pipeline.stages import (
    AudioAnalysisExecutor,
    AudioExtractionExecutor,
    IngestExecutor,
    ProbeExecutor,
    TranscriptionExecutor,
    TranscriptNormalizationExecutor,
)
from app.services.storage import StorageService
from app.transcription.engine import TranscriptionResult
from app.transcription.service import TranscriptionOptions


class FixedEngine:
    def transcribe(self, _path: Path, _options: TranscriptionOptions) -> TranscriptionResult:
        return TranscriptionResult(
            language="ar",
            language_probability=0.95,
            raw_text="أهلا hello",
            segments=[{"start": 0.0, "end": 1.0, "text": "أهلا hello"}],
            word_segments=[{"start": 0.0, "end": 0.3, "word": "أهلا"}],
            duration=1.0,
        )


def test_generated_owned_media_reaches_ready_for_analysis(
    sqlite_engine: object, tmp_path: Path
) -> None:
    """Concrete FFmpeg stages plus durable transitions work without a real ASR download."""
    source_path = tmp_path / "generated.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-shortest",
            str(source_path),
        ],
        check=True,
        capture_output=True,
    )
    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri=str(source_path), content_hash="generated")
        session.add(source)
        session.commit()
        executors = {
            PipelineStage.INGEST: IngestExecutor(),
            PipelineStage.PROBE: ProbeExecutor(FFprobe()),
            PipelineStage.AUDIO_EXTRACTION: AudioExtractionExecutor(
                AudioExtractor(session=session, storage=storage)
            ),
            PipelineStage.TRANSCRIPTION: TranscriptionExecutor(
                session=session,
                engine=FixedEngine(),
                options=TranscriptionOptions("small", "cpu", "int8", 5),
                storage=storage,
            ),
            PipelineStage.TRANSCRIPT_NORMALIZATION: TranscriptNormalizationExecutor(
                session=session
            ),
            PipelineStage.AUDIO_ANALYSIS: AudioAnalysisExecutor(session=session, storage=storage),
        }
        runner = PipelineRunner(session, executors)
        for stage in (
            PipelineStage.INGEST,
            PipelineStage.PROBE,
            PipelineStage.AUDIO_EXTRACTION,
            PipelineStage.TRANSCRIPTION,
            PipelineStage.TRANSCRIPT_NORMALIZATION,
            PipelineStage.AUDIO_ANALYSIS,
        ):
            runner.run(source.id, stage)

        session.refresh(source)
        assert source.lifecycle_state is PipelineStage.READY_FOR_ANALYSIS
        assert source.transcript is not None
        assert source.transcript.chunks[0].text == "أهلا hello"
        assert source.audio_analysis is not None
        assert source.quality_assessment is not None
