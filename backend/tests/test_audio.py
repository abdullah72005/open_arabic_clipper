import subprocess
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models import AudioArtifact, SourceVideo
from app.services.storage import StorageService


class FakeAudioCommand:
    """Writes deterministic WAV-like bytes in place of the external FFmpeg process."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, args: list[str]) -> None:
        self.calls += 1
        Path(args[-1]).write_bytes(b"RIFFtest-wav")


class MissingStreamCommand:
    """Simulates FFmpeg's missing-audio diagnostic."""

    def __call__(self, args: list[str]) -> None:
        raise subprocess.CalledProcessError(
            1, args, stderr="Output file #0 does not contain any stream"
        )


def test_audio_extraction_caches_a_mono_analysis_artifact(
    sqlite_engine: object, tmp_path: Path
) -> None:
    """A second extraction reuses a verified artifact instead of calling FFmpeg again."""

    from app.media.audio import AudioExtractor

    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri=str(tmp_path / "source.mp4"), content_hash="source-hash")
        Path(source.source_uri).write_bytes(b"source")
        session.add(source)
        session.commit()
        runner = FakeAudioCommand()
        extractor = AudioExtractor(storage=storage, command_runner=runner, session=session)

        artifact = extractor.extract(source)
        persisted = session.get(AudioArtifact, artifact.id)
        assert persisted is not None
        assert persisted.source_video_id == source.id
        assert persisted.source_content_hash == "source-hash"
        assert storage.resolve("sources", persisted.output_path).is_file()
        cached = extractor.extract(source)

        assert artifact.sample_rate == 16_000
        assert artifact.duration == 0.0
        assert artifact.output_path == cached.output_path
        assert runner.calls == 1


def test_audio_extraction_reports_missing_audio_stream(
    sqlite_engine: object, tmp_path: Path
) -> None:
    """A source without audio fails clearly instead of silently creating an unusable artifact."""

    from app.media.audio import AudioExtractor, MissingAudioStreamError

    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    source_path = tmp_path / "silent.mp4"
    source_path.write_bytes(b"source")
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri=str(source_path), content_hash="silent-source")
        session.add(source)
        session.commit()

        extractor = AudioExtractor(
            storage=storage, command_runner=MissingStreamCommand(), session=session
        )

        with pytest.raises(MissingAudioStreamError, match="no usable audio stream"):
            extractor.extract(source)
