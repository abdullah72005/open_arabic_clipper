"""Real-FFmpeg regression for the bounded Stage 3.5 refinement audio window.

Exercises ``CandidateAudioWindowService.extract`` with the real command runner
and its temporary ``.tmp`` output name. Before the explicit WAV muxer was added,
FFmpeg could not infer a format from the ``.tmp`` extension and aborted, so the
extraction must be proven to actually produce, validate, hash and atomically
publish a mono 16 kHz PCM window for both CANDIDATE and FINAL_CLIP priorities.
"""

from __future__ import annotations

import math
import shutil
import wave
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.core.enums import CandidateDisposition, RefinementPriority
from app.db.base import Base
from app.models import AudioArtifact, ClipCandidate, SourceVideo
from app.refinement.audio_window import (
    CandidateAudioWindowService,
    _run_command,
)
from app.refinement.policy import DEFAULT_CONFIG
from app.services.hashing import sha256_file
from app.services.storage import StorageCategory, StorageService

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg unavailable")

_SAMPLE_RATE = 16_000


def _write_tone(path: Path, duration: float, *, frequency: float = 440.0) -> None:
    """Write a deterministic mono 16 kHz PCM tone."""

    frames = max(1, int(round(duration * _SAMPLE_RATE)))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    for index in range(frames):
        value = int(0.5 * 32767 * math.sin(2 * math.pi * frequency * index / _SAMPLE_RATE))
        payload += value.to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(bytes(payload))


def _setup_candidate(
    storage: StorageService,
    session: Session,
    tmp_path: Path,
    *,
    source_duration: float = 30.0,
    coarse: tuple[float, float] = (10.0, 12.0),
) -> tuple[SourceVideo, ClipCandidate]:
    source_path = tmp_path / "source.wav"
    _write_tone(source_path, source_duration)
    source = SourceVideo(source_uri=str(source_path), content_hash=sha256_file(source_path))
    session.add(source)
    session.flush()

    relative = f"{source.id}/speech-analysis.wav"
    artifact_path = storage.resolve(StorageCategory.SOURCES, relative)
    _write_tone(artifact_path, source_duration)
    session.add(
        AudioArtifact(
            source_video_id=source.id,
            output_path=relative,
            content_hash=sha256_file(artifact_path),
            source_content_hash=source.content_hash,
            sample_rate=_SAMPLE_RATE,
            duration=source_duration,
        )
    )
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key="candidate-1",
        disposition=CandidateDisposition.CANDIDATE,
        start_time=coarse[0],
        end_time=coarse[1],
        start_segment_index=0,
        end_segment_index=0,
    )
    session.add(candidate)
    session.commit()
    return source, candidate


@requires_ffmpeg
@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("priority", "expected_duration"),
    [
        (RefinementPriority.CANDIDATE, 12.0),
        (RefinementPriority.FINAL_CLIP, 18.0),
    ],
)
def test_real_extraction_writes_valid_wav_from_temporary_name(
    sqlite_engine: object,
    tmp_path: Path,
    priority: RefinementPriority,
    expected_duration: float,
) -> None:
    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    calls: list[list[str]] = []

    def runner(args: list[str]) -> None:
        calls.append(list(args))
        _run_command(args)

    with Session(sqlite_engine) as session:
        source, candidate = _setup_candidate(storage, session, tmp_path)
        service = CandidateAudioWindowService(
            storage=storage,
            session=session,
            config=DEFAULT_CONFIG,
            command_runner=runner,
        )
        window = service.extract(source=source, candidate=candidate, priority=priority)
        destination = storage.resolve(StorageCategory.SOURCES, window.relative_path)

        # The extraction helper was exercised through its temporary filename.
        assert len(calls) == 1
        assert calls[0][-1].endswith(".tmp")
        assert not destination.exists() or destination.is_file()
        # No temporary residue survives the atomic replacement.
        assert not list(destination.parent.glob("*.tmp"))

        assert destination.is_file()
        with wave.open(str(destination), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == _SAMPLE_RATE
            assert wav.getnframes() / _SAMPLE_RATE == pytest.approx(expected_duration, abs=0.25)

        assert math.isfinite(window.duration)
        assert window.duration == pytest.approx(expected_duration, abs=0.25)
        assert window.content_hash == sha256_file(destination)
        assert destination.with_name(destination.name + ".fingerprint").is_file()

        # Cache reuse: a second identical request performs no new FFmpeg work.
        before = len(calls)
        second = service.extract(source=source, candidate=candidate, priority=priority)
        assert len(calls) == before
        assert second.content_hash == window.content_hash


@requires_ffmpeg
def test_real_extraction_reports_ffmpeg_failure_for_unreadable_source(
    sqlite_engine: object, tmp_path: Path
) -> None:
    from app.refinement.audio_window import RefinementAudioError

    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    with Session(sqlite_engine) as session:
        source, candidate = _setup_candidate(storage, session, tmp_path)
        Path(source.source_uri).write_bytes(b"not-real-audio")
        service = CandidateAudioWindowService(
            storage=storage,
            session=session,
            config=DEFAULT_CONFIG,
            command_runner=_run_command,
        )
        with pytest.raises(RefinementAudioError):
            service.extract(
                source=source,
                candidate=candidate,
                priority=RefinementPriority.CANDIDATE,
            )

    destination = storage.resolve(
        StorageCategory.SOURCES,
        f"{source.id}/candidate-refinements/{candidate.id}/CANDIDATE.wav",
    )
    assert not destination.exists()
    assert not list(destination.parent.glob("*.tmp"))
