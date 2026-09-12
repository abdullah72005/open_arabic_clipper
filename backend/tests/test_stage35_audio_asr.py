"""Deterministic Stage 3.5 tests for bounded audio extraction and targeted ASR."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.core.enums import CandidateDisposition, RefinementPriority
from app.db.base import Base
from app.models import AudioArtifact, ClipCandidate, SourceVideo
from app.services.hashing import sha256_file
from app.services.storage import StorageCategory, StorageService
from app.transcription.engine import TranscriptionResult
from app.transcription.service import TranscriptionOptions


def _write_wav(path: Path, duration: float, sample_rate: int = 16_000) -> None:
    """Write a real mono 16 kHz PCM WAV so duration validation is exercised."""

    frames = max(1, int(round(duration * sample_rate)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


class FakeRefinementCommand:
    """Records exact FFmpeg arguments and writes a real bounded WAV to args[-1]."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> None:
        self.calls.append(list(args))
        _write_wav(Path(args[-1]), float(_arg_value(args, "-t")))


class EmptyOutputCommand:
    """Simulates FFmpeg producing an unusable empty file."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, args: list[str]) -> None:
        self.calls += 1
        Path(args[-1]).write_bytes(b"")


class FailingCommand:
    """Simulates an FFmpeg process failure."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, args: list[str]) -> None:
        self.calls += 1
        raise subprocess.CalledProcessError(1, args, stderr="boom")


def _arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _service(
    storage: StorageService,
    session: Session,
    *,
    runner: object | None = None,
    config: object | None = None,
):
    from app.refinement.audio_window import CandidateAudioWindowService
    from app.refinement.policy import DEFAULT_CONFIG

    return CandidateAudioWindowService(
        storage=storage,
        session=session,
        config=config or DEFAULT_CONFIG,
        command_runner=runner or FakeRefinementCommand(),  # type: ignore[arg-type]
    )


def _setup_candidate(
    storage: StorageService,
    session: Session,
    tmp_path: Path,
    *,
    source_hash: str = "source-hash",
    artifact_source_hash: str | None = None,
    artifact_duration: float = 60.0,
    coarse: tuple[float, float] = (10.0, 20.0),
) -> tuple[SourceVideo, AudioArtifact, ClipCandidate]:
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source-media")
    source = SourceVideo(source_uri=str(source_path), content_hash=source_hash)
    session.add(source)
    session.flush()
    relative = f"{source.id}/speech-analysis.wav"
    artifact_path = storage.resolve(StorageCategory.SOURCES, relative)
    _write_wav(artifact_path, artifact_duration)
    artifact = AudioArtifact(
        source_video_id=source.id,
        output_path=relative,
        content_hash=sha256_file(artifact_path),
        source_content_hash=(
            artifact_source_hash if artifact_source_hash is not None else source_hash
        ),
        sample_rate=16_000,
        duration=artifact_duration,
    )
    session.add(artifact)
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
    return source, artifact, candidate


def test_context_bounds_pads_by_priority_and_clamps(sqlite_engine: object, tmp_path: Path) -> None:
    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    with Session(sqlite_engine) as session:
        service = _service(storage, session)
        candidate = service.context_bounds(
            coarse_start=100.0,
            coarse_end=110.0,
            source_duration=200.0,
            priority=RefinementPriority.CANDIDATE,
        )
        final = service.context_bounds(
            coarse_start=100.0,
            coarse_end=110.0,
            source_duration=200.0,
            priority=RefinementPriority.FINAL_CLIP,
        )
        clamped = service.context_bounds(
            coarse_start=1.0,
            coarse_end=3.0,
            source_duration=10.0,
            priority=RefinementPriority.CANDIDATE,
        )
        tail = service.context_bounds(
            coarse_start=98.0,
            coarse_end=100.0,
            source_duration=100.0,
            priority=RefinementPriority.FINAL_CLIP,
        )

    assert candidate == (95.0, 115.0)
    assert final == (92.0, 118.0)
    assert clamped == (0.0, 8.0)
    assert tail == (90.0, 100.0)


def test_context_bounds_shrinks_oversized_windows(sqlite_engine: object, tmp_path: Path) -> None:
    from app.refinement.policy import DEFAULT_CONFIG

    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    config = DEFAULT_CONFIG.with_overrides(max_refinement_window_seconds=10.0)
    with Session(sqlite_engine) as session:
        service = _service(storage, session, config=config)
        start, end = service.context_bounds(
            coarse_start=0.0,
            coarse_end=100.0,
            source_duration=200.0,
            priority=RefinementPriority.CANDIDATE,
        )

    assert end - start == pytest.approx(10.0)
    assert start == pytest.approx(45.0)
    assert end == pytest.approx(55.0)
    assert 0.0 <= start < end <= 200.0


def test_extract_uses_exact_bounded_ffmpeg_arguments(sqlite_engine: object, tmp_path: Path) -> None:
    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    runner = FakeRefinementCommand()
    with Session(sqlite_engine) as session:
        source, _, candidate = _setup_candidate(storage, session, tmp_path)
        service = _service(storage, session, runner=runner)
        window = service.extract(
            source=source, candidate=candidate, priority=RefinementPriority.CANDIDATE
        )

    assert len(runner.calls) == 1
    args = runner.calls[0]
    assert _arg_value(args, "-ss") == "5.0"
    assert _arg_value(args, "-i") == str(Path(source.source_uri))
    assert _arg_value(args, "-i") != str(
        storage.resolve(StorageCategory.SOURCES, source.id.__str__() + "/speech-analysis.wav")
    )
    assert _arg_value(args, "-t") == "20.0"
    assert "-vn" in args
    assert _arg_value(args, "-ac") == "1"
    assert _arg_value(args, "-ar") == "16000"
    assert _arg_value(args, "-c:a") == "pcm_s16le"
    assert args[-1].endswith(".tmp")
    assert "speech-analysis.wav" not in args
    assert window.context_start == 5.0
    assert window.context_end == 25.0
    assert window.relative_path == f"{source.id}/candidate-refinements/{candidate.id}/CANDIDATE.wav"
    assert storage.resolve(StorageCategory.SOURCES, window.relative_path).is_file()
    assert window.duration == pytest.approx(20.0, abs=0.1)
    assert window.content_hash == sha256_file(
        storage.resolve(StorageCategory.SOURCES, window.relative_path)
    )


def test_extract_reuses_cached_window_without_ffmpeg(sqlite_engine: object, tmp_path: Path) -> None:
    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    runner = FakeRefinementCommand()
    with Session(sqlite_engine) as session:
        source, _, candidate = _setup_candidate(storage, session, tmp_path)
        service = _service(storage, session, runner=runner)
        first = service.extract(
            source=source, candidate=candidate, priority=RefinementPriority.CANDIDATE
        )
        second = service.extract(
            source=source, candidate=candidate, priority=RefinementPriority.CANDIDATE
        )

    assert len(runner.calls) == 1
    assert first.input_fingerprint == second.input_fingerprint
    assert first.relative_path == second.relative_path
    assert first.content_hash == second.content_hash


def test_extract_removes_partial_destination_on_failure(
    sqlite_engine: object, tmp_path: Path
) -> None:
    from app.refinement.audio_window import RefinementAudioError

    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    failing = FailingCommand()
    empty = EmptyOutputCommand()
    with Session(sqlite_engine) as session:
        source, _, candidate = _setup_candidate(storage, session, tmp_path)
        for runner in (failing, empty):
            service = _service(storage, session, runner=runner)
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
    assert failing.calls == 1
    assert empty.calls == 1
    assert not destination.exists()
    assert not list(destination.parent.glob("*.tmp"))


def test_extract_rejects_stale_artifact_without_ffmpeg(
    sqlite_engine: object, tmp_path: Path
) -> None:
    from app.refinement.audio_window import RefinementAudioError

    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    runner = FakeRefinementCommand()
    with Session(sqlite_engine) as session:
        source, _, candidate = _setup_candidate(
            storage, session, tmp_path, artifact_source_hash="stale-hash"
        )
        service = _service(storage, session, runner=runner)
        with pytest.raises(RefinementAudioError):
            service.extract(
                source=source,
                candidate=candidate,
                priority=RefinementPriority.CANDIDATE,
            )

    assert len(runner.calls) == 0


class FakeASREngine:
    """Returns a fixed transcription result without loading Whisper."""

    def __init__(self, result: TranscriptionResult) -> None:
        self.result = result
        self.calls = 0
        self.last_options: TranscriptionOptions | None = None

    def transcribe(
        self,
        audio_path: Path,
        options: TranscriptionOptions,
        cancel_event: object = None,
    ) -> TranscriptionResult:
        del audio_path, cancel_event
        self.calls += 1
        self.last_options = options
        return self.result


class RaisingASREngine:
    """Raises immediately to exercise lease cleanup."""

    def transcribe(
        self, audio_path: Path, options: TranscriptionOptions, cancel_event: object = None
    ) -> TranscriptionResult:
        del audio_path, options, cancel_event
        raise RuntimeError("boom")


def _options_for(_priority: RefinementPriority) -> TranscriptionOptions:
    return TranscriptionOptions(model="small", device="cpu", compute_type="int8", beam_size=5)


def _result(
    word_segments: list[dict[str, object]], raw_text: str = "مرحبا بك"
) -> TranscriptionResult:
    return TranscriptionResult(
        language="ar",
        language_probability=0.97,
        raw_text=raw_text,
        duration=2.0,
        segments=[],
        word_segments=word_segments,
    )


def test_targeted_asr_converts_words_to_source_time() -> None:
    from app.refinement.asr import TargetedASREngine

    result = _result(
        [
            {"start": 0.5, "end": 0.9, "word": "مرحبا", "probability": 0.8},
            {"start": 1.0, "end": 1.4, "word": "بك", "probability": 1.0},
        ]
    )
    engine = TargetedASREngine(options_for=_options_for, engine=FakeASREngine(result))

    targeted = engine.transcribe(
        Path("window.wav"),
        context_start=100.0,
        context_end=102.0,
        priority=RefinementPriority.CANDIDATE,
    )

    assert [word.start for word in targeted.word_timestamps] == [100.5, 101.0]
    assert [word.end for word in targeted.word_timestamps] == [100.9, 101.4]
    assert targeted.provider == "faster-whisper"
    assert targeted.model == "small"
    assert targeted.language == "ar"
    assert targeted.confidence == pytest.approx(0.9)
    assert targeted.rejected_reason is None
    assert targeted.fingerprint


def test_targeted_asr_drops_out_of_window_words() -> None:
    from app.refinement.asr import TargetedASREngine

    result = _result(
        [
            {"start": 0.5, "end": 0.9, "word": "مرحبا", "probability": 0.8},
            {"start": 5.0, "end": 5.4, "word": "خارجي", "probability": 0.7},
        ]
    )
    engine = TargetedASREngine(options_for=_options_for, engine=FakeASREngine(result))

    targeted = engine.transcribe(
        Path("window.wav"),
        context_start=100.0,
        context_end=102.0,
        priority=RefinementPriority.CANDIDATE,
    )

    assert len(targeted.word_timestamps) == 1
    assert targeted.word_timestamps[0].text == "مرحبا"
    assert targeted.rejected_reason is not None


class CountingLease:
    def __init__(self, factory: CountingLeaseFactory, ownership_lost: bool = False) -> None:
        self._factory = factory
        self.ownership_lost = ownership_lost

    def __enter__(self) -> CountingLease:
        self._factory.enters += 1
        return self

    def __exit__(self, *args: object) -> bool:
        self._factory.releases += 1
        return False


class CountingLeaseFactory:
    def __init__(self, ownership_lost: bool = False) -> None:
        self._ownership_lost = ownership_lost
        self.acquires = 0
        self.enters = 0
        self.releases = 0
        self.purposes: list[str] = []

    def acquire(self, *, purpose: str, on_ownership_lost: object = None) -> CountingLease:
        self.acquires += 1
        self.purposes.append(purpose)
        return CountingLease(self, self._ownership_lost)


def test_targeted_asr_releases_lease_when_engine_raises() -> None:
    from app.refinement.asr import TargetedASREngine

    factory = CountingLeaseFactory()
    engine = TargetedASREngine(
        options_for=_options_for, engine=RaisingASREngine(), lease_factory=factory
    )

    with pytest.raises(RuntimeError, match="boom"):
        engine.transcribe(
            Path("window.wav"),
            context_start=100.0,
            context_end=102.0,
            priority=RefinementPriority.CANDIDATE,
        )

    assert factory.acquires == 1
    assert factory.enters == 1
    assert factory.releases == 1
    assert factory.purposes == ["targeted-asr"]


def test_targeted_asr_raises_when_lease_ownership_is_lost() -> None:
    from app.refinement.asr import TargetedASREngine, TargetedASRError

    factory = CountingLeaseFactory(ownership_lost=True)
    result = _result([{"start": 0.5, "end": 0.9, "word": "مرحبا", "probability": 0.8}])
    engine = TargetedASREngine(
        options_for=_options_for, engine=FakeASREngine(result), lease_factory=factory
    )

    with pytest.raises(TargetedASRError, match="lease"):
        engine.transcribe(
            Path("window.wav"),
            context_start=100.0,
            context_end=102.0,
            priority=RefinementPriority.CANDIDATE,
        )

    assert factory.releases == 1


def test_targeted_asr_applies_context_terms_without_mutating_options() -> None:
    from app.refinement.asr import TargetedASREngine

    original = _options_for(RefinementPriority.CANDIDATE)
    fake_engine = FakeASREngine(_result([]))
    engine = TargetedASREngine(options_for=lambda _p: original, engine=fake_engine)

    engine.transcribe(
        Path("window.wav"),
        context_start=100.0,
        context_end=102.0,
        priority=RefinementPriority.CANDIDATE,
        context_terms=("خلي بالك", "المشروع"),
    )

    assert original.hotwords is None
    assert fake_engine.last_options is not None
    assert fake_engine.last_options.hotwords == "خلي بالك المشروع"
