from dataclasses import replace
from pathlib import Path

from sqlalchemy.orm import Session

from app.transcription.service import TranscriptionOptions
from app.workers.celery_app import celery_app


class FakeWord:
    start = 0.0
    end = 0.2
    word = "أهلا"
    probability = 0.9


class FakeSegment:
    start = 0.0
    end = 1.25
    text = "أهلا"
    tokens = [50364, 1234, 50414]
    avg_logprob = -0.1
    compression_ratio = 1.17
    no_speech_prob = 0.01
    temperature = 0.2
    words = [FakeWord()]


class FakeInfo:
    language = "ar"
    language_probability = 0.97
    duration = 0.8


class FakeModel:
    def transcribe(self, path: str, **kwargs: object) -> tuple[list[FakeSegment], FakeInfo]:
        assert path == "speech.wav"
        assert kwargs["word_timestamps"] is True
        return [FakeSegment()], FakeInfo()


def test_transcription_fingerprint_changes_for_material_settings_only() -> None:
    """Cache keys change when speech-recognition output can change."""

    options = TranscriptionOptions(model="small", device="cpu", compute_type="int8", beam_size=5)

    assert options.fingerprint("a" * 64) == options.fingerprint("a" * 64)
    assert options.fingerprint("a" * 64) != replace(options, beam_size=1).fingerprint("a" * 64)


def test_transcription_options_include_forced_language_in_cache_key() -> None:
    """Forced-language output cannot reuse an auto-detected transcript."""

    automatic = TranscriptionOptions("small", "cpu", "int8", 5)
    forced_arabic = replace(automatic, language="ar")

    assert automatic.fingerprint("b" * 64) != forced_arabic.fingerprint("b" * 64)


def test_celery_routes_transcription_work_to_dedicated_queue() -> None:
    """Large model work cannot compete with ordinary media tasks by default."""

    route = celery_app.conf.task_routes

    assert route("clipfactory.run_pipeline_stage", ("id", "TRANSCRIPTION"), {}, {}) == {
        "queue": "transcription"
    }
    assert route("clipfactory.run_pipeline_stage", ("id", "INGEST"), {}, {}) is None
    assert route("clipfactory.run_pipeline_stage", ("id", "PROBE"), {}, {}) is None
    assert route("clipfactory.worker_heartbeat", (), {}, {}) is None


def test_engine_falls_back_to_cpu_int8_and_preserves_word_timestamps() -> None:
    """Auto device selection remains usable without CUDA and retains Whisper evidence."""

    from app.transcription.engine import WhisperEngine

    created: list[tuple[str, str, str]] = []

    def model_factory(model: str, device: str, compute_type: str) -> FakeModel:
        created.append((model, device, compute_type))
        return FakeModel()

    result = WhisperEngine(model_factory=model_factory, cuda_available=lambda: False).transcribe(
        Path("speech.wav"), TranscriptionOptions("small", "auto", "auto", 5)
    )

    assert created == [("small", "cpu", "int8")]
    assert result.language == "ar"
    assert result.language_probability == 0.97
    assert result.segments[0]["tokens"] == [50364, 1234, 50414]
    assert result.segments[0]["compression_ratio"] == 1.17
    assert result.segments[0]["temperature"] == 0.2
    assert result.segments[0]["start"] == 0.0
    assert result.segments[0]["end"] == 1.25
    assert result.segments[0]["words"][0]["start"] == 0.0


def test_engine_passes_explicit_safe_decoding_options() -> None:
    """Optional Whisper controls are explicit, output-affecting, and still local."""

    from app.transcription.engine import WhisperEngine

    received: dict[str, object] = {}

    class RecordingModel:
        def transcribe(self, _path: str, **kwargs: object) -> tuple[list[FakeSegment], FakeInfo]:
            received.update(kwargs)
            return [FakeSegment()], FakeInfo()

    WhisperEngine(
        model_factory=lambda *_args: RecordingModel(), cuda_available=lambda: False
    ).transcribe(
        Path("speech.wav"),
        TranscriptionOptions(
            "small",
            "cpu",
            "int8",
            5,
            temperature=(0.0, 0.4),
            condition_on_previous_text=False,
            vad_filter=True,
            initial_prompt="مصري",
            hotwords="خلي بالك",
        ),
    )

    assert received["temperature"] == (0.0, 0.4)
    assert received["condition_on_previous_text"] is False
    assert received["vad_filter"] is True
    assert received["initial_prompt"] == "مصري"
    assert received["hotwords"] == "خلي بالك"


def test_transcription_executor_persists_raw_timestamped_result(
    sqlite_engine: object, tmp_path: Path
) -> None:
    """Worker-stage execution stores raw evidence and detected language for reuse."""

    from app.db.base import Base
    from app.models import AudioArtifact, SourceVideo
    from app.pipeline.stages import TranscriptionExecutor
    from app.transcription.engine import TranscriptionResult

    class FakeEngine:
        def transcribe(self, path: Path, options: TranscriptionOptions) -> TranscriptionResult:
            return TranscriptionResult(
                language="ar",
                language_probability=0.9,
                raw_text="أهلا hello",
                duration=1.0,
                segments=[{"start": 0.0, "end": 1.0, "text": "أهلا hello", "words": []}],
                word_segments=[],
            )

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri=str(tmp_path / "source.mp4"), content_hash="source")
        session.add(source)
        session.flush()
        audio_path = tmp_path / "speech.wav"
        audio_path.write_bytes(b"wav")
        artifact = AudioArtifact(
            source_video_id=source.id,
            output_path=str(audio_path),
            content_hash="audio",
            source_content_hash="source",
            sample_rate=16000,
            duration=1.0,
        )
        session.add(artifact)
        session.commit()

        transcript = TranscriptionExecutor(session=session, engine=FakeEngine()).execute(source)

        assert transcript.language == "ar"
        assert transcript.segments[0]["start"] == 0.0


def test_normalization_preserves_egyptian_arabic_and_embedded_english() -> None:
    """Display cleanup removes noise without translating dialect or Latin tokens."""

    from app.transcription.normalization import normalize_transcript

    assert normalize_transcript("  أنا  okay\n\nأنا  ") == "أنا okay أنا"


def test_normalization_and_audio_analysis_persist_reusable_signals(
    sqlite_engine: object, tmp_path: Path
) -> None:
    from app.db.base import Base
    from app.models import AudioArtifact, SourceVideo, Transcript
    from app.pipeline.stages import AudioAnalysisExecutor, TranscriptNormalizationExecutor
    from app.services.storage import StorageCategory, StorageService

    Base.metadata.create_all(sqlite_engine)
    storage = StorageService(tmp_path / "storage")
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri=str(tmp_path / "source.mp4"), content_hash="source")
        session.add(source)
        session.flush()
        artifact_path = storage.resolve(StorageCategory.SOURCES, f"{source.id}/speech-analysis.wav")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"wav")
        session.add(
            AudioArtifact(
                source_video_id=source.id,
                output_path=f"{source.id}/speech-analysis.wav",
                content_hash="audio",
                sample_rate=16_000,
                duration=10.0,
            )
        )
        session.add(
            Transcript(
                source_video_id=source.id,
                whisper_model="small",
                input_fingerprint="fingerprint",
                raw_text="  أهلا  hello\n",
                segments=[{"start": 0.0, "end": 2.0, "text": "  أهلا  hello "}],
                word_segments=[{"start": 0.0, "end": 0.2, "word": "أهلا"}],
                duration=10.0,
            )
        )
        session.commit()

        transcript = TranscriptNormalizationExecutor(session=session).execute(source)
        analysis = AudioAnalysisExecutor(
            session=session,
            storage=storage,
            command_runner=lambda _args: (
                "silence_start: 2\\nsilence_end: 4 | silence_duration: 2\\n"
            ),
        ).execute(source)

        assert transcript.normalized_text == "أهلا hello"
        assert transcript.segments[0]["normalized_text"] == "أهلا hello"
        assert transcript.chunks[0].segment_indexes == [0]
        assert analysis.silence_ratio == 0.2
        assert analysis.speech_density == 0.8


def test_normalization_persists_raw_corrected_final_text_and_timestamp(
    sqlite_engine: object, tmp_path: Path
) -> None:
    """Stage 2.5 persists derived correction state without altering ASR timestamps."""

    from app.db.base import Base
    from app.models import SourceVideo, Transcript
    from app.pipeline.stages import TranscriptNormalizationExecutor

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri=str(tmp_path / "source.mp4"), content_hash="source")
        session.add(source)
        session.flush()
        session.add(
            Transcript(
                source_video_id=source.id,
                whisper_model="small",
                input_fingerprint="fingerprint",
                raw_text="خطي بالك",
                normalized_text="خطي بالك",
                segments=[{"start": 0.0, "end": 1.0, "text": "خطي بالك", "words": []}],
                word_segments=[],
                duration=1.0,
            )
        )
        session.commit()

        transcript = TranscriptNormalizationExecutor(session=session).execute(source)

        assert transcript.raw_text == "خطي بالك"
        assert transcript.corrected_text == "خلي بالك"
        assert transcript.final_text == "خلي بالك"
        assert transcript.segments[0]["raw_text"] == "خطي بالك"
        assert transcript.segments[0]["corrected_text"] == "خلي بالك"
        assert transcript.segments[0]["final_text"] == "خلي بالك"
        assert transcript.segments[0]["correction_applied"] is True
        assert transcript.segments[0]["start"] == 0.0
        assert transcript.segments[0]["end"] == 1.0
        assert transcript.corrected_segment_ratio == 1.0
        assert transcript.uncertain_segment_ratio == 0.0


def test_reconstruction_persists_derived_text_without_replacing_prior_evidence(
    sqlite_engine: object, tmp_path: Path
) -> None:
    """Stage 2.7 keeps ASR, correction, and manual values separate from final display text."""

    from app.db.base import Base
    from app.models import SourceVideo, Transcript
    from app.pipeline.stages import ContextualReconstructionExecutor
    from app.transcription.reconstruction import ContextualReconstructor

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri=str(tmp_path / "source.mp4"), content_hash="source")
        session.add(source)
        session.flush()
        session.add(
            Transcript(
                source_video_id=source.id,
                whisper_model="small",
                input_fingerprint="fingerprint",
                raw_text="خطي بالك يا صحبي",
                corrected_text="خلي بالك يا صاحبي",
                segments=[
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "خطي بالك",
                        "raw_text": "خطي بالك",
                        "corrected_text": "خلي بالك",
                    },
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "text": "يا صحبي",
                        "raw_text": "يا صحبي",
                        "corrected_text": "يا صاحبي",
                        "operator_text": "يا صديقي",
                    },
                ],
                word_segments=[],
                duration=2.0,
            )
        )
        session.commit()

        transcript = ContextualReconstructionExecutor(
            session=session, reconstructor=ContextualReconstructor(provider=None)
        ).execute(source)

        assert transcript.raw_text == "خطي بالك يا صحبي"
        assert transcript.corrected_text == "خلي بالك يا صاحبي"
        assert transcript.contextual_reconstructed_text == "خلي بالك يا صاحبي"
        assert transcript.final_text == "خلي بالك يا صديقي"
        assert transcript.segments[0]["raw_text"] == "خطي بالك"
        assert transcript.segments[0]["corrected_text"] == "خلي بالك"
        assert transcript.segments[0]["contextual_reconstructed_text"] == "خلي بالك"
        assert transcript.segments[1]["final_text"] == "يا صديقي"
        assert transcript.reconstruction_method == "stage2_5_fallback"
        assert transcript.chunks[0].text == "خلي بالك يا صديقي"


def test_benchmark_reports_actual_transcription_throughput(tmp_path: Path) -> None:
    from app.transcription.benchmark import benchmark_transcription
    from app.transcription.engine import TranscriptionResult

    class FixedEngine:
        def resolved_hardware(self, _options: TranscriptionOptions) -> tuple[str, str]:
            return "cpu", "int8"

        def transcribe(self, _path: Path, _options: TranscriptionOptions) -> TranscriptionResult:
            return TranscriptionResult(
                language="ar",
                language_probability=0.9,
                raw_text="أهلا",
                segments=[],
                word_segments=[],
                duration=30.0,
            )

    report = benchmark_transcription(
        tmp_path / "sample.wav", FixedEngine(), TranscriptionOptions("small", "cpu", "int8", 5)
    )

    assert report.source_audio_seconds == 30.0
    assert report.device == "cpu"
    assert report.compute_type == "int8"
    assert report.real_time_factor >= 0.0
    assert report.audio_minutes_per_wall_minute > 0.0


def test_ingest_and_probe_executors_validate_local_source(tmp_path: Path) -> None:
    from app.models import SourceVideo
    from app.pipeline.stages import IngestExecutor, ProbeExecutor

    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"media")
    source = SourceVideo(source_uri=str(source_path))

    class RecordingProbe:
        def __init__(self) -> None:
            self.paths: list[Path] = []

        def probe(self, path: Path) -> object:
            self.paths.append(path)
            return object()

    probe = RecordingProbe()
    assert IngestExecutor().execute(source) is source
    assert ProbeExecutor(probe).execute(source) is not None
    assert probe.paths == [source_path]


def test_chunking_uses_segment_boundaries_and_neighbor_context() -> None:
    """Later analysis gets coherent timestamp ranges rather than character slices."""

    from app.transcription.chunking import ChunkConfig, build_chunks

    segments = [
        {"start": 0.0, "end": 10.0, "text": "First sentence."},
        {"start": 10.0, "end": 20.0, "text": "Second sentence."},
        {"start": 20.0, "end": 30.0, "text": "Third sentence."},
    ]
    chunks = build_chunks(segments, ChunkConfig(target_seconds=20, context_segments=1))

    assert chunks[0].segment_indexes == [0, 1]
    assert chunks[0].following_context == "Third sentence."
