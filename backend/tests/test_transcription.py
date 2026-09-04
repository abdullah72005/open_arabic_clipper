from dataclasses import replace
from pathlib import Path

from app.transcription.service import TranscriptionOptions
from app.workers.celery_app import celery_app


class FakeWord:
    start = 0.0
    end = 0.2
    word = "أهلا"
    probability = 0.9


class FakeSegment:
    start = 0.0
    end = 0.8
    text = "أهلا"
    avg_logprob = -0.1
    no_speech_prob = 0.01
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

    assert celery_app.conf.task_routes["clipfactory.run_transcription"] == {
        "queue": "transcription"
    }


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
    assert result.segments[0]["words"][0]["start"] == 0.0
