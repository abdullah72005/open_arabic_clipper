from dataclasses import replace

from app.transcription.service import TranscriptionOptions
from app.workers.celery_app import celery_app


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
