import json
from pathlib import Path

import pytest

from app.services.storage import StorageCategory, StorageService
from app.transcription.engine import TranscriptionResult
from app.transcription.reconstruction.capture import (
    ASRCapture,
    CaptureValidationError,
    build_capture,
    build_decoder_identity,
    capture_hash,
    load_capture,
    parse_capture,
    save_capture,
    serialize_capture,
    verify_replay_provenance,
)
from app.transcription.service import TranscriptionOptions


def _options() -> TranscriptionOptions:
    return TranscriptionOptions(
        model="large-v3-turbo",
        device="cpu",
        compute_type="int8",
        beam_size=5,
        language="ar",
    )


def _result(clip_id: str) -> TranscriptionResult:
    return TranscriptionResult(
        language="ar",
        language_probability=0.99,
        raw_text=f"raw-{clip_id}",
        duration=30.0,
        segments=[
            {
                "start": 0.0,
                "end": 1.0,
                "text": f"raw-{clip_id}-0",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.0,
                "words": [{"start": 0.0, "end": 0.5, "word": "x", "probability": 0.9}],
            },
            {"start": 1.0, "end": 2.0, "text": f"raw-{clip_id}-1", "words": []},
        ],
        word_segments=[],
    )


def _capture() -> ASRCapture:
    return build_capture(
        capture_id="cap-1",
        clips=[("clip-0", "source-a", 0.0, 30.0), ("clip-1", "source-a", 30.0, 60.0)],
        clip_hashes={"clip-0": "clip0hash", "clip-1": "clip1hash"},
        source_hashes={"source-a": "abc123"},
        results={"clip-0": _result("clip-0"), "clip-1": _result("clip-1")},
        options=_options(),
        wall_clock_seconds=10.0,
    )


def _mutated(**changes: object) -> ASRCapture:
    payload = json.loads(serialize_capture(_capture()))
    payload.update(changes)
    return parse_capture(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def test_capture_roundtrip_preserves_content_and_hash() -> None:
    capture = _capture()
    restored = parse_capture(serialize_capture(capture))
    assert capture_hash(restored) == capture_hash(capture)
    assert restored.clips[0].segments[0]["text"] == "raw-clip-0-0"
    assert restored.decoder.whisper_model == "large-v3-turbo"
    assert restored.decoder.faster_whisper_version


def test_build_decoder_identity_captures_output_affecting_options() -> None:
    identity = build_decoder_identity(_options())
    assert identity.whisper_model == "large-v3-turbo"
    assert identity.device == "cpu"
    assert identity.beam_size == 5
    assert identity.language == "ar"
    assert identity.word_timestamps is True


def test_capture_hash_changes_when_raw_text_changes() -> None:
    base = _capture()
    changed = _mutated()
    changed.clips[0].segments[0]["text"] = "different raw"  # type: ignore[index]
    assert capture_hash(changed) != capture_hash(base)


def test_capture_hash_changes_when_decoder_model_changes() -> None:
    base = _capture()
    payload = json.loads(serialize_capture(base))
    payload["decoder"]["whisper_model"] = "large-v3"
    changed = parse_capture(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    assert capture_hash(changed) != capture_hash(base)


def test_capture_hash_is_stable_for_identical_content() -> None:
    assert capture_hash(_capture()) == capture_hash(_capture())


def test_capture_rejects_unknown_schema_version() -> None:
    with pytest.raises(CaptureValidationError, match="schema version"):
        _mutated(schema_version="asr-capture-v2")


def test_capture_rejects_duplicate_clip_ids() -> None:
    payload = json.loads(serialize_capture(_capture()))
    payload["clips"].append(payload["clips"][0])
    with pytest.raises(CaptureValidationError, match="duplicate clip"):
        parse_capture(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def test_capture_rejects_missing_source_hash() -> None:
    payload = json.loads(serialize_capture(_capture()))
    payload["clips"][0]["source_hash"] = ""
    with pytest.raises(CaptureValidationError, match="source hash"):
        parse_capture(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def test_capture_rejects_inverted_timestamps() -> None:
    payload = json.loads(serialize_capture(_capture()))
    payload["clips"][0]["segments"][0]["start"] = 2.0
    payload["clips"][0]["segments"][0]["end"] = 1.0
    with pytest.raises(CaptureValidationError, match="inverted"):
        parse_capture(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def test_capture_rejects_overlapping_timestamps() -> None:
    payload = json.loads(serialize_capture(_capture()))
    payload["clips"][0]["segments"][1]["start"] = 0.5
    with pytest.raises(CaptureValidationError, match="overlaps"):
        parse_capture(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def test_capture_rejects_missing_clip_segments() -> None:
    payload = json.loads(serialize_capture(_capture()))
    payload["clips"][0]["segments"] = []
    with pytest.raises(CaptureValidationError, match="no raw segments"):
        parse_capture(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def test_save_and_load_capture_roundtrip_via_storage(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    capture = _capture()

    path = save_capture(storage, capture, name="cap-1")
    loaded = load_capture(storage, "cap-1")

    assert capture_hash(loaded) == capture_hash(capture)
    assert path.is_relative_to(storage.category_root(StorageCategory.BENCHMARKS))


def _provenance_kwargs() -> dict[str, object]:
    return {
        "clip_id": "clip-0",
        "source_id": "source-a",
        "source_hash": "abc123",
        "clip_hash": "clip0hash",
        "start_seconds": 0.0,
        "end_seconds": 30.0,
        "expected_decoder": build_decoder_identity(_options()),
    }


def test_replay_provenance_accepts_exact_matching_identity() -> None:
    verify_replay_provenance(_capture(), **_provenance_kwargs())  # type: ignore[arg-type]


def test_replay_provenance_rejects_wrong_source_id() -> None:
    kwargs = _provenance_kwargs()
    kwargs["source_id"] = "source-b"
    with pytest.raises(CaptureValidationError, match="source"):
        verify_replay_provenance(_capture(), **kwargs)  # type: ignore[arg-type]


def test_replay_provenance_rejects_wrong_original_media_hash() -> None:
    kwargs = _provenance_kwargs()
    kwargs["source_hash"] = "different"
    with pytest.raises(CaptureValidationError, match="original-media hash"):
        verify_replay_provenance(_capture(), **kwargs)  # type: ignore[arg-type]


def test_replay_provenance_rejects_wrong_clip_audio_hash() -> None:
    kwargs = _provenance_kwargs()
    kwargs["clip_hash"] = "different"
    with pytest.raises(CaptureValidationError, match="clip audio hash"):
        verify_replay_provenance(_capture(), **kwargs)  # type: ignore[arg-type]


def test_replay_provenance_rejects_wrong_start_or_end_bounds() -> None:
    kwargs = _provenance_kwargs()
    kwargs["start_seconds"] = 5.0
    with pytest.raises(CaptureValidationError, match="bounds"):
        verify_replay_provenance(_capture(), **kwargs)  # type: ignore[arg-type]

    kwargs = _provenance_kwargs()
    kwargs["end_seconds"] = 40.0
    with pytest.raises(CaptureValidationError, match="bounds"):
        verify_replay_provenance(_capture(), **kwargs)  # type: ignore[arg-type]


def test_replay_provenance_rejects_missing_clip_in_capture() -> None:
    kwargs = _provenance_kwargs()
    kwargs["clip_id"] = "clip-9"
    with pytest.raises(CaptureValidationError, match="missing or duplicated"):
        verify_replay_provenance(_capture(), **kwargs)  # type: ignore[arg-type]


def test_replay_provenance_rejects_decoder_identity_drift() -> None:
    import dataclasses

    changed = dataclasses.replace(_options(), model="large-v3")
    kwargs = _provenance_kwargs()
    kwargs["expected_decoder"] = build_decoder_identity(changed)
    with pytest.raises(CaptureValidationError, match="decoder identity"):
        verify_replay_provenance(_capture(), **kwargs)  # type: ignore[arg-type]


def test_replay_provenance_rejects_unsupported_schema_version() -> None:
    kwargs = _provenance_kwargs()
    base = _capture()
    altered = ASRCapture(
        schema_version="asr-capture-v2",
        capture_id=base.capture_id,
        decoder=base.decoder,
        clips=base.clips,
        wall_clock_seconds=base.wall_clock_seconds,
        memory_snapshots=base.memory_snapshots,
    )
    with pytest.raises(CaptureValidationError, match="schema"):
        verify_replay_provenance(altered, **kwargs)  # type: ignore[arg-type]


def test_multi_clip_same_source_keeps_distinct_clip_hashes_and_bounds() -> None:
    """Multiple clips from one source never share or overwrite clip identity."""

    capture = _capture()
    first, second = capture.clips

    assert first.clip_id == "clip-0"
    assert second.clip_id == "clip-1"
    assert first.source_id == second.source_id == "source-a"
    assert first.source_hash == second.source_hash == "abc123"
    assert first.clip_hash == "clip0hash"
    assert second.clip_hash == "clip1hash"
    assert first.clip_hash != second.clip_hash
    assert first.start_seconds == 0.0 and first.end_seconds == 30.0
    assert second.start_seconds == 30.0 and second.end_seconds == 60.0
