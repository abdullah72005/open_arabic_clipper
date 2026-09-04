from pathlib import Path

from app.media.analysis import SilenceInterval, parse_silencedetect


def test_parse_silencedetect_pairs_start_end_and_duration() -> None:
    """FFmpeg silence logs become timestamped intervals for clip-boundary work."""

    parsed = parse_silencedetect(
        "[silencedetect] silence_start: 1.0\n"
        "[silencedetect] silence_end: 2.5 | silence_duration: 1.5"
    )

    assert parsed == [SilenceInterval(start=1.0, end=2.5, duration=1.5)]


def test_windowed_rms_reads_cached_pcm_audio(tmp_path: Path) -> None:
    import wave

    from app.media.analysis import windowed_rms

    audio_path = tmp_path / "speech.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes((1000).to_bytes(2, "little", signed=True) * 16_000)

    assert windowed_rms(audio_path)[0]["rms"] == 1000.0
