from app.media.analysis import SilenceInterval, parse_silencedetect


def test_parse_silencedetect_pairs_start_end_and_duration() -> None:
    """FFmpeg silence logs become timestamped intervals for clip-boundary work."""

    parsed = parse_silencedetect(
        "[silencedetect] silence_start: 1.0\n"
        "[silencedetect] silence_end: 2.5 | silence_duration: 1.5"
    )

    assert parsed == [SilenceInterval(start=1.0, end=2.5, duration=1.5)]
