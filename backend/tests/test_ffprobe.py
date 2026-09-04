from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.media.ffprobe import FFprobe, ProbeParseError, parse_ffprobe_json


def test_parse_ffprobe_json_extracts_typed_media_metadata() -> None:
    payload = {
        "format": {"duration": "12.5"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
            },
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
        ],
    }

    metadata = parse_ffprobe_json(json.dumps(payload))

    assert metadata.duration_seconds == 12.5
    assert metadata.video_codec == "h264"
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.frames_per_second == pytest.approx(30000 / 1001)
    assert metadata.audio_codec == "aac"
    assert metadata.audio_sample_rate == 48000


@pytest.mark.parametrize(
    "payload",
    ["not json", '{"format": {"duration": "1"}, "streams": []}'],
)
def test_parse_ffprobe_json_rejects_malformed_or_invalid_payloads(payload: str) -> None:
    with pytest.raises(ProbeParseError):
        parse_ffprobe_json(payload)


def test_parse_ffprobe_json_rejects_zero_frame_rate_denominator() -> None:
    payload = {
        "format": {"duration": "1"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1,
                "height": 1,
                "avg_frame_rate": "1/0",
            }
        ],
    }

    with pytest.raises(ProbeParseError):
        parse_ffprobe_json(json.dumps(payload))


@pytest.mark.parametrize("duration", ["inf", "-inf", "nan"])
def test_parse_ffprobe_json_rejects_non_finite_numbers(duration: str) -> None:
    payload = {
        "format": {"duration": duration},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1,
                "height": 1,
                "avg_frame_rate": "24/1",
            }
        ],
    }

    with pytest.raises(ProbeParseError):
        parse_ffprobe_json(json.dumps(payload))


def test_ffprobe_uses_safe_argument_vector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media_path = tmp_path / "video.mp4"
    media_path.write_bytes(b"synthetic")
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args[0], kwargs))
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(
                {
                    "format": {"duration": "1"},
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 1,
                            "height": 1,
                            "avg_frame_rate": "24/1",
                        }
                    ],
                }
            ),
        )

    monkeypatch.setattr("app.media.ffprobe.subprocess.run", fake_run)

    FFprobe(binary="ffprobe-test").probe(media_path)

    command, keyword_arguments = calls[0]
    assert isinstance(command, list)
    assert command[0] == "ffprobe-test"
    assert keyword_arguments == {"check": True, "capture_output": True, "text": True}
