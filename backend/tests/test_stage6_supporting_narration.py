"""Stage 6 supporting-asset rights gating and narration validation."""

from __future__ import annotations

from typing import cast

import pytest

from app.materialization.narration import (
    FakeNarrationSynthesizer,
    inspect_wav,
    narration_requests,
    transcription_matches,
    validate_narration_audio,
    wrap_pcm_as_wav,
)
from app.materialization.policy import DEFAULT_CONFIG
from app.materialization.supporting import (
    MANIFEST_SCHEMA_VERSION,
    SupportingAsset,
    SupportingNeed,
    parse_manifest,
    select_supporting_asset,
)
from app.materialization.types import NarrationSynthesisRequest


def _asset(**overrides: object) -> SupportingAsset:
    defaults: dict[str, object] = {
        "asset_id": "asset-1",
        "relative_path": "supporting/asset-1.mp4",
        "sha256": "abc",
        "kind": "ABSTRACT_LOOP",
        "duration_seconds": 8.0,
        "width": 1920,
        "height": 1080,
        "owner": "studio",
        "creator": "studio",
        "origin": "internal",
        "license": "internal",
        "permission": "granted",
        "allowed_uses": ("background",),
        "required_attribution": "",
        "restrictions": (),
        "suitability_tags": ("calm",),
        "rights": "EXPLICIT_COMPATIBLE",
    }
    defaults.update(overrides)
    return SupportingAsset(**defaults)  # type: ignore[arg-type]


def test_manifest_requires_schema_and_identity() -> None:
    with pytest.raises(ValueError):
        parse_manifest({"assets": []})
    with pytest.raises(ValueError):
        parse_manifest({"schema_version": MANIFEST_SCHEMA_VERSION, "assets": [{"asset_id": "x"}]})
    assets = parse_manifest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "assets": [
                {
                    "asset_id": "a1",
                    "relative_path": "supporting/a1.mp4",
                    "sha256": "deadbeef",
                    "rights": "EXPLICIT_COMPATIBLE",
                }
            ],
        }
    )
    assert assets[0].usable


def test_off_mode_declines_with_reason() -> None:
    selection = select_supporting_asset(
        "OFF", "AUTO", SupportingNeed(block_index=1, content_type="STORY"), [_asset()]
    )
    assert selection.resolution == "DECLINED"
    assert selection.reason_code == "SUPPORTING_DISABLED"


def test_unknown_rights_asset_is_never_selected() -> None:
    selection = select_supporting_asset(
        "ON",
        "AUTO",
        SupportingNeed(block_index=1, content_type="STORY", required=True),
        [_asset(rights="UNKNOWN")],
    )
    assert selection.resolution == "DECLINED"
    assert selection.reason_code == "NO_RIGHTS_CLEARED_ASSET"


def test_auto_selects_rights_cleared_asset_deterministically() -> None:
    assets = [
        _asset(asset_id="b", suitability_tags=("calm",)),
        _asset(asset_id="a", suitability_tags=("calm", "STORY")),
    ]
    selection = select_supporting_asset(
        "AUTO", "AUTO", SupportingNeed(block_index=2, content_type="STORY"), assets
    )
    assert selection.resolution == "FULFILLED"
    assert selection.asset_id == "a"


def test_auto_declines_when_usefulness_uncertain() -> None:
    selection = select_supporting_asset("AUTO", "AUTO", SupportingNeed(block_index=2), [_asset()])
    assert selection.resolution == "DECLINED"
    assert selection.reason_code == "UNCERTAIN_USEFULNESS"


def test_distraction_risk_declines_optional_asset() -> None:
    selection = select_supporting_asset(
        "AUTO",
        "AUTO",
        SupportingNeed(block_index=2, content_type="STORY", readability_priority="HIGH"),
        [_asset(suitability_tags=("busy",))],
    )
    assert selection.resolution == "DECLINED"
    assert selection.reason_code == "DISTRACTION_RISK"


def test_wav_roundtrip_and_silence_detection() -> None:
    import struct

    frames = 24_000
    pcm = struct.pack(f"<{frames}h", *([1000] * frames))
    wav = wrap_pcm_as_wav(pcm, sample_rate=24_000, channels=1)
    info = inspect_wav(wav)
    assert info.sample_rate == 24_000
    assert info.channels == 1
    assert abs(info.duration_seconds - 1.0) < 1e-3
    assert not info.silent
    validation = validate_narration_audio(info, expected_text="hello", config=DEFAULT_CONFIG)
    assert validation["ok"] is True

    silence = wrap_pcm_as_wav(b"\x00" * 4800, sample_rate=24_000, channels=1)
    silent_info = inspect_wav(silence)
    assert silent_info.silent
    silent_validation = validate_narration_audio(
        silent_info, expected_text="hello", config=DEFAULT_CONFIG
    )
    assert "SILENT_AUDIO" in cast("list[str]", silent_validation["findings"])


def test_fake_synthesizer_produces_valid_lossless_audio() -> None:
    synth = FakeNarrationSynthesizer()
    request = NarrationSynthesisRequest(
        chunk_id="c1",
        block_index=1,
        text="The mechanism is mentorship loss",
        provider="gemini",
        model="gemini-3.8-flash-tts",
        voice="Kore",
    )
    audio = synth.synthesize(request)
    assert audio.container == "wav"
    assert audio.sample_rate == 24_000
    assert audio.provider_identity["voice"] == "Kore"
    assert len(synth.calls) == 1


def test_narration_requests_are_bounded_and_skip_empty_copy() -> None:
    requests = narration_requests(
        [
            {"block_index": 1, "copy": "hello"},
            {"block_index": 2, "copy": "   "},
            {"block_index": 3, "copy": "world"},
        ],
        identity={"provider": "gemini", "model": "m", "voice": "Kore"},
        config=DEFAULT_CONFIG,
    )
    assert [request.block_index for request in requests] == [1, 3]


def test_transcription_match_detects_negation_and_names() -> None:
    assert transcription_matches("John did not agree", "john did not agree")
    assert not transcription_matches("John did not agree", "john agreed completely")
