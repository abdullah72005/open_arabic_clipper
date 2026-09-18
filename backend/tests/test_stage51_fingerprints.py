"""Stage 5.1 fingerprint determinism, namespace separation, and boundaries."""

from __future__ import annotations

from app.composition.fingerprints import (
    analysis_fingerprint,
    ass_fingerprint,
    build_stage51_input_payload,
    caption_plan_fingerprint,
    framing_fingerprint,
    source_media_identity_fingerprint,
    visual_composition_input_fingerprint,
    visual_composition_output_fingerprint,
)
from app.composition.policy import DETECTOR_IDENTITY, Stage51Config, stage51_config_payload


def _base(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": "candidate-1",
        "candidate_key": "key-1",
        "candidate_is_current": True,
        "disposition": "CANDIDATE",
        "analysis_fingerprint": "analysis-fp",
        "source_id": "source-1",
        "source_media_identity": {
            "source_id": "source-1",
            "content_hash": "",
            "relative_path": "sources/source-1/source.mp4",
            "size_bytes": 2048,
            "mtime_ns": 1,
        },
        "contract_input_fingerprint": "contract-in",
        "contract_output_fingerprint": "contract-out",
        "contract_status": "READY_FOR_RENDER_PLANNING",
        "contract_ready": True,
        "contract_is_current": True,
        "contract_live_freshness": "CURRENT",
        "contract_effective": True,
        "selected_plan_id": "plan-1",
        "selection_id": "selection-1",
        "final_refinement_id": "final-1",
        "display_geometry": {
            "encoded_width": 1920,
            "encoded_height": 1080,
            "rotation_degrees": 0,
            "display_width": 1920,
            "display_height": 1080,
            "display_aspect": 1.777778,
            "pixel_aspect_ratio": 1.0,
            "square_pixels_applied": True,
            "exotic_pixel_aspect": False,
        },
        "rotation_degrees": 0,
        "bound_spans": [
            {
                "block_index": 0,
                "start": 1.0,
                "end": 2.0,
                "word_start_index": 0,
                "word_end_index": 3,
                "source_role": "HERO",
                "is_hero": True,
            }
        ],
        "caption_source_fingerprint": "caption-fp",
        "caption_payload": {
            "final_transcript": "hello world",
            "word_timestamps": [{"index": 0, "text": "hello", "start": 1.0, "end": 1.5}],
            "dialect_profile": "EGYPTIAN",
            "dialect_confidence": 0.9,
            "code_switch_evidence": {"suspected": False, "tokens": []},
        },
        "output_profile": {
            "profile_key": "SHORTS_1080X1920",
            "width": 1080,
            "height": 1920,
        },
        "config": Stage51Config(),
    }
    payload.update(overrides)
    return payload


def _fp(**overrides: object) -> str:
    payload = build_stage51_input_payload(**_base(**overrides))  # type: ignore[arg-type]
    return visual_composition_input_fingerprint(payload)


def _payload(**overrides: object) -> dict[str, object]:
    return build_stage51_input_payload(**_base(**overrides))  # type: ignore[arg-type]


def test_input_fingerprint_is_deterministic() -> None:
    assert _fp() == _fp()


def test_fingerprints_are_deterministic() -> None:
    payload = _payload()
    assert visual_composition_output_fingerprint(payload) == (
        visual_composition_output_fingerprint(payload)
    )
    assert framing_fingerprint(payload) == framing_fingerprint(payload)
    assert ass_fingerprint(payload) == ass_fingerprint(payload)


def test_fingerprint_namespaces_are_separate() -> None:
    payload = _payload()
    fingerprints = {
        visual_composition_input_fingerprint(payload),
        visual_composition_output_fingerprint(payload),
        analysis_fingerprint(payload),
        framing_fingerprint(payload),
        caption_plan_fingerprint(payload),
        ass_fingerprint(payload),
        source_media_identity_fingerprint({"source_id": "source-1"}),
    }
    assert len(fingerprints) == 7


def test_built_payload_contains_required_sections() -> None:
    payload = _payload()
    assert set(payload) >= {
        "candidate",
        "source",
        "contract",
        "display",
        "spans",
        "caption",
        "output_profile",
        "safe_zone",
        "policy",
        "detector",
    }


def test_tts_publishing_codec_render_and_analytics_are_excluded() -> None:
    excluded = {
        "tts": {"provider": "gemini", "model": "tts-1", "voice": "narrator"},
        "publishing": {"title": "T", "schedule": "2026-01-01", "description": "d"},
        "codec": {"encoder": "libx264", "audio_codec": "aac"},
        "final_render": {"artifact_hash": "deadbeef", "mp4_path": "/tmp/out.mp4"},
        "analytics": {"enabled": True, "variant": "A"},
    }
    assert _fp() == _fp(excluded_runtime_context=excluded)
    assert _fp() == _fp(excluded_runtime_context={"tts": excluded["tts"]})


def test_built_payload_contains_no_excluded_runtime_keys() -> None:
    serialized = str(_payload(excluded_runtime_context={"tts": {"voice": "x"}})).casefold()
    for forbidden in ("tts", "voice", "speaker", "publish", "analytics", "artifact", "encoder"):
        assert forbidden not in serialized


def test_input_fingerprint_changes_with_source_media_identity() -> None:
    identity = dict(_base()["source_media_identity"])  # type: ignore[arg-type]
    identity["mtime_ns"] = 42
    assert _fp() != _fp(source_media_identity=identity)


def test_input_fingerprint_changes_with_contract_fingerprint() -> None:
    assert _fp() != _fp(contract_input_fingerprint="changed-in")
    assert _fp() != _fp(contract_output_fingerprint="changed-out")


def test_input_fingerprint_changes_with_caption_fingerprint() -> None:
    assert _fp() != _fp(caption_source_fingerprint="changed-caption")


def test_input_fingerprint_changes_with_contract_status() -> None:
    assert _fp() != _fp(contract_status="MATERIALIZATION_REQUIRED")
    assert _fp() != _fp(contract_live_freshness="STALE", contract_effective=False)


def test_input_fingerprint_changes_with_policy_version() -> None:
    changed = stage51_config_payload(Stage51Config())
    changed["policy_version"] = "stage5.1-v99"
    assert _fp() != _fp(stage51_config=changed)
    assert _fp() != _fp(framing_policy_version="stage5.1-framing-v99")
    assert _fp() != _fp(caption_layout_policy_version="stage5.1-caption-layout-v99")
    assert _fp() != _fp(ass_policy_version="stage5.1-ass-v99")


def test_input_fingerprint_changes_with_safe_zone_and_style() -> None:
    assert _fp() != _fp(safe_zone_key="GENERIC_CONSERVATIVE_SAFE_ZONE_V1")
    generic = stage51_config_payload(
        Stage51Config(safe_zone_profile_key="GENERIC_CONSERVATIVE_SAFE_ZONE_V1")
    )
    assert _fp() != _fp(stage51_config=generic)
    restyled = stage51_config_payload(Stage51Config(caption_font_family="Other Font"))
    assert _fp() != _fp(stage51_config=restyled)


def test_input_fingerprint_changes_with_detector_identity() -> None:
    changed = dict(DETECTOR_IDENTITY)
    changed["input_size"] = 320
    assert _fp() != _fp(detector_identity=changed)


def test_input_fingerprint_changes_with_bound_spans_and_rotation() -> None:
    spans = [
        {
            "block_index": 0,
            "start": 1.0,
            "end": 2.0,
            "word_start_index": 0,
            "word_end_index": 3,
            "source_role": "HERO",
            "is_hero": True,
        },
        {
            "block_index": 1,
            "start": 2.0,
            "end": 3.0,
            "word_start_index": 4,
            "word_end_index": 6,
            "source_role": "SUPPORT",
            "is_hero": False,
        },
    ]
    assert _fp() != _fp(bound_spans=spans)
    rotated = dict(_base()["display_geometry"])  # type: ignore[arg-type]
    rotated["rotation_degrees"] = 90
    rotated["display_width"], rotated["display_height"] = (
        rotated["display_height"],
        rotated["display_width"],
    )
    assert _fp() != _fp(display_geometry=rotated, rotation_degrees=90)
