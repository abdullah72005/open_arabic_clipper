"""Stage 5.0 fingerprint separation and invalidation boundaries."""

from __future__ import annotations

from app.render.fingerprints import (
    build_render_contract_input_payload,
    render_contract_input_fingerprint,
)
from app.render.policy import Stage50Config, stage50_config_payload


def _payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "candidate_id": "candidate-1",
        "candidate_key": "key-1",
        "candidate_is_current": True,
        "disposition": "CANDIDATE",
        "analysis_fingerprint": "analysis-fp",
        "source_id": "source-1",
        "source_media_identity": {
            "source_id": "source-1",
            "content_hash": "hash",
            "relative_path": "sources/source-1/source.mp4",
            "size_bytes": 100,
            "mtime_ns": 1,
        },
        "selection_id": "selection-1",
        "selection_status": "PLAN_SELECTED",
        "selection_with_caution": False,
        "selection_input_fingerprint": "sel-in",
        "selection_output_fingerprint": "sel-out",
        "governance_input_fingerprint": "gov-in",
        "governance_output_fingerprint": "gov-out",
        "governor_policy_version": "stage4.2-v1",
        "governor_validation_version": "stage4.2-validation-v1",
        "platform_policy_profile_version": "stage4.2-profile-v1",
        "selected_plan_id": "plan-1",
        "selected_plan_fingerprint": "plan-fp",
        "selected_plan_output_fingerprint": "plan-fp",
        "selected_plan_is_current": True,
        "planning_refinement_id": "ref-1",
        "planning_refinement_priority": "CANDIDATE",
        "planning_refinement_quality_level": "CANDIDATE",
        "planning_refinement_output_fingerprint": "planning-ref-fp",
        "final_refinement_id": "final-1",
        "final_refinement_status": "FINAL_TRANSCRIPT_READY",
        "final_refinement_quality_level": "FINAL_CLIP",
        "final_refinement_output_fingerprint": "final-fp",
        "live_caption_source_fingerprint": "caption-fp",
        "verification_state": "GROUNDED_IN_SOURCE",
        "verification_unresolved": False,
        "profile_key": "SHORTS_1080X1920",
        "profile_version": "stage5.0-render-profile-v1",
        "stage50_config": {"profile_key": "SHORTS_1080X1920", "max_frame_rate": 60.0},
    }
    base.update(overrides)
    return base


def _fp(**overrides: object) -> str:
    return render_contract_input_fingerprint(
        build_render_contract_input_payload(**_payload(**overrides))  # type: ignore[arg-type]
    )


def test_input_fingerprint_changes_with_final_clip_fingerprint() -> None:
    assert _fp() != _fp(final_refinement_output_fingerprint="changed")


def test_input_fingerprint_changes_with_selected_plan() -> None:
    assert _fp() != _fp(selected_plan_id="plan-2", selected_plan_fingerprint="plan-fp-2")


def test_input_fingerprint_changes_with_source_artifact_identity() -> None:
    changed = dict(_payload()["source_media_identity"])  # type: ignore[arg-type]
    changed["mtime_ns"] = 999
    assert _fp() != _fp(source_media_identity=changed)


def test_input_fingerprint_changes_with_profile_version() -> None:
    assert _fp() != _fp(profile_version="stage5.0-render-profile-v2")


def test_input_fingerprint_changes_with_policy_config() -> None:
    assert _fp() != _fp(stage50_config={"profile_key": "SHORTS_1080X1920", "max_frame_rate": 30.0})


def test_tts_voice_provider_and_caption_style_are_not_fingerprint_inputs() -> None:
    config_payload = stage50_config_payload(Stage50Config())
    text = str(config_payload).casefold()
    for forbidden in ("voice", "tts", "speaker", "font", "caption_style", "codec"):
        assert forbidden not in text
    payload = build_render_contract_input_payload(**_payload())  # type: ignore[arg-type]
    serialized = str(payload).casefold()
    for forbidden in ("voice", "tts", "speaker", "font", "publish", "title", "description"):
        assert forbidden not in serialized


def test_stage50_config_payload_is_deterministic() -> None:
    assert stage50_config_payload(Stage50Config()) == stage50_config_payload(Stage50Config())
