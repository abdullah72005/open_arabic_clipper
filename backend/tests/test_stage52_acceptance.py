"""Stage 5.2 acceptance-boundary tests.

These prove that the production manual-acceptance path fails closed unless it is
backed by genuine persisted, audio-backed lineage, and that synthetic evidence
cannot be relabelled as production acceptance.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from types import SimpleNamespace

import pytest
from stage52_support import crop_scene, make_spec, occurred

from app.render.execution.acceptance import (
    PRODUCTION_MANUAL_ACCEPTANCE,
    SYNTHETIC_REGRESSION,
    AcceptanceValidationError,
    validate_production_evidence,
)
from app.render.execution.types import AssAsset, RenderSpec

_HEX = "a" * 64


def _production_spec() -> RenderSpec:
    scene = crop_scene(
        0, 0, 10.0, 12.0, "TRACKED_CROP", [(10.0, 0.5, 0.5, 0.8), (12.0, 0.6, 0.5, 0.7)]
    )
    occ = occurred("block-0", 0, 10.0, 12.0, 0.0, (scene,))
    spec = make_spec(occurrences=(occ,), caption_events=())
    return replace(
        spec,
        render_execution_id=str(uuid.uuid4()),
        candidate_id=str(uuid.uuid4()),
        source_id=str(uuid.uuid4()),
        render_contract_id=str(uuid.uuid4()),
        visual_plan_id=str(uuid.uuid4()),
        plan_input_fingerprint=_HEX,
        plan_output_fingerprint=_HEX,
        ass=AssAsset(
            relative_path="sources/source/visual-composition/fp.ass",
            sha256=_HEX,
            event_count=0,
            line_count=1,
            policy_version="stage5.1-ass-v5",
        ),
    )


def _refinement(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": uuid.uuid4(),
        "priority": SimpleNamespace(value="FINAL_CLIP"),
        "status": SimpleNamespace(value="FINAL_TRANSCRIPT_READY"),
        "audio_relative_path": "sources/source/refinements/final.wav",
        "audio_content_hash": _HEX,
        "word_timestamps": [{"word": "مرحبا", "start": 10.1, "end": 10.4}],
        "final_transcript": "مرحبا بالعالم",
        "component_fingerprints": {"local_asr": "fp", "audio": "fp"},
        "confidence": 0.95,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _plan_payload(*, scenes: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "scenes": scenes
        if scenes is not None
        else [{"block_index": 0, "framing_mode": "TRACKED_CROP"}]
    }


def _contract_payload() -> dict[str, object]:
    return {"blocks": [{"block_index": 0, "block_type": "SOURCE_EXCERPT"}]}


def _validate(
    *,
    spec: RenderSpec | None = None,
    plan_payload: dict[str, object] | None = None,
    refinement: object | None = None,
) -> None:
    validate_production_evidence(
        spec=spec if spec is not None else _production_spec(),
        contract_payload=_contract_payload(),
        plan_payload=plan_payload if plan_payload is not None else _plan_payload(),
        refinement=refinement if refinement is not None else _refinement(),
    )


def test_acceptance_classifications_are_distinct() -> None:
    assert SYNTHETIC_REGRESSION != PRODUCTION_MANUAL_ACCEPTANCE


def test_production_evidence_accepts_genuine_lineage() -> None:
    _validate()


def test_acceptance_rejects_placeholder_identities() -> None:
    synthetic = make_spec(caption_events=())
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(spec=synthetic)
    assert error.value.reason_code == "CANDIDATE_IDENTITY_MISSING"


def test_acceptance_rejects_missing_plan_fingerprints() -> None:
    spec = replace(_production_spec(), plan_input_fingerprint="")
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(spec=spec)
    assert error.value.reason_code == "PLAN_INPUT_FINGERPRINT_MISSING"


def test_acceptance_rejects_missing_ass_hash() -> None:
    spec = replace(_production_spec(), ass=replace(_production_spec().ass, sha256=""))
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(spec=spec)
    assert error.value.reason_code == "ASS_HASH_MISSING"


def test_acceptance_rejects_missing_final_clip_refinement() -> None:
    with pytest.raises(AcceptanceValidationError) as error:
        validate_production_evidence(
            spec=_production_spec(),
            contract_payload=_contract_payload(),
            plan_payload=_plan_payload(),
            refinement=None,
        )
    assert error.value.reason_code == "FINAL_CLIP_REFINEMENT_MISSING"


def test_acceptance_rejects_non_final_clip_refinement() -> None:
    refinement = _refinement(priority=SimpleNamespace(value="CANDIDATE"))
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_REFINEMENT_MISSING"


def test_acceptance_rejects_seeded_non_audio_backed_evidence() -> None:
    # A seeded FINAL_CLIP row (as the plumbing tests create) has no audio path.
    refinement = _refinement(audio_relative_path=None, audio_content_hash=None)
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_NOT_AUDIO_BACKED"


def test_acceptance_rejects_invented_evenly_spaced_timings() -> None:
    refinement = _refinement(word_timestamps=[])
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_WORD_TIMESTAMPS_MISSING"


def test_acceptance_rejects_missing_audio_component_provenance() -> None:
    refinement = _refinement(component_fingerprints={"plan": "fp"})
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_NOT_AUDIO_BACKED"


def test_acceptance_rejects_multi_span_without_per_occurrence_decisions() -> None:
    scene = crop_scene(0, 0, 10.0, 12.0, "TRACKED_CROP", [(10.0, 0.5, 0.5, 0.8)])
    occ_a = occurred("block-0", 0, 10.0, 12.0, 0.0, (scene,))
    occ_b = occurred("block-1", 1, 30.0, 32.0, 2.0, ())
    spec = replace(_production_spec(), occurrences=(occ_a, occ_b))
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(spec=spec)
    assert error.value.reason_code == "MULTI_SPAN_VISUAL_DECISIONS_MISSING"


def test_acceptance_rejects_plan_missing_scene_decisions() -> None:
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(plan_payload={"scenes": []})
    assert error.value.reason_code == "MULTI_SPAN_VISUAL_DECISIONS_MISSING"
