"""Stage 5.2 acceptance-boundary tests.

These prove that the production manual-acceptance path fails closed unless it is
backed by genuine persisted, audio-backed lineage, and that synthetic evidence
cannot be relabelled as production acceptance. Component *names* never establish
provenance: only a managed audio file whose SHA-256 matches the persisted hash
and a recomputed Stage 3.5 dependency fingerprint do.
"""

from __future__ import annotations

import json
import uuid
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from stage52_support import crop_scene, make_spec, occurred

from app.refinement.fingerprints import component_fingerprint
from app.render.execution.acceptance import (
    PRODUCTION_MANUAL_ACCEPTANCE,
    SYNTHETIC_REGRESSION,
    AcceptanceValidationError,
    validate_production_evidence,
)
from app.render.execution.types import AssAsset, RenderSpec
from app.services.hashing import sha256_file
from app.services.storage import StorageCategory, StorageService

_HEX = "a" * 64
_AUDIO_INPUT = "b" * 64
_LOCAL_ASR = "c" * 64

_CANDIDATE_ID = str(uuid.uuid4())
_SOURCE_ID = str(uuid.uuid4())


def _write_wav(path: Path, duration: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(duration * 16_000))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * frames)


def _production_spec() -> RenderSpec:
    scene = crop_scene(
        0, 0, 10.0, 12.0, "TRACKED_CROP", [(10.0, 0.5, 0.5, 0.8), (12.0, 0.6, 0.5, 0.7)]
    )
    occ = occurred("block-0", 0, 10.0, 12.0, 0.0, (scene,))
    spec = make_spec(occurrences=(occ,), caption_events=())
    return replace(
        spec,
        render_execution_id=str(uuid.uuid4()),
        candidate_id=_CANDIDATE_ID,
        source_id=_SOURCE_ID,
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


def _audio_relative() -> str:
    return f"{_SOURCE_ID}/candidate-refinements/{_CANDIDATE_ID}/FINAL_CLIP.wav"


def _storage(tmp_path: Path) -> StorageService:
    storage = StorageService(tmp_path / "storage")
    _write_wav(storage.resolve(StorageCategory.SOURCES, _audio_relative()))
    return storage


def _audio_hash(storage: StorageService) -> str:
    return sha256_file(storage.resolve(StorageCategory.SOURCES, _audio_relative()))


def _audio_component(audio_hash: str) -> str:
    return component_fingerprint(
        "audio-extraction",
        {"audio_input_fingerprint": _AUDIO_INPUT, "content_hash": audio_hash},
    )


def _refinement(storage: StorageService, **overrides: object) -> SimpleNamespace:
    audio_hash = _audio_hash(storage)
    values: dict[str, object] = {
        "id": uuid.uuid4(),
        "clip_candidate_id": _CANDIDATE_ID,
        "source_video_id": _SOURCE_ID,
        "priority": SimpleNamespace(value="FINAL_CLIP"),
        "status": SimpleNamespace(value="FINAL_TRANSCRIPT_READY"),
        "audio_relative_path": _audio_relative(),
        "audio_content_hash": audio_hash,
        "audio_input_fingerprint": _AUDIO_INPUT,
        "word_timestamps": [{"word": "مرحبا", "start": 10.1, "end": 10.4}],
        "final_transcript": "مرحبا بالعالم",
        "component_fingerprints": {
            "audio_extraction": _audio_component(audio_hash),
            "local_asr": _LOCAL_ASR,
        },
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
    storage: StorageService,
    *,
    spec: RenderSpec | None = None,
    plan_payload: dict[str, object] | None = None,
    refinement: object | None = None,
) -> None:
    validate_production_evidence(
        spec=spec if spec is not None else _production_spec(),
        contract_payload=_contract_payload(),
        plan_payload=plan_payload if plan_payload is not None else _plan_payload(),
        refinement=refinement if refinement is not None else _refinement(storage),
        storage=storage,
    )


def test_acceptance_classifications_are_distinct() -> None:
    assert SYNTHETIC_REGRESSION != PRODUCTION_MANUAL_ACCEPTANCE


def test_production_evidence_accepts_genuine_lineage(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    _validate(storage)


def test_acceptance_rejects_placeholder_identities(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    synthetic = make_spec(caption_events=())
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, spec=synthetic)
    assert error.value.reason_code == "CANDIDATE_IDENTITY_MISSING"


def test_acceptance_rejects_missing_plan_fingerprints(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    spec = replace(_production_spec(), plan_input_fingerprint="")
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, spec=spec)
    assert error.value.reason_code == "PLAN_INPUT_FINGERPRINT_MISSING"


def test_acceptance_rejects_missing_ass_hash(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    spec = replace(_production_spec(), ass=replace(_production_spec().ass, sha256=""))
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, spec=spec)
    assert error.value.reason_code == "ASS_HASH_MISSING"


def test_acceptance_rejects_missing_final_clip_refinement(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    with pytest.raises(AcceptanceValidationError) as error:
        validate_production_evidence(
            spec=_production_spec(),
            contract_payload=_contract_payload(),
            plan_payload=_plan_payload(),
            refinement=None,
            storage=storage,
        )
    assert error.value.reason_code == "FINAL_CLIP_REFINEMENT_MISSING"


def test_acceptance_rejects_non_final_clip_refinement(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    refinement = _refinement(storage, priority=SimpleNamespace(value="CANDIDATE"))
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_REFINEMENT_MISSING"


def test_acceptance_rejects_seeded_non_audio_backed_evidence(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    refinement = _refinement(storage, audio_relative_path=None, audio_content_hash=None)
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_NOT_AUDIO_BACKED"


def test_acceptance_rejects_absent_audio_file(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    refinement = _refinement(storage, audio_relative_path=f"{_SOURCE_ID}/missing.wav")
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_AUDIO_MISSING"


def test_acceptance_rejects_corrupt_audio_file(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    refinement = _refinement(storage)
    path = storage.resolve(StorageCategory.SOURCES, _audio_relative())
    path.write_bytes(b"corrupt-not-the-hashed-audio")
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_AUDIO_HASH_MISMATCH"


def test_acceptance_rejects_unmanaged_audio_path(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    refinement = _refinement(storage, audio_relative_path="../../etc/passwd")
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_AUDIO_UNMANAGED"


def test_acceptance_rejects_invented_evenly_spaced_timings(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    refinement = _refinement(storage, word_timestamps=[])
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_WORD_TIMESTAMPS_MISSING"


def test_acceptance_rejects_mismatched_audio_extraction_provenance(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    audio_hash = _audio_hash(storage)
    refinement = _refinement(
        storage,
        component_fingerprints={
            "audio_extraction": _HEX,
            "local_asr": _LOCAL_ASR,
        },
    )
    assert audio_hash
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_AUDIO_PROVENANCE_MISMATCH"


def test_acceptance_rejects_component_name_substrings(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    refinement = _refinement(
        storage,
        component_fingerprints={
            "audio": _HEX,
            "local_asr_whisper_transcription": _HEX,
        },
    )
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_AUDIO_PROVENANCE_MISMATCH"


def test_acceptance_rejects_missing_local_asr_dependency(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    audio_hash = _audio_hash(storage)
    refinement = _refinement(
        storage,
        component_fingerprints={"audio_extraction": _audio_component(audio_hash)},
    )
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, refinement=refinement)
    assert error.value.reason_code == "FINAL_CLIP_LOCAL_ASR_MISSING"


def test_acceptance_rejects_mismatched_refinement_identity(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    refinement = _refinement(storage, clip_candidate_id=str(uuid.uuid4()))
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, refinement=refinement)
    assert error.value.reason_code == "REFINEMENT_IDENTITY_MISMATCH"


def test_acceptance_rejects_multi_span_without_per_occurrence_decisions(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    scene = crop_scene(0, 0, 10.0, 12.0, "TRACKED_CROP", [(10.0, 0.5, 0.5, 0.8)])
    occ_a = occurred("block-0", 0, 10.0, 12.0, 0.0, (scene,))
    occ_b = occurred("block-1", 1, 30.0, 32.0, 2.0, ())
    spec = replace(_production_spec(), occurrences=(occ_a, occ_b))
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, spec=spec)
    assert error.value.reason_code == "MULTI_SPAN_VISUAL_DECISIONS_MISSING"


def test_acceptance_rejects_plan_missing_scene_decisions(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    with pytest.raises(AcceptanceValidationError) as error:
        _validate(storage, plan_payload={"scenes": []})
    assert error.value.reason_code == "MULTI_SPAN_VISUAL_DECISIONS_MISSING"


def test_manifest_upsert_keeps_two_candidates_and_is_stable(tmp_path: Path) -> None:
    from app.render.execution.acceptance import _merge_production_manifest

    output = tmp_path / "manual-acceptance"
    case_a = {"candidate_id": "candidate-a", "output_sha256": "a" * 64}
    case_b = {"candidate_id": "candidate-b", "output_sha256": "b" * 64}
    first = _merge_production_manifest(output, case_a)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(first), encoding="utf-8")
    second = _merge_production_manifest(output, case_b)
    (output / "manifest.json").write_text(json.dumps(second), encoding="utf-8")
    assert {case["candidate_id"] for case in second["cases"]} == {"candidate-a", "candidate-b"}

    rerun_a = dict(case_a, output_sha256="c" * 64)
    third = _merge_production_manifest(output, rerun_a)
    assert len(third["cases"]) == 2
    by_id = {case["candidate_id"]: case for case in third["cases"]}
    assert by_id["candidate-a"]["output_sha256"] == "c" * 64
    assert by_id["candidate-b"]["output_sha256"] == "b" * 64


def test_load_manifest_absent_is_empty(tmp_path: Path) -> None:
    from app.render.execution.acceptance import _load_manifest

    assert _load_manifest(tmp_path) == {}


def test_load_manifest_invalid_json_fails_closed(tmp_path: Path) -> None:
    from app.render.execution.acceptance import _load_manifest

    (tmp_path / "manifest.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(AcceptanceValidationError) as error:
        _load_manifest(tmp_path)
    assert error.value.reason_code == "MANIFEST_INVALID"


def test_load_manifest_non_object_fails_closed(tmp_path: Path) -> None:
    from app.render.execution.acceptance import _load_manifest

    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(AcceptanceValidationError) as error:
        _load_manifest(tmp_path)
    assert error.value.reason_code == "MANIFEST_INVALID"


def test_load_manifest_invalid_cases_fails_closed(tmp_path: Path) -> None:
    from app.render.execution.acceptance import _load_manifest

    (tmp_path / "manifest.json").write_text('{"cases": "nope"}', encoding="utf-8")
    with pytest.raises(AcceptanceValidationError):
        _load_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text('{"cases": [{"no_candidate": 1}]}', encoding="utf-8")
    with pytest.raises(AcceptanceValidationError) as error:
        _load_manifest(tmp_path)
    assert error.value.reason_code == "MANIFEST_INVALID"


def test_load_manifest_unreadable_fails_closed(tmp_path: Path, monkeypatch: object) -> None:
    from app.render.execution.acceptance import _load_manifest

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    real_read_text = Path.read_text

    def flaky(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "manifest.json":
            raise OSError("simulated read failure")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky)  # type: ignore[attr-defined]
    with pytest.raises(AcceptanceValidationError) as error:
        _load_manifest(tmp_path)
    assert error.value.reason_code == "MANIFEST_UNREADABLE"


def _seed_package(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "package"
    case_dir = output / "candidate-a"
    case_dir.mkdir(parents=True)
    (case_dir / "output.mp4").write_bytes(b"OLD-MEDIA")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "classification": "PRODUCTION_MANUAL_ACCEPTANCE",
                "cases": [{"candidate_id": "candidate-a", "output_sha256": "old"}],
            }
        ),
        encoding="utf-8",
    )
    (output / "README.md").write_text("OLD README", encoding="utf-8")
    return output, case_dir


def _staged_case(output: Path) -> Path:
    staging = output / ".staging-candidate-a-deadbeef"
    staging.mkdir(parents=True)
    (staging / "output.mp4").write_bytes(b"NEW-MEDIA")
    return staging


def test_publish_package_manifest_failure_restores_previous(
    tmp_path: Path, monkeypatch: object
) -> None:
    import app.render.execution.acceptance as acceptance

    output, case_dir = _seed_package(tmp_path)
    staging = _staged_case(output)
    manifest = {
        "classification": "PRODUCTION_MANUAL_ACCEPTANCE",
        "cases": [{"candidate_id": "candidate-a", "output_sha256": "new"}],
    }
    real_write_json = acceptance._write_json

    def flaky(path: Path, payload: object) -> None:
        if path.name == "manifest.json":
            raise RuntimeError("simulated manifest publication failure")
        real_write_json(path, payload)

    monkeypatch.setattr(acceptance, "_write_json", flaky)  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        acceptance._publish_package(output, staging, case_dir, manifest)

    assert (case_dir / "output.mp4").read_bytes() == b"OLD-MEDIA"
    assert json.loads((output / "manifest.json").read_text())["cases"][0]["output_sha256"] == "old"
    assert (output / "README.md").read_text() == "OLD README"
    assert not list(output.glob("*.previous-*"))
    assert not staging.exists()


def test_publish_package_readme_failure_restores_previous(
    tmp_path: Path, monkeypatch: object
) -> None:
    import app.render.execution.acceptance as acceptance

    output, case_dir = _seed_package(tmp_path)
    staging = _staged_case(output)
    manifest = {
        "classification": "PRODUCTION_MANUAL_ACCEPTANCE",
        "cases": [{"candidate_id": "candidate-a", "output_sha256": "new"}],
    }

    def boom(_manifest: object) -> str:
        raise RuntimeError("simulated README publication failure")

    monkeypatch.setattr(acceptance, "_readme", boom)  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        acceptance._publish_package(output, staging, case_dir, manifest)

    assert (case_dir / "output.mp4").read_bytes() == b"OLD-MEDIA"
    assert json.loads((output / "manifest.json").read_text())["cases"][0]["output_sha256"] == "old"
    assert (output / "README.md").read_text() == "OLD README"
    assert not list(output.glob("*.previous-*"))
    assert not staging.exists()
