"""F1 authoritative manual text must reach executable captions unchanged."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stage43_support
from sqlalchemy import select
from sqlalchemy.orm import Session
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import FakeProber, seed_stage50
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    seed_stage51,
)
from stage52_support import fake_runtime
from test_stage6_vertical_slice import (
    _ANNOTATION_COPY,
    _PROFILE,
    _stream_probe,
)

from app.composition.ass import serialize_ass
from app.composition.captions import build_caption_plan
from app.composition.executor import build_visual_composition_executor
from app.composition.policy import CaptionStyle, Stage51Config, safe_zone_for
from app.composition.queue import get_or_create_plan_row
from app.composition.service import get_current_visual_composition, read_visual_composition
from app.core.enums import JobKind, RefinementPriority, RefinementStatus
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.executor import build_materialization_executor
from app.materialization.providers import FakeCritiqueProvider, FakeWritingProvider
from app.materialization.rendering import build_transformed_render_spec
from app.materialization.service import MaterializationStaleError, resolve_prerequisites
from app.models import CandidateRefinement, ClipCandidate, ClipMaterialization, ProcessingJob
from app.refinement.queue import apply_manual_transcript
from app.render.execution.compiler import compile_render
from app.render.execution.service import RenderExecutionError, validate_candidate_for_render
from app.render.policy import MANUAL_TRANSCRIPT_DISPLAY_ALIGNMENT_UNRESOLVED
from app.render.service import create_render_contract, read_render_contract
from app.services.storage import StorageService
from app.transformation.executor import build_transformation_executor


def _run_composition(session: Session, fixture: Any, candidate_id: Any) -> Any:
    candidate = session.get(ClipCandidate, candidate_id)
    assert candidate is not None
    row = get_or_create_plan_row(session, candidate)
    job = ProcessingJob(
        source_video_id=candidate.source_video_id,
        kind=JobKind.VISUAL_COMPOSITION,
        visual_composition_plan_id=row.id,
    )
    session.add(job)
    session.commit()
    executor = build_visual_composition_executor(
        session,
        fixture.stage50.storage,
        fixture.settings,
        display_probe=fixture.display_probe,
        frame_sampler=FakeFrameSampler(),
        scene_cut_detector=FakeSceneCutDetector(),
        detector=FakeDetector(),
    )
    executor.set_active_job(job.id)
    executor.execute(row.id)
    session.refresh(row)
    assert row.plan_ready is True
    return row


def _seed_manual_chain(session: Session, monkeypatch: Any) -> tuple[Any, Any, dict[str, Any]]:
    original_seed = stage43_support.seed_stage41
    state: dict[str, Any] = {}

    def manual_before_planning(*args: Any, **kwargs: Any) -> Any:
        seed = original_seed(*args, **kwargs)
        source, candidate, final, analysis, strategies = seed
        final.priority = RefinementPriority.FINAL_CLIP
        final.quality_level = "FINAL_CLIP"
        final.status = RefinementStatus.FINAL_TRANSCRIPT_READY
        session.flush()
        raw_words = [dict(word) for word in final.word_timestamps or []]
        corrected = final.automatic_transcript.replace("sharply.", "slightly.")
        apply_manual_transcript(session, final.id, corrected)
        assert final.status is RefinementStatus.FINAL_TRANSCRIPT_READY
        analysis.refinement_priority = final.priority.value
        analysis.refinement_quality_level = final.quality_level
        analysis.input_fingerprint = build_transformation_executor(
            session, kwargs["settings"]
        ).input_fingerprint(candidate)
        session.commit()
        state["final"] = final
        state["raw_words"] = raw_words
        return seed

    monkeypatch.setattr(stage43_support, "seed_stage41", manual_before_planning)
    fixture = seed_stage51(session, monkeypatch, planning_on_final=False)
    candidate_id = fixture.stage50.selection.candidate.id
    visual = _run_composition(session, fixture, candidate_id)
    return fixture, visual, state


def test_manual_final_text_reaches_contract_composition_stage6_and_compiler(
    sqlite_engine: Any, monkeypatch: Any
) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        fixture, visual, state = _seed_manual_chain(session, monkeypatch)
        final = state["final"]
        raw_words = state["raw_words"]

        assert final.manual_transcript.endswith("slightly.")
        assert final.final_transcript.endswith("slightly.")
        assert final.word_timestamps == raw_words

        caption_input = fixture.contract.row.contract_payload["caption_input"]
        assert caption_input["transcript_text"] == final.manual_transcript
        assert caption_input["manual_transcript"] == final.manual_transcript
        assert caption_input["display_projection"]["status"] == "REPLACEMENT"
        corrected_index = caption_input["display_projection"]["corrected_token_index"]
        assert corrected_index == len(raw_words) - 1
        display_words = caption_input["word_timestamps"]
        assert display_words[corrected_index]["text"] == "slightly."
        for index, raw in enumerate(raw_words):
            assert display_words[index]["start"] == raw["start"]
            assert display_words[index]["end"] == raw["end"]
            if index != corrected_index:
                assert display_words[index]["text"] == raw["text"]

        events = visual.plan_payload["captions"]["events"]
        displayed = " ".join(event["text"] for event in events)
        assert "slightly." in displayed
        assert "sharply." not in displayed

        ass_path = (
            Path(fixture.stage50.storage.storage_root) / visual.plan_payload["ass"]["asset_path"]
        )
        ass_text = ass_path.read_text(encoding="utf-8")
        assert "slightly." in ass_text
        assert "sharply." not in ass_text

        candidate = fixture.stage50.selection.candidate
        job = ProcessingJob(
            source_video_id=candidate.source_video_id,
            kind=JobKind.MATERIALIZATION,
        )
        session.add(job)
        session.commit()
        executor = build_materialization_executor(
            session,
            fixture.stage50.storage,
            get_settings(),
            writing=FakeWritingProvider({1: _ANNOTATION_COPY}),
            critique=FakeCritiqueProvider(accepted={1: True}),
        )
        executor.set_active_job(job.id)
        executor.execute(str(candidate.id))
        materialization = session.scalars(
            select(ClipMaterialization).where(
                ClipMaterialization.clip_candidate_id == candidate.id,
                ClipMaterialization.is_current.is_(True),
            )
        ).one()
        spec = build_transformed_render_spec(
            session,
            candidate,
            materialization_id=materialization.id,
            expected_output_fingerprint=materialization.output_fingerprint,
            delivery_profile_key=_PROFILE,
            settings=get_settings(),
            storage=fixture.stage50.storage,
            stream_probe=_stream_probe,
        )
        assert spec.ass.sha256 == visual.plan_payload["ass"]["sha256"]
        compiled = compile_render(spec, fake_runtime())
        assert compiled.compiler_version.startswith("stage6-authored")


def _seed_legacy_ready(
    session: Session,
    monkeypatch: Any,
    *,
    manual: str,
    automatic: str,
    words: list[dict[str, object]],
) -> Any:
    original_seed = stage43_support.seed_stage41

    def legacy_optimistic_ready(*args: Any, **kwargs: Any) -> Any:
        seed = original_seed(*args, **kwargs)
        source, candidate, final, analysis, strategies = seed
        final.priority = RefinementPriority.FINAL_CLIP
        final.quality_level = "FINAL_CLIP"
        final.status = RefinementStatus.FINAL_TRANSCRIPT_READY
        final.manual_transcript = manual
        final.final_transcript = manual
        final.automatic_transcript = automatic
        final.word_timestamps = [dict(word) for word in words]
        session.flush()
        analysis.refinement_priority = final.priority.value
        analysis.refinement_quality_level = final.quality_level
        analysis.input_fingerprint = build_transformation_executor(
            session, kwargs["settings"]
        ).input_fingerprint(candidate)
        session.commit()
        return seed

    monkeypatch.setattr(stage43_support, "seed_stage41", legacy_optimistic_ready)
    governance = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, governance)
    return seed_stage50(session, settings=governance, planning_on_final=False)


def test_legacy_optimistic_ready_manual_row_refuses_executable_readiness(
    sqlite_engine: Any, monkeypatch: Any
) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        fixture = _seed_legacy_ready(
            session,
            monkeypatch,
            manual="completely unrelated replacement words here",
            automatic="rates fell sharply.",
            words=[
                {"index": 0, "text": "rates", "start": 10.0, "end": 10.4},
                {"index": 1, "text": "fell", "start": 10.4, "end": 10.8},
                {"index": 2, "text": "sharply.", "start": 10.8, "end": 11.2},
            ],
        )
        view = create_render_contract(
            session,
            fixture.selection.candidate.id,
            storage=fixture.storage,
            prober=fixture.prober,
        )
        assert view is not None
        assert view.row.status.value == "FINAL_CLIP_REFINEMENT_REQUIRED", view.row.reason_codes
        assert view.row.contract_ready is False
        assert MANUAL_TRANSCRIPT_DISPLAY_ALIGNMENT_UNRESOLVED in view.row.reason_codes
        final = session.get(CandidateRefinement, fixture.selection.refinement.id)
        assert final is not None
        assert final.final_transcript == "completely unrelated replacement words here"


def test_legacy_stage50_preflight_refuses_malformed_intervening_word(
    sqlite_engine: Any, monkeypatch: Any
) -> None:
    raw_words: list[dict[str, object]] = [
        {"index": 0, "text": "rates", "start": 10.0, "end": 10.4},
        {"text": "really"},
        {"index": 2, "text": "fell", "start": 11.0, "end": 11.4},
        {"index": 3, "text": "sharply.", "start": 12.0, "end": 12.4},
    ]
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        fixture = _seed_legacy_ready(
            session,
            monkeypatch,
            manual="rates fell slightly.",
            automatic="rates fell sharply.",
            words=raw_words,
        )
        view = create_render_contract(
            session,
            fixture.selection.candidate.id,
            storage=fixture.storage,
            prober=fixture.prober,
        )
        assert view is not None
        assert view.row.status.value == "FINAL_CLIP_REFINEMENT_REQUIRED", view.row.reason_codes
        assert view.row.contract_ready is False
        assert MANUAL_TRANSCRIPT_DISPLAY_ALIGNMENT_UNRESOLVED in view.row.reason_codes
        final = session.get(CandidateRefinement, fixture.selection.refinement.id)
        assert final is not None
        assert final.manual_transcript == "rates fell slightly."
        assert final.final_transcript == "rates fell slightly."
        assert final.word_timestamps == raw_words


def test_manual_edit_after_downstream_artifacts_invalidates_currentness(
    sqlite_engine: Any, monkeypatch: Any
) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as setup:
        fixture = seed_stage51(setup, monkeypatch)
        candidate_id = fixture.stage50.selection.candidate.id
        final_id = fixture.stage50.selection.refinement.id
        old_contract_id = fixture.contract.row.id
        old_visual = _run_composition(setup, fixture, candidate_id)
        old_ass_sha = old_visual.plan_payload["ass"]["sha256"]
        setup.commit()

    with Session(sqlite_engine) as fresh:
        final = fresh.get(CandidateRefinement, final_id)
        assert final is not None
        corrected = final.automatic_transcript.replace("sharply.", "slightly.")
        apply_manual_transcript(fresh, final_id, corrected)
        fresh.commit()

    with Session(sqlite_engine) as stale_check:
        contract_view = read_render_contract(stale_check, candidate_id, settings=get_settings())
        assert contract_view is not None
        assert contract_view.live_freshness == "STALE"
        assert contract_view.effective is False

        visual_view = read_visual_composition(
            stale_check,
            candidate_id,
            display_probe=fixture.display_probe,
            config=fixture.settings.config,
        )
        assert visual_view is not None
        assert visual_view.live_freshness == "STALE"
        assert visual_view.effective is False

        with pytest.raises(MaterializationStaleError):
            resolve_prerequisites(stale_check, candidate_id)
        with pytest.raises(RenderExecutionError):
            validate_candidate_for_render(stale_check, candidate_id, settings=get_settings())

        current_visual = get_current_visual_composition(stale_check, candidate_id)
        assert current_visual is not None
        assert current_visual.plan_payload["ass"]["sha256"] == old_ass_sha

    with Session(sqlite_engine) as rebuild:
        storage = StorageService(get_settings().storage_root)
        blocked = create_render_contract(
            rebuild,
            candidate_id,
            storage=storage,
            prober=FakeProber(),
        )
        assert blocked is not None
        assert blocked.row.contract_ready is False
        assert blocked.row.id != old_contract_id
        assert blocked.row.status.value != "READY_FOR_RENDER_PLANNING"


def test_nonmanual_unchanged_path_remains_usable(sqlite_engine: Any, monkeypatch: Any) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        fixture = seed_stage51(session, monkeypatch)
        final = fixture.stage50.selection.refinement
        caption_input = fixture.contract.row.contract_payload["caption_input"]
        assert caption_input["manual_transcript"] is None
        assert caption_input["display_projection"]["status"] == "UNCHANGED"
        assert [word["text"] for word in caption_input["word_timestamps"]] == [
            str(word["text"]) for word in final.word_timestamps or []
        ]
        visual = _run_composition(session, fixture, fixture.stage50.selection.candidate.id)
        assert visual.plan_ready is True


def _font_family() -> str | None:
    if shutil.which("fc-match") is None:
        return None
    result = subprocess.run(
        ["fc-match", "-f", "%{family}", "Noto Sans Arabic"],
        check=False,
        capture_output=True,
        text=True,
    )
    family = result.stdout.strip()
    return family or None


def _has_libx264() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], check=False, capture_output=True, text=True
    )
    return "libx264" in result.stdout


_SKIP_RENDER = None
if shutil.which("ffmpeg") is None:
    _SKIP_RENDER = "ffmpeg is not installed"
elif _font_family() is None:
    _SKIP_RENDER = "no usable font is installed"
elif not _has_libx264():
    _SKIP_RENDER = "ffmpeg lacks libx264"


def _caption_ass(text: str, font_family: str) -> bytes:
    tokens = text.split()
    words = [
        {
            "index": index,
            "text": token,
            "start": index * 0.4,
            "end": index * 0.4 + 0.35,
        }
        for index, token in enumerate(tokens)
    ]
    style = CaptionStyle(font_family=font_family)
    plan = build_caption_plan(
        [(0, 0.0, len(tokens) * 0.4, 0, len(tokens) - 1, "HERO")],
        words,
        style,
        Stage51Config(),
    )
    return serialize_ass(plan, style, safe_zone_for("SHORTS_VERTICAL_SAFE_ZONE_V1"))


def _render_png(ass_bytes: bytes, destination: Path) -> None:
    ass_path = destination.with_suffix(".ass")
    ass_path.write_bytes(ass_bytes)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1080x1920:d=1",
            "-vf",
            f"ass={ass_path}",
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )


def _encode_mp4(ass_bytes: bytes, destination: Path) -> None:
    ass_path = destination.with_suffix(".ass")
    ass_path.write_bytes(ass_bytes)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1080x1920:d=1:r=5",
            "-vf",
            f"ass={ass_path}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "5",
            "-t",
            "1",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )


def _raw_rgb(path: Path, *, seek: float | None = None) -> np.ndarray:
    command = ["ffmpeg", "-v", "error"]
    if seek is not None:
        command += ["-ss", str(seek)]
    command += ["-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    result = subprocess.run(command, check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(1920, 1080, 3).astype(np.int32)


@pytest.mark.skipif(_SKIP_RENDER is not None, reason=_SKIP_RENDER or "render unavailable")
def test_corrected_display_reaches_encoded_mp4(tmp_path: Path) -> None:
    font_family = _font_family()
    assert font_family is not None
    corrected_ass = _caption_ass("promotion rates fell slightly.", font_family)
    asr_ass = _caption_ass("promotion rates fell sharply.", font_family)

    corrected_png = tmp_path / "corrected.png"
    asr_png = tmp_path / "asr.png"
    encoded = tmp_path / "corrected.mp4"
    _render_png(corrected_ass, corrected_png)
    _render_png(asr_ass, asr_png)
    _encode_mp4(corrected_ass, encoded)

    corrected_frame = _raw_rgb(corrected_png)
    asr_frame = _raw_rgb(asr_png)
    decoded_frame = _raw_rgb(encoded, seek=0.2)

    assert not np.array_equal(corrected_frame, asr_frame)
    assert int(np.abs(corrected_frame - corrected_frame[0, 0]).sum(axis=2).max()) > 40
    corrected_distance = float(np.abs(decoded_frame - corrected_frame).mean())
    asr_distance = float(np.abs(decoded_frame - asr_frame).mean())
    assert corrected_distance < asr_distance
