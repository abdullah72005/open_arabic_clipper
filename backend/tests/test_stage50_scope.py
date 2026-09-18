"""Stage 5.0 scope-boundary and frozen-stage regression tests."""

from __future__ import annotations

from pathlib import Path

from app.core.enums import JobKind, PipelineStage, RenderContractStatus
from app.core.settings import Settings
from app.workers.tasks import _NEXT_STAGE

RENDER_ROOT = Path(__file__).parents[1] / "app" / "render"


def _render_sources() -> str:
    return "\n".join(path.read_text() for path in RENDER_ROOT.glob("*.py"))


def test_no_provider_vision_or_tts_imports() -> None:
    lowered = _render_sources().casefold()
    for forbidden in (
        "google.genai",
        "app.candidates.gemini",
        "app.candidates.local",
        "app.refinement.hosted",
        "app.transcription.reconstruction.ollama",
        "faster_whisper",
        "import cv2",
        "import ffmpeg",
        "generate_speech",
        "synthesize_speech",
        "text_to_speech(",
        "import whisper",
        "ollama",
    ):
        assert forbidden.casefold() not in lowered


def test_no_rendering_or_caption_code_paths() -> None:
    lowered = _render_sources().casefold()
    for forbidden in (
        "def render(",
        "render_video",
        "subprocess",
        "libass",
        "import ass",
        "write_ass",
        "face_tracking",
        "crop_path",
        "bidi",
        "\\u202a",
        "\\u202b",
        "\\u2066",
    ):
        assert forbidden not in lowered


def test_no_new_job_kind_or_pipeline_stage() -> None:
    kinds = {kind.value for kind in JobKind}
    assert "RENDER_CONTRACT" not in kinds
    assert "RENDERING" not in kinds
    stage_values = {stage.value for stage in PipelineStage}
    assert all("RENDER" not in value for value in stage_values)
    assert "RENDER_CONTRACT" not in {k.value for k in _NEXT_STAGE}


def test_render_status_has_no_compatibility_check_state() -> None:
    assert "COMPATIBILITY_CHECK_REQUIRED" not in {item.value for item in RenderContractStatus}


def test_stage50_settings_are_provider_free() -> None:
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_root="/tmp/scope50",
        render_contract_profile="SHORTS_1080X1920",
    )
    config = settings.stage50_config()
    assert config.profile_key == "SHORTS_1080X1920"
    assert config.max_frame_rate == 60.0
    assert config.probe_reuse_enabled is True


def test_no_stage_5_1_or_6_implementation_flags() -> None:
    source = (RENDER_ROOT / "handoff.py").read_text()
    assert "stage5_1_implemented" in source
    assert "stage5_2_implemented" in source
    assert "stage6_implemented" in source
    assert "False" in source


def test_handoff_returns_none_for_missing_candidate(tmp_path: Path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.db.base import Base
    from app.render.handoff import build_stage5_1_handoff
    from app.render.service import create_render_contract, read_render_contract

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'scope50.sqlite3'}")
    Base.metadata.create_all(engine)
    missing = "00000000-0000-0000-0000-000000000000"
    with Session(engine) as session:
        assert build_stage5_1_handoff(session, missing) is None
        assert read_render_contract(session, missing) is None
        assert create_render_contract(session, missing) is None
