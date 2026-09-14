"""Stage 4.1 scope-boundary and frozen-stage regression tests."""

from __future__ import annotations

from pathlib import Path

from app.core.enums import JobKind, PipelineStage, PlanSemanticOutcome, SemanticProviderMode
from app.core.settings import Settings
from app.transformation.planning.handoff import build_stage4_2_handoff
from app.workers.tasks import _NEXT_STAGE


def test_no_channel_or_tts_persistence_model_added() -> None:
    from app import models

    names = set(dir(models))
    assert "ChannelTTSConfig" not in names
    assert "ChannelConfig" not in names
    assert "VoiceConfig" not in names
    assert all("Channel" not in name for name in names)


def test_no_tts_or_rendering_modules_added() -> None:
    planning_root = Path(__file__).parents[1] / "app" / "transformation" / "planning"
    sources = "\n".join(path.read_text() for path in planning_root.glob("*.py"))
    lowered = sources.casefold()
    # No TTS generation or media/rendering implementation may live in Stage 4.1.
    for forbidden in (
        "import ffmpeg",
        "subprocess",
        "generate_speech",
        "synthesize_speech",
        "text_to_speech(",
        "def render",
        "ffmpeg_operation",
    ):
        assert forbidden not in lowered
    # The boundary markers exist only to reject such provider output.
    assert "tts_selection_markers" in lowered
    assert "rendering_instruction_markers" in lowered


def test_no_stage_4_2_or_4_3_implemented() -> None:
    import app.transformation.planning.handoff as handoff

    source = Path(handoff.__file__).read_text()
    assert "stage4_2_implemented" in source
    assert "stage4_3_implemented" in source
    assert "governor" not in source.casefold() or "stage4_2_implemented" in source


def test_no_automatic_next_stage_entry() -> None:
    assert "TRANSFORMATION_PLANNING" not in {k.value for k in _NEXT_STAGE}
    assert "TRANSFORMATION_ELIGIBILITY" not in {k.value for k in _NEXT_STAGE}
    stage_values = {stage.value for stage in PipelineStage}
    assert all("PLANNING" not in value for value in stage_values)


def test_no_platform_evasion_tactics() -> None:
    planning_root = Path(__file__).parents[1] / "app" / "transformation" / "planning"
    sources = "\n".join(path.read_text().casefold() for path in planning_root.glob("*.py"))
    # The planning instruction must explicitly prohibit platform-evasion work
    # and the deterministic boundary must reject evasion markers.
    assert "platform-detection evasion" in sources
    assert "platform_evasion_markers" in sources
    assert "reject_evasion" in sources
    for forbidden in ("anti-detection", "evasion_strategy", "def evade"):
        assert forbidden not in sources


def test_qwen_disabled_by_default_and_never_adaptive_fallback() -> None:
    settings = Settings(database_url="sqlite+pysqlite:///:memory:", storage_root="/tmp/scope41")
    assert settings.local_qwen_enabled is False
    assert settings.transformation_planning_semantic_mode() is SemanticProviderMode.ADAPTIVE
    # Adaptive builds Gemini only when a key exists; with no key it returns None
    # and never the local provider.
    settings_no_key = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_root="/tmp/scope41",
        gemini_api_key=None,
    )
    assert settings_no_key.transformation_planning_provider() is None
    assert settings_no_key.local_transformation_planning_provider_instance() is None


def test_planning_mode_decoupled_from_stage40_mode() -> None:
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_root="/tmp/scope41",
        transformation_provider_mode="deterministic",
        transformation_planning_mode="adaptive",
    )
    assert settings.transformation_semantic_mode() is SemanticProviderMode.DETERMINISTIC
    assert settings.transformation_planning_semantic_mode() is SemanticProviderMode.ADAPTIVE


def test_no_source_lifecycle_transition_values_added() -> None:
    stage_values = {stage.value for stage in PipelineStage}
    assert "TRANSFORMATION_PLANNING" not in stage_values
    assert "READY_FOR_TRANSFORMATION" not in stage_values
    assert "CANDIDATE_PLANNED" not in stage_values


def test_plan_semantic_outcomes_are_not_failures() -> None:
    finite = {item.value for item in PlanSemanticOutcome}
    assert "PLANNING_DEFERRED" in finite
    assert "NO_VALID_PLAN_FROM_STRATEGY" in finite
    assert "PROVIDER_UNAVAILABLE" in finite


def test_job_kind_extends_existing_platform_only() -> None:
    kinds = {kind.value for kind in JobKind}
    assert "TRANSFORMATION_PLANNING" in kinds
    assert "TRANSFORMATION_ELIGIBILITY" in kinds
    # No new job platform: only enum extension, no pipeline stage.
    assert "PIPELINE_RUN" not in kinds


def test_handoff_callable_signature(tmp_path: Path) -> None:
    # build_stage4_2_handoff returns None for a missing candidate.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.db.base import Base

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'scope.sqlite3'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert build_stage4_2_handoff(session, "00000000-0000-0000-0000-000000000000") is None
