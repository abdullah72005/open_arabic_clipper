"""Stage 6 production-path service: orchestration and persistence."""

from __future__ import annotations

from dataclasses import replace

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from test_stage6_materialization_core import make_inputs

from app.core.enums import (
    CandidateDisposition,
    ContentType,
    MaterializationLifecycle,
    MaterializationOutcome,
    OriginalityRisk,
    RightsRisk,
)
from app.db.base import Base
from app.materialization.inputs import MaterializationInputs
from app.materialization.narration import FakeNarrationSynthesizer
from app.materialization.policy import DEFAULT_CONFIG, CallLedger
from app.materialization.providers import FakeCritiqueProvider, FakeWritingProvider
from app.materialization.service import (
    MaterializationResult,
    persist_materialization,
    run_materialization,
)
from app.models import ClipCandidate, ClipMaterialization, SourceVideo

NOVEL_COPY = "The mechanism is that mentorship loss compounds"


def _inputs(**overrides: object) -> MaterializationInputs:
    inputs = make_inputs()
    blocks = list(inputs.blocks)
    blocks[1] = {**blocks[1], "grounding_refs": ["block:0"]}
    base = replace(inputs, blocks=tuple(blocks))
    return replace(base, **overrides)


def _run(inputs: MaterializationInputs) -> MaterializationResult:
    return run_materialization(
        inputs,
        provider=FakeWritingProvider(copies={1: NOVEL_COPY}),
        critique_provider=FakeCritiqueProvider(accepted={1: True}),
        narration_synthesizer=FakeNarrationSynthesizer(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        ledger=CallLedger(),
    )


def test_orchestration_materializes_timeline_and_assessment() -> None:
    result = _run(_inputs())
    assert result.script.accepted
    assert result.timeline.segments
    assert result.audio_recipe.entries
    # Pre-render, the content is materialized but not yet human-review-ready.
    assert result.assessment.outcome == MaterializationOutcome.MATERIALIZED_WITH_CAUTION.value
    assert result.assessment.human_review_required is False
    assert result.assessment.publication_ready is False


def test_orchestration_synthesizes_required_narration() -> None:
    inputs = _inputs(
        narration={"need": "RECOMMENDED", "requirements": {"placement_block_index": 1}},
        profile=replace(make_inputs().profile, narration_allowed=True),
    )
    result = _run(inputs)
    assert result.narration
    card = next(seg for seg in result.timeline.segments if seg.kind == "AUTHORED_CARD")
    assert card.narration_chunk_id is not None
    assert card.output_duration >= result.narration[0].duration_seconds - 0.01


def test_orchestration_defers_when_provider_unavailable() -> None:
    inputs = _inputs()
    result = run_materialization(
        inputs,
        provider=FakeWritingProvider(available=False),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        ledger=CallLedger(),
    )
    assert result.script.deferred
    assert result.assessment.outcome == MaterializationOutcome.DEFERRED_PROVIDER.value


def test_persist_materialization_awaits_render() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        source = SourceVideo(source_uri="/tmp/stage6.mp4", content_hash="stage6-hash")
        session.add(source)
        session.flush()
        candidate = ClipCandidate(
            source_video_id=source.id,
            candidate_key="stage6-candidate",
            disposition=CandidateDisposition.CANDIDATE,
            start_time=0.0,
            end_time=10.0,
            start_segment_index=0,
            end_segment_index=0,
            primary_content_type=ContentType.STORY,
            rights_risk=RightsRisk.LOW,
            originality_risk=OriginalityRisk.NOT_INDICATED,
        )
        session.add(candidate)
        session.flush()

        inputs = _inputs(source_id=str(source.id), candidate_id=str(candidate.id))
        result = _run(inputs)
        row = persist_materialization(session, inputs, result)

        assert row.lifecycle is MaterializationLifecycle.AWAITING_RENDER
        assert row.publication_ready is False
        assert row.is_current is True
        assert row.input_fingerprint
        assert row.component_checkpoints["SCRIPT"]["fingerprint"]
        assert row.semantic_outcome is MaterializationOutcome.MATERIALIZED_WITH_CAUTION

        # Identical work reuses the same row (no duplicate history row).
        result2 = _run(inputs)
        row2 = persist_materialization(session, inputs, result2)
        assert row2.id == row.id
        assert (
            session.query(ClipMaterialization)
            .filter(ClipMaterialization.is_current.is_(True))
            .count()
            == 1
        )

        # A genuinely changed input (different priority) creates a new current
        # row and demotes the previous one.
        changed = _inputs(
            source_id=str(source.id), candidate_id=str(candidate.id), priority="FINAL_CLIP"
        )
        result3 = _run(changed)
        row3 = persist_materialization(session, changed, result3)
        assert row3.id != row.id
        old_current = session.scalars(
            select(ClipMaterialization.is_current).where(ClipMaterialization.id == row.id)
        ).one()
        assert old_current is False
        assert row3.is_current is True
        assert (
            session.query(ClipMaterialization)
            .filter(ClipMaterialization.is_current.is_(True))
            .count()
            == 1
        )
