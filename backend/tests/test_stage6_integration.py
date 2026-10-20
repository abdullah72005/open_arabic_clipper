"""Stage 6 production-path integration: persisted ORM assembly + executor.

Exercises ``assemble_inputs`` through genuine persisted upstream relationships
(source -> candidate -> FINAL_CLIP refinement -> eligibility analysis ->
strategy -> plan set -> selected plan -> render contract -> ready visual
composition), then drives the materialization executor end to end with fake
seams. No FFmpeg, no network, no provider.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    seed_stage51,
)

from app.composition.executor import build_visual_composition_executor
from app.composition.queue import get_or_create_plan_row
from app.core.enums import JobKind, JobStatus
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization import service
from app.materialization.executor import build_materialization_executor
from app.materialization.providers import DeterministicWritingProvider
from app.materialization.service import assemble_inputs
from app.models import ClipMaterialization, ProcessingJob
from app.models.transformation_plan import TransformationPlan, TransformationPlanSet


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _ready_composition(session: Session, fixture: Any) -> Any:
    candidate = fixture.stage50.selection.candidate
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


def test_assemble_inputs_reads_persisted_plan_set_context(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch, narration="NONE")
    candidate = fixture.stage50.selection.candidate
    _ready_composition(session, fixture)

    inputs = assemble_inputs(session, candidate.id)
    plan = session.get(TransformationPlan, uuid.UUID(inputs.plan_id))
    assert plan is not None
    plan_set = session.get(TransformationPlanSet, plan.plan_set_id)
    assert plan_set is not None
    assert inputs.plan_target_context == service._plan_target_context(plan_set)
    assert inputs.final_clip.status == "FINAL_TRANSCRIPT_READY"
    assert inputs.contract_output_fingerprint == fixture.contract.row.output_fingerprint


def test_materialization_executor_persists_fenced_current_row(
    session: Session, monkeypatch: Any
) -> None:
    fixture = seed_stage51(session, monkeypatch, narration="NONE")
    candidate = fixture.stage50.selection.candidate
    _ready_composition(session, fixture)

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
        writing=DeterministicWritingProvider(),
    )
    executor.set_active_job(job.id)
    executor.execute(str(candidate.id))

    session.refresh(job)
    assert job.status is JobStatus.SUCCEEDED
    assert job.claim_version >= 1
    row = session.scalars(
        select(ClipMaterialization).where(ClipMaterialization.is_current.is_(True))
    ).one()
    assert row.is_current is True
    assert row.publication_ready is False
    assert row.input_fingerprint
    assert job.materialization_id == row.id
