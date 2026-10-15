"""Stage 6 persisted vertical slice: materialization -> typed transformed spec.

Drives the *persisted* production path from genuine Stage 4/5 relationships
through the real materialization executor, then builds the exact
TRANSFORMED_SHORT render request via ``build_transformed_render_spec`` and
compiles it with the frozen renderer. This is a synthetic fixture (the source
media is a generated stub), not product acceptance.
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
from stage52_support import fake_runtime

from app.composition.executor import build_visual_composition_executor
from app.composition.queue import get_or_create_plan_row
from app.core.enums import JobKind
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.executor import build_materialization_executor
from app.materialization.providers import DeterministicWritingProvider
from app.materialization.rendering import build_transformed_render_spec
from app.models import ClipMaterialization, ProcessingJob
from app.render.execution.compiler import compile_render
from app.render.execution.policy import TRANSFORMED_SHORT
from app.render.execution.service import SourceStreamFacts

_PROFILE = "MP4_H264_AAC_1080X1920_V1"


def _stream_probe(_path: Any, _settings: Any) -> SourceStreamFacts:
    # Synthetic stub source: report an executable stereo stream origin.
    return SourceStreamFacts(0.0, 0.0, 2, True)


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


def _persisted_materialization(session: Session, monkeypatch: Any) -> tuple[Any, Any]:
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
    row = session.scalars(
        select(ClipMaterialization).where(ClipMaterialization.is_current.is_(True))
    ).one()
    return fixture, row


def test_persisted_materialization_builds_executable_transformed_spec(
    session: Session, monkeypatch: Any
) -> None:
    fixture, row = _persisted_materialization(session, monkeypatch)
    assert row.output_fingerprint
    assert row.timeline_payload
    assert row.audio_recipe
    candidate = fixture.stage50.selection.candidate

    spec = build_transformed_render_spec(
        session,
        candidate,
        materialization_id=row.id,
        expected_output_fingerprint=row.output_fingerprint,
        delivery_profile_key=_PROFILE,
        settings=get_settings(),
        storage=fixture.stage50.storage,
        stream_probe=_stream_probe,
    )
    assert spec.artifact_purpose == TRANSFORMED_SHORT
    assert spec.authored is not None
    payload = spec.authored
    assert payload.materialization_id == str(row.id)
    assert payload.materialization_output_fingerprint == row.output_fingerprint
    # At least one SOURCE unit and the complete hero are present.
    assert any(unit.kind == "SOURCE" for unit in payload.units)
    assert any(occurrence.is_hero for occurrence in spec.occurrences)
    # Every SOURCE occurrence has exactly one unit at its authored position.
    source_occurrence_ids = {unit.occurrence_id for unit in payload.units if unit.kind == "SOURCE"}
    assert source_occurrence_ids == {occ.occurrence_id for occ in spec.occurrences}
    # The authored ASS was built and stored with a verifiable hash.
    ass_path = fixture.stage50.storage.storage_root / payload.authored_ass.relative_path
    assert ass_path.is_file()

    # The persisted request is genuinely executable by the frozen renderer.
    compiled = compile_render(spec, fake_runtime())
    assert compiled.compiler_version.startswith("stage6-authored")
    assert compiled.audio_preparation is not None
    assert compiled.authored_ass_localized_name == "authored.ass"


def test_transformed_spec_fails_closed_on_wrong_output_fingerprint(
    session: Session, monkeypatch: Any
) -> None:
    fixture, row = _persisted_materialization(session, monkeypatch)
    candidate = fixture.stage50.selection.candidate
    from app.render.execution.service import RenderExecutionError

    with pytest.raises(RenderExecutionError):
        build_transformed_render_spec(
            session,
            candidate,
            materialization_id=row.id,
            expected_output_fingerprint="0" * 64,
            delivery_profile_key=_PROFILE,
            settings=get_settings(),
            storage=fixture.stage50.storage,
        )


def test_transformed_spec_fails_closed_on_unknown_materialization(
    session: Session, monkeypatch: Any
) -> None:
    fixture, _row = _persisted_materialization(session, monkeypatch)
    candidate = fixture.stage50.selection.candidate
    from app.render.execution.service import RenderExecutionError

    with pytest.raises(RenderExecutionError):
        build_transformed_render_spec(
            session,
            candidate,
            materialization_id=uuid.uuid4(),
            expected_output_fingerprint="a" * 64,
            delivery_profile_key=_PROFILE,
            settings=get_settings(),
            storage=fixture.stage50.storage,
        )
