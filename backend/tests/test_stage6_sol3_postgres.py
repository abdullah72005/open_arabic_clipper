"""Stage 6 ownership/currentness regressions against disposable PostgreSQL.

Gated on ``CLIPFACTORY_TEST_POSTGRES_URL``. PostgreSQL enforces the scoped
partial unique index for real, so this proves the scope fencing (changed-input
supersession and stale-worker rejection) under genuine database constraints.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from dataclasses import replace

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from test_stage6_sol3_regressions import _inputs, _run

from app.core.enums import CandidateDisposition, ContentType, OriginalityRisk, RightsRisk
from app.db.base import Base
from app.materialization.service import MaterializationStaleError, persist_materialization
from app.models import ClipCandidate, ClipMaterialization, SourceVideo

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL, reason="CLIPFACTORY_TEST_POSTGRES_URL is required for Stage 6 Postgres"
)

_DATABASE = f"clipfactory_stage6_{uuid.uuid4().hex[:12]}"


def _admin_url() -> str:
    assert _URL is not None
    return str(make_url(_URL).set(database="postgres").render_as_string(hide_password=False))


def _target_url() -> str:
    assert _URL is not None
    return str(make_url(_URL).set(database=_DATABASE).render_as_string(hide_password=False))


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    admin = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        from sqlalchemy import text

        connection.execute(text(f'DROP DATABASE IF EXISTS "{_DATABASE}" WITH (FORCE)'))
        connection.execute(text(f'CREATE DATABASE "{_DATABASE}"'))
    admin.dispose()
    target = create_engine(_target_url())
    # PostgreSQL enforces the scoped partial unique index defined on the model.
    Base.metadata.create_all(target)
    try:
        yield target
    finally:
        target.dispose()
        admin = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
        with admin.connect() as connection:
            from sqlalchemy import text

            connection.execute(text(f'DROP DATABASE IF EXISTS "{_DATABASE}" WITH (FORCE)'))
        admin.dispose()


def _seed(session: Session) -> tuple[SourceVideo, ClipCandidate]:
    unique = uuid.uuid4().hex
    source = SourceVideo(source_uri=f"/tmp/stage6p-{unique}.mp4", content_hash=f"stage6p-{unique}")
    session.add(source)
    session.flush()
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key=f"stage6p-{unique}",
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
    return source, candidate


def test_changed_input_supersession_on_postgres(engine: Engine) -> None:
    with Session(engine) as session:
        source, candidate = _seed(session)
        inputs_a = _inputs(source_id=str(source.id), candidate_id=str(candidate.id))
        changed = replace(inputs_a, priority="FINAL_CLIP")
        row_a = persist_materialization(session, inputs_a, _run(inputs_a), claim_version=1)
        row_b = persist_materialization(session, changed, _run(changed), claim_version=2)
        assert row_a.is_current is False
        assert row_b.is_current is True
        with pytest.raises(MaterializationStaleError):
            persist_materialization(session, inputs_a, _run(inputs_a), claim_version=1)
        current = session.scalars(
            select(ClipMaterialization).where(ClipMaterialization.is_current.is_(True))
        ).one()
        assert current.id == row_b.id


def test_scope_has_exactly_one_current_row_on_postgres(engine: Engine) -> None:
    with Session(engine) as session:
        source, candidate = _seed(session)
        inputs_a = _inputs(source_id=str(source.id), candidate_id=str(candidate.id))
        changed = replace(inputs_a, priority="FINAL_CLIP")
        persist_materialization(session, inputs_a, _run(inputs_a), claim_version=1)
        persist_materialization(session, changed, _run(changed), claim_version=2)
        persist_materialization(session, inputs_a, _run(inputs_a), claim_version=3)
        currents = session.scalars(
            select(ClipMaterialization).where(
                ClipMaterialization.clip_candidate_id == candidate.id,
                ClipMaterialization.is_current.is_(True),
            )
        ).all()
        assert len(currents) == 1


def test_worker_mutation_requires_running_job_on_postgres(engine: Engine) -> None:
    from app.core.enums import JobKind, JobStatus
    from app.models import ProcessingJob

    with Session(engine) as session:
        source, candidate = _seed(session)
        inputs = _inputs(source_id=str(source.id), candidate_id=str(candidate.id))
        job = ProcessingJob(source_video_id=source.id, kind=JobKind.MATERIALIZATION)
        session.add(job)
        session.flush()
        with pytest.raises(MaterializationStaleError):
            persist_materialization(session, inputs, _run(inputs), active_job_id=job.id)
        job.status = JobStatus.RUNNING
        job.claim_version = 1
        session.commit()
        row = persist_materialization(
            session, inputs, _run(inputs), active_job_id=job.id, claim_version=1
        )
        assert row.active_job_id == job.id
