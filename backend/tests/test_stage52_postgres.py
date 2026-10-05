"""Stage 5.2 PostgreSQL-gated concurrency and admission tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.enums import (
    CandidateDisposition,
    ContentType,
    OriginalityRisk,
    RenderArtifactPurpose,
    RenderExecutionLifecycle,
    RightsRisk,
)
from app.db.base import Base
from app.models import ClipCandidate, SourceVideo
from app.models.render_execution import RenderExecution
from app.render.execution.concurrency import PostgresRenderAdmission

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not _URL, reason="CLIPFACTORY_TEST_POSTGRES_URL is required for PostgreSQL tests"
)

_DATABASE = "clipfactory_stage52_concurrency_test"


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    assert _URL is not None
    admin_url = str(make_url(_URL).set(database="postgres").render_as_string(hide_password=False))
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{_DATABASE}" WITH (FORCE)'))
            connection.execute(text(f'CREATE DATABASE "{_DATABASE}"'))
    finally:
        admin.dispose()
    target = str(make_url(_URL).set(database=_DATABASE).render_as_string(hide_password=False))
    engine = create_engine(target)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()
        cleanup = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        try:
            with cleanup.connect() as connection:
                connection.execute(text(f'DROP DATABASE IF EXISTS "{_DATABASE}" WITH (FORCE)'))
        finally:
            cleanup.dispose()


def _candidate(session: Session) -> ClipCandidate:
    source = SourceVideo(source_uri="/tmp/stage52-concurrency.mp4", content_hash="h")
    session.add(source)
    session.flush()
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key="stage52-concurrency",
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
    return candidate


def test_global_render_admission_excludes_second_connection(engine: Engine) -> None:
    first = PostgresRenderAdmission(engine)
    second = PostgresRenderAdmission(engine)
    assert first.acquire(wait_seconds=0, cancel_check=lambda: False) is True
    assert second.acquire(wait_seconds=0, cancel_check=lambda: False) is False
    first.release()
    assert second.acquire(wait_seconds=0, cancel_check=lambda: False) is True
    second.release()


def test_scoped_current_unique_index_rejects_duplicate_scope(engine: Engine) -> None:
    with Session(engine) as session:
        candidate = _candidate(session)
        session.add(
            RenderExecution(
                source_video_id=candidate.source_video_id,
                clip_candidate_id=candidate.id,
                artifact_purpose=RenderArtifactPurpose.CORE_SOURCE_VALIDATION,
                lifecycle=RenderExecutionLifecycle.QUEUED,
                is_current=True,
                delivery_profile_key="MP4_H264_AAC_1080X1920_V1",
            )
        )
        session.commit()
        session.add(
            RenderExecution(
                source_video_id=candidate.source_video_id,
                clip_candidate_id=candidate.id,
                artifact_purpose=RenderArtifactPurpose.CORE_SOURCE_VALIDATION,
                lifecycle=RenderExecutionLifecycle.QUEUED,
                is_current=True,
                delivery_profile_key="MP4_H264_AAC_1080X1920_V1",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
