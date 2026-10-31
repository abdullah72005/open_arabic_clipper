"""PostgreSQL locking tests for operator candidate registration.

These tests are gated on ``CLIPFACTORY_TEST_POSTGRES_URL`` and must run against a
disposable isolated PostgreSQL instance, never the operational acceptance or
default database.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from uuid import UUID

import pytest
from registration_support import (
    EXPECTED_KEY,
    OPERATOR_REASON,
    SELECTED_END,
    SELECTED_START,
    install_source,
    make_draft,
    make_outcome,
)
from sqlalchemy import Engine, create_engine, func, select, text
from sqlalchemy.orm import sessionmaker

from app.candidates.executor import CandidateAnalysisExecutor
from app.candidates.policy import DEFAULT_CONFIG
from app.candidates.registration import register_operator_candidate
from app.db.base import Base
from app.models import ClipCandidate, SourceVideo, Transcript

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL, reason="CLIPFACTORY_TEST_POSTGRES_URL is required for registration locking"
)


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    assert _URL is not None
    test_engine = create_engine(_URL)
    with test_engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    Base.metadata.create_all(test_engine)
    try:
        yield test_engine
    finally:
        with test_engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        test_engine.dispose()


def test_concurrent_registration_converges_on_one_identity(engine: Engine) -> None:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        source, _ = install_source(session)
        source_id = source.id

    barrier = threading.Barrier(2)
    results: list[tuple[bool, UUID]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with factory() as session:
                barrier.wait(timeout=10)
                result = register_operator_candidate(
                    session, source_id, SELECTED_START, SELECTED_END, OPERATOR_REASON
                )
                session.commit()
                results.append((result.created, result.candidate.id))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not errors, errors
    assert len(results) == 2
    assert sum(1 for created, _ in results if created) == 1
    assert len({row_id for _, row_id in results}) == 1

    with factory() as session:
        rows = list(
            session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source_id))
        )
        assert len(rows) == 1
        assert rows[0].candidate_key == EXPECTED_KEY
        assert rows[0].is_current is True
        marker = rows[0].provenance_snapshot["operator_selection"]
        assert marker["origin"] == "OPERATOR"


def test_registration_and_persistence_lock_wait_preserves_ownership(
    engine: Engine,
) -> None:
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        source, transcript = install_source(session)
        source_id = source.id
        transcript_id = transcript.id

    registered = threading.Event()
    release = threading.Event()
    persist_started = threading.Event()
    persist_done = threading.Event()
    errors: list[BaseException] = []

    def register_worker() -> None:
        try:
            with factory() as session:
                register_operator_candidate(
                    session, source_id, SELECTED_START, SELECTED_END, OPERATOR_REASON
                )
                registered.set()
                assert release.wait(timeout=20)
                session.commit()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    draft = make_draft(
        candidate_key=EXPECTED_KEY,
        start_segment_index=230,
        end_segment_index=245,
        start_time=SELECTED_START,
        end_time=SELECTED_END,
        text="same-key automatic draft under lock",
    )

    def persist_worker() -> None:
        try:
            with factory() as session:
                assert registered.wait(timeout=20)
                transcript_row = session.get(Transcript, transcript_id)
                source_obj = session.get(SourceVideo, source_id)
                assert transcript_row is not None
                assert source_obj is not None
                persist_started.set()
                executor = CandidateAnalysisExecutor(session=session, config=DEFAULT_CONFIG)
                executor._persist(
                    source_obj,
                    transcript_row,
                    make_outcome(draft),
                    "input-fp",
                    0.1,
                )
                persist_done.set()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
            persist_done.set()

    register_thread = threading.Thread(target=register_worker)
    persist_thread = threading.Thread(target=persist_worker)
    register_thread.start()
    persist_thread.start()

    assert registered.wait(timeout=20)
    assert persist_started.wait(timeout=20)
    time.sleep(0.3)
    assert not persist_done.is_set(), "persistence must wait for the registration source lock"
    release.set()

    register_thread.join(timeout=20)
    persist_thread.join(timeout=20)

    assert not errors, errors
    assert persist_done.is_set()

    with factory() as session:
        rows = list(
            session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source_id))
        )
        assert len(rows) == 1
        assert rows[0].id is not None
        assert rows[0].candidate_key == EXPECTED_KEY
        assert rows[0].is_current is True
        assert rows[0].candidate_analysis_id is None
        marker = rows[0].provenance_snapshot["operator_selection"]
        assert marker["origin"] == "OPERATOR"
        assert marker["raw_span_fingerprint"]
        assert session.scalar(select(func.count()).select_from(ClipCandidate)) == 1
