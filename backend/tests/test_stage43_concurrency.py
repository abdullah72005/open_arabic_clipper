"""Stage 4.3 PostgreSQL concurrency and one-current constraint validation.

SQLite cannot meaningfully validate the partial unique index or row locking, so
this module is gated on ``CLIPFACTORY_TEST_POSTGRES_URL`` and skips otherwise.
Run it against the repository's compose PostgreSQL with that variable set.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from stage43_support import (
    FakeGovernanceSettings,
    install_selection_settings,
    make_result_spec,
    seed_selection_fixture,
)

from app.core.enums import GovernancePlanStatus, TransformationSelectionStatus
from app.db.base import Base
from app.models import TransformationPlanSelection
from app.transformation.selection.service import select_transformation_plan

_URL = os.environ.get("CLIPFACTORY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not _URL, reason="CLIPFACTORY_TEST_POSTGRES_URL is required for PostgreSQL concurrency"
)


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    assert _URL is not None
    test_engine = create_engine(_URL)
    _drop_non_portable_boolean_checks()
    Base.metadata.drop_all(test_engine)
    Base.metadata.create_all(test_engine)
    try:
        yield test_engine
    finally:
        Base.metadata.drop_all(test_engine)
        test_engine.dispose()


def _drop_non_portable_boolean_checks() -> None:
    """Remove SQLite-style ``boolean IN (0, 1)`` checks before PostgreSQL DDL.

    The frozen Stage 4.0/4.2 models declare booleans with SQLite-style
    ``IN (0, 1)`` checks that PostgreSQL rejects. This is a test-only DDL
    accommodation so the PostgreSQL partial-unique-index and row-locking
    behavior can be validated; it changes no production model or migration.
    """

    import re

    from sqlalchemy import CheckConstraint

    boolean_int = re.compile(r"(?<![<>!=])=\s*[01]\b|IN\s*\(\s*0\s*,\s*1\s*\)")
    for table in Base.metadata.tables.values():
        for constraint in list(table.constraints):
            if not isinstance(constraint, CheckConstraint):
                continue
            if boolean_int.search(str(constraint.sqltext)):
                table.constraints.discard(constraint)


def test_concurrent_selection_converges_on_one_current_row(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        fixture = seed_selection_fixture(
            session,
            settings=settings,
            result_specs=[
                make_result_spec(status=GovernancePlanStatus.APPROVED_FOR_SELECTION.value)
            ],
        )
        candidate_id = fixture.candidate.id
        session.commit()

    results: list[object] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with factory() as session:
                view = select_transformation_plan(session, candidate_id)
                assert view is not None
                row_id = view.row.id
                session.commit()
                results.append(row_id)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert len(results) == 2
    assert results[0] == results[1]

    with factory() as session:
        rows = session.scalars(
            select(TransformationPlanSelection).where(
                TransformationPlanSelection.clip_candidate_id == candidate_id
            )
        ).all()
        current = [row for row in rows if row.is_current]
        assert len(current) == 1
        assert current[0].status is TransformationSelectionStatus.PLAN_SELECTED
        assert current[0].selected_plan_id is not None


def test_partial_unique_index_rejects_second_current_row(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        fixture = seed_selection_fixture(
            session,
            settings=settings,
            result_specs=[
                make_result_spec(status=GovernancePlanStatus.APPROVED_FOR_SELECTION.value)
            ],
        )
        candidate_id = fixture.candidate.id
        source_id = fixture.source.id
        session.commit()

    with factory() as session:
        session.add(
            TransformationPlanSelection(
                source_video_id=source_id,
                clip_candidate_id=candidate_id,
                status=TransformationSelectionStatus.NO_SELECTABLE_PLAN,
                is_current=True,
                input_fingerprint="concurrent-a",
            )
        )
        session.commit()
        session.add(
            TransformationPlanSelection(
                source_video_id=source_id,
                clip_candidate_id=candidate_id,
                status=TransformationSelectionStatus.NO_SELECTABLE_PLAN,
                is_current=True,
                input_fingerprint="concurrent-b",
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()
