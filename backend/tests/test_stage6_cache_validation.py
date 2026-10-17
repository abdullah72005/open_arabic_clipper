"""Stage 6 completed-cache verification regressions (SOL cache-validation blocker).

A ``COMPLETE`` + ``cache_eligible`` materialization is only a *cache candidate*:
the normal POST path must still verify the exact effective bound transformed
render, the complete generation/output/repair binding, the live render scope, and
the actual artifact integrity before it may be claimed as cached work. A demoted
bound render, a deleted/corrupted artifact, or a stale output/repair binding must
never be returned as verified cache; the accepted components, provider budgets,
repair usage, and historical artifacts are preserved and the exact transform is
technically recovered through the existing render reactivation/re-handoff
boundary.

Real FFmpeg media; SYNTHETIC INTEGRATION evidence, not product acceptance.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session
from stage51_support import (
    FakeDetector,
    FakeFrameSampler,
    FakeSceneCutDetector,
    seed_stage51,
)

from app.api.app import _materialization_response
from app.composition.executor import build_visual_composition_executor
from app.composition.queue import get_or_create_plan_row
from app.core.enums import JobKind, JobStatus
from app.core.settings import get_settings
from app.db.base import Base
from app.db.session import create_session_factory
from app.materialization import queue as queue_module
from app.materialization.executor import build_materialization_executor
from app.materialization.providers import FakeCritiqueProvider, FakeWritingProvider
from app.materialization.queue import (
    PHASE_FINALIZE,
    RENDER_DISPATCH_FAILED_MARKER,
    queue_materialization,
)
from app.models import ClipMaterialization, ProcessingJob
from app.models.render_execution import RenderExecution

_ANNOTATION_COPY = (
    "That drop looks milder once you control for which teams already promoted heavily."
)
_TERMINAL = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


class _Capture:
    def __init__(self) -> None:
        self.rendered: list[tuple[str, str]] = []
        self.finalized: list[tuple[str, str]] = []

    def dispatch(self, render_id: object, job_id: object, force: bool = False) -> None:
        self.rendered.append((str(render_id), str(job_id)))

    def dispatch_finalization(self, candidate_id: str, job_id: str) -> None:
        self.finalized.append((candidate_id, job_id))


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _session() -> Session:
    return create_session_factory()()


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


def _prepare_to_awaiting_render(
    engine: Engine, monkeypatch: Any, *, media_writer: Any
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID, uuid.UUID, Any]:
    """Run prepare through the real materialization executor to AWAITING_RENDER.

    Stops *before* executing the queued render. Returns ``(candidate_id,
    materialization_id, render_id, render_job_id, storage)``.
    """

    monkeypatch.setattr(queue_module, "_dispatch", lambda *args, **kwargs: None)
    # Keep any accidental broker delivery out of the loop (the capture dispatcher
    # still records render/finalizer deliveries on the POST under test).
    from app.workers import tasks as tasks_module

    monkeypatch.setattr(tasks_module.run_render_execution, "delay", lambda *a, **k: None)
    monkeypatch.setattr(
        tasks_module.run_materialization_finalization, "delay", lambda *a, **k: None
    )
    monkeypatch.setattr(tasks_module.run_materialization_job, "delay", lambda *a, **k: None)
    settings = get_settings()
    from app.services.storage import StorageService

    storage = StorageService(settings.storage_root)

    session = _session()
    try:
        fixture = seed_stage51(
            session, monkeypatch, narration="NONE", media_writer=media_writer
        )
        candidate = fixture.stage50.selection.candidate
        candidate_id = candidate.id
        _ready_composition(session, fixture)
        session.commit()
        prepare = ProcessingJob(
            source_video_id=candidate.source_video_id, kind=JobKind.MATERIALIZATION
        )
        session.add(prepare)
        session.commit()
        executor = build_materialization_executor(
            session,
            storage,
            settings,
            writing=FakeWritingProvider({1: _ANNOTATION_COPY}),
            critique=FakeCritiqueProvider(accepted={1: True}),
        )
        executor.set_active_job(prepare.id)
        executor.execute(str(candidate_id))
        session.commit()
        envelope = session.scalars(
            select(ClipMaterialization).where(ClipMaterialization.is_current.is_(True))
        ).one()
        materialization_id = envelope.id
        assert envelope.lifecycle.value == "AWAITING_RENDER", envelope.reason_codes
        render_id = envelope.render_execution_id
        assert render_id is not None
        render_job = session.scalars(
            select(ProcessingJob).where(ProcessingJob.render_execution_id == render_id)
        ).first()
        assert render_job is not None
        render_job_id = render_job.id
    finally:
        session.close()

    return (
        uuid.UUID(str(candidate_id)),
        materialization_id,
        render_id,
        uuid.UUID(str(render_job_id)),
        storage,
    )


def _complete_materialization(
    engine: Engine, monkeypatch: Any, *, media_writer: Any
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID, Any]:
    """Run prepare -> real render -> callback -> finalizer to a verified COMPLETE.

    Returns ``(candidate_id, materialization_id, render_id, storage)``.
    """

    from app.workers.tasks import run_render_execution

    candidate_id, materialization_id, render_id, render_job_id, storage = (
        _prepare_to_awaiting_render(engine, monkeypatch, media_writer=media_writer)
    )

    run_render_execution(str(render_id), str(render_job_id))

    session = _session()
    try:
        render = session.get(RenderExecution, render_id)
        assert render is not None
        assert render.lifecycle.value == "COMPLETE", render.reason_codes
        assert render.is_current is True
        candidate_uuid = render.clip_candidate_id
    finally:
        session.close()

    from app.workers.tasks import _request_finalization_after_render

    _request_finalization_after_render(render_id)

    session = _session()
    try:
        finalizer = session.scalars(
            select(ProcessingJob)
            .where(
                ProcessingJob.materialization_id == materialization_id,
                ProcessingJob.kind == JobKind.MATERIALIZATION,
                ProcessingJob.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
            )
            .order_by(ProcessingJob.created_at.desc())
        ).first()
        assert finalizer is not None
        finalizer_id = finalizer.id
    finally:
        session.close()

    from app.workers.tasks import run_materialization_finalization

    run_materialization_finalization(str(candidate_uuid), str(finalizer_id))

    session = _session()
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        assert envelope.lifecycle.value == "COMPLETE", envelope.reason_codes
        assert envelope.cache_eligible is True
        assert envelope.human_review_required is True
        assert envelope.publication_ready is False
        assert (envelope.metrics or {}).get("assessment_fingerprint")
    finally:
        session.close()

    return uuid.UUID(str(candidate_id)), materialization_id, render_id, storage


def _post(
    candidate_id: uuid.UUID, capture: _Capture
) -> Any:
    session = _session()
    try:
        outcome = queue_materialization(
            session, candidate_id, priority="CANDIDATE", render_dispatcher=capture
        )
        session.commit()
        return outcome
    finally:
        session.close()


def _fresh_state(materialization_id: uuid.UUID) -> dict[str, object]:
    """Read the persisted envelope on a brand-new session (no identity-map state)."""

    session = _session()
    try:
        row = session.get(ClipMaterialization, materialization_id)
        assert row is not None
        return {
            "lifecycle": row.lifecycle.value,
            "semantic_outcome": row.semantic_outcome.value if row.semantic_outcome else None,
            "human_review_required": bool(row.human_review_required),
            "publication_ready": bool(row.publication_ready),
            "cache_eligible": bool(row.cache_eligible),
            "render_execution_id": row.render_execution_id,
            "assessment_fingerprint": (row.metrics or {}).get("assessment_fingerprint"),
            "component_checkpoints": dict(row.component_checkpoints or {}),
            "call_ledger": dict(row.call_ledger or {}),
            "repair_count": int(row.repair_count or 0),
            "reason_codes": list(row.reason_codes or []),
        }
    finally:
        session.close()


def _api_response(materialization_id: uuid.UUID) -> Any:
    """Fresh read through the exact GET response constructor (no writes)."""

    session = _session()
    try:
        row = session.get(ClipMaterialization, materialization_id)
        assert row is not None
        return _materialization_response(session, row)
    finally:
        session.close()


def test_valid_completed_replay_is_cached_without_duplicated_work(
    engine: Engine, monkeypatch: Any
) -> None:
    from test_stage6_real_executor import _write_source

    candidate_id, materialization_id, render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )

    before_jobs = _materialization_job_ids(materialization_id)
    capture = _Capture()
    outcome = _post(candidate_id, capture)

    assert outcome.cached is True
    assert outcome.job_id is None
    assert outcome.queued is False
    # Zero duplicated work: no render re-dispatched, no finalizer queued, no new job.
    assert capture.rendered == []
    assert capture.finalized == []
    assert _materialization_job_ids(materialization_id) == before_jobs
    session = _session()
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None and envelope.render_execution_id == render_id
        assert envelope.lifecycle.value == "COMPLETE"
    finally:
        session.close()


def test_demoted_bound_render_is_recovered_not_cached(
    engine: Engine, monkeypatch: Any
) -> None:
    from test_stage6_real_executor import _write_source

    candidate_id, materialization_id, render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    before = _fresh_state(materialization_id)
    assert before["human_review_required"] is True
    assert before["assessment_fingerprint"]
    with Session(engine) as session:
        render = session.get(RenderExecution, render_id)
        assert render is not None
        render.is_current = False
        session.commit()

    capture = _Capture()
    outcome = _post(candidate_id, capture)

    assert outcome.cached is False
    session = _session()
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        # The exact transform is recovered, not regenerated semantically: no new
        # materialization prepare job, and the bound render identity is restored.
        render = session.get(RenderExecution, render_id)
        assert render is not None and render.is_current is True
        assert envelope.render_execution_id == render_id
    finally:
        session.close()
    # An intact-artifact reactivation proceeds straight to a finalizer.
    assert outcome.phase == PHASE_FINALIZE
    assert outcome.job_id is not None
    assert capture.finalized and not capture.rendered
    # Review eligibility is cleared while finalization is pending (no stale review).
    pending = _fresh_state(materialization_id)
    assert pending["lifecycle"] == "AWAITING_RENDER"
    assert pending["human_review_required"] is False
    assert pending["publication_ready"] is False

    # A verified current final assessment restores the correct flags, with no
    # duplicated provider work (identical assessment identity; untouched ledger).
    from app.workers.tasks import run_materialization_finalization

    run_materialization_finalization(str(candidate_id), str(outcome.job_id))
    after = _fresh_state(materialization_id)
    assert after["lifecycle"] == "COMPLETE"
    assert after["human_review_required"] is True
    assert after["publication_ready"] is False
    assert after["cache_eligible"] is True
    assert after["assessment_fingerprint"] == before["assessment_fingerprint"]
    assert after["component_checkpoints"] == before["component_checkpoints"]
    assert after["call_ledger"] == before["call_ledger"]
    assert after["repair_count"] == before["repair_count"]


@pytest.mark.parametrize("mode", ["deleted", "corrupted"])
def test_missing_or_corrupted_artifact_is_recovered_not_cached(
    engine: Engine, monkeypatch: Any, mode: str
) -> None:
    from test_stage6_real_executor import _write_source

    candidate_id, materialization_id, render_id, storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    session = _session()
    try:
        render = session.get(RenderExecution, render_id)
        assert render is not None
        reference = render.artifact_reference or {}
        artifact = storage.storage_root / str(reference["relative_path"])
        assert artifact.is_file()
    finally:
        session.close()
    if mode == "deleted":
        artifact.unlink()
    else:
        artifact.write_bytes(artifact.read_bytes() + b"corruption")

    capture = _Capture()
    outcome = _post(candidate_id, capture)

    assert outcome.cached is False
    # A bounded technical re-render is queued through the existing render boundary;
    # the accepted semantic components are preserved (no provider prepare job).
    assert capture.rendered
    # Invalidating the completed assessment atomically clears current final-review
    # eligibility: AWAITING_RENDER is pending finalization, never review-ready or
    # publication-ready, while the historical assessment fingerprint, accepted
    # components, and ledger survive for the recovered generation.
    state = _fresh_state(materialization_id)
    assert state["lifecycle"] == "AWAITING_RENDER"
    assert state["human_review_required"] is False
    assert state["publication_ready"] is False
    assert state["semantic_outcome"] is None
    assert state["cache_eligible"] is False
    assert state["assessment_fingerprint"]
    assert state["component_checkpoints"]
    response = _api_response(materialization_id)
    assert response.lifecycle == "AWAITING_RENDER"
    assert response.human_review_required is False
    assert response.publication_ready is False
    assert response.semantic_outcome is None


def test_stale_output_binding_is_recovered_not_cached(
    engine: Engine, monkeypatch: Any
) -> None:
    from test_stage6_real_executor import _write_source

    candidate_id, materialization_id, render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        # The bound render's expected output fingerprint no longer matches the
        # current envelope output/repair generation.
        envelope.output_fingerprint = "9" * 64
        session.commit()

    capture = _Capture()
    outcome = _post(candidate_id, capture)

    assert outcome.cached is False
    # A fresh handoff for the changed generation is queued; the exact binding is
    # never returned as a verified cache claim.
    assert capture.rendered
    session = _session()
    try:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        assert envelope.render_execution_id is not None
        assert envelope.render_execution_id != render_id
    finally:
        session.close()
    state = _fresh_state(materialization_id)
    assert state["human_review_required"] is False
    assert state["publication_ready"] is False


def test_failed_render_handoff_delivery_is_recovered_by_next_post(
    engine: Engine, monkeypatch: Any
) -> None:
    """A recorded broker-delivery failure is recovered by the next normal POST.

    The prepare job commits AWAITING_RENDER and its exact render binding; if the
    broker delivery of that render fails, the durable ``RENDER_DISPATCH_FAILED``
    marker lets the next POST redeliver the *same* pending render job with no new
    generation and no semantic regeneration. An unfinished generation never
    requests human review.
    """

    from test_stage6_real_executor import _write_source

    candidate_id, materialization_id, render_id, render_job_id, _storage = (
        _prepare_to_awaiting_render(engine, monkeypatch, media_writer=_write_source)
    )
    # Simulate the prepare->render handoff dispatch having failed.
    with Session(engine) as session:
        job = session.get(ProcessingJob, render_job_id)
        assert job is not None
        job.error_code = RENDER_DISPATCH_FAILED_MARKER
        session.commit()

    pending = _fresh_state(materialization_id)
    assert pending["lifecycle"] == "AWAITING_RENDER"
    assert pending["human_review_required"] is False
    assert pending["publication_ready"] is False

    capture = _Capture()
    outcome = _post(candidate_id, capture)

    assert outcome.cached is False
    # The exact pending render job is redelivered (not a new generation).
    assert capture.rendered and not capture.finalized
    with Session(engine) as session:
        job = session.get(ProcessingJob, render_job_id)
        assert job is not None
        assert job.error_code is None
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        assert envelope.render_execution_id == render_id
        assert envelope.human_review_required is False
        assert envelope.publication_ready is False
    response = _api_response(materialization_id)
    assert response.lifecycle == "AWAITING_RENDER"
    assert response.human_review_required is False
    assert response.publication_ready is False


def _materialization_job_ids(materialization_id: uuid.UUID) -> set[str]:
    session = _session()
    try:
        return {
            str(job.id)
            for job in session.scalars(
                select(ProcessingJob).where(
                    ProcessingJob.materialization_id == materialization_id
                )
            ).all()
        }
    finally:
        session.close()


def _remove_committed_assessment(materialization_id: uuid.UUID) -> None:
    session = _session()
    try:
        row = session.get(ClipMaterialization, materialization_id)
        assert row is not None
        metrics = dict(row.metrics or {})
        assert metrics.pop("assessment_fingerprint", None)
        row.metrics = metrics
        session.commit()
    finally:
        session.close()


def test_missing_committed_assessment_is_recovered_not_cached(
    engine: Engine, monkeypatch: Any
) -> None:
    """A COMPLETE envelope whose committed assessment is gone is not a cache hit.

    The exact completed render remains valid, so recovery runs ONE finalizer over
    that same render -- no provider repeat and no re-render.
    """

    from test_stage6_real_executor import _write_source

    candidate_id, materialization_id, render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    before = _fresh_state(materialization_id)
    assert before["assessment_fingerprint"]
    before_jobs = _materialization_job_ids(materialization_id)
    _remove_committed_assessment(materialization_id)

    capture = _Capture()
    outcome = _post(candidate_id, capture)

    assert outcome.cached is False
    # One finalizer over the exact render; no new technical render.
    assert capture.finalized and not capture.rendered
    pending = _fresh_state(materialization_id)
    assert pending["lifecycle"] == "AWAITING_RENDER"
    assert pending["human_review_required"] is False
    assert pending["publication_ready"] is False
    assert pending["render_execution_id"] == render_id
    assert len(_materialization_job_ids(materialization_id)) == len(before_jobs) + 1

    # Running the recovered finalizer restores the committed assessment with no
    # duplicated provider work (untouched ledger/checkpoints/repair usage).
    from app.workers.tasks import run_materialization_finalization

    assert outcome.job_id is not None
    run_materialization_finalization(str(candidate_id), str(outcome.job_id))
    after = _fresh_state(materialization_id)
    assert after["lifecycle"] == "COMPLETE"
    assert after["human_review_required"] is True
    assert after["cache_eligible"] is True
    assert after["assessment_fingerprint"]
    assert after["component_checkpoints"] == before["component_checkpoints"]
    assert after["call_ledger"] == before["call_ledger"]
    assert after["repair_count"] == before["repair_count"]


def test_resume_valid_completed_is_verified_cache_without_work(
    engine: Engine, monkeypatch: Any
) -> None:
    """Resume of a genuinely complete generation converges on verified cache."""

    from test_stage6_real_executor import _write_source

    from app.materialization.queue import resume_materialization_by_id

    candidate_id, materialization_id, _render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    before_jobs = _materialization_job_ids(materialization_id)
    capture = _Capture()
    session = _session()
    try:
        outcome = resume_materialization_by_id(
            session, materialization_id, settings=get_settings(), dispatcher=capture
        )
        session.commit()
    finally:
        session.close()
    assert outcome.cached is True
    assert outcome.job_id is None
    assert capture.rendered == [] and capture.finalized == []
    assert _materialization_job_ids(materialization_id) == before_jobs


def test_resume_missing_assessment_recovers_one_finalizer_without_render(
    engine: Engine, monkeypatch: Any
) -> None:
    """Resume of a valid render with a missing assessment runs one finalizer."""

    from test_stage6_real_executor import _write_source

    from app.materialization.queue import resume_materialization_by_id

    candidate_id, materialization_id, render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    before = _fresh_state(materialization_id)
    _remove_committed_assessment(materialization_id)
    capture = _Capture()
    session = _session()
    try:
        outcome = resume_materialization_by_id(
            session, materialization_id, settings=get_settings(), dispatcher=capture
        )
        session.commit()
    finally:
        session.close()
    assert outcome.cached is False
    assert capture.finalized and not capture.rendered
    pending = _fresh_state(materialization_id)
    assert pending["lifecycle"] == "AWAITING_RENDER"
    assert pending["render_execution_id"] == render_id

    from app.workers.tasks import run_materialization_finalization

    assert outcome.job_id is not None
    run_materialization_finalization(str(candidate_id), str(outcome.job_id))
    after = _fresh_state(materialization_id)
    assert after["lifecycle"] == "COMPLETE"
    assert after["assessment_fingerprint"] == before["assessment_fingerprint"]
    assert after["call_ledger"] == before["call_ledger"]


def _tampering_copyfile(monkeypatch: Any) -> None:
    """Corrupt the destination authored.ass AFTER an otherwise-correct copy."""

    import shutil as _shutil

    real = _shutil.copyfile

    def tamper(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        result = real(src, dst, *args, **kwargs)
        if str(dst).endswith("authored.ass"):
            with open(dst, "ab") as handle:
                handle.write(b"; corrupted-after-copy")
        return result

    monkeypatch.setattr(_shutil, "copyfile", tamper)


def test_export_verifies_copied_destination_bytes(
    engine: Engine, monkeypatch: Any
) -> None:
    """A genuine export verifies the COPIED destination bytes, not the source."""

    import json
    from pathlib import Path

    from test_stage6_real_executor import _write_source

    from app.materialization.acceptance_export import export_materializations

    _candidate_id, materialization_id, _render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    settings = get_settings()
    session = _session()
    try:
        genuine = export_materializations(session, settings, [materialization_id])
    finally:
        session.close()
    genuine_case = genuine["cases"][0]
    assert genuine_case["genuine_acceptance"] is True, genuine_case
    genuine_index = json.loads(
        (Path(genuine_case["directory"]) / "index.json").read_text(encoding="utf-8")
    )
    assert genuine_index["genuine_acceptance"] is True
    assert not [
        reason
        for reason in genuine_index["genuine_acceptance_reasons"]
        if reason.startswith("EXPORT_ASSET")
    ]

    # Corrupt only the destination copy of the bound ASS: the copied bytes no longer
    # match the authoritative expected hash, so acceptance must be invalidated.
    _tampering_copyfile(monkeypatch)
    session = _session()
    try:
        tampered = export_materializations(session, settings, [materialization_id])
    finally:
        session.close()
    tampered_case = tampered["cases"][0]
    assert tampered_case["genuine_acceptance"] is False
    tampered_index = json.loads(
        (Path(tampered_case["directory"]) / "index.json").read_text(encoding="utf-8")
    )
    assert any(
        reason.startswith("EXPORT_ASSET_MISMATCH")
        for reason in tampered_index["genuine_acceptance_reasons"]
    )
    assert tampered_index["authored_ass_sha256"] != tampered_index["authored_ass_expected_sha256"]


def test_diagnostic_export_preserves_last_valid_package(
    engine: Engine, monkeypatch: Any
) -> None:
    """A diagnostic export never replaces a valid published package or its catalog reference."""

    import json
    from pathlib import Path

    from test_stage6_real_executor import _write_source

    from app.materialization.acceptance_export import export_materializations

    _candidate_id, materialization_id, _render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    settings = get_settings()
    session = _session()
    try:
        genuine = export_materializations(session, settings, [materialization_id])
    finally:
        session.close()
    genuine_case = genuine["cases"][0]
    assert genuine_case["genuine_acceptance"] is True
    case = genuine_case["case"]
    valid_identity = genuine_case["package_identity"]
    valid_directory = Path(genuine_case["directory"])
    # Published into an immutable versioned directory: packages/<case>/<identity>/.
    assert valid_directory.parts[-3:] == ("packages", case, valid_identity)
    valid_index_before = (valid_directory / "index.json").read_text(encoding="utf-8")
    root = valid_directory.parents[2]

    _tampering_copyfile(monkeypatch)
    session = _session()
    try:
        diagnostic = export_materializations(session, settings, [materialization_id])
    finally:
        session.close()
    diagnostic_case = diagnostic["cases"][0]
    assert diagnostic_case["genuine_acceptance"] is False
    assert diagnostic_case["published"] is False
    diagnostic_directory = Path(str(diagnostic_case["diagnostic_directory"]))
    assert diagnostic_directory.is_dir()
    assert diagnostic_directory != valid_directory
    diagnostic_index = json.loads(
        (diagnostic_directory / "index.json").read_text(encoding="utf-8")
    )
    assert diagnostic_index["diagnostic"] is True
    # The last valid package is preserved untouched (immutable versioned directory).
    assert (valid_directory / "index.json").read_text(encoding="utf-8") == valid_index_before
    preserved = json.loads((valid_directory / "index.json").read_text(encoding="utf-8"))
    assert preserved["genuine_acceptance"] is True

    # The versioned catalog keeps the published valid reference and records the
    # diagnostic attempt separately; the published reference is never replaced.
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["version"] == 2
    entry = catalog["cases"][case]
    assert entry["published"]["identity"] == valid_identity
    assert entry["diagnostic_attempts"]
    assert all(
        attempt["genuine_acceptance"] is False for attempt in entry["diagnostic_attempts"]
    )


def test_export_catalog_merge_preserves_unrelated_cases(
    engine: Engine, monkeypatch: Any
) -> None:
    """A later export merges into the catalog without dropping unrelated cases."""

    import json
    from pathlib import Path

    from test_stage6_real_executor import _write_source

    from app.materialization.acceptance_export import export_materializations

    settings = get_settings()
    _candidate_id, materialization_id, _render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    session = _session()
    try:
        first_export = export_materializations(
            session, settings, [materialization_id], case_labels=["case-alpha"]
        )
    finally:
        session.close()
    first_case = first_export["cases"][0]["case"]
    first_identity = first_export["cases"][0]["package_identity"]
    assert first_case == "case-alpha"

    session = _session()
    try:
        second_export = export_materializations(
            session, settings, [materialization_id], case_labels=["case-beta"]
        )
    finally:
        session.close()
    second_case = second_export["cases"][0]["case"]
    assert second_case == "case-beta"

    root = Path(first_export["root"])
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    # Both cases survive: the later export did not drop the earlier one.
    assert first_case in catalog["cases"]
    assert second_case in catalog["cases"]
    assert catalog["cases"][first_case]["published"]["identity"] == first_identity
    # Both immutable packages remain on disk.
    assert (Path(first_export["cases"][0]["directory"]) / "index.json").is_file()
    assert (Path(second_export["cases"][0]["directory"]) / "index.json").is_file()


def test_export_failure_before_catalog_publication_keeps_previous_reference(
    engine: Engine, monkeypatch: Any
) -> None:
    """A catalog-publication failure leaves the published reference and bytes usable."""

    import json
    from pathlib import Path

    import pytest
    from test_stage6_real_executor import _write_source

    import app.materialization.acceptance_export as export_module
    from app.materialization.acceptance_export import export_materializations

    _candidate_id, materialization_id, _render_id, _storage = _complete_materialization(
        engine, monkeypatch, media_writer=_write_source
    )
    settings = get_settings()
    session = _session()
    try:
        genuine = export_materializations(session, settings, [materialization_id])
    finally:
        session.close()
    case = genuine["cases"][0]["case"]
    valid_identity = genuine["cases"][0]["package_identity"]
    valid_directory = Path(genuine["cases"][0]["directory"])
    valid_index_before = (valid_directory / "index.json").read_text(encoding="utf-8")
    root = valid_directory.parents[2]

    # A later (diagnostic-causing) export stages and publishes its package, but the
    # catalog replacement fails: the previous reference and bytes must survive.
    _tampering_copyfile(monkeypatch)

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("catalog write failed")

    monkeypatch.setattr(export_module, "_write_catalog_atomic", fail_write)
    session = _session()
    try:
        with pytest.raises(RuntimeError):
            export_materializations(session, settings, [materialization_id])
    finally:
        session.close()

    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["cases"][case]["published"]["identity"] == valid_identity
    assert catalog["cases"][case]["diagnostic_attempts"] == []
    assert (valid_directory / "index.json").read_text(encoding="utf-8") == valid_index_before
