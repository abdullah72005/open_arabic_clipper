"""Stage 5.2 render-execution executor with durable claim fencing.

Duplicate/redelivered Celery invocations are fenced by an atomic
``QUEUED/FAILED -> RUNNING`` claim that advances ``claim_version``. Every
worker-owned mutation is fenced on that token and ``status == RUNNING``, so a
superseded worker can never finalize, cancel, or fail a newer run. Cancellation
is cooperative and read through a fresh scalar query; the narrow global
admission lock is held on a dedicated connection for the render/QC lifetime and
released on every exit path.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.core.enums import JobStatus, RenderExecutionLifecycle
from app.core.settings import Settings
from app.db.session import create_session_factory
from app.models import ProcessingJob
from app.models.render_execution import RenderExecution
from app.pipeline.executor import StageCancelled
from app.render.execution.compiler import compile_render
from app.render.execution.concurrency import RenderAdmission, render_admission_for
from app.render.execution.policy import (
    RENDER_ADMISSION_UNAVAILABLE,
)
from app.render.execution.qc import check_render_artifact
from app.render.execution.runner import (
    RenderCancelled,
    RenderProcessError,
    RenderTimeout,
    run_compiled_render,
)
from app.render.execution.service import (
    RenderExecutionError,
    build_render_spec,
    cache_hit,
    finalize_success,
    mark_blocked,
    mark_cancelled,
    mark_failed,
    request_input_fingerprint,
    resolve_runtime_identity,
)
from app.render.execution.types import AttemptContext
from app.services.storage import StorageService

_STALE_HEARTBEAT_SECONDS = 120.0
_CLAIMABLE_STATUSES = (JobStatus.QUEUED, JobStatus.FAILED)


class RenderAdmissionUnavailable(RuntimeError):
    """The global render admission could not be acquired (transient)."""

    retryable = True


class _LivenessHeartbeat:
    def __init__(
        self, session_factory: Callable[[], Session], job_id: uuid.UUID, claim_version: int
    ) -> None:
        self._factory = session_factory
        self._job_id = job_id
        self._claim_version = claim_version
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        from datetime import datetime, timezone

        while not self._stop.wait(5.0):
            session = self._factory()
            try:
                session.execute(
                    update(ProcessingJob)
                    .where(
                        ProcessingJob.id == self._job_id,
                        ProcessingJob.claim_version == self._claim_version,
                        ProcessingJob.status == JobStatus.RUNNING,
                    )
                    .values(heartbeat_at=datetime.now(timezone.utc))
                    .execution_options(synchronize_session=False)
                )
                session.commit()
            except Exception:
                session.rollback()
            finally:
                session.close()


class RenderExecutionExecutor:
    def __init__(
        self,
        session: Session,
        storage: StorageService,
        settings: Settings,
        *,
        admission: RenderAdmission | None = None,
        session_factory: Callable[[], Session] | None = None,
        compiler: Callable[..., Any] | None = None,
        runner: Callable[..., Any] | None = None,
        qc_checker: Callable[..., Any] | None = None,
        spec_builder: Callable[..., Any] | None = None,
        runtime_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._session = session
        self._storage = storage
        self._settings = settings
        self._config = settings.stage52_config()
        self._admission = admission or render_admission_for(session.get_bind())
        self._session_factory = session_factory or create_session_factory()
        self._job_id: uuid.UUID | None = None
        self._claim_version: int = -1
        self._row_id: uuid.UUID | None = None
        self.skipped_duplicate = False
        self._admission_held = False
        self._compiler = compiler or compile_render
        self._runner = runner or run_compiled_render
        self._qc_checker = qc_checker or check_render_artifact
        self._spec_builder = spec_builder or build_render_spec
        self._runtime_factory = runtime_factory or resolve_runtime_identity

    def set_active_job(self, job_id: uuid.UUID | None) -> None:
        self._job_id = job_id

    # Claim fencing

    def _claim_job(self) -> bool:
        if self._job_id is None:
            return True
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        stale_before = now.timestamp() - _STALE_HEARTBEAT_SECONDS
        job = self._session.get(ProcessingJob, self._job_id)
        if job is None:
            return False
        claimable = job.status in _CLAIMABLE_STATUSES
        if (
            not claimable
            and job.status is JobStatus.RUNNING
            and (job.heartbeat_at is None or job.heartbeat_at.timestamp() <= stale_before)
        ):
            claimable = True
        if not claimable:
            self.skipped_duplicate = True
            self._claim_version = job.claim_version
            return False
        result = self._session.execute(
            update(ProcessingJob)
            .where(
                ProcessingJob.id == self._job_id,
                ProcessingJob.claim_version == job.claim_version,
            )
            .values(
                status=JobStatus.RUNNING,
                started_at=now,
                heartbeat_at=now,
                claim_version=ProcessingJob.claim_version + 1,
                error_code=None,
                error_message=None,
            )
            .execution_options(synchronize_session=False)
        )
        if int(result.rowcount) != 1:
            self.skipped_duplicate = True
            return False
        self._session.commit()
        self._session.refresh(job)
        self._claim_version = job.claim_version
        return True

    def _claim_is_current(self, *, require_running: bool = True) -> bool:
        if self._job_id is None:
            return True
        job = self._fresh_job()
        if job is None:
            return False
        if job.claim_version != self._claim_version:
            return False
        if require_running and job.status is not JobStatus.RUNNING:
            return False
        return True

    def _fresh_job(self) -> ProcessingJob | None:
        if self._job_id is None:
            return None
        self._session.expire_all()
        return self._session.get(ProcessingJob, self._job_id)

    def _job_cancelled(self) -> bool:
        job = self._fresh_job()
        return job is not None and job.status is JobStatus.CANCELLED

    def _finish_job(self, status: JobStatus, *, error_code: str | None = None) -> None:
        if self._job_id is None or not self._claim_is_current(require_running=False):
            return
        from datetime import datetime, timezone

        self._session.execute(
            update(ProcessingJob)
            .where(
                ProcessingJob.id == self._job_id,
                ProcessingJob.claim_version == self._claim_version,
            )
            .values(status=status, completed_at=datetime.now(timezone.utc), error_code=error_code)
            .execution_options(synchronize_session=False)
        )
        self._session.commit()

    # Entry point

    def execute(
        self, render_execution_id: uuid.UUID | str, *, force: bool = False
    ) -> RenderExecution:
        row = self._session.get(RenderExecution, _as_uuid(render_execution_id))
        if row is None:
            raise RenderExecutionError("RENDER_EXECUTION_NOT_FOUND")
        self._row_id = row.id
        if self._job_id is not None:
            pending = self._session.get(ProcessingJob, self._job_id)
            if pending is not None and pending.status is JobStatus.CANCELLED:
                mark_cancelled(self._session, row)
                self._session.commit()
                raise StageCancelled("render cancelled while queued")
        claimed = self._claim_job()
        heartbeat: _LivenessHeartbeat | None = None
        try:
            if not claimed and self.skipped_duplicate:
                return row
            if self._job_id is not None:
                heartbeat = _LivenessHeartbeat(
                    self._session_factory, self._job_id, self._claim_version
                )
                heartbeat.start()
            if (
                not force
                and row.lifecycle is RenderExecutionLifecycle.COMPLETE
                and row.cache_eligible
            ):
                cached = cache_hit(self._session, row.clip_candidate_id, self._settings)
                if cached is not None and cached.id == row.id:
                    self._finish_job(JobStatus.SUCCEEDED)
                    return row
            row = self._run(row)
            return row
        except StageCancelled:
            self._mark_cancelled(row)
            raise
        except RenderCancelled as error:
            self._mark_cancelled(row)
            raise StageCancelled(str(error)) from error
        finally:
            if self._admission_held:
                self._admission_held = False
                self._admission.release()
            if heartbeat is not None:
                heartbeat.stop()

    def _run(self, row: RenderExecution) -> RenderExecution:
        if self._job_cancelled():
            raise StageCancelled("render cancelled before start")
        candidate = row.clip_candidate
        if candidate is None:
            from app.models import ClipCandidate

            candidate = self._session.get(ClipCandidate, row.clip_candidate_id)
        if candidate is None:
            self._block(row, "CANDIDATE_NOT_CURRENT")
            return row
        try:
            spec = self._spec_builder(
                self._session,
                candidate,
                artifact_purpose=row.artifact_purpose.value,
                delivery_profile_key=row.delivery_profile_key,
                settings=self._settings,
                storage=self._storage,
            )
        except RenderExecutionError as error:
            if error.blocked:
                self._block(row, error.reason_code)
            else:
                self._fail(row, error.reason_code)
            return row
        fingerprint = request_input_fingerprint(spec, self._config)
        if fingerprint != row.input_fingerprint:
            self._block(row, "REQUEST_INPUT_CHANGED")
            return row

        if not self._claim_is_current():
            self.skipped_duplicate = True
            return row

        if not self._admission.acquire(
            wait_seconds=self._config.admission_wait_seconds,
            cancel_check=self._job_cancelled,
        ):
            if self._job_cancelled():
                raise StageCancelled("render cancelled while waiting for admission")
            self._finish_job(JobStatus.QUEUED)
            raise RenderAdmissionUnavailable(RENDER_ADMISSION_UNAVAILABLE)
        self._admission_held = True

        attempt_directory = self._storage.render_attempt_directory(
            row.source_video_id,
            row.id,
            job_id=self._job_id or row.id,
            claim_version=max(0, self._claim_version),
        )
        source_absolute = str(
            (self._storage.storage_root / spec.source_media_relative_path).resolve()
        )
        runtime = self._runtime_factory(
            self._settings,
            source_absolute_path=source_absolute,
            attempt_directory=str(attempt_directory),
        )
        try:
            compiled = self._compiler(spec, runtime)
        except Exception as error:  # noqa: BLE001
            self._fail(row, getattr(error, "reason_code", "RENDER_PROCESS_FAILED"))
            raise

        ass_relative = spec.ass.relative_path
        ass_data = (self._storage.storage_root / ass_relative).read_bytes()

        self._set_lifecycle(row, RenderExecutionLifecycle.RENDERING)
        context = AttemptContext(
            attempt_directory=attempt_directory,
            cancel_check=self._job_cancelled,
            timeout_seconds=self._config.max_render_seconds,
            ass_bytes=ass_data,
        )
        try:
            artifacts = self._runner(compiled, context)
        except RenderTimeout:
            self._fail(row, "RENDER_TIMEOUT")
            raise
        except RenderProcessError as error:
            self._fail(row, error.reason_code)
            raise
        except RenderCancelled:
            self._mark_cancelled(row)
            raise

        self._set_lifecycle(row, RenderExecutionLifecycle.QC_RUNNING)
        qc = self._qc_checker(
            artifacts,
            artifacts.manifest,
            self._config,
            source_path=Path(source_absolute),
        )
        if qc.status == "FAIL":
            self._fail(row, qc.reason_codes[0] if qc.reason_codes else "QC_FAILED")
            return row

        # Revalidate source identity and currentness before publishing.
        restat = Path(source_absolute).stat()
        if restat.st_size != spec.source_size_bytes or restat.st_mtime_ns != spec.source_mtime_ns:
            self._fail(row, "SOURCE_MEDIA_CHANGED")
            return row
        if self._job_cancelled():
            raise StageCancelled("render cancelled before publish")
        if not self._claim_is_current():
            self.skipped_duplicate = True
            return row

        finalize_success(
            self._session,
            row,
            artifacts=artifacts,
            qc=qc,
            compiled_fingerprint_value=compiled.fingerprint,
            runtime_fp=_runtime_fp(runtime),
            storage=self._storage,
            config=self._config,
        )
        self._session.commit()
        self._finish_job(JobStatus.SUCCEEDED)
        return row

    # Lifecycle helpers

    def _set_lifecycle(self, row: RenderExecution, lifecycle: RenderExecutionLifecycle) -> None:
        if not self._claim_is_current():
            self.skipped_duplicate = True
            raise StageCancelled("render ownership lost")
        row.lifecycle = lifecycle
        self._session.commit()

    def _block(self, row: RenderExecution, reason_code: str) -> None:
        mark_blocked(self._session, row, reason_code)
        self._session.commit()
        self._finish_job(JobStatus.FAILED, error_code=reason_code)

    def _fail(self, row: RenderExecution, reason_code: str) -> None:
        mark_failed(self._session, row, reason_code)
        self._session.commit()
        self._finish_job(JobStatus.FAILED, error_code=reason_code)

    def _mark_cancelled(self, row: RenderExecution) -> None:
        if not self._claim_is_current(require_running=False):
            return
        mark_cancelled(self._session, row)
        self._session.commit()
        if self._job_id is not None and self._claim_is_current(require_running=False):
            self._session.execute(
                update(ProcessingJob)
                .where(
                    ProcessingJob.id == self._job_id,
                    ProcessingJob.claim_version == self._claim_version,
                )
                .values(status=JobStatus.CANCELLED)
                .execution_options(synchronize_session=False)
            )
            self._session.commit()

    def record_failure(self, error: Exception) -> None:
        row = self._session.get(RenderExecution, self._row_id) if self._row_id else None
        if row is not None:
            self._fail(row, getattr(error, "reason_code", "RENDER_PROCESS_FAILED"))


def _runtime_fp(runtime: Any) -> str:
    from app.render.execution.service import runtime_fingerprint

    return runtime_fingerprint(runtime)


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def build_render_execution_executor(
    session: Session,
    storage: StorageService,
    settings: Settings,
    *,
    admission: RenderAdmission | None = None,
    session_factory: Callable[[], Session] | None = None,
    compiler: Callable[..., Any] | None = None,
    runner: Callable[..., Any] | None = None,
    qc_checker: Callable[..., Any] | None = None,
    spec_builder: Callable[..., Any] | None = None,
    runtime_factory: Callable[..., Any] | None = None,
) -> RenderExecutionExecutor:
    return RenderExecutionExecutor(
        session,
        storage,
        settings,
        admission=admission,
        session_factory=session_factory,
        compiler=compiler,
        runner=runner,
        qc_checker=qc_checker,
        spec_builder=spec_builder,
        runtime_factory=runtime_factory,
    )


__all__ = [
    "RenderAdmissionUnavailable",
    "RenderExecutionExecutor",
    "build_render_execution_executor",
]
