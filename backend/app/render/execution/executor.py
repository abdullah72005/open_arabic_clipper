"""Stage 5.2 render-execution executor with durable claim fencing.

Duplicate/redelivered Celery invocations are fenced by an atomic
``QUEUED/FAILED -> RUNNING`` claim that advances ``claim_version`` in the same
statement. Every worker-owned mutation is fenced on the execution row's
authoritative ``active_job_id`` plus the executing job's ``claim_version`` and
permitted ``RUNNING`` status, so a superseded worker can never finalize, cancel,
fail, or release a newer run. Cancellation and lost ownership are read through a
fresh scalar query and the narrow global admission lock is verified against its
dedicated connection; both are polled at a fixed cadence and the active FFmpeg
child is reaped on every exit path.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from sqlalchemy import and_, or_, update
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
from app.render.execution.qc import QCCancelled, check_render_artifact
from app.render.execution.runner import (
    RenderCancelled,
    RenderProcessError,
    RenderTimeout,
    run_compiled_render,
)
from app.render.execution.service import (
    RenderExecutionError,
    begin_attempt,
    build_render_spec,
    cache_hit,
    cancel_execution,
    finalize_success,
    mark_blocked,
    mark_cancelled,
    mark_failed,
    request_input_fingerprint,
    resolve_runtime_identity,
    set_lifecycle_fenced,
)
from app.render.execution.types import AttemptContext, TechnicalQCResult
from app.services.storage import StorageService

_STALE_HEARTBEAT_SECONDS = 120.0
_CLAIMABLE_STATUSES = (JobStatus.QUEUED, JobStatus.FAILED)


class RenderAdmissionUnavailable(RuntimeError):
    """The global render admission could not be acquired (transient)."""

    retryable = True


class _LivenessHeartbeat:
    def __init__(
        self,
        session_factory: Callable[[], Session],
        job_id: uuid.UUID,
        claim_version: int,
        *,
        on_lost: Callable[[], None] | None = None,
    ) -> None:
        self._factory = session_factory
        self._job_id = job_id
        self._claim_version = claim_version
        self._on_lost = on_lost
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        from datetime import datetime, timezone

        failures = 0
        while not self._stop.wait(5.0):
            session = self._factory()
            try:
                result = session.execute(
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
                if int(result.rowcount) != 1:
                    if self._on_lost is not None:
                        self._on_lost()
                    return
                failures = 0
            except Exception:
                session.rollback()
                failures += 1
                if failures >= 3 and self._on_lost is not None:
                    self._on_lost()
                    return
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
        self._claimed = False
        self._ownership_lost = False
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
        stale_before = datetime.fromtimestamp(
            now.timestamp() - _STALE_HEARTBEAT_SECONDS, tz=timezone.utc
        )
        job = self._session.get(ProcessingJob, self._job_id)
        if job is None:
            return False
        observed = int(job.claim_version)
        claimable = job.status in _CLAIMABLE_STATUSES or (
            job.status is JobStatus.RUNNING
            and (job.heartbeat_at is None or job.heartbeat_at <= stale_before)
        )
        if not claimable:
            self.skipped_duplicate = True
            self._claim_version = observed
            return False
        result = self._session.execute(
            update(ProcessingJob)
            .where(
                ProcessingJob.id == self._job_id,
                ProcessingJob.claim_version == observed,
                or_(
                    ProcessingJob.status.in_(_CLAIMABLE_STATUSES),
                    and_(
                        ProcessingJob.status == JobStatus.RUNNING,
                        or_(
                            ProcessingJob.heartbeat_at.is_(None),
                            ProcessingJob.heartbeat_at <= stale_before,
                        ),
                    ),
                ),
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
        self._claim_version = observed + 1
        self._claimed = True
        return True

    def _claim_is_current(self, *, require_running: bool = True) -> bool:
        if self._job_id is None:
            return True
        job = self._fresh_job()
        if job is None:
            return False
        if int(job.claim_version) != self._claim_version:
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

    def _ownership_ok(self) -> bool:
        if not self._claim_is_current(require_running=False):
            self._ownership_lost = True
            return False
        if self._admission_held and not self._admission.held():
            self._ownership_lost = True
            return False
        return True

    def _should_stop(self) -> bool:
        """Polled by the runner: True on cancellation or lost ownership."""

        if self._job_cancelled():
            return True
        return not self._ownership_ok()

    def _finish_job(self, status: JobStatus, *, error_code: str | None = None) -> None:
        if self._job_id is None or not self._claim_is_current(require_running=False):
            return
        from datetime import datetime, timezone

        self._session.execute(
            update(ProcessingJob)
            .where(
                ProcessingJob.id == self._job_id,
                ProcessingJob.claim_version == self._claim_version,
                ProcessingJob.status == JobStatus.RUNNING,
            )
            .values(status=status, completed_at=datetime.now(timezone.utc), error_code=error_code)
            .execution_options(synchronize_session=False)
        )
        self._session.commit()

    # Entry point

    def execute(
        self, render_execution_id: uuid.UUID | str, *, force: bool = False
    ) -> RenderExecution:
        row = cast(
            "RenderExecution | None",
            self._session.get(RenderExecution, _as_uuid(render_execution_id)),
        )
        if row is None:
            raise RenderExecutionError("RENDER_EXECUTION_NOT_FOUND")
        self._row_id = row.id
        if self._job_id is not None:
            pending = self._session.get(ProcessingJob, self._job_id)
            if pending is not None and pending.status is JobStatus.CANCELLED:
                cancel_execution(self._session, row.id, job_id=self._job_id)
                self._session.commit()
                raise StageCancelled("render cancelled while queued")
        claimed = self._claim_job()
        heartbeat: _LivenessHeartbeat | None = None
        try:
            if not claimed and self.skipped_duplicate:
                return row
            if self._job_id is not None:
                heartbeat = _LivenessHeartbeat(
                    self._session_factory,
                    self._job_id,
                    self._claim_version,
                    on_lost=self._mark_ownership_lost,
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
            return self._run(row)
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

    def _mark_ownership_lost(self) -> None:
        self._ownership_lost = True

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

        ass_data = self._verified_ass_bytes(spec)

        if not self._begin_attempt(row):
            self.skipped_duplicate = True
            return row
        context = AttemptContext(
            attempt_directory=attempt_directory,
            cancel_check=self._should_stop,
            timeout_seconds=self._config.max_render_seconds,
            poll_seconds=self._config.cancel_poll_seconds,
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

        if not self._set_qc_running(row):
            self.skipped_duplicate = True
            return row
        try:
            qc = self._qc_checker(
                artifacts,
                artifacts.manifest,
                self._config,
                source_path=Path(source_absolute),
                cancel_check=self._should_stop,
            )
        except QCCancelled:
            self._mark_cancelled(row)
            raise StageCancelled("render cancelled during QC") from None
        if qc.status == "FAIL":
            self._fail(row, qc.reason_codes[0] if qc.reason_codes else "QC_FAILED", qc=qc)
            return row

        if not self._revalidate_before_publish(row, candidate, source_absolute, spec):
            return row

        if self._job_cancelled():
            raise StageCancelled("render cancelled before publish")
        if not self._ownership_ok():
            if self._job_cancelled():
                raise StageCancelled("render cancelled before publish")
            self.skipped_duplicate = True
            return row

        published = self._finalize_success(row, artifacts, qc, compiled.fingerprint, runtime)
        if not published:
            self.skipped_duplicate = True
            return row
        self._finish_job(JobStatus.SUCCEEDED)
        return row

    def _begin_attempt(self, row: RenderExecution) -> bool:
        if self._job_id is None:
            row.lifecycle = RenderExecutionLifecycle.RENDERING
            row.cache_eligible = False
            row.qc_status = None
            row.artifact_reference = {}
            row.execution_manifest = {}
            row.qc_result = {}
            row.output_fingerprint = ""
            row.reason_codes = []
            row.error_code = None
            row.error_message = None
            self._session.flush()
            self._session.commit()
            return True
        claimed = begin_attempt(
            self._session, row.id, job_id=self._job_id, claim_version=self._claim_version
        )
        self._session.commit()
        if not claimed:
            self.skipped_duplicate = True
            return False
        self._session.expire_all()
        return True

    def _set_qc_running(self, row: RenderExecution) -> bool:
        if self._job_id is None:
            row.lifecycle = RenderExecutionLifecycle.QC_RUNNING
            self._session.flush()
            self._session.commit()
            return True
        claimed = set_lifecycle_fenced(
            self._session,
            row.id,
            RenderExecutionLifecycle.QC_RUNNING,
            job_id=self._job_id,
            claim_version=self._claim_version,
        )
        self._session.commit()
        if not claimed:
            self.skipped_duplicate = True
            return False
        self._session.expire_all()
        return True

    def _verified_ass_bytes(self, spec: Any) -> bytes:
        relative = spec.ass.relative_path
        path = (self._storage.storage_root / relative).resolve()
        try:
            path.relative_to(self._storage.storage_root)
        except ValueError as error:
            raise RenderExecutionError("ASS_MISSING") from error
        data = path.read_bytes()
        expected = (spec.ass.sha256 or "").lower()
        import hashlib

        actual = hashlib.sha256(data).hexdigest()
        if not expected or actual != expected:
            raise RenderExecutionError("ASS_HASH_MISMATCH")
        return bytes(data)

    def _revalidate_before_publish(
        self, row: RenderExecution, candidate: Any, source_absolute: str, spec: Any
    ) -> bool:
        try:
            fresh = self._spec_builder(
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
            return False
        if request_input_fingerprint(fresh, self._config) != row.input_fingerprint:
            self._block(row, "REQUEST_INPUT_CHANGED")
            return False
        stat = Path(source_absolute).stat()
        if stat.st_size != spec.source_size_bytes or stat.st_mtime_ns != spec.source_mtime_ns:
            self._fail(row, "SOURCE_MEDIA_CHANGED")
            return False
        return True

    # Lifecycle helpers

    def _block(self, row: RenderExecution, reason_code: str) -> None:
        if self._job_id is None:
            mark_blocked(self._session, row, reason_code)
            self._commit_or_rollback()
            return
        if not self._claimed:
            return
        try:
            marked = mark_blocked(
                self._session,
                row,
                reason_code,
                job_id=self._job_id,
                claim_version=self._claim_version,
            )
            self._session.commit()
            if not marked:
                self.skipped_duplicate = True
                return
        except Exception:
            self._safe_rollback()
            return
        self._finish_job(JobStatus.FAILED, error_code=reason_code)

    def _fail(
        self,
        row: RenderExecution,
        reason_code: str,
        *,
        qc: TechnicalQCResult | None = None,
    ) -> None:
        if self._job_id is None:
            mark_failed(self._session, row, reason_code, qc=qc)
            self._commit_or_rollback()
            return
        if not self._claimed:
            return
        try:
            marked = mark_failed(
                self._session,
                row,
                reason_code,
                qc=qc,
                job_id=self._job_id,
                claim_version=self._claim_version,
            )
            self._session.commit()
            if not marked:
                self.skipped_duplicate = True
                return
        except Exception:
            self._safe_rollback()
            return
        self._finish_job(JobStatus.FAILED, error_code=reason_code)

    def _finalize_success(
        self,
        row: RenderExecution,
        artifacts: Any,
        qc: TechnicalQCResult,
        compiled_fingerprint_value: str,
        runtime: Any,
    ) -> bool:
        from app.render.execution.service import runtime_fingerprint

        try:
            published = finalize_success(
                self._session,
                row,
                artifacts=artifacts,
                qc=qc,
                compiled_fingerprint_value=compiled_fingerprint_value,
                runtime_fp=runtime_fingerprint(runtime),
                storage=self._storage,
                config=self._config,
                job_id=self._job_id,
                claim_version=self._claim_version,
            )
            self._session.commit()
        except Exception:
            self._safe_rollback()
            return False
        return bool(published)

    def _mark_cancelled(self, row: RenderExecution) -> None:
        if self._job_id is None:
            mark_cancelled(self._session, row)
            self._commit_or_rollback()
            return
        if not self._claim_is_current(require_running=False):
            return
        try:
            marked = mark_cancelled(
                self._session, row, job_id=self._job_id, claim_version=self._claim_version
            )
            if marked:
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
        except Exception:
            self._safe_rollback()

    def _commit_or_rollback(self) -> None:
        try:
            self._session.commit()
        except Exception:
            self._safe_rollback()

    def _safe_rollback(self) -> None:
        try:
            self._session.rollback()
        except Exception:  # pragma: no cover - defensive
            pass

    def record_failure(self, error: Exception) -> None:
        self._safe_rollback()
        if self._row_id is None or not self._claimed:
            return
        row = self._session.get(RenderExecution, self._row_id)
        if row is None:
            return
        self._fail(row, getattr(error, "reason_code", "RENDER_PROCESS_FAILED"))


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
