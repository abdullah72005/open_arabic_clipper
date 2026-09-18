"""Durable Stage 5.1 candidate-scoped visual-composition executor.

Claims one plan row, resolves the current executable Stage 5.0 contract, runs the
bounded deterministic CPU-local planner, and persists a truthful result.
Observes the exact executing job for cooperative cancellation, fences every
persistence path with the durable ``claim_version`` token, and releases owned
detector/FFmpeg resources on every exit path (including cache hits).

Fencing, liveness heartbeat, cancellation, and stale-claim reclamation mirror the
frozen Stage 4.1 executor exactly; the only schema used is the existing
``processing_jobs`` ``claim_version``/``heartbeat_at``/``started_at`` columns.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from time import monotonic

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.composition.detector import FaceDetector
from app.composition.geometry import DisplayProbe, FFprobeDisplayProbe
from app.composition.policy import (
    Stage51Config,
    VisualCompositionExecutionStatus,
    VisualCompositionStatus,
)
from app.composition.service import (
    CompositionCancelled,
    CompositionError,
    CompositionInputError,
    composition_input_fingerprint,
    execute_visual_composition,
    resolve_planner_inputs,
)
from app.composition.types import PlannerInputs
from app.core.enums import JobStatus
from app.core.settings import Settings
from app.models import ProcessingJob
from app.models.visual_composition_plan import VisualCompositionPlan as VisualCompositionPlanRow
from app.pipeline.executor import StageCancelled, StageExecutionResult
from app.services.storage import StorageService

_JOB_CLAIM_STALE_SECONDS = 3_600.0
_JOB_HEARTBEAT_INTERVAL_SECONDS = 30.0
_CLAIMABLE_JOB_STATUSES = (JobStatus.QUEUED, JobStatus.FAILED)


class _LivenessHeartbeat:
    """Renew one job claim's durable liveness from a background thread."""

    def __init__(
        self,
        *,
        bind: object,
        job_id: object,
        claim_version: int,
        interval_seconds: float,
    ) -> None:
        self._factory = sessionmaker(bind=bind)
        self._job_id = job_id
        self._claim_version = claim_version
        self._interval = max(0.01, float(interval_seconds))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="stage51-liveness-heartbeat", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._renew()

    def _renew(self) -> None:
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
            try:
                session.rollback()
            except Exception:
                pass
        finally:
            session.close()


def _sanitize_error_message(error: Exception | None) -> str:
    if error is None:
        return "Stage 5.1 visual composition failed"
    if isinstance(error, CompositionInputError):
        return str(error)[:300]
    return type(error).__name__[:300]


def _sanitize_error_code(error: Exception | None) -> str:
    if error is None:
        return "VISUAL_COMPOSITION_FAILED"
    if isinstance(error, CompositionInputError):
        return "COMPOSITION_INPUT"
    return type(error).__name__[:128]


class VisualCompositionExecutor:
    """Execute one candidate-scoped Stage 5.1 visual-composition run."""

    def __init__(
        self,
        *,
        session: Session,
        settings: Settings,
        config: Stage51Config,
        storage: StorageService,
        display_probe: object | None = None,
        frame_sampler: object | None = None,
        scene_cut_detector: object | None = None,
        detector: FaceDetector | None = None,
        job_claim_stale_seconds: float = _JOB_CLAIM_STALE_SECONDS,
        heartbeat_interval_seconds: float = _JOB_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._session = session
        self._settings = settings
        self._config = config
        self._storage = storage
        self._display_probe = display_probe
        self._frame_sampler = frame_sampler
        self._scene_cut_detector = scene_cut_detector
        self._detector = detector
        self._job_claim_stale_seconds = max(0.0, float(job_claim_stale_seconds))
        self._heartbeat_interval_seconds = max(0.01, float(heartbeat_interval_seconds))
        self._active_job_id: object | None = None
        self._job_owner = False
        self._claim_version: int | None = None
        self._plan_id: object | None = None
        self.claim_lost = False
        self.skipped_duplicate = False

    # -- job claim / ownership -------------------------------------------------

    def set_active_job(self, job_id: object | None) -> None:
        self._active_job_id = job_id

    def _job_status(self) -> JobStatus | None:
        if self._active_job_id is None:
            return None
        return self._session.scalar(
            select(ProcessingJob.status).where(ProcessingJob.id == self._active_job_id)
        )

    def _job_cancelled(self) -> bool:
        return self._job_status() is JobStatus.CANCELLED

    def _claim_is_current(self) -> bool:
        if self.claim_lost:
            return False
        if self._active_job_id is None:
            return True
        if self._job_owner and self._claim_version is not None:
            stored = self._session.scalar(
                select(ProcessingJob.claim_version).where(ProcessingJob.id == self._active_job_id)
            )
            if stored != self._claim_version:
                self._job_owner = False
                self.claim_lost = True
                return False
            return True
        return self._job_status() is not JobStatus.RUNNING

    def _fence_claim(self, *, require_running: bool = True) -> bool:
        if self._active_job_id is None:
            return True
        if not self._job_owner or self._claim_version is None:
            self.claim_lost = True
            return False
        conditions = [
            ProcessingJob.id == self._active_job_id,
            ProcessingJob.claim_version == self._claim_version,
        ]
        if require_running:
            conditions.append(ProcessingJob.status == JobStatus.RUNNING)
        result = self._session.execute(
            update(ProcessingJob)
            .where(*conditions)
            .values(claim_version=self._claim_version)
            .execution_options(synchronize_session=False)
        )
        if int(result.rowcount) == 1:
            return True
        stored = self._session.scalar(
            select(ProcessingJob.claim_version).where(ProcessingJob.id == self._active_job_id)
        )
        if stored != self._claim_version:
            self._job_owner = False
            self.claim_lost = True
        return False

    def _claim_guard(self, *, require_running: bool = True) -> bool:
        if self.claim_lost:
            return False
        if self._active_job_id is None:
            return True
        if self._job_owner and self._claim_version is not None:
            return self._fence_claim(require_running=require_running)
        return self._job_status() is not JobStatus.RUNNING

    def _claim_job(self) -> bool:
        """Atomically claim the executing job and advance its durable token."""

        if self._active_job_id is None:
            self._job_owner = True
            self._claim_version = None
            return True
        now = datetime.now(timezone.utc)
        stale_before = now - timedelta(seconds=self._job_claim_stale_seconds)
        result = self._session.execute(
            update(ProcessingJob)
            .where(
                ProcessingJob.id == self._active_job_id,
                or_(
                    ProcessingJob.status.in_(_CLAIMABLE_JOB_STATUSES),
                    and_(
                        ProcessingJob.status == JobStatus.RUNNING,
                        or_(
                            and_(
                                ProcessingJob.heartbeat_at.is_not(None),
                                ProcessingJob.heartbeat_at < stale_before,
                            ),
                            and_(
                                ProcessingJob.heartbeat_at.is_(None),
                                ProcessingJob.started_at.is_not(None),
                                ProcessingJob.started_at < stale_before,
                            ),
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
                completed_at=None,
            )
            .execution_options(synchronize_session=False)
        )
        claimed = int(result.rowcount) == 1
        self._session.commit()
        self._job_owner = claimed
        if claimed:
            self._claim_version = self._session.scalar(
                select(ProcessingJob.claim_version).where(ProcessingJob.id == self._active_job_id)
            )
        return claimed

    def _finish_job(self, status: JobStatus, error: Exception | None = None) -> None:
        if self._active_job_id is None or not self._job_owner:
            return
        values: dict[str, object] = {
            "status": status,
            "completed_at": datetime.now(timezone.utc),
        }
        if status is JobStatus.FAILED:
            values["error_code"] = _sanitize_error_code(error)
            values["error_message"] = _sanitize_error_message(error)
        result = self._session.execute(
            update(ProcessingJob)
            .where(
                ProcessingJob.id == self._active_job_id,
                ProcessingJob.claim_version == self._claim_version,
                ProcessingJob.status == JobStatus.RUNNING,
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if int(result.rowcount) == 1:
            self._session.commit()
        else:
            self._session.rollback()
            self._job_owner = False
            self.claim_lost = True

    def _start_heartbeat(self) -> _LivenessHeartbeat | None:
        if self._active_job_id is None or not self._job_owner or self._claim_version is None:
            return None
        heartbeat = _LivenessHeartbeat(
            bind=self._session.get_bind(),
            job_id=self._active_job_id,
            claim_version=self._claim_version,
            interval_seconds=self._heartbeat_interval_seconds,
        )
        heartbeat.start()
        return heartbeat

    # -- row lifecycle ---------------------------------------------------------

    def _reconcile_cancelled(self, row: VisualCompositionPlanRow) -> None:
        if not self._claim_guard(require_running=False):
            self._session.rollback()
            return
        row.execution_status = VisualCompositionExecutionStatus.CANCELLED
        row.active_job_id = None
        self._session.commit()

    def _mark_cancelled(self, row: VisualCompositionPlanRow) -> None:
        if not self._claim_guard(require_running=True):
            if self.claim_lost or self._job_status() is not JobStatus.CANCELLED:
                self._session.rollback()
                return
            self._reconcile_cancelled(row)
            return
        row.execution_status = VisualCompositionExecutionStatus.CANCELLED
        row.active_job_id = None
        self._session.commit()

    def _mark_failed(self, row: VisualCompositionPlanRow, error: Exception | None = None) -> None:
        if not self._claim_guard(require_running=True):
            if not self.claim_lost and self._job_status() is JobStatus.CANCELLED:
                self._reconcile_cancelled(row)
            else:
                self._session.rollback()
            return
        if self._job_cancelled():
            row.execution_status = VisualCompositionExecutionStatus.CANCELLED
        else:
            row.execution_status = VisualCompositionExecutionStatus.FAILED
        row.active_job_id = None
        self._session.commit()

    def _claim_row(self, row: VisualCompositionPlanRow) -> bool:
        if not self._claim_guard():
            self._session.rollback()
            return False
        row.active_job_id = self._active_job_id
        row.execution_status = VisualCompositionExecutionStatus.ANALYZING
        self._session.commit()
        return True

    def _should_stop(self) -> bool:
        return self._job_cancelled() or not self._claim_is_current()

    # -- execution -------------------------------------------------------------

    def input_fingerprint(self, row: VisualCompositionPlanRow) -> str:
        inputs = resolve_planner_inputs(
            self._session,
            row.clip_candidate_id,
            display_probe=self._probe(),
            config=self._config,
        )
        if inputs is None:
            return ""
        return composition_input_fingerprint(inputs, self._config)

    def is_cache_hit(self, row: VisualCompositionPlanRow, *, force: bool = False) -> bool:
        if force or not row.cache_eligible or not row.plan_ready or not row.input_fingerprint:
            return False
        try:
            current = self.input_fingerprint(row)
        except Exception:
            return False
        return bool(current) and current == row.input_fingerprint

    def execute(self, plan_id: object, *, force: bool = False) -> StageExecutionResult:
        self.skipped_duplicate = False
        self.claim_lost = False
        self._plan_id = plan_id
        row = self._session.get(VisualCompositionPlanRow, plan_id)
        if row is None:
            raise CompositionInputError("visual-composition plan is missing")
        if self._job_cancelled():
            self._mark_cancelled(row)
            self._finish_job(JobStatus.CANCELLED)
            raise CompositionCancelled("Stage 5.1 visual composition cancelled before start")
        if not self._claim_job():
            self.skipped_duplicate = True
            return StageExecutionResult(row.output_fingerprint, row)
        heartbeat = self._start_heartbeat()
        try:
            inputs: PlannerInputs | None
            try:
                inputs = resolve_planner_inputs(
                    self._session,
                    row.clip_candidate_id,
                    display_probe=self._probe(),
                    config=self._config,
                )
            except Exception as error:
                self._mark_failed(row, error)
                self._finish_job(JobStatus.FAILED, error)
                raise
            if inputs is None:
                prerequisite_error = CompositionInputError(
                    "candidate has no current live-effective executable Stage 5.0 contract"
                )
                self._mark_failed(row, prerequisite_error)
                self._finish_job(JobStatus.FAILED, prerequisite_error)
                raise prerequisite_error
            if self.is_cache_hit(row, force=force):
                self._finish_job(JobStatus.SUCCEEDED)
                return StageExecutionResult(row.output_fingerprint, row)
            if not self._claim_is_current():
                return StageExecutionResult(row.output_fingerprint, row)
            if not self._claim_row(row):
                return StageExecutionResult(row.output_fingerprint, row)
            started = monotonic()
            try:
                result_row = execute_visual_composition(
                    self._session,
                    row.id,
                    storage=self._storage,
                    settings=self._settings,
                    force=force,
                    display_probe=self._probe(),
                    frame_sampler=self._frame_sampler,
                    scene_cut_detector=self._scene_cut_detector,
                    detector=self._detector,
                    cancel_check=self._should_stop,
                    persist_guard=lambda: self._claim_guard(require_running=True),
                )
            except CompositionCancelled as error:
                self._mark_cancelled(row)
                self._finish_job(JobStatus.CANCELLED)
                raise error
            except StageCancelled as error:
                self._mark_cancelled(row)
                self._finish_job(JobStatus.CANCELLED)
                raise error
            except Exception as error:
                self._mark_failed(row, error)
                self._finish_job(JobStatus.FAILED, error)
                raise
            if result_row is None:
                # Claim superseded or cancelled before persistence.
                if self._job_cancelled():
                    self._mark_cancelled(row)
                    self._finish_job(JobStatus.CANCELLED)
                    raise CompositionCancelled(
                        "Stage 5.1 visual composition cancelled before persistence"
                    )
                self._session.rollback()
                return StageExecutionResult(row.output_fingerprint, row)
            elapsed = monotonic() - started
            result_row.metrics = {
                **dict(result_row.metrics or {}),
                "processing_wall_seconds": round(elapsed, 6),
            }
            if result_row.status is VisualCompositionStatus.FAILED:
                plan_error = CompositionError("Stage 5.1 planner produced a failed plan")
                self._session.commit()
                self._finish_job(JobStatus.FAILED, plan_error)
                return StageExecutionResult(result_row.output_fingerprint, result_row)
            self._session.commit()
            self._finish_job(JobStatus.SUCCEEDED)
            return StageExecutionResult(result_row.output_fingerprint, result_row)
        finally:
            if heartbeat is not None:
                heartbeat.stop()
            self._release_resources()

    def record_failure(self, error: Exception) -> None:
        """Best-effort fenced sanitized FAILED finalization for the task wrapper."""

        try:
            self._session.rollback()
        except Exception:
            pass
        row = (
            self._session.get(VisualCompositionPlanRow, self._plan_id)
            if self._plan_id is not None
            else None
        )
        if self._active_job_id is None:
            if row is not None:
                self._mark_failed(row, error)
            return
        if self._claim_version is not None:
            stored = self._session.scalar(
                select(ProcessingJob.claim_version).where(ProcessingJob.id == self._active_job_id)
            )
            if stored == self._claim_version:
                status = self._job_status()
                if status is JobStatus.RUNNING:
                    self._job_owner = True
                    if row is not None:
                        self._mark_failed(row, error)
                    self._finish_job(JobStatus.FAILED, error)
                elif status is JobStatus.CANCELLED:
                    if row is not None:
                        self._reconcile_cancelled(row)
                return
            return
        result = self._session.execute(
            update(ProcessingJob)
            .where(
                ProcessingJob.id == self._active_job_id,
                ProcessingJob.status == JobStatus.QUEUED,
            )
            .values(
                status=JobStatus.FAILED,
                completed_at=datetime.now(timezone.utc),
                error_code=_sanitize_error_code(error),
                error_message=_sanitize_error_message(error),
            )
            .execution_options(synchronize_session=False)
        )
        if int(result.rowcount) == 1:
            if row is not None:
                row.execution_status = VisualCompositionExecutionStatus.FAILED
                row.active_job_id = None
            self._session.commit()
        else:
            self._session.rollback()

    # -- resources -------------------------------------------------------------

    def _probe(self) -> DisplayProbe:
        if self._display_probe is None:
            self._display_probe = FFprobeDisplayProbe(binary=self._settings.ffprobe_binary)
        return self._display_probe  # type: ignore[return-value]

    def _release_resources(self) -> None:
        for target in (self._detector, self._frame_sampler, self._scene_cut_detector):
            if target is None:
                continue
            for name in ("release", "close"):
                closer = getattr(target, name, None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass
                    break
        try:
            self._storage.cleanup_temporary_files(older_than_seconds=0, limit=200)
        except Exception:
            pass


def build_visual_composition_executor(
    session: Session,
    storage: StorageService,
    settings: Settings,
    *,
    display_probe: object | None = None,
    frame_sampler: object | None = None,
    scene_cut_detector: object | None = None,
    detector: FaceDetector | None = None,
) -> VisualCompositionExecutor:
    """Build the production Stage 5.1 executor lazily and network-free."""

    return VisualCompositionExecutor(
        session=session,
        settings=settings,
        config=settings.stage51_config(),
        storage=storage,
        display_probe=display_probe,
        frame_sampler=frame_sampler,
        scene_cut_detector=scene_cut_detector,
        detector=detector,
    )


__all__ = [
    "VisualCompositionExecutor",
    "build_visual_composition_executor",
]
