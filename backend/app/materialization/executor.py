"""Durable Stage 6 candidate-scoped materialization executor.

Claims one ``MATERIALIZATION`` job, re-derives every current live-effective
upstream prerequisite, assembles immutable inputs through the persisted ORM
relationships, runs the bounded deterministic semantic pipeline with the durable
call ledger (resuming any checkpointed work), and persists a truthful, fenced
result. Observes the exact executing job for cooperative cancellation and
releases owned provider resources on every exit path.

Fencing, liveness heartbeat, cancellation, and stale-claim reclamation mirror the
frozen Stage 5.1 executor; the only schema used is the existing
``processing_jobs`` ``claim_version``/``heartbeat_at``/``started_at`` columns.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.enums import (
    JobStatus,
    MaterializationPriority,
)
from app.core.settings import Settings
from app.materialization.components import (
    ComponentContext,
    ComponentProviders,
    prepare_materialization,
)
from app.materialization.inputs import MaterializationInputs
from app.materialization.policy import Stage6Config
from app.materialization.providers import (
    DeterministicWritingProvider,
    MaterializationCritiqueProvider,
    MaterializationWritingProvider,
)
from app.materialization.repository import (
    MaterializationOwner,
    bind_active_job,
    claim_materialization,
    commit_prepared,
    get_or_create_envelope,
)
from app.materialization.service import (
    MaterializationError,
    assemble_inputs,
)
from app.models import ClipMaterialization, ProcessingJob
from app.pipeline.executor import StageCancelled, StageExecutionResult
from app.services.storage import StorageService

_JOB_CLAIM_STALE_SECONDS = 3_600.0
_JOB_HEARTBEAT_INTERVAL_SECONDS = 30.0
_CLAIMABLE_JOB_STATUSES = (JobStatus.QUEUED, JobStatus.FAILED)


@dataclass
class MaterializationProviders:
    """Provider bundle for one Stage 6 run (all optional except writing)."""

    writing: MaterializationWritingProvider
    critique: MaterializationCritiqueProvider | None = None
    narration: object | None = None


def build_materialization_providers(
    settings: Settings,
    *,
    writing: MaterializationWritingProvider | None = None,
    critique: MaterializationCritiqueProvider | None = None,
    narration: object | None = None,
) -> MaterializationProviders:
    """Build the production provider bundle.

    The deterministic writing provider makes zero hosted calls and can only
    reuse already-authored plan wording; no hosted writing/critique/TTS adapter
    is wired yet, so the default bundle is deterministic and network-free.
    """

    return MaterializationProviders(
        writing=writing or DeterministicWritingProvider(),
        critique=critique,
        narration=narration,
    )


class _LivenessHeartbeat:
    """Renew one job claim's durable liveness from a background thread."""

    def __init__(
        self, *, bind: object, job_id: object, claim_version: int, interval_seconds: float
    ) -> None:
        self._factory = sessionmaker(bind=bind)
        self._job_id = job_id
        self._claim_version = claim_version
        self._interval = max(0.01, float(interval_seconds))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="stage6-liveness-heartbeat", daemon=True
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


def _sanitize_error_code(error: Exception | None) -> str:
    if error is None:
        return "MATERIALIZATION_FAILED"
    if isinstance(error, MaterializationError):
        return "MATERIALIZATION_INPUT"
    return type(error).__name__[:128]


def _sanitize_error_message(error: Exception | None) -> str:
    if error is None:
        return "Stage 6 materialization failed"
    if isinstance(error, MaterializationError):
        return str(error)[:300]
    return type(error).__name__[:300]


class MaterializationExecutor:
    """Execute one candidate-scoped Stage 6 materialization run."""

    def __init__(
        self,
        *,
        session: Session,
        settings: Settings,
        config: Stage6Config,
        storage: StorageService,
        providers: MaterializationProviders,
        job_claim_stale_seconds: float = _JOB_CLAIM_STALE_SECONDS,
        heartbeat_interval_seconds: float = _JOB_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._session = session
        self._settings = settings
        self._config = config
        self._storage = storage
        self._providers = providers
        self._job_claim_stale_seconds = max(0.0, float(job_claim_stale_seconds))
        self._heartbeat_interval_seconds = max(0.01, float(heartbeat_interval_seconds))
        self._active_job_id: object | None = None
        self._job_owner = False
        self._claim_version: int | None = None
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

    def _should_stop(self) -> bool:
        return self._job_cancelled() or not self._claim_is_current()

    # -- execution -------------------------------------------------------------

    def _claim_envelope(
        self,
        inputs: MaterializationInputs,
        *,
        priority: str,
        profile_key: str | None,
        supporting_requested: str,
    ) -> MaterializationOwner | None:
        """Create/bind the durable envelope and claim it for the executing job."""

        if self._active_job_id is None:
            return None
        request_identity = {
            "priority": priority,
            "profile_key": inputs.profile.profile_key,
            "supporting_requested": supporting_requested,
        }
        envelope = get_or_create_envelope(self._session, inputs, request_identity)
        job = self._session.get(ProcessingJob, self._active_job_id)
        if job is None:
            return None
        if job.materialization_id != envelope.id:
            if envelope.active_job_id is None:
                bind_active_job(self._session, envelope.id, job.id)
            job.materialization_id = envelope.id
            self._session.commit()
        owner = claim_materialization(
            self._session, envelope.id, uuid.UUID(str(self._active_job_id))
        )
        return owner

    def _component_context(
        self, inputs: MaterializationInputs, owner: MaterializationOwner
    ) -> ComponentContext:
        return ComponentContext(
            session=self._session,
            owner=owner,
            inputs=inputs,
            config=self._config,
            storage=self._storage,
            providers=ComponentProviders(
                writing=self._providers.writing,
                critique=self._providers.critique,
                narration=self._providers.narration,  # type: ignore[arg-type]
            ),
            allow_hosted=self._providers.critique is not None,
            cancelled=self._should_stop,
        )

    def execute(
        self,
        candidate_id: str,
        *,
        priority: str = MaterializationPriority.CANDIDATE.value,
        profile_key: str | None = None,
        supporting_requested: str = "OFF",
        force: bool = False,
    ) -> StageExecutionResult:
        self.skipped_duplicate = False
        self.claim_lost = False
        if self._job_cancelled():
            self._finish_job(JobStatus.CANCELLED)
            raise StageCancelled("Stage 6 materialization cancelled before start")
        if not self._claim_job():
            self.skipped_duplicate = True
            return StageExecutionResult("", None)
        heartbeat = self._start_heartbeat()
        try:
            try:
                inputs = assemble_inputs(
                    self._session,
                    candidate_id,
                    priority=priority,
                    profile_key=profile_key,
                    supporting_requested=supporting_requested,
                )
            except MaterializationError:
                self._session.rollback()
                raise
            if self._should_stop():
                self._finish_job(JobStatus.CANCELLED)
                raise StageCancelled("Stage 6 materialization cancelled before execution")
            owner = self._claim_envelope(
                inputs,
                priority=priority,
                profile_key=profile_key,
                supporting_requested=supporting_requested,
            )
            if owner is None:
                self.skipped_duplicate = True
                return StageExecutionResult("", None)
            context = self._component_context(inputs, owner)
            try:
                prepared = prepare_materialization(context)
            except Exception as error:
                self._finish_job(JobStatus.FAILED, error)
                raise
            if self._job_cancelled():
                self._finish_job(JobStatus.CANCELLED)
                raise StageCancelled("Stage 6 materialization cancelled before persistence")
            if not self._claim_guard(require_running=True):
                self._session.rollback()
                return StageExecutionResult("", None)
            envelope = self._session.get(ClipMaterialization, owner.materialization_id)
            if envelope is None:
                self._session.rollback()
                return StageExecutionResult("", None)
            if not commit_prepared(self._session, owner, prepared):
                self._session.rollback()
                return StageExecutionResult("", None)
            self._session.commit()
            self._finish_job(JobStatus.SUCCEEDED)
            return StageExecutionResult(prepared.output_fingerprint, envelope)
        except StageCancelled:
            self._finish_job(JobStatus.CANCELLED)
            raise
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
        if self._active_job_id is None:
            return
        if self._claim_version is not None:
            stored = self._session.scalar(
                select(ProcessingJob.claim_version).where(ProcessingJob.id == self._active_job_id)
            )
            if stored == self._claim_version:
                if self._job_status() is JobStatus.RUNNING:
                    self._job_owner = True
                    self._finish_job(JobStatus.FAILED, error)
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
            self._session.commit()
        else:
            self._session.rollback()

    def _release_resources(self) -> None:
        for target in (
            self._providers.writing,
            self._providers.critique,
            self._providers.narration,
        ):
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


def build_materialization_executor(
    session: Session,
    storage: StorageService,
    settings: Settings,
    *,
    writing: MaterializationWritingProvider | None = None,
    critique: MaterializationCritiqueProvider | None = None,
    narration: object | None = None,
) -> MaterializationExecutor:
    """Build the production Stage 6 executor lazily and network-free."""

    return MaterializationExecutor(
        session=session,
        settings=settings,
        config=settings.stage6_config(),
        storage=storage,
        providers=build_materialization_providers(
            settings, writing=writing, critique=critique, narration=narration
        ),
    )


__all__ = [
    "MaterializationExecutor",
    "MaterializationProviders",
    "build_materialization_executor",
    "build_materialization_providers",
]
