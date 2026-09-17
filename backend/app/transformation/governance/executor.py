"""Durable Stage 4.2 candidate-scoped governance executor.

Claims one governance set, resolves the current Stage 4.1 plan set and selected
Stage 3.5 refinement, runs the bounded deterministic+provider governance service,
and persists independent per-plan results. Observes the exact executing job for
cooperative cancellation, routes hosted work through the shared HIGH admission
gate, binds local inference to the shared heavy-model lease, and releases owned
providers on every exit path.
"""

from __future__ import annotations

import threading
import uuid as _uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from time import monotonic

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.enums import (
    AdmissionPriority,
    GovernanceExecutionStatus,
    JobStatus,
    SemanticProviderMode,
)
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    ProcessingJob,
    TransformationGovernanceResult,
    TransformationGovernanceSet,
    TransformationPlan,
    TransformationPlanSet,
)
from app.pipeline.executor import StageCancelled, StageExecutionResult
from app.runtime.heavy_model_lease import (
    HeavyModelLeaseBusy,
    HeavyModelLeaseFactory,
    NoopHeavyModelLeaseFactory,
)
from app.transformation.governance.fingerprints import (
    build_governance_input_payload,
    governance_input_fingerprint,
)
from app.transformation.governance.inputs import (
    GovernanceInputError,
    build_governance_inputs,
)
from app.transformation.governance.policy import (
    DEFAULT_CONFIG,
    GOVERNOR_POLICY_VERSION,
    GOVERNOR_SCHEMA_VERSION,
    GOVERNOR_VALIDATION_VERSION,
    PLATFORM_POLICY_CHECKED_AT,
    PLATFORM_POLICY_PROFILE_VERSION,
    Stage42Config,
)
from app.transformation.governance.providers import (
    DeterministicGovernanceProvider,
    GovernanceProvider,
    GovernanceProviderError,
    GovernanceRequest,
)
from app.transformation.governance.service import GovernanceService
from app.transformation.governance.types import (
    GovernanceAttempt,
    GovernanceInputs,
    GovernanceOutcome,
    PlanGovernance,
)

_READY_STATUSES = {
    GovernanceExecutionStatus.COMPLETE,
    GovernanceExecutionStatus.PROVIDER_DEGRADED,
}
_ACTIVE_JOB_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}
_JOB_CLAIM_STALE_SECONDS = 3_600.0
# Public alias: the queue uses the same heartbeat-abandonment rule to recover a
# genuinely abandoned RUNNING job.
JOB_CLAIM_STALE_SECONDS = _JOB_CLAIM_STALE_SECONDS
_JOB_HEARTBEAT_INTERVAL_SECONDS = 30.0
_CLAIMABLE_JOB_STATUSES = (JobStatus.QUEUED, JobStatus.FAILED)


class GovernanceCancelled(StageCancelled):
    """Cooperative cancellation while Stage 4.2 governance was running."""


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
            target=self._run, name="stage42-liveness-heartbeat", daemon=True
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
        return "Stage 4.2 governance failed"
    if isinstance(error, GovernanceInputError):
        return str(error)[:300]
    if isinstance(error, GovernanceProviderError):
        return f"transformation_governance_provider_{error.category}"[:300]
    return type(error).__name__[:300]


def _sanitize_error_code(error: Exception | None) -> str:
    if error is None:
        return "GOVERNANCE_FAILED"
    if isinstance(error, GovernanceInputError):
        return "GOVERNANCE_INPUT"
    if isinstance(error, GovernanceProviderError):
        return f"PROVIDER_{error.category}"[:128]
    return type(error).__name__[:128]


class _LeaseBoundGovernanceProvider:
    """Acquire the shared heavy-model lease lazily around real local inference."""

    def __init__(self, inner: GovernanceProvider, lease_factory: object) -> None:
        self._inner = inner
        self._lease_factory = lease_factory
        self._lease: object | None = None
        self.provider_name = getattr(inner, "provider_name", "ollama")
        self.model = getattr(inner, "model", None)
        self.hosted_provider = False

    def _enter(self) -> object:
        if self._lease is None:
            lease = self._lease_factory.acquire(purpose="ollama")  # type: ignore[attr-defined]
            lease.__enter__()
            self._lease = lease
        return self._lease

    def govern(self, request: GovernanceRequest, tier: str = "ROUTINE") -> object:
        lease = self._enter()
        result = self._inner.govern(request, tier)
        if getattr(lease, "ownership_lost", False):
            raise HeavyModelLeaseBusy(
                "heavy-model lease was lost during Stage 4.2 governance; retry"
            )
        return result

    def release(self) -> None:
        if self._lease is None:
            self._inner.release()
            return
        lease = self._lease
        self._lease = None
        try:
            self._inner.release()
        finally:
            lease.__exit__(None, None, None)  # type: ignore[attr-defined]

    def runtime_identity(self) -> dict[str, object]:
        return self._inner.runtime_identity()

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self._inner.refresh_runtime_identity()

    def usage_summary(self) -> dict[str, int]:
        usage = getattr(self._inner, "usage_summary", None)
        return usage() if callable(usage) else {}

    def raw_call_count(self) -> int:
        counter = getattr(self._inner, "raw_call_count", None)
        return int(counter()) if callable(counter) else 0


class _AdmissionBoundGovernanceProvider:
    """Route Stage 4.2 hosted governance through the shared HIGH gate."""

    def __init__(self, inner: GovernanceProvider, admission: object) -> None:
        self._inner = inner
        self._admission = admission
        self.provider_name = getattr(inner, "provider_name", "gemini")
        self.model = getattr(inner, "model", None)
        self.hosted_provider = bool(getattr(inner, "hosted_provider", False))
        self._released = False

    def govern(self, request: GovernanceRequest, tier: str = "ROUTINE") -> object:
        try:
            decision = self._admission.acquire(AdmissionPriority.HIGH)  # type: ignore[attr-defined]
        except Exception as error:
            raise GovernanceProviderError("PROVIDER_ERROR", "admission gate unavailable") from error
        if not getattr(decision, "admitted", False):
            raise GovernanceProviderError("RATE_LIMITED", "admission denied")
        return self._inner.govern(request, tier)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._inner.release()

    def runtime_identity(self) -> dict[str, object]:
        return dict(self._inner.runtime_identity())

    def refresh_runtime_identity(self) -> dict[str, object]:
        return dict(self._inner.refresh_runtime_identity())

    def usage_summary(self) -> dict[str, int]:
        usage = getattr(self._inner, "usage_summary", None)
        result = usage() if callable(usage) else {}
        return {str(key): int(value) for key, value in dict(result).items()}

    def raw_call_count(self) -> int:
        counter = getattr(self._inner, "raw_call_count", None)
        return int(counter()) if callable(counter) else 0


class TransformationGovernanceExecutor:
    """Execute one candidate-scoped Stage 4.2 governance run and persist it."""

    def __init__(
        self,
        *,
        session: Session,
        settings: object,
        config: Stage42Config = DEFAULT_CONFIG,
        provider: GovernanceProvider | None = None,
        provider_identity: Mapping[str, object] | None = None,
        mode: SemanticProviderMode = SemanticProviderMode.DETERMINISTIC,
        lease_factory: HeavyModelLeaseFactory | NoopHeavyModelLeaseFactory | None = None,
        admission: object | None = None,
        job_claim_stale_seconds: float = _JOB_CLAIM_STALE_SECONDS,
        heartbeat_interval_seconds: float = _JOB_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._session = session
        self._settings = settings
        self._config = config
        self._provider = provider
        self._configured_identity = (
            dict(provider_identity) if provider_identity is not None else None
        )
        self._mode = mode
        self._lease_factory = lease_factory or NoopHeavyModelLeaseFactory()
        self._admission = admission
        self._job_claim_stale_seconds = max(0.0, float(job_claim_stale_seconds))
        self._heartbeat_interval_seconds = max(0.01, float(heartbeat_interval_seconds))
        self._active_job_id: object | None = None
        self._effective: GovernanceProvider | None = None
        self._job_owner = False
        self._claim_version: int | None = None
        self._governance_set_id: object | None = None
        self.claim_lost = False
        self.skipped_duplicate = False

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

    def _reconcile_cancelled(self, governance_set: TransformationGovernanceSet) -> None:
        if not self._claim_guard(require_running=False):
            self._session.rollback()
            return
        governance_set.execution_status = GovernanceExecutionStatus.CANCELLED
        governance_set.active_job_id = None
        self._session.commit()

    def _should_stop(self) -> bool:
        return self._job_cancelled() or not self._claim_is_current()

    def _provider_identity(self) -> dict[str, object]:
        if self._configured_identity is not None:
            return dict(self._configured_identity)
        if self._provider is None or self._mode is SemanticProviderMode.DETERMINISTIC:
            return DeterministicGovernanceProvider().runtime_identity()
        return dict(self._provider.runtime_identity())

    def _effective_provider(self) -> GovernanceProvider | None:
        if self._effective is not None:
            return self._effective
        if self._mode is SemanticProviderMode.DETERMINISTIC:
            return None
        if self._provider is None:
            return None
        if self._mode is SemanticProviderMode.LOCAL_ONLY:
            self._effective = _LeaseBoundGovernanceProvider(  # type: ignore[assignment]
                self._provider, self._lease_factory
            )
        elif self._admission is not None:
            self._effective = _AdmissionBoundGovernanceProvider(  # type: ignore[assignment]
                self._provider, self._admission
            )
        else:
            self._effective = self._provider
        return self._effective

    def _candidate_and_plan_set(
        self, governance_set: TransformationGovernanceSet
    ) -> tuple[ClipCandidate, TransformationPlanSet]:
        candidate = self._session.get(ClipCandidate, governance_set.clip_candidate_id)
        plan_set = self._session.get(
            TransformationPlanSet, governance_set.transformation_plan_set_id
        )
        if candidate is None or plan_set is None:
            raise GovernanceInputError("candidate or Stage 4.1 plan set is missing")
        return candidate, plan_set

    def _resolve(
        self, governance_set: TransformationGovernanceSet
    ) -> tuple[ClipCandidate, CandidateRefinement, GovernanceInputs, str]:
        candidate, plan_set = self._candidate_and_plan_set(governance_set)
        from app.transformation.inputs import resolve_effective_refinement

        refinement = resolve_effective_refinement(self._session, candidate)
        if refinement is None:
            raise GovernanceInputError("no usable Stage 3.5 refinement is available")
        inputs = build_governance_inputs(
            self._session,
            candidate,
            plan_set,
            self._settings,
            self._config,
            self._stage41_config(),
        )
        fingerprint = self._input_fingerprint(inputs)
        return candidate, refinement, inputs, fingerprint

    def _stage41_config(self) -> object:
        factory = getattr(self._settings, "stage41_config", None)
        if callable(factory):
            return factory()
        from app.transformation.planning.policy import DEFAULT_CONFIG as STAGE41_DEFAULT

        return STAGE41_DEFAULT

    def _input_fingerprint(self, inputs: GovernanceInputs) -> str:
        payload = build_governance_input_payload(
            inputs=inputs,
            config=self._config,
            provider_mode=self._mode.value,
            provider_identity=self._provider_identity(),
        )
        return governance_input_fingerprint(payload)

    def input_fingerprint(self, governance_set: TransformationGovernanceSet) -> str:
        try:
            _candidate, _refinement, inputs, fingerprint = self._resolve(governance_set)
        except Exception:
            return ""
        return fingerprint

    def is_cache_hit(
        self, governance_set: TransformationGovernanceSet, *, force: bool = False
    ) -> bool:
        if force or not governance_set.cache_eligible:
            return False
        if governance_set.execution_status not in _READY_STATUSES:
            return False
        if not governance_set.input_fingerprint:
            return False
        try:
            current = self.input_fingerprint(governance_set)
        except Exception:
            return False
        return bool(current) and current == governance_set.input_fingerprint

    def _claimed_by_other_job(self, governance_set: TransformationGovernanceSet) -> bool:
        owner = governance_set.active_job_id
        if not owner or self._active_job_id is None or str(self._active_job_id) == owner:
            return False
        try:
            owner_id: object = _uuid.UUID(owner)
        except (TypeError, ValueError):
            return False
        status = self._session.scalar(
            select(ProcessingJob.status).where(ProcessingJob.id == owner_id)
        )
        return bool(status in _ACTIVE_JOB_STATUSES)

    def _claim(self, governance_set: TransformationGovernanceSet) -> bool:
        if not self._claim_guard():
            self._session.rollback()
            return False
        governance_set.active_job_id = str(self._active_job_id) if self._active_job_id else None
        governance_set.execution_status = GovernanceExecutionStatus.GOVERNING
        self._session.commit()
        return True

    def _mark_cancelled(self, governance_set: TransformationGovernanceSet) -> None:
        if not self._claim_guard(require_running=True):
            if self.claim_lost or self._job_status() is not JobStatus.CANCELLED:
                self._session.rollback()
                return
            self._reconcile_cancelled(governance_set)
            return
        governance_set.execution_status = GovernanceExecutionStatus.CANCELLED
        governance_set.active_job_id = None
        self._session.commit()

    def _mark_failed(
        self, governance_set: TransformationGovernanceSet, error: Exception | None = None
    ) -> None:
        if not self._claim_guard(require_running=True):
            if not self.claim_lost and self._job_status() is JobStatus.CANCELLED:
                self._reconcile_cancelled(governance_set)
            else:
                self._session.rollback()
            return
        if self._job_cancelled():
            governance_set.execution_status = GovernanceExecutionStatus.CANCELLED
        else:
            governance_set.execution_status = GovernanceExecutionStatus.FAILED
        governance_set.active_job_id = None
        self._session.commit()

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

    def record_failure(self, error: Exception) -> None:
        try:
            self._session.rollback()
        except Exception:
            pass
        governance_set = (
            self._session.get(TransformationGovernanceSet, self._governance_set_id)
            if self._governance_set_id is not None
            else None
        )
        if self._active_job_id is None:
            if governance_set is not None:
                self._mark_failed(governance_set, error)
            return
        if self._claim_version is not None:
            stored = self._session.scalar(
                select(ProcessingJob.claim_version).where(ProcessingJob.id == self._active_job_id)
            )
            if stored == self._claim_version:
                status = self._job_status()
                if status is JobStatus.RUNNING:
                    self._job_owner = True
                    if governance_set is not None:
                        self._mark_failed(governance_set, error)
                    self._finish_job(JobStatus.FAILED, error)
                elif status is JobStatus.CANCELLED:
                    if governance_set is not None:
                        self._reconcile_cancelled(governance_set)
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
            if governance_set is not None:
                governance_set.execution_status = GovernanceExecutionStatus.FAILED
                governance_set.active_job_id = None
            self._session.commit()
        else:
            self._session.rollback()

    def execute(self, governance_set_id: object, *, force: bool = False) -> StageExecutionResult:
        self.skipped_duplicate = False
        self.claim_lost = False
        self._governance_set_id = governance_set_id
        governance_set = self._session.get(TransformationGovernanceSet, governance_set_id)
        if governance_set is None:
            raise GovernanceInputError("transformation governance set is missing")
        if self._job_cancelled():
            self._mark_cancelled(governance_set)
            self._finish_job(JobStatus.CANCELLED)
            raise GovernanceCancelled("Stage 4.2 governance cancelled before start")
        if not self._claim_job():
            self.skipped_duplicate = True
            return StageExecutionResult(governance_set.output_fingerprint, governance_set)
        heartbeat = self._start_heartbeat()
        try:
            try:
                _candidate, refinement, inputs, fingerprint = self._resolve(governance_set)
            except GovernanceInputError as error:
                self._mark_failed(governance_set, error)
                self._finish_job(JobStatus.FAILED, error)
                raise
            if self.is_cache_hit(governance_set, force=force):
                self._finish_job(JobStatus.SUCCEEDED)
                return StageExecutionResult(governance_set.output_fingerprint, governance_set)
            if self._claimed_by_other_job(governance_set):
                self._finish_job(JobStatus.SUCCEEDED)
                return StageExecutionResult(governance_set.output_fingerprint, governance_set)
            if not self._claim_is_current():
                return StageExecutionResult(governance_set.output_fingerprint, governance_set)
            if not self._claim(governance_set):
                return StageExecutionResult(governance_set.output_fingerprint, governance_set)
            started = monotonic()
            outcome = self._run_service(governance_set, inputs, fingerprint)
            if self._job_cancelled():
                self._mark_cancelled(governance_set)
                self._finish_job(JobStatus.CANCELLED)
                raise GovernanceCancelled("Stage 4.2 governance cancelled before persistence")
            result = self._persist(
                governance_set, refinement, inputs, outcome, monotonic() - started
            )
            if result is None:
                return StageExecutionResult(governance_set.output_fingerprint, governance_set)
            self._finish_job(JobStatus.SUCCEEDED)
            return result
        finally:
            if heartbeat is not None:
                heartbeat.stop()
            self._release_owned_providers()

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

    def _run_service(
        self,
        governance_set: TransformationGovernanceSet,
        inputs: GovernanceInputs,
        fingerprint: str,
    ) -> GovernanceOutcome:
        checkpoints = self._checkpoints(governance_set)
        service = GovernanceService(
            config=self._config,
            provider=self._effective_provider(),
            provider_identity=self._provider_identity(),
            mode=self._mode,
            is_cancelled=self._should_stop,
        )
        try:
            return service.govern(inputs, input_fingerprint=fingerprint, checkpoints=checkpoints)
        except StageCancelled:
            self._mark_cancelled(governance_set)
            self._finish_job(JobStatus.CANCELLED)
            raise
        except Exception as error:
            self._mark_failed(governance_set, error)
            self._finish_job(JobStatus.FAILED, error)
            raise

    def _checkpoints(
        self, governance_set: TransformationGovernanceSet
    ) -> dict[str, dict[str, object]]:
        checkpoints: dict[str, dict[str, object]] = {}
        for attempt in governance_set.plan_attempts or []:
            if not isinstance(attempt, dict):
                continue
            checkpoint = attempt.get("checkpoint")
            if not isinstance(checkpoint, dict):
                continue
            plan_id = str(attempt.get("plan_id", ""))
            if plan_id:
                checkpoints[plan_id] = dict(checkpoint)
        return checkpoints

    def _persist(
        self,
        governance_set: TransformationGovernanceSet,
        refinement: CandidateRefinement,
        inputs: GovernanceInputs,
        outcome: GovernanceOutcome,
        processing_duration: float,
    ) -> StageExecutionResult | None:
        if not self._claim_guard(require_running=True):
            self._session.rollback()
            if not self.claim_lost and self._job_status() is JobStatus.CANCELLED:
                self._reconcile_cancelled(governance_set)
            return None
        governance_set.refinement_id = refinement.id
        governance_set.refinement_priority = refinement.priority.value
        governance_set.refinement_quality_level = refinement.quality_level
        governance_set.execution_status = (
            GovernanceExecutionStatus.PROVIDER_DEGRADED
            if outcome.execution_status == "PROVIDER_DEGRADED"
            else GovernanceExecutionStatus.COMPLETE
        )
        governance_set.governance_outcome = outcome.semantic_outcome
        governance_set.outcome_reasons = list(outcome.outcome_reasons)
        governance_set.summary_counts = dict(outcome.summary_counts)
        governance_set.stage40_snapshot = {
            "analysis_id": inputs.stage40_analysis_id,
            "input_fingerprint": inputs.stage40_input_fingerprint,
            "output_fingerprint": inputs.stage40_output_fingerprint,
            "policy_version": inputs.stage40_policy_version,
            "assessments": dict(inputs.stage40_assessments),
            "platform_risk": dict(inputs.stage40_platform_risk),
        }
        governance_set.target_context = inputs.target_context.as_dict()
        governance_set.provider_mode = self._mode
        governance_set.provider_identity = dict(outcome.provider_identity)
        governance_set.provider_status = outcome.provider_status
        governance_set.provider_evidence = dict(outcome.provider_evidence)
        governance_set.plan_attempts = [_attempt_payload(attempt) for attempt in outcome.attempts]
        governance_set.platform_policy_profile_version = PLATFORM_POLICY_PROFILE_VERSION
        governance_set.platform_policy_checked_at = PLATFORM_POLICY_CHECKED_AT
        governance_set.input_fingerprint = outcome.input_fingerprint
        governance_set.output_fingerprint = outcome.output_fingerprint
        governance_set.policy_version = GOVERNOR_POLICY_VERSION
        governance_set.schema_version = GOVERNOR_SCHEMA_VERSION
        governance_set.validation_version = GOVERNOR_VALIDATION_VERSION
        governance_set.cache_eligible = outcome.cache_eligible
        governance_set.metrics = dict(outcome.metrics)
        governance_set.processing_duration = processing_duration
        governance_set.active_job_id = None
        self._session.flush()

        self._persist_results(governance_set, outcome.plans)
        self._session.commit()
        self._session.refresh(governance_set)
        return StageExecutionResult(outcome.output_fingerprint, governance_set)

    def _persist_results(
        self,
        governance_set: TransformationGovernanceSet,
        plans: Sequence[PlanGovernance],
    ) -> None:
        plan_ids = [_uuid.UUID(plan.plan_id) for plan in plans]
        rows = {
            row.id: row
            for row in self._session.scalars(
                select(TransformationPlan).where(TransformationPlan.id.in_(plan_ids))
            )
        }
        existing = {
            row.transformation_plan_id: row
            for row in self._session.scalars(
                select(TransformationGovernanceResult).where(
                    TransformationGovernanceResult.governance_set_id == governance_set.id
                )
            )
        }
        for plan in plans:
            plan_uuid = _uuid.UUID(plan.plan_id)
            if plan_uuid not in rows:
                continue
            row = existing.get(plan_uuid)
            if row is None:
                row = TransformationGovernanceResult(
                    governance_set_id=governance_set.id,
                    transformation_plan_id=plan_uuid,
                )
                self._session.add(row)
            row.plan_output_fingerprint = plan.plan_output_fingerprint
            row.status = plan.status
            row.eligible_for_stage4_3 = plan.eligible_for_stage4_3
            row.severity = plan.severity.value
            row.hard_gates = [dict(item) for item in plan.hard_gates]
            row.dimensions = dict(plan.dimensions)
            row.verification = dict(plan.verification)
            row.platform_risk = dict(plan.platform_risk)
            row.reason_codes = list(plan.reason_codes)
            row.warnings = [dict(item) for item in plan.warnings]
            row.remediation = [dict(item) for item in plan.remediation]
            row.governance_provider_evidence = dict(plan.provider_evidence)
            row.input_fingerprint = plan.input_fingerprint
            row.output_fingerprint = plan.output_fingerprint

    def _release_owned_providers(self) -> None:
        effective = self._effective
        self._effective = None
        target = effective if effective is not None else self._provider
        if target is None:
            return
        release = getattr(target, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass


def _attempt_payload(attempt: GovernanceAttempt) -> dict[str, object]:
    return {
        "plan_id": attempt.plan_id,
        "plan_output_fingerprint": attempt.plan_output_fingerprint,
        "status": attempt.status,
        "reasons": list(attempt.reasons),
        "provider_input_fingerprint": attempt.provider_input_fingerprint,
        "checkpoint": dict(attempt.checkpoint) if attempt.checkpoint else None,
    }


def build_transformation_governance_executor(
    session: Session,
    settings: object,
) -> "TransformationGovernanceExecutor":
    """Build the production governance executor from settings, lazily and network-free."""

    identity_factory = getattr(settings, "transformation_governance_provider_identity", None)
    provider_identity = identity_factory() if callable(identity_factory) else None
    return TransformationGovernanceExecutor(
        session=session,
        settings=settings,
        config=settings.stage42_config(),  # type: ignore[attr-defined]
        provider=settings.transformation_governance_provider(),  # type: ignore[attr-defined]
        provider_identity=provider_identity,
        mode=settings.transformation_governance_semantic_mode(),  # type: ignore[attr-defined]
        lease_factory=settings.heavy_model_lease_factory(),  # type: ignore[attr-defined]
        admission=settings.gemini_admission_controller(),  # type: ignore[attr-defined]
    )


__all__ = [
    "GovernanceCancelled",
    "JOB_CLAIM_STALE_SECONDS",
    "TransformationGovernanceExecutor",
    "build_transformation_governance_executor",
]
