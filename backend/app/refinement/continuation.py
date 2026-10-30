"""One-shot, operator-only continuation for an admitted local observation.

This module runs exactly one normal ``CANDIDATE_REFINEMENT`` task for the pinned
``FINAL_CLIP`` target. It does not admit evidence and it does not use a broker.
Instead it captures the normal queue dispatch, then executes the same Celery
task synchronously in this process while a deny-only ASR collaborator is
installed on the task's production executor factory.

The guard is deliberately narrow: the real factory still owns the decoder and
input identity, the normal task still owns job claim/failure/cancellation and
cleanup, and the only replaced collaborator raises before model work when the
accepted/rejected observation reuse path misses. It never returns a fake
transcript.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.enums import JobStatus, RefinementPriority, RefinementStatus
from app.models import CandidateRefinement, ClipCandidate, ProcessingJob
from app.refinement.queue import (
    Stage35QueueError,
    queue_candidate_refinement,
    validate_candidate_for_refinement,
)
from app.refinement.recovery import LocalObservationExpectation
from app.services.storage import StorageService

_READY_STATUSES = {
    RefinementStatus.CANDIDATE_REFINED,
    RefinementStatus.FINAL_TRANSCRIPT_READY,
}
_FAILED_STATUSES = {
    RefinementStatus.REFINEMENT_FAILED,
    RefinementStatus.CANCELLED,
}


class ContinuationError(RuntimeError):
    """A continuation could not be prepared or executed truthfully."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ContinuationConfigurationError(ContinuationError):
    """The environment or exact target is incompatible with the guarded run."""


class ContinuationModelCallDenied(RuntimeError):
    """Checkpoint reuse missed; no local or hosted model work is permitted."""


class DenyOnlyASR:
    """Deny-only ASR collaborator installed for the one guarded invocation."""

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.calls += 1
        raise ContinuationModelCallDenied(
            "checkpoint reuse missed; targeted ASR is denied in this continuation"
        )


@dataclass(frozen=True)
class PreparedContinuation:
    job_id: UUID
    task_args: tuple[object, ...]


@dataclass(frozen=True)
class GuardedExecution:
    cancelled: bool
    error: str | None
    denial_calls: int


@dataclass(frozen=True)
class ContinuationResult:
    candidate_id: UUID
    refinement_id: UUID
    job_id: UUID
    succeeded: bool
    cancelled: bool
    denial_calls: int
    job_status: str | None
    refinement_status: str | None
    metrics: dict[str, object]
    error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": str(self.candidate_id),
            "refinement_id": str(self.refinement_id),
            "job_id": str(self.job_id),
            "succeeded": self.succeeded,
            "cancelled": self.cancelled,
            "denial_calls": self.denial_calls,
            "job_status": self.job_status,
            "refinement_status": self.refinement_status,
            "metrics": dict(self.metrics),
            "error": self.error,
        }


def validate_continuation_settings(settings: object) -> None:
    """Fail closed unless the continuation is local-only and provider-free."""

    if getattr(settings, "refinement_routing_mode", None) != "local_only":
        raise ContinuationConfigurationError(
            "routing_mode_not_local_only",
            "continuation requires CLIPFACTORY_REFINEMENT_ROUTING_MODE=local_only",
        )
    if bool(getattr(settings, "local_qwen_enabled", False)):
        raise ContinuationConfigurationError(
            "local_qwen_enabled",
            "continuation requires CLIPFACTORY_LOCAL_QWEN_ENABLED=false",
        )
    if bool(getattr(settings, "gemini_api_key_present", False)):
        raise ContinuationConfigurationError(
            "hosted_key_present",
            "continuation requires no configured Gemini key",
        )


@contextmanager
def deny_only_asr_guard(deny: DenyOnlyASR) -> Iterator[None]:
    """Install a process-local deny-only ASR on the task's executor factory."""

    from app.workers import tasks

    original = tasks.build_candidate_refinement_executor

    def guarded(session: Session, storage: StorageService, settings: object) -> object:
        executor = original(session, storage, settings)
        if getattr(executor, "_asr", None) is None:
            raise ContinuationConfigurationError(
                "asr_unavailable",
                "production executor has no targeted ASR collaborator to guard",
            )
        executor._asr = deny  # type: ignore[attr-defined]
        return executor

    tasks.build_candidate_refinement_executor = guarded  # type: ignore[assignment]
    try:
        yield
    finally:
        tasks.build_candidate_refinement_executor = original  # type: ignore[assignment]


def _target(
    session: Session, expectation: LocalObservationExpectation
) -> tuple[ClipCandidate, CandidateRefinement]:
    try:
        candidate = validate_candidate_for_refinement(session, expectation.candidate_id)
    except Stage35QueueError as error:
        raise ContinuationConfigurationError("candidate_refused", str(error)) from error
    refinement = session.get(CandidateRefinement, expectation.refinement_id)
    if refinement is None:
        raise ContinuationConfigurationError(
            "refinement_not_found", "pinned candidate refinement does not exist"
        )
    if candidate.source_video_id != expectation.source_id:
        raise ContinuationConfigurationError(
            "candidate_source_mismatch", "candidate is not from the pinned source"
        )
    if refinement.source_video_id != expectation.source_id:
        raise ContinuationConfigurationError(
            "refinement_source_mismatch", "refinement is not from the pinned source"
        )
    if refinement.clip_candidate_id != candidate.id:
        raise ContinuationConfigurationError(
            "refinement_candidate_mismatch", "refinement is not the pinned candidate"
        )
    if (
        refinement.priority is not RefinementPriority.FINAL_CLIP
        or refinement.quality_level != RefinementPriority.FINAL_CLIP.value
    ):
        raise ContinuationConfigurationError(
            "refinement_priority", "continuation only supports a FINAL_CLIP refinement"
        )
    if refinement.cache_eligible or refinement.status in _READY_STATUSES:
        raise ContinuationConfigurationError(
            "refinement_ready",
            "refinement is already ready; do not run the guarded continuation",
        )
    return candidate, refinement


def prepare_continuation_job(
    session: Session,
    storage: StorageService,
    settings: object,
    expectation: LocalObservationExpectation,
) -> PreparedContinuation:
    """Queue exactly one normal FINAL_CLIP job without dispatching to a broker."""

    validate_continuation_settings(settings)
    candidate, refinement = _target(session, expectation)
    from app.refinement import queue as queue_module

    captured: list[tuple[object, ...]] = []
    original_dispatch = queue_module._dispatch

    def capture(*args: object, **kwargs: object) -> None:
        del kwargs
        captured.append(tuple(args))

    queue_module._dispatch = capture  # type: ignore[assignment]
    try:
        outcome = queue_candidate_refinement(
            session,
            storage,
            candidate,
            RefinementPriority.FINAL_CLIP,
            force=False,
        )
    except Stage35QueueError as error:
        raise ContinuationConfigurationError("queue_refused", str(error)) from error
    finally:
        queue_module._dispatch = original_dispatch  # type: ignore[assignment]

    if outcome.active:
        raise ContinuationConfigurationError(
            "refinement_active", "refinement already owns an active job"
        )
    if outcome.cached:
        raise ContinuationConfigurationError(
            "refinement_ready", "refinement already matches a cached result"
        )
    if not outcome.queued or outcome.job_id is None:
        raise ContinuationConfigurationError(
            "job_not_queued", "normal queue did not create a continuation job"
        )
    if len(captured) != 1:
        raise ContinuationConfigurationError(
            "dispatch_capture", "normal queue did not produce exactly one dispatch"
        )
    raw = captured[0]
    if len(raw) != 5:
        raise ContinuationConfigurationError(
            "dispatch_mismatch", "captured dispatch is not the exact pinned target"
        )
    args = (
        str(raw[0]),
        getattr(raw[1], "value", str(raw[1])),
        str(raw[2]),
        str(raw[3]),
        bool(raw[4]),
    )
    if (
        args[0] != str(candidate.id)
        or args[1] != RefinementPriority.FINAL_CLIP.value
        or args[2] != str(refinement.id)
        or args[3] != str(outcome.job_id)
        or args[4] is not False
    ):
        raise ContinuationConfigurationError(
            "dispatch_mismatch", "captured dispatch is not the exact pinned target"
        )
    return PreparedContinuation(outcome.job_id, args)


def execute_guarded_job(
    prepared: PreparedContinuation,
    *,
    expected_candidate_id: UUID,
    expected_refinement_id: UUID,
) -> GuardedExecution:
    """Execute the captured task synchronously under the deny-only ASR guard."""

    args = prepared.task_args
    if (
        len(args) != 5
        or str(args[0]) != str(expected_candidate_id)
        or str(args[2]) != str(expected_refinement_id)
        or args[1] != RefinementPriority.FINAL_CLIP.value
        or str(args[3]) != str(prepared.job_id)
        or args[4] is not False
    ):
        raise ContinuationConfigurationError(
            "target_mismatch", "guarded task arguments are not the exact pinned target"
        )

    from app.workers.tasks import run_candidate_refinement

    deny = DenyOnlyASR()
    error: str | None = None
    cancelled = False
    with deny_only_asr_guard(deny):
        try:
            result = run_candidate_refinement.run(*args)
        except Exception as exc:  # noqa: BLE001 - truthful failure is the contract
            error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        else:
            cancelled = bool(result.get("cancelled")) if isinstance(result, dict) else False
    return GuardedExecution(cancelled=cancelled, error=error, denial_calls=deny.calls)


def run_guarded_continuation(
    session: Session,
    storage: StorageService,
    settings: object,
    expectation: LocalObservationExpectation,
) -> ContinuationResult:
    """Run the exact pinned FINAL_CLIP continuation once and report truthfully."""

    prepared = prepare_continuation_job(session, storage, settings, expectation)
    execution = execute_guarded_job(
        prepared,
        expected_candidate_id=expectation.candidate_id,
        expected_refinement_id=expectation.refinement_id,
    )

    session.expire_all()
    job = session.get(ProcessingJob, prepared.job_id)
    refinement = session.get(CandidateRefinement, expectation.refinement_id)
    job_status = job.status.value if job is not None else None
    refinement_status = refinement.status.value if refinement is not None else None
    error = execution.error
    if error is None and job is not None and job.status is JobStatus.FAILED:
        error = job.error_message or "continuation job failed"
    succeeded = bool(
        not execution.cancelled
        and error is None
        and job is not None
        and job.status is JobStatus.SUCCEEDED
        and refinement is not None
        and refinement.status not in _FAILED_STATUSES
    )
    return ContinuationResult(
        candidate_id=expectation.candidate_id,
        refinement_id=expectation.refinement_id,
        job_id=prepared.job_id,
        succeeded=succeeded,
        cancelled=execution.cancelled,
        denial_calls=execution.denial_calls,
        job_status=job_status,
        refinement_status=refinement_status,
        metrics=dict(refinement.metrics or {}) if refinement is not None else {},
        error=error,
    )
