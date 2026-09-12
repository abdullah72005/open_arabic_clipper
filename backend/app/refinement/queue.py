"""Queueing, idempotency, manual resolution, and batch entry points."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import (
    CandidateDisposition,
    JobKind,
    JobStatus,
    RefinementPriority,
)
from app.models import CandidateRefinement, ClipCandidate, ProcessingJob, SourceVideo
from app.refinement.executor import build_candidate_refinement_executor
from app.refinement.types import priority_is_stage35
from app.services.storage import StorageService

_VALID_DISPOSITIONS = {
    CandidateDisposition.CANDIDATE,
    CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
}
_ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}


class Stage35QueueError(ValueError):
    """A Stage 3.5 queue request was stale, rejected, cross-source, or malformed."""


@dataclass(frozen=True)
class QueueOutcome:
    refinement_id: uuid.UUID
    job_id: uuid.UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool
    skipped_reason: str | None = None


def validate_candidate_for_refinement(
    session: Session, candidate_id: uuid.UUID | str
) -> ClipCandidate:
    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        raise Stage35QueueError("candidate does not exist")
    if not candidate.is_current:
        raise Stage35QueueError("candidate is stale and cannot be refined")
    if candidate.disposition not in _VALID_DISPOSITIONS:
        raise Stage35QueueError("candidate was not retained for refinement")
    return candidate


def get_or_create_refinement(
    session: Session,
    candidate: ClipCandidate,
    priority: RefinementPriority,
) -> CandidateRefinement:
    if not priority_is_stage35(priority):
        raise Stage35QueueError("Stage 3.5 only supports CANDIDATE/FINAL_CLIP priorities")
    existing = session.scalar(
        select(CandidateRefinement).where(
            CandidateRefinement.clip_candidate_id == candidate.id,
            CandidateRefinement.priority == priority,
        )
    )
    if existing is not None:
        return existing
    refinement = CandidateRefinement(
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        priority=priority,
        coarse_start=float(candidate.start_time),
        coarse_end=float(candidate.end_time),
        context_start=float(candidate.start_time),
        context_end=float(candidate.end_time),
    )
    session.add(refinement)
    session.commit()
    session.refresh(refinement)
    return refinement


def _active_job(session: Session, refinement: CandidateRefinement) -> ProcessingJob | None:
    jobs = session.scalars(
        select(ProcessingJob)
        .where(
            ProcessingJob.candidate_refinement_id == refinement.id,
            ProcessingJob.kind == JobKind.CANDIDATE_REFINEMENT,
            ProcessingJob.status.in_(_ACTIVE_STATUSES),
        )
        .order_by(ProcessingJob.created_at.desc())
    ).all()
    return jobs[0] if jobs else None


def _matches_cache(
    session: Session,
    storage: StorageService,
    candidate: ClipCandidate,
    refinement: CandidateRefinement,
    priority: RefinementPriority,
) -> bool:
    if not refinement.cache_eligible or not refinement.input_fingerprint:
        return False
    executor = build_candidate_refinement_executor(session, storage, _settings())
    try:
        current = executor.input_fingerprint(candidate, priority)
    except Exception:
        return False
    return bool(current == refinement.input_fingerprint)


def queue_candidate_refinement(
    session: Session,
    storage: StorageService,
    candidate: ClipCandidate,
    priority: RefinementPriority,
    *,
    force: bool = False,
) -> QueueOutcome:
    refinement = get_or_create_refinement(session, candidate, priority)
    active = _active_job(session, refinement)
    if active is not None:
        return QueueOutcome(
            refinement_id=refinement.id,
            job_id=active.id,
            status=refinement.status.value,
            queued=False,
            cached=False,
            active=True,
        )
    if _matches_cache(session, storage, candidate, refinement, priority):
        return QueueOutcome(
            refinement_id=refinement.id,
            job_id=None,
            status=refinement.status.value,
            queued=False,
            cached=True,
            active=False,
        )
    job = ProcessingJob(
        source_video_id=candidate.source_video_id,
        kind=JobKind.CANDIDATE_REFINEMENT,
        candidate_refinement_id=refinement.id,
    )
    session.add(job)
    session.flush()
    refinement.active_job_id = str(job.id)
    session.commit()
    session.refresh(job)
    _dispatch(candidate.id, priority, refinement.id, job.id, force)
    return QueueOutcome(
        refinement_id=refinement.id,
        job_id=job.id,
        status=refinement.status.value,
        queued=True,
        cached=False,
        active=False,
    )


def queue_candidate_batch(
    session: Session,
    storage: StorageService,
    source_id: uuid.UUID | str,
    *,
    limit: int | None = None,
    force: bool = False,
) -> list[QueueOutcome]:
    resolved_source = _as_uuid(source_id)
    if session.get(SourceVideo, resolved_source) is None:
        raise Stage35QueueError("source does not exist")
    from app.core.settings import get_settings

    settings = get_settings()
    default_limit = settings.refinement_batch_default_limit
    max_limit = settings.refinement_batch_max_limit
    effective = min(max(limit or default_limit, 1), max_limit)
    candidates = session.scalars(
        select(ClipCandidate)
        .where(
            ClipCandidate.source_video_id == resolved_source,
            ClipCandidate.is_current.is_(True),
            ClipCandidate.disposition.in_(_VALID_DISPOSITIONS),
        )
        .order_by(ClipCandidate.clip_score.desc(), ClipCandidate.start_time.asc())
    ).all()
    outcomes: list[QueueOutcome] = []
    for candidate in candidates:
        if len(outcomes) >= effective:
            break
        existing = session.scalar(
            select(CandidateRefinement).where(
                CandidateRefinement.clip_candidate_id == candidate.id,
                CandidateRefinement.priority == RefinementPriority.CANDIDATE,
            )
        )
        if existing is not None and _active_job(session, existing) is not None:
            continue
        if (
            existing is not None
            and existing.status.value
            in {
                "CANDIDATE_REFINED",
                "FINAL_TRANSCRIPT_READY",
            }
            and existing.cache_eligible
        ):
            continue
        outcomes.append(
            queue_candidate_refinement(
                session,
                storage,
                candidate,
                RefinementPriority.CANDIDATE,
                force=force,
            )
        )
    return outcomes


def apply_manual_transcript(
    session: Session,
    refinement_id: uuid.UUID | str,
    text: str,
    resolutions: dict[str, str] | None = None,
) -> CandidateRefinement:
    refinement = session.get(CandidateRefinement, _as_uuid(refinement_id))
    if refinement is None:
        raise Stage35QueueError("candidate refinement does not exist")
    cleaned = text.strip()
    if not cleaned:
        raise Stage35QueueError("manual transcript must not be empty")
    refinement.manual_transcript = cleaned
    refinement.final_transcript = cleaned
    pending_critical = False
    for span in refinement.unresolved_spans or []:
        if not isinstance(span, dict):
            continue
        span_id = str(span.get("span_id", ""))
        if resolutions and span_id in resolutions:
            span["resolution_state"] = "RESOLVED"
            span["operator_resolution"] = resolutions[span_id]
            continue
        if span.get("meaning_critical") and span.get("resolution_state") != "RESOLVED":
            pending_critical = True
    if pending_critical or (
        refinement.priority is RefinementPriority.FINAL_CLIP
        and not (refinement.word_timestamps or [])
    ):
        refinement.status = _review_status(refinement)
    else:
        refinement.status = _ready_status(refinement)
    refinement.cache_eligible = refinement.status.value in {
        "CANDIDATE_REFINED",
        "FINAL_TRANSCRIPT_READY",
    }
    session.commit()
    session.refresh(refinement)
    return refinement


def list_refinements(session: Session, candidate_id: uuid.UUID | str) -> list[CandidateRefinement]:
    return list(
        session.scalars(
            select(CandidateRefinement)
            .where(CandidateRefinement.clip_candidate_id == _as_uuid(candidate_id))
            .order_by(CandidateRefinement.priority.asc())
        ).all()
    )


def get_refinement(session: Session, refinement_id: uuid.UUID | str) -> CandidateRefinement | None:
    return session.get(CandidateRefinement, _as_uuid(refinement_id))


def _ready_status(refinement: CandidateRefinement) -> object:
    from app.core.enums import RefinementStatus

    if refinement.priority is RefinementPriority.FINAL_CLIP:
        return RefinementStatus.FINAL_TRANSCRIPT_READY
    return RefinementStatus.CANDIDATE_REFINED


def _review_status(refinement: CandidateRefinement) -> object:
    from app.core.enums import RefinementStatus

    if refinement.priority is RefinementPriority.FINAL_CLIP:
        return RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
    return RefinementStatus.CANDIDATE_REFINED


def _dispatch(
    candidate_id: uuid.UUID,
    priority: RefinementPriority,
    refinement_id: uuid.UUID,
    job_id: uuid.UUID,
    force: bool,
) -> None:
    from app.workers.tasks import run_candidate_refinement

    run_candidate_refinement.delay(
        str(candidate_id), priority.value, str(refinement_id), str(job_id), force
    )


def _settings() -> object:
    from app.core.settings import get_settings

    return get_settings()


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as error:
        raise Stage35QueueError("invalid identifier") from error
