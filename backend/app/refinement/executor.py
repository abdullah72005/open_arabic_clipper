"""Durable Stage 3.5 candidate-refinement executor and cleanup.

The executor atomically claims one ``(candidate, priority)`` refinement row,
runs the bounded refinement service, and persists a truthful result. It observes
the exact executing ``ProcessingJob`` for cooperative cancellation, preserves
accepted checkpoints, and releases owned providers on every exit path.
"""

from __future__ import annotations

from time import monotonic

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import (
    JobStatus,
    RefinementPriority,
    RefinementStatus,
)
from app.models import CandidateRefinement, ClipCandidate, ProcessingJob
from app.pipeline.executor import StageCancelled, StageExecutionResult
from app.refinement.audio_window import CandidateAudioWindowService
from app.refinement.policy import DEFAULT_CONFIG, REFINEMENT_SCHEMA_VERSION, Stage35Config
from app.refinement.service import CandidateRefinementService, RefinementError
from app.refinement.types import RefinementOutcome, priority_is_stage35
from app.services.storage import StorageService

_READY_STATUSES = {
    RefinementStatus.CANDIDATE_REFINED,
    RefinementStatus.FINAL_TRANSCRIPT_READY,
    RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW,
    RefinementStatus.PROVIDER_DEGRADED,
}
_ACTIVE_JOB_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}


class CandidateRefinementCancelled(StageCancelled):
    """Cooperative cancellation while Stage 3.5 refinement was running."""


class CandidateRefinementExecutor:
    """Execute one candidate-scoped refinement and persist its bounded result."""

    def __init__(
        self,
        *,
        session: Session,
        storage: StorageService,
        config: Stage35Config = DEFAULT_CONFIG,
        audio_service: CandidateAudioWindowService | None = None,
        asr_engine: object | None = None,
        hosted_provider: object | None = None,
        adjudication_provider: object | None = None,
        admission: object | None = None,
        routing_mode: str = "adaptive",
        local_qwen_enabled: bool = False,
        qwen_reconstructor: object | None = None,
        local_identity: dict[str, object] | None = None,
    ) -> None:
        self._session = session
        self._storage = storage
        self._config = config
        self._audio_service = audio_service
        self._asr = asr_engine
        self._hosted = hosted_provider
        self._adjudicator = adjudication_provider
        self._admission = admission
        self._routing_mode = routing_mode
        self._local_qwen_enabled = local_qwen_enabled
        self._qwen = qwen_reconstructor
        self._local_identity = local_identity or {}
        self._active_job_id: object | None = None

    def set_active_job(self, job_id: object | None) -> None:
        self._active_job_id = job_id

    def _job_cancelled(self) -> bool:
        if self._active_job_id is None:
            return False
        status = self._session.scalar(
            select(ProcessingJob.status).where(ProcessingJob.id == self._active_job_id)
        )
        return status is JobStatus.CANCELLED

    def _service(self) -> CandidateRefinementService:
        return CandidateRefinementService(
            session=self._session,
            storage=self._storage,
            config=self._config,
            audio_service=self._audio_service,
            asr_engine=self._asr,
            hosted_provider=self._hosted,
            adjudication_provider=self._adjudicator,
            admission=self._admission,
            routing_mode=self._routing_mode,
            local_qwen_enabled=self._local_qwen_enabled,
            qwen_reconstructor=self._qwen,
            local_identity=self._local_identity,
            is_cancelled=self._job_cancelled,
        )

    def input_fingerprint(self, candidate: ClipCandidate, priority: RefinementPriority) -> str:
        return self._service().input_fingerprint(candidate, priority)

    def _candidate(self, refinement: CandidateRefinement) -> ClipCandidate:
        candidate = self._session.get(ClipCandidate, refinement.clip_candidate_id)
        if candidate is None:
            raise RefinementError("candidate is missing for refinement")
        return candidate

    def is_cache_hit(self, refinement: CandidateRefinement, *, force: bool = False) -> bool:
        if force or not refinement.cache_eligible:
            return False
        if refinement.status not in {
            RefinementStatus.CANDIDATE_REFINED,
            RefinementStatus.FINAL_TRANSCRIPT_READY,
        }:
            return False
        if not refinement.input_fingerprint:
            return False
        candidate = self._candidate(refinement)
        return bool(
            self.input_fingerprint(candidate, refinement.priority) == refinement.input_fingerprint
        )

    def execute(self, refinement_id: object, *, force: bool = False) -> StageExecutionResult:
        refinement = self._session.get(CandidateRefinement, refinement_id)
        if refinement is None:
            raise RefinementError("candidate refinement is missing")
        candidate = self._candidate(refinement)
        if not priority_is_stage35(refinement.priority):
            raise RefinementError("Stage 3.5 only supports CANDIDATE/FINAL_CLIP priorities")
        if self._job_cancelled():
            self._mark_cancelled(refinement)
            raise CandidateRefinementCancelled("candidate refinement cancelled before start")
        if self.is_cache_hit(refinement, force=force):
            return StageExecutionResult(refinement.output_fingerprint, refinement)
        if self._claimed_by_other_job(refinement):
            return StageExecutionResult(refinement.output_fingerprint, refinement)

        self._claim(refinement)
        started = monotonic()
        service = self._service()
        try:
            outcome = service.execute(
                candidate, priority=refinement.priority, force=force, prior=refinement
            )
        except StageCancelled:
            self._mark_cancelled(refinement)
            raise
        except Exception:
            self._mark_failed(refinement)
            raise
        finally:
            self._release_owned_providers()
        if self._job_cancelled():
            self._mark_cancelled(refinement)
            raise CandidateRefinementCancelled("candidate refinement cancelled before persistence")
        return self._persist(refinement, outcome, monotonic() - started)

    # ------------------------------------------------------------------
    # claiming / persistence

    def _claimed_by_other_job(self, refinement: CandidateRefinement) -> bool:
        owner = refinement.active_job_id
        if not owner or self._active_job_id is None or str(self._active_job_id) == owner:
            return False
        try:
            import uuid as _uuid

            owner_id: object = _uuid.UUID(owner)
        except (TypeError, ValueError):
            return False
        status = self._session.scalar(
            select(ProcessingJob.status).where(ProcessingJob.id == owner_id)
        )
        return bool(status in _ACTIVE_JOB_STATUSES)

    def _claim(self, refinement: CandidateRefinement) -> None:
        refinement.active_job_id = str(self._active_job_id) if self._active_job_id else None
        refinement.status = RefinementStatus.REFINING
        self._session.commit()

    def _mark_cancelled(self, refinement: CandidateRefinement) -> None:
        refinement.status = RefinementStatus.CANCELLED
        refinement.active_job_id = None
        self._session.commit()

    def _mark_failed(self, refinement: CandidateRefinement) -> None:
        if self._job_cancelled():
            refinement.status = RefinementStatus.CANCELLED
        else:
            refinement.status = RefinementStatus.REFINEMENT_FAILED
        refinement.active_job_id = None
        self._session.commit()

    def _persist(
        self,
        refinement: CandidateRefinement,
        outcome: RefinementOutcome,
        processing_duration: float,
    ) -> StageExecutionResult:
        refinement.status = RefinementStatus(outcome.status)
        refinement.quality_level = outcome.quality_level
        refinement.coarse_start = outcome.coarse_start
        refinement.coarse_end = outcome.coarse_end
        refinement.context_start = outcome.context_start
        refinement.context_end = outcome.context_end
        refinement.refined_start = outcome.refined_start
        refinement.refined_end = outcome.refined_end
        refinement.audio_relative_path = outcome.audio_relative_path
        refinement.audio_content_hash = outcome.audio_content_hash
        refinement.audio_input_fingerprint = outcome.audio_input_fingerprint
        refinement.automatic_transcript = outcome.automatic_transcript
        if outcome.manual_transcript is not None:
            refinement.manual_transcript = outcome.manual_transcript
        refinement.final_transcript = outcome.final_transcript
        refinement.word_timestamps = [word.as_dict for word in outcome.word_timestamps]
        refinement.confidence = outcome.confidence
        refinement.dialect_profile = outcome.dialect_profile
        refinement.dialect_confidence = outcome.dialect_confidence
        refinement.code_switch_evidence = dict(outcome.code_switch_evidence)
        refinement.transcript_evidence = [
            record.as_dict() for record in outcome.transcript_evidence
        ]
        refinement.entity_evidence = [
            {
                "text": mention.text,
                "normalized": mention.normalized,
                "entity_type": mention.entity_type,
                "start": mention.start,
                "end": mention.end,
                "evidence_fingerprints": list(mention.evidence_fingerprints),
            }
            for mention in outcome.entity_evidence
        ]
        refinement.unresolved_spans = [span.as_dict() for span in outcome.unresolved_spans]
        refinement.provider_evidence = dict(outcome.provider_evidence)
        refinement.routing_evidence = dict(outcome.routing_evidence)
        refinement.input_fingerprint = outcome.input_fingerprint
        refinement.output_fingerprint = outcome.output_fingerprint
        refinement.component_fingerprints = dict(outcome.component_fingerprints)
        refinement.policy_version = REFINEMENT_SCHEMA_VERSION
        refinement.schema_version = REFINEMENT_SCHEMA_VERSION
        refinement.cache_eligible = outcome.cache_eligible
        refinement.metrics = dict(outcome.metrics)
        refinement.processing_duration = processing_duration
        refinement.active_job_id = None
        self._session.commit()
        self._session.refresh(refinement)
        return StageExecutionResult(outcome.output_fingerprint, refinement)

    def _release_owned_providers(self) -> None:
        for provider in (self._hosted, self._adjudicator):
            release = getattr(provider, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass


def build_candidate_refinement_executor(
    session: Session,
    storage: StorageService,
    settings: object,
) -> CandidateRefinementExecutor:
    """Build the production executor from settings, lazily and without network."""

    from app.transcription.engine import WhisperEngine

    engine = WhisperEngine()
    priority_names = {
        RefinementPriority.CANDIDATE: "CANDIDATE",
        RefinementPriority.FINAL_CLIP: "FINAL_CLIP",
    }

    def options_for(priority: RefinementPriority):  # type: ignore[no-untyped-def]
        return settings.targeted_transcription_options(priority_names[priority])  # type: ignore[attr-defined]

    asr = None
    try:
        from app.refinement.asr import TargetedASREngine

        lease_factory = settings.heavy_model_lease_factory()  # type: ignore[attr-defined]
        asr = TargetedASREngine(options_for=options_for, engine=engine, lease_factory=lease_factory)
    except Exception:
        asr = None

    qwen = None
    routing_mode = getattr(settings, "refinement_routing_mode", "adaptive")
    local_qwen_enabled = bool(getattr(settings, "local_qwen_enabled", False))
    if routing_mode == "local_only" and local_qwen_enabled:
        try:
            qwen = settings.contextual_reconstructor()  # type: ignore[attr-defined]
        except Exception:
            qwen = None

    local_identity = {
        "provider": "faster-whisper",
        "model": getattr(settings, "whisper_model", None),
        "device": getattr(settings, "whisper_device", None),
        "condition_on_previous_text": False,
        "vad_filter": False,
        "candidate_beam_size": getattr(settings, "refinement_candidate_beam_size", 5),
        "final_beam_size": getattr(settings, "refinement_final_beam_size", 8),
    }
    return CandidateRefinementExecutor(
        session=session,
        storage=storage,
        config=settings.stage35_config(),  # type: ignore[attr-defined]
        asr_engine=asr,
        hosted_provider=settings.hosted_transcription_provider(),  # type: ignore[attr-defined]
        adjudication_provider=settings.adjudication_provider(),  # type: ignore[attr-defined]
        admission=settings.gemini_admission_controller(),  # type: ignore[attr-defined]
        routing_mode=routing_mode,
        local_qwen_enabled=local_qwen_enabled,
        qwen_reconstructor=qwen,
        local_identity=local_identity,
    )
