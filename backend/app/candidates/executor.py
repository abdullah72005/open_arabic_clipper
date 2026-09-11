"""Durable Stage 3 candidate-analysis executor."""

from __future__ import annotations

from collections.abc import Sequence
from time import monotonic

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.candidates.fingerprints import novelty_corpus_digest
from app.candidates.novelty import NoveltyItem, idea_signature, topic_signature
from app.candidates.policy import (
    DEFAULT_CONFIG,
    POLICY_VERSION,
    SCORING_VERSION,
    Stage3Config,
)
from app.candidates.providers import (
    DeterministicSemanticProvider,
    SemanticEvaluationRequest,
    SemanticEvaluationResult,
    SemanticProvider,
    semantic_prompt_hash,
)
from app.candidates.service import (
    CandidateAnalysisService,
    build_input_fingerprint_payload,
    semantic_result_from_evidence,
)
from app.candidates.types import CandidateAnalysisOutcome, CandidateDraft
from app.core.enums import (
    CandidateDisposition,
    JobStatus,
    SemanticProviderMode,
)
from app.models import (
    AudioAnalysis,
    CandidateAnalysis,
    ClipCandidate,
    ProcessingJob,
    SourceQualityAssessment,
    SourceVideo,
    Transcript,
)
from app.pipeline.executor import StageCancelled, StageExecutionResult
from app.pipeline.fingerprints import canonical_fingerprint
from app.pipeline.runner import StageExecutionError
from app.runtime.heavy_model_lease import (
    HeavyModelLeaseBusy,
    HeavyModelLeaseFactory,
    NoopHeavyModelLeaseFactory,
)
from app.transcription.dialect import DIALECT_POLICY_VERSION

_RETAINED_DISPOSITIONS = {
    CandidateDisposition.CANDIDATE,
    CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
}


class CandidateAnalysisCancelled(StageCancelled):
    """Cooperative cancellation while Stage 3 candidate analysis was running."""


class _LeaseBoundSemanticProvider:
    """Acquire the shared heavy-model lease lazily around real local inference."""

    def __init__(self, inner: SemanticProvider, lease_factory: object) -> None:
        self._inner = inner
        self._lease_factory = lease_factory
        self._lease: object | None = None
        self.provider_name = getattr(inner, "provider_name", "ollama")
        self.model = getattr(inner, "model", None)

    def _enter(self) -> object:
        if self._lease is None:
            lease = self._lease_factory.acquire(purpose="ollama")  # type: ignore[attr-defined]
            lease.__enter__()
            self._lease = lease
        return self._lease

    def evaluate(
        self, requests: Sequence[SemanticEvaluationRequest]
    ) -> dict[str, SemanticEvaluationResult]:
        lease = self._enter()
        result = self._inner.evaluate(requests)
        if getattr(lease, "ownership_lost", False):
            raise HeavyModelLeaseBusy(
                "heavy-model lease was lost during candidate analysis; retry the stage"
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


class CandidateAnalysisExecutor:
    """Persist a bounded, atomic Stage 3 candidate-analysis result."""

    def __init__(
        self,
        *,
        session: Session,
        config: Stage3Config = DEFAULT_CONFIG,
        provider: SemanticProvider | None = None,
        mode: SemanticProviderMode = SemanticProviderMode.DETERMINISTIC,
        lease_factory: HeavyModelLeaseFactory | NoopHeavyModelLeaseFactory | None = None,
    ) -> None:
        self._session = session
        self._config = config
        self._provider = provider
        self._mode = mode
        self._lease_factory = lease_factory or NoopHeavyModelLeaseFactory()
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

    # ------------------------------------------------------------------
    # lifecycle

    def _provider_identity(self) -> dict[str, object]:
        if self._provider is None:
            return DeterministicSemanticProvider().runtime_identity()
        return self._provider.runtime_identity()

    def input_fingerprint(self, source: SourceVideo) -> str:
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        analysis = self._session.scalar(
            select(AudioAnalysis).where(AudioAnalysis.source_video_id == source.id)
        )
        if transcript is None or analysis is None:
            return ""
        quality = self._session.scalar(
            select(SourceQualityAssessment).where(
                SourceQualityAssessment.source_video_id == source.id
            )
        )
        corpus = self._historical_corpus(source.id)
        return build_input_fingerprint_payload(
            source_id=str(source.id),
            content_hash=source.content_hash,
            duration=transcript.duration,
            rights_status=source.rights_status,
            media_origin=source.media_origin,
            provenance_metadata=source.provenance_metadata or {},
            transcript_input_fingerprint=transcript.input_fingerprint,
            transcription_revision=transcript.transcription_revision,
            normalization_fingerprint=transcript.normalization_fingerprint,
            reconstruction_fingerprint=transcript.reconstruction_fingerprint,
            reconstruction_status=(
                transcript.reconstruction_status.value
                if hasattr(transcript.reconstruction_status, "value")
                else str(transcript.reconstruction_status)
            ),
            reconstruction_version=transcript.reconstruction_version,
            segments=transcript.segments,
            language=transcript.language,
            dialect_profile=(
                transcript.dialect_profile.value if transcript.dialect_profile else None
            ),
            dialect_confidence=transcript.dialect_confidence,
            dialect_policy_version=DIALECT_POLICY_VERSION,
            audio_input_fingerprint=analysis.input_fingerprint,
            silence_intervals=analysis.silence_intervals,
            audio_features=analysis.features,
            quality_input_fingerprint=quality.input_fingerprint if quality else "",
            config=self._config,
            provider_identity=self._provider_identity(),
            novelty_digest=novelty_corpus_digest([item.key for item in corpus]),
        )

    def skip_is_allowed(self, source: SourceVideo) -> bool:
        analysis = self._session.scalar(
            select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
        )
        if analysis is None:
            return False
        return analysis.cache_eligible is True

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        analysis = self._session.scalar(
            select(AudioAnalysis).where(AudioAnalysis.source_video_id == source.id)
        )
        if transcript is None:
            raise StageExecutionError("normalized transcript is required for candidate analysis")
        if analysis is None:
            raise StageExecutionError("audio analysis is required for candidate analysis")
        if self._job_cancelled():
            raise CandidateAnalysisCancelled("candidate analysis cancelled before start")

        input_fingerprint = self.input_fingerprint(source)
        started = monotonic()
        corpus = self._historical_corpus(source.id)
        reuse = self._reuse_map(source.id)
        provider = self._effective_provider()
        service = CandidateAnalysisService(
            config=self._config,
            provider=provider,
            mode=self._mode,
            is_cancelled=self._job_cancelled,
        )
        try:
            outcome = service.analyze(
                source_id=str(source.id),
                segments=transcript.segments,
                duration=transcript.duration,
                language=transcript.language,
                dialect_profile=(
                    transcript.dialect_profile.value if transcript.dialect_profile else None
                ),
                dialect_confidence=transcript.dialect_confidence,
                silence_intervals=analysis.silence_intervals,
                audio_features=analysis.features,
                rights_status=source.rights_status,
                media_origin=source.media_origin,
                provenance_metadata=source.provenance_metadata or {},
                historical_corpus=corpus,
                reuse=reuse,
            )
        finally:
            self._release_provider(provider)
        if self._job_cancelled():
            raise CandidateAnalysisCancelled("candidate analysis cancelled before persistence")
        return self._persist(source, transcript, outcome, input_fingerprint, monotonic() - started)

    def _effective_provider(self) -> SemanticProvider | None:
        if self._mode is SemanticProviderMode.DETERMINISTIC:
            return DeterministicSemanticProvider()
        if self._provider is None:
            return DeterministicSemanticProvider()
        if self._mode is SemanticProviderMode.LOCAL_ONLY:
            return _LeaseBoundSemanticProvider(self._provider, self._lease_factory)
        return self._provider

    def _release_provider(self, provider: SemanticProvider | None) -> None:
        if provider is None:
            return
        try:
            provider.release()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # persistence

    def _persist(
        self,
        source: SourceVideo,
        transcript: Transcript,
        outcome: CandidateAnalysisOutcome,
        input_fingerprint: str,
        processing_duration: float,
    ) -> StageExecutionResult:
        if self._job_cancelled():
            raise CandidateAnalysisCancelled("candidate analysis cancelled before commit")
        analysis = self._session.scalar(
            select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
        )
        if analysis is None:
            analysis = CandidateAnalysis(source_video_id=source.id)
            self._session.add(analysis)
        analysis.input_fingerprint = input_fingerprint
        analysis.output_fingerprint = outcome.output_fingerprint
        analysis.policy_version = POLICY_VERSION
        analysis.scoring_version = SCORING_VERSION
        analysis.semantic_provider_mode = self._mode
        analysis.provider_identity = dict(self._provider_identity())
        analysis.provider_status = outcome.provider_status
        analysis.cache_eligible = outcome.cache_eligible
        analysis.metrics = dict(outcome.metrics)
        analysis.processing_duration = processing_duration
        self._session.flush()

        existing = {
            row.candidate_key: row
            for row in self._session.scalars(
                select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
            )
        }
        emitted: set[str] = set()
        for draft in outcome.candidates:
            emitted.add(draft.candidate_key)
            row = existing.get(draft.candidate_key)
            if row is None:
                row = ClipCandidate(source_video_id=source.id, candidate_key=draft.candidate_key)
                self._session.add(row)
            self._apply_draft(row, draft, analysis, input_fingerprint)
        for key, row in existing.items():
            if key not in emitted:
                row.is_current = False
        self._session.commit()
        self._session.refresh(analysis)
        return StageExecutionResult(outcome.output_fingerprint, analysis)

    def _apply_draft(
        self,
        row: ClipCandidate,
        draft: CandidateDraft,
        analysis: CandidateAnalysis,
        input_fingerprint: str,
    ) -> None:
        proposal = draft.proposal
        scores = draft.scores
        row.candidate_analysis_id = analysis.id
        row.is_current = True
        row.disposition = draft.disposition
        row.start_time = proposal.start_time
        row.end_time = proposal.end_time
        row.start_segment_index = proposal.start_segment_index
        row.end_segment_index = proposal.end_segment_index
        row.segment_indexes = list(proposal.segment_indexes)
        row.transcript_excerpt = draft.transcript_excerpt
        row.evidence_snapshot = dict(draft.evidence_snapshot)
        row.primary_content_type = draft.content.primary
        row.secondary_content_types = [item.value for item in draft.content.secondary]
        row.clip_score = scores.clip_score
        row.short_form_score = scores.short_form_score
        row.moment_density_score = scores.moment_density_score
        row.boredom_risk_score = scores.boredom_risk_score
        row.ending_quality_score = scores.ending_quality_score
        row.loopability_score = scores.loopability_score
        row.engagement_confidence = scores.engagement_confidence
        row.transcript_confidence = scores.transcript_confidence
        row.audio_confidence = scores.audio_confidence
        row.boundary_confidence = scores.boundary_confidence
        row.uncertainty_severity = scores.uncertainty_severity
        row.idea_novelty_score = scores.idea_novelty_score
        row.topic_novelty_score = scores.topic_novelty_score
        row.recent_semantic_similarity_risk = scores.recent_semantic_similarity_risk
        row.refinement_reasons = [reason.value for reason in draft.refinement_reasons]
        row.refinement_evidence = dict(draft.refinement_evidence)
        row.provenance_snapshot = dict(draft.provenance_snapshot)
        row.rights_risk = draft.rights_risk
        row.originality_risk = draft.originality_risk
        row.dialect_profile = draft.dialect_profile
        row.dialect_confidence = draft.dialect_confidence
        row.code_switch_suspected = draft.code_switch_suspected
        row.hooks = [hook.as_dict() for hook in draft.hooks]
        row.idea_summary = draft.idea_summary
        row.topic_summary = draft.topic_summary
        row.idea_signature = idea_signature(draft.idea_summary or draft.transcript_excerpt)
        row.topic_signature = topic_signature(draft.topic_summary or draft.transcript_excerpt)
        row.provider_input_fingerprint = draft.provider_input_fingerprint
        row.provider_evidence = dict(draft.provider_evidence)
        row.analysis_fingerprint = canonical_fingerprint(
            "candidate-analysis-record",
            POLICY_VERSION,
            {
                "input_fingerprint": input_fingerprint,
                "candidate_key": draft.candidate_key,
                "provider_input_fingerprint": draft.provider_input_fingerprint,
                "scoring_version": SCORING_VERSION,
                "prompt_hash": semantic_prompt_hash(),
            },
        )
        row.policy_version = POLICY_VERSION

    # ------------------------------------------------------------------
    # corpus / reuse

    def _historical_corpus(self, source_id: object) -> list[NoveltyItem]:
        rows = self._session.scalars(
            select(ClipCandidate)
            .where(
                ClipCandidate.is_current.is_(True),
                ClipCandidate.source_video_id != source_id,
                ClipCandidate.disposition.in_(_RETAINED_DISPOSITIONS),
            )
            .order_by(ClipCandidate.updated_at.desc())
            .limit(self._config.novelty_corpus_limit)
        )
        return [
            NoveltyItem(
                key=row.candidate_key,
                source_id=str(row.source_video_id),
                idea_text=row.idea_summary or row.transcript_excerpt,
                topic_text=row.topic_summary or row.transcript_excerpt,
                clip_score=row.clip_score,
            )
            for row in rows
        ]

    def _reuse_map(self, source_id: object) -> dict[str, tuple[str, SemanticEvaluationResult]]:
        rows = self._session.scalars(
            select(ClipCandidate).where(
                ClipCandidate.source_video_id == source_id,
                ClipCandidate.is_current.is_(True),
            )
        )
        reuse: dict[str, tuple[str, SemanticEvaluationResult]] = {}
        for row in rows:
            if not row.provider_input_fingerprint:
                continue
            result = semantic_result_from_evidence(row.provider_evidence or {}, row.candidate_key)
            if result is None:
                continue
            reuse[row.candidate_key] = (row.provider_input_fingerprint, result)
        return reuse
