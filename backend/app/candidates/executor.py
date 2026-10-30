"""Durable Stage 3 candidate-analysis executor."""

from __future__ import annotations

from collections.abc import Sequence
from time import monotonic

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.candidates.fingerprints import novelty_corpus_digest
from app.candidates.novelty import NoveltyItem
from app.candidates.persistence import apply_candidate_fields
from app.candidates.policy import (
    DEFAULT_CONFIG,
    POLICY_VERSION,
    SCORING_VERSION,
    Stage3Config,
)
from app.candidates.providers import (
    DeterministicSemanticProvider,
    ProviderErrorCategory,
    SemanticEvaluationRequest,
    SemanticEvaluationResult,
    SemanticProvider,
    SemanticProviderError,
    semantic_prompt_hash,
)
from app.candidates.registration import (
    operator_draft_matches_window,
    operator_identity_is_current,
    operator_selection_marker,
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
    AudioArtifact,
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


class _AdmissionBoundSemanticProvider:
    """Route Stage 3 hosted semantic evaluation through the shared MEDIUM gate."""

    def __init__(self, inner: SemanticProvider, admission: object) -> None:
        self._inner = inner
        self._admission = admission
        self.provider_name = getattr(inner, "provider_name", "gemini")
        self.model = getattr(inner, "model", None)
        self._released = False

    def evaluate(
        self, requests: Sequence[SemanticEvaluationRequest]
    ) -> dict[str, SemanticEvaluationResult]:
        from app.core.enums import AdmissionPriority

        try:
            decision = self._admission.acquire(AdmissionPriority.MEDIUM)  # type: ignore[attr-defined]
        except Exception as error:
            raise SemanticProviderError(
                ProviderErrorCategory.PROVIDER_ERROR, "admission gate unavailable"
            ) from error
        if not getattr(decision, "admitted", False):
            raise SemanticProviderError(ProviderErrorCategory.RATE_LIMITED, "admission denied")
        return dict(self._inner.evaluate(requests))

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
        admission: object | None = None,
    ) -> None:
        self._session = session
        self._config = config
        self._provider = provider
        self._mode = mode
        self._lease_factory = lease_factory or NoopHeavyModelLeaseFactory()
        self._admission = admission
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
            semantic_provider_mode=self._mode.value,
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
        if self._admission is not None:
            return _AdmissionBoundSemanticProvider(self._provider, self._admission)
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
        locked_source = self._session.scalar(
            select(SourceVideo).where(SourceVideo.id == source.id).with_for_update()
        )
        if locked_source is None:
            raise StageExecutionError("source disappeared before candidate persistence")
        source = locked_source
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
        audio = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        operator_rows = {
            key: row for key, row in existing.items() if operator_selection_marker(row) is not None
        }
        operator_preserved = 0
        operator_staled = 0
        for row in operator_rows.values():
            if operator_identity_is_current(row, source, transcript, audio):
                if row.is_current:
                    operator_preserved += 1
                continue
            if row.is_current:
                row.is_current = False
                operator_staled += 1

        emitted: set[str] = set()
        applied = 0
        suppressed_matched = 0
        suppressed_collided = 0
        for draft in outcome.candidates:
            emitted.add(draft.candidate_key)
            row = existing.get(draft.candidate_key)
            if row is not None and operator_selection_marker(row) is not None:
                if operator_draft_matches_window(row, draft):
                    suppressed_matched += 1
                else:
                    suppressed_collided += 1
                continue
            if row is None:
                row = ClipCandidate(source_video_id=source.id, candidate_key=draft.candidate_key)
                self._session.add(row)
            self._apply_draft(row, draft, analysis, input_fingerprint)
            applied += 1
        for key, row in existing.items():
            if key in emitted:
                continue
            if operator_selection_marker(row) is not None:
                continue
            row.is_current = False

        analysis.metrics = {
            **dict(outcome.metrics),
            "operator_rows_preserved": operator_preserved,
            "operator_rows_staled_raw_identity": operator_staled,
            "operator_same_key_drafts_matched": suppressed_matched,
            "operator_same_key_drafts_collided": suppressed_collided,
            "automatic_drafts_suppressed": suppressed_matched + suppressed_collided,
            "automatic_candidates_applied": applied,
            "persistence_scope": "automatic_drafts_excluding_operator_owned",
        }
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
        row.candidate_analysis_id = analysis.id
        row.is_current = True
        apply_candidate_fields(row, draft)
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
