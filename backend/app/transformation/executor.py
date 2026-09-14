"""Durable Stage 4.0 candidate-scoped eligibility executor.

Claims one analysis row, resolves the selected Stage 3.5 refinement, runs the
bounded deterministic+provider service, and persists a truthful result. It
observes the exact executing job for cooperative cancellation, routes hosted
work through the shared HIGH admission gate, binds local inference to the shared
heavy-model lease, and releases owned providers on every exit path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import monotonic

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import (
    AdmissionPriority,
    JobStatus,
    SemanticProviderMode,
    TransformationExecutionStatus,
)
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    ProcessingJob,
    TransformationEligibilityAnalysis,
    TransformationStrategyCandidate,
)
from app.pipeline.executor import StageCancelled, StageExecutionResult
from app.runtime.heavy_model_lease import (
    HeavyModelLeaseBusy,
    HeavyModelLeaseFactory,
    NoopHeavyModelLeaseFactory,
)
from app.transformation.fingerprints import (
    build_input_fingerprint_payload,
    transformation_input_fingerprint,
)
from app.transformation.inputs import (
    TransformationInputError,
    build_transformation_inputs,
    resolve_effective_refinement,
)
from app.transformation.policy import (
    DEFAULT_CONFIG,
    POLICY_VERSION,
    SCHEMA_VERSION,
    VALIDATION_VERSION,
    Stage40Config,
)
from app.transformation.providers import (
    DeterministicTransformationProvider,
    TransformationProvider,
    TransformationProviderError,
    deserialize_provider_result,
)
from app.transformation.service import (
    TransformationEligibilityService,
    build_strategy_fingerprint,
    compute_provider_route,
)
from app.transformation.types import (
    StrategyDraft,
    TransformationInputs,
    TransformationOutcome,
    TransformationProviderResult,
)

_READY_STATUSES = {
    TransformationExecutionStatus.COMPLETE,
    TransformationExecutionStatus.PROVIDER_DEGRADED,
}
_ACTIVE_JOB_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}
_DEGRADED_STATUSES = {"PROVIDER_DEGRADED", "RATE_LIMITED"}


class TransformationCancelled(StageCancelled):
    """Cooperative cancellation while Stage 4.0 analysis was running."""


class _LeaseBoundTransformationProvider:
    """Acquire the shared heavy-model lease lazily around real local inference."""

    def __init__(self, inner: TransformationProvider, lease_factory: object) -> None:
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

    def discover(self, requests: Sequence[object]) -> dict[str, TransformationProviderResult]:
        lease = self._enter()
        result = self._inner.discover(requests)  # type: ignore[arg-type]
        if getattr(lease, "ownership_lost", False):
            raise HeavyModelLeaseBusy("heavy-model lease was lost during Stage 4.0 analysis; retry")
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


class _AdmissionBoundTransformationProvider:
    """Route Stage 4.0 hosted discovery through the shared HIGH gate."""

    def __init__(self, inner: TransformationProvider, admission: object) -> None:
        self._inner = inner
        self._admission = admission
        self.provider_name = getattr(inner, "provider_name", "gemini")
        self.model = getattr(inner, "model", None)
        self._released = False

    def discover(self, requests: Sequence[object]) -> dict[str, TransformationProviderResult]:
        try:
            decision = self._admission.acquire(AdmissionPriority.HIGH)  # type: ignore[attr-defined]
        except Exception as error:
            raise TransformationProviderError("PROVIDER_ERROR", "admission gate unavailable") from (
                error
            )
        if not getattr(decision, "admitted", False):
            raise TransformationProviderError("RATE_LIMITED", "admission denied")
        return dict(self._inner.discover(requests))  # type: ignore[arg-type]

    def select_tier(self, requests: Sequence[object]) -> str:
        select = getattr(self._inner, "select_tier", None)
        return str(select(requests)) if callable(select) else ""

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


class TransformationEligibilityExecutor:
    """Execute one candidate-scoped Stage 4.0 analysis and persist it."""

    def __init__(
        self,
        *,
        session: Session,
        config: Stage40Config = DEFAULT_CONFIG,
        provider: TransformationProvider | None = None,
        provider_identity: Mapping[str, object] | None = None,
        mode: SemanticProviderMode = SemanticProviderMode.DETERMINISTIC,
        lease_factory: HeavyModelLeaseFactory | NoopHeavyModelLeaseFactory | None = None,
        admission: object | None = None,
    ) -> None:
        self._session = session
        self._config = config
        self._provider = provider
        # Stable configured identity, independent of transient availability.
        self._configured_identity = (
            dict(provider_identity) if provider_identity is not None else None
        )
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

    # ---- inputs -----------------------------------------------------------

    def _candidate(self, analysis: TransformationEligibilityAnalysis) -> ClipCandidate:
        candidate = self._session.get(ClipCandidate, analysis.clip_candidate_id)
        if candidate is None:
            raise TransformationInputError("candidate is missing for analysis")
        return candidate

    def _resolve(
        self, candidate: ClipCandidate
    ) -> tuple[CandidateRefinement, TransformationInputs, str]:
        refinement = resolve_effective_refinement(self._session, candidate)
        if refinement is None:
            raise TransformationInputError(
                "no usable Stage 3.5 refinement is available for this candidate"
            )
        inputs = build_transformation_inputs(self._session, candidate, refinement, self._config)
        structure = self._structure_for_route(inputs, self._config)
        necessity = self._necessity(inputs)
        route = compute_provider_route(inputs, structure, necessity)
        fingerprint = self._input_fingerprint(inputs, route)
        return refinement, inputs, fingerprint

    def _structure_for_route(self, inputs: TransformationInputs, config: Stage40Config):  # type: ignore[no-untyped-def]
        from app.transformation.eligibility import derive_source_moment

        return derive_source_moment(inputs, config)

    def _provider_identity(self) -> dict[str, object]:
        provider = self._effective_provider_identity()
        return provider

    def _effective_provider_identity(self) -> dict[str, object]:
        # Configured identity wins even when the provider is temporarily
        # unavailable (missing key/outage), so accepted analysis is not
        # invalidated by transient availability. Real model/prompt/schema
        # changes still alter this identity and invalidate correctly.
        if self._configured_identity is not None:
            return dict(self._configured_identity)
        if self._provider is None or self._mode is SemanticProviderMode.DETERMINISTIC:
            return DeterministicTransformationProvider().runtime_identity()
        return dict(self._provider.runtime_identity())

    def _input_fingerprint(self, inputs: TransformationInputs, route: str | None) -> str:
        payload = build_input_fingerprint_payload(
            inputs=inputs,
            config=self._config,
            provider_identity=self._provider_identity(),
            provider_mode=self._mode.value,
            provider_route=route,
        )
        return transformation_input_fingerprint(payload)

    def input_fingerprint(self, candidate: ClipCandidate) -> str:
        refinement = resolve_effective_refinement(self._session, candidate)
        if refinement is None:
            return ""
        inputs = build_transformation_inputs(self._session, candidate, refinement, self._config)
        structure = self._structure_for_route(inputs, self._config)
        route = compute_provider_route(inputs, structure, self._necessity(inputs))
        return self._input_fingerprint(inputs, route)

    def _necessity(self, inputs: TransformationInputs) -> float:
        from app.transformation.eligibility import transformation_necessity

        return transformation_necessity(inputs)

    # ---- service / provider ----------------------------------------------

    def _service(self) -> TransformationEligibilityService:
        return TransformationEligibilityService(
            config=self._config,
            provider=self._effective_provider(),
            provider_identity=self._provider_identity(),
            mode=self._mode,
            is_cancelled=self._job_cancelled,
        )

    def _effective_provider(self) -> TransformationProvider | None:
        if self._mode is SemanticProviderMode.DETERMINISTIC:
            return None
        if self._provider is None:
            return None
        if self._mode is SemanticProviderMode.LOCAL_ONLY:
            return _LeaseBoundTransformationProvider(self._provider, self._lease_factory)
        if self._admission is not None:
            return _AdmissionBoundTransformationProvider(self._provider, self._admission)
        return self._provider

    # ---- cache / claim ----------------------------------------------------

    def is_cache_hit(
        self,
        analysis: TransformationEligibilityAnalysis,
        candidate: ClipCandidate,
        *,
        force: bool = False,
    ) -> bool:
        if force or not analysis.cache_eligible:
            return False
        if analysis.execution_status not in _READY_STATUSES:
            return False
        if not analysis.input_fingerprint:
            return False
        try:
            current = self.input_fingerprint(candidate)
        except Exception:
            return False
        return bool(current) and current == analysis.input_fingerprint

    def _claimed_by_other_job(self, analysis: TransformationEligibilityAnalysis) -> bool:
        owner = analysis.active_job_id
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

    def _claim(self, analysis: TransformationEligibilityAnalysis) -> None:
        analysis.active_job_id = str(self._active_job_id) if self._active_job_id else None
        analysis.execution_status = TransformationExecutionStatus.ANALYZING
        self._session.commit()

    def _mark_cancelled(self, analysis: TransformationEligibilityAnalysis) -> None:
        analysis.execution_status = TransformationExecutionStatus.CANCELLED
        analysis.active_job_id = None
        self._session.commit()

    def _mark_failed(self, analysis: TransformationEligibilityAnalysis) -> None:
        if self._job_cancelled():
            analysis.execution_status = TransformationExecutionStatus.CANCELLED
        else:
            analysis.execution_status = TransformationExecutionStatus.FAILED
        analysis.active_job_id = None
        self._session.commit()

    # ---- execute / persist ------------------------------------------------

    def execute(self, analysis_id: object, *, force: bool = False) -> StageExecutionResult:
        analysis = self._session.get(TransformationEligibilityAnalysis, analysis_id)
        if analysis is None:
            raise TransformationInputError("transformation analysis is missing")
        candidate = self._candidate(analysis)
        if self._job_cancelled():
            self._mark_cancelled(analysis)
            raise TransformationCancelled("Stage 4.0 analysis cancelled before start")
        try:
            refinement, inputs, fingerprint = self._resolve(candidate)
        except TransformationInputError:
            self._mark_failed(analysis)
            raise
        if self.is_cache_hit(analysis, candidate, force=force):
            return StageExecutionResult(analysis.output_fingerprint, analysis)
        if self._claimed_by_other_job(analysis):
            return StageExecutionResult(analysis.output_fingerprint, analysis)
        self._claim(analysis)

        reuse = self._reuse_provider_result(analysis)
        started = monotonic()
        service = self._service()
        try:
            outcome = service.evaluate(
                inputs,
                input_fingerprint=fingerprint,
                reuse=reuse,
            )
        except StageCancelled:
            self._mark_cancelled(analysis)
            raise
        except Exception:
            self._mark_failed(analysis)
            raise
        finally:
            self._release_owned_providers()
        if self._job_cancelled():
            self._mark_cancelled(analysis)
            raise TransformationCancelled("Stage 4.0 analysis cancelled before persistence")
        return self._persist(analysis, refinement, outcome, monotonic() - started)

    def _reuse_provider_result(
        self, analysis: TransformationEligibilityAnalysis
    ) -> tuple[str, TransformationProviderResult] | None:
        fp = analysis.provider_input_fingerprint
        evidence = analysis.provider_evidence or {}
        if not fp or not evidence:
            return None
        raw = evidence.get("provider_result")
        result = deserialize_provider_result(raw, str(analysis.clip_candidate_id))
        if result is None:
            return None
        return fp, result

    def _persist(
        self,
        analysis: TransformationEligibilityAnalysis,
        refinement: CandidateRefinement,
        outcome: TransformationOutcome,
        processing_duration: float,
    ) -> StageExecutionResult:
        analysis.refinement_id = refinement.id
        analysis.refinement_priority = refinement.priority.value
        analysis.refinement_quality_level = refinement.quality_level
        analysis.execution_status = (
            TransformationExecutionStatus.PROVIDER_DEGRADED
            if outcome.provider_status in _DEGRADED_STATUSES
            else TransformationExecutionStatus.COMPLETE
        )
        analysis.eligibility_outcome = outcome.eligibility_outcome
        analysis.eligibility_reasons = list(outcome.eligibility_reasons)
        analysis.assessments = dict(outcome.assessments.as_dict())
        analysis.source_moment = dict(outcome.source_moment.as_dict())
        analysis.platform_risk = dict(outcome.platform_risk)
        analysis.transformation_intensity = outcome.intensity
        analysis.provider_mode = self._mode
        analysis.provider_identity = dict(self._provider_identity())
        analysis.provider_status = outcome.provider_status
        analysis.provider_evidence = dict(outcome.provider_evidence)
        analysis.provider_input_fingerprint = outcome.provider_input_fingerprint
        analysis.input_fingerprint = outcome.input_fingerprint
        analysis.output_fingerprint = outcome.output_fingerprint
        analysis.policy_version = POLICY_VERSION
        analysis.schema_version = SCHEMA_VERSION
        analysis.validation_version = VALIDATION_VERSION
        analysis.cache_eligible = outcome.cache_eligible
        analysis.metrics = dict(outcome.metrics)
        analysis.processing_duration = processing_duration
        analysis.active_job_id = None
        self._session.flush()

        self._persist_strategies(analysis, outcome.strategies)
        self._session.commit()
        self._session.refresh(analysis)
        return StageExecutionResult(outcome.output_fingerprint, analysis)

    def _persist_strategies(
        self,
        analysis: TransformationEligibilityAnalysis,
        strategies: Sequence[StrategyDraft],
    ) -> None:
        existing = {
            row.strategy_type: row
            for row in self._session.scalars(
                select(TransformationStrategyCandidate).where(
                    TransformationStrategyCandidate.analysis_id == analysis.id
                )
            )
        }
        emitted: set[object] = set()
        for draft in strategies:
            emitted.add(draft.strategy_type)
            row = existing.get(draft.strategy_type)
            if row is None:
                row = TransformationStrategyCandidate(
                    analysis_id=analysis.id,
                    strategy_type=draft.strategy_type,
                )
                self._session.add(row)
            row.strategy_key = f"{analysis.id}:{draft.strategy_type.value}"
            row.is_current = True
            row.disposition = draft.disposition
            row.rank = draft.rank
            row.intensity = draft.intensity
            row.direction_summary = draft.direction_summary
            row.added_value_focus = draft.added_value_focus
            row.substantive_value_kind = draft.substantive_value_kind
            row.source_moment_role = draft.source_moment_role
            row.preservation_requirements = list(draft.preservation_requirements)
            row.retention_preservation = draft.assessments.retention_preservation
            row.source_moment_damage_risk = draft.assessments.source_moment_damage_risk
            row.added_value_density = draft.assessments.added_value_density
            row.originality_potential = draft.assessments.originality_potential
            row.source_dominance_risk = draft.assessments.source_dominance_risk
            row.generic_filler_risk = draft.assessments.generic_filler_risk
            row.redundant_commentary_risk = draft.assessments.redundant_commentary_risk
            row.template_staleness_risk = draft.assessments.template_staleness_risk
            row.external_verification_requirement = draft.external_verification_requirement
            row.verification_requirements = list(draft.verification_requirements)
            row.rejection_reasons = list(draft.rejection_reasons)
            row.confidence = draft.confidence
            row.origin = draft.origin
            row.provider_evidence = dict(draft.provider_evidence)
            row.strategy_fingerprint = build_strategy_fingerprint(
                draft, analysis.output_fingerprint, POLICY_VERSION
            )
            row.policy_version = POLICY_VERSION
        for strategy_type, row in existing.items():
            if strategy_type not in emitted:
                row.is_current = False

    def _release_owned_providers(self) -> None:
        provider = self._provider
        release = getattr(provider, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass


def build_transformation_executor(
    session: Session,
    settings: object,
) -> "TransformationEligibilityExecutor":
    """Build the production executor from settings, lazily and without network."""

    identity_factory = getattr(settings, "transformation_provider_identity", None)
    provider_identity = identity_factory() if callable(identity_factory) else None
    return TransformationEligibilityExecutor(
        session=session,
        config=settings.stage40_config(),  # type: ignore[attr-defined]
        provider=settings.transformation_provider(),  # type: ignore[attr-defined]
        provider_identity=provider_identity,
        mode=settings.transformation_semantic_mode(),  # type: ignore[attr-defined]
        lease_factory=settings.heavy_model_lease_factory(),  # type: ignore[attr-defined]
        admission=settings.gemini_admission_controller(),  # type: ignore[attr-defined]
    )
