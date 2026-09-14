"""Durable Stage 4.1 candidate-scoped planning executor.

Claims one plan set, resolves the current Stage 4.0 handoff and selected Stage
3.5 refinement, runs the bounded deterministic+provider planning service, and
persists a truthful result. Observes the exact executing job for cooperative
cancellation, routes hosted work through the shared HIGH admission gate, binds
local inference to the shared heavy-model lease, and releases owned providers on
every exit path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import monotonic

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import (
    AdmissionPriority,
    JobStatus,
    PlanExecutionStatus,
    SemanticProviderMode,
)
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    ProcessingJob,
    TransformationEligibilityAnalysis,
    TransformationPlan,
    TransformationPlanSet,
)
from app.pipeline.executor import StageCancelled, StageExecutionResult
from app.runtime.heavy_model_lease import (
    HeavyModelLeaseBusy,
    HeavyModelLeaseFactory,
    NoopHeavyModelLeaseFactory,
)
from app.transformation.inputs import resolve_effective_refinement
from app.transformation.planning.fingerprints import (
    build_plan_set_input_payload,
    planning_input_fingerprint,
)
from app.transformation.planning.inputs import (
    PlanningInputError,
    build_planning_inputs,
)
from app.transformation.planning.policy import (
    DEFAULT_CONFIG,
    POLICY_VERSION,
    SCHEMA_VERSION,
    VALIDATION_VERSION,
    Stage41Config,
)
from app.transformation.planning.providers import (
    DeterministicPlanningProvider,
    PlanningProvider,
    PlanningProviderError,
)
from app.transformation.planning.service import PlanningService
from app.transformation.planning.types import PlanningInputs, PlanningOutcome, ValidatedPlan

_READY_STATUSES = {
    PlanExecutionStatus.COMPLETE,
    PlanExecutionStatus.PROVIDER_DEGRADED,
}
_ACTIVE_JOB_STATUSES = {JobStatus.QUEUED, JobStatus.RUNNING}
_DEGRADED_PROVIDER_STATUSES = {"PROVIDER_DEGRADED", "RATE_LIMITED"}


class PlanningCancelled(StageCancelled):
    """Cooperative cancellation while Stage 4.1 planning was running."""


class _LeaseBoundPlanningProvider:
    """Acquire the shared heavy-model lease lazily around real local inference."""

    def __init__(self, inner: PlanningProvider, lease_factory: object) -> None:
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

    def plan(self, requests: Sequence[object], tier: str = "ROUTINE") -> dict[str, object]:
        lease = self._enter()
        result = self._inner.plan(requests, tier)  # type: ignore[arg-type]
        if getattr(lease, "ownership_lost", False):
            raise HeavyModelLeaseBusy("heavy-model lease was lost during Stage 4.1 planning; retry")
        return dict(result)

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


class _AdmissionBoundPlanningProvider:
    """Route Stage 4.1 hosted planning through the shared HIGH gate."""

    def __init__(self, inner: PlanningProvider, admission: object) -> None:
        self._inner = inner
        self._admission = admission
        self.provider_name = getattr(inner, "provider_name", "gemini")
        self.model = getattr(inner, "model", None)
        self._released = False

    def plan(self, requests: Sequence[object], tier: str = "ROUTINE") -> dict[str, object]:
        try:
            decision = self._admission.acquire(AdmissionPriority.HIGH)  # type: ignore[attr-defined]
        except Exception as error:
            raise PlanningProviderError("PROVIDER_ERROR", "admission gate unavailable") from error
        if not getattr(decision, "admitted", False):
            raise PlanningProviderError("RATE_LIMITED", "admission denied")
        return dict(self._inner.plan(requests, tier))  # type: ignore[arg-type]

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


class TransformationPlanningExecutor:
    """Execute one candidate-scoped Stage 4.1 planning run and persist it."""

    def __init__(
        self,
        *,
        session: Session,
        settings: object,
        config: Stage41Config = DEFAULT_CONFIG,
        provider: PlanningProvider | None = None,
        provider_identity: Mapping[str, object] | None = None,
        mode: SemanticProviderMode = SemanticProviderMode.DETERMINISTIC,
        lease_factory: HeavyModelLeaseFactory | NoopHeavyModelLeaseFactory | None = None,
        admission: object | None = None,
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

    def _provider_identity(self) -> dict[str, object]:
        if self._configured_identity is not None:
            return dict(self._configured_identity)
        if self._provider is None or self._mode is SemanticProviderMode.DETERMINISTIC:
            return DeterministicPlanningProvider().runtime_identity()
        return dict(self._provider.runtime_identity())

    def _effective_provider(self) -> PlanningProvider | None:
        if self._mode is SemanticProviderMode.DETERMINISTIC:
            return None
        if self._provider is None:
            return None
        if self._mode is SemanticProviderMode.LOCAL_ONLY:
            return _LeaseBoundPlanningProvider(self._provider, self._lease_factory)  # type: ignore[return-value]
        if self._admission is not None:
            return _AdmissionBoundPlanningProvider(self._provider, self._admission)  # type: ignore[return-value]
        return self._provider

    def _candidate_and_analysis(
        self, plan_set: TransformationPlanSet
    ) -> tuple[ClipCandidate, TransformationEligibilityAnalysis]:
        candidate = self._session.get(ClipCandidate, plan_set.clip_candidate_id)
        analysis = self._session.get(
            TransformationEligibilityAnalysis, plan_set.transformation_analysis_id
        )
        if candidate is None or analysis is None:
            raise PlanningInputError("candidate or Stage 4.0 analysis is missing")
        return candidate, analysis

    def _resolve(
        self, plan_set: TransformationPlanSet
    ) -> tuple[
        ClipCandidate, TransformationEligibilityAnalysis, CandidateRefinement, PlanningInputs, str
    ]:
        candidate, analysis = self._candidate_and_analysis(plan_set)
        refinement = resolve_effective_refinement(self._session, candidate)
        if refinement is None:
            raise PlanningInputError("no usable Stage 3.5 refinement is available")
        settings = self._settings
        inputs = build_planning_inputs(
            self._session, candidate, analysis, refinement, settings, self._config
        )
        fingerprint = self._input_fingerprint(inputs)
        return candidate, analysis, refinement, inputs, fingerprint

    def _input_fingerprint(self, inputs: PlanningInputs) -> str:
        payload = build_plan_set_input_payload(
            inputs=inputs,
            config=self._config,
            provider_mode=self._mode.value,
            provider_identity=self._provider_identity(),
        )
        return planning_input_fingerprint(payload)

    def input_fingerprint(self, plan_set: TransformationPlanSet) -> str:
        candidate, analysis = self._candidate_and_analysis(plan_set)
        refinement = resolve_effective_refinement(self._session, candidate)
        if refinement is None:
            return ""
        inputs = build_planning_inputs(
            self._session, candidate, analysis, refinement, self._settings, self._config
        )
        return self._input_fingerprint(inputs)

    def is_cache_hit(
        self,
        plan_set: TransformationPlanSet,
        *,
        force: bool = False,
    ) -> bool:
        if force or not plan_set.cache_eligible:
            return False
        if plan_set.execution_status not in _READY_STATUSES:
            return False
        if not plan_set.input_fingerprint:
            return False
        try:
            current = self.input_fingerprint(plan_set)
        except Exception:
            return False
        return bool(current) and current == plan_set.input_fingerprint

    def _claimed_by_other_job(self, plan_set: TransformationPlanSet) -> bool:
        owner = plan_set.active_job_id
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

    def _claim(self, plan_set: TransformationPlanSet) -> None:
        plan_set.active_job_id = str(self._active_job_id) if self._active_job_id else None
        plan_set.execution_status = PlanExecutionStatus.PLANNING
        self._session.commit()

    def _mark_cancelled(self, plan_set: TransformationPlanSet) -> None:
        plan_set.execution_status = PlanExecutionStatus.CANCELLED
        plan_set.active_job_id = None
        self._session.commit()

    def _mark_failed(self, plan_set: TransformationPlanSet) -> None:
        if self._job_cancelled():
            plan_set.execution_status = PlanExecutionStatus.CANCELLED
        else:
            plan_set.execution_status = PlanExecutionStatus.FAILED
        plan_set.active_job_id = None
        self._session.commit()

    def execute(self, plan_set_id: object, *, force: bool = False) -> StageExecutionResult:
        plan_set = self._session.get(TransformationPlanSet, plan_set_id)
        if plan_set is None:
            raise PlanningInputError("transformation plan set is missing")
        if self._job_cancelled():
            self._mark_cancelled(plan_set)
            raise PlanningCancelled("Stage 4.1 planning cancelled before start")
        try:
            _candidate, _analysis, refinement, inputs, fingerprint = self._resolve(plan_set)
        except PlanningInputError:
            self._mark_failed(plan_set)
            raise
        if self.is_cache_hit(plan_set, force=force):
            self._release_owned_providers()
            return StageExecutionResult(plan_set.output_fingerprint, plan_set)
        if self._claimed_by_other_job(plan_set):
            self._release_owned_providers()
            return StageExecutionResult(plan_set.output_fingerprint, plan_set)
        self._claim(plan_set)

        checkpoints = self._checkpoints(plan_set)
        started = monotonic()
        service = PlanningService(
            config=self._config,
            provider=self._effective_provider(),
            provider_identity=self._provider_identity(),
            mode=self._mode,
            is_cancelled=self._job_cancelled,
        )
        try:
            outcome = service.plan(
                inputs,
                input_fingerprint=fingerprint,
                checkpoints=checkpoints,
            )
        except StageCancelled:
            self._mark_cancelled(plan_set)
            raise
        except Exception:
            self._mark_failed(plan_set)
            raise
        finally:
            self._release_owned_providers()
        if self._job_cancelled():
            self._mark_cancelled(plan_set)
            raise PlanningCancelled("Stage 4.1 planning cancelled before persistence")
        return self._persist(plan_set, refinement, inputs, outcome, monotonic() - started)

    def _checkpoints(self, plan_set: TransformationPlanSet) -> dict[str, dict[str, object]]:
        checkpoints: dict[str, dict[str, object]] = {}
        for attempt in plan_set.strategy_attempts or []:
            if not isinstance(attempt, dict):
                continue
            checkpoint = attempt.get("checkpoint")
            if not isinstance(checkpoint, dict):
                continue
            strategy_id = str(attempt.get("strategy_id", ""))
            if strategy_id:
                checkpoints[strategy_id] = dict(checkpoint)
        return checkpoints

    def _persist(
        self,
        plan_set: TransformationPlanSet,
        refinement: CandidateRefinement,
        inputs: PlanningInputs,
        outcome: PlanningOutcome,
        processing_duration: float,
    ) -> StageExecutionResult:
        plan_set.refinement_id = refinement.id
        plan_set.refinement_priority = refinement.priority.value
        plan_set.refinement_quality_level = refinement.quality_level
        plan_set.execution_status = (
            PlanExecutionStatus.PROVIDER_DEGRADED
            if outcome.execution_status == "PROVIDER_DEGRADED"
            else PlanExecutionStatus.COMPLETE
        )
        plan_set.planning_outcome = outcome.semantic_outcome
        plan_set.outcome_reasons = list(outcome.outcome_reasons)
        plan_set.stage40_snapshot = {
            "analysis_id": inputs.stage40_analysis_id,
            "input_fingerprint": inputs.stage40_input_fingerprint,
            "output_fingerprint": inputs.stage40_output_fingerprint,
            "policy_version": inputs.stage40_policy_version,
            "assessments": dict(inputs.stage40_assessments),
            "source_moment": dict(inputs.source_moment),
        }
        plan_set.target_context = inputs.planning_context.as_dict()
        plan_set.provider_mode = self._mode
        plan_set.provider_identity = dict(outcome.provider_identity)
        plan_set.provider_status = outcome.provider_status
        plan_set.provider_evidence = dict(outcome.provider_evidence)
        plan_set.strategy_attempts = [attempt.as_dict() for attempt in outcome.attempts]
        plan_set.input_fingerprint = outcome.input_fingerprint
        plan_set.output_fingerprint = outcome.output_fingerprint
        plan_set.policy_version = POLICY_VERSION
        plan_set.schema_version = SCHEMA_VERSION
        plan_set.validation_version = VALIDATION_VERSION
        plan_set.cache_eligible = outcome.cache_eligible
        plan_set.metrics = dict(outcome.metrics)
        plan_set.processing_duration = processing_duration
        plan_set.active_job_id = None
        self._session.flush()

        self._persist_plans(plan_set, inputs, outcome.plans)
        self._session.commit()
        self._session.refresh(plan_set)
        return StageExecutionResult(outcome.output_fingerprint, plan_set)

    def _persist_plans(
        self,
        plan_set: TransformationPlanSet,
        inputs: PlanningInputs,
        plans: Sequence[ValidatedPlan],
    ) -> None:
        strategies = {str(item.get("id")): item for item in inputs.stage40_strategies}
        existing = {
            row.strategy_candidate_id: row
            for row in self._session.scalars(
                select(TransformationPlan).where(TransformationPlan.plan_set_id == plan_set.id)
            )
        }
        emitted: set[object] = set()
        for plan in plans:
            strategy = strategies.get(plan.strategy_id)
            if strategy is None:
                continue
            import uuid as _uuid

            strategy_id = _uuid.UUID(plan.strategy_id)
            emitted.add(strategy_id)
            row = existing.get(strategy_id)
            if row is None:
                row = TransformationPlan(plan_set_id=plan_set.id, strategy_candidate_id=strategy_id)
                self._session.add(row)
            row.plan_key = plan.plan_key
            row.is_current = True
            row.status = plan.status
            row.generation_rank = plan.generation_rank
            row.strategy_type = plan.strategy_type
            row.intensity = plan.intensity
            row.strategy_fingerprint = plan.strategy_fingerprint
            row.strategy_snapshot = dict(strategy)
            row.source_dialect = dict(plan.source_dialect)
            row.target_audience = dict(plan.target_intent)
            row.blocks = [block.as_dict() for block in plan.blocks]
            row.hero_block_index = plan.hero_block_index
            row.hero_source_start = plan.hero_source_start
            row.hero_source_end = plan.hero_source_end
            row.hero_appearance_time = plan.hero_appearance_time
            row.preservation_constraints = list(plan.preservation_constraints)
            row.original_value_kinds = [item.value for item in plan.original_value_kinds]
            row.original_value_reasons = list(plan.original_value_reasons)
            row.narration_need = plan.narration.need.value
            row.narration_requirements = plan.narration.as_dict()
            row.external_fact_dependencies = [
                dict(item) for item in plan.external_fact_dependencies
            ]
            row.required_context = list(plan.required_context)
            row.derived_durations = dict(plan.derived_durations)
            row.hook_payoff_evidence = dict(plan.hook_payoff_evidence)
            row.degraded_rules = list(plan.degraded_rules)
            row.stage40_risk = dict(plan.stage40_risk)
            row.planner_confidence = plan.planner_confidence
            row.generation_origin = plan.generation_origin
            row.planning_provider_evidence = dict(plan.provider_evidence)
            row.provider_input_fingerprint = plan.provider_input_fingerprint
            row.plan_output_fingerprint = plan.plan_output_fingerprint
            row.structure_signature = plan.structure_signature
            row.policy_version = POLICY_VERSION
        for strategy_id, row in existing.items():
            if strategy_id not in emitted:
                row.is_current = False

    def _release_owned_providers(self) -> None:
        provider = self._provider
        release = getattr(provider, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass


def build_transformation_planning_executor(
    session: Session,
    settings: object,
) -> "TransformationPlanningExecutor":
    """Build the production planning executor from settings, lazily and network-free."""

    identity_factory = getattr(settings, "transformation_planning_provider_identity", None)
    provider_identity = identity_factory() if callable(identity_factory) else None
    return TransformationPlanningExecutor(
        session=session,
        settings=settings,
        config=settings.stage41_config(),  # type: ignore[attr-defined]
        provider=settings.transformation_planning_provider(),  # type: ignore[attr-defined]
        provider_identity=provider_identity,
        mode=settings.transformation_planning_semantic_mode(),  # type: ignore[attr-defined]
        lease_factory=settings.heavy_model_lease_factory(),  # type: ignore[attr-defined]
        admission=settings.gemini_admission_controller(),  # type: ignore[attr-defined]
    )
