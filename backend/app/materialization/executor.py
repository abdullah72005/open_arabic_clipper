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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import cast

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.enums import (
    JobStatus,
    MaterializationLifecycle,
    MaterializationOutcome,
    MaterializationPriority,
)
from app.core.settings import Settings
from app.materialization.components import (
    ComponentContext,
    ComponentProviders,
    prepare_materialization,
)
from app.materialization.fingerprints import input_fingerprint
from app.materialization.inputs import MaterializationInputs
from app.materialization.policy import REASON_UPSTREAM_REVALIDATION_REQUIRED, Stage6Config
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
    finish_owner_job,
    get_or_create_envelope,
    guarded_update,
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
    #: Whether hosted providers are permitted for this bundle. This is an
    #: explicit configuration decision, never inferred from a critique object's
    #: mere presence.
    allow_hosted: bool = False
    #: Shared admit/deny callable (HIGH gate) consulted before any hosted call.
    admission: Callable[[], bool] | None = field(default=None)
    #: Notify the shared admission controller of a rate/quota failure.
    on_rate_limited: Callable[[], None] | None = field(default=None)
    #: Per-run narration synthesizer builder (needs the resolved source id). Used
    #: only when no narration provider is injected directly.
    narration_factory: Callable[[MaterializationInputs], object | None] | None = field(default=None)
    #: Optional local audio-backed fidelity validator applied to synthesized audio.
    narration_fidelity: Callable[[object], Mapping[str, object]] | None = field(default=None)


def _is_hosted_provider(provider: object) -> bool:
    return bool(getattr(provider, "provider_name", "deterministic") != "deterministic")


def build_materialization_providers(
    settings: Settings,
    *,
    writing: MaterializationWritingProvider | None = None,
    critique: MaterializationCritiqueProvider | None = None,
    narration: object | None = None,
    narration_fidelity: Callable[[object], Mapping[str, object]] | None = None,
    storage: StorageService | None = None,
) -> MaterializationProviders:
    """Build the production provider bundle from the configured materialization mode.

    Injected providers (tests/fixtures) are honored exactly and determine hosted
    permission from the providers themselves. When nothing is injected, an
    explicit ``adaptive`` mode with a configured key builds the hosted writing and
    critique adapters routed through the shared HIGH admission gate; a
    ``deterministic`` mode (or a missing key) builds the network-free deterministic
    writing provider and makes zero hosted calls.
    """

    from app.core.enums import SemanticProviderMode

    if writing is not None or critique is not None:
        # Hosted permission follows the *writing* provider that would make the
        # hosted call; it is never inferred from the mere presence of a critique
        # object. Injected providers are a test/fixture seam and are not routed
        # through the production admission backend.
        hosted = _is_hosted_provider(writing)
        return MaterializationProviders(
            writing=writing or DeterministicWritingProvider(),
            critique=critique,
            narration=narration,
            allow_hosted=hosted,
            narration_fidelity=narration_fidelity,
        )

    mode = settings.materialization_semantic_mode()
    if mode is SemanticProviderMode.ADAPTIVE:
        writing = cast(
            "MaterializationWritingProvider | None", settings.materialization_writing_provider()
        )
        critique = cast(
            "MaterializationCritiqueProvider | None", settings.materialization_critique_provider()
        )
        hosted = writing is not None or critique is not None
        admission = None
        on_rate_limited = None
        if hosted:
            controller = settings.gemini_admission_controller()
            admission = _admission_callable(controller)
            on_rate_limited = _rate_limit_notifier(controller)
        return MaterializationProviders(
            writing=writing or DeterministicWritingProvider(),
            critique=critique,
            narration=narration,
            allow_hosted=hosted,
            admission=admission,
            on_rate_limited=on_rate_limited,
            narration_factory=(
                _narration_factory(settings, storage) if storage is not None else None
            ),
            narration_fidelity=(
                narration_fidelity
                if narration_fidelity is not None
                else (
                    _narration_fidelity_validator(settings, storage)
                    if storage is not None
                    else None
                )
            ),
        )
    # Deterministic (or an unavailable local-only adapter): zero hosted calls.
    return MaterializationProviders(
        writing=writing or DeterministicWritingProvider(),
        critique=critique,
        narration=narration,
        allow_hosted=False,
        narration_fidelity=narration_fidelity,
    )


def _narration_factory(
    settings: Settings, storage: StorageService
) -> Callable[[MaterializationInputs], object | None]:
    """Build a per-run Gemini narration synthesizer bound to the exact source."""

    def build(inputs: MaterializationInputs) -> object | None:
        profile = inputs.profile
        if not profile.narration_allowed:
            return None
        provider = (profile.tts_provider or "").strip().lower()
        if provider not in {"gemini-tts", "gemini"}:
            # No configured (or a non-Gemini) narrator: TTS is simply unavailable.
            return None
        key = settings.gemini_api_key
        if key is None or not key.get_secret_value():
            return None
        from app.materialization.tts_gemini import GeminiNarrationSynthesizer

        return GeminiNarrationSynthesizer(
            api_key=key.get_secret_value(),
            model=profile.tts_model or settings.materialization_tts_model,
            api_version=profile.tts_api_version or settings.materialization_tts_api_version,
            voice=profile.tts_voice or settings.materialization_tts_default_voice,
            style=profile.tts_style,
            storage=storage,
            source_id=inputs.source_id,
        )

    return build


def _narration_fidelity_validator(
    settings: Settings,
    storage: StorageService,
    *,
    cancel_event: object | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Callable[[object], Mapping[str, object]]:
    """Build the local audio-backed fidelity validator under the heavy-model lease."""

    from app.materialization.tts_gemini import validate_narration_fidelity
    from app.transcription.engine import WhisperEngine

    def runtime_identity() -> dict[str, object]:
        """The effective ASR runtime/options that participate in validation identity."""

        try:
            options = settings.targeted_transcription_options("FINAL_CLIP")
        except Exception:
            return {"provider": "faster-whisper"}
        return {
            "provider": "faster-whisper",
            "model": getattr(options, "model", ""),
            "device": getattr(options, "device", ""),
            "compute_type": getattr(options, "compute_type", ""),
            "beam_size": getattr(options, "beam_size", None),
        }

    def validate(audio: object) -> Mapping[str, object]:
        relative = str(getattr(audio, "relative_path", "") or "")
        expected = str(getattr(audio, "text", "") or "")
        if not relative or not expected:
            return {"status": "DEFERRED", "findings": ["MISSING_NARRATION_EVIDENCE"]}
        try:
            path = storage.resolve_relative(relative)
            engine = WhisperEngine()
            options = settings.targeted_transcription_options("FINAL_CLIP")
            lease_factory = settings.heavy_model_lease_factory()
        except Exception:
            return {"status": "DEFERRED", "findings": ["FIDELITY_UNAVAILABLE"]}
        try:
            result = validate_narration_fidelity(
                path,
                expected_text=expected,
                engine=engine,
                options=options,
                lease_factory=lease_factory,
                cancel_event=cancel_event,  # type: ignore[arg-type]
                is_cancelled=is_cancelled,
            )
        except Exception:
            return {"status": "DEFERRED", "findings": ["FIDELITY_UNAVAILABLE"]}
        return result.as_dict()

    #: Marks this as the settings-built validator so the executor may rebind it to
    #: the per-run cancellation event; injected test validators are left untouched.
    validate._stage6_rebindable = True  # type: ignore[attr-defined]
    validate.runtime_identity = runtime_identity  # type: ignore[attr-defined]
    return validate


def build_local_narration_fidelity_validator(
    settings: Settings,
    storage: StorageService,
    *,
    cancel_event: object | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Callable[[object], Mapping[str, object]]:
    """Public factory for the settings-built local audio-backed fidelity validator.

    Used by the production executor and by the CLI TTS audition so both validate
    generated audio through the same leased Whisper engine and strict comparison.
    """

    return _narration_fidelity_validator(
        settings, storage, cancel_event=cancel_event, is_cancelled=is_cancelled
    )


def _admission_callable(controller: object) -> Callable[[], bool]:
    """Wrap the shared admission controller as an admit/deny boolean (HIGH gate)."""

    from app.core.enums import AdmissionPriority

    priority = AdmissionPriority.HIGH

    def admit() -> bool:
        try:
            decision = controller.acquire(priority)  # type: ignore[attr-defined]
        except Exception:
            return False
        return bool(getattr(decision, "admitted", False))

    return admit


def _rate_limit_notifier(controller: object) -> Callable[[], None]:
    """Wrap the shared controller so a rate/quota failure applies its cooldown."""

    def notify() -> None:
        recorder = getattr(controller, "record_rate_limit", None)
        if callable(recorder):
            try:
                recorder()
            except Exception:
                return

    return notify


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


class _NoDispatch:
    """A no-op broker seam for non-worker executor use (no delivery)."""

    def dispatch(self, *_args: object, **_kwargs: object) -> None:
        return None

    def dispatch_finalization(self, *_args: object, **_kwargs: object) -> None:
        return None


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
        #: Shared cancellation event for this run: set when the job is cancelled or
        #: the claim is lost, and observed by the local ASR child.
        self._cancel_event = threading.Event()
        #: A narration synthesizer created by the per-run factory, so it is released
        #: with the other owned provider resources on every exit.
        self._owned_narration: object | None = None
        # Default: no broker delivery (safe for pure/unit reuse). Production
        # workers inject the real broker seam via ``set_render_dispatcher``.
        self._render_dispatcher: object | None = _NoDispatch()

    def set_render_dispatcher(self, dispatcher: object | None) -> None:
        """Set the broker seam for the render handoff.

        ``None`` selects the real Celery broker; a ``_NoDispatch`` default keeps
        direct/unit executor use from touching the broker.
        """

        self._render_dispatcher = dispatcher

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
        stopping = self._job_cancelled() or not self._claim_is_current()
        if stopping:
            self._cancel_event.set()
        return stopping

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
            "requested_profile_key": profile_key,
            "supporting_requested": supporting_requested,
        }
        envelope = get_or_create_envelope(self._session, inputs, request_identity)
        job = self._session.get(ProcessingJob, self._active_job_id, populate_existing=True)
        if job is None:
            return None
        if job.materialization_id != envelope.id:
            if envelope.active_job_id is None:
                bind_active_job(self._session, envelope.id, job.id)
            job.materialization_id = envelope.id
            self._session.commit()
        if self._claim_version is None:
            return None
        owner = claim_materialization(
            self._session,
            envelope.id,
            uuid.UUID(str(self._active_job_id)),
            expected_job_claim_version=self._claim_version,
        )
        return owner

    def _revalidate_upstream(
        self,
        candidate_id: str,
        *,
        priority: str,
        profile_key: str | None,
        supporting_requested: str,
        inputs: MaterializationInputs,
    ) -> bool:
        """Freshly re-derive upstream effectiveness and exact input identity.

        Owner predicates alone do not prove the upstream candidate/plan/contract/
        FINAL_CLIP/composition are still current. Expire cached state, re-run the
        frozen prerequisite read helpers, and compare the exact input fingerprint
        before any authoritative output is written.
        """

        try:
            self._session.expire_all()
            current = assemble_inputs(
                self._session,
                candidate_id,
                priority=priority,
                profile_key=profile_key,
                supporting_requested=supporting_requested,
            )
        except MaterializationError:
            return False
        return input_fingerprint(current) == input_fingerprint(inputs)

    def _finish_superseded(self, owner: MaterializationOwner) -> None:
        """Record a superseded run truthfully: no output, one atomic transition."""

        try:
            envelope = self._session.get(
                ClipMaterialization, owner.materialization_id, populate_existing=True
            )
            reasons = list(envelope.reason_codes or []) if envelope is not None else []
            if REASON_UPSTREAM_REVALIDATION_REQUIRED not in reasons:
                reasons.append(REASON_UPSTREAM_REVALIDATION_REQUIRED)
            ok = guarded_update(
                self._session,
                owner,
                {
                    "lifecycle": MaterializationLifecycle.BLOCKED,
                    "semantic_outcome": MaterializationOutcome.UPSTREAM_REVALIDATION_REQUIRED,
                    "output_fingerprint": "",
                    "cache_eligible": False,
                    "human_review_required": False,
                    "publication_ready": False,
                    "active_job_id": None,
                    "reason_codes": reasons,
                },
            )
            if not ok:
                self._session.rollback()
                return
            finish_owner_job(self._session, owner, JobStatus.SUCCEEDED)
            self._session.commit()
        except Exception:
            self._session.rollback()

    def _component_context(
        self, inputs: MaterializationInputs, owner: MaterializationOwner
    ) -> ComponentContext:
        narration = self._providers.narration
        if narration is None and self._providers.narration_factory is not None:
            narration = self._providers.narration_factory(inputs)
            if narration is not None:
                self._owned_narration = narration
        from app.materialization.narration import NarrationSynthesizer

        narration_provider = narration if isinstance(narration, NarrationSynthesizer) else None
        narration_fidelity = self._providers.narration_fidelity
        if (
            narration_provider is not None
            and narration_fidelity is not None
            and getattr(narration_fidelity, "_stage6_rebindable", False)
        ):
            # Rebind the settings-built validator so a lost lease or a cancelled
            # job sets the same cancellation event the Whisper child observes.
            narration_fidelity = _narration_fidelity_validator(
                self._settings,
                self._storage,
                cancel_event=self._cancel_event,
                is_cancelled=self._should_stop,
            )
        return ComponentContext(
            session=self._session,
            owner=owner,
            inputs=inputs,
            config=self._config,
            storage=self._storage,
            providers=ComponentProviders(
                writing=self._providers.writing,
                critique=self._providers.critique,
                narration=narration_provider,
                narration_fidelity=(
                    narration_fidelity if narration_provider is not None else None
                ),
            ),
            allow_hosted=self._providers.allow_hosted,
            admission=self._providers.admission,
            on_rate_limited=self._providers.on_rate_limited,
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
        self._cancel_event.clear()
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
            # Upstream can change during provider work; revalidate before any
            # authoritative output is written or a render is bound.
            if not self._revalidate_upstream(
                candidate_id,
                priority=priority,
                profile_key=profile_key,
                supporting_requested=supporting_requested,
                inputs=inputs,
            ):
                self._session.rollback()
                self._finish_superseded(owner)
                return StageExecutionResult("", None)
            envelope = self._session.get(
                ClipMaterialization, owner.materialization_id, populate_existing=True
            )
            if envelope is None:
                self._session.rollback()
                return StageExecutionResult("", None)
            if not commit_prepared(self._session, owner, prepared):
                self._session.rollback()
                return StageExecutionResult("", None)
            if not finish_owner_job(self._session, owner, JobStatus.SUCCEEDED):
                self._session.rollback()
                return StageExecutionResult("", None)
            self._session.commit()
            self._job_owner = False

            # The commit used synchronize_session=False and the production session
            # has expire_on_commit=False, so re-read the envelope freshly before
            # deciding whether to dispatch the render handoff.
            envelope = self._session.get(
                ClipMaterialization, owner.materialization_id, populate_existing=True
            )
            if envelope is None:
                return StageExecutionResult(prepared.output_fingerprint, None)

            # Render handoff: a renderable outcome dispatches the existing
            # TRANSFORMED_SHORT renderer for this exact revision. A non-renderable
            # outcome never dispatches. Broker failure is recoverable through
            # POST/resume; the durable AWAITING_RENDER + binding is the marker.
            if (
                envelope.lifecycle is MaterializationLifecycle.AWAITING_RENDER
                and envelope.output_fingerprint
            ):
                try:
                    from app.materialization.finalization import dispatch_transformed_render

                    dispatch_transformed_render(
                        self._session,
                        envelope,
                        candidate_id,
                        settings=self._settings,
                        storage=self._storage,
                        dispatcher=self._render_dispatcher,
                    )
                    self._session.commit()
                except Exception:
                    self._session.rollback()
            return StageExecutionResult(prepared.output_fingerprint, envelope)
        except StageCancelled:
            self._finish_job(JobStatus.CANCELLED)
            raise
        finally:
            if heartbeat is not None:
                heartbeat.stop()
            self._release_resources()

    def finalize(self, candidate_id: str) -> StageExecutionResult:
        """Claim an AWAITING_RENDER envelope and apply the final assessment.

        Branches to finalization without rerunning any accepted writing or
        synthesis: the same MATERIALIZATION job kind and paired job/envelope
        fencing are reused.
        """

        from app.materialization.finalization import finalize_materialization
        from app.materialization.repository import claim_materialization

        self.skipped_duplicate = False
        self.claim_lost = False
        if self._job_cancelled():
            self._finish_job(JobStatus.CANCELLED)
            raise StageCancelled("Stage 6 finalization cancelled before start")
        if not self._claim_job():
            self.skipped_duplicate = True
            return StageExecutionResult("", None)
        heartbeat = self._start_heartbeat()
        try:
            job = self._session.get(ProcessingJob, self._active_job_id)
            materialization_id = job.materialization_id if job is not None else None
            if materialization_id is None:
                self._finish_job(
                    JobStatus.FAILED, MaterializationError("finalization envelope missing")
                )
                return StageExecutionResult("", None)
            owner = claim_materialization(
                self._session,
                materialization_id,
                self._active_job_id,
                expected_job_claim_version=self._claim_version or 0,
            )
            if owner is None:
                self.skipped_duplicate = True
                return StageExecutionResult("", None)
            if not self._claim_guard(require_running=True):
                self._session.rollback()
                return StageExecutionResult("", None)
            result = finalize_materialization(
                self._session, owner, settings=self._settings, storage=self._storage
            )
            self._job_owner = False
            try:
                envelope = self._session.get(ClipMaterialization, materialization_id)
            except Exception:
                envelope = None
            return StageExecutionResult(result.assessment_fingerprint, envelope)
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
            self._owned_narration,
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
        # This worker owns no temporary files: intermediate artifacts are written
        # atomically through StorageService, which manages its own temporary
        # siblings. A global sweep here would delete *another* worker's fresh
        # in-progress ``.tmp`` files, so the executor never runs one.


def build_materialization_executor(
    session: Session,
    storage: StorageService,
    settings: Settings,
    *,
    writing: MaterializationWritingProvider | None = None,
    critique: MaterializationCritiqueProvider | None = None,
    narration: object | None = None,
    narration_fidelity: Callable[[object], Mapping[str, object]] | None = None,
) -> MaterializationExecutor:
    """Build the production Stage 6 executor lazily and network-free."""

    return MaterializationExecutor(
        session=session,
        settings=settings,
        config=settings.stage6_config(),
        storage=storage,
        providers=build_materialization_providers(
            settings,
            writing=writing,
            critique=critique,
            narration=narration,
            narration_fidelity=narration_fidelity,
            storage=storage,
        ),
    )


__all__ = [
    "MaterializationExecutor",
    "MaterializationProviders",
    "build_materialization_executor",
    "build_materialization_providers",
]
