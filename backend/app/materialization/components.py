"""Durable Stage 6 component execution and replay through the repository.

The production executor drives *checkpointed components* here rather than the
all-in-one :func:`run_materialization` fixture helper. Every raw provider attempt
is reserved durably *before* invocation under the owner fence, and every accepted
subcomponent (a writing response, a critique, a synthesized chunk) is persisted
as soon as it is produced, so a crash resumes from the last accepted work without
repeating providers.

The pure deterministic helpers (validation, timeline, audio recipe, assessment)
are reused unchanged: this module only adds the durability boundary and replay.
A replayed writing/critique response is re-validated deterministically, which is
free and makes zero provider calls.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy.orm import Session

from app.core.enums import MaterializationLifecycle, MaterializationOutcome
from app.core.settings import Settings
from app.materialization.delivery import (
    RATE_LIMITED_REASON,
    delivery_status,
    raised_attempt_status,
)
from app.materialization.fingerprints import output_fingerprint, stable_digest
from app.materialization.inputs import MaterializationInputs
from app.materialization.narration import NarrationSynthesizer
from app.materialization.policy import (
    SEMANTIC_STRONG_TIER,
    CallLedger,
    Stage6Config,
)
from app.materialization.providers import (
    BlockCritique,
    CritiqueRequest,
    CritiqueResult,
    CritiqueTarget,
    MaterializationCritiqueProvider,
    MaterializationProviderError,
    MaterializationScript,
    MaterializationWritingProvider,
    WritingRequest,
    WritingResult,
)
from app.materialization.repository import (
    ATTEMPT_RESERVED,
    ATTEMPT_RESPONDED,
    ATTEMPT_UNCERTAIN,
    PRE_RENDER_CONTEXT_METRIC,
    PRE_RENDER_CONTEXT_VERSION,
    MaterializationOwner,
    MaterializationOwnershipError,
    PreparedMaterialization,
    ledger_counts,
    ledger_entries,
    merge_component_checkpoint,
    owner_is_current,
    record_attempt_response,
    reserve_attempt,
    save_attempt_result,
)
from app.materialization.service import run_materialization
from app.materialization.types import NarrationAudio, NarrationSynthesisRequest
from app.services.storage import StorageService


@dataclass
class ComponentProviders:
    """The provider bundle used by one component run."""

    writing: MaterializationWritingProvider
    critique: MaterializationCritiqueProvider | None = None
    narration: NarrationSynthesizer | None = None
    #: Optional local audio-backed fidelity validator for synthesized audio. It
    #: returns a mapping with at least ``status`` (ACCEPTED/DEFERRED/REJECTED).
    narration_fidelity: Callable[[NarrationAudio], Mapping[str, object]] | None = None


@dataclass
class ComponentContext:
    """Everything a component run needs, all derived from the owned envelope."""

    session: Session
    owner: MaterializationOwner
    inputs: MaterializationInputs
    config: Stage6Config
    storage: StorageService
    providers: ComponentProviders
    allow_hosted: bool
    supporting_assets: Sequence[object] = ()
    settings: Settings | None = None
    cancelled: Callable[[], bool] = field(default=lambda: False)
    admission: Callable[[], bool] | None = None
    #: Notify the shared admission controller of a rate/quota failure so its
    #: cooldown is applied (setting a provider's flag alone is not enough).
    on_rate_limited: Callable[[], None] | None = None


#: The pure delivery classification is shared with the bounded TTS audition so
#: both interpret a provider outcome identically (delivery outcome vs content
#: validity). See :mod:`app.materialization.delivery`.
_RATE_LIMITED_REASON = RATE_LIMITED_REASON
_delivery_status = delivery_status
_raised_attempt_status = raised_attempt_status
#: A prior attempt left RESERVED (crash in the reserve→call gap) or UNCERTAIN
#: (ambiguous transport loss) is unresolved uncertainty: a later ordinary
#: recovery/redelivery must never silently invoke the provider again.
_UNCERTAIN_UNRESOLVED_REASON = "PROVIDER_UNCERTAIN_UNRESOLVED"
#: Version of the bounded critique request/evidence binding persisted with each
#: accepted critique checkpoint. A checkpoint without it predates request binding
#: and can never authorize a new generation.
_CRITIQUE_EVIDENCE_VERSION = "stage6-critique-request-v1"


def script_dependency_inputs(
    inputs: MaterializationInputs,
    writing_identity: Mapping[str, object],
    critique_identity: Mapping[str, object] | None,
    config: Stage6Config,
) -> dict[str, object]:
    """The exact SCRIPT generation dependency set, including resolved evidence.

    This is the canonical value the finalizer recomputes from *live* inputs (and
    the persisted writer/critic identities) to prove a stored accepted SCRIPT
    checkpoint belongs to this generation. It is a first-class, recomputable
    identity boundary: a self-consistent content digest alone can never authorize
    review readiness, because it does not tie the accepted copy to the actual
    plan/source/verification/narration dependencies it was written from.
    """

    return {
        "component": "SCRIPT",
        "validation_version": config.validation_version,
        "plan_output_fingerprint": inputs.plan_output_fingerprint,
        "final_clip_output_fingerprint": inputs.final_clip.output_fingerprint,
        "blocks": [dict(block) for block in inputs.blocks],
        "plan_target_context": dict(inputs.plan_target_context),
        "profile_semantic_context": {
            "target_market": inputs.profile.target_market,
            "output_language_policy": inputs.profile.output_language_policy,
            "register_intent": inputs.profile.register_intent,
        },
        "narration": dict(inputs.narration),
        "operator_authored_copy": [dict(item) for item in inputs.operator_authored_copy],
        "verification_dependencies": [dict(item) for item in inputs.verification_dependencies],
        "writing_identity": dict(writing_identity),
        "critique_identity": dict(critique_identity or {}),
    }


def _script_dependency_fingerprint(
    inputs: MaterializationInputs,
    writing_identity: Mapping[str, object],
    critique_identity: Mapping[str, object] | None,
    config: Stage6Config,
) -> str:
    return stable_digest(
        script_dependency_inputs(inputs, writing_identity, critique_identity, config)
    )


def _narration_chunk_fingerprint(
    inputs: MaterializationInputs,
    request: NarrationSynthesisRequest,
    config: Stage6Config,
    synth_identity: Mapping[str, object] | None = None,
) -> str:
    return stable_digest(
        {
            "component": "NARRATION",
            "chunk_id": request.chunk_id,
            "text": request.text,
            "block_index": request.block_index,
            "identity": request.identity(),
            "language_hint": request.language_hint,
            "narration_policy_version": config.naturalness_policy_version,
            # The EFFECTIVE synthesizer runtime identity (resolved model/API/voice/
            # style/settings), not only the request/profile fields: a changed
            # synthesizer must invalidate the accepted audio rather than silently
            # reuse another model's output.
            "synthesizer_identity": dict(synth_identity or {}),
        }
    )


class DurableRequestGateway:
    """Reserve-then-invoke provider calls under the durable owner fence."""

    def __init__(self, context: ComponentContext) -> None:
        self._context = context

    @property
    def context(self) -> ComponentContext:
        return self._context

    @property
    def _config(self) -> Stage6Config:
        return self._context.config

    def _cancel(self) -> bool:
        if self._context.cancelled():
            return True
        return not owner_is_current(self._context.session, self._context.owner)

    def _ledger(self) -> Mapping[str, object] | None:
        from app.models import ClipMaterialization

        row = self._context.session.get(
            ClipMaterialization,
            self._context.owner.materialization_id,
            populate_existing=True,
        )
        return row.call_ledger if row is not None else None

    def _budget_allows(self, kind: str, tier: str | None) -> bool:
        counts = ledger_counts(self._ledger())
        if kind == "synthesis":
            return counts["synthesis"] < self._config.max_raw_synthesis_calls
        if kind in {"semantic", "critique"}:
            if counts["semantic"] >= self._config.max_raw_semantic_calls:
                return False
            if tier == SEMANTIC_STRONG_TIER and counts["strong"] >= (
                self._config.max_strong_tier_semantic_calls
            ):
                return False
        return True

    def run(
        self,
        *,
        component: str,
        dependency_fingerprint: str,
        kind: str,
        tier: str | None,
        hosted: bool,
        operation: Callable[[], Any],
        on_response: Callable[[Any], Mapping[str, object] | None] | None = None,
        checkpoint_component: str | None = None,
    ) -> tuple[Any | None, str, str | None]:
        """Run one raw attempt. Returns ``(response, status, attempt_id)``.

        ``status`` is ``RESPONDED`` or ``DEFERRED``. A ``DEFERRED`` result means
        no provider response is available (denied admission, exhausted budget, or
        cancellation) and, when admission was denied, no local attempt was
        consumed. A reserved attempt whose provider call raised is durably
        ``UNCERTAIN`` (the request may have reached the provider).

        ``on_response`` computes the reusable checkpoint patch for an accepted
        response; it returns ``None`` for a deferred/rejected response so nothing
        replayable is persisted. The attempt outcome and its checkpoint commit
        atomically.
        """

        if self._cancel():
            return None, "DEFERRED", None
        if hosted:
            if not self._budget_allows(kind, tier):
                return None, "DEFERRED", None
            if self._context.admission is not None and not self._context.admission():
                return None, "DEFERRED", None
            # Never silently retry matching uncertain/reserved work: an ordinary
            # recovery or redelivery defers truthfully instead (the prior attempt
            # already consumed its budget). A crash-left reservation is recovered
            # durably as UNCERTAIN ("a reservation without durable response").
            if self.unresolved_attempt(component, dependency_fingerprint):
                self._recover_reservations(component, dependency_fingerprint)
                return None, "DEFERRED", None
            entry = reserve_attempt(
                self._context.session,
                self._context.owner,
                component,
                dependency_fingerprint,
                kind,
                tier,
            )
            attempt_id = str(entry["attempt_id"])
            if self._cancel():
                return None, "DEFERRED", attempt_id
            try:
                response = operation()
            except MaterializationProviderError as error:
                if str(getattr(error, "category", "")) == _RATE_LIMITED_REASON:
                    self._notify_rate_limited()
                save_attempt_result(
                    self._context.session,
                    self._context.owner,
                    attempt_id,
                    status=_raised_attempt_status(error),
                )
                raise
            except Exception:
                save_attempt_result(
                    self._context.session,
                    self._context.owner,
                    attempt_id,
                    status=ATTEMPT_UNCERTAIN,
                )
                raise
            if self._cancel():
                save_attempt_result(
                    self._context.session,
                    self._context.owner,
                    attempt_id,
                    status=ATTEMPT_UNCERTAIN,
                )
                return None, "DEFERRED", attempt_id
            patch = on_response(response) if on_response is not None else None
            delivery = _delivery_status(response)
            if str(getattr(response, "reason_code", "") or "") == _RATE_LIMITED_REASON:
                self._notify_rate_limited()
            record_attempt_response(
                self._context.session,
                self._context.owner,
                attempt_id,
                status=delivery,
                component=checkpoint_component or component,
                response_fingerprint=stable_digest(_response_identity(response)),
                checkpoint_patch=patch,
            )
            if delivery != ATTEMPT_RESPONDED:
                return response, "DEFERRED", attempt_id
            return response, "RESPONDED", attempt_id
        return operation(), "RESPONDED", None

    def _notify_rate_limited(self) -> None:
        notify = self._context.on_rate_limited
        if notify is None:
            return
        try:
            notify()
        except Exception:
            return

    def unresolved_attempt(self, component: str, dependency_fingerprint: str) -> bool:
        """True when a matching attempt is already unresolved uncertainty.

        A ``RESERVED`` attempt is a crash in the reserve→call gap; an ``UNCERTAIN``
        attempt is an ambiguous transport loss. Either way the request may have
        reached the provider, so a later ordinary run must never silently retry
        it; the generation defers honestly instead. Only a *new* explicitly
        authorized attempt (allocated under the remaining durable budget) may
        retry, and that path never consults this guard.
        """

        for entry in ledger_entries(self._ledger()):
            if (
                entry.get("component") == component
                and entry.get("dependency_fingerprint") == dependency_fingerprint
                and entry.get("status") in {ATTEMPT_RESERVED, ATTEMPT_UNCERTAIN}
            ):
                return True
        return False

    #: Backwards-compatible alias for callers that only reasoned about UNCERTAIN.
    has_uncertain_attempt = unresolved_attempt

    def _recover_reservations(self, component: str, dependency_fingerprint: str) -> None:
        """Durably mark crash-left RESERVED attempts as UNCERTAIN.

        A reservation without a durable response is unresolved uncertainty, never
        permission to invoke again. Normalizing it here keeps the persisted ledger
        truthful for observers and later recovery.
        """

        for entry in ledger_entries(self._ledger()):
            if (
                entry.get("component") == component
                and entry.get("dependency_fingerprint") == dependency_fingerprint
                and entry.get("status") == ATTEMPT_RESERVED
            ):
                save_attempt_result(
                    self._context.session,
                    self._context.owner,
                    str(entry.get("attempt_id")),
                    status=ATTEMPT_UNCERTAIN,
                )


def _response_identity(response: object) -> object:
    if isinstance(response, WritingResult):
        return _writing_result_to_dict(response)
    if isinstance(response, CritiqueResult):
        return _critique_result_to_dict(response)
    if isinstance(response, NarrationAudio):
        return response.as_dict()
    return repr(response)


# --- (de)serialization for accepted subcomponent responses ------------------


def _writing_result_to_dict(result: WritingResult) -> dict[str, object]:
    return {
        "copies": [
            {"index": int(index), "copy": str(copy)} for index, copy in result.copies.items()
        ],
        "provider_identity": dict(result.provider_identity),
        "notes": result.notes,
        "deferred": bool(result.deferred),
        "reason_code": result.reason_code,
    }


def _writing_result_from_dict(data: Mapping[str, object]) -> WritingResult:
    copies: dict[int, str] = {}
    raw = data.get("copies")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                index = item.get("index")
                if isinstance(index, int) and not isinstance(index, bool):
                    copies[index] = str(item.get("copy", ""))
    identity = data.get("provider_identity")
    reason_code = data.get("reason_code")
    return WritingResult(
        copies=copies,
        provider_identity=dict(identity) if isinstance(identity, Mapping) else {},
        notes=str(data.get("notes", "")),
        deferred=bool(data.get("deferred", False)),
        reason_code=reason_code if isinstance(reason_code, str) else None,
    )


def _critique_result_to_dict(result: CritiqueResult) -> dict[str, object]:
    return {
        # Persist booleans only alongside complete categorical evidence. A bare
        # positive boolean is not authoritative and must not survive a checkpoint
        # as if it were.
        "accepted": [
            {"index": int(index), "value": bool(value)}
            for index, value in result.accepted.items()
            if index in result.blocks
        ],
        "blocks": [block.as_dict() for block in result.blocks.values()],
        "provider_identity": dict(result.provider_identity),
        "notes": result.notes,
        "deferred": bool(result.deferred),
        "reason_code": result.reason_code,
    }


def _critique_result_from_dict(data: Mapping[str, object]) -> CritiqueResult:
    accepted: dict[int, bool] = {}
    raw = data.get("accepted")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                index = item.get("index")
                if isinstance(index, int) and not isinstance(index, bool):
                    accepted[index] = bool(item.get("value", False))
    blocks: dict[int, BlockCritique] = {}
    raw_blocks = data.get("blocks")
    if isinstance(raw_blocks, list):
        for item in raw_blocks:
            if isinstance(item, Mapping):
                block = BlockCritique.from_dict(item)
                if block.block_index >= 0:
                    blocks[block.block_index] = block
    # A legacy/malformed boolean-only entry has no complete categorical
    # evidence. Drop it and defer truthfully so replay can never synthesize
    # approval or resend the provider request.
    boolean_only = [index for index in accepted if index not in blocks]
    for index in boolean_only:
        del accepted[index]
    identity = data.get("provider_identity")
    reason_code = data.get("reason_code")
    return CritiqueResult(
        accepted=accepted,
        provider_identity=dict(identity) if isinstance(identity, Mapping) else {},
        notes=str(data.get("notes", "")),
        deferred=bool(data.get("deferred", False)) or bool(boolean_only),
        reason_code=(
            reason_code
            if isinstance(reason_code, str)
            else ("MALFORMED_CRITIQUE_EVIDENCE" if boolean_only else None)
        ),
        blocks=blocks,
    )


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _narration_audio_from_dict(data: Mapping[str, object]) -> NarrationAudio:
    provider_identity = data.get("provider_identity")
    validation = data.get("validation")
    return NarrationAudio(
        chunk_id=str(data.get("chunk_id", "")),
        block_index=_as_int(data.get("block_index"), -1),
        text=str(data.get("text", "")),
        relative_path=str(data.get("relative_path", "")),
        sha256=str(data.get("sha256", "")),
        size_bytes=_as_int(data.get("size_bytes")),
        duration_seconds=_as_float(data.get("duration_seconds")),
        sample_rate=_as_int(data.get("sample_rate")),
        channels=_as_int(data.get("channels")),
        container=str(data.get("container", "")),
        provider_identity=dict(provider_identity) if isinstance(provider_identity, Mapping) else {},
        validation=dict(validation) if isinstance(validation, Mapping) else {},
    )


def _script_checkpoint(gateway: DurableRequestGateway) -> Mapping[str, object]:
    from app.models import ClipMaterialization

    row = gateway.context.session.get(
        ClipMaterialization, gateway.context.owner.materialization_id, populate_existing=True
    )
    if row is None or not isinstance(row.component_checkpoints, Mapping):
        return {}
    entry = row.component_checkpoints.get("SCRIPT")
    return entry if isinstance(entry, Mapping) else {}


def _critique_request_fingerprint(request: CritiqueRequest) -> str:
    """Fingerprint the exact bounded critique request a judgment reviewed."""

    return stable_digest(
        {
            "component": "SCRIPT.CRITIQUE.REQUEST",
            "evidence_version": _CRITIQUE_EVIDENCE_VERSION,
            "request": request.as_dict(),
        }
    )


def _reviewed_targets_payload(request: CritiqueRequest) -> list[dict[str, object]]:
    """Snapshot the bounded target facts exactly as the provider saw them."""

    return [target.as_dict() for target in request.targets]


def critique_request_fingerprint(
    reviewed_targets: Sequence[Mapping[str, object]],
    source_context: Mapping[str, object],
) -> str | None:
    """Recompute the exact bounded critique request fingerprint for a replay.

    The persisted target snapshots must be canonical (no missing/extra keys) so a
    replayed judgment is bound to the precise facts the provider reviewed. Returns
    ``None`` when the snapshot is malformed.
    """

    targets: list[CritiqueTarget] = []
    for data in reviewed_targets:
        target = CritiqueTarget.from_dict(data)
        if target is None or dict(data) != target.as_dict():
            return None
        targets.append(target)
    request = CritiqueRequest(targets=tuple(targets), source_context=dict(source_context))
    return _critique_request_fingerprint(request)


def parse_critique_result(response: Mapping[str, object]) -> CritiqueResult:
    """Parse a persisted critique response with legacy/malformed deferral."""

    return _critique_result_from_dict(response)


# --- checkpointing provider decorators --------------------------------------


class CheckpointedWritingProvider:
    """Writing provider that reserves durably and replays accepted responses."""

    def __init__(
        self,
        provider: MaterializationWritingProvider,
        gateway: DurableRequestGateway,
        dependency_fingerprint: str,
    ) -> None:
        self._provider = provider
        self.provider_name = getattr(provider, "provider_name", "writing")
        self._gateway = gateway
        self._dep = dependency_fingerprint
        self.reused = False
        self.deferred_reason: str | None = None

    def available(self) -> bool:
        return bool(self._provider.available())

    def runtime_identity(self) -> dict[str, object]:
        return dict(self._provider.runtime_identity())

    def write(self, request: WritingRequest) -> WritingResult:
        stored = self._stored_writing()
        if stored is not None:
            self.reused = True
            return stored
        hosted = self._provider.provider_name != "deterministic"

        def on_response(result: Any) -> Mapping[str, object] | None:
            if not isinstance(result, WritingResult) or result.deferred:
                # Deferred/rejected output is never reusable: record the status
                # truthfully without an accepted response so a later run retries.
                return {
                    "dependency_fingerprint": self._dep,
                    "status": "DEFERRED",
                }
            return {
                "dependency_fingerprint": self._dep,
                "status": "PENDING",
                "writing": {
                    "dependency_fingerprint": self._dep,
                    "response": _writing_result_to_dict(result),
                },
            }

        response, status, _attempt_id = self._gateway.run(
            component="SCRIPT.WRITE",
            dependency_fingerprint=self._dep,
            kind="semantic",
            tier=request.tier,
            hosted=hosted,
            operation=lambda: self._provider.write(request),
            on_response=on_response,
            checkpoint_component="SCRIPT",
        )
        if isinstance(response, WritingResult) and response.deferred:
            # Preserve the provider's truthful delivery reason (e.g. UNCERTAIN),
            # never collapsing it into a generic deferral.
            return response
        if status != "RESPONDED" or response is None:
            self.deferred_reason = "DEFERRED_PROVIDER"
            return WritingResult(
                copies={},
                provider_identity=self._provider.runtime_identity(),
                deferred=True,
                reason_code="DEFERRED_PROVIDER",
            )
        result = response
        assert isinstance(result, WritingResult)
        return result

    def _stored_writing(self) -> WritingResult | None:
        # A valid reusable response exists only when the accepted writing
        # subcomponent was persisted; deferred/rejected calls store no `writing`.
        writing = _script_checkpoint(self._gateway).get("writing")
        if not isinstance(writing, Mapping):
            return None
        if writing.get("dependency_fingerprint") != self._dep:
            return None
        response = writing.get("response")
        if not isinstance(response, Mapping):
            return None
        return _writing_result_from_dict(response)

    def release(self) -> None:
        return None


class CheckpointedCritiqueProvider:
    """Critique provider that reserves durably and replays accepted responses."""

    def __init__(
        self,
        provider: MaterializationCritiqueProvider,
        gateway: DurableRequestGateway,
        dependency_fingerprint: str,
    ) -> None:
        self._provider = provider
        self.provider_name = getattr(provider, "provider_name", "critique")
        self._gateway = gateway
        self._dep = dependency_fingerprint
        self.reused = False
        self.deferred_reason: str | None = None

    def available(self) -> bool:
        return bool(self._provider.available())

    def runtime_identity(self) -> dict[str, object]:
        return dict(self._provider.runtime_identity())

    def critique(self, request: CritiqueRequest) -> CritiqueResult:
        request_fingerprint = _critique_request_fingerprint(request)
        stored = self._stored_critique(request_fingerprint)
        if stored is not None:
            if stored.deferred:
                # A stored judgment exists but cannot be re-used for this exact
                # request (or predates binding). Defer truthfully; never relabel
                # it with current facts and never resend automatically.
                self.deferred_reason = stored.reason_code
                return stored
            self.reused = True
            return stored

        def on_response(result: Any) -> Mapping[str, object] | None:
            if not isinstance(result, CritiqueResult) or result.deferred:
                return {
                    "dependency_fingerprint": self._dep,
                    "status": "DEFERRED",
                }
            return {
                "dependency_fingerprint": self._dep,
                "status": "PENDING",
                "critique": {
                    "dependency_fingerprint": self._dep,
                    "request_fingerprint": request_fingerprint,
                    "reviewed_targets": _reviewed_targets_payload(request),
                    "response": _critique_result_to_dict(result),
                },
            }

        response, status, _attempt_id = self._gateway.run(
            component="SCRIPT.CRITIQUE",
            dependency_fingerprint=self._dep,
            kind="critique",
            tier=request.tier,
            hosted=True,
            operation=lambda: self._provider.critique(request),
            on_response=on_response,
            checkpoint_component="SCRIPT",
        )
        if isinstance(response, CritiqueResult) and response.deferred:
            return response
        if status != "RESPONDED" or response is None:
            return CritiqueResult(
                accepted={},
                provider_identity=self._provider.runtime_identity(),
                deferred=True,
                reason_code="DEFERRED_PROVIDER",
            )
        result = response
        assert isinstance(result, CritiqueResult)
        return result

    def _stored_critique(self, request_fingerprint: str) -> CritiqueResult | None:
        critique = _script_checkpoint(self._gateway).get("critique")
        if not isinstance(critique, Mapping):
            return None
        if critique.get("dependency_fingerprint") != self._dep:
            return None
        response = critique.get("response")
        if not isinstance(response, Mapping):
            # Deferred/absent responses are not reusable; an ordinary later run
            # may invoke the provider under the existing budget rules.
            return None
        stored_request = critique.get("request_fingerprint")
        if stored_request != request_fingerprint:
            # A response produced for different facts (or an older unbound
            # checkpoint) must never be presented as a judgment of this request.
            return CritiqueResult(
                accepted={},
                provider_identity=self._provider.runtime_identity(),
                deferred=True,
                reason_code=(
                    "CRITIQUE_CHECKPOINT_UNBOUND"
                    if not isinstance(stored_request, str)
                    else "CRITIQUE_CHECKPOINT_REQUEST_MISMATCH"
                ),
            )
        result = _critique_result_from_dict(response)
        if result.deferred:
            # Malformed/legacy positive evidence defers without a resend.
            return result
        reviewed: dict[int, Mapping[str, object]] = {}
        raw_targets = critique.get("reviewed_targets")
        if isinstance(raw_targets, list):
            for item in raw_targets:
                if not isinstance(item, Mapping):
                    continue
                index = item.get("block_index")
                if isinstance(index, int) and not isinstance(index, bool):
                    reviewed[index] = dict(item)
        if reviewed:
            result = replace(result, reviewed_targets=reviewed)
        return result

    def release(self) -> None:
        return None


class CheckpointedNarrationSynthesizer:
    """Synthesis wrapper that reserves durably and replays accepted audio."""

    def __init__(
        self,
        synthesizer: NarrationSynthesizer,
        gateway: DurableRequestGateway,
        inputs: MaterializationInputs,
        validator: Callable[[NarrationAudio], Mapping[str, object]] | None = None,
    ) -> None:
        self._synthesizer = synthesizer
        self.provider_name = getattr(synthesizer, "provider_name", "narration")
        self._gateway = gateway
        self._inputs = inputs
        self._validator = validator
        self.reused = False

    def available(self) -> bool:
        return bool(self._synthesizer.available())

    def runtime_identity(self) -> dict[str, object]:
        return dict(self._synthesizer.runtime_identity())

    def synthesize(self, request: NarrationSynthesisRequest) -> NarrationAudio:
        dep = _narration_chunk_fingerprint(
            self._inputs, request, self._gateway._config, self.runtime_identity()
        )
        stored = self._stored_chunk(request.chunk_id, dep)
        if stored is not None:
            self.reused = True
            if not self._ensure_fidelity(stored, request, dep):
                raise RuntimeError("NARRATION_FIDELITY_FAILED")
            return stored

        def on_response(result: Any) -> Mapping[str, object] | None:
            if not isinstance(result, NarrationAudio):
                return None
            return {
                "dependency_fingerprint": dep,
                "chunks": {
                    request.chunk_id: {
                        "dependency_fingerprint": dep,
                        "synthesis": {
                            "status": "ACCEPTED",
                            "asset": result.as_dict(),
                        },
                    }
                },
            }

        response, status, _attempt_id = self._gateway.run(
            component=f"NARRATION.{request.chunk_id}",
            dependency_fingerprint=dep,
            kind="synthesis",
            tier=None,
            hosted=True,
            operation=lambda: self._synthesizer.synthesize(request),
            on_response=on_response,
            checkpoint_component="NARRATION",
        )
        if status != "RESPONDED" or response is None:
            raise RuntimeError("TTS_UNAVAILABLE")
        audio = response
        assert isinstance(audio, NarrationAudio)
        if not self._ensure_fidelity(audio, request, dep):
            raise RuntimeError("NARRATION_FIDELITY_FAILED")
        return audio

    def _validation_input_fingerprint(
        self, audio: NarrationAudio, request: NarrationSynthesisRequest
    ) -> str:
        """Fingerprint the *validation* inputs, separate from the synthesis inputs.

        An accepted synthesis is keyed by the spoken script, language/register,
        and effective synthesizer identity. Validation is keyed independently by
        the actual immutable audio hash, the exact expected script, the
        validation-policy version, and the effective ASR runtime/options, so a
        changed ASR configuration reruns local validation (spending no TTS call)
        even though the accepted synthesis is preserved.
        """

        runtime: dict[str, object] = {}
        identity_fn = getattr(self._validator, "runtime_identity", None)
        if callable(identity_fn):
            try:
                runtime = dict(identity_fn())
            except Exception:
                # Missing runtime evidence must never imply a reusable validation:
                # a sentinel makes the fingerprint mismatch any recorded identity.
                runtime = {"runtime_identity": "ERROR"}
        else:
            runtime = {"runtime_identity": "ABSENT"}
        return stable_digest(
            {
                "component": "NARRATION.VALIDATION",
                "chunk_id": audio.chunk_id,
                "audio_sha256": audio.sha256,
                "expected_text": audio.text or request.text,
                "validation_policy_version": self._gateway._config.naturalness_policy_version,
                "runtime_identity": runtime,
            }
        )

    def _ensure_fidelity(
        self, audio: NarrationAudio, request: NarrationSynthesisRequest, dep: str
    ) -> bool:
        """Run the separate, idempotently-checkpointed local fidelity validation.

        The synthesis checkpoint is persisted before this call. An already
        ACCEPTED validation is reused only when its persisted validation-input
        fingerprint (audio hash, exact script, policy version, effective ASR
        runtime/options) matches the current one; a changed ASR configuration
        reruns validation without repeating synthesis. A missing production
        validator defers, never returns True. Unknown (deferred) or rejected
        fidelity never accepts the chunk.
        """

        if self._validator is None:
            return False
        fingerprint = self._validation_input_fingerprint(audio, request)
        stored = self._stored_validation(request.chunk_id, dep)
        if (
            stored is not None
            and str(stored.get("status") or "") == "ACCEPTED"
            and str(stored.get("input_fingerprint") or "") == fingerprint
        ):
            return True
        validation = dict(self._validator(audio))
        validation["input_fingerprint"] = fingerprint
        self._record_validation(request.chunk_id, dep, validation)
        return str(validation.get("status") or "") == "ACCEPTED"

    def _stored_validation(self, chunk_id: str, dep: str) -> Mapping[str, object] | None:
        from app.models import ClipMaterialization

        row = self._gateway.context.session.get(
            ClipMaterialization,
            self._gateway.context.owner.materialization_id,
            populate_existing=True,
        )
        if row is None or not isinstance(row.component_checkpoints, Mapping):
            return None
        entry = row.component_checkpoints.get("NARRATION")
        if not isinstance(entry, Mapping):
            return None
        chunks = entry.get("chunks")
        if not isinstance(chunks, Mapping):
            return None
        chunk = chunks.get(chunk_id)
        if not isinstance(chunk, Mapping) or chunk.get("dependency_fingerprint") != dep:
            return None
        validation = chunk.get("validation")
        return dict(validation) if isinstance(validation, Mapping) else None

    def _record_validation(self, chunk_id: str, dep: str, validation: Mapping[str, object]) -> None:
        from app.models import ClipMaterialization

        session = self._gateway.context.session
        owner = self._gateway.context.owner
        row = session.get(ClipMaterialization, owner.materialization_id, populate_existing=True)
        checkpoints = dict(row.component_checkpoints or {}) if row is not None else {}
        entry = checkpoints.get("NARRATION")
        merged = dict(entry) if isinstance(entry, Mapping) else {}
        chunks = dict(merged.get("chunks") or {})
        chunk = dict(chunks.get(chunk_id) or {})
        chunk["dependency_fingerprint"] = dep
        chunk["validation"] = dict(validation)
        chunks[chunk_id] = chunk
        merged["chunks"] = chunks
        merge_component_checkpoint(session, owner, "NARRATION", merged)

    def _stored_chunk(self, chunk_id: str, dep: str) -> NarrationAudio | None:
        from app.models import ClipMaterialization

        row = self._gateway.context.session.get(
            ClipMaterialization,
            self._gateway.context.owner.materialization_id,
            populate_existing=True,
        )
        if row is None or not isinstance(row.component_checkpoints, Mapping):
            return None
        entry = row.component_checkpoints.get("NARRATION")
        if not isinstance(entry, Mapping):
            return None
        chunks = entry.get("chunks")
        if not isinstance(chunks, Mapping):
            return None
        chunk = chunks.get(chunk_id)
        if not isinstance(chunk, Mapping) or chunk.get("dependency_fingerprint") != dep:
            return None
        synthesis = chunk.get("synthesis")
        if not isinstance(synthesis, Mapping):
            return None
        asset = synthesis.get("asset")
        if not isinstance(asset, Mapping):
            return None
        audio = _narration_audio_from_dict(asset)
        if not self._verify_stored_asset(audio):
            return None
        return audio

    def _verify_stored_asset(self, audio: NarrationAudio) -> bool:
        """Confirm stored bytes match the recorded hash before any checkpoint reuse."""

        storage = self._gateway.context.storage
        try:
            path = storage.resolve_relative(audio.relative_path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return False
        return bool(audio.sha256) and digest == audio.sha256

    def release(self) -> None:
        return None


# --- orchestration ----------------------------------------------------------


def script_dependency_fingerprint(
    context: ComponentContext,
    *,
    critique_provider: MaterializationCritiqueProvider | None,
) -> str:
    critique_identity = critique_provider.runtime_identity() if critique_provider else None
    return _script_dependency_fingerprint(
        context.inputs,
        context.providers.writing.runtime_identity(),
        critique_identity,
        context.config,
    )


def _narration_required(inputs: MaterializationInputs) -> bool:
    need = str(inputs.narration.get("need", "NONE"))
    return need in {"RECOMMENDED", "REQUIRED"}


def prepare_materialization(context: ComponentContext) -> PreparedMaterialization:
    """Execute/replay all pre-render components under the durable gateway."""

    if not owner_is_current(context.session, context.owner):
        raise MaterializationOwnershipError("prepare_materialization lost ownership")

    gateway = DurableRequestGateway(context)
    providers = context.providers
    script_dep_inputs = script_dependency_inputs(
        context.inputs,
        providers.writing.runtime_identity(),
        providers.critique.runtime_identity() if providers.critique else None,
        context.config,
    )
    script_dep = stable_digest(script_dep_inputs)

    writing = CheckpointedWritingProvider(providers.writing, gateway, script_dep)
    critique = (
        CheckpointedCritiqueProvider(providers.critique, gateway, script_dep)
        if providers.critique is not None
        else None
    )
    narration = (
        CheckpointedNarrationSynthesizer(
            providers.narration,
            gateway,
            context.inputs,
            validator=providers.narration_fidelity,
        )
        if providers.narration is not None
        else None
    )

    result = run_materialization(
        context.inputs,
        provider=writing,
        critique_provider=critique,
        narration_synthesizer=narration,
        supporting_assets=context.supporting_assets,  # type: ignore[arg-type]
        config=context.config,
        allow_hosted=context.allow_hosted,
        ledger=CallLedger(),
    )
    _record_script_checkpoint(context, script_dep, script_dep_inputs, result.script)
    return _prepared_from_result(context, result, script_dep)


def _record_script_checkpoint(
    context: ComponentContext,
    script_dep: str,
    script_dep_inputs: Mapping[str, object],
    script: MaterializationScript,
) -> None:
    if not owner_is_current(context.session, context.owner):
        return
    if script.accepted:
        status = "ACCEPTED"
        result_fingerprint: str | None = stable_digest(
            [block.as_dict() for block in script.authored_blocks]
        )
    elif script.deferred:
        status = "DEFERRED"
        result_fingerprint = None
    else:
        status = "REJECTED"
        result_fingerprint = None
    patch: dict[str, object] = {
        "dependency_fingerprint": script_dep,
        "dependency_inputs": dict(script_dep_inputs),
        "status": status,
        "provider_identity": dict(script.provider_identity),
        "result": {
            "authored_blocks": [block.as_dict() for block in script.authored_blocks],
        },
    }
    if result_fingerprint is not None:
        patch["result_fingerprint"] = result_fingerprint
    merge_component_checkpoint(context.session, context.owner, "SCRIPT", patch)


def _prepared_from_result(
    context: ComponentContext, result: Any, script_dep: str
) -> PreparedMaterialization:
    inputs = context.inputs
    checkpoints = dict(result.component_checkpoints)
    from app.models import ClipMaterialization

    row = context.session.get(
        ClipMaterialization, context.owner.materialization_id, populate_existing=True
    )
    if row is not None and isinstance(row.component_checkpoints, Mapping):
        for name in ("SCRIPT", "NARRATION"):
            stored = row.component_checkpoints.get(name)
            if isinstance(stored, Mapping):
                existing = checkpoints.get(name)
                merged = dict(existing) if isinstance(existing, Mapping) else {}
                merged.update(dict(stored))
                checkpoints[name] = merged

    outcome = result.assessment.outcome
    lifecycle = (
        MaterializationLifecycle.AWAITING_RENDER
        if outcome
        in {
            MaterializationOutcome.READY_FOR_HUMAN_REVIEW.value,
            MaterializationOutcome.MATERIALIZED_WITH_CAUTION.value,
        }
        else MaterializationLifecycle.COMPLETE
    )
    output_fp = _output_fingerprint(inputs, checkpoints)
    return PreparedMaterialization(
        lifecycle=lifecycle,
        semantic_outcome=outcome,
        # Review readiness requires actual media: the pre-render prepare step
        # never requests human review, even for a cautionary semantic outcome.
        human_review_required=(
            False
            if lifecycle is MaterializationLifecycle.AWAITING_RENDER
            else bool(result.assessment.human_review_required)
        ),
        repair_count=int(row.repair_count or 0) if row is not None else 0,
        reason_codes=tuple(result.reason_codes),
        component_checkpoints=checkpoints,
        authored_blocks=[block.as_dict() for block in result.script.authored_blocks],
        claim_ledger=[
            block.as_dict() for block in result.script.authored_blocks if block.claim_class
        ],
        slot_resolutions=[item.as_dict() for item in result.slot_resolutions],
        narration_references=[chunk.as_dict() for chunk in result.narration],
        asset_references=[item.as_dict() for item in result.supporting],
        timeline_payload=result.timeline.as_dict(),
        audio_recipe=result.audio_recipe.as_dict(),
        quality_findings=[finding.as_dict() for finding in result.assessment.findings],
        provider_identity=dict(result.script.provider_identity),
        upstream_snapshot=dict(result.upstream_snapshot),
        metrics={
            "segments": len(result.timeline.segments),
            "output_duration": round(result.timeline.output_duration, 6),
            "removals": [dict(item) for item in result.removals],
            PRE_RENDER_CONTEXT_METRIC: {
                "version": PRE_RENDER_CONTEXT_VERSION,
                "materialization_output_fingerprint": output_fp,
                "context": dict(result.assessment_context),
            },
        },
        output_fingerprint=output_fp,
    )


def _output_fingerprint(inputs: MaterializationInputs, checkpoints: Mapping[str, object]) -> str:
    def fp(name: str) -> str:
        entry = checkpoints.get(name)
        if isinstance(entry, Mapping):
            value = entry.get("fingerprint")
            if isinstance(value, str):
                return value
        return ""

    return output_fingerprint(
        inputs,
        script_fp=fp("SCRIPT"),
        narration_fp=fp("NARRATION"),
        supporting_fp=fp("SUPPORTING"),
        timeline_fp=fp("TIMELINE"),
    )


__all__ = [
    "CheckpointedCritiqueProvider",
    "CheckpointedNarrationSynthesizer",
    "CheckpointedWritingProvider",
    "ComponentContext",
    "ComponentProviders",
    "DurableRequestGateway",
    "prepare_materialization",
    "script_dependency_fingerprint",
    "script_dependency_inputs",
]
