"""Batch contextual reconstruction with safe Stage 2.5 fallback."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

from app.core.enums import ReconstructionStatus, RefinementPriority
from app.pipeline.executor import ReconstructionCancelled
from app.pipeline.fingerprints import reconstruction_output_fingerprint
from app.transcription.reconstruction.confidence import (
    CONFIDENCE_POLICY_VERSION,
    decide_candidate,
    is_near_acceptance,
)
from app.transcription.reconstruction.entities import SourceEntityMemory, build_entity_memory
from app.transcription.reconstruction.gemini import (
    GeminiErrorCategory,
    GeminiProviderError,
    GeminiReconstructionProvider,
)
from app.transcription.reconstruction.providers import (
    ProviderResponseError,
    ReconstructionProvider,
    ReconstructionRequest,
)
from app.transcription.reconstruction.routing import (
    ADAPTIVE_ROUTING_VERSION,
    AdaptiveRoutingConfig,
    AdaptiveRoutingDecision,
    ReconstructionRoute,
    RoutingMode,
    route_adaptive,
)
from app.transcription.reconstruction.types import (
    ConfidenceLevel,
    ProviderHealth,
    QualityFlag,
    ReconstructionCandidate,
    ReconstructionResult,
    SegmentReconstruction,
    UnloadOutcome,
)
from app.transcription.reconstruction.validation import VALIDATION_VERSION, validate_candidate
from app.transcription.reconstruction.windows import acoustic_evidence, build_reconstruction_window

_MANUAL_METHOD = "operator:manual"
_STAGE25_METHOD = "stage25"
_INDEX_METHOD = "index_deferred"
_INDEX_DEFERRED_REASON = "index_priority_deferred"
_PROVIDER_DISABLED_METHOD = "provider:disabled"
_PROVIDER_DISABLED_REASON = "no_provider_configured"

_PERMANENT_GEMINI_STOP = frozenset(
    {
        GeminiErrorCategory.RATE_LIMITED,
        GeminiErrorCategory.AUTHENTICATION,
        GeminiErrorCategory.MODEL_NOT_FOUND,
        GeminiErrorCategory.INVALID_REQUEST,
    }
)


@dataclass(frozen=True)
class FingerprintCheck:
    """Stable output fingerprint plus whether all identities resolved."""

    fingerprint: str
    resolved: bool


class ContextualReconstructor:
    """Resolve bounded local alternatives without overwriting Stage 2.5 evidence."""

    def __init__(
        self,
        provider: ReconstructionProvider | None,
        *,
        gemini_provider: GeminiReconstructionProvider | None = None,
        routing: AdaptiveRoutingConfig | None = None,
        gemini_budget: int = 0,
        batch_windows: int | None = None,
        batch_characters: int | None = None,
        local_max_targets: int | None = None,
        local_wall_seconds: float | None = None,
        priority: RefinementPriority = RefinementPriority.INDEX,
        monotonic: Callable[[], float] = time.monotonic,
        is_cancelled: Callable[[], bool] | None = None,
        checkpoint: Callable[[dict[int, "SegmentReconstruction"], dict[str, object]], None]
        | None = None,
    ) -> None:
        self._provider = provider
        self._gemini = gemini_provider
        self._routing = routing or AdaptiveRoutingConfig()
        self._gemini_budget = max(0, gemini_budget)
        self._batch_windows = batch_windows
        self._batch_characters = batch_characters
        self._local_max_targets = local_max_targets
        self._local_wall_seconds = local_wall_seconds
        self._priority = priority
        self._monotonic = monotonic
        self._is_cancelled = is_cancelled
        self._checkpoint = checkpoint

    def with_local_provider(
        self, provider: ReconstructionProvider | None
    ) -> "ContextualReconstructor":
        """Return a reconstructor sharing Gemini/routing but with a new local provider."""

        return ContextualReconstructor(
            provider,
            gemini_provider=self._gemini,
            routing=self._routing,
            gemini_budget=self._gemini_budget,
            batch_windows=self._batch_windows,
            batch_characters=self._batch_characters,
            local_max_targets=self._local_max_targets,
            local_wall_seconds=self._local_wall_seconds,
            priority=self._priority,
            monotonic=self._monotonic,
            is_cancelled=self._is_cancelled,
            checkpoint=self._checkpoint,
        )

    def with_priority(self, priority: RefinementPriority) -> "ContextualReconstructor":
        """Return a copy that runs at the given refinement priority."""

        return ContextualReconstructor(
            self._provider,
            gemini_provider=self._gemini,
            routing=self._routing,
            gemini_budget=self._gemini_budget,
            batch_windows=self._batch_windows,
            batch_characters=self._batch_characters,
            local_max_targets=self._local_max_targets,
            local_wall_seconds=self._local_wall_seconds,
            priority=priority,
            monotonic=self._monotonic,
            is_cancelled=self._is_cancelled,
            checkpoint=self._checkpoint,
        )

    @property
    def priority(self) -> RefinementPriority:
        """The refinement priority this reconstructor is configured to run at."""

        return self._priority

    def with_orchestration(
        self,
        *,
        is_cancelled: Callable[[], bool] | None = None,
        checkpoint: Callable[[dict[int, "SegmentReconstruction"], dict[str, object]], None]
        | None = None,
    ) -> "ContextualReconstructor":
        """Return a copy wired to cooperative cancellation and durable checkpoints."""

        return ContextualReconstructor(
            self._provider,
            gemini_provider=self._gemini,
            routing=self._routing,
            gemini_budget=self._gemini_budget,
            batch_windows=self._batch_windows,
            batch_characters=self._batch_characters,
            local_max_targets=self._local_max_targets,
            local_wall_seconds=self._local_wall_seconds,
            priority=self._priority,
            monotonic=self._monotonic,
            is_cancelled=is_cancelled or self._is_cancelled,
            checkpoint=checkpoint or self._checkpoint,
        )

    def runtime_identity(self) -> dict[str, object]:
        """Return the stable runtime identity for Stage 2.7 output dependencies."""

        identity: dict[str, object] = {}
        if self._provider is not None:
            identity.update(dict(self._provider.runtime_identity()))
        else:
            identity["provider"] = "disabled"
            identity["confidence_policy_version"] = CONFIDENCE_POLICY_VERSION
            identity["validation_version"] = VALIDATION_VERSION
        identity["routing_mode"] = self._routing.mode.value
        identity["routing_policy_version"] = ADAPTIVE_ROUTING_VERSION
        identity["routing_policy"] = self._routing.as_dict()
        identity["gemini_budget"] = self._gemini_budget
        identity["local_batch_windows"] = self._batch_windows
        identity["local_batch_characters"] = self._batch_characters
        identity["local_max_targets_per_job"] = self._local_max_targets
        identity["local_wall_seconds"] = self._local_wall_seconds
        identity["priority"] = self._priority.value
        identity["gemini"] = (
            dict(self._gemini.runtime_identity())
            if self._gemini is not None
            else {"provider": "not_configured"}
        )
        return identity

    def refresh_runtime_identity(self) -> dict[str, object]:
        """Resolve live digests and return the refreshed runtime identity.

        INDEX runs never resolve live digests: whole-source indexing does not
        load or probe Qwen, so no local or Gemini network call happens for it.
        """

        if self._priority is RefinementPriority.INDEX:
            return self.runtime_identity()
        if self._provider is not None:
            self._provider.refresh_runtime_identity()
        if self._gemini is not None and self._routing.mode is not RoutingMode.LOCAL_ONLY:
            self._gemini.refresh_runtime_identity()
        return self.runtime_identity()

    def _prompt_diagnostics(self) -> dict[str, object]:
        sizes = getattr(self._provider, "last_request_sizes", lambda: ())()
        serialized = [item.serialized_bytes for item in sizes]
        tokens = [item.estimated_input_tokens for item in sizes]
        return {
            "average_serialized_bytes": sum(serialized) / len(serialized) if serialized else None,
            "max_serialized_bytes": max(serialized) if serialized else None,
            "average_input_tokens": sum(tokens) / len(tokens) if tokens else None,
            "max_input_tokens": max(tokens) if tokens else None,
        }

    def reconstruct(
        self,
        segments: Sequence[Mapping[str, object]],
        *,
        language: str | None,
        transcription_fingerprint: str,
        correction_version: str,
        resolved: Mapping[int, SegmentReconstruction] | None = None,
        target_indexes: Sequence[int] | None = None,
        priority: RefinementPriority | None = None,
    ) -> ReconstructionResult:
        """Resolve refinement targets with bounded provider work.

        ``target_indexes`` restricts which transcript segments are mutation
        targets; all other segments remain immutable context only. ``priority``
        selects the quality tier: INDEX resolves nothing through a provider and
        defers uncertainty truthfully; CANDIDATE/FINAL_CLIP run the bounded
        adaptive provider pipeline.
        """

        effective_priority = priority or self._priority
        if effective_priority is RefinementPriority.INDEX:
            return self._index_reconstruct(
                segments,
                language=language,
                transcription_fingerprint=transcription_fingerprint,
                correction_version=correction_version,
                target_indexes=target_indexes,
                resolved=resolved,
            )
        if self._provider is None and self._gemini is None:
            identity = self.runtime_identity()
            fingerprint = reconstruction_output_fingerprint(
                provider_identity=identity,
                segments=segments,
                language=language,
                transcription_fingerprint=transcription_fingerprint,
                correction_version=correction_version,
                target_indexes=target_indexes,
            )
            targets = (
                list(target_indexes) if target_indexes is not None else list(range(len(segments)))
            )
            results = tuple(
                self._providerless_segment(
                    index,
                    segments[index],
                    language=language,
                    reason=_PROVIDER_DISABLED_REASON,
                    method=_PROVIDER_DISABLED_METHOD,
                )
                for index in targets
            )
            disabled_metadata: dict[str, object] = {
                "runtime_identity": identity,
                "cache_eligible": True,
                "priority": effective_priority.value,
                "provider_calls": 0,
                "gemini_calls": 0,
            }
            return ReconstructionResult(results, _joined(results), fingerprint, disabled_metadata)
        started_at = self._monotonic()
        result: ReconstructionResult = ReconstructionResult((), "", "")
        try:
            local_health = self._local_health()
            gemini_health = (
                self._gemini_health()
                if self._gemini is not None and self._routing.mode is not RoutingMode.LOCAL_ONLY
                else None
            )
            local_available = (
                local_health is not None and local_health.availability.value == "AVAILABLE"
            )
            gemini_available = (
                gemini_health is not None and gemini_health.availability.value == "AVAILABLE"
            )
            identity = self.runtime_identity()
            fingerprint = reconstruction_output_fingerprint(
                provider_identity=identity,
                segments=segments,
                language=language,
                transcription_fingerprint=transcription_fingerprint,
                correction_version=correction_version,
                target_indexes=target_indexes,
            )
            result = ReconstructionResult((), "", fingerprint)
            memory = build_entity_memory(segments)
            decisions = [
                route_adaptive(segment, config=self._routing, language=language)
                for segment in segments
            ]
            targets = (
                list(target_indexes) if target_indexes is not None else list(range(len(segments)))
            )
            requests = {
                index: _reconstruction_request(segments, index, language, memory)
                for index in targets
            }
            state = _JobState(defaultdict(int), self._gemini_budget, False, None, self._monotonic)
            state.total_segments = len(targets)
            by_index, cancelled = self._run_phases(
                segments,
                decisions,
                requests,
                memory,
                local_available,
                gemini_available,
                local_health,
                gemini_health,
                state,
                resolved=resolved,
                target_indexes=targets,
            )
            if not cancelled:
                # Final cooperative cancellation poll immediately before a
                # successful return (e.g. cancellation landing after the last
                # provider batch).
                cancelled = self._poll_cancelled(by_index, state)
            if cancelled:
                self._checkpoint_results(by_index, state)
                raise ReconstructionCancelled("reconstruction cancelled")
            ordered = tuple(by_index[index] for index in targets)
            cache_eligible = self._cache_eligible(state, ordered)
            metadata: dict[str, object] = {
                "runtime_identity": identity,
                "provider_available": local_available,
                "gemini_available": gemini_available,
                "wall_seconds": self._monotonic() - started_at,
                "prompt_diagnostics": self._prompt_diagnostics(),
                "routing_counts": dict(state.counts),
                "cache_eligible": cache_eligible,
                "local_budget": self._local_budget_metadata(state),
                "progress": self._progress(by_index, state),
                "priority": effective_priority.value,
            }
            if self._gemini is not None:
                metadata["gemini_usage"] = self._gemini.usage_summary()
            result = ReconstructionResult(ordered, _joined(ordered), fingerprint, metadata)
        finally:
            result = self._cleanup(result)
        return result

    def _index_reconstruct(
        self,
        segments: Sequence[Mapping[str, object]],
        *,
        language: str | None,
        transcription_fingerprint: str,
        correction_version: str,
        target_indexes: Sequence[int] | None,
        resolved: Mapping[int, SegmentReconstruction] | None,
    ) -> ReconstructionResult:
        """INDEX: preserve evidence and defer all provider reconstruction.

        This is the default whole-source path. It never loads or calls Qwen,
        never constructs/uses Gemini, and never makes a provider network call.
        Unresolved text is acceptable: segments that the adaptive router would
        have sent to a provider are marked unresolved/deferred (never a provider
        failure) so later targeted refinement can pick them up from saved
        evidence. Clean NO_LLM and manual segments keep their existing truthful
        states.
        """

        identity = self.runtime_identity()
        fingerprint = reconstruction_output_fingerprint(
            provider_identity=identity,
            segments=segments,
            language=language,
            transcription_fingerprint=transcription_fingerprint,
            correction_version=correction_version,
            target_indexes=target_indexes,
        )
        targets = list(target_indexes) if target_indexes is not None else list(range(len(segments)))
        resolved = resolved or {}
        results: dict[int, SegmentReconstruction] = {}
        for index in targets:
            if index in resolved:
                results[index] = resolved[index]
            else:
                results[index] = self._index_segment(index, segments[index], language=language)
        ordered = tuple(results[index] for index in targets)
        deferred = sum(1 for item in ordered if item.escalation_reason == _INDEX_DEFERRED_REASON)
        metadata: dict[str, object] = {
            "runtime_identity": identity,
            "priority": RefinementPriority.INDEX.value,
            "index_deferred": True,
            "index_deferred_segments": deferred,
            "reconstruction_method": _INDEX_METHOD,
            "routing_counts": {"index_deferred": deferred},
            "provider_calls": 0,
            "gemini_calls": 0,
            "wall_seconds": 0.0,
            "cache_eligible": True,
        }
        return ReconstructionResult(ordered, _joined(ordered), fingerprint, metadata)

    def _index_segment(
        self,
        index: int,
        segment: Mapping[str, object],
        *,
        language: str | None,
    ) -> SegmentReconstruction:
        """One INDEX target: route cheaply and defer provider work truthfully."""

        operator_text = segment.get("operator_text")
        decision = route_adaptive(segment, config=self._routing, language=language)
        if operator_text:
            return self._manual(index, segment, decision)
        if decision.route is ReconstructionRoute.NO_LLM:
            return self._no_llm(index, segment, decision)
        return self._fallback(
            index,
            segment,
            status=ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
            method=_INDEX_METHOD,
            decision=decision,
            route=decision.route.value,
            escalation_reason=_INDEX_DEFERRED_REASON,
            final_provider=_STAGE25_METHOD,
        )

    def _providerless_segment(
        self,
        index: int,
        segment: Mapping[str, object],
        *,
        language: str | None,
        reason: str,
        method: str,
    ) -> SegmentReconstruction:
        """Truthful unresolved result when no provider is configured for an
        active (non-INDEX) refinement tier."""

        operator_text = segment.get("operator_text")
        decision = route_adaptive(segment, config=self._routing, language=language)
        if operator_text:
            return self._manual(index, segment, decision)
        if decision.route is ReconstructionRoute.NO_LLM:
            return self._no_llm(index, segment, decision)
        return self._fallback(
            index,
            segment,
            status=ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
            method=method,
            decision=decision,
            route=decision.route.value,
            escalation_reason=reason,
            final_provider=_STAGE25_METHOD,
        )

    def _cleanup(self, result: ReconstructionResult) -> ReconstructionResult:
        try:
            outcome = cast(object, self._provider.release()) if self._provider is not None else None
        except Exception:
            result = replace(
                result,
                metadata={**result.metadata, "release_warning": "provider_release_failed"},
            )
        else:
            if isinstance(outcome, UnloadOutcome):
                metadata = {
                    **result.metadata,
                    "unload_outcome": {
                        "requested": outcome.requested,
                        "confirmed": outcome.confirmed,
                        "elapsed_seconds": outcome.elapsed_seconds,
                    },
                }
                if outcome.warning is not None:
                    metadata["release_warning"] = outcome.warning
                result = replace(result, metadata=metadata)
        if self._gemini is not None:
            try:
                self._gemini.release()
            except Exception:
                result = replace(
                    result,
                    metadata={
                        **result.metadata,
                        "gemini_cleanup_warning": "gemini_release_failed",
                    },
                )
        return result

    def _local_health(self) -> ProviderHealth | None:
        if self._provider is None:
            return None
        try:
            return self._provider.health()
        except (OSError, ProviderResponseError):
            return None

    def _gemini_health(self) -> ProviderHealth | None:
        if self._gemini is None:
            return None
        try:
            return self._gemini.health()
        except (OSError, ProviderResponseError):
            return None

    def output_fingerprint(
        self,
        segments: Sequence[Mapping[str, object]],
        *,
        language: str | None,
        transcription_fingerprint: str,
        correction_version: str,
        target_indexes: Sequence[int] | None = None,
    ) -> FingerprintCheck:
        """Compute the stable output fingerprint with no generation call.

        Health probes are cheap metadata lookups used only to resolve identity
        digests; availability itself is excluded from the fingerprint. A matching
        stored fingerprint plus stored cache eligibility lets the executor skip
        provider calls. INDEX runs never probe providers and always resolve.
        """

        if self._priority is RefinementPriority.INDEX:
            fingerprint = reconstruction_output_fingerprint(
                provider_identity=self.runtime_identity(),
                segments=segments,
                language=language,
                transcription_fingerprint=transcription_fingerprint,
                correction_version=correction_version,
                target_indexes=target_indexes,
            )
            return FingerprintCheck(fingerprint, True)
        local_health = self._local_health()
        gemini_health = (
            self._gemini_health()
            if self._gemini is not None and self._routing.mode is not RoutingMode.LOCAL_ONLY
            else None
        )
        local_resolved = local_health is None or local_health.availability.value == "AVAILABLE"
        gemini_resolved = True
        if self._gemini is not None and self._routing.mode is not RoutingMode.LOCAL_ONLY:
            gemini_resolved = (
                gemini_health is not None and gemini_health.availability.value == "AVAILABLE"
            )
        fingerprint = reconstruction_output_fingerprint(
            provider_identity=self.runtime_identity(),
            segments=segments,
            language=language,
            transcription_fingerprint=transcription_fingerprint,
            correction_version=correction_version,
            target_indexes=target_indexes,
        )
        return FingerprintCheck(fingerprint, local_resolved and gemini_resolved)

    def _gemini_missing_reason(self, state: "_JobState") -> str:
        """Whether the current Gemini gap is configured-but-down or not configured."""

        if self._gemini is None:
            return "gemini_not_configured"
        if state.gemini_exhausted:
            return state.gemini_stop_reason or "gemini_budget_exhausted"
        return "gemini_unavailable"

    def _record_gemini_unavailable(self, state: "_JobState") -> None:
        if self._gemini is not None:
            state.counts["gemini_unavailable"] += 1

    def _run_phases(
        self,
        segments: Sequence[Mapping[str, object]],
        decisions: Sequence[AdaptiveRoutingDecision],
        requests: Mapping[int, ReconstructionRequest],
        memory: SourceEntityMemory,
        local_available: bool,
        gemini_available: bool,
        local_health: ProviderHealth | None,
        gemini_health: ProviderHealth | None,
        state: "_JobState",
        resolved: Mapping[int, SegmentReconstruction] | None = None,
        target_indexes: Sequence[int] | None = None,
    ) -> tuple[dict[int, SegmentReconstruction], bool]:
        """Resolve every target in bounded phases and restore transcript order.

        ``resolved`` supplies already-accepted per-target outputs from a prior
        checkpoint whose dependency fingerprint is unchanged; those targets are
        reused without any provider call. ``target_indexes`` restricts which
        segments are mutation targets (context segments are never targets).
        Returns the completed per-index results plus whether cooperative
        cancellation was requested mid-run.
        """

        resolved = resolved or {}
        targets = list(target_indexes) if target_indexes is not None else list(range(len(segments)))
        results: dict[int, SegmentReconstruction] = {}
        direct_ids: list[int] = []
        local_ids: list[int] = []
        for index in targets:
            decision = decisions[index]
            if index in resolved:
                state.counts["reused"] += 1
                results[index] = resolved[index]
            elif segments[index].get("operator_text"):
                state.counts["manual"] += 1
                results[index] = self._manual(index, segments[index], decision)
            elif decision.route is ReconstructionRoute.NO_LLM:
                state.counts["no_llm"] += 1
                results[index] = self._no_llm(index, segments[index], decision)
            elif decision.route is ReconstructionRoute.GEMINI_DIRECT:
                direct_ids.append(index)
            else:
                local_ids.append(index)
        mode = self._routing.mode
        if mode is RoutingMode.GEMINI_ONLY:
            self._gemini_only_phase(
                segments,
                decisions,
                requests,
                memory,
                results,
                direct_ids + local_ids,
                gemini_available,
                gemini_health,
                state,
            )
        elif mode is RoutingMode.LOCAL_ONLY:
            self._local_only_phase(
                segments,
                decisions,
                requests,
                memory,
                results,
                direct_ids,
                local_ids,
                local_available,
                local_health,
                state,
            )
        else:
            self._adaptive_phase(
                segments,
                decisions,
                requests,
                memory,
                results,
                direct_ids,
                local_ids,
                local_available,
                gemini_available,
                local_health,
                gemini_health,
                state,
            )
        for index in targets:
            if index not in results:
                results[index] = self._fallback(index, segments[index], decision=decisions[index])
        return results, state.cancelled

    def _gemini_only_phase(
        self,
        segments: Sequence[Mapping[str, object]],
        decisions: Sequence[AdaptiveRoutingDecision],
        requests: Mapping[int, ReconstructionRequest],
        memory: SourceEntityMemory,
        results: dict[int, SegmentReconstruction],
        targets: list[int],
        gemini_available: bool,
        gemini_health: ProviderHealth | None,
        state: "_JobState",
    ) -> None:
        ranked = sorted(targets, key=lambda index: (-decisions[index].severity, index))
        for index in ranked:
            if self._poll_cancelled(results, state):
                break
            decision = decisions[index]
            state.current_phase = "gemini"
            if (
                gemini_available
                and state.gemini_budget_remaining > 0
                and not state.gemini_exhausted
            ):
                results[index] = self._gemini_attempt(
                    index,
                    segments[index],
                    requests[index],
                    decision,
                    memory,
                    "gemini",
                    state,
                    escalation_reason=None,
                    route_override=ReconstructionRoute.GEMINI_DIRECT.value,
                    gemini_health=gemini_health,
                )
                self._checkpoint_results(results, state)
                if self._poll_cancelled(results, state):
                    break
                continue
            state.counts["gemini_budget_skips"] += 1
            if not gemini_available:
                self._record_gemini_unavailable(state)
                reason = self._gemini_missing_reason(state)
                provider_error = True
                status = ReconstructionStatus.PROVIDER_UNAVAILABLE
            else:
                reason = state.gemini_stop_reason or "gemini_budget_exhausted"
                provider_error = False
                status = ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
            results[index] = self._fallback(
                index,
                segments[index],
                provider_error=provider_error,
                status=status,
                method="gemini",
                decision=decision,
                route=decision.route.value,
                escalation_reason=reason,
            )

    def _local_only_phase(
        self,
        segments: Sequence[Mapping[str, object]],
        decisions: Sequence[AdaptiveRoutingDecision],
        requests: Mapping[int, ReconstructionRequest],
        memory: SourceEntityMemory,
        results: dict[int, SegmentReconstruction],
        direct_ids: list[int],
        local_ids: list[int],
        local_available: bool,
        local_health: ProviderHealth | None,
        state: "_JobState",
    ) -> None:
        local_method = (
            _provider_method(local_health) if local_health is not None else "provider:unknown"
        )
        work: list[tuple[int, str | None, bool]] = []
        for index in direct_ids:
            state.counts["gemini_budget_skips"] += 1
            work.append((index, "gemini_blocked_local_only", False))
        for index in local_ids:
            work.append((index, None, False))
        local_results, _ = self._run_local_queue(
            segments,
            decisions,
            requests,
            memory,
            work,
            local_available,
            local_method,
            state,
            allow_escalations=False,
        )
        results.update(local_results)

    def _adaptive_phase(
        self,
        segments: Sequence[Mapping[str, object]],
        decisions: Sequence[AdaptiveRoutingDecision],
        requests: Mapping[int, ReconstructionRequest],
        memory: SourceEntityMemory,
        results: dict[int, SegmentReconstruction],
        direct_ids: list[int],
        local_ids: list[int],
        local_available: bool,
        gemini_available: bool,
        local_health: ProviderHealth | None,
        gemini_health: ProviderHealth | None,
        state: "_JobState",
    ) -> None:
        local_method = (
            _provider_method(local_health) if local_health is not None else "provider:unknown"
        )
        # 1-4. Direct candidates: spend the strongest-first Gemini budget first.
        # Targets that cannot use Gemini are queued for bounded local work and
        # never re-escalate to Gemini (they already exhausted or lack it).
        work: list[tuple[int, str | None, bool]] = []
        for index in sorted(direct_ids, key=lambda index: (-decisions[index].severity, index)):
            if self._poll_cancelled(results, state):
                break
            decision = decisions[index]
            state.current_phase = "gemini"
            if (
                gemini_available
                and state.gemini_budget_remaining > 0
                and not state.gemini_exhausted
            ):
                state.counts["gemini_direct"] += 1
                results[index] = self._gemini_attempt(
                    index,
                    segments[index],
                    requests[index],
                    decision,
                    memory,
                    local_method,
                    state,
                    escalation_reason=None,
                    route_override=decision.route.value,
                    gemini_health=gemini_health,
                    fallback_to_local=local_available,
                )
                self._checkpoint_results(results, state)
                if self._poll_cancelled(results, state):
                    break
                continue
            state.counts["gemini_budget_skips"] += 1
            if not gemini_available:
                self._record_gemini_unavailable(state)
                reason = self._gemini_missing_reason(state)
            else:
                reason = state.gemini_stop_reason or "gemini_budget_exhausted"
            work.append((index, reason, False))
        # 6. Local-suitable candidates through Qwen in bounded micro-batches.
        for index in local_ids:
            work.append((index, None, True))
        local_results, escalations = self._run_local_queue(
            segments,
            decisions,
            requests,
            memory,
            work,
            local_available,
            local_method,
            state,
            allow_escalations=True,
        )
        results.update(local_results)
        # 7-9. Rank escalations and spend remaining budget on the strongest.
        ranked_escalations = sorted(
            escalations,
            key=lambda item: (
                -self._escalation_priority(decisions[item[0]], item[1], item[2]),
                item[0],
            ),
        )
        for position, (index, local_seg, escalation_reason) in enumerate(ranked_escalations):
            if self._poll_cancelled(results, state):
                break
            decision = decisions[index]
            state.current_phase = "gemini_escalation"
            backlog_active = self._local_backlog_active(state)
            if (
                gemini_available
                and state.gemini_budget_remaining > 0
                and not state.gemini_exhausted
                and backlog_active
            ):
                results[index] = self._gemini_attempt(
                    index,
                    segments[index],
                    requests[index],
                    decision,
                    memory,
                    local_seg.final_provider or _STAGE25_METHOD if local_seg else local_method,
                    state,
                    escalation_reason=escalation_reason,
                    route_override="LOCAL_THEN_GEMINI",
                    gemini_health=gemini_health,
                    local_seg=local_seg,
                )
                self._checkpoint_results(results, state)
                if self._poll_cancelled(results, state):
                    break
                continue
            if not backlog_active:
                # Local wall ceiling expired: invalidate every remaining queued
                # local-origin Gemini escalation. No Gemini call is made for the
                # local backlog; safe Stage 2.5/current results are preserved and
                # marked unresolved/manual review.
                for drop_index, drop_seg, _ in ranked_escalations[position:]:
                    results[drop_index] = self._drop_local_escalation(
                        drop_index,
                        segments[drop_index],
                        decisions[drop_index],
                        drop_seg,
                        state,
                    )
                break
            state.counts["gemini_budget_skips"] += 1
            if not gemini_available:
                self._record_gemini_unavailable(state)
                block_reason = self._gemini_missing_reason(state)
            else:
                block_reason = state.gemini_stop_reason or "gemini_budget_exhausted"
            if local_seg is not None:
                state.counts["unresolved"] += 1
                results[index] = replace(
                    local_seg,
                    escalation_reason=_merge_escalation(local_seg.escalation_reason, block_reason),
                )
            else:
                state.counts["unresolved"] += 1
                results[index] = self._fallback(
                    index,
                    segments[index],
                    provider_error=True,
                    status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                    method="provider:unknown",
                    decision=decision,
                    route=decision.route.value,
                    escalation_reason="local_provider_unavailable",
                )

    def _run_local_queue(
        self,
        segments: Sequence[Mapping[str, object]],
        decisions: Sequence[AdaptiveRoutingDecision],
        requests: Mapping[int, ReconstructionRequest],
        memory: SourceEntityMemory,
        work: Sequence[tuple[int, str | None, bool]],
        local_available: bool,
        local_method: str,
        state: "_JobState",
        allow_escalations: bool,
    ) -> tuple[
        dict[int, SegmentReconstruction],
        list[tuple[int, SegmentReconstruction | None, str]],
    ]:
        """Run the bounded local micro-batch scheduler.

        ``work`` carries ``(index, fallback_reason, escalate)``. Targets are
        ranked strongest-first, capped by the per-job local target ceiling,
        processed in deterministic micro-batches bounded by window and character
        limits, and checkpointed after every batch. Each micro-batch is planned
        immediately before it executes (never eagerly for future batches), after
        a cooperative cancellation poll and a hard local wall-time ceiling check:
        once the ceiling expires no further batch is planned or dispatched, no
        target is classified unfit, and no Gemini escalation is enqueued. A
        later target that cannot fit context is isolated on its own and earlier
        or later valid work still runs. One invalid candidate never rejects
        valid siblings, and a malformed batch degrades to isolated unresolved
        results without recursive splitting or retry storms.
        """

        results: dict[int, SegmentReconstruction] = {}
        escalations: list[tuple[int, SegmentReconstruction | None, str]] = []
        if not work:
            return results, escalations
        if not local_available:
            for index, fallback_reason, escalate in work:
                if escalate and allow_escalations:
                    escalations.append((index, None, "local_provider_unavailable"))
                else:
                    state.counts["unresolved"] += 1
                    results[index] = self._fallback(
                        index,
                        segments[index],
                        provider_error=True,
                        status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                        method="provider:unknown",
                        decision=decisions[index],
                        route=decisions[index].route.value,
                        escalation_reason=fallback_reason or "local_provider_unavailable",
                    )
            return results, escalations
        state.current_phase = "local"
        if state.local_wall_started is None:
            state.local_wall_started = self._monotonic()
        selected, skipped = self._select_local_targets(work, decisions, state)
        for index in skipped:
            state.counts["unresolved"] += 1
            state.local_target_budget_exhausted = True
            results[index] = self._local_budget_skip(
                index,
                segments[index],
                decisions[index],
                "local_target_budget_exhausted",
            )
        attempted: set[int] = set()
        for batch in self._plan_batches(selected, requests):
            if self._poll_cancelled(results, state):
                break
            if not self._local_budget_available(state):
                # Hard wall-time ceiling: stop before planning this micro-batch so
                # a later irreducible target is never planned, never classified
                # unfit, never counts an attempt/failure, and never escalates.
                state.local_time_budget_exhausted = True
                break
            batch_requests = [requests[index] for index in batch]
            units, irreducible = self._plan_local_units(batch_requests)
            if irreducible:
                # A planner rejection (a target still over context after bounded
                # shrinking) isolates only that target; earlier valid work stays
                # checkpointed and later valid units still run, so one oversized
                # target never aborts the whole local phase.
                state.counts["local_failures"] += len(irreducible)
                for index in irreducible:
                    state.counts["local_attempts"] += 1
                    state.local_targets_used += 1
                    attempted.add(index)
                    escalate = next((flag for item, _, flag in work if item == index), False)
                    failed = self._fallback(
                        index,
                        segments[index],
                        provider_error=True,
                        status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                        method=local_method,
                        decision=decisions[index],
                        route=decisions[index].route.value,
                        local_attempted=True,
                        local_result_state="unfit",
                        final_provider=_STAGE25_METHOD,
                    )
                    results[index] = failed
                    if escalate and allow_escalations:
                        if self._local_backlog_active(state):
                            escalations.append((index, failed, "local_context_unfit"))
                        else:
                            results[index] = self._drop_local_escalation(
                                index,
                                segments[index],
                                decisions[index],
                                failed,
                                state,
                            )
                    else:
                        state.counts["unresolved"] += 1
                self._checkpoint_results(results, state)
                if self._poll_cancelled(results, state):
                    break
            first_unit = True
            for unit_requests in units:
                unit_indices = [request.segment_index for request in unit_requests]
                if self._poll_cancelled(results, state):
                    break
                if not first_unit and not self._local_budget_available(state):
                    state.local_time_budget_exhausted = True
                    break
                first_unit = False
                self._checkpoint_results(results, state)
                try:
                    candidates = self._provider.reconstruct_segments(unit_requests)  # type: ignore[union-attr]
                except (OSError, ProviderResponseError):
                    # Only this actual request's targets fail; earlier actual requests
                    # stay checkpointed and reusable.
                    state.counts["local_failures"] += len(unit_indices)
                    for index in unit_indices:
                        state.counts["local_attempts"] += 1
                        state.local_targets_used += 1
                        attempted.add(index)
                        escalate = next((flag for item, _, flag in work if item == index), False)
                        failed = self._fallback(
                            index,
                            segments[index],
                            provider_error=True,
                            status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                            method=local_method,
                            decision=decisions[index],
                            route=decisions[index].route.value,
                            local_attempted=True,
                            local_result_state="failure",
                            final_provider=_STAGE25_METHOD,
                        )
                        results[index] = failed
                        if escalate and allow_escalations:
                            if self._local_backlog_active(state):
                                escalations.append((index, failed, "local_provider_error"))
                            else:
                                results[index] = self._drop_local_escalation(
                                    index,
                                    segments[index],
                                    decisions[index],
                                    failed,
                                    state,
                                )
                        else:
                            state.counts["unresolved"] += 1
                    self._checkpoint_results(results, state)
                    if self._poll_cancelled(results, state):
                        break
                    continue
                for index in unit_indices:
                    state.counts["local_attempts"] += 1
                    state.local_targets_used += 1
                    attempted.add(index)
                    raw = str(segments[index].get("raw_text", segments[index].get("text", "")))
                    candidate = candidates.get(index, ReconstructionCandidate("raw", raw))
                    fallback_reason = next(
                        (reason for item, reason, _ in work if item == index), None
                    )
                    escalate = next((flag for item, _, flag in work if item == index), False)
                    segment_result = self._decide(
                        index,
                        segments[index],
                        candidate,
                        memory,
                        local_method,
                        routing=decisions[index],
                        route=decisions[index].route.value,
                        local_attempted=True,
                        local_result_state="accepted",
                        final_provider=local_method,
                    )
                    if fallback_reason is not None:
                        segment_result = replace(segment_result, escalation_reason=fallback_reason)
                    if (
                        segment_result.applied
                        and segment_result.confidence_level is ConfidenceLevel.HIGH
                    ):
                        state.counts["local_accepted"] += 1
                        results[index] = segment_result
                    else:
                        state.counts["local_unaccepted"] += 1
                        segment_result = replace(segment_result, final_provider=_STAGE25_METHOD)
                        results[index] = segment_result
                        if escalate and allow_escalations:
                            if self._local_backlog_active(state):
                                escalations.append(
                                    (
                                        index,
                                        segment_result,
                                        _local_escalation_reason(segment_result),
                                    )
                                )
                            else:
                                results[index] = self._drop_local_escalation(
                                    index,
                                    segments[index],
                                    decisions[index],
                                    segment_result,
                                    state,
                                )
                        else:
                            state.counts["unresolved"] += 1
                self._checkpoint_results(results, state)
                if self._poll_cancelled(results, state):
                    break
            if state.cancelled or state.local_time_budget_exhausted:
                break
        if not state.cancelled and state.local_time_budget_exhausted:
            for index in selected:
                if index in attempted:
                    continue
                state.counts["unresolved"] += 1
                results[index] = self._local_budget_skip(
                    index,
                    segments[index],
                    decisions[index],
                    "local_time_budget_exhausted",
                )
        return results, escalations

    def _select_local_targets(
        self,
        work: Sequence[tuple[int, str | None, bool]],
        decisions: Sequence[AdaptiveRoutingDecision],
        state: "_JobState",
    ) -> tuple[list[int], list[int]]:
        """Rank local candidates strongest-first and cap by the target ceiling."""

        ordered = sorted(work, key=lambda item: (-decisions[item[0]].severity, item[0]))
        if self._local_max_targets is None:
            return [item[0] for item in ordered], []
        available = max(0, self._local_max_targets - state.local_targets_used)
        selected = [item[0] for item in ordered[:available]]
        skipped = [item[0] for item in ordered[available:]]
        return selected, skipped

    def _plan_batches(
        self, indices: Sequence[int], requests: Mapping[int, ReconstructionRequest]
    ) -> list[list[int]]:
        """Deterministic greedy micro-batches respecting window and character limits.

        Without explicit batch configuration each target is sent as its own
        bounded request so a provider failure is isolated per target.
        """

        if self._batch_windows is None and self._batch_characters is None:
            return [[index] for index in indices]
        batches: list[list[int]] = []
        current: list[int] = []
        current_characters = 0
        for index in indices:
            size = len(json.dumps({"targets": [requests[index].to_payload()]}, ensure_ascii=False))
            if current and self._batch_windows is not None and len(current) >= self._batch_windows:
                batches.append(current)
                current, current_characters = [], 0
            if (
                current
                and self._batch_characters is not None
                and current_characters + size > self._batch_characters
            ):
                batches.append(current)
                current, current_characters = [], 0
            current.append(index)
            current_characters += size
        if current:
            batches.append(current)
        return batches

    def _plan_local_units(
        self, batch_requests: list[ReconstructionRequest]
    ) -> tuple[list[list[ReconstructionRequest]], list[int]]:
        """Plan one outer micro-batch into actual provider request units lazily.

        Planning runs immediately before the batch executes, never for future
        batches, so a later irreducible target cannot abort earlier/later local
        work during eager planning. The provider's pure ``plan_aggregate_batches``
        helper performs the aggregate context-envelope split without executing
        any HTTP call. If it rejects a target as irreducible (still over context
        after bounded shrinking), only that target is isolated and reported for
        fallback/escalation; valid siblings are still scheduled as their own
        actual units so no hidden provider loop can issue requests that escape
        cancellation/budget/checkpoint handling.
        """

        planner = getattr(self._provider, "plan_aggregate_batches", None)
        if planner is None:
            return [batch_requests], []
        try:
            return planner(batch_requests), []
        except ProviderResponseError:
            units: list[list[ReconstructionRequest]] = []
            irreducible: list[int] = []
            for request in batch_requests:
                try:
                    planned = planner([request])
                except ProviderResponseError:
                    irreducible.append(request.segment_index)
                    continue
                units.extend(planned)
            return units, irreducible

    def _local_budget_available(self, state: "_JobState") -> bool:
        """False once the bounded local wall-time budget is exhausted."""

        if self._local_wall_seconds is None or state.local_wall_started is None:
            return True
        return (self._monotonic() - state.local_wall_started) < self._local_wall_seconds

    def _local_backlog_active(self, state: "_JobState") -> bool:
        """Whether local-origin Gemini escalations may still be enqueued/run.

        Once the local wall-time ceiling has expired, no new local-origin Gemini
        escalation may be enqueued and queued ones are invalidated. This is the
        correctness boundary that prevents a local failure backlog from spending
        Gemini quota after the local ceiling.
        """

        if not self._local_budget_available(state):
            state.local_time_budget_exhausted = True
            return False
        return not state.local_time_budget_exhausted

    def _drop_local_escalation(
        self,
        index: int,
        segment: Mapping[str, object],
        decision: AdaptiveRoutingDecision,
        local_seg: SegmentReconstruction | None,
        state: "_JobState",
    ) -> SegmentReconstruction:
        """Convert a local-origin Gemini escalation into a truthful unresolved
        result after the local wall ceiling expired.

        The safe Stage 2.5/current result is preserved and the target is marked
        unresolved/manual review; no Gemini call is made for the backlog.
        """

        state.counts["unresolved"] += 1
        state.counts["local_escalations_dropped"] += 1
        if local_seg is not None:
            return replace(
                local_seg,
                escalation_reason=_merge_escalation(
                    local_seg.escalation_reason, "local_time_budget_exhausted"
                ),
            )
        return self._fallback(
            index,
            segment,
            status=ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
            decision=decision,
            route=decision.route.value,
            escalation_reason="local_time_budget_exhausted",
            final_provider=_STAGE25_METHOD,
        )

    def _consume_local_target(self, state: "_JobState") -> bool:
        """Reserve one local target attempt within the configured ceilings."""

        if (
            self._local_max_targets is not None
            and state.local_targets_used >= self._local_max_targets
        ):
            state.local_target_budget_exhausted = True
            return False
        if self._local_wall_seconds is not None:
            if state.local_wall_started is None:
                state.local_wall_started = self._monotonic()
            if (self._monotonic() - state.local_wall_started) >= self._local_wall_seconds:
                state.local_time_budget_exhausted = True
                return False
        state.local_targets_used += 1
        return True

    def _local_budget_skip(
        self,
        index: int,
        segment: Mapping[str, object],
        decision: AdaptiveRoutingDecision,
        reason: str,
    ) -> SegmentReconstruction:
        """Truthful unresolved result when a local work ceiling is reached."""

        return self._fallback(
            index,
            segment,
            status=ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
            decision=decision,
            route=decision.route.value,
            escalation_reason=reason,
            final_provider=_STAGE25_METHOD,
        )

    def _poll_cancelled(
        self, results: dict[int, SegmentReconstruction], state: "_JobState"
    ) -> bool:
        """One centralized cooperative-cancellation poll.

        On detection it records the requested state, checkpoints the completed
        safe results so far, and returns True so the caller stops scheduling any
        further Qwen or Gemini work. Subsequent polls short-circuit on the
        already-set ``state.cancelled`` so a single request is never double
        counted.
        """

        if state.cancelled:
            return True
        if self._is_cancelled is None or not self._is_cancelled():
            return False
        state.cancelled = True
        state.counts["cancellation_requested"] += 1
        self._checkpoint_results(results, state)
        return True

    def _checkpoint_results(
        self, results: dict[int, SegmentReconstruction], state: "_JobState"
    ) -> None:
        """Persist completed per-target results through the executor checkpoint."""

        if self._checkpoint is None:
            return
        self._checkpoint(dict(results), self._progress(results, state))

    def _progress(
        self,
        results: dict[int, SegmentReconstruction],
        state: "_JobState",
    ) -> dict[str, object]:
        """Lightweight durable progress used by checkpoints and final metadata."""

        local_budget_remaining: int | None = None
        if self._local_max_targets is not None:
            local_budget_remaining = max(0, self._local_max_targets - state.local_targets_used)
        wall_remaining: float | None = None
        if self._local_wall_seconds is not None and state.local_wall_started is not None:
            wall_remaining = max(
                0.0, self._local_wall_seconds - (self._monotonic() - state.local_wall_started)
            )
        return {
            "total_segments": state.total_segments,
            "phase": state.current_phase,
            "no_llm_completed": state.counts["no_llm"] + state.counts["manual"],
            "local_eligible": state.counts["local_attempts"] + state.counts["unresolved"],
            "local_completed": state.counts["local_accepted"] + state.counts["local_unaccepted"],
            "gemini_eligible": state.counts["gemini_direct"] + state.counts["gemini_escalations"],
            "gemini_completed": state.counts["gemini_accepted"] + state.counts["gemini_rejected"],
            "unresolved": state.counts["unresolved"],
            "local_target_budget_remaining": local_budget_remaining,
            "local_wall_seconds_remaining": wall_remaining,
            "cancellation_requested": bool(state.cancelled),
            "routing_counts": dict(state.counts),
        }

    def _local_budget_metadata(self, state: "_JobState") -> dict[str, object]:
        return {
            "max_targets_per_job": self._local_max_targets,
            "max_wall_seconds": self._local_wall_seconds,
            "targets_used": state.local_targets_used,
            "target_budget_exhausted": state.local_target_budget_exhausted,
            "time_budget_exhausted": state.local_time_budget_exhausted,
            "wall_seconds_used": (
                self._monotonic() - state.local_wall_started
                if state.local_wall_started is not None
                else None
            ),
        }

    def _manual(
        self, index: int, segment: Mapping[str, object], decision: AdaptiveRoutingDecision
    ) -> SegmentReconstruction:
        raw = str(segment.get("raw_text", segment.get("text", "")))
        corrected = str(segment.get("corrected_text", raw))
        return SegmentReconstruction(
            index,
            raw,
            corrected,
            corrected,
            None,
            False,
            1.0,
            ConfidenceLevel.HIGH,
            (),
            ReconstructionStatus.MANUAL_OVERRIDE,
            reconstruction_method=_MANUAL_METHOD,
            routing_score=decision.score,
            routing_reasons=(decision.reason,),
            focus_spans=decision.focus_spans,
            route="MANUAL",
            routing_evidence=decision.evidence,
            final_provider=_MANUAL_METHOD,
        )

    def _no_llm(
        self, index: int, segment: Mapping[str, object], decision: AdaptiveRoutingDecision
    ) -> SegmentReconstruction:
        raw = str(segment.get("raw_text", segment.get("text", "")))
        corrected = str(segment.get("corrected_text", raw))
        return SegmentReconstruction(
            index,
            raw,
            corrected,
            corrected,
            None,
            False,
            0.0,
            ConfidenceLevel.HIGH,
            (),
            ReconstructionStatus.NOT_REQUIRED
            if decision.priority.value == "leave"
            else ReconstructionStatus.UNCHANGED_HIGH_CONFIDENCE,
            routing_score=decision.score,
            routing_reasons=(decision.reason,),
            focus_spans=decision.focus_spans,
            route=decision.route.value,
            routing_evidence=decision.evidence,
            final_provider=_STAGE25_METHOD,
        )

    def _local_attempt(
        self,
        index: int,
        segment: Mapping[str, object],
        request: ReconstructionRequest,
        decision: AdaptiveRoutingDecision,
        memory: SourceEntityMemory,
        method: str,
        state: "_JobState",
    ) -> SegmentReconstruction:
        if not self._consume_local_target(state):
            state.counts["unresolved"] += 1
            reason = (
                "local_time_budget_exhausted"
                if state.local_time_budget_exhausted
                else "local_target_budget_exhausted"
            )
            return self._local_budget_skip(index, segment, decision, reason)
        state.counts["local_attempts"] += 1
        raw = str(segment.get("raw_text", segment.get("text", "")))
        try:
            generated = self._provider.reconstruct_segments([request])  # type: ignore[union-attr]
            candidate = generated.get(index, ReconstructionCandidate("raw", raw))
        except (OSError, ProviderResponseError):
            state.counts["local_failures"] += 1
            return self._fallback(
                index,
                segment,
                provider_error=True,
                status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                method=method,
                decision=decision,
                route=decision.route.value,
                local_attempted=True,
                local_result_state="failure",
                final_provider=_STAGE25_METHOD,
            )
        segment_result = self._decide(
            index,
            segment,
            candidate,
            memory,
            method,
            routing=decision,
            route=decision.route.value,
            local_attempted=True,
            local_result_state="accepted",
            final_provider=method,
        )
        if segment_result.applied and segment_result.confidence_level is ConfidenceLevel.HIGH:
            state.counts["local_accepted"] += 1
            return segment_result
        state.counts["local_unaccepted"] += 1
        return replace(segment_result, final_provider=_STAGE25_METHOD)

    def _gemini_attempt(
        self,
        index: int,
        segment: Mapping[str, object],
        request: ReconstructionRequest,
        decision: AdaptiveRoutingDecision,
        memory: SourceEntityMemory,
        local_method: str,
        state: "_JobState",
        escalation_reason: str | None,
        route_override: str,
        gemini_health: ProviderHealth | None = None,
        fallback_to_local: bool = False,
        local_seg: SegmentReconstruction | None = None,
    ) -> SegmentReconstruction:
        state.gemini_budget_remaining -= 1
        if escalation_reason is not None:
            state.counts["gemini_escalations"] += 1
        gemini_model = (
            gemini_health.model
            if gemini_health is not None and gemini_health.model
            else getattr(self._gemini, "model", "unknown")
        )
        method = f"gemini:{gemini_model}"
        try:
            generated = self._gemini.reconstruct_segments([request])  # type: ignore[union-attr]
            candidate = generated.get(
                index,
                ReconstructionCandidate(
                    "raw", str(segment.get("raw_text", segment.get("text", "")))
                ),
            )
        except GeminiProviderError as error:
            state.counts["gemini_failures"] += 1
            if error.category is GeminiErrorCategory.RATE_LIMITED:
                state.counts["gemini_rate_limited"] += 1
                state.gemini_exhausted = True
                state.gemini_stop_reason = "gemini_rate_limit_exhausted"
            elif error.category is GeminiErrorCategory.AUTHENTICATION:
                state.counts["gemini_authentication_failed"] += 1
                state.gemini_exhausted = True
                state.gemini_stop_reason = "gemini_authentication_failed"
            elif error.category in {
                GeminiErrorCategory.MODEL_NOT_FOUND,
                GeminiErrorCategory.INVALID_REQUEST,
            }:
                state.gemini_exhausted = True
                state.gemini_stop_reason = f"gemini_{error.category.value}"
            if local_seg is not None:
                state.counts["unresolved"] += 1
                return replace(
                    local_seg,
                    gemini_attempted=True,
                    gemini_result_state=f"failure:{error.category.value}",
                    route="LOCAL_THEN_GEMINI",
                    escalation_reason=_merge_escalation(
                        local_seg.escalation_reason, f"gemini_{error.category.value}"
                    ),
                )
            failed = self._fallback(
                index,
                segment,
                provider_error=True,
                status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                method=method,
                decision=decision,
                route=route_override,
                gemini_attempted=True,
                gemini_result_state=f"failure:{error.category.value}",
                final_provider=_STAGE25_METHOD,
                escalation_reason=escalation_reason,
            )
            if fallback_to_local and self._provider is not None:
                local_seg = self._local_attempt(
                    index, segment, request, decision, memory, local_method, state
                )
                return replace(
                    local_seg,
                    gemini_attempted=True,
                    gemini_result_state=f"failure:{error.category.value}",
                )
            state.counts["unresolved"] += 1
            return failed
        except (OSError, ProviderResponseError):
            state.counts["gemini_failures"] += 1
            if local_seg is not None:
                state.counts["unresolved"] += 1
                return replace(
                    local_seg,
                    gemini_attempted=True,
                    gemini_result_state="failure:provider_error",
                    route="LOCAL_THEN_GEMINI",
                    escalation_reason=_merge_escalation(
                        local_seg.escalation_reason, "gemini_provider_error"
                    ),
                )
            failed = self._fallback(
                index,
                segment,
                provider_error=True,
                status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                method=method,
                decision=decision,
                route=route_override,
                gemini_attempted=True,
                gemini_result_state="failure:provider_error",
                final_provider=_STAGE25_METHOD,
                escalation_reason=escalation_reason,
            )
            if fallback_to_local and self._provider is not None:
                local_seg = self._local_attempt(
                    index, segment, request, decision, memory, local_method, state
                )
                return replace(
                    local_seg,
                    gemini_attempted=True,
                    gemini_result_state="failure:provider_error",
                )
            state.counts["unresolved"] += 1
            return failed
        segment_result = self._decide(
            index,
            segment,
            candidate,
            memory,
            method,
            routing=decision,
            route=route_override,
            gemini_attempted=True,
            final_provider=method,
            escalation_reason=escalation_reason,
        )
        if segment_result.applied and segment_result.confidence_level is ConfidenceLevel.HIGH:
            state.counts["gemini_accepted"] += 1
            return segment_result
        state.counts["gemini_rejected"] += 1
        segment_result = replace(segment_result, final_provider=_STAGE25_METHOD)
        if local_seg is not None:
            state.counts["unresolved"] += 1
            return replace(
                local_seg,
                gemini_attempted=True,
                gemini_result_state="rejected",
                route="LOCAL_THEN_GEMINI",
                escalation_reason=_merge_escalation(local_seg.escalation_reason, "gemini_rejected"),
            )
        if fallback_to_local and self._provider is not None:
            local_seg = self._local_attempt(
                index, segment, request, decision, memory, local_method, state
            )
            return replace(
                local_seg,
                gemini_attempted=True,
                gemini_result_state="rejected",
            )
        state.counts["unresolved"] += 1
        return segment_result

    def _fallback(
        self,
        index: int,
        segment: Mapping[str, object],
        provider_error: bool = False,
        status: ReconstructionStatus | None = None,
        method: str | None = None,
        decision: AdaptiveRoutingDecision | None = None,
        route: str | None = None,
        local_attempted: bool = False,
        local_result_state: str | None = None,
        gemini_attempted: bool = False,
        gemini_result_state: str | None = None,
        final_provider: str | None = None,
        escalation_reason: str | None = None,
    ) -> SegmentReconstruction:
        raw = str(segment.get("raw_text", segment.get("text", "")))
        corrected = str(segment.get("corrected_text", raw))
        operator_text = segment.get("operator_text")
        if operator_text:
            return SegmentReconstruction(
                index,
                raw,
                corrected,
                corrected,
                None,
                False,
                1.0,
                ConfidenceLevel.HIGH,
                (),
                ReconstructionStatus.MANUAL_OVERRIDE,
                reconstruction_method=_MANUAL_METHOD,
            )
        flags = (QualityFlag.RECONSTRUCTION_PROVIDER_ERROR,) if provider_error else ()
        return SegmentReconstruction(
            index,
            raw,
            corrected,
            corrected,
            None,
            False,
            0.0,
            ConfidenceLevel.LOW,
            flags,
            status
            or (
                ReconstructionStatus.PROVIDER_UNAVAILABLE
                if provider_error
                else ReconstructionStatus.UNCHANGED_HIGH_CONFIDENCE
            ),
            reconstruction_method=method,
            routing_score=decision.score if decision is not None else None,
            routing_reasons=(decision.reason,) if decision is not None else (),
            focus_spans=decision.focus_spans if decision is not None else (),
            route=route,
            routing_evidence=decision.evidence if decision is not None else (),
            local_attempted=local_attempted,
            local_result_state=local_result_state,
            gemini_attempted=gemini_attempted,
            gemini_result_state=gemini_result_state,
            final_provider=final_provider,
            escalation_reason=escalation_reason,
        )

    def _decide(
        self,
        index: int,
        segment: Mapping[str, object],
        candidate: ReconstructionCandidate,
        memory: SourceEntityMemory,
        provider_method: str = "provider:unknown",
        routing: AdaptiveRoutingDecision | None = None,
        route: str | None = None,
        local_attempted: bool = False,
        local_result_state: str | None = None,
        gemini_attempted: bool = False,
        gemini_result_state: str | None = None,
        final_provider: str | None = None,
        escalation_reason: str | None = None,
    ) -> SegmentReconstruction:
        raw = str(segment.get("raw_text", segment.get("text", "")))
        corrected = str(segment.get("corrected_text", raw))
        operator_text = segment.get("operator_text")
        if operator_text:
            return SegmentReconstruction(
                index,
                raw,
                corrected,
                corrected,
                None,
                False,
                1.0,
                ConfidenceLevel.HIGH,
                (),
                ReconstructionStatus.MANUAL_OVERRIDE,
                reconstruction_method=_MANUAL_METHOD,
            )
        decision = routing or route_adaptive(segment)
        score = decision.score
        if candidate.candidate_id == "raw" or candidate.text == corrected or candidate.text == raw:
            return SegmentReconstruction(
                index,
                raw,
                corrected,
                corrected,
                None,
                False,
                0.0,
                ConfidenceLevel.HIGH,
                (),
                ReconstructionStatus.NOT_REQUIRED
                if decision.priority.value == "leave"
                else ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
                routing_score=score,
                routing_reasons=(decision.reason,),
                focus_spans=decision.focus_spans,
                route=route,
                routing_evidence=decision.evidence,
                local_attempted=local_attempted,
                local_result_state=local_result_state,
                gemini_attempted=gemini_attempted,
                gemini_result_state=gemini_result_state,
                final_provider=final_provider,
                escalation_reason=escalation_reason,
            )
        validation = validate_candidate(raw, candidate, memory)
        if not validation.accepted:
            return SegmentReconstruction(
                index,
                raw,
                corrected,
                corrected,
                candidate.text,
                False,
                0.0,
                ConfidenceLevel.LOW,
                (QualityFlag.LOW_CONFIDENCE_UNRESOLVED,),
                ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
                routing_score=score,
                routing_reasons=(decision.reason,),
                focus_spans=decision.focus_spans,
                validated_changes=candidate.changes,
                candidate_id=candidate.candidate_id,
                reconstruction_method=provider_method,
                validation_reason=validation.reason,
                route=route,
                routing_evidence=decision.evidence,
                local_attempted=local_attempted,
                local_result_state=local_result_state,
                gemini_attempted=gemini_attempted,
                gemini_result_state=gemini_result_state,
                final_provider=final_provider,
                escalation_reason=escalation_reason,
            )
        decided = decide_candidate(
            provider_confidence=candidate.provider_confidence,
            phonetic_similarity=validation.phonetic_similarity,
            raw_acoustic_confidence=acoustic_evidence(segment).confidence,
            edit_ratio=validation.edit_ratio,
            token_delta=validation.token_delta,
        )
        text = candidate.text if decided.applied else corrected
        flags = (
            (QualityFlag.MULTIWORD_RECONSTRUCTION,)
            if len(raw.split()) > 1 or len(candidate.text.split()) > 1
            else ()
        )
        status = (
            ReconstructionStatus.APPLIED
            if decided.applied
            else ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
        )
        near = is_near_acceptance(
            decided,
            phonetic_similarity=validation.phonetic_similarity,
            margin=self._routing.escalation_near_threshold_margin,
        )
        return SegmentReconstruction(
            index,
            raw,
            corrected,
            text,
            candidate.text,
            decided.applied,
            decided.provider_confidence,
            decided.level,
            flags,
            status,
            routing_score=score,
            routing_reasons=(decision.reason,),
            focus_spans=decision.focus_spans,
            validated_changes=candidate.changes,
            reconstruction_method=provider_method,
            candidate_id=candidate.candidate_id,
            confidence_margin=0.0,
            explanation=candidate.explanation,
            decision_reason=decided.reason,
            route=route,
            routing_evidence=decision.evidence,
            local_attempted=local_attempted,
            local_result_state=local_result_state,
            gemini_attempted=gemini_attempted,
            gemini_result_state=gemini_result_state,
            final_provider=final_provider,
            escalation_reason=escalation_reason,
            near_acceptance=near,
        )

    def _escalation_priority(
        self,
        decision: AdaptiveRoutingDecision,
        local_seg: SegmentReconstruction | None,
        escalation_reason: str,
    ) -> float:
        """Deterministic escalation ranking: severity, then local evidence."""

        priority = decision.severity
        if local_seg is not None and local_seg.near_acceptance:
            priority += 0.25
        if escalation_reason in {
            "local_provider_error",
            "local_unchanged",
            "local_unresolved",
            "local_provider_unavailable",
        }:
            priority += 0.10
        return priority

    def _cache_eligible(self, state: "_JobState", results: Sequence[SegmentReconstruction]) -> bool:
        """True only when the run completed without transient provider failure.

        Deterministic rejections and genuine unresolved outcomes stay eligible;
        provider-unavailable runs, quota/rate-limit exhaustion, local work
        ceilings, and any provider failure remain retryable on a later run.
        """

        if state.gemini_exhausted:
            return False
        if state.local_target_budget_exhausted or state.local_time_budget_exhausted:
            return False
        if state.counts["gemini_failures"] or state.counts["local_failures"]:
            return False
        if state.counts["gemini_unavailable"]:
            return False
        if any(item.status is ReconstructionStatus.PROVIDER_UNAVAILABLE for item in results):
            return False
        return True


def _local_escalation_reason(segment: SegmentReconstruction) -> str:
    if segment.status is ReconstructionStatus.PROVIDER_UNAVAILABLE:
        return "local_provider_error"
    if segment.near_acceptance:
        return "local_near_acceptance"
    if segment.candidate_text is None or segment.candidate_text == segment.corrected_text:
        return "local_unchanged"
    if segment.validation_reason:
        return "local_validation_rejected"
    return "local_unresolved"


def _merge_escalation(current: str | None, addition: str) -> str:
    if current is None or current == "":
        return addition
    if addition in current:
        return current
    return f"{current};{addition}"


def _provider_method(health: ProviderHealth) -> str:
    return f"{health.provider}:{health.model or 'unknown'}"


def select_final_text(
    *,
    operator_text: str | None,
    reconstructed: str,
    reconstruction_applied: bool,
    level: ConfidenceLevel,
    corrected: str,
    raw: str,
) -> str:
    """Apply the immutable final-text priority shared by pipeline and API writes."""

    if operator_text:
        return operator_text
    if reconstruction_applied and level is ConfidenceLevel.HIGH:
        return reconstructed
    return corrected or raw


class _JobState:
    def __init__(
        self,
        counts: dict[str, int],
        budget: int,
        exhausted: bool,
        stop_reason: str | None,
        monotonic: Callable[[], float],
    ) -> None:
        self.counts = counts
        self.gemini_budget_remaining = budget
        self.gemini_exhausted = exhausted
        self.gemini_stop_reason = stop_reason
        self.monotonic = monotonic
        self.total_segments = 0
        self.current_phase = "idle"
        self.local_targets_used = 0
        self.local_wall_started: float | None = None
        self.local_target_budget_exhausted = False
        self.local_time_budget_exhausted = False
        self.cancelled = False


def _reconstruction_request(
    segments: Sequence[Mapping[str, object]],
    index: int,
    language: str | None,
    memory: SourceEntityMemory,
) -> ReconstructionRequest:
    window = build_reconstruction_window(segments, index)
    target = next(
        item for item in window.segments if item.segment_index == window.target_segment_index
    )
    position = next(
        i
        for i, item in enumerate(window.segments)
        if item.segment_index == window.target_segment_index
    )
    ordered = window.segments
    previous = tuple(item.corrected_text for item in ordered[:position])
    following = tuple(item.corrected_text for item in ordered[position + 1 :])
    entities = tuple(
        form
        for form in memory.occurrences
        if any(form in item.raw_text or form in item.corrected_text for item in ordered)
    )
    routing = route_adaptive(segments[index], language=language)
    return ReconstructionRequest(
        segment_index=index,
        raw_text=target.raw_text,
        corrected_text=target.corrected_text,
        previous=previous[-2:],
        following=following[:2],
        word_evidence=target.word_evidence,
        acoustic=target.acoustic,
        entities=entities,
        routing_reasons=(routing.reason,),
        focus_spans=routing.focus_spans,
        language=language,
    )


def _joined(results: Sequence[SegmentReconstruction]) -> str:
    return " ".join(item.contextual_reconstructed_text for item in results).strip()
