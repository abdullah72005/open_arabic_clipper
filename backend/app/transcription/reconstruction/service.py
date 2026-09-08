"""Batch contextual reconstruction with safe Stage 2.5 fallback."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

from app.core.enums import ReconstructionStatus
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
    ) -> None:
        self._provider = provider
        self._gemini = gemini_provider
        self._routing = routing or AdaptiveRoutingConfig()
        self._gemini_budget = max(0, gemini_budget)

    def with_local_provider(
        self, provider: ReconstructionProvider | None
    ) -> "ContextualReconstructor":
        """Return a reconstructor sharing Gemini/routing but with a new local provider."""

        return ContextualReconstructor(
            provider,
            gemini_provider=self._gemini,
            routing=self._routing,
            gemini_budget=self._gemini_budget,
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
        identity["gemini"] = (
            dict(self._gemini.runtime_identity())
            if self._gemini is not None
            else {"provider": "not_configured"}
        )
        return identity

    def refresh_runtime_identity(self) -> dict[str, object]:
        """Resolve live digests and return the refreshed runtime identity."""

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
    ) -> ReconstructionResult:
        if self._provider is None and self._gemini is None:
            identity = self.runtime_identity()
            fingerprint = reconstruction_output_fingerprint(
                provider_identity=identity,
                segments=segments,
                language=language,
                transcription_fingerprint=transcription_fingerprint,
                correction_version=correction_version,
            )
            results = tuple(
                self._fallback(index, segment) for index, segment in enumerate(segments)
            )
            disabled_metadata: dict[str, object] = {
                "runtime_identity": identity,
                "cache_eligible": True,
            }
            return ReconstructionResult(results, _joined(results), fingerprint, disabled_metadata)
        started_at = time.monotonic()
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
            )
            result = ReconstructionResult((), "", fingerprint)
            memory = build_entity_memory(segments)
            decisions = [
                route_adaptive(segment, config=self._routing, language=language)
                for segment in segments
            ]
            requests = [
                _reconstruction_request(segments, index, language, memory)
                for index in range(len(segments))
            ]
            state = _JobState(defaultdict(int), self._gemini_budget, False, None)
            by_index = self._run_phases(
                segments,
                decisions,
                requests,
                memory,
                local_available,
                gemini_available,
                local_health,
                gemini_health,
                state,
            )
            ordered = tuple(by_index[index] for index in range(len(segments)))
            cache_eligible = self._cache_eligible(state, ordered)
            metadata: dict[str, object] = {
                "runtime_identity": identity,
                "provider_available": local_available,
                "gemini_available": gemini_available,
                "wall_seconds": time.monotonic() - started_at,
                "prompt_diagnostics": self._prompt_diagnostics(),
                "routing_counts": dict(state.counts),
                "cache_eligible": cache_eligible,
            }
            if self._gemini is not None:
                metadata["gemini_usage"] = self._gemini.usage_summary()
            result = ReconstructionResult(ordered, _joined(ordered), fingerprint, metadata)
        finally:
            result = self._cleanup(result)
        return result

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
    ) -> FingerprintCheck:
        """Compute the stable output fingerprint with no generation call.

        Health probes are cheap metadata lookups used only to resolve identity
        digests; availability itself is excluded from the fingerprint. A matching
        stored fingerprint plus stored cache eligibility lets the executor skip
        provider calls.
        """

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
        requests: Sequence[ReconstructionRequest],
        memory: SourceEntityMemory,
        local_available: bool,
        gemini_available: bool,
        local_health: ProviderHealth | None,
        gemini_health: ProviderHealth | None,
        state: "_JobState",
    ) -> dict[int, SegmentReconstruction]:
        """Resolve every target in bounded phases and restore transcript order."""

        results: dict[int, SegmentReconstruction] = {}
        direct_ids: list[int] = []
        local_ids: list[int] = []
        for index, decision in enumerate(decisions):
            if segments[index].get("operator_text"):
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
        for index in range(len(segments)):
            if index not in results:
                results[index] = self._fallback(index, segments[index], decision=decisions[index])
        return results

    def _gemini_only_phase(
        self,
        segments: Sequence[Mapping[str, object]],
        decisions: Sequence[AdaptiveRoutingDecision],
        requests: Sequence[ReconstructionRequest],
        memory: SourceEntityMemory,
        results: dict[int, SegmentReconstruction],
        targets: list[int],
        gemini_available: bool,
        gemini_health: ProviderHealth | None,
        state: "_JobState",
    ) -> None:
        ranked = sorted(targets, key=lambda index: (-decisions[index].severity, index))
        for index in ranked:
            decision = decisions[index]
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
        requests: Sequence[ReconstructionRequest],
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
        for index in direct_ids:
            state.counts["gemini_budget_skips"] += 1
            results[index] = self._local_path(
                index,
                segments[index],
                requests[index],
                decisions[index],
                memory,
                local_available,
                local_health,
                local_method,
                state,
                fallback_reason="gemini_blocked_local_only",
            )
        for index in local_ids:
            results[index] = self._local_path(
                index,
                segments[index],
                requests[index],
                decisions[index],
                memory,
                local_available,
                local_health,
                local_method,
                state,
            )

    def _adaptive_phase(
        self,
        segments: Sequence[Mapping[str, object]],
        decisions: Sequence[AdaptiveRoutingDecision],
        requests: Sequence[ReconstructionRequest],
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
        ranked_direct = sorted(direct_ids, key=lambda index: (-decisions[index].severity, index))
        for index in ranked_direct:
            decision = decisions[index]
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
                continue
            state.counts["gemini_budget_skips"] += 1
            if not gemini_available:
                self._record_gemini_unavailable(state)
                reason = self._gemini_missing_reason(state)
            else:
                reason = state.gemini_stop_reason or "gemini_budget_exhausted"
            results[index] = self._local_path(
                index,
                segments[index],
                requests[index],
                decision,
                memory,
                local_available,
                local_health,
                local_method,
                state,
                fallback_reason=reason,
            )
        # 6. Local-suitable candidates through Qwen; collect escalations.
        escalations: list[tuple[int, SegmentReconstruction | None, str]] = []
        for index in local_ids:
            decision = decisions[index]
            if not local_available:
                escalation_reason = "local_provider_unavailable"
                if (
                    gemini_available
                    and state.gemini_budget_remaining > 0
                    and not state.gemini_exhausted
                ):
                    escalations.append((index, None, escalation_reason))
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
                continue
            segment_result = self._local_attempt(
                index,
                segments[index],
                requests[index],
                decision,
                memory,
                local_method,
                state,
            )
            if segment_result.applied and segment_result.confidence_level is ConfidenceLevel.HIGH:
                results[index] = segment_result
            else:
                escalations.append(
                    (index, segment_result, _local_escalation_reason(segment_result))
                )
        # 7-9. Rank escalations and spend remaining budget on the strongest.
        ranked_escalations = sorted(
            escalations,
            key=lambda item: (
                -self._escalation_priority(decisions[item[0]], item[1], item[2]),
                item[0],
            ),
        )
        for index, local_seg, escalation_reason in ranked_escalations:
            decision = decisions[index]
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
                    local_seg.final_provider or _STAGE25_METHOD if local_seg else local_method,
                    state,
                    escalation_reason=escalation_reason,
                    route_override="LOCAL_THEN_GEMINI",
                    gemini_health=gemini_health,
                    local_seg=local_seg,
                )
                continue
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

    def _local_path(
        self,
        index: int,
        segment: Mapping[str, object],
        request: ReconstructionRequest,
        decision: AdaptiveRoutingDecision,
        memory: SourceEntityMemory,
        local_available: bool,
        local_health: ProviderHealth | None,
        local_method: str,
        state: "_JobState",
        fallback_reason: str | None = None,
    ) -> SegmentReconstruction:
        if not local_available:
            state.counts["unresolved"] += 1
            return self._fallback(
                index,
                segment,
                provider_error=True,
                status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                method="provider:unknown",
                decision=decision,
                route=decision.route.value,
                escalation_reason=fallback_reason or "local_provider_unavailable",
            )
        segment_result = self._local_attempt(
            index, segment, request, decision, memory, local_method, state
        )
        if fallback_reason is not None:
            segment_result = replace(segment_result, escalation_reason=fallback_reason)
        if segment_result.applied and segment_result.confidence_level is ConfidenceLevel.HIGH:
            return segment_result
        state.counts["unresolved"] += 1
        return segment_result

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
        provider-unavailable runs, quota/rate-limit exhaustion, and any provider
        failure remain retryable on a later run.
        """

        if state.gemini_exhausted:
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
    ) -> None:
        self.counts = counts
        self.gemini_budget_remaining = budget
        self.gemini_exhausted = exhausted
        self.gemini_stop_reason = stop_reason


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
