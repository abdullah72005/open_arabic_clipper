"""Batch contextual reconstruction with safe Stage 2.5 fallback."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
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
                provider_available=False,
                segments=segments,
                language=language,
                transcription_fingerprint=transcription_fingerprint,
                correction_version=correction_version,
            )
            results = tuple(
                self._fallback(index, segment) for index, segment in enumerate(segments)
            )
            return ReconstructionResult(results, _joined(results), fingerprint)
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
                provider_available=local_available,
                segments=segments,
                language=language,
                transcription_fingerprint=transcription_fingerprint,
                correction_version=correction_version,
                gemini_available=gemini_available,
            )
            result = ReconstructionResult((), "", fingerprint)
            memory = build_entity_memory(segments)
            windows = [build_reconstruction_window([segment], 0) for segment in segments]
            decisions = [
                route_adaptive(window, config=self._routing, language=language)
                for window in windows
            ]
            requests = [
                _reconstruction_request(segments, index, language, memory)
                for index in range(len(segments))
            ]
            state = _JobState(defaultdict(int), self._gemini_budget, False)
            results = tuple(
                self._resolve_segment(
                    index,
                    segments[index],
                    decisions[index],
                    requests[index],
                    memory,
                    local_health,
                    gemini_health,
                    local_available,
                    gemini_available,
                    state,
                )
                for index in range(len(segments))
            )
            metadata: dict[str, object] = {
                "runtime_identity": identity,
                "provider_available": local_available,
                "gemini_available": gemini_available,
                "wall_seconds": time.monotonic() - started_at,
                "prompt_diagnostics": self._prompt_diagnostics(),
                "routing_counts": dict(state.counts),
            }
            if self._gemini is not None:
                metadata["gemini_usage"] = self._gemini.usage_summary()
            result = ReconstructionResult(results, _joined(results), fingerprint, metadata)
        finally:
            try:
                outcome = (
                    cast(object, self._provider.release()) if self._provider is not None else None
                )
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
    ) -> str:
        """Compute the Stage 2.7 output fingerprint without any generation call.

        Health probes are cheap metadata lookups (never content generation), so
        a matching stored fingerprint lets the executor skip provider calls.
        """

        local_health = self._local_health()
        gemini_health = self._gemini_health() if self._gemini is not None else None
        local_available = (
            local_health is not None and local_health.availability.value == "AVAILABLE"
        )
        gemini_available = (
            gemini_health is not None and gemini_health.availability.value == "AVAILABLE"
        )
        return reconstruction_output_fingerprint(
            provider_identity=self.runtime_identity(),
            provider_available=local_available,
            segments=segments,
            language=language,
            transcription_fingerprint=transcription_fingerprint,
            correction_version=correction_version,
            gemini_available=gemini_available,
        )

    def _resolve_segment(
        self,
        index: int,
        segment: Mapping[str, object],
        decision: AdaptiveRoutingDecision,
        request: ReconstructionRequest,
        memory: SourceEntityMemory,
        local_health: ProviderHealth | None,
        gemini_health: ProviderHealth | None,
        local_available: bool,
        gemini_available: bool,
        state: "_JobState",
    ) -> SegmentReconstruction:
        if segment.get("operator_text"):
            state.counts["manual"] += 1
            return self._manual(index, segment, decision)
        if decision.route is ReconstructionRoute.NO_LLM:
            state.counts["no_llm"] += 1
            return self._no_llm(index, segment, decision)
        if self._routing.mode is RoutingMode.GEMINI_ONLY:
            return self._gemini_only_segment(
                index, segment, request, decision, memory, gemini_available, state
            )
        if self._routing.mode is RoutingMode.LOCAL_ONLY:
            return self._local_only_segment(
                index,
                segment,
                request,
                decision,
                memory,
                local_available,
                local_health,
                state,
            )
        return self._adaptive_segment(
            index,
            segment,
            request,
            decision,
            memory,
            local_available,
            gemini_available,
            local_health,
            gemini_health,
            state,
        )

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
            final_provider="stage25",
        )

    def _local_only_segment(
        self,
        index: int,
        segment: Mapping[str, object],
        request: ReconstructionRequest,
        decision: AdaptiveRoutingDecision,
        memory: SourceEntityMemory,
        local_available: bool,
        local_health: ProviderHealth | None,
        state: "_JobState",
    ) -> SegmentReconstruction:
        if decision.route is ReconstructionRoute.GEMINI_DIRECT:
            state.counts["gemini_budget_skips"] += 1
            local_reason = "gemini_blocked_local_only"
        else:
            local_reason = None
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
                escalation_reason="local_provider_unavailable",
            )
        assert local_health is not None
        method = _provider_method(local_health)
        return self._local_attempt(
            index,
            segment,
            request,
            decision,
            memory,
            method,
            escalate_to_gemini=False,
            gemini_available=False,
            state=state,
            fallback_reason=local_reason,
        )

    def _gemini_only_segment(
        self,
        index: int,
        segment: Mapping[str, object],
        request: ReconstructionRequest,
        decision: AdaptiveRoutingDecision,
        memory: SourceEntityMemory,
        gemini_available: bool,
        state: "_JobState",
    ) -> SegmentReconstruction:
        if decision.route is ReconstructionRoute.GEMINI_DIRECT:
            state.counts["gemini_direct"] += 1
        if gemini_available and state.gemini_budget_remaining > 0 and not state.gemini_exhausted:
            return self._gemini_attempt(
                index,
                segment,
                request,
                decision,
                memory,
                "gemini",
                state,
                escalation_reason=None,
                route_override=decision.route.value,
            )
        if not gemini_available:
            state.counts["gemini_unavailable"] += 1
            reason = "gemini_unavailable"
        else:
            state.counts["gemini_budget_skips"] += 1
            reason = "gemini_budget_exhausted"
        state.counts["unresolved"] += 1
        return self._fallback(
            index,
            segment,
            provider_error=not gemini_available,
            status=(
                ReconstructionStatus.PROVIDER_UNAVAILABLE
                if not gemini_available
                else ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
            ),
            method="gemini",
            decision=decision,
            route=decision.route.value,
            escalation_reason=reason,
        )

    def _adaptive_segment(
        self,
        index: int,
        segment: Mapping[str, object],
        request: ReconstructionRequest,
        decision: AdaptiveRoutingDecision,
        memory: SourceEntityMemory,
        local_available: bool,
        gemini_available: bool,
        local_health: ProviderHealth | None,
        gemini_health: ProviderHealth | None,
        state: "_JobState",
    ) -> SegmentReconstruction:
        local_method = _provider_method(local_health) if local_health else "provider:unknown"
        if decision.route is ReconstructionRoute.GEMINI_DIRECT:
            state.counts["gemini_direct"] += 1
            if (
                gemini_available
                and state.gemini_budget_remaining > 0
                and not state.gemini_exhausted
            ):
                return self._gemini_attempt(
                    index,
                    segment,
                    request,
                    decision,
                    memory,
                    local_method,
                    state,
                    escalation_reason=None,
                    route_override=decision.route.value,
                    gemini_health=gemini_health,
                    fallback_to_local=local_available,
                )
            reason = (
                "gemini_unavailable"
                if not gemini_available
                else "gemini_rate_limit_exhausted"
                if state.gemini_exhausted
                else "gemini_budget_exhausted"
            )
            if not gemini_available:
                state.counts["gemini_unavailable"] += 1
            else:
                state.counts["gemini_budget_skips"] += 1
            if local_available:
                return self._local_attempt(
                    index,
                    segment,
                    request,
                    decision,
                    memory,
                    local_method,
                    escalate_to_gemini=False,
                    gemini_available=False,
                    state=state,
                    fallback_reason=reason,
                )
            state.counts["unresolved"] += 1
            return self._fallback(
                index,
                segment,
                provider_error=not gemini_available,
                status=(
                    ReconstructionStatus.PROVIDER_UNAVAILABLE
                    if not gemini_available
                    else ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
                ),
                method="gemini",
                decision=decision,
                route=decision.route.value,
                escalation_reason=reason,
            )
        if not local_available:
            if (
                gemini_available
                and state.gemini_budget_remaining > 0
                and not state.gemini_exhausted
            ):
                return self._gemini_attempt(
                    index,
                    segment,
                    request,
                    decision,
                    memory,
                    local_method,
                    state,
                    escalation_reason="local_provider_unavailable",
                    route_override="LOCAL_THEN_GEMINI",
                    gemini_health=gemini_health,
                )
            state.counts["unresolved"] += 1
            return self._fallback(
                index,
                segment,
                provider_error=True,
                status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                method="provider:unknown",
                decision=decision,
                route=decision.route.value,
                escalation_reason="local_provider_unavailable",
            )
        return self._local_attempt(
            index,
            segment,
            request,
            decision,
            memory,
            local_method,
            escalate_to_gemini=True,
            gemini_available=gemini_available,
            state=state,
            gemini_health=gemini_health,
        )

    def _local_attempt(
        self,
        index: int,
        segment: Mapping[str, object],
        request: ReconstructionRequest,
        decision: AdaptiveRoutingDecision,
        memory: SourceEntityMemory,
        method: str,
        escalate_to_gemini: bool,
        gemini_available: bool,
        state: "_JobState",
        gemini_health: ProviderHealth | None = None,
        fallback_reason: str | None = None,
    ) -> SegmentReconstruction:
        state.counts["local_attempts"] += 1
        raw = str(segment.get("raw_text", segment.get("text", "")))
        try:
            generated = self._provider.reconstruct_segments([request])  # type: ignore[union-attr]
            candidate = generated.get(index, ReconstructionCandidate("raw", raw))
        except (OSError, ProviderResponseError):
            state.counts["local_failures"] += 1
            local_seg = self._fallback(
                index,
                segment,
                provider_error=True,
                status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                method=method,
                decision=decision,
                route=decision.route.value,
                local_attempted=True,
                local_result_state="failure",
                final_provider=method,
                escalation_reason=fallback_reason,
            )
            if escalate_to_gemini:
                return self._escalate(
                    index,
                    segment,
                    request,
                    decision,
                    memory,
                    local_seg,
                    gemini_available,
                    state,
                    escalation_reason="local_provider_error",
                    gemini_health=gemini_health,
                )
            state.counts["unresolved"] += 1
            return local_seg
        segment_result = self._decide(
            index,
            segment,
            candidate,
            memory,
            method,
            routing=decision,
            route=decision.route.value,
            local_attempted=True,
            final_provider=method,
            escalation_reason=fallback_reason,
        )
        if segment_result.applied and segment_result.confidence_level is ConfidenceLevel.HIGH:
            state.counts["local_accepted"] += 1
            return segment_result
        state.counts["local_unaccepted"] += 1
        if escalate_to_gemini:
            return self._escalate(
                index,
                segment,
                request,
                decision,
                memory,
                segment_result,
                gemini_available,
                state,
                escalation_reason=_local_escalation_reason(segment_result),
                gemini_health=gemini_health,
            )
        state.counts["unresolved"] += 1
        return segment_result

    def _escalate(
        self,
        index: int,
        segment: Mapping[str, object],
        request: ReconstructionRequest,
        decision: AdaptiveRoutingDecision,
        memory: SourceEntityMemory,
        local_seg: SegmentReconstruction,
        gemini_available: bool,
        state: "_JobState",
        escalation_reason: str,
        gemini_health: ProviderHealth | None = None,
    ) -> SegmentReconstruction:
        if not gemini_available:
            state.counts["gemini_unavailable"] += 1
            state.counts["unresolved"] += 1
            return replace(
                local_seg,
                escalation_reason="gemini_unavailable",
                final_provider=local_seg.final_provider or "stage25",
            )
        if state.gemini_exhausted:
            state.counts["gemini_budget_skips"] += 1
            state.counts["unresolved"] += 1
            return replace(local_seg, escalation_reason="gemini_rate_limit_exhausted")
        if state.gemini_budget_remaining <= 0:
            state.counts["gemini_budget_skips"] += 1
            state.counts["unresolved"] += 1
            return replace(local_seg, escalation_reason="gemini_budget_exhausted")
        return self._gemini_attempt(
            index,
            segment,
            request,
            decision,
            memory,
            local_seg.final_provider or "stage25",
            state,
            escalation_reason=escalation_reason,
            route_override="LOCAL_THEN_GEMINI",
            gemini_health=gemini_health,
            local_seg=local_seg,
        )

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
            if error.category in {
                GeminiErrorCategory.RATE_LIMITED,
                GeminiErrorCategory.AUTHENTICATION,
            }:
                state.counts["gemini_rate_limited"] += 1
                state.gemini_exhausted = True
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
                final_provider=method,
                escalation_reason=escalation_reason,
            )
            if fallback_to_local and self._provider is not None:
                local_seg = self._local_attempt(
                    index,
                    segment,
                    request,
                    decision,
                    memory,
                    local_method,
                    escalate_to_gemini=False,
                    gemini_available=False,
                    state=state,
                    fallback_reason=f"gemini_{error.category.value}",
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
                final_provider=method,
                escalation_reason=escalation_reason,
            )
            if fallback_to_local and self._provider is not None:
                local_seg = self._local_attempt(
                    index,
                    segment,
                    request,
                    decision,
                    memory,
                    local_method,
                    escalate_to_gemini=False,
                    gemini_available=False,
                    state=state,
                    fallback_reason="gemini_provider_error",
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
                index,
                segment,
                request,
                decision,
                memory,
                local_method,
                escalate_to_gemini=False,
                gemini_available=False,
                state=state,
                fallback_reason="gemini_rejected",
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
        decision = routing or route_adaptive(build_reconstruction_window([segment], 0))
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
    def __init__(self, counts: dict[str, int], budget: int, exhausted: bool) -> None:
        self.counts = counts
        self.gemini_budget_remaining = budget
        self.gemini_exhausted = exhausted


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
    routing = route_adaptive(window, language=language)
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
