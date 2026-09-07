"""Batch contextual reconstruction with safe Stage 2.5 fallback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from app.core.enums import ReconstructionStatus
from app.pipeline.fingerprints import reconstruction_output_fingerprint
from app.transcription.reconstruction.confidence import decide_candidate
from app.transcription.reconstruction.entities import SourceEntityMemory, build_entity_memory
from app.transcription.reconstruction.providers import (
    ProviderResponseError,
    ReconstructionProvider,
    ReconstructionRequest,
)
from app.transcription.reconstruction.routing import route_segment
from app.transcription.reconstruction.types import (
    ConfidenceLevel,
    QualityFlag,
    ReconstructionCandidate,
    ReconstructionResult,
    SegmentReconstruction,
    UnloadOutcome,
)
from app.transcription.reconstruction.validation import validate_candidate
from app.transcription.reconstruction.windows import acoustic_evidence, build_reconstruction_window


class ContextualReconstructor:
    """Resolve bounded local alternatives without overwriting Stage 2.5 evidence."""

    def __init__(self, provider: ReconstructionProvider | None) -> None:
        self._provider = provider

    def runtime_identity(self) -> dict[str, object]:
        """Return the stable runtime identity for Stage 2.7 output dependencies."""

        if self._provider is None:
            return {"provider": "disabled"}
        return dict(self._provider.runtime_identity())

    def refresh_runtime_identity(self) -> dict[str, object]:
        """Resolve the live model digest and return the refreshed runtime identity."""

        if self._provider is None:
            return {"provider": "disabled"}
        return dict(self._provider.refresh_runtime_identity())

    def reconstruct(
        self,
        segments: Sequence[Mapping[str, object]],
        *,
        language: str | None,
        transcription_fingerprint: str,
        correction_version: str,
    ) -> ReconstructionResult:
        if self._provider is None:
            fingerprint = reconstruction_output_fingerprint(
                provider_identity={"provider": "disabled"},
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
        result: ReconstructionResult = ReconstructionResult((), "", "")
        try:
            try:
                health = self._provider.health()
            except (OSError, ProviderResponseError):
                health = None
            provider_method = (
                f"{health.provider}:{health.model or 'unknown'}"
                if health is not None
                else "unknown"
            )
            provider_available = health is not None and health.availability.value == "AVAILABLE"
            identity = self.runtime_identity()
            fingerprint = reconstruction_output_fingerprint(
                provider_identity=identity,
                provider_available=provider_available,
                segments=segments,
                language=language,
                transcription_fingerprint=transcription_fingerprint,
                correction_version=correction_version,
            )
            result = ReconstructionResult((), "", fingerprint)
            if not provider_available:
                results = tuple(
                    self._fallback(
                        index,
                        segment,
                        provider_error=True,
                        status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                        method=provider_method,
                    )
                    for index, segment in enumerate(segments)
                )
                result = ReconstructionResult(
                    results,
                    _joined(results),
                    fingerprint,
                    metadata={"runtime_identity": identity, "provider_available": False},
                )
                return result
            memory = build_entity_memory(segments)
            routing_decisions = [
                route_segment(build_reconstruction_window([segment], 0), language=language)
                for segment in segments
            ]
            routed_indexes = [
                index
                for index, decision in enumerate(routing_decisions)
                if decision.priority.value != "leave"
            ]
            requests = [
                _reconstruction_request(segments, index, language, memory)
                for index in routed_indexes
            ]
            generated: dict[int, ReconstructionCandidate] = {}
            failed_targets: set[int] = set()
            for batch in _batch_requests(requests, max_windows=1):
                for request in batch:
                    try:
                        generated.update(self._provider.reconstruct_segments([request]))
                    except (OSError, ProviderResponseError):
                        failed_targets.add(request.segment_index)
            results = tuple(
                self._fallback(
                    index,
                    segment,
                    provider_error=True,
                    status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                    method=provider_method,
                )
                if index in failed_targets
                else self._decide(
                    index,
                    segment,
                    generated.get(
                        index,
                        ReconstructionCandidate(
                            "raw",
                            str(segment.get("raw_text", segment.get("text", ""))),
                        ),
                    ),
                    memory,
                    provider_method,
                    routing=routing_decisions[index],
                )
                for index, segment in enumerate(segments)
            )
            result = ReconstructionResult(
                results,
                _joined(results),
                fingerprint,
                metadata={"runtime_identity": identity, "provider_available": True},
            )
        finally:
            try:
                outcome = self._provider.release()
            except Exception:
                # Cleanup is best effort; never replace valid or fallback output.
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

    def _fallback(
        self,
        index: int,
        segment: Mapping[str, object],
        provider_error: bool = False,
        status: ReconstructionStatus | None = None,
        method: str | None = None,
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
                reconstruction_method="operator:manual",
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
        )

    def _decide(
        self,
        index: int,
        segment: Mapping[str, object],
        candidate: ReconstructionCandidate,
        memory: SourceEntityMemory,
        provider_method: str = "provider:unknown",
        routing: object | None = None,
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
                reconstruction_method="operator:manual",
            )
        routing = routing or route_segment(build_reconstruction_window([segment], 0))
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
                if routing.priority.value == "leave"
                else ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
                routing_score=routing.evidence.score,
                routing_reasons=(routing.reason,),
                focus_spans=routing.focus_spans,
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
                routing_score=routing.evidence.score,
                routing_reasons=(routing.reason,),
                focus_spans=routing.focus_spans,
                validated_changes=candidate.changes,
                candidate_id=candidate.candidate_id,
                reconstruction_method=provider_method,
                validation_reason=validation.reason,
            )
        decision = decide_candidate(
            provider_confidence=candidate.provider_confidence,
            phonetic_similarity=validation.phonetic_similarity,
            raw_acoustic_confidence=acoustic_evidence(segment).confidence,
            edit_ratio=validation.edit_ratio,
            token_delta=validation.token_delta,
        )
        text = candidate.text if decision.applied else corrected
        flags = (
            (QualityFlag.MULTIWORD_RECONSTRUCTION,)
            if len(raw.split()) > 1 or len(candidate.text.split()) > 1
            else ()
        )
        status = (
            ReconstructionStatus.APPLIED
            if decision.applied
            else ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
        )
        return SegmentReconstruction(
            index,
            raw,
            corrected,
            text,
            candidate.text,
            decision.applied,
            decision.provider_confidence,
            decision.level,
            flags,
            status,
            routing_score=routing.evidence.score,
            routing_reasons=(routing.reason,),
            focus_spans=routing.focus_spans,
            validated_changes=candidate.changes,
            reconstruction_method=provider_method,
            candidate_id=candidate.candidate_id,
            confidence_margin=0.0,
            explanation=candidate.explanation,
            decision_reason=decision.reason,
        )


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
    routing = route_segment(window, language=language)
    return ReconstructionRequest(
        segment_index=index,
        raw_text=target.raw_text,
        corrected_text=target.corrected_text,
        previous=previous[-2:],
        following=following[:2],
        word_evidence=target.word_evidence,
        acoustic=target.acoustic,
        entities=entities,
        routing_reasons=routing.reasons,
        focus_spans=routing.focus_spans,
        language=language,
    )


def _joined(results: Sequence[SegmentReconstruction]) -> str:
    return " ".join(item.contextual_reconstructed_text for item in results).strip()


def _batch_requests(
    requests: list[ReconstructionRequest], *, max_windows: int
) -> list[list[ReconstructionRequest]]:
    """Split deterministic requests into bounded provider batches."""

    if max_windows < 1:
        max_windows = 1
    return [requests[i : i + max_windows] for i in range(0, len(requests), max_windows)]
