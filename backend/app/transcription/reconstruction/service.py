"""Batch contextual reconstruction with safe Stage 2.5 fallback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from app.core.enums import ReconstructionStatus
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
    ResolutionScores,
    SegmentReconstruction,
)
from app.transcription.reconstruction.validation import validate_candidate
from app.transcription.reconstruction.windows import acoustic_evidence, build_reconstruction_window


class ContextualReconstructor:
    """Resolve bounded local alternatives without overwriting Stage 2.5 evidence."""

    def __init__(self, provider: ReconstructionProvider | None) -> None:
        self._provider = provider

    def reconstruct(
        self,
        segments: Sequence[Mapping[str, object]],
        *,
        language: str | None,
        transcription_fingerprint: str,
        correction_version: str,
    ) -> ReconstructionResult:
        fingerprint = reconstruction_fingerprint(
            segments, language, transcription_fingerprint, correction_version
        )
        if self._provider is None:
            results = tuple(
                self._fallback(index, segment) for index, segment in enumerate(segments)
            )
            return ReconstructionResult(results, _joined(results), fingerprint)
        result: ReconstructionResult = ReconstructionResult((), "", fingerprint)
        try:
            health = self._provider.health()
            provider_method = f"{health.provider}:{health.model or 'unknown'}"
            if health.availability.value != "AVAILABLE":
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
                result = ReconstructionResult(results, _joined(results), fingerprint)
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
            for batch in _batch_requests(requests, max_windows=1):
                generated.update(self._provider.reconstruct_segments(batch))
            results = tuple(
                self._decide(
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
            result = ReconstructionResult(results, _joined(results), fingerprint)
        except (OSError, ProviderResponseError):
            results = tuple(
                self._fallback(
                    index,
                    segment,
                    provider_error=True,
                    status=ReconstructionStatus.PROVIDER_UNAVAILABLE,
                )
                for index, segment in enumerate(segments)
            )
            result = ReconstructionResult(results, _joined(results), fingerprint)
        finally:
            try:
                self._provider.release()
            except Exception:
                # Cleanup is best effort; never replace valid or fallback output.
                result = replace(result, metadata={"release_warning": "provider_release_failed"})
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
                str(operator_text),
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
                str(operator_text),
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
        scores = candidate.scores or _neutral_scores()
        margin = scores.selection_confidence
        decision = decide_candidate(
            phonetic_similarity=validation.phonetic_similarity,
            resolution=scores,
            raw_acoustic_confidence=acoustic_evidence(segment).confidence,
            edit_ratio=validation.edit_ratio,
            margin=margin,
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
            decision.score,
            decision.level,
            flags,
            status,
            routing_score=routing.evidence.score,
            routing_reasons=(routing.reason,),
            focus_spans=routing.focus_spans,
            validated_changes=candidate.changes,
            reconstruction_method=provider_method,
            candidate_id=candidate.candidate_id,
            confidence_margin=margin,
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


def reconstruction_fingerprint(
    segments: Sequence[Mapping[str, object]],
    language: str | None,
    transcription_fingerprint: str,
    correction_version: str,
) -> str:
    payload = {
        "language": language,
        "transcription_fingerprint": transcription_fingerprint,
        "correction_version": correction_version,
        "segments": [
            {
                "raw": segment.get("raw_text", segment.get("text", "")),
                "corrected": segment.get("corrected_text"),
                "start": segment.get("start"),
                "end": segment.get("end"),
            }
            for segment in segments
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


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


def _neutral_scores() -> ResolutionScores:
    return ResolutionScores(0.5, 0.5, 0.5, 0.5, 0.5)
