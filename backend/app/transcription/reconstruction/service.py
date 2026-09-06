"""Batch contextual reconstruction with safe Stage 2.5 fallback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from app.transcription.reconstruction.confidence import decide_candidate
from app.transcription.reconstruction.entities import SourceEntityMemory, build_entity_memory
from app.transcription.reconstruction.providers import (
    GenerationRequest,
    ProviderResponseError,
    ReconstructionProvider,
    ResolutionChoice,
    ResolutionRequest,
)
from app.transcription.reconstruction.routing import route_segment
from app.transcription.reconstruction.types import (
    ConfidenceLevel,
    QualityFlag,
    ReconstructionCandidate,
    ReconstructionResult,
    SegmentReconstruction,
)
from app.transcription.reconstruction.validation import validate_candidate
from app.transcription.reconstruction.windows import acoustic_evidence, build_reconstruction_window
from app.core.enums import ReconstructionStatus


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
            if health.availability.value != "AVAILABLE":
                results = tuple(self._fallback(index, segment, provider_error=True, status=ReconstructionStatus.PROVIDER_UNAVAILABLE, method=f"{health.provider}:{health.model or 'unknown'}") for index, segment in enumerate(segments))
                result = ReconstructionResult(results, _joined(results), fingerprint)
                return result
            memory = build_entity_memory(segments)
            requests = [
                _generation_request(segments, index, language, tuple(memory.occurrences))
                for index in range(len(segments))
            ]
            generated = self._provider.generate_candidates(requests)
            resolved = self._provider.resolve_candidates(
                [
                    ResolutionRequest(
                        request.segment_index,
                        request.raw_text,
                        request.previous,
                        request.following,
                        tuple(
                            _candidate_set(
                                request.raw_text,
                                segments[request.segment_index],
                                generated[request.segment_index],
                            )
                        ),
                    )
                    for request in requests
                ]
            )
            memory = build_entity_memory(segments)
            results = tuple(
                self._decide(index, segment, generated[index], resolved[index], memory)
                for index, segment in enumerate(segments)
            )
            result = ReconstructionResult(results, _joined(results), fingerprint)
        except (OSError, ProviderResponseError):
            results = tuple(
                self._fallback(index, segment, provider_error=True, status=ReconstructionStatus.PROVIDER_UNAVAILABLE)
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
        self, index: int, segment: Mapping[str, object], provider_error: bool = False,
        status: ReconstructionStatus | None = None, method: str | None = None,
    ) -> SegmentReconstruction:
        raw = str(segment.get("raw_text", segment.get("text", "")))
        corrected = str(segment.get("corrected_text", raw))
        operator_text = segment.get("operator_text")
        if operator_text:
            return SegmentReconstruction(index, raw, corrected, str(operator_text), None, False,
                1.0, ConfidenceLevel.HIGH, (), ReconstructionStatus.MANUAL_OVERRIDE,
                reconstruction_method="operator:manual")
        flags = (QualityFlag.RECONSTRUCTION_PROVIDER_ERROR,) if provider_error else ()
        return SegmentReconstruction(index, raw, corrected, corrected, None, False, 0.0,
            ConfidenceLevel.LOW, flags, status or (ReconstructionStatus.PROVIDER_UNAVAILABLE if provider_error else ReconstructionStatus.UNCHANGED_HIGH_CONFIDENCE),
            reconstruction_method=method)

    def _decide(
        self,
        index: int,
        segment: Mapping[str, object],
        generated: list[ReconstructionCandidate],
        choice: ResolutionChoice,
        memory: SourceEntityMemory,
    ) -> SegmentReconstruction:
        raw = str(segment.get("raw_text", segment.get("text", "")))
        corrected = str(segment.get("corrected_text", raw))
        candidate = next(
            (
                item
                for item in _candidate_set(raw, segment, generated)
                if item.candidate_id == choice.candidate_id
            ),
            None,
        )
        routing = route_segment(build_reconstruction_window([segment], 0))
        if candidate is None or candidate.candidate_id in {"raw", "stage25"}:
            return SegmentReconstruction(index, raw, corrected, corrected, None, False, 0.0,
                ConfidenceLevel.HIGH, (), ReconstructionStatus.NOT_REQUIRED if routing.priority.value == "leave" else ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
                routing_score=routing.evidence.score, routing_reasons=(routing.reason,), focus_spans=routing.focus_spans)
        validation = validate_candidate(raw, candidate, memory)
        if not validation.accepted:
            return SegmentReconstruction(index, raw, corrected, corrected, candidate.text, False, 0.0,
                ConfidenceLevel.LOW, (QualityFlag.LOW_CONFIDENCE_UNRESOLVED,), ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
                routing_score=routing.evidence.score, routing_reasons=(routing.reason,), focus_spans=routing.focus_spans,
                validated_changes=candidate.changes, candidate_id=candidate.candidate_id)
        decision = decide_candidate(
            phonetic_similarity=validation.phonetic_similarity,
            resolution=choice.scores,
            raw_acoustic_confidence=acoustic_evidence(segment).confidence,
            edit_ratio=validation.edit_ratio,
            margin=choice.margin,
            token_delta=validation.token_delta,
        )
        text = candidate.text if decision.applied else corrected
        flags = (QualityFlag.MULTIWORD_RECONSTRUCTION,) if len(raw.split()) > 1 or len(candidate.text.split()) > 1 else ()
        status = ReconstructionStatus.APPLIED if decision.applied else ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
        return SegmentReconstruction(
            index,
            raw,
            corrected,
            text,
            candidate.text,
            decision.applied,
            decision.score,
            decision.level,
            flags, status, routing_score=routing.evidence.score, routing_reasons=(routing.reason,),
            focus_spans=routing.focus_spans, validated_changes=candidate.changes,
            reconstruction_method=f"provider:{getattr(self._provider, 'model', 'unknown')}",
            candidate_id=candidate.candidate_id, confidence_margin=choice.margin,
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


def _generation_request(
    segments: Sequence[Mapping[str, object]],
    index: int,
    language: str | None = None,
    entity_forms: tuple[str, ...] = (),
) -> GenerationRequest:
    window = build_reconstruction_window(segments, index)
    return GenerationRequest(
        window=window,
        language=language,
        entity_forms=entity_forms,
        routing_decision=route_segment(window, language=language),
    )


def _candidate_set(
    raw: str, segment: Mapping[str, object], generated: list[ReconstructionCandidate]
) -> list[ReconstructionCandidate]:
    corrected = str(segment.get("corrected_text", raw))
    candidates = [ReconstructionCandidate("raw", raw)]
    if corrected != raw:
        candidates.append(ReconstructionCandidate("stage25", corrected))
    for candidate in generated:
        if candidate.text not in {item.text for item in candidates} and len(candidates) < 3:
            candidates.append(candidate)
    return candidates


def _joined(results: Sequence[SegmentReconstruction]) -> str:
    return " ".join(item.contextual_reconstructed_text for item in results).strip()
