"""Batch contextual reconstruction with safe Stage 2.5 fallback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from app.transcription.reconstruction.confidence import decide_candidate
from app.transcription.reconstruction.entities import build_entity_memory
from app.transcription.reconstruction.providers import (
    GenerationRequest,
    ReconstructionProvider,
    ResolutionRequest,
)
from app.transcription.reconstruction.types import (
    ConfidenceLevel,
    QualityFlag,
    ReconstructionCandidate,
    ReconstructionResult,
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
        requests = [_generation_request(segments, index) for index in range(len(segments))]
        try:
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
        except Exception:
            results = tuple(
                self._fallback(index, segment, provider_error=True)
                for index, segment in enumerate(segments)
            )
            return ReconstructionResult(results, _joined(results), fingerprint)
        memory = build_entity_memory(segments)
        results = tuple(
            self._decide(index, segment, generated[index], resolved[index], memory)
            for index, segment in enumerate(segments)
        )
        return ReconstructionResult(results, _joined(results), fingerprint)

    def _fallback(
        self, index: int, segment: Mapping[str, object], provider_error: bool = False
    ) -> SegmentReconstruction:
        raw = str(segment.get("raw_text", segment.get("text", "")))
        corrected = str(segment.get("corrected_text", raw))
        flags = (QualityFlag.RECONSTRUCTION_PROVIDER_ERROR,) if provider_error else ()
        return SegmentReconstruction(
            index, raw, corrected, corrected, None, False, 0.0, ConfidenceLevel.LOW, flags
        )

    def _decide(
        self,
        index: int,
        segment: Mapping[str, object],
        generated: list[ReconstructionCandidate],
        choice: object,
        memory: object,
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
        if candidate is None or candidate.candidate_id in {"raw", "stage25"}:
            return self._fallback(index, segment)
        validation = validate_candidate(raw, candidate, memory)
        if not validation.accepted:
            return self._fallback(index, segment)
        decision = decide_candidate(
            phonetic_similarity=validation.phonetic_similarity,
            resolution=choice.scores,
            raw_acoustic_confidence=acoustic_evidence(segment).confidence,
            edit_ratio=validation.edit_ratio,
            margin=0.12,
            token_delta=validation.token_delta,
        )
        text = candidate.text if decision.applied else corrected
        return SegmentReconstruction(
            index,
            raw,
            corrected,
            text,
            candidate.text,
            decision.applied,
            decision.score,
            decision.level,
            (),
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


def _generation_request(segments: Sequence[Mapping[str, object]], index: int) -> GenerationRequest:
    window = build_reconstruction_window(segments, index)
    current = next(item for item in window.segments if item.segment_index == index)
    position = next(
        position for position, item in enumerate(window.segments) if item.segment_index == index
    )
    return GenerationRequest(
        index,
        current.raw_text,
        tuple(item.raw_text for item in window.segments[:position]),
        tuple(item.raw_text for item in window.segments[position + 1 :]),
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
