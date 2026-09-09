"""Reusable bounded targeted-window refinement for future Stage 3/3.5 callers.

Whole-source ingestion runs at INDEX priority and defers all provider
reconstruction. When a short source-time region is actually selected for
candidate or final-clip work, a caller requests refinement through
``refine_transcript_window``, which operates only on the bounded requested
region, reuses the existing Stage 2.5 and Stage 2.7 provider/routing/validation/
checkpoint mechanisms, preserves raw ASR and all timestamps exactly, and never
silently mutates unrelated segments.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.enums import ReconstructionStatus, RefinementPriority
from app.models import Transcript, TranscriptChunk
from app.pipeline.fingerprints import reconstruction_target_fingerprint
from app.pipeline.stages import (
    _reconstruction_method,
    _segment_code_switch_suspected,
    _segment_needs_refinement,
    _segment_reconstruction_status,
    _target_cache_eligible,
)
from app.transcription.chunking import ChunkConfig, build_chunks
from app.transcription.normalization import normalize_transcript
from app.transcription.reconstruction.service import (
    ContextualReconstructor,
    select_final_text,
)
from app.transcription.reconstruction.status import aggregate_reconstruction_status
from app.transcription.reconstruction.types import (
    ConfidenceLevel,
    ReconstructionResult,
    SegmentReconstruction,
)

_DURATION_TOLERANCE_SECONDS = 1e-6


class RefinementError(ValueError):
    """A targeted refinement request is outside the permitted bounds."""


@dataclass(frozen=True)
class RefinementOutcome:
    """Structured result of one targeted window refinement.

    ``results`` are ordered by transcript segment index and carry refined text
    where accepted plus confidence, status, provider/routing evidence, unresolved
    state, and the target/window identity. ``accepted_indexes`` and
    ``unresolved_indexes`` are derived conveniences for downstream callers.
    """

    source_id: UUID
    priority: RefinementPriority
    start_time: float
    end_time: float
    target_indexes: tuple[int, ...]
    results: tuple[SegmentReconstruction, ...]
    fingerprint: str
    metadata: dict[str, object]

    @property
    def accepted_indexes(self) -> tuple[int, ...]:
        return tuple(
            result.segment_index
            for result in self.results
            if result.applied and result.confidence_level is ConfidenceLevel.HIGH
        )

    @property
    def unresolved_indexes(self) -> tuple[int, ...]:
        return tuple(
            result.segment_index
            for result in self.results
            if result.status
            in {
                ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
                ReconstructionStatus.PROVIDER_UNAVAILABLE,
                ReconstructionStatus.FAILED,
            }
            or result.escalation_reason
        )


def refine_transcript_window(
    session: Session,
    source_id: UUID,
    start_time: float,
    end_time: float,
    priority: RefinementPriority,
    reconstructor: ContextualReconstructor,
    *,
    max_targets: int | None = None,
    max_window_seconds: float | None = None,
    checkpoint: Callable[[dict[int, SegmentReconstruction], dict[str, object]], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> RefinementOutcome:
    """Refine only the bounded source-time window that intersects the request.

    The requested interval selects target segments from immutable source
    timestamps; a small bounded nearby context window is included in provider
    prompts without ever making context segments mutation targets. Results are
    persisted only to the selected segments and returned structured so a future
    Stage 3/3.5 caller can drive candidate adjudication without redesign.
    """

    from app.core.settings import get_settings

    settings = get_settings()
    if max_targets is None:
        max_targets = settings.reconstruction_refinement_max_targets
    if max_window_seconds is None:
        max_window_seconds = settings.reconstruction_refinement_max_window_seconds

    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise RefinementError("start_time and end_time must be finite")
    if start_time < 0.0 or end_time <= start_time:
        raise RefinementError("refinement window must satisfy 0 <= start_time < end_time")
    if end_time - start_time > max_window_seconds:
        raise RefinementError(
            f"refinement window of {end_time - start_time:.2f}s exceeds the "
            f"{max_window_seconds:.2f}s bound"
        )

    transcript = session.scalar(select(Transcript).where(Transcript.source_video_id == source_id))
    if transcript is None:
        raise RefinementError("source has no transcript to refine")

    duration = float(transcript.duration or 0.0)
    if duration > 0.0 and end_time > duration + _DURATION_TOLERANCE_SECONDS:
        raise RefinementError("refinement window end_time exceeds the transcript duration")

    segments = transcript.segments
    targets: tuple[int, ...] = tuple(
        index
        for index, segment in enumerate(segments)
        if float(segment.get("end", 0.0)) > start_time
        and float(segment.get("start", 0.0)) < end_time
    )
    if len(targets) > max_targets:
        raise RefinementError(
            f"refinement window intersects {len(targets)} segments, exceeding the "
            f"{max_targets} target bound; narrow the window"
        )

    reconstructor = reconstructor.with_priority(priority)
    reconstructor = reconstructor.with_orchestration(
        is_cancelled=is_cancelled, checkpoint=checkpoint
    )
    result = reconstructor.reconstruct(
        segments,
        language=transcript.language,
        transcription_fingerprint=transcript.input_fingerprint,
        correction_version=transcript.correction_version,
        target_indexes=targets,
        priority=priority,
    )

    identity = result.metadata.get("runtime_identity")
    if not isinstance(identity, dict):
        identity = reconstructor.runtime_identity()

    updated = list(segments)
    statuses: list[ReconstructionStatus] = []
    for reconstruction in result.segments:
        index = reconstruction.segment_index
        if not 0 <= index < len(updated):
            raise RefinementError("refinement produced an out-of-range segment index")
        status = _segment_reconstruction_status(updated[index], reconstruction)
        statuses.append(status)
        updated[index] = _apply_refinement_segment(
            updated[index],
            reconstruction,
            status,
            identity,
            updated,
            language=transcript.language,
            transcription_fingerprint=transcript.input_fingerprint,
            correction_version=transcript.correction_version,
            priority=priority,
        )

    transcript.segments = updated
    transcript.contextual_reconstructed_text = " ".join(
        str(
            segment.get("contextual_reconstructed_text")
            or segment.get("corrected_text")
            or segment.get("text")
            or ""
        )
        for segment in updated
    ).strip()
    transcript.final_text = " ".join(
        str(segment.get("final_text") or segment.get("corrected_text") or segment.get("text") or "")
        for segment in updated
    ).strip()
    transcript.normalized_text = normalize_transcript(transcript.final_text)
    applied = [item for item in result.segments if item.applied]
    transcript.reconstruction_fingerprint = result.fingerprint
    transcript.reconstruction_status = aggregate_reconstruction_status(statuses)
    transcript.reconstruction_confidence = (
        sum(item.confidence for item in applied) / len(applied) if applied else 0.0
    )
    transcript.reconstructed_segment_ratio = len(applied) / len(targets) if targets else 0.0
    transcript.reconstruction_method = (
        result.metadata.get("reconstruction_method")
        if isinstance(result.metadata.get("reconstruction_method"), str)
        else _reconstruction_method(result.segments, result.metadata)
    )
    transcript.reconstruction_version = "stage2.7-v1"
    transcript.reconstruction_metadata = _merge_refinement_metadata(
        transcript, result, priority, start_time, end_time, targets
    )
    session.execute(delete(TranscriptChunk).where(TranscriptChunk.transcript_id == transcript.id))
    session.add_all(
        TranscriptChunk(
            transcript_id=transcript.id,
            sequence=sequence,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            text=chunk.text,
            segment_indexes=chunk.segment_indexes,
            preceding_context=chunk.preceding_context,
            following_context=chunk.following_context,
        )
        for sequence, chunk in enumerate(build_chunks(updated, ChunkConfig()))
    )
    session.commit()
    session.refresh(transcript)

    return RefinementOutcome(
        source_id=source_id,
        priority=priority,
        start_time=start_time,
        end_time=end_time,
        target_indexes=targets,
        results=result.segments,
        fingerprint=result.fingerprint,
        metadata=transcript.reconstruction_metadata,
    )


def _apply_refinement_segment(
    segment: dict[str, object],
    reconstruction: SegmentReconstruction,
    status: ReconstructionStatus,
    identity: dict[str, object],
    segments: list[dict[str, object]],
    *,
    language: str | None,
    transcription_fingerprint: str,
    correction_version: str,
    priority: RefinementPriority,
) -> dict[str, object]:
    """Merge one targeted refinement outcome into its persisted segment record.

    The field set mirrors the whole-source executor's ``_apply_segment`` so the
    two persistence paths stay consistent and reusable downstream.
    """

    raw = str(segment.get("raw_text", segment.get("text", "")))
    corrected = str(segment.get("corrected_text", raw))
    operator_text = segment.get("operator_text")
    operator = str(operator_text) if operator_text else None
    final_text = select_final_text(
        operator_text=operator,
        reconstructed=reconstruction.contextual_reconstructed_text,
        reconstruction_applied=reconstruction.applied,
        level=reconstruction.confidence_level,
        corrected=corrected,
        raw=raw,
    )
    target_fingerprint = reconstruction_target_fingerprint(
        provider_identity=identity,
        segments=segments,
        target_index=reconstruction.segment_index,
        language=language,
        transcription_fingerprint=transcription_fingerprint,
        correction_version=correction_version,
    )
    return {
        **segment,
        "contextual_reconstructed_text": reconstruction.contextual_reconstructed_text,
        "reconstruction_candidate_text": reconstruction.candidate_text,
        "reconstruction_applied": reconstruction.applied,
        "reconstruction_confidence": reconstruction.confidence,
        "reconstruction_confidence_level": reconstruction.confidence_level.value,
        "reconstruction_quality_flags": [flag.value for flag in reconstruction.quality_flags],
        "routing_score": reconstruction.routing_score,
        "routing_reasons": list(reconstruction.routing_reasons),
        "focus_spans": [
            {
                "word": word.text,
                "start": word.start,
                "end": word.end,
                "probability": word.probability,
            }
            for word in reconstruction.focus_spans
        ],
        "reconstruction_status": status.value,
        "reconstruction_method": reconstruction.reconstruction_method,
        "final_text": final_text,
        "normalized_text": normalize_transcript(final_text),
        "reconstruction_route": reconstruction.route,
        "routing_evidence": list(reconstruction.routing_evidence),
        "local_attempted": reconstruction.local_attempted,
        "local_result_state": reconstruction.local_result_state,
        "gemini_attempted": reconstruction.gemini_attempted,
        "gemini_result_state": reconstruction.gemini_result_state,
        "final_provider": reconstruction.final_provider,
        "escalation_reason": reconstruction.escalation_reason,
        "near_acceptance": reconstruction.near_acceptance,
        "refinement_priority": priority.value,
        "needs_refinement": _segment_needs_refinement(status, reconstruction),
        "code_switch_suspected": _segment_code_switch_suspected(segment),
        "reconstruction_target_fingerprint": target_fingerprint,
        "reconstruction_cache_eligible": _target_cache_eligible(reconstruction),
    }


def _merge_refinement_metadata(
    transcript: Transcript,
    result: ReconstructionResult,
    priority: RefinementPriority,
    start_time: float,
    end_time: float,
    targets: tuple[int, ...],
) -> dict[str, object]:
    """Extend prior transcript metadata with the targeted refinement record."""

    metadata: dict[str, object] = dict(transcript.reconstruction_metadata)
    if priority is not RefinementPriority.INDEX:
        # A prior whole-source INDEX run's deferred markers are stale once an
        # active refinement touched the transcript.
        metadata.pop("index_deferred", None)
        metadata.pop("index_deferred_segments", None)
    result_metadata = result.metadata
    metadata["priority"] = priority.value
    metadata["cache_eligible"] = result_metadata.get("cache_eligible") is True
    metadata["window"] = {"start": start_time, "end": end_time}
    metadata["target_indexes"] = list(targets)
    if isinstance(result_metadata.get("runtime_identity"), dict):
        metadata["runtime_identity"] = result_metadata["runtime_identity"]
    if isinstance(result_metadata.get("routing_counts"), dict):
        metadata["routing_counts"] = result_metadata["routing_counts"]
    if isinstance(result_metadata.get("gemini_usage"), dict):
        metadata["gemini_usage"] = result_metadata["gemini_usage"]
    return metadata
