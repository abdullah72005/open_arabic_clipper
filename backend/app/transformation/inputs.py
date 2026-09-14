"""Bounded Stage 4.0 input assembly from persisted Stage 3 / Stage 3.5 evidence.

Reads only: Stage 3 candidate evidence, the selected Stage 3.5 refinement, the
source provenance, and a small bounded nearby context assembled from relevant
transcript segments. Invalidates nothing; rewrites nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import ContentType, RefinementPriority, RefinementStatus
from app.models import CandidateRefinement, ClipCandidate, SourceVideo, Transcript
from app.transformation.policy import Stage40Config
from app.transformation.types import TransformationInputs

_FINAL_READY = {
    RefinementStatus.FINAL_TRANSCRIPT_READY.value,
}
_CANDIDATE_USABLE = {
    RefinementStatus.CANDIDATE_REFINED.value,
    "NEEDS_MANUAL_TRANSCRIPT_REVIEW",
    RefinementStatus.PROVIDER_DEGRADED.value,
}
_TEXT_KEYS = ("final_text", "contextual_reconstructed_text", "corrected_text", "text", "raw_text")


class TransformationInputError(ValueError):
    """A Stage 4.0 input prerequisite is missing or stale."""


def _as_uuid(value: object) -> object:
    import uuid as _uuid

    if isinstance(value, _uuid.UUID):
        return value
    return _uuid.UUID(str(value))


def resolve_effective_refinement(
    session: Session,
    candidate: ClipCandidate,
) -> CandidateRefinement | None:
    """Prefer a usable FINAL_CLIP row; otherwise a usable CANDIDATE row.

    A queued, failed, cancelled, empty, or unresolved final row never hides a
    usable candidate-grade row. Final refinement is never enqueued here.
    """

    rows = list(
        session.scalars(
            select(CandidateRefinement).where(CandidateRefinement.clip_candidate_id == candidate.id)
        ).all()
    )
    final = _pick(rows, RefinementPriority.FINAL_CLIP)
    candidate_row = _pick(rows, RefinementPriority.CANDIDATE)
    if final is not None and _usable(final, final_ready=True):
        return final
    if candidate_row is not None and _usable(candidate_row, final_ready=False):
        return candidate_row
    return None


def _pick(
    rows: Sequence[CandidateRefinement], priority: RefinementPriority
) -> CandidateRefinement | None:
    matches = [row for row in rows if row.priority is priority]
    if not matches:
        return None
    matches.sort(key=lambda row: row.updated_at, reverse=True)
    return matches[0]


def _usable(row: CandidateRefinement, *, final_ready: bool) -> bool:
    status = row.status.value
    if not (row.final_transcript or "").strip():
        return False
    if final_ready:
        return status in _FINAL_READY
    return status in _CANDIDATE_USABLE


def _segment_text(segment: Mapping[str, object]) -> str:
    for key in _TEXT_KEYS:
        value = segment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _bounded_context(
    segments: Sequence[Mapping[str, object]],
    start_index: int,
    end_index: int,
    config: Stage40Config,
) -> tuple[str, ...]:
    radius = max(1, config.max_context_segments // 2)
    before = [
        _segment_text(segment) for segment in segments[max(0, start_index - radius) : start_index]
    ]
    after = [_segment_text(segment) for segment in segments[end_index + 1 : end_index + 1 + radius]]
    context: list[str] = []
    budget = config.max_context_characters
    for text in [*before, *after]:
        if not text:
            continue
        text = text[:budget]
        context.append(text)
        budget -= len(text)
        if budget <= 0 or len(context) >= config.max_context_segments:
            break
    return tuple(context)


def build_transformation_inputs(
    session: Session,
    candidate: ClipCandidate,
    refinement: CandidateRefinement,
    config: Stage40Config,
) -> TransformationInputs:
    source = session.get(SourceVideo, candidate.source_video_id)
    if source is None:
        raise TransformationInputError("source is missing for candidate")
    transcript = session.scalar(
        select(Transcript).where(Transcript.source_video_id == candidate.source_video_id)
    )
    segments: Sequence[Mapping[str, object]] = ()
    if transcript is not None and isinstance(transcript.segments, list):
        segments = [item for item in transcript.segments if isinstance(item, Mapping)]
    context = _bounded_context(
        segments,
        int(candidate.start_segment_index),
        int(candidate.end_segment_index),
        config,
    )
    code_switch = dict(refinement.code_switch_evidence or {})
    return TransformationInputs(
        candidate_id=str(candidate.id),
        candidate_key=candidate.candidate_key,
        source_id=str(candidate.source_video_id),
        disposition=candidate.disposition.value,
        content_type=candidate.primary_content_type,
        secondary_content_types=tuple(
            _content_type(item) for item in (candidate.secondary_content_types or [])
        ),
        coarse_start=float(candidate.start_time),
        coarse_end=float(candidate.end_time),
        refined_start=refinement.refined_start,
        refined_end=refinement.refined_end,
        transcript=(refinement.final_transcript or "").strip()[: config.max_context_characters],
        transcript_confidence=float(refinement.confidence),
        refinement_confidence=float(refinement.confidence),
        word_timestamps=tuple(
            item for item in (refinement.word_timestamps or []) if isinstance(item, Mapping)
        ),
        unresolved_spans=tuple(
            item for item in (refinement.unresolved_spans or []) if isinstance(item, Mapping)
        ),
        entity_evidence=tuple(
            item for item in (refinement.entity_evidence or []) if isinstance(item, Mapping)
        ),
        dialect_profile=refinement.dialect_profile or candidate.dialect_profile,
        dialect_confidence=float(
            refinement.dialect_confidence or candidate.dialect_confidence or 0.0
        ),
        code_switch=code_switch,
        context_segments=context,
        clip_score=float(candidate.clip_score),
        short_form_score=float(candidate.short_form_score),
        moment_density_score=float(candidate.moment_density_score),
        ending_quality_score=float(candidate.ending_quality_score),
        loopability_score=float(candidate.loopability_score),
        idea_summary=candidate.idea_summary or "",
        topic_summary=candidate.topic_summary or "",
        hooks=tuple(item for item in (candidate.hooks or []) if isinstance(item, Mapping)),
        rights_risk=candidate.rights_risk,
        originality_risk=candidate.originality_risk,
        rights_status=source.rights_status.value,
        media_origin=source.media_origin.value,
        provenance_snapshot=dict(candidate.provenance_snapshot or {}),
        refinement_priority=refinement.priority.value,
        refinement_quality_level=refinement.quality_level,
        refinement_status=refinement.status.value,
        refinement_output_fingerprint=refinement.output_fingerprint or "",
        stage3_analysis_fingerprint=candidate.analysis_fingerprint or "",
        stage3_policy_version=candidate.policy_version or "stage3-v1",
    )


def _content_type(value: object) -> ContentType:
    try:
        return ContentType(str(value))
    except ValueError:
        return ContentType.OTHER
