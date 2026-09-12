"""Typed read-only Stage 3.5 -> Stage 4 handoff builder.

This performs no transformation planning. It exposes the refined transcript and
exact boundaries truthfully: a valid FINAL_CLIP row is preferred, otherwise the
candidate-grade output is exposed and clearly labeled as candidate quality.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import RefinementPriority, RefinementStatus
from app.models import CandidateRefinement, ClipCandidate

_FINAL_READY = {RefinementStatus.FINAL_TRANSCRIPT_READY.value}


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def build_stage4_handoff(session: Session, candidate_id: uuid.UUID | str) -> dict[str, Any] | None:
    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        return None
    refinements = list(
        session.scalars(
            select(CandidateRefinement).where(CandidateRefinement.clip_candidate_id == candidate.id)
        ).all()
    )
    final_row = _pick(refinements, RefinementPriority.FINAL_CLIP)
    candidate_row = _pick(refinements, RefinementPriority.CANDIDATE)
    selected = final_row if final_row is not None else candidate_row
    if selected is None:
        refined: dict[str, Any] = {
            "quality_level": None,
            "final_ready": False,
            "status": None,
            "effective_transcript": None,
            "refined_start": candidate.start_time,
            "refined_end": candidate.end_time,
            "confidence": 0.0,
            "word_timestamps": [],
            "unresolved_spans": [],
            "manual_review_required": False,
            "dialect_profile": candidate.dialect_profile,
            "dialect_confidence": candidate.dialect_confidence,
            "code_switch": {"suspected": candidate.code_switch_suspected, "recovered": []},
            "entity_evidence": [],
            "provider_evidence": {},
            "routing_evidence": {},
            "input_fingerprint": None,
            "output_fingerprint": None,
            "component_fingerprints": {},
        }
    else:
        final_ready = (
            selected.priority is RefinementPriority.FINAL_CLIP
            and selected.status.value in _FINAL_READY
        )
        refined = {
            "quality_level": selected.quality_level,
            "final_ready": final_ready,
            "status": selected.status.value,
            "effective_transcript": selected.final_transcript,
            "refined_start": selected.refined_start,
            "refined_end": selected.refined_end,
            "confidence": selected.confidence,
            "word_timestamps": list(selected.word_timestamps or []),
            "unresolved_spans": [
                span
                for span in (selected.unresolved_spans or [])
                if isinstance(span, dict) and span.get("resolution_state") != "RESOLVED"
            ],
            "manual_review_required": selected.status.value
            == RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW.value,
            "dialect_profile": selected.dialect_profile,
            "dialect_confidence": selected.dialect_confidence,
            "code_switch": dict(selected.code_switch_evidence or {}),
            "entity_evidence": list(selected.entity_evidence or []),
            "provider_evidence": dict(selected.provider_evidence or {}),
            "routing_evidence": dict(selected.routing_evidence or {}),
            "input_fingerprint": selected.input_fingerprint,
            "output_fingerprint": selected.output_fingerprint,
            "component_fingerprints": dict(selected.component_fingerprints or {}),
        }
    return {
        "candidate": {
            "id": str(candidate.id),
            "candidate_key": candidate.candidate_key,
            "source_id": str(candidate.source_video_id),
            "disposition": candidate.disposition.value,
            "coarse_start": candidate.start_time,
            "coarse_end": candidate.end_time,
            "start_segment_index": candidate.start_segment_index,
            "end_segment_index": candidate.end_segment_index,
            "segment_indexes": list(candidate.segment_indexes or []),
        },
        "stage3": {
            "clip_score": candidate.clip_score,
            "short_form_score": candidate.short_form_score,
            "moment_density_score": candidate.moment_density_score,
            "ending_quality_score": candidate.ending_quality_score,
            "loopability_score": candidate.loopability_score,
            "uncertainty_severity": candidate.uncertainty_severity,
            "primary_content_type": candidate.primary_content_type.value,
            "secondary_content_types": list(candidate.secondary_content_types or []),
            "hooks": list(candidate.hooks or []),
            "idea_novelty_score": candidate.idea_novelty_score,
            "topic_novelty_score": candidate.topic_novelty_score,
            "provenance_snapshot": dict(candidate.provenance_snapshot or {}),
            "rights_risk": candidate.rights_risk.value,
            "originality_risk": candidate.originality_risk.value,
        },
        "refinement": refined,
        "stage4_implemented": False,
    }


def _pick(
    refinements: list[CandidateRefinement], priority: RefinementPriority
) -> CandidateRefinement | None:
    matches = [row for row in refinements if row.priority is priority]
    if not matches:
        return None
    matches.sort(key=lambda row: row.updated_at, reverse=True)
    return matches[0]
