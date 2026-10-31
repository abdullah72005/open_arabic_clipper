"""Small shared mapping from a scored Stage 3 draft to candidate row fields."""

from __future__ import annotations

from app.candidates.novelty import idea_signature, topic_signature
from app.candidates.types import CandidateDraft
from app.models import ClipCandidate


def apply_candidate_fields(row: ClipCandidate, draft: CandidateDraft) -> None:
    """Copy scored draft metadata while leaving identity/ownership to the caller."""

    proposal = draft.proposal
    scores = draft.scores
    row.disposition = draft.disposition
    row.start_time = proposal.start_time
    row.end_time = proposal.end_time
    row.start_segment_index = proposal.start_segment_index
    row.end_segment_index = proposal.end_segment_index
    row.segment_indexes = list(proposal.segment_indexes)
    row.transcript_excerpt = draft.transcript_excerpt
    row.evidence_snapshot = dict(draft.evidence_snapshot)
    row.primary_content_type = draft.content.primary
    row.secondary_content_types = [item.value for item in draft.content.secondary]
    row.clip_score = scores.clip_score
    row.short_form_score = scores.short_form_score
    row.moment_density_score = scores.moment_density_score
    row.boredom_risk_score = scores.boredom_risk_score
    row.ending_quality_score = scores.ending_quality_score
    row.loopability_score = scores.loopability_score
    row.engagement_confidence = scores.engagement_confidence
    row.transcript_confidence = scores.transcript_confidence
    row.audio_confidence = scores.audio_confidence
    row.boundary_confidence = scores.boundary_confidence
    row.uncertainty_severity = scores.uncertainty_severity
    row.idea_novelty_score = scores.idea_novelty_score
    row.topic_novelty_score = scores.topic_novelty_score
    row.recent_semantic_similarity_risk = scores.recent_semantic_similarity_risk
    row.refinement_reasons = [reason.value for reason in draft.refinement_reasons]
    row.refinement_evidence = dict(draft.refinement_evidence)
    row.provenance_snapshot = dict(draft.provenance_snapshot)
    row.rights_risk = draft.rights_risk
    row.originality_risk = draft.originality_risk
    row.dialect_profile = draft.dialect_profile
    row.dialect_confidence = draft.dialect_confidence
    row.code_switch_suspected = draft.code_switch_suspected
    row.hooks = [hook.as_dict() for hook in draft.hooks]
    row.idea_summary = draft.idea_summary
    row.topic_summary = draft.topic_summary
    row.idea_signature = idea_signature(draft.idea_summary or draft.transcript_excerpt)
    row.topic_signature = topic_signature(draft.topic_summary or draft.transcript_excerpt)
    row.provider_input_fingerprint = draft.provider_input_fingerprint
    row.provider_evidence = dict(draft.provider_evidence)
