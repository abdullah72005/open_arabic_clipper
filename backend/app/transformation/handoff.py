"""Typed read-only Stage 4.0 -> Stage 4.1 handoff builder.

Performs no planning. Exposes eligibility, strategies, bounded evidence, and a
truthful stale/ready state. If current upstream input no longer matches the
analysis input fingerprint, the handoff reports stale rather than silently
combining current transcript data with old strategies.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.core.enums import StrategyDisposition
from app.core.settings import get_settings
from app.models import CandidateRefinement, ClipCandidate
from app.transformation.executor import build_transformation_executor
from app.transformation.queue import (
    get_analysis_for_candidate,
    list_strategies,
)


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def build_stage4_1_handoff(
    session: Session, candidate_id: uuid.UUID | str
) -> dict[str, Any] | None:
    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        return None
    analysis = get_analysis_for_candidate(session, candidate.id)
    base: dict[str, Any] = {
        "candidate": {
            "id": str(candidate.id),
            "candidate_key": candidate.candidate_key,
            "source_id": str(candidate.source_video_id),
            "disposition": candidate.disposition.value,
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
            "idea_summary": candidate.idea_summary,
            "topic_summary": candidate.topic_summary,
            "rights_risk": candidate.rights_risk.value,
            "originality_risk": candidate.originality_risk.value,
            "provenance_snapshot": dict(candidate.provenance_snapshot or {}),
        },
        "analysis_id": None,
        "output_fingerprint": None,
        "selected_refinement": None,
        "effective_transcript": None,
        "transcript_confidence": None,
        "transcript_status": None,
        "refined_bounds": None,
        "word_timing_evidence": [],
        "unresolved_spans": [],
        "dialect": {
            "profile": candidate.dialect_profile,
            "confidence": candidate.dialect_confidence,
            "code_switch_suspected": candidate.code_switch_suspected,
        },
        "entity_evidence": [],
        "eligibility_outcome": None,
        "eligibility_reasons": [],
        "transformation_necessity": None,
        "transformation_potential": None,
        "platform_risk": {},
        "recommended_strategies": [],
        "rejected_strategies": [],
        "intensity": None,
        "retention_preservation": None,
        "source_moment_damage_risk": None,
        "source_dominance_risk": None,
        "added_value_density": None,
        "originality_potential": None,
        "verification_requirements": [],
        "provider": {},
        "cache_eligible": False,
        "current": False,
        "stale": False,
        "ready_for_stage4_1": False,
        "stage4_1_implemented": False,
    }
    if analysis is None:
        base["reason"] = "NO_ANALYSIS"
        return base

    base["analysis_id"] = str(analysis.id)
    base["output_fingerprint"] = analysis.output_fingerprint
    base["eligibility_outcome"] = (
        analysis.eligibility_outcome.value if analysis.eligibility_outcome else None
    )
    base["eligibility_reasons"] = list(analysis.eligibility_reasons or [])
    assessments = analysis.assessments or {}
    base["transformation_necessity"] = assessments.get("transformation_necessity")
    base["transformation_potential"] = assessments.get("transformation_potential")
    base["platform_risk"] = dict(analysis.platform_risk or {})
    base["intensity"] = (
        analysis.transformation_intensity.value if analysis.transformation_intensity else None
    )
    base["retention_preservation"] = assessments.get("retention_preservation")
    base["source_moment_damage_risk"] = assessments.get("source_moment_damage_risk")
    base["source_dominance_risk"] = assessments.get("source_dominance_risk")
    base["added_value_density"] = assessments.get("added_value_density")
    base["originality_potential"] = assessments.get("originality_potential")
    base["cache_eligible"] = bool(analysis.cache_eligible)
    base["provider"] = {
        "mode": analysis.provider_mode.value,
        "status": analysis.provider_status,
        "identity": dict(analysis.provider_identity or {}),
        "input_fingerprint": analysis.provider_input_fingerprint,
    }

    refinement = (
        session.get(CandidateRefinement, analysis.refinement_id) if analysis.refinement_id else None
    )
    if refinement is not None:
        base["selected_refinement"] = {
            "id": str(refinement.id),
            "priority": refinement.priority.value,
            "quality_level": refinement.quality_level,
            "status": refinement.status.value,
            "confidence": refinement.confidence,
        }
        base["effective_transcript"] = refinement.final_transcript
        base["transcript_confidence"] = refinement.confidence
        base["transcript_status"] = refinement.status.value
        base["refined_bounds"] = {
            "start": refinement.refined_start,
            "end": refinement.refined_end,
        }
        base["word_timing_evidence"] = list(refinement.word_timestamps or [])
        base["unresolved_spans"] = [
            span
            for span in (refinement.unresolved_spans or [])
            if isinstance(span, dict) and span.get("resolution_state") != "RESOLVED"
        ]
        base["entity_evidence"] = list(refinement.entity_evidence or [])
        base["dialect"] = {
            "profile": refinement.dialect_profile or candidate.dialect_profile,
            "confidence": refinement.dialect_confidence or candidate.dialect_confidence,
            "code_switch": dict(refinement.code_switch_evidence or {}),
            "code_switch_suspected": candidate.code_switch_suspected,
        }

    strategies = list_strategies(session, analysis.id)
    recommended = []
    rejected = []
    verification: list[str] = []
    for row in strategies:
        if not row.is_current:
            continue
        entry = _strategy_dict(row)
        if row.disposition is StrategyDisposition.RECOMMENDED:
            recommended.append(entry)
            verification.extend(row.verification_requirements or [])
        else:
            rejected.append(entry)
    base["recommended_strategies"] = recommended
    base["rejected_strategies"] = rejected
    base["verification_requirements"] = list(dict.fromkeys(verification))
    stale, current = _staleness(session, candidate, analysis)
    base["stale"] = stale
    base["current"] = current
    base["ready_for_stage4_1"] = bool(
        current
        and not stale
        and analysis.execution_status.value == "COMPLETE"
        and bool(recommended)
    )
    return base


def _strategy_dict(row: Any) -> dict[str, object]:
    return {
        "id": str(row.id),
        "strategy_key": row.strategy_key,
        "strategy_type": row.strategy_type.value,
        "disposition": row.disposition.value,
        "rank": row.rank,
        "intensity": row.intensity.value,
        "direction_summary": row.direction_summary,
        "added_value_focus": row.added_value_focus,
        "substantive_value_kind": row.substantive_value_kind.value,
        "source_moment_role": row.source_moment_role,
        "preservation_requirements": list(row.preservation_requirements or []),
        "retention_preservation": row.retention_preservation,
        "source_moment_damage_risk": row.source_moment_damage_risk,
        "added_value_density": row.added_value_density,
        "originality_potential": row.originality_potential,
        "source_dominance_risk": row.source_dominance_risk,
        "generic_filler_risk": row.generic_filler_risk,
        "redundant_commentary_risk": row.redundant_commentary_risk,
        "template_staleness_risk": row.template_staleness_risk,
        "external_verification_requirement": row.external_verification_requirement.value,
        "verification_requirements": list(row.verification_requirements or []),
        "rejection_reasons": list(row.rejection_reasons or []),
        "confidence": row.confidence,
        "origin": row.origin.value,
        "strategy_fingerprint": row.strategy_fingerprint,
    }


def _staleness(session: Session, candidate: ClipCandidate, analysis: Any) -> tuple[bool, bool]:
    """Compare the stored input fingerprint to the current runtime identity.

    Freshness uses the same settings-derived Stage 4.0 config, provider mode, and
    provider runtime identity that execution uses, so a just-completed analysis
    is not immediately stale and a relevant policy/config/provider/model/prompt
    change is detected. Unrelated rendering/publishing settings are absent from
    the fingerprint entirely.
    """

    if not analysis.input_fingerprint:
        return False, False
    try:
        executor = build_transformation_executor(session, get_settings())
        current = executor.input_fingerprint(candidate)
    except Exception:
        return False, False
    if not current:
        return False, False
    return (current != analysis.input_fingerprint), True
