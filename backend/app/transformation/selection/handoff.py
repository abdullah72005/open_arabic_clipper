"""Read-only Stage 4.3 execution handoff builder.

Composes exactly one selected plan (or none) for a later execution boundary. It
never renders, synthesizes speech, uploads, ingests, transcribes, or calls a
provider, and it never claims platform safety or monetization. Execution
readiness is a live property of current FINAL_CLIP evidence, never a persisted
snapshot.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import RefinementPriority, RefinementStatus, TransformationSelectionStatus
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    TransformationPlan,
    TransformationPlanSet,
)
from app.transformation.governance.handoff import (
    FRESHNESS_VERIFIED_CURRENT,
    build_stage4_3_handoff,
)
from app.transformation.planning.queue import get_plan_set_for_candidate
from app.transformation.selection.service import as_uuid, read_selection

_FINAL_READY = {RefinementStatus.FINAL_TRANSCRIPT_READY.value}


def build_execution_handoff(
    session: Session, candidate_id: uuid.UUID | str
) -> dict[str, Any] | None:
    candidate = session.get(ClipCandidate, as_uuid(candidate_id))
    if candidate is None:
        return None

    view = read_selection(session, candidate.id)
    selection = view.row if view is not None else None
    stage43 = build_stage4_3_handoff(session, candidate.id) or {}
    plan_set = get_plan_set_for_candidate(session, candidate.id)
    plan_set_row = session.get(TransformationPlanSet, plan_set.id) if plan_set is not None else None
    planning_refinement = _planning_refinement(plan_set_row)
    final_refinement = _usable_final_refinement(session, candidate.id)

    selected_plan_id = selection.selected_plan_id if selection is not None else None
    selected_plan = (
        session.get(TransformationPlan, selected_plan_id) if selected_plan_id is not None else None
    )
    governance_snapshot = (
        dict(selection.selected_governance_snapshot or {}) if selection is not None else {}
    )

    same_identity = bool(
        final_refinement is not None
        and planning_refinement is not None
        and final_refinement.id == as_uuid(planning_refinement.get("id"))
    )
    effective = bool(view.effective) if view is not None else False
    live_freshness = view.live_freshness if view is not None else "NOT_CURRENT"

    readiness = _readiness(
        selection=selection,
        selected_plan=selected_plan,
        effective=effective,
        live_freshness=live_freshness,
        final_refinement=final_refinement,
        planning_refinement=planning_refinement,
        governance_snapshot=governance_snapshot,
    )

    base: dict[str, Any] = {
        "candidate": {
            "id": str(candidate.id),
            "candidate_key": candidate.candidate_key,
            "source_id": str(candidate.source_video_id),
            "disposition": candidate.disposition.value,
        },
        "selection": _selection_summary(selection, view, live_freshness),
        "rights_and_provenance": {
            "rights_status": (
                candidate.source_video.rights_status.value
                if candidate.source_video is not None
                else None
            ),
            "media_origin": (
                candidate.source_video.media_origin.value
                if candidate.source_video is not None
                else None
            ),
            "provenance": dict(candidate.provenance_snapshot or {}),
            "rights_risk": candidate.rights_risk.value,
            "platform_originality_risk": candidate.originality_risk.value,
        },
        "stage40": {
            "analysis_id": (stage43.get("stage40") or {}).get("analysis_id"),
            "output_fingerprint": (stage43.get("stage40") or {}).get("output_fingerprint"),
            "risk_snapshot": dict((stage43.get("stage40") or {}).get("risk_snapshot") or {}),
        },
        "stage41": dict(stage43.get("stage41") or {}),
        "stage42": {
            "governance_set_id": (stage43.get("governance_set") or {}).get("id"),
            "input_fingerprint": (stage43.get("governance_set") or {}).get("input_fingerprint"),
            "output_fingerprint": (stage43.get("governance_set") or {}).get("output_fingerprint"),
            "policy_version": (stage43.get("governance_set") or {}).get("policy_version"),
            "platform_policy_profile_version": (stage43.get("governance_set") or {}).get(
                "platform_policy_profile_version"
            ),
        },
        "selected_plan": None,
        "strategy": None,
        "blocks": [],
        "hero": None,
        "preservation_constraints": [],
        "original_value_kinds": [],
        "original_value_reasons": [],
        "narration": None,
        "target_context": dict(plan_set_row.target_context or {}) if plan_set_row else {},
        "source_dialect": {},
        "verification": {},
        "duration_evidence": {},
        "governance": {
            "hard_gates": list(governance_snapshot.get("hard_gates") or []),
            "dimensions": dict(governance_snapshot.get("dimensions") or {}),
            "reason_codes": list(governance_snapshot.get("reason_codes") or []),
            "warnings": list(governance_snapshot.get("warnings") or []),
            "remediation": list(governance_snapshot.get("remediation") or []),
            "provider_evidence": dict(governance_snapshot.get("provider_evidence") or {}),
        },
        "platform_risk": {
            "policy_profile_version": governance_snapshot.get("platform_policy_profile_version"),
            "youtube": dict((governance_snapshot.get("platform_risk") or {}).get("youtube") or {}),
            "facebook": dict(
                (governance_snapshot.get("platform_risk") or {}).get("facebook") or {}
            ),
        },
        "planning_refinement": (
            {
                "id": planning_refinement.get("id"),
                "priority": planning_refinement.get("priority"),
                "quality_level": planning_refinement.get("quality_level"),
                "output_fingerprint": planning_refinement.get("output_fingerprint"),
            }
            if planning_refinement is not None
            else None
        ),
        "final_clip_refinement": (
            {
                "id": str(final_refinement.id),
                "status": final_refinement.status.value,
                "quality_level": final_refinement.quality_level,
                "output_fingerprint": final_refinement.output_fingerprint,
            }
            if final_refinement is not None
            else None
        ),
        "final_clip_refinement_available": final_refinement is not None,
        "final_clip_refinement_required": not (final_refinement is not None and same_identity),
        "selection_based_on_final_clip": bool(
            planning_refinement is not None
            and planning_refinement.get("priority") == RefinementPriority.FINAL_CLIP.value
        ),
        "same_refinement_identity": same_identity,
        "compatibility_recheck_required": readiness
        == "REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK",
        "upstream_chain_stale": not effective,
        "execution_readiness": readiness,
        "fingerprints": {
            "selection_input_fingerprint": (
                selection.input_fingerprint if selection is not None else ""
            ),
            "selection_output_fingerprint": (
                selection.output_fingerprint if selection is not None else ""
            ),
            "governance_input_fingerprint": (
                selection.governance_input_fingerprint if selection is not None else ""
            ),
            "governance_output_fingerprint": (
                selection.governance_output_fingerprint if selection is not None else ""
            ),
            "selected_plan_fingerprint": (
                selection.selected_plan_fingerprint if selection is not None else ""
            ),
        },
        "stage5_implemented": False,
        "stage6_tts_implemented": False,
    }

    if selected_plan is not None:
        base["selected_plan"] = {
            "plan_id": str(selected_plan.id),
            "plan_key": selected_plan.plan_key,
            "plan_output_fingerprint": selected_plan.plan_output_fingerprint,
            "governance_result_id": (
                str(selection.selected_governance_result_id) if selection is not None else None
            ),
        }
        base["strategy"] = {
            "type": selected_plan.strategy_type.value,
            "intensity": selected_plan.intensity.value,
        }
        base["blocks"] = [dict(block) for block in (selected_plan.blocks or [])]
        base["hero"] = {
            "block_index": selected_plan.hero_block_index,
            "source_start": selected_plan.hero_source_start,
            "source_end": selected_plan.hero_source_end,
            "appearance_time": selected_plan.hero_appearance_time,
        }
        base["preservation_constraints"] = list(selected_plan.preservation_constraints or [])
        base["original_value_kinds"] = list(selected_plan.original_value_kinds or [])
        base["original_value_reasons"] = list(selected_plan.original_value_reasons or [])
        base["narration"] = {
            "need": selected_plan.narration_need,
            "requirements": dict(selected_plan.narration_requirements or {}),
        }
        base["source_dialect"] = dict(selected_plan.source_dialect or {})
        base["verification"] = {
            "state": (governance_snapshot.get("verification") or {}).get("claim_state"),
            "claims": list((governance_snapshot.get("verification") or {}).get("claims") or []),
            "unresolved": bool((governance_snapshot.get("verification") or {}).get("unresolved")),
            "dependencies": list(selected_plan.external_fact_dependencies or []),
        }
        base["duration_evidence"] = dict(selected_plan.derived_durations or {})
    else:
        base["verification"] = {}
        base["duration_evidence"] = {}

    return base


def _usable_final_refinement(
    session: Session, candidate_id: uuid.UUID
) -> CandidateRefinement | None:
    rows: list[CandidateRefinement] = list(
        session.scalars(
            select(CandidateRefinement).where(CandidateRefinement.clip_candidate_id == candidate_id)
        ).all()
    )
    finals = [row for row in rows if row.priority is RefinementPriority.FINAL_CLIP]
    if not finals:
        return None
    finals.sort(key=lambda row: row.updated_at, reverse=True)
    for row in finals:
        if (row.final_transcript or "").strip() and row.status.value in _FINAL_READY:
            return row
    return None


def _planning_refinement(plan_set: TransformationPlanSet | None) -> dict[str, object] | None:
    if plan_set is None or plan_set.refinement_id is None:
        return None
    return {
        "id": str(plan_set.refinement_id),
        "priority": plan_set.refinement_priority,
        "quality_level": plan_set.refinement_quality_level,
        "output_fingerprint": "",
    }


def _readiness(
    *,
    selection: Any,
    selected_plan: Any,
    effective: bool,
    live_freshness: str,
    final_refinement: CandidateRefinement | None,
    planning_refinement: dict[str, object] | None,
    governance_snapshot: dict[str, object],
) -> str:
    if selection is None or selected_plan is None:
        return "BLOCKED"
    if selection.status in {
        TransformationSelectionStatus.NO_SELECTABLE_PLAN,
        TransformationSelectionStatus.SELECTION_DEFERRED,
        TransformationSelectionStatus.STALE_SELECTION_INPUT,
    }:
        return "BLOCKED"
    verification = governance_snapshot.get("verification")
    if isinstance(verification, dict) and verification.get("unresolved"):
        return "BLOCKED"
    same_identity = bool(
        final_refinement is not None
        and planning_refinement is not None
        and final_refinement.id == as_uuid(planning_refinement.get("id"))
    )
    if final_refinement is not None and same_identity:
        if not effective or live_freshness != FRESHNESS_VERIFIED_CURRENT:
            return "BLOCKED"
        return "READY_FOR_EXECUTION_PREP"
    # A CANDIDATE-based selection with a newer usable FINAL_CLIP must fail
    # closed toward a compatibility check rather than become render-ready, even
    # when the newer FINAL_CLIP truthfully invalidates the frozen upstream chain.
    if final_refinement is not None:
        return "REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK"
    if not effective or live_freshness != FRESHNESS_VERIFIED_CURRENT:
        return "BLOCKED"
    return "READY_FOR_FINAL_REFINEMENT"


def _selection_summary(selection: Any, view: Any, live_freshness: str) -> dict[str, object] | None:
    if selection is None:
        return {
            "selection_id": None,
            "status": None,
            "reason_codes": [],
            "selected_with_caution": False,
            "live_freshness": live_freshness,
            "current": False,
            "effective": False,
            "historical": False,
        }
    return {
        "selection_id": str(selection.id),
        "status": selection.status.value,
        "reason_codes": list(selection.selection_reason_codes or []),
        "selected_with_caution": bool(selection.selected_with_caution),
        "live_freshness": live_freshness,
        "current": bool(selection.is_current),
        "effective": bool(view.effective) if view is not None else False,
        "historical": not bool(selection.is_current),
    }


__all__ = ["build_execution_handoff"]
