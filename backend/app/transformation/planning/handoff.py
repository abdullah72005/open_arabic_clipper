"""Typed read-only Stage 4.1 -> Stage 4.2 handoff builder.

Performs no governor scoring and no plan selection. Exposes the exact ordered
blocks, hero span, narration semantics, verification dependencies, and Stage 4.0
risk so Stage 4.2 never has to reconstruct block order or plan semantics from
prose.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.models import ClipCandidate
from app.transformation.planning.executor import build_transformation_planning_executor
from app.transformation.planning.queue import (
    get_plan_set_for_candidate,
    list_plans,
)


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _plan_dict(row: Any) -> dict[str, object]:
    return {
        "plan_id": str(row.id),
        "plan_key": row.plan_key,
        "plan_output_fingerprint": row.plan_output_fingerprint,
        "provider_input_fingerprint": row.provider_input_fingerprint,
        "status": row.status.value,
        "generation_rank": row.generation_rank,
        "planner_confidence": row.planner_confidence,
        "is_current": row.is_current,
        "strategy": {
            "id": str(row.strategy_candidate_id),
            "key": (row.strategy_snapshot or {}).get("strategy_key"),
            "type": row.strategy_type.value,
            "fingerprint": row.strategy_fingerprint,
            "intensity": row.intensity.value,
            "rank": (row.strategy_snapshot or {}).get("rank"),
            "direction_summary": (row.strategy_snapshot or {}).get("direction_summary"),
            "added_value_focus": (row.strategy_snapshot or {}).get("added_value_focus"),
            "preservation_requirements": list(
                (row.strategy_snapshot or {}).get("preservation_requirements") or []
            ),
            "external_verification_requirement": (row.strategy_snapshot or {}).get(
                "external_verification_requirement"
            ),
            "verification_requirements": list(
                (row.strategy_snapshot or {}).get("verification_requirements") or []
            ),
        },
        "source_dialect": dict(row.source_dialect or {}),
        "target_audience": dict(row.target_audience or {}),
        "blocks": list(row.blocks or []),
        "hero": {
            "block_index": row.hero_block_index,
            "source_start": row.hero_source_start,
            "source_end": row.hero_source_end,
            "appearance_time": row.hero_appearance_time,
        },
        "hook_payoff_evidence": dict(row.hook_payoff_evidence or {}),
        "original_value_kinds": list(row.original_value_kinds or []),
        "original_value_reasons": list(row.original_value_reasons or []),
        "narration": {
            "need": row.narration_need,
            "requirements": dict(row.narration_requirements or {}),
        },
        "verification_dependencies": list(row.external_fact_dependencies or []),
        "derived_durations": dict(row.derived_durations or {}),
        "required_context": list(row.required_context or []),
        "preservation_constraints": list(row.preservation_constraints or []),
        "degraded_rules": list(row.degraded_rules or []),
        "stage40_risk": dict(row.stage40_risk or {}),
        "planning_provider": {
            "origin": row.generation_origin.value,
            "evidence": dict(row.planning_provider_evidence or {}),
        },
    }


def build_stage4_2_handoff(
    session: Session, candidate_id: uuid.UUID | str
) -> dict[str, Any] | None:
    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        return None
    base: dict[str, Any] = {
        "candidate": {
            "id": str(candidate.id),
            "candidate_key": candidate.candidate_key,
            "source_id": str(candidate.source_video_id),
            "disposition": candidate.disposition.value,
        },
        "stage40": {},
        "selected_refinement": None,
        "plan_set": None,
        "plans": [],
        "current": False,
        "stale": False,
        "cache_eligible": False,
        "stage4_2_implemented": False,
        "stage4_3_implemented": False,
    }
    plan_set = get_plan_set_for_candidate(session, candidate.id)
    if plan_set is None:
        base["reason"] = "NO_PLAN_SET"
        return base
    base["stage40"] = {
        "analysis_id": str(plan_set.transformation_analysis_id),
        "snapshot": dict(plan_set.stage40_snapshot or {}),
    }
    if plan_set.refinement_id is not None:
        base["selected_refinement"] = {
            "id": str(plan_set.refinement_id),
            "priority": plan_set.refinement_priority,
            "quality_level": plan_set.refinement_quality_level,
        }
    base["plan_set"] = {
        "id": str(plan_set.id),
        "execution_status": plan_set.execution_status.value,
        "semantic_outcome": (
            plan_set.planning_outcome.value if plan_set.planning_outcome else None
        ),
        "outcome_reasons": list(plan_set.outcome_reasons or []),
        "provider_mode": plan_set.provider_mode.value,
        "provider_status": plan_set.provider_status,
        "provider_identity": dict(plan_set.provider_identity or {}),
        "provider_evidence": dict(plan_set.provider_evidence or {}),
        "strategy_attempts": list(plan_set.strategy_attempts or []),
        "input_fingerprint": plan_set.input_fingerprint,
        "output_fingerprint": plan_set.output_fingerprint,
        "target_context": dict(plan_set.target_context or {}),
    }
    base["cache_eligible"] = bool(plan_set.cache_eligible)
    base["plans"] = [_plan_dict(row) for row in list_plans(session, plan_set.id) if row.is_current]
    stale, current = _staleness(session, plan_set)
    base["stale"] = stale
    base["current"] = current
    return base


def _staleness(session: Session, plan_set: Any) -> tuple[bool, bool]:
    if not plan_set.input_fingerprint:
        return False, False
    try:
        executor = build_transformation_planning_executor(session, get_settings())
        current = executor.input_fingerprint(plan_set)
    except Exception:
        return False, False
    if not current:
        return False, False
    return (current != plan_set.input_fingerprint), True
