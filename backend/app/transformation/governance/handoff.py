"""Typed read-only Stage 4.2 -> Stage 4.3 handoff builder.

Performs no selection, ranking, rendering, or publication work. Exposes every
current Stage 4.1 plan with its independent governance result so Stage 4.3 can
filter eligible plans without reconstructing governance from prose. There is no
winner, no selected plan, no render-ready state, and no publication approval.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.models import ClipCandidate
from app.transformation.governance.queue import (
    get_governance_set_for_candidate,
    list_results,
)
from app.transformation.planning.handoff import build_stage4_2_handoff

# Truthful fail-closed governance freshness. Only VERIFIED_CURRENT may retain
# Stage 4.3 eligibility; STALE/NOT_CURRENT/UNVERIFIABLE force ineligibility.
FRESHNESS_VERIFIED_CURRENT = "VERIFIED_CURRENT"
FRESHNESS_STALE = "STALE"
FRESHNESS_NOT_CURRENT = "NOT_CURRENT"
FRESHNESS_UNVERIFIABLE = "UNVERIFIABLE"

_CURRENT_GOVERNANCE_STATUSES = {"COMPLETE", "PROVIDER_DEGRADED"}


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _result_dict(row: Any) -> dict[str, object]:
    return {
        "result_id": str(row.id),
        "plan_id": str(row.transformation_plan_id),
        "plan_output_fingerprint": row.plan_output_fingerprint,
        "status": row.status.value,
        "eligible_for_stage4_3": bool(row.eligible_for_stage4_3),
        "severity": row.severity,
        "hard_gates": list(row.hard_gates or []),
        "dimensions": dict(row.dimensions or {}),
        "verification": dict(row.verification or {}),
        "platform_risk": dict(row.platform_risk or {}),
        "reason_codes": list(row.reason_codes or []),
        "warnings": list(row.warnings or []),
        "remediation": list(row.remediation or []),
        "provider_evidence": dict(row.governance_provider_evidence or {}),
        "input_fingerprint": row.input_fingerprint,
        "output_fingerprint": row.output_fingerprint,
    }


def build_stage4_3_handoff(
    session: Session, candidate_id: uuid.UUID | str
) -> dict[str, Any] | None:
    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        return None

    stage41 = build_stage4_2_handoff(session, candidate_id)
    governance_set = get_governance_set_for_candidate(session, candidate.id)

    base: dict[str, Any] = {
        "candidate": {
            "id": str(candidate.id),
            "candidate_key": candidate.candidate_key,
            "source_id": str(candidate.source_video_id),
            "disposition": candidate.disposition.value,
        },
        "rights_and_provenance": {
            "rights_status": candidate.source_video.rights_status.value
            if candidate.source_video is not None
            else None,
            "media_origin": candidate.source_video.media_origin.value
            if candidate.source_video is not None
            else None,
            "provenance": dict(candidate.provenance_snapshot or {}),
            "rights_risk": candidate.rights_risk.value,
            "originality_risk": candidate.originality_risk.value,
        },
        "selected_refinement": (
            stage41.get("selected_refinement") if stage41 is not None else None
        ),
        "stage40": (
            {
                "analysis_id": (stage41.get("stage40") or {}).get("analysis_id"),
                "input_fingerprint": None,
                "output_fingerprint": (stage41.get("stage40") or {}).get("output_fingerprint"),
                "risk_snapshot": dict((stage41.get("stage40") or {}).get("snapshot") or {}),
            }
            if stage41 is not None
            else {}
        ),
        "stage41": (
            {
                "plan_set_id": (stage41.get("plan_set") or {}).get("id"),
                "input_fingerprint": (stage41.get("plan_set") or {}).get("input_fingerprint"),
                "output_fingerprint": (stage41.get("plan_set") or {}).get("output_fingerprint"),
                "current": bool(stage41.get("current")),
                "stale": bool(stage41.get("stale")),
            }
            if stage41 is not None
            else {}
        ),
        "governance_set": None,
        "plans": [],
        "stage4_3_implemented": True,
    }
    if governance_set is None:
        base["reason"] = "NO_GOVERNANCE_SET"
        return base

    results = {
        str(row.transformation_plan_id): row for row in list_results(session, governance_set.id)
    }
    freshness = _governance_freshness(session, governance_set)
    verified_current = freshness == FRESHNESS_VERIFIED_CURRENT
    base["governance_set"] = {
        "id": str(governance_set.id),
        "execution_status": governance_set.execution_status.value,
        "semantic_outcome": (
            governance_set.governance_outcome.value if governance_set.governance_outcome else None
        ),
        "policy_version": governance_set.policy_version,
        "platform_policy_profile_version": governance_set.platform_policy_profile_version,
        "platform_policy_checked_at": governance_set.platform_policy_checked_at,
        "provider": {
            "mode": governance_set.provider_mode.value,
            "status": governance_set.provider_status,
            "identity": dict(governance_set.provider_identity or {}),
            "evidence": dict(governance_set.provider_evidence or {}),
        },
        "input_fingerprint": governance_set.input_fingerprint,
        "output_fingerprint": governance_set.output_fingerprint,
        "cache_eligible": bool(governance_set.cache_eligible),
        "summary_counts": dict(governance_set.summary_counts or {}),
        "outcome_reasons": list(governance_set.outcome_reasons or []),
        # Truthful fail-closed freshness. A stale, not-current, or unverifiable
        # result is never represented as currently eligible for Stage 4.3.
        "freshness": freshness,
        "current": verified_current,
        "stale": freshness == FRESHNESS_STALE,
    }

    plans: list[dict[str, object]] = []
    if stage41 is not None:
        for plan in stage41.get("plans", []):
            plan_id = str(plan.get("plan_id"))
            result = results.get(plan_id)
            governance = _result_dict(result) if result is not None else None
            if governance is not None:
                governance["freshness"] = freshness
                governance["stale"] = freshness == FRESHNESS_STALE
                if not verified_current:
                    governance["eligible_for_stage4_3"] = False
            plans.append(
                {
                    "plan_id": plan_id,
                    "strategy": plan.get("strategy"),
                    "intensity": (plan.get("strategy") or {}).get("intensity"),
                    "blocks": plan.get("blocks"),
                    "governance": governance,
                }
            )
    base["plans"] = plans
    return base


def _governance_freshness(session: Session, governance_set: Any) -> str:
    """Fail-closed governance freshness for the read-only Stage 4.3 handoff.

    Returns exactly one of ``VERIFIED_CURRENT``, ``STALE``, ``NOT_CURRENT``, or
    ``UNVERIFIABLE``. Anything that is not an explicitly verified current result
    forces Stage 4.3 ineligibility.
    """

    if not governance_set.input_fingerprint:
        return FRESHNESS_UNVERIFIABLE
    if (
        governance_set.execution_status.value not in _CURRENT_GOVERNANCE_STATUSES
        or governance_set.governance_outcome is None
    ):
        return FRESHNESS_NOT_CURRENT
    from app.transformation.governance.executor import (
        build_transformation_governance_executor,
    )

    try:
        executor = build_transformation_governance_executor(session, get_settings())
        current = executor.input_fingerprint(governance_set)
    except Exception:
        return FRESHNESS_UNVERIFIABLE
    if not current:
        return FRESHNESS_UNVERIFIABLE
    if current != governance_set.input_fingerprint:
        return FRESHNESS_STALE
    return FRESHNESS_VERIFIED_CURRENT


__all__ = [
    "FRESHNESS_NOT_CURRENT",
    "FRESHNESS_STALE",
    "FRESHNESS_UNVERIFIABLE",
    "FRESHNESS_VERIFIED_CURRENT",
    "build_stage4_3_handoff",
]
