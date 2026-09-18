"""Canonical deterministic Stage 4.3 selection fingerprints.

The input fingerprint covers every selection-relevant dependency but never the
Gemini key/availability, Qwen/Ollama availability, TTS provider/model/voice or
speaker identity, channel voice configuration, caption font, render
resolution/configuration, B-roll choice, or publishing title/description/
schedule/metadata. A changed Stage 4.2 fingerprint, Stage 4.1 plan fingerprint,
verification state, warning, governor policy/profile, relevant target context, or
Stage 4.3 policy version produces a new current selection result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from app.pipeline.fingerprints import canonical_fingerprint
from app.transformation.selection.policy import (
    CAUTION_ALLOWLIST,
    SELECTION_FINGERPRINT_VERSION,
    SELECTION_POLICY_VERSION,
    SELECTION_SCHEMA_VERSION,
)


class _PlanRow(Protocol):
    id: object
    plan_output_fingerprint: str
    generation_rank: int
    planner_confidence: float
    strategy_type: object
    intensity: object
    is_current: bool


def selection_input_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "transformation-selection-input", SELECTION_FINGERPRINT_VERSION, dict(payload)
        )
    )


def selection_output_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "transformation-selection-output", SELECTION_FINGERPRINT_VERSION, dict(payload)
        )
    )


def build_selection_input_payload(
    *,
    candidate_id: str,
    candidate_key: str,
    source_id: str,
    disposition: str,
    is_current: bool,
    analysis_fingerprint: str,
    handoff: Mapping[str, object],
    plan_rows: Sequence[_PlanRow],
) -> dict[str, object]:
    stage40 = _mapping(handoff.get("stage40"))
    stage41 = _mapping(handoff.get("stage41"))
    governance_set = _mapping(handoff.get("governance_set"))
    selected_refinement = _mapping(handoff.get("selected_refinement") or {})
    governance_by_plan = _governance_by_plan(handoff)

    plans: list[dict[str, object]] = []
    for row in sorted(plan_rows, key=lambda item: str(item.id)):
        plans.append(
            {
                "plan_id": str(row.id),
                "plan_output_fingerprint": row.plan_output_fingerprint,
                "generation_rank": row.generation_rank,
                "planner_confidence": round(float(row.planner_confidence), 6),
                "strategy_type": str(getattr(row.strategy_type, "value", row.strategy_type)),
                "intensity": str(getattr(row.intensity, "value", row.intensity)),
                "is_current": bool(row.is_current),
            }
        )

    governance: list[dict[str, object]] = []
    for plan_id in sorted(governance_by_plan):
        result = governance_by_plan[plan_id]
        governance.append(_governance_payload(plan_id, result))

    return {
        "candidate": {
            "id": candidate_id,
            "key": candidate_key,
            "source_id": source_id,
            "disposition": disposition,
            "is_current": is_current,
            "analysis_fingerprint": analysis_fingerprint,
        },
        "stage40": {
            "analysis_id": stage40.get("analysis_id"),
            "output_fingerprint": stage40.get("output_fingerprint"),
        },
        "stage41": {
            "plan_set_id": stage41.get("plan_set_id"),
            "input_fingerprint": stage41.get("input_fingerprint"),
            "output_fingerprint": stage41.get("output_fingerprint"),
            "current": bool(stage41.get("current")),
            "stale": bool(stage41.get("stale")),
        },
        "stage42": {
            "governance_set_id": governance_set.get("id"),
            "input_fingerprint": governance_set.get("input_fingerprint"),
            "output_fingerprint": governance_set.get("output_fingerprint"),
            "policy_version": governance_set.get("policy_version"),
            "platform_policy_profile_version": governance_set.get(
                "platform_policy_profile_version"
            ),
            "freshness": governance_set.get("freshness"),
            "semantic_outcome": governance_set.get("semantic_outcome"),
        },
        "refinement": {
            "id": selected_refinement.get("id"),
            "priority": selected_refinement.get("priority"),
            "quality_level": selected_refinement.get("quality_level"),
        },
        "plans": plans,
        "governance": governance,
        "policy": {
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "schema_version": SELECTION_SCHEMA_VERSION,
            "caution_allowlist": sorted(CAUTION_ALLOWLIST),
        },
    }


def build_selection_output_payload(
    *,
    status: str,
    selected_plan_id: str | None,
    selected_governance_result_id: str | None,
    selected_with_caution: bool,
    selected_plan_fingerprint: str,
    selection_reason_codes: Sequence[str],
    selected_governance_snapshot: Mapping[str, object],
    arbitration_evidence: Mapping[str, object],
    alternative_dispositions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "status": status,
        "selected_plan_id": selected_plan_id,
        "selected_governance_result_id": selected_governance_result_id,
        "selected_with_caution": selected_with_caution,
        "selected_plan_fingerprint": selected_plan_fingerprint,
        "selection_reason_codes": list(selection_reason_codes),
        "selected_governance_snapshot": dict(selected_governance_snapshot),
        "arbitration_evidence": dict(arbitration_evidence),
        "alternative_dispositions": [dict(item) for item in alternative_dispositions],
    }


def _governance_by_plan(handoff: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    plans = handoff.get("plans")
    if not isinstance(plans, list):
        return result
    for plan in plans:
        if not isinstance(plan, Mapping):
            continue
        plan_id = str(plan.get("plan_id") or "")
        governance = plan.get("governance")
        if plan_id and isinstance(governance, Mapping):
            result[plan_id] = governance
    return result


def _governance_payload(plan_id: str, result: Mapping[str, object]) -> dict[str, object]:
    return {
        "plan_id": plan_id,
        "result_id": result.get("result_id"),
        "plan_output_fingerprint": result.get("plan_output_fingerprint"),
        "status": result.get("status"),
        "eligible_for_stage4_3": bool(result.get("eligible_for_stage4_3")),
        "severity": result.get("severity"),
        "hard_gates": _as_list(result.get("hard_gates")),
        "dimensions": dict(_mapping(result.get("dimensions"))),
        "verification": dict(_mapping(result.get("verification"))),
        "platform_risk": dict(_mapping(result.get("platform_risk"))),
        "reason_codes": _as_list(result.get("reason_codes")),
        "warnings": _as_list(result.get("warnings")),
        "remediation": _as_list(result.get("remediation")),
        "output_fingerprint": result.get("output_fingerprint"),
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


__all__ = [
    "build_selection_input_payload",
    "build_selection_output_payload",
    "selection_input_fingerprint",
    "selection_output_fingerprint",
]
