"""Versioned Stage 4.3 selection policy: statuses, caution allowlist, ordering.

Stage 4.3 is deterministic and provider-free. It never replans, re-governs,
rewrites, researches, or renders. This module owns every output-affecting
constant so the deterministic hierarchy is auditable and fingerprintable. It
never computes a weighted aggregate score: arbitration is a readable
lexicographic comparison over persisted Stage 4.2 evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

SELECTION_POLICY_VERSION = "stage4.3-v1"
SELECTION_SCHEMA_VERSION = "stage4.3-schema-v1"
SELECTION_FINGERPRINT_VERSION = "1"

# Approved-for-selection is the clean tier; approved-with-caution is considered
# only when no valid clean approval exists.
CLEAN_STATUS = "APPROVED_FOR_SELECTION"
CAUTION_STATUS = "APPROVED_WITH_CAUTION"
DEFERRED_STATUS = "GOVERNANCE_DEFERRED"
BLOCKED_STATUS = "BLOCKED_PENDING_VERIFICATION"
REVISION_STATUS = "REVISION_REQUIRED"
REJECTED_STATUS = "REJECTED_BY_GOVERNOR"

TERMINAL_NON_SELECTABLE_STATUSES = frozenset({BLOCKED_STATUS, REVISION_STATUS, REJECTED_STATUS})

# Conservative automatic caution allowlist. A caution plan is auto-selectable
# only when *every* warning-level reason is in this allowlist and the persisted
# dimension is MODERATE (never HIGH/UNKNOWN), plus the extra conditions below.
CAUTION_ALLOWLIST = frozenset({"SOURCE_DOMINANCE_CONCERN", "TEMPLATE_MASS_PRODUCED_FEEL"})

# Explicitly non-auto-selectable caution codes (and the default for every unknown
# warning code). These may still be reported, never automatically committed.
NON_AUTO_SELECTABLE_CAUTION_CODES = frozenset(
    {
        "SEMANTIC_FIDELITY_CONCERN",
        "UNSUPPORTED_CRITICAL_CLAIM",
        "SOURCE_MOMENT_SEVERELY_DAMAGED",
        "PLAN_INCOHERENT",
    }
)

RESOLVED_VERIFICATION_STATES = frozenset({"GROUNDED_IN_SOURCE", "NOT_APPLICABLE"})

# Closed, bounded Stage 4.3 reason codes.
SELECTED_CLEAN = "PLAN_SELECTED_CLEAN"
SELECTED_CAUTION = "PLAN_SELECTED_WITH_CAUTION"
NO_ELIGIBLE_SURVIVOR = "NO_ELIGIBLE_SURVIVOR"
NO_SELECTABLE_PLAN = "NO_SELECTABLE_PLAN"
GOVERNANCE_NOT_AVAILABLE = "GOVERNANCE_NOT_AVAILABLE"
GOVERNANCE_STALE = "GOVERNANCE_STALE"
GOVERNANCE_NOT_CURRENT = "GOVERNANCE_NOT_CURRENT"
GOVERNANCE_UNVERIFIABLE = "GOVERNANCE_UNVERIFIABLE"
INCONSISTENT_GOVERNANCE_EVIDENCE = "INCONSISTENT_GOVERNANCE_EVIDENCE"
SEMANTIC_GOVERNANCE_UNFINISHED = "SEMANTIC_GOVERNANCE_UNFINISHED"
STATUS_NOT_APPROVED = "STATUS_NOT_APPROVED"
NOT_ELIGIBLE_FOR_STAGE4_3 = "NOT_ELIGIBLE_FOR_STAGE4_3"
HARD_GATES_PRESENT = "HARD_GATES_PRESENT"
PLAN_FINGERPRINT_MISMATCH = "PLAN_FINGERPRINT_MISMATCH"
VERIFICATION_UNRESOLVED = "VERIFICATION_UNRESOLVED"
SEMANTIC_FIDELITY_INSUFFICIENT = "SEMANTIC_FIDELITY_INSUFFICIENT"
MISSING_DECISION_EVIDENCE = "MISSING_DECISION_EVIDENCE"
CAUTION_NOT_ALLOWLISTED = "CAUTION_NOT_ALLOWLISTED"
CAUTION_DIMENSION_NOT_MODERATE = "CAUTION_DIMENSION_NOT_MODERATE"
RETENTION_DAMAGED_OR_MIXED = "RETENTION_DAMAGED_OR_MIXED"
COHERENCE_NOT_ACCEPTABLE = "COHERENCE_NOT_ACCEPTABLE"
LOST_IN_ARBITRATION = "LOST_IN_ARBITRATION"
NOT_IN_ACTIVE_TIER = "NOT_IN_ACTIVE_TIER"
MISSING_GOVERNANCE_RESULT = "MISSING_GOVERNANCE_RESULT"

# Categorical order maps. Lower is always "better" so a plain tuple comparison
# is the whole comparator.
EVIDENCE_STRENGTH_ORDER: Mapping[str, int] = {
    "STRONG": 0,
    "ADEQUATE": 1,
    "WEAK": 2,
    "NONE": 3,
    "UNKNOWN": 4,
}
LEVEL_ORDER: Mapping[str, int] = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "UNKNOWN": 3}
PLATFORM_RISK_ORDER: Mapping[str, int] = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "UNDETERMINED": 3,
}
INTENSITY_ORDER: Mapping[str, int] = {"MINIMAL": 0, "MODERATE": 1, "STRONG": 2}
NARRATION_BURDEN_ORDER: Mapping[str, int] = {
    "APPROPRIATE": 0,
    "NOT_NEEDED": 1,
    "EXCESSIVE": 2,
    "REDUNDANT": 3,
    "POSITION_DAMAGING": 4,
    "UNKNOWN": 5,
}

_WORST = 99

# Ordered arbitration dimensions (each mapped so lower is better). Persisted as
# evidence and used for the first-material-distinction explanation.
COMPARISON_DIMENSIONS: tuple[str, ...] = (
    "semantic_fidelity",
    "retention_preservation",
    "source_moment_damage",
    "substantive_originality",
    "platform_reuse_risk",
    "platform_high_moderate_count",
    "plan_coherence",
    "generic_filler_risk",
    "redundant_commentary_risk",
    "narration_burden",
    "transformation_proportionality",
    "transformation_intensity",
    "stage41_generation_rank",
    "stage41_planner_confidence",
    "plan_identity",
)


DEFAULT_POLICY_SUMMARY: Mapping[str, object] = {
    "selection_policy_version": SELECTION_POLICY_VERSION,
    "schema_version": SELECTION_SCHEMA_VERSION,
    "caution_allowlist": sorted(CAUTION_ALLOWLIST),
    "hierarchy": list(COMPARISON_DIMENSIONS),
}


def _rank(order: Mapping[str, int], value: object) -> int:
    if isinstance(value, str):
        return order.get(value, _WORST)
    return _WORST


def source_dominance_dimension(result: Mapping[str, object]) -> str:
    return _dimension(result, "source_dominance")


def template_dimension(result: Mapping[str, object]) -> str:
    return _dimension(result, "template_mass_produced_feel")


def _dimension(result: Mapping[str, object], key: str) -> str:
    dimensions = result.get("dimensions")
    if isinstance(dimensions, Mapping):
        value = dimensions.get(key)
        if isinstance(value, str):
            return value
    return "UNKNOWN"


def _verification_resolved(result: Mapping[str, object]) -> bool:
    verification = result.get("verification")
    if not isinstance(verification, Mapping):
        return False
    claim_state = verification.get("claim_state")
    if claim_state not in RESOLVED_VERIFICATION_STATES:
        return False
    return not bool(verification.get("unresolved"))


def _plan_fingerprint_matches(result: Mapping[str, object], plan_fingerprint: str) -> bool:
    governed = str(result.get("plan_output_fingerprint") or "")
    return bool(governed) and governed == plan_fingerprint


def _hard_gates(result: Mapping[str, object]) -> list[object]:
    gates = result.get("hard_gates")
    return list(gates) if isinstance(gates, list) else []


def _warning_codes(result: Mapping[str, object]) -> list[str]:
    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        return []
    codes: list[str] = []
    for warning in warnings:
        if isinstance(warning, Mapping):
            code = warning.get("code")
            if isinstance(code, str) and code:
                codes.append(code)
    return codes


def _common_selectability_reasons(
    result: Mapping[str, object], plan_fingerprint: str, expected_status: str
) -> list[str]:
    codes: list[str] = []
    if result.get("status") != expected_status:
        codes.append(STATUS_NOT_APPROVED)
    if not bool(result.get("eligible_for_stage4_3")):
        codes.append(NOT_ELIGIBLE_FOR_STAGE4_3)
    if _hard_gates(result):
        codes.append(HARD_GATES_PRESENT)
    if not _plan_fingerprint_matches(result, plan_fingerprint):
        codes.append(PLAN_FINGERPRINT_MISMATCH)
    if not _verification_resolved(result):
        codes.append(VERIFICATION_UNRESOLVED)
    fidelity = _dimension(result, "semantic_fidelity")
    if EVIDENCE_STRENGTH_ORDER.get(fidelity, _WORST) != 0:
        codes.append(SEMANTIC_FIDELITY_INSUFFICIENT)
    if not _has_decision_evidence(result):
        codes.append(MISSING_DECISION_EVIDENCE)
    return codes


def _has_decision_evidence(result: Mapping[str, object]) -> bool:
    return (
        isinstance(result.get("result_id"), (str, type(None)))
        and isinstance(result.get("dimensions"), Mapping)
        and isinstance(result.get("verification"), Mapping)
        and isinstance(result.get("platform_risk"), Mapping)
        and isinstance(result.get("plan_output_fingerprint"), str)
        and bool(result.get("plan_output_fingerprint"))
    )


def clean_selectability(
    result: Mapping[str, object], plan_fingerprint: str
) -> tuple[bool, tuple[str, ...]]:
    """True only when a result is a clean, fully verified approved plan."""

    codes = _common_selectability_reasons(result, plan_fingerprint, CLEAN_STATUS)
    return (not codes), tuple(dict.fromkeys(codes))


def caution_selectability(
    result: Mapping[str, object], plan_fingerprint: str
) -> tuple[bool, tuple[str, ...]]:
    """Conservative automatic caution selection.

    ``APPROVED_WITH_CAUTION`` is not automatically selectable merely because the
    broad Stage 4.2 handoff flag is true.
    """

    codes = _common_selectability_reasons(result, plan_fingerprint, CAUTION_STATUS)
    warning_codes = _warning_codes(result)
    for code in warning_codes:
        if code not in CAUTION_ALLOWLIST or code in NON_AUTO_SELECTABLE_CAUTION_CODES:
            codes.append(CAUTION_NOT_ALLOWLISTED)

    if "SOURCE_DOMINANCE_CONCERN" in warning_codes:
        if source_dominance_dimension(result) != "MODERATE":
            codes.append(CAUTION_DIMENSION_NOT_MODERATE)
    if "TEMPLATE_MASS_PRODUCED_FEEL" in warning_codes:
        if template_dimension(result) != "MODERATE":
            codes.append(CAUTION_DIMENSION_NOT_MODERATE)

    if _dimension(result, "source_moment_damage") != "LOW":
        codes.append(RETENTION_DAMAGED_OR_MIXED)
    if EVIDENCE_STRENGTH_ORDER.get(_dimension(result, "retention_preservation"), _WORST) > 1:
        codes.append(RETENTION_DAMAGED_OR_MIXED)
    if EVIDENCE_STRENGTH_ORDER.get(_dimension(result, "plan_coherence"), _WORST) > 1:
        codes.append(COHERENCE_NOT_ACCEPTABLE)
    if "UNSUPPORTED_CRITICAL_CLAIM" in warning_codes:
        codes.append(CAUTION_NOT_ALLOWLISTED)

    return (not codes), tuple(dict.fromkeys(codes))


def governance_evidence_inconsistent(
    result: Mapping[str, object], plan_fingerprint: str
) -> tuple[bool, tuple[str, ...]]:
    """Fail-closed integrity check for an approved/caution result.

    A current governance set with internally inconsistent evidence must defer,
    never "repair" the evidence at the selection boundary.
    """

    status = result.get("status")
    if status not in {CLEAN_STATUS, CAUTION_STATUS}:
        return False, ()
    reasons: list[str] = []
    if _hard_gates(result):
        reasons.append(HARD_GATES_PRESENT)
    if not _plan_fingerprint_matches(result, plan_fingerprint):
        reasons.append(PLAN_FINGERPRINT_MISMATCH)
    if not _verification_resolved(result):
        reasons.append(VERIFICATION_UNRESOLVED)
    if not _has_decision_evidence(result):
        reasons.append(MISSING_DECISION_EVIDENCE)
    return bool(reasons), tuple(dict.fromkeys(reasons))


def platform_risk_profile(platform_risk: Mapping[str, object]) -> tuple[int, int]:
    """Return ``(worst_level_rank, high_moderate_count)`` across platform risks.

    Compares YouTube reused/inauthentic/spam, Facebook unoriginal/spam, and the
    generic source-dominance level. Never claims safety or monetization.
    """

    levels: list[str] = []
    for platform, keys in (
        ("youtube", ("reused_content", "inauthentic_mass_produced", "spam_deceptive_practices")),
        ("facebook", ("unoriginal_content", "spam_repetitive")),
    ):
        section = platform_risk.get(platform)
        if not isinstance(section, Mapping):
            continue
        for key in keys:
            entry = section.get(key)
            if isinstance(entry, Mapping):
                level = entry.get("level")
                if isinstance(level, str):
                    levels.append(level)
    generic = platform_risk.get("generic")
    if isinstance(generic, Mapping):
        dominance = generic.get("source_dominance")
        if isinstance(dominance, Mapping) and isinstance(dominance.get("level"), str):
            levels.append(str(dominance["level"]))
    if not levels:
        return _WORST, _WORST
    worst = max(PLATFORM_RISK_ORDER.get(level, _WORST) for level in levels)
    count = sum(1 for level in levels if level in {"MODERATE", "HIGH"})
    return worst, count


def comparison_vector(
    result: Mapping[str, object],
    *,
    plan_fingerprint: str,
    generation_rank: int,
    planner_confidence: float,
) -> tuple[object, ...]:
    """Deterministic lexicographic key; lower is better on every element."""

    worst_risk, risk_count = platform_risk_profile(_platform_risk(result))
    confidence_rank = int(round(max(0.0, min(1.0, planner_confidence)) * 1_000_000))
    return (
        _rank(EVIDENCE_STRENGTH_ORDER, _dimension(result, "semantic_fidelity")),
        _rank(EVIDENCE_STRENGTH_ORDER, _dimension(result, "retention_preservation")),
        _rank(LEVEL_ORDER, _dimension(result, "source_moment_damage")),
        _rank(EVIDENCE_STRENGTH_ORDER, _dimension(result, "substantive_originality")),
        worst_risk,
        risk_count,
        _rank(EVIDENCE_STRENGTH_ORDER, _dimension(result, "plan_coherence")),
        _rank(LEVEL_ORDER, _dimension(result, "generic_filler_risk")),
        _rank(LEVEL_ORDER, _dimension(result, "redundant_commentary_risk")),
        _rank(NARRATION_BURDEN_ORDER, _dimension(result, "narration_burden")),
        _rank(LEVEL_ORDER, _dimension(result, "transformation_proportionality")),
        _intensity_rank(result),
        max(1, generation_rank),
        -confidence_rank,
        str(result.get("plan_id") or ""),
    )


def _platform_risk(result: Mapping[str, object]) -> Mapping[str, object]:
    value = result.get("platform_risk")
    return value if isinstance(value, Mapping) else {}


def _intensity_rank(result: Mapping[str, object]) -> int:
    intensity = result.get("intensity")
    if isinstance(intensity, str):
        return INTENSITY_ORDER.get(intensity, _WORST)
    return _WORST


def first_material_distinction(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    left_vector: Sequence[object],
    right_vector: Sequence[object],
) -> dict[str, object] | None:
    for index, (left_value, right_value) in enumerate(zip(left_vector, right_vector)):
        if left_value != right_value:
            return {
                "dimension": COMPARISON_DIMENSIONS[index],
                "winner_plan_id": str(left.get("plan_id") or ""),
                "runner_up_plan_id": str(right.get("plan_id") or ""),
                "winner_value": _jsonable(left_value),
                "runner_up_value": _jsonable(right_value),
            }
    return None


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "CAUTION_ALLOWLIST",
    "CAUTION_STATUS",
    "CLEAN_STATUS",
    "COMPARISON_DIMENSIONS",
    "DEFAULT_POLICY_SUMMARY",
    "DEFERRED_STATUS",
    "EVIDENCE_STRENGTH_ORDER",
    "INTENSITY_ORDER",
    "LEVEL_ORDER",
    "NARRATION_BURDEN_ORDER",
    "NON_AUTO_SELECTABLE_CAUTION_CODES",
    "PLATFORM_RISK_ORDER",
    "RESOLVED_VERIFICATION_STATES",
    "SELECTION_FINGERPRINT_VERSION",
    "SELECTION_POLICY_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "TERMINAL_NON_SELECTABLE_STATUSES",
    "caution_selectability",
    "clean_selectability",
    "comparison_vector",
    "first_material_distinction",
    "governance_evidence_inconsistent",
    "platform_risk_profile",
]
