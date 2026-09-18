"""Stage 4.2 semantic-operator fidelity hardening (no live providers).

Meaning-changing negation, exclusivity, and modality/certainty/obligation
operators must never be silently discarded during deterministic source-support
comparison. A claim that adds, drops, or changes a protected operator relative
to its cited wording must not become GROUNDED_IN_SOURCE through lexical overlap
alone; it defers to semantic review instead.
"""

from __future__ import annotations

from stage42_support import make_inputs, make_plan, original_block, source_block

from app.transformation.governance.policy import DEFAULT_CONFIG
from app.transformation.governance.validation import evaluate_plan

_GROUNDED = "GROUNDED_IN_SOURCE"


def _plan(source_text: str, claim: str, *, kind: str = "EXPLANATION"):
    return make_plan(
        blocks=[
            source_block(0, text=source_text),
            original_block(1, intent=claim, kind=kind, grounding=("block:0",)),
        ],
        original_value_kinds=(kind,),
    )


def _evaluate(source_text: str, claim: str, *, kind: str = "EXPLANATION"):
    plan = _plan(source_text, claim, kind=kind)
    return evaluate_plan(
        plan,
        make_inputs([plan]),
        DEFAULT_CONFIG,
        provider_mode_deterministic=True,
    )


def _claim_state(source_text: str, claim: str) -> str:
    return str(_evaluate(source_text, claim).verification["claim_state"])


# --- English: added exclusivity --------------------------------------------


def test_added_exclusivity_is_not_grounded():
    assert (
        _claim_state(
            "Customers can request refunds.",
            "Only customers can request refunds.",
        )
        != _GROUNDED
    )


# --- English: added negation -----------------------------------------------


def test_added_negation_is_not_grounded():
    assert (
        _claim_state(
            "Customers can request refunds.",
            "Customers cannot request refunds.",
        )
        != _GROUNDED
    )


def test_contracted_negation_is_not_grounded():
    assert (
        _claim_state(
            "Customers can request refunds.",
            "Customers can't request refunds.",
        )
        != _GROUNDED
    )


# --- English: changed modality / obligation --------------------------------


def test_changed_modality_is_not_grounded():
    assert (
        _claim_state(
            "We may launch next month.",
            "We will launch next month.",
        )
        != _GROUNDED
    )


def test_changed_obligation_is_not_grounded():
    assert (
        _claim_state(
            "We can launch next month.",
            "We must launch next month.",
        )
        != _GROUNDED
    )


# --- English: dropped operator (other direction) ---------------------------


def test_dropped_negation_is_not_grounded():
    assert (
        _claim_state(
            "Customers cannot request refunds.",
            "Customers can request refunds.",
        )
        != _GROUNDED
    )


def test_dropped_obligation_is_not_grounded():
    assert (
        _claim_state(
            "We shall launch next month.",
            "We launch next month.",
        )
        != _GROUNDED
    )


# --- English: permitted framing and faithful restatements stay grounded -----


def test_permitted_framing_remains_grounded():
    assert (
        _claim_state(
            "Customers can request refunds.",
            "According to the source, customers can request refunds.",
        )
        == _GROUNDED
    )


def test_identical_exclusivity_is_grounded():
    assert (
        _claim_state(
            "Only customers can request refunds.",
            "Only customers can request refunds.",
        )
        == _GROUNDED
    )


def test_identical_negation_is_grounded():
    assert (
        _claim_state(
            "Customers cannot request refunds.",
            "Customers cannot request refunds.",
        )
        == _GROUNDED
    )


def test_identical_modality_is_grounded():
    assert (
        _claim_state(
            "We may launch next month.",
            "We may launch next month.",
        )
        == _GROUNDED
    )


# --- English: quote plus an added operator must not inherit support ---------


def test_quote_with_added_exclusivity_is_not_grounded():
    assert (
        _claim_state(
            "Customers can request refunds.",
            'The source says "customers can request refunds," but only customers qualify.',
        )
        != _GROUNDED
    )


def test_quote_with_added_negation_is_not_grounded():
    assert (
        _claim_state(
            "Customers can request refunds.",
            'The source says "customers can request refunds," but non-customers cannot.',
        )
        != _GROUNDED
    )


def test_quote_restated_with_added_exclusivity_is_not_grounded():
    assert (
        _claim_state(
            "Customers can request refunds.",
            'The source says "customers can request refunds" but only customers '
            "can request refunds",
        )
        != _GROUNDED
    )


def test_framed_claim_dropping_modality_is_not_grounded():
    assert (
        _claim_state(
            "We may launch next month.",
            "According to the source, we launch next month",
        )
        != _GROUNDED
    )


# --- Arabic: operators -----------------------------------------------------


def test_arabic_added_exclusivity_is_not_grounded():
    assert (
        _claim_state(
            "العملاء يمكنهم طلب استرجاع الأموال",
            "فقط العملاء يمكنهم طلب استرجاع الأموال",
        )
        != _GROUNDED
    )


def test_arabic_added_negation_is_not_grounded():
    assert (
        _claim_state(
            "العملاء يمكنهم طلب استرجاع الأموال",
            "العملاء لا يمكنهم طلب استرجاع الأموال",
        )
        != _GROUNDED
    )


def test_arabic_changed_modality_is_not_grounded():
    assert (
        _claim_state(
            "قد نطلق المنتج الشهر القادم",
            "سوف نطلق المنتج الشهر القادم",
        )
        != _GROUNDED
    )


def test_arabic_permitted_framing_remains_grounded():
    assert (
        _claim_state(
            "العملاء يمكنهم طلب استرجاع الأموال",
            "بحسب المصدر، العملاء يمكنهم طلب استرجاع الأموال",
        )
        == _GROUNDED
    )


# --- Stage 4.2 hard-gate: no deterministic approval without support --------


def test_operator_tampering_never_approves_or_becomes_eligible():
    from app.core.enums import GovernancePlanStatus
    from app.transformation.governance.validation import build_plan_governance

    plan = _plan("We may launch next month.", "We will launch next month.")
    evaluation = evaluate_plan(
        plan,
        make_inputs([plan]),
        DEFAULT_CONFIG,
        provider_mode_deterministic=True,
    )
    governance = build_plan_governance(evaluation, input_fingerprint="in", output_fingerprint="out")
    assert governance.status is not GovernancePlanStatus.APPROVED_FOR_SELECTION
    assert governance.status is not GovernancePlanStatus.APPROVED_WITH_CAUTION
    assert governance.eligible_for_stage4_3 is False
