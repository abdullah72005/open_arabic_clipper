"""Stage 4.2 provider boundary: tolerant parsing and forbidden content."""

from __future__ import annotations

from stage42_support import make_inputs, make_plan

from app.transformation.governance.policy import DEFAULT_CONFIG
from app.transformation.governance.providers import (
    GovernanceProviderError,
    build_governance_request,
    deserialize_critique,
    parse_governance_result,
    serialize_critique,
)


def _request(plans=None):
    plans = plans or [make_plan()]
    return build_governance_request(make_inputs(plans), DEFAULT_CONFIG)


def _entry(plan, **overrides):
    payload = {
        "plan_id": plan.plan_id,
        "plan_output_fingerprint": plan.plan_output_fingerprint,
        "fidelity": "PASS",
        "value": "DISTINCT",
        "retention": "PRESERVED",
        "coherence": "COHERENT",
        "narration": "APPROPRIATE",
        "template_feel": "LOW",
        "unsupported_claim": "NONE",
        "finding_codes": [],
        "block_indexes": [0],
        "summary": "bounded review",
        "confidence": 0.7,
    }
    payload.update(overrides)
    return payload


def test_valid_siblings_survive_one_malformed_item():
    plan_a = make_plan(plan_id="a" * 8)
    plan_b = make_plan(plan_id="b" * 8, fingerprint="fp-b")
    request = _request([plan_a, plan_b])
    content = {
        "critiques": [
            _entry(plan_a),
            {**_entry(plan_b), "fidelity": "NOT_A_VALUE"},
        ]
    }
    result = parse_governance_result(content, request)
    ids = {critique.plan_id for critique in result.critiques}
    assert ids == {plan_a.plan_id}


def test_unknown_plan_id_is_dropped():
    plan = make_plan()
    request = _request([plan])
    result = parse_governance_result({"critiques": [_entry(plan, plan_id="unknown")]}, request)
    assert result.critiques == ()


def test_mismatched_fingerprint_is_dropped():
    plan = make_plan()
    request = _request([plan])
    result = parse_governance_result(
        {"critiques": [_entry(plan, plan_output_fingerprint="other")]}, request
    )
    assert result.critiques == ()


def test_duplicate_items_are_dropped():
    plan = make_plan()
    request = _request([plan])
    result = parse_governance_result({"critiques": [_entry(plan), _entry(plan)]}, request)
    assert len(result.critiques) == 1


def test_platform_guarantee_text_is_rejected():
    plan = make_plan()
    request = _request([plan])
    result = parse_governance_result(
        {"critiques": [_entry(plan, summary="This plan is safe for YouTube")]}, request
    )
    assert result.critiques == ()


def test_provider_status_assignment_is_rejected():
    plan = make_plan()
    request = _request([plan])
    result = parse_governance_result(
        {"critiques": [_entry(plan, summary="status: APPROVED_FOR_SELECTION")]}, request
    )
    assert result.critiques == ()


def test_provider_voice_selection_is_rejected():
    plan = make_plan()
    request = _request([plan])
    result = parse_governance_result(
        {"critiques": [_entry(plan, finding_codes=["Use Gemini voice Charon"])]}, request
    )
    assert result.critiques == ()


def test_provider_evasion_tactic_is_rejected():
    plan = make_plan()
    request = _request([plan])
    result = parse_governance_result(
        {"critiques": [_entry(plan, summary="mirror the video to evade detection")]}, request
    )
    assert result.critiques == ()


def test_malformed_top_level_raises():
    request = _request()
    try:
        parse_governance_result({"critiques": "not-a-list"}, request)
    except GovernanceProviderError as error:
        assert error.category == "MALFORMED_OUTPUT"
    else:  # pragma: no cover
        raise AssertionError("expected GovernanceProviderError")


def test_serialize_round_trip():
    plan = make_plan()
    request = _request([plan])
    result = parse_governance_result({"critiques": [_entry(plan)]}, request)
    critique = result.critiques[0]
    restored = deserialize_critique(serialize_critique(critique))
    assert restored is not None
    assert restored.plan_id == critique.plan_id
    assert restored.fidelity == critique.fidelity


def test_unbounded_summary_is_truncated():
    plan = make_plan()
    request = _request([plan])
    result = parse_governance_result({"critiques": [_entry(plan, summary="x" * 4000)]}, request)
    assert len(result.critiques[0].summary) <= 600


def test_arbitrary_finding_codes_are_discarded():
    plan = make_plan()
    request = _request([plan])
    result = parse_governance_result(
        {
            "critiques": [
                _entry(
                    plan,
                    finding_codes=["totally made up code", "CONTEXT_DISTORTION", "unbounded text"],
                )
            ]
        },
        request,
    )
    assert result.critiques[0].finding_codes == ("CONTEXT_DISTORTION",)


def test_out_of_range_block_indexes_are_dropped():
    plan = make_plan()
    request = _request([plan])
    block_count = len(plan.blocks)
    result = parse_governance_result(
        {"critiques": [_entry(plan, block_indexes=[0, 1, 99, -1])]}, request
    )
    indexes = result.critiques[0].block_indexes
    assert all(0 <= index < block_count for index in indexes)
    assert 99 not in indexes and -1 not in indexes
