"""Stage 6 V4 intent-fulfillment / redundancy regressions (offline).

These test the review contract, not live model judgment: recorded/fake responses
can authorize only when the new intent evidence is structurally valid.
"""

from __future__ import annotations

from test_stage6_gemini_boundary import (
    NOVEL_COPY,
    _CategoricalCritiqueProvider,
    _inputs_with_draft,
)
from test_stage6_materialization_core import make_inputs

from app.materialization.policy import DEFAULT_CONFIG, CallLedger
from app.materialization.providers import (
    CRITIQUE_DIMENSIONS,
    BlockCritique,
    CritiqueDimensionFinding,
    DeterministicWritingProvider,
    build_writing_request,
    materialize_script,
)

#: The failed acceptance copy: dialect source restated as a general claim about
#: excessive demands, without the promised mechanism.
FAILED_COPY = "الطلبات والذهب الكثير يعكسان المبالغة في الاحتياجات."


def _block(
    *,
    intent_quote: str = NOVEL_COPY,
    intent_refs: tuple[str, ...] = ("block:0",),
    intent_relation: str = "The copy states the mechanism behind the observed drop.",
    **overrides: str,
) -> BlockCritique:
    findings: list[CritiqueDimensionFinding] = []
    for dimension in CRITIQUE_DIMENSIONS:
        level = overrides.get(dimension, "PASS")
        if dimension == "INTENT_FULFILLMENT":
            findings.append(
                CritiqueDimensionFinding(
                    dimension,
                    level,
                    copy_quote=intent_quote,
                    grounding_refs=intent_refs,
                    added_relation=intent_relation,
                )
            )
        else:
            findings.append(CritiqueDimensionFinding(dimension, level))
    return BlockCritique(block_index=1, findings=tuple(findings), substantive=True)


def _materialize(inputs: object, block: BlockCritique):
    return materialize_script(
        DeterministicWritingProvider(),
        inputs,  # type: ignore[arg-type]
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        critique_provider=_CategoricalCritiqueProvider({1: block}),
    )


def test_all_pass_without_intent_evidence_defers() -> None:
    block = BlockCritique(
        block_index=1,
        findings=tuple(
            CritiqueDimensionFinding(dimension, "PASS") for dimension in CRITIQUE_DIMENSIONS
        ),
        substantive=True,
    )
    script = _materialize(_inputs_with_draft(NOVEL_COPY), block)
    assert script.deferred
    assert script.authored_blocks == ()


def test_intent_quote_must_be_an_exact_substring_of_actual_copy() -> None:
    block = _block(intent_quote="A quote that was never authored.")
    script = _materialize(_inputs_with_draft(NOVEL_COPY), block)
    assert script.deferred
    assert any(
        "INVALID_INTENT_FULFILLMENT_COPY_QUOTE" in finding.reason_codes
        for finding in script.findings
    )


def test_intent_grounding_must_stay_within_the_offered_evidence() -> None:
    block = _block(intent_refs=("words:999-1000",))
    script = _materialize(_inputs_with_draft(NOVEL_COPY), block)
    assert script.deferred
    assert any(
        "OUT_OF_SCOPE_INTENT_FULFILLMENT_GROUNDING" in finding.reason_codes
        for finding in script.findings
    )


def test_intent_relation_is_required() -> None:
    block = _block(intent_relation="")
    script = _materialize(_inputs_with_draft(NOVEL_COPY), block)
    assert script.deferred
    assert any(
        "MISSING_INTENT_FULFILLMENT_RELATION" in finding.reason_codes for finding in script.findings
    )


def test_redundancy_concern_rejects_a_substantive_block() -> None:
    block = _block(REDUNDANCY="CONCERN")
    script = _materialize(_inputs_with_draft(NOVEL_COPY), block)
    assert not script.accepted
    assert script.authored_blocks == ()
    assert any("CRITIQUE_REDUNDANCY_CONCERN" in code for code in script.reason_codes)


def test_redundancy_unknown_defers_a_substantive_block() -> None:
    block = _block(REDUNDANCY="UNKNOWN")
    script = _materialize(_inputs_with_draft(NOVEL_COPY), block)
    assert script.deferred
    assert script.authored_blocks == ()


def test_standalone_delivery_requires_a_length_necessity_judgment() -> None:
    inputs = make_inputs()
    blocks = list(inputs.blocks)
    blocks[1] = {
        **blocks[1],
        "draft_line": NOVEL_COPY,
        "grounding_refs": ["block:0"],
        "delivery_intent": "ON_SCREEN_TEXT",
    }
    standalone = make_inputs(blocks=tuple(blocks))

    deferred = _materialize(standalone, _block(LENGTH_NECESSITY="UNKNOWN"))
    assert deferred.deferred

    accepted = _materialize(standalone, _block(LENGTH_NECESSITY="PASS"))
    assert accepted.accepted
    assert any(
        finding.dimension == "PACING_CONTEXT_DENSITY"
        and finding.level == "SATISFIED"
        and "CRITIQUE_LENGTH_NECESSITY_PASS" in finding.reason_codes
        for finding in accepted.findings
    )


def test_failed_acceptance_copy_cannot_be_passed_without_intent_evidence() -> None:
    inputs = _inputs_with_draft(FAILED_COPY)
    block = BlockCritique(
        block_index=1,
        findings=tuple(
            CritiqueDimensionFinding(dimension, "PASS") for dimension in CRITIQUE_DIMENSIONS
        ),
        substantive=True,
    )
    script = _materialize(inputs, block)
    assert script.deferred
    assert script.authored_blocks == ()


def test_required_context_is_not_writing_evidence() -> None:
    request = build_writing_request(_inputs_with_draft(NOVEL_COPY), config=DEFAULT_CONFIG)
    target = request.targets[0]
    assert "remote work debate" not in target.source_evidence
    assert target.proposed_delivery["delivery"] in {"AUTHORED_TEXT", "STANDALONE_CARD"}


def test_planning_request_separates_rendered_from_nonrendered_evidence() -> None:
    from types import SimpleNamespace

    from app.transformation.planning.providers import _source_selection_payload

    inputs = SimpleNamespace(
        word_coverage_sufficient=True,
        refined_start=10.0,
        refined_end=14.0,
        planning_start=10.0,
        planning_end=14.0,
        has_operator_window=False,
        approved_source_start=None,
        approved_source_end=None,
    )
    config = SimpleNamespace(min_source_excerpt_seconds=0.4, max_blocks_per_plan=8)
    words = [
        {"index": 0, "text": "كتير", "start": 10.0, "end": 10.4, "probability": 0.9},
    ]
    payload = _source_selection_payload(inputs, config, words, False, 1.5)
    scope = payload["evidence_scope"]
    assert "rendered_on_selection" in scope
    assert "refined_transcript" in scope["nonrendered"]
    assert "content_sufficiency_requirement" in scope


def test_governance_request_carries_rendered_selected_evidence_only() -> None:
    from stage42_support import make_inputs, make_plan

    from app.transformation.governance.policy import DEFAULT_CONFIG
    from app.transformation.governance.providers import build_governance_request

    request = build_governance_request(make_inputs([make_plan()]), DEFAULT_CONFIG)
    payload = request.to_payload()
    assert payload["evidence_scope"]["nonrendered"]
    plan = payload["plans"][0]
    assert "selected_evidence" in plan
    assert plan["required_context_is_not_rendered_evidence"] is True
    # The full transcript remains available as explicitly nonrendered context.
    assert request.transcript
