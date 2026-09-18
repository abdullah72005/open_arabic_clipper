"""Deterministic Stage 4.2 governor behavior (no live providers)."""

from __future__ import annotations

from stage42_support import (
    make_critique,
    make_inputs,
    make_plan,
    original_block,
    source_block,
    verification_block,
)

from app.core.enums import GovernancePlanStatus
from app.transformation.governance.policy import DEFAULT_CONFIG
from app.transformation.governance.validation import (
    SEMANTIC_AVAILABLE,
    apply_semantic_review,
    build_plan_governance,
    evaluate_plan,
    finalize,
)


def _evaluate(plan, inputs=None, *, provider_mode_deterministic=True):
    return evaluate_plan(
        plan,
        inputs or make_inputs([plan]),
        DEFAULT_CONFIG,
        provider_mode_deterministic=provider_mode_deterministic,
    )


def _status(evaluation):
    return finalize(evaluation)[0]


def _governance(plan, inputs=None, *, provider_mode_deterministic=True, critique=None):
    inputs = inputs or make_inputs([plan])
    evaluation = evaluate_plan(
        plan, inputs, DEFAULT_CONFIG, provider_mode_deterministic=provider_mode_deterministic
    )
    if critique is not None:
        apply_semantic_review(evaluation, critique, SEMANTIC_AVAILABLE)
    return build_plan_governance(evaluation, input_fingerprint="in", output_fingerprint="out")


def test_third_party_source_as_evidence_with_real_analysis_passes():
    plan = make_plan(
        blocks=[source_block(0), original_block(1)],
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    inputs = make_inputs(
        [plan],
        rights_risk="ELEVATED",
        originality_risk="UNDETERMINED",
        rights_status="THIRD_PARTY_UNKNOWN",
    )
    governance = _governance(plan, inputs)
    assert governance.status is GovernancePlanStatus.APPROVED_FOR_SELECTION
    assert governance.eligible_for_stage4_3 is True


def test_third_party_presentation_only_is_rejected():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(1, intent="Add captions, crop and zoom the video", kind="SYNTHESIS"),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    inputs = make_inputs([plan], rights_risk="ELEVATED", originality_risk="UNDETERMINED")
    governance = _governance(plan, inputs)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR
    assert "PRESENTATION_ONLY_TRANSFORMATION" in governance.reason_codes


def test_paraphrase_only_requires_revision():
    transcript = (
        "The guest argues that remote work collapsed productivity because managers lost "
        "the ability to mentor junior staff."
    )
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(1, intent=transcript, kind="SYNTHESIS"),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    inputs = make_inputs([plan], transcript=transcript)
    governance = _governance(plan, inputs)
    assert governance.status is GovernancePlanStatus.REVISION_REQUIRED
    assert "REDUNDANT_PARAPHRASE_ONLY" in governance.reason_codes


def test_five_second_explanation_after_hero_preserves_retention():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="Explain why the promotion rates fell sharply for the remote-work debate",
                kind="EXPLANATION",
                duration=5.0,
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan)
    assert governance.status in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }
    assert governance.dimensions["retention_preservation"] in {"STRONG", "ADEQUATE"}


def test_twelve_second_generic_preamble_fails():
    plan = make_plan(
        blocks=[
            original_block(
                0,
                intent="This is interesting and important to note",
                kind="SYNTHESIS",
                duration=12.0,
            ),
            source_block(1),
        ],
        hero_index=1,
        original_value_kinds=("SYNTHESIS",),
    )
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR
    assert "NO_SUBSTANTIVE_VALUE" in governance.reason_codes


def test_narration_interrupting_punchline_is_revision():
    plan = make_plan(
        blocks=[source_block(0), original_block(1, kind="EXPLANATION")],
        narration_need="RECOMMENDED",
        narration={
            "estimated_duration": 6.0,
            "placement_block_index": 0,
            "overlaps_source_audio": True,
            "essential": True,
        },
        original_value_kinds=("EXPLANATION",),
    )
    inputs = make_inputs([plan], source_moment_structure="JOKE")
    governance = _governance(plan, inputs)
    assert governance.status is GovernancePlanStatus.REVISION_REQUIRED
    assert "NARRATION_POSITION_DAMAGING" in governance.reason_codes
    assert governance.dimensions["source_moment_damage"] == "HIGH"


def test_funny_source_led_plan_with_no_narration_passes():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="Explain the context for the promotion rates fell sharply",
                kind="MISSING_CONTEXT",
            ),
        ],
        original_value_kinds=("MISSING_CONTEXT",),
    )
    inputs = make_inputs([plan], source_moment_structure="JOKE")
    governance = _governance(plan, inputs)
    assert governance.status in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }


def test_educational_clip_with_concise_explanation_passes():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="Explain why the promotion rates fell sharply",
                kind="EXPLANATION",
            ),
        ],
        strategy_type="EXPLANATORY",
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.APPROVED_FOR_SELECTION


def test_interview_claim_as_evidence_passes():
    plan = make_plan(
        strategy_type="SOURCE_AS_EVIDENCE",
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.APPROVED_FOR_SELECTION


def test_decontextualized_extreme_quote_hard_fails_fidelity():
    plan = make_plan()
    critique = make_critique(plan, fidelity="FAIL")
    governance = _governance(plan, provider_mode_deterministic=False, critique=critique)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR
    assert "SEMANTIC_DISTORTION" in governance.reason_codes


def test_sarcasm_literalized_fails_fidelity():
    plan = make_plan()
    critique = make_critique(plan, fidelity="FAIL")
    governance = _governance(plan, provider_mode_deterministic=False, critique=critique)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR


def test_speculation_to_fact_hard_fails():
    plan = make_plan()
    critique = make_critique(plan, fidelity="FAIL", unsupported_claim="CLEAR")
    governance = _governance(plan, provider_mode_deterministic=False, critique=critique)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR
    assert "UNSUPPORTED_CRITICAL_CLAIM" in governance.reason_codes


def test_counterpoint_with_linked_unverified_fact_is_blocked():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="Counter the claim using a linked external statistic",
                kind="COUNTERPOINT",
                grounding=(),
                dependency_ids=("claim-1",),
            ),
            verification_block(2, claim_id="claim-1", dependent=(1,)),
        ],
        original_value_kinds=("COUNTERPOINT",),
        verification_dependencies=(
            {"claim_id": "claim-1", "block_indexes": [1], "must_verify_before_execution": True},
        ),
    )
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION
    assert "EXTERNAL_VERIFICATION_REQUIRED" in governance.reason_codes
    assert governance.eligible_for_stage4_3 is False


def test_fabricated_numeric_claim_is_rejected():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="Sales in the region grew by 47 percent last year",
                kind="EXPLANATION",
                grounding=(),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR
    assert "UNSUPPORTED_CRITICAL_CLAIM" in governance.reason_codes


def test_high_source_ratio_with_source_as_evidence_not_auto_rejected():
    plan = make_plan(
        blocks=[
            source_block(0, duration=30.0, start=20.0),
            original_block(1, duration=5.0, kind="SOURCE_AS_EVIDENCE"),
        ],
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    inputs = make_inputs([plan])
    governance = _governance(plan, inputs)
    assert governance.dimensions["source_dominance"] == "LOW"
    assert governance.status is not GovernancePlanStatus.REJECTED_BY_GOVERNOR


def test_low_source_ratio_generic_filler_is_not_rewarded():
    plan = make_plan(
        blocks=[
            source_block(0, duration=5.0, start=20.0),
            original_block(
                1, duration=25.0, intent="very interesting insightful value", kind="SYNTHESIS"
            ),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    governance = _governance(plan)
    assert governance.status is not GovernancePlanStatus.APPROVED_FOR_SELECTION


def test_narration_none_is_valid():
    plan = make_plan(narration_need="NONE")
    governance = _governance(plan)
    assert governance.dimensions["narration_burden"] == "APPROPRIATE"
    assert governance.status is GovernancePlanStatus.APPROVED_FOR_SELECTION


def test_plan_level_template_structure_produces_template_risk():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(1, intent="interesting important note", kind="SYNTHESIS"),
            original_block(2, intent="very useful value", kind="SYNTHESIS"),
            original_block(3, intent="insightful important", kind="SYNTHESIS"),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    governance = _governance(plan)
    assert governance.dimensions["template_mass_produced_feel"] == "HIGH"
    assert "TEMPLATE_MASS_PRODUCED_FEEL" in governance.reason_codes


def test_account_level_repetition_is_deferred_to_stage7():
    plan = make_plan()
    governance = _governance(plan)
    assert governance.platform_risk["account_level_repetition"] == "DEFERRED_TO_STAGE_7"


def test_youtube_reused_content_risk_from_shared_evidence():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(1, intent="Add captions only", kind="SYNTHESIS"),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    governance = _governance(plan)
    youtube = governance.platform_risk["youtube"]
    assert youtube["reused_content"]["level"] == "HIGH"


def test_facebook_unoriginal_content_risk_is_independent():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(1, intent="Add captions only", kind="SYNTHESIS"),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    governance = _governance(plan)
    facebook = governance.platform_risk["facebook"]
    assert facebook["unoriginal_content"]["level"] == "HIGH"


def test_platform_results_contain_no_algorithm_certainty():
    plan = make_plan()
    governance = _governance(plan)
    import json

    serialized = json.dumps(governance.platform_risk).casefold()
    for phrase in (
        "safe for youtube",
        "safe for facebook",
        "guaranteed",
        "will not be flagged",
        "algorithm safe",
    ):
        assert phrase not in serialized
    assert governance.platform_risk["limitations"]


def test_platform_high_risk_alone_does_not_globally_hard_reject():
    plan = make_plan(
        blocks=[
            source_block(0, duration=20.0, start=20.0),
            original_block(1, duration=5.0, kind="SOURCE_AS_EVIDENCE"),
        ],
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    inputs = make_inputs([plan], rights_risk="ELEVATED", originality_risk="UNDETERMINED")
    governance = _governance(plan, inputs)
    assert governance.status is not GovernancePlanStatus.REJECTED_BY_GOVERNOR
    assert governance.status is not GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION


def test_ambiguous_plan_defers_when_provider_missing():
    plan = make_plan(
        blocks=[source_block(0), original_block(1, kind="EXPLANATION")],
        narration_need="RECOMMENDED",
        narration={"estimated_duration": 4.0, "placement_block_index": 1},
        original_value_kinds=("EXPLANATION",),
    )
    evaluation = _evaluate(plan, provider_mode_deterministic=False)
    assert evaluation.requires_semantic_review is True
    assert _status(evaluation) is GovernancePlanStatus.GOVERNANCE_DEFERRED


def test_strong_plan_does_not_require_semantic_review():
    plan = make_plan(
        strategy_type="SOURCE_AS_EVIDENCE",
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    evaluation = _evaluate(plan, provider_mode_deterministic=False)
    assert evaluation.requires_semantic_review is False
    assert _status(evaluation) is GovernancePlanStatus.APPROVED_FOR_SELECTION


def test_integrity_invalid_is_rejected():
    plan = make_plan(hero_index=5)
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR
    assert "PLAN_INTEGRITY_INVALID" in governance.reason_codes


def test_platform_guarantee_text_in_plan_is_rejected():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="This plan is safe for YouTube and guaranteed monetizable",
                kind="SYNTHESIS",
            ),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR


def test_tts_identity_text_in_plan_is_rejected():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="Use Gemini voice Charon to narrate this explanation",
                kind="SYNTHESIS",
            ),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR
    assert "TTS_IDENTITY_FORBIDDEN" in governance.reason_codes


def test_platform_evasion_text_in_plan_is_rejected():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="Mirror the video and pitch shift it to evade detection",
                kind="SYNTHESIS",
            ),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR
    assert "PLATFORM_EVASION_TACTIC" in governance.reason_codes


def test_provider_none_value_is_rejected():
    plan = make_plan()
    critique = make_critique(plan, value="NONE")
    governance = _governance(plan, provider_mode_deterministic=False, critique=critique)
    assert governance.status is GovernancePlanStatus.REJECTED_BY_GOVERNOR


def test_provider_unknown_on_required_evidence_defers():
    plan = make_plan(
        blocks=[source_block(0), original_block(1, kind="EXPLANATION")],
        narration_need="RECOMMENDED",
        narration={"estimated_duration": 4.0, "placement_block_index": 1},
        original_value_kinds=("EXPLANATION",),
    )
    critique = make_critique(plan, fidelity="UNKNOWN")
    governance = _governance(plan, provider_mode_deterministic=False, critique=critique)
    assert governance.status is GovernancePlanStatus.GOVERNANCE_DEFERRED


def test_ungrounded_non_numeric_external_claim_cannot_be_approved():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="The merger closes next Monday",
                kind="EXPLANATION",
                grounding=(),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION
    assert governance.status is not GovernancePlanStatus.APPROVED_FOR_SELECTION
    assert governance.eligible_for_stage4_3 is False
    assert governance.verification["claim_state"] == "EXTERNAL_REQUIRED_UNRESOLVED"
    assert "EXTERNAL_VERIFICATION_REQUIRED" in governance.reason_codes


def test_structural_citation_to_unrelated_evidence_is_not_grounded():
    plan = make_plan(
        blocks=[
            source_block(0, text="The source speaker discussed the weather and sports."),
            original_block(
                1,
                intent="The merger closes next Monday",
                kind="EXPLANATION",
                grounding=("block:0", "source:21-27"),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan)
    assert governance.verification["claim_state"] != "GROUNDED_IN_SOURCE"
    assert governance.status is GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION
    assert governance.eligible_for_stage4_3 is False


def test_factual_statement_supported_by_cited_wording_is_grounded():
    plan = make_plan(
        blocks=[
            source_block(0, text="The merger closes next Monday according to the filing."),
            original_block(
                1,
                intent="The merger closes next Monday",
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan)
    assert governance.verification["claim_state"] == "GROUNDED_IN_SOURCE"
    assert governance.status is GovernancePlanStatus.APPROVED_FOR_SELECTION


def test_faithful_near_exact_restatement_passes():
    plan = make_plan(
        blocks=[
            source_block(
                0,
                text="Remote work collapsed productivity as promotion rates fell sharply.",
            ),
            original_block(
                1,
                intent="Remote work collapsed productivity as promotion rates fell sharply",
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan)
    assert governance.verification["claim_state"] == "GROUNDED_IN_SOURCE"
    # Grounding is established; a near-exact restatement may still be flagged as
    # paraphrase by the independent redundancy dimension.
    assert governance.status not in {
        GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION,
        GovernancePlanStatus.GOVERNANCE_DEFERRED,
        GovernancePlanStatus.REJECTED_BY_GOVERNOR,
    }


def test_shared_entity_with_changed_predicate_is_not_grounded():
    plan = make_plan(
        blocks=[
            source_block(0, text="Microsoft announced a new product launch."),
            original_block(
                1,
                intent="Microsoft files bankruptcy",
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    evaluation = evaluate_plan(
        plan, make_inputs([plan]), DEFAULT_CONFIG, provider_mode_deterministic=False
    )
    assert evaluation.requires_semantic_review is True
    governance = _governance(plan, provider_mode_deterministic=False)
    assert governance.verification["claim_state"] != "GROUNDED_IN_SOURCE"
    assert governance.status is not GovernancePlanStatus.APPROVED_FOR_SELECTION
    assert governance.eligible_for_stage4_3 is False


def test_changed_predicate_with_same_entities_is_not_grounded():
    plan = make_plan(
        blocks=[
            source_block(0, text="The merger delays Friday."),
            original_block(
                1,
                intent="The merger closes Friday",
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan, provider_mode_deterministic=False)
    assert governance.verification["claim_state"] != "GROUNDED_IN_SOURCE"
    assert governance.status is not GovernancePlanStatus.APPROVED_FOR_SELECTION
    assert governance.eligible_for_stage4_3 is False


def test_direct_quote_supported_by_cited_source_passes():
    plan = make_plan(
        blocks=[
            source_block(0, text="We will double production by 2030."),
            original_block(
                1,
                intent='The speaker said "we will double production by 2030"',
                kind="SOURCE_AS_EVIDENCE",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    governance = _governance(plan)
    assert governance.verification["claim_state"] == "GROUNDED_IN_SOURCE"
    assert governance.status in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }


def test_non_factual_explanation_tied_to_cited_evidence_passes():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="Infer why the promotion rates fell sharply",
                kind="INFERENCE",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("INFERENCE",),
    )
    governance = _governance(plan)
    assert governance.verification["claim_state"] == "GROUNDED_IN_SOURCE"
    assert governance.status is GovernancePlanStatus.APPROVED_FOR_SELECTION
    assert "EXTERNAL_VERIFICATION_REQUIRED" not in governance.reason_codes


def test_ambiguous_claim_support_without_provider_never_approves():
    plan = make_plan(
        blocks=[
            source_block(0, text="The team discussed the annual plan in a closed meeting."),
            original_block(
                1,
                intent="Explain how the plan reflects broader strategy",
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    inputs = make_inputs([plan])
    evaluation = evaluate_plan(plan, inputs, DEFAULT_CONFIG, provider_mode_deterministic=False)
    assert evaluation.requires_semantic_review is True
    governance = _governance(plan, provider_mode_deterministic=False)
    assert governance.status is GovernancePlanStatus.GOVERNANCE_DEFERRED
    assert governance.status not in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }


def test_supported_claim_requires_no_semantic_review():
    plan = make_plan()
    inputs = make_inputs([plan])
    evaluation = evaluate_plan(plan, inputs, DEFAULT_CONFIG, provider_mode_deterministic=False)
    assert evaluation.requires_semantic_review is False


def test_arbitrary_grounding_labels_do_not_approve_external_claim():
    for label in ("strategy", "source_excerpt", "excerpt", "evidence", "label"):
        plan = make_plan(
            fingerprint=f"plan-fingerprint-{label.replace(' ', '-')}",
            blocks=[
                source_block(0),
                original_block(
                    1,
                    intent="The merger closes next Monday",
                    kind="EXPLANATION",
                    grounding=(label,),
                ),
            ],
            original_value_kinds=("EXPLANATION",),
        )
        governance = _governance(plan)
        assert governance.status is GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION
        assert governance.eligible_for_stage4_3 is False


def test_external_claim_with_unresolvable_grounding_remains_ineligible():
    plan = make_plan(
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="The merger closes next Monday",
                kind="EXPLANATION",
                grounding=("block:99", "word:900-950", "span:1-2"),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan)
    assert governance.status is GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION
    assert governance.eligible_for_stage4_3 is False
    assert governance.verification["claim_state"] == "EXTERNAL_REQUIRED_UNRESOLVED"


def test_strict_hero_thresholds_are_config_governed():
    from app.transformation.governance.policy import Stage42Config

    plan = make_plan(
        blocks=[
            original_block(0, kind="EXPLANATION", duration=2.0, grounding=("block:1",)),
            source_block(1, duration=6.0),
        ],
        hero_index=1,
        original_value_kinds=("EXPLANATION",),
    )
    inputs = make_inputs([plan], stage40_assessments={"moment_density": 0.5})
    default_eval = evaluate_plan(plan, inputs, DEFAULT_CONFIG, provider_mode_deterministic=True)
    assert finalize(default_eval)[0] is GovernancePlanStatus.REVISION_REQUIRED

    relaxed = Stage42Config(short_moment_seconds=3.0, high_moment_density_floor=0.9)
    relaxed_eval = evaluate_plan(plan, inputs, relaxed, provider_mode_deterministic=True)
    assert finalize(relaxed_eval)[0] is GovernancePlanStatus.APPROVED_FOR_SELECTION


def test_strict_hero_thresholds_invalidate_fingerprint():
    from app.transformation.governance.fingerprints import (
        build_governance_input_payload,
        governance_input_fingerprint,
    )
    from app.transformation.governance.policy import Stage42Config

    plan = make_plan()
    inputs = make_inputs([plan])
    base = governance_input_fingerprint(
        build_governance_input_payload(
            inputs=inputs,
            config=DEFAULT_CONFIG,
            provider_mode="deterministic",
            provider_identity={"provider": "deterministic"},
        )
    )
    changed = governance_input_fingerprint(
        build_governance_input_payload(
            inputs=inputs,
            config=Stage42Config(short_moment_seconds=9.0),
            provider_mode="deterministic",
            provider_identity={"provider": "deterministic"},
        )
    )
    assert base != changed


def test_direct_quote_with_attribution_is_grounded_and_eligible():
    plan = make_plan(
        blocks=[
            source_block(0, text="We will double production by 2030."),
            original_block(
                1,
                intent='According to the source, "we will double production by 2030"',
                kind="SOURCE_AS_EVIDENCE",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    governance = _governance(plan)
    assert governance.verification["claim_state"] == "GROUNDED_IN_SOURCE"
    assert governance.status in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }
    assert governance.eligible_for_stage4_3 is True


def test_quote_with_added_factual_predicate_is_not_grounded():
    plan = make_plan(
        blocks=[
            source_block(0, text="Microsoft launches product today."),
            original_block(
                1,
                intent="According to the source, Microsoft launches product and files bankruptcy",
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    evaluation = evaluate_plan(
        plan, make_inputs([plan]), DEFAULT_CONFIG, provider_mode_deterministic=False
    )
    assert evaluation.requires_semantic_review is True
    assert evaluation.verification["claim_state"] == "SUPPORT_UNVERIFIED"
    governance = _governance(plan, provider_mode_deterministic=False)
    assert governance.verification["claim_state"] != "GROUNDED_IN_SOURCE"
    assert governance.status is GovernancePlanStatus.GOVERNANCE_DEFERRED
    assert governance.status not in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }
    assert governance.eligible_for_stage4_3 is False


def test_quote_with_changed_predicate_is_not_grounded():
    plan = make_plan(
        blocks=[
            source_block(0, text='The speaker said "we will expand in 2026".'),
            original_block(
                1,
                intent='The speaker said "we will expand in 2026" and the merger closes Friday',
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan, provider_mode_deterministic=False)
    assert governance.verification["claim_state"] != "GROUNDED_IN_SOURCE"
    assert governance.status is not GovernancePlanStatus.APPROVED_FOR_SELECTION
    assert governance.eligible_for_stage4_3 is False


def test_quote_with_unsupported_number_is_not_grounded():
    plan = make_plan(
        blocks=[
            source_block(0, text="We will double production."),
            original_block(
                1,
                intent='According to the source, "we will double production" by 47 percent',
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan, provider_mode_deterministic=False)
    assert governance.verification["claim_state"] != "GROUNDED_IN_SOURCE"
    assert governance.status not in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }
    assert governance.eligible_for_stage4_3 is False


def test_quote_with_unsupported_event_is_not_grounded():
    plan = make_plan(
        blocks=[
            source_block(0, text='The speaker said "we will expand in 2026".'),
            original_block(
                1,
                intent='The speaker said "we will expand in 2026" at the annual conference',
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan, provider_mode_deterministic=False)
    assert governance.verification["claim_state"] != "GROUNDED_IN_SOURCE"
    assert governance.eligible_for_stage4_3 is False


def test_quote_with_interpretive_framing_remains_grounded():
    plan = make_plan(
        blocks=[
            source_block(0, text="We will double production by 2030."),
            original_block(
                1,
                intent='The source explains that "we will double production by 2030"',
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
    )
    governance = _governance(plan)
    assert governance.verification["claim_state"] == "GROUNDED_IN_SOURCE"
    assert governance.status in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }


def test_quote_with_added_claim_is_blocked_when_external_dependency_exists():
    plan = make_plan(
        blocks=[
            source_block(0, text="Microsoft launches product today."),
            original_block(
                1,
                intent="According to the source, Microsoft launches product and files bankruptcy",
                kind="EXPLANATION",
                grounding=("block:0",),
            ),
        ],
        original_value_kinds=("EXPLANATION",),
        verification_dependencies=(
            {"claim_id": "claim-9", "block_indexes": [1], "must_verify_before_execution": True},
        ),
    )
    governance = _governance(plan)
    assert governance.verification["claim_state"] == "EXTERNAL_REQUIRED_UNRESOLVED"
    assert governance.status is GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION
    assert governance.eligible_for_stage4_3 is False
