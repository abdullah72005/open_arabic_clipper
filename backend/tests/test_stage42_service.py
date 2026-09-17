"""Stage 4.2 service orchestration: routing, checkpoints, summary precedence."""

from __future__ import annotations

from stage42_support import (
    FakeGovernanceProvider,
    make_critique,
    make_inputs,
    make_plan,
    original_block,
    source_block,
    verification_block,
)

from app.core.enums import GovernancePlanStatus, GovernanceSemanticOutcome, SemanticProviderMode
from app.transformation.governance.policy import DEFAULT_CONFIG
from app.transformation.governance.service import GovernanceService


def _inputs(plans):
    return make_inputs(plans)


def test_candidate_outcome_precedence_and_independent_counts():
    approved = make_plan(plan_id="a" * 8, fingerprint="fp-a")
    blocked = make_plan(
        plan_id="b" * 8,
        fingerprint="fp-b",
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
    )
    rejected = make_plan(
        plan_id="c" * 8,
        fingerprint="fp-c",
        blocks=[
            source_block(0),
            original_block(1, intent="Add captions and crop the video", kind="SYNTHESIS"),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    inputs = _inputs([approved, blocked, rejected])
    service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=None,
        provider_identity={"provider": "deterministic"},
        mode=SemanticProviderMode.DETERMINISTIC,
    )
    outcome = service.govern(inputs, input_fingerprint="in")
    assert outcome.semantic_outcome is GovernanceSemanticOutcome.PLANS_ELIGIBLE_FOR_SELECTION
    assert outcome.summary_counts["approved"] == 1
    assert outcome.summary_counts["verification_blocked"] == 1
    assert outcome.summary_counts["rejected"] == 1
    statuses = {plan.plan_id: plan.status for plan in outcome.plans}
    assert statuses[approved.plan_id] is GovernancePlanStatus.APPROVED_FOR_SELECTION
    assert statuses[blocked.plan_id] is GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION
    assert statuses[rejected.plan_id] is GovernancePlanStatus.REJECTED_BY_GOVERNOR


def test_no_governor_approved_plan_when_all_ineligible():
    blocked = make_plan(
        plan_id="b" * 8,
        fingerprint="fp-b",
        blocks=[
            source_block(0),
            original_block(
                1,
                intent="Counter with a linked external statistic",
                kind="COUNTERPOINT",
                grounding=(),
                dependency_ids=("claim-1",),
            ),
            verification_block(2, claim_id="claim-1", dependent=(1,)),
        ],
        original_value_kinds=("COUNTERPOINT",),
    )
    rejected = make_plan(
        plan_id="c" * 8,
        fingerprint="fp-c",
        blocks=[
            source_block(0),
            original_block(1, intent="Add captions only", kind="SYNTHESIS"),
        ],
        original_value_kinds=("SYNTHESIS",),
    )
    service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=None,
        provider_identity={"provider": "deterministic"},
        mode=SemanticProviderMode.DETERMINISTIC,
    )
    outcome = service.govern(_inputs([blocked, rejected]), input_fingerprint="in")
    assert outcome.semantic_outcome is GovernanceSemanticOutcome.NO_GOVERNOR_APPROVED_PLAN
    assert outcome.summary_counts["deferred"] == 0
    assert outcome.summary_counts["verification_blocked"] == 1
    assert outcome.summary_counts["rejected"] == 1


def test_governance_deferred_outcome_when_only_deferred():
    plan = make_plan(
        plan_id="d" * 8,
        fingerprint="fp-d",
        blocks=[source_block(0), original_block(1, kind="EXPLANATION")],
        narration_need="RECOMMENDED",
        narration={"estimated_duration": 4.0, "placement_block_index": 1},
        original_value_kinds=("EXPLANATION",),
    )
    service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=None,
        provider_identity={"provider": "deterministic"},
        mode=SemanticProviderMode.ADAPTIVE,
    )
    outcome = service.govern(_inputs([plan]), input_fingerprint="in")
    assert outcome.semantic_outcome is GovernanceSemanticOutcome.GOVERNANCE_DEFERRED
    assert outcome.cache_eligible is False
    assert outcome.summary_counts["deferred"] == 1


def test_approved_with_caution_is_cache_eligible_and_eligible():
    plan = make_plan(
        plan_id="e" * 8,
        fingerprint="fp-e",
        blocks=[source_block(0), original_block(1, kind="SOURCE_AS_EVIDENCE")],
        narration_need="RECOMMENDED",
        narration={"estimated_duration": 4.0, "placement_block_index": 1},
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )
    critique = make_critique(plan, fidelity="CONCERN")
    provider = FakeGovernanceProvider({plan.plan_id: critique})
    service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=provider,
        provider_identity={"provider": "fake"},
        mode=SemanticProviderMode.ADAPTIVE,
    )
    outcome = service.govern(_inputs([plan]), input_fingerprint="in")
    governance = outcome.plans[0]
    assert governance.status is GovernancePlanStatus.APPROVED_WITH_CAUTION
    assert governance.eligible_for_stage4_3 is True
    assert outcome.cache_eligible is True
    assert provider.calls == 1


def test_provider_call_ceiling_is_two_raw_calls():
    first = make_plan(
        plan_id="f" * 8,
        fingerprint="fp-f",
        blocks=[source_block(0), original_block(1, kind="EXPLANATION")],
        narration_need="RECOMMENDED",
        narration={"estimated_duration": 4.0, "placement_block_index": 1},
        original_value_kinds=("EXPLANATION",),
    )
    second = make_plan(
        plan_id="g" * 8,
        fingerprint="fp-g",
        blocks=[source_block(0), original_block(1, kind="EXPLANATION")],
        narration_need="RECOMMENDED",
        narration={"estimated_duration": 4.0, "placement_block_index": 1},
        original_value_kinds=("EXPLANATION",),
    )
    unknown_a = make_critique(first, fidelity="UNKNOWN")
    unknown_b = make_critique(second, fidelity="UNKNOWN")
    provider = FakeGovernanceProvider({first.plan_id: unknown_a, second.plan_id: unknown_b})
    service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=provider,
        provider_identity={"provider": "fake"},
        mode=SemanticProviderMode.ADAPTIVE,
    )
    outcome = service.govern(_inputs([first, second]), input_fingerprint="in")
    assert provider.calls <= DEFAULT_CONFIG.max_hosted_raw_calls
    assert outcome.semantic_outcome is GovernanceSemanticOutcome.GOVERNANCE_DEFERRED


def test_second_strong_call_only_for_unknown_plans():
    strong = make_plan(
        plan_id="h" * 8,
        fingerprint="fp-h",
        strategy_type="ANALYSIS",
        blocks=[source_block(0), original_block(1, kind="AUTHORED_THESIS")],
        narration_need="RECOMMENDED",
        narration={"estimated_duration": 4.0, "placement_block_index": 1},
        original_value_kinds=("AUTHORED_THESIS",),
    )
    known = make_critique(strong, fidelity="PASS")
    provider = FakeGovernanceProvider({strong.plan_id: known})
    service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=provider,
        provider_identity={"provider": "fake"},
        mode=SemanticProviderMode.ADAPTIVE,
    )
    outcome = service.govern(_inputs([strong]), input_fingerprint="in")
    # Strong-tier first call only; a PASS critique needs no second call.
    assert provider.tiers == ["STRONG"]
    assert outcome.plans[0].status in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }


def test_checkpoints_are_reused_without_provider_call():
    plan = make_plan(
        plan_id="i" * 8,
        fingerprint="fp-i",
        blocks=[source_block(0), original_block(1, kind="EXPLANATION")],
        narration_need="RECOMMENDED",
        narration={"estimated_duration": 4.0, "placement_block_index": 1},
        original_value_kinds=("EXPLANATION",),
    )
    inputs = _inputs([plan])
    provider = FakeGovernanceProvider(auto=True)
    service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=provider,
        provider_identity={"provider": "fake"},
        mode=SemanticProviderMode.ADAPTIVE,
    )
    first = service.govern(inputs, input_fingerprint="in")
    assert provider.calls == 1
    fingerprint = first.attempts[0].provider_input_fingerprint
    checkpoints = {
        plan.plan_id: {
            "provider_input_fingerprint": fingerprint,
            "critique": first.attempts[0].checkpoint["critique"],  # type: ignore[index]
        }
    }
    second_provider = FakeGovernanceProvider(auto=True)
    second_service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=second_provider,
        provider_identity={"provider": "fake"},
        mode=SemanticProviderMode.ADAPTIVE,
    )
    second = second_service.govern(inputs, input_fingerprint="in", checkpoints=checkpoints)
    assert second_provider.calls == 0
    assert second.attempts[0].status == "REUSED"


def test_second_strong_call_checkpoint_is_persisted_and_reused():
    plan = make_plan(
        plan_id="j" * 8,
        fingerprint="fp-j",
        blocks=[source_block(0), original_block(1, kind="EXPLANATION")],
        narration_need="RECOMMENDED",
        narration={"estimated_duration": 4.0, "placement_block_index": 1},
        original_value_kinds=("EXPLANATION",),
    )
    inputs = _inputs([plan])
    unknown = make_critique(plan, fidelity="UNKNOWN")
    resolved = make_critique(plan, fidelity="PASS")
    provider = FakeGovernanceProvider(responses=[{plan.plan_id: unknown}, {plan.plan_id: resolved}])
    service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=provider,
        provider_identity={"provider": "fake"},
        mode=SemanticProviderMode.ADAPTIVE,
    )
    outcome = service.govern(inputs, input_fingerprint="in")
    assert provider.calls == 2
    persisted = outcome.attempts[-1]
    assert persisted.checkpoint is not None
    assert persisted.checkpoint["critique"]["fidelity"] == "PASS"  # type: ignore[index]

    replay_provider = FakeGovernanceProvider(auto=True)
    replay_service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=replay_provider,
        provider_identity={"provider": "fake"},
        mode=SemanticProviderMode.ADAPTIVE,
    )
    checkpoints = {
        plan.plan_id: {
            "provider_input_fingerprint": persisted.provider_input_fingerprint,
            "critique": persisted.checkpoint["critique"],  # type: ignore[index]
        }
    }
    replay = replay_service.govern(inputs, input_fingerprint="in", checkpoints=checkpoints)
    assert replay_provider.calls == 0
    assert replay.attempts[0].status == "REUSED"


def test_second_strong_call_failure_stays_truthfully_deferred():
    plan = make_plan(
        plan_id="k" * 8,
        fingerprint="fp-k",
        blocks=[source_block(0), original_block(1, kind="EXPLANATION")],
        narration_need="RECOMMENDED",
        narration={"estimated_duration": 4.0, "placement_block_index": 1},
        original_value_kinds=("EXPLANATION",),
    )
    inputs = _inputs([plan])
    unknown = make_critique(plan, fidelity="UNKNOWN")
    provider = FakeGovernanceProvider(responses=[{plan.plan_id: unknown}, None])
    service = GovernanceService(
        config=DEFAULT_CONFIG,
        provider=provider,
        provider_identity={"provider": "fake"},
        mode=SemanticProviderMode.ADAPTIVE,
    )
    outcome = service.govern(inputs, input_fingerprint="in")
    assert provider.calls == 2
    assert outcome.semantic_outcome is GovernanceSemanticOutcome.GOVERNANCE_DEFERRED
    # The first accepted critique remains checkpointed for a later retry.
    assert outcome.attempts[-1].checkpoint is not None
