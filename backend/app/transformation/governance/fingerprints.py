"""Stable Stage 4.2 governance fingerprint composition.

Input identity covers every output-relevant semantic input but deliberately
excludes the Gemini key, transient admission/cooldown/outage state, TTS
provider/model/voice/fallback, speech-generation settings, rendering
configuration, publishing schedule/settings, metadata, and unrelated frontend
settings. Output identity covers the complete ordered governance results and the
candidate summary, excluding transient timing, metrics, and availability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.pipeline.fingerprints import canonical_fingerprint
from app.transformation.governance.policy import (
    INPUT_FINGERPRINT_VERSION,
    OUTPUT_FINGERPRINT_VERSION,
    PLAN_GOVERNANCE_FINGERPRINT_VERSION,
    PROVIDER_INPUT_FINGERPRINT_VERSION,
    Stage42Config,
    governance_config_payload,
    governance_policy_payload,
    platform_policy_payload,
)
from app.transformation.governance.types import (
    GovernanceAttempt,
    GovernanceInputs,
    PlanEvidence,
    PlanGovernance,
)


def governance_input_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint(
        "transformation-governance-input", INPUT_FINGERPRINT_VERSION, dict(payload)
    )


def governance_output_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint(
        "transformation-governance-output", OUTPUT_FINGERPRINT_VERSION, dict(payload)
    )


def plan_governance_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint(
        "transformation-governance-plan", PLAN_GOVERNANCE_FINGERPRINT_VERSION, dict(payload)
    )


def provider_input_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint(
        "transformation-governance-provider-input",
        PROVIDER_INPUT_FINGERPRINT_VERSION,
        dict(payload),
    )


def build_governance_input_payload(
    *,
    inputs: GovernanceInputs,
    config: Stage42Config,
    provider_mode: str,
    provider_identity: Mapping[str, object],
) -> dict[str, object]:
    return {
        "candidate": {
            "id": inputs.candidate_id,
            "key": inputs.candidate_key,
            "source_id": inputs.source_id,
            "disposition": inputs.disposition,
            "content_type": inputs.content_type,
            "source_moment_structure": inputs.source_moment_structure,
        },
        "governance_scope": {"plan_set_id": inputs.plan_set_id},
        "stage41": {
            "plan_set_input_fingerprint": inputs.plan_set_input_fingerprint,
            "plan_set_output_fingerprint": inputs.plan_set_output_fingerprint,
            "semantic_outcome": inputs.plan_set_semantic_outcome,
        },
        "stage40": {
            "analysis_id": inputs.stage40_analysis_id,
            "input_fingerprint": inputs.stage40_input_fingerprint,
            "output_fingerprint": inputs.stage40_output_fingerprint,
            "policy_version": inputs.stage40_policy_version,
            "assessments": dict(inputs.stage40_assessments),
            "platform_risk": dict(inputs.stage40_platform_risk),
            "strategies": [
                {
                    "plan_id": plan.plan_id,
                    "strategy_id": plan.strategy_id,
                    "strategy_type": plan.strategy_type,
                    "strategy_fingerprint": plan.strategy_fingerprint,
                    "intensity": plan.intensity,
                }
                for plan in inputs.plans
            ],
        },
        "plans": [_plan_input_payload(plan) for plan in inputs.plans],
        "refinement": {
            "id": inputs.refinement_id,
            "priority": inputs.refinement_priority,
            "quality_level": inputs.refinement_quality_level,
            "status": inputs.refinement_status,
            "output_fingerprint": inputs.refinement_output_fingerprint,
            "transcript": inputs.transcript,
            "transcript_confidence": inputs.transcript_confidence,
            "refined_start": inputs.refined_start,
            "refined_end": inputs.refined_end,
            "context": list(inputs.context_segments),
            "dialect_profile": inputs.dialect_profile,
            "dialect_confidence": inputs.dialect_confidence,
            "code_switch": dict(inputs.code_switch),
            "idea_summary": inputs.idea_summary,
            "topic_summary": inputs.topic_summary,
            "hooks": [dict(item) for item in inputs.hooks],
            "stage3_risk": dict(inputs.stage3_risk),
        },
        "provenance": {
            "rights_status": inputs.rights_status,
            "rights_risk": inputs.rights_risk,
            "originality_risk": inputs.originality_risk,
            "media_origin": inputs.media_origin,
            "snapshot": dict(inputs.provenance_snapshot),
        },
        "target_context": inputs.target_context.semantic_payload(),
        "policy": governance_policy_payload(),
        "platform_policy": platform_policy_payload(),
        "config": governance_config_payload(config),
        "provider": {"mode": provider_mode, "identity": dict(provider_identity)},
    }


def _plan_input_payload(plan: PlanEvidence) -> dict[str, object]:
    return {
        "plan_id": plan.plan_id,
        "plan_key": plan.plan_key,
        "plan_output_fingerprint": plan.plan_output_fingerprint,
        "provider_input_fingerprint": plan.provider_input_fingerprint,
        "strategy": {
            "id": plan.strategy_id,
            "key": plan.strategy_key,
            "type": plan.strategy_type,
            "fingerprint": plan.strategy_fingerprint,
            "rank": plan.strategy_rank,
            "intensity": plan.intensity,
            "snapshot": dict(plan.strategy_snapshot),
        },
        "status": plan.status,
        "generation_rank": plan.generation_rank,
        "blocks": [dict(block) for block in plan.blocks],
        "hero": {
            "block_index": plan.hero_block_index,
            "source_start": plan.hero_source_start,
            "source_end": plan.hero_source_end,
            "appearance_time": plan.hero_appearance_time,
        },
        "narration": dict(plan.narration),
        "verification_dependencies": [dict(item) for item in plan.verification_dependencies],
        "derived_durations": dict(plan.derived_durations),
        "hook_payoff_evidence": dict(plan.hook_payoff_evidence),
        "original_value_kinds": list(plan.original_value_kinds),
        "original_value_reasons": list(plan.original_value_reasons),
        "preservation_constraints": list(plan.preservation_constraints),
        "required_context": list(plan.required_context),
        "stage40_risk": dict(plan.stage40_risk),
        "source_dialect": dict(plan.source_dialect),
    }


def build_governance_output_payload(
    *,
    semantic_outcome: str,
    plans: Sequence[PlanGovernance],
    summary_counts: Mapping[str, int],
    attempts: Sequence[GovernanceAttempt],
) -> dict[str, object]:
    return {
        "semantic_outcome": semantic_outcome,
        "summary_counts": dict(summary_counts),
        "plans": [
            {
                "plan_id": plan.plan_id,
                "plan_output_fingerprint": plan.plan_output_fingerprint,
                "status": plan.status.value,
                "eligible_for_stage4_3": plan.eligible_for_stage4_3,
                "output_fingerprint": plan.output_fingerprint,
                "reason_codes": list(plan.reason_codes),
            }
            for plan in plans
        ],
        "attempts": [
            {
                "plan_id": attempt.plan_id,
                "plan_output_fingerprint": attempt.plan_output_fingerprint,
                "status": attempt.status,
                "reasons": list(attempt.reasons),
            }
            for attempt in attempts
        ],
    }


def build_provider_input_payload(
    *,
    request: Mapping[str, object],
    provider_identity: Mapping[str, object],
    route: str,
) -> dict[str, object]:
    return {
        "request": dict(request),
        "provider_identity": dict(provider_identity),
        "route": route,
    }


def build_plan_governance_output_payload(plan: PlanGovernance) -> dict[str, object]:
    return {
        "plan_id": plan.plan_id,
        "plan_output_fingerprint": plan.plan_output_fingerprint,
        "status": plan.status.value,
        "eligible_for_stage4_3": plan.eligible_for_stage4_3,
        "severity": plan.severity.value,
        "hard_gates": [dict(item) for item in plan.hard_gates],
        "dimensions": dict(plan.dimensions),
        "verification": dict(plan.verification),
        "platform_risk": dict(plan.platform_risk),
        "reason_codes": list(plan.reason_codes),
        "warnings": [dict(item) for item in plan.warnings],
        "remediation": [dict(item) for item in plan.remediation],
    }
