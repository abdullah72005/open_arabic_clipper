"""Stable Stage 4.1 fingerprint composition.

Plan-set input identity covers every output-relevant semantic, but deliberately
excludes the Gemini key, transient admission/cooldown/outage state, TTS
provider/model/voice/fallback voice, speech-generation settings, rendering
configuration, publishing schedule, and unrelated frontend settings. Output
identity covers the ordered current validated plan representation and terminal
per-strategy outcomes, excluding transient timing, metrics, and availability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.pipeline.fingerprints import canonical_fingerprint
from app.transformation.planning.policy import (
    INPUT_FINGERPRINT_VERSION,
    OUTPUT_FINGERPRINT_VERSION,
    PLAN_FINGERPRINT_VERSION,
    PROVIDER_INPUT_FINGERPRINT_VERSION,
    Stage41Config,
    stage41_config_payload,
    stage41_policy_payload,
)
from app.transformation.planning.types import (
    PlanningContext,
    PlanningInputs,
    StrategyAttempt,
    ValidatedPlan,
)


def planning_input_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint(
        "transformation-planning-input", INPUT_FINGERPRINT_VERSION, dict(payload)
    )


def planning_output_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint(
        "transformation-planning-output", OUTPUT_FINGERPRINT_VERSION, dict(payload)
    )


def plan_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint("transformation-plan", PLAN_FINGERPRINT_VERSION, dict(payload))


def provider_input_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint(
        "transformation-planning-provider-input",
        PROVIDER_INPUT_FINGERPRINT_VERSION,
        dict(payload),
    )


def context_semantic_payload(context: PlanningContext) -> dict[str, object]:
    """Planning-semantic context only; never TTS/voice/rendering fields."""

    return context.semantic_payload()


def build_plan_set_input_payload(
    *,
    inputs: PlanningInputs,
    config: Stage41Config,
    provider_mode: str,
    provider_identity: Mapping[str, object],
) -> dict[str, object]:
    return {
        "candidate": {
            "id": inputs.candidate_id,
            "key": inputs.candidate_key,
            "source_id": inputs.source_id,
            "content_type": inputs.content_type.value,
        },
        "stage40": {
            "analysis_id": inputs.stage40_analysis_id,
            "input_fingerprint": inputs.stage40_input_fingerprint,
            "output_fingerprint": inputs.stage40_output_fingerprint,
            "policy_version": inputs.stage40_policy_version,
            "assessments": dict(inputs.stage40_assessments),
            "platform_risk": dict(inputs.stage40_platform_risk),
            "source_moment": dict(inputs.source_moment),
            "strategies": [
                {
                    "id": item.get("id"),
                    "key": item.get("strategy_key"),
                    "type": item.get("strategy_type"),
                    "rank": item.get("rank"),
                    "fingerprint": item.get("strategy_fingerprint"),
                    "external_verification_requirement": item.get(
                        "external_verification_requirement"
                    ),
                }
                for item in inputs.stage40_strategies
            ],
        },
        "refinement": {
            "priority": inputs.refinement_priority,
            "quality_level": inputs.refinement_quality_level,
            "status": inputs.refinement_status,
            "output_fingerprint": inputs.refinement_output_fingerprint,
            "transcript": inputs.transcript,
            "transcript_confidence": inputs.transcript_confidence,
            "refined_start": inputs.refined_start,
            "refined_end": inputs.refined_end,
            "words": [word.as_dict() for word in inputs.words],
            "word_coverage_sufficient": inputs.word_coverage_sufficient,
            "unresolved_spans": [dict(item) for item in inputs.unresolved_spans],
            "entities": [dict(item) for item in inputs.entities],
            "dialect_profile": inputs.dialect_profile,
            "dialect_confidence": inputs.dialect_confidence,
            "code_switch": dict(inputs.code_switch),
        },
        "context": list(inputs.context_segments),
        "provenance": {
            "rights_status": inputs.rights_status,
            "rights_risk": inputs.rights_risk,
            "originality_risk": inputs.originality_risk,
            "media_origin": inputs.media_origin,
            "snapshot": dict(inputs.provenance_snapshot),
            "stage3_risk": dict(inputs.stage3_risk),
        },
        "target_context": context_semantic_payload(inputs.planning_context),
        "policy": stage41_policy_payload(),
        "config": stage41_config_payload(config),
        "provider": {
            "mode": provider_mode,
            "identity": dict(provider_identity),
        },
    }


def build_plan_set_output_payload(
    *,
    semantic_outcome: str,
    plans: Sequence[ValidatedPlan],
    attempts: Sequence[StrategyAttempt],
) -> dict[str, object]:
    ordered = sorted(plans, key=lambda item: item.generation_rank)
    return {
        "semantic_outcome": semantic_outcome,
        "plans": [
            {
                "plan_key": plan.plan_key,
                "strategy_id": plan.strategy_id,
                "status": plan.status.value,
                "generation_rank": plan.generation_rank,
                "plan_output_fingerprint": plan.plan_output_fingerprint,
                "structure_signature": plan.structure_signature,
            }
            for plan in ordered
        ],
        "attempts": [
            {
                "strategy_id": attempt.strategy_id,
                "strategy_key": attempt.strategy_key,
                "status": attempt.status,
                "reasons": list(attempt.reasons),
            }
            for attempt in sorted(attempts, key=lambda item: item.strategy_key)
        ],
    }


def build_plan_output_payload(plan: ValidatedPlan) -> dict[str, object]:
    return {
        "strategy_id": plan.strategy_id,
        "strategy_key": plan.strategy_key,
        "strategy_type": plan.strategy_type.value,
        "status": plan.status.value,
        "blocks": [block.as_dict() for block in plan.blocks],
        "hero_block_index": plan.hero_block_index,
        "hero_source_start": round(plan.hero_source_start, 3),
        "hero_source_end": round(plan.hero_source_end, 3),
        "narration": plan.narration.as_dict(),
        "external_fact_dependencies": [dict(item) for item in plan.external_fact_dependencies],
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
