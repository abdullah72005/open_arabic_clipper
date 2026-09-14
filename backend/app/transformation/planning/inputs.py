"""Bounded Stage 4.1 input assembly from a current Stage 4.0 handoff.

Uses the frozen Stage 4.0 handoff as the readiness/freshness gate, reloads the
selected analysis/refinement by persisted identity, and recovers the exact
bounded nearby context through Stage 4.0's own bounded input assembly. Never
loads a whole video or a whole multi-hour transcript, and never recomputes
Stage 4.0 eligibility or strategy ranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.core.enums import SourceMomentStructure, StrategyDisposition
from app.core.settings import Settings
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    TransformationEligibilityAnalysis,
    TransformationStrategyCandidate,
)
from app.transformation.handoff import build_stage4_1_handoff
from app.transformation.inputs import build_transformation_inputs
from app.transformation.planning.policy import Stage41Config
from app.transformation.planning.types import (
    PlanningContext,
    PlanningInputs,
    WordEvidence,
    word_evidence_from_mapping,
)
from app.transformation.queue import list_strategies


class PlanningInputError(ValueError):
    """A Stage 4.1 input prerequisite is missing or stale."""


def _recommended_current(
    analysis: TransformationEligibilityAnalysis, session: object
) -> list[TransformationStrategyCandidate]:
    rows = list_strategies(session, analysis.id)  # type: ignore[arg-type]
    return [
        row for row in rows if row.is_current and row.disposition is StrategyDisposition.RECOMMENDED
    ]


def _bounded_words(
    raw: Sequence[object], config: Stage41Config
) -> tuple[tuple[WordEvidence, ...], bool]:
    words: list[WordEvidence] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        parsed = word_evidence_from_mapping(item, len(words))
        if parsed is None:
            continue
        words.append(parsed)
        if len(words) >= config.provider_max_words:
            break
    return tuple(words), bool(words)


def _coverage_sufficient(
    words: Sequence[WordEvidence], refined_start: float, refined_end: float
) -> bool:
    if not words:
        return False
    duration = max(0.0, refined_end - refined_start)
    if duration <= 0:
        return False
    span = max(word.end for word in words) - min(word.start for word in words)
    return (span / duration) >= 0.5


def build_planning_inputs(
    session: object,
    candidate: ClipCandidate,
    analysis: TransformationEligibilityAnalysis,
    refinement: CandidateRefinement,
    settings: Settings,
    config: Stage41Config,
) -> PlanningInputs:
    """Assemble bounded, output-relevant Stage 4.1 inputs or raise."""

    handoff = build_stage4_1_handoff(session, candidate.id)
    if handoff is None:
        raise PlanningInputError("candidate is missing")
    if not handoff.get("ready_for_stage4_1"):
        reason = "STALE_STAGE40" if handoff.get("stale") else "STAGE40_NOT_READY"
        raise PlanningInputError(reason)
    if handoff.get("analysis_id") != str(analysis.id):
        raise PlanningInputError("STALE_STAGE40")

    recommended = _recommended_current(analysis, session)
    if not recommended:
        raise PlanningInputError("NO_CURRENT_RECOMMENDED_STRATEGY")
    if len(recommended) > config.max_plans:
        recommended = recommended[: config.max_plans]

    stage40 = build_transformation_inputs(
        session,
        candidate,
        refinement,
        settings.stage40_config(),  # type: ignore[arg-type]
    )

    raw_words = tuple(
        item for item in (refinement.word_timestamps or []) if isinstance(item, Mapping)
    )
    words, _has_words = _bounded_words(raw_words, config)
    coverage = _coverage_sufficient(words, stage40.effective_start, stage40.effective_end)

    strategies: list[Mapping[str, object]] = []
    for row in recommended:
        strategies.append(
            {
                "id": str(row.id),
                "strategy_key": row.strategy_key,
                "strategy_type": row.strategy_type.value,
                "rank": row.rank,
                "intensity": row.intensity.value,
                "direction_summary": row.direction_summary,
                "added_value_focus": row.added_value_focus,
                "substantive_value_kind": row.substantive_value_kind.value,
                "source_moment_role": row.source_moment_role,
                "preservation_requirements": list(row.preservation_requirements or []),
                "external_verification_requirement": row.external_verification_requirement.value,
                "verification_requirements": list(row.verification_requirements or []),
                "strategy_fingerprint": row.strategy_fingerprint,
                "confidence": row.confidence,
            }
        )

    structure_value = str(analysis.source_moment.get("structure", "UNKNOWN"))
    try:
        structure = SourceMomentStructure(structure_value)
    except ValueError:
        structure = SourceMomentStructure.UNKNOWN

    return PlanningInputs(
        candidate_id=str(candidate.id),
        candidate_key=candidate.candidate_key,
        source_id=str(candidate.source_video_id),
        content_type=stage40.content_type,
        source_moment_structure=structure,
        source_moment=dict(analysis.source_moment or {}),
        transcript=stage40.transcript[: config.provider_max_input_characters],
        transcript_confidence=stage40.transcript_confidence,
        refined_start=stage40.effective_start,
        refined_end=stage40.effective_end,
        words=words,
        word_coverage_sufficient=coverage,
        context_segments=tuple(stage40.context_segments),
        dialect_profile=stage40.dialect_profile,
        dialect_confidence=stage40.dialect_confidence,
        code_switch=dict(stage40.code_switch),
        entities=tuple(stage40.entity_evidence),
        unresolved_spans=tuple(stage40.unresolved_spans),
        idea_summary=stage40.idea_summary,
        topic_summary=stage40.topic_summary,
        hooks=tuple(stage40.hooks),
        rights_risk=stage40.rights_risk.value,
        originality_risk=stage40.originality_risk.value,
        rights_status=stage40.rights_status,
        media_origin=stage40.media_origin,
        provenance_snapshot=dict(stage40.provenance_snapshot),
        stage3_risk={
            "clip_score": stage40.clip_score,
            "short_form_score": stage40.short_form_score,
            "moment_density_score": stage40.moment_density_score,
            "ending_quality_score": stage40.ending_quality_score,
            "loopability_score": stage40.loopability_score,
        },
        stage40_assessments=dict(analysis.assessments or {}),
        stage40_platform_risk=dict(analysis.platform_risk or {}),
        stage40_analysis_id=str(analysis.id),
        stage40_input_fingerprint=analysis.input_fingerprint,
        stage40_output_fingerprint=analysis.output_fingerprint,
        stage40_policy_version=analysis.policy_version,
        stage40_strategies=tuple(strategies),
        planning_context=PlanningContextResolver.default(),
        refinement_priority=refinement.priority.value,
        refinement_quality_level=refinement.quality_level,
        refinement_status=refinement.status.value,
        refinement_output_fingerprint=refinement.output_fingerprint or "",
    )


class PlanningContextResolver:
    """Future integration seam: project channel config into planning semantics.

    The repository has no channel/account/target-market persistence model and
    Stage 4.1 must not add one. When a future channel mapping exists, only its
    planning-semantic fields cross this seam; TTS provider/model/voice and
    rendering settings are intentionally dropped.
    """

    @staticmethod
    def default() -> PlanningContext:
        return PlanningContext()

    @staticmethod
    def from_mapping(mapping: Mapping[str, object] | None) -> PlanningContext:
        if not mapping:
            return PlanningContext()
        return PlanningContext(
            target_market=str(mapping.get("target_market", PlanningContext().target_market)),
            output_language_policy=str(
                mapping.get("output_language_policy", PlanningContext().output_language_policy)
            ),
            register_intent=str(mapping.get("register_intent", PlanningContext().register_intent)),
            narration_allowed=bool(
                mapping.get("narration_allowed", PlanningContext().narration_allowed)
            ),
        )
