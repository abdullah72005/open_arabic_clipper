"""Bounded Stage 4.1 input assembly from a current Stage 4.0 handoff.

Uses the frozen Stage 4.0 handoff as the readiness/freshness gate, reloads the
selected analysis/refinement by persisted identity, and recovers the exact
bounded nearby context through Stage 4.0's own bounded input assembly. Never
loads a whole video or a whole multi-hour transcript, and never recomputes
Stage 4.0 eligibility or strategy ranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sqlalchemy import select

from app.core.enums import SourceMomentStructure, StrategyDisposition
from app.core.settings import Settings
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    Transcript,
    TransformationEligibilityAnalysis,
    TransformationStrategyCandidate,
)
from app.transformation.handoff import build_stage4_1_handoff
from app.transformation.hook_payoff import (
    MANDATORY_COVERAGE_TOLERANCE_SECONDS,
    MandatoryEvidence,
    resolve_mandatory_evidence,
)
from app.transformation.inputs import build_transformation_inputs
from app.transformation.planning.policy import Stage41Config
from app.transformation.planning.request import (
    OperatorPlanningRequest,
    normalize_operator_request,
)
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


# Matches Stage 4.1 span validation: a selected span may sit this far outside the
# refined window before it is rejected as ``INVALID_SOURCE_SPAN``.
_SPAN_TOLERANCE_SECONDS = 0.75


def _selectable_window_words(
    raw: Sequence[object], refined_start: float, refined_end: float
) -> tuple[Mapping[str, object], ...]:
    """Keep only words that a provider may legally select as a source span.

    The refinement carries bounded nearby context outside the refined window.
    A provider that selects those context words produces an out-of-window span
    that validation rejects, so context words are never offered as selectable
    indexes. Index ``0`` is therefore always the first in-window word.
    """

    window_start = refined_start - _SPAN_TOLERANCE_SECONDS
    window_end = refined_end + _SPAN_TOLERANCE_SECONDS
    selected: list[Mapping[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if float(start) >= window_start and float(end) <= window_end:
            selected.append(item)
    return tuple(selected)


_BOUND_EPSILON = 1e-6


def _operator_window_words(
    raw: Sequence[object], source_start: float, source_end: float
) -> tuple[Mapping[str, object], ...]:
    """Offer only words wholly inside the explicit operator window."""

    selected: list[Mapping[str, object]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        start = item.get("start")
        end = item.get("end")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            continue
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            continue
        if (
            float(start) >= source_start - _BOUND_EPSILON
            and float(end) <= source_end + _BOUND_EPSILON
        ):
            selected.append(item)
    return tuple(selected)


def _source_evidence_bounds(
    raw_words: Sequence[Mapping[str, object]],
    source_segments: Sequence[Mapping[str, object]],
) -> tuple[float, float] | None:
    starts: list[float] = []
    ends: list[float] = []
    for item in (*raw_words, *source_segments):
        start = item.get("start")
        end = item.get("end")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            continue
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            continue
        starts.append(float(start))
        ends.append(float(end))
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def _mandatory_segments(
    mandatory: MandatoryEvidence,
) -> tuple[tuple[str, tuple[int, float, float] | None, str | None], ...]:
    return (
        ("hook", mandatory.hook_segment, mandatory.hook_error),
        ("payoff", mandatory.payoff_segment, mandatory.payoff_error),
    )


def _validate_operator_window(
    request: OperatorPlanningRequest,
    raw_words: Sequence[Mapping[str, object]],
    source_segments: Sequence[Mapping[str, object]],
    mandatory: MandatoryEvidence,
) -> None:
    """Reject a request whose window cannot preserve the mandatory evidence."""

    if not request.has_window:
        return
    assert request.source_start is not None and request.source_end is not None
    bounds = _source_evidence_bounds(raw_words, source_segments)
    if bounds is None:
        raise PlanningInputError("OPERATOR_WINDOW_HAS_NO_SOURCE_EVIDENCE")
    if (
        request.source_start < bounds[0] - _BOUND_EPSILON
        or request.source_end > bounds[1] + _BOUND_EPSILON
    ):
        raise PlanningInputError("OPERATOR_WINDOW_OUT_OF_SOURCE_BOUNDS")
    for label, segment, error in _mandatory_segments(mandatory):
        if error == "MALFORMED":
            raise PlanningInputError(f"{label.upper()}_EVIDENCE_MALFORMED")
        if error == "UNRESOLVED":
            raise PlanningInputError(f"{label.upper()}_EVIDENCE_UNRESOLVED")
        if segment is None:
            continue
        if (
            segment[1] < request.source_start - _BOUND_EPSILON
            or segment[2] > request.source_end + _BOUND_EPSILON
        ):
            raise PlanningInputError(f"{label.upper()}_OUTSIDE_APPROVED_WINDOW")


def _window_can_cover(words: Sequence[WordEvidence], start: float, end: float) -> bool:
    tolerance = MANDATORY_COVERAGE_TOLERANCE_SECONDS
    start_indexes = [
        index
        for index, word in enumerate(words)
        if word.start <= start + tolerance and word.end >= start - tolerance
    ]
    end_indexes = [
        index
        for index, word in enumerate(words)
        if word.start <= end + tolerance and word.end >= end - tolerance
    ]
    if not start_indexes or not end_indexes:
        return False
    return min(start_indexes) <= max(end_indexes)


def _require_operator_window_coverage(
    request: OperatorPlanningRequest,
    words: Sequence[WordEvidence],
    mandatory: MandatoryEvidence,
) -> None:
    if not request.has_window:
        return
    if not words:
        raise PlanningInputError("OPERATOR_WINDOW_HAS_NO_WORD_EVIDENCE")
    for label, segment, _error in _mandatory_segments(mandatory):
        if segment is None:
            continue
        if not _window_can_cover(words, segment[1], segment[2]):
            raise PlanningInputError(f"{label.upper()}_NOT_COVERABLE_WITHIN_APPROVED_WINDOW")


def build_planning_inputs(
    session: object,
    candidate: ClipCandidate,
    analysis: TransformationEligibilityAnalysis,
    refinement: CandidateRefinement,
    settings: Settings,
    config: Stage41Config,
    operator_request: OperatorPlanningRequest | Mapping[str, object] | None = None,
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

    transcript_row = session.scalar(
        select(Transcript).where(Transcript.source_video_id == candidate.source_video_id)
    )
    source_segments: tuple[Mapping[str, object], ...] = ()
    if transcript_row is not None and isinstance(transcript_row.segments, list):
        source_segments = tuple(
            item for item in transcript_row.segments if isinstance(item, Mapping)
        )
    mandatory = resolve_mandatory_evidence(
        str(candidate.source_video_id),
        analysis.source_moment or {},
        source_segments,
    )

    request = normalize_operator_request(operator_request)
    # Consume the same effective word view as Stage 4.0/captions instead of
    # rereading raw refinement words that may still carry a superseded spelling.
    raw_words = tuple(item for item in (stage40.word_timestamps or ()) if isinstance(item, Mapping))
    if request is not None and request.has_window:
        assert request.source_start is not None and request.source_end is not None
        _validate_operator_window(request, raw_words, source_segments, mandatory)
        raw_words = _operator_window_words(raw_words, request.source_start, request.source_end)
        selectable_start = request.source_start
        selectable_end = request.source_end
    else:
        raw_words = _selectable_window_words(
            raw_words, stage40.effective_start, stage40.effective_end
        )
        selectable_start = stage40.effective_start
        selectable_end = stage40.effective_end
    words, _has_words = _bounded_words(raw_words, config)
    coverage = _coverage_sufficient(words, selectable_start, selectable_end)
    if request is not None and request.has_window:
        _require_operator_window_coverage(request, words, mandatory)
        if not coverage:
            raise PlanningInputError("OPERATOR_WINDOW_WORD_COVERAGE_INSUFFICIENT")

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
        mandatory_evidence=mandatory.as_dict(),
        operator_request_version=request.version if request is not None else "",
        approved_source_start=(request.source_start if request is not None else None),
        approved_source_end=request.source_end if request is not None else None,
        editorial_intent=request.editorial_intent if request is not None else None,
        selectable_start=selectable_start,
        selectable_end=selectable_end,
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
