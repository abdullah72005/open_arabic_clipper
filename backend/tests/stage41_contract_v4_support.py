"""Authentic-evidence helpers for the Stage 4.1 producer-contract regressions.

Fixtures under ``fixtures/stage41_contract_v4`` are deliberate, bounded
extractions from the saved read-only ``clipfactory_stage6_acceptance`` snapshot
(``/tmp/oac-stage41-audit-e7e067f.json``, SHA-256
``259d3128301b130da8573676211a84e1c7fc92de6039d28577210ec4998d92da``). They
preserve authentic candidate/refinement/analysis/strategy identity, ordered
source segments, bounds, words and Arabic text. They never contain a recovered
provider reply; authored provider-format responses in tests are clearly labelled
offline fixtures.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.core.enums import (
    CandidateDisposition,
    ContentType,
    OriginalityRisk,
    RefinementPriority,
    RefinementStatus,
    RightsRisk,
    StrategyDisposition,
    TransformationEligibilityOutcome,
    TransformationExecutionStatus,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.models import (
    CandidateRefinement,
    ClipCandidate,
    SourceVideo,
    Transcript,
    TransformationEligibilityAnalysis,
    TransformationStrategyCandidate,
)
from app.transformation.planning.types import PlanProviderResult

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "stage41_contract_v4"

CANDIDATES = (
    "1827c524-e893-4769-8b22-31411c10b6e8",
    "ac0670c4-dfd4-4609-b9b8-b2736b96b095",
    "64e7807c-3b58-4ea7-9b80-d2ec3e543572",
    "a725e62d-bec3-445d-93b4-76815febfde4",
)
# The representative case uses an authentic CONTEXT_HOOK strategy whose source
# excerpt spans the hook segment and the following demand phrase. That lets the
# authored explanation state a grounded relationship between two authentic
# phrases instead of merely quoting and attributing one phrase.
REPRESENTATIVE = "1827c524-e893-4769-8b22-31411c10b6e8"
REPRESENTATIVE_STRATEGY = TransformationStrategyType.CONTEXT_HOOK


def load_case(candidate_id: str) -> dict[str, Any]:
    path = FIXTURE_DIR / f"{candidate_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def seed_authentic_case(
    session: Any,
    candidate_id: str = REPRESENTATIVE,
    *,
    strategy_type: TransformationStrategyType | None = None,
    settings: Any | None = None,
) -> tuple[Any, Any, Any, Any, list[Any]]:
    """Insert one authentic saved candidate/refinement/analysis/strategy set.

    Only the selected strategy identity (when overridden) is a test concern; all
    source evidence, bounds, words and identities come from the saved rows.
    """

    from stage41_support import FakeStage41Settings

    from app.transformation.executor import build_transformation_executor

    case = load_case(candidate_id)
    c = case["candidate"]
    r = case["refinement"]
    a = case["analysis"]

    settings = settings or FakeStage41Settings()
    source = SourceVideo(
        id=uuid.UUID(c["source_video_id"]),
        source_uri=f"/tmp/stage41-authentic-{c['source_video_id']}.mp4",
        content_hash=f"authentic-{c['source_video_id']}",
    )
    session.add(source)
    session.flush()
    session.add(
        Transcript(
            source_video_id=source.id,
            whisper_model="authentic-saved",
            transcription_options={},
            input_fingerprint="t" * 64,
            raw_text=r.get("automatic_transcript") or "",
            normalized_text=r.get("final_transcript") or "",
            corrected_text=r.get("final_transcript") or "",
            final_text=r.get("final_transcript") or "",
            segments=[
                {"start": item.get("start"), "end": item.get("end"), "text": item.get("text")}
                for item in case["source_segments"]
            ],
            duration=float(case["source_segments"][-1]["end"]),
            language="ar",
        )
    )
    candidate = ClipCandidate(
        id=uuid.UUID(c["id"]),
        source_video_id=source.id,
        candidate_key=c["candidate_key"],
        is_current=bool(c.get("is_current", True)),
        disposition=CandidateDisposition(c["disposition"]),
        start_time=float(c["start_time"]),
        end_time=float(c["end_time"]),
        start_segment_index=c.get("start_segment_index"),
        end_segment_index=c.get("end_segment_index"),
        segment_indexes=list(c.get("segment_indexes") or []),
        primary_content_type=(
            ContentType(c["primary_content_type"]) if c.get("primary_content_type") else None
        ),
        secondary_content_types=list(c.get("secondary_content_types") or []),
        idea_summary=c.get("idea_summary") or "",
        topic_summary=c.get("topic_summary") or "",
        hooks=list(c.get("hooks") or []),
        clip_score=float(c.get("clip_score") or 0.0),
        short_form_score=float(c.get("short_form_score") or 0.0),
        moment_density_score=float(c.get("moment_density_score") or 0.0),
        ending_quality_score=float(c.get("ending_quality_score") or 0.0),
        loopability_score=float(c.get("loopability_score") or 0.0),
        rights_risk=RightsRisk(c.get("rights_risk") or "LOW"),
        originality_risk=OriginalityRisk(c.get("originality_risk") or "NOT_INDICATED"),
        dialect_profile=c.get("dialect_profile"),
        dialect_confidence=float(c.get("dialect_confidence") or 0.0),
        code_switch_suspected=bool(c.get("code_switch_suspected")),
        analysis_fingerprint=c.get("analysis_fingerprint") or "",
        policy_version=c.get("policy_version") or "stage3-v1",
        provenance_snapshot=dict(c.get("provenance_snapshot") or {}),
    )
    session.add(candidate)
    session.flush()
    refinement = CandidateRefinement(
        id=uuid.UUID(r["id"]),
        source_video_id=source.id,
        clip_candidate_id=candidate.id,
        priority=RefinementPriority(r["priority"]),
        status=RefinementStatus(r["status"]),
        coarse_start=float(r["coarse_start"]),
        coarse_end=float(r["coarse_end"]),
        context_start=float(r["context_start"]),
        context_end=float(r["context_end"]),
        refined_start=float(r["refined_start"]),
        refined_end=float(r["refined_end"]),
        automatic_transcript=r.get("automatic_transcript") or "",
        manual_transcript=r.get("manual_transcript") or "",
        final_transcript=r.get("final_transcript") or "",
        word_timestamps=[dict(item) for item in (r.get("word_timestamps") or [])],
        confidence=float(r.get("confidence") or 0.0),
        quality_level=r.get("quality_level") or "FINAL_CLIP",
        dialect_profile=r.get("dialect_profile"),
        dialect_confidence=float(r.get("dialect_confidence") or 0.0),
        code_switch_evidence=dict(r.get("code_switch_evidence") or {}),
        transcript_evidence=(
            list(r.get("transcript_evidence") or [])
            if isinstance(r.get("transcript_evidence"), list)
            else dict(r.get("transcript_evidence") or {})
        ),
        entity_evidence=list(r.get("entity_evidence") or []),
        unresolved_spans=list(r.get("unresolved_spans") or []),
        input_fingerprint=r.get("input_fingerprint") or "",
        output_fingerprint=r.get("output_fingerprint") or "",
        policy_version=r.get("policy_version") or "",
        schema_version=r.get("schema_version") or "",
        validation_version=r.get("validation_version") or "",
        cache_eligible=bool(r.get("cache_eligible")),
    )
    session.add(refinement)
    session.flush()
    analysis = TransformationEligibilityAnalysis(
        id=uuid.UUID(a["id"]),
        source_video_id=source.id,
        clip_candidate_id=candidate.id,
        refinement_id=refinement.id,
        refinement_priority=refinement.priority.value,
        refinement_quality_level=refinement.quality_level,
        execution_status=TransformationExecutionStatus(a["execution_status"]),
        eligibility_outcome=TransformationEligibilityOutcome(a["eligibility_outcome"]),
        eligibility_reasons=list(a.get("eligibility_reasons") or []),
        assessments=dict(a.get("assessments") or {}),
        source_moment=dict(a.get("source_moment") or {}),
        platform_risk=dict(a.get("platform_risk") or {}),
        transformation_intensity=(
            TransformationIntensity(a["transformation_intensity"])
            if a.get("transformation_intensity")
            else TransformationIntensity.MODERATE
        ),
        policy_version=a.get("policy_version") or "stage4.0-v1",
        output_fingerprint=a.get("output_fingerprint") or "",
        cache_eligible=bool(a.get("cache_eligible")),
    )
    session.add(analysis)
    session.flush()

    strategies: list[TransformationStrategyCandidate] = []
    for index, row in enumerate(case["strategies"]):
        if row.get("disposition") != "RECOMMENDED" or not row.get("is_current"):
            continue
        row_type = TransformationStrategyType(row["strategy_type"])
        if strategy_type is not None and row_type is not strategy_type:
            continue
        strategies.append(
            TransformationStrategyCandidate(
                id=uuid.UUID(row["id"]),
                analysis_id=analysis.id,
                strategy_key=row["strategy_key"],
                strategy_type=row_type,
                is_current=True,
                disposition=StrategyDisposition.RECOMMENDED,
                rank=int(row.get("rank") or index + 1),
                intensity=TransformationIntensity(row.get("intensity") or "MODERATE"),
                direction_summary=row.get("direction_summary") or "",
                added_value_focus=row.get("added_value_focus") or "",
                substantive_value_kind=row.get("substantive_value_kind"),
                source_moment_role=row.get("source_moment_role"),
                preservation_requirements=list(row.get("preservation_requirements") or []),
                external_verification_requirement=row.get("external_verification_requirement")
                or "NOT_REQUIRED",
                verification_requirements=list(row.get("verification_requirements") or []),
                confidence=float(row.get("confidence") or 0.0),
                strategy_fingerprint=row.get("strategy_fingerprint") or "",
                policy_version=row.get("policy_version") or "stage4.0-v1",
            )
        )
    assert strategies, "authentic fixture has no recommended strategy"
    session.add_all(strategies)
    session.flush()
    analysis.input_fingerprint = build_transformation_executor(session, settings).input_fingerprint(
        candidate
    )
    session.commit()
    return source, candidate, refinement, analysis, strategies


class AuthoredProvider:
    """Offline provider that feeds an authored response through the real parser."""

    provider_name = "offline-authored"
    hosted_provider = True

    def __init__(self, content: Mapping[str, object], *, model: str = "offline-authored") -> None:
        self._content = dict(content)
        self.model = model
        self.calls = 0
        self.tiers: list[str] = []

    def plan(self, requests: Sequence[Any], tier: str = "ROUTINE") -> dict[str, PlanProviderResult]:
        from app.transformation.planning.providers import parse_plan_results

        self.calls += 1
        self.tiers.append(tier)
        return parse_plan_results(self._content, requests)

    def release(self) -> None:
        return None

    def raw_call_count(self) -> int:
        return self.calls

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": self.provider_name, "model": self.model}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


def selectable_planning_words(refinement: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return the authentic selectable words in the offered planning order."""

    refined_start = float(refinement["refined_start"])
    refined_end = float(refinement["refined_end"])
    selectable: list[Mapping[str, object]] = []
    for item in refinement.get("word_timestamps") or []:
        if not isinstance(item, Mapping):
            continue
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            continue
        if not isinstance(end, (int, float)) or isinstance(end, bool):
            continue
        if float(start) >= refined_start - 0.75 and float(end) <= refined_end + 0.75:
            # Match WordEvidence.as_dict rounding so the helper sees exactly the
            # planning bounds the production request exposes.
            selectable.append(
                {**item, "start": round(float(start), 4), "end": round(float(end), 4)}
            )
    return selectable


def planning_word_range_for_span(
    refinement: Mapping[str, object], span: tuple[float, float]
) -> tuple[int, int] | None:
    """Map authentic span bounds to offered planning-word indexes read-only."""

    indexes = [
        index
        for index, item in enumerate(selectable_planning_words(refinement))
        if float(item["end"]) > span[0] and float(item["start"]) < span[1]
    ]
    if not indexes:
        return None
    return indexes[0], indexes[-1]


def planning_word_range_for_segment(
    refinement: Mapping[str, object], segment: tuple[float, float]
) -> tuple[int, int] | None:
    """Map authentic segment bounds to offered planning-word indexes read-only."""

    return planning_word_range_for_span(refinement, segment)


def planning_word_texts(
    refinement: Mapping[str, object], start_index: int, end_index: int
) -> tuple[str, ...]:
    words = selectable_planning_words(refinement)
    return tuple(str(item.get("text", "")).strip() for item in words[start_index : end_index + 1])


def representative_response(
    *,
    strategy_id: str,
    strategy_key: str,
    word_start: int,
    word_end: int,
    excerpt_quote: str,
    demand_quote: str,
) -> dict[str, object]:
    """Authored offline provider-format response grounded in authentic evidence.

    The authored contribution is an explanatory claim about the relation between
    two authentic phrases in the selected excerpt: the gold and diamonds are the
    family's demands, not an unrelated list. The quoted phrases are the exact
    saved planning-word text, and the only non-source tokens are the permitted
    reference/framing words "source", "context", and "explains". Therefore the
    claim adds no new factual predicate and inherits support from the cited
    excerpt rather than laundering a quotation as substantive value.
    """

    semantic_intent = f'The source context explains "{excerpt_quote}" as "{demand_quote}".'
    return {
        "plans": [
            {
                "strategy_id": strategy_id,
                "strategy_key": strategy_key,
                "confidence": 0.72,
                "no_valid_plan": False,
                "no_valid_reason": "",
                "planner_notes": "offline authored substantive-value fixture",
                "preservation_constraints": ["Keep the source speakers and dialect unchanged."],
                "narration": {
                    "need": "NONE",
                    "purposes": [],
                    "language": None,
                    "register_intent": None,
                    "estimated_duration": 0.0,
                    "overlaps_source_audio": False,
                    "verification_dependency_ids": [],
                },
                "blocks": [
                    {
                        "block_type": "SOURCE_EXCERPT",
                        "purpose": "Authentic hook and demand excerpt",
                        "estimated_duration": 3.36,
                        "source_role": "HERO",
                        "word_start_index": word_start,
                        "word_end_index": word_end,
                        "use_full_window": False,
                        "interrupts_source": False,
                        "preservation_constraints": [],
                        "dependency_ids": [],
                        "semantic_intent": "",
                        "why_unavailable": "",
                        "grounding_refs": [],
                        "dependent_block_ids": [],
                    },
                    {
                        "block_type": "ORIGINAL_VALUE",
                        "purpose": "Grounded explanation of the demand list",
                        "estimated_duration": 3.0,
                        "substantive_value_kind": "EXPLANATION",
                        "semantic_intent": semantic_intent,
                        "why_unavailable": (
                            "The selected excerpt states the diamonds, gold, and the family's "
                            "demands as separate phrases; it does not itself state the "
                            "explanatory relation that the diamonds and gold are those demands."
                        ),
                        "grounding_refs": [f"words:{word_start}-{word_end}"],
                        "delivery_intent": "ON_SCREEN_TEXT",
                        "interrupts_source": False,
                        "preservation_constraints": [],
                        "dependency_ids": [],
                        "dependent_block_ids": [],
                    },
                ],
            }
        ]
    }


def replace_strategy_id(
    content: Mapping[str, object], strategy_id: str, key: str
) -> dict[str, Any]:
    payload = json.loads(json.dumps(content))
    payload["plans"][0]["strategy_id"] = strategy_id
    payload["plans"][0]["strategy_key"] = key
    return payload


__all__ = [
    "CANDIDATES",
    "REPRESENTATIVE",
    "REPRESENTATIVE_STRATEGY",
    "AuthoredProvider",
    "load_case",
    "planning_word_range_for_segment",
    "planning_word_range_for_span",
    "planning_word_texts",
    "representative_response",
    "replace_strategy_id",
    "seed_authentic_case",
    "selectable_planning_words",
]
