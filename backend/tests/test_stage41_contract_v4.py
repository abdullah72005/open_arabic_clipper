"""Stage 4.1 producer-contract V4 regressions on authentic saved evidence.

All provider-format responses here are offline authored fixtures. The source
evidence, bounds, words, identities and Arabic text come from the saved
read-only acceptance snapshot described in ``stage41_contract_v4_support``.
No test starts a worker, model, renderer, or network client.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage41_contract_v4_support import (
    CANDIDATES,
    AuthoredProvider,
    load_case,
    planning_word_range_for_segment,
    representative_response,
    seed_authentic_case,
)
from stage41_support import (
    FakeStage41Settings,
    install_stage41_settings,
)

from app.core.enums import (
    ContentType,
    JobKind,
    NarrationNeed,
    PlanBlockType,
    PlanStatus,
    SemanticProviderMode,
    SourceExcerptRole,
    SourceMomentStructure,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.db.base import Base
from app.materialization.grounding import parse_grounding_refs, resolve_grounding
from app.materialization.inputs import (
    FinalClipEvidence,
    MaterializationInputs,
    ProfileContext,
    SourceMediaEvidence,
    VisualPlanEvidence,
)
from app.materialization.inputs import (
    WordEvidence as FinalWordEvidence,
)
from app.models import ProcessingJob
from app.transformation.governance.inputs import build_governance_inputs
from app.transformation.governance.policy import DEFAULT_CONFIG as STAGE42_CONFIG
from app.transformation.governance.validation import (
    evaluate_plan,
    finalize,
    revalidate_integrity,
)
from app.transformation.hook_payoff import resolve_mandatory_evidence
from app.transformation.planning.executor import TransformationPlanningExecutor
from app.transformation.planning.inputs import build_planning_inputs
from app.transformation.planning.policy import (
    DEFAULT_CONFIG as STAGE41_CONFIG,
)
from app.transformation.planning.policy import (
    SCHEMA_VERSION,
    strategy_value_kinds,
)
from app.transformation.planning.providers import (
    PLANNING_OUTPUT_SCHEMA,
    PLANNING_SYSTEM_INSTRUCTION,
    PlanningOutputSchema,
    build_planning_request,
    planning_prompt_hash,
)
from app.transformation.planning.queue import get_or_create_plan_set, list_plans
from app.transformation.planning.types import (
    NarrationRequirement,
    PlanningContext,
    PlanningInputs,
    PlanProviderBlock,
    PlanProviderPlan,
)
from app.transformation.planning.validation import validate_provider_plan
from app.transformation.selection.service import _mandatory_failures


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _authentic_inputs(
    session: Session, settings: FakeStage41Settings
) -> tuple[PlanningInputs, tuple[object, ...], dict[str, object]]:
    source, candidate, refinement, analysis, strategies = seed_authentic_case(
        session,
        strategy_type=TransformationStrategyType.ANALYSIS,
        settings=settings,
    )
    inputs = build_planning_inputs(
        session, candidate, analysis, refinement, settings, settings.stage41_config()
    )
    return (
        inputs,
        (source, candidate, refinement, analysis, strategies),
        load_case(str(candidate.id)),
    )


def _strategy(inputs: PlanningInputs) -> dict[str, object]:
    return next(
        item
        for item in inputs.stage40_strategies
        if item.get("strategy_type") == TransformationStrategyType.ANALYSIS.value
    )


def test_fixtures_are_deliberate_authentic_extractions() -> None:
    expected_hash = "259d3128301b130da8573676211a84e1c7fc92de6039d28577210ec4998d92da"
    for candidate_id in CANDIDATES:
        case = load_case(candidate_id)
        provenance = case["provenance"]
        assert provenance["source_database"] == "clipfactory_stage6_acceptance"
        assert provenance["snapshot_sha256"] == expected_hash
        assert provenance["candidate_id"] == candidate_id
        assert provenance["source_segment_count"] == len(case["source_segments"])
        assert case["candidate"]["id"] == candidate_id
        assert case["refinement"]["clip_candidate_id"] == candidate_id
        assert case["analysis"]["clip_candidate_id"] == candidate_id
        assert all(
            item["strategy_key"].startswith(case["analysis"]["id"]) for item in case["strategies"]
        )


def test_request_exposes_authoritative_mandatory_evidence_and_limits(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    inputs, _, case = _authentic_inputs(session, settings)
    strategy = _strategy(inputs)
    request = build_planning_request(strategy, inputs, settings.stage41_config())
    payload = request.to_payload()

    hook = case["analysis"]["source_moment"]["hook_index"]
    segment = case["source_segments"][hook]
    assert payload["source_selection"]["words_offered"] == 117
    assert payload["source_selection"]["word_index_scope"].startswith("Inclusive")
    assert payload["mandatory_evidence"]["hook"]["state"] == "RESOLVED"
    assert payload["mandatory_evidence"]["hook"]["segment_index"] == hook
    assert payload["mandatory_evidence"]["hook"]["source_start"] == segment["start"]
    assert (
        payload["mandatory_evidence"]["hook"]["planning_word_start_index"],
        payload["mandatory_evidence"]["hook"]["planning_word_end_index"],
    ) == planning_word_range_for_segment(case["refinement"], (segment["start"], segment["end"]))
    assert payload["mandatory_evidence"]["hook"]["covered_by_offered_words"] is True
    assert payload["mandatory_evidence"]["payoff"]["state"] == "ABSENT"
    assert set(payload["strategy_constraints"]["allowed_value_kinds"]) == {
        item.value for item in strategy_value_kinds(TransformationStrategyType.ANALYSIS)
    }
    assert payload["source_selection"]["min_source_excerpt_seconds"] > 0
    assert payload["source_selection"]["strict_hero_cap_seconds"] > 0
    assert payload["nonselectable_context"]["suggested_hook_copy"]
    assert "context_text" in payload["nonselectable_context"]


def test_actual_provider_schema_and_instruction_are_the_authoritative_contract(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    inputs, _, _ = _authentic_inputs(session, settings)
    request = build_planning_request(_strategy(inputs), inputs, settings.stage41_config())

    captured: dict[str, object] = {}

    class _Models:
        def generate_content(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return type(
                "_Response",
                (),
                {
                    "parsed": None,
                    "text": json.dumps({"plans": []}),
                    "usage_metadata": None,
                    "prompt_feedback": None,
                    "candidates": [type("_C", (), {"finish_reason": None})()],
                },
            )()

    class _Client:
        def __init__(self) -> None:
            self.models = _Models()

        def close(self) -> None:
            return None

    from app.transformation.planning.gemini import GeminiPlanningProvider

    provider = GeminiPlanningProvider(api_key="offline", client_factory=lambda: _Client())
    provider.plan([request], "ROUTINE")
    provider.release()

    config = captured["config"]
    assert config.system_instruction == PLANNING_SYSTEM_INSTRUCTION
    assert config.response_schema is PlanningOutputSchema
    assert PLANNING_OUTPUT_SCHEMA == PlanningOutputSchema.model_json_schema()

    block_required = set(PLANNING_OUTPUT_SCHEMA["$defs"]["PlanningBlockSchema"]["required"])
    plan_required = set(PLANNING_OUTPUT_SCHEMA["$defs"]["PlanningPlanSchema"]["required"])
    assert {
        "block_type",
        "purpose",
        "estimated_duration",
        "grounding_refs",
        "dependent_block_ids",
    } <= block_required
    assert {
        "strategy_id",
        "strategy_key",
        "confidence",
        "no_valid_plan",
        "no_valid_reason",
        "blocks",
        "narration",
    } <= plan_required
    descriptions = json.dumps(PLANNING_OUTPUT_SCHEMA, ensure_ascii=False)
    for phrase in ("block:N", "words:N-M", "source:A-B", "hook", "payoff", "grounding"):
        assert phrase in descriptions
    assert "Do not use object references or final_clip" in descriptions
    for phrase in (
        "mandatory_evidence",
        "hook",
        "grounding",
        "words:N-M",
        "block:N",
        "source:A-B",
        "allowed_value_kinds",
    ):
        assert phrase in PLANNING_SYSTEM_INSTRUCTION

    expected_hash = (
        __import__("hashlib")
        .sha256(
            (
                PLANNING_SYSTEM_INSTRUCTION
                + json.dumps(PlanningOutputSchema.model_json_schema(), sort_keys=True)
            ).encode("utf-8")
        )
        .hexdigest()
    )
    assert planning_prompt_hash() == expected_hash
    assert provider.runtime_identity()["schema_version"] == SCHEMA_VERSION


def _run_authentic_planning(
    session: Session,
    settings: FakeStage41Settings,
    provider: object,
    *,
    job: ProcessingJob | None = None,
) -> tuple[object, object, object, object]:
    from app.transformation.planning.queue import get_or_create_plan_set

    source, candidate, refinement, analysis, strategies = seed_authentic_case(
        session,
        strategy_type=TransformationStrategyType.ANALYSIS,
        settings=settings,
    )
    plan_set = get_or_create_plan_set(session, candidate, analysis)
    executor = TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=provider,  # type: ignore[arg-type]
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=SemanticProviderMode.ADAPTIVE,
        config=STAGE41_CONFIG,
        lease_factory=settings.heavy_model_lease_factory(),  # type: ignore[arg-type]
    )
    if job is not None:
        executor.set_active_job(job.id)
    executor.execute(plan_set.id)
    session.refresh(plan_set)
    return plan_set, candidate, refinement, analysis


def test_representative_authentic_plan_passes_stage41_and_downstream(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    case = load_case("a725e62d-bec3-445d-93b4-76815febfde4")
    segment = case["source_segments"][case["analysis"]["source_moment"]["hook_index"]]
    word_range = planning_word_range_for_segment(
        case["refinement"], (segment["start"], segment["end"])
    )
    assert word_range is not None
    word_start, word_end = word_range[0], min(word_range[1] + 2, 116)
    assert word_start > 0  # a nonzero planning index space, not an identity shortcut
    response = representative_response(
        strategy_id="",
        strategy_key="",
        word_start=word_start,
        word_end=word_end,
    )
    PlanningOutputSchema.model_validate(response)
    # The authored response must match the authentic strategy identity.
    authentic_strategy = next(
        item
        for item in case["strategies"]
        if item["strategy_type"] == "ANALYSIS" and item["is_current"]
    )
    response["plans"][0]["strategy_id"] = authentic_strategy["id"]
    response["plans"][0]["strategy_key"] = authentic_strategy["strategy_key"]

    plan_set, candidate, refinement, analysis = _run_authentic_planning(
        session, settings, AuthoredProvider(response)
    )
    assert plan_set.execution_status.value == "COMPLETE"
    plans = list_plans(session, plan_set.id)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.status is PlanStatus.PLAN_GENERATED
    hero = plan.blocks[0]
    assert hero["block_type"] == "SOURCE_EXCERPT"
    assert hero["word_start_index"] == word_start
    assert hero["word_end_index"] == word_end
    assert len(hero["planning_words"]) == word_end - word_start + 1
    assert hero["planning_words"][0]["start"] >= 0
    # Planning indexes are not FINAL_CLIP indexes: the first planning word maps
    # to an authentic refinement word at a different stored index.
    final_index = next(
        index
        for index, item in enumerate(refinement.word_timestamps)
        if float(item["start"]) == float(hero["planning_words"][0]["start"])
        and float(item["end"]) == float(hero["planning_words"][0]["end"])
    )
    assert final_index != hero["word_start_index"]

    governance = build_governance_inputs(
        session,
        candidate,
        plan_set,
        settings,
        STAGE42_CONFIG,
        settings.stage41_config(),
    )
    evidence = governance.plans[0]
    ok, reasons = revalidate_integrity(
        evidence, STAGE42_CONFIG, mandatory_evidence=governance.mandatory_evidence
    )
    assert ok, reasons
    evaluation = evaluate_plan(
        evidence,
        governance,
        STAGE42_CONFIG,
        provider_mode_deterministic=True,
    )
    assert not evaluation.hard_failures, evaluation.hard_failures
    assert not evaluation.blocking, evaluation.blocking
    status, _severity = finalize(evaluation)
    assert status.value in {"APPROVED_FOR_SELECTION", "APPROVED_WITH_CAUTION"}

    assert _mandatory_failures(session, candidate, [plan]) == {}

    from app.render.compatibility import evaluate_compatibility
    from app.render.types import ClipWord, FinalClipEvidence

    final_words = tuple(
        ClipWord(
            index=index,
            text=item["text"].strip(),
            start=float(item["start"]),
            end=float(item["end"]),
        )
        for index, item in enumerate(refinement.word_timestamps)
        if isinstance(item.get("text"), str) and item.get("text", "").strip()
    )
    final = FinalClipEvidence(
        refinement_id=str(refinement.id),
        priority=refinement.priority.value,
        quality_level=refinement.quality_level,
        status=refinement.status.value,
        output_fingerprint=refinement.output_fingerprint or "",
        final_transcript=refinement.final_transcript or "",
        words=final_words,
        refined_start=float(refinement.refined_start),
        refined_end=float(refinement.refined_end),
    )
    result = evaluate_compatibility(
        blocks=[dict(block) for block in plan.blocks],
        hero_block_index=plan.hero_block_index,
        hook_payoff_evidence=dict(plan.hook_payoff_evidence),
        final=final,
        source_segments=[
            {"start": item["start"], "end": item["end"], "text": item["text"]}
            for item in case["source_segments"]
        ],
        planning_refined_start=float(refinement.refined_start),
        planning_refined_end=float(refinement.refined_end),
        exact_identity_match=True,
        caption_source_fingerprint="caption",
        planning_refinement_id=str(refinement.id),
        planning_refinement_priority=refinement.priority.value,
        planning_refinement_quality_level=refinement.quality_level,
        planning_output_fingerprint=plan.plan_output_fingerprint,
    )
    assert result.outcome == "EXACT_MATCH", result.reason_codes
    assert "HOOK_NOT_COVERED" not in result.reason_codes

    materialization = _materialization_inputs(plan, refinement, final_words)
    refs = parse_grounding_refs([f"words:{word_start + 1}-{word_end - 1}"])
    resolution = resolve_grounding(materialization, refs)
    assert resolution.unresolved == ()
    assert resolution.text == " ".join(
        item["text"].strip() for item in hero["planning_words"][1:-1]
    )


def _materialization_inputs(
    plan: object, refinement: object, final_words: tuple[object, ...]
) -> MaterializationInputs:
    blocks = tuple(dict(block) for block in plan.blocks)  # type: ignore[attr-defined]
    final_text = " ".join(word.text for word in final_words)
    bindings = tuple(
        {
            "block_index": block["index"],
            "final_clip_start": block.get("source_start"),
            "final_clip_end": block.get("source_end"),
            "final_clip_word_start": None,
            "final_clip_word_end": None,
            "final_clip_text": block.get("source_text") or "",
        }
        for block in blocks
        if block.get("block_type") == "SOURCE_EXCERPT"
    )
    return MaterializationInputs(
        candidate_id="candidate",
        candidate_key="candidate",
        source_id="source",
        priority="CANDIDATE",
        selection_id="selection",
        selection_output_fingerprint="selection-fp",
        selected_plan_fingerprint="plan-fp",
        plan_id="plan",
        plan_output_fingerprint=str(plan.plan_output_fingerprint),  # type: ignore[attr-defined]
        strategy_type=str(plan.strategy_type.value),  # type: ignore[attr-defined]
        intensity=str(plan.intensity.value),  # type: ignore[attr-defined]
        blocks=blocks,
        hero_block_index=int(plan.hero_block_index),  # type: ignore[attr-defined]
        narration={"need": "NONE", "requirements": {}},
        verification_dependencies=(),
        original_value_kinds=("EXPLANATION",),
        required_context=(),
        preservation_constraints=(),
        plan_target_context={},
        contract_id="contract",
        contract_status="MATERIALIZATION_REQUIRED",
        contract_input_fingerprint="contract-in",
        contract_output_fingerprint="contract-out",
        execution_slots=(),
        contract_source_bindings=bindings,
        final_clip=FinalClipEvidence(
            refinement_id=str(refinement.id),  # type: ignore[attr-defined]
            priority=refinement.priority.value,  # type: ignore[attr-defined]
            quality_level=refinement.quality_level,  # type: ignore[attr-defined]
            status=refinement.status.value,  # type: ignore[attr-defined]
            output_fingerprint=refinement.output_fingerprint or "",  # type: ignore[attr-defined]
            text=final_text,
            words=tuple(
                FinalWordEvidence(index=index, text=word.text, start=word.start, end=word.end)
                for index, word in enumerate(final_words)
            ),
            start=float(refinement.refined_start),  # type: ignore[attr-defined]
            end=float(refinement.refined_end),  # type: ignore[attr-defined]
        ),
        visual_plan=VisualPlanEvidence(
            visual_plan_id="vis",
            output_fingerprint="vis-out",
            plan_ready=True,
            ass_relative_path="captions.ass",
            ass_sha256="a" * 64,
            ass_policy_version="v1",
            caption_source_fingerprint="caption",
            framing_fingerprint="framing",
            source_media_fingerprint="media",
        ),
        source_media=SourceMediaEvidence(
            relative_path="source.webm",
            content_hash="hash",
            size_bytes=1,
            mtime_ns=1,
            duration=900.0,
            frame_rate=Fraction(25, 1),
            display_width=1920,
            display_height=1080,
            encoded_width=1920,
            encoded_height=1080,
            sample_rate=48000,
            channels=2,
        ),
        profile=ProfileContext(
            profile_id="profile",
            profile_key="offline-contract",
            target_market="UNSPECIFIED",
            output_language_policy="SOURCE_LANGUAGE",
            register_intent="SOURCE_COMPATIBLE",
            narration_allowed=False,
        ),
    )


def test_rejection_diagnostics_are_bounded_and_survive_a_later_run(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    case = load_case("a725e62d-bec3-445d-93b4-76815febfde4")
    segment = case["source_segments"][71]
    word_range = planning_word_range_for_segment(
        case["refinement"], (segment["start"], segment["end"])
    )
    assert word_range is not None
    strategy = next(
        item
        for item in case["strategies"]
        if item["strategy_type"] == "ANALYSIS" and item["is_current"]
    )
    invalid = representative_response(
        strategy_id=strategy["id"],
        strategy_key=strategy["strategy_key"],
        word_start=word_range[0],
        word_end=word_range[1] + 2,
    )
    invalid["plans"][0]["blocks"][1]["grounding_refs"] = ["793.32", "block:0"]

    source, candidate, refinement, analysis, _strategies = seed_authentic_case(
        session, strategy_type=TransformationStrategyType.ANALYSIS, settings=settings
    )
    plan_set = get_or_create_plan_set(session, candidate, analysis)
    job = ProcessingJob(
        source_video_id=source.id,
        transformation_plan_set_id=plan_set.id,
        kind=JobKind.TRANSFORMATION_PLANNING,
    )
    session.add(job)
    session.commit()

    executor = TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=AuthoredProvider(invalid),
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=SemanticProviderMode.ADAPTIVE,
        config=STAGE41_CONFIG,
        lease_factory=settings.heavy_model_lease_factory(),  # type: ignore[arg-type]
    )
    executor.set_active_job(job.id)
    executor.execute(plan_set.id)
    session.refresh(plan_set)

    entries = dict(plan_set.provider_evidence or {}).get("attempt_diagnostics")
    assert isinstance(entries, list) and entries
    entry = entries[-1]
    assert entry["job_id"] == str(job.id)
    assert entry["outcome"] == "INVALID"
    assert entry["phase"] == "CONTRACT_REJECTION"
    assert any("GROUNDING_EVIDENCE_UNRESOLVED" in code for code in entry["failure_codes"])
    assert entry["strategy_key"] == strategy["strategy_key"]
    assert entry["route"] == "STRONG"
    assert entry["provider_input_fingerprint"]
    assert entry["mandatory_evidence"]["hook"]["state"] == "RESOLVED"
    selectors = entry["selectors"]
    assert selectors[0]["block_type"] == "SOURCE_EXCERPT"
    assert selectors[0]["resolved"]["word_start_index"] == word_range[0]
    # Quote declarations are retained as digests, never as transcript text.
    assert all("أوه" not in json.dumps(item, ensure_ascii=False) for item in selectors)

    # A later normal run keeps the earlier per-job diagnostic evidence.
    valid = representative_response(
        strategy_id=strategy["id"],
        strategy_key=strategy["strategy_key"],
        word_start=word_range[0],
        word_end=word_range[1],
    )
    second_job = ProcessingJob(
        source_video_id=source.id,
        transformation_plan_set_id=plan_set.id,
        kind=JobKind.TRANSFORMATION_PLANNING,
    )
    session.add(second_job)
    session.commit()
    executor2 = TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=AuthoredProvider(valid),
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=SemanticProviderMode.ADAPTIVE,
        config=STAGE41_CONFIG,
        lease_factory=settings.heavy_model_lease_factory(),  # type: ignore[arg-type]
    )
    executor2.set_active_job(second_job.id)
    executor2.execute(plan_set.id, force=True)
    session.refresh(plan_set)
    entries = dict(plan_set.provider_evidence or {}).get("attempt_diagnostics")
    assert isinstance(entries, list)
    # The earlier rejected job remains inspectable; a later accepted job adds no
    # rejection diagnostic and never reuses the rejected attempt as a checkpoint.
    assert str(job.id) in {item["job_id"] for item in entries}
    assert str(second_job.id) not in {item["job_id"] for item in entries}
    assert any(item["outcome"] == "INVALID" for item in entries)
    assert list_plans(session, plan_set.id)


def _authentic_planning_inputs(session: Session) -> PlanningInputs:
    case = load_case("a725e62d-bec3-445d-93b4-76815febfde4")
    refinement = case["refinement"]
    mandatory = resolve_mandatory_evidence(
        case["candidate"]["source_video_id"],
        case["analysis"]["source_moment"],
        case["source_segments"],
    )
    raw_words = refinement["word_timestamps"]
    selectable = [
        item
        for item in raw_words
        if float(item["start"]) >= float(refinement["refined_start"]) - 0.75
        and float(item["end"]) <= float(refinement["refined_end"]) + 0.75
    ]
    from app.transformation.planning.types import WordEvidence

    words = tuple(
        WordEvidence(
            index=index,
            text=item["text"].strip(),
            start=float(item["start"]),
            end=float(item["end"]),
            probability=item.get("probability"),
        )
        for index, item in enumerate(selectable)
    )
    return PlanningInputs(
        candidate_id=case["candidate"]["id"],
        candidate_key=case["candidate"]["candidate_key"],
        source_id=case["candidate"]["source_video_id"],
        content_type=ContentType(case["candidate"]["primary_content_type"]),
        source_moment_structure=SourceMomentStructure(
            case["analysis"]["source_moment"]["structure"]
        ),
        source_moment=dict(case["analysis"]["source_moment"]),
        transcript=str(refinement["final_transcript"]),
        transcript_confidence=float(refinement["confidence"]),
        refined_start=float(refinement["refined_start"]),
        refined_end=float(refinement["refined_end"]),
        words=words,
        word_coverage_sufficient=True,
        context_segments=(),
        dialect_profile=refinement["dialect_profile"],
        dialect_confidence=float(refinement["dialect_confidence"]),
        code_switch={},
        entities=(),
        unresolved_spans=(),
        idea_summary=case["candidate"]["idea_summary"],
        topic_summary=case["candidate"]["topic_summary"],
        hooks=tuple(case["candidate"]["hooks"]),
        rights_risk=case["candidate"]["rights_risk"],
        originality_risk=case["candidate"]["originality_risk"],
        rights_status="OWNED",
        media_origin="OTHER",
        provenance_snapshot={},
        stage3_risk={"moment_density_score": case["candidate"]["moment_density_score"]},
        stage40_assessments={},
        stage40_platform_risk={},
        stage40_analysis_id=case["analysis"]["id"],
        stage40_input_fingerprint="in",
        stage40_output_fingerprint="out",
        stage40_policy_version=case["analysis"]["policy_version"],
        stage40_strategies=(),
        planning_context=PlanningContext(),
        refinement_priority="FINAL_CLIP",
        refinement_quality_level="FINAL_CLIP",
        refinement_status="FINAL_TRANSCRIPT_READY",
        refinement_output_fingerprint=refinement["output_fingerprint"],
        mandatory_evidence=mandatory.as_dict(),
    )


def _strategy_dict() -> dict[str, object]:
    return {
        "id": "strategy-1",
        "strategy_key": "candidate:ANALYSIS",
        "strategy_type": "ANALYSIS",
        "intensity": "MODERATE",
        "rank": 1,
        "strategy_fingerprint": "strategy-fp",
        "external_verification_requirement": "NOT_REQUIRED",
        "confidence": 0.8,
    }


def _plan(
    grounding: object, *, hero: tuple[int, int] = (3, 7), draft: str | None = None
) -> PlanProviderPlan:
    return PlanProviderPlan(
        strategy_id="strategy-1",
        strategy_key="candidate:ANALYSIS",
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                source_role=SourceExcerptRole.HERO,
                word_start_index=hero[0],
                word_end_index=hero[1],
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.ORIGINAL_VALUE,
                estimated_duration=2.0,
                substantive_value_kind=SubstantiveValueKind.EXPLANATION,
                semantic_intent="The speaker repeats the dish question as a question.",
                why_unavailable="The raw excerpt does not name the repetition as a question.",
                grounding_refs=tuple(grounding) if isinstance(grounding, (list, tuple)) else (),
                grounding_refs_invalid=not isinstance(grounding, (list, tuple)),
                draft_line=draft,
            ),
        ),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )


@pytest.mark.parametrize(
    ("case_name", "grounding", "expected"),
    [
        ("bare_decimal", ["793.32"], "GROUNDING_EVIDENCE_UNRESOLVED"),
        ("substring_quote", ["نزلك إيه؟"], "GROUNDING_EVIDENCE_UNRESOLVED"),
        ("unknown_block", ["block:9"], "GROUNDING_EVIDENCE_UNRESOLVED"),
        ("huge_word_range", ["words:3-400"], "GROUNDING_EVIDENCE_UNRESOLVED"),
        ("partial_word_range", ["words:2-4"], "GROUNDING_EVIDENCE_UNRESOLVED"),
        ("reversed_word_range", ["words:5-3"], "GROUNDING_DECLARATION_MALFORMED"),
        ("reversed_time_range", ["source:150-100"], "GROUNDING_DECLARATION_MALFORMED"),
        ("mixed_valid_invalid", ["block:0", "793.32"], "GROUNDING_EVIDENCE_UNRESOLVED"),
        ("empty_grounding", [], "GROUNDING_REQUIRED_MISSING"),
    ],
)
def test_authentic_negative_grounding_controls_still_fail(
    session: Session, case_name: str, grounding: object, expected: str
) -> None:
    inputs = _authentic_planning_inputs(session)
    result = validate_provider_plan(
        _plan(grounding),
        _strategy_dict(),
        inputs,
        STAGE41_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="provider-fp",
    )
    assert result.plan is None, case_name
    assert any(expected in reason for reason in result.reasons), result.reasons


def test_authentic_missing_hook_and_declared_missing_payoff_still_fail(session: Session) -> None:
    inputs = _authentic_planning_inputs(session)
    missing_hook = validate_provider_plan(
        _plan(["block:0"], hero=(40, 45)),
        _strategy_dict(),
        inputs,
        STAGE41_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="provider-fp",
    )
    assert missing_hook.plan is None
    assert any("HOOK_NOT_COVERED" in reason for reason in missing_hook.reasons)

    case = load_case("a725e62d-bec3-445d-93b4-76815febfde4")
    payoff_index = 73
    mandatory = resolve_mandatory_evidence(
        case["candidate"]["source_video_id"],
        {"hook_index": 71, "payoff_index": payoff_index},
        case["source_segments"],
    )
    with_payoff = replace(
        inputs,
        source_moment={"hook_index": 71, "payoff_index": payoff_index},
        mandatory_evidence=mandatory.as_dict(),
    )
    missing_payoff = validate_provider_plan(
        _plan(["block:0"], hero=(3, 6)),
        _strategy_dict(),
        with_payoff,
        STAGE41_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="provider-fp",
    )
    assert missing_payoff.plan is None
    assert any("PAYOFF_NOT_COVERED" in reason for reason in missing_payoff.reasons)
    assert not any("HOOK_NOT_COVERED" in reason for reason in missing_payoff.reasons)


def test_authentic_malformed_mandatory_identity_and_unsupported_claim_fail(
    session: Session,
) -> None:
    inputs = _authentic_planning_inputs(session)
    malformed = resolve_mandatory_evidence(
        inputs.source_id, {"hook_index": "343", "payoff_index": None}, []
    )
    malformed_inputs = replace(
        inputs,
        source_moment={"hook_index": "343", "payoff_index": None},
        mandatory_evidence=malformed.as_dict(),
    )
    result = validate_provider_plan(
        _plan(["block:0"]),
        _strategy_dict(),
        malformed_inputs,
        STAGE41_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="provider-fp",
    )
    assert result.plan is None
    assert any("HOOK_EVIDENCE_MALFORMED" in reason for reason in result.reasons)

    unsupported = validate_provider_plan(
        _plan(["block:0"], draft="The clip shows 47 percent of the negotiation."),
        _strategy_dict(),
        inputs,
        STAGE41_CONFIG,
        provider_evidence={},
        provider_input_fingerprint="provider-fp",
    )
    assert unsupported.plan is None
    assert any("UNSUPPORTED_FACT" in reason for reason in unsupported.reasons)
