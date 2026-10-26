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
from types import SimpleNamespace

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage41_contract_v4_support import (
    CANDIDATES,
    REPRESENTATIVE,
    REPRESENTATIVE_STRATEGY,
    AuthoredProvider,
    load_case,
    planning_word_range_for_segment,
    planning_word_range_for_span,
    planning_word_texts,
    representative_response,
    seed_authentic_case,
    selectable_planning_words,
)
from stage41_support import (
    FakeStage41Settings,
    install_stage41_settings,
)

from app.core.enums import (
    ContentType,
    DeliveryIntent,
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
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.grounding import parse_grounding_refs, resolve_grounding
from app.materialization.inputs import (
    MaterializationInputs,
    ProfileContext,
    SourceMediaEvidence,
    VisualPlanEvidence,
)
from app.materialization.service import _final_clip_evidence, _merge_blocks
from app.models import ProcessingJob
from app.render.binding import build_bound_spans
from app.render.compatibility import evaluate_compatibility
from app.render.policy import render_profile_for
from app.render.service import _assemble_contract, _final_evidence
from app.transformation.governance.executor import (
    build_transformation_governance_executor,
)
from app.transformation.governance.inputs import build_governance_inputs
from app.transformation.governance.policy import DEFAULT_CONFIG as STAGE42_CONFIG
from app.transformation.governance.queue import get_or_create_governance_set
from app.transformation.governance.validation import (
    evaluate_plan,
    finalize,
    revalidate_integrity,
)
from app.transformation.hook_payoff import resolve_mandatory_evidence
from app.transformation.planning.executor import TransformationPlanningExecutor
from app.transformation.planning.fingerprints import (
    build_provider_input_payload,
    provider_input_fingerprint,
)
from app.transformation.planning.inputs import _coverage_sufficient, build_planning_inputs
from app.transformation.planning.policy import (
    DEFAULT_CONFIG as STAGE41_CONFIG,
)
from app.transformation.planning.policy import (
    SCHEMA_VERSION,
    strategy_value_kinds,
)
from app.transformation.planning.providers import (
    PLANNING_OUTPUT_SCHEMA,
    PLANNING_SCHEMA_BINDING_VERSION,
    PLANNING_SYSTEM_INSTRUCTION,
    PlanningOutputSchema,
    build_planning_output_schema,
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
from app.transformation.selection.service import (
    _mandatory_failures,
    select_transformation_plan,
)


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
        strategy_type=REPRESENTATIVE_STRATEGY,
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
    return dict(inputs.stage40_strategies[0])


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
    assert payload["source_selection"]["words_offered"] == _EXPECTED_PLANNING_WORDS[REPRESENTATIVE]
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
        item.value
        for item in strategy_value_kinds(TransformationStrategyType(str(strategy["strategy_type"])))
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
    bound_schema = build_planning_output_schema([request])
    assert config.response_schema.model_json_schema() == bound_schema.model_json_schema()
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
                + PLANNING_SCHEMA_BINDING_VERSION
                + json.dumps(PlanningOutputSchema.model_json_schema(), sort_keys=True)
            ).encode("utf-8")
        )
        .hexdigest()
    )
    assert planning_prompt_hash() == expected_hash
    assert provider.runtime_identity()["schema_version"] == SCHEMA_VERSION
    assert provider.runtime_identity()["schema_binding_version"] == PLANNING_SCHEMA_BINDING_VERSION


_EXPECTED_PLANNING_WORDS = {
    "1827c524-e893-4769-8b22-31411c10b6e8": 113,
    "ac0670c4-dfd4-4609-b9b8-b2736b96b095": 114,
    "64e7807c-3b58-4ea7-9b80-d2ec3e543572": 127,
    "a725e62d-bec3-445d-93b4-76815febfde4": 117,
}


def _capture_gemini_request(request: object) -> dict[str, object]:
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
    provider.plan([request], "ROUTINE")  # type: ignore[list-item]
    provider.release()
    return captured


def _coverage_plan(strategy: dict[str, object], selection: dict[str, object]) -> PlanProviderPlan:
    kind = strategy_value_kinds(TransformationStrategyType(str(strategy["strategy_type"])))[0]
    method = str(selection.get("method"))
    hero = PlanProviderBlock(
        block_type=PlanBlockType.SOURCE_EXCERPT,
        purpose="Hero source moment",
        source_role=SourceExcerptRole.HERO,
        use_full_window=method == "FULL_WINDOW",
        word_start_index=selection.get("planning_word_start_index"),
        word_end_index=selection.get("planning_word_end_index"),
    )
    value = PlanProviderBlock(
        block_type=PlanBlockType.ORIGINAL_VALUE,
        purpose="Grounded substantive value",
        estimated_duration=2.0,
        substantive_value_kind=kind,
        semantic_intent="Synthesize the source evidence for the viewer",
        why_unavailable="The excerpt states the moment but not the synthesis",
        grounding_refs=("block:0",),
        delivery_intent=DeliveryIntent.ON_SCREEN_TEXT,
    )
    return PlanProviderPlan(
        strategy_id=str(strategy["id"]),
        strategy_key=str(strategy["strategy_key"]),
        blocks=(hero, value),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )


@pytest.mark.parametrize("candidate_id", CANDIDATES)  # type: ignore[untyped-decorator]
def test_actual_request_feasibility_and_gemini_boundary_agree_for_every_fixture(
    session: Session, monkeypatch: pytest.MonkeyPatch, candidate_id: str
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    case = load_case(candidate_id)
    _source, candidate, refinement, analysis, _strategies = seed_authentic_case(
        session,
        candidate_id=candidate_id,
        strategy_type=None,
        settings=settings,
    )
    inputs = build_planning_inputs(
        session, candidate, analysis, refinement, settings, settings.stage41_config()
    )
    assert len(inputs.words) == _EXPECTED_PLANNING_WORDS[candidate_id]
    hook_index = case["analysis"]["source_moment"]["hook_index"]
    hook_segment = case["source_segments"][hook_index]
    expected_word_range = planning_word_range_for_segment(
        case["refinement"], (hook_segment["start"], hook_segment["end"])
    )
    assert inputs.stage40_strategies
    for strategy in inputs.stage40_strategies:
        request = build_planning_request(strategy, inputs, settings.stage41_config())
        payload = request.to_payload()
        hook = payload["mandatory_evidence"]["hook"]
        assert hook["state"] == "RESOLVED"
        assert hook["segment_index"] == hook_index
        assert hook["source_start"] == hook_segment["start"]
        assert hook["source_end"] == hook_segment["end"]
        assert (
            hook["planning_word_start_index"],
            hook["planning_word_end_index"],
        ) == expected_word_range
        assert hook["mandatory_coverage_feasible"] is True
        assert hook["supported_selection"]["method"] in {"WORD_SELECTION", "FULL_WINDOW"}
        assert (
            payload["source_selection"]["words_offered"] == _EXPECTED_PLANNING_WORDS[candidate_id]
        )
        assert set(payload["strategy_constraints"]["allowed_value_kinds"]) == {
            item.value
            for item in strategy_value_kinds(
                TransformationStrategyType(str(strategy["strategy_type"]))
            )
        }
        assert payload["nonselectable_context"]["suggested_hook_copy"]
        assert "context_text" in payload["nonselectable_context"]

        identity = settings.transformation_planning_provider_identity()
        fingerprint = provider_input_fingerprint(
            build_provider_input_payload(
                request=payload, provider_identity=identity, route="ROUTINE"
            )
        )
        altered = json.loads(json.dumps(payload))
        altered["mandatory_evidence"]["hook"]["mandatory_coverage_feasible"] = False
        assert (
            provider_input_fingerprint(
                build_provider_input_payload(
                    request=altered, provider_identity=identity, route="ROUTINE"
                )
            )
            != fingerprint
        )

        captured = _capture_gemini_request(request)
        config = captured["config"]
        assert config.system_instruction == PLANNING_SYSTEM_INSTRUCTION
        assert (
            config.response_schema.model_json_schema()
            == build_planning_output_schema([request]).model_json_schema()
        )
        contents = json.loads(captured["contents"])
        assert contents["plans_requested"] == [payload]
        assert (
            contents["plans_requested"][0]["mandatory_evidence"]["hook"][
                "mandatory_coverage_feasible"
            ]
            is True
        )

        validation = validate_provider_plan(
            _coverage_plan(dict(strategy), hook["supported_selection"]),
            strategy,
            inputs,
            settings.stage41_config(),
            provider_evidence={},
            provider_input_fingerprint="offline-fixture",
        )
        assert validation.plan is not None, validation.reasons
        assert not any("HOOK_NOT_COVERED" in reason for reason in validation.reasons)


def test_ac0670c4_inter_word_endpoint_is_feasible_without_false_decline(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_id = "ac0670c4-dfd4-4609-b9b8-b2736b96b095"
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    _source, candidate, refinement, analysis, _strategies = seed_authentic_case(
        session, candidate_id=candidate_id, strategy_type=None, settings=settings
    )
    inputs = build_planning_inputs(
        session, candidate, analysis, refinement, settings, settings.stage41_config()
    )
    strategy = dict(inputs.stage40_strategies[0])
    assert inputs.word_coverage_sufficient is True
    payload = build_planning_request(strategy, inputs, settings.stage41_config()).to_payload()
    hook = payload["mandatory_evidence"]["hook"]
    # The hook end lies between word indexes 2 and 3; its word mapping alone is
    # incomplete, but the allowed excerpt still covers the endpoint tolerance.
    assert hook["end_endpoint_in_word"] is False
    assert hook["covered_by_offered_words"] is False
    assert hook["mandatory_coverage_feasible"] is True
    assert hook["supported_selection"]["method"] == "WORD_SELECTION"
    assert hook["supported_selection"]["planning_word_start_index"] == 0
    assert hook["supported_selection"]["planning_word_end_index"] >= 2
    validation = validate_provider_plan(
        _coverage_plan(strategy, hook["supported_selection"]),
        strategy,
        inputs,
        settings.stage41_config(),
        provider_evidence={},
        provider_input_fingerprint="offline-fixture",
    )
    assert validation.plan is not None, validation.reasons


def test_insufficient_nonempty_authentic_words_advertise_full_window_selection(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    case = load_case(REPRESENTATIVE)
    _source, candidate, refinement, analysis, _strategies = seed_authentic_case(
        session, strategy_type=REPRESENTATIVE_STRATEGY, settings=settings
    )
    inputs = build_planning_inputs(
        session, candidate, analysis, refinement, settings, settings.stage41_config()
    )
    hook_index = case["analysis"]["source_moment"]["hook_index"]
    hook_segment = case["source_segments"][hook_index]
    hook_start = float(hook_segment["start"])
    hook_end = float(hook_segment["end"])
    hook_word_indexes = [
        index
        for index, word in enumerate(inputs.words)
        if word.end > hook_start and word.start < hook_end
    ]
    assert hook_word_indexes
    # A genuine nonempty subset of the authentic planning words: it covers the
    # hook locally but not enough of the refined window to resolve word spans.
    subset = inputs.words[hook_word_indexes[0] : hook_word_indexes[-1] + 1]
    coverage = _coverage_sufficient(subset, inputs.refined_start, inputs.refined_end)
    assert subset
    assert coverage is False
    bounded = replace(inputs, words=subset, word_coverage_sufficient=coverage)
    strategy = dict(bounded.stage40_strategies[0])
    payload = build_planning_request(strategy, bounded, settings.stage41_config()).to_payload()
    hook = payload["mandatory_evidence"]["hook"]
    assert hook["state"] == "RESOLVED"
    assert hook["mandatory_coverage_feasible"] is True
    assert hook["supported_selection"]["method"] == "FULL_WINDOW"
    validation = validate_provider_plan(
        _coverage_plan(strategy, hook["supported_selection"]),
        strategy,
        bounded,
        settings.stage41_config(),
        provider_evidence={},
        provider_input_fingerprint="offline-fixture",
    )
    assert validation.plan is not None, validation.reasons


def test_wordless_full_window_case_is_supported_and_declines_only_when_impossible(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    case = load_case(REPRESENTATIVE)
    _source, candidate, refinement, analysis, _strategies = seed_authentic_case(
        session, strategy_type=REPRESENTATIVE_STRATEGY, settings=settings
    )
    inputs = build_planning_inputs(
        session, candidate, analysis, refinement, settings, settings.stage41_config()
    )
    wordless = replace(inputs, words=(), word_coverage_sufficient=False)
    strategy = dict(wordless.stage40_strategies[0])
    payload = build_planning_request(strategy, wordless, settings.stage41_config()).to_payload()
    hook = payload["mandatory_evidence"]["hook"]
    assert hook["mandatory_coverage_feasible"] is True
    assert hook["supported_selection"]["method"] == "FULL_WINDOW"
    assert "full-window fallback" in PLANNING_SYSTEM_INSTRUCTION
    validation = validate_provider_plan(
        _coverage_plan(strategy, hook["supported_selection"]),
        strategy,
        wordless,
        settings.stage41_config(),
        provider_evidence={},
        provider_input_fingerprint="offline-fixture",
    )
    assert validation.plan is not None, validation.reasons

    # Truly uncoverable mandatory evidence is not feasible and cannot pass the
    # validator through either a word selection or the full-window fallback.
    far_index = next(
        index
        for index, segment in enumerate(case["source_segments"])
        if float(segment["start"]) > float(refinement.refined_end) + 5.0
    )
    mandatory = resolve_mandatory_evidence(
        str(candidate.source_video_id),
        {"hook_index": far_index, "payoff_index": None},
        case["source_segments"],
    )
    uncoverable = replace(
        inputs,
        source_moment={"hook_index": far_index, "payoff_index": None},
        mandatory_evidence=mandatory.as_dict(),
    )
    payload = build_planning_request(strategy, uncoverable, settings.stage41_config()).to_payload()
    hook = payload["mandatory_evidence"]["hook"]
    assert hook["state"] == "RESOLVED"
    assert hook["mandatory_coverage_feasible"] is False
    assert hook["supported_selection"]["method"] == "NONE"
    rejected = validate_provider_plan(
        _coverage_plan(strategy, {"method": "FULL_WINDOW"}),
        strategy,
        uncoverable,
        settings.stage41_config(),
        provider_evidence={},
        provider_input_fingerprint="offline-fixture",
    )
    assert rejected.plan is None
    assert any("HOOK_NOT_COVERED" in reason for reason in rejected.reasons)


def test_malformed_and_unresolved_mandatory_evidence_remain_distinct(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    case = load_case(REPRESENTATIVE)
    _source, candidate, refinement, analysis, _strategies = seed_authentic_case(
        session, strategy_type=REPRESENTATIVE_STRATEGY, settings=settings
    )
    inputs = build_planning_inputs(
        session, candidate, analysis, refinement, settings, settings.stage41_config()
    )
    strategy = dict(inputs.stage40_strategies[0])

    malformed_evidence = resolve_mandatory_evidence(
        str(candidate.source_video_id),
        {"hook_index": "343", "payoff_index": None},
        case["source_segments"],
    )
    malformed = replace(
        inputs,
        source_moment={"hook_index": "343", "payoff_index": None},
        mandatory_evidence=malformed_evidence.as_dict(),
    )
    malformed_payload = build_planning_request(
        strategy, malformed, settings.stage41_config()
    ).to_payload()
    assert malformed_payload["mandatory_evidence"]["hook"]["state"] == "MALFORMED"
    assert malformed_payload["mandatory_evidence"]["hook"]["mandatory_coverage_feasible"] is False

    unresolved_evidence = resolve_mandatory_evidence(
        str(candidate.source_video_id),
        {"hook_index": len(case["source_segments"]) + 5, "payoff_index": None},
        case["source_segments"],
    )
    unresolved = replace(
        inputs,
        source_moment={"hook_index": len(case["source_segments"]) + 5, "payoff_index": None},
        mandatory_evidence=unresolved_evidence.as_dict(),
    )
    unresolved_payload = build_planning_request(
        strategy, unresolved, settings.stage41_config()
    ).to_payload()
    assert unresolved_payload["mandatory_evidence"]["hook"]["state"] == "UNRESOLVED"
    assert unresolved_payload["mandatory_evidence"]["hook"]["mandatory_coverage_feasible"] is False


def _run_authentic_planning(
    session: Session,
    settings: FakeStage41Settings,
    provider: object,
    *,
    job: ProcessingJob | None = None,
    strategy_type: TransformationStrategyType | None = REPRESENTATIVE_STRATEGY,
) -> tuple[object, object, object, object]:
    from app.transformation.planning.queue import get_or_create_plan_set

    source, candidate, refinement, analysis, strategies = seed_authentic_case(
        session,
        strategy_type=strategy_type,
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
    from stage43_support import FakeGovernanceSettings, install_selection_settings

    settings = FakeGovernanceSettings(governance_mode="deterministic")
    install_selection_settings(monkeypatch, settings)
    case = load_case(REPRESENTATIVE)
    hook_index = case["analysis"]["source_moment"]["hook_index"]
    hook_segment = case["source_segments"][hook_index]
    demand_segment = case["source_segments"][hook_index + 1]
    word_range = planning_word_range_for_span(
        case["refinement"], (hook_segment["start"], demand_segment["end"])
    )
    assert word_range is not None
    word_start, word_end = word_range
    assert word_start > 0  # a nonzero planning index space, not an identity shortcut
    texts = planning_word_texts(case["refinement"], word_start, word_end)
    assert len(texts) >= 6
    excerpt_quote = " ".join(texts[:4])
    demand_quote = " ".join(texts[4:])
    authentic_strategy = next(
        item
        for item in case["strategies"]
        if item["strategy_type"] == "CONTEXT_HOOK"
        and item["is_current"]
        and item["disposition"] == "RECOMMENDED"
    )
    response = representative_response(
        strategy_id=authentic_strategy["id"],
        strategy_key=authentic_strategy["strategy_key"],
        word_start=word_start,
        word_end=word_end,
        excerpt_quote=excerpt_quote,
        demand_quote=demand_quote,
    )
    PlanningOutputSchema.model_validate(response)

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
    assert hero["source_text"] == " ".join(texts)
    assert len(hero["planning_words"]) == word_end - word_start + 1
    # Planning indexes are not FINAL_CLIP indexes: the first planning word maps
    # to an authentic refinement word at a different stored index.
    final_index = next(
        index
        for index, item in enumerate(refinement.word_timestamps)
        if float(item["start"]) == float(hero["planning_words"][0]["start"])
        and float(item["end"]) == float(hero["planning_words"][0]["end"])
    )
    assert final_index != hero["word_start_index"]

    # Inspect the authored contribution directly. It states a relation between
    # two authentic phrases: the quoted gold/diamonds are the quoted demands.
    contribution = plan.blocks[1]["semantic_intent"]
    assert contribution == f'The source context explains "{excerpt_quote}" as "{demand_quote}".'
    assert excerpt_quote in hero["source_text"] and demand_quote in hero["source_text"]
    assert plan.blocks[1]["substantive_value_kind"] == "EXPLANATION"

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
    assert not evaluation.revisions, evaluation.revisions
    assert evaluation.verification["claim_state"] == "GROUNDED_IN_SOURCE"
    status, _severity = finalize(evaluation)
    assert status.value in {"APPROVED_FOR_SELECTION", "APPROVED_WITH_CAUTION"}

    assert _mandatory_failures(session, candidate, [plan]) == {}

    # Persist the real deterministic Stage 4.2 result. No approval row is
    # fabricated. Deterministic governance cannot supply the semantic review
    # dimension, so Stage 4.3 correctly declines automatic selection; that is a
    # truthful offline blocker for a persisted contract, not something this
    # milestone may paper over with a manufactured favorable critique.
    governance_set = get_or_create_governance_set(session, candidate, plan_set)
    build_transformation_governance_executor(session, settings).execute(governance_set.id)
    session.refresh(governance_set)
    assert governance_set.execution_status.value == "COMPLETE"
    assert governance_set.governance_outcome.value == "PLANS_ELIGIBLE_FOR_SELECTION"
    selection_view = select_transformation_plan(session, candidate.id)
    assert selection_view is not None and selection_view.effective
    assert selection_view.row.status.value == "NO_SELECTABLE_PLAN"
    assert "NO_SELECTABLE_PLAN" in list(selection_view.row.selection_reason_codes or [])

    # Real Stage 5.0 compatibility, binding, and contract assembly primitives.
    final = _final_evidence(refinement)
    compatibility = evaluate_compatibility(
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
        caption_source_fingerprint="offline-caption",
        planning_refinement_id=str(refinement.id),
        planning_refinement_priority=refinement.priority.value,
        planning_refinement_quality_level=refinement.quality_level,
        planning_output_fingerprint=plan.plan_output_fingerprint,
    )
    assert compatibility.outcome == "EXACT_MATCH", compatibility.reason_codes
    bound_spans = build_bound_spans(
        [dict(block) for block in plan.blocks],
        {verdict.block_index: verdict for verdict in compatibility.per_block},
        hero_block_index=plan.hero_block_index,
        words=final.words,
    )
    contract_config = get_settings().stage50_config()
    profile = render_profile_for(contract_config.profile_key)
    facts = SimpleNamespace(
        duration_seconds=90.0,
        width=1920,
        height=1080,
        frames_per_second=30.0,
        audio_sample_rate=48000,
        audio_channels=2,
        as_dict=lambda: {
            "duration_seconds": 90.0,
            "width": 1920,
            "height": 1080,
            "frames_per_second": 30.0,
            "audio_sample_rate": 48000,
            "audio_channels": 2,
        },
    )
    gathered = SimpleNamespace(
        candidate=candidate,
        handoff={},
        selection=selection_view.row,
        selection_view=selection_view,
        selected_plan=plan,
        planning_refinement=refinement,
        final_refinement=refinement,
        source_segments=[dict(item) for item in case["source_segments"]],
        governance_snapshot={},
    )
    assembly = _assemble_contract(
        candidate=candidate,
        gathered=gathered,
        final_row=refinement,
        compatibility=compatibility,
        bound_spans=bound_spans,
        facts=facts,
        media_identity={
            "relative_path": "offline/source.mp4",
            "content_hash": "offline-hash",
            "size_bytes": 1,
            "mtime_ns": 1,
        },
        caption_fingerprint="offline-caption",
        profile=profile,
        config=contract_config,
    )
    contract = SimpleNamespace(contract_payload=assembly.payload)
    contract_source = next(
        item for item in assembly.payload["blocks"] if item.get("block_type") == "SOURCE_EXCERPT"
    )
    binding = contract_source["source_binding"]
    assert binding["final_clip_word_start"] is not None
    assert binding["final_clip_word_start"] != word_start
    assert binding["final_clip_text"] == " ".join(texts)

    # Production contract-to-materialization merge (_merge_blocks) and the real
    # grounding resolver. The merged blocks and bindings are not hand-authored;
    # only unrelated media/profile scaffolding is synthetic.
    merged_blocks, bindings = _merge_blocks(contract, plan)
    merged = next(item for item in merged_blocks if item.get("block_type") == "SOURCE_EXCERPT")
    assert merged["planning_word_start_index"] == word_start
    assert merged["planning_word_end_index"] == word_end
    assert [item["text"] for item in merged["planning_words"]] == list(texts)
    assert merged["word_start_index"] == binding["final_clip_word_start"]
    assert merged["word_end_index"] == binding["final_clip_word_end"]
    assert merged["word_start_index"] != merged["planning_word_start_index"]
    assert bindings[0]["final_clip_word_start"] == binding["final_clip_word_start"]
    final_clip = _final_clip_evidence(refinement)
    assert final_clip.start == float(refinement.refined_start)
    materialization = MaterializationInputs(
        candidate_id=str(candidate.id),
        candidate_key=candidate.candidate_key,
        source_id=str(candidate.source_video_id),
        priority="CANDIDATE",
        selection_id=str(selection_view.row.id),
        selection_output_fingerprint=selection_view.row.output_fingerprint,
        selected_plan_fingerprint=plan.plan_output_fingerprint,
        plan_id=str(plan.id),
        plan_output_fingerprint=plan.plan_output_fingerprint,
        strategy_type=plan.strategy_type.value,
        intensity=plan.intensity.value,
        blocks=merged_blocks,
        hero_block_index=int(plan.hero_block_index),
        narration={
            "need": plan.narration_need,
            "requirements": dict(plan.narration_requirements or {}),
        },
        verification_dependencies=(),
        original_value_kinds=tuple(plan.original_value_kinds or []),
        required_context=tuple(plan.required_context or []),
        preservation_constraints=tuple(plan.preservation_constraints or []),
        plan_target_context={},
        contract_id="offline-contract",
        contract_status="MATERIALIZATION_REQUIRED",
        contract_input_fingerprint="offline-contract-input",
        contract_output_fingerprint="offline-contract-output",
        execution_slots=(),
        contract_source_bindings=bindings,
        final_clip=final_clip,
        visual_plan=VisualPlanEvidence(
            visual_plan_id="offline-visual",
            output_fingerprint="offline-visual-output",
            plan_ready=True,
            ass_relative_path="offline/captions.ass",
            ass_sha256="a" * 64,
            ass_policy_version="offline",
            caption_source_fingerprint="offline-caption",
            framing_fingerprint="offline-framing",
            source_media_fingerprint="offline-media",
        ),
        source_media=SourceMediaEvidence(
            relative_path="offline/source.mp4",
            content_hash="offline-hash",
            size_bytes=1,
            mtime_ns=1,
            duration=facts.duration_seconds,
            frame_rate=Fraction(str(facts.frames_per_second)).limit_denominator(1001),
            display_width=facts.width,
            display_height=facts.height,
            encoded_width=facts.width,
            encoded_height=facts.height,
            sample_rate=facts.audio_sample_rate,
            channels=2,
        ),
        profile=ProfileContext(
            profile_id="offline-profile",
            profile_key="offline-contract",
            target_market="UNSPECIFIED",
            output_language_policy="SOURCE_LANGUAGE",
            register_intent="SOURCE_COMPATIBLE",
            narration_allowed=False,
        ),
    )

    refs = parse_grounding_refs([f"words:{word_start + 1}-{word_end - 1}"])
    resolution = resolve_grounding(materialization, refs)
    assert resolution.unresolved == ()
    assert resolution.text == " ".join(texts[1:-1])

    # Removing the preserved planning-word mapping makes a partial citation
    # unresolvable; the accepted full-block citation still works.
    without_words = replace(
        materialization,
        blocks=tuple(
            {**item, "planning_words": []} if item.get("block_type") == "SOURCE_EXCERPT" else item
            for item in materialization.blocks
        ),
    )
    broken = resolve_grounding(without_words, refs)
    assert broken.unresolved
    assert "exact_word_evidence_unavailable" in broken.unresolved[0]
    assert resolve_grounding(without_words, parse_grounding_refs(["block:0"])).unresolved == ()

    # Corrupting the bound source span fails closed.
    corrupt_binding = replace(
        materialization,
        blocks=tuple(
            {**item, "source_start": 0.0, "source_end": 1.0}
            if item.get("block_type") == "SOURCE_EXCERPT"
            else item
            for item in materialization.blocks
        ),
    )
    corrupt = resolve_grounding(corrupt_binding, parse_grounding_refs(["block:0"]))
    assert corrupt.unresolved
    assert "out_of_scope_block" in corrupt.unresolved[0]

    # A planning word admitted by the 0.75-second selection tolerance can lie
    # outside the strict FINAL_CLIP scope. It remains rejected as final evidence.
    tolerance_word = selectable_planning_words(case["refinement"])[0]
    assert float(tolerance_word["start"]) >= float(refinement.refined_start) - 0.75
    assert float(tolerance_word["start"]) < float(refinement.refined_start)
    out_of_scope = replace(
        materialization,
        blocks=tuple(
            {
                **item,
                "source_start": float(tolerance_word["start"]),
                "source_end": float(tolerance_word["end"]),
            }
            if item.get("block_type") == "SOURCE_EXCERPT"
            else item
            for item in materialization.blocks
        ),
    )
    rejected = resolve_grounding(out_of_scope, parse_grounding_refs(["block:0"]))
    assert rejected.unresolved
    assert "out_of_scope_block" in rejected.unresolved[0]


def test_rejection_diagnostics_are_bounded_and_survive_a_later_run(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    case = load_case(REPRESENTATIVE)
    hook_index = case["analysis"]["source_moment"]["hook_index"]
    segment = case["source_segments"][hook_index]
    next_segment = case["source_segments"][hook_index + 1]
    word_range = planning_word_range_for_span(
        case["refinement"], (segment["start"], next_segment["end"])
    )
    assert word_range is not None
    strategy = next(
        item
        for item in case["strategies"]
        if item["strategy_type"] == "CONTEXT_HOOK" and item["is_current"]
    )
    texts = planning_word_texts(case["refinement"], word_range[0], word_range[1])
    invalid = representative_response(
        strategy_id=strategy["id"],
        strategy_key=strategy["strategy_key"],
        word_start=word_range[0],
        word_end=word_range[1],
        excerpt_quote=" ".join(texts[:4]),
        demand_quote=" ".join(texts[4:]),
    )
    invalid["plans"][0]["blocks"][1]["grounding_refs"] = ["793.32", "block:0"]

    source, candidate, refinement, analysis, _strategies = seed_authentic_case(
        session, strategy_type=REPRESENTATIVE_STRATEGY, settings=settings
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
    assert entry["route"] == "ROUTINE"
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
        excerpt_quote=" ".join(texts[:4]),
        demand_quote=" ".join(texts[4:]),
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


def _representative_content(
    case: dict[str, object],
    strategy: dict[str, object],
    *,
    wrong_id: bool = False,
) -> dict[str, object]:
    hook_index = case["analysis"]["source_moment"]["hook_index"]  # type: ignore[index]
    hook = case["source_segments"][hook_index]  # type: ignore[index]
    following = case["source_segments"][hook_index + 1]  # type: ignore[index]
    word_range = planning_word_range_for_span(
        case["refinement"],
        (hook["start"], following["end"]),  # type: ignore[index]
    )
    assert word_range is not None
    texts = planning_word_texts(case["refinement"], word_range[0], word_range[1])  # type: ignore[arg-type]
    content = representative_response(
        strategy_id="not-the-requested-id" if wrong_id else str(strategy["id"]),
        strategy_key=str(strategy["strategy_key"]),
        word_start=word_range[0],
        word_end=word_range[1],
        excerpt_quote=" ".join(texts[:4]),
        demand_quote=" ".join(texts[4:]),
    )
    return content


def _seed_planning_job(
    session: Session,
    settings: FakeStage41Settings,
    *,
    candidate_id: str = REPRESENTATIVE,
    strategy_type: TransformationStrategyType | None = REPRESENTATIVE_STRATEGY,
) -> tuple[object, object, object, ProcessingJob]:
    source, candidate, refinement, analysis, _strategies = seed_authentic_case(
        session, candidate_id=candidate_id, strategy_type=strategy_type, settings=settings
    )
    plan_set = get_or_create_plan_set(session, candidate, analysis)
    job = ProcessingJob(
        source_video_id=source.id,
        transformation_plan_set_id=plan_set.id,
        kind=JobKind.TRANSFORMATION_PLANNING,
    )
    session.add(job)
    session.commit()
    return source, candidate, plan_set, job


def _execute_with_content(
    session: Session,
    settings: FakeStage41Settings,
    plan_set: object,
    job: ProcessingJob,
    provider: object,
    *,
    force: bool = False,
) -> None:
    executor = TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=provider,  # type: ignore[arg-type]
        provider_identity=settings.transformation_planning_provider_identity(),
        mode=SemanticProviderMode.ADAPTIVE,
        config=STAGE41_CONFIG,
        lease_factory=settings.heavy_model_lease_factory(),  # type: ignore[arg-type]
    )
    executor.set_active_job(job.id)
    executor.execute(plan_set.id, force=force)  # type: ignore[attr-defined]
    session.refresh(plan_set)


def test_identity_rejection_is_diagnosed_not_treated_as_omission_and_not_reused(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    case = load_case(REPRESENTATIVE)
    strategy = next(
        item
        for item in case["strategies"]
        if item["strategy_type"] == "CONTEXT_HOOK" and item["is_current"]
    )
    source, candidate, plan_set, first_job = _seed_planning_job(session, settings)
    invalid = _representative_content(case, strategy, wrong_id=True)
    _execute_with_content(session, settings, plan_set, first_job, AuthoredProvider(invalid))

    assert list_plans(session, plan_set.id) == []  # type: ignore[attr-defined]
    assert plan_set.planning_outcome.value == "PLANNING_DEFERRED"  # type: ignore[attr-defined]
    entries = dict(plan_set.provider_evidence or {}).get("attempt_diagnostics")  # type: ignore[attr-defined]
    assert isinstance(entries, list) and entries
    entry = entries[-1]
    assert entry["phase"] == "PARSER_REJECTION"
    assert "STRATEGY_ID_MISMATCH" in entry["failure_codes"]
    assert entry["parser_rejections"][0]["code"] == "STRATEGY_ID_MISMATCH"
    assert entry["parser_rejections"][0]["requested_strategy_key"] == strategy["strategy_key"]
    assert entry["response_evidence"]["returned_plan_count"] == 1
    assert entry["response_evidence"]["content_digest"]
    assert entry["response_evidence"]["content_bytes"] > 0
    # The returned identity is never persisted verbatim.
    assert "not-the-requested-id" not in json.dumps(entry, ensure_ascii=False)
    attempts = [dict(item) for item in plan_set.strategy_attempts or []]  # type: ignore[attr-defined]
    assert attempts and attempts[0]["checkpoint"] is None

    # A later valid run adds a plan, keeps the earlier diagnostic, and never
    # reuses the rejected attempt as a checkpoint.
    valid = _representative_content(case, strategy)
    second_job = ProcessingJob(
        source_video_id=source.id,
        transformation_plan_set_id=plan_set.id,  # type: ignore[attr-defined]
        kind=JobKind.TRANSFORMATION_PLANNING,
    )
    session.add(second_job)
    session.commit()
    _execute_with_content(
        session,
        settings,
        plan_set,
        second_job,
        AuthoredProvider(valid),
        force=True,
    )
    plans = list_plans(session, plan_set.id)  # type: ignore[attr-defined]
    assert len(plans) == 1
    entries = dict(plan_set.provider_evidence or {}).get("attempt_diagnostics")  # type: ignore[attr-defined]
    assert isinstance(entries, list)
    assert str(first_job.id) in {item["job_id"] for item in entries}
    assert str(second_job.id) not in {item["job_id"] for item in entries}
    assert any(item["phase"] == "PARSER_REJECTION" for item in entries)


def test_mixed_accepted_and_identity_rejected_siblings_are_isolated(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    case = load_case(REPRESENTATIVE)
    recommended = {
        item["strategy_type"]: item
        for item in case["strategies"]
        if item["is_current"] and item["disposition"] == "RECOMMENDED"
    }
    hook_strategy = recommended["CONTEXT_HOOK"]
    minimal_strategy = recommended["SOURCE_LED_MINIMAL"]
    content = _representative_content(case, hook_strategy)
    rejected_plan = _representative_content(case, minimal_strategy, wrong_id=True)["plans"][0]
    content["plans"].append(rejected_plan)

    _source, _candidate, plan_set, job = _seed_planning_job(session, settings, strategy_type=None)
    _execute_with_content(session, settings, plan_set, job, AuthoredProvider(content))

    plans = list_plans(session, plan_set.id)  # type: ignore[attr-defined]
    assert len(plans) == 1
    assert plans[0].strategy_type.value == "CONTEXT_HOOK"
    attempts = {
        item["strategy_key"]: dict(item)
        for item in plan_set.strategy_attempts or []  # type: ignore[attr-defined]
    }
    assert attempts[hook_strategy["strategy_key"]]["status"] == "GENERATED"
    rejected = attempts[minimal_strategy["strategy_key"]]
    assert rejected["status"] == "DEFERRED"
    assert rejected["diagnostics"]["phase"] == "PARSER_REJECTION"
    assert rejected["diagnostics"]["parser_rejections"][0]["code"] == "STRATEGY_ID_MISMATCH"
    assert rejected["checkpoint"] is None
    assert plan_set.planning_outcome.value == "PLANS_GENERATED"  # type: ignore[attr-defined]


def test_true_omission_and_transport_failure_remain_distinct(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    _source, _candidate, plan_set, omission_job = _seed_planning_job(session, settings)
    _execute_with_content(
        session, settings, plan_set, omission_job, AuthoredProvider({"plans": []})
    )
    entries = dict(plan_set.provider_evidence or {}).get("attempt_diagnostics")  # type: ignore[attr-defined]
    omission = entries[-1]
    assert omission["phase"] == "OMISSION"
    assert omission["failure_codes"] == ["PROVIDER_OMITTED"]
    assert omission["response_evidence"] == {}
    assert omission["parser_rejections"] == []

    class _TransportFailure:
        hosted_provider = True

        def plan(self, requests: object, tier: str = "ROUTINE") -> object:
            from app.transformation.planning.providers import PlanningProviderError

            raise PlanningProviderError("TIMEOUT")

        def release(self) -> None:
            return None

        def runtime_identity(self) -> dict[str, object]:
            return {"provider": "offline-transport-failure"}

        def refresh_runtime_identity(self) -> dict[str, object]:
            return self.runtime_identity()

        def raw_call_count(self) -> int:
            return 0

    transport_job = ProcessingJob(
        source_video_id=omission_job.source_video_id,
        transformation_plan_set_id=plan_set.id,  # type: ignore[attr-defined]
        kind=JobKind.TRANSFORMATION_PLANNING,
    )
    session.add(transport_job)
    session.commit()
    _execute_with_content(
        session,
        settings,
        plan_set,
        transport_job,
        _TransportFailure(),
        force=True,
    )
    entries = dict(plan_set.provider_evidence or {}).get("attempt_diagnostics")  # type: ignore[attr-defined]
    transport = entries[-1]
    assert transport["phase"] == "TRANSPORT"
    assert transport["failure_codes"] == ["TIMEOUT"]
    assert transport["response_evidence"] == {}
    assert transport["parser_rejections"] == []


def test_parser_retains_malformed_and_unknown_entry_evidence(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.transformation.planning.providers import parse_plan_results

    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    inputs, _, _case = _authentic_inputs(session, settings)
    strategy = _strategy(inputs)
    request = build_planning_request(strategy, inputs, settings.stage41_config())
    key = request.strategy_key
    valid_plan = {
        "strategy_id": request.strategy_id,
        "strategy_key": key,
        "blocks": [
            {
                "block_type": "SOURCE_EXCERPT",
                "source_role": "HERO",
                "use_full_window": True,
            },
            {
                "block_type": "ORIGINAL_VALUE",
                "substantive_value_kind": "EXPLANATION",
                "semantic_intent": "Synthesize the source evidence for the viewer",
                "why_unavailable": "The excerpt states the moment but not the synthesis",
                "grounding_refs": ["block:0"],
                "estimated_duration": 2.0,
            },
        ],
    }
    duplicate = json.loads(json.dumps({"plans": [valid_plan, valid_plan]}))
    results = parse_plan_results(duplicate, [request])
    assert len(results[key].plans) == 1
    assert any(item["code"] == "DUPLICATE_STRATEGY_KEY" for item in results[key].rejections)

    malformed_container = {
        "plans": [{"strategy_id": request.strategy_id, "strategy_key": key, "blocks": "nope"}]
    }
    results = parse_plan_results(malformed_container, [request])
    assert results[key].plans[0].blocks == ()
    assert any(item["code"] == "MALFORMED_BLOCKS_CONTAINER" for item in results[key].rejections)

    unknown = {"plans": [{"strategy_id": "other", "strategy_key": "unknown-key", "blocks": []}]}
    results = parse_plan_results(unknown, [request])
    assert key in results and results[key].plans == ()
    assert results[key].rejections[0]["code"] == "UNKNOWN_STRATEGY_KEY"
    assert results[key].response_evidence["unmatched_entry_count"] == 1
    assert "unknown-key" not in json.dumps(dict(results[key].response_evidence))

    non_mapping = {"plans": ["not-a-mapping"]}
    results = parse_plan_results(non_mapping, [request])
    assert results[key].plans == ()
    assert results[key].rejections[0]["code"] == "MALFORMED_ENTRY"


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
