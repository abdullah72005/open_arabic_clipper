"""Offline Stage 4.1 strategy/value-kind provider contract regressions.

Every provider response here is an isolated controlled fixture. No test starts a
worker, model, network client, renderer, or acceptance run, and no fixture is
production evidence.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage41_support import (
    FakePlanningProvider,
    FakeStage41Settings,
    install_stage41_settings,
    make_source_value_plan,
    run_planning,
    seed_stage41,
)

from app.core.enums import SubstantiveValueKind, TransformationStrategyType
from app.db.base import Base
from app.transformation.planning.fingerprints import (
    build_provider_input_payload,
    provider_input_fingerprint,
)
from app.transformation.planning.gemini import GeminiPlanningProvider
from app.transformation.planning.inputs import build_planning_inputs
from app.transformation.planning.policy import (
    SCHEMA_VERSION,
    strategy_value_kinds,
)
from app.transformation.planning.providers import (
    PLANNING_SCHEMA_BINDING_VERSION,
    PlanningRequest,
    build_planning_output_schema,
    planning_output_schema_digest,
    planning_prompt_hash,
)
from app.transformation.planning.validation import (
    REJECT_VALUE_KIND_MISMATCH,
    validate_provider_plan,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _request(
    strategy_type: TransformationStrategyType,
    strategy_id: str,
    strategy_key: str,
) -> PlanningRequest:
    allowed = strategy_value_kinds(strategy_type)
    return PlanningRequest(
        strategy_id=strategy_id,
        strategy_key=strategy_key,
        strategy_type=strategy_type.value,
        intensity="MODERATE",
        direction_summary="d",
        added_value_focus="Explain the moment without inventing facts",
        substantive_value_kind=(allowed[0] if allowed else SubstantiveValueKind.SYNTHESIS).value,
        preservation_requirements=(),
        external_verification_requirement="NOT_REQUIRED",
        verification_requirements=(),
        content_type="ANALYSIS",
        source_moment_structure="CLAIM",
        dialect_profile="EGYPTIAN",
        code_switch_tokens=(),
        target_market="UNSPECIFIED",
        output_language_policy="SOURCE_LANGUAGE",
        register_intent="SOURCE_COMPATIBLE",
        narration_allowed=True,
        max_blocks=8,
        strict_hero_cap_seconds=1.5,
        refined_transcript="text",
        refined_start=0.0,
        refined_end=10.0,
        context_text="",
        idea_summary="",
        topic_summary="",
        hooks=(),
        words=(),
        strategy_constraints={"allowed_value_kinds": [item.value for item in allowed]},
    )


def _definition_name(reference: str) -> str:
    return reference.rsplit("/", 1)[-1]


def _schema_value_kinds(schema: Mapping[str, Any], strategy_key: str) -> set[str]:
    """Resolve the per-strategy block enum from the serialized schema."""

    definitions = schema["$defs"]
    items = schema["properties"]["plans"]["items"]
    variants = items.get("anyOf") or [items]
    found: dict[str, set[str]] = {}
    for variant in variants:
        reference = variant.get("$ref")
        if not reference:
            continue
        plan = definitions[_definition_name(reference)]
        key_spec = plan["properties"]["strategy_key"]
        keys = key_spec.get("enum") or ([key_spec["const"]] if "const" in key_spec else [])
        block_reference = plan["properties"]["blocks"]["items"]["$ref"]
        block = definitions[_definition_name(block_reference)]
        kind_spec = block["properties"]["substantive_value_kind"]
        kind_items = kind_spec.get("anyOf") or [kind_spec]
        kinds: set[str] = set()
        for item in kind_items:
            if "enum" in item:
                kinds.update(item["enum"])
            elif "const" in item:
                kinds.add(item["const"])
        for key in keys:
            found[key] = kinds
    assert strategy_key in found, found
    return found[strategy_key]


def _block(kind: str) -> dict[str, object]:
    return {
        "block_type": "ORIGINAL_VALUE",
        "purpose": "value",
        "estimated_duration": 1.0,
        "interrupts_source": False,
        "preservation_constraints": [],
        "dependency_ids": [],
        "source_role": None,
        "word_start_index": None,
        "word_end_index": None,
        "use_full_window": False,
        "continuity_rationale": None,
        "substantive_value_kind": kind,
        "semantic_intent": "Explain the causal relationship in the selected moment",
        "why_unavailable": "The excerpt states the outcome without the cause",
        "grounding_refs": ["block:0"],
        "delivery_intent": "ON_SCREEN_TEXT",
        "draft_line": None,
        "claim_dependency": None,
        "verification_rationale": None,
        "intended_use": None,
        "must_verify_before_execution": False,
        "dependent_block_ids": [],
    }


def _narration() -> dict[str, object]:
    return {
        "need": "NONE",
        "purposes": [],
        "language": None,
        "register_intent": None,
        "estimated_duration": 0.0,
        "placement_block_index": None,
        "max_source_interruption_seconds": 0.0,
        "overlaps_source_audio": False,
        "replaces_silence": False,
        "essential": False,
        "verification_dependency_ids": [],
    }


def _plan_for(strategy_id: str, strategy_key: str, kind: str) -> dict[str, object]:
    return {
        "strategy_id": strategy_id,
        "strategy_key": strategy_key,
        "confidence": 0.7,
        "no_valid_plan": False,
        "no_valid_reason": "",
        "planner_notes": "",
        "preservation_constraints": [],
        "blocks": [_block(kind)],
        "narration": _narration(),
    }


def _plan(request: PlanningRequest, kind: str) -> dict[str, object]:
    return _plan_for(request.strategy_id, request.strategy_key, kind)


def _plan_with_source(strategy_id: str, strategy_key: str, kind: str) -> dict[str, object]:
    plan = _plan_for(strategy_id, strategy_key, kind)
    source = _block(kind)
    source.update(
        {
            "block_type": "SOURCE_EXCERPT",
            "source_role": "HERO",
            "use_full_window": True,
            "substantive_value_kind": None,
        }
    )
    plan["blocks"] = [source, _block(kind)]
    return plan


def _response(plans: list[dict[str, object]]) -> object:
    return type(
        "_Response",
        (),
        {
            "parsed": None,
            "text": json.dumps({"plans": plans}, ensure_ascii=False),
            "usage_metadata": None,
            "prompt_feedback": None,
            "candidates": [type("_C", (), {"finish_reason": None})()],
        },
    )()


def _client(response: object, capture: dict[str, object]) -> object:
    class _Models:
        calls = 0

        def generate_content(self, **kwargs: object) -> object:
            _Models.calls += 1
            capture.update(kwargs)
            return response

    class _Client:
        models = _Models()

        def close(self) -> None:
            return None

    return _Client()


def test_serialized_schema_excludes_unsupported_strategy_value_kinds() -> None:
    counterpoint = _request(TransformationStrategyType.COUNTERPOINT, "id-c", "key-c")
    summary = _request(TransformationStrategyType.SUMMARY, "id-s", "key-s")
    schema = build_planning_output_schema([counterpoint, summary]).model_json_schema()

    assert _schema_value_kinds(schema, "key-c") == {"COUNTERPOINT"}
    assert _schema_value_kinds(schema, "key-s") == {"SYNTHESIS", "USEFUL_TAKEAWAY"}
    serialized = json.dumps(schema, sort_keys=True)
    # The unsupported value kind is not admitted anywhere in the COUNTERPOINT
    # plan variant, even though it is valid for the mixed SUMMARY strategy.
    assert '"key-c"' in serialized
    assert '"COUNTERPOINT"' in serialized


def test_schema_accepts_valid_and_rejects_cross_strategy_combinations() -> None:
    counterpoint = _request(TransformationStrategyType.COUNTERPOINT, "id-c", "key-c")
    summary = _request(TransformationStrategyType.SUMMARY, "id-s", "key-s")
    model = build_planning_output_schema([counterpoint, summary])

    model.model_validate(
        {
            "plans": [
                _plan(counterpoint, "COUNTERPOINT"),
                _plan(summary, "USEFUL_TAKEAWAY"),
            ]
        }
    )

    with pytest.raises(ValidationError):
        model.model_validate({"plans": [_plan(counterpoint, "USEFUL_TAKEAWAY")]})
    with pytest.raises(ValidationError):
        model.model_validate({"plans": [_plan(summary, "COUNTERPOINT")]})


def test_google_sdk_conversion_preserves_per_strategy_enums() -> None:
    from google.genai import _transformers

    counterpoint = _request(TransformationStrategyType.COUNTERPOINT, "id-c", "key-c")
    summary = _request(TransformationStrategyType.SUMMARY, "id-s", "key-s")
    model = build_planning_output_schema([counterpoint, summary])

    converted = _transformers.t_schema(None, model)
    variants = converted.properties["plans"].items.any_of
    assert variants
    mapping: dict[str, set[str]] = {}
    for variant in variants:
        key = variant.properties["strategy_key"].enum[0]
        kind = variant.properties["blocks"].items.properties["substantive_value_kind"]
        mapping[key] = set(kind.enum or [])
        for option in kind.any_of or ():
            mapping[key].update(option.enum or [])
    assert mapping == {
        "key-c": {"COUNTERPOINT"},
        "key-s": {"SYNTHESIS", "USEFUL_TAKEAWAY"},
    }


def test_mixed_batch_uses_bound_schema_and_one_provider_call() -> None:
    counterpoint = _request(TransformationStrategyType.COUNTERPOINT, "id-c", "key-c")
    summary = _request(TransformationStrategyType.SUMMARY, "id-s", "key-s")
    response = _response(
        [
            _plan(counterpoint, "COUNTERPOINT"),
            _plan(summary, "USEFUL_TAKEAWAY"),
        ]
    )
    capture: dict[str, object] = {}
    client = _client(response, capture)
    provider = GeminiPlanningProvider(api_key="offline", client_factory=lambda: client)

    results = provider.plan([counterpoint, summary], "ROUTINE")
    provider.release()

    assert set(results) == {"key-c", "key-s"}
    assert client.models.calls == 1  # type: ignore[attr-defined]
    assert provider.raw_call_count() == 1
    assert capture["config"].response_schema.model_json_schema() == (  # type: ignore[attr-defined]
        build_planning_output_schema([counterpoint, summary]).model_json_schema()
    )
    for result in results.values():
        assert result.schema_rejections == ()
        assert result.response_evidence["output_schema_digest"] == (
            planning_output_schema_digest([counterpoint, summary])
        )


def test_sdk_schema_rejection_is_bounded_and_does_not_retry() -> None:
    counterpoint = _request(TransformationStrategyType.COUNTERPOINT, "id-c", "key-c")
    capture: dict[str, object] = {}
    client = _client(_response([_plan(counterpoint, "USEFUL_TAKEAWAY")]), capture)
    provider = GeminiPlanningProvider(api_key="offline", client_factory=lambda: client)

    results = provider.plan([counterpoint], "ROUTINE")
    provider.release()

    assert client.models.calls == 1  # type: ignore[attr-defined]
    assert provider.raw_call_count() == 1
    rejection = results["key-c"].schema_rejections[0]
    assert rejection["origin"] == "SDK_PARSING"
    assert rejection["code"] == "SCHEMA_VALUE_NOT_ALLOWED"
    assert rejection["strategy_id"] == "id-c"
    assert rejection["strategy_key"] == "key-c"
    assert rejection["block_index"] == 0
    assert rejection["field"] == "substantive_value_kind"
    assert rejection["actual_value"] == "USEFUL_TAKEAWAY"
    assert rejection["allowed_values"] == ["COUNTERPOINT"]
    # Bounded evidence, never the full raw response.
    serialized = json.dumps(rejection, sort_keys=True)
    assert "plans_requested" not in serialized
    assert len(serialized) < 2_000


def test_application_validation_reports_actual_and_allowed_values(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    seed = seed_stage41(
        session,
        settings=settings,
        strategy_type=TransformationStrategyType.COUNTERPOINT,
        value_kind=SubstantiveValueKind.COUNTERPOINT,
    )
    _source, candidate, refinement, analysis, _strategies = seed
    inputs = build_planning_inputs(
        session, candidate, analysis, refinement, settings, settings.stage41_config()
    )
    strategy = dict(inputs.stage40_strategies[0])
    plan = make_source_value_plan(
        str(strategy["id"]),
        str(strategy["strategy_key"]),
        kind=SubstantiveValueKind.USEFUL_TAKEAWAY,
    )

    result = validate_provider_plan(
        plan,
        strategy,
        inputs,
        settings.stage41_config(),
        provider_evidence={},
        provider_input_fingerprint="offline-fixture",
    )

    assert result.plan is None
    assert result.reasons == (REJECT_VALUE_KIND_MISMATCH,)
    detail = result.rejection_details[0]
    assert detail["origin"] == "APPLICATION_VALIDATION"
    assert detail["strategy_id"] == str(strategy["id"])
    assert detail["strategy_key"] == str(strategy["strategy_key"])
    assert detail["block_index"] == 1
    assert detail["field"] == "substantive_value_kind"
    assert detail["actual_value"] == "USEFUL_TAKEAWAY"
    assert detail["allowed_values"] == ["COUNTERPOINT"]


def test_service_diagnostics_keep_application_rejection_without_retry(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    seed = seed_stage41(
        session,
        settings=settings,
        strategy_type=TransformationStrategyType.COUNTERPOINT,
        value_kind=SubstantiveValueKind.COUNTERPOINT,
    )
    strategy = seed[4][0]
    invalid = make_source_value_plan(
        str(strategy.id),
        str(strategy.strategy_key),
        kind=SubstantiveValueKind.USEFUL_TAKEAWAY,
    )
    provider = FakePlanningProvider([invalid], hosted=False)

    plan_set = run_planning(session, settings, seed, provider)

    assert provider.calls == 1
    entries = dict(plan_set.provider_evidence or {}).get("attempt_diagnostics")
    assert isinstance(entries, list) and entries
    entry = entries[-1]
    assert entry["phase"] == "CONTRACT_REJECTION"
    assert "VALUE_KIND_INCONSISTENT_WITH_STRATEGY" in entry["failure_codes"]
    assert entry["rejection_origins"] == ["APPLICATION_VALIDATION"]
    detail = entry["validation_rejections"][0]
    assert detail["origin"] == "APPLICATION_VALIDATION"
    assert detail["actual_value"] == "USEFUL_TAKEAWAY"
    assert detail["allowed_values"] == ["COUNTERPOINT"]


def test_service_records_sdk_and_application_origins_without_retry(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    seed = seed_stage41(
        session,
        settings=settings,
        strategy_type=TransformationStrategyType.COUNTERPOINT,
        value_kind=SubstantiveValueKind.COUNTERPOINT,
    )

    class _Models:
        calls = 0

        def generate_content(self, **kwargs: object) -> object:
            _Models.calls += 1
            payload = json.loads(str(kwargs["contents"]))
            plans = [
                _plan_with_source(item["strategy_id"], item["strategy_key"], "USEFUL_TAKEAWAY")
                for item in payload["plans_requested"]
            ]
            return type(
                "_Response",
                (),
                {
                    "parsed": None,
                    "text": json.dumps({"plans": plans}, ensure_ascii=False),
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

    provider = GeminiPlanningProvider(api_key="offline", client_factory=lambda: _Client())
    plan_set = run_planning(session, settings, seed, provider)

    assert _Models.calls == 1
    assert provider.raw_call_count() == 1
    entries = dict(plan_set.provider_evidence or {}).get("attempt_diagnostics")
    assert isinstance(entries, list) and entries
    entry = entries[-1]
    assert entry["phase"] == "CONTRACT_REJECTION"
    assert entry["rejection_origins"] == ["SDK_PARSING", "APPLICATION_VALIDATION"]
    sdk = next(
        item for item in entry["schema_rejections"] if item.get("field") == "substantive_value_kind"
    )
    assert sdk["origin"] == "SDK_PARSING"
    assert sdk["actual_value"] == "USEFUL_TAKEAWAY"
    assert sdk["allowed_values"] == ["COUNTERPOINT"]
    application = next(
        item
        for item in entry["validation_rejections"]
        if item.get("field") == "substantive_value_kind"
    )
    assert application["origin"] == "APPLICATION_VALIDATION"
    assert application["actual_value"] == "USEFUL_TAKEAWAY"
    assert application["allowed_values"] == ["COUNTERPOINT"]


def test_schema_binding_preserves_scope_fingerprint_and_call_budget() -> None:
    counterpoint = _request(TransformationStrategyType.COUNTERPOINT, "id-c", "key-c")
    summary = _request(TransformationStrategyType.SUMMARY, "id-s", "key-s")

    assert strategy_value_kinds(TransformationStrategyType.COUNTERPOINT) == (
        SubstantiveValueKind.COUNTERPOINT,
    )
    assert planning_output_schema_digest([counterpoint]) != planning_output_schema_digest([summary])
    identity = {
        "prompt_hash": planning_prompt_hash(),
        "schema_version": SCHEMA_VERSION,
        "schema_binding_version": PLANNING_SCHEMA_BINDING_VERSION,
    }
    first = provider_input_fingerprint(
        build_provider_input_payload(
            request=counterpoint.to_payload(), provider_identity=identity, route="ROUTINE"
        )
    )
    second = provider_input_fingerprint(
        build_provider_input_payload(
            request=summary.to_payload(), provider_identity=identity, route="ROUTINE"
        )
    )
    assert first != second

    capture: dict[str, object] = {}
    client = _client(_response([_plan(counterpoint, "COUNTERPOINT")]), capture)
    provider = GeminiPlanningProvider(
        api_key="offline", client_factory=lambda: client, max_raw_calls=1
    )
    provider.plan([counterpoint], "ROUTINE")
    with pytest.raises(Exception) as error:
        provider.plan([counterpoint], "ROUTINE")
    assert "budget" in getattr(error.value, "detail", str(error.value))
    assert client.models.calls == 1  # type: ignore[attr-defined]
    assert provider.raw_call_count() == 1
