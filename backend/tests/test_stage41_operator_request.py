"""Operator planning-request scope and editorial-intent contract tests.

These tests are hermetic: every provider is a local fake, no Gemini/TTS/Qwen
client is constructed, and no hosted endpoint is contacted.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from stage41_support import (
    FakePlanningProvider,
    FakeStage41Settings,
    install_stage41_settings,
    seed_stage41,
)
from stage42_support import FakeGovernanceSettings
from stage43_support import install_selection_settings, make_result_spec
from stage50_support import FakeProber, seed_stage50
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.core.enums import (
    PlanBlockType,
    SemanticProviderMode,
    SourceExcerptRole,
    SubstantiveValueKind,
)
from app.db.base import Base
from app.models import ProcessingJob, TransformationPlanSet
from app.transformation.governance.inputs import build_governance_inputs
from app.transformation.governance.policy import DEFAULT_CONFIG as STAGE42_CONFIG
from app.transformation.governance.providers import build_governance_request
from app.transformation.planning.executor import TransformationPlanningExecutor
from app.transformation.planning.inputs import (
    PlanningInputError,
    build_planning_inputs,
)
from app.transformation.planning.policy import DEFAULT_CONFIG
from app.transformation.planning.queue import (
    list_plans,
    queue_transformation_planning,
)
from app.transformation.planning.request import (
    OperatorPlanningRequest,
    OperatorRequestError,
    parse_operator_request,
)
from app.transformation.planning.types import (
    PlanProviderBlock,
    PlanProviderPlan,
    WordEvidence,
)
from app.transformation.planning.validation import REJECT_INVALID_SPAN, resolve_source_span


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def _no_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.workers.tasks.run_transformation_planning.delay", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.workers.tasks.run_transformation_governance.delay", lambda *a, **k: None
    )


class CapturingPlanningProvider(FakePlanningProvider):
    """Fake provider that records the exact serialized requests it receives."""

    def __init__(self, plans: list[PlanProviderPlan]) -> None:
        super().__init__(plans)
        self.requests: list[dict[str, object]] = []

    def plan(self, requests: Any, tier: str = "ROUTINE") -> Any:
        self.requests.extend(request.to_payload() for request in requests)
        return super().plan(requests, tier)


def _factory(tmp_path: Any) -> sessionmaker[Session]:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'operator-request.sqlite3'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _scoped_plan(strategy: Any) -> PlanProviderPlan:
    return PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        confidence=0.8,
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                purpose="Hero source moment",
                use_full_window=False,
                word_start_index=0,
                word_end_index=1,
                source_role=SourceExcerptRole.HERO,
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.ORIGINAL_VALUE,
                purpose="Practical reciprocal move",
                estimated_duration=4.0,
                substantive_value_kind=SubstantiveValueKind.SOURCE_AS_EVIDENCE,
                semantic_intent=("Turn the complaint into a practical reciprocal listening move"),
                why_unavailable=(
                    "The excerpt states the complaint but not the reciprocal practice"
                ),
                grounding_refs=("block:0",),
            ),
        ),
    )


def _execute(
    factory: sessionmaker[Session],
    settings: FakeStage41Settings,
    plan_set_id: uuid.UUID,
    job_id: uuid.UUID,
    provider: CapturingPlanningProvider,
) -> None:
    with factory() as session:
        executor = TransformationPlanningExecutor(
            session=session,
            settings=settings,
            provider=provider,
            provider_identity=settings.transformation_planning_provider_identity(),
            mode=SemanticProviderMode.ADAPTIVE,
            config=DEFAULT_CONFIG,
            lease_factory=settings.heavy_model_lease_factory(),  # type: ignore[arg-type]
        )
        executor.set_active_job(job_id)
        executor.execute(plan_set_id)


def _queue(
    factory: sessionmaker[Session],
    settings: FakeStage41Settings,
    candidate_id: uuid.UUID,
    **kwargs: Any,
) -> Any:
    options: dict[str, object] = {
        "operator_source_start": 21.0,
        "operator_source_end": 24.0,
        "editorial_intent": (
            "Make the conversation reciprocal by inviting the other person's experience."
        ),
    }
    options.update(kwargs)
    with factory() as session:
        from app.transformation.planning.queue import validate_candidate_for_planning

        candidate, analysis = validate_candidate_for_planning(session, candidate_id)
        outcome = queue_transformation_planning(session, candidate, analysis, **options)
        session.commit()
        return outcome


def test_cli_queue_executor_preserves_request_across_worker_boundary(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _factory(tmp_path)
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    monkeypatch.setattr("app.cli.create_session_factory", lambda: factory)
    monkeypatch.setattr("app.workers.tasks.run_transformation_planning.delay", lambda *a, **k: None)
    with factory() as session:
        seed = seed_stage41(session, settings=settings)
        candidate_id = seed[1].id
        strategy = seed[4][0]
        session.commit()

    result = CliRunner().invoke(
        cli_app,
        [
            "transformation-plan-generate",
            str(candidate_id),
            "--source-start",
            "21.0",
            "--source-end",
            "24.0",
            "--editorial-intent",
            "Make the conversation reciprocal by inviting the other person's experience.",
        ],
    )
    assert result.exit_code == 0, result.output
    queued = json.loads(result.stdout)
    assert queued["queued"] is True

    with factory() as session:
        plan_set = session.get(TransformationPlanSet, uuid.UUID(queued["plan_set_id"]))
        assert plan_set is not None
        assert plan_set.operator_request == {
            "version": "stage4.1-operator-request-v1",
            "source_start": 21.0,
            "source_end": 24.0,
            "editorial_intent": (
                "Make the conversation reciprocal by inviting the other person's experience."
            ),
        }

    provider = CapturingPlanningProvider([_scoped_plan(strategy)])
    _execute(
        factory,
        settings,
        uuid.UUID(queued["plan_set_id"]),
        uuid.UUID(queued["job_id"]),
        provider,
    )

    assert provider.requests
    payload = provider.requests[0]
    assert payload["operator_direction"] == {
        "role": "OPERATOR_DIRECTION_NOT_SOURCE_EVIDENCE",
        "version": "stage4.1-operator-request-v1",
        "approved_source_start": 21.0,
        "approved_source_end": 24.0,
        "editorial_intent": (
            "Make the conversation reciprocal by inviting the other person's experience."
        ),
        "constraints": [
            "Every SOURCE_EXCERPT must be fully inside the approved source window.",
            "Do not treat operator direction as source evidence, approved copy, "
            "or a guaranteed strategy.",
            "Do not invent psychology or unsupported factual claims.",
        ],
    }
    assert payload["refined_start"] == 21.0
    assert payload["refined_end"] == 24.0
    assert payload["source_selection"]["approved_source_start"] == 21.0
    assert payload["source_selection"]["approved_source_end"] == 24.0
    assert payload["source_selection"]["operator_window_is_authoritative"] is True
    assert all(word["start"] >= 21.0 and word["end"] <= 24.0 for word in payload["words"])

    with factory() as session:
        plan_set = session.get(TransformationPlanSet, uuid.UUID(queued["plan_set_id"]))
        assert plan_set is not None
        assert plan_set.execution_status.value == "COMPLETE"
        rows = list_plans(session, plan_set.id)
        assert rows
        hero = rows[0].blocks[0]
        assert hero["source_start"] >= 21.0
        assert hero["source_end"] <= 24.0


def test_out_of_window_fragment_is_rejected_even_if_presented(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _factory(tmp_path)
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    with factory() as session:
        seed = seed_stage41(
            session,
            settings=settings,
            refined=(259.16, 271.0),
            words=[
                {"text": "fragment", "start": 258.36, "end": 260.06, "probability": 0.9},
                {"text": "inside", "start": 259.16, "end": 259.9, "probability": 0.9},
                {"text": "later", "start": 270.5, "end": 271.0, "probability": 0.9},
            ],
        )
        _source, candidate, refinement, analysis, _strategies = seed
        request = OperatorPlanningRequest(
            source_start=259.16,
            source_end=271.0,
            editorial_intent="Keep the approved exchange.",
        )
        inputs = build_planning_inputs(
            session, candidate, analysis, refinement, settings, DEFAULT_CONFIG, request
        )
        assert all(word.start >= 259.16 for word in inputs.words)
        planted = replace(
            inputs,
            words=(WordEvidence(0, "fragment", 258.36, 260.06),),
            word_coverage_sufficient=True,
        )
        span, error = resolve_source_span(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                use_full_window=False,
                word_start_index=0,
                word_end_index=0,
                source_role=SourceExcerptRole.HERO,
            ),
            planted,
            DEFAULT_CONFIG,
        )
        assert span is None
        assert error == REJECT_INVALID_SPAN


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("start", "end", "expected"),
    [
        (20.9, 22.0, REJECT_INVALID_SPAN),
        (21.0, 23.0, None),
        (23.0, 24.1, REJECT_INVALID_SPAN),
        (20.9999, 23.0, REJECT_INVALID_SPAN),
        (23.0, 24.0001, REJECT_INVALID_SPAN),
    ],
)
def test_operator_window_rejects_both_edges_without_tolerance(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    start: float,
    end: float,
    expected: str | None,
) -> None:
    factory = _factory(tmp_path)
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    with factory() as session:
        seed = seed_stage41(session, settings=settings)
        _source, candidate, refinement, analysis, _strategies = seed
        request = OperatorPlanningRequest(
            source_start=21.0, source_end=24.0, editorial_intent="Keep the reciprocal move."
        )
        inputs = build_planning_inputs(
            session, candidate, analysis, refinement, settings, DEFAULT_CONFIG, request
        )
        planted = replace(
            inputs,
            words=(WordEvidence(0, "edge", start, end),),
            word_coverage_sufficient=True,
        )
        span, error = resolve_source_span(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                use_full_window=False,
                word_start_index=0,
                word_end_index=0,
                source_role=SourceExcerptRole.HERO,
            ),
            planted,
            DEFAULT_CONFIG,
        )
        if expected is None:
            assert span is not None and error is None
        else:
            assert span is None
            assert error == expected


def test_request_excluding_mandatory_hook_is_rejected_before_provider(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _factory(tmp_path)
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    with factory() as session:
        seed = seed_stage41(
            session,
            settings=settings,
            hook_index=1,
            source_segments=[
                {"start": 0.0, "end": 20.0, "text": "Intro"},
                {"start": 20.0, "end": 24.0, "text": "Mandatory hook outside the approved window"},
                {"start": 24.0, "end": 90.0, "text": "Outro"},
            ],
        )
        candidate_id = seed[1].id
        session.commit()

    from app.transformation.planning.queue import validate_candidate_for_planning

    with factory() as session:
        candidate, analysis = validate_candidate_for_planning(session, candidate_id)
        outcome = queue_transformation_planning(
            session,
            candidate,
            analysis,
            operator_source_start=21.0,
            operator_source_end=24.0,
            editorial_intent="Keep only the approved exchange.",
        )
        session.commit()
        job_id = outcome.job_id
        plan_set_id = outcome.plan_set_id

    provider = CapturingPlanningProvider([])
    with pytest.raises(PlanningInputError) as raised:
        _execute(factory, settings, plan_set_id, job_id, provider)
    assert "HOOK_OUTSIDE_APPROVED_WINDOW" in str(raised.value)
    assert provider.calls == 0

    with factory() as session:
        plan_set = session.get(TransformationPlanSet, plan_set_id)
        assert plan_set is not None
        assert plan_set.execution_status.value == "FAILED"
        job = session.get(ProcessingJob, job_id)
        assert job is not None
        assert job.status.value == "FAILED"


def test_changed_scope_or_intent_invalidates_cache_and_checkpoints(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _factory(tmp_path)
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    with factory() as session:
        seed = seed_stage41(session, settings=settings)
        candidate_id = seed[1].id
        strategy = seed[4][0]
        session.commit()

    first = _queue(factory, settings, candidate_id)
    _execute(
        factory,
        settings,
        first.plan_set_id,
        first.job_id,
        CapturingPlanningProvider([_scoped_plan(strategy)]),
    )

    with factory() as session:
        plan_set = session.get(TransformationPlanSet, first.plan_set_id)
        assert plan_set is not None
        first_fingerprint = plan_set.input_fingerprint
        assert plan_set.cache_eligible is True

    changed_intent = _queue(
        factory,
        settings,
        candidate_id,
        editorial_intent="Use a different reciprocal practice.",
    )
    assert changed_intent.cached is False
    assert changed_intent.queued is True
    provider = CapturingPlanningProvider([_scoped_plan(strategy)])
    _execute(
        factory,
        settings,
        changed_intent.plan_set_id,
        changed_intent.job_id,
        provider,
    )
    assert provider.calls > 0

    with factory() as session:
        plan_set = session.get(TransformationPlanSet, changed_intent.plan_set_id)
        assert plan_set is not None
        assert plan_set.input_fingerprint != first_fingerprint

    changed_scope = _queue(
        factory,
        settings,
        candidate_id,
        operator_source_start=21.0,
        operator_source_end=23.0,
    )
    assert changed_scope.cached is False
    assert changed_scope.queued is True

    identical = _queue(
        factory,
        settings,
        candidate_id,
        operator_source_start=21.0,
        operator_source_end=23.0,
    )
    assert identical.cached is True or identical.queued is False


def test_unscoped_legacy_caller_keeps_empty_request_and_old_payload(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _factory(tmp_path)
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    with factory() as session:
        seed = seed_stage41(session, settings=settings)
        candidate_id = seed[1].id
        strategy = seed[4][0]
        session.commit()

    with factory() as session:
        from app.transformation.planning.queue import validate_candidate_for_planning

        candidate, analysis = validate_candidate_for_planning(session, candidate_id)
        outcome = queue_transformation_planning(session, candidate, analysis)
        session.commit()
        plan_set_id, job_id = outcome.plan_set_id, outcome.job_id

    provider = CapturingPlanningProvider(
        [
            PlanProviderPlan(
                strategy_id=str(strategy.id),
                strategy_key=str(strategy.strategy_key),
                confidence=0.8,
                blocks=(
                    PlanProviderBlock(
                        block_type=PlanBlockType.SOURCE_EXCERPT,
                        purpose="Hero source moment",
                        use_full_window=True,
                        source_role=SourceExcerptRole.HERO,
                    ),
                    PlanProviderBlock(
                        block_type=PlanBlockType.ORIGINAL_VALUE,
                        purpose="Substantive value",
                        estimated_duration=4.0,
                        substantive_value_kind=SubstantiveValueKind.SOURCE_AS_EVIDENCE,
                        semantic_intent="Frame the source as evidence for the debate",
                        why_unavailable="The excerpt states the claim but not its relevance",
                        grounding_refs=("block:0",),
                    ),
                ),
            )
        ]
    )
    _execute(factory, settings, plan_set_id, job_id, provider)
    assert provider.requests
    payload = provider.requests[0]
    assert "operator_direction" not in payload
    assert payload["refined_start"] == 20.5
    assert payload["refined_end"] == 44.5
    with factory() as session:
        plan_set = session.get(TransformationPlanSet, plan_set_id)
        assert plan_set is not None
        assert plan_set.operator_request == {}


def test_governance_recovers_same_scoped_constraints(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _factory(tmp_path)
    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    with factory() as session:
        seed = seed_stage41(session, settings=settings)
        candidate_id = seed[1].id
        strategy = seed[4][0]
        session.commit()

    outcome = _queue(factory, settings, candidate_id)
    _execute(
        factory,
        settings,
        outcome.plan_set_id,
        outcome.job_id,
        CapturingPlanningProvider([_scoped_plan(strategy)]),
    )

    with factory() as session:
        candidate = session.get(type(seed[1]), candidate_id)
        plan_set = session.get(TransformationPlanSet, outcome.plan_set_id)
        assert candidate is not None and plan_set is not None
        governance = build_governance_inputs(
            session,
            candidate,
            plan_set,
            settings,  # type: ignore[arg-type]
            STAGE42_CONFIG,
            DEFAULT_CONFIG,
        )
        assert governance.approved_source_start == 21.0
        assert governance.approved_source_end == 24.0
        assert governance.editorial_intent == (
            "Make the conversation reciprocal by inviting the other person's experience."
        )
        request = build_governance_request(governance, STAGE42_CONFIG)
        payload = request.to_payload()
        assert payload["operator_direction"]["approved_source_start"] == 21.0
        assert payload["operator_direction"]["approved_source_end"] == 24.0
        assert payload["operator_direction"]["editorial_intent"] == (
            "Make the conversation reciprocal by inviting the other person's experience."
        )


def test_scoped_plan_reaches_selection_and_contract_without_word_index_drift(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'operator-downstream.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr("app.render.service.FFprobe", lambda **kwargs: FakeProber())
    monkeypatch.setattr("app.cli.create_session_factory", lambda: factory)

    request = OperatorPlanningRequest(
        source_start=21.0,
        source_end=24.0,
        editorial_intent=(
            "Make the conversation reciprocal by inviting the other person's experience."
        ),
    )

    def scoped_plan_factory(strategy_id: str, strategy_key: str) -> PlanProviderPlan:
        return _scoped_plan(
            type(
                "Strategy",
                (),
                {"id": strategy_id, "strategy_key": strategy_key},
            )()
        )

    with factory() as session:
        fixture = seed_stage50(
            session,
            settings=settings,
            result_specs=[make_result_spec()],
            operator_request=request,
            first_provider_plan_factory=scoped_plan_factory,
        )
        candidate_id = fixture.selection.candidate.id
        session.commit()

    result = CliRunner().invoke(cli_app, ["render-contract", str(candidate_id)])
    assert result.exit_code == 0, result.output
    contract = json.loads(result.stdout)
    assert contract["contract_ready"] is True
    assert contract["live_freshness"] == "CURRENT"
    payload = contract["contract_payload"]
    source_blocks = [
        block for block in payload["blocks"] if block["block_type"] == "SOURCE_EXCERPT"
    ]
    assert source_blocks
    binding = source_blocks[0]["source_binding"]
    assert binding["planning_source_start"] >= 21.0
    assert binding["planning_source_end"] <= 24.0
    words = payload["caption_input"]["word_timestamps"]
    start_index = binding["final_clip_word_start"]
    end_index = binding["final_clip_word_end"]
    assert start_index is not None and end_index is not None
    assert 0 <= start_index <= end_index < len(words)
    bound_text = " ".join(str(word["text"]) for word in words[start_index : end_index + 1])
    assert bound_text == binding["planning_text"]


def test_operator_request_validation_rejects_invalid_bounds() -> None:
    with pytest.raises(OperatorRequestError):
        parse_operator_request(source_start=float("nan"), source_end=24.0, editorial_intent="x")
    with pytest.raises(OperatorRequestError):
        parse_operator_request(source_start=24.0, source_end=21.0, editorial_intent="x")
    with pytest.raises(OperatorRequestError):
        parse_operator_request(source_start=-1.0, source_end=24.0, editorial_intent="x")
    with pytest.raises(OperatorRequestError):
        parse_operator_request(source_start=21.0, source_end=None, editorial_intent="x")
    with pytest.raises(OperatorRequestError):
        parse_operator_request(source_start=21.0, source_end=24.0, editorial_intent="")
