"""Offline semantic-claim contract regressions across Stages 4.1, 4.2 and 6.

Diagnostic inputs mirror the rejected production plan: the source discusses
forced humour, self-imposition and talking about oneself; it does not establish
that listening prevents self-absorption. These tests never call a provider.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage41_support import (
    FakePlanningProvider,
    FakeStage41Settings,
    install_stage41_settings,
    seed_stage41,
)
from stage42_support import (
    make_critique,
    make_plan,
    original_block,
    source_block,
)
from stage42_support import (
    make_inputs as make_governance_inputs,
)
from test_stage6_materialization_core import make_inputs as make_materialization_inputs

from app.core.enums import (
    GovernancePlanStatus,
    NarrationNeed,
    PlanBlockType,
    PlanSemanticOutcome,
    SemanticProviderMode,
    SourceExcerptRole,
    SubstantiveValueKind,
    TransformationStrategyType,
)
from app.materialization.inputs import FinalClipEvidence, WordEvidence
from app.materialization.policy import DEFAULT_CONFIG as STAGE6_CONFIG
from app.materialization.policy import CallLedger
from app.materialization.providers import (
    BlockCritique,
    CritiqueDimensionFinding,
    CritiqueResult,
    FakeCritiqueProvider,
    FakeWritingProvider,
    build_writing_request,
    materialize_script,
)
from app.transformation.governance.validation import (
    SEMANTIC_AVAILABLE,
    apply_semantic_review,
    evaluate_plan,
    finalize,
)
from app.transformation.planning.policy import DEFAULT_CONFIG as STAGE41_CONFIG
from app.transformation.planning.queue import get_or_create_plan_set, list_plans
from app.transformation.planning.types import (
    NarrationRequirement,
    PlanProviderBlock,
    PlanProviderPlan,
)

SOURCE_EXCERPT = (
    "إن شاء الله خير لو أنت عادة مع صحابتك وفيه واحد جديد داخل عليك وإيه أكتر حاجة "
    "ممكن تقفلك منه؟ الاستظراف أه أو يخش مثلا يفرد نفسه أو يكون تانك أو يعود يتكلم "
    "عن نفسه كتير"
)
FAILED_COPY = "الاستماع الحقيقي هو الحل المتبادل لتجنب التمركز حول الذات."
FAILED_INTENT = (
    "The viewer gains an actionable communication takeaway: balancing self-expression "
    "with active curiosity about the other person's experience prevents the perceived "
    "self-absorption described in the source."
)
POSITIVE_SUGGESTION = (
    "اقتراح للحوار: بعد ما تحكي موقفك، اسأل اللي قدامك عن تجربة شبهه وسيب له مساحة يرد."
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    from app.db.base import Base

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _arabic_words() -> tuple[WordEvidence, ...]:
    tokens = SOURCE_EXCERPT.split()
    step = (271.0 - 259.16) / max(1, len(tokens))
    return tuple(
        WordEvidence(
            index=index,
            text=token,
            start=259.16 + index * step,
            end=259.16 + (index + 1) * step,
        )
        for index, token in enumerate(tokens)
    )


def _stage6_inputs(
    *,
    intent: str = FAILED_INTENT,
    draft: str | None = FAILED_COPY,
    kind: str = SubstantiveValueKind.SOURCE_AS_EVIDENCE.value,
) -> Any:
    inputs = make_materialization_inputs()
    blocks = (
        {
            **inputs.blocks[0],
            "source_text": SOURCE_EXCERPT,
            "source_start": 259.16,
            "source_end": 271.0,
            "word_start_index": 0,
            "word_end_index": len(SOURCE_EXCERPT.split()) - 1,
        },
        {
            **inputs.blocks[1],
            "semantic_intent": intent,
            "draft_line": draft,
            "substantive_value_kind": kind,
            "grounding_refs": ["block:0"],
        },
    )
    final_clip = FinalClipEvidence(
        refinement_id=inputs.final_clip.refinement_id,
        priority=inputs.final_clip.priority,
        quality_level=inputs.final_clip.quality_level,
        status=inputs.final_clip.status,
        output_fingerprint=inputs.final_clip.output_fingerprint,
        text=SOURCE_EXCERPT,
        words=_arabic_words(),
        start=259.16,
        end=271.0,
    )
    return make_materialization_inputs(blocks=blocks, final_clip=final_clip)


def _materialize(inputs: Any, copy: str) -> Any:
    provider = FakeWritingProvider(copies={1: copy})
    script = materialize_script(
        provider,
        inputs,
        ledger=CallLedger(),
        config=STAGE6_CONFIG,
        allow_hosted=True,
        critique_provider=FakeCritiqueProvider(accepted={1: True}),
    )
    return provider, script


def _stage41_seed(session: Session) -> tuple[Any, ...]:
    settings = FakeStage41Settings()
    from _pytest.monkeypatch import MonkeyPatch

    monkeypatch = MonkeyPatch()
    install_stage41_settings(monkeypatch, settings)
    return cast(
        tuple[Any, ...],
        seed_stage41(
            session,
            settings=settings,
            transcript=SOURCE_EXCERPT,
            refined=(259.16, 271.0),
        ),
    )


def _failed_stage41_plan(seed: tuple[Any, ...]) -> PlanProviderPlan:
    strategy = seed[4][0]
    return PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        confidence=0.7,
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                use_full_window=True,
                source_role=SourceExcerptRole.HERO,
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.ORIGINAL_VALUE,
                estimated_duration=3.0,
                substantive_value_kind=SubstantiveValueKind.SOURCE_AS_EVIDENCE,
                semantic_intent=FAILED_INTENT,
                why_unavailable=(
                    "The source identifies self-focused behaviour but does not outline "
                    "a remedy or a psychological mechanism."
                ),
                grounding_refs=("block:0",),
                draft_line=FAILED_COPY,
            ),
        ),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )


# Stage 4.1 -----------------------------------------------------------------


def test_stage41_rejects_unsupported_remedy_in_intent_and_draft(session: Session) -> None:
    from app.transformation.planning.executor import TransformationPlanningExecutor

    seed = _stage41_seed(session)
    source, candidate, _refinement, analysis, _strategies = seed
    plan_set = get_or_create_plan_set(session, candidate, analysis)
    executor = TransformationPlanningExecutor(
        session=session,
        settings=FakeStage41Settings(),
        provider=FakePlanningProvider([_failed_stage41_plan(seed)]),
        provider_identity={"provider": "fake", "model": "fake-model"},
        mode=SemanticProviderMode.ADAPTIVE,
        config=STAGE41_CONFIG,
    )
    executor.execute(plan_set.id)
    session.refresh(plan_set)

    assert list_plans(session, plan_set.id) == []
    assert plan_set.planning_outcome is PlanSemanticOutcome.PLANNING_DEFERRED


def test_stage41_accepts_authored_source_linked_suggestion(session: Session) -> None:
    from app.transformation.planning.executor import TransformationPlanningExecutor

    settings = FakeStage41Settings()
    from _pytest.monkeypatch import MonkeyPatch

    monkeypatch = MonkeyPatch()
    install_stage41_settings(monkeypatch, settings)
    seed = seed_stage41(
        session,
        settings=settings,
        strategy_type=TransformationStrategyType.SUMMARY,
        value_kind=SubstantiveValueKind.USEFUL_TAKEAWAY,
        transcript=SOURCE_EXCERPT,
        refined=(259.16, 271.0),
    )
    strategy = seed[4][0]
    plan = PlanProviderPlan(
        strategy_id=str(strategy.id),
        strategy_key=strategy.strategy_key,
        confidence=0.7,
        blocks=(
            PlanProviderBlock(
                block_type=PlanBlockType.SOURCE_EXCERPT,
                use_full_window=True,
                source_role=SourceExcerptRole.HERO,
            ),
            PlanProviderBlock(
                block_type=PlanBlockType.ORIGINAL_VALUE,
                estimated_duration=4.0,
                substantive_value_kind=SubstantiveValueKind.USEFUL_TAKEAWAY,
                semantic_intent=(
                    "Offer an authored, source-linked conversational suggestion "
                    "that invites the other person's perspective."
                ),
                why_unavailable=(
                    "The source names self-focused behaviour but does not offer a "
                    "concrete conversational action."
                ),
                grounding_refs=("block:0",),
                draft_line=POSITIVE_SUGGESTION,
            ),
        ),
        narration=NarrationRequirement(need=NarrationNeed.NONE),
    )
    source, candidate, _refinement, analysis, _strategies = seed
    plan_set = get_or_create_plan_set(session, candidate, analysis)
    executor = TransformationPlanningExecutor(
        session=session,
        settings=settings,
        provider=FakePlanningProvider([plan]),
        provider_identity={"provider": "fake", "model": "fake-model"},
        mode=SemanticProviderMode.ADAPTIVE,
        config=STAGE41_CONFIG,
    )
    executor.execute(plan_set.id)
    session.refresh(plan_set)

    rows = list_plans(session, plan_set.id)
    assert len(rows) == 1
    assert rows[0].blocks[1]["substantive_value_kind"] == "USEFUL_TAKEAWAY"
    assert rows[0].blocks[1]["draft_only"] is True


# Stage 4.2 -----------------------------------------------------------------


def _stage42_failed_plan() -> Any:
    block = {
        **original_block(1, intent=FAILED_INTENT, kind="SOURCE_AS_EVIDENCE"),
        "draft_line": FAILED_COPY,
    }
    return make_plan(
        blocks=[source_block(0, text=SOURCE_EXCERPT), block],
        original_value_kinds=("SOURCE_AS_EVIDENCE",),
    )


def test_stage42_rejects_remedy_even_with_favorable_critique() -> None:
    plan = _stage42_failed_plan()
    inputs = make_governance_inputs([plan], transcript=SOURCE_EXCERPT)
    evaluation = evaluate_plan(
        plan,
        inputs,
        _governance_config(),
        provider_mode_deterministic=False,
    )
    apply_semantic_review(evaluation, make_critique(plan), SEMANTIC_AVAILABLE)

    status = finalize(evaluation)[0]
    assert status is GovernancePlanStatus.REJECTED_BY_GOVERNOR


def test_stage42_accepts_authored_suggestion_after_favorable_review() -> None:
    block = {
        **original_block(
            1,
            intent=(
                "Offer an authored, source-linked conversational suggestion that "
                "invites the other person's perspective."
            ),
            kind="USEFUL_TAKEAWAY",
        ),
        "draft_line": POSITIVE_SUGGESTION,
    }
    plan = make_plan(
        blocks=[source_block(0, text=SOURCE_EXCERPT), block],
        original_value_kinds=("USEFUL_TAKEAWAY",),
    )
    inputs = make_governance_inputs([plan], transcript=SOURCE_EXCERPT)
    evaluation = evaluate_plan(
        plan,
        inputs,
        _governance_config(),
        provider_mode_deterministic=False,
    )
    apply_semantic_review(evaluation, make_critique(plan), SEMANTIC_AVAILABLE)

    assert finalize(evaluation)[0] in {
        GovernancePlanStatus.APPROVED_FOR_SELECTION,
        GovernancePlanStatus.APPROVED_WITH_CAUTION,
    }


def _governance_config() -> Any:
    from app.transformation.governance.policy import DEFAULT_CONFIG

    return DEFAULT_CONFIG


# Stage 6 -------------------------------------------------------------------


def test_stage6_rejects_exact_failed_copy_even_with_favorable_critique() -> None:
    inputs = _stage6_inputs(intent="Offer a practical takeaway.")
    _provider, script = _materialize(inputs, FAILED_COPY)

    assert script.authored_blocks == ()
    assert any("UNSUPPORTED" in code for code in script.reason_codes)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "copy",
    [
        "اقتراح: " + FAILED_COPY,
        "Try this: active listening prevents self-absorption.",
        "جرّب: " + FAILED_COPY,
    ],
)
def test_stage6_suggestion_prefix_does_not_bypass_unsupported_claim(copy: str) -> None:
    _provider, script = _materialize(
        _stage6_inputs(intent="Offer a practical takeaway."),
        copy,
    )

    assert script.authored_blocks == ()


def test_stage6_rejects_invented_motive() -> None:
    copy = "هو يتكلم عن نفسه كتير لأنه يحتاج تقدير ويحاول يثبت نفسه."
    _provider, script = _materialize(_stage6_inputs(intent="Explain the behaviour."), copy)

    assert script.authored_blocks == ()


def test_stage6_rejects_generic_irrelevant_advice() -> None:
    _provider, script = _materialize(
        _stage6_inputs(intent="Offer a practical takeaway.", kind="USEFUL_TAKEAWAY"),
        "ابتسم دائما وكن واثقا في نفسك.",
    )

    assert script.authored_blocks == ()


def test_stage6_rejects_concealed_external_fact_in_source_as_evidence() -> None:
    copy = "اقتراح: جرب الاستماع لمدة 7 دقائق."
    _provider, script = _materialize(
        _stage6_inputs(intent="Offer a practical takeaway."),
        copy,
    )

    assert script.authored_blocks == ()
    assert any("UNSUPPORTED_ESSENTIAL_CLAIM" in code for code in script.reason_codes)


def test_stage6_rejects_unverified_proven_effectiveness() -> None:
    copy = "الاستماع الحقيقي حل فعال ومضمون لكل مشاكل التواصل."
    _provider, script = _materialize(_stage6_inputs(intent="Offer a practical takeaway."), copy)

    assert script.authored_blocks == ()


def test_stage6_rejects_source_paraphrase_as_original_value() -> None:
    inputs = _stage6_inputs(intent="Offer a practical takeaway.")
    _provider, script = _materialize(inputs, SOURCE_EXCERPT)

    assert script.authored_blocks == ()


def test_stage6_reports_upstream_revalidation_without_calling_writer() -> None:
    provider, script = _materialize(_stage6_inputs(), FAILED_COPY)

    assert script.deferred
    assert "UPSTREAM_REVALIDATION_REQUIRED" in script.reason_codes
    assert provider.calls == []


def test_stage6_accepts_structured_source_linked_suggestion_after_full_critique() -> None:
    inputs = _stage6_inputs(
        intent="Offer an authored, source-linked conversational suggestion.",
        draft=None,
        kind="USEFUL_TAKEAWAY",
    )
    request = build_writing_request(inputs, config=STAGE6_CONFIG)
    assert request.targets[0].claim_class == "AUTHORED_SOURCE_LINKED_SUGGESTION"
    assert request.targets[0].draft_only is False

    _provider, script = _materialize(inputs, POSITIVE_SUGGESTION)

    assert script.accepted
    assert script.authored_blocks[0].copy == POSITIVE_SUGGESTION
    assert script.authored_blocks[0].claim_class == "AUTHORED_SOURCE_LINKED_SUGGESTION"


def test_writing_request_marks_draft_as_untrusted_material() -> None:
    inputs = _stage6_inputs(
        intent="Offer an authored, source-linked conversational suggestion.",
        draft=FAILED_COPY,
        kind="USEFUL_TAKEAWAY",
    )
    target = build_writing_request(inputs, config=STAGE6_CONFIG).targets[0]

    assert target.draft_line == FAILED_COPY
    assert target.draft_only is True
    assert target.claim_class == "AUTHORED_SOURCE_LINKED_SUGGESTION"


def test_governance_request_carries_draft_and_rendered_evidence_separately() -> None:
    from app.transformation.governance.policy import DEFAULT_CONFIG as GOVERNANCE_CONFIG
    from app.transformation.governance.providers import build_governance_request

    plan = _stage42_failed_plan()
    request = build_governance_request(
        make_governance_inputs([plan], transcript=SOURCE_EXCERPT),
        GOVERNANCE_CONFIG,
    )
    payload = request.to_payload()
    rendered = payload["plans"][0]["selected_evidence"]
    assert rendered == [SOURCE_EXCERPT]
    assert payload["plans"][0]["blocks"][1]["draft_line"] == FAILED_COPY
    assert payload["plans"][0]["required_context_is_not_rendered_evidence"] is True


def test_shared_semantic_contract_is_in_every_stage_instruction() -> None:
    from app.core.semantic_contract import SEMANTIC_CLAIM_CONTRACT
    from app.materialization.gemini import CRITIQUE_SYSTEM_INSTRUCTION, WRITING_SYSTEM_INSTRUCTION
    from app.transformation.governance.providers import GOVERNANCE_SYSTEM_INSTRUCTION
    from app.transformation.planning.providers import PLANNING_SYSTEM_INSTRUCTION

    for instruction in (
        PLANNING_SYSTEM_INSTRUCTION,
        GOVERNANCE_SYSTEM_INSTRUCTION,
        WRITING_SYSTEM_INSTRUCTION,
        CRITIQUE_SYSTEM_INSTRUCTION,
    ):
        assert SEMANTIC_CLAIM_CONTRACT in instruction
        assert "draft_line" in instruction or instruction is CRITIQUE_SYSTEM_INSTRUCTION
    assert "AUTHORED SOURCE-LINKED SUGGESTION" in SEMANTIC_CLAIM_CONTRACT


def test_stage6_unfavorable_complete_critique_still_rejects() -> None:
    inputs = _stage6_inputs(
        intent="Offer an authored, source-linked conversational suggestion.",
        draft=None,
        kind="USEFUL_TAKEAWAY",
    )
    provider = FakeWritingProvider(copies={1: POSITIVE_SUGGESTION})
    script = materialize_script(
        provider,
        inputs,
        ledger=CallLedger(),
        config=STAGE6_CONFIG,
        allow_hosted=True,
        critique_provider=FakeCritiqueProvider(accepted={1: False}),
    )

    assert script.authored_blocks == ()
    assert not script.accepted


def test_stage6_rejects_concrete_but_irrelevant_suggestion() -> None:
    from app.materialization.providers import CRITIQUE_DIMENSIONS

    class IrrelevantIntentProvider:
        provider_name = "irrelevant-intent"

        def available(self) -> bool:
            return True

        def runtime_identity(self) -> dict[str, object]:
            return {"provider": "irrelevant-intent", "model": "fake", "api_version": "v1"}

        def critique(self, request: Any) -> CritiqueResult:
            target = request.targets[0]
            findings = tuple(
                CritiqueDimensionFinding(
                    dimension=dimension,
                    level="FAIL" if dimension == "INTENT_FULFILLMENT" else "PASS",
                )
                for dimension in CRITIQUE_DIMENSIONS
            )
            return CritiqueResult(
                accepted={target.block_index: False},
                provider_identity=self.runtime_identity(),
                blocks={
                    target.block_index: BlockCritique(
                        block_index=target.block_index,
                        findings=findings,
                        substantive=True,
                    )
                },
            )

        def release(self) -> None:
            return None

    inputs = _stage6_inputs(
        intent="Offer an authored, source-linked conversational suggestion.",
        draft=None,
        kind="USEFUL_TAKEAWAY",
    )
    provider = FakeWritingProvider(copies={1: "اقتراح: اشتري هدية للشخص الجديد."})
    script = materialize_script(
        provider,
        inputs,
        ledger=CallLedger(),
        config=STAGE6_CONFIG,
        allow_hosted=True,
        critique_provider=IrrelevantIntentProvider(),
    )

    assert script.authored_blocks == ()
    assert any("CRITIQUE_INTENT_FULFILLMENT_FAIL" in code for code in script.reason_codes)


def test_stage6_incomplete_critique_defers_without_accepting() -> None:
    class IncompleteCritiqueProvider:
        provider_name = "incomplete-critique"

        def available(self) -> bool:
            return True

        def runtime_identity(self) -> dict[str, object]:
            return {"provider": "incomplete-critique", "model": "fake", "api_version": "v1"}

        def critique(self, request: Any) -> CritiqueResult:
            target = request.targets[0]
            return CritiqueResult(
                accepted={target.block_index: True},
                provider_identity=self.runtime_identity(),
                blocks={
                    target.block_index: BlockCritique(
                        block_index=target.block_index,
                        findings=(
                            CritiqueDimensionFinding(
                                dimension="GROUNDING_ENTAILMENT",
                                level="PASS",
                                evidence="only one dimension",
                            ),
                        ),
                    )
                },
            )

        def release(self) -> None:
            return None

    inputs = _stage6_inputs(
        intent="Offer an authored, source-linked conversational suggestion.",
        draft=None,
        kind="USEFUL_TAKEAWAY",
    )
    provider = FakeWritingProvider(copies={1: POSITIVE_SUGGESTION})
    script = materialize_script(
        provider,
        inputs,
        ledger=CallLedger(),
        config=STAGE6_CONFIG,
        allow_hosted=True,
        critique_provider=IncompleteCritiqueProvider(),
    )

    assert script.authored_blocks == ()
    assert script.deferred
