"""Stage 4.3 deterministic selection: eligibility, tiers, arbitration, freshness."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from stage43_support import (
    FakeGovernanceSettings,
    gate,
    install_selection_settings,
    make_result_spec,
    platform_risk,
    seed_selection_fixture,
    warning,
)

from app.core.enums import (
    GovernanceExecutionStatus,
    GovernancePlanStatus,
    RefinementPriority,
    RefinementStatus,
    TransformationSelectionStatus,
)
from app.db.base import Base
from app.models import CandidateRefinement, ProcessingJob, TransformationPlan
from app.transformation.selection.service import (
    get_current_selection,
    read_selection,
    select_transformation_plan,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _clean(**overrides: Any) -> dict[str, object]:
    return make_result_spec(
        **{"status": GovernancePlanStatus.APPROVED_FOR_SELECTION.value, **overrides}
    )


def _caution(**overrides: Any) -> dict[str, object]:
    return make_result_spec(
        **{"status": GovernancePlanStatus.APPROVED_WITH_CAUTION.value, **overrides}
    )


def _select(session: Session, candidate_id: Any) -> Any:
    view = select_transformation_plan(session, candidate_id)
    assert view is not None
    session.commit()
    return view


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_one_clean_approved_plan_is_selected(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.PLAN_SELECTED
    assert view.row.selected_plan_id == fixture.plans[0].id
    assert view.row.selected_with_caution is False
    assert view.row.selection_reason_codes == ["PLAN_SELECTED_CLEAN"]


def test_clean_approved_plus_caution_prefers_clean(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            _caution(
                dimensions={"source_dominance": "MODERATE"},
                warnings=[warning("SOURCE_DOMINANCE_CONCERN")],
            ),
            _clean(),
        ],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.PLAN_SELECTED
    assert view.row.selected_plan_id == fixture.plans[1].id
    assert view.row.arbitration_evidence["approval_tier"] == "CLEAN"


def test_allowlisted_moderate_caution_selected_with_warnings_preserved(
    session: Session, monkeypatch: Any
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            _caution(
                dimensions={"source_dominance": "MODERATE"},
                warnings=[warning("SOURCE_DOMINANCE_CONCERN")],
            )
        ],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.PLAN_SELECTED_WITH_CAUTION
    assert view.row.selected_with_caution is True
    assert view.row.selected_plan_id == fixture.plans[0].id
    snapshot = view.row.selected_governance_snapshot
    assert snapshot["warnings"] == [warning("SOURCE_DOMINANCE_CONCERN")]


def test_non_allowlisted_caution_only_is_not_selected(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            _caution(
                dimensions={"semantic_fidelity": "WEAK"},
                warnings=[warning("SEMANTIC_FIDELITY_CONCERN")],
            )
        ],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.NO_SELECTABLE_PLAN
    assert view.row.selected_plan_id is None


def test_approved_plus_verification_blocked_selects_approved(
    session: Session, monkeypatch: Any
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            make_result_spec(status=GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION.value),
            _clean(),
        ],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.PLAN_SELECTED
    assert view.row.selected_plan_id == fixture.plans[1].id


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "status",
    [
        GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION.value,
        GovernancePlanStatus.REVISION_REQUIRED.value,
        GovernancePlanStatus.REJECTED_BY_GOVERNOR.value,
    ],
)
def test_terminal_non_selectable_only_yields_no_selection(
    session: Session, monkeypatch: Any, status: str
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session, settings=settings, result_specs=[make_result_spec(status=status)]
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.NO_SELECTABLE_PLAN
    assert view.row.selected_plan_id is None


def test_governance_deferred_only_yields_deferred_selection(
    session: Session, monkeypatch: Any
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[make_result_spec(status=GovernancePlanStatus.GOVERNANCE_DEFERRED.value)],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.SELECTION_DEFERRED
    assert view.row.selection_reason_codes == ["SEMANTIC_GOVERNANCE_UNFINISHED"]


def test_mixed_terminal_failures_yield_no_selectable_plan(
    session: Session, monkeypatch: Any
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            make_result_spec(status=GovernancePlanStatus.REJECTED_BY_GOVERNOR.value),
            make_result_spec(status=GovernancePlanStatus.REVISION_REQUIRED.value),
        ],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.NO_SELECTABLE_PLAN


# ---------------------------------------------------------------------------
# Deterministic arbitration
# ---------------------------------------------------------------------------


def test_better_retention_wins_at_equal_value(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            _clean(dimensions={"retention_preservation": "ADEQUATE"}),
            _clean(dimensions={"retention_preservation": "STRONG"}),
        ],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.selected_plan_id == fixture.plans[1].id


def test_higher_narration_burden_loses_with_no_greater_value(
    session: Session, monkeypatch: Any
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            _clean(dimensions={"narration_burden": "EXCESSIVE"}),
            _clean(dimensions={"narration_burden": "APPROPRIATE"}),
        ],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.selected_plan_id == fixture.plans[1].id


def test_lower_platform_reuse_risk_wins_when_comparable(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            _clean(risk=platform_risk(youtube_reused="MODERATE", facebook_unoriginal="MODERATE")),
            _clean(risk=platform_risk()),
        ],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.selected_plan_id == fixture.plans[1].id
    distinction = view.row.arbitration_evidence["first_material_distinction"]
    assert distinction["dimension"] == "platform_reuse_risk"


def test_worse_stage42_evidence_cannot_win_through_better_stage41_rank(
    session: Session, monkeypatch: Any
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            _clean(dimensions={"substantive_originality": "ADEQUATE"}),
            _clean(dimensions={"substantive_originality": "STRONG"}),
        ],
    )
    # Plan 0 has generation_rank 1; plan 1 rank 2. Evidence must override rank.
    assert fixture.plans[0].generation_rank == 1
    view = _select(session, fixture.candidate.id)
    assert view.row.selected_plan_id == fixture.plans[1].id


def test_planner_confidence_cannot_override_blocking(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            _clean(),
            make_result_spec(status=GovernancePlanStatus.BLOCKED_PENDING_VERIFICATION.value),
        ],
    )
    fixture.plans[1].planner_confidence = 1.0
    session.flush()
    view = _select(session, fixture.candidate.id)
    assert view.row.selected_plan_id == fixture.plans[0].id


def test_arbitration_has_no_opaque_aggregate_score(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean(), _clean()])
    view = _select(session, fixture.candidate.id)
    evidence = view.row.arbitration_evidence
    assert evidence["comparison_dimensions"][0] == "semantic_fidelity"
    serialized = json.dumps(evidence).casefold()
    for forbidden in ("viral_score", "final_plan_score", "weighted_score", "overall_score"):
        assert forbidden not in serialized
    assert "competition_rule" in evidence


# ---------------------------------------------------------------------------
# Integrity fail-closed
# ---------------------------------------------------------------------------


def test_semantic_fidelity_failure_cannot_be_selected(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            _clean(dimensions={"semantic_fidelity": "NONE"}, reason_codes=["SEMANTIC_DISTORTION"]),
            _clean(dimensions={"semantic_fidelity": "WEAK"}),
        ],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.selected_plan_id is None
    assert view.row.status in {
        TransformationSelectionStatus.NO_SELECTABLE_PLAN,
        TransformationSelectionStatus.SELECTION_DEFERRED,
    }


def test_unresolved_verification_cannot_be_selected(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[
            _clean(
                verification={
                    "claim_state": "EXTERNAL_REQUIRED_UNRESOLVED",
                    "claims": [],
                    "unresolved": True,
                    "reason_codes": ["EXTERNAL_VERIFICATION_REQUIRED"],
                }
            )
        ],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.selected_plan_id is None
    assert view.row.status is TransformationSelectionStatus.SELECTION_DEFERRED
    assert view.row.selection_reason_codes == ["INCONSISTENT_GOVERNANCE_EVIDENCE"]


def test_non_empty_hard_gates_cannot_be_bypassed(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[_clean(hard_gates=[gate("PLAN_INTEGRITY_INVALID")])],
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.selected_plan_id is None
    assert view.row.status is TransformationSelectionStatus.SELECTION_DEFERRED


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


def test_stale_governance_produces_no_selection(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    fixture.governance_set.input_fingerprint = "0" * 64
    session.flush()
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.STALE_SELECTION_INPUT
    assert view.row.selected_plan_id is None


def test_not_current_governance_produces_no_selection(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    fixture.governance_set.execution_status = GovernanceExecutionStatus.QUEUED
    session.flush()
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.SELECTION_DEFERRED
    assert view.row.selection_reason_codes == ["GOVERNANCE_NOT_CURRENT"]


def test_unverifiable_governance_produces_no_selection(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    fixture.governance_set.input_fingerprint = ""
    session.flush()
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.SELECTION_DEFERRED
    assert view.row.selection_reason_codes == ["GOVERNANCE_UNVERIFIABLE"]


def test_missing_governance_set_is_deferred(session: Session, monkeypatch: Any) -> None:
    from stage41_support import seed_stage41

    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    source, candidate, *_rest = seed_stage41(session, settings=settings)
    session.commit()
    view = _select(session, candidate.id)
    assert view.row.status is TransformationSelectionStatus.SELECTION_DEFERRED
    assert view.row.selection_reason_codes == ["GOVERNANCE_NOT_AVAILABLE"]


# ---------------------------------------------------------------------------
# Idempotency / historical rows / invalidation
# ---------------------------------------------------------------------------


def test_stable_input_produces_same_plan_and_reasons(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean(), _clean()])
    first = _select(session, fixture.candidate.id)
    second = _select(session, fixture.candidate.id)
    assert first.row.id == second.row.id
    assert first.row.selected_plan_id == second.row.selected_plan_id
    assert first.row.input_fingerprint == second.row.input_fingerprint


def test_repeated_post_returns_same_row(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    first = _select(session, fixture.candidate.id)
    second = _select(session, fixture.candidate.id)
    assert first.row.id == second.row.id
    rows = session.scalars(
        select(TransformationPlan).where(TransformationPlan.is_current.is_(True))
    ).all()
    assert rows


def test_upstream_change_creates_new_current_and_marks_old_historical(
    session: Session, monkeypatch: Any
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    first = _select(session, fixture.candidate.id)
    plan = fixture.plans[0]
    plan.blocks = [dict(block) for block in (plan.blocks or [])] + [
        {"index": 99, "block_type": "TRANSITION", "estimated_duration": 1.0}
    ]
    session.flush()
    second = _select(session, fixture.candidate.id)
    assert second.row.status is TransformationSelectionStatus.STALE_SELECTION_INPUT
    assert second.row.id != first.row.id
    session.refresh(first.row)
    assert first.row.is_current is False
    assert second.row.is_current is True


def test_plan_fingerprint_change_invalidates_selection(session: Session, monkeypatch: Any) -> None:
    from app.models import TransformationGovernanceResult

    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    first = _select(session, fixture.candidate.id)
    plan = fixture.plans[0]
    result = session.scalars(
        select(TransformationGovernanceResult).where(
            TransformationGovernanceResult.transformation_plan_id == plan.id
        )
    ).one()
    plan.plan_output_fingerprint = "changed-plan-fingerprint"
    result.plan_output_fingerprint = "changed-plan-fingerprint"
    session.flush()
    from app.transformation.governance.executor import (
        build_transformation_governance_executor,
    )

    fixture.governance_set.input_fingerprint = build_transformation_governance_executor(
        session, settings
    ).input_fingerprint(fixture.governance_set)
    session.flush()
    second = _select(session, fixture.candidate.id)
    assert second.row.status is TransformationSelectionStatus.PLAN_SELECTED
    assert second.row.input_fingerprint != first.row.input_fingerprint
    assert second.row.selected_plan_fingerprint == "changed-plan-fingerprint"


def test_platform_policy_profile_change_invalidates_selection(
    session: Session, monkeypatch: Any
) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    first = _select(session, fixture.candidate.id)
    fixture.governance_set.platform_policy_profile_version = "stage4.2-platform-policy-changed"
    session.flush()
    second = _select(session, fixture.candidate.id)
    assert second.row.input_fingerprint != first.row.input_fingerprint
    assert second.row.id != first.row.id


def test_selection_fingerprint_excludes_tts_render_and_publishing(
    session: Session, monkeypatch: Any
) -> None:
    from stage42_support import install_stage42_settings

    from app.transformation.selection.fingerprints import build_selection_input_payload

    settings = FakeGovernanceSettings()
    install_stage42_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    fixture.candidate  # keep reference
    # A read-only recompute: payload must contain no TTS/render/publishing keys.
    view_before = read_selection(session, fixture.candidate.id)
    assert view_before is None
    _select(session, fixture.candidate.id)
    # Change settings that only affect TTS/render/publishing identity.
    setattr(settings, "stage43_tts_voice", "Charon")
    setattr(settings, "stage43_render_resolution", "1080x1920")
    re_view = select_transformation_plan(session, fixture.candidate.id)
    assert re_view is not None
    payload = build_selection_input_payload(
        candidate_id=str(fixture.candidate.id),
        candidate_key=fixture.candidate.candidate_key,
        source_id=str(fixture.candidate.source_video_id),
        disposition=fixture.candidate.disposition.value,
        is_current=True,
        analysis_fingerprint=fixture.candidate.analysis_fingerprint or "",
        handoff={},
        plan_rows=[],
    )
    serialized = json.dumps(payload).casefold()
    for token in ("tts", "voice", "render", "publish", "caption", "b_roll"):
        assert token not in serialized


# ---------------------------------------------------------------------------
# FINAL_CLIP / execution readiness
# ---------------------------------------------------------------------------


def _add_final_refinement(
    session: Session,
    fixture: Any,
    *,
    status: RefinementStatus = RefinementStatus.FINAL_TRANSCRIPT_READY,
    transcript: str = "final transcript text",
) -> CandidateRefinement:
    row = CandidateRefinement(
        source_video_id=fixture.source.id,
        clip_candidate_id=fixture.candidate.id,
        priority=RefinementPriority.FINAL_CLIP,
        status=status,
        coarse_start=fixture.refinement.coarse_start,
        coarse_end=fixture.refinement.coarse_end,
        context_start=fixture.refinement.context_start,
        context_end=fixture.refinement.context_end,
        refined_start=fixture.refinement.refined_start,
        refined_end=fixture.refinement.refined_end,
        automatic_transcript=transcript,
        final_transcript=transcript,
        confidence=0.95,
        quality_level="FINAL_CLIP",
        output_fingerprint="final-output-fp",
    )
    session.add(row)
    session.flush()
    return row


def test_candidate_planning_is_selectable_but_needs_final_refinement(
    session: Session, monkeypatch: Any
) -> None:
    from app.transformation.selection.handoff import build_execution_handoff

    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    _select(session, fixture.candidate.id)
    handoff = build_execution_handoff(session, fixture.candidate.id)
    assert handoff is not None
    assert handoff["selection"]["status"] == "PLAN_SELECTED"
    assert handoff["selected_plan"] is not None
    assert handoff["final_clip_refinement_available"] is False
    assert handoff["final_clip_refinement_required"] is True
    assert handoff["selection_based_on_final_clip"] is False
    assert handoff["execution_readiness"] == "READY_FOR_FINAL_REFINEMENT"


def test_current_valid_final_clip_is_detected(session: Session, monkeypatch: Any) -> None:
    from app.transformation.selection.handoff import build_execution_handoff

    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session, settings=settings, result_specs=[_clean()], planning_on_final=True
    )
    _select(session, fixture.candidate.id)
    handoff = build_execution_handoff(session, fixture.candidate.id)
    assert handoff is not None
    assert handoff["final_clip_refinement_available"] is True
    assert handoff["same_refinement_identity"] is True
    assert handoff["execution_readiness"] == "READY_FOR_EXECUTION_PREP"


def test_newer_final_clip_requires_compatibility_check(session: Session, monkeypatch: Any) -> None:
    from app.transformation.selection.handoff import build_execution_handoff

    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    _select(session, fixture.candidate.id)
    _add_final_refinement(session, fixture)
    handoff = build_execution_handoff(session, fixture.candidate.id)
    assert handoff is not None
    assert handoff["final_clip_refinement_available"] is True
    assert handoff["same_refinement_identity"] is False
    assert handoff["execution_readiness"] == "REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK"


def test_selection_never_queues_refinement_work(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    before = session.scalar(
        select(ProcessingJob).where(ProcessingJob.source_video_id == fixture.source.id)
    )
    _select(session, fixture.candidate.id)
    after = session.scalar(
        select(ProcessingJob).where(ProcessingJob.source_video_id == fixture.source.id)
    )
    assert before is None and after is None


# ---------------------------------------------------------------------------
# Preservation / scope
# ---------------------------------------------------------------------------


def test_selection_preserves_plan_blocks_and_governor_rows(
    session: Session, monkeypatch: Any
) -> None:
    from app.models import TransformationGovernanceResult
    from app.transformation.governance.queue import list_results

    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean(), _clean()])
    blocks_before = [list(plan.blocks or []) for plan in fixture.plans]
    results_before = [
        (row.status, dict(row.dimensions), list(row.warnings))
        for row in list_results(session, fixture.governance_set.id)
    ]
    _select(session, fixture.candidate.id)
    session.expire_all()
    for plan, blocks in zip(fixture.plans, blocks_before):
        session.refresh(plan)
        assert list(plan.blocks or []) == blocks
    results_after = [
        (row.status, dict(row.dimensions), list(row.warnings))
        for row in list_results(session, fixture.governance_set.id)
    ]
    assert results_after == results_before
    assert session.scalars(select(TransformationGovernanceResult)).all()


def test_narration_semantics_preserved_without_voice(session: Session, monkeypatch: Any) -> None:
    from app.transformation.selection.handoff import build_execution_handoff

    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session, settings=settings, result_specs=[_clean()], plan_narration="RECOMMENDED"
    )
    _select(session, fixture.candidate.id)
    handoff = build_execution_handoff(session, fixture.candidate.id)
    assert handoff is not None
    assert handoff["narration"]["need"] == "RECOMMENDED"
    serialized = json.dumps(handoff).casefold()
    for token in ("voice_id", "voice", "speaker_identity", "tts_provider"):
        assert token not in serialized


def test_selection_makes_no_provider_calls(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    view = _select(session, fixture.candidate.id)
    metrics = view.row.metrics
    assert metrics.get("gemini_calls", 0) in (0, None)
    assert "qwen" not in json.dumps(metrics).casefold()


def test_third_party_source_can_be_selected_without_merging_risk_kinds(
    session: Session, monkeypatch: Any
) -> None:
    from app.transformation.selection.handoff import build_execution_handoff

    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(
        session,
        settings=settings,
        result_specs=[_clean()],
        rights_risk="ELEVATED",
        originality_risk="TRANSFORMATION_REQUIRED",
        rights_status="THIRD_PARTY_REUSE",
    )
    view = _select(session, fixture.candidate.id)
    assert view.row.status is TransformationSelectionStatus.PLAN_SELECTED
    handoff = build_execution_handoff(session, fixture.candidate.id)
    assert handoff is not None
    assert handoff["rights_and_provenance"]["rights_risk"] == "ELEVATED"
    assert (
        handoff["rights_and_provenance"]["platform_originality_risk"] == "TRANSFORMATION_REQUIRED"
    )


def test_no_platform_guarantee_text_in_selection_or_handoff(
    session: Session, monkeypatch: Any
) -> None:
    from app.transformation.selection.handoff import build_execution_handoff

    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    view = _select(session, fixture.candidate.id)
    handoff = build_execution_handoff(session, fixture.candidate.id)
    serialized = json.dumps({"row": view.row.arbitration_evidence, "handoff": handoff}).casefold()
    for forbidden in (
        "safe_for_youtube",
        "safe_for_facebook",
        "will_be_monetized",
        "will_not_be_flagged",
        "algorithm_safe",
    ):
        assert forbidden not in serialized


def test_stage5_and_stage6_are_not_implemented(session: Session, monkeypatch: Any) -> None:
    from app.transformation.selection.handoff import build_execution_handoff

    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    _select(session, fixture.candidate.id)
    handoff = build_execution_handoff(session, fixture.candidate.id)
    assert handoff is not None
    assert handoff["stage5_implemented"] is False
    assert handoff["stage6_tts_implemented"] is False


def test_get_current_selection_returns_current_row(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    view = _select(session, fixture.candidate.id)
    current = get_current_selection(session, fixture.candidate.id)
    assert current is not None
    assert current.id == view.row.id


def test_concurrent_selection_is_unique_on_sqlite(session: Session, monkeypatch: Any) -> None:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    fixture = seed_selection_fixture(session, settings=settings, result_specs=[_clean()])
    first = _select(session, fixture.candidate.id)
    second = _select(session, fixture.candidate.id)
    assert first.row.id == second.row.id
