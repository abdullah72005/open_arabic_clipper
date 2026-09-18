"""Stage 5.0 execution preflight readiness and status semantics."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import (
    FakeProber,
    add_final,
    corrupt_prober,
    make_verification_required_plan,
    managed_source,
    no_video_prober,
    seed_stage50,
)

from app.core.enums import PlanStatus, RenderContractStatus
from app.core.settings import Settings
from app.core.settings import get_settings as core_get_settings
from app.db.base import Base
from app.media.ffprobe import MediaMetadata
from app.models import RenderContract, TransformationPlanSelection
from app.render.handoff import build_stage5_1_handoff
from app.render.policy import (
    EXCERPT_OUT_OF_BOUNDS,
    MATERIALIZATION_REQUIRED,
    READY_FOR_RENDER_PLANNING,
    SOURCE_MEDIA_CORRUPT,
    SOURCE_MEDIA_ZERO_BYTES,
    UNRESOLVED_REQUIRED_VERIFICATION,
)
from app.render.service import (
    _required_verification_unresolved,
    create_render_contract,
    get_current_render_contract,
    read_render_contract,
)


@pytest.fixture  # type: ignore[untyped-decorator]
def session(sqlite_engine: Engine) -> Iterator[Session]:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        yield session


def _install(monkeypatch: Any) -> FakeGovernanceSettings:
    settings = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, settings)
    return settings


def test_no_selected_plan_is_blocked(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    selection = session.query(TransformationPlanSelection).one()
    session.delete(selection)
    session.flush()
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.BLOCKED
    assert view.row.contract_ready is False


def test_stale_selection_has_no_executable_contract(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    selection = session.query(TransformationPlanSelection).one()
    selection.input_fingerprint = "stale-fingerprint"
    session.flush()
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.contract_ready is False
    assert view.row.status is RenderContractStatus.BLOCKED


def test_missing_final_clip_requires_refinement_without_scheduling(
    session: Session, monkeypatch: Any
) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings, planning_on_final=False)
    before = session.query(TransformationPlanSelection).count()
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.FINAL_CLIP_REFINEMENT_REQUIRED
    assert view.row.contract_ready is False
    # No new selection or refinement rows are created.
    assert session.query(TransformationPlanSelection).count() == before


def test_compatible_final_clip_is_executable_with_materialization(
    session: Session, monkeypatch: Any
) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.MATERIALIZATION_REQUIRED
    assert view.row.contract_ready is True
    payload = view.row.contract_payload
    assert payload["materialization"]["required"] is True
    assert payload["caption_input"]["logical_order_preserved"] is True


def test_ready_readiness_mapping_is_ready_for_stage_5_1() -> None:
    from app.render.service import _readiness

    ready = _readiness(
        status=READY_FOR_RENDER_PLANNING,
        executable=True,
        materialization_required=False,
        source_media_ready=True,
        transcript_ready=True,
        compatibility_ready=True,
        verification_ready=True,
        reason_codes=[],
    )
    assert ready["final_render_ready"] is True
    assert ready["next_action"] == "PROCEED_TO_STAGE_5_1"
    assert ready["downstream_stage_eligibility"] == "READY_FOR_DOWNSTREAM_COMPOSITION"

    pending = _readiness(
        status=MATERIALIZATION_REQUIRED,
        executable=True,
        materialization_required=True,
        source_media_ready=True,
        transcript_ready=True,
        compatibility_ready=True,
        verification_ready=True,
        reason_codes=["NARRATION_MATERIALIZATION_REQUIRED"],
    )
    assert pending["final_render_ready"] is False
    assert pending["next_action"] == "STAGE_6_MATERIALIZATION_REQUIRED"


def test_negation_change_requires_upstream_revalidation(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings, planning_on_final=False)
    plan_text = fixture.selection.refinement.final_transcript
    add_final(
        session,
        fixture,
        transcript=plan_text.replace("collapsed productivity", "did not collapse productivity"),
        output_fingerprint="newer-final-fp",
    )
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.UPSTREAM_REVALIDATION_REQUIRED
    assert view.row.contract_ready is False


def test_missing_source_media_is_unavailable(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    fixture.source_path.unlink()
    session.flush()
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.SOURCE_MEDIA_UNAVAILABLE


def test_unmanaged_source_path_is_unavailable(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    fixture.selection.source.source_uri = "/tmp/not-managed-source.mp4"
    session.flush()
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.SOURCE_MEDIA_UNAVAILABLE


def test_no_video_stream_is_unavailable(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    view = create_render_contract(
        session,
        fixture.selection.candidate.id,
        storage=fixture.storage,
        prober=no_video_prober(),
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.SOURCE_MEDIA_UNAVAILABLE


def test_source_span_beyond_duration_is_invalid(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    short = FakeProber(
        MediaMetadata(
            duration_seconds=10.0,
            video_codec="h264",
            width=1920,
            height=1080,
            frames_per_second=30.0,
            audio_codec="aac",
            audio_sample_rate=48_000,
        )
    )
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=short
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.INVALID_SOURCE_BINDING


def test_repeated_request_reuses_same_current_contract(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    first = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    second = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert first is not None and second is not None
    assert first.row.id == second.row.id
    assert second.row.metrics.get("contract_cache_hits", 0) >= 1


def test_probe_reuse_across_contract_versions(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings, planning_on_final=False)
    final = add_final(
        session,
        fixture,
        transcript=fixture.selection.refinement.final_transcript,
        output_fingerprint="final-fp-a",
    )
    prober = FakeProber()
    first = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=prober
    )
    assert first is not None
    assert prober.calls == 1
    final.output_fingerprint = "final-fp-b"
    session.flush()
    second = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=prober
    )
    assert second is not None
    assert second.row.id != first.row.id
    # Cached probe facts are reused for the unchanged source media identity.
    assert prober.calls == 1


def test_read_freshness_and_current_contract(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    created = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert created is not None
    view = read_render_contract(session, fixture.selection.candidate.id)
    assert view is not None
    assert view.row.id == created.row.id
    assert view.live_freshness == "CURRENT"
    assert view.effective is True
    assert get_current_render_contract(session, fixture.selection.candidate.id) is not None


def test_read_freshness_is_stale_after_final_clip_change(
    session: Session, monkeypatch: Any
) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    created = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert created is not None and created.row.contract_ready is True
    final = fixture.selection.refinement
    final.final_transcript = final.final_transcript + " changed"
    final.output_fingerprint = "changed-final-fp"
    session.flush()
    view = read_render_contract(session, fixture.selection.candidate.id)
    assert view is not None
    assert view.live_freshness == "STALE"
    assert view.effective is False
    handoff = build_stage5_1_handoff(session, fixture.selection.candidate.id)
    assert handoff is not None
    assert handoff["contract"]["effective"] is False
    assert handoff["contract"]["live_freshness"] == "STALE"


def test_read_freshness_is_stale_after_plan_change(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    fixture.selection.plans[0].plan_output_fingerprint = "changed-plan-fp"
    session.flush()
    view = read_render_contract(session, fixture.selection.candidate.id)
    assert view is not None
    assert view.live_freshness == "STALE"
    assert view.effective is False


def test_read_freshness_is_stale_after_config_change(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    storage_root = core_get_settings().storage_root
    changed = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_root=storage_root,
        render_contract_max_frame_rate=30.0,
    )
    monkeypatch.setattr("app.render.service.get_settings", lambda: changed)
    view = read_render_contract(session, fixture.selection.candidate.id)
    assert view is not None
    assert view.live_freshness == "STALE"
    assert view.effective is False


def test_readiness_constants_are_consistent(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    readiness = view.row.readiness
    if view.row.status is RenderContractStatus.MATERIALIZATION_REQUIRED:
        assert readiness["next_action"] == "STAGE_6_MATERIALIZATION_REQUIRED"
        assert readiness["final_render_ready"] is False
    assert view.row.status.value in {MATERIALIZATION_REQUIRED, READY_FOR_RENDER_PLANNING}


def test_preflight_does_not_mutate_upstream_rows(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    plan = fixture.selection.plans[0]
    selection = fixture.selection.governance_set
    before_blocks = [dict(block) for block in plan.blocks]
    before_plan_fp = plan.plan_output_fingerprint
    before_governance = dict(selection.provider_evidence or {})
    before_selection_status = session.query(TransformationPlanSelection).one().status

    create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )

    session.refresh(plan)
    assert [dict(block) for block in plan.blocks] == before_blocks
    assert plan.plan_output_fingerprint == before_plan_fp
    assert dict(selection.provider_evidence or {}) == before_governance
    assert session.query(TransformationPlanSelection).one().status == before_selection_status


def test_only_one_current_contract_after_repeated_requests(
    session: Session, monkeypatch: Any
) -> None:
    _install(monkeypatch)
    fixture = seed_stage50(session)
    create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    current = [
        row
        for row in session.query(RenderContract).all()
        if row.clip_candidate_id == fixture.selection.candidate.id and row.is_current
    ]
    assert len(current) == 1


def test_status_enum_has_no_compatibility_check_required(session: Session) -> None:
    assert "COMPATIBILITY_CHECK_REQUIRED" not in {item.value for item in RenderContractStatus}


def test_unusable_final_status_requires_refinement(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings, planning_on_final=False)
    add_final(session, fixture, transcript="", output_fingerprint="empty-final-fp")
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.FINAL_CLIP_REFINEMENT_REQUIRED


def test_zero_byte_source_media_is_unavailable(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    # The managed-source helper writes a zero-byte artifact inside the managed dir.
    managed_source(fixture.selection.source, size=0)
    session.flush()
    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.SOURCE_MEDIA_UNAVAILABLE
    assert view.row.contract_ready is False
    assert SOURCE_MEDIA_ZERO_BYTES in view.row.reason_codes


def test_corrupt_source_media_is_unavailable(session: Session, monkeypatch: Any) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings, prober=corrupt_prober())
    view = create_render_contract(
        session,
        fixture.selection.candidate.id,
        storage=fixture.storage,
        prober=fixture.prober,
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.SOURCE_MEDIA_UNAVAILABLE
    assert view.row.contract_ready is False
    assert SOURCE_MEDIA_CORRUPT in view.row.reason_codes


def test_reversed_persisted_plan_block_span_is_invalid_source_binding(
    session: Session, monkeypatch: Any
) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(session, settings=settings)
    plan = fixture.selection.plans[0]
    blocks = [dict(block) for block in plan.blocks]
    blocks[0]["source_start"] = 30.0
    blocks[0]["source_end"] = 25.0
    plan.blocks = blocks
    session.flush()
    # A mutated plan is caught by the fail-closed upstream freshness gate unless
    # the current governance/selection boundaries are re-established first, so
    # refresh the governance input fingerprint and re-run selection to make the
    # reversed plan the truthful current upstream and exercise Stage 5.0 binding.
    from app.transformation.governance.executor import (
        build_transformation_governance_executor,
    )
    from app.transformation.selection.service import select_transformation_plan

    governance = fixture.selection.governance_set
    executor = build_transformation_governance_executor(session, settings)
    governance.input_fingerprint = executor.input_fingerprint(governance)
    session.flush()
    assert select_transformation_plan(session, fixture.selection.candidate.id) is not None

    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.INVALID_SOURCE_BINDING
    assert view.row.contract_ready is False
    # The public path reports the structural excerpt failure. The internal
    # SPAN_REVERSED_OR_NEGATIVE media-bound code is unreachable: a reversed span
    # never yields the ordered rebound range that media-bound validation checks.
    assert EXCERPT_OUT_OF_BOUNDS in view.row.reason_codes


def _verification_plan(
    *,
    status: PlanStatus = PlanStatus.PLAN_GENERATED,
    dependencies: list[dict[str, object]] | None = None,
    blocks: list[dict[str, object]] | None = None,
) -> Any:
    return SimpleNamespace(
        status=status,
        external_fact_dependencies=list(dependencies or []),
        blocks=list(blocks or []),
    )


_RESOLVED_GOVERNANCE: dict[str, object] = {
    "verification": {"claim_state": "GROUNDED_IN_SOURCE", "unresolved": False}
}
_UNRESOLVED_GOVERNANCE: dict[str, object] = {
    "verification": {"claim_state": "EXTERNAL_REQUIRED_UNRESOLVED", "unresolved": True}
}


def test_required_verification_unresolved_unit_matrix() -> None:
    required_status = PlanStatus.PLAN_GENERATED_WITH_VERIFICATION_REQUIRED
    dependency = [{"dependency": "claim-1", "must_verify_before_execution": True}]
    essential_placeholder = [
        {"block_type": "FACT_VERIFICATION_PLACEHOLDER", "must_verify_before_execution": True}
    ]
    optional_placeholder = [
        {"block_type": "FACT_VERIFICATION_PLACEHOLDER", "must_verify_before_execution": False}
    ]

    # No requirement trigger: never blocks, even when verification is unresolved.
    assert _required_verification_unresolved(_verification_plan(), _UNRESOLVED_GOVERNANCE) is False
    assert _required_verification_unresolved(_verification_plan(), _RESOLVED_GOVERNANCE) is False
    assert _required_verification_unresolved(_verification_plan(), {}) is False

    # PLAN_GENERATED_WITH_VERIFICATION_REQUIRED: blocks only when unresolved.
    assert (
        _required_verification_unresolved(
            _verification_plan(status=required_status), _RESOLVED_GOVERNANCE
        )
        is False
    )
    assert (
        _required_verification_unresolved(
            _verification_plan(status=required_status), _UNRESOLVED_GOVERNANCE
        )
        is True
    )
    # Missing verification evidence fails closed.
    assert _required_verification_unresolved(_verification_plan(status=required_status), {}) is True

    # must_verify_before_execution dependency.
    assert (
        _required_verification_unresolved(
            _verification_plan(dependencies=dependency), _RESOLVED_GOVERNANCE
        )
        is False
    )
    assert (
        _required_verification_unresolved(
            _verification_plan(dependencies=dependency), _UNRESOLVED_GOVERNANCE
        )
        is True
    )

    # Essential verification placeholder.
    assert (
        _required_verification_unresolved(
            _verification_plan(blocks=essential_placeholder), _RESOLVED_GOVERNANCE
        )
        is False
    )
    assert (
        _required_verification_unresolved(
            _verification_plan(blocks=essential_placeholder), _UNRESOLVED_GOVERNANCE
        )
        is True
    )
    # A non-essential placeholder is not a requirement.
    assert (
        _required_verification_unresolved(
            _verification_plan(blocks=optional_placeholder), _UNRESOLVED_GOVERNANCE
        )
        is False
    )


def test_unresolved_required_verification_after_selection_is_blocked(
    session: Session, monkeypatch: Any
) -> None:
    settings = _install(monkeypatch)
    fixture = seed_stage50(
        session,
        settings=settings,
        first_provider_plan_factory=make_verification_required_plan,
    )
    plan = fixture.selection.plans[0]
    assert plan.status.value == "PLAN_GENERATED_WITH_VERIFICATION_REQUIRED"
    # The selected-governance snapshot is mutated to unresolved after selection;
    # the snapshot is not part of the upstream freshness fingerprint, so the
    # truthful Stage 5.0 verification gate (not a stale-input gate) must fire.
    selection = session.query(TransformationPlanSelection).one()
    snapshot = dict(selection.selected_governance_snapshot or {})
    verification = dict(snapshot.get("verification") or {})
    verification["claim_state"] = "EXTERNAL_REQUIRED_UNRESOLVED"
    verification["unresolved"] = True
    snapshot["verification"] = verification
    selection.selected_governance_snapshot = snapshot
    session.flush()

    view = create_render_contract(
        session, fixture.selection.candidate.id, storage=fixture.storage, prober=fixture.prober
    )
    assert view is not None
    assert view.row.status is RenderContractStatus.BLOCKED
    assert view.row.contract_ready is False
    assert UNRESOLVED_REQUIRED_VERIFICATION in view.row.reason_codes
