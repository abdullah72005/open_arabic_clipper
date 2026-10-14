"""Stage 6 DB input assembly, deterministic orchestration, and persistence.

This is the production path between persisted upstream evidence and a durable
``ClipMaterialization``. It assembles immutable inputs from the current Stage 4
selection/plan, the executable Stage 5.0 contract, the current Stage 5.1
composition, the FINAL_CLIP refinement, the source media, and the channel
profile; it revalidates currentness; it runs the bounded deterministic
materialization pipeline; and it persists the result with independent component
checkpoints, a durable call ledger, and a truthful outcome.

Rendering itself (authored cards, mixed/mastered audio, final-purpose QC) is a
separate Stage 5.2 concern; until it is integrated a materialization that is
otherwise complete is persisted as ``AWAITING_RENDER``.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.enums import (
    MaterializationLifecycle,
    MaterializationOutcome,
    MaterializationPriority,
    RefinementPriority,
)
from app.materialization.acceptance import AssessmentContext, assess_materialization
from app.materialization.fingerprints import (
    input_fingerprint,
    narration_fingerprint,
    output_fingerprint,
    script_fingerprint,
    supporting_fingerprint,
    timeline_fingerprint,
)
from app.materialization.inputs import (
    FinalClipEvidence,
    MaterializationInputs,
    SourceMediaEvidence,
    VisualPlanEvidence,
    WordEvidence,
)
from app.materialization.narration import (
    NarrationSynthesizer,
    narration_requests,
)
from app.materialization.policy import (
    STAGE6_POLICY_VERSION,
    STAGE6_SCHEMA_VERSION,
    STAGE6_VALIDATION_VERSION,
    CallLedger,
    Stage6Config,
)
from app.materialization.profiles import profile_from_row
from app.materialization.providers import (
    MaterializationCritiqueProvider,
    MaterializationScript,
    MaterializationWritingProvider,
    materialize_script,
)
from app.materialization.scripting import (
    all_required_slots_fulfilled,
    resolve_slots,
)
from app.materialization.supporting import (
    SupportingAsset,
    SupportingNeed,
    select_supporting_asset,
)
from app.materialization.timeline import (
    build_audio_recipe,
    build_timeline,
    hero_is_protected,
)
from app.materialization.types import (
    AudioRecipe,
    AuthoredTimeline,
    MaterializationAssessment,
    NarrationAudio,
    SlotResolution,
    SupportingAssetSelection,
)
from app.models import (
    CandidateRefinement,
    ChannelProductionProfile,
    ClipCandidate,
    ClipMaterialization,
    RenderContract,
    SourceVideo,
    TransformationPlan,
    TransformationPlanSelection,
)
from app.models.visual_composition_plan import VisualCompositionPlan as VisualCompositionPlanRow
from app.render.policy import EXECUTABLE_STATUSES
from app.render.service import get_current_render_contract

_FINAL_READY = {"FINAL_TRANSCRIPT_READY", "CANDIDATE_REFINED", "NEEDS_MANUAL_TRANSCRIPT_REVIEW"}


class MaterializationError(ValueError):
    """A Stage 6 request was rejected or missing a prerequisite."""


class MaterializationStaleError(MaterializationError):
    """An upstream input changed or is no longer current."""


@dataclass(frozen=True)
class MaterializationPrerequisites:
    candidate: ClipCandidate
    source: SourceVideo
    selection: TransformationPlanSelection
    plan: TransformationPlan
    contract: RenderContract
    final_clip: CandidateRefinement
    visual_plan: VisualCompositionPlanRow
    profile: ChannelProductionProfile | None


@dataclass
class MaterializationResult:
    """Deterministic materialization outputs (pre-render)."""

    script: MaterializationScript
    slot_resolutions: tuple[SlotResolution, ...]
    narration: tuple[NarrationAudio, ...]
    supporting: tuple[SupportingAssetSelection, ...]
    timeline: AuthoredTimeline
    audio_recipe: AudioRecipe
    assessment: MaterializationAssessment
    component_checkpoints: dict[str, object]
    ledger: CallLedger
    reason_codes: tuple[str, ...]
    upstream_snapshot: dict[str, object] = field(default_factory=dict)
    timeline_conflicts: tuple[Mapping[str, object], ...] = ()
    removals: tuple[Mapping[str, object], ...] = ()


# ---------------------------------------------------------------------------
# DB assembly
# ---------------------------------------------------------------------------


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as error:
        raise MaterializationError("invalid identifier") from error


def _usable_final(session: Session, candidate_id: uuid.UUID) -> CandidateRefinement | None:
    rows = list(
        session.scalars(
            select(CandidateRefinement).where(CandidateRefinement.clip_candidate_id == candidate_id)
        ).all()
    )
    finals = [row for row in rows if row.priority is RefinementPriority.FINAL_CLIP]
    finals.sort(key=lambda row: row.updated_at, reverse=True)
    for row in finals:
        if (row.final_transcript or "").strip() and row.status.value in _FINAL_READY:
            return row
    return None


def resolve_prerequisites(
    session: Session, candidate_id: uuid.UUID | str, *, profile_key: str | None = None
) -> MaterializationPrerequisites:
    """Validate and return every current upstream prerequisite for Stage 6."""

    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        raise MaterializationError("candidate does not exist")
    if not candidate.is_current:
        raise MaterializationStaleError("candidate is stale")
    if candidate.disposition.value not in {"CANDIDATE", "CANDIDATE_NEEDS_REFINEMENT"}:
        raise MaterializationError("candidate was not retained")

    from app.transformation.selection.service import read_selection

    selection_view = read_selection(session, candidate.id)
    selection = selection_view.row if selection_view is not None else None
    if selection is None or selection.selected_plan_id is None:
        raise MaterializationError("candidate has no current selected Stage 4 plan")
    if selection.status.value not in {"PLAN_SELECTED", "PLAN_SELECTED_WITH_CAUTION"}:
        raise MaterializationError("selection is not a selected plan")

    plan = session.get(TransformationPlan, selection.selected_plan_id)
    if plan is None:
        raise MaterializationStaleError("selected plan no longer exists")

    contract = get_current_render_contract(session, candidate.id)
    if contract is None:
        raise MaterializationError("candidate has no Stage 5.0 render contract")
    if not contract.contract_ready or contract.status.value not in EXECUTABLE_STATUSES:
        raise MaterializationError("Stage 5.0 render contract is not executable")

    final_clip = _usable_final(session, candidate.id)
    if final_clip is None:
        raise MaterializationError("candidate has no usable FINAL_CLIP refinement")

    visual_plan = session.scalars(
        select(VisualCompositionPlanRow)
        .where(
            VisualCompositionPlanRow.clip_candidate_id == candidate.id,
            VisualCompositionPlanRow.is_current.is_(True),
        )
        .order_by(VisualCompositionPlanRow.created_at.desc())
    ).first()
    if visual_plan is None or not visual_plan.plan_ready:
        raise MaterializationError("candidate has no current ready Stage 5.1 composition")

    profile = _current_profile(session, profile_key)
    source = session.get(SourceVideo, candidate.source_video_id)
    if source is None:
        raise MaterializationStaleError("source video no longer exists")

    return MaterializationPrerequisites(
        candidate=candidate,
        source=source,
        selection=selection,
        plan=plan,
        contract=contract,
        final_clip=final_clip,
        visual_plan=visual_plan,
        profile=profile,
    )


def _current_profile(session: Session, profile_key: str | None) -> ChannelProductionProfile | None:
    if profile_key is None:
        return None
    return session.scalars(
        select(ChannelProductionProfile).where(
            ChannelProductionProfile.channel_key == profile_key,
            ChannelProductionProfile.is_current.is_(True),
        )
    ).first()


def _final_clip_evidence(row: CandidateRefinement) -> FinalClipEvidence:
    words: list[WordEvidence] = []
    for index, item in enumerate(row.word_timestamps or []):
        if not isinstance(item, Mapping):
            continue
        text = item.get("text", item.get("word", ""))
        start = item.get("start")
        end = item.get("end")
        if not isinstance(text, str) or not text.strip():
            continue
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            continue
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            continue
        confidence = item.get("probability", item.get("confidence"))
        words.append(
            WordEvidence(
                index=index,
                text=text.strip(),
                start=float(start),
                end=float(end),
                confidence=float(confidence)
                if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                else None,
            )
        )
    return FinalClipEvidence(
        refinement_id=str(row.id),
        priority=row.priority.value,
        quality_level=row.quality_level,
        status=row.status.value,
        output_fingerprint=row.output_fingerprint,
        text=row.final_transcript or "",
        words=tuple(words),
        start=float(row.refined_start or row.coarse_start),
        end=float(row.refined_end or row.coarse_end),
        usable=True,
        manual_override=bool(row.manual_transcript),
    )


def _as_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return -1
    return value


def _contract_blocks(contract: RenderContract) -> tuple[Mapping[str, object], ...]:
    payload = dict(contract.contract_payload or {})
    blocks = payload.get("blocks")
    return (
        tuple(item for item in blocks if isinstance(item, Mapping))
        if isinstance(blocks, list)
        else ()
    )


def _contract_slots(contract: RenderContract) -> tuple[Mapping[str, object], ...]:
    payload = dict(contract.contract_payload or {})
    materialization = payload.get("materialization")
    if not isinstance(materialization, Mapping):
        return ()
    slots = materialization.get("slots")
    return (
        tuple(item for item in slots if isinstance(item, Mapping))
        if isinstance(slots, list)
        else ()
    )


def _merge_blocks(
    contract: RenderContract, plan: TransformationPlan
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    """Merge Stage 4.1 plan intents with Stage 5.0 contract bindings by index."""

    plan_blocks = {
        _as_index(block.get("index")): block
        for block in (plan.blocks or [])
        if isinstance(block, Mapping)
    }
    merged: list[dict[str, object]] = []
    bindings: list[dict[str, object]] = []
    for contract_block in _contract_blocks(contract):
        index = _as_index(contract_block.get("block_index"))
        plan_block = plan_blocks.get(index, {})
        block_type = str(contract_block.get("block_type", plan_block.get("block_type", "")))
        binding = contract_block.get("source_binding")
        source_text = ""
        if isinstance(binding, Mapping):
            source_text = str(binding.get("final_clip_text", "") or "")
            bindings.append(dict(binding))
        merged.append(
            {
                "index": index,
                "block_type": block_type,
                "semantic_intent": plan_block.get("semantic_intent", ""),
                "draft_line": plan_block.get("draft_line"),
                "substantive_value_kind": plan_block.get("substantive_value_kind", ""),
                "grounding_refs": plan_block.get("grounding_refs", ()),
                "dependency_ids": plan_block.get("dependency_ids", ()),
                "source_text": source_text,
                "estimated_duration": plan_block.get(
                    "estimated_duration", contract_block.get("estimated_duration_seconds", 0.0)
                ),
                "source_start": binding.get("final_clip_start")
                if isinstance(binding, Mapping)
                else None,
                "source_end": binding.get("final_clip_end")
                if isinstance(binding, Mapping)
                else None,
                "word_start_index": binding.get("final_clip_word_start")
                if isinstance(binding, Mapping)
                else None,
                "word_end_index": binding.get("final_clip_word_end")
                if isinstance(binding, Mapping)
                else None,
            }
        )
    return tuple(merged), tuple(bindings)


def assemble_inputs(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    priority: str = MaterializationPriority.CANDIDATE.value,
    profile_key: str | None = None,
    operator_authored_copy: Sequence[Mapping[str, object]] = (),
    supporting_requested: str = "OFF",
) -> MaterializationInputs:
    """Assemble immutable Stage 6 inputs from current persisted evidence."""

    if priority not in {
        MaterializationPriority.CANDIDATE.value,
        MaterializationPriority.FINAL_CLIP.value,
    }:
        raise MaterializationError("priority must be CANDIDATE or FINAL_CLIP")
    prereq = resolve_prerequisites(session, candidate_id, profile_key=profile_key)
    plan = prereq.plan
    blocks, bindings = _merge_blocks(prereq.contract, plan)

    source_media = _source_media_evidence(prereq.contract)
    visual_plan = _visual_plan_evidence(prereq.visual_plan)
    profile = profile_from_row(prereq.profile)

    return MaterializationInputs(
        candidate_id=str(prereq.candidate.id),
        candidate_key=prereq.candidate.candidate_key,
        source_id=str(prereq.candidate.source_video_id),
        priority=priority,
        selection_id=str(prereq.selection.id),
        selection_output_fingerprint=prereq.selection.output_fingerprint,
        selected_plan_fingerprint=prereq.plan.plan_output_fingerprint,
        plan_id=str(plan.id),
        plan_output_fingerprint=plan.plan_output_fingerprint,
        strategy_type=plan.strategy_type.value,
        intensity=plan.intensity.value,
        blocks=blocks,
        hero_block_index=int(plan.hero_block_index),
        narration={
            "need": plan.narration_need,
            "requirements": dict(plan.narration_requirements or {}),
        },
        verification_dependencies=tuple(
            dict(item)
            for item in (plan.external_fact_dependencies or [])
            if isinstance(item, Mapping)
        ),
        original_value_kinds=tuple(plan.original_value_kinds or []),
        required_context=tuple(plan.required_context or []),
        preservation_constraints=tuple(plan.preservation_constraints or []),
        plan_target_context=dict(plan.target_context or {}),
        contract_id=str(prereq.contract.id),
        contract_status=prereq.contract.status.value,
        contract_input_fingerprint=prereq.contract.input_fingerprint,
        contract_output_fingerprint=prereq.contract.output_fingerprint,
        execution_slots=_contract_slots(prereq.contract),
        contract_source_bindings=bindings,
        final_clip=_final_clip_evidence(prereq.final_clip),
        visual_plan=visual_plan,
        source_media=source_media,
        profile=profile,
        operator_authored_copy=tuple(dict(item) for item in operator_authored_copy),
        supporting_requested=supporting_requested,
    )


def _source_media_evidence(contract: RenderContract) -> SourceMediaEvidence:
    payload = dict(contract.contract_payload or {})
    source_media = (
        payload.get("source_media") if isinstance(payload.get("source_media"), Mapping) else {}
    )
    assert isinstance(source_media, Mapping)
    identity = (
        source_media.get("identity") if isinstance(source_media.get("identity"), Mapping) else {}
    )
    probe = source_media.get("probe") if isinstance(source_media.get("probe"), Mapping) else {}
    assert isinstance(identity, Mapping) and isinstance(probe, Mapping)
    return SourceMediaEvidence(
        relative_path=str(
            source_media.get("managed_relative_path") or identity.get("relative_path") or ""
        ),
        content_hash=str(source_media.get("content_hash") or identity.get("content_hash") or ""),
        size_bytes=_int(identity.get("size_bytes")),
        mtime_ns=_int(identity.get("mtime_ns")),
        duration=float(probe.get("duration_seconds") or 0.0),
        frame_rate=_frame_rate(probe.get("frames_per_second")),
        display_width=_int(probe.get("width")),
        display_height=_int(probe.get("height")),
        encoded_width=_int(probe.get("width")),
        encoded_height=_int(probe.get("height")),
        sample_rate=_int(probe.get("audio_sample_rate")) or 48_000,
        channels=2,
    )


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _frame_rate(value: object) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return Fraction(30, 1)
    return Fraction(str(value)).limit_denominator(1001)


def _visual_plan_evidence(row: VisualCompositionPlanRow) -> VisualPlanEvidence:
    payload = dict(row.plan_payload or {})
    ass = payload.get("ass") if isinstance(payload.get("ass"), Mapping) else {}
    assert isinstance(ass, Mapping)
    return VisualPlanEvidence(
        visual_plan_id=str(row.id),
        output_fingerprint=row.output_fingerprint,
        plan_ready=bool(row.plan_ready),
        ass_relative_path=str(ass.get("asset_path") or ""),
        ass_sha256=str(ass.get("sha256") or ""),
        ass_policy_version=str(ass.get("policy_version") or ""),
        caption_source_fingerprint=row.caption_source_fingerprint,
        framing_fingerprint=row.framing_fingerprint,
        source_media_fingerprint=row.source_media_fingerprint,
    )


# ---------------------------------------------------------------------------
# Deterministic orchestration
# ---------------------------------------------------------------------------


def run_materialization(
    inputs: MaterializationInputs,
    *,
    provider: MaterializationWritingProvider,
    critique_provider: MaterializationCritiqueProvider | None = None,
    narration_synthesizer: NarrationSynthesizer | None = None,
    supporting_assets: Sequence[SupportingAsset] = (),
    config: Stage6Config,
    allow_hosted: bool,
    ledger: CallLedger | None = None,
) -> MaterializationResult:
    """Run the bounded deterministic materialization pipeline over inputs."""

    active_ledger = ledger or CallLedger()
    script = materialize_script(
        provider,
        inputs,
        ledger=active_ledger,
        config=config,
        allow_hosted=allow_hosted,
        critique_provider=critique_provider,
    )

    narration: tuple[NarrationAudio, ...] = ()
    tts_available = narration_synthesizer is not None and narration_synthesizer.available()
    narration_required = _narration_required(inputs)
    if script.accepted and narration_required and inputs.profile.narration_allowed:
        narration = _synthesize_narration(
            inputs, script, narration_synthesizer, active_ledger, config
        )

    narration_blocks = frozenset(chunk.block_index for chunk in narration)
    slot_resolutions = resolve_slots(
        inputs, script.authored_blocks, narration_satisfied_blocks=narration_blocks
    )

    supporting = _select_supporting(inputs, supporting_assets, config)

    build = build_timeline(
        inputs,
        script.authored_blocks,
        narration,
        slot_resolutions=slot_resolutions,
        config=config,
    )
    recipe, audio_conflicts = build_audio_recipe(inputs, build.timeline, narration, config=config)

    script_fp = script_fingerprint(
        inputs,
        [block.as_dict() for block in script.authored_blocks],
        prompt_hash=STAGE6_VALIDATION_VERSION,
        provider_identity=script.provider_identity,
    )
    narration_fp = narration_fingerprint(
        script_fp,
        [chunk.as_dict() for chunk in narration],
        tts_identity=inputs.profile.tts_identity(),
    )
    supporting_fp = supporting_fingerprint([item.as_dict() for item in supporting])
    timeline_fp = timeline_fingerprint(
        inputs,
        [block.as_dict() for block in script.authored_blocks],
        recipe.as_dict(),
        segment_identity=[segment.as_dict() for segment in build.timeline.segments],
    )

    context = AssessmentContext(
        profile_mismatch_fields=_profile_mismatch(inputs),
        required_slots_fulfilled=all_required_slots_fulfilled(slot_resolutions),
        required_narration_required=narration_required and inputs.profile.narration_allowed,
        narration_accepted=bool(narration) or not narration_required,
        tts_available=tts_available,
        provider_deferred=script.deferred,
        unsupported_essential="UNSUPPORTED_ESSENTIAL_CLAIM" in script.reason_codes,
        hero_protected=hero_is_protected(build.timeline, inputs),
        timeline_conflicts=build.conflicts,
        audio_conflicts=audio_conflicts,
        critical_evidence_current=True,
        render_complete=False,
    )
    assessment = assess_materialization(
        context, output_duration=build.timeline.output_duration, config=config
    )

    checkpoints = _component_checkpoints(
        inputs, script, narration, supporting, script_fp, narration_fp, supporting_fp, timeline_fp
    )
    reasons = tuple(dict.fromkeys((*script.reason_codes, *assessment.reason_codes)))
    return MaterializationResult(
        script=script,
        slot_resolutions=slot_resolutions,
        narration=narration,
        supporting=supporting,
        timeline=build.timeline,
        audio_recipe=recipe,
        assessment=assessment,
        component_checkpoints=checkpoints,
        ledger=active_ledger,
        reason_codes=reasons,
        timeline_conflicts=build.conflicts,
        removals=build.removals,
    )


def _narration_required(inputs: MaterializationInputs) -> bool:
    need = str(inputs.narration.get("need", "NONE"))
    return need in {"RECOMMENDED", "REQUIRED"}


def _synthesize_narration(
    inputs: MaterializationInputs,
    script: MaterializationScript,
    synthesizer: NarrationSynthesizer | None,
    ledger: CallLedger,
    config: Stage6Config,
) -> tuple[NarrationAudio, ...]:
    if synthesizer is None or not synthesizer.available():
        return ()
    requests = narration_requests(
        [block.as_dict() for block in script.authored_blocks],
        identity=inputs.profile.tts_identity(),
        config=config,
    )
    chunks: list[NarrationAudio] = []
    for request in requests:
        if ledger.synthesis >= config.max_raw_synthesis_calls:
            break
        ledger.synthesis += 1
        ledger.ledger_entries.append({"kind": "synthesis", "chunk_id": request.chunk_id})
        try:
            chunks.append(synthesizer.synthesize(request))
        except Exception:
            break
    return tuple(chunks)


def _select_supporting(
    inputs: MaterializationInputs,
    assets: Sequence[SupportingAsset],
    config: Stage6Config,
) -> tuple[SupportingAssetSelection, ...]:
    mode = inputs.supporting_requested or inputs.profile.supporting_visual_mode
    if mode == "OFF":
        return ()
    need = _supporting_need(inputs)
    selection = select_supporting_asset(
        mode,
        inputs.profile.supporting_visual_style,
        need,
        assets,
        style_preference=inputs.profile.supporting_visual_style,
    )
    return (selection,)


def _supporting_need(inputs: MaterializationInputs) -> SupportingNeed | None:
    for block in inputs.blocks:
        if str(block.get("block_type", "")) in {"ORIGINAL_VALUE", "TEXTUAL_ANNOTATION"}:
            return SupportingNeed(block_index=int(block.get("index", -1)))
    return None


def _profile_mismatch(inputs: MaterializationInputs) -> tuple[str, ...]:
    from app.materialization.profiles import semantic_context_matches

    ok, mismatches = semantic_context_matches(inputs.profile, inputs.plan_target_context)
    return () if ok else mismatches


def _component_checkpoints(
    inputs: MaterializationInputs,
    script: MaterializationScript,
    narration: Sequence[NarrationAudio],
    supporting: Sequence[SupportingAssetSelection],
    script_fp: str,
    narration_fp: str,
    supporting_fp: str,
    timeline_fp: str,
) -> dict[str, object]:
    return {
        "SCRIPT": {
            "fingerprint": script_fp,
            "provider_identity": dict(script.provider_identity),
            "validation_version": STAGE6_VALIDATION_VERSION,
        },
        "NARRATION": {
            "fingerprint": narration_fp,
            "chunks": [chunk.chunk_id for chunk in narration],
        },
        "SUPPORTING": {
            "fingerprint": supporting_fp,
            "assets": [item.asset_id for item in supporting if item.asset_id],
        },
        "TIMELINE": {
            "fingerprint": timeline_fp,
            "segments": len(script.authored_blocks),
        },
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_materialization(
    session: Session,
    inputs: MaterializationInputs,
    result: MaterializationResult,
    *,
    render_execution_id: uuid.UUID | None = None,
    runtime_identity: Mapping[str, object] | None = None,
) -> ClipMaterialization:
    """Persist one durable materialization row (history-preserving).

    Identical ``(candidate, profile, input fingerprint)`` work updates the
    existing row in place instead of inserting a duplicate; a genuinely changed
    input creates a new history row and demotes the previous current one.
    """

    outcome = result.assessment.outcome
    lifecycle = (
        MaterializationLifecycle.AWAITING_RENDER
        if outcome
        in {
            MaterializationOutcome.READY_FOR_HUMAN_REVIEW.value,
            MaterializationOutcome.MATERIALIZED_WITH_CAUTION.value,
        }
        else MaterializationLifecycle.COMPLETE
    )
    input_fp = input_fingerprint(inputs)
    candidate_uuid = _as_uuid(inputs.candidate_id)
    # Demote every current row in this scope *before* creating/updating the
    # target row, so the scoped partial unique index is never transiently
    # violated when a changed input creates a new history row.
    session.execute(
        update(ClipMaterialization)
        .where(
            ClipMaterialization.clip_candidate_id == candidate_uuid,
            ClipMaterialization.profile_key == inputs.profile.profile_key,
        )
        .values(is_current=False)
        .execution_options(synchronize_session="fetch")
    )
    row = session.scalars(
        select(ClipMaterialization).where(
            ClipMaterialization.clip_candidate_id == candidate_uuid,
            ClipMaterialization.profile_key == inputs.profile.profile_key,
            ClipMaterialization.input_fingerprint == input_fp,
        )
    ).first()
    if row is None:
        row = ClipMaterialization(
            source_video_id=_as_uuid(inputs.source_id),
            clip_candidate_id=candidate_uuid,
            profile_key=inputs.profile.profile_key,
        )
        session.add(row)

    row.channel_production_profile_id = _optional_uuid(inputs.profile.profile_id)
    row.selection_id = _optional_uuid(inputs.selection_id)
    row.selected_plan_id = _optional_uuid(inputs.plan_id)
    row.refinement_id = _optional_uuid(inputs.final_clip.refinement_id)
    row.render_contract_id = _optional_uuid(inputs.contract_id)
    row.visual_composition_plan_id = _optional_uuid(inputs.visual_plan.visual_plan_id)
    row.render_execution_id = render_execution_id
    row.priority = MaterializationPriority(inputs.priority)
    row.lifecycle = lifecycle
    row.semantic_outcome = MaterializationOutcome(outcome)
    row.human_review_required = bool(result.assessment.human_review_required)
    row.publication_ready = False
    row.cache_eligible = False
    row.is_current = True
    row.repair_count = 0
    row.reason_codes = list(result.reason_codes)
    row.upstream_fingerprints = {
        "selection_output_fingerprint": inputs.selection_output_fingerprint,
        "selected_plan_fingerprint": inputs.selected_plan_fingerprint,
        "contract_output_fingerprint": inputs.contract_output_fingerprint,
        "final_clip_output_fingerprint": inputs.final_clip.output_fingerprint,
        "visual_plan_output_fingerprint": inputs.visual_plan.output_fingerprint,
        "source_media_fingerprint": inputs.visual_plan.source_media_fingerprint,
    }
    row.upstream_snapshot = dict(result.upstream_snapshot)
    row.request_payload = {
        "priority": inputs.priority,
        "profile_key": inputs.profile.profile_key,
        "supporting_requested": inputs.supporting_requested,
    }
    row.input_fingerprint = input_fp
    row.output_fingerprint = output_fingerprint(
        inputs,
        script_fp=_checkpoint_fingerprint(result.component_checkpoints, "SCRIPT"),
        narration_fp=_checkpoint_fingerprint(result.component_checkpoints, "NARRATION"),
        supporting_fp=_checkpoint_fingerprint(result.component_checkpoints, "SUPPORTING"),
        timeline_fp=_checkpoint_fingerprint(result.component_checkpoints, "TIMELINE"),
    )
    row.component_checkpoints = dict(result.component_checkpoints)
    row.call_ledger = result.ledger.as_dict()
    row.authored_blocks = [block.as_dict() for block in result.script.authored_blocks]
    row.claim_ledger = [
        block.as_dict() for block in result.script.authored_blocks if block.claim_class
    ]
    row.slot_resolutions = [item.as_dict() for item in result.slot_resolutions]
    row.narration_references = [chunk.as_dict() for chunk in result.narration]
    row.asset_references = [item.as_dict() for item in result.supporting]
    row.provider_identity = dict(result.script.provider_identity)
    row.runtime_identity = dict(runtime_identity or {})
    row.timeline_payload = result.timeline.as_dict()
    row.audio_recipe = result.audio_recipe.as_dict()
    row.quality_findings = [finding.as_dict() for finding in result.assessment.findings]
    row.repair_actions = []
    row.metrics = {
        "segments": len(result.timeline.segments),
        "output_duration": round(result.timeline.output_duration, 6),
        "removals": [dict(item) for item in result.removals],
    }
    row.policy_version = STAGE6_POLICY_VERSION
    row.schema_version = STAGE6_SCHEMA_VERSION

    session.flush()
    session.commit()
    session.refresh(row)
    return row


def _checkpoint_fingerprint(checkpoints: Mapping[str, object], name: str) -> str:
    entry = checkpoints.get(name)
    if isinstance(entry, Mapping):
        fingerprint = entry.get("fingerprint")
        if isinstance(fingerprint, str):
            return fingerprint
    return ""


def _optional_uuid(value: object) -> uuid.UUID | None:
    if value is None:
        return None
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


__all__ = [
    "MaterializationError",
    "MaterializationPrerequisites",
    "MaterializationResult",
    "MaterializationStaleError",
    "assemble_inputs",
    "persist_materialization",
    "resolve_prerequisites",
    "run_materialization",
]
