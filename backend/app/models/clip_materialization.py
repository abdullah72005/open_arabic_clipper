from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import (
    MaterializationLifecycle,
    MaterializationOutcome,
    MaterializationPriority,
)
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.candidate_refinement import CandidateRefinement
    from app.models.channel_production_profile import ChannelProductionProfile
    from app.models.clip_candidate import ClipCandidate
    from app.models.render_contract import RenderContract
    from app.models.render_execution import RenderExecution
    from app.models.source_video import SourceVideo
    from app.models.transformation_plan import TransformationPlan
    from app.models.transformation_selection import TransformationPlanSelection
    from app.models.visual_composition_plan import VisualCompositionPlan

_CACHE_OK = (
    "lifecycle = 'COMPLETE' AND semantic_outcome IN "
    "('READY_FOR_HUMAN_REVIEW', 'MATERIALIZED_WITH_CAUTION')"
)


class ClipMaterialization(Base):
    """One durable Stage 6 transformed-short materialization for a candidate.

    Preserves history: one row per ``(clip_candidate_id, profile_key,
    input_fingerprint)`` with exactly one database-current row per
    ``(clip_candidate_id, profile_key)`` scope, enforced by a scoped partial
    unique index. Detailed evidence lives in bounded validated JSON; there is no
    table per sentence, caption, provider call, repair, or timeline event.
    ``publication_ready`` is always ``false``: a materialized short is
    human-review-ready, never publication-authorised.
    """

    __tablename__ = "clip_materializations"
    __table_args__ = (
        UniqueConstraint(
            "clip_candidate_id",
            "profile_key",
            "input_fingerprint",
            name="uq_clip_materializations_candidate_profile_input",
        ),
        Index(
            "uq_clip_materializations_current_scope",
            "clip_candidate_id",
            "profile_key",
            unique=True,
            postgresql_where=text("is_current"),
            sqlite_where=text("is_current"),
        ),
        CheckConstraint(
            "is_current IN (true, false)", name="ck_clip_materializations_current_bool"
        ),
        CheckConstraint(
            "publication_ready IN (true, false)",
            name="ck_clip_materializations_publication_ready_bool",
        ),
        CheckConstraint(
            "human_review_required IN (true, false)",
            name="ck_clip_materializations_human_review_bool",
        ),
        CheckConstraint(
            "cache_eligible IN (true, false)",
            name="ck_clip_materializations_cache_eligible_bool",
        ),
        CheckConstraint(
            "repair_count >= 0 AND repair_count <= 1",
            name="ck_clip_materializations_repair_count_bounded",
        ),
        CheckConstraint(
            f"(cache_eligible AND {_CACHE_OK}) OR (NOT cache_eligible)",
            name="ck_clip_materializations_cache_consistency",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clip_candidate_id: Mapped[UUID] = mapped_column(
        ForeignKey("clip_candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    channel_production_profile_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("channel_production_profiles.id", ondelete="SET NULL"), index=True
    )
    # Non-null stable scope key (the profile's channel_key, or "default"), so the
    # one-current-row scope never depends on a NULLable FK.
    profile_key: Mapped[str] = mapped_column(String(120), nullable=False, default="default")

    selection_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("transformation_plan_selections.id", ondelete="SET NULL"), index=True
    )
    selected_plan_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("transformation_plans.id", ondelete="SET NULL"), index=True
    )
    refinement_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("candidate_refinements.id", ondelete="SET NULL"), index=True
    )
    render_contract_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("render_contracts.id", ondelete="SET NULL"), index=True
    )
    visual_composition_plan_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("visual_composition_plans.id", ondelete="SET NULL"), index=True
    )
    render_execution_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("render_executions.id", ondelete="SET NULL"), index=True
    )
    active_job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True
    )
    # Durable ownership token. Every successful claim advances it, so a
    # superseded worker can never replace a newer current materialization.
    claim_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    priority: Mapped[MaterializationPriority] = mapped_column(
        Enum(
            MaterializationPriority,
            name="materialization_priority",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=MaterializationPriority.CANDIDATE,
        index=True,
    )
    lifecycle: Mapped[MaterializationLifecycle] = mapped_column(
        Enum(
            MaterializationLifecycle,
            name="materialization_lifecycle",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=MaterializationLifecycle.QUEUED,
        index=True,
    )
    semantic_outcome: Mapped[MaterializationOutcome | None] = mapped_column(
        Enum(
            MaterializationOutcome,
            name="materialization_outcome",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=True,
        index=True,
    )
    human_review_required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    publication_ready: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    cache_eligible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true", index=True
    )
    repair_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    reason_codes: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2048))

    upstream_fingerprints: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    upstream_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    request_payload: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )

    component_checkpoints: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    call_ledger: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    authored_blocks: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    claim_ledger: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    slot_resolutions: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    narration_references: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    asset_references: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    provider_identity: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    runtime_identity: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    timeline_payload: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    audio_recipe: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    quality_findings: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    repair_actions: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    artifact_reference: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    metrics: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )

    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage6-v1", server_default="stage6-v1"
    )
    schema_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage6-schema-v1", server_default="stage6-schema-v1"
    )
    fingerprint_version: Mapped[str] = mapped_column(
        String(16), nullable=False, default="1", server_default="1"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    source_video: Mapped["SourceVideo"] = relationship()
    clip_candidate: Mapped["ClipCandidate"] = relationship()
    channel_production_profile: Mapped["ChannelProductionProfile | None"] = relationship(
        back_populates="materializations"
    )
    selection: Mapped["TransformationPlanSelection | None"] = relationship()
    selected_plan: Mapped["TransformationPlan | None"] = relationship()
    refinement: Mapped["CandidateRefinement | None"] = relationship()
    render_contract: Mapped["RenderContract | None"] = relationship()
    visual_composition_plan: Mapped["VisualCompositionPlan | None"] = relationship()
    render_execution: Mapped["RenderExecution | None"] = relationship(
        foreign_keys=[render_execution_id]
    )


__all__ = ["ClipMaterialization"]
