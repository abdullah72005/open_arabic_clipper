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
    String,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.composition.policy import (
    VisualCompositionExecutionStatus,
    VisualCompositionStatus,
)
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.candidate_refinement import CandidateRefinement
    from app.models.clip_candidate import ClipCandidate
    from app.models.render_contract import RenderContract
    from app.models.source_video import SourceVideo
    from app.models.transformation_plan import TransformationPlan
    from app.models.transformation_selection import TransformationPlanSelection


class VisualCompositionPlan(Base):
    """One durable Stage 5.1 deterministic visual-composition plan.

    Preserves history: one row per ``(clip_candidate_id, input_fingerprint)``
    with exactly one database-current row per candidate, enforced by a partial
    unique index. The plan binds a current executable Stage 5.0 render contract,
    its Stage 4.3 selection, Stage 4.1 selected plan, and FINAL_CLIP refinement.
    It is a plan only: it never carries a rendered artifact, TTS voice/provider/
    model, final codec setting, or publishing metadata.
    """

    __tablename__ = "visual_composition_plans"
    __table_args__ = (
        UniqueConstraint(
            "clip_candidate_id",
            "input_fingerprint",
            name="uq_visual_composition_plans_candidate_input",
        ),
        Index(
            "uq_visual_composition_plans_current",
            "clip_candidate_id",
            unique=True,
            postgresql_where=text("is_current"),
            sqlite_where=text("is_current"),
        ),
        CheckConstraint(
            "is_current IN (true, false)",
            name="ck_visual_composition_plans_current_bool",
        ),
        CheckConstraint(
            "plan_ready IN (true, false)",
            name="ck_visual_composition_plans_ready_bool",
        ),
        CheckConstraint(
            "cache_eligible IN (true, false)",
            name="ck_visual_composition_plans_cache_eligible_bool",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clip_candidate_id: Mapped[UUID] = mapped_column(
        ForeignKey("clip_candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    render_contract_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("render_contracts.id", ondelete="SET NULL"), index=True
    )
    transformation_selection_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("transformation_plan_selections.id", ondelete="SET NULL"), index=True
    )
    selected_plan_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("transformation_plans.id", ondelete="SET NULL"), index=True
    )
    final_refinement_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("candidate_refinements.id", ondelete="SET NULL"), index=True
    )

    status: Mapped[VisualCompositionStatus] = mapped_column(
        Enum(
            VisualCompositionStatus,
            name="visual_composition_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        index=True,
    )
    execution_status: Mapped[VisualCompositionExecutionStatus] = mapped_column(
        Enum(
            VisualCompositionExecutionStatus,
            name="visual_composition_execution_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=VisualCompositionExecutionStatus.QUEUED,
        index=True,
    )
    plan_ready: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true", index=True
    )
    active_job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True
    )
    reason_codes: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )

    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    contract_input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    contract_output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    caption_source_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    source_media_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    source_media_identity: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    analysis_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    framing_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    ass_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )

    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage5.1-v1", server_default="stage5.1-v1"
    )
    schema_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="stage5.1-schema-v1",
        server_default="stage5.1-schema-v1",
    )
    fingerprint_version: Mapped[str] = mapped_column(
        String(16), nullable=False, default="1", server_default="1"
    )

    plan_payload: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    readiness: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    metrics: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    cache_eligible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    source_video: Mapped["SourceVideo"] = relationship()
    clip_candidate: Mapped["ClipCandidate"] = relationship()
    render_contract: Mapped["RenderContract | None"] = relationship()
    selection: Mapped["TransformationPlanSelection | None"] = relationship()
    selected_plan: Mapped["TransformationPlan | None"] = relationship()
    final_refinement: Mapped["CandidateRefinement | None"] = relationship()


__all__ = ["VisualCompositionPlan"]
