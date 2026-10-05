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

from app.core.enums import (
    RenderArtifactPurpose,
    RenderExecutionLifecycle,
    RenderQCStatus,
)
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.candidate_refinement import CandidateRefinement
    from app.models.clip_candidate import ClipCandidate
    from app.models.render_contract import RenderContract
    from app.models.source_video import SourceVideo
    from app.models.transformation_plan import TransformationPlan
    from app.models.transformation_selection import TransformationPlanSelection
    from app.models.visual_composition_plan import VisualCompositionPlan

_COMPLETE_OK = "lifecycle = 'COMPLETE' AND qc_status IN ('PASS', 'WARN')"


class RenderExecution(Base):
    """One durable Stage 5.2 render execution (source-validation artifact).

    Preserves history: one row per ``(clip_candidate_id, input_fingerprint)``
    with one database-current row per ``(candidate, artifact_purpose,
    delivery_profile_key)``, enforced by a scoped partial unique index.
    """

    __tablename__ = "render_executions"
    __table_args__ = (
        UniqueConstraint(
            "clip_candidate_id",
            "input_fingerprint",
            name="uq_render_executions_candidate_input",
        ),
        Index(
            "uq_render_executions_current_scope",
            "clip_candidate_id",
            "artifact_purpose",
            "delivery_profile_key",
            unique=True,
            postgresql_where=text("is_current"),
            sqlite_where=text("is_current"),
        ),
        CheckConstraint("is_current IN (true, false)", name="ck_render_executions_current_bool"),
        CheckConstraint(
            "publication_ready IN (true, false)",
            name="ck_render_executions_publication_ready_bool",
        ),
        CheckConstraint(
            "stage6_implemented IN (true, false)",
            name="ck_render_executions_stage6_bool",
        ),
        CheckConstraint(
            "cache_eligible IN (true, false)", name="ck_render_executions_cache_eligible_bool"
        ),
        CheckConstraint(
            f"(cache_eligible AND {_COMPLETE_OK}) OR (NOT cache_eligible)",
            name="ck_render_executions_cache_consistency",
        ),
        CheckConstraint(
            f"({_COMPLETE_OK}) OR (lifecycle <> 'COMPLETE')",
            name="ck_render_executions_complete_qc",
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
    visual_composition_plan_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("visual_composition_plans.id", ondelete="SET NULL"), index=True
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

    artifact_purpose: Mapped[RenderArtifactPurpose] = mapped_column(
        Enum(
            RenderArtifactPurpose,
            name="render_artifact_purpose",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=RenderArtifactPurpose.CORE_SOURCE_VALIDATION,
        index=True,
    )
    lifecycle: Mapped[RenderExecutionLifecycle] = mapped_column(
        Enum(
            RenderExecutionLifecycle,
            name="render_execution_lifecycle",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=RenderExecutionLifecycle.QUEUED,
        index=True,
    )
    qc_status: Mapped[RenderQCStatus | None] = mapped_column(
        Enum(
            RenderQCStatus,
            name="render_qc_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=True,
        index=True,
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true", index=True
    )
    active_job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True
    )
    publication_ready: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    stage6_implemented: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    cache_eligible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    reason_codes: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2048))

    delivery_profile_key: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    delivery_profile_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    runtime_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    compiler_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    qc_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )

    request_payload: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    execution_manifest: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    qc_result: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    artifact_reference: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    omitted_requirements: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    metrics: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )

    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage5.2-v1", server_default="stage5.2-v1"
    )
    schema_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="stage5.2-schema-v1",
        server_default="stage5.2-schema-v1",
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
    render_contract: Mapped["RenderContract | None"] = relationship()
    visual_composition_plan: Mapped["VisualCompositionPlan | None"] = relationship()
    selection: Mapped["TransformationPlanSelection | None"] = relationship()
    selected_plan: Mapped["TransformationPlan | None"] = relationship()
    final_refinement: Mapped["CandidateRefinement | None"] = relationship()


__all__ = ["RenderExecution"]
