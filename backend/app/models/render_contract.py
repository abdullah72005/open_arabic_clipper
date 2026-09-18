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

from app.core.enums import FinalClipCompatibilityOutcome, RenderContractStatus
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.candidate_refinement import CandidateRefinement
    from app.models.clip_candidate import ClipCandidate
    from app.models.source_video import SourceVideo
    from app.models.transformation_plan import TransformationPlan
    from app.models.transformation_selection import TransformationPlanSelection

_EXECUTABLE_STATUSES = "('READY_FOR_RENDER_PLANNING', 'MATERIALIZATION_REQUIRED')"


class RenderContract(Base):
    """One durable Stage 5.0 deterministic execution/render contract.

    Preserves history: one row per ``(clip_candidate_id, input_fingerprint)``
    with exactly one database-current row per candidate, enforced by a partial
    unique index. The contract binds a selected Stage 4.1 plan and Stage 4.3
    selection to current FINAL_CLIP evidence and a present managed source
    artifact. It is never a rendered artifact and never carries a TTS voice,
    provider, model, caption file, crop path, or publishing metadata.
    """

    __tablename__ = "render_contracts"
    __table_args__ = (
        UniqueConstraint(
            "clip_candidate_id",
            "input_fingerprint",
            name="uq_render_contracts_candidate_input",
        ),
        Index(
            "uq_render_contracts_current",
            "clip_candidate_id",
            unique=True,
            postgresql_where=text("is_current"),
            sqlite_where=text("is_current"),
        ),
        CheckConstraint(
            "is_current IN (true, false)",
            name="ck_render_contracts_current_bool",
        ),
        CheckConstraint(
            "contract_ready IN (true, false)",
            name="ck_render_contracts_ready_bool",
        ),
        CheckConstraint(
            f"(contract_ready AND status IN {_EXECUTABLE_STATUSES}) "
            f"OR (NOT contract_ready AND status NOT IN {_EXECUTABLE_STATUSES})",
            name="ck_render_contracts_ready_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clip_candidate_id: Mapped[UUID] = mapped_column(
        ForeignKey("clip_candidates.id", ondelete="CASCADE"), nullable=False, index=True
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

    status: Mapped[RenderContractStatus] = mapped_column(
        Enum(
            RenderContractStatus,
            name="render_contract_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        index=True,
    )
    compatibility_outcome: Mapped[FinalClipCompatibilityOutcome | None] = mapped_column(
        Enum(
            FinalClipCompatibilityOutcome,
            name="final_clip_compatibility_outcome",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=True,
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true", index=True
    )
    contract_ready: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    reason_codes: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    compatibility_evidence: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    source_media_identity: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    source_probe: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    readiness: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    contract_payload: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )

    selected_plan_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    planning_refinement_output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    final_refinement_output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    caption_source_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    source_media_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    probe_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )

    profile_key: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    profile_version: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage5.0-v2", server_default="stage5.0-v2"
    )
    schema_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="stage5.0-schema-v1",
        server_default="stage5.0-schema-v1",
    )
    fingerprint_version: Mapped[str] = mapped_column(
        String(16), nullable=False, default="1", server_default="1"
    )
    metrics: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    source_video: Mapped["SourceVideo"] = relationship()
    clip_candidate: Mapped["ClipCandidate"] = relationship()
    selection: Mapped["TransformationPlanSelection | None"] = relationship()
    selected_plan: Mapped["TransformationPlan | None"] = relationship()
    final_refinement: Mapped["CandidateRefinement | None"] = relationship()


__all__ = ["RenderContract"]
