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

from app.core.enums import TransformationSelectionStatus
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.candidate_refinement import CandidateRefinement
    from app.models.clip_candidate import ClipCandidate
    from app.models.source_video import SourceVideo
    from app.models.transformation_eligibility import TransformationEligibilityAnalysis
    from app.models.transformation_governance import (
        TransformationGovernanceResult,
        TransformationGovernanceSet,
    )
    from app.models.transformation_plan import TransformationPlan, TransformationPlanSet

_SELECTED_STATUSES = "('PLAN_SELECTED', 'PLAN_SELECTED_WITH_CAUTION')"
_CAUTION_STATUS = "PLAN_SELECTED_WITH_CAUTION"


class TransformationPlanSelection(Base):
    """One durable Stage 4.3 selection result for a candidate.

    Preserves selection history: one row per ``(clip_candidate_id,
    input_fingerprint)`` with exactly one database-current row per candidate,
    enforced by a partial unique index. ``selected_plan_id`` is a nullable
    pointer, never a child plan copy; the Stage 4.1 plan and Stage 4.2 result are
    never mutated. The selected governance evidence is snapshotted immutably
    because Stage 4.2 rows may be refreshed later.

    Execution readiness is deliberately *not* persisted here: it is a live
    property of current FINAL_CLIP evidence, computed by the read-only execution
    handoff, so a newly created refinement can never be masked by a stale stored
    snapshot.
    """

    __tablename__ = "transformation_plan_selections"
    __table_args__ = (
        UniqueConstraint(
            "clip_candidate_id",
            "input_fingerprint",
            name="uq_transformation_plan_selections_candidate_input",
        ),
        Index(
            "uq_transformation_plan_selections_current",
            "clip_candidate_id",
            unique=True,
            postgresql_where=text("is_current"),
            sqlite_where=text("is_current"),
        ),
        Index(
            "ix_plan_selections_transformation_governance_set_id",
            "transformation_governance_set_id",
        ),
        Index(
            "ix_plan_selections_selected_governance_result_id",
            "selected_governance_result_id",
        ),
        CheckConstraint(
            "is_current IN (true, false)",
            name="ck_transformation_plan_selections_current_bool",
        ),
        CheckConstraint(
            "selected_with_caution IN (true, false)",
            name="ck_transformation_plan_selections_caution_bool",
        ),
        CheckConstraint(
            "(selected_plan_id IS NULL AND selected_governance_result_id IS NULL) "
            "OR (selected_plan_id IS NOT NULL AND selected_governance_result_id IS NOT NULL)",
            name="ck_transformation_plan_selections_selected_pair",
        ),
        CheckConstraint(
            f"(status IN {_SELECTED_STATUSES} AND selected_plan_id IS NOT NULL) "
            f"OR (status NOT IN {_SELECTED_STATUSES} AND selected_plan_id IS NULL)",
            name="ck_transformation_plan_selections_status_selected",
        ),
        CheckConstraint(
            f"(selected_with_caution AND status = '{_CAUTION_STATUS}') "
            f"OR (NOT selected_with_caution AND status != '{_CAUTION_STATUS}')",
            name="ck_transformation_plan_selections_caution_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clip_candidate_id: Mapped[UUID] = mapped_column(
        ForeignKey("clip_candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    transformation_analysis_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("transformation_eligibility_analyses.id", ondelete="SET NULL"), index=True
    )
    transformation_plan_set_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("transformation_plan_sets.id", ondelete="SET NULL"), index=True
    )
    transformation_governance_set_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("transformation_governance_sets.id", ondelete="SET NULL")
    )
    selected_plan_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("transformation_plans.id", ondelete="SET NULL"), index=True
    )
    selected_governance_result_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("transformation_governance_results.id", ondelete="SET NULL")
    )
    refinement_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("candidate_refinements.id", ondelete="SET NULL"), index=True
    )
    refinement_priority: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    refinement_quality_level: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    refinement_output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )

    status: Mapped[TransformationSelectionStatus] = mapped_column(
        Enum(
            TransformationSelectionStatus,
            name="transformation_selection_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        index=True,
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true", index=True
    )
    selected_with_caution: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    selection_reason_codes: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    arbitration_evidence: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    selected_governance_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    alternative_dispositions: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )

    governance_input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    governance_output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    governor_policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    governor_validation_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    platform_policy_profile_version: Mapped[str] = mapped_column(
        String(80), nullable=False, default="", server_default=""
    )
    selected_plan_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage4.3-v1", server_default="stage4.3-v1"
    )
    schema_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="stage4.3-schema-v1",
        server_default="stage4.3-schema-v1",
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
    analysis: Mapped["TransformationEligibilityAnalysis | None"] = relationship()
    plan_set: Mapped["TransformationPlanSet | None"] = relationship()
    governance_set: Mapped["TransformationGovernanceSet | None"] = relationship()
    selected_plan: Mapped["TransformationPlan | None"] = relationship()
    selected_governance_result: Mapped["TransformationGovernanceResult | None"] = relationship()
    refinement: Mapped["CandidateRefinement | None"] = relationship()


__all__ = ["TransformationPlanSelection"]
