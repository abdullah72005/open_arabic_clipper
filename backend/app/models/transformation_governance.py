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
    Float,
    ForeignKey,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import (
    GovernanceExecutionStatus,
    GovernancePlanStatus,
    GovernanceSemanticOutcome,
    SemanticProviderMode,
)
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.candidate_refinement import CandidateRefinement
    from app.models.clip_candidate import ClipCandidate
    from app.models.source_video import SourceVideo
    from app.models.transformation_eligibility import TransformationEligibilityAnalysis
    from app.models.transformation_plan import TransformationPlan, TransformationPlanSet

_ELIGIBLE_STATUSES = "('APPROVED_FOR_SELECTION', 'APPROVED_WITH_CAUTION')"


class TransformationGovernanceSet(Base):
    """One durable Stage 4.2 governance envelope per Stage 4.1 plan set.

    Owns execution lifecycle, candidate-level semantic outcome, fingerprints,
    provider identity/evidence, policy profile, cache state, metrics, active job
    ownership, and bounded per-plan attempt/checkpoint state. It never selects a
    plan and never mutates the Stage 4.1 plan set.
    """

    __tablename__ = "transformation_governance_sets"
    __table_args__ = (
        CheckConstraint(
            "cache_eligible IN (true, false)",
            name="ck_transformation_governance_sets_cache_eligible_bool",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clip_candidate_id: Mapped[UUID] = mapped_column(
        ForeignKey("clip_candidates.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    transformation_plan_set_id: Mapped[UUID] = mapped_column(
        ForeignKey("transformation_plan_sets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    transformation_analysis_id: Mapped[UUID] = mapped_column(
        ForeignKey("transformation_eligibility_analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    refinement_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("candidate_refinements.id", ondelete="SET NULL"), index=True
    )
    refinement_priority: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    refinement_quality_level: Mapped[str] = mapped_column(String(32), nullable=False, default="")

    execution_status: Mapped[GovernanceExecutionStatus] = mapped_column(
        Enum(
            GovernanceExecutionStatus,
            name="transformation_governance_execution_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=GovernanceExecutionStatus.QUEUED,
        index=True,
    )
    governance_outcome: Mapped[GovernanceSemanticOutcome | None] = mapped_column(
        Enum(
            GovernanceSemanticOutcome,
            name="transformation_governance_semantic_outcome",
            native_enum=False,
            create_constraint=True,
        ),
        index=True,
    )
    outcome_reasons: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    summary_counts: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    stage40_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    target_context: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )

    provider_mode: Mapped[SemanticProviderMode] = mapped_column(
        Enum(
            SemanticProviderMode,
            name="transformation_governance_provider_mode",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=SemanticProviderMode.DETERMINISTIC,
        server_default=SemanticProviderMode.DETERMINISTIC.value,
    )
    provider_identity: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    provider_status: Mapped[str] = mapped_column(
        String(64), nullable=False, default="DETERMINISTIC", server_default="DETERMINISTIC"
    )
    provider_evidence: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    plan_attempts: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )

    platform_policy_profile_version: Mapped[str] = mapped_column(
        String(80), nullable=False, default="", server_default=""
    )
    platform_policy_checked_at: Mapped[str] = mapped_column(
        String(16), nullable=False, default="", server_default=""
    )
    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage4.2-v1", server_default="stage4.2-v1"
    )
    schema_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="stage4.2-schema-v1",
        server_default="stage4.2-schema-v1",
    )
    validation_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="stage4.2-validation-v1",
        server_default="stage4.2-validation-v1",
    )
    cache_eligible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    active_job_id: Mapped[str | None] = mapped_column(String(64), index=True)
    metrics: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    processing_duration: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    source_video: Mapped["SourceVideo"] = relationship()
    clip_candidate: Mapped["ClipCandidate"] = relationship()
    plan_set: Mapped["TransformationPlanSet"] = relationship()
    analysis: Mapped["TransformationEligibilityAnalysis"] = relationship()
    refinement: Mapped["CandidateRefinement | None"] = relationship()
    results: Mapped[list["TransformationGovernanceResult"]] = relationship(
        back_populates="governance_set", cascade="all, delete-orphan"
    )


class TransformationGovernanceResult(Base):
    """Independent Stage 4.2 governance result for one current Stage 4.1 plan.

    Stores only validated, bounded, explainable governance evidence. The Stage
    4.1 ``TransformationPlan`` row is never modified.
    """

    __tablename__ = "transformation_governance_results"
    __table_args__ = (
        UniqueConstraint(
            "governance_set_id",
            "transformation_plan_id",
            name="uq_transformation_governance_results_set_plan",
        ),
        CheckConstraint(
            "eligible_for_stage4_3 IN (true, false)",
            name="ck_transformation_governance_results_eligible_bool",
        ),
        CheckConstraint(
            f"(eligible_for_stage4_3 AND status IN {_ELIGIBLE_STATUSES}) "
            f"OR (NOT eligible_for_stage4_3 AND status NOT IN {_ELIGIBLE_STATUSES})",
            name="ck_transformation_governance_results_status_eligibility",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    governance_set_id: Mapped[UUID] = mapped_column(
        ForeignKey("transformation_governance_sets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    transformation_plan_id: Mapped[UUID] = mapped_column(
        ForeignKey("transformation_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan_output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    status: Mapped[GovernancePlanStatus] = mapped_column(
        Enum(
            GovernancePlanStatus,
            name="transformation_governance_plan_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
    )
    eligible_for_stage4_3: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false", index=True
    )
    severity: Mapped[str] = mapped_column(
        String(32), nullable=False, default="ADVISORY", server_default="ADVISORY"
    )

    hard_gates: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    dimensions: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    verification: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    platform_risk: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    reason_codes: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    warnings: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    remediation: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    governance_provider_evidence: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    governance_set: Mapped["TransformationGovernanceSet"] = relationship(back_populates="results")
    plan: Mapped["TransformationPlan"] = relationship()
