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
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import (
    PlanExecutionStatus,
    PlanSemanticOutcome,
    PlanStatus,
    SemanticProviderMode,
    StrategyOrigin,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.candidate_refinement import CandidateRefinement
    from app.models.clip_candidate import ClipCandidate
    from app.models.source_video import SourceVideo
    from app.models.transformation_eligibility import (
        TransformationEligibilityAnalysis,
        TransformationStrategyCandidate,
    )


class TransformationPlanSet(Base):
    """One durable Stage 4.1 planning envelope per candidate/current analysis.

    Represents successful zero-plan, deferred, degraded, cancelled, and failed
    execution truthfully without manufacturing fake ``TransformationPlan`` rows.
    """

    __tablename__ = "transformation_plan_sets"
    __table_args__ = (
        CheckConstraint(
            "cache_eligible IN (0, 1)",
            name="ck_transformation_plan_sets_cache_eligible_bool",
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
    transformation_analysis_id: Mapped[UUID] = mapped_column(
        ForeignKey("transformation_eligibility_analyses.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    refinement_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("candidate_refinements.id", ondelete="SET NULL"), index=True
    )
    refinement_priority: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    refinement_quality_level: Mapped[str] = mapped_column(String(32), nullable=False, default="")

    execution_status: Mapped[PlanExecutionStatus] = mapped_column(
        Enum(
            PlanExecutionStatus,
            name="transformation_plan_execution_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=PlanExecutionStatus.QUEUED,
        index=True,
    )
    planning_outcome: Mapped[PlanSemanticOutcome | None] = mapped_column(
        Enum(
            PlanSemanticOutcome,
            name="transformation_plan_semantic_outcome",
            native_enum=False,
            create_constraint=True,
        ),
        index=True,
    )
    outcome_reasons: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
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
            name="transformation_planning_provider_mode",
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
    strategy_attempts: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )

    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage4.1-v1", server_default="stage4.1-v1"
    )
    schema_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="stage4.1-schema-v1",
        server_default="stage4.1-schema-v1",
    )
    validation_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="stage4.1-validation-v1",
        server_default="stage4.1-validation-v1",
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
    analysis: Mapped["TransformationEligibilityAnalysis"] = relationship()
    refinement: Mapped["CandidateRefinement | None"] = relationship()
    plans: Mapped[list["TransformationPlan"]] = relationship(
        back_populates="plan_set", cascade="all, delete-orphan"
    )


class TransformationPlan(Base):
    """One validated concrete Stage 4.1 plan for a current Stage 4.0 strategy."""

    __tablename__ = "transformation_plans"
    __table_args__ = (
        UniqueConstraint(
            "plan_set_id",
            "strategy_candidate_id",
            name="uq_transformation_plans_set_strategy",
        ),
        CheckConstraint("generation_rank >= 1", name="ck_transformation_plans_rank_positive"),
        CheckConstraint("hero_block_index >= 0", name="ck_transformation_plans_hero_nonnegative"),
        CheckConstraint(
            "planner_confidence >= 0 AND planner_confidence <= 1",
            name="ck_transformation_plans_confidence",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plan_set_id: Mapped[UUID] = mapped_column(
        ForeignKey("transformation_plan_sets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strategy_candidate_id: Mapped[UUID] = mapped_column(
        ForeignKey("transformation_strategy_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan_key: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true", index=True
    )
    status: Mapped[PlanStatus] = mapped_column(
        Enum(
            PlanStatus,
            name="transformation_plan_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
    )
    generation_rank: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    strategy_type: Mapped[TransformationStrategyType] = mapped_column(
        Enum(
            TransformationStrategyType,
            name="transformation_plan_strategy_type",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
    )
    intensity: Mapped[TransformationIntensity] = mapped_column(
        Enum(
            TransformationIntensity,
            name="transformation_plan_intensity",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=TransformationIntensity.MINIMAL,
    )
    strategy_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    strategy_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    source_dialect: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    target_audience: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )

    blocks: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    hero_block_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hero_source_start: Mapped[float | None] = mapped_column(Float)
    hero_source_end: Mapped[float | None] = mapped_column(Float)
    hero_appearance_time: Mapped[float] = mapped_column(Float, nullable=False, default=0)

    preservation_constraints: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    original_value_kinds: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    original_value_reasons: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    narration_need: Mapped[str] = mapped_column(
        String(32), nullable=False, default="NONE", server_default="NONE"
    )
    narration_requirements: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    external_fact_dependencies: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    required_context: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    derived_durations: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    hook_payoff_evidence: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    degraded_rules: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    stage40_risk: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )

    planner_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    generation_origin: Mapped[StrategyOrigin] = mapped_column(
        Enum(
            StrategyOrigin,
            name="transformation_plan_origin",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=StrategyOrigin.DETERMINISTIC,
    )
    planning_provider_evidence: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    provider_input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    plan_output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    structure_signature: Mapped[str] = mapped_column(
        String(512), nullable=False, default="", server_default=""
    )
    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage4.1-v1", server_default="stage4.1-v1"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    plan_set: Mapped["TransformationPlanSet"] = relationship(back_populates="plans")
    strategy_candidate: Mapped["TransformationStrategyCandidate"] = relationship()
