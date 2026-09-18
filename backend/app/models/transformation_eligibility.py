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
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import (
    ExternalFactRequirement,
    SemanticProviderMode,
    StrategyDisposition,
    StrategyOrigin,
    SubstantiveValueKind,
    TransformationEligibilityOutcome,
    TransformationExecutionStatus,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.candidate_refinement import CandidateRefinement
    from app.models.clip_candidate import ClipCandidate
    from app.models.source_video import SourceVideo


class TransformationEligibilityAnalysis(Base):
    """One reusable current Stage 4.0 eligibility analysis per candidate.

    Holds bounded decision evidence only: the selected refinement identity/quality,
    eligibility state, independent assessments, source-moment evidence, platform
    risk snapshot, provider identity/status, fingerprints, and cache lifecycle.
    The full transcript is never duplicated here; the selected refinement is
    referenced instead.
    """

    __tablename__ = "transformation_eligibility_analyses"
    __table_args__ = (
        CheckConstraint(
            "cache_eligible IN (true, false)",
            name="ck_transformation_analyses_cache_eligible_bool",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clip_candidate_id: Mapped[UUID] = mapped_column(
        ForeignKey("clip_candidates.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    refinement_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("candidate_refinements.id", ondelete="SET NULL"), index=True
    )
    refinement_priority: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    refinement_quality_level: Mapped[str] = mapped_column(String(32), nullable=False, default="")

    execution_status: Mapped[TransformationExecutionStatus] = mapped_column(
        Enum(
            TransformationExecutionStatus,
            name="transformation_execution_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=TransformationExecutionStatus.QUEUED,
        index=True,
    )
    eligibility_outcome: Mapped[TransformationEligibilityOutcome | None] = mapped_column(
        Enum(
            TransformationEligibilityOutcome,
            name="transformation_eligibility_outcome",
            native_enum=False,
            create_constraint=True,
        ),
        index=True,
    )
    eligibility_reasons: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    assessments: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    source_moment: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    platform_risk: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    transformation_intensity: Mapped[TransformationIntensity | None] = mapped_column(
        Enum(
            TransformationIntensity,
            name="transformation_intensity",
            native_enum=False,
            create_constraint=True,
        )
    )

    provider_mode: Mapped[SemanticProviderMode] = mapped_column(
        Enum(
            SemanticProviderMode,
            name="transformation_provider_mode",
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
    provider_input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )

    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage4.0-v1", server_default="stage4.0-v1"
    )
    schema_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="stage4.0-schema-v1",
        server_default="stage4.0-schema-v1",
    )
    validation_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="stage4.0-validation-v1",
        server_default="stage4.0-validation-v1",
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
    refinement: Mapped["CandidateRefinement | None"] = relationship()
    strategies: Mapped[list["TransformationStrategyCandidate"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


class TransformationStrategyCandidate(Base):
    """One stable Stage 4.0 strategy direction for an eligibility analysis.

    Exactly one current row per ``(analysis_id, strategy_type)``. Reruns upsert
    matching types (stable UUIDs); strategies absent from a successful finalized
    analysis are marked non-current, never deleted mid-history.
    """

    __tablename__ = "transformation_strategy_candidates"
    __table_args__ = (
        UniqueConstraint(
            "analysis_id",
            "strategy_type",
            name="uq_transformation_strategies_analysis_type",
        ),
        CheckConstraint("rank >= 0", name="ck_transformation_strategies_rank_nonnegative"),
        CheckConstraint(
            "retention_preservation >= 0 AND retention_preservation <= 1",
            name="ck_transformation_strategies_retention",
        ),
        CheckConstraint(
            "source_moment_damage_risk >= 0 AND source_moment_damage_risk <= 1",
            name="ck_transformation_strategies_damage",
        ),
        CheckConstraint(
            "added_value_density >= 0 AND added_value_density <= 1",
            name="ck_transformation_strategies_added_value",
        ),
        CheckConstraint(
            "originality_potential >= 0 AND originality_potential <= 1",
            name="ck_transformation_strategies_originality",
        ),
        CheckConstraint(
            "source_dominance_risk >= 0 AND source_dominance_risk <= 1",
            name="ck_transformation_strategies_source_dominance",
        ),
        CheckConstraint(
            "generic_filler_risk >= 0 AND generic_filler_risk <= 1",
            name="ck_transformation_strategies_filler",
        ),
        CheckConstraint(
            "redundant_commentary_risk >= 0 AND redundant_commentary_risk <= 1",
            name="ck_transformation_strategies_redundancy",
        ),
        CheckConstraint(
            "template_staleness_risk >= 0 AND template_staleness_risk <= 1",
            name="ck_transformation_strategies_template",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_transformation_strategies_confidence",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[UUID] = mapped_column(
        ForeignKey("transformation_eligibility_analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strategy_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    strategy_type: Mapped[TransformationStrategyType] = mapped_column(
        Enum(
            TransformationStrategyType,
            name="transformation_strategy_type",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        index=True,
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true", index=True
    )
    disposition: Mapped[StrategyDisposition] = mapped_column(
        Enum(
            StrategyDisposition,
            name="transformation_strategy_disposition",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        index=True,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    intensity: Mapped[TransformationIntensity] = mapped_column(
        Enum(
            TransformationIntensity,
            name="transformation_intensity",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=TransformationIntensity.MINIMAL,
    )

    direction_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    added_value_focus: Mapped[str] = mapped_column(Text, nullable=False, default="")
    substantive_value_kind: Mapped[SubstantiveValueKind] = mapped_column(
        Enum(
            SubstantiveValueKind,
            name="transformation_substantive_value_kind",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=SubstantiveValueKind.MISSING_CONTEXT,
    )
    source_moment_role: Mapped[str] = mapped_column(String(64), nullable=False, default="HERO")
    preservation_requirements: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )

    retention_preservation: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    source_moment_damage_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    added_value_density: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    originality_potential: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    source_dominance_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    generic_filler_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    redundant_commentary_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    template_staleness_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0)

    external_verification_requirement: Mapped[ExternalFactRequirement] = mapped_column(
        Enum(
            ExternalFactRequirement,
            name="transformation_external_fact_requirement",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=ExternalFactRequirement.NOT_REQUIRED,
    )
    verification_requirements: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    rejection_reasons: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    origin: Mapped[StrategyOrigin] = mapped_column(
        Enum(
            StrategyOrigin,
            name="transformation_strategy_origin",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=StrategyOrigin.DETERMINISTIC,
    )
    provider_evidence: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    strategy_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage4.0-v1", server_default="stage4.0-v1"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    analysis: Mapped["TransformationEligibilityAnalysis"] = relationship(
        back_populates="strategies"
    )
