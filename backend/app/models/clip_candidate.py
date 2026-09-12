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
    CandidateDisposition,
    ContentType,
    OriginalityRisk,
    RightsRisk,
)
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.candidate_analysis import CandidateAnalysis
    from app.models.candidate_refinement import CandidateRefinement
    from app.models.source_video import SourceVideo

_SCORE_COLUMNS = (
    "clip_score",
    "short_form_score",
    "moment_density_score",
    "boredom_risk_score",
    "ending_quality_score",
    "loopability_score",
    "engagement_confidence",
    "transcript_confidence",
    "audio_confidence",
    "boundary_confidence",
    "uncertainty_severity",
    "idea_novelty_score",
    "topic_novelty_score",
    "recent_semantic_similarity_risk",
)


class ClipCandidate(Base):
    """A persisted Stage 3 coarse candidate or explainable rejected proposal."""

    __tablename__ = "clip_candidates"
    __table_args__ = (
        CheckConstraint(
            "start_time >= 0 AND end_time > start_time", name="ck_candidates_time_range"
        ),
        CheckConstraint(
            "start_segment_index >= 0 AND end_segment_index >= start_segment_index",
            name="ck_candidates_segment_range",
        ),
        CheckConstraint(
            "dialect_confidence >= 0 AND dialect_confidence <= 1", name="ck_cand_dialect"
        ),
        UniqueConstraint("candidate_key", name="uq_clip_candidates_candidate_key"),
        *(
            CheckConstraint(
                f"{column} >= 0 AND {column} <= 1",
                name=f"ck_clip_candidates_{column}_bounds",
            )
            for column in _SCORE_COLUMNS
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    candidate_analysis_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("candidate_analyses.id", ondelete="SET NULL"), index=True
    )
    # Deterministic identity from source UUID + immutable contiguous segment indexes.
    candidate_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true", index=True
    )
    disposition: Mapped[CandidateDisposition] = mapped_column(
        Enum(
            CandidateDisposition,
            name="candidate_disposition",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        index=True,
    )
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    start_segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    end_segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    segment_indexes: Mapped[list[int]] = mapped_column(JSON, nullable=False, default=list)
    transcript_excerpt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evidence_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    primary_content_type: Mapped[ContentType] = mapped_column(
        Enum(ContentType, name="content_type", native_enum=False, create_constraint=True),
        nullable=False,
        default=ContentType.OTHER,
    )
    secondary_content_types: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    clip_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    short_form_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    moment_density_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    boredom_risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    ending_quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    loopability_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    engagement_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    transcript_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    audio_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    boundary_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    uncertainty_severity: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    idea_novelty_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    topic_novelty_score: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    recent_semantic_similarity_risk: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    refinement_reasons: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    refinement_evidence: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    provenance_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    rights_risk: Mapped[RightsRisk] = mapped_column(
        Enum(RightsRisk, name="rights_risk", native_enum=False, create_constraint=True),
        nullable=False,
        default=RightsRisk.UNDETERMINED,
    )
    originality_risk: Mapped[OriginalityRisk] = mapped_column(
        Enum(
            OriginalityRisk,
            name="originality_risk",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=OriginalityRisk.UNDETERMINED,
    )
    dialect_profile: Mapped[str | None] = mapped_column(String(32))
    dialect_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    code_switch_suspected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    hooks: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    idea_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    topic_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    idea_signature: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    topic_signature: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    provider_input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    provider_evidence: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    analysis_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False, default="stage3-v1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    source_video: Mapped["SourceVideo"] = relationship(back_populates="candidates")
    candidate_analysis: Mapped["CandidateAnalysis | None"] = relationship()
    refinements: Mapped[list["CandidateRefinement"]] = relationship(
        back_populates="clip_candidate", cascade="all, delete-orphan"
    )
