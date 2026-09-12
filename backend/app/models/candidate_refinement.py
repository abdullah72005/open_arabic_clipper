from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import RefinementPriority, RefinementStatus
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.clip_candidate import ClipCandidate
    from app.models.source_video import SourceVideo


class CandidateRefinement(Base):
    """One durable candidate-scoped refinement at a single quality level.

    A ``(clip_candidate_id, priority)`` pair has at most one row. Stage 3.5 only
    accepts the ``CANDIDATE`` and ``FINAL_CLIP`` priorities; ``INDEX`` remains a
    whole-source Stage 2.7 concept. Coarse Stage 3 bounds, context-window bounds,
    and refined clip bounds are persisted separately so context audio never
    silently becomes the final clip.
    """

    __tablename__ = "candidate_refinements"
    __table_args__ = (
        UniqueConstraint(
            "clip_candidate_id",
            "priority",
            name="uq_candidate_refinements_candidate_priority",
        ),
        CheckConstraint(
            "coarse_start >= 0 AND coarse_end > coarse_start",
            name="ck_candidate_refinements_coarse_range",
        ),
        CheckConstraint(
            "context_start >= 0 AND context_end > context_start",
            name="ck_candidate_refinements_context_range",
        ),
        CheckConstraint(
            "refined_start >= 0 AND refined_end >= refined_start",
            name="ck_candidate_refinements_refined_range",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_candidate_refinements_confidence",
        ),
        CheckConstraint(
            "dialect_confidence >= 0 AND dialect_confidence <= 1",
            name="ck_candidate_refinements_dialect_confidence",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clip_candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("clip_candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    priority: Mapped[RefinementPriority] = mapped_column(
        Enum(
            RefinementPriority,
            name="refinement_priority",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        index=True,
    )
    status: Mapped[RefinementStatus] = mapped_column(
        Enum(
            RefinementStatus,
            name="refinement_status",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=RefinementStatus.QUEUED,
        index=True,
    )
    active_job_id: Mapped[str | None] = mapped_column(String(64), index=True)

    coarse_start: Mapped[float] = mapped_column(Float, nullable=False)
    coarse_end: Mapped[float] = mapped_column(Float, nullable=False)
    context_start: Mapped[float] = mapped_column(Float, nullable=False)
    context_end: Mapped[float] = mapped_column(Float, nullable=False)
    refined_start: Mapped[float | None] = mapped_column(Float)
    refined_end: Mapped[float | None] = mapped_column(Float)

    audio_relative_path: Mapped[str | None] = mapped_column(String(1024))
    audio_content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    audio_input_fingerprint: Mapped[str | None] = mapped_column(String(64))

    automatic_transcript: Mapped[str] = mapped_column(Text, nullable=False, default="")
    manual_transcript: Mapped[str | None] = mapped_column(Text)
    final_transcript: Mapped[str] = mapped_column(Text, nullable=False, default="")
    word_timestamps: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list
    )

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    quality_level: Mapped[str] = mapped_column(String(32), nullable=False, default="CANDIDATE")

    dialect_profile: Mapped[str | None] = mapped_column(String(32))
    dialect_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    code_switch_evidence: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    transcript_evidence: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    entity_evidence: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    unresolved_spans: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    provider_evidence: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    routing_evidence: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)

    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    output_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    component_fingerprints: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False, default="stage3.5-v1")
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False, default="stage3.5-v1")
    validation_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage3.5-validation-v1"
    )
    cache_eligible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    metrics: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    processing_duration: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    source_video: Mapped["SourceVideo"] = relationship(back_populates="candidate_refinements")
    clip_candidate: Mapped["ClipCandidate"] = relationship(back_populates="refinements")
