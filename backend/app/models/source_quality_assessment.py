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
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.source_video import SourceVideo


class SourceQualityAssessment(Base):
    """Advisory source-level quality signals for later candidate analysis."""

    __tablename__ = "source_quality_assessments"
    __table_args__ = (
        CheckConstraint(
            "transcript_quality_score >= 0 AND transcript_quality_score <= 1",
            name="ck_source_quality_transcript_quality_score_bounds",
        ),
        CheckConstraint(
            "low_confidence_word_ratio >= 0 AND low_confidence_word_ratio <= 1",
            name="ck_source_quality_low_confidence_word_ratio_bounds",
        ),
        CheckConstraint(
            "unresolved_segment_ratio >= 0 AND unresolved_segment_ratio <= 1",
            name="ck_source_quality_unresolved_segment_ratio_bounds",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), unique=True, index=True
    )
    transcript_confidence: Mapped[float] = mapped_column(Float, default=0)
    transcript_quality_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0, server_default="0"
    )
    low_confidence_word_ratio: Mapped[float] = mapped_column(
        Float, nullable=False, default=0, server_default="0"
    )
    unresolved_segment_ratio: Mapped[float] = mapped_column(
        Float, nullable=False, default=0, server_default="0"
    )
    manual_review_required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    speech_density: Mapped[float] = mapped_column(Float, default=0)
    silence_ratio: Mapped[float] = mapped_column(Float, default=0)
    audio_quality_score: Mapped[float] = mapped_column(Float, default=0)
    preliminary_visual_quality_score: Mapped[float | None] = mapped_column(Float)
    repetition_score: Mapped[float] = mapped_column(Float, default=0)
    estimated_candidate_density: Mapped[float | None] = mapped_column(Float)
    language_confidence: Mapped[float] = mapped_column(Float, default=0)
    overall_source_quality_score: Mapped[float] = mapped_column(Float, default=0)
    reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    source_video: Mapped["SourceVideo"] = relationship(back_populates="quality_assessment")
