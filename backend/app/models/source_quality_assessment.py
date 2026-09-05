from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.source_video import SourceVideo


class SourceQualityAssessment(Base):
    """Advisory source-level quality signals for later candidate analysis."""

    __tablename__ = "source_quality_assessments"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), unique=True, index=True
    )
    transcript_confidence: Mapped[float] = mapped_column(Float, default=0)
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
