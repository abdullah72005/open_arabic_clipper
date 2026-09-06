from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.source_video import SourceVideo


class AudioAnalysis(Base):
    """Cached silence and lightweight audio features for a source."""

    __tablename__ = "audio_analyses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), unique=True, index=True
    )
    audio_hash: Mapped[str] = mapped_column(index=True)
    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    silence_intervals: Mapped[list[dict[str, float]]] = mapped_column(JSON, default=list)
    features: Mapped[list[dict[str, float]]] = mapped_column(JSON, default=list)
    silence_ratio: Mapped[float] = mapped_column(Float, default=0)
    speech_density: Mapped[float] = mapped_column(Float, default=0)
    speech_rate: Mapped[float] = mapped_column(Float, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    source_video: Mapped["SourceVideo"] = relationship(back_populates="audio_analysis")
