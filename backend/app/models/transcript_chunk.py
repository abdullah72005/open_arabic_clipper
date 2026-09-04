"""Persisted semantic transcript chunks for later analysis stages."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.transcript import Transcript


class TranscriptChunk(Base):
    """A timestamp-bound, context-preserving set of transcript segments."""

    __tablename__ = "transcript_chunks"
    __table_args__ = (UniqueConstraint("transcript_id", "sequence"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    transcript_id: Mapped[UUID] = mapped_column(
        ForeignKey("transcripts.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    segment_indexes: Mapped[list[int]] = mapped_column(JSON, default=list)
    preceding_context: Mapped[str] = mapped_column(Text, default="")
    following_context: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    transcript: Mapped["Transcript"] = relationship(back_populates="chunks")
