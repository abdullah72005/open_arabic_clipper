from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.source_video import SourceVideo
    from app.models.transcript_chunk import TranscriptChunk


class Transcript(Base):
    """Current reusable timestamped transcript for a source."""

    __tablename__ = "transcripts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), unique=True, index=True
    )
    language: Mapped[str | None] = mapped_column(String(32))
    detected_language_probability: Mapped[float | None] = mapped_column(Float)
    whisper_model: Mapped[str] = mapped_column(String(64))
    transcription_options: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    input_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    normalized_text: Mapped[str] = mapped_column(Text, default="")
    corrected_text: Mapped[str] = mapped_column(Text, default="")
    contextual_reconstructed_text: Mapped[str] = mapped_column(Text, default="")
    final_text: Mapped[str] = mapped_column(Text, default="")
    raw_transcript_confidence: Mapped[float] = mapped_column(Float, default=0)
    correction_confidence: Mapped[float] = mapped_column(Float, default=0)
    corrected_segment_ratio: Mapped[float] = mapped_column(Float, default=0)
    uncertain_segment_ratio: Mapped[float] = mapped_column(Float, default=0)
    correction_method: Mapped[str] = mapped_column(String(64), default="pending")
    correction_version: Mapped[str] = mapped_column(String(64), default="pending")
    reconstruction_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    reconstruction_confidence: Mapped[float] = mapped_column(Float, default=0)
    reconstructed_segment_ratio: Mapped[float] = mapped_column(Float, default=0)
    reconstruction_method: Mapped[str] = mapped_column(String(64), default="pending")
    reconstruction_version: Mapped[str] = mapped_column(String(64), default="pending")
    reconstruction_processing_duration: Mapped[float | None] = mapped_column(Float)
    reconstruction_metadata: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    segments: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    word_segments: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    duration: Mapped[float] = mapped_column(Float, default=0)
    processing_duration: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    source_video: Mapped["SourceVideo"] = relationship(back_populates="transcript")
    chunks: Mapped[list["TranscriptChunk"]] = relationship(
        back_populates="transcript",
        cascade="all, delete-orphan",
        order_by="TranscriptChunk.sequence",
    )
