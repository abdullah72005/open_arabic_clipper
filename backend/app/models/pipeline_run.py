from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import PipelineRunStatus, PipelineStage
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.source_video import SourceVideo


class PipelineRun(Base):
    """One resumable attempt to execute a pipeline stage for a source."""

    __tablename__ = "pipeline_runs"
    __table_args__ = (
        CheckConstraint("attempt >= 1", name="ck_pipeline_runs_attempt_positive"),
        UniqueConstraint(
            "source_video_id", "stage", "attempt", name="uq_pipeline_runs_source_stage_attempt"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[PipelineStage] = mapped_column(
        Enum(PipelineStage, name="pipeline_stage", native_enum=False, create_constraint=True),
        nullable=False,
        default=PipelineStage.INGEST,
        index=True,
    )
    status: Mapped[PipelineRunStatus] = mapped_column(
        Enum(
            PipelineRunStatus, name="pipeline_run_status", native_enum=False, create_constraint=True
        ),
        nullable=False,
        default=PipelineRunStatus.QUEUED,
        index=True,
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    error_message: Mapped[str | None] = mapped_column(Text())
    input_fingerprint: Mapped[str | None] = mapped_column(String(64))
    output_fingerprint: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    source_video: Mapped["SourceVideo"] = relationship(back_populates="pipeline_runs")
