from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, Enum, ForeignKey, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import JobKind, JobStatus
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.source_video import SourceVideo


class ProcessingJob(Base):
    """A retryable background operation attached to a source video."""

    __tablename__ = "processing_jobs"
    __table_args__ = (
        CheckConstraint("retry_count >= 0", name="ck_processing_jobs_retry_count_nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[JobKind] = mapped_column(
        Enum(JobKind, name="job_kind", native_enum=False, create_constraint=True),
        nullable=False,
        default=JobKind.INGEST,
        index=True,
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", native_enum=False, create_constraint=True),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True,
    )
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    task_id: Mapped[str | None] = mapped_column(String(255), index=True)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(2048))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    source_video: Mapped["SourceVideo"] = relationship(back_populates="jobs")
