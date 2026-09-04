from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import PipelineStage, RightsStatus
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.pipeline_run import PipelineRun
    from app.models.processing_job import ProcessingJob


class SourceVideo(Base):
    """A source accepted for the local ingest and probe pipeline."""

    __tablename__ = "source_videos"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_uri: Mapped[str] = mapped_column(String(2048), nullable=False, index=True)
    original_filename: Mapped[str | None] = mapped_column(String(512))
    content_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    rights_status: Mapped[RightsStatus] = mapped_column(
        Enum(RightsStatus, name="rights_status", native_enum=False, create_constraint=True),
        nullable=False,
        default=RightsStatus.UNKNOWN,
    )
    lifecycle_state: Mapped[PipelineStage] = mapped_column(
        Enum(
            PipelineStage, name="source_lifecycle_state", native_enum=False, create_constraint=True
        ),
        nullable=False,
        default=PipelineStage.INGEST,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="source_video", cascade="all, delete-orphan"
    )
    pipeline_runs: Mapped[list["PipelineRun"]] = relationship(
        back_populates="source_video", cascade="all, delete-orphan"
    )
