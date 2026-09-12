from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import SemanticProviderMode
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.source_video import SourceVideo


class CandidateAnalysis(Base):
    """One-to-one Stage 3 analysis summary for a source.

    Holds only execution/cache-lifecycle fields: fingerprints, policy/scoring
    versions, stable provider identity (never credentials or transient
    availability), a sanitized provider status, cache eligibility, bounded
    metrics, and timing.
    """

    __tablename__ = "candidate_analyses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), unique=True, index=True
    )
    input_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    output_fingerprint: Mapped[str] = mapped_column(
        String(64), nullable=False, default="", server_default=""
    )
    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage3-v1", server_default="stage3-v1"
    )
    scoring_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage3-scoring-v1", server_default="stage3-scoring-v1"
    )
    semantic_provider_mode: Mapped[SemanticProviderMode] = mapped_column(
        Enum(
            SemanticProviderMode,
            name="semantic_provider_mode",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=SemanticProviderMode.DETERMINISTIC,
        server_default=SemanticProviderMode.DETERMINISTIC.value,
    )
    provider_identity: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    provider_status: Mapped[str] = mapped_column(
        String(64), nullable=False, default="DETERMINISTIC", server_default="DETERMINISTIC"
    )
    cache_eligible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    metrics: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    processing_duration: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    source_video: Mapped["SourceVideo"] = relationship(back_populates="candidate_analysis")
