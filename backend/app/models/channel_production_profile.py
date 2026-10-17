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
    Enum,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import SupportingVisualMode, SupportingVisualStyle
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.clip_materialization import ClipMaterialization


class ChannelProductionProfile(Base):
    """Minimal local channel production configuration for Stage 6.

    This is deliberately *not* social-account management: it carries only the
    semantic context, optional narration voice, and supporting-visual defaults a
    transformed short is authored under. Secrets (API keys) never live here; the
    durable identity is the stable ``channel_key``. Each ``(channel_key,
    revision)`` pair is unique with exactly one database-current row per
    ``channel_key``.
    """

    __tablename__ = "channel_production_profiles"
    __table_args__ = (
        UniqueConstraint(
            "channel_key",
            "revision",
            name="uq_channel_production_profiles_key_revision",
        ),
        Index(
            "uq_channel_production_profiles_current_key",
            "channel_key",
            unique=True,
            postgresql_where=text("is_current"),
            sqlite_where=text("is_current"),
        ),
        CheckConstraint(
            "is_current IN (true, false)",
            name="ck_channel_production_profiles_current_bool",
        ),
        CheckConstraint(
            "narration_allowed IN (true, false)",
            name="ck_channel_production_profiles_narration_allowed_bool",
        ),
        CheckConstraint("revision >= 1", name="ck_channel_production_profiles_revision_positive"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_key: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true", index=True
    )
    label: Mapped[str] = mapped_column(String(200), nullable=False, default="", server_default="")

    # Neutral semantic defaults: never a deployment-wide dialect assumption.
    target_market: Mapped[str] = mapped_column(
        String(80), nullable=False, default="UNSPECIFIED", server_default="UNSPECIFIED"
    )
    output_language_policy: Mapped[str] = mapped_column(
        String(80), nullable=False, default="SOURCE_LANGUAGE", server_default="SOURCE_LANGUAGE"
    )
    register_intent: Mapped[str] = mapped_column(
        String(80), nullable=False, default="SOURCE_COMPATIBLE", server_default="SOURCE_COMPATIBLE"
    )
    narration_allowed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    # Persistent narrator identity. Never a random voice per clip.
    tts_provider: Mapped[str] = mapped_column(
        String(80), nullable=False, default="", server_default=""
    )
    tts_model: Mapped[str] = mapped_column(
        String(120), nullable=False, default="", server_default=""
    )
    tts_api_version: Mapped[str] = mapped_column(
        String(40), nullable=False, default="", server_default=""
    )
    tts_voice: Mapped[str] = mapped_column(
        String(120), nullable=False, default="", server_default=""
    )
    tts_style: Mapped[str] = mapped_column(
        String(120), nullable=False, default="", server_default=""
    )
    tts_settings: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )

    supporting_visual_mode: Mapped[SupportingVisualMode] = mapped_column(
        Enum(
            SupportingVisualMode,
            name="supporting_visual_mode",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=SupportingVisualMode.OFF,
        server_default="OFF",
    )
    supporting_visual_style: Mapped[SupportingVisualStyle] = mapped_column(
        Enum(
            SupportingVisualStyle,
            name="supporting_visual_style",
            native_enum=False,
            create_constraint=True,
        ),
        nullable=False,
        default=SupportingVisualStyle.AUTO,
        server_default="AUTO",
    )
    supporting_visual_defaults: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )

    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage6-v1", server_default="stage6-v1"
    )
    schema_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage6-schema-v1", server_default="stage6-schema-v1"
    )
    fingerprint_version: Mapped[str] = mapped_column(
        String(16), nullable=False, default="1", server_default="1"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    materializations: Mapped[list["ClipMaterialization"]] = relationship(
        back_populates="channel_production_profile"
    )


__all__ = ["ChannelProductionProfile"]
