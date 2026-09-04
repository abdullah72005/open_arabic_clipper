"""Add durable Stage 2 transcription and analysis state.

Revision ID: 20260904_0003
Revises: 20260904_0002
Create Date: 2026-09-04 16:15:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260904_0003"
down_revision: str | None = "20260904_0002"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_LIFECYCLE_BEFORE = ("INGEST", "PROBE", "READY_FOR_TRANSCRIPTION")
_LIFECYCLE_AFTER = (
    *_LIFECYCLE_BEFORE,
    "AUDIO_EXTRACTION",
    "TRANSCRIPTION",
    "TRANSCRIPT_NORMALIZATION",
    "AUDIO_ANALYSIS",
    "READY_FOR_ANALYSIS",
)
_JOB_BEFORE = ("INGEST", "PROBE")
_JOB_AFTER = (*_JOB_BEFORE, "TRANSCRIPTION")
_PIPELINE_BEFORE = ("INGEST", "PROBE", "READY_FOR_TRANSCRIPTION")
_PIPELINE_AFTER = (
    *_PIPELINE_BEFORE,
    "AUDIO_EXTRACTION",
    "TRANSCRIPTION",
    "TRANSCRIPT_NORMALIZATION",
    "AUDIO_ANALYSIS",
    "READY_FOR_ANALYSIS",
)


def _enum(name: str, values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True)


def upgrade() -> None:
    with op.batch_alter_table("source_videos") as batch:
        batch.alter_column(
            "lifecycle_state",
            existing_type=_enum("source_lifecycle_state", _LIFECYCLE_BEFORE),
            type_=_enum("source_lifecycle_state", _LIFECYCLE_AFTER),
            existing_nullable=False,
        )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_BEFORE),
            type_=_enum("job_kind", _JOB_AFTER),
            existing_nullable=False,
        )
    with op.batch_alter_table("pipeline_runs") as batch:
        batch.alter_column(
            "stage",
            existing_type=_enum("pipeline_stage", _PIPELINE_BEFORE),
            type_=_enum("pipeline_stage", _PIPELINE_AFTER),
            existing_nullable=False,
        )

    op.create_table(
        "audio_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_video_id", sa.Uuid(), nullable=False),
        sa.Column("output_path", sa.String(length=1024), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("source_content_hash", sa.String(length=64)),
        sa.Column("sample_rate", sa.Integer(), nullable=False),
        sa.Column("duration", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_video_id"], ["source_videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_video_id"),
    )
    op.create_index("ix_audio_artifacts_source_video_id", "audio_artifacts", ["source_video_id"])
    op.create_index("ix_audio_artifacts_content_hash", "audio_artifacts", ["content_hash"])
    op.create_index(
        "ix_audio_artifacts_source_content_hash", "audio_artifacts", ["source_content_hash"]
    )

    op.create_table(
        "transcripts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_video_id", sa.Uuid(), nullable=False),
        sa.Column("language", sa.String(length=32)),
        sa.Column("detected_language_probability", sa.Float()),
        sa.Column("whisper_model", sa.String(length=64), nullable=False),
        sa.Column("transcription_options", sa.JSON(), nullable=False),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("segments", sa.JSON(), nullable=False),
        sa.Column("word_segments", sa.JSON(), nullable=False),
        sa.Column("duration", sa.Float(), nullable=False),
        sa.Column("processing_duration", sa.Float()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_video_id"], ["source_videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_video_id"),
    )
    op.create_index("ix_transcripts_source_video_id", "transcripts", ["source_video_id"])
    op.create_index("ix_transcripts_input_fingerprint", "transcripts", ["input_fingerprint"])

    op.create_table(
        "audio_analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_video_id", sa.Uuid(), nullable=False),
        sa.Column("audio_hash", sa.String(length=64), nullable=False),
        sa.Column("silence_intervals", sa.JSON(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("silence_ratio", sa.Float(), nullable=False),
        sa.Column("speech_density", sa.Float(), nullable=False),
        sa.Column("speech_rate", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_video_id"], ["source_videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_video_id"),
    )
    op.create_index("ix_audio_analyses_source_video_id", "audio_analyses", ["source_video_id"])
    op.create_index("ix_audio_analyses_audio_hash", "audio_analyses", ["audio_hash"])

    op.create_table(
        "source_quality_assessments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_video_id", sa.Uuid(), nullable=False),
        sa.Column("transcript_confidence", sa.Float(), nullable=False),
        sa.Column("speech_density", sa.Float(), nullable=False),
        sa.Column("silence_ratio", sa.Float(), nullable=False),
        sa.Column("audio_quality_score", sa.Float(), nullable=False),
        sa.Column("preliminary_visual_quality_score", sa.Float()),
        sa.Column("repetition_score", sa.Float(), nullable=False),
        sa.Column("estimated_candidate_density", sa.Float()),
        sa.Column("language_confidence", sa.Float(), nullable=False),
        sa.Column("overall_source_quality_score", sa.Float(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_video_id"], ["source_videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_video_id"),
    )
    op.create_index(
        "ix_source_quality_assessments_source_video_id",
        "source_quality_assessments",
        ["source_video_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_source_quality_assessments_source_video_id", table_name="source_quality_assessments"
    )
    op.drop_table("source_quality_assessments")
    op.drop_index("ix_audio_analyses_audio_hash", table_name="audio_analyses")
    op.drop_index("ix_audio_analyses_source_video_id", table_name="audio_analyses")
    op.drop_table("audio_analyses")
    op.drop_index("ix_transcripts_input_fingerprint", table_name="transcripts")
    op.drop_index("ix_transcripts_source_video_id", table_name="transcripts")
    op.drop_table("transcripts")
    op.drop_index("ix_audio_artifacts_source_content_hash", table_name="audio_artifacts")
    op.drop_index("ix_audio_artifacts_content_hash", table_name="audio_artifacts")
    op.drop_index("ix_audio_artifacts_source_video_id", table_name="audio_artifacts")
    op.drop_table("audio_artifacts")

    with op.batch_alter_table("pipeline_runs") as batch:
        batch.alter_column(
            "stage",
            existing_type=_enum("pipeline_stage", _PIPELINE_AFTER),
            type_=_enum("pipeline_stage", _PIPELINE_BEFORE),
            existing_nullable=False,
        )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_AFTER),
            type_=_enum("job_kind", _JOB_BEFORE),
            existing_nullable=False,
        )
    with op.batch_alter_table("source_videos") as batch:
        batch.alter_column(
            "lifecycle_state",
            existing_type=_enum("source_lifecycle_state", _LIFECYCLE_AFTER),
            type_=_enum("source_lifecycle_state", _LIFECYCLE_BEFORE),
            existing_nullable=False,
        )
