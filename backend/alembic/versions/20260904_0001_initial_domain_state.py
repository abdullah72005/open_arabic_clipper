"""Create durable Stage 1 domain state.

Revision ID: 20260904_0001
Revises:
Create Date: 2026-09-04 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260904_0001"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    rights_status = sa.Enum(
        "OWNED",
        "AUTHORIZED",
        "UNKNOWN",
        name="rights_status",
        native_enum=False,
        create_constraint=True,
    )
    source_lifecycle_state = sa.Enum(
        "INGEST",
        "PROBE",
        "READY_FOR_TRANSCRIPTION",
        name="source_lifecycle_state",
        native_enum=False,
        create_constraint=True,
    )
    job_kind = sa.Enum(
        "INGEST", "PROBE", name="job_kind", native_enum=False, create_constraint=True
    )
    job_status = sa.Enum(
        "QUEUED",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        name="job_status",
        native_enum=False,
        create_constraint=True,
    )
    pipeline_stage = sa.Enum(
        "INGEST",
        "PROBE",
        "READY_FOR_TRANSCRIPTION",
        name="pipeline_stage",
        native_enum=False,
        create_constraint=True,
    )
    pipeline_run_status = sa.Enum(
        "QUEUED",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        name="pipeline_run_status",
        native_enum=False,
        create_constraint=True,
    )

    op.create_table(
        "source_videos",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_uri", sa.String(length=2048), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("rights_status", rights_status, nullable=False),
        sa.Column("lifecycle_state", source_lifecycle_state, nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_hash"),
    )
    op.create_index(
        "ix_source_videos_lifecycle_state", "source_videos", ["lifecycle_state"], unique=False
    )
    op.create_index("ix_source_videos_source_uri", "source_videos", ["source_uri"], unique=False)

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_video_id", sa.Uuid(), nullable=False),
        sa.Column("kind", job_kind, nullable=False),
        sa.Column("status", job_status, nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.String(length=255), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.String(length=2048), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint("retry_count >= 0", name="ck_processing_jobs_retry_count_nonnegative"),
        sa.ForeignKeyConstraint(["source_video_id"], ["source_videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_processing_jobs_kind", "processing_jobs", ["kind"], unique=False)
    op.create_index(
        "ix_processing_jobs_source_video_id", "processing_jobs", ["source_video_id"], unique=False
    )
    op.create_index("ix_processing_jobs_status", "processing_jobs", ["status"], unique=False)
    op.create_index("ix_processing_jobs_task_id", "processing_jobs", ["task_id"], unique=False)

    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_video_id", sa.Uuid(), nullable=False),
        sa.Column("stage", pipeline_stage, nullable=False),
        sa.Column("status", pipeline_run_status, nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint("attempt >= 1", name="ck_pipeline_runs_attempt_positive"),
        sa.ForeignKeyConstraint(["source_video_id"], ["source_videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_video_id", "stage", "attempt", name="uq_pipeline_runs_source_stage_attempt"
        ),
    )
    op.create_index(
        "ix_pipeline_runs_source_video_id", "pipeline_runs", ["source_video_id"], unique=False
    )
    op.create_index("ix_pipeline_runs_stage", "pipeline_runs", ["stage"], unique=False)
    op.create_index("ix_pipeline_runs_status", "pipeline_runs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_pipeline_runs_status", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_stage", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_source_video_id", table_name="pipeline_runs")
    op.drop_table("pipeline_runs")
    op.drop_index("ix_processing_jobs_task_id", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_status", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_source_video_id", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_kind", table_name="processing_jobs")
    op.drop_table("processing_jobs")
    op.drop_index("ix_source_videos_source_uri", table_name="source_videos")
    op.drop_index("ix_source_videos_lifecycle_state", table_name="source_videos")
    op.drop_table("source_videos")
