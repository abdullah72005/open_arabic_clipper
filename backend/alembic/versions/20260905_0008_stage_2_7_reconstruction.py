"""Persist Stage 2.7 contextual reconstruction state.

Revision ID: 20260905_0008
Revises: 20260905_0007
Create Date: 2026-09-05 22:15:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260905_0008"
down_revision: str | None = "20260905_0007"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


_LIFECYCLE_PREVIOUS = (
    "INGEST",
    "PROBE",
    "READY_FOR_TRANSCRIPTION",
    "AUDIO_EXTRACTION",
    "TRANSCRIPTION",
    "TRANSCRIPT_NORMALIZATION",
    "AUDIO_ANALYSIS",
    "READY_FOR_ANALYSIS",
)
_LIFECYCLE_CURRENT = (
    *_LIFECYCLE_PREVIOUS[:6],
    "CONTEXTUAL_RECONSTRUCTION",
    *_LIFECYCLE_PREVIOUS[6:],
)
_JOB_PREVIOUS = ("INGEST", "PROBE", "TRANSCRIPTION")
_JOB_CURRENT = (*_JOB_PREVIOUS, "RECONSTRUCTION")


def _enum(name: str, values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True)


def upgrade() -> None:
    with op.batch_alter_table("source_videos") as batch:
        batch.alter_column(
            "lifecycle_state",
            existing_type=_enum("source_lifecycle_state", _LIFECYCLE_PREVIOUS),
            type_=_enum("source_lifecycle_state", _LIFECYCLE_CURRENT),
            existing_nullable=False,
        )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_PREVIOUS),
            type_=_enum("job_kind", _JOB_CURRENT),
            existing_nullable=False,
        )
    with op.batch_alter_table("pipeline_runs") as batch:
        batch.alter_column(
            "stage",
            existing_type=_enum("pipeline_stage", _LIFECYCLE_PREVIOUS),
            type_=_enum("pipeline_stage", _LIFECYCLE_CURRENT),
            existing_nullable=False,
        )
    with op.batch_alter_table("transcripts") as batch:
        batch.add_column(
            sa.Column("contextual_reconstructed_text", sa.Text(), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column(
                "reconstruction_fingerprint",
                sa.String(length=64),
                nullable=False,
                server_default="",
            )
        )
        batch.add_column(
            sa.Column("reconstruction_confidence", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column("reconstructed_segment_ratio", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column(
                "reconstruction_method",
                sa.String(length=64),
                nullable=False,
                server_default="pending",
            )
        )
        batch.add_column(
            sa.Column(
                "reconstruction_version",
                sa.String(length=64),
                nullable=False,
                server_default="pending",
            )
        )
        batch.add_column(sa.Column("reconstruction_processing_duration", sa.Float()))
        batch.add_column(
            sa.Column("reconstruction_metadata", sa.JSON(), nullable=False, server_default="{}")
        )


def downgrade() -> None:
    with op.batch_alter_table("transcripts") as batch:
        batch.drop_column("reconstruction_metadata")
        batch.drop_column("reconstruction_processing_duration")
        batch.drop_column("reconstruction_version")
        batch.drop_column("reconstruction_method")
        batch.drop_column("reconstructed_segment_ratio")
        batch.drop_column("reconstruction_confidence")
        batch.drop_column("reconstruction_fingerprint")
        batch.drop_column("contextual_reconstructed_text")
    with op.batch_alter_table("pipeline_runs") as batch:
        batch.alter_column(
            "stage",
            existing_type=_enum("pipeline_stage", _LIFECYCLE_CURRENT),
            type_=_enum("pipeline_stage", _LIFECYCLE_PREVIOUS),
            existing_nullable=False,
        )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_CURRENT),
            type_=_enum("job_kind", _JOB_PREVIOUS),
            existing_nullable=False,
        )
    with op.batch_alter_table("source_videos") as batch:
        batch.alter_column(
            "lifecycle_state",
            existing_type=_enum("source_lifecycle_state", _LIFECYCLE_CURRENT),
            type_=_enum("source_lifecycle_state", _LIFECYCLE_PREVIOUS),
            existing_nullable=False,
        )
