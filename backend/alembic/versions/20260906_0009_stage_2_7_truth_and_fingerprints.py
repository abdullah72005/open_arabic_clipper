"""Persist truthful Stage 2.7 state and dependency fingerprints.

Revision ID: 20260906_0009
Revises: 20260905_0008
Create Date: 2026-09-06 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260906_0009"
down_revision: str | None = "20260905_0008"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_RECONSTRUCTION_STATUSES = (
    "NOT_REQUIRED",
    "APPLIED",
    "UNCHANGED_HIGH_CONFIDENCE",
    "LOW_CONFIDENCE_UNRESOLVED",
    "PROVIDER_UNAVAILABLE",
    "FAILED",
    "MANUAL_OVERRIDE",
)


def _reconstruction_status_enum() -> sa.Enum:
    return sa.Enum(
        *_RECONSTRUCTION_STATUSES,
        name="reconstruction_status",
        native_enum=False,
        create_constraint=True,
    )


def upgrade() -> None:
    with op.batch_alter_table("pipeline_runs") as batch:
        batch.add_column(sa.Column("input_fingerprint", sa.String(length=64)))
        batch.add_column(sa.Column("output_fingerprint", sa.String(length=64)))

    with op.batch_alter_table("transcripts") as batch:
        batch.add_column(
            sa.Column("transcription_revision", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column(
                "normalization_fingerprint",
                sa.String(length=64),
                nullable=False,
                server_default="",
            )
        )
        batch.add_column(
            sa.Column(
                "reconstruction_status",
                _reconstruction_status_enum(),
                nullable=False,
                server_default="NOT_REQUIRED",
            )
        )
        batch.create_check_constraint(
            "ck_transcripts_transcription_revision_nonnegative",
            "transcription_revision >= 0",
        )

    with op.batch_alter_table("audio_analyses") as batch:
        batch.add_column(
            sa.Column("input_fingerprint", sa.String(length=64), nullable=False, server_default="")
        )

    with op.batch_alter_table("source_quality_assessments") as batch:
        batch.add_column(
            sa.Column("transcript_quality_score", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column("low_confidence_word_ratio", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column("unresolved_segment_ratio", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column(
                "manual_review_required", sa.Boolean(), nullable=False, server_default=sa.true()
            )
        )
        batch.add_column(
            sa.Column("input_fingerprint", sa.String(length=64), nullable=False, server_default="")
        )
        batch.create_check_constraint(
            "ck_source_quality_transcript_quality_score_bounds",
            "transcript_quality_score >= 0 AND transcript_quality_score <= 1",
        )
        batch.create_check_constraint(
            "ck_source_quality_low_confidence_word_ratio_bounds",
            "low_confidence_word_ratio >= 0 AND low_confidence_word_ratio <= 1",
        )
        batch.create_check_constraint(
            "ck_source_quality_unresolved_segment_ratio_bounds",
            "unresolved_segment_ratio >= 0 AND unresolved_segment_ratio <= 1",
        )


def downgrade() -> None:
    with op.batch_alter_table("source_quality_assessments") as batch:
        batch.drop_constraint("ck_source_quality_unresolved_segment_ratio_bounds", type_="check")
        batch.drop_constraint("ck_source_quality_low_confidence_word_ratio_bounds", type_="check")
        batch.drop_constraint("ck_source_quality_transcript_quality_score_bounds", type_="check")
        batch.drop_column("input_fingerprint")
        batch.drop_column("manual_review_required")
        batch.drop_column("unresolved_segment_ratio")
        batch.drop_column("low_confidence_word_ratio")
        batch.drop_column("transcript_quality_score")

    with op.batch_alter_table("audio_analyses") as batch:
        batch.drop_column("input_fingerprint")

    with op.batch_alter_table("transcripts") as batch:
        batch.drop_constraint("ck_transcripts_transcription_revision_nonnegative", type_="check")
        batch.drop_constraint("reconstruction_status", type_="check")
        batch.drop_column("reconstruction_status")
        batch.drop_column("normalization_fingerprint")
        batch.drop_column("transcription_revision")

    with op.batch_alter_table("pipeline_runs") as batch:
        batch.drop_column("output_fingerprint")
        batch.drop_column("input_fingerprint")
