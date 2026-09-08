"""Persist Stage 2.5 corrected transcript state without replacing raw ASR evidence.

Revision ID: 20260905_0006
Revises: 20260904_0005
Create Date: 2026-09-05 15:40:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260905_0006"
down_revision: str | None = "20260904_0005"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("transcripts") as batch:
        batch.add_column(sa.Column("corrected_text", sa.Text(), nullable=False, server_default=""))
        batch.add_column(sa.Column("final_text", sa.Text(), nullable=False, server_default=""))
        batch.add_column(
            sa.Column("raw_transcript_confidence", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("correction_confidence", sa.Float(), nullable=False, server_default="0"))
        batch.add_column(
            sa.Column("corrected_segment_ratio", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column("uncertain_segment_ratio", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column("correction_method", sa.String(length=64), nullable=False, server_default="pending")
        )
        batch.add_column(
            sa.Column("correction_version", sa.String(length=64), nullable=False, server_default="pending")
        )
    op.execute("UPDATE transcripts SET corrected_text = normalized_text, final_text = normalized_text")


def downgrade() -> None:
    with op.batch_alter_table("transcripts") as batch:
        batch.drop_column("correction_version")
        batch.drop_column("correction_method")
        batch.drop_column("uncertain_segment_ratio")
        batch.drop_column("corrected_segment_ratio")
        batch.drop_column("correction_confidence")
        batch.drop_column("raw_transcript_confidence")
        batch.drop_column("final_text")
        batch.drop_column("corrected_text")
