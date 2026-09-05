"""Persist reusable Stage 2 semantic transcript chunks.

Revision ID: 20260904_0004
Revises: 20260904_0003
Create Date: 2026-09-04 20:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260904_0004"
down_revision: str | None = "20260904_0003"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "transcript_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("transcript_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("segment_indexes", sa.JSON(), nullable=False),
        sa.Column("preceding_context", sa.Text(), nullable=False),
        sa.Column("following_context", sa.Text(), nullable=False),
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
        sa.ForeignKeyConstraint(["transcript_id"], ["transcripts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("transcript_id", "sequence"),
    )
    op.create_index("ix_transcript_chunks_transcript_id", "transcript_chunks", ["transcript_id"])


def downgrade() -> None:
    op.drop_index("ix_transcript_chunks_transcript_id", table_name="transcript_chunks")
    op.drop_table("transcript_chunks")
