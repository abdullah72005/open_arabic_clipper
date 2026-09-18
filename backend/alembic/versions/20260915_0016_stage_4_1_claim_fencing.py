"""Stage 4.1 durable per-execution claim fencing.

Revision ID: 20260915_0016
Revises: 20260914_0015
Create Date: 2026-09-15 12:00:00.000000

Adds a single non-null integer ``processing_jobs.claim_version`` ownership
token. Every successful claim advances it, so a worker whose stale RUNNING claim
was reclaimed can never persist, cancel, fail, or finalize the newer run. It
adds no ``PipelineStage``, no ``PipelineRun``, and never changes source
lifecycle values. The downgrade only removes the column and preserves every
Stage 1-4.1 row.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260915_0016"
down_revision: str | None = "20260914_0015"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(
            sa.Column("claim_version", sa.Integer(), nullable=False, server_default="0")
        )


def downgrade() -> None:
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_column("claim_version")
