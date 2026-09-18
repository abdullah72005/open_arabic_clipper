"""Stage 4.1 renewable execution heartbeat for stale-claim reclaim.

Revision ID: 20260915_0017
Revises: 20260915_0016
Create Date: 2026-09-15 18:00:00.000000

Adds one nullable ``processing_jobs.heartbeat_at`` liveness timestamp. A worker
actively executing refreshes it, so a live worker stalled inside provider work
is never reclaimed merely because its original ``started_at`` is old, while a
genuinely abandoned run becomes reclaimable once its heartbeat goes stale. It
adds no ``PipelineStage``, no ``PipelineRun``, and never changes source
lifecycle values. The downgrade only removes the column and preserves every
Stage 1-4.1 row.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260915_0017"
down_revision: str | None = "20260915_0016"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_column("heartbeat_at")
