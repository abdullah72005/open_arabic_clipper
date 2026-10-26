"""Stage 4.1 operator planning-request scope and editorial intent.

Revision ID: 20261024_0025
Revises: 20261014_0024
Create Date: 2026-10-24 15:00:00.000000

Adds a durable ``transformation_plan_sets.operator_request`` JSON object. It
binds the explicit source window and editorial direction to the existing Stage
4.1 plan-set envelope so a separate worker recovers the same constraints. It
adds no PipelineStage, PipelineRun, or source-lifecycle change.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20261024_0025"
down_revision: str | None = "20261014_0024"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("transformation_plan_sets") as batch:
        batch.add_column(
            sa.Column(
                "operator_request",
                sa.JSON(),
                nullable=False,
                server_default="{}",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("transformation_plan_sets") as batch:
        batch.drop_column("operator_request")
