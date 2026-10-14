"""Stage 6 materialization fencing token and deferred-evidence outcome.

Revision ID: 20261014_0024
Revises: 20261014_0023
Create Date: 2026-10-14 18:00:00.000000

Adds a durable ``clip_materializations.claim_version`` ownership token (so a
superseded worker can never replace a newer current materialization) and expands
the ``materialization_outcome`` vocabulary with ``DEFERRED_EVIDENCE`` (final
review deferred on missing/unknown critical evidence). It adds no
``PipelineStage``, ``PipelineRun``, or source-lifecycle change.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20261014_0024"
down_revision: str | None = "20261014_0023"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_OUTCOME_PREVIOUS = (
    "READY_FOR_HUMAN_REVIEW",
    "MATERIALIZED_WITH_CAUTION",
    "NO_OUTPUT",
    "DEFERRED_PROVIDER",
    "TTS_UNAVAILABLE",
    "BLOCKED_VERIFICATION",
    "REJECTED_QUALITY",
    "UPSTREAM_REVALIDATION_REQUIRED",
    "NEEDS_MANUAL_TRANSCRIPT_REVIEW",
    "FAILED",
)
_OUTCOME_CURRENT = (
    "READY_FOR_HUMAN_REVIEW",
    "MATERIALIZED_WITH_CAUTION",
    "NO_OUTPUT",
    "DEFERRED_PROVIDER",
    "DEFERRED_EVIDENCE",
    "TTS_UNAVAILABLE",
    "BLOCKED_VERIFICATION",
    "REJECTED_QUALITY",
    "UPSTREAM_REVALIDATION_REQUIRED",
    "NEEDS_MANUAL_TRANSCRIPT_REVIEW",
    "FAILED",
)


def _enum(name: str, values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True)


def upgrade() -> None:
    with op.batch_alter_table("clip_materializations") as batch:
        batch.add_column(
            sa.Column("claim_version", sa.Integer(), nullable=False, server_default="0")
        )
        batch.alter_column(
            "semantic_outcome",
            existing_type=_enum("materialization_outcome", _OUTCOME_PREVIOUS),
            type_=_enum("materialization_outcome", _OUTCOME_CURRENT),
            existing_nullable=True,
        )


def downgrade() -> None:
    op.execute(
        "DELETE FROM clip_materializations WHERE semantic_outcome = 'DEFERRED_EVIDENCE'"
    )
    with op.batch_alter_table("clip_materializations") as batch:
        batch.alter_column(
            "semantic_outcome",
            existing_type=_enum("materialization_outcome", _OUTCOME_CURRENT),
            type_=_enum("materialization_outcome", _OUTCOME_PREVIOUS),
            existing_nullable=True,
        )
        batch.drop_column("claim_version")
