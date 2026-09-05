"""Allow durable pipeline-run records for the ingest stage.

Revision ID: 20260904_0005
Revises: 20260904_0004
Create Date: 2026-09-04 21:15:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260904_0005"
down_revision: str | None = "20260904_0004"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_BEFORE = (
    "PROBE",
    "READY_FOR_TRANSCRIPTION",
    "AUDIO_EXTRACTION",
    "TRANSCRIPTION",
    "TRANSCRIPT_NORMALIZATION",
    "AUDIO_ANALYSIS",
    "READY_FOR_ANALYSIS",
)
_AFTER = ("INGEST", *_BEFORE)


def _enum(values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name="pipeline_stage", native_enum=False, create_constraint=True)


def upgrade() -> None:
    with op.batch_alter_table("pipeline_runs") as batch:
        batch.alter_column(
            "stage",
            existing_type=_enum(_BEFORE),
            type_=_enum(_AFTER),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("pipeline_runs") as batch:
        batch.alter_column(
            "stage",
            existing_type=_enum(_AFTER),
            type_=_enum(_BEFORE),
            existing_nullable=False,
        )
