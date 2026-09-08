"""Add truthful third-party provenance states.

Revision ID: 20260905_0007
Revises: 20260905_0006
Create Date: 2026-09-05 20:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260905_0007"
down_revision: str | None = "20260905_0006"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


_PREVIOUS = (
    "UNKNOWN",
    "OWNED",
    "LICENSED",
    "PERMISSION",
    "PUBLIC_DOMAIN",
    "OTHER_ALLOWED",
)
_CURRENT = (*_PREVIOUS, "THIRD_PARTY_UNKNOWN", "THIRD_PARTY_REUSE")


def _rights_status(values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name="rights_status", native_enum=False, create_constraint=True)


def upgrade() -> None:
    with op.batch_alter_table("source_videos") as batch:
        batch.alter_column(
            "rights_status",
            existing_type=_rights_status(_PREVIOUS),
            type_=_rights_status(_CURRENT),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("source_videos") as batch:
        batch.alter_column(
            "rights_status",
            existing_type=_rights_status(_CURRENT),
            type_=_rights_status(_PREVIOUS),
            existing_nullable=False,
        )
