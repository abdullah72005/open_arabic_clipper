"""Expand the operator rights-status vocabulary.

Revision ID: 20260904_0002
Revises: 20260904_0001
Create Date: 2026-09-04 11:55:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260904_0002"
down_revision: str | None = "20260904_0001"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_CURRENT = ("OWNED", "AUTHORIZED", "UNKNOWN")
_EXPANDED = ("UNKNOWN", "OWNED", "LICENSED", "PERMISSION", "PUBLIC_DOMAIN", "OTHER_ALLOWED")


def _rights_status(values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name="rights_status", native_enum=False, create_constraint=True)


def upgrade() -> None:
    with op.batch_alter_table("source_videos") as batch:
        batch.alter_column(
            "rights_status",
            existing_type=_rights_status(_CURRENT),
            type_=_rights_status(_EXPANDED),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("source_videos") as batch:
        batch.alter_column(
            "rights_status",
            existing_type=_rights_status(_EXPANDED),
            type_=_rights_status(_CURRENT),
            existing_nullable=False,
        )
