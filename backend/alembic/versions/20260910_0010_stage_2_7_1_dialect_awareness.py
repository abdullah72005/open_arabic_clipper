"""Persist Stage 2.7.1 dialect awareness and code-switch evidence.

Revision ID: 20260910_0010
Revises: 20260906_0009
Create Date: 2026-09-10 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260910_0010"
down_revision: str | None = "20260906_0009"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_DIALECT_PROFILES = (
    "EGYPTIAN",
    "SAUDI",
    "GULF",
    "LEVANTINE",
    "MSA",
    "UNKNOWN_ARABIC",
)


def _dialect_enum() -> sa.Enum:
    return sa.Enum(
        *_DIALECT_PROFILES,
        name="arabic_dialect_profile",
        native_enum=False,
        create_constraint=True,
    )


def upgrade() -> None:
    with op.batch_alter_table("source_videos") as batch:
        batch.add_column(sa.Column("dialect_profile_override", _dialect_enum(), nullable=True))

    with op.batch_alter_table("transcripts") as batch:
        batch.add_column(sa.Column("dialect_profile", _dialect_enum(), nullable=True))
        batch.add_column(
            sa.Column("dialect_confidence", sa.Float(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column("dialect_evidence", sa.JSON(), nullable=False, server_default="{}")
        )
        batch.add_column(
            sa.Column("code_switch_suspected", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch.create_check_constraint(
            "ck_transcripts_dialect_confidence_bounds",
            "dialect_confidence >= 0 AND dialect_confidence <= 1",
        )

    # Backfill existing Arabic-language transcripts conservatively as
    # UNKNOWN_ARABIC with confidence 0. Clearly non-Arabic transcripts keep a
    # nullable profile. Evidence defaults to an empty object, code-switch
    # suspicion to false, and no transcript text is rewritten.
    op.execute(
        "UPDATE transcripts SET dialect_profile = 'UNKNOWN_ARABIC' "
        "WHERE dialect_profile IS NULL AND (language LIKE 'ar%' OR language LIKE 'AR%')"
    )


def downgrade() -> None:
    with op.batch_alter_table("transcripts") as batch:
        batch.drop_constraint("ck_transcripts_dialect_confidence_bounds", type_="check")
        batch.drop_constraint("arabic_dialect_profile", type_="check")
        batch.drop_column("code_switch_suspected")
        batch.drop_column("dialect_evidence")
        batch.drop_column("dialect_confidence")
        batch.drop_column("dialect_profile")

    with op.batch_alter_table("source_videos") as batch:
        batch.drop_constraint("arabic_dialect_profile", type_="check")
        batch.drop_column("dialect_profile_override")