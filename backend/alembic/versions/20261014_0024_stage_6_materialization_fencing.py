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
    # Preserve history: a ``DEFERRED_EVIDENCE`` sample cannot exist under the
    # older outcome vocabulary, so map it to the older ``FAILED`` outcome while
    # retaining ``DEFERRED_EVIDENCE`` in reason_codes/history, forcing lifecycle
    # BLOCKED and every review/cache/publication flag false. Rows and their
    # foreign-key relationships are never deleted.
    import json

    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT id, reason_codes FROM clip_materializations "
            "WHERE semantic_outcome = 'DEFERRED_EVIDENCE'"
        )
    ).fetchall()
    for row in rows:
        reasons = row.reason_codes
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except ValueError:
                reasons = []
        if not isinstance(reasons, list):
            reasons = []
        if "DEFERRED_EVIDENCE" not in reasons:
            reasons.append("DEFERRED_EVIDENCE")
        bind.execute(
            sa.text(
                "UPDATE clip_materializations SET semantic_outcome = 'FAILED', "
                "lifecycle = 'BLOCKED', cache_eligible = false, "
                "human_review_required = false, publication_ready = false, "
                "reason_codes = :reasons WHERE id = :row_id"
            ),
            {"reasons": json.dumps(reasons), "row_id": row.id},
        )
    with op.batch_alter_table("clip_materializations") as batch:
        batch.alter_column(
            "semantic_outcome",
            existing_type=_enum("materialization_outcome", _OUTCOME_CURRENT),
            type_=_enum("materialization_outcome", _OUTCOME_PREVIOUS),
            existing_nullable=True,
        )
        batch.drop_column("claim_version")
