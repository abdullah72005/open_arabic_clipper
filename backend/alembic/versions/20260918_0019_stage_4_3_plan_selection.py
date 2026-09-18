"""Stage 4.3 deterministic final-plan selection persistence.

Revision ID: 20260918_0019
Revises: 20260917_0018
Create Date: 2026-09-18 12:00:00.000000

Adds one table (``transformation_plan_selections``) with a partial unique index
guaranteeing at most one database-current selection per candidate. It adds no
``PipelineStage``, no ``PipelineRun`` column, no ``ProcessingJob`` kind, and never
changes source lifecycle values. The downgrade removes only Stage 4.3 selection
rows/schema and preserves every Stage 1-4.2 source, transcript, candidate,
refinement, analysis, strategy, plan set, plan, governance set, and governance
result.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260918_0019"
down_revision: str | None = "20260917_0018"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_SELECTION_STATUS = (
    "PLAN_SELECTED",
    "PLAN_SELECTED_WITH_CAUTION",
    "NO_SELECTABLE_PLAN",
    "SELECTION_DEFERRED",
    "STALE_SELECTION_INPUT",
)
_SELECTED_STATUSES = "('PLAN_SELECTED', 'PLAN_SELECTED_WITH_CAUTION')"
_CAUTION_STATUS = "PLAN_SELECTED_WITH_CAUTION"


def _enum(name: str, values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True)


def upgrade() -> None:
    op.create_table(
        "transformation_plan_selections",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "source_video_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("source_videos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "clip_candidate_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("clip_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "transformation_analysis_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_eligibility_analyses.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "transformation_plan_set_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_plan_sets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "transformation_governance_set_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_governance_sets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "selected_plan_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_plans.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "selected_governance_result_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_governance_results.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "refinement_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("candidate_refinements.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("refinement_priority", sa.String(32), nullable=False, server_default=""),
        sa.Column("refinement_quality_level", sa.String(32), nullable=False, server_default=""),
        sa.Column(
            "refinement_output_fingerprint", sa.String(64), nullable=False, server_default=""
        ),
        sa.Column(
            "status",
            _enum("transformation_selection_status", _SELECTION_STATUS),
            nullable=False,
        ),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("selected_with_caution", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("selection_reason_codes", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("arbitration_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("selected_governance_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("alternative_dispositions", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("governance_input_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column(
            "governance_output_fingerprint", sa.String(64), nullable=False, server_default=""
        ),
        sa.Column("governor_policy_version", sa.String(64), nullable=False, server_default=""),
        sa.Column("governor_validation_version", sa.String(64), nullable=False, server_default=""),
        sa.Column(
            "platform_policy_profile_version", sa.String(80), nullable=False, server_default=""
        ),
        sa.Column("selected_plan_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("input_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("output_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("policy_version", sa.String(64), nullable=False, server_default="stage4.3-v1"),
        sa.Column(
            "schema_version", sa.String(64), nullable=False, server_default="stage4.3-schema-v1"
        ),
        sa.Column("fingerprint_version", sa.String(16), nullable=False, server_default="1"),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "clip_candidate_id",
            "input_fingerprint",
            name="uq_transformation_plan_selections_candidate_input",
        ),
        sa.CheckConstraint(
            "is_current IN (0, 1)",
            name="ck_transformation_plan_selections_current_bool",
        ),
        sa.CheckConstraint(
            "selected_with_caution IN (0, 1)",
            name="ck_transformation_plan_selections_caution_bool",
        ),
        sa.CheckConstraint(
            "(selected_plan_id IS NULL AND selected_governance_result_id IS NULL) "
            "OR (selected_plan_id IS NOT NULL AND selected_governance_result_id IS NOT NULL)",
            name="ck_transformation_plan_selections_selected_pair",
        ),
        sa.CheckConstraint(
            f"(status IN {_SELECTED_STATUSES} AND selected_plan_id IS NOT NULL) "
            f"OR (status NOT IN {_SELECTED_STATUSES} AND selected_plan_id IS NULL)",
            name="ck_transformation_plan_selections_status_selected",
        ),
        sa.CheckConstraint(
            f"(selected_with_caution = 1 AND status = '{_CAUTION_STATUS}') "
            f"OR (selected_with_caution = 0 AND status != '{_CAUTION_STATUS}')",
            name="ck_transformation_plan_selections_caution_status",
        ),
    )
    op.create_index(
        "ix_transformation_plan_selections_source_video_id",
        "transformation_plan_selections",
        ["source_video_id"],
    )
    op.create_index(
        "ix_transformation_plan_selections_clip_candidate_id",
        "transformation_plan_selections",
        ["clip_candidate_id"],
    )
    op.create_index(
        "ix_transformation_plan_selections_transformation_analysis_id",
        "transformation_plan_selections",
        ["transformation_analysis_id"],
    )
    op.create_index(
        "ix_transformation_plan_selections_transformation_plan_set_id",
        "transformation_plan_selections",
        ["transformation_plan_set_id"],
    )
    op.create_index(
        "ix_transformation_plan_selections_transformation_governance_set_id",
        "transformation_plan_selections",
        ["transformation_governance_set_id"],
    )
    op.create_index(
        "ix_transformation_plan_selections_selected_plan_id",
        "transformation_plan_selections",
        ["selected_plan_id"],
    )
    op.create_index(
        "ix_transformation_plan_selections_selected_governance_result_id",
        "transformation_plan_selections",
        ["selected_governance_result_id"],
    )
    op.create_index(
        "ix_transformation_plan_selections_refinement_id",
        "transformation_plan_selections",
        ["refinement_id"],
    )
    op.create_index(
        "ix_transformation_plan_selections_status",
        "transformation_plan_selections",
        ["status"],
    )
    op.create_index(
        "ix_transformation_plan_selections_is_current",
        "transformation_plan_selections",
        ["is_current"],
    )
    op.create_index(
        "uq_transformation_plan_selections_current",
        "transformation_plan_selections",
        ["clip_candidate_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
        sqlite_where=sa.text("is_current"),
    )


def downgrade() -> None:
    for index_name in (
        "uq_transformation_plan_selections_current",
        "ix_transformation_plan_selections_is_current",
        "ix_transformation_plan_selections_status",
        "ix_transformation_plan_selections_refinement_id",
        "ix_transformation_plan_selections_selected_governance_result_id",
        "ix_transformation_plan_selections_selected_plan_id",
        "ix_transformation_plan_selections_transformation_governance_set_id",
        "ix_transformation_plan_selections_transformation_plan_set_id",
        "ix_transformation_plan_selections_transformation_analysis_id",
        "ix_transformation_plan_selections_clip_candidate_id",
        "ix_transformation_plan_selections_source_video_id",
    ):
        op.drop_index(index_name, table_name="transformation_plan_selections")
    op.drop_table("transformation_plan_selections")
