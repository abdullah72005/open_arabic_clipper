"""Stage 5.0 deterministic execution preflight render-contract persistence.

Revision ID: 20260918_0020
Revises: 20260918_0019
Create Date: 2026-09-18 13:00:00.000000

Adds one table (``render_contracts``) with a partial unique index guaranteeing at
most one database-current Stage 5.0 render contract per candidate. It adds no
``PipelineStage``, no ``PipelineRun`` column, no ``ProcessingJob`` kind, and never
changes source lifecycle values. The downgrade removes only Stage 5.0 render
contracts and preserves every Stage 1-4.3 source, transcript, candidate,
refinement, analysis, strategy, plan, governance, and selection row.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260918_0020"
down_revision: str | None = "20260918_0019"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_RENDER_STATUS = (
    "READY_FOR_RENDER_PLANNING",
    "MATERIALIZATION_REQUIRED",
    "FINAL_CLIP_REFINEMENT_REQUIRED",
    "UPSTREAM_REVALIDATION_REQUIRED",
    "SOURCE_MEDIA_UNAVAILABLE",
    "INVALID_SOURCE_BINDING",
    "BLOCKED",
)
_COMPATIBILITY_OUTCOME = (
    "EXACT_MATCH",
    "COMPATIBLE_NON_MATERIAL_CHANGE",
    "MATERIAL_SEMANTIC_CHANGE",
    "MATERIAL_TIMING_CHANGE",
    "SOURCE_SPAN_NO_LONGER_VALID",
    "UNRESOLVED_COMPATIBILITY",
)
_EXECUTABLE_STATUSES = "('READY_FOR_RENDER_PLANNING', 'MATERIALIZATION_REQUIRED')"


def _enum(name: str, values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True)


def upgrade() -> None:
    op.create_table(
        "render_contracts",
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
            "transformation_selection_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_plan_selections.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "selected_plan_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_plans.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "final_refinement_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("candidate_refinements.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", _enum("render_contract_status", _RENDER_STATUS), nullable=False),
        sa.Column(
            "compatibility_outcome",
            _enum("final_clip_compatibility_outcome", _COMPATIBILITY_OUTCOME),
            nullable=True,
        ),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("contract_ready", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reason_codes", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("compatibility_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("source_media_identity", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("source_probe", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("readiness", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("contract_payload", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("selected_plan_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column(
            "planning_refinement_output_fingerprint",
            sa.String(64),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "final_refinement_output_fingerprint",
            sa.String(64),
            nullable=False,
            server_default="",
        ),
        sa.Column("caption_source_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("source_media_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("probe_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("input_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("output_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("profile_key", sa.String(64), nullable=False, server_default=""),
        sa.Column("profile_version", sa.String(80), nullable=False, server_default=""),
        sa.Column("policy_version", sa.String(64), nullable=False, server_default="stage5.0-v3"),
        sa.Column(
            "schema_version", sa.String(64), nullable=False, server_default="stage5.0-schema-v1"
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
            name="uq_render_contracts_candidate_input",
        ),
        sa.CheckConstraint(
            "is_current IN (true, false)",
            name="ck_render_contracts_current_bool",
        ),
        sa.CheckConstraint(
            "contract_ready IN (true, false)",
            name="ck_render_contracts_ready_bool",
        ),
        sa.CheckConstraint(
            f"(contract_ready AND status IN {_EXECUTABLE_STATUSES}) "
            f"OR (NOT contract_ready AND status NOT IN {_EXECUTABLE_STATUSES})",
            name="ck_render_contracts_ready_status",
        ),
    )
    op.create_index(
        "ix_render_contracts_source_video_id",
        "render_contracts",
        ["source_video_id"],
    )
    op.create_index(
        "ix_render_contracts_clip_candidate_id",
        "render_contracts",
        ["clip_candidate_id"],
    )
    op.create_index(
        "ix_render_contracts_transformation_selection_id",
        "render_contracts",
        ["transformation_selection_id"],
    )
    op.create_index(
        "ix_render_contracts_selected_plan_id",
        "render_contracts",
        ["selected_plan_id"],
    )
    op.create_index(
        "ix_render_contracts_final_refinement_id",
        "render_contracts",
        ["final_refinement_id"],
    )
    op.create_index(
        "ix_render_contracts_status",
        "render_contracts",
        ["status"],
    )
    op.create_index(
        "ix_render_contracts_is_current",
        "render_contracts",
        ["is_current"],
    )
    op.create_index(
        "uq_render_contracts_current",
        "render_contracts",
        ["clip_candidate_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
        sqlite_where=sa.text("is_current"),
    )


def downgrade() -> None:
    for index_name in (
        "uq_render_contracts_current",
        "ix_render_contracts_is_current",
        "ix_render_contracts_status",
        "ix_render_contracts_final_refinement_id",
        "ix_render_contracts_selected_plan_id",
        "ix_render_contracts_transformation_selection_id",
        "ix_render_contracts_clip_candidate_id",
        "ix_render_contracts_source_video_id",
    ):
        op.drop_index(index_name, table_name="render_contracts")
    op.drop_table("render_contracts")
