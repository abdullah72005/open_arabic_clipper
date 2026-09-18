"""Stage 5.1 deterministic visual-composition plan persistence.

Revision ID: 20260918_0021
Revises: 20260918_0020
Create Date: 2026-09-18 14:00:00.000000

Adds a candidate-scoped ``VISUAL_COMPOSITION`` job kind, one table
(``visual_composition_plans``) with a partial unique index guaranteeing at most
one database-current Stage 5.1 plan per candidate, and one nullable
``processing_jobs.visual_composition_plan_id`` FK. It adds no ``PipelineStage``,
no ``PipelineRun`` column, and never changes source lifecycle values. The
downgrade removes only Stage 5.1 jobs/schema and preserves every Stage 1-5.0
source, transcript, candidate, refinement, analysis, strategy, plan, governance,
selection, job, pipeline run, and render contract.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260918_0021"
down_revision: str | None = "20260918_0020"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_JOB_PREVIOUS = (
    "INGEST",
    "TRANSCRIPTION",
    "RECONSTRUCTION",
    "PROBE",
    "CANDIDATE_ANALYSIS",
    "CANDIDATE_REFINEMENT",
    "TRANSFORMATION_ELIGIBILITY",
    "TRANSFORMATION_PLANNING",
    "TRANSFORMATION_GOVERNANCE",
)
_JOB_CURRENT = (*_JOB_PREVIOUS, "VISUAL_COMPOSITION")

_COMPOSITION_STATUS = (
    "READY_FOR_VISUAL_EXECUTION",
    "BLOCKED",
    "FAILED",
)
_COMPOSITION_EXECUTION_STATUS = (
    "QUEUED",
    "ANALYZING",
    "COMPLETE",
    "FAILED",
    "CANCELLED",
)


def _enum(name: str, values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True)


def upgrade() -> None:
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_PREVIOUS),
            type_=_enum("job_kind", _JOB_CURRENT),
            existing_nullable=False,
        )

    op.create_table(
        "visual_composition_plans",
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
            "render_contract_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("render_contracts.id", ondelete="SET NULL"),
            nullable=True,
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
        sa.Column(
            "status",
            _enum("visual_composition_status", _COMPOSITION_STATUS),
            nullable=False,
        ),
        sa.Column(
            "execution_status",
            _enum("visual_composition_execution_status", _COMPOSITION_EXECUTION_STATUS),
            nullable=False,
            server_default="QUEUED",
        ),
        sa.Column("plan_ready", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "active_job_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("processing_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("reason_codes", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("input_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("output_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("contract_input_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("contract_output_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("caption_source_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("source_media_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("source_media_identity", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("analysis_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("framing_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("ass_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("policy_version", sa.String(64), nullable=False, server_default="stage5.1-v1"),
        sa.Column(
            "schema_version", sa.String(64), nullable=False, server_default="stage5.1-schema-v1"
        ),
        sa.Column("fingerprint_version", sa.String(16), nullable=False, server_default="1"),
        sa.Column("plan_payload", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("readiness", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("cache_eligible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "clip_candidate_id",
            "input_fingerprint",
            name="uq_visual_composition_plans_candidate_input",
        ),
        sa.CheckConstraint(
            "is_current IN (true, false)",
            name="ck_visual_composition_plans_current_bool",
        ),
        sa.CheckConstraint(
            "plan_ready IN (true, false)",
            name="ck_visual_composition_plans_ready_bool",
        ),
        sa.CheckConstraint(
            "cache_eligible IN (true, false)",
            name="ck_visual_composition_plans_cache_eligible_bool",
        ),
    )
    op.create_index(
        "ix_visual_composition_plans_source_video_id",
        "visual_composition_plans",
        ["source_video_id"],
    )
    op.create_index(
        "ix_visual_composition_plans_clip_candidate_id",
        "visual_composition_plans",
        ["clip_candidate_id"],
    )
    op.create_index(
        "ix_visual_composition_plans_render_contract_id",
        "visual_composition_plans",
        ["render_contract_id"],
    )
    op.create_index(
        "ix_visual_composition_plans_transformation_selection_id",
        "visual_composition_plans",
        ["transformation_selection_id"],
    )
    op.create_index(
        "ix_visual_composition_plans_selected_plan_id",
        "visual_composition_plans",
        ["selected_plan_id"],
    )
    op.create_index(
        "ix_visual_composition_plans_final_refinement_id",
        "visual_composition_plans",
        ["final_refinement_id"],
    )
    op.create_index(
        "ix_visual_composition_plans_status",
        "visual_composition_plans",
        ["status"],
    )
    op.create_index(
        "ix_visual_composition_plans_execution_status",
        "visual_composition_plans",
        ["execution_status"],
    )
    op.create_index(
        "ix_visual_composition_plans_is_current",
        "visual_composition_plans",
        ["is_current"],
    )
    op.create_index(
        "ix_visual_composition_plans_active_job_id",
        "visual_composition_plans",
        ["active_job_id"],
    )
    op.create_index(
        "uq_visual_composition_plans_current",
        "visual_composition_plans",
        ["clip_candidate_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
        sqlite_where=sa.text("is_current"),
    )

    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(
            sa.Column("visual_composition_plan_id", sa.Uuid(as_uuid=True), nullable=True)
        )
        batch.create_foreign_key(
            "fk_processing_jobs_visual_composition_plan_id",
            "visual_composition_plans",
            ["visual_composition_plan_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        "ix_processing_jobs_visual_composition_plan_id",
        "processing_jobs",
        ["visual_composition_plan_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_processing_jobs_visual_composition_plan_id", table_name="processing_jobs")
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_constraint("fk_processing_jobs_visual_composition_plan_id", type_="foreignkey")
        batch.drop_column("visual_composition_plan_id")

    for index_name in (
        "uq_visual_composition_plans_current",
        "ix_visual_composition_plans_active_job_id",
        "ix_visual_composition_plans_is_current",
        "ix_visual_composition_plans_execution_status",
        "ix_visual_composition_plans_status",
        "ix_visual_composition_plans_final_refinement_id",
        "ix_visual_composition_plans_selected_plan_id",
        "ix_visual_composition_plans_transformation_selection_id",
        "ix_visual_composition_plans_render_contract_id",
        "ix_visual_composition_plans_clip_candidate_id",
        "ix_visual_composition_plans_source_video_id",
    ):
        op.drop_index(index_name, table_name="visual_composition_plans")
    op.drop_table("visual_composition_plans")

    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_CURRENT),
            type_=_enum("job_kind", _JOB_PREVIOUS),
            existing_nullable=False,
        )
