"""Stage 5.2 deterministic render-execution persistence.

Revision ID: 20260918_0022
Revises: 20260918_0021
Create Date: 2026-10-05 12:00:00.000000

Adds a candidate-scoped ``RENDER_EXECUTION`` job kind, one table
(``render_executions``) with a scoped partial unique index guaranteeing at most
one database-current execution per ``(candidate, artifact purpose, delivery
profile)``, and one nullable ``processing_jobs.render_execution_id`` FK. It adds
no ``PipelineStage``, no ``PipelineRun`` column, and never changes source
lifecycle values. The downgrade removes only Stage 5.2 jobs/schema and preserves
every Stage 1-5.1 source, transcript, candidate, refinement, analysis, strategy,
plan, governance, selection, render contract, visual-composition plan, job, and
pipeline run.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260918_0022"
down_revision: str | None = "20260918_0021"
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
    "VISUAL_COMPOSITION",
)
_JOB_CURRENT = (*_JOB_PREVIOUS, "RENDER_EXECUTION")

_ARTIFACT_PURPOSE = ("CORE_SOURCE_VALIDATION",)
_LIFECYCLE = ("QUEUED", "RENDERING", "QC_RUNNING", "COMPLETE", "BLOCKED", "FAILED", "CANCELLED")
_QC_STATUS = ("PASS", "WARN", "FAIL")

_COMPLETE_OK = "lifecycle = 'COMPLETE' AND qc_status IN ('PASS', 'WARN')"


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
        "render_executions",
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
            "visual_composition_plan_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("visual_composition_plans.id", ondelete="SET NULL"),
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
            "artifact_purpose", _enum("render_artifact_purpose", _ARTIFACT_PURPOSE), nullable=False
        ),
        sa.Column("lifecycle", _enum("render_execution_lifecycle", _LIFECYCLE), nullable=False),
        sa.Column("qc_status", _enum("render_qc_status", _QC_STATUS), nullable=True),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "active_job_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("processing_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("publication_ready", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("stage6_implemented", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("cache_eligible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reason_codes", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("error_message", sa.String(2048), nullable=True),
        sa.Column("delivery_profile_key", sa.String(80), nullable=False, server_default=""),
        sa.Column("delivery_profile_version", sa.String(64), nullable=False, server_default=""),
        sa.Column("input_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("output_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("runtime_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("compiler_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("qc_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("request_payload", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("execution_manifest", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("qc_result", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("artifact_reference", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("omitted_requirements", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("policy_version", sa.String(64), nullable=False, server_default="stage5.2-v3"),
        sa.Column(
            "schema_version", sa.String(64), nullable=False, server_default="stage5.2-schema-v3"
        ),
        sa.Column("fingerprint_version", sa.String(16), nullable=False, server_default="3"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "clip_candidate_id",
            "input_fingerprint",
            name="uq_render_executions_candidate_input",
        ),
        sa.CheckConstraint("is_current IN (true, false)", name="ck_render_executions_current_bool"),
        sa.CheckConstraint(
            "publication_ready IN (true, false)",
            name="ck_render_executions_publication_ready_bool",
        ),
        sa.CheckConstraint(
            "stage6_implemented IN (true, false)",
            name="ck_render_executions_stage6_bool",
        ),
        sa.CheckConstraint(
            "cache_eligible IN (true, false)",
            name="ck_render_executions_cache_eligible_bool",
        ),
        sa.CheckConstraint(
            f"(cache_eligible AND {_COMPLETE_OK}) OR (NOT cache_eligible)",
            name="ck_render_executions_cache_consistency",
        ),
        sa.CheckConstraint(
            f"({_COMPLETE_OK}) OR (lifecycle <> 'COMPLETE')",
            name="ck_render_executions_complete_qc",
        ),
    )
    for column in (
        "source_video_id",
        "clip_candidate_id",
        "render_contract_id",
        "visual_composition_plan_id",
        "transformation_selection_id",
        "selected_plan_id",
        "final_refinement_id",
        "artifact_purpose",
        "lifecycle",
        "qc_status",
        "is_current",
        "active_job_id",
    ):
        op.create_index(f"ix_render_executions_{column}", "render_executions", [column])
    op.create_index(
        "uq_render_executions_current_scope",
        "render_executions",
        ["clip_candidate_id", "artifact_purpose", "delivery_profile_key"],
        unique=True,
        postgresql_where=sa.text("is_current"),
        sqlite_where=sa.text("is_current"),
    )

    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(sa.Column("render_execution_id", sa.Uuid(as_uuid=True), nullable=True))
        batch.create_foreign_key(
            "fk_processing_jobs_render_execution_id",
            "render_executions",
            ["render_execution_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        "ix_processing_jobs_render_execution_id",
        "processing_jobs",
        ["render_execution_id"],
    )


def downgrade() -> None:
    # Remove Stage 5.2 rows/jobs ourselves so the narrowed ``job_kind`` check
    # constraint can be recreated on populated databases. This preserves every
    # Stage 1-5.1 job and schema object, and never requires test-side cleanup.
    op.execute("DELETE FROM render_executions")
    op.execute("DELETE FROM processing_jobs WHERE kind = 'RENDER_EXECUTION'")

    op.drop_index("ix_processing_jobs_render_execution_id", table_name="processing_jobs")
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_constraint("fk_processing_jobs_render_execution_id", type_="foreignkey")
        batch.drop_column("render_execution_id")

    index_names = [
        "uq_render_executions_current_scope",
        *[
            f"ix_render_executions_{column}"
            for column in reversed(
                [
                    "source_video_id",
                    "clip_candidate_id",
                    "render_contract_id",
                    "visual_composition_plan_id",
                    "transformation_selection_id",
                    "selected_plan_id",
                    "final_refinement_id",
                    "artifact_purpose",
                    "lifecycle",
                    "qc_status",
                    "is_current",
                    "active_job_id",
                ]
            )
        ],
    ]
    for index_name in index_names:
        op.drop_index(index_name, table_name="render_executions")
    op.drop_table("render_executions")

    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_CURRENT),
            type_=_enum("job_kind", _JOB_PREVIOUS),
            existing_nullable=False,
        )
