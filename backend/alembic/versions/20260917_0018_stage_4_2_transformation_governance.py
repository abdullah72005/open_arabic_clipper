"""Stage 4.2 retention/originality/platform-risk governor persistence.

Revision ID: 20260917_0018
Revises: 20260915_0017
Create Date: 2026-09-17 12:00:00.000000

Adds a candidate-scoped ``TRANSFORMATION_GOVERNANCE`` job kind plus two tables
(``transformation_governance_sets`` and ``transformation_governance_results``)
and one nullable ``processing_jobs.transformation_governance_set_id`` FK. It
adds no ``PipelineStage``, no ``PipelineRun`` column, never changes source
lifecycle values, and never selects a plan. The downgrade removes only Stage 4.2
jobs/schema and preserves every Stage 1-4.1 source, transcript, candidate,
refinement, analysis, strategy, plan set, plan, job, and pipeline run.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260917_0018"
down_revision: str | None = "20260915_0017"
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
)
_JOB_CURRENT = (*_JOB_PREVIOUS, "TRANSFORMATION_GOVERNANCE")

_GOVERNANCE_EXECUTION_STATUS = (
    "QUEUED",
    "GOVERNING",
    "COMPLETE",
    "PROVIDER_DEGRADED",
    "FAILED",
    "CANCELLED",
)
_GOVERNANCE_SEMANTIC_OUTCOME = (
    "PLANS_ELIGIBLE_FOR_SELECTION",
    "NO_GOVERNOR_APPROVED_PLAN",
    "GOVERNANCE_DEFERRED",
)
_GOVERNANCE_PLAN_STATUS = (
    "APPROVED_FOR_SELECTION",
    "APPROVED_WITH_CAUTION",
    "BLOCKED_PENDING_VERIFICATION",
    "REVISION_REQUIRED",
    "REJECTED_BY_GOVERNOR",
    "GOVERNANCE_DEFERRED",
)
_PROVIDER_MODE = ("deterministic", "adaptive", "local_only")
_ELIGIBLE = "('APPROVED_FOR_SELECTION', 'APPROVED_WITH_CAUTION')"


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
        "transformation_governance_sets",
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
            unique=True,
        ),
        sa.Column(
            "transformation_plan_set_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_plan_sets.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "transformation_analysis_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_eligibility_analyses.id", ondelete="CASCADE"),
            nullable=False,
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
            "execution_status",
            _enum("transformation_governance_execution_status", _GOVERNANCE_EXECUTION_STATUS),
            nullable=False,
            server_default="QUEUED",
        ),
        sa.Column(
            "governance_outcome",
            _enum("transformation_governance_semantic_outcome", _GOVERNANCE_SEMANTIC_OUTCOME),
            nullable=True,
        ),
        sa.Column("outcome_reasons", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("summary_counts", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("stage40_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("target_context", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "provider_mode",
            _enum("transformation_governance_provider_mode", _PROVIDER_MODE),
            nullable=False,
            server_default="deterministic",
        ),
        sa.Column("provider_identity", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("provider_status", sa.String(64), nullable=False, server_default="DETERMINISTIC"),
        sa.Column("provider_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("plan_attempts", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column(
            "platform_policy_profile_version", sa.String(80), nullable=False, server_default=""
        ),
        sa.Column("platform_policy_checked_at", sa.String(16), nullable=False, server_default=""),
        sa.Column("input_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("output_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("policy_version", sa.String(64), nullable=False, server_default="stage4.2-v1"),
        sa.Column(
            "schema_version", sa.String(64), nullable=False, server_default="stage4.2-schema-v1"
        ),
        sa.Column(
            "validation_version",
            sa.String(64),
            nullable=False,
            server_default="stage4.2-validation-v1",
        ),
        sa.Column("cache_eligible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("active_job_id", sa.String(64), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("processing_duration", sa.Float(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "cache_eligible IN (true, false)",
            name="ck_transformation_governance_sets_cache_eligible_bool",
        ),
    )
    op.create_index(
        "ix_transformation_governance_sets_source_video_id",
        "transformation_governance_sets",
        ["source_video_id"],
    )
    op.create_index(
        "ix_transformation_governance_sets_clip_candidate_id",
        "transformation_governance_sets",
        ["clip_candidate_id"],
    )
    op.create_index(
        "ix_transformation_governance_sets_transformation_plan_set_id",
        "transformation_governance_sets",
        ["transformation_plan_set_id"],
    )
    op.create_index(
        "ix_transformation_governance_sets_transformation_analysis_id",
        "transformation_governance_sets",
        ["transformation_analysis_id"],
    )
    op.create_index(
        "ix_transformation_governance_sets_refinement_id",
        "transformation_governance_sets",
        ["refinement_id"],
    )
    op.create_index(
        "ix_transformation_governance_sets_execution_status",
        "transformation_governance_sets",
        ["execution_status"],
    )
    op.create_index(
        "ix_transformation_governance_sets_governance_outcome",
        "transformation_governance_sets",
        ["governance_outcome"],
    )
    op.create_index(
        "ix_transformation_governance_sets_active_job_id",
        "transformation_governance_sets",
        ["active_job_id"],
    )

    op.create_table(
        "transformation_governance_results",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "governance_set_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_governance_sets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "transformation_plan_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_plans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("plan_output_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column(
            "status",
            _enum("transformation_governance_plan_status", _GOVERNANCE_PLAN_STATUS),
            nullable=False,
        ),
        sa.Column("eligible_for_stage4_3", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("severity", sa.String(32), nullable=False, server_default="ADVISORY"),
        sa.Column("hard_gates", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("dimensions", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("verification", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("platform_risk", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("reason_codes", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("warnings", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("remediation", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("governance_provider_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("input_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("output_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "governance_set_id",
            "transformation_plan_id",
            name="uq_transformation_governance_results_set_plan",
        ),
        sa.CheckConstraint(
            "eligible_for_stage4_3 IN (true, false)",
            name="ck_transformation_governance_results_eligible_bool",
        ),
        sa.CheckConstraint(
            f"(eligible_for_stage4_3 AND status IN {_ELIGIBLE}) "
            f"OR (NOT eligible_for_stage4_3 AND status NOT IN {_ELIGIBLE})",
            name="ck_transformation_governance_results_status_eligibility",
        ),
    )
    op.create_index(
        "ix_transformation_governance_results_governance_set_id",
        "transformation_governance_results",
        ["governance_set_id"],
    )
    op.create_index(
        "ix_transformation_governance_results_transformation_plan_id",
        "transformation_governance_results",
        ["transformation_plan_id"],
    )
    op.create_index(
        "ix_transformation_governance_results_eligible_for_stage4_3",
        "transformation_governance_results",
        ["eligible_for_stage4_3"],
    )

    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(
            sa.Column("transformation_governance_set_id", sa.Uuid(as_uuid=True), nullable=True)
        )
        batch.create_foreign_key(
            "fk_processing_jobs_transformation_governance_set_id",
            "transformation_governance_sets",
            ["transformation_governance_set_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        "ix_processing_jobs_transformation_governance_set_id",
        "processing_jobs",
        ["transformation_governance_set_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_processing_jobs_transformation_governance_set_id", table_name="processing_jobs"
    )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_constraint(
            "fk_processing_jobs_transformation_governance_set_id", type_="foreignkey"
        )
        batch.drop_column("transformation_governance_set_id")

    op.drop_index(
        "ix_transformation_governance_results_eligible_for_stage4_3",
        table_name="transformation_governance_results",
    )
    op.drop_index(
        "ix_transformation_governance_results_transformation_plan_id",
        table_name="transformation_governance_results",
    )
    op.drop_index(
        "ix_transformation_governance_results_governance_set_id",
        table_name="transformation_governance_results",
    )
    op.drop_table("transformation_governance_results")

    for index_name in (
        "ix_transformation_governance_sets_active_job_id",
        "ix_transformation_governance_sets_governance_outcome",
        "ix_transformation_governance_sets_execution_status",
        "ix_transformation_governance_sets_refinement_id",
        "ix_transformation_governance_sets_transformation_analysis_id",
        "ix_transformation_governance_sets_transformation_plan_set_id",
        "ix_transformation_governance_sets_clip_candidate_id",
        "ix_transformation_governance_sets_source_video_id",
    ):
        op.drop_index(index_name, table_name="transformation_governance_sets")
    op.drop_table("transformation_governance_sets")

    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_CURRENT),
            type_=_enum("job_kind", _JOB_PREVIOUS),
            existing_nullable=False,
        )
