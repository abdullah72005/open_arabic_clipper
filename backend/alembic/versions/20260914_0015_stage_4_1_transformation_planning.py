"""Stage 4.1 transformation plan generation persistence and job wiring.

Revision ID: 20260914_0015
Revises: 20260914_0014
Create Date: 2026-09-14 15:00:00.000000

Adds a candidate-scoped Stage 4.1 job kind plus two tables. It adds no
``PipelineStage``, no ``PipelineRun`` column, and never changes source lifecycle
values. The downgrade removes only Stage 4.1 jobs/schema and preserves every
Stage 1-4.0 source, transcript, candidate, refinement, analysis, strategy, job,
and pipeline run.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260914_0015"
down_revision: str | None = "20260914_0014"
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
)
_JOB_CURRENT = (*_JOB_PREVIOUS, "TRANSFORMATION_PLANNING")

_PLAN_EXECUTION_STATUS = (
    "QUEUED",
    "PLANNING",
    "COMPLETE",
    "PROVIDER_DEGRADED",
    "FAILED",
    "CANCELLED",
)
_PLAN_SEMANTIC_OUTCOME = (
    "PLANS_GENERATED",
    "PLANS_GENERATED_WITH_VERIFICATION_REQUIRED",
    "PLANNING_DEFERRED",
    "NO_VALID_PLAN_FROM_STRATEGY",
    "PROVIDER_UNAVAILABLE",
)
_PLAN_STATUS = (
    "PLAN_GENERATED",
    "PLAN_GENERATED_WITH_VERIFICATION_REQUIRED",
)
_PROVIDER_MODE = ("deterministic", "adaptive", "local_only")
_STRATEGY_TYPE = (
    "CONTEXT_HOOK",
    "HOOK_PLUS_TAKEAWAY",
    "EXPLANATORY",
    "COMMENTARY",
    "ANALYSIS",
    "SUMMARY",
    "COMPARISON",
    "COUNTERPOINT",
    "REACTION_FRAMING",
    "QUESTION_EXPLANATION_TAKEAWAY",
    "CLAIM_CONTEXT_CONCLUSION",
    "DEBATE_CONTEXT",
    "NEWS_CONTEXT",
    "SOURCE_AS_EVIDENCE",
    "SOURCE_LED_MINIMAL",
)
_INTENSITY = ("MINIMAL", "MODERATE", "STRONG")
_ORIGIN = ("DETERMINISTIC", "PROVIDER")


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
        "transformation_plan_sets",
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
            "transformation_analysis_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_eligibility_analyses.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "refinement_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("candidate_refinements.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("refinement_priority", sa.String(length=32), nullable=False, server_default=""),
        sa.Column(
            "refinement_quality_level", sa.String(length=32), nullable=False, server_default=""
        ),
        sa.Column(
            "execution_status",
            _enum("transformation_plan_execution_status", _PLAN_EXECUTION_STATUS),
            nullable=False,
            server_default="QUEUED",
        ),
        sa.Column(
            "planning_outcome",
            _enum("transformation_plan_semantic_outcome", _PLAN_SEMANTIC_OUTCOME),
            nullable=True,
        ),
        sa.Column("outcome_reasons", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("stage40_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("target_context", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "provider_mode",
            _enum("transformation_planning_provider_mode", _PROVIDER_MODE),
            nullable=False,
            server_default="deterministic",
        ),
        sa.Column("provider_identity", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "provider_status", sa.String(length=64), nullable=False, server_default="DETERMINISTIC"
        ),
        sa.Column("provider_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("strategy_attempts", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("output_fingerprint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column(
            "policy_version", sa.String(length=64), nullable=False, server_default="stage4.1-v1"
        ),
        sa.Column(
            "schema_version",
            sa.String(length=64),
            nullable=False,
            server_default="stage4.1-schema-v1",
        ),
        sa.Column(
            "validation_version",
            sa.String(length=64),
            nullable=False,
            server_default="stage4.1-validation-v1",
        ),
        sa.Column("cache_eligible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("active_job_id", sa.String(length=64)),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("processing_duration", sa.Float()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "cache_eligible IN (0, 1)",
            name="ck_transformation_plan_sets_cache_eligible_bool",
        ),
    )
    for column in (
        "source_video_id",
        "clip_candidate_id",
        "transformation_analysis_id",
        "refinement_id",
        "execution_status",
        "planning_outcome",
        "active_job_id",
    ):
        op.create_index(
            f"ix_transformation_plan_sets_{column}", "transformation_plan_sets", [column]
        )

    op.create_table(
        "transformation_plans",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "plan_set_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_plan_sets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "strategy_candidate_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_strategy_candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("plan_key", sa.String(length=160), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "status",
            _enum("transformation_plan_status", _PLAN_STATUS),
            nullable=False,
        ),
        sa.Column("generation_rank", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "strategy_type",
            _enum("transformation_plan_strategy_type", _STRATEGY_TYPE),
            nullable=False,
        ),
        sa.Column(
            "intensity",
            _enum("transformation_plan_intensity", _INTENSITY),
            nullable=False,
            server_default="MINIMAL",
        ),
        sa.Column("strategy_fingerprint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("strategy_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("source_dialect", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("target_audience", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("blocks", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("hero_block_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("hero_source_start", sa.Float()),
        sa.Column("hero_source_end", sa.Float()),
        sa.Column("hero_appearance_time", sa.Float(), nullable=False, server_default="0"),
        sa.Column("preservation_constraints", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("original_value_kinds", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("original_value_reasons", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("narration_need", sa.String(length=32), nullable=False, server_default="NONE"),
        sa.Column("narration_requirements", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("external_fact_dependencies", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("required_context", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("derived_durations", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("hook_payoff_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("degraded_rules", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("stage40_risk", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("planner_confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "generation_origin",
            _enum("transformation_plan_origin", _ORIGIN),
            nullable=False,
            server_default="DETERMINISTIC",
        ),
        sa.Column("planning_provider_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "provider_input_fingerprint",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "plan_output_fingerprint", sa.String(length=64), nullable=False, server_default=""
        ),
        sa.Column("structure_signature", sa.String(length=512), nullable=False, server_default=""),
        sa.Column(
            "policy_version", sa.String(length=64), nullable=False, server_default="stage4.1-v1"
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "plan_set_id",
            "strategy_candidate_id",
            name="uq_transformation_plans_set_strategy",
        ),
        sa.CheckConstraint("generation_rank >= 1", name="ck_transformation_plans_rank_positive"),
        sa.CheckConstraint(
            "hero_block_index >= 0", name="ck_transformation_plans_hero_nonnegative"
        ),
        sa.CheckConstraint(
            "planner_confidence >= 0 AND planner_confidence <= 1",
            name="ck_transformation_plans_confidence",
        ),
    )
    for column in (
        "plan_set_id",
        "strategy_candidate_id",
        "plan_key",
        "is_current",
    ):
        op.create_index(f"ix_transformation_plans_{column}", "transformation_plans", [column])

    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(
            sa.Column("transformation_plan_set_id", sa.Uuid(as_uuid=True), nullable=True)
        )
        batch.create_foreign_key(
            "fk_processing_jobs_transformation_plan_set_id",
            "transformation_plan_sets",
            ["transformation_plan_set_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        "ix_processing_jobs_transformation_plan_set_id",
        "processing_jobs",
        ["transformation_plan_set_id"],
    )


def downgrade() -> None:
    op.execute("DELETE FROM processing_jobs WHERE kind = 'TRANSFORMATION_PLANNING'")
    op.execute(
        "UPDATE processing_jobs SET transformation_plan_set_id = NULL "
        "WHERE transformation_plan_set_id IS NOT NULL"
    )
    op.drop_index(
        "ix_processing_jobs_transformation_plan_set_id",
        table_name="processing_jobs",
    )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_column("transformation_plan_set_id")
    op.drop_table("transformation_plans")
    op.drop_table("transformation_plan_sets")
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_CURRENT),
            type_=_enum("job_kind", _JOB_PREVIOUS),
            existing_nullable=False,
        )
