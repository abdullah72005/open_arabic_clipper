"""Stage 4.0 transformation eligibility and strategy persistence and job wiring.

Revision ID: 20260914_0014
Revises: 20260914_0013
Create Date: 2026-09-14 12:00:00.000000

Adds a candidate-scoped Stage 4.0 job kind plus two tables. It adds no
``PipelineStage``, no ``PipelineRun`` column, and never changes source
lifecycle values. The downgrade removes only Stage 4.0 rows and preserves every
Stage 1-3.7 source, transcript, candidate, refinement, job, and pipeline run.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260914_0014"
down_revision: str | None = "20260914_0013"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_JOB_PREVIOUS = (
    "INGEST",
    "TRANSCRIPTION",
    "RECONSTRUCTION",
    "PROBE",
    "CANDIDATE_ANALYSIS",
    "CANDIDATE_REFINEMENT",
)
_JOB_CURRENT = (*_JOB_PREVIOUS, "TRANSFORMATION_ELIGIBILITY")

_EXECUTION_STATUS = (
    "QUEUED",
    "ANALYZING",
    "COMPLETE",
    "PROVIDER_DEGRADED",
    "FAILED",
    "CANCELLED",
)
_ELIGIBILITY_OUTCOME = (
    "ELIGIBLE_FOR_TRANSFORMATION",
    "ELIGIBLE_WITH_CAUTION",
    "TRANSFORMATION_REQUIRED",
    "NO_TRANSFORMATION_STRATEGY_WORTH_USING",
    "INSUFFICIENT_TRANSCRIPT_CONFIDENCE",
    "INSUFFICIENT_CONTEXT",
    "UNRESOLVED_POLICY_OR_PROVENANCE_RISK",
)
_INTENSITY = ("MINIMAL", "MODERATE", "STRONG")
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
_DISPOSITION = ("RECOMMENDED", "REJECTED")
_ORIGIN = ("DETERMINISTIC", "PROVIDER")
_SUBSTANTIVE_VALUE = (
    "MISSING_CONTEXT",
    "INFERENCE",
    "EXPLANATION",
    "COMPARISON",
    "COUNTERPOINT",
    "VERIFICATION_CORRECTION",
    "SYNTHESIS",
    "AUTHORED_THESIS",
    "USEFUL_TAKEAWAY",
    "SOURCE_AS_EVIDENCE",
)
_EXTERNAL_FACT = ("NOT_REQUIRED", "REQUIRES_EXTERNAL_FACT_VERIFICATION")
_PROVIDER_MODE = ("deterministic", "adaptive", "local_only")


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
        "transformation_eligibility_analyses",
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
            _enum("transformation_execution_status", _EXECUTION_STATUS),
            nullable=False,
            server_default="QUEUED",
        ),
        sa.Column(
            "eligibility_outcome",
            _enum("transformation_eligibility_outcome", _ELIGIBILITY_OUTCOME),
            nullable=True,
        ),
        sa.Column("eligibility_reasons", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("assessments", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("source_moment", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("platform_risk", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "transformation_intensity",
            _enum("transformation_intensity", _INTENSITY),
            nullable=True,
        ),
        sa.Column(
            "provider_mode",
            _enum("transformation_provider_mode", _PROVIDER_MODE),
            nullable=False,
            server_default="deterministic",
        ),
        sa.Column("provider_identity", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "provider_status", sa.String(length=64), nullable=False, server_default="DETERMINISTIC"
        ),
        sa.Column("provider_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "provider_input_fingerprint",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("output_fingerprint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column(
            "policy_version", sa.String(length=64), nullable=False, server_default="stage4.0-v1"
        ),
        sa.Column(
            "schema_version",
            sa.String(length=64),
            nullable=False,
            server_default="stage4.0-schema-v1",
        ),
        sa.Column(
            "validation_version",
            sa.String(length=64),
            nullable=False,
            server_default="stage4.0-validation-v1",
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
            name="ck_transformation_analyses_cache_eligible_bool",
        ),
    )
    op.create_index(
        "ix_transformation_eligibility_analyses_source_video_id",
        "transformation_eligibility_analyses",
        ["source_video_id"],
    )
    op.create_index(
        "ix_transformation_eligibility_analyses_clip_candidate_id",
        "transformation_eligibility_analyses",
        ["clip_candidate_id"],
    )
    op.create_index(
        "ix_transformation_eligibility_analyses_refinement_id",
        "transformation_eligibility_analyses",
        ["refinement_id"],
    )
    op.create_index(
        "ix_transformation_eligibility_analyses_execution_status",
        "transformation_eligibility_analyses",
        ["execution_status"],
    )
    op.create_index(
        "ix_transformation_eligibility_analyses_eligibility_outcome",
        "transformation_eligibility_analyses",
        ["eligibility_outcome"],
    )
    op.create_index(
        "ix_transformation_eligibility_analyses_active_job_id",
        "transformation_eligibility_analyses",
        ["active_job_id"],
    )

    op.create_table(
        "transformation_strategy_candidates",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "analysis_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("transformation_eligibility_analyses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("strategy_key", sa.String(length=128), nullable=False),
        sa.Column(
            "strategy_type",
            _enum("transformation_strategy_type", _STRATEGY_TYPE),
            nullable=False,
        ),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "disposition",
            _enum("transformation_strategy_disposition", _DISPOSITION),
            nullable=False,
        ),
        sa.Column("rank", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "intensity",
            _enum("transformation_intensity", _INTENSITY),
            nullable=False,
            server_default="MINIMAL",
        ),
        sa.Column("direction_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("added_value_focus", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "substantive_value_kind",
            _enum("transformation_substantive_value_kind", _SUBSTANTIVE_VALUE),
            nullable=False,
            server_default="MISSING_CONTEXT",
        ),
        sa.Column("source_moment_role", sa.String(length=64), nullable=False, server_default="HERO"),
        sa.Column("preservation_requirements", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("retention_preservation", sa.Float(), nullable=False, server_default="0"),
        sa.Column("source_moment_damage_risk", sa.Float(), nullable=False, server_default="0"),
        sa.Column("added_value_density", sa.Float(), nullable=False, server_default="0"),
        sa.Column("originality_potential", sa.Float(), nullable=False, server_default="0"),
        sa.Column("source_dominance_risk", sa.Float(), nullable=False, server_default="0"),
        sa.Column("generic_filler_risk", sa.Float(), nullable=False, server_default="0"),
        sa.Column("redundant_commentary_risk", sa.Float(), nullable=False, server_default="0"),
        sa.Column("template_staleness_risk", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "external_verification_requirement",
            _enum("transformation_external_fact_requirement", _EXTERNAL_FACT),
            nullable=False,
            server_default="NOT_REQUIRED",
        ),
        sa.Column("verification_requirements", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("rejection_reasons", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "origin",
            _enum("transformation_strategy_origin", _ORIGIN),
            nullable=False,
            server_default="DETERMINISTIC",
        ),
        sa.Column("provider_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("strategy_fingerprint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column(
            "policy_version", sa.String(length=64), nullable=False, server_default="stage4.0-v1"
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "analysis_id",
            "strategy_type",
            name="uq_transformation_strategies_analysis_type",
        ),
        sa.CheckConstraint("rank >= 0", name="ck_transformation_strategies_rank_nonnegative"),
        sa.CheckConstraint(
            "retention_preservation >= 0 AND retention_preservation <= 1",
            name="ck_transformation_strategies_retention",
        ),
        sa.CheckConstraint(
            "source_moment_damage_risk >= 0 AND source_moment_damage_risk <= 1",
            name="ck_transformation_strategies_damage",
        ),
        sa.CheckConstraint(
            "added_value_density >= 0 AND added_value_density <= 1",
            name="ck_transformation_strategies_added_value",
        ),
        sa.CheckConstraint(
            "originality_potential >= 0 AND originality_potential <= 1",
            name="ck_transformation_strategies_originality",
        ),
        sa.CheckConstraint(
            "source_dominance_risk >= 0 AND source_dominance_risk <= 1",
            name="ck_transformation_strategies_source_dominance",
        ),
        sa.CheckConstraint(
            "generic_filler_risk >= 0 AND generic_filler_risk <= 1",
            name="ck_transformation_strategies_filler",
        ),
        sa.CheckConstraint(
            "redundant_commentary_risk >= 0 AND redundant_commentary_risk <= 1",
            name="ck_transformation_strategies_redundancy",
        ),
        sa.CheckConstraint(
            "template_staleness_risk >= 0 AND template_staleness_risk <= 1",
            name="ck_transformation_strategies_template",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_transformation_strategies_confidence",
        ),
    )
    op.create_index(
        "ix_transformation_strategy_candidates_analysis_id",
        "transformation_strategy_candidates",
        ["analysis_id"],
    )
    op.create_index(
        "ix_transformation_strategy_candidates_strategy_key",
        "transformation_strategy_candidates",
        ["strategy_key"],
    )
    op.create_index(
        "ix_transformation_strategy_candidates_strategy_type",
        "transformation_strategy_candidates",
        ["strategy_type"],
    )
    op.create_index(
        "ix_transformation_strategy_candidates_is_current",
        "transformation_strategy_candidates",
        ["is_current"],
    )
    op.create_index(
        "ix_transformation_strategy_candidates_disposition",
        "transformation_strategy_candidates",
        ["disposition"],
    )

    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(
            sa.Column("transformation_analysis_id", sa.Uuid(as_uuid=True), nullable=True)
        )
        batch.create_foreign_key(
            "fk_processing_jobs_transformation_analysis_id",
            "transformation_eligibility_analyses",
            ["transformation_analysis_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        "ix_processing_jobs_transformation_analysis_id",
        "processing_jobs",
        ["transformation_analysis_id"],
    )


def downgrade() -> None:
    # Stage 4.0-only jobs are the only rows that may reference Stage 4.0
    # analyses. Remove them and clear residual references before the FK column
    # and the parent tables disappear. Stage 1-3.7 rows are untouched and no
    # source lifecycle value is changed.
    op.execute("DELETE FROM processing_jobs WHERE kind = 'TRANSFORMATION_ELIGIBILITY'")
    op.execute(
        "UPDATE processing_jobs SET transformation_analysis_id = NULL "
        "WHERE transformation_analysis_id IS NOT NULL"
    )
    op.drop_index(
        "ix_processing_jobs_transformation_analysis_id",
        table_name="processing_jobs",
    )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_column("transformation_analysis_id")
    op.drop_table("transformation_strategy_candidates")
    op.drop_table("transformation_eligibility_analyses")
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_CURRENT),
            type_=_enum("job_kind", _JOB_PREVIOUS),
            existing_nullable=False,
        )
