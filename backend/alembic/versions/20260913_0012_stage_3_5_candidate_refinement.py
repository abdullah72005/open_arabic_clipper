"""Stage 3.5 candidate-scoped refinement persistence and job wiring.

Revision ID: 20260913_0012
Revises: 20260912_0011
Create Date: 2026-09-13 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260913_0012"
down_revision: str | None = "20260912_0011"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_JOB_PREVIOUS = (
    "INGEST",
    "TRANSCRIPTION",
    "RECONSTRUCTION",
    "PROBE",
    "CANDIDATE_ANALYSIS",
)
_JOB_CURRENT = (*_JOB_PREVIOUS, "CANDIDATE_REFINEMENT")

_REFINEMENT_PRIORITY = ("INDEX", "CANDIDATE", "FINAL_CLIP")
_REFINEMENT_STATUS = (
    "QUEUED",
    "REFINING",
    "CANDIDATE_REFINED",
    "FINAL_TRANSCRIPT_READY",
    "NEEDS_MANUAL_TRANSCRIPT_REVIEW",
    "PROVIDER_DEGRADED",
    "REFINEMENT_FAILED",
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
        "candidate_refinements",
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
            "priority",
            _enum("refinement_priority", _REFINEMENT_PRIORITY),
            nullable=False,
        ),
        sa.Column(
            "status",
            _enum("refinement_status", _REFINEMENT_STATUS),
            nullable=False,
        ),
        sa.Column("active_job_id", sa.String(length=64)),
        sa.Column("coarse_start", sa.Float(), nullable=False),
        sa.Column("coarse_end", sa.Float(), nullable=False),
        sa.Column("context_start", sa.Float(), nullable=False),
        sa.Column("context_end", sa.Float(), nullable=False),
        sa.Column("refined_start", sa.Float()),
        sa.Column("refined_end", sa.Float()),
        sa.Column("audio_relative_path", sa.String(length=1024)),
        sa.Column("audio_content_hash", sa.String(length=64)),
        sa.Column("audio_input_fingerprint", sa.String(length=64)),
        sa.Column("automatic_transcript", sa.Text(), nullable=False),
        sa.Column("manual_transcript", sa.Text()),
        sa.Column("final_transcript", sa.Text(), nullable=False),
        sa.Column("word_timestamps", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("quality_level", sa.String(length=32), nullable=False),
        sa.Column("dialect_profile", sa.String(length=32)),
        sa.Column("dialect_confidence", sa.Float(), nullable=False),
        sa.Column("code_switch_evidence", sa.JSON(), nullable=False),
        sa.Column("transcript_evidence", sa.JSON(), nullable=False),
        sa.Column("entity_evidence", sa.JSON(), nullable=False),
        sa.Column("unresolved_spans", sa.JSON(), nullable=False),
        sa.Column("provider_evidence", sa.JSON(), nullable=False),
        sa.Column("routing_evidence", sa.JSON(), nullable=False),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("output_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("component_fingerprints", sa.JSON(), nullable=False),
        sa.Column("policy_version", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("validation_version", sa.String(length=64), nullable=False),
        sa.Column("cache_eligible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("processing_duration", sa.Float()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "clip_candidate_id",
            "priority",
            name="uq_candidate_refinements_candidate_priority",
        ),
        sa.CheckConstraint(
            "coarse_start >= 0 AND coarse_end > coarse_start",
            name="ck_candidate_refinements_coarse_range",
        ),
        sa.CheckConstraint(
            "context_start >= 0 AND context_end > context_start",
            name="ck_candidate_refinements_context_range",
        ),
        sa.CheckConstraint(
            "refined_start >= 0 AND refined_end >= refined_start",
            name="ck_candidate_refinements_refined_range",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_candidate_refinements_confidence",
        ),
        sa.CheckConstraint(
            "dialect_confidence >= 0 AND dialect_confidence <= 1",
            name="ck_candidate_refinements_dialect_confidence",
        ),
    )
    op.create_index(
        "ix_candidate_refinements_source_video_id",
        "candidate_refinements",
        ["source_video_id"],
    )
    op.create_index(
        "ix_candidate_refinements_clip_candidate_id",
        "candidate_refinements",
        ["clip_candidate_id"],
    )
    op.create_index("ix_candidate_refinements_priority", "candidate_refinements", ["priority"])
    op.create_index("ix_candidate_refinements_status", "candidate_refinements", ["status"])
    op.create_index(
        "ix_candidate_refinements_audio_content_hash",
        "candidate_refinements",
        ["audio_content_hash"],
    )
    op.create_index(
        "ix_candidate_refinements_active_job_id",
        "candidate_refinements",
        ["active_job_id"],
    )

    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(sa.Column("candidate_refinement_id", sa.Uuid(as_uuid=True), nullable=True))
        batch.create_foreign_key(
            "fk_processing_jobs_candidate_refinement_id",
            "candidate_refinements",
            ["candidate_refinement_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        "ix_processing_jobs_candidate_refinement_id",
        "processing_jobs",
        ["candidate_refinement_id"],
    )


def downgrade() -> None:
    # Stage 3.5-only jobs are the only rows that may reference refinements.
    # Remove them and clear any residual references before the FK/column and
    # the parent table disappear. Stage 1-3 rows are untouched.
    op.execute("DELETE FROM processing_jobs WHERE kind = 'CANDIDATE_REFINEMENT'")
    op.execute(
        "UPDATE processing_jobs SET candidate_refinement_id = NULL "
        "WHERE candidate_refinement_id IS NOT NULL"
    )
    op.drop_index(
        "ix_processing_jobs_candidate_refinement_id",
        table_name="processing_jobs",
    )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_column("candidate_refinement_id")
    op.drop_table("candidate_refinements")
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_CURRENT),
            type_=_enum("job_kind", _JOB_PREVIOUS),
            existing_nullable=False,
        )
