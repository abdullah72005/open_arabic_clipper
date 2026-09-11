"""Stage 3 candidate analysis persistence and source provenance.

Revision ID: 20260912_0011
Revises: 20260910_0010
Create Date: 2026-09-12 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260912_0011"
down_revision: str | None = "20260910_0010"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_LIFECYCLE_PREVIOUS = (
    "INGEST",
    "PROBE",
    "READY_FOR_TRANSCRIPTION",
    "AUDIO_EXTRACTION",
    "TRANSCRIPTION",
    "TRANSCRIPT_NORMALIZATION",
    "CONTEXTUAL_RECONSTRUCTION",
    "AUDIO_ANALYSIS",
    "READY_FOR_ANALYSIS",
)
_LIFECYCLE_CURRENT = (
    *_LIFECYCLE_PREVIOUS,
    "CANDIDATE_ANALYSIS",
    "READY_FOR_REFINEMENT",
)
_JOB_PREVIOUS = ("INGEST", "PROBE", "TRANSCRIPTION", "RECONSTRUCTION")
_JOB_CURRENT = (*_JOB_PREVIOUS, "CANDIDATE_ANALYSIS")
_MEDIA_ORIGIN = (
    "YOUTUBE_CREATOR_VIDEO",
    "PODCAST_INTERVIEW",
    "MOVIE_TV",
    "NEWS_CLIP",
    "SPORTS_BROADCAST",
    "OTHER",
)
_CONTENT_TYPE = (
    "EDUCATIONAL",
    "CONTROVERSIAL_OPINION",
    "FUNNY",
    "STORY",
    "SURPRISING_FACT",
    "EMOTIONAL",
    "NEWS_CURRENT_EVENT",
    "INTERVIEW_INSIGHT",
    "DEBATE",
    "MOTIVATIONAL",
    "TUTORIAL",
    "ANALYSIS",
    "REACTION_WORTHY",
    "OTHER",
)
_DISPOSITION = (
    "CANDIDATE",
    "CANDIDATE_NEEDS_REFINEMENT",
    "DO_NOT_CLIP",
    "DO_NOT_CLIP_RECENTLY_REDUNDANT",
)
_ORIGINALITY = ("NOT_INDICATED", "UNDETERMINED", "TRANSFORMATION_REQUIRED")
_RIGHTS_RISK = ("LOW", "UNDETERMINED", "ELEVATED")
_SEMANTIC_MODE = ("deterministic", "adaptive", "local_only")

_SCORE_COLUMNS = (
    "clip_score",
    "short_form_score",
    "moment_density_score",
    "boredom_risk_score",
    "ending_quality_score",
    "loopability_score",
    "engagement_confidence",
    "transcript_confidence",
    "audio_confidence",
    "boundary_confidence",
    "uncertainty_severity",
    "idea_novelty_score",
    "topic_novelty_score",
    "recent_semantic_similarity_risk",
)


def _enum(name: str, values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True)


def upgrade() -> None:
    with op.batch_alter_table("source_videos") as batch:
        batch.alter_column(
            "lifecycle_state",
            existing_type=_enum("source_lifecycle_state", _LIFECYCLE_PREVIOUS),
            type_=_enum("source_lifecycle_state", _LIFECYCLE_CURRENT),
            existing_nullable=False,
        )
        batch.add_column(
            sa.Column(
                "media_origin",
                _enum("media_origin_type", _MEDIA_ORIGIN),
                nullable=False,
                server_default="OTHER",
            )
        )
        batch.add_column(
            sa.Column(
                "provenance_metadata",
                sa.JSON(),
                nullable=False,
                server_default="{}",
            )
        )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_PREVIOUS),
            type_=_enum("job_kind", _JOB_CURRENT),
            existing_nullable=False,
        )
    with op.batch_alter_table("pipeline_runs") as batch:
        batch.alter_column(
            "stage",
            existing_type=_enum("pipeline_stage", _LIFECYCLE_PREVIOUS),
            type_=_enum("pipeline_stage", _LIFECYCLE_CURRENT),
            existing_nullable=False,
        )

    op.create_table(
        "candidate_analyses",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "source_video_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("source_videos.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("output_fingerprint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("policy_version", sa.String(length=64), nullable=False, server_default="stage3-v1"),
        sa.Column(
            "scoring_version",
            sa.String(length=64),
            nullable=False,
            server_default="stage3-scoring-v1",
        ),
        sa.Column(
            "semantic_provider_mode",
            _enum("semantic_provider_mode", _SEMANTIC_MODE),
            nullable=False,
            server_default="deterministic",
        ),
        sa.Column("provider_identity", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "provider_status", sa.String(length=64), nullable=False, server_default="DETERMINISTIC"
        ),
        sa.Column("cache_eligible", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("processing_duration", sa.Float()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_candidate_analyses_source_video_id", "candidate_analyses", ["source_video_id"]
    )

    op.create_table(
        "clip_candidates",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "source_video_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("source_videos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "candidate_analysis_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("candidate_analyses.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("candidate_key", sa.String(length=128), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("disposition", _enum("candidate_disposition", _DISPOSITION), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("start_segment_index", sa.Integer(), nullable=False),
        sa.Column("end_segment_index", sa.Integer(), nullable=False),
        sa.Column("segment_indexes", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("transcript_excerpt", sa.Text(), nullable=False, server_default=""),
        sa.Column("evidence_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("primary_content_type", _enum("content_type", _CONTENT_TYPE), nullable=False),
        sa.Column("secondary_content_types", sa.JSON(), nullable=False, server_default="[]"),
        *(
            sa.Column(column, sa.Float(), nullable=False, server_default="0")
            for column in _SCORE_COLUMNS
        ),
        sa.Column("refinement_reasons", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("refinement_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("provenance_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("rights_risk", _enum("rights_risk", _RIGHTS_RISK), nullable=False),
        sa.Column(
            "originality_risk", _enum("originality_risk", _ORIGINALITY), nullable=False
        ),
        sa.Column("dialect_profile", sa.String(length=32)),
        sa.Column("dialect_confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("code_switch_suspected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("hooks", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("idea_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("topic_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("idea_signature", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("topic_signature", sa.String(length=64), nullable=False, server_default=""),
        sa.Column(
            "provider_input_fingerprint", sa.String(length=64), nullable=False, server_default=""
        ),
        sa.Column("provider_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("analysis_fingerprint", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("policy_version", sa.String(length=64), nullable=False, server_default="stage3-v1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "start_time >= 0 AND end_time > start_time", name="ck_candidates_time_range"
        ),
        sa.CheckConstraint(
            "start_segment_index >= 0 AND end_segment_index >= start_segment_index",
            name="ck_candidates_segment_range",
        ),
        sa.CheckConstraint(
            "dialect_confidence >= 0 AND dialect_confidence <= 1", name="ck_cand_dialect"
        ),
        *(
            sa.CheckConstraint(
                f"{column} >= 0 AND {column} <= 1",
                name=f"ck_clip_candidates_{column}_bounds",
            )
            for column in _SCORE_COLUMNS
        ),
        sa.UniqueConstraint("candidate_key", name="uq_clip_candidates_candidate_key"),
    )
    op.create_index("ix_clip_candidates_source_video_id", "clip_candidates", ["source_video_id"])
    op.create_index("ix_clip_candidates_candidate_key", "clip_candidates", ["candidate_key"])
    op.create_index("ix_clip_candidates_is_current", "clip_candidates", ["is_current"])
    op.create_index("ix_clip_candidates_disposition", "clip_candidates", ["disposition"])
    op.create_index(
        "ix_clip_candidates_candidate_analysis_id", "clip_candidates", ["candidate_analysis_id"]
    )


def downgrade() -> None:
    op.drop_table("clip_candidates")
    op.drop_table("candidate_analyses")
    # Remove Stage-3-only history and map live source state back to the closest
    # valid pre-Stage-3 lifecycle before narrowing any enum/check constraint.
    # Pre-existing Stage 1/2/2.5/2.7/2.7.1 source/transcript/audio data is kept.
    op.execute(
        "DELETE FROM pipeline_runs WHERE stage IN ('CANDIDATE_ANALYSIS', 'READY_FOR_REFINEMENT')"
    )
    op.execute("DELETE FROM processing_jobs WHERE kind = 'CANDIDATE_ANALYSIS'")
    op.execute(
        "UPDATE source_videos SET lifecycle_state = 'READY_FOR_ANALYSIS' "
        "WHERE lifecycle_state IN ('CANDIDATE_ANALYSIS', 'READY_FOR_REFINEMENT')"
    )
    with op.batch_alter_table("pipeline_runs") as batch:
        batch.alter_column(
            "stage",
            existing_type=_enum("pipeline_stage", _LIFECYCLE_CURRENT),
            type_=_enum("pipeline_stage", _LIFECYCLE_PREVIOUS),
            existing_nullable=False,
        )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_CURRENT),
            type=_enum("job_kind", _JOB_PREVIOUS),
            existing_nullable=False,
        )
    with op.batch_alter_table("source_videos") as batch:
        batch.drop_constraint("media_origin_type", type_="check")
        batch.drop_column("provenance_metadata")
        batch.drop_column("media_origin")
        batch.alter_column(
            "lifecycle_state",
            existing_type=_enum("source_lifecycle_state", _LIFECYCLE_CURRENT),
            type=_enum("source_lifecycle_state", _LIFECYCLE_PREVIOUS),
            existing_nullable=False,
        )
