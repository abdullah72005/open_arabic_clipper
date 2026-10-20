"""Stage 6 transformed-short materialization persistence.

Revision ID: 20261014_0023
Revises: 20260918_0022
Create Date: 2026-10-14 12:00:00.000000

Adds a candidate-scoped ``MATERIALIZATION`` job kind, the ``TRANSFORMED_SHORT``
render artifact purpose, two tables (``channel_production_profiles``,
``clip_materializations``), one nullable ``processing_jobs.materialization_id``
FK, and one nullable ``render_executions.materialization_id`` FK. It adds no
``PipelineStage``, no ``PipelineRun`` column, and never changes source lifecycle
values. The downgrade removes only Stage 6 jobs/schema and preserves every
Stage 1-5.2 source, transcript, candidate, refinement, analysis, strategy, plan,
governance, selection, render contract, visual-composition plan, render
execution, job, and pipeline run.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20261014_0023"
down_revision: str | None = "20260918_0022"
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
    "RENDER_EXECUTION",
)
_JOB_CURRENT = (*_JOB_PREVIOUS, "MATERIALIZATION")

_ARTIFACT_PREVIOUS = ("CORE_SOURCE_VALIDATION",)
_ARTIFACT_CURRENT = ("CORE_SOURCE_VALIDATION", "TRANSFORMED_SHORT")

_MATERIALIZATION_PRIORITY = ("CANDIDATE", "FINAL_CLIP")
_MATERIALIZATION_LIFECYCLE = (
    "QUEUED",
    "RUNNING",
    "AWAITING_RENDER",
    "COMPLETE",
    "BLOCKED",
    "FAILED",
    "CANCELLED",
)
_MATERIALIZATION_OUTCOME = (
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
_SUPPORTING_VISUAL_MODE = ("OFF", "AUTO", "ON")
_SUPPORTING_VISUAL_STYLE = ("AUTO", "ABSTRACT_LOOP", "STOCK_BROLL", "CUSTOM_LIBRARY")

_CACHE_OK = (
    "lifecycle = 'COMPLETE' AND semantic_outcome IN "
    "('READY_FOR_HUMAN_REVIEW', 'MATERIALIZED_WITH_CAUTION')"
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
    with op.batch_alter_table("render_executions") as batch:
        batch.alter_column(
            "artifact_purpose",
            existing_type=_enum("render_artifact_purpose", _ARTIFACT_PREVIOUS),
            type_=_enum("render_artifact_purpose", _ARTIFACT_CURRENT),
            existing_nullable=False,
        )

    op.create_table(
        "channel_production_profiles",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("channel_key", sa.String(120), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("label", sa.String(200), nullable=False, server_default=""),
        sa.Column("target_market", sa.String(80), nullable=False, server_default="UNSPECIFIED"),
        sa.Column(
            "output_language_policy",
            sa.String(80),
            nullable=False,
            server_default="SOURCE_LANGUAGE",
        ),
        sa.Column(
            "register_intent", sa.String(80), nullable=False, server_default="SOURCE_COMPATIBLE"
        ),
        sa.Column("narration_allowed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("tts_provider", sa.String(80), nullable=False, server_default=""),
        sa.Column("tts_model", sa.String(120), nullable=False, server_default=""),
        sa.Column("tts_api_version", sa.String(40), nullable=False, server_default=""),
        sa.Column("tts_voice", sa.String(120), nullable=False, server_default=""),
        sa.Column("tts_style", sa.String(120), nullable=False, server_default=""),
        sa.Column("tts_settings", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "supporting_visual_mode",
            _enum("supporting_visual_mode", _SUPPORTING_VISUAL_MODE),
            nullable=False,
            server_default="OFF",
        ),
        sa.Column(
            "supporting_visual_style",
            _enum("supporting_visual_style", _SUPPORTING_VISUAL_STYLE),
            nullable=False,
            server_default="AUTO",
        ),
        sa.Column("supporting_visual_defaults", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("policy_version", sa.String(64), nullable=False, server_default="stage6-v1"),
        sa.Column(
            "schema_version", sa.String(64), nullable=False, server_default="stage6-schema-v1"
        ),
        sa.Column("fingerprint_version", sa.String(16), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "channel_key", "revision", name="uq_channel_production_profiles_key_revision"
        ),
        sa.CheckConstraint(
            "is_current IN (true, false)",
            name="ck_channel_production_profiles_current_bool",
        ),
        sa.CheckConstraint(
            "narration_allowed IN (true, false)",
            name="ck_channel_production_profiles_narration_allowed_bool",
        ),
        sa.CheckConstraint(
            "revision >= 1", name="ck_channel_production_profiles_revision_positive"
        ),
    )
    op.create_index(
        "ix_channel_production_profiles_channel_key",
        "channel_production_profiles",
        ["channel_key"],
    )
    op.create_index(
        "ix_channel_production_profiles_is_current",
        "channel_production_profiles",
        ["is_current"],
    )
    op.create_index(
        "uq_channel_production_profiles_current_key",
        "channel_production_profiles",
        ["channel_key"],
        unique=True,
        postgresql_where=sa.text("is_current"),
        sqlite_where=sa.text("is_current"),
    )

    op.create_table(
        "clip_materializations",
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
            "channel_production_profile_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("channel_production_profiles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("profile_key", sa.String(120), nullable=False, server_default="default"),
        sa.Column(
            "selection_id",
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
            "refinement_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("candidate_refinements.id", ondelete="SET NULL"),
            nullable=True,
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
            "render_execution_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("render_executions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "active_job_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("processing_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "priority",
            _enum("materialization_priority", _MATERIALIZATION_PRIORITY),
            nullable=False,
            server_default="CANDIDATE",
        ),
        sa.Column(
            "lifecycle",
            _enum("materialization_lifecycle", _MATERIALIZATION_LIFECYCLE),
            nullable=False,
            server_default="QUEUED",
        ),
        sa.Column(
            "semantic_outcome",
            _enum("materialization_outcome", _MATERIALIZATION_OUTCOME),
            nullable=True,
        ),
        sa.Column(
            "human_review_required", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("publication_ready", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("cache_eligible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("repair_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reason_codes", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("error_message", sa.String(2048), nullable=True),
        sa.Column("upstream_fingerprints", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("upstream_snapshot", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("request_payload", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("input_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("output_fingerprint", sa.String(64), nullable=False, server_default=""),
        sa.Column("component_checkpoints", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("call_ledger", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("authored_blocks", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("claim_ledger", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("slot_resolutions", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("narration_references", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("asset_references", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("provider_identity", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("runtime_identity", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("timeline_payload", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("audio_recipe", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("quality_findings", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("repair_actions", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("artifact_reference", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("policy_version", sa.String(64), nullable=False, server_default="stage6-v1"),
        sa.Column(
            "schema_version", sa.String(64), nullable=False, server_default="stage6-schema-v1"
        ),
        sa.Column("fingerprint_version", sa.String(16), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "clip_candidate_id",
            "profile_key",
            "input_fingerprint",
            name="uq_clip_materializations_candidate_profile_input",
        ),
        sa.CheckConstraint(
            "is_current IN (true, false)", name="ck_clip_materializations_current_bool"
        ),
        sa.CheckConstraint(
            "publication_ready IN (true, false)",
            name="ck_clip_materializations_publication_ready_bool",
        ),
        sa.CheckConstraint(
            "human_review_required IN (true, false)",
            name="ck_clip_materializations_human_review_bool",
        ),
        sa.CheckConstraint(
            "cache_eligible IN (true, false)",
            name="ck_clip_materializations_cache_eligible_bool",
        ),
        sa.CheckConstraint(
            "repair_count >= 0 AND repair_count <= 1",
            name="ck_clip_materializations_repair_count_bounded",
        ),
        sa.CheckConstraint(
            f"(cache_eligible AND {_CACHE_OK}) OR (NOT cache_eligible)",
            name="ck_clip_materializations_cache_consistency",
        ),
    )
    for column in (
        "source_video_id",
        "clip_candidate_id",
        "channel_production_profile_id",
        "selection_id",
        "selected_plan_id",
        "refinement_id",
        "render_contract_id",
        "visual_composition_plan_id",
        "render_execution_id",
        "active_job_id",
        "priority",
        "lifecycle",
        "semantic_outcome",
        "is_current",
    ):
        op.create_index(f"ix_clip_materializations_{column}", "clip_materializations", [column])
    op.create_index(
        "uq_clip_materializations_current_scope",
        "clip_materializations",
        ["clip_candidate_id", "profile_key"],
        unique=True,
        postgresql_where=sa.text("is_current"),
        sqlite_where=sa.text("is_current"),
    )

    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(sa.Column("materialization_id", sa.Uuid(as_uuid=True), nullable=True))
        batch.create_foreign_key(
            "fk_processing_jobs_materialization_id",
            "clip_materializations",
            ["materialization_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        "ix_processing_jobs_materialization_id",
        "processing_jobs",
        ["materialization_id"],
    )

    with op.batch_alter_table("render_executions") as batch:
        batch.add_column(sa.Column("materialization_id", sa.Uuid(as_uuid=True), nullable=True))
        batch.create_foreign_key(
            "fk_render_executions_materialization_id",
            "clip_materializations",
            ["materialization_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index(
        "ix_render_executions_materialization_id",
        "render_executions",
        ["materialization_id"],
    )


def downgrade() -> None:
    # Remove Stage 6 rows/jobs ourselves so the narrowed enum check constraints
    # can be recreated on populated databases, preserving every Stage 1-5.2 row.
    op.execute("DELETE FROM clip_materializations")
    op.execute("DELETE FROM processing_jobs WHERE kind = 'MATERIALIZATION'")
    op.execute("DELETE FROM render_executions WHERE artifact_purpose = 'TRANSFORMED_SHORT'")
    op.execute("DELETE FROM channel_production_profiles")

    op.drop_index("ix_render_executions_materialization_id", table_name="render_executions")
    with op.batch_alter_table("render_executions") as batch:
        batch.drop_constraint("fk_render_executions_materialization_id", type_="foreignkey")
        batch.drop_column("materialization_id")
    op.drop_index("ix_processing_jobs_materialization_id", table_name="processing_jobs")
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_constraint("fk_processing_jobs_materialization_id", type_="foreignkey")
        batch.drop_column("materialization_id")

    index_names = [
        "uq_clip_materializations_current_scope",
        *[
            f"ix_clip_materializations_{column}"
            for column in reversed(
                [
                    "source_video_id",
                    "clip_candidate_id",
                    "channel_production_profile_id",
                    "selection_id",
                    "selected_plan_id",
                    "refinement_id",
                    "render_contract_id",
                    "visual_composition_plan_id",
                    "render_execution_id",
                    "active_job_id",
                    "priority",
                    "lifecycle",
                    "semantic_outcome",
                    "is_current",
                ]
            )
        ],
    ]
    for index_name in index_names:
        op.drop_index(index_name, table_name="clip_materializations")
    op.drop_table("clip_materializations")

    for index_name in (
        "uq_channel_production_profiles_current_key",
        "ix_channel_production_profiles_is_current",
        "ix_channel_production_profiles_channel_key",
    ):
        op.drop_index(index_name, table_name="channel_production_profiles")
    op.drop_table("channel_production_profiles")

    with op.batch_alter_table("render_executions") as batch:
        batch.alter_column(
            "artifact_purpose",
            existing_type=_enum("render_artifact_purpose", _ARTIFACT_CURRENT),
            type_=_enum("render_artifact_purpose", _ARTIFACT_PREVIOUS),
            existing_nullable=False,
        )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.alter_column(
            "kind",
            existing_type=_enum("job_kind", _JOB_CURRENT),
            type_=_enum("job_kind", _JOB_PREVIOUS),
            existing_nullable=False,
        )
