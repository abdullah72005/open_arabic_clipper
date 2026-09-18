"""Local operator commands using the same storage and health services as HTTP."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import typer

from app.candidates.novelty import NoveltyItem
from app.candidates.service import CandidateAnalysisService
from app.core.enums import (
    CandidateDisposition,
    JobKind,
    PipelineStage,
    RefinementPriority,
    SemanticProviderMode,
)
from app.core.settings import get_settings
from app.db.session import create_session_factory
from app.models import (
    AudioAnalysis,
    AudioArtifact,
    CandidateAnalysis,
    ClipCandidate,
    ProcessingJob,
    SourceVideo,
    Transcript,
)
from app.refinement.handoff import build_stage4_handoff
from app.refinement.queue import (
    Stage35QueueError,
    list_refinements,
    queue_candidate_batch,
    queue_candidate_refinement,
    validate_candidate_for_refinement,
)
from app.runtime.heavy_model_lease import HeavyModelLeaseBusy, HeavyModelUnsafe
from app.runtime.memory import MemoryReadError, capture_memory
from app.services.health import HealthService
from app.services.storage import StorageCategory, StorageService
from app.transcription.benchmark import benchmark_transcription, transcribe_for_benchmark
from app.transcription.engine import WhisperEngine
from app.transcription.performance_replay import (
    CandidateSnapshot,
    replay_index_candidates,
)
from app.transcription.reconstruction.benchmark import (
    BenchmarkRunner,
    evaluate_completion_gate,
    load_benchmark_manifest,
    prompt_settings_fingerprint,
)
from app.transcription.reconstruction.capture import capture_hash, load_capture, save_capture
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import ProviderAvailability, ProviderHealth
from app.transformation.governance.handoff import build_stage4_3_handoff
from app.transformation.governance.queue import (
    GovernanceQueueError,
    get_governance_set_for_candidate,
    list_results,
    queue_transformation_governance,
    validate_candidate_for_governance,
)
from app.transformation.handoff import build_stage4_1_handoff
from app.transformation.planning.handoff import build_stage4_2_handoff
from app.transformation.planning.queue import (
    PlanningQueueError,
    get_plan_set_for_candidate,
    list_plans,
    queue_transformation_planning,
    validate_candidate_for_planning,
)
from app.transformation.queue import (
    TransformationQueueError,
    get_analysis_for_candidate,
    list_strategies,
    queue_transformation_analysis,
    validate_candidate_for_transformation,
)
from app.workers.tasks import run_pipeline_stage

app = typer.Typer(no_args_is_help=True)
_KNOWN_REGRESSION_MANIFEST_NAME = "stage-2-7/known-regression-v1.json"


def _storage() -> StorageService:
    return StorageService(get_settings().storage_root)


@app.command("diagnose-memory")
def diagnose_memory(
    json_output: bool = typer.Option(False, "--json/--text", help="Emit machine-readable JSON."),
) -> None:
    """Print labeled host/container memory facts. Read-only."""

    try:
        snapshot = capture_memory()
    except MemoryReadError as error:
        typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False))
        raise typer.Exit(code=1) from error
    if json_output:
        payload = asdict(snapshot)
        payload["effective_capacity"] = snapshot.effective_capacity
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    typer.echo(
        f"linux_total={snapshot.linux_total / 1024**3:.2f} GiB "
        f"linux_available={snapshot.linux_available / 1024**3:.2f} GiB"
    )
    typer.echo(
        f"swap_total={snapshot.swap_total / 1024**3:.2f} GiB "
        f"swap_free={snapshot.swap_free / 1024**3:.2f} GiB"
    )
    limit = (
        f"{snapshot.cgroup_limit / 1024**3:.2f} GiB"
        if snapshot.cgroup_limit is not None
        else "unlimited"
    )
    typer.echo(f"cgroup_limit={limit} cgroup_peak={snapshot.cgroup_peak}")
    typer.echo(
        f"process_rss={snapshot.process_rss / 1024**3:.3f} GiB "
        f"effective_capacity={snapshot.effective_capacity / 1024**3:.2f} GiB"
    )


@app.command()
def health() -> None:
    report = HealthService(_storage()).report()
    typer.echo(report.status.value)


@app.command("reconstruction-health")
def reconstruction_health() -> None:
    """Verify configured reconstruction endpoint and exact model availability."""

    settings = get_settings()
    provider = settings.reconstruction_provider_instance()
    report = (
        provider.health()
        if provider is not None
        else ProviderHealth(
            ProviderAvailability.MISCONFIGURED,
            settings.reconstruction_provider,
            settings.reconstruction_provider_model,
            None,
            "local Qwen reconstruction is disabled by default; set "
            "CLIPFACTORY_LOCAL_QWEN_ENABLED=true to enable local providers",
        )
    )
    typer.echo(
        json.dumps(
            {
                "availability": report.availability.value,
                "provider": report.provider,
                "model": report.model,
                "digest": report.model_digest,
                "detail": report.detail,
            }
        )
    )
    if report.availability is not ProviderAvailability.AVAILABLE:
        raise typer.Exit(1)


@app.command("recover-heavy-model")
def recover_heavy_model() -> None:
    """Clear unsafe heavy-model state only after the model is no longer resident."""

    settings = get_settings()
    factory = settings.heavy_model_lease_factory()
    if not factory.unsafe_recorded():
        typer.echo(json.dumps({"status": "CLEAR", "detail": "no unsafe heavy-model state"}))
        return
    reason = getattr(factory, "unsafe_reason", lambda: "")() or ""
    owner = re.search(r'"owner_pid"\s*:\s*(\d+)', reason)
    if "lease retained for benchmark-asr" in reason and owner is not None:
        try:
            os.kill(int(owner.group(1)), 0)
        except ProcessLookupError:
            factory.recover()
            typer.echo(json.dumps({"status": "RECOVERED", "detail": "benchmark child is absent"}))
            return
        except PermissionError:
            pass
    provider = settings.reconstruction_provider_instance()
    resident = provider.is_model_resident() if provider is not None else True
    if resident:
        typer.echo(
            json.dumps(
                {
                    "status": "UNSAFE",
                    "detail": factory.unsafe_reason(),
                    "model": settings.reconstruction_provider_model,
                }
            )
        )
        raise typer.Exit(1)
    factory.recover()
    typer.echo(json.dumps({"status": "RECOVERED", "detail": "model confirmed not resident"}))


@app.command()
def add(path: Path) -> None:
    if not path.is_file():
        raise typer.BadParameter("path must be a readable file")
    typer.echo(str(path.resolve()))


@app.command()
def status(source_id: UUID) -> None:
    typer.echo(str(source_id))


@app.command()
def retry(source_id: UUID) -> None:
    typer.echo(str(source_id))


@app.command()
def cleanup(older_than_seconds: int = 3600, limit: int = 100) -> None:
    removed = _storage().cleanup_temporary_files(
        older_than_seconds=older_than_seconds,
        limit=limit,
    )
    typer.echo(str(removed))


def _queue_transcription(source_id: UUID, *, force: bool) -> UUID:
    with create_session_factory()() as session:
        if session.get(SourceVideo, source_id) is None:
            raise typer.BadParameter("source does not exist")
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.TRANSCRIPTION)
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id
    run_pipeline_stage.delay(str(source_id), PipelineStage.TRANSCRIPTION.value, str(job_id), force)
    return job_id


def _queue_reconstruction(source_id: UUID, *, force: bool) -> UUID:
    with create_session_factory()() as session:
        if session.get(SourceVideo, source_id) is None:
            raise typer.BadParameter("source does not exist")
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.RECONSTRUCTION)
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id
    run_pipeline_stage.delay(
        str(source_id), PipelineStage.CONTEXTUAL_RECONSTRUCTION.value, str(job_id), force
    )
    return job_id


@app.command()
def transcribe(source_id: UUID) -> None:
    """Queue local transcription, reusing a valid fingerprinted transcript."""
    typer.echo(str(_queue_transcription(source_id, force=False)))


@app.command()
def retranscribe(source_id: UUID, force: bool = typer.Option(True, "--force/--no-force")) -> None:
    """Queue transcription and, by default, bypass the transcript cache."""
    typer.echo(str(_queue_transcription(source_id, force=force)))


@app.command()
def reconstruct(source_id: UUID, force: bool = typer.Option(False, "--force/--no-force")) -> None:
    """Queue bounded contextual reconstruction, reusing its current fingerprint by default."""
    typer.echo(str(_queue_reconstruction(source_id, force=force)))


def _queue_candidate_analysis(source_id: UUID, *, force: bool) -> UUID:
    with create_session_factory()() as session:
        if session.get(SourceVideo, source_id) is None:
            raise typer.BadParameter("source does not exist")
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.CANDIDATE_ANALYSIS)
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id
    run_pipeline_stage.delay(
        str(source_id), PipelineStage.CANDIDATE_ANALYSIS.value, str(job_id), force
    )
    return job_id


@app.command("candidate-analysis")
def candidate_analysis(
    source_id: UUID, force: bool = typer.Option(False, "--force/--no-force")
) -> None:
    """Queue bounded Stage 3 candidate analysis for a source."""

    typer.echo(str(_queue_candidate_analysis(source_id, force=force)))


@app.command("candidates")
def candidates(
    source_id: UUID,
    limit: int = typer.Option(20, min=1, max=200),
    include_rejected: bool = typer.Option(False, "--include-rejected/--accepted-only"),
) -> None:
    """Print bounded current candidate summaries for local inspection."""

    with create_session_factory()() as session:
        statement = (
            session.query(ClipCandidate)
            .filter(ClipCandidate.source_video_id == source_id)
            .filter(ClipCandidate.is_current.is_(True))
            .order_by(ClipCandidate.clip_score.desc())
        )
        if not include_rejected:
            statement = statement.filter(
                ClipCandidate.disposition.in_(
                    [
                        CandidateDisposition.CANDIDATE,
                        CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
                    ]
                )
            )
        rows = statement.limit(limit).all()
        typer.echo(
            json.dumps(
                [
                    {
                        "id": str(row.id),
                        "candidate_key": row.candidate_key,
                        "disposition": row.disposition.value,
                        "start_time": row.start_time,
                        "end_time": row.end_time,
                        "segment_indexes": [
                            row.start_segment_index,
                            row.end_segment_index,
                        ],
                        "clip_score": row.clip_score,
                        "transcript_confidence": row.transcript_confidence,
                        "primary_content_type": row.primary_content_type.value,
                        "refinement_reasons": row.refinement_reasons,
                        "rights_risk": row.rights_risk.value,
                        "originality_risk": row.originality_risk.value,
                        "excerpt": row.transcript_excerpt[:200],
                    }
                    for row in rows
                ],
                ensure_ascii=False,
            )
        )


@app.command("candidate-refine")
def candidate_refine(
    candidate_id: UUID,
    priority: str = typer.Option("CANDIDATE", "--priority"),
    force: bool = typer.Option(False, "--force/--no-force"),
) -> None:
    """Queue one explicit candidate-scoped Stage 3.5 refinement."""

    try:
        parsed = RefinementPriority(priority.upper())
    except ValueError as error:
        raise typer.BadParameter("priority must be CANDIDATE or FINAL_CLIP") from error
    with create_session_factory()() as session:
        try:
            candidate = validate_candidate_for_refinement(session, candidate_id)
        except Stage35QueueError as error:
            raise typer.BadParameter(str(error)) from error
        outcome = queue_candidate_refinement(session, _storage(), candidate, parsed, force=force)
    typer.echo(
        json.dumps(
            {
                "refinement_id": str(outcome.refinement_id),
                "job_id": str(outcome.job_id) if outcome.job_id else None,
                "status": outcome.status,
                "queued": outcome.queued,
                "cached": outcome.cached,
                "active": outcome.active,
            }
        )
    )


@app.command("candidate-refine-batch")
def candidate_refine_batch(
    source_id: UUID,
    limit: int = typer.Option(5, min=1, max=10),
    force: bool = typer.Option(False, "--force/--no-force"),
) -> None:
    """Queue a bounded, score-ordered candidate-grade refinement batch."""

    with create_session_factory()() as session:
        try:
            outcomes = queue_candidate_batch(
                session, _storage(), source_id, limit=limit, force=force
            )
        except Stage35QueueError as error:
            raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            [
                {
                    "refinement_id": str(outcome.refinement_id),
                    "job_id": str(outcome.job_id) if outcome.job_id else None,
                    "queued": outcome.queued,
                    "cached": outcome.cached,
                    "active": outcome.active,
                }
                for outcome in outcomes
            ]
        )
    )


@app.command("candidate-refinements")
def candidate_refinements(candidate_id: UUID) -> None:
    """Print both quality-level refinements for a candidate."""

    with create_session_factory()() as session:
        rows = list_refinements(session, candidate_id)
        typer.echo(
            json.dumps(
                [
                    {
                        "id": str(row.id),
                        "priority": row.priority.value,
                        "status": row.status.value,
                        "quality_level": row.quality_level,
                        "confidence": row.confidence,
                        "refined_start": row.refined_start,
                        "refined_end": row.refined_end,
                        "final_transcript": row.final_transcript[:500],
                        "cache_eligible": row.cache_eligible,
                    }
                    for row in rows
                ],
                ensure_ascii=False,
            )
        )


@app.command("candidate-handoff")
def candidate_handoff(candidate_id: UUID) -> None:
    """Print the typed read-only Stage 4 handoff for a candidate."""

    with create_session_factory()() as session:
        handoff = build_stage4_handoff(session, candidate_id)
        if handoff is None:
            raise typer.BadParameter("candidate does not exist")
    typer.echo(json.dumps(handoff, ensure_ascii=False, default=str))


@app.command("transformation-analyze")
def transformation_analyze(
    candidate_id: UUID,
    force: bool = typer.Option(False, "--force/--no-force"),
) -> None:
    """Queue one explicit candidate-scoped Stage 4.0 eligibility analysis."""

    with create_session_factory()() as session:
        try:
            candidate = validate_candidate_for_transformation(session, candidate_id)
        except TransformationQueueError as error:
            raise typer.BadParameter(str(error)) from error
        outcome = queue_transformation_analysis(session, candidate, force=force)
    typer.echo(
        json.dumps(
            {
                "analysis_id": str(outcome.analysis_id),
                "job_id": str(outcome.job_id) if outcome.job_id else None,
                "status": outcome.status,
                "queued": outcome.queued,
                "cached": outcome.cached,
                "active": outcome.active,
            }
        )
    )


def _transformation_strategy_payload(row: object) -> dict[str, object]:
    return {
        "id": str(row.id),  # type: ignore[attr-defined]
        "strategy_type": row.strategy_type.value,  # type: ignore[attr-defined]
        "disposition": row.disposition.value,  # type: ignore[attr-defined]
        "is_current": row.is_current,  # type: ignore[attr-defined]
        "rank": row.rank,  # type: ignore[attr-defined]
        "intensity": row.intensity.value,  # type: ignore[attr-defined]
        "direction_summary": row.direction_summary,  # type: ignore[attr-defined]
        "added_value_focus": row.added_value_focus,  # type: ignore[attr-defined]
        "substantive_value_kind": row.substantive_value_kind.value,  # type: ignore[attr-defined]
        "external_verification_requirement": row.external_verification_requirement.value,  # type: ignore[attr-defined]
        "verification_requirements": list(row.verification_requirements or []),  # type: ignore[attr-defined]
        "rejection_reasons": list(row.rejection_reasons or []),  # type: ignore[attr-defined]
        "strategy_fingerprint": row.strategy_fingerprint,  # type: ignore[attr-defined]
    }


@app.command("transformation-analysis")
def transformation_analysis(candidate_id: UUID) -> None:
    """Print the current Stage 4.0 eligibility analysis and strategies."""

    with create_session_factory()() as session:
        analysis = get_analysis_for_candidate(session, candidate_id)
        if analysis is None:
            raise typer.BadParameter("transformation analysis does not exist")
        rows = list_strategies(session, analysis.id)
        outcome = analysis.eligibility_outcome
        payload = {
            "id": str(analysis.id),
            "execution_status": analysis.execution_status.value,
            "eligibility_outcome": outcome.value if outcome else None,
            "eligibility_reasons": list(analysis.eligibility_reasons or []),
            "assessments": analysis.assessments,
            "source_moment": analysis.source_moment,
            "platform_risk": analysis.platform_risk,
            "transformation_intensity": (
                analysis.transformation_intensity.value
                if analysis.transformation_intensity
                else None
            ),
            "provider_mode": analysis.provider_mode.value,
            "provider_status": analysis.provider_status,
            "input_fingerprint": analysis.input_fingerprint,
            "output_fingerprint": analysis.output_fingerprint,
            "cache_eligible": analysis.cache_eligible,
            "strategies": [_transformation_strategy_payload(row) for row in rows],
        }
    typer.echo(json.dumps(payload, ensure_ascii=False, default=str))


@app.command("transformation-handoff")
def transformation_handoff(candidate_id: UUID) -> None:
    """Print the read-only Stage 4.0 -> Stage 4.1 handoff."""

    with create_session_factory()() as session:
        handoff = build_stage4_1_handoff(session, candidate_id)
        if handoff is None:
            raise typer.BadParameter("candidate does not exist")
    typer.echo(json.dumps(handoff, ensure_ascii=False, default=str))


@app.command("transformation-plan-generate")
def transformation_plan_generate(
    candidate_id: UUID,
    force: bool = typer.Option(False, "--force/--no-force"),
) -> None:
    """Queue one explicit candidate-scoped Stage 4.1 planning run."""

    with create_session_factory()() as session:
        try:
            candidate, analysis = validate_candidate_for_planning(session, candidate_id)
        except PlanningQueueError as error:
            raise typer.BadParameter(str(error)) from error
        outcome = queue_transformation_planning(session, candidate, analysis, force=force)
    typer.echo(
        json.dumps(
            {
                "plan_set_id": str(outcome.plan_set_id),
                "job_id": str(outcome.job_id) if outcome.job_id else None,
                "status": outcome.status,
                "queued": outcome.queued,
                "cached": outcome.cached,
                "active": outcome.active,
            }
        )
    )


def _plan_payload(row: object) -> dict[str, object]:
    return {
        "id": str(row.id),  # type: ignore[attr-defined]
        "plan_key": row.plan_key,  # type: ignore[attr-defined]
        "is_current": row.is_current,  # type: ignore[attr-defined]
        "status": row.status.value,  # type: ignore[attr-defined]
        "generation_rank": row.generation_rank,  # type: ignore[attr-defined]
        "strategy_candidate_id": str(row.strategy_candidate_id),  # type: ignore[attr-defined]
        "strategy_type": row.strategy_type.value,  # type: ignore[attr-defined]
        "intensity": row.intensity.value,  # type: ignore[attr-defined]
        "blocks": list(row.blocks or []),  # type: ignore[attr-defined]
        "hero_block_index": row.hero_block_index,  # type: ignore[attr-defined]
        "hero_source_start": row.hero_source_start,  # type: ignore[attr-defined]
        "hero_source_end": row.hero_source_end,  # type: ignore[attr-defined]
        "hero_appearance_time": row.hero_appearance_time,  # type: ignore[attr-defined]
        "original_value_kinds": list(row.original_value_kinds or []),  # type: ignore[attr-defined]
        "narration_need": row.narration_need,  # type: ignore[attr-defined]
        "narration_requirements": row.narration_requirements,  # type: ignore[attr-defined]
        "external_fact_dependencies": list(row.external_fact_dependencies or []),  # type: ignore[attr-defined]
        "derived_durations": row.derived_durations,  # type: ignore[attr-defined]
        "provider_input_fingerprint": row.provider_input_fingerprint,  # type: ignore[attr-defined]
        "plan_output_fingerprint": row.plan_output_fingerprint,  # type: ignore[attr-defined]
    }


@app.command("transformation-plans")
def transformation_plans(candidate_id: UUID) -> None:
    """Print the current Stage 4.1 plan set and its plans."""

    with create_session_factory()() as session:
        plan_set = get_plan_set_for_candidate(session, candidate_id)
        if plan_set is None:
            raise typer.BadParameter("transformation plan set does not exist")
        rows = list_plans(session, plan_set.id)
        outcome = plan_set.planning_outcome
        payload = {
            "id": str(plan_set.id),
            "execution_status": plan_set.execution_status.value,
            "planning_outcome": outcome.value if outcome else None,
            "outcome_reasons": list(plan_set.outcome_reasons or []),
            "provider_mode": plan_set.provider_mode.value,
            "provider_status": plan_set.provider_status,
            "strategy_attempts": list(plan_set.strategy_attempts or []),
            "input_fingerprint": plan_set.input_fingerprint,
            "output_fingerprint": plan_set.output_fingerprint,
            "cache_eligible": plan_set.cache_eligible,
            "plans": [_plan_payload(row) for row in rows],
        }
    typer.echo(json.dumps(payload, ensure_ascii=False, default=str))


@app.command("transformation-plan-handoff")
def transformation_plan_handoff(candidate_id: UUID) -> None:
    """Print the read-only Stage 4.1 -> Stage 4.2 handoff."""

    with create_session_factory()() as session:
        handoff = build_stage4_2_handoff(session, candidate_id)
        if handoff is None:
            raise typer.BadParameter("candidate does not exist")
    typer.echo(json.dumps(handoff, ensure_ascii=False, default=str))


@app.command("transformation-govern")
def transformation_govern(
    candidate_id: UUID,
    force: bool = typer.Option(False, "--force/--no-force"),
) -> None:
    """Queue one explicit candidate-scoped Stage 4.2 governance run."""

    with create_session_factory()() as session:
        try:
            candidate, plan_set = validate_candidate_for_governance(session, candidate_id)
        except GovernanceQueueError as error:
            raise typer.BadParameter(str(error)) from error
        outcome = queue_transformation_governance(session, candidate, plan_set, force=force)
    typer.echo(
        json.dumps(
            {
                "governance_set_id": str(outcome.governance_set_id),
                "job_id": str(outcome.job_id) if outcome.job_id else None,
                "status": outcome.status,
                "queued": outcome.queued,
                "cached": outcome.cached,
                "active": outcome.active,
            }
        )
    )


def _governance_result_payload(row: object) -> dict[str, object]:
    return {
        "transformation_plan_id": str(row.transformation_plan_id),  # type: ignore[attr-defined]
        "plan_output_fingerprint": row.plan_output_fingerprint,  # type: ignore[attr-defined]
        "status": row.status.value,  # type: ignore[attr-defined]
        "eligible_for_stage4_3": row.eligible_for_stage4_3,  # type: ignore[attr-defined]
        "severity": row.severity,  # type: ignore[attr-defined]
        "hard_gates": list(row.hard_gates or []),  # type: ignore[attr-defined]
        "dimensions": dict(row.dimensions or {}),  # type: ignore[attr-defined]
        "verification": dict(row.verification or {}),  # type: ignore[attr-defined]
        "platform_risk": dict(row.platform_risk or {}),  # type: ignore[attr-defined]
        "reason_codes": list(row.reason_codes or []),  # type: ignore[attr-defined]
        "warnings": list(row.warnings or []),  # type: ignore[attr-defined]
        "remediation": list(row.remediation or []),  # type: ignore[attr-defined]
        "output_fingerprint": row.output_fingerprint,  # type: ignore[attr-defined]
    }


@app.command("transformation-governance")
def transformation_governance(candidate_id: UUID) -> None:
    """Print the current Stage 4.2 governance set and its per-plan results."""

    with create_session_factory()() as session:
        governance_set = get_governance_set_for_candidate(session, candidate_id)
        if governance_set is None:
            raise typer.BadParameter("transformation governance set does not exist")
        rows = list_results(session, governance_set.id)
        outcome = governance_set.governance_outcome
        payload = {
            "id": str(governance_set.id),
            "execution_status": governance_set.execution_status.value,
            "semantic_outcome": outcome.value if outcome else None,
            "outcome_reasons": list(governance_set.outcome_reasons or []),
            "summary_counts": dict(governance_set.summary_counts or {}),
            "provider_mode": governance_set.provider_mode.value,
            "provider_status": governance_set.provider_status,
            "plan_attempts": list(governance_set.plan_attempts or []),
            "platform_policy_profile_version": governance_set.platform_policy_profile_version,
            "platform_policy_checked_at": governance_set.platform_policy_checked_at,
            "input_fingerprint": governance_set.input_fingerprint,
            "output_fingerprint": governance_set.output_fingerprint,
            "cache_eligible": governance_set.cache_eligible,
            "results": [_governance_result_payload(row) for row in rows],
        }
    typer.echo(json.dumps(payload, ensure_ascii=False, default=str))


@app.command("transformation-governance-handoff")
def transformation_governance_handoff(candidate_id: UUID) -> None:
    """Print the read-only Stage 4.2 -> Stage 4.3 handoff (no winner)."""

    with create_session_factory()() as session:
        handoff = build_stage4_3_handoff(session, candidate_id)
        if handoff is None:
            raise typer.BadParameter("candidate does not exist")
    typer.echo(json.dumps(handoff, ensure_ascii=False, default=str))


@app.command()
def transcript(source_id: UUID) -> None:
    """Print the current timestamped transcript as JSON."""

    with create_session_factory()() as session:
        current = session.query(Transcript).filter_by(source_video_id=source_id).one_or_none()
        if current is None:
            raise typer.BadParameter("transcript is not ready")
        typer.echo(
            json.dumps(
                {
                    "language": current.language,
                    "raw_text": current.raw_text,
                    "normalized_text": current.normalized_text,
                    "segments": current.segments,
                    "word_segments": current.word_segments,
                    "reconstruction_status": current.reconstruction_status.value,
                },
                ensure_ascii=False,
            )
        )


@app.command()
def benchmark(audio_path: Path) -> None:
    """Measure local configured faster-whisper throughput on representative audio."""
    if not audio_path.is_file():
        raise typer.BadParameter("audio_path must be a readable local file")
    settings = get_settings()
    report = benchmark_transcription(audio_path, WhisperEngine(), settings.transcription_options())
    typer.echo(json.dumps(report.as_dict()))


@app.command("benchmark-index-replay")
def benchmark_index_replay(source_id: UUID) -> None:
    """Run INDEX ASR once and replay Stage 2.5/Stage 3 without durable writes."""

    settings = get_settings()
    with create_session_factory()() as session:
        source = session.get(SourceVideo, source_id)
        transcript = (
            session.query(Transcript).filter(Transcript.source_video_id == source_id).one_or_none()
        )
        artifact = (
            session.query(AudioArtifact)
            .filter(AudioArtifact.source_video_id == source_id)
            .one_or_none()
        )
        analysis = (
            session.query(AudioAnalysis)
            .filter(AudioAnalysis.source_video_id == source_id)
            .one_or_none()
        )
        candidate_analysis = (
            session.query(CandidateAnalysis)
            .filter(CandidateAnalysis.source_video_id == source_id)
            .one_or_none()
        )
        if source is None or transcript is None or artifact is None or analysis is None:
            raise typer.BadParameter("source requires transcript, cached audio, and audio analysis")
        if (
            candidate_analysis is None
            or candidate_analysis.semantic_provider_mode is not SemanticProviderMode.DETERMINISTIC
        ):
            raise typer.BadParameter("source requires deterministic Stage 3 analysis for replay")
        baseline = [
            CandidateSnapshot(
                row.candidate_key,
                row.disposition,
                tuple(row.refinement_reasons or ()),
                row.clip_score,
                row.disposition is CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
                row.refinement_evidence or {},
            )
            for row in session.query(ClipCandidate)
            .filter(ClipCandidate.source_video_id == source_id)
            .filter(ClipCandidate.is_current.is_(True))
            .all()
        ]
        audio_path = _storage().resolve(StorageCategory.SOURCES, artifact.output_path)
        rights_status = source.rights_status
        media_origin = source.media_origin
        provenance_metadata = source.provenance_metadata or {}
        dialect_override = source.dialect_profile_override
        silence_intervals = analysis.silence_intervals
        audio_features = analysis.features
        transcription_fingerprint = transcript.input_fingerprint
        correction_version = transcript.correction_version
        historical_corpus = [
            NoveltyItem(
                key=row.candidate_key,
                source_id=str(row.source_video_id),
                idea_text=row.idea_summary or row.transcript_excerpt,
                topic_text=row.topic_summary or row.transcript_excerpt,
                clip_score=row.clip_score,
            )
            for row in session.query(ClipCandidate)
            .filter(
                ClipCandidate.is_current.is_(True),
                ClipCandidate.source_video_id != source_id,
                ClipCandidate.disposition.in_(
                    [
                        CandidateDisposition.CANDIDATE,
                        CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
                    ]
                ),
            )
            .order_by(ClipCandidate.updated_at.desc())
            .limit(settings.stage3_config().novelty_corpus_limit)
            .all()
        ]

    engine = WhisperEngine()
    cancel_event = threading.Event()
    try:
        with settings.heavy_model_lease_factory().acquire(
            purpose="benchmark-asr", on_ownership_lost=cancel_event.set
        ) as lease:
            report, result = transcribe_for_benchmark(
                audio_path, engine, settings.transcription_options(), cancel_event
            )
            if lease.ownership_lost:
                raise HeavyModelLeaseBusy("heavy-model lease was lost during benchmark ASR")
    except (HeavyModelLeaseBusy, HeavyModelUnsafe) as error:
        raise typer.BadParameter(str(error)) from error
    replay = replay_index_candidates(
        source_id=str(source_id),
        result=result,
        duration=result.duration,
        silence_intervals=silence_intervals,
        audio_features=audio_features,
        rights_status=rights_status,
        media_origin=media_origin,
        provenance_metadata=provenance_metadata,
        dialect_override=dialect_override,
        baseline_candidates=baseline,
        service=CandidateAnalysisService(config=settings.stage3_config()),
        corrector=settings.contextual_corrector(),
        historical_corpus=historical_corpus,
        reconstructor=settings.contextual_reconstructor(),
        transcription_fingerprint=transcription_fingerprint,
        correction_version=correction_version,
    )
    typer.echo(
        json.dumps(
            {
                "benchmark": report.as_dict(),
                "child_peak_rss_bytes": engine.last_child_peak_rss(),
                "replay": replay.as_dict(),
            },
            ensure_ascii=False,
        )
    )


@app.command("benchmark-reconstruction")
def benchmark_reconstruction(
    manifest_name: str,
    model: str | None = typer.Option(None, "--model"),
    allow_known_regression_set: bool = typer.Option(False, "--allow-known-regression-set"),
    capture_asr: bool = typer.Option(False, "--capture-asr", help="Capture immutable ASR only."),
    from_capture: str | None = typer.Option(
        None, "--from-capture", help="Replay a stored capture without a transcriber."
    ),
) -> None:
    """Run a private, authorized reconstruction benchmark through production stages."""

    settings = get_settings()
    storage = _storage()
    if allow_known_regression_set and manifest_name != _KNOWN_REGRESSION_MANIFEST_NAME:
        raise typer.BadParameter(
            "--allow-known-regression-set is reserved for the Chernobyl diagnostic manifest"
        )
    manifest_path = storage.resolve(StorageCategory.BENCHMARKS, manifest_name)
    manifest = load_benchmark_manifest(
        manifest_path,
        allow_known_regression_set=allow_known_regression_set,
        known_regression_manifest_path=storage.resolve(
            StorageCategory.BENCHMARKS, _KNOWN_REGRESSION_MANIFEST_NAME
        ),
    )
    provider = settings.reconstruction_provider_instance(model=model)
    health = provider.health() if provider is not None else None
    whisper_options = asdict(settings.transcription_options())
    fingerprint = prompt_settings_fingerprint(
        provider=settings.reconstruction_provider,
        model=health.model if health is not None else None,
        digest=health.model_digest if health is not None else None,
        whisper_options=whisper_options,
    )
    runner = BenchmarkRunner(
        storage=storage,
        whisper_engine=WhisperEngine() if from_capture is None else None,
        corrector=settings.contextual_corrector(),
        reconstructor=ContextualReconstructor(provider),
        transcription_options=settings.transcription_options(),
        provider_health=health,
        prompt_settings_fingerprint=fingerprint,
    )
    lease_factory = settings.heavy_model_lease_factory()
    if capture_asr:
        try:
            with lease_factory.acquire(purpose="benchmark-asr") as _heavy_lease:
                capture = runner.capture_asr(manifest)
        except HeavyModelLeaseBusy as error:
            typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False))
            raise typer.Exit(code=1) from error
        except HeavyModelUnsafe as error:
            typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False))
            raise typer.Exit(code=1) from error
        path = save_capture(storage, capture, name=capture.capture_id)
        typer.echo(
            json.dumps(
                {
                    "capture_id": capture.capture_id,
                    "capture_hash": capture_hash(capture),
                    "path": str(path),
                    "status": "CAPTURED",
                },
                ensure_ascii=False,
            )
        )
        return
    try:
        with lease_factory.acquire(purpose="benchmark") as _heavy_lease:
            report = runner.run(
                manifest, capture=load_capture(storage, from_capture) if from_capture else None
            )
    except HeavyModelLeaseBusy as error:
        typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False))
        raise typer.Exit(code=1) from error
    except HeavyModelUnsafe as error:
        typer.echo(json.dumps({"error": str(error)}, ensure_ascii=False))
        raise typer.Exit(code=1) from error
    passed, reasons = evaluate_completion_gate(
        report, expected_prompt_settings_fingerprint=fingerprint
    )
    typer.echo(
        json.dumps(
            {
                "model": report.model_identifier,
                "model_digest": report.model_digest,
                "provider_available": report.provider_available,
                "human_labels_complete": report.human_labels_complete,
                "model_feasible": report.model_feasible,
                "semantic_correct_stage25": report.semantic_correct_stage25,
                "semantic_correct_stage27": report.semantic_correct_stage27,
                "improved": report.improved,
                "unchanged_correct": report.unchanged_correct,
                "unchanged_wrong": report.unchanged_wrong,
                "regressed": report.regressed,
                "hallucinated": report.hallucinated,
                "changed_wrong": report.changed_wrong,
                "unresolved": report.unresolved,
                "unreviewed": report.unreviewed,
                "exact_improved": report.exact_improved,
                "exact_unchanged_correct": report.exact_unchanged_correct,
                "exact_unchanged_wrong": report.exact_unchanged_wrong,
                "exact_regressed": report.exact_regressed,
                "exact_changed_wrong": report.exact_changed_wrong,
                "reconstruction_wall_seconds": report.reconstruction_wall_seconds,
                "average_serialized_prompt_bytes": report.average_serialized_prompt_bytes,
                "max_serialized_prompt_bytes": report.max_serialized_prompt_bytes,
                "average_estimated_input_tokens": report.average_estimated_input_tokens,
                "max_estimated_input_tokens": report.max_estimated_input_tokens,
                "unload_confirmed": report.unload_confirmed,
                "unload_warning": report.unload_warning,
                "source_audio_seconds": report.source_audio_seconds,
                "wall_clock_seconds": report.wall_clock_seconds,
                "peak_ram_bytes": report.peak_ram_bytes,
                "peak_vram_bytes": report.peak_vram_bytes,
                "comparison_path": str(report.comparison_path) if report.comparison_path else None,
                "report_path": str(report.report_path) if report.report_path else None,
                "worksheet_path": str(report.worksheet_path) if report.worksheet_path else None,
                "passed": passed,
                "reasons": reasons,
                "status": "READY FOR STAGE 3" if passed else "STAGE 2.7 MUST CONTINUE",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    app()
