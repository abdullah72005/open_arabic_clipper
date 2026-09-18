"""FastAPI application factory for asynchronous local media ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.enums import (
    CandidateDisposition,
    ContentType,
    JobKind,
    JobStatus,
    MediaOriginType,
    OriginalityRisk,
    PipelineStage,
    ReconstructionStatus,
    RefinementPriority,
    RightsRisk,
    RightsStatus,
    StrategyDisposition,
)
from app.core.settings import get_settings
from app.db.session import create_session_factory
from app.models import (
    CandidateAnalysis,
    CandidateRefinement,
    ClipCandidate,
    ProcessingJob,
    SourceQualityAssessment,
    SourceVideo,
    Transcript,
    TranscriptChunk,
    TransformationPlanSelection,
)
from app.refinement.handoff import build_stage4_handoff
from app.refinement.queue import (
    Stage35QueueError,
    apply_manual_transcript,
    get_refinement,
    list_refinements,
    queue_candidate_batch,
    queue_candidate_refinement,
    validate_candidate_for_refinement,
)
from app.services.health import CheckStatus, HealthService
from app.services.source_adapters import SourceValidationError, normalize_source_url
from app.services.storage import StorageCategory, StorageService
from app.transcription.chunking import ChunkConfig, build_chunks
from app.transcription.dialect import ArabicDialectProfile
from app.transcription.normalization import normalize_transcript
from app.transcription.reconstruction.providers import ReconstructionProvider
from app.transformation.governance.handoff import build_stage4_3_handoff
from app.transformation.governance.queue import (
    GovernanceQueueError,
    get_governance_set,
    get_governance_set_for_candidate,
    list_results,
    queue_transformation_governance,
    validate_candidate_for_governance,
)
from app.transformation.handoff import build_stage4_1_handoff
from app.transformation.planning.handoff import build_stage4_2_handoff
from app.transformation.planning.queue import (
    PlanningQueueError,
    get_plan_set,
    get_plan_set_for_candidate,
    list_plans,
    queue_transformation_planning,
    validate_candidate_for_planning,
)
from app.transformation.queue import (
    TransformationQueueError,
    get_analysis,
    get_analysis_for_candidate,
    list_strategies,
    queue_transformation_analysis,
    validate_candidate_for_transformation,
)
from app.transformation.selection.handoff import build_execution_handoff
from app.transformation.selection.service import (
    get_current_selection,
    get_selection,
    read_selection,
    select_transformation_plan,
)
from app.workers.tasks import run_pipeline_stage

UPLOAD_CHUNK_BYTES = 1024 * 1024


class Dispatcher(Protocol):
    def dispatch(self, source_id: UUID, job_id: UUID) -> None: ...


class CeleryDispatcher:
    def dispatch(self, source_id: UUID, job_id: UUID) -> None:
        from app.workers.tasks import run_pipeline_stage

        run_pipeline_stage.delay(str(source_id), PipelineStage.INGEST.value, str(job_id))


class SourceURLRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    rights_status: RightsStatus = RightsStatus.UNKNOWN
    dialect_profile_override: ArabicDialectProfile | None = None
    media_origin: MediaOriginType = MediaOriginType.OTHER
    provenance_metadata: dict[str, str] = Field(default_factory=dict)


class SourceResponse(BaseModel):
    id: UUID
    source_uri: str
    original_filename: str | None
    dialect_profile_override: ArabicDialectProfile | None
    rights_status: RightsStatus
    media_origin: MediaOriginType
    provenance_metadata: dict[str, object]
    lifecycle_state: PipelineStage
    created_at: datetime

    model_config = {"from_attributes": True}


class ProvenanceUpdateRequest(BaseModel):
    rights_status: RightsStatus | None = None
    media_origin: MediaOriginType | None = None
    provenance_metadata: dict[str, str] | None = None


class CandidateResponse(BaseModel):
    id: UUID
    source_video_id: UUID
    candidate_key: str
    is_current: bool
    disposition: CandidateDisposition
    start_time: float
    end_time: float
    start_segment_index: int
    end_segment_index: int
    segment_indexes: list[int]
    transcript_excerpt: str
    primary_content_type: ContentType
    secondary_content_types: list[str]
    clip_score: float
    short_form_score: float
    moment_density_score: float
    boredom_risk_score: float
    ending_quality_score: float
    loopability_score: float
    engagement_confidence: float
    transcript_confidence: float
    audio_confidence: float
    boundary_confidence: float
    uncertainty_severity: float
    idea_novelty_score: float
    topic_novelty_score: float
    recent_semantic_similarity_risk: float
    refinement_reasons: list[str]
    refinement_evidence: dict[str, object]
    rights_risk: RightsRisk
    originality_risk: OriginalityRisk
    dialect_profile: str | None
    dialect_confidence: float
    code_switch_suspected: bool
    hooks: list[dict[str, object]]
    idea_summary: str
    topic_summary: str
    provider_evidence: dict[str, object]
    policy_version: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CandidateAnalysisResponse(BaseModel):
    provider_status: str
    semantic_provider_mode: str
    cache_eligible: bool
    metrics: dict[str, object]


class CandidateRefinementResponse(BaseModel):
    id: UUID
    source_video_id: UUID
    clip_candidate_id: UUID
    priority: RefinementPriority
    status: str
    quality_level: str
    coarse_start: float
    coarse_end: float
    context_start: float
    context_end: float
    refined_start: float | None
    refined_end: float | None
    automatic_transcript: str
    manual_transcript: str | None
    final_transcript: str
    word_timestamps: list[dict[str, object]]
    confidence: float
    dialect_profile: str | None
    dialect_confidence: float
    code_switch_evidence: dict[str, object]
    transcript_evidence: list[dict[str, object]]
    entity_evidence: list[dict[str, object]]
    unresolved_spans: list[dict[str, object]]
    provider_evidence: dict[str, object]
    routing_evidence: dict[str, object]
    input_fingerprint: str
    output_fingerprint: str
    cache_eligible: bool
    metrics: dict[str, object]
    processing_duration: float | None
    created_at: datetime
    updated_at: datetime


class RefinementQueueResponse(BaseModel):
    refinement_id: UUID
    job_id: UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool


class ManualTranscriptRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    resolutions: dict[str, str] = Field(default_factory=dict)


class Stage4HandoffResponse(BaseModel):
    candidate: dict[str, object]
    stage3: dict[str, object]
    refinement: dict[str, object]
    stage4_implemented: bool


class TransformationStrategyResponse(BaseModel):
    id: str
    strategy_key: str
    strategy_type: str
    disposition: str
    is_current: bool
    rank: int
    intensity: str
    direction_summary: str
    added_value_focus: str
    substantive_value_kind: str
    source_moment_role: str
    preservation_requirements: list[str]
    retention_preservation: float
    source_moment_damage_risk: float
    added_value_density: float
    originality_potential: float
    source_dominance_risk: float
    generic_filler_risk: float
    redundant_commentary_risk: float
    template_staleness_risk: float
    external_verification_requirement: str
    verification_requirements: list[str]
    rejection_reasons: list[str]
    confidence: float
    origin: str
    strategy_fingerprint: str


class TransformationAnalysisResponse(BaseModel):
    id: UUID
    source_video_id: UUID
    clip_candidate_id: UUID
    refinement_id: UUID | None
    refinement_priority: str
    refinement_quality_level: str
    execution_status: str
    eligibility_outcome: str | None
    eligibility_reasons: list[str]
    assessments: dict[str, object]
    source_moment: dict[str, object]
    platform_risk: dict[str, object]
    transformation_intensity: str | None
    provider_mode: str
    provider_status: str
    provider_evidence: dict[str, object]
    provider_input_fingerprint: str
    input_fingerprint: str
    output_fingerprint: str
    cache_eligible: bool
    metrics: dict[str, object]
    processing_duration: float | None
    recommended_strategies: list[TransformationStrategyResponse]
    rejected_strategies: list[TransformationStrategyResponse]


class TransformationQueueResponse(BaseModel):
    analysis_id: UUID
    job_id: UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool


class Stage41HandoffResponse(BaseModel):
    candidate: dict[str, object]
    stage3: dict[str, object]
    analysis_id: str | None
    output_fingerprint: str | None
    eligibility_outcome: str | None
    eligibility_reasons: list[str]
    platform_risk: dict[str, object]
    recommended_strategies: list[dict[str, object]]
    rejected_strategies: list[dict[str, object]]
    verification_requirements: list[str]
    provider: dict[str, object]
    cache_eligible: bool
    current: bool
    stale: bool
    ready_for_stage4_1: bool
    stage4_1_implemented: bool
    model_config = {"extra": "allow"}


class TransformationPlanResponse(BaseModel):
    id: UUID
    plan_key: str
    is_current: bool
    status: str
    generation_rank: int
    strategy_candidate_id: UUID
    strategy_type: str
    intensity: str
    strategy_fingerprint: str
    blocks: list[dict[str, object]]
    hero_block_index: int
    hero_source_start: float | None
    hero_source_end: float | None
    hero_appearance_time: float
    original_value_kinds: list[str]
    narration_need: str
    narration_requirements: dict[str, object]
    external_fact_dependencies: list[dict[str, object]]
    derived_durations: dict[str, object]
    hook_payoff_evidence: dict[str, object]
    stage40_risk: dict[str, object]
    source_dialect: dict[str, object]
    target_audience: dict[str, object]
    planner_confidence: float
    generation_origin: str
    planning_provider_evidence: dict[str, object]
    provider_input_fingerprint: str
    plan_output_fingerprint: str


class TransformationPlanSetResponse(BaseModel):
    id: UUID
    source_video_id: UUID
    clip_candidate_id: UUID
    transformation_analysis_id: UUID
    refinement_id: UUID | None
    refinement_priority: str
    refinement_quality_level: str
    execution_status: str
    planning_outcome: str | None
    outcome_reasons: list[str]
    stage40_snapshot: dict[str, object]
    target_context: dict[str, object]
    provider_mode: str
    provider_status: str
    provider_identity: dict[str, object]
    provider_evidence: dict[str, object]
    strategy_attempts: list[dict[str, object]]
    input_fingerprint: str
    output_fingerprint: str
    cache_eligible: bool
    metrics: dict[str, object]
    processing_duration: float | None
    plans: list[TransformationPlanResponse]


class PlanningQueueResponse(BaseModel):
    plan_set_id: UUID
    job_id: UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool


class Stage42HandoffResponse(BaseModel):
    candidate: dict[str, object]
    stage40: dict[str, object]
    selected_refinement: dict[str, object] | None
    plan_set: dict[str, object] | None
    plans: list[dict[str, object]]
    current: bool
    stale: bool
    cache_eligible: bool
    stage4_2_implemented: bool
    stage4_3_implemented: bool
    model_config = {"extra": "allow"}


class GovernanceQueueResponse(BaseModel):
    governance_set_id: UUID
    job_id: UUID | None
    status: str
    queued: bool
    cached: bool
    active: bool


class GovernanceResultResponse(BaseModel):
    id: UUID
    transformation_plan_id: UUID
    plan_output_fingerprint: str
    status: str
    eligible_for_stage4_3: bool
    severity: str
    hard_gates: list[dict[str, object]]
    dimensions: dict[str, object]
    verification: dict[str, object]
    platform_risk: dict[str, object]
    reason_codes: list[str]
    warnings: list[dict[str, object]]
    remediation: list[dict[str, object]]
    provider_evidence: dict[str, object]
    input_fingerprint: str
    output_fingerprint: str


class GovernanceSetResponse(BaseModel):
    id: UUID
    source_video_id: UUID
    clip_candidate_id: UUID
    transformation_plan_set_id: UUID
    transformation_analysis_id: UUID
    refinement_id: UUID | None
    refinement_priority: str
    refinement_quality_level: str
    execution_status: str
    semantic_outcome: str | None
    outcome_reasons: list[str]
    summary_counts: dict[str, object]
    stage40_snapshot: dict[str, object]
    target_context: dict[str, object]
    provider_mode: str
    provider_status: str
    provider_identity: dict[str, object]
    provider_evidence: dict[str, object]
    plan_attempts: list[dict[str, object]]
    platform_policy_profile_version: str
    platform_policy_checked_at: str
    input_fingerprint: str
    output_fingerprint: str
    policy_version: str
    schema_version: str
    validation_version: str
    cache_eligible: bool
    metrics: dict[str, object]
    processing_duration: float | None
    results: list[GovernanceResultResponse]


class Stage43HandoffResponse(BaseModel):
    candidate: dict[str, object]
    rights_and_provenance: dict[str, object]
    selected_refinement: dict[str, object] | None
    stage40: dict[str, object]
    stage41: dict[str, object]
    governance_set: dict[str, object] | None
    plans: list[dict[str, object]]
    stage4_3_implemented: bool
    model_config = {"extra": "allow"}


class TransformationSelectionResponse(BaseModel):
    id: UUID
    source_video_id: UUID
    clip_candidate_id: UUID
    transformation_analysis_id: UUID | None
    transformation_plan_set_id: UUID | None
    transformation_governance_set_id: UUID | None
    selected_plan_id: UUID | None
    selected_governance_result_id: UUID | None
    refinement_id: UUID | None
    refinement_priority: str
    refinement_quality_level: str
    refinement_output_fingerprint: str
    status: str
    is_current: bool
    selected_with_caution: bool
    selection_reason_codes: list[str]
    arbitration_evidence: dict[str, object]
    selected_governance_snapshot: dict[str, object]
    alternative_dispositions: list[dict[str, object]]
    governance_input_fingerprint: str
    governance_output_fingerprint: str
    governor_policy_version: str
    governor_validation_version: str
    platform_policy_profile_version: str
    selected_plan_fingerprint: str
    input_fingerprint: str
    output_fingerprint: str
    policy_version: str
    schema_version: str
    fingerprint_version: str
    live_freshness: str
    effective: bool
    created_at: datetime
    updated_at: datetime
    model_config = {"extra": "allow"}


class ExecutionHandoffResponse(BaseModel):
    candidate: dict[str, object]
    selection: dict[str, object] | None
    rights_and_provenance: dict[str, object]
    selected_plan: dict[str, object] | None
    execution_readiness: str
    stage5_implemented: bool
    stage6_tts_implemented: bool
    model_config = {"extra": "allow"}


class JobResponse(BaseModel):
    id: UUID
    source_video_id: UUID
    kind: JobKind
    status: JobStatus
    retry_count: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: CheckStatus
    checks: list[dict[str, str]]


class StorageResponse(BaseModel):
    total_bytes: int
    used_bytes: int
    free_bytes: int


class TranscriptResponse(BaseModel):
    source_video_id: UUID
    language: str | None
    detected_language_probability: float | None
    whisper_model: str
    transcription_options: dict[str, object]
    dialect_profile: ArabicDialectProfile | None
    dialect_confidence: float
    dialect_evidence: dict[str, object]
    code_switch_suspected: bool
    raw_text: str
    normalized_text: str
    corrected_text: str
    final_text: str
    raw_transcript_confidence: float
    correction_confidence: float
    corrected_segment_ratio: float
    uncertain_segment_ratio: float
    correction_method: str
    correction_version: str
    contextual_reconstructed_text: str
    reconstruction_fingerprint: str
    reconstruction_confidence: float
    reconstructed_segment_ratio: float
    reconstruction_method: str
    reconstruction_version: str
    reconstruction_processing_duration: float | None
    reconstruction_metadata: dict[str, object]
    reconstruction_status: ReconstructionStatus
    segments: list[dict[str, object]]
    word_segments: list[dict[str, object]]
    duration: float
    processing_duration: float | None


class QualityMetricsResponse(BaseModel):
    audio_quality_score: float
    transcript_quality_score: float
    low_confidence_word_ratio: float
    unresolved_segment_ratio: float
    manual_review_required: bool
    conservative_source_floor: float


class QualityResponse(BaseModel):
    reconstruction_status: ReconstructionStatus | None
    quality: QualityMetricsResponse | None


class TranscriptSearchResponse(BaseModel):
    segments: list[dict[str, object]]


class TranscriptOverrideRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


def create_app(
    *,
    session_factory: sessionmaker[Session] | None = None,
    storage: StorageService | None = None,
    dispatcher: Dispatcher | None = None,
    health: HealthService | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    settings = get_settings()
    factory = session_factory or create_session_factory()
    storage_service = storage or StorageService(settings.storage_root)
    task_dispatcher = dispatcher or CeleryDispatcher()
    upload_limit = max_upload_bytes or settings.max_upload_bytes
    health_service = health or _default_health(
        storage_service,
        factory,
        settings.ffmpeg_binary,
        settings.ffprobe_binary,
        settings.reconstruction_provider_instance(),
    )
    app = FastAPI(title="ClipFactory API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    def session() -> Iterator[Session]:
        database = factory()
        try:
            yield database
        finally:
            database.close()

    @app.post("/sources/upload", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
    def upload_source(
        response: Response,
        file: UploadFile = File(...),
        rights_status: RightsStatus = Form(RightsStatus.UNKNOWN),
        dialect_profile_override: ArabicDialectProfile | None = Form(None),
        media_origin: MediaOriginType = Form(MediaOriginType.OTHER),
        provenance_metadata: str | None = Form(None),
        database: Session = Depends(session),
    ) -> SourceResponse:
        provenance = _parse_provenance_form(provenance_metadata)
        filename = _safe_filename(file.filename)
        temporary_path = storage_service.resolve(StorageCategory.TEMPORARY, f"upload-{uuid4()}.tmp")
        digest = hashlib.sha256()
        bytes_written = 0

        def chunks() -> Iterator[bytes]:
            nonlocal bytes_written
            while chunk := file.file.read(UPLOAD_CHUNK_BYTES):
                bytes_written += len(chunk)
                if bytes_written > upload_limit:
                    raise HTTPException(
                        status_code=413, detail="upload exceeds configured size limit"
                    )
                storage_service.ensure_capacity(len(chunk))
                digest.update(chunk)
                yield chunk

        try:
            storage_service.atomic_write(temporary_path, chunks())
            if bytes_written == 0:
                raise HTTPException(status_code=422, detail="upload must not be empty")
            existing = database.scalar(
                select(SourceVideo).where(SourceVideo.content_hash == digest.hexdigest())
            )
            if existing is not None:
                response.status_code = status.HTTP_200_OK
                return _duplicate_response(existing)
            source = SourceVideo(
                source_uri="",
                original_filename=filename,
                content_hash=digest.hexdigest(),
                dialect_profile_override=dialect_profile_override,
                rights_status=rights_status,
                media_origin=media_origin,
                provenance_metadata=provenance,
            )
            database.add(source)
            database.flush()
            destination = storage_service.source_directory(source.id) / filename
            os.replace(temporary_path, destination)
            source.source_uri = str(destination)
            job = _new_job(source.id)
            database.add(job)
            database.commit()
            database.refresh(source)
            task_dispatcher.dispatch(source.id, job.id)
            return SourceResponse.model_validate(source)
        finally:
            temporary_path.unlink(missing_ok=True)

    @app.post("/sources/url", response_model=SourceResponse, status_code=status.HTTP_202_ACCEPTED)
    def create_url_source(
        request: SourceURLRequest, response: Response, database: Session = Depends(session)
    ) -> SourceResponse:
        try:
            normalized = normalize_source_url(request.url)
        except SourceValidationError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        existing = database.scalar(select(SourceVideo).where(SourceVideo.source_uri == normalized))
        if existing is not None:
            response.status_code = status.HTTP_200_OK
            return _duplicate_response(existing)
        source = SourceVideo(
            source_uri=normalized,
            dialect_profile_override=request.dialect_profile_override,
            rights_status=request.rights_status,
            media_origin=request.media_origin,
            provenance_metadata=_validate_provenance_metadata(request.provenance_metadata),
        )
        database.add(source)
        database.flush()
        job = _new_job(source.id)
        database.add(job)
        database.commit()
        database.refresh(source)
        task_dispatcher.dispatch(source.id, job.id)
        return SourceResponse.model_validate(source)

    @app.get("/sources", response_model=list[SourceResponse])
    def list_sources(database: Session = Depends(session)) -> list[SourceResponse]:
        return [
            SourceResponse.model_validate(source)
            for source in database.scalars(
                select(SourceVideo).order_by(SourceVideo.created_at.desc())
            )
        ]

    @app.get("/sources/{source_id}", response_model=SourceResponse)
    def get_source(source_id: UUID, database: Session = Depends(session)) -> SourceResponse:
        return SourceResponse.model_validate(_source_or_404(database, source_id))

    @app.get("/api/sources/{source_id}/transcript", response_model=TranscriptResponse)
    def get_transcript(source_id: UUID, database: Session = Depends(session)) -> TranscriptResponse:
        transcript = _transcript_or_404(database, source_id)
        return _transcript_response(transcript)

    @app.get("/api/sources/{source_id}/quality", response_model=QualityResponse)
    def get_source_quality(
        source_id: UUID, database: Session = Depends(session)
    ) -> QualityResponse:
        source = _source_or_404(database, source_id)
        transcript = source.transcript
        assessment = source.quality_assessment
        return QualityResponse(
            reconstruction_status=(transcript.reconstruction_status if transcript else None),
            quality=_quality_metrics_response(assessment) if assessment else None,
        )

    @app.get("/api/sources/{source_id}/media")
    def get_source_media(source_id: UUID, database: Session = Depends(session)) -> FileResponse:
        """Serve only the storage-owned local original for timestamp playback."""
        source = _source_or_404(database, source_id)
        source_path = Path(source.source_uri)
        source_directory = storage_service.source_directory(source.id).resolve()
        try:
            source_path.resolve().relative_to(source_directory)
        except (OSError, ValueError):
            raise HTTPException(
                status_code=404, detail="local source media is unavailable"
            ) from None
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="local source media is unavailable")
        return FileResponse(source_path)

    @app.get(
        "/api/sources/{source_id}/transcript/segments",
        response_model=TranscriptSearchResponse,
    )
    def get_transcript_segments(
        source_id: UUID, offset: int = 0, limit: int = 200, database: Session = Depends(session)
    ) -> TranscriptSearchResponse:
        transcript = _transcript_or_404(database, source_id)
        bounded_offset = max(offset, 0)
        bounded_limit = min(max(limit, 1), 500)
        return TranscriptSearchResponse(
            segments=transcript.segments[bounded_offset : bounded_offset + bounded_limit]
        )

    @app.post("/api/sources/{source_id}/transcript/segments/{segment_index}/override")
    def override_transcript_segment(
        source_id: UUID,
        segment_index: int,
        request: TranscriptOverrideRequest,
        database: Session = Depends(session),
    ) -> dict[str, object]:
        """Persist operator feedback without changing raw or automatic transcript evidence."""

        transcript = _transcript_or_404(database, source_id)
        segments = _copy_segments(transcript)
        segment = _segment_or_404(segments, segment_index)
        segment["operator_text"] = request.text.strip()
        segment["final_text"] = segment["operator_text"]
        _persist_final_segments(database, transcript, segments)
        database.commit()
        database.refresh(transcript)
        return transcript.segments[segment_index]

    @app.delete("/api/sources/{source_id}/transcript/segments/{segment_index}/override")
    def clear_transcript_segment_override(
        source_id: UUID, segment_index: int, database: Session = Depends(session)
    ) -> dict[str, object]:
        """Restore automatic corrected text while keeping the feedback audit fields intact."""

        transcript = _transcript_or_404(database, source_id)
        segments = _copy_segments(transcript)
        segment = _segment_or_404(segments, segment_index)
        segment["operator_text"] = None
        segment["final_text"] = _automatic_segment_text(segment)
        _persist_final_segments(database, transcript, segments)
        database.commit()
        database.refresh(transcript)
        return transcript.segments[segment_index]

    @app.get("/api/sources/{source_id}/transcript/search", response_model=TranscriptSearchResponse)
    def search_transcript(
        source_id: UUID,
        q: str = Query(min_length=1, max_length=256),
        database: Session = Depends(session),
    ) -> TranscriptSearchResponse:
        transcript = _transcript_or_404(database, source_id)
        query = q.casefold()
        return TranscriptSearchResponse(
            segments=[
                segment
                for segment in transcript.segments
                if query in _final_segment_text(segment).casefold()
            ]
        )

    @app.post(
        "/api/sources/{source_id}/retranscribe",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def retranscribe_source(
        source_id: UUID,
        force: bool = False,
        database: Session = Depends(session),
    ) -> JobResponse:
        """Queue a fresh local ASR run; option changes invalidate its transcript cache."""
        _source_or_404(database, source_id)
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.TRANSCRIPTION)
        database.add(job)
        database.commit()
        database.refresh(job)
        run_pipeline_stage.delay(
            str(source_id), PipelineStage.TRANSCRIPTION.value, str(job.id), force
        )
        return JobResponse.model_validate(job)

    @app.post(
        "/api/sources/{source_id}/reconstruct",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def reconstruct_source(
        source_id: UUID,
        force: bool = False,
        database: Session = Depends(session),
    ) -> JobResponse:
        """Queue Stage 2.7 reconstruction while leaving ASR and correction caches intact."""

        _source_or_404(database, source_id)
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.RECONSTRUCTION)
        database.add(job)
        database.commit()
        database.refresh(job)
        run_pipeline_stage.delay(
            str(source_id), PipelineStage.CONTEXTUAL_RECONSTRUCTION.value, str(job.id), force
        )
        return JobResponse.model_validate(job)

    @app.post(
        "/api/sources/{source_id}/candidate-analysis",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def queue_candidate_analysis(
        source_id: UUID,
        force: bool = False,
        database: Session = Depends(session),
    ) -> JobResponse:
        """Queue Stage 3 candidate analysis without touching Stage 2 caches."""

        _source_or_404(database, source_id)
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.CANDIDATE_ANALYSIS)
        database.add(job)
        database.commit()
        database.refresh(job)
        run_pipeline_stage.delay(
            str(source_id), PipelineStage.CANDIDATE_ANALYSIS.value, str(job.id), force
        )
        return JobResponse.model_validate(job)

    @app.get(
        "/api/sources/{source_id}/candidate-analysis",
        response_model=CandidateAnalysisResponse,
    )
    def get_candidate_analysis(
        source_id: UUID, database: Session = Depends(session)
    ) -> CandidateAnalysisResponse:
        _source_or_404(database, source_id)
        analysis = database.scalar(
            select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source_id)
        )
        if analysis is None:
            raise HTTPException(status_code=404, detail="candidate analysis not found")
        return CandidateAnalysisResponse(
            provider_status=analysis.provider_status,
            semantic_provider_mode=analysis.semantic_provider_mode.value,
            cache_eligible=analysis.cache_eligible,
            metrics=_without_secrets(analysis.metrics),
        )

    @app.get("/api/sources/{source_id}/candidates", response_model=list[CandidateResponse])
    def list_candidates(
        source_id: UUID,
        offset: int = 0,
        limit: int = 50,
        include_rejected: bool = False,
        database: Session = Depends(session),
    ) -> list[CandidateResponse]:
        """List current candidates with bounded pagination and rejected filtering."""

        _source_or_404(database, source_id)
        bounded_offset = max(offset, 0)
        bounded_limit = min(max(limit, 1), 200)
        statement = select(ClipCandidate).where(
            ClipCandidate.source_video_id == source_id,
            ClipCandidate.is_current.is_(True),
        )
        if not include_rejected:
            statement = statement.where(
                ClipCandidate.disposition.in_(
                    [
                        CandidateDisposition.CANDIDATE,
                        CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
                    ]
                )
            )
        statement = (
            statement.order_by(ClipCandidate.clip_score.desc(), ClipCandidate.start_time.asc())
            .offset(bounded_offset)
            .limit(bounded_limit)
        )
        return [_candidate_response(candidate) for candidate in database.scalars(statement)]

    @app.get("/api/candidates/{candidate_id}", response_model=CandidateResponse)
    def get_candidate(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> CandidateResponse:
        candidate = database.get(ClipCandidate, candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        return _candidate_response(candidate)

    @app.post(
        "/api/candidates/{candidate_id}/refinements",
        response_model=RefinementQueueResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def queue_candidate_refinement_endpoint(
        candidate_id: UUID,
        priority: RefinementPriority = Query(...),
        force: bool = False,
        database: Session = Depends(session),
    ) -> RefinementQueueResponse:
        """Explicitly queue one candidate refinement at CANDIDATE or FINAL_CLIP."""

        try:
            candidate = validate_candidate_for_refinement(database, candidate_id)
            outcome = queue_candidate_refinement(
                database, storage_service, candidate, priority, force=force
            )
        except Stage35QueueError as error:
            raise _stage35_http(error) from error
        return RefinementQueueResponse(
            refinement_id=outcome.refinement_id,
            job_id=outcome.job_id,
            status=outcome.status,
            queued=outcome.queued,
            cached=outcome.cached,
            active=outcome.active,
        )

    @app.get(
        "/api/refinements/{refinement_id}",
        response_model=CandidateRefinementResponse,
    )
    def get_candidate_refinement(
        refinement_id: UUID, database: Session = Depends(session)
    ) -> CandidateRefinementResponse:
        refinement = get_refinement(database, refinement_id)
        if refinement is None:
            raise HTTPException(status_code=404, detail="candidate refinement not found")
        return _refinement_response(refinement)

    @app.get(
        "/api/candidates/{candidate_id}/refinements",
        response_model=list[CandidateRefinementResponse],
    )
    def list_candidate_refinements(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> list[CandidateRefinementResponse]:
        if database.get(ClipCandidate, candidate_id) is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        return [_refinement_response(row) for row in list_refinements(database, candidate_id)]

    @app.get(
        "/api/candidates/{candidate_id}/stage4-handoff",
        response_model=Stage4HandoffResponse,
    )
    def get_stage4_handoff(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> Stage4HandoffResponse:
        handoff = build_stage4_handoff(database, candidate_id)
        if handoff is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        return Stage4HandoffResponse(**handoff)

    @app.post(
        "/api/refinements/{refinement_id}/manual",
        response_model=CandidateRefinementResponse,
    )
    def submit_manual_transcript(
        refinement_id: UUID,
        request: ManualTranscriptRequest,
        database: Session = Depends(session),
    ) -> CandidateRefinementResponse:
        """Submit authoritative manual text and explicitly resolve named ambiguities."""

        try:
            refinement = apply_manual_transcript(
                database, refinement_id, request.text, request.resolutions
            )
        except Stage35QueueError as error:
            raise _stage35_http(error) from error
        return _refinement_response(refinement)

    @app.post(
        "/api/sources/{source_id}/candidate-refinements/batch",
        response_model=list[RefinementQueueResponse],
        status_code=status.HTTP_202_ACCEPTED,
    )
    def queue_candidate_refinement_batch(
        source_id: UUID,
        limit: int | None = None,
        force: bool = False,
        database: Session = Depends(session),
    ) -> list[RefinementQueueResponse]:
        """Queue a bounded, score-ordered candidate-grade batch (no FINAL_CLIP)."""

        try:
            outcomes = queue_candidate_batch(
                database, storage_service, source_id, limit=limit, force=force
            )
        except Stage35QueueError as error:
            raise _stage35_http(error) from error
        return [
            RefinementQueueResponse(
                refinement_id=outcome.refinement_id,
                job_id=outcome.job_id,
                status=outcome.status,
                queued=outcome.queued,
                cached=outcome.cached,
                active=outcome.active,
            )
            for outcome in outcomes
        ]

    @app.post(
        "/api/candidates/{candidate_id}/transformation-analyses",
        response_model=TransformationQueueResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def queue_transformation_analysis_endpoint(
        candidate_id: UUID,
        force: bool = False,
        database: Session = Depends(session),
    ) -> TransformationQueueResponse:
        """Explicitly queue one candidate-scoped Stage 4.0 eligibility analysis."""

        try:
            candidate = validate_candidate_for_transformation(database, candidate_id)
            outcome = queue_transformation_analysis(database, candidate, force=force)
        except TransformationQueueError as error:
            raise _transformation_http(error) from error
        return TransformationQueueResponse(
            analysis_id=outcome.analysis_id,
            job_id=outcome.job_id,
            status=outcome.status,
            queued=outcome.queued,
            cached=outcome.cached,
            active=outcome.active,
        )

    @app.get(
        "/api/candidates/{candidate_id}/transformation-analysis",
        response_model=TransformationAnalysisResponse,
    )
    def get_candidate_transformation_analysis(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> TransformationAnalysisResponse:
        if database.get(ClipCandidate, candidate_id) is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        analysis = get_analysis_for_candidate(database, candidate_id)
        if analysis is None:
            raise HTTPException(status_code=404, detail="transformation analysis not found")
        return _transformation_response(database, analysis)

    @app.get(
        "/api/transformation-analyses/{analysis_id}",
        response_model=TransformationAnalysisResponse,
    )
    def get_transformation_analysis(
        analysis_id: UUID, database: Session = Depends(session)
    ) -> TransformationAnalysisResponse:
        analysis = get_analysis(database, analysis_id)
        if analysis is None:
            raise HTTPException(status_code=404, detail="transformation analysis not found")
        return _transformation_response(database, analysis)

    @app.get(
        "/api/candidates/{candidate_id}/stage4-1-handoff",
        response_model=Stage41HandoffResponse,
    )
    def get_stage4_1_handoff(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> Stage41HandoffResponse:
        handoff = build_stage4_1_handoff(database, candidate_id)
        if handoff is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        return Stage41HandoffResponse(**handoff)

    @app.post(
        "/api/candidates/{candidate_id}/transformation-plans",
        response_model=PlanningQueueResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def queue_transformation_plans_endpoint(
        candidate_id: UUID,
        force: bool = False,
        database: Session = Depends(session),
    ) -> PlanningQueueResponse:
        """Explicitly queue one candidate-scoped Stage 4.1 planning run."""

        try:
            candidate, analysis = validate_candidate_for_planning(database, candidate_id)
            outcome = queue_transformation_planning(database, candidate, analysis, force=force)
        except PlanningQueueError as error:
            raise _planning_http(error) from error
        return PlanningQueueResponse(
            plan_set_id=outcome.plan_set_id,
            job_id=outcome.job_id,
            status=outcome.status,
            queued=outcome.queued,
            cached=outcome.cached,
            active=outcome.active,
        )

    @app.get(
        "/api/candidates/{candidate_id}/transformation-plans",
        response_model=TransformationPlanSetResponse,
    )
    def get_candidate_transformation_plans(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> TransformationPlanSetResponse:
        if database.get(ClipCandidate, candidate_id) is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        plan_set = get_plan_set_for_candidate(database, candidate_id)
        if plan_set is None:
            raise HTTPException(status_code=404, detail="transformation plan set not found")
        return _plan_set_response(database, plan_set)

    @app.get(
        "/api/transformation-plan-sets/{plan_set_id}",
        response_model=TransformationPlanSetResponse,
    )
    def get_transformation_plan_set(
        plan_set_id: UUID, database: Session = Depends(session)
    ) -> TransformationPlanSetResponse:
        plan_set = get_plan_set(database, plan_set_id)
        if plan_set is None:
            raise HTTPException(status_code=404, detail="transformation plan set not found")
        return _plan_set_response(database, plan_set)

    @app.get(
        "/api/candidates/{candidate_id}/stage4-2-handoff",
        response_model=Stage42HandoffResponse,
    )
    def get_stage4_2_handoff(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> Stage42HandoffResponse:
        handoff = build_stage4_2_handoff(database, candidate_id)
        if handoff is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        return Stage42HandoffResponse(**handoff)

    @app.post(
        "/api/candidates/{candidate_id}/transformation-governance",
        response_model=GovernanceQueueResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def queue_transformation_governance_endpoint(
        candidate_id: UUID,
        force: bool = False,
        database: Session = Depends(session),
    ) -> GovernanceQueueResponse:
        """Explicitly queue one candidate-scoped Stage 4.2 governance run."""

        try:
            candidate, plan_set = validate_candidate_for_governance(database, candidate_id)
            outcome = queue_transformation_governance(database, candidate, plan_set, force=force)
        except GovernanceQueueError as error:
            raise _governance_http(error) from error
        return GovernanceQueueResponse(
            governance_set_id=outcome.governance_set_id,
            job_id=outcome.job_id,
            status=outcome.status,
            queued=outcome.queued,
            cached=outcome.cached,
            active=outcome.active,
        )

    @app.get(
        "/api/candidates/{candidate_id}/transformation-governance",
        response_model=GovernanceSetResponse,
    )
    def get_candidate_transformation_governance(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> GovernanceSetResponse:
        if database.get(ClipCandidate, candidate_id) is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        governance_set = get_governance_set_for_candidate(database, candidate_id)
        if governance_set is None:
            raise HTTPException(status_code=404, detail="transformation governance set not found")
        return _governance_set_response(database, governance_set)

    @app.get(
        "/api/transformation-governance-sets/{governance_set_id}",
        response_model=GovernanceSetResponse,
    )
    def get_transformation_governance_set(
        governance_set_id: UUID, database: Session = Depends(session)
    ) -> GovernanceSetResponse:
        governance_set = get_governance_set(database, governance_set_id)
        if governance_set is None:
            raise HTTPException(status_code=404, detail="transformation governance set not found")
        return _governance_set_response(database, governance_set)

    @app.get(
        "/api/candidates/{candidate_id}/stage4-3-handoff",
        response_model=Stage43HandoffResponse,
    )
    def get_stage4_3_handoff(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> Stage43HandoffResponse:
        handoff = build_stage4_3_handoff(database, candidate_id)
        if handoff is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        return Stage43HandoffResponse(**handoff)

    @app.post(
        "/api/candidates/{candidate_id}/transformation-selection",
        response_model=TransformationSelectionResponse,
    )
    def create_transformation_selection(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> TransformationSelectionResponse:
        """Synchronously select or reuse one current Stage 4.3 plan outcome."""

        view = select_transformation_plan(database, candidate_id)
        if view is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        response = _selection_response(view.row, view.live_freshness, view.effective)
        database.commit()
        return response

    @app.get(
        "/api/candidates/{candidate_id}/transformation-selection",
        response_model=TransformationSelectionResponse,
    )
    def get_candidate_transformation_selection(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> TransformationSelectionResponse:
        view = read_selection(database, candidate_id)
        if view is None:
            raise HTTPException(status_code=404, detail="transformation selection not found")
        return _selection_response(view.row, view.live_freshness, view.effective)

    @app.get(
        "/api/transformation-selections/{selection_id}",
        response_model=TransformationSelectionResponse,
    )
    def get_transformation_selection(
        selection_id: UUID, database: Session = Depends(session)
    ) -> TransformationSelectionResponse:
        row = get_selection(database, selection_id)
        if row is None:
            raise HTTPException(status_code=404, detail="transformation selection not found")
        # Always serialize the requested row and compute its own state. A
        # historical row must never be replaced by the candidate's current row.
        current = get_current_selection(database, row.clip_candidate_id)
        view = read_selection(database, row.clip_candidate_id)
        live_freshness = view.live_freshness if view is not None else "NOT_CURRENT"
        is_current_row = current is not None and current.id == row.id
        effective = bool(is_current_row and view is not None and view.effective)
        return _selection_response(row, live_freshness, effective)

    @app.get(
        "/api/candidates/{candidate_id}/execution-handoff",
        response_model=ExecutionHandoffResponse,
    )
    def get_execution_handoff(
        candidate_id: UUID, database: Session = Depends(session)
    ) -> ExecutionHandoffResponse:
        handoff = build_execution_handoff(database, candidate_id)
        if handoff is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        return ExecutionHandoffResponse(**handoff)

    @app.patch("/api/sources/{source_id}/provenance", response_model=SourceResponse)
    def update_source_provenance(
        source_id: UUID,
        request: ProvenanceUpdateRequest,
        database: Session = Depends(session),
    ) -> SourceResponse:
        """Explicitly update provenance; Stage 3 is invalidated, Stage 2 is not."""

        source = _source_or_404(database, source_id)
        if request.rights_status is not None:
            source.rights_status = request.rights_status
        if request.media_origin is not None:
            source.media_origin = request.media_origin
        if request.provenance_metadata is not None:
            source.provenance_metadata = _validate_provenance_metadata(request.provenance_metadata)
        database.commit()
        database.refresh(source)
        return SourceResponse.model_validate(source)

    @app.delete("/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_source(source_id: UUID, database: Session = Depends(session)) -> None:
        source = _source_or_404(database, source_id)
        if any(job.status in {JobStatus.QUEUED, JobStatus.RUNNING} for job in source.jobs):
            raise HTTPException(status_code=409, detail="cannot delete source with active jobs")
        if source.source_uri and not source.source_uri.startswith(("http://", "https://")):
            path = storage_service.source_directory(source.id)
            if path.exists():
                shutil.rmtree(path)
        database.delete(source)
        database.commit()

    @app.post(
        "/sources/{source_id}/process",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def process_source(source_id: UUID, database: Session = Depends(session)) -> JobResponse:
        _source_or_404(database, source_id)
        job = _new_job(source_id)
        database.add(job)
        database.commit()
        database.refresh(job)
        task_dispatcher.dispatch(source_id, job.id)
        return JobResponse.model_validate(job)

    @app.post(
        "/sources/{source_id}/retry",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def retry_source(source_id: UUID, database: Session = Depends(session)) -> JobResponse:
        latest = database.scalar(
            select(ProcessingJob)
            .where(ProcessingJob.source_video_id == source_id)
            .order_by(ProcessingJob.created_at.desc())
        )
        if latest is None or latest.status not in {JobStatus.FAILED, JobStatus.CANCELLED}:
            raise HTTPException(status_code=409, detail="source has no failed or cancelled job")
        latest.status = JobStatus.QUEUED
        latest.retry_count += 1
        latest.error_code = None
        latest.error_message = None
        database.commit()
        database.refresh(latest)
        task_dispatcher.dispatch(source_id, latest.id)
        return JobResponse.model_validate(latest)

    @app.get("/jobs", response_model=list[JobResponse])
    def list_jobs(database: Session = Depends(session)) -> list[JobResponse]:
        return [
            JobResponse.model_validate(job)
            for job in database.scalars(
                select(ProcessingJob).order_by(ProcessingJob.created_at.desc())
            )
        ]

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: UUID, database: Session = Depends(session)) -> JobResponse:
        return JobResponse.model_validate(_job_or_404(database, job_id))

    @app.post("/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel_job(job_id: UUID, database: Session = Depends(session)) -> JobResponse:
        job = _job_or_404(database, job_id)
        if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise HTTPException(status_code=409, detail="job cannot be cancelled")
        job.status = JobStatus.CANCELLED
        database.commit()
        database.refresh(job)
        return JobResponse.model_validate(job)

    @app.get("/system/health", response_model=HealthResponse)
    def health_report() -> HealthResponse:
        report = health_service.report()
        return HealthResponse(
            status=report.status,
            checks=[
                {"name": check.name, "status": check.status.value, "detail": check.detail}
                for check in report.checks
            ],
        )

    @app.get("/system/storage", response_model=StorageResponse)
    def storage_report() -> StorageResponse:
        report = health_service.storage_report()
        return StorageResponse(**report.__dict__)

    return app


def _safe_filename(filename: str | None) -> str:
    candidate = (filename or "upload.bin").replace("\\", "/").split("/")[-1]
    if candidate in {"", ".", ".."} or len(candidate) > 512:
        raise HTTPException(status_code=422, detail="invalid upload filename")
    return candidate


PROVENANCE_MAX_KEYS = 12
PROVENANCE_MAX_KEY_LENGTH = 64
PROVENANCE_MAX_VALUE_LENGTH = 2048


def _parse_provenance_form(value: str | None) -> dict[str, str]:
    if value is None or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail="provenance_metadata must be JSON") from error
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="provenance_metadata must be an object")
    return _validate_provenance_metadata(parsed)


def _validate_provenance_metadata(value: Mapping[str, object]) -> dict[str, str]:
    if len(value) > PROVENANCE_MAX_KEYS:
        raise HTTPException(status_code=422, detail="provenance_metadata has too many keys")
    validated: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or len(key) > PROVENANCE_MAX_KEY_LENGTH:
            raise HTTPException(status_code=422, detail="provenance_metadata key is too long")
        text = str(item)
        if len(text) > PROVENANCE_MAX_VALUE_LENGTH:
            raise HTTPException(status_code=422, detail="provenance_metadata value is too long")
        validated[key] = text
    return validated


def _candidate_response(candidate: ClipCandidate) -> CandidateResponse:
    return CandidateResponse(
        id=candidate.id,
        source_video_id=candidate.source_video_id,
        candidate_key=candidate.candidate_key,
        is_current=candidate.is_current,
        disposition=candidate.disposition,
        start_time=candidate.start_time,
        end_time=candidate.end_time,
        start_segment_index=candidate.start_segment_index,
        end_segment_index=candidate.end_segment_index,
        segment_indexes=list(candidate.segment_indexes or []),
        transcript_excerpt=candidate.transcript_excerpt[:4000],
        primary_content_type=candidate.primary_content_type,
        secondary_content_types=list(candidate.secondary_content_types or []),
        clip_score=candidate.clip_score,
        short_form_score=candidate.short_form_score,
        moment_density_score=candidate.moment_density_score,
        boredom_risk_score=candidate.boredom_risk_score,
        ending_quality_score=candidate.ending_quality_score,
        loopability_score=candidate.loopability_score,
        engagement_confidence=candidate.engagement_confidence,
        transcript_confidence=candidate.transcript_confidence,
        audio_confidence=candidate.audio_confidence,
        boundary_confidence=candidate.boundary_confidence,
        uncertainty_severity=candidate.uncertainty_severity,
        idea_novelty_score=candidate.idea_novelty_score,
        topic_novelty_score=candidate.topic_novelty_score,
        recent_semantic_similarity_risk=candidate.recent_semantic_similarity_risk,
        refinement_reasons=list(candidate.refinement_reasons or []),
        refinement_evidence=_without_secrets(dict(candidate.refinement_evidence or {})),
        rights_risk=candidate.rights_risk,
        originality_risk=candidate.originality_risk,
        dialect_profile=candidate.dialect_profile,
        dialect_confidence=candidate.dialect_confidence,
        code_switch_suspected=candidate.code_switch_suspected,
        hooks=list(candidate.hooks or []),
        idea_summary=candidate.idea_summary,
        topic_summary=candidate.topic_summary,
        provider_evidence=_without_secrets(dict(candidate.provider_evidence or {})),
        policy_version=candidate.policy_version,
        created_at=candidate.created_at,
    )


def _new_job(source_id: UUID) -> ProcessingJob:
    return ProcessingJob(source_video_id=source_id, kind=JobKind.INGEST, status=JobStatus.QUEUED)


def _stage35_http(error: Stage35QueueError) -> HTTPException:
    detail = str(error)
    code = 404 if "does not exist" in detail else 409
    return HTTPException(status_code=code, detail=detail)


def _transformation_http(error: TransformationQueueError) -> HTTPException:
    detail = str(error)
    code = 404 if "does not exist" in detail else 409
    return HTTPException(status_code=code, detail=detail)


def _strategy_response(row: object) -> TransformationStrategyResponse:
    return TransformationStrategyResponse(
        id=str(row.id),  # type: ignore[attr-defined]
        strategy_key=row.strategy_key,  # type: ignore[attr-defined]
        strategy_type=row.strategy_type.value,  # type: ignore[attr-defined]
        disposition=row.disposition.value,  # type: ignore[attr-defined]
        is_current=row.is_current,  # type: ignore[attr-defined]
        rank=row.rank,  # type: ignore[attr-defined]
        intensity=row.intensity.value,  # type: ignore[attr-defined]
        direction_summary=row.direction_summary,  # type: ignore[attr-defined]
        added_value_focus=row.added_value_focus,  # type: ignore[attr-defined]
        substantive_value_kind=row.substantive_value_kind.value,  # type: ignore[attr-defined]
        source_moment_role=row.source_moment_role,  # type: ignore[attr-defined]
        preservation_requirements=list(row.preservation_requirements or []),  # type: ignore[attr-defined]
        retention_preservation=row.retention_preservation,  # type: ignore[attr-defined]
        source_moment_damage_risk=row.source_moment_damage_risk,  # type: ignore[attr-defined]
        added_value_density=row.added_value_density,  # type: ignore[attr-defined]
        originality_potential=row.originality_potential,  # type: ignore[attr-defined]
        source_dominance_risk=row.source_dominance_risk,  # type: ignore[attr-defined]
        generic_filler_risk=row.generic_filler_risk,  # type: ignore[attr-defined]
        redundant_commentary_risk=row.redundant_commentary_risk,  # type: ignore[attr-defined]
        template_staleness_risk=row.template_staleness_risk,  # type: ignore[attr-defined]
        external_verification_requirement=row.external_verification_requirement.value,  # type: ignore[attr-defined]
        verification_requirements=list(row.verification_requirements or []),  # type: ignore[attr-defined]
        rejection_reasons=list(row.rejection_reasons or []),  # type: ignore[attr-defined]
        confidence=row.confidence,  # type: ignore[attr-defined]
        origin=row.origin.value,  # type: ignore[attr-defined]
        strategy_fingerprint=row.strategy_fingerprint,  # type: ignore[attr-defined]
    )


def _transformation_response(database: Session, analysis: object) -> TransformationAnalysisResponse:
    rows = list_strategies(database, analysis.id)  # type: ignore[attr-defined]
    recommended = [
        _strategy_response(row)
        for row in rows
        if row.is_current and row.disposition is StrategyDisposition.RECOMMENDED
    ]
    rejected = [
        _strategy_response(row)
        for row in rows
        if row.is_current and row.disposition is not StrategyDisposition.RECOMMENDED
    ]
    outcome = analysis.eligibility_outcome  # type: ignore[attr-defined]
    intensity = analysis.transformation_intensity  # type: ignore[attr-defined]
    return TransformationAnalysisResponse(
        id=analysis.id,  # type: ignore[attr-defined]
        source_video_id=analysis.source_video_id,  # type: ignore[attr-defined]
        clip_candidate_id=analysis.clip_candidate_id,  # type: ignore[attr-defined]
        refinement_id=analysis.refinement_id,  # type: ignore[attr-defined]
        refinement_priority=analysis.refinement_priority,  # type: ignore[attr-defined]
        refinement_quality_level=analysis.refinement_quality_level,  # type: ignore[attr-defined]
        execution_status=analysis.execution_status.value,  # type: ignore[attr-defined]
        eligibility_outcome=outcome.value if outcome else None,
        eligibility_reasons=list(analysis.eligibility_reasons or []),  # type: ignore[attr-defined]
        assessments=_without_secrets(dict(analysis.assessments or {})),  # type: ignore[attr-defined]
        source_moment=dict(analysis.source_moment or {}),  # type: ignore[attr-defined]
        platform_risk=_without_secrets(dict(analysis.platform_risk or {})),  # type: ignore[attr-defined]
        transformation_intensity=intensity.value if intensity else None,
        provider_mode=analysis.provider_mode.value,  # type: ignore[attr-defined]
        provider_status=analysis.provider_status,  # type: ignore[attr-defined]
        provider_evidence=_without_secrets(dict(analysis.provider_evidence or {})),  # type: ignore[attr-defined]
        provider_input_fingerprint=analysis.provider_input_fingerprint,  # type: ignore[attr-defined]
        input_fingerprint=analysis.input_fingerprint,  # type: ignore[attr-defined]
        output_fingerprint=analysis.output_fingerprint,  # type: ignore[attr-defined]
        cache_eligible=analysis.cache_eligible,  # type: ignore[attr-defined]
        metrics=_without_secrets(dict(analysis.metrics or {})),  # type: ignore[attr-defined]
        processing_duration=analysis.processing_duration,  # type: ignore[attr-defined]
        recommended_strategies=recommended,
        rejected_strategies=rejected,
    )


def _planning_http(error: PlanningQueueError) -> HTTPException:
    detail = str(error)
    code = 404 if "does not exist" in detail else 409
    return HTTPException(status_code=code, detail=detail)


def _plan_response(row: object) -> TransformationPlanResponse:
    return TransformationPlanResponse(
        id=row.id,  # type: ignore[attr-defined]
        plan_key=row.plan_key,  # type: ignore[attr-defined]
        is_current=row.is_current,  # type: ignore[attr-defined]
        status=row.status.value,  # type: ignore[attr-defined]
        generation_rank=row.generation_rank,  # type: ignore[attr-defined]
        strategy_candidate_id=row.strategy_candidate_id,  # type: ignore[attr-defined]
        strategy_type=row.strategy_type.value,  # type: ignore[attr-defined]
        intensity=row.intensity.value,  # type: ignore[attr-defined]
        strategy_fingerprint=row.strategy_fingerprint,  # type: ignore[attr-defined]
        blocks=list(row.blocks or []),  # type: ignore[attr-defined]
        hero_block_index=row.hero_block_index,  # type: ignore[attr-defined]
        hero_source_start=row.hero_source_start,  # type: ignore[attr-defined]
        hero_source_end=row.hero_source_end,  # type: ignore[attr-defined]
        hero_appearance_time=row.hero_appearance_time,  # type: ignore[attr-defined]
        original_value_kinds=list(row.original_value_kinds or []),  # type: ignore[attr-defined]
        narration_need=row.narration_need,  # type: ignore[attr-defined]
        narration_requirements=_without_secrets(dict(row.narration_requirements or {})),  # type: ignore[attr-defined]
        external_fact_dependencies=[
            dict(item)
            for item in (row.external_fact_dependencies or [])  # type: ignore[attr-defined]
        ],
        derived_durations=dict(row.derived_durations or {}),  # type: ignore[attr-defined]
        hook_payoff_evidence=dict(row.hook_payoff_evidence or {}),  # type: ignore[attr-defined]
        stage40_risk=_without_secrets(dict(row.stage40_risk or {})),  # type: ignore[attr-defined]
        source_dialect=dict(row.source_dialect or {}),  # type: ignore[attr-defined]
        target_audience=dict(row.target_audience or {}),  # type: ignore[attr-defined]
        planner_confidence=row.planner_confidence,  # type: ignore[attr-defined]
        generation_origin=row.generation_origin.value,  # type: ignore[attr-defined]
        planning_provider_evidence=_without_secrets(
            dict(row.planning_provider_evidence or {})  # type: ignore[attr-defined]
        ),
        provider_input_fingerprint=row.provider_input_fingerprint,  # type: ignore[attr-defined]
        plan_output_fingerprint=row.plan_output_fingerprint,  # type: ignore[attr-defined]
    )


def _plan_set_response(database: Session, plan_set: object) -> TransformationPlanSetResponse:
    outcome = plan_set.planning_outcome  # type: ignore[attr-defined]
    return TransformationPlanSetResponse(
        id=plan_set.id,  # type: ignore[attr-defined]
        source_video_id=plan_set.source_video_id,  # type: ignore[attr-defined]
        clip_candidate_id=plan_set.clip_candidate_id,  # type: ignore[attr-defined]
        transformation_analysis_id=plan_set.transformation_analysis_id,  # type: ignore[attr-defined]
        refinement_id=plan_set.refinement_id,  # type: ignore[attr-defined]
        refinement_priority=plan_set.refinement_priority,  # type: ignore[attr-defined]
        refinement_quality_level=plan_set.refinement_quality_level,  # type: ignore[attr-defined]
        execution_status=plan_set.execution_status.value,  # type: ignore[attr-defined]
        planning_outcome=outcome.value if outcome else None,
        outcome_reasons=list(plan_set.outcome_reasons or []),  # type: ignore[attr-defined]
        stage40_snapshot=_without_secrets(dict(plan_set.stage40_snapshot or {})),  # type: ignore[attr-defined]
        target_context=dict(plan_set.target_context or {}),  # type: ignore[attr-defined]
        provider_mode=plan_set.provider_mode.value,  # type: ignore[attr-defined]
        provider_status=plan_set.provider_status,  # type: ignore[attr-defined]
        provider_identity=dict(plan_set.provider_identity or {}),  # type: ignore[attr-defined]
        provider_evidence=_without_secrets(dict(plan_set.provider_evidence or {})),  # type: ignore[attr-defined]
        strategy_attempts=[dict(item) for item in (plan_set.strategy_attempts or [])],  # type: ignore[attr-defined]
        input_fingerprint=plan_set.input_fingerprint,  # type: ignore[attr-defined]
        output_fingerprint=plan_set.output_fingerprint,  # type: ignore[attr-defined]
        cache_eligible=plan_set.cache_eligible,  # type: ignore[attr-defined]
        metrics=_without_secrets(dict(plan_set.metrics or {})),  # type: ignore[attr-defined]
        processing_duration=plan_set.processing_duration,  # type: ignore[attr-defined]
        plans=[_plan_response(row) for row in list_plans(database, plan_set.id)],  # type: ignore[attr-defined]
    )


def _governance_http(error: GovernanceQueueError) -> HTTPException:
    detail = str(error)
    code = 404 if "does not exist" in detail else 409
    return HTTPException(status_code=code, detail=detail)


def _governance_result_response(row: object) -> GovernanceResultResponse:
    return GovernanceResultResponse(
        id=row.id,  # type: ignore[attr-defined]
        transformation_plan_id=row.transformation_plan_id,  # type: ignore[attr-defined]
        plan_output_fingerprint=row.plan_output_fingerprint,  # type: ignore[attr-defined]
        status=row.status.value,  # type: ignore[attr-defined]
        eligible_for_stage4_3=row.eligible_for_stage4_3,  # type: ignore[attr-defined]
        severity=row.severity,  # type: ignore[attr-defined]
        hard_gates=[dict(item) for item in (row.hard_gates or [])],  # type: ignore[attr-defined]
        dimensions=dict(row.dimensions or {}),  # type: ignore[attr-defined]
        verification=dict(row.verification or {}),  # type: ignore[attr-defined]
        platform_risk=_without_secrets(dict(row.platform_risk or {})),  # type: ignore[attr-defined]
        reason_codes=list(row.reason_codes or []),  # type: ignore[attr-defined]
        warnings=[_public_metadata_value(item) for item in (row.warnings or [])],  # type: ignore[attr-defined]
        remediation=[_public_metadata_value(item) for item in (row.remediation or [])],  # type: ignore[attr-defined]
        provider_evidence=_without_secrets(  # type: ignore[attr-defined]
            dict(row.governance_provider_evidence or {})
        ),
        input_fingerprint=row.input_fingerprint,  # type: ignore[attr-defined]
        output_fingerprint=row.output_fingerprint,  # type: ignore[attr-defined]
    )


def _governance_set_response(database: Session, governance_set: object) -> GovernanceSetResponse:
    outcome = governance_set.governance_outcome  # type: ignore[attr-defined]
    return GovernanceSetResponse(
        id=governance_set.id,  # type: ignore[attr-defined]
        source_video_id=governance_set.source_video_id,  # type: ignore[attr-defined]
        clip_candidate_id=governance_set.clip_candidate_id,  # type: ignore[attr-defined]
        transformation_plan_set_id=governance_set.transformation_plan_set_id,  # type: ignore[attr-defined]
        transformation_analysis_id=governance_set.transformation_analysis_id,  # type: ignore[attr-defined]
        refinement_id=governance_set.refinement_id,  # type: ignore[attr-defined]
        refinement_priority=governance_set.refinement_priority,  # type: ignore[attr-defined]
        refinement_quality_level=governance_set.refinement_quality_level,  # type: ignore[attr-defined]
        execution_status=governance_set.execution_status.value,  # type: ignore[attr-defined]
        semantic_outcome=outcome.value if outcome else None,
        outcome_reasons=list(governance_set.outcome_reasons or []),  # type: ignore[attr-defined]
        summary_counts=dict(governance_set.summary_counts or {}),  # type: ignore[attr-defined]
        stage40_snapshot=_without_secrets(  # type: ignore[attr-defined]
            dict(governance_set.stage40_snapshot or {})
        ),
        target_context=dict(governance_set.target_context or {}),  # type: ignore[attr-defined]
        provider_mode=governance_set.provider_mode.value,  # type: ignore[attr-defined]
        provider_status=governance_set.provider_status,  # type: ignore[attr-defined]
        provider_identity=dict(governance_set.provider_identity or {}),  # type: ignore[attr-defined]
        provider_evidence=_without_secrets(  # type: ignore[attr-defined]
            dict(governance_set.provider_evidence or {})
        ),
        plan_attempts=[  # type: ignore[attr-defined]
            _public_metadata_value(item) for item in (governance_set.plan_attempts or [])
        ],
        platform_policy_profile_version=governance_set.platform_policy_profile_version,  # type: ignore[attr-defined]
        platform_policy_checked_at=governance_set.platform_policy_checked_at,  # type: ignore[attr-defined]
        input_fingerprint=governance_set.input_fingerprint,  # type: ignore[attr-defined]
        output_fingerprint=governance_set.output_fingerprint,  # type: ignore[attr-defined]
        policy_version=governance_set.policy_version,  # type: ignore[attr-defined]
        schema_version=governance_set.schema_version,  # type: ignore[attr-defined]
        validation_version=governance_set.validation_version,  # type: ignore[attr-defined]
        cache_eligible=governance_set.cache_eligible,  # type: ignore[attr-defined]
        metrics=_without_secrets(dict(governance_set.metrics or {})),  # type: ignore[attr-defined]
        processing_duration=governance_set.processing_duration,  # type: ignore[attr-defined]
        results=[
            _governance_result_response(row)
            for row in list_results(database, governance_set.id)  # type: ignore[attr-defined]
        ],
    )


def _selection_response(
    row: TransformationPlanSelection, live_freshness: str, effective: bool
) -> TransformationSelectionResponse:
    return TransformationSelectionResponse(
        id=row.id,
        source_video_id=row.source_video_id,
        clip_candidate_id=row.clip_candidate_id,
        transformation_analysis_id=row.transformation_analysis_id,
        transformation_plan_set_id=row.transformation_plan_set_id,
        transformation_governance_set_id=row.transformation_governance_set_id,
        selected_plan_id=row.selected_plan_id,
        selected_governance_result_id=row.selected_governance_result_id,
        refinement_id=row.refinement_id,
        refinement_priority=row.refinement_priority,
        refinement_quality_level=row.refinement_quality_level,
        refinement_output_fingerprint=row.refinement_output_fingerprint,
        status=row.status.value,
        is_current=bool(row.is_current),
        selected_with_caution=bool(row.selected_with_caution),
        selection_reason_codes=list(row.selection_reason_codes or []),
        arbitration_evidence=_without_secrets(dict(row.arbitration_evidence or {})),
        selected_governance_snapshot=_without_secrets(dict(row.selected_governance_snapshot or {})),
        alternative_dispositions=[
            _public_metadata_value(item) for item in (row.alternative_dispositions or [])
        ],
        governance_input_fingerprint=row.governance_input_fingerprint,
        governance_output_fingerprint=row.governance_output_fingerprint,
        governor_policy_version=row.governor_policy_version,
        governor_validation_version=row.governor_validation_version,
        platform_policy_profile_version=row.platform_policy_profile_version,
        selected_plan_fingerprint=row.selected_plan_fingerprint,
        input_fingerprint=row.input_fingerprint,
        output_fingerprint=row.output_fingerprint,
        policy_version=row.policy_version,
        schema_version=row.schema_version,
        fingerprint_version=row.fingerprint_version,
        live_freshness=live_freshness,
        effective=effective,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _refinement_response(row: CandidateRefinement) -> CandidateRefinementResponse:
    return CandidateRefinementResponse(
        id=row.id,
        source_video_id=row.source_video_id,
        clip_candidate_id=row.clip_candidate_id,
        priority=row.priority,
        status=row.status.value if hasattr(row.status, "value") else str(row.status),
        quality_level=row.quality_level,
        coarse_start=row.coarse_start,
        coarse_end=row.coarse_end,
        context_start=row.context_start,
        context_end=row.context_end,
        refined_start=row.refined_start,
        refined_end=row.refined_end,
        automatic_transcript=row.automatic_transcript,
        manual_transcript=row.manual_transcript,
        final_transcript=row.final_transcript,
        word_timestamps=[dict(item) for item in (row.word_timestamps or [])],
        confidence=row.confidence,
        dialect_profile=row.dialect_profile,
        dialect_confidence=row.dialect_confidence,
        code_switch_evidence=_without_secrets(dict(row.code_switch_evidence or {})),
        transcript_evidence=[
            _public_metadata_value(item) for item in (row.transcript_evidence or [])
        ],
        entity_evidence=[_public_metadata_value(item) for item in (row.entity_evidence or [])],
        unresolved_spans=[_public_metadata_value(item) for item in (row.unresolved_spans or [])],
        provider_evidence=_without_secrets(dict(row.provider_evidence or {})),
        routing_evidence=_without_secrets(dict(row.routing_evidence or {})),
        input_fingerprint=row.input_fingerprint,
        output_fingerprint=row.output_fingerprint,
        cache_eligible=row.cache_eligible,
        metrics=_without_secrets(dict(row.metrics or {})),
        processing_duration=row.processing_duration,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _source_or_404(database: Session, source_id: UUID) -> SourceVideo:
    source = database.get(SourceVideo, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="source not found")
    return source


def _transcript_or_404(database: Session, source_id: UUID) -> Transcript:
    _source_or_404(database, source_id)
    transcript = database.scalar(select(Transcript).where(Transcript.source_video_id == source_id))
    if transcript is None:
        raise HTTPException(status_code=404, detail="transcript not found")
    return transcript


def _transcript_response(transcript: Transcript) -> TranscriptResponse:
    return TranscriptResponse(
        source_video_id=transcript.source_video_id,
        language=transcript.language,
        detected_language_probability=transcript.detected_language_probability,
        whisper_model=transcript.whisper_model,
        transcription_options=transcript.transcription_options,
        dialect_profile=transcript.dialect_profile,
        dialect_confidence=transcript.dialect_confidence,
        dialect_evidence=transcript.dialect_evidence,
        code_switch_suspected=transcript.code_switch_suspected,
        raw_text=transcript.raw_text,
        normalized_text=transcript.normalized_text,
        corrected_text=transcript.corrected_text,
        final_text=transcript.final_text,
        raw_transcript_confidence=transcript.raw_transcript_confidence,
        correction_confidence=transcript.correction_confidence,
        corrected_segment_ratio=transcript.corrected_segment_ratio,
        uncertain_segment_ratio=transcript.uncertain_segment_ratio,
        correction_method=transcript.correction_method,
        correction_version=transcript.correction_version,
        contextual_reconstructed_text=transcript.contextual_reconstructed_text,
        reconstruction_fingerprint=transcript.reconstruction_fingerprint,
        reconstruction_confidence=transcript.reconstruction_confidence,
        reconstructed_segment_ratio=transcript.reconstructed_segment_ratio,
        reconstruction_method=transcript.reconstruction_method,
        reconstruction_version=transcript.reconstruction_version,
        reconstruction_processing_duration=transcript.reconstruction_processing_duration,
        reconstruction_metadata=_public_reconstruction_metadata(transcript),
        reconstruction_status=transcript.reconstruction_status,
        segments=transcript.segments,
        word_segments=transcript.word_segments,
        duration=transcript.duration,
        processing_duration=transcript.processing_duration,
    )


def _quality_metrics_response(
    assessment: SourceQualityAssessment,
) -> QualityMetricsResponse:
    return QualityMetricsResponse(
        audio_quality_score=assessment.audio_quality_score,
        transcript_quality_score=assessment.transcript_quality_score,
        low_confidence_word_ratio=assessment.low_confidence_word_ratio,
        unresolved_segment_ratio=assessment.unresolved_segment_ratio,
        manual_review_required=assessment.manual_review_required,
        conservative_source_floor=assessment.overall_source_quality_score,
    )


def _public_reconstruction_metadata(transcript: Transcript) -> dict[str, object]:
    metadata = _without_secrets(transcript.reconstruction_metadata)
    availability = metadata.get("provider_availability")
    if (
        availability is None
        and transcript.reconstruction_status is ReconstructionStatus.PROVIDER_UNAVAILABLE
    ):
        availability = "UNAVAILABLE"
    metadata["reconstruction_status"] = transcript.reconstruction_status.value
    metadata["provider_health"] = {
        "availability": availability or "UNKNOWN",
        "model": metadata.get("model"),
        "model_digest": metadata.get("model_digest"),
    }
    return metadata


def _without_secrets(metadata: dict[str, object]) -> dict[str, object]:
    sensitive_fragments = ("api_key", "authorization", "password", "secret", "token")
    return {
        key: _public_metadata_value(value)
        for key, value in metadata.items()
        if not any(fragment in key.casefold() for fragment in sensitive_fragments)
    }


def _public_metadata_value(value: object) -> object:
    if isinstance(value, dict):
        return _without_secrets({str(key): item for key, item in value.items()})
    if isinstance(value, list):
        return [_public_metadata_value(item) for item in value]
    return value


def _job_or_404(database: Session, job_id: UUID) -> ProcessingJob:
    job = database.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _copy_segments(transcript: Transcript) -> list[dict[str, object]]:
    return [dict(segment) for segment in transcript.segments]


def _segment_or_404(segments: list[dict[str, object]], segment_index: int) -> dict[str, object]:
    if segment_index < 0 or segment_index >= len(segments):
        raise HTTPException(status_code=404, detail="transcript segment not found")
    return segments[segment_index]


def _automatic_segment_text(segment: dict[str, object]) -> str:
    if (
        segment.get("reconstruction_applied")
        and segment.get("reconstruction_confidence_level") == "HIGH"
    ):
        return str(segment.get("contextual_reconstructed_text") or "")
    return str(
        segment.get("corrected_text") or segment.get("normalized_text") or segment.get("text", "")
    )


def _final_segment_text(segment: dict[str, object]) -> str:
    return str(segment.get("final_text") or _automatic_segment_text(segment))


def _persist_final_segments(
    database: Session, transcript: Transcript, segments: list[dict[str, object]]
) -> None:
    """Atomically refresh only derived display/chunk state after manual text feedback."""

    transcript.segments = segments
    transcript.final_text = " ".join(_final_segment_text(segment) for segment in segments).strip()
    transcript.normalized_text = normalize_transcript(transcript.final_text)
    database.execute(delete(TranscriptChunk).where(TranscriptChunk.transcript_id == transcript.id))
    database.add_all(
        TranscriptChunk(
            transcript_id=transcript.id,
            sequence=sequence,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            text=chunk.text,
            segment_indexes=chunk.segment_indexes,
            preceding_context=chunk.preceding_context,
            following_context=chunk.following_context,
        )
        for sequence, chunk in enumerate(build_chunks(segments, ChunkConfig()))
    )


def _duplicate_response(source: SourceVideo) -> SourceResponse:
    return SourceResponse.model_validate(source)


def _default_health(
    storage: StorageService,
    factory: sessionmaker[Session],
    ffmpeg: str,
    ffprobe: str,
    reconstruction_provider: ReconstructionProvider | None,
) -> HealthService:
    def database() -> tuple[CheckStatus, str]:
        with factory() as session:
            session.execute(select(1))
        return CheckStatus.HEALTHY, "connected"

    def binary(name: str) -> tuple[CheckStatus, str]:
        return (
            (CheckStatus.HEALTHY, "available")
            if shutil.which(name)
            else (CheckStatus.DEGRADED, "not found")
        )

    def storage_check() -> tuple[CheckStatus, str]:
        try:
            report = HealthService(storage).storage_report()
        except OSError as err:
            return CheckStatus.FAILED, str(err)
        return CheckStatus.HEALTHY, f"{report.free_bytes} bytes free"

    checks = {
        "database": database,
        "redis": lambda: (CheckStatus.DEGRADED, "not checked"),
        "worker": lambda: (CheckStatus.DEGRADED, "heartbeat unavailable"),
        "ffmpeg": lambda: binary(ffmpeg),
        "ffprobe": lambda: binary(ffprobe),
        "storage": storage_check,
    }
    if reconstruction_provider is None:
        checks["reconstruction_provider"] = lambda: (
            CheckStatus.DEGRADED,
            "local Qwen reconstruction is disabled by default; "
            "set CLIPFACTORY_LOCAL_QWEN_ENABLED=true to enable local providers",
        )
    return HealthService(
        storage,
        checks,
        reconstruction_provider=reconstruction_provider,
    )
