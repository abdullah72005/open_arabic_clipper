"""FastAPI application factory for asynchronous local media ingestion."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterator
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

from app.core.enums import JobKind, JobStatus, PipelineStage, RightsStatus
from app.core.settings import get_settings
from app.db.session import create_session_factory
from app.models import ProcessingJob, SourceVideo, Transcript, TranscriptChunk
from app.services.health import CheckStatus, HealthService
from app.services.source_adapters import SourceValidationError, normalize_source_url
from app.services.storage import StorageCategory, StorageService
from app.transcription.chunking import ChunkConfig, build_chunks
from app.transcription.normalization import normalize_transcript
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


class SourceResponse(BaseModel):
    id: UUID
    source_uri: str
    original_filename: str | None
    rights_status: RightsStatus
    lifecycle_state: PipelineStage
    created_at: datetime

    model_config = {"from_attributes": True}


class JobResponse(BaseModel):
    id: UUID
    source_video_id: UUID
    kind: JobKind
    status: JobStatus
    retry_count: int
    error_code: str | None
    error_message: str | None

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
    segments: list[dict[str, object]]
    word_segments: list[dict[str, object]]
    duration: float
    processing_duration: float | None


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
        storage_service, factory, settings.ffmpeg_binary, settings.ffprobe_binary
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
        database: Session = Depends(session),
    ) -> SourceResponse:
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
                rights_status=rights_status,
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
        source = SourceVideo(source_uri=normalized, rights_status=request.rights_status)
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
        if force:
            transcript = database.scalar(
                select(Transcript).where(Transcript.source_video_id == source_id)
            )
            if transcript is not None:
                transcript.input_fingerprint = ""
        job = ProcessingJob(source_video_id=source_id, kind=JobKind.TRANSCRIPTION)
        database.add(job)
        database.commit()
        database.refresh(job)
        run_pipeline_stage.delay(str(source_id), PipelineStage.TRANSCRIPTION.value, str(job.id))
        return JobResponse.model_validate(job)

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


def _new_job(source_id: UUID) -> ProcessingJob:
    return ProcessingJob(source_video_id=source_id, kind=JobKind.INGEST, status=JobStatus.QUEUED)


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
        segments=transcript.segments,
        word_segments=transcript.word_segments,
        duration=transcript.duration,
        processing_duration=transcript.processing_duration,
    )


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
    storage: StorageService, factory: sessionmaker[Session], ffmpeg: str, ffprobe: str
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

    return HealthService(
        storage,
        {
            "database": database,
            "redis": lambda: (CheckStatus.DEGRADED, "not checked"),
            "worker": lambda: (CheckStatus.DEGRADED, "heartbeat unavailable"),
            "ffmpeg": lambda: binary(ffmpeg),
            "ffprobe": lambda: binary(ffprobe),
            "storage": storage_check,
        },
    )
