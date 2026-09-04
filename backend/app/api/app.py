"""FastAPI application factory for asynchronous local media ingestion."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator
from datetime import datetime
from typing import Protocol
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.enums import JobKind, JobStatus, PipelineStage, RightsStatus
from app.core.settings import get_settings
from app.db.session import create_session_factory
from app.models import ProcessingJob, SourceVideo
from app.services.health import CheckStatus, HealthService
from app.services.source_adapters import SourceValidationError, normalize_source_url
from app.services.storage import StorageService


class Dispatcher(Protocol):
    def dispatch(self, job_id: UUID) -> None: ...


class CeleryDispatcher:
    def dispatch(self, job_id: UUID) -> None:
        from app.workers.tasks import run_pipeline_stage

        run_pipeline_stage.delay("", PipelineStage.INGEST.value, str(job_id))


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
        content = file.file.read(upload_limit + 1)
        if not content:
            raise HTTPException(status_code=422, detail="upload must not be empty")
        if len(content) > upload_limit:
            raise HTTPException(status_code=413, detail="upload exceeds configured size limit")
        digest = hashlib.sha256(content).hexdigest()
        existing = database.scalar(select(SourceVideo).where(SourceVideo.content_hash == digest))
        if existing is not None:
            response.status_code = status.HTTP_200_OK
            return _duplicate_response(existing)
        source = SourceVideo(
            source_uri="",
            original_filename=filename,
            content_hash=digest,
            rights_status=rights_status,
        )
        database.add(source)
        database.flush()
        destination = storage_service.source_directory(source.id) / filename
        storage_service.ensure_capacity(len(content))
        storage_service.atomic_write(destination, iter((content,)))
        source.source_uri = str(destination)
        job = _new_job(source.id)
        database.add(job)
        database.commit()
        database.refresh(source)
        task_dispatcher.dispatch(job.id)
        return SourceResponse.model_validate(source)

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
        task_dispatcher.dispatch(job.id)
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
        task_dispatcher.dispatch(job.id)
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
        task_dispatcher.dispatch(latest.id)
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


def _job_or_404(database: Session, job_id: UUID) -> ProcessingJob:
    job = database.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


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
