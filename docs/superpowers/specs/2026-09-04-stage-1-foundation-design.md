# Stage 1 Foundation Design

## Purpose

Build the local-first base for `open_arabic_clipper`: accepted video sources
are stored safely, probed asynchronously, and made ready for a later
transcription stage without starting any AI, render, or publishing behavior.

## Architecture

The backend is a FastAPI application backed by PostgreSQL. API writes create a
`SourceVideo` and a `ProcessingJob`; Celery workers perform acquisition and
probing, persist a pipeline-run record per stage, and update job progress.
Redis is both the Celery broker and result backend. The API only schedules
heavy work, so downloads and media analysis never occupy request handlers.

`StorageService` owns every persistent path below a configurable root, deriving
safe source-ID directories and offering atomic writes, capacity checks, and
temporary-file cleanup. Source adapters accept only validated inputs and use
argument-vector subprocess calls. The media service uses ffprobe JSON parsing
into typed metadata.

The explicit initial state machine is `INGEST -> PROBE -> READY_FOR_TRANSCRIPTION`.
Each completed stage is immutable unless an operator retry resets a failed or
cancelled run. Later stages are represented only by extension-point interfaces.
Their policy boundary rejects automated candidate generation, rendering, and
publishing for `UNKNOWN` rights unless a later explicit authorization policy
allows it.

## Domain and API

`SourceVideo` stores provenance, rights status, content hash, media metadata,
and lifecycle state. `ProcessingJob` tracks a stage-specific asynchronous
operation, retries, timings, cancellation, and error metadata. `PipelineRun`
records each stage execution so restarts can skip completed work.

The API offers source submission (URL and multipart upload), source listing and
detail, process/retry/delete controls, job listing/detail/cancellation, health,
and storage reporting. The CLI exposes corresponding safe local operations.

## Operations and UI

Compose runs PostgreSQL, Redis, backend, worker, and frontend with conservative
worker concurrency. The frontend is a functional shell with dashboard, sources,
add-source, source-detail, jobs, and settings pages consuming API data.
Health aggregates process, database, Redis, worker heartbeat, media binaries,
and storage directory status into HEALTHY/DEGRADED/FAILED results.

## Constraints

- Python 3.12 in supported containers; local host dependencies are configurable.
- No paid SaaS, Kubernetes, DRM/paywall/login/CAPTCHA bypassing, or untrusted
  shell interpolation.
- URL processing uses yt-dlp only for permitted public sources.
- All new observable behavior is test-first; routine tests use synthetic or
  mocked bounded media data, never public large downloads.
