# Stage 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a portable local-first source ingestion and media-probing application foundation for open_arabic_clipper.

**Architecture:** FastAPI writes durable source/job records to PostgreSQL and delegates heavyweight acquisition and probing to Celery through Redis. A StorageService owns all paths; adapters and media services have small typed interfaces; a Next.js dashboard reads the API.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Alembic, Pydantic, Celery, Redis, PostgreSQL, pytest, Ruff, Next.js, TypeScript, Tailwind CSS, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-04-stage-1-foundation-design.md`

## Global Constraints

- Docker is the supported runtime for this machine; it must run Python 3.12 and include FFmpeg/ffprobe.
- The application must also support native execution with dependencies supplied by the operator.
- Never use shell interpolation for user input; `subprocess.run` receives argument lists only.
- Stage 1 ends at `READY_FOR_TRANSCRIPTION`; no transcription, selection, rendering, or publishing implementation.
- UNKNOWN rights may ingest/probe but future automatic public-output stages require an explicit policy authorization.
- Tests must not download public media or require a GPU.

## File Structure

- `backend/app/core/`: settings, logging, errors, enums.
- `backend/app/db/`: SQLAlchemy base, engine/session lifecycle, Alembic metadata.
- `backend/app/models/`: source, job, pipeline-run persistence models.
- `backend/app/services/`: storage, hashing, source adapters, health service.
- `backend/app/media/`: ffprobe command and typed response parser.
- `backend/app/pipeline/`: stage definitions, authorization boundary, resilient runner.
- `backend/app/workers/`: Celery configuration and task wrappers.
- `backend/app/api/`: FastAPI routes and Pydantic request/response contracts.
- `backend/tests/`: isolated unit/API tests with temporary paths and SQLite fixtures.
- `frontend/`: Next.js dashboard pages and typed API client.
- root config/docs: Compose, environment example, CLI, status, architecture, setup, troubleshooting, README.

## Task 1: Repository and backend test harness

- [ ] Create `backend/pyproject.toml` with Python `>=3.12`, runtime dependencies (FastAPI, SQLAlchemy, Alembic, psycopg, Celery, Redis, Pydantic Settings, Typer, yt-dlp) and dev dependencies (pytest, pytest-asyncio, httpx, Ruff, mypy).
- [ ] Create `backend/tests/test_settings.py` first; assert an explicit test environment returns a storage root from `CLIPFACTORY_STORAGE_ROOT` and rejects a non-positive upload limit.
- [ ] Run `docker compose run --rm backend pytest tests/test_settings.py` and confirm it fails because settings do not exist.
- [ ] Implement `backend/app/core/settings.py` with a cached `Settings` class, required URLs/binary names/storage limits/CORS origins, and validation of positive limits.
- [ ] Re-run the test and add `backend/tests/conftest.py` that supplies a temporary SQLite database and storage root.
- [ ] Add Ruff/mypy configuration and format the backend.

## Task 2: Durable domain state and migration

- [ ] Write `backend/tests/test_models.py` first; create a `SourceVideo` with `UNKNOWN` rights and assert defaults, unique content hash behavior, a queued ingest job, and a pipeline-run state transition.
- [ ] Run the isolated test and confirm imports/models are missing.
- [ ] Implement `backend/app/core/enums.py`, SQLAlchemy declarative base/session modules, and models for `SourceVideo`, `ProcessingJob`, and `PipelineRun` with UUID IDs, timestamps, constraint-backed enum fields, and indexed lookup fields.
- [ ] Create Alembic environment/config and a checked-in initial migration reflecting those three tables.
- [ ] Run the model test against SQLite and run `alembic upgrade head` inside the backend container against PostgreSQL.

## Task 3: Storage and duplicate primitives

- [ ] Write `backend/tests/test_storage.py` first; assert a stable source directory stays below each category, traversal input raises a domain validation error, atomic writing replaces a complete final file, cleanup only removes old temp files, and insufficient free bytes raises a recoverable storage error.
- [ ] Run it to establish failure.
- [ ] Implement `backend/app/services/storage.py` with configured category roots, UUID validation, `shutil.disk_usage`, safe relative-path resolution, atomic temp-and-rename writes, and bounded cleanup.
- [ ] Add `backend/app/services/hashing.py` to stream SHA-256 with a fixed chunk size.
- [ ] Re-run storage tests and Ruff.

## Task 4: Source adapters and media metadata

- [ ] Write `backend/tests/test_adapters.py` first; assert URL validation rejects unsupported protocols/credentials, local-file acquisition copies a permitted real file to the source directory, and yt-dlp commands are argument vectors without `shell=True`.
- [ ] Write `backend/tests/test_ffprobe.py` first; assert valid JSON extracts duration/codecs/dimensions/fps/audio/sample rate and malformed/zero-denominator payloads return a typed probe error.
- [ ] Run both test modules and observe the missing adapter/media failures.
- [ ] Implement `SourceAdapter`, `LocalFileAdapter`, and `YtDlpAdapter` with explicit permission messaging, normal URL normalization, metadata inspection, sanitized names, disk-space checking, and no authentication bypass behavior.
- [ ] Implement typed ffprobe parsing plus a wrapper that calls configured ffprobe with `subprocess.run([...], check=True, capture_output=True, text=True)`.
- [ ] Run adapter/media tests and Ruff.

## Task 5: Pipeline and Celery workers

- [ ] Write `backend/tests/test_pipeline.py` first; assert a completed stage is skipped, failures persist stage/job error information, a retry advances retry count without replacing source identity, and the later AUTOPILOT authorization function rejects UNKNOWN rights.
- [ ] Run it and confirm it fails.
- [ ] Implement `PipelineStage`, `PipelineRunner`, a stage-executor protocol, and an explicit future authorization boundary in `backend/app/pipeline/`.
- [ ] Implement Celery configuration with JSON serialization, late acknowledgement, `worker_prefetch_multiplier=1`, default concurrency one, and task wrappers that create/update durable job progress and retry only retryable errors.
- [ ] Add a worker heartbeat task and run pipeline tests plus a container worker ping check.

## Task 6: API, health, and CLI

- [ ] Write `backend/tests/test_api_sources.py` first; assert upload validation, source creation, duplicate response behavior, URL request validation, safe delete semantics, job cancel behavior, and non-blocking task scheduling through an injected dispatcher.
- [ ] Write `backend/tests/test_health.py` first; assert individual unhealthy dependencies produce DEGRADED/FAILED aggregate health and storage reporting includes capacity values.
- [ ] Run these tests and confirm failures.
- [ ] Implement FastAPI app/router, Pydantic schemas, dependency-injected session/task dispatcher, source/job endpoints, and intentional local CORS configuration.
- [ ] Implement health checks for database, Redis, Celery worker heartbeat, FFmpeg/ffprobe, and storage, returning per-check results and aggregate severity.
- [ ] Implement Typer commands `health`, `add`, `status`, `retry`, and `cleanup` against the same service layer.
- [ ] Run API/health tests, all backend tests, Ruff, and mypy.

## Task 7: Compose, frontend, and docs

- [ ] Create Dockerfiles and Compose services for PostgreSQL, Redis, backend, worker, and frontend; keep mounts/configuration appropriate for local storage and conservative worker execution.
- [ ] Create `.env.example`, `.gitignore`, `STATUS.md`, and base storage category directories with `.gitkeep` files.
- [ ] Scaffold a Next.js TypeScript/Tailwind app, then add tests or type-level checks for its typed API client before creating dashboard pages.
- [ ] Implement Dashboard, Sources, Add Source (multiline URL input and upload form), Source Detail, Jobs, and Settings pages with usable loading/error states and no unavailable future-stage controls.
- [ ] Write `README.md`, `docs/ARCHITECTURE.md`, `docs/PIPELINE.md`, `docs/LOCAL_SETUP.md`, and `docs/TROUBLESHOOTING.md` with actual Docker/native commands and the legal/rights policy.
- [ ] Build frontend and validate Compose configuration.

## Task 8: Integration verification and remediation

- [ ] Start Compose services and run migration, backend health, Redis ping, Celery inspect ping, and media binary checks in their respective containers.
- [ ] Create a tiny synthetic MP4 with the backend image's FFmpeg, submit it through the local-upload API, wait for its ingest/probe jobs, and assert persisted metadata and `READY_FOR_TRANSCRIPTION` source state.
- [ ] Submit the same file again and assert duplicate detection returns the original source instead of duplicating it.
- [ ] Run the complete backend test suite, lint/type checks, frontend build, and `docker compose config`.
- [ ] Record exact evidence and any environment limitation in `STATUS.md`; repair failures before a completion claim.
