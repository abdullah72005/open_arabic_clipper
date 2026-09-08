# Processing and Provenance Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow public URL ingestion and all local analysis regardless of truthful provenance state while retaining provenance for future publishing review.

**Architecture:** Remove the rights check from the ingest stage only. Extend the rights vocabulary without changing existing values, make UNKNOWN the truthful default, and use the existing retry endpoint. The source UI obtains job state from the jobs API and exposes the latest matching job.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Celery, pytest, Next.js, TypeScript.

**Spec:** `docs/superpowers/specs/2026-09-05-processing-provenance-separation-design.md`

## Task 1: Processing eligibility regression tests

- [ ] Replace the test that expects UNKNOWN public URL rejection with parameterized tests for UNKNOWN, THIRD_PARTY_REUSE, and OWNED acquisition.
- [ ] Run `pytest tests/test_stage2_pipeline_e2e.py` and confirm the new behavior fails before implementation.
- [ ] Remove the ingest rights gate and rerun the focused test.

## Task 2: Provenance and retry behavior

- [ ] Add truthful third-party provenance enum values and migration coverage.
- [ ] Add API tests that default URL sources to UNKNOWN and retry a failed source without recreating it.
- [ ] Implement the smallest API/model changes required by those tests.

## Task 3: Source job visibility

- [ ] Add a frontend API method for source retry and a source-detail testable job-state selection helper.
- [ ] Render current job status, failure reason, and retry button on source details.
- [ ] Run frontend lint and tests.

## Task 4: Documentation and full verification

- [ ] Update AGENTS.md and STATUS.md with the processing/publishing boundary.
- [ ] Run backend tests, formatting, linting, frontend checks, migrations, and the real public URL completion test.
