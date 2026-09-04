# Task 2 Report: Durable Domain State and Migration

## Result

Implemented the Stage 1 SQLAlchemy domain foundation: enum definitions,
declarative base/session factories, `SourceVideo`, `ProcessingJob`, and
`PipelineRun` models, isolated SQLite fixture, Alembic environment, and initial
domain-state migration.

## Test-first evidence

1. Added `backend/tests/test_models.py` before production model code.
2. Initial red run failed because the Task 1 virtual environment lacked
   SQLAlchemy. After installing the already-declared database dependencies into
   that environment, the unchanged test failed as intended with
   `ModuleNotFoundError: No module named 'app'`.
3. Green run: `cd backend && .venv/bin/python -m pytest tests/test_models.py -q`
   returned `3 passed`.
4. Final backend test run: `5 passed`.

## Files

- `backend/app/core/enums.py`
- `backend/app/db/base.py`, `backend/app/db/session.py`
- `backend/app/models/{source_video,processing_job,pipeline_run}.py`
- `backend/app/models/__init__.py`
- `backend/alembic.ini`, `backend/alembic/env.py`
- `backend/alembic/versions/20260904_0001_initial_domain_state.py`
- `backend/tests/conftest.py`, `backend/tests/test_models.py`

## Verification

- Ruff lint: `All checks passed!`
- Ruff format: `16 files already formatted`
- Pytest: `5 passed`
- Mypy: `Success: no issues found in 14 source files`
- SQLite Alembic upgrade: revision `20260904_0001` applied successfully and
  reports head on a subsequent invocation.
- `git diff --check` passed.

## Concerns / deferrals

- PostgreSQL/container migration execution is deferred to Task 7: Compose has
  not yet been created (`docker compose config` reports no configuration file),
  so no backend container or PostgreSQL service exists to target.
- The existing virtual environment is Python 3.10 and initially lacked pip and
  SQLAlchemy; it was used only as a compatibility test environment. Supported
  Python 3.12 container verification remains part of Compose integration.
