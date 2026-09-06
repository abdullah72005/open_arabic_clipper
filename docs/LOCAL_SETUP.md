# Local setup

## Docker (supported)

Docker Desktop/Compose must be running. Then run:

```bash
cp .env.example .env
docker compose build
docker compose up -d postgres redis backend worker frontend
docker compose exec backend alembic upgrade head
docker compose ps
```

Use `docker compose logs -f backend worker` to observe jobs. The dashboard is
on `http://localhost:3301`; API docs are on `http://localhost:8300/docs`.

## Optional local reconstruction (Stage 2.7)

Stage 2.7 defaults to the managed local Ollama provider, which runs under the
optional `reconstruction` Compose profile. Starting the profile does not
download a model; the operator must pull the configured model explicitly:

```bash
docker compose --profile reconstruction up -d ollama
docker compose exec ollama ollama pull qwen3:8b
docker compose exec backend python -m app.cli reconstruction-health
```

`reconstruction-health` verifies endpoint reachability, exact model presence,
and model digest; it exits non-zero for a missing configuration, endpoint, or
model. Set `CLIPFACTORY_RECONSTRUCTION_PROVIDER=disabled` to run without a
provider, or `openai_compatible` for a different operator-configured local
endpoint. This machine has 7.4 GiB RAM and no CUDA, so CPU-only operation is the
supported default; the provider unloads the model after each run by default.

## Native backend

Native use requires Python 3.12, PostgreSQL, Redis, FFmpeg, and ffprobe.
From `backend/`, create a Python 3.12 environment, install with
`pip install -e '.[dev]'`, set `CLIPFACTORY_DATABASE_URL`,
`CLIPFACTORY_REDIS_URL`, and `CLIPFACTORY_STORAGE_ROOT`, then run
`alembic upgrade head`, `uvicorn app.main:app --reload`, and
`celery -A app.workers.celery_app:celery_app worker --concurrency=1`.

For the dashboard, run `npm install && npm run dev` from `frontend/`. Set
`NEXT_PUBLIC_API_BASE_URL=http://localhost:8300` before its build or dev server.
