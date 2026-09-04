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
on `http://localhost:3000`; API docs are on `http://localhost:8000/docs`.

## Native backend

Native use requires Python 3.12, PostgreSQL, Redis, FFmpeg, and ffprobe.
From `backend/`, create a Python 3.12 environment, install with
`pip install -e '.[dev]'`, set `CLIPFACTORY_DATABASE_URL`,
`CLIPFACTORY_REDIS_URL`, and `CLIPFACTORY_STORAGE_ROOT`, then run
`alembic upgrade head`, `uvicorn app.main:app --reload`, and
`celery -A app.workers.celery_app:celery_app worker --concurrency=1`.

For the dashboard, run `npm install && npm run dev` from `frontend/`. Set
`NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` before its build or dev server.
