# ClipFactory / open_arabic_clipper

ClipFactory is a local-first foundation for safely ingesting media and probing
its metadata. Stage 1 ends at `READY_FOR_TRANSCRIPTION`: it does not transcribe,
select clips, reframe, render, publish, or automatically authorize content.

Only process material you own or are explicitly authorized to process. URL
ingest accepts permitted public sources only. The software does not bypass DRM,
logins, paywalls, CAPTCHAs, or platform protections.

## Quick start (Docker)

```bash
cp .env.example .env
docker compose up --build -d
docker compose exec backend alembic upgrade head
docker compose exec backend python -m app.cli health
```

Open `http://localhost:3301` for the dashboard and `http://localhost:8300/docs`
for the API. Stop services with `docker compose down`; add `-v` only when you
intentionally want to remove database and Redis volumes. Local media remains
under `./storage`.

## Development checks

GitHub Actions runs these quality gates on every push and pull request: backend
tests with a coverage report, Ruff format/lint checks, frontend tests/lint/build,
and Docker Compose configuration validation.

```bash
docker compose config
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest pytest-asyncio httpx coverage ruff && \
   coverage run --source=app -m pytest && ruff format --check app tests && ruff check app tests"
(cd frontend && npm ci && npm test && npm run lint && npm run build)
```

The frontend image is built for the browser API base URL configured by
`NEXT_PUBLIC_API_BASE_URL`; use `http://localhost:8300` for local browser use.

Read [local setup](docs/LOCAL_SETUP.md), [architecture](docs/ARCHITECTURE.md),
[pipeline](docs/PIPELINE.md), and [troubleshooting](docs/TROUBLESHOOTING.md)
before using external media sources.
