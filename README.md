# ClipFactory / open_arabic_clipper

ClipFactory is a local-first foundation for safely ingesting media, probing its
metadata, and transcribing owned or authorized media. Stage 2 extracts a cached
mono 16 kHz WAV, runs local faster-whisper with automatic Arabic (Egyptian/MSA),
English, and mixed-speech detection, normalizes transcript text conservatively,
and records silence/quality signals through `READY_FOR_ANALYSIS`. It does not
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

## Stage 2 transcription

Workers need FFmpeg/ffprobe and the local `faster-whisper` dependency. Configure
`CLIPFACTORY_WHISPER_MODEL` (`tiny`, `base`, `small`, `medium`, or `large-v3`),
`CLIPFACTORY_WHISPER_DEVICE` (`auto`, `cpu`, or `cuda`), and optionally
`CLIPFACTORY_WHISPER_FORCED_LANGUAGE` (`ar` or `en`). `auto` uses CUDA only when
available and otherwise uses CPU `int8` inference.

Use `GET /api/sources/{id}/transcript` for the persisted raw and normalized
evidence, `GET /api/sources/{id}/transcript/search?q=...` for timestamped
segments, and `POST /api/sources/{id}/retranscribe` to queue a new local ASR job.
Arabic transcript panels render RTL when Arabic is detected; mixed segments retain
their original Unicode text. Source detail pages play storage-owned local media;
selecting a transcript segment seeks playback to its timestamp.

Operator commands are available from the backend environment: `python -m app.cli
transcribe SOURCE_ID`, `python -m app.cli transcript SOURCE_ID`, and `python -m
app.cli retranscribe SOURCE_ID`. The latter bypasses the cache by default.

Before selecting a deployment default, run `python -m app.cli benchmark
REPRESENTATIVE_AUTHORIZED_AUDIO.wav` on the target machine. It prints the source
duration, wall-clock time, real-time factor, audio-minutes-per-wall-minute,
model, device, and compute type from the actual local run. No representative
licensed Arabic sample is bundled with this repository, so benchmark figures are
intentionally not fabricated.

The current local cached-model benchmark is recorded in
[benchmark results](docs/BENCHMARKS.md).
