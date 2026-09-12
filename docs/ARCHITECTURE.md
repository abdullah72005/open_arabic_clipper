# Architecture

The browser dashboard talks to FastAPI. FastAPI persists sources, jobs, and
pipeline runs in PostgreSQL, then schedules heavyweight work through Celery.
Redis is the Celery broker/result backend. A worker has concurrency one and
late acknowledgements so local media work is conservative and retryable.

`StorageService` is the sole application owner of filesystem paths below
`CLIPFACTORY_STORAGE_ROOT` (mounted as `./storage` in Compose). It creates
per-source folders, validates paths, checks capacity, and writes atomically.
The backend image uses Python 3.12 and installs FFmpeg/ffprobe from Debian.

Compose services are `postgres`, `redis`, `backend`, `worker`, `frontend`, and
the optional `ollama` service under the `reconstruction` profile. Postgres and
Redis use named volumes; source media uses the visible local `storage/` mount so
an operator can inspect or back it up. Ollama stores models in the
`ollama_models` volume and exposes no public port.

Stage 3 lives in `app/candidates/`: deterministic proposal generation, content
classification, scoring/uncertainty, hooks, novelty, a Stage 3-specific semantic
provider boundary (deterministic/adaptive-Gemini/local-Qwen), fingerprints, and
a durable `CandidateAnalysisExecutor`. The runner and worker add a
`CANDIDATE_ANALYSIS` stage and job kind; `cancellation` is generalized from the
Stage 2.7 reconstruction marker without changing Stage 2.7 behavior.
`candidate_analyses` and `clip_candidates` persist the result, and small API/CLI
surfaces expose provenance, queueing, and bounded inspection. There is no
separate queue, service process, orchestration subsystem, or analytics engine.
