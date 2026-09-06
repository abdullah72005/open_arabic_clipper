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
