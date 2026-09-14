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

## Stage 4.0 transformation eligibility

Stage 4.0 lives in `app/transformation/` and is explicit, candidate-scoped work
after a candidate has a usable Stage 3.5 refinement. It decides whether a
credible substantive transformation path exists, produces at most three
recommended (and three useful rejected) strategy directions for Stage 4.1, and
returns `NO_TRANSFORMATION_STRATEGY_WORTH_USING` as a normal successful outcome
when no natural strategy clears the hard gates. It extends the existing
Celery/`ProcessingJob` platform with a `TRANSFORMATION_ELIGIBILITY` job kind and
a nullable `processing_jobs.transformation_analysis_id` FK. It adds **no**
`PipelineStage`, **no** `PipelineRun`, **no** `_NEXT_STAGE` entry, and no source
lifecycle change. `transformation_eligibility_analyses` (one current row per
candidate) and `transformation_strategy_candidates` (one current row per
`(analysis, strategy_type)`) persist the bounded result.

Deterministic logic owns prerequisites, transcript/context sufficiency,
transformation necessity, source-moment structure, content-to-strategy
suitability, presentation-only zero credit, hard gates, validation, final
eligibility, safe fallback, and deterministic ranking. An optional Stage
4-specific provider (hosted Gemini routine/strong tiers or an explicit
`local_only` Qwen) may only assess candidates that survived those gates, runs at
most one hosted call per analysis through the shared HIGH admission gate, and
never overrides a hard blocker. Rights/provenance risk and
originality/transformation risk stay separate; the platform-risk snapshot is
decision support, never a legal or monetization guarantee. Stage 4.1 receives a
typed read-only handoff only.
