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

## Stage 4.1 transformation plan generation

Stage 4.1 lives in `app/transformation/planning/` and is explicit,
candidate-scoped work after a current, non-stale Stage 4.0 analysis with at least
one current recommended strategy. It consumes only that handoff, accepts a usable
`CANDIDATE` Stage 3.5 refinement (never requires `FINAL_CLIP`), and produces zero
to three concrete validated plans — normally at most one per current recommended
strategy. It does not select, approve, authorize, or mark a winning plan, and it
adds **no** `PipelineStage`, **no** `PipelineRun`, **no** `_NEXT_STAGE` entry,
and no source lifecycle change. It extends the existing Celery/`ProcessingJob`
platform with a `TRANSFORMATION_PLANNING` job kind and a nullable
`processing_jobs.transformation_plan_set_id` FK. `transformation_plan_sets` (one
durable planning envelope per candidate) and `transformation_plans` (one stable
row per `(plan_set, strategy_candidate)`) persist the result; a zero-plan,
deferred, degraded, or provider-unavailable plan set is recorded truthfully
without fake plan rows. Bounded structured blocks are stored as validated JSON on
the plan rather than a block table.

Deterministic logic owns readiness, input bounds, routing, source-span
resolution, hero placement, duration arithmetic, substantive-value validation,
paraphrase/cosmetic rejection, narration/TTS separation, verification dependency
enforcement, material distinction, persistence eligibility, and cache/fingerprint
composition. An optional planning provider (hosted Gemini routine/strong tiers or
an explicit `local_only` Qwen) may only turn an already-approved strategy into a
concrete plan; the provider selects source spans by indexed word references or a
full-window sentinel and never supplies timestamps, quotes, TTS choices, or
rendering instructions. Hosted calls are batched per tier (at most one call per
tier and two per plan-set run) through the shared HIGH admission gate with
temperature 0 and strict structured validation. A missing key/outage/429/quota/
safety refusal/malformed output never fails the source: accepted per-strategy
checkpoints survive, only unfinished work stays non-cache-eligible, and a later
normal request retries it. Narration is an abstract semantic requirement; Stage
4.1 decides WHAT narration communicates, channel configuration will decide WHO
the persistent narrator is, and Stage 6 generates speech.
