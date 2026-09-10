# Stage 2.7 operations

Stage 2.7 performs bounded contextual reconstruction after Stage 2.5. It is
local-first and never overwrites raw ASR text, segment timing, word timing, or
Stage 2.5 evidence.

## Refinement priorities: whole-source is INDEX

Transcript refinement uses an explicit quality ladder. Whole-source ingestion
always runs at **INDEX** (indexing quality): it preserves raw ASR, Stage 2.5,
timestamps, words, confidences, and correction/reconstruction evidence, and
defers every provider reconstruction truthfully (segments are marked
unresolved/manual-review with `escalation_reason=index_priority_deferred`,
never as a provider failure). Normal ingestion therefore pays for ASR + Stage
2.5 + cheap uncertainty bookkeeping only — no Qwen load, no Gemini calls, no
broad reconstruction. Unresolved INDEX text is not a pipeline failure and the
transcript still reaches analysis-ready when otherwise valid.

A shortlisted window is **CANDIDATE** (semantic quality) and a selected final
clip is **FINAL_CLIP** (publication/caption quality). Expensive work is deferred
until a short region is actually close to publication, at which point a caller
uses the reusable `refine_transcript_window(source_id, start_time, end_time,
priority)` service entry point (see below).

## Dialect awareness (Stage 2.7.1)

Stage 2.7.1 adds conservative source-level Arabic dialect awareness and exact
Arabic-English code-switch preservation. During Stage 2.5 normalization a pure,
deterministic detector classifies the source from immutable raw segment text:

- **Profiles.** `EGYPTIAN`, `SAUDI`, `GULF`, `LEVANTINE`, `MSA`, or
  `UNKNOWN_ARABIC`; `None` means no Arabic evidence. `UNKNOWN_ARABIC` means
  Arabic is present but uncertain/mixed and always favors no change. Dialect is
  source evidence, not a target audience; there is no deployment-wide default.
- **Lightweight.** No network, no LLM, no model loading, no audio decoding, no
  per-segment classifier. Marker scoring comes from a bounded representative
  sample of at most 48 segments and stores no transcript bodies; Arabic
  applicability itself scans every immutable raw segment, so an Arabic segment
  outside the sample is never lost (it resolves to `UNKNOWN_ARABIC`, not
  `None`).
- **Override.** An optional `dialect_profile_override` may be provided when a
  source is created through URL ingest JSON or multipart upload. It wins with
  confidence 1.0, is stored on the source, participates in normalization
  fingerprints, and an explicit `UNKNOWN_ARABIC` override means "do not force a
  dialect." The initial implementation accepts the override at source creation
  only; there is no post-ingest editing workflow in this stage.
- **Egyptian isolation.** The Egyptian Stage 2.5 lexicon and its optional
  provider apply only when the effective profile is confidently or explicitly
  EGYPTIAN. Other profiles and non-Arabic material pass through unchanged with
  zero optional Stage 2.5 provider calls.
- **Code switching.** Latin words/names, abbreviations, technical tokens, and
  numbers are preserved exactly through Stage 2.5 and every accepted Stage 2.7
  candidate; slash/`+`/`#`/URL technical forms are kept as exact atomic tokens
  (URLs include query, fragment, parameter, and percent-encoded syntax).
  `code_switch_suspected` is true only for a segment that itself contains both
  Arabic-script evidence and Latin-letter protected tokens (numbers alone and
  English-only segments are not flagged).
  Omitted-English audio recovery is deferred to Stage 3.5.
- **Shared provider contract.** Local Qwen and hosted Gemini receive the same
  dialect-neutral, preservation-first instruction plus validated profile-specific
  addenda; the shared reconstruction request carries the target segment's
  inherited effective dialect profile, and local prompt planning sizes the exact
  profile-specific instruction.

## Provider operation

The default configuration is the optional local Ollama provider at
`http://ollama:11434` using `qwen3.5:4b`. **Automatic local Qwen use is disabled
by default** (`CLIPFACTORY_LOCAL_QWEN_ENABLED=false`), so no local provider is
constructed and INDEX ingestion never loads or calls Qwen. An operator
explicitly re-enables local Qwen for targeted CANDIDATE/FINAL_CLIP refinement:

```bash
CLIPFACTORY_LOCAL_QWEN_ENABLED=true
```

Starting the Compose profile does not pull any model. An operator must
explicitly obtain the configured model before targeted refinement is available.

```bash
docker compose --profile reconstruction up -d ollama
docker compose exec ollama ollama pull qwen3.5:4b
docker compose exec backend python -m app.cli reconstruction-health
```

With Qwen disabled (the default), `reconstruction-health` reports
`MISCONFIGURED` with a message explaining the re-enable flag; this is the
expected default and is not an outage. When local-only is intentionally selected
but Qwen is disabled/unavailable, targeted work defers safely (unresolved/
provider-unavailable semantics) and never silently uses Gemini.

## Hosted Gemini provider and routing modes

Stage 2.7 also supports an optional hosted Gemini provider
(`gemini-3.8-flash` by default) through the official Google Gen AI SDK
(`google-genai`). The Gemini API key is read through application settings from
`GEMINI_API_KEY` or `CLIPFACTORY_GEMINI_API_KEY` in the local `.env`; it is held
as a Pydantic `SecretStr`, unwrapped only when constructing the SDK client, and
masked in settings repr/serialization/validation errors. The key is never
logged, exposed through the API or fingerprints, or committed. If the key is
absent the application starts and runs normally with the local path only.

The SDK client is constructed **lazily**, only when a generation request is
actually about to run. Availability is configuration-level (key present + model
configured); there is no per-job `models.get` metadata probe, and the bounded
generation request is the real availability check with classified failure
handling. A cache hit, an all-`NO_LLM` job, and `LOCAL_ONLY` work therefore make
zero Gemini generation, metadata, or network calls and ideally construct no
Gemini client. Generation is deterministic for reconstruction: temperature
defaults to `0` and the stable v1 API version is used explicitly.

Three routing modes are controlled by `CLIPFACTORY_RECONSTRUCTION_ROUTING_MODE`:

| Mode | Behavior |
| --- | --- |
| `local_only` | Never calls Gemini; uses Qwen for targets needing reconstruction; safe unresolved fallback. |
| `adaptive` (default) | Qwen for normal uncertainty; Gemini directly for clearly difficult targets; Qwen failures escalate to Gemini; safe local/unresolved degradation when Gemini is unavailable. |
| `gemini_only` | Skips Qwen; sends only targets that need LLM reconstruction to Gemini; still permits `NO_LLM`; degrades safely. |

**Cloud snippet disclosure.** `adaptive` and `gemini_only` may send short
transcript snippets and bounded context to Google Gemini. Configure the key and
mode explicitly; this is a cloud-processing configuration decision and is
separate from rights/provenance eligibility, which is evaluated independently.
Free-tier limits are shared per Google project, so all Gemini consumers
(including future project stages) draw from the same quota.

### Routing on real Stage 2.5 trust

`NO_LLM` is chosen only from affirmative evidence, and the decision is made on
the raw-ASR words Stage 2.5 did **not** resolve:

- **Clean unchanged result.** An unchanged Stage 2.5 result
  (`correction_method="unchanged"`) routes to `NO_LLM` when it is backed by
  adequate word coverage (`evidence_coverage_min` 0.80 of words carrying a
  probability), a high average word probability (`clean_average_probability`
  0.85), no low-probability span, no protected-token ambiguity, no contiguous
  uncertain span, no suspicious multi-word corruption, and no hard-corruption
  indicator. A clean high-probability segment never calls Qwen or Gemini merely
  because Stage 2.5 left it unchanged.
- **Trusted repair.** A high-confidence accepted Stage 2.5 repair
  (`correction_applied`, `correction_confidence >= 0.90`) removes the
  low-probability words its `correction_changes` cover from the residual
  evidence. A clean remainder routes to `NO_LLM` instead of reprocessing the
  repair. If an independent severe span remains unresolved, the remaining
  evidence still routes to the appropriate LLM path.
- **Missing evidence is not trust.** A segment without adequate word coverage
  stays eligible for a conservative local check and is never Gemini-direct
  merely because evidence is absent.

A single isolated low-confidence word is reported accurately and never labeled
`no_low_probability_evidence`. Routing thresholds and the policy version
(`adaptive-routing-v3`) participate in runtime identity and fingerprints, so any
threshold or coverage change invalidates prior runs.

### Strongest-first Gemini budget

`CLIPFACTORY_GEMINI_MAX_TARGETS_PER_JOB` (default `5`) caps Gemini reconstruction
targets per job. The budget is spent on the **five strongest eligible targets**,
not the first five in transcript order. Routing severity is deterministic
(existing routing score, bounded per-contiguous-very-low-word bonus, protected
overlap, multiple severe indicators), and ties break by segment index. Direct
Gemini candidates are ranked and allocated first; Qwen runs afterward; then
local failures/rejections are ranked by severity, protected ambiguity, local
outcome, and near-acceptance evidence and the remaining budget is spent on the
strongest escalations. Direct-Gemini targets never run Qwen before Gemini.

Operators may raise the budget for long videos after checking active AI Studio
quota. The cap reserves Gemini capacity for later project stages; it is not an
account-wide billing or quota manager. Future Gemini consumers will need
coordinated project-level budgeting because quotas are shared per Google project.

### Stable fingerprints, outages, and cache eligibility

Reconstruction fingerprints cover stable dependency identity only: provider
configuration, model identity/digest, prompt/schema, routing policy and
thresholds, local batching and work-ceiling settings, budgets, Gemini API
version and temperature, thinking level, validation/confidence versions, source
and context. Each segment's fingerprint also covers Stage 2.5 method, confidence,
applied state, a stable change digest, word probabilities, acoustic evidence,
language, dialect-profile carrier, and bounded surrounding context. Transient
provider availability is execution state and is excluded, so a temporary Gemini
outage cannot invalidate accepted output or change the stable model identity
(Gemini identity is configuration-derived; no `models.get` probe runs).

Each run also records `cache_eligible`: a run that completed without transient
provider failure or budget exhaustion is reusable; a run that degraded to
provider-unavailable fallback, hit quota exhaustion, or exhausted a local work
ceiling is retried on a later run. **Per-target reuse** means a restart after
cancellation or a late provider failure keeps already-accepted per-target work
(stored per-target dependency fingerprint plus per-target eligibility) without
re-calling a provider; only failed or eligible-unresolved targets are
reconsidered. Checkpoints persist completed batches through the existing
transcript metadata, so a cancellation or worker restart does not discard every
preceding expensive result. Manual overrides stay terminal. No generation call
is made to check a cache key.

Every executor exit path — a fresh cache hit, cached-output preservation during
an outage, all-NO_LLM, normal completion, cancellation, and provider failure —
releases owned provider resources idempotently. A fresh cache-hit worker
therefore scrubs the Gemini API key and closes owned SDK clients without ever
building a client or making a network call; unowned injected clients are never
closed.

### Heavy-model lease

The Ollama heavy-model lease is acquired lazily, only around actual local Qwen
inference and local model release. `NO_LLM`, `GEMINI_ONLY`, and direct-Gemini
work never acquire it; a direct-Gemini call runs before any optional local
fallback lease. Real local inference and unload remain lease-protected, and
unsafe-model markers and lost-lease behavior still fail closed.

### Gemini configuration settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `CLIPFACTORY_RECONSTRUCTION_ROUTING_MODE` | `adaptive` | Provider selection policy. |
| `GEMINI_API_KEY` / `CLIPFACTORY_GEMINI_API_KEY` | — | Secret, presence-detected API key. |
| `CLIPFACTORY_GEMINI_MODEL` | `gemini-3.8-flash` | Hosted model for reconstruction. |
| `CLIPFACTORY_GEMINI_THINKING_LEVEL` | `low` | Bounded reasoning (`low`/`medium`/`high`). |
| `CLIPFACTORY_GEMINI_TEMPERATURE` | `0` | Deterministic reconstruction generation. |
| `CLIPFACTORY_GEMINI_API_VERSION` | `v1` | Explicit stable Gemini API version. |
| `CLIPFACTORY_GEMINI_TIMEOUT_SECONDS` | `30` | Per-request timeout. |
| `CLIPFACTORY_GEMINI_RETRY_ATTEMPTS` | `1` | Bounded retries for transient failures only. |
| `CLIPFACTORY_GEMINI_RETRY_BACKOFF_SECONDS` | `1.5` | Backoff between retries. |
| `CLIPFACTORY_GEMINI_MAX_TARGETS_PER_JOB` | `5` | Per-job Gemini target budget (strongest first). |
| `CLIPFACTORY_GEMINI_MAX_OUTPUT_TOKENS` | `1024` | Tight structured-output budget. |

When Gemini is unavailable, misconfigured, budget-blocked, or fails, a safe
accepted local result is retained when one exists, otherwise safe Stage 2.5
text is preserved and the segment is marked unresolved/manual review. Provider
failures are never silently reported as success. Permanent 400-class request
failures, authentication, model-not-found, 429, malformed output, and safety
refusal are never retried; connection, timeout, and eligible 5xx (including
HTTP 503, surfaced as the precise sanitized `SERVICE_UNAVAILABLE` category) get
at most one bounded retry — an initial request plus one retry, so a single
generation never exceeds two attempts.

## Memory diagnostics

`scripts/diagnose-memory.sh` is a read-only host/container diagnostic that
labels host RAM, the WSL VM limit, the container cgroup limit, process and
container usage, and Ollama residency. It never edits configuration. Run it
before any heavy-model trial and after any WSL/Docker change:

```bash
./scripts/diagnose-memory.sh
```

The current machine ceiling is the WSL2 VM allocation (about 7.44 GiB with the
default 50% of a 16 GB host). Raising it requires the operator to write
`%UserProfile%\.wslconfig` with `memory=11GB` and `swap=4GB`, run
`wsl --shutdown`, restart Docker Desktop, and rerun the diagnostic; the
repository never performs that change itself. See `docs/ENVIRONMENT.md` for the
measured values and conclusion.

## Measured sequential lifecycle (2026-09-07, 10.69 GiB envelope)

Three sequential transcription-plus-reconstruction trials ran on an
operator-authorized 51.5 s source (`ca6cb88a…`) with `large-v3-turbo` (int8,
CPU) and `qwen3.5:4b` through the managed Ollama provider. The worker runs
Celery with `--pool=solo --concurrency=1 --max-tasks-per-child=1` and
`PYTHONPATH=/app`; the pre-fork pool is not used because its daemonic workers
cannot spawn the Whisper child process. A Redis-backed heavy-model lease
(`clipfactory:heavy-model`) serializes Whisper and Ollama.

Per-trial container RSS peaks (`docker stats --no-stream`, GiB):

| Trial | Whisper child peak | Whisper after child exit | Ollama peak | Ollama after unload | `ollama ps` after |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 1.97 | ≈ 0.17 | n/a (observed 0.33 idle) | 0.33 | empty |
| 2 | 2.46 | ≈ 0.26 | 5.85 | runtime empty (2.2 page cache) | empty |
| 3 | 3.36 | ≈ 0.47 | 5.93 | runtime empty (2.3 page cache) | empty |

Whisper child peak memory is measured inside the spawned child after the model
work finishes (`ru_maxrss`), never from pre-load parent RSS. If the child exits
abnormally (timeout, cancellation, or death without a result envelope) and no
trustworthy peak exists, the peak is reported as `UNKNOWN` rather than a
fabricated value.

Observations across all three trials:

- No Whisper/Ollama overlap: the lease serialized them, and the Whisper child
  exited before Ollama began loading.
- No OOM event; `MemAvailable` never fell below about 4.6 GiB.
- Swap growth per run was effectively zero (4 GiB swap, `SwapFree` stayed above
  3.99 GiB throughout).
- `ollama ps` was empty after every reconstruction; the persisted
  `unload_outcome` metadata recorded `requested=true`, `confirmed=true`, and
  sub-second elapsed time. The Ollama container's residual ≈ 2.2 GiB RSS is
  kernel page cache attributed to the container, not a resident model runtime.
- The worker process itself stayed small; the container RSS includes the
  spawned child and reclaimable page cache.

The health command and API expose provider availability, provider name, model,
and model digest only. They do not expose provider response bodies, prompts,
transcript text, credentials, or API keys. If the provider is unavailable,
misconfigured, or fails during release, reconstruction persists a truthful
status and falls back safely to earlier evidence.

## Persistent unsafe state and recovery

Unsafe model residency (an Ollama unload that is not confirmed, or a heavy-model
lease lost while a model may still be resident) is recorded in a persistent
Redis marker, `clipfactory:heavy-model:unsafe`, with no TTL. The marker survives
worker restart, CLI exit, and the lease TTL, so no new heavy-model work can
start until an operator explicitly clears it. A worker or CLI that finds the
marker raises `HeavyModelUnsafe` instead of acquiring the lease.

Acquisition and recovery are each a single Redis-side Lua script, so the unsafe
marker check, the lease acquisition, the marker clear, and the stale-lease
delete never interleave with each other:

- Acquisition first checks the unsafe marker and only then takes the lease
  with `NX` + TTL inside one atomic script; a marker written while a waiter
  retries still blocks that waiter, and a marker present before the script runs
  returns `HeavyModelUnsafe` without acquiring.
- Recovery deletes the stale lease and clears the unsafe marker in one atomic
  script, so no acquisition can slip into the gap between the two operations.
  The script refuses to run when no unsafe marker exists, so it never deletes a
  valid newly acquired lease.

Recovery clears the marker only after confirming the model is no longer
resident:

```bash
docker compose exec backend python -m app.cli recover-heavy-model
```

The command polls Ollama `/api/ps`; a resident model or an unreadable listing
keeps the marker in place and changes no Redis state. When the model is
confirmed gone, the atomic recovery script clears the marker and any stale
lease key. See `docs/ENVIRONMENT.md`.

## Lease-loss handling

If the Redis lease renewal fails or ownership is lost while Whisper work is
active, the spawned child is cancelled and reaped immediately and the lease loss
is recorded as persistent unsafe state; the stage fails closed and no other
heavy task can start until recovery. Reconstruction (Ollama HTTP calls) records
the same persistent unsafe block on ownership loss. Overlapping Whisper/Ollama
jobs cannot start after a lease expiry because the unsafe marker blocks the next
acquisition.

## Provider confidence contract

Model output is untrusted data at one strict adapter boundary. Provider
confidence is valid only when its JSON value is a non-boolean finite real number
in the inclusive range `[0.0, 1.0]`; `NaN`, infinity, overflowed exponents,
booleans, strings, and out-of-range values are rejected as a contained provider
failure. The one-pass model emits one scalar confidence, so a candidate stores
`provider_confidence` only; no fabricated acoustic, phonetic, semantic, or
margin dimensions are created. Confidence never bypasses deterministic
validation.

The provisional apply policy is `score = provider_confidence -
0.20 * raw_acoustic_confidence * edit_ratio`, with `HIGH` at `score >= 0.82` and
`phonetic_similarity >= 0.72`. The `0.82` threshold is provisional and is not
tuned in the correctness plan.

## Per-segment failure isolation

Local Qwen work runs in deterministic micro-batches bounded by
`CLIPFACTORY_RECONSTRUCTION_PROVIDER_BATCH_WINDOWS` (default 8 targets) and
`CLIPFACTORY_RECONSTRUCTION_PROVIDER_BATCH_CHARACTERS` (default 24 000 serialized
characters), in stable order with exact segment-ID mapping. Each candidate still
passes shared validation independently: one invalid candidate never rejects a
valid sibling. A malformed or failed batch degrades safely to isolated
unresolved/escalation results; there is no recursive splitting or retry storm
(one provider request per planned batch). A failure for one target (provider
error, network timeout) falls back only that segment to Stage 2.5 with
`PROVIDER_UNAVAILABLE`; earlier successful segments keep their applied
reconstruction. A provider-health failure before any call still falls back all
targets because no call can safely start. Provider release runs in one outer
`finally`; a release failure is logged but never replaces a valid result.

### Aggregate context safety at the request boundary

The window and character ceilings are secondary bounds. The real correctness
bound is the aggregate context envelope, and context-safe planning is owned by
orchestration. Each window/character micro-batch is planned **immediately before
it executes**, never eagerly for future batches, so a later target that cannot
fit context cannot abort earlier or later valid work during planning. For each
micro-batch, orchestration calls the provider's pure planning helper
(`plan_aggregate_batches`, which never executes an HTTP call) to split the
targets into actual request units: each unit is greedily sized so the exact
envelope that will be sent — the system instruction, the full combined
`{"targets": [...]}` JSON payload, the chat-framing reserve, the safety reserve,
and the scaled output budget for that unit — stays within `max_context_tokens`
(default 4096). Each unit is then one **visible** actual provider request that
orchestration schedules, polls cooperative cancellation and the local wall-time
budget around, and checkpoints after. A single target that still cannot fit
after bounded shrinking is isolated on its own: it falls back/escalates/unresolves
through the existing safe paths (Gemini escalation only when policy and budget
permit) and valid siblings are still scheduled, so one oversized target never
aborts the whole local phase; `reconstruct_segments` itself never hides
additional sequential HTTP calls. Segment order, IDs, per-target validation, and
safe malformed-batch fallback are preserved; the output-token budget scales with
the number of targets in each sent unit. If one actual request fails after
earlier ones succeeded, only that request's targets fail/escalate/unresolve — the
earlier targets' accepted candidates are already checkpointed and remain
reusable.

## Local work ceilings

Local Qwen work is hard-bounded per job so a long source can never produce
unbounded inference:

- `CLIPFACTORY_LOCAL_RECONSTRUCTION_MAX_TARGETS_PER_JOB` (default `64`) caps the
  number of local target attempts. Candidates are ranked strongest-first by
  deterministic routing severity, so the limited budget goes to the most
  uncertain targets rather than the earliest segments.
- `CLIPFACTORY_LOCAL_RECONSTRUCTION_MAX_WALL_SECONDS` (default `1200`) caps local
  reconstruction wall time.

When a ceiling is reached, scheduling of additional Qwen requests stops, the
job neither fails nor stalls, safe Stage 2.5 text is preserved, skipped targets
are marked unresolved/manual review with `local_target_budget_exhausted` or
`local_time_budget_exhausted`, the transcript still reaches a terminal state,
and no additional Gemini quota is spent to compensate. The wall-time ceiling is
enforced before each micro-batch is even planned: once it expires, no further
batch is planned or dispatched, no target is classified unfit, and no Gemini
escalation is enqueued, so a later oversized target can never spend Gemini quota
after the ceiling. `cache_eligible` is false
so a later run reconsiders eligible unresolved work (accepted per-target results
are reused, not repeated). A clean transcript may legitimately make zero Qwen
and zero Gemini calls.

## Local wall-time ceiling is authoritative over the Gemini backlog

The local wall-time ceiling also invalidates local-origin Gemini escalations
that were queued before it expired. The problematic sequence — Qwen fails before
the ceiling, the target is queued for Gemini escalation, the ceiling then
expires, and the queued escalation still calls Gemini — is closed at two points:

- **No enqueue after expiry.** A local failure/unresolved target is only added
  to the escalation backlog while the local wall-time ceiling is still active.
- **Drop queued backlog.** Immediately before each queued local-origin
  escalation executes, the ceiling is rechecked; once it has expired, every
  remaining queued escalation is invalidated (counted as
  `local_escalations_dropped`), no Gemini call is made for the backlog, the safe
  Stage 2.5/current result is preserved, and the target is marked
  unresolved/manual review with `local_time_budget_exhausted`.

Already accepted/checkpointed per-target work is never discarded, and
cancellation semantics and resource cleanup are unchanged. Direct-Gemini work
(not local-origin) is not gated by the local ceiling.

## Targeted window refinement entry point

Future Stage 3/3.5 callers request higher-quality reconstruction for an
already-selected short window through the reusable service entry point:

```python
refine_transcript_window(
    session, source_id, start_time, end_time, priority,
    reconstructor,
    max_targets=..., max_window_seconds=..., checkpoint=..., is_cancelled=...,
)
```

`priority` accepts `CANDIDATE` and `FINAL_CLIP` (and `INDEX`, which defers).
Behavior:

- Only the bounded requested source-time region is processed. Target segments
  are selected from immutable source timestamps (`start`/`end`), never from
  derived text.
- A small bounded nearby context window is included in provider prompts without
  making context segments mutation targets.
- Bounds are enforced: `0 <= start_time < end_time`, the window must not exceed
  the transcript duration, `CLIPFACTORY_RECONSTRUCTION_REFINEMENT_MAX_WINDOW_SECONDS`
  (default 300) caps the interval, and
  `CLIPFACTORY_RECONSTRUCTION_REFINEMENT_MAX_TARGETS` (default 32) caps the
  target count.
- It reuses the existing Stage 2.5 and Stage 2.7 provider, routing, validation,
  and checkpoint mechanisms; cancellation and per-target checkpoints are wired
  through when supplied.
- Raw ASR text and all timestamps are preserved exactly; manual override remains
  authoritative for `final_text`.
- It returns a structured `RefinementOutcome` (refined text where accepted,
  confidence, status, provider/routing evidence, unresolved state, and
  target/window identity) and persists results only to the selected target
  segments.
- Priority and window scope participate in output fingerprints, so a whole-source
  INDEX result can never satisfy a CANDIDATE/FINAL_CLIP request and one window
  can never satisfy another.
- The transcript-level summary stays truthful for the whole source, not just the
  window: `reconstruction_status`, `reconstruction_confidence`,
  `reconstructed_segment_ratio`, and `reconstruction_metadata.cache_eligible`
  are recomputed over every segment. A CANDIDATE refinement of a subset never
  reports the source as `APPLIED`/ratio `1.0`/cache-eligible while untouched
  segments remain unresolved, and the `index_deferred`/`index_deferred_segments`
  markers reflect only the segments still deferred. Window-specific detail
  (requested bounds, target indexes, applied/unresolved counts) is carried in the
  `RefinementOutcome.metadata`, separate from the source-wide summary.

Per-segment handoff signals future stages can consume: `refinement_priority`,
`needs_refinement` (derived from status/escalation evidence), and
`code_switch_suspected` (true only when the segment itself contains both
Arabic-script and Latin-letter protected-token evidence; numbers alone and
English-only segments are not flagged; no recovery logic).

## Ollama hardware safeguards (Compose)

The Ollama container pins safe, operator-tunable limits for the documented
~10.7 GiB WSL environment:

| Control | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_NUM_PARALLEL` | `1` | One inference at a time. |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | One loaded model. |
| `OLLAMA_MAX_QUEUE` | `4` | Bounded request queue. |
| `OLLAMA_CONTEXT_LENGTH` | `4096` | Matches the reconstruction context budget. |
| `OLLAMA_CPUS` | `6` | Compose CPU cap (protects machine responsiveness; may raise per-inference latency). |
| `OLLAMA_MEM_LIMIT` | `6g` | Compose memory cap. |
| `OLLAMA_MEMSWAP_LIMIT` | `8g` | Bounded memory+swap allowance instead of unlimited swapping. |

If the resource ceiling prevents Qwen from loading or completing, a truthful
provider failure is recorded, Stage 2.5 output is preserved, escalation happens
only within the existing Gemini policy and budget, otherwise the target is
marked unresolved/manual review, and nothing retries indefinitely. Routing,
batching, caching, and the work ceilings above provide the main wall-time
improvement; the CPU cap primarily protects machine responsiveness.

## Cancellation and retry

Cancelling a running or queued reconstruction job (`POST /jobs/{job_id}/cancel`)
is cooperative:

- A single centralized poll is checked before and after every provider attempt
  or actual request: each local Qwen actual request, each Gemini-only attempt,
  each direct-Gemini attempt, each local-to-Gemini escalation, and once
  immediately before a successful return (so cancellation landing after the
  final local request is never missed). No new Qwen or Gemini work is scheduled
  after cancellation, and the in-flight HTTP request is bounded by the
  configured provider timeout.
- Cancellation reads the exact currently executing job's status with a fresh
  scalar column query, so a `CANCELLED` state committed by the API in a
  separate session is observed by the worker on its next poll (no reliance on
  the worker's ORM identity map or session commit timing).
- A cancelled job stays `CANCELLED`; it is never overwritten as successful, and
  the next pipeline stage is never scheduled. The source stays truthful and
  retryable.
- Already-checkpointed per-target results are preserved. A retry (or a fresh
  reconstruction run) reuses accepted eligible targets without re-calling a
  provider and reconsiders only failed or eligible-unresolved targets.
- Provider and heavy-model lease cleanup still runs on cancellation, and raw ASR
  and timestamps are never altered.

### Degraded-run retry semantics

A fully cache-eligible reconstruction run may be skipped on a later normal
request (the runner matches the succeeded run's input fingerprint and the
executor confirms `cache_eligible`). A degraded run — one with
unresolved/provider-failed/rate-limited/local-ceiling targets, so
`cache_eligible` is false — is **not** skipped: `PipelineRunner` re-enters the
reconstruction executor on a later non-force request (only for the
reconstruction stage; other stages skip as before). The executor then reuses the
accepted per-target results without provider calls and retries only the eligible
unfinished targets. A safe degraded result may remain terminal for the current
attempt, but it never suppresses a later recovery attempt and never requires
`--force`.

## Prompt budgeting

`max_context_tokens` applies to the complete chat envelope: the stable system
instruction, the serialized user wrapper, a chat-framing reserve, the configured
output budget (256), and a safety reserve. The adapter uses a conservative UTF-8
estimate, then deterministically shrinks following context, previous context,
entities, and word evidence in that order, never dropping the target segment ID
or its raw/Stage 2.5 text. An irreducible request (still over budget after
shrinking) is isolated per target by orchestration — it raises a contained
planner error caught at the micro-batch boundary and falls back/escalates
without affecting valid siblings or later batches — never before any HTTP
dispatch of valid work.

## Runtime identity and fingerprints

Every output-affecting dependency participates in the reconstruction runtime
identity: provider protocol, configured model, live model digest (or an explicit
`digest_unavailable` marker, never a fake tag), one-pass prompt hash and schema
version, full context/output budget, local batch and work-ceiling settings,
routing mode/policy version and thresholds, Gemini model/schema/API version and
temperature, budgets, confidence-policy version, deterministic validation
version. Both the executor input fingerprint and the reconstruction output
fingerprint include this identity with canonical JSON ordering. The output
fingerprint additionally covers every route-relevant per-segment input: raw and
corrected text, bounds, Stage 2.5 method/confidence/applied state and stable
change digest, word probabilities, acoustic evidence, language, dialect-profile
carrier, and bounded surrounding context. A model, digest, prompt, policy,
validation, correction, or evidence change invalidates prior successful Stage 2.7
runs, and a provider-unavailable run has a distinct fingerprint that cannot
collide with an available-model run.

## Text and quality truth

Raw ASR, Stage 2.5 corrected text, Stage 2.7 reconstructed text, and manual
operator text are separate evidence. Final text priority is manual override,
then an applied high-confidence reconstruction, then Stage 2.5, then raw ASR.

`GET /api/sources/{source_id}/transcript` includes reconstruction status and
public derived metadata. `GET /api/sources/{source_id}/quality` reports audio
quality separately from transcript/reconstruction quality. The compatibility
aggregate is the lower of those scores, so clean audio cannot hide unresolved
speech evidence. The dashboard presents the same status, reasons, and bounded
routing focus spans.

## Resuming or forcing work

Pipeline reuse requires matching canonical dependency fingerprints. A changed
input reruns the affected stage and downstream derived stages; historic null or
legacy fingerprints never create a cache hit. `--force` requests execution of
the selected stage without erasing persisted cache fields.

```bash
docker compose exec backend python -m app.cli reconstruct SOURCE_ID --force
docker compose exec backend python -m app.cli retranscribe SOURCE_ID --force
```

These commands affect only application-owned derived state. They do not modify
raw text or timing evidence.

## Benchmark boundary

Readiness requires a private, authorized, human-reviewed unseen-audio manifest
under storage-owned `benchmarks/`. Do not treat synthetic fixture metrics or a
known Chernobyl diagnostic set as readiness evidence. The real runner and
unseen-audio gate remain the next implementation/verification work.

Evaluation separates four axes and reports them independently:

1. reconstruction runtime status (`APPLIED`, `LOW_CONFIDENCE_UNRESOLVED`, ...);
2. deterministic text comparison against a human reference (normalization plus
   the reviewed word-pair lexicon);
3. literal exact-string comparison (NFC equality after whitespace collapse only);
4. a validated human semantic/safety label.

Referenced unresolved rows stay in the Stage 2.5 and Stage 2.7 correctness
denominators; runtime status never removes a referenced row from the
denominator. Unreferenced rows are `unreviewed`, never implicitly safe.
Automated changed-but-wrong output is `changed_wrong`; `hallucinated` is a human
safety label. Human labels override semantic/safety counts only and never
override exact counts. Unknown human labels are rejected when loading a manifest
or review worksheet.

## Immutable ASR capture and replay

`benchmark-reconstruction` supports capturing immutable ASR once and replaying
it so reconstruction models are compared on identical raw segments:

```bash
docker compose exec backend python -m app.cli benchmark-reconstruction \
  stage-2-7/known-regression-v1.json --allow-known-regression-set --capture-asr
docker compose exec backend python -m app.cli benchmark-reconstruction \
  stage-2-7/known-regression-v1.json --allow-known-regression-set \
  --model qwen3.5:4b --from-capture <capture-id>
```

`--capture-asr` runs Whisper exactly once per clip and writes a hashed,
read-only capture under `storage/benchmarks/stage-2-7/captures/`.
`--from-capture` replays a stored capture and never constructs a transcriber.
Whisper and reconstruction runs always hold the `clipfactory:heavy-model`
lease, so models never overlap.

A replay is accepted only after full provenance verification. For every clip the
runner verifies the clip id, source id, original-media SHA-256, the exact
extracted clip audio SHA-256, the exact start/end bounds, the capture schema
version, and the decoder identity against the current transcription options. A
matching clip id alone never authorizes a replay; a stale media file, wrong
source, drifted bounds, or changed decoder identity rejects the run. Multiple
clips from the same source each keep their own clip audio hash and bounds.
