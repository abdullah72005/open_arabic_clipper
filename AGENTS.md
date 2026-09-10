# open_arabic_clipper — Project Memory

## Product and stage

- Product name: `open_arabic_clipper` (working product label: ClipFactory).
- Current scope: Stage 2.7.1 — local-first ingest/probe, cached audio,
  faster-whisper transcription, conservative dialect-aware Arabic correction,
  bounded contextual reconstruction through a managed local provider, storage,
  jobs, dashboard, and operational tooling through `READY_FOR_ANALYSIS`.
- Explicitly out of scope until later stages: Stage 3 AI clip selection,
  advanced rendering/reframing, social publishing, and automatic authorization.
- Process only media the operator owns or is authorized to process. Never add
  DRM, login, paywall, CAPTCHA, or platform-protection circumvention.

## Technical decisions

- Runtime: Python 3.12, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL, Redis,
  Celery, and structured JSON logging.
- UI: Next.js, TypeScript, and Tailwind CSS.
- Media interfaces use FFmpeg/ffprobe through safe argument arrays; GPU use is
  optional and must never be required.
- The storage service is the sole owner of application filesystem paths.
- Pipeline stages are persisted, idempotent, retryable, and resumable. Stage 2.5
  preserves raw ASR text/timestamps and derives correction/final fields without
  realignment. Stage 2.7 preserves raw ASR text, segment timestamps, and word
  timestamps and derives reconstruction fields through a managed local provider.
  Reconstruction output carries one `provider_confidence` scalar and is validated
  at a strict provider boundary; failures are isolated per segment. Stage 2.7
  input/output fingerprints include the full runtime identity (provider, model,
  digest, prompt hash/schema, budgets, batching and local-work ceilings, routing
  mode/policy version and thresholds, Gemini schema/API version and temperature,
  confidence-policy and validation versions) plus every route-relevant segment
  input (Stage 2.5 method/confidence/applied state and change digest, word and
  acoustic evidence, bounded context), so any of those changes invalidates prior
  Stage 2.7 runs. Routing is deterministic: clean, well-covered unchanged Stage
  2.5 results and trusted Stage 2.5 repairs that resolved their uncertainty use
  `NO_LLM`; residual evidence routes normal uncertainty to Qwen (batched and
  hard-bounded per job) and clearly difficult spans to Gemini under a finite
  strongest-first budget. Aggregate context-safe local planning is owned by
  orchestration: a pure provider planner (never an HTTP call) splits each
  window/character micro-batch — planned immediately before it executes, never
  eagerly for future batches and only after a hard local wall-time ceiling
  check, so once the ceiling expires no further batch is planned, no target is
  classified unfit, and no Gemini escalation is enqueued — into visible actual
  requests whose exact
  combined chat envelope (system instruction, full `{"targets": [...]}` payload,
  framing/safety reserves, scaled output budget) never exceeds
  `max_context_tokens`, and `reconstruct_segments` executes exactly one HTTP
  call per actual request. A target that still cannot fit after bounded
  shrinking is isolated on its own (fallback/escalation/unresolved per policy)
  so it never aborts earlier or later valid local work. A degraded (not
  cache-eligible)
  reconstruction run is never skipped by the runner: it re-enters the executor
  on a later normal request and reuses accepted per-target work without
  repeating providers. Cancellation is cooperative and polled before and after
  every actual provider request and once before a successful return; it reads
  the exact executing job's status with a fresh scalar query so an API-session
  cancel is visible to the worker without a commit; it keeps the job
  `CANCELLED`, never schedules the next stage, and preserves checkpointed
  accepted per-target work for restart via per-target fingerprints. Every
  executor exit path (including a fresh cache hit) releases owned provider
  resources and scrubs the Gemini key. `contextual_reconstructed_text` joins each segment's
  actual Stage 2.7 output; manual overrides change only `final_text`.
  Rights/provenance are tracked throughout the pipeline but do not block local
  analysis; publishing eligibility is evaluated separately. Stage 2.7 also
  supports an optional hosted Gemini provider through
  `CLIPFACTORY_RECONSTRUCTION_ROUTING_MODE`
  (`local_only`/`adaptive`/`gemini_only`, default `adaptive`). The shared request
  carries a future optional dialect/language-profile hint; Gemini candidates pass
  the same shared validation/confidence gates and a deterministic router enforces
  a finite per-job Gemini target budget. Gemini availability is
  configuration-level with a lazily constructed SDK client: there are no
  per-job metadata probes, generation is deterministic (`temperature=0`, API
  version `v1`), and cache hits/all-`NO_LLM`/`LOCAL_ONLY` jobs make zero Gemini
  network calls. Cancellation is cooperative: it is checked around every
  provider batch, keeps the job `CANCELLED`, never schedules the next stage, and
  preserves checkpointed accepted per-target work for restart via per-target
  fingerprints. `adaptive` and `gemini_only` may send
  short transcript snippets to Google Gemini; the API key is presence-checked
  only, unwrapped only at lazy client construction, and never logged, exposed, or
  committed. Unsafe heavy-model residency is a persistent
  Redis marker with no TTL that survives restart, CLI exit, and lease TTL
  expiry; `python -m app.cli recover-heavy-model` clears it only after
  confirming the model is no longer resident. A lost heavy-model lease cancels
  the active Whisper child and records unsafe state so overlapping jobs cannot
  start. Immutable ASR capture replay verifies clip id, source id,
  original-media SHA-256, exact clip audio SHA-256, exact bounds, schema, and
  decoder identity. Whisper child peak memory is measured in the child after
  model work; abnormal exits report UNKNOWN.
- Transcript refinement has an explicit priority ladder: whole-source is
  `INDEX` (indexing quality), a shortlisted window is `CANDIDATE` (semantic
  quality), and a selected final clip is `FINAL_CLIP` (publication/caption
  quality). Normal whole-source ingestion always runs at INDEX: it preserves raw
  ASR, Stage 2.5, timestamps, word and acoustic evidence, and defers every
  provider reconstruction truthfully (unresolved, never a provider failure), so
  it pays for ASR + Stage 2.5 + cheap uncertainty bookkeeping only. Automatic
  local Qwen use is disabled by default (`CLIPFACTORY_LOCAL_QWEN_ENABLED=false`);
  the operator re-enables it explicitly for targeted work, and an intentionally
  selected `local_only` mode still uses Qwen and never Gemini. Gemini is
  reserved for targeted candidate/final-clip ambiguity through the existing
  bounded routing/budget gates and is never used for blanket whole-source
  cleanup. A reusable `refine_transcript_window(source_id, start_time,
  end_time, priority)` entry point refines only the bounded requested region
  (target segments selected from immutable timestamps, bounded nearby context
  that is never a mutation target), reuses the Stage 2.5/2.7 provider, routing,
  validation, and checkpoint mechanisms, preserves raw ASR/timestamps, keeps
  manual override authoritative, and returns a structured outcome (refined text
  where accepted, confidence, status, provider/routing evidence, unresolved
  state, target/window identity). Priority and window scope participate in
  output fingerprints, so an INDEX result can never satisfy a CANDIDATE/FINAL_CLIP
  request and one window can never satisfy another. The local wall-time ceiling
  is authoritative over the local-origin Gemini backlog: once it expires no new
  escalation is enqueued and queued escalations are invalidated (no Gemini call
  for the backlog, safe Stage 2.5/current text preserved, unresolved/manual
  review). Stage 2.7.1 adds dialect-aware preservation: a pure deterministic
  detector (no network, no LLM, no model loading, no audio decoding) classifies
  the source from immutable raw segment text into `EGYPTIAN`, `SAUDI`, `GULF`,
  `LEVANTINE`, `MSA`, or `UNKNOWN_ARABIC`; `None` means no Arabic evidence and
  `UNKNOWN_ARABIC` means Arabic is present but uncertain/mixed and always favors
  no change. The effective profile/confidence/evidence persist on the transcript
  and are inherited by every derived segment. The Egyptian Stage 2.5 lexicon and
  its optional provider apply only when the profile is confidently or explicitly
  EGYPTIAN; every other profile passes valid text through unchanged. Exact
  protected tokens (Latin words/names, abbreviations, technical forms, Western
  and Arabic-Indic numbers) are preserved through Stage 2.5 and every accepted
  Stage 2.7 candidate; `code_switch_suspected` flags Latin-bearing evidence
  inside an Arabic source (numbers alone and English-only sources are not
  flagged), and omitted-English audio recovery is deferred to Stage 3.5. The
  shared reconstruction instruction is dialect-neutral and preservation-first
  for both local and Gemini providers, with validated profile-specific addenda.
  An optional source-creation `dialect_profile_override` wins with confidence
  1.0 and participates in normalization fingerprints. Dialect is source
  evidence, not a target audience, and there is no deployment-wide dialect
  default.

## Local development facts (not product requirements)

- This workspace is a WSL2 checkout on a Windows-mounted drive.
- Docker Desktop/Compose, Node/npm, and internet access are available.
- Host Python is 3.10.12; use Docker's Python 3.12 runtime or install Python
  3.12 before native backend work.
- FFmpeg and ffprobe are currently absent from the host PATH; Docker images
  install them, and native setup documentation must cover installation.
- No NVIDIA/CUDA tooling was detected. Do not make GPU assumptions.

## Working agreements

- Keep implementation and tests in the monorepo boundaries documented below.
- Add tests for each feature or bugfix; run formatting, linting, and relevant
  tests before completion claims.
- Update README and operational docs whenever commands or configuration change.

# Repository Guidelines

## Project Structure & Module Organization

This repository is currently a minimal scaffold. `README.md` introduces the project and `LICENSE` contains its license. Keep implementation code in a top-level directory that matches the selected stack (for example, `src/`), with automated tests in `tests/` or colocated as `*.test.*`. Put static, non-code files in `assets/`. Update `README.md` whenever a new build tool, entry point, or required service is introduced.

Repository-local AI workflows live in `.agents/skills/`; do not edit installed skill files unless intentionally maintaining them. `skills-lock.json` records their sources and should be committed with skill changes.

## Build, Test, and Development Commands

No application runtime, package manifest, or test suite exists yet. Do not document or rely on imaginary commands. Once tooling is added, expose the standard development, lint, test, and production-build commands in `README.md` and keep this section synchronized. Useful repository checks today are:

```bash
git status --short       # show pending changes
npx skills list --json   # inspect project AI skills
```

## Coding Style & Naming Conventions

Follow the formatter, linter, and conventions of the language selected for the project; add their configuration at the repository root. Use descriptive, lowercase, hyphenated names for documentation and assets (`api-reference.md`, `logo-mark.svg`). Use the language’s conventional source-file naming and avoid unrelated refactors in focused changes. Format and lint modified files before requesting review.

## Testing Guidelines

Add tests alongside each new feature or bug fix. Name tests after observable behavior, such as `clips_selected_text_when_triggered`. Keep test inputs deterministic and avoid network-dependent tests unless they are explicitly integration tests. When a test runner is introduced, document the exact local command and any coverage threshold in `README.md`.

## Commit & Pull Request Guidelines

The current history contains only `Initial commit`, so no established convention exists. Use concise, imperative subjects: `Add clipboard parser` or `Fix empty selection handling`. Keep commits narrowly scoped. Pull requests should explain the change and test evidence, link related issues, and include screenshots or recordings for visible UI changes. Call out configuration, migration, or security implications explicitly.

## AI-Assisted Work

Superpowers and Caveman are installed locally for Codex-compatible agents. Use Superpowers skills only when task scope matches their trigger conditions; do not invoke them merely because they are installed. Use Caveman by default for agent responses, except where clarity or safety requires normal prose. Update skills with `npx skills update -y` and review the resulting diff before committing.
