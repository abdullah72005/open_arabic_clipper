# open_arabic_clipper — Project Memory

## Product and stage

- Product name: `open_arabic_clipper` (working product label: ClipFactory).
- Current scope: Stage 2.7 — local-first ingest/probe, cached audio,
  faster-whisper transcription, conservative contextual Egyptian correction,
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
  digest, prompt hash/schema, budgets, confidence-policy and validation
  versions), so any of those changes invalidates prior Stage 2.7 runs.
  `contextual_reconstructed_text` joins each segment's actual Stage 2.7 output;
  manual overrides change only `final_text`. Rights/provenance are tracked
  throughout the pipeline but do not block local analysis; publishing eligibility
  is evaluated separately. Stage 2.7 also supports an optional hosted Gemini
  provider through `CLIPFACTORY_RECONSTRUCTION_ROUTING_MODE`
  (`local_only`/`adaptive`/`gemini_only`, default `adaptive`). The shared request
  carries a future optional dialect/language-profile hint; Gemini candidates pass
  the same shared validation/confidence gates and a deterministic router enforces
  a finite per-job Gemini target budget. `adaptive` and `gemini_only` may send
  short transcript snippets to Google Gemini; the API key is presence-checked
  only and never logged, exposed, or committed. Unsafe heavy-model residency is a persistent
  Redis marker with no TTL that survives restart, CLI exit, and lease TTL
  expiry; `python -m app.cli recover-heavy-model` clears it only after
  confirming the model is no longer resident. A lost heavy-model lease cancels
  the active Whisper child and records unsafe state so overlapping jobs cannot
  start. Immutable ASR capture replay verifies clip id, source id,
  original-media SHA-256, exact clip audio SHA-256, exact bounds, schema, and
  decoder identity. Whisper child peak memory is measured in the child after
  model work; abnormal exits report UNKNOWN.

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
