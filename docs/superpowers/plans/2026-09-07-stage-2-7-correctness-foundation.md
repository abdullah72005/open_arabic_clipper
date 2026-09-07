# Stage 2.7 Correctness Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make provider failure handling, confidence decisions, persistence, fingerprints, and benchmark reporting trustworthy before any more model comparisons.

**Architecture:** Treat model output as untrusted data at one strict adapter boundary. Preserve successful per-segment work when another call fails. Represent the one scalar the model actually returns, keep deterministic safety validation independent, persist the selected reconstruction exactly, and evaluate runtime, literal equality, deterministic equivalence, and human safety on separate axes.

**Tech Stack:** Python 3.12, dataclasses, Pydantic, FastAPI application services, SQLAlchemy 2, pytest, Ruff, mypy, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-07-stage-2-7-accuracy-runtime-recovery-design.md`

## Global Constraints

- Execute only this plan. Do not start memory work or download/run models.
- Use test-driven development: add one focused failing test, run it and see the expected failure, implement the minimum fix, then rerun it.
- Preserve raw ASR text, timestamps, word evidence, Stage 2.5 text, manual overrides, and final-text priority.
- Do not change the provisional `0.82` apply policy, add phrase-specific fixes, expand a dialect lexicon, or hard-code known benchmark phrases.
- Do not weaken protected-token, entity, length, or phonetic validation.
- Run Python commands in the Docker Python 3.12 environment. Route long shell output through lean-ctx.
- After every task, inspect `git diff --check` and commit only that coherent task. Stop immediately if an existing unrelated user change overlaps a target hunk.

---

## Task 1: Make malformed provider output a contained provider failure

**Files:**

- Modify: `backend/app/transcription/reconstruction/providers.py`
- Test: `backend/tests/test_reconstruction_provider.py`
- Test: `backend/tests/test_reconstruction_service.py`

- [ ] Add parameterized provider tests for response content containing: no JSON object, truncated JSON, missing target result, duplicate target result, string confidence, boolean confidence, `-0.01`, `1.01`, `NaN`, `Infinity`, and an overflow exponent. Assert every case raises `ProviderResponseError`, never bare `ValueError`, `TypeError`, or `KeyError`.
- [ ] Run:

  ```bash
  docker compose run --rm --no-deps backend pytest -q tests/test_reconstruction_provider.py
  ```

  Expected: the new cases fail against the current permissive parser.
- [ ] Add one service test whose fake provider raises `ProviderResponseError` and assert the affected segment falls back to Stage 2.5 with `PROVIDER_UNAVAILABLE` while the call returns normally.
- [ ] In `_extract_json_object`, translate the exhausted scan to `ProviderResponseError` while preserving the original cause. In `_parse_reconstructions`, accept confidence only when `isinstance(value, (int, float))`, `not isinstance(value, bool)`, `math.isfinite(float(value))`, and `0.0 <= value <= 1.0`.
- [ ] Validate response collection shape and exact requested-ID coverage before creating candidates. Reject duplicate, missing, or unexpected IDs.
- [ ] Rerun both focused test files and confirm pass.
- [ ] Commit: `Fix reconstruction provider response validation`

## Task 2: Represent one-pass confidence honestly without changing policy

**Files:**

- Modify: `backend/app/transcription/reconstruction/types.py`
- Modify: `backend/app/transcription/reconstruction/confidence.py`
- Modify: `backend/app/transcription/reconstruction/providers.py`
- Modify: `backend/app/transcription/reconstruction/service.py`
- Test: `backend/tests/test_reconstruction_confidence.py`
- Test: `backend/tests/test_reconstruction_service.py`
- Test: `backend/tests/test_reconstruction_provider.py`

- [ ] Add tests proving the parsed candidate contains one `provider_confidence` value and no fabricated acoustic/phonetic/semantic/margin fields.
- [ ] Add decision tests for the current effective boundaries: valid candidate at `0.82` is HIGH/APPLY; immediately below `0.82` is not applied; deterministic validation failure is never applied regardless of confidence; unchanged text remains unchanged.
- [ ] Run the three focused files and record the expected structural failures.
- [ ] Replace `ReconstructionCandidate.scores: ResolutionScores` with `provider_confidence: float`. Keep `ResolutionScores` only if another real subsystem still produces independent inputs; otherwise delete it and update imports/tests.
- [ ] Change `ReconstructionDecision` to expose `provider_confidence`, `level`, `apply`, and a stable `reason`. Implement the one-scalar decision explicitly rather than passing the same number through weighted score and margin formulas.
- [ ] Preserve the effective HIGH boundary of `0.82`; do not tune MID/LOW behavior in this task. Give the policy a constant version string such as `one-pass-provider-confidence-v1` for fingerprinting.
- [ ] Rerun focused tests and then:

  ```bash
  docker compose run --rm --no-deps backend pytest -q tests/test_reconstruction_*.py
  ```

- [ ] Commit: `Model one-pass reconstruction confidence honestly`

## Task 3: Isolate failures per target and preserve earlier successes

**Files:**

- Modify: `backend/app/transcription/reconstruction/service.py`
- Test: `backend/tests/test_reconstruction_service.py`

- [ ] Add a three-target test with a scripted provider: target 1 returns an accepted repair, target 2 raises `ProviderResponseError`, target 3 returns unchanged. Assert target 1 stays applied, target 2 alone falls back as unavailable, and target 3 is processed.
- [ ] Add the equivalent test for an `OSError`/timeout on target 2.
- [ ] Run the focused tests and confirm the current batch-wide exception scope loses valid work.
- [ ] Move recoverable exception handling inside the per-request loop. Keep an initial provider-health failure as a full fallback because no target call can safely start.
- [ ] Put `provider.release()` in one outer `finally` and ensure a release failure is logged but cannot replace a successful reconstruction result.
- [ ] Rerun the focused test file and reconstruction regression tests.
- [ ] Commit: `Isolate reconstruction failures by segment`

## Task 4: Budget the complete chat envelope

**Files:**

- Modify: `backend/app/core/settings.py`
- Modify: `backend/app/transcription/reconstruction/providers.py`
- Modify: `backend/app/transcription/reconstruction/types.py`
- Test: `backend/tests/test_reconstruction_provider.py`
- Test: `backend/tests/test_settings.py`

- [ ] Add tests showing `estimated_tokens` includes the stable system instruction, serialized user wrapper, chat framing reserve, configured output budget, and safety reserve—not only the payload.
- [ ] Add a test that deterministic shrinking removes following context, previous context, entities, then word evidence in the documented order while retaining target ID/raw/Stage 2.5 text.
- [ ] Add a test that an irreducible request raises `ProviderResponseError` before the HTTP transport is called.
- [ ] Introduce named settings for output-token budget and reserves with positive bounded validation. Keep the current output budget at 256.
- [ ] Store request-size measurements on the provider result or a small diagnostics object so the benchmark can report serialized bytes and estimated input tokens without parsing logs.
- [ ] Rerun provider/settings tests.
- [ ] Commit: `Budget complete reconstruction prompts`

## Task 5: Separate exact, deterministic, runtime, and human evaluation

**Files:**

- Modify: `backend/app/transcription/reconstruction/benchmark.py`
- Modify: `backend/app/transcription/fixtures/egyptian_ar_reconstruction.json`
- Test: `backend/tests/test_reconstruction_audio_benchmark.py`

- [ ] Add literal exact tests proving that alef variants, `ة/ه`, `ى/ي`, punctuation, and diacritics are not exact matches. Permit only Unicode NFC normalization and the explicitly documented outer/duplicate-whitespace rule.
- [ ] Add counterexample tests proving deterministic equivalence rejects `كتاب/كتابي`, `حسن/حسني`, `ثامر/تامر`, `عمل/يعمل`, changed numbers, and changed Latin tokens.
- [ ] Replace generic edge-edit and dental-shift rules with a small reviewed pair set stored as data, bidirectional and exact-word only. Include only already justified dialect orthography pairs; do not add benchmark phrases as repairs.
- [ ] Add aggregation tests where semantic human labels differ from exact status. Assert human labels affect semantic/safety counts only and exact counts always derive from exact status.
- [ ] Add a referenced unresolved row and assert it stays in the Stage 2.5 and Stage 2.7 denominators. Add an unreferenced row and assert it is reported as `unreviewed`, never implicitly safe.
- [ ] Rename changed-but-wrong automated output to `changed_wrong`. Reserve `hallucinated` for a validated human safety label. Reject unknown human labels when loading the manifest/worksheet.
- [ ] Rerun the benchmark tests and inspect one generated synthetic report for internally consistent totals.
- [ ] Commit: `Make reconstruction evaluation auditable`

## Task 6: Invalidate stale runs and persist the actual reconstruction

**Files:**

- Modify: `backend/app/transcription/reconstruction/providers.py`
- Modify: `backend/app/transcription/reconstruction/service.py`
- Modify: `backend/app/pipeline/stages.py`
- Modify: `backend/app/pipeline/fingerprints.py`
- Test: `backend/tests/test_pipeline_fingerprints.py`
- Test: `backend/tests/test_reconstruction_persistence.py`
- Test: `backend/tests/test_stage2_pipeline_e2e.py`

- [ ] Define `ReconstructionProvider.runtime_identity()` and add fake-provider tests for a stable dictionary containing provider protocol, configured model ID, optional live digest, prompt hash, schema version, full context/output budget, confidence-policy version, deterministic-validation version, and correction fingerprint.
- [ ] Add fingerprint tests that independently change model ID, digest, prompt version, confidence-policy version, validation version, or context setting and assert a different Stage 2.7 input fingerprint.
- [ ] Ensure provider unavailability has a distinct identity and cannot collide with a prior available model run.
- [ ] Add persistence tests where raw, Stage 2.5, Stage 2.7, manual, and final text are deliberately different. Assert `contextual_reconstructed_text` joins the Stage 2.7 values and manual text changes only final text.
- [ ] Implement runtime identity in the OpenAI-compatible provider. If the live digest endpoint is unavailable, record an explicit `digest_unavailable` marker; do not pretend the configured tag is a digest.
- [ ] Include runtime identity in both executor input fingerprint and reconstruction output fingerprint using canonical JSON ordering.
- [ ] Fix transcript aggregation in `ContextualReconstructionExecutor._apply` to use persisted reconstruction output.
- [ ] Rerun all three focused files.
- [ ] Commit: `Fingerprint and persist Stage 2.7 outputs correctly`

## Task 7: Correctness checkpoint and documentation

**Files:**

- Modify: `docs/BENCHMARKS.md`
- Modify: `docs/STAGE_2_7_OPERATIONS.md`
- Modify: `STATUS.md`
- Modify: `AGENTS.md` only if the public persistence/fingerprint contract changed materially

- [ ] Update docs to mark pre-fix reports non-comparable. State that the old semantic and exact aggregates are invalid evidence and retain their run IDs only as historical artifacts.
- [ ] Document the four evaluation axes, denominator rules, provider confidence contract, and per-segment fallback.
- [ ] Run formatting, lint, scoped typing, and backend reconstruction/pipeline tests:

  ```bash
  docker compose run --rm --no-deps backend sh -lc \
    "pytest -q tests/test_reconstruction_*.py tests/test_pipeline_fingerprints.py tests/test_stage2_pipeline_e2e.py && \
     ruff format --check app tests && ruff check app tests && \
     mypy app/transcription/reconstruction app/pipeline/stages.py"
  ```

- [ ] Run `git diff --check` and `git status --short`. Record exact test counts and any pre-existing failures; do not describe an unrun suite as passing.
- [ ] Commit: `Document Stage 2.7 correctness foundation`
- [ ] Stop and hand the diff, commits, and verification output to a reviewer. Do not start the memory plan until this checkpoint is accepted.

