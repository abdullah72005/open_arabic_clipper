# Local observation recovery — G1/T1 fix result

Date: 2026-10-29
Status: **READY FOR ARCHITECT REVIEW**
Brief: [G1/T1 correction](2026-10-29-local-observation-recovery-v4-fix-r1.md).
Review: [Architect review](2026-10-29-local-observation-recovery-sol-review.md).

## Outcome

G1 now has an executable, operator-only continuation runner:

```bash
python -m app.cli candidate-recovery-continue --expectation MANIFEST.json
```

It captures the normal queue dispatch, then runs the same
`run_candidate_refinement` Celery task synchronously in this process with no
broker/backlog consumer. The task's production executor factory is wrapped only
for this invocation: the real factory still builds the decoder/input identity,
and the only replaced collaborator is a deny-only ASR object that raises before
model work if accepted/rejected observation reuse misses. The runner fails
closed unless routing is `local_only`, local Qwen is disabled, and no Gemini key
is present. It does not admit evidence, submit manual text, or dispatch a normal
broker job.

T1 now separates hash-integrity regressions from semantic regressions. Hash
cases keep the original pins and assert `bundle_hash_mismatch`; semantic cases
mutate only the isolated synthetic fixture file (or the isolated expectation),
re-pin only the changed file where applicable, and assert the named refusal
code plus no writes.

No real recovery, real row import, operational DB/Redis/storage access, model or
provider call, download, migration, commit, or push was performed.

## Repository state

- Branch: `p5`.
- HEAD: `d326f426a7bf6d19c0fe47442172add706b755ee`.
- Shared uncommitted worktree preserved; unrelated files untouched.
- No commit created.

## G1 implementation

Added:

- `backend/app/refinement/continuation.py`
  - `DenyOnlyASR`: counts calls and raises `ContinuationModelCallDenied` before
    any model work.
  - `deny_only_asr_guard(...)`: process-local wrapper around
    `app.workers.tasks.build_candidate_refinement_executor`; calls the real
    factory, refuses an executor with `_asr=None`, replaces only `_asr`, and
    restores the factory in `finally`.
  - `validate_continuation_settings(...)`: fails closed on non-`local_only`
    routing, enabled local Qwen, or a configured Gemini key.
  - `prepare_continuation_job(...)`: validates the exact pinned
    candidate/refinement/source and `FINAL_CLIP`; calls the existing
    `queue_candidate_refinement` with `force=False`; captures `_dispatch` instead
    of calling Celery; verifies the captured task args are exactly the pinned
    target.
  - `execute_guarded_job(...)`: runs `run_candidate_refinement.run(*args)` under
    the guard and reports cancellation/error/denial count.
  - `run_guarded_continuation(...)`: returns a truthful result with job status,
    refinement status, metrics, denial count, cancellation and error.

Changed:

- `backend/app/cli.py`: added `candidate-recovery-continue`. Exit codes: `0`
  success, `2` configuration/environment refusal, `1` job failure/cancellation
  or execution error.
- `docs/STAGE_3_5_OPERATIONS.md`: replaced the old "run a normal job" step with
  the exact guarded command; explicitly states `--require-reuse` is a necessary
  read-only preflight, **not** the runtime guard.

The existing `candidate-recovery --apply` path is unchanged and still never
runs the continuation.

## G1 evidence

New file: `backend/tests/test_stage35_recovery_continuation.py` (7 tests).

- Reuse hit after admission: preflight succeeds, guarded job succeeds,
  `denial_calls=0`, job `SUCCEEDED`, refinement `NEEDS_MANUAL_TRANSCRIPT_REVIEW`,
  `metrics.component_reuse=1`, no `local_asr_runs`, approved 67-word view at the
  refined bounds, rejected context retained, no context leakage.
- Reuse miss after a successful preflight: changed the isolated `FINAL_CLIP`
  decoder beam from 8 to 7. The guarded job fails before model work:
  `denial_calls=1`, job `FAILED`, refinement `REFINEMENT_FAILED`,
  `error` contains `ContinuationModelCallDenied`, no manual text, unchanged
  final transcript, no retry, only the original and continuation jobs exist.
- Cancellation boundary: prepare a queued job, mark it `CANCELLED`, then run the
  guarded path. The task returns cancelled, `denial_calls=0`, job stays
  `CANCELLED`, refinement remains `NEEDS_MANUAL_TRANSCRIPT_REVIEW`.
- Environment refusals: `routing_mode_not_local_only`, `local_qwen_enabled`,
  `hosted_key_present`; no continuation job is created.
- CLI: the documented `candidate-recovery-continue` invocation exits 0 and
  reports the guarded reuse result.

## T1 changes

`backend/tests/test_stage35_recovery.py`:

- `test_hash_tampering_refuses_before_writes` (3 cases):
  `decoder_hash`, `result_hash`, `prior_hash` keep the original pins and assert
  `bundle_hash_mismatch`.
- `test_semantic_evidence_changes_refuse_at_named_gate` (12 cases):
  - `candidate_id` → `bundle_candidate_mismatch`
  - `refinement_id` → `bundle_refinement_identity`
  - `job_id` → `bundle_job_mismatch`
  - `coarse_bounds` → `candidate_bounds`
  - `context_bounds` → `bundle_refinement_bounds`
  - `source_hash` → `prior_source_mismatch`
  - `audio_hash` → `prior_audio_mismatch`
  - `decoder_options` → `decoder_contract_mismatch`
  - `incomplete_raw` → `raw_record_count`
  - `zero_inside_clip` → `derived_view_refused`
  - `candidate_bounds` → `candidate_bounds`
  - `candidate_segments` → `candidate_segments`

Candidate/refinement/job/decoder/incomplete/zero cases mutate and re-pin only
the affected synthetic bundle file. Source/audio cases mutate and re-pin only
`prior-evidence.json`. Bounds cases change only the isolated expectation. No real
bundle pin or archived byte was changed.

## Verification

Network-disabled cached Python 3.12 container (`oac-stage6-test:latest`), backend
mounted read-only, isolated SQLite/storage:

```bash
docker run --rm --pull never --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/app/tests \
  -v "$PWD/backend:/app:ro" -v /tmp/oac-recovery-combined:/audit \
  -w /app --entrypoint python oac-stage6-test:latest -m pytest \
  tests/test_stage35_recovery.py tests/test_stage35_recovery_continuation.py \
  -q -p no:cacheprovider --basetemp=/audit/pytest
```

Result: **30 passed** (23 recovery + 7 continuation).

Affected Stage 3.5 scope/manual/ASR plus queue/cancellation selection:

```bash
docker run --rm --pull never --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/app/tests \
  -v "$PWD/backend:/app:ro" -v /tmp/oac-recovery-affected-r1:/audit \
  -w /app --entrypoint python oac-stage6-test:latest -m pytest \
  tests/test_stage35_scope.py tests/test_stage35_audio_asr.py \
  tests/test_stage35_refinement.py tests/test_stage35_api.py \
  -q -p no:cacheprovider --basetemp=/audit/pytest \
  -k 'scope_refusal or supported_manual_correction or rejected_observation or rejected_candidate_retry or raw_sequence_with_nonrecord or production_ or scope or captured_context or zero_duration or tampered or manual or asr or queue or cancellation'
```

Result: **53 passed, 23 deselected**.

Other checks:

- `ruff check` and `ruff format --check` on all changed Python files: passed.
- `git diff --check`: passed.
- Frozen renderer diff (`backend/app/render/execution/`): empty.
- `python -m app.cli candidate-recovery-continue --help`: registered.

## Runtime outcomes

- Reuse hit: zero model/provider work; normal task/executor claim, persistence,
  and cleanup; approved 67-word view; rejected context preserved.
- Reuse miss: one denied ASR attempt, zero actual model/provider work; truthful
  `FAILED` job and `REFINEMENT_FAILED` refinement; no manual submission; no
  unguarded retry.
- Cancellation: queued cancellation preserved; zero denial calls.
- Incompatible environment: refused before job creation.

## Limits

- The guard is process-local to the documented runner. Manually queueing a
  normal broker job outside this command bypasses it; the runbook requires the
  runner.
- The runner invokes the Celery task synchronously via `.run`, not through a
  live worker/broker. Normal task/executor code and cancellation handling are
  exercised, but broker-level delivery/redelivery is not.
- The cancellation boundary marks the queued job cancelled before the task;
  mid-execution cancellation remains covered by existing executor tests.
- The real pinned operational identity, PostgreSQL row identity, and real
  source/audio bytes remain unverified. The real read-only preview and guarded
  continuation remain the operational gate.
- No real recovery or provider acceptance is claimed.

## Unresolved blockers

None for this correction. Real admission/continuation/manual submission remains
unauthorized until independent Architect review.
