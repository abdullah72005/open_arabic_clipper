# Local observation recovery — implementation result

Date: 2026-10-29
Status: **READY FOR ARCHITECT REVIEW**
Brief: [local observation recovery v4](2026-10-29-local-observation-recovery-v4.md).
Scope correction: [FINAL_CLIP scope approval](2026-10-28-final-clip-scope-sol-approval.md).
Captured run: [selected-window local recovery v4 result](2026-10-28-selected-window-local-recovery-v4-result.md).

## Outcome

Implemented one operator-only recovery service and a thin CLI command that
validates a hash-pinned local decoder observation, previews the exact checkpoint
delta, and (with explicit `--apply`) admits the raw `FINAL_CLIP` checkpoint as
`REJECTED` with complete raw observations and recovery provenance. A normal
subsequent refinement job can reuse that checkpoint without a model call and
derive the validated 67-word clip view.

No real-row import, operational DB/Redis access, ASR/Gemini/Qwen/TTS call,
provider construction, worker dispatch, migration, commit, or push was made in
this batch. The existing shared worktree changes were preserved.

## Repository state

- Branch: `p5`.
- HEAD: `d326f426a7bf6d19c0fe47442172add706b755ee`.
- The worktree already contained unrelated modified/untracked files before this
  batch; they remain untouched. This batch modified/added only the files below.
- No commit was created.

## Implementation locations

Added:

- `backend/app/refinement/recovery.py` — validation, preview, apply, snapshot,
  provenance, manifest parsing, and read-only reuse preflight.
- `backend/tests/test_stage35_recovery.py` — focused positive, negative,
  idempotency, rollback, zero-model, manual, Stage 4/caption, and CLI tests.
- `backend/tests/fixtures/final_clip_scope/prior-evidence.json` — pinned copy of
  the captured prior evidence used by the offline fixture.
- `docs/superpowers/handoffs/2026-10-29-local-observation-recovery-v4-result.md`
  — this result.

Changed:

- `backend/app/refinement/asr.py` — extracted the production raw-result
  conversion (`convert_transcription_result`) and source-evidenced hotword
  helper (`effective_transcription_options`); `TargetedASREngine` now delegates
  to the same pure helpers. No model-loading behavior changed.
- `backend/app/refinement/service.py` — exposed `seed_bounds(...)`, extracted
  `source_evidenced_context_terms(...)`, and added the read-only
  `local_observation_reuse_ready(...)` preflight. Existing execution behavior is
  unchanged.
- `backend/app/refinement/executor.py` — extracted `build_local_identity(...)`
  and reused it in the production factory.
- `backend/app/cli.py` — added `candidate-recovery` with preview default,
  explicit `--apply`, and `--require-reuse` exit guard.
- `docs/STAGE_3_5_OPERATIONS.md` — added the recovery runbook and pinned
  manifest.

## Service behavior

Validation runs before any mutation and uses the bundle path through
`StorageService` (`TEMPORARY`). It verifies:

- All three pinned bundle file hashes.
- Current source row hash, source media bytes, cached audio artifact identity and
  bytes, and exact context WAV bytes.
- Candidate/source/refinement/original-job identities, `FINAL_CLIP` priority,
  coarse/context/refined bounds, and audio path/hash.
- Operator marker and recomputed current raw-span identity.
- The current effective `FINAL_CLIP` decoder contract (all
  `TranscriptionOptions` fields) plus source-evidenced hotwords and context
  bounds.
- Complete 107-record raw response through the production conversion, including
  record 105's zero duration and the real `nonpositive word duration` reason.
- The derived clip view is `ACCEPTED`, has 67 words, the approved text SHA-256
  `6cc761f9...`, and context-only invalid record `[105]`.
- The bundle's selected words match the derived view exactly.

`--apply` revalidates under source/refinement locks, refuses active/conflicting
jobs, ready/accepted checkpoints, or differing manual text, writes a uniquely
named snapshot under
`storage/temporary/diagnostics/recovery-admissions/`, and replaces only:

- the single `TARGETED_LOCAL_ASR` evidence record (state `REJECTED`, complete raw
  observations, real reason, recovery provenance),
- `component_fingerprints.audio_extraction` and `.local_asr`,
- `provider_evidence.recovery`,
- status `NEEDS_MANUAL_TRANSCRIPT_REVIEW` and `cache_eligible=false`.

It does not set final text, executable clip words, manual text, ready status,
metrics, jobs, source lifecycle, or publishing approvals. Repeated identical
apply is a no-op. A snapshot failure prevents mutation; caller rollback preserves
the old row.

## Verification (offline, network-disabled Python 3.12 container)

Image: cached `oac-stage6-test:latest`; backend mounted read-only; isolated
SQLite/storage; no operational DB/Redis; no real model/provider call.

```bash
docker run --rm --pull never --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/app/tests \
  -v "$PWD/backend:/app:ro" \
  -v /tmp/oac-recovery-full:/audit \
  -w /app --entrypoint python oac-stage6-test:latest -m pytest \
  tests/test_stage35_recovery.py -q -p no:cacheprovider --basetemp=/audit/pytest
```

Result: **23 passed**.

Affected Stage 3.5 scope/manual identity selection:

```bash
docker run --rm --pull never --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/app/tests \
  -v "$PWD/backend:/app:ro" \
  -v /tmp/oac-recovery-affected-final:/audit \
  -w /app --entrypoint python oac-stage6-test:latest -m pytest \
  tests/test_stage35_scope.py tests/test_stage35_audio_asr.py \
  tests/test_stage35_refinement.py \
  -q -p no:cacheprovider --basetemp=/audit/pytest \
  -k 'scope_refusal or supported_manual_correction or rejected_observation or rejected_candidate_retry or raw_sequence_with_nonrecord or production_ or scope or captured_context or zero_duration or tampered or manual or asr'
```

Result: **43 passed, 21 deselected**.

Other checks:

- `ruff check` on all changed Python files: passed.
- `ruff format --check` on all changed Python files: passed.
- `git diff --check`: passed.
- `git diff --exit-code -- backend/app/render/execution/`: empty (frozen
  renderer unchanged).
- `python -m app.cli candidate-recovery --help`: command registered.

## Acceptance evidence

- Preview reports zero writes, `derived_view.state=ACCEPTED`, 67 selected words,
  approved SHA, context-only invalid `[105]`, and the exact proposed delta.
- Apply admits one `REJECTED` local record with 107 complete raw observations,
  real reason, bundle hashes, source/audio/decoder identity, and original job;
  writes a snapshot; does not change final/manual text, word timestamps, metrics,
  jobs, or source state.
- A normal production executor over the admitted checkpoint uses zero ASR calls
  (fail-on-call spy), derives the 67-word view at the refined bounds, keeps the
  context evidence rejected, and records `component_reuse` without a new
  `local_asr_runs`.
- The existing manual endpoint with the exact approved text yields
  `FINAL_TRANSCRIPT_READY` through existing rules; F1 projection is unchanged;
  Stage 4 effective input and caption payload both equal the approved text with
  no context-word leakage.
- Tampered bundle/source/audio hashes, wrong identities/bounds/segments, changed
  decoder options, incomplete raw data, zero-duration timing inside the clip,
  stale source, active jobs, conflicting checkpoints, differing manual text, and
  snapshot failure all refuse before mutation.
- Repeated apply is idempotent. Rollback preserves the old evidence/manual/jobs/
  source state.

## Proposed operator commands (not executed in this batch)

Manifest content and environment are in
[`docs/STAGE_3_5_OPERATIONS.md`](../../STAGE_3_5_OPERATIONS.md#local-observation-recovery).
Dry run:

```bash
python -m app.cli candidate-recovery --expectation /secure/local-recovery-expectation.json
```

Apply (only after independent review and authorization):

```bash
python -m app.cli candidate-recovery --expectation /secure/local-recovery-expectation.json --apply
```

Hard preflight before the later normal job:

```bash
python -m app.cli candidate-recovery --expectation /secure/local-recovery-expectation.json --require-reuse
```

Exit 0 and `reuse_ready=true` are required. If it exits non-zero, stop; do not
queue the normal refinement job. The later sequence is: admission -> one normal
guarded `FINAL_CLIP` job under the same environment -> existing manual endpoint
with the exact approved text -> readiness/currentness and Stage 4/caption checks.
That sequence is not authorized by this implementation-only batch.

## Admitted field delta

- `transcript_evidence`: replace the single `TARGETED_LOCAL_ASR` record with the
  reconstructed `REJECTED` record (107 complete raw observations, real
  `nonpositive word duration`, recovery provenance in `entity_metadata.recovery`).
- `component_fingerprints.audio_extraction` and `.local_asr`: set to the current
  audio component and reconstructed local fingerprint.
- `provider_evidence.recovery`: add the admission provenance.
- `status`: `NEEDS_MANUAL_TRANSCRIPT_REVIEW`; `cache_eligible=false`;
  `active_job_id=null`.
- Untouched: manual/final/automatic text, executable word timestamps, metrics,
  entity/unresolved evidence, routing evidence, source lifecycle, existing jobs,
  publishing approvals.

## Limitations

- The offline positive fixture keeps the captured decoder response, 107 records,
  record 105, selected words, and approved SHA, but adapts source/audio/operator
  identity to an isolated synthetic DB because the operational source transcript
  and database are not available offline. The real pinned raw-span/candidate
  identity and real source/audio bytes must be validated by the read-only preview
  against the real storage/DB before `--apply`.
- The normal executor's boundary reproduction was exercised on the isolated
  fixture, not on the real operational row. The real preview and guarded normal
  job remain the acceptance gate.
- No PostgreSQL or live-provider acceptance is claimed. No real row, provider,
  or Stage 4–6 work was run.
- The `--require-reuse` preflight is the hard zero-model-call guard for the later
  normal job. The environment/evidence must not change between preflight and job;
  if reuse misses, the normal job may call Whisper, so do not assume a retry is
  free.

## Unresolved blockers

None for the code batch. The operational admission/refinement/manual sequence
remains unauthorized until independent Architect review.
