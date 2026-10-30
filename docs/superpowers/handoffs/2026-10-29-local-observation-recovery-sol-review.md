# Architect review — local observation recovery

**Disposition: CHANGES REQUIRED.** Admission and evidence reconstruction pass the
focused review; the required zero-model continuation is not yet supplied. This
is a narrow follow-up, not a reopening of FINAL_CLIP scope, F1/B1 or R3.

## Baseline and reviewed scope

- Branch `p5`; BASE and live HEAD both
  `d326f426a7bf6d19c0fe47442172add706b755ee`.
- Shared modified/untracked worktree; no commit or push. Existing unrelated work
  preserved. Review includes untracked recovery code/tests, not just HEAD diff.
- Contract: [implementation brief](2026-10-29-local-observation-recovery-v4.md).
- Implementer evidence: [result](2026-10-29-local-observation-recovery-v4-result.md).
- Inspected recovery validation/admission, production conversion and identity
  helpers, CLI, continuation runbook, and relevant tests. Independent delegated
  provenance/decoder review found no admission bypass in its assigned scope.

## Passing requirements

The service verifies pinned files, current source/audio bytes, operator candidate
and raw-span identity, original job, bounds, decoder options and selected view.
It reconstructs the complete rejected context through production conversion.
Preview does not mutate; apply locks and revalidates, snapshots before mutation,
admits raw evidence only, and leaves final/manual/executable caption fields alone.
The isolated normal executor reuses the checkpoint and derives the 67-word view;
the existing manual endpoint then supplies readiness and caption/planning parity.
No new schema, HTTP API, renderer changes or provider execution were found in
this batch.

## G1 — P1: required runtime guard missing from the continuation

**Owner:** operator continuation/runbook, not the scope or projection algorithm.

**Locations:** `docs/STAGE_3_5_OPERATIONS.md:472–480`,
`backend/app/cli.py:545–598`, and
`backend/app/refinement/service.py:989–1021`.

`--require-reuse` validates the current checkpoint and exits. It neither installs
a guard on the later job nor constrains that job's ASR engine. The documented
next step runs a normal factory-built executor; if its accepted/rejected reuse
checks miss, `_local_asr` calls `self._asr.transcribe`. The runbook explicitly
acknowledges that changed evidence/settings can call Whisper. This fails item 8
of the approved brief: stop before model work on a reuse miss. It could consume
an unauthorized additional local model pass. No actual model call was made to
demonstrate this; the missing boundary is established by the executable path.

**Smallest correction:** supply and offline-test an executable one-shot
continuation harness that installs a deny-only ASR collaborator on the actual
normal job execution path. Keep production decoder identity and all normal job
claim, failure/cancellation and cleanup behavior. The guard must raise before
model work, never return invented evidence. Preserve local-only/no-Qwen/no-key
constraints. No general retry/recovery platform or new persisted job mode.

**Regression:** same guarded runner succeeds through reuse with zero model
calls; a reuse miss after a successful preflight fails the job before model work
and before manual submission. A preflight-only test does not prove this gate.

## T1 — P2: negative regressions stop before their named semantic gates

**Owner:** `backend/tests/test_stage35_recovery.py:833–910`.

The parameterized test rewrites bundle JSON while keeping its old pinned hash
and asserts only `LocalObservationRecoveryError`. Candidate/refinement/job,
decoder, incomplete-record and selected-invalid-timing cases therefore stop at
`bundle_hash_mismatch`, not the semantic condition advertised by the test/report.
The implementation is not shown defective by this test issue.

Independent network-disabled probes re-pinned only their isolated synthetic test
files, then reached these actual refusal codes:

| Case | Refusal beyond the file hash |
| --- | --- |
| Candidate identity | `bundle_candidate_mismatch` |
| Refinement identity | `bundle_refinement_identity` |
| Original job | `bundle_job_mismatch` |
| Decoder beam | `decoder_contract_mismatch` |
| Missing raw record | `raw_record_count` |
| Zero-duration record inside selected clip | `derived_view_refused` |

Thus current deeper validation works in these probes. Correct the permanent
tests in the same bounded follow-up: retain original hashes for hash-tampering
cases; for semantic tests update only isolated fixture pins and assert the exact
intended code. No production pins or archived evidence may be changed.

## Fresh verification and limits

Architect reran, using cached Python 3.12 and `--network none`:

```bash
docker run --rm --pull never --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app:/app/tests \
  -v "$PWD/backend:/app:ro" -v /tmp/oac-recovery-sol-review:/audit \
  -w /app --entrypoint python oac-stage6-test:latest -m pytest \
  tests/test_stage35_recovery.py -q -p no:cacheprovider \
  --basetemp=/audit/pytest --junitxml=/audit/review.xml
```

JUnit: **23 tests, 0 failures, 0 errors, 0 skipped; 75.027 seconds**.
Independent semantic probe: `/tmp/oac-recovery-validation-probe.py`, with results
above. `git diff --check` passed and the frozen renderer diff was empty.
V4's 43 affected tests and Ruff/format results remain implementer evidence;
the Architect did not repeat that full selection.

SQLite/synthetic identity adaptation is a known limitation. Real pinned storage,
PostgreSQL row identity, and normal boundary reproduction remain unverified.
No operational DB/Redis access, real recovery, ASR, hosted call, migration,
commit or push occurred during this review. Caption wording approval does not
accept a transformed Stage 6 video. Stage 6 stays unfrozen; Stage 7 stays gated.

## Next action and stop

Execute only [G1/T1 follow-up](2026-10-29-local-observation-recovery-v4-fix-r1.md).
After focused independent verification, close this implementation scope and
prepare the concrete authorized operational sequence. This review does not
authorize real admission/refinement/manual submission or provider spending.
