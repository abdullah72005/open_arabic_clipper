# V4 — finish the zero-model recovery continuation

**Implementation-only correction. No operational execution.**

Read [Architect review](2026-10-29-local-observation-recovery-sol-review.md) and
the [original brief](2026-10-29-local-observation-recovery-v4.md). Preserve the
reviewed admission design. Scope is only G1 and T1 below.

## Baseline and constraints

Repo `open_arabic_clipper`, branch `p5`, BASE/HEAD
`d326f426a7bf6d19c0fe47442172add706b755ee`. Existing changes are shared and
uncommitted: preserve them. No commit/push, migrations, new HTTP routes, new job
types, broad workers, provider/model calls, downloads, operational DB/Redis or
storage access, real imports/admission, manual submission, or Stage 4–6 work.
Use isolated fixtures and cached network-disabled Python 3.12 containers.
FINAL_CLIP scope, F1/B1/R3, accepted-checkpoint policy and frozen Stage 5.2 stay
unchanged. No speculative refactoring or fresh audit.

## G1: executable runtime denial, not just a preflight

Supply one small, operator-only continuation runner, with an exact invocation in
`docs/STAGE_3_5_OPERATIONS.md`. A checked-in operational script or executable
runbook harness is sufficient; this does not need a new application feature.

Use the existing normal queue/task/executor path for exactly the pinned
candidate/refinement, `FINAL_CLIP`, `force=false`, with no backlog consumer. The
existing production factory and Settings must own decoder/input identity.
Install a process-local deny-only ASR collaborator on the executor actually used
by this one job. It must raise before model work if checkpoint reuse misses;
it must never return a fake transcript or stand in for accepted evidence. Keep
the guard present through the whole invocation. Do not set ASR to `None`: the
service checks this before attempting reuse.

A process-scoped wrapper around the task's production executor factory is an
acceptable narrow mechanism. Limit it to this one-shot process and exact target;
retain the real factory's effective identity, normal ProcessingJob handling,
claim/fencing, cancellation and resource cleanup. Do not rewrite these systems.
Avoid a persisted checkpoint-only job mode or general cache framework.

Require local-only routing, local Qwen disabled and no hosted key/provider
execution; fail closed on incompatible settings. Runtime denial must remain
effective even if the earlier `--require-reuse` check passed and a later
identity/evidence mismatch makes reuse fail. The job must fail truthfully with
zero actual model/provider calls and must not proceed to manual submission.
No model work or model download may occur while installing the guard.

Keep preview/admission separate from job dispatch. `candidate-recovery --apply`
must retain its existing narrow field delta and never run the continuation.
Correct documentation/report wording: `--require-reuse` is a necessary read-only
preflight, not the runtime guard itself.

### Focused evidence

Test the same runner/guard code documented for later operator use:

1. Isolated admitted checkpoint -> preflight succeeds -> guarded normal job
   succeeds by reuse, derives approved 67-word view, retains rejected context,
   records no new ASR run and makes zero model/provider calls.
2. Successful preflight -> alter an isolated reuse dependency (for example the
   decoder identity) -> guarded normal job refuses before any model/provider
   work. Assert truthful failed job/refinement outcome, no ready/manual text
   manufactured, no retry to an unguarded execution, and normal cleanup.
3. Existing cancellation/ownership behavior is preserved on this guarded path;
   use relevant focused existing coverage plus one boundary check where needed.

Do not settle for a test-only spy that is absent from the operator invocation.

## T1: make existing negative tests reach their advertised gate

In `backend/tests/test_stage35_recovery.py`, split hash-integrity cases from
semantic cases. Hash cases keep original pins. Semantic cases mutate isolated
fixture content, re-pin only the changed synthetic file in that fixture's
expectation, and assert the intended specific refusal code plus no writes.
Do not rewrite unaffected fixture files. Include candidate/refinement/job,
decoder options, incomplete raw records, selected invalid timing, and existing
bounds/source/audio/operator cases as appropriate. Never change real bundle
pins or original archived bytes to make these tests pass.

The Architect already verified the deeper refusals; this is regression-fidelity
work, not a request to redesign validators or add a broad security suite.

## Verification, ownership and stop

One owner for the runner/runbook. T1 may run in parallel with explicit ownership
of the existing recovery test file; keep new runner tests separate if delegated.
No overlapping shared-file edits or automatic review loops.

Run new/changed guard and recovery tests, only directly affected existing job or
Stage 3.5 checks, scoped Ruff/format, `git diff --check`, and frozen renderer diff.
Do not repeat full Stage 1–6 suites or the closed scope audit.

Save `docs/superpowers/handoffs/2026-10-29-local-observation-recovery-v4-fix-r1-result.md`.
Return a copyable Architect report with changed locations, exact runner command,
both runtime reuse-hit/miss outcomes, refusal codes, actual test counts, and
limits. Explicitly confirm zero operational/model/provider work and no commit.
Stop for independent review. Do not apply the real recovery or run its job.
