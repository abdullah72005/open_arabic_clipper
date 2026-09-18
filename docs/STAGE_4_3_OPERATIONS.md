# Stage 4.3 operations: deterministic final-plan selection

Stage 4.3 commits each candidate's current Stage 4.2 governance set to exactly
zero or one current survivor. It is explicit, candidate-scoped, **synchronous**,
transaction-safe, and **provider-free**. It never replans, re-governs, rewrites,
researches, refines transcripts, or renders, and it adds **no** Celery task,
`ProcessingJob` kind, queue/executor, `PipelineStage`, `PipelineRun`, or
`_NEXT_STAGE` entry.

- Package: `app/transformation/selection/`
  (`policy.py`, `types.py`, `fingerprints.py`, `service.py`, `handoff.py`).
- Migration: `20260918_0019_stage_4_3_plan_selection`.
- Persistence: `transformation_plan_selections`.
- Versions: policy `stage4.3-v1`, schema `stage4.3-schema-v1`, fingerprint `1`.

## Authoritative input

`build_stage4_3_handoff()` in `app/transformation/governance/handoff.py` is the
authoritative Stage 4.2 -> 4.3 contract. Its freshness values are exactly
`VERIFIED_CURRENT`, `STALE`, `NOT_CURRENT`, and `UNVERIFIABLE`. Stage 4.3 also
reads the current Stage 4.1 `TransformationPlan` rows for generation rank,
planner confidence, strategy, and intensity. The handoff now truthfully reports
`stage4_3_implemented=true`; no Stage 4.2 evaluation, threshold, status,
evidence, platform interpretation, or provider behavior changed.

## Outcomes

| Status | Meaning |
| --- | --- |
| `PLAN_SELECTED` | exactly one clean approved plan committed |
| `PLAN_SELECTED_WITH_CAUTION` | exactly one allowlisted caution plan committed |
| `NO_SELECTABLE_PLAN` | current terminal results exist but none is selectable |
| `SELECTION_DEFERRED` | missing/unfinished/not-current/unverifiable/inconsistent input |
| `STALE_SELECTION_INPUT` | governance freshness is `STALE` |

All outcomes are successful semantic results once the candidate exists and the
input can be represented truthfully. Candidate-not-found remains `404`.
Verification-blocked-only, revision-required-only, rejected-only, or a completed
mix of those terminal results is `NO_SELECTABLE_PLAN`; a deferred semantic result
that could still change the answer keeps `SELECTION_DEFERRED`.

If a current governance set has internally inconsistent evidence - an approved
result with non-empty hard gates, a mismatched plan fingerprint, unresolved
essential verification, or missing decision-critical fields - Stage 4.3 fails
closed as `SELECTION_DEFERRED` with `INCONSISTENT_GOVERNANCE_EVIDENCE`. It never
repairs the evidence.

## Eligibility

A clean plan is selectable only when all are true:

- governance freshness is `VERIFIED_CURRENT`;
- the Stage 4.1 plan is current and its fingerprint matches the governed
  fingerprint;
- status is `APPROVED_FOR_SELECTION` and `eligible_for_stage4_3=true`;
- hard gates are empty;
- verification is resolved (`GROUNDED_IN_SOURCE` or `NOT_APPLICABLE`, and
  `unresolved=false`);
- semantic fidelity is strong (not failed, weak, none, or unknown);
- no essential verification dependency remains unresolved.

`APPROVED_WITH_CAUTION` is not automatically selectable merely because the broad
Stage 4.2 handoff flag is true.

## Caution allowlist

Automatic caution selection accepts only every-warning-allowlisted results with
the persisted dimension at `MODERATE`:

- `SOURCE_DOMINANCE_CONCERN` when `dimensions.source_dominance == MODERATE`;
- `TEMPLATE_MASS_PRODUCED_FEEL` when
  `dimensions.template_mass_produced_feel == MODERATE`.

Additionally: hard gates empty, verification resolved, semantic fidelity strong,
retention not damaged/mixed, coherence not mixed/incoherent, and no
unsupported-claim warning. Explicitly non-auto-selectable cautions are
`SEMANTIC_FIDELITY_CONCERN`, `UNSUPPORTED_CRITICAL_CLAIM`,
`SOURCE_MOMENT_SEVERELY_DAMAGED`, `PLAN_INCOHERENT`, and any unknown/future
warning code. All warnings are preserved even when an allowlisted caution wins.
If at least one valid clean approval exists, the clean pool arbitrates first;
cautions are considered only when no valid clean approval exists.

## Deterministic hierarchy

No weighted aggregate score, "viral score", or opaque FinalPlanScore is
computed or persisted. Arbitration is a readable lexicographic comparison
(lower is better on every element):

1. semantic fidelity;
2. retention preservation and lower source-moment damage;
3. substantive originality/value;
4. lower platform/source-dominance risk (worst YouTube/Facebook level first,
   then the count of `HIGH`/`MODERATE` findings);
5. plan coherence;
6. lower generic filler and redundant-commentary risk;
7. lower unnecessary narration burden;
8. lower transformation proportionality/intensity when substantive value is not
   worse (least intrusive sufficient transformation);
9. lower Stage 4.1 generation rank, then higher planner confidence;
10. stable plan identity.

Stage 4.2 hard gates dominate; planner rank/confidence never override Stage 4.2
evidence or eligibility. Explainability persists the approval tier, eligible and
excluded plan IDs, ordered comparison dimensions, the first material distinction
(including why a more intrusive plan lost when it added no greater substantive
value), whether the stable identity tie-break was reached, and per-alternative
dispositions.

## Fingerprints and cache

The selection input fingerprint canonically includes candidate/source identity
and currentness, Stage 4.0 identity/output, Stage 4.1 plan-set identity and
fingerprints plus target semantic context, Stage 4.2 set identity, fingerprints,
governor/validation/schema and platform-policy profile versions, planning
refinement identity/priority/quality/output fingerprint, every current plan's ID,
fingerprint, rank, confidence, strategy and intensity, every governance result's
ID/status/eligibility/hard gates/dimensions/verification/platform risk/warnings/
reason codes/output fingerprint, and the Stage 4.3 policy version and caution
policy.

It deliberately excludes the Gemini key/availability, Qwen/Ollama availability,
TTS provider/model/voice/speaker identity, channel voice configuration, caption
font, render resolution/configuration, B-roll choice, and publishing
title/description/schedule/metadata. A changed Stage 4.2 fingerprint, Stage 4.1
plan fingerprint, verification state, warning, governor policy/profile, relevant
target context, or Stage 4.3 policy version produces a new current selection
result and marks the prior row historical; TTS voice/provider/model, render
configuration, and publishing configuration changes never invalidate selection.

Repeated identical `POST`s return the existing row with no duplicate provider
work. A read-only `GET` never mutates rows and recomputes effective freshness
truthfully.

## Idempotency and concurrency

- `transformation_plan_selections` stores one row per
  `(clip_candidate_id, input_fingerprint)` with a unique constraint.
- A PostgreSQL/SQLite partial unique index allows at most one
  `is_current=true` row per candidate.
- A `POST` locks the candidate row (`SELECT ... FOR UPDATE` on PostgreSQL),
  revalidates freshness after the lock, reuses an exact-fingerprint row when
  present, otherwise demotes the prior current row and inserts the new one,
  recovering from a concurrent insert via a savepoint and re-query.
- Check constraints enforce: selected statuses require non-null selected
  plan/result IDs; non-selected statuses require null; the selected pair is
  all-or-nothing; and `selected_with_caution` is true only for
  `PLAN_SELECTED_WITH_CAUTION`.
- No job platform or fencing is needed because the work is synchronous and
  bounded to at most three persisted plans.

## Selection versus execution readiness

Selection and execution readiness are separate. Selection may be made from
`CANDIDATE`-grade Stage 3.5 evidence and never enqueues refinement. The live
readiness enum is:

- `READY_FOR_EXECUTION_PREP`: the selected plan is based on the same current
  usable `FINAL_CLIP` (`FINAL_TRANSCRIPT_READY`, non-empty final transcript) with
  the same refinement ID **and** output fingerprint as the stored planning
  evidence, and the selection is still effective.
- `READY_FOR_FINAL_REFINEMENT`: the selected plan is based on `CANDIDATE`
  evidence with no usable `FINAL_CLIP`.
- `REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK`: a newer usable `FINAL_CLIP`
  exists after `CANDIDATE`-grade planning, or the same `(candidate, priority)`
  row's output fingerprint changed since planning (Stage 3.5 updates rows in
  place), so a later execution boundary must fail closed if it materially
  contradicts planning evidence. This is reported even when the new `FINAL_CLIP`
  truthfully invalidates the frozen upstream fingerprint chain.
- `BLOCKED`: no selected plan, stale/unverifiable governance, stale selection,
  unresolved verification, or inconsistent evidence.

The planning refinement output fingerprint used by the selected plan/governance
is preserved on the selection row and exposed truthfully in the execution
handoff; readiness never compares row identity alone.
`final_clip_refinement_required` is true only when no usable `FINAL_CLIP`
refinement exists. When a usable `FINAL_CLIP` exists but differs from planning
evidence, refinement is already complete:
`final_clip_refinement_available=true`, `final_clip_refinement_required=false`,
`compatibility_recheck_required=true`, and readiness is
`REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK`.

A newly created `FINAL_CLIP` that makes the frozen Stage 4.0/4.1/4.2 input chain
stale is reported truthfully (`upstream_chain_stale=true`); Stage 4.3 never
reruns those stages and never erases the historical selected-plan snapshot.

## TTS/voice separation

Narration remains semantic only. The execution handoff contains no TTS provider,
model, voice ID, random voice, or speaker identity, and never claims platform
safety or monetization (`SAFE_FOR_YOUTUBE`, `WILL_BE_MONETIZED`, and similar are
never emitted or persisted). Voice/provider/model changes cannot invalidate a
selection.

## API and CLI

- `POST /api/candidates/{candidate_id}/transformation-selection`
- `GET /api/candidates/{candidate_id}/transformation-selection`
- `GET /api/transformation-selections/{selection_id}`
- `GET /api/candidates/{candidate_id}/execution-handoff`
- `python -m app.cli transformation-selection CANDIDATE_ID`
- `python -m app.cli transformation-selection-handoff CANDIDATE_ID`

`POST` / `transformation-selection` performs or reuses selection synchronously
and returns the durable result; no fake queued job is returned. `GET` by
candidate returns the latest effective selection record with live freshness;
`GET` by ID can return historical rows while identifying whether they are still
effective.

## Execution handoff

The read-only handoff exposes: selection ID/status/live freshness/current or
historical state/reason codes; candidate/source identity; separate rights/
provenance risk and platform originality/spam risk; Stage 4.0/4.1/4.2 identities
and fingerprints; the selected plan ID or null; strategy and intensity; exact
unchanged ordered blocks; hero span and appearance time; preservation
constraints; original-value kinds/reasons; abstract narration need and
requirements; target language/register/audience context; source dialect; exact
verification state and dependencies; duration/ratio evidence; Stage 4.2 hard
gates, dimensions, reason codes, warnings, remediation and provider evidence;
YouTube/Facebook risk snapshots with policy profile/version; planning-refinement
identity; current `FINAL_CLIP` availability/identity; final-refinement and
compatibility requirements; readiness; fingerprints; and
`stage5_implemented=false` / `stage6_tts_implemented=false`.

If there is no selected plan, the handoff returns the selection outcome and
reasons with `selected_plan=null`; it never fabricates blocks or readiness.

## PostgreSQL validation

SQLite cannot meaningfully validate the partial unique index or row locking, and
Alembic DDL must be proven on the real engine. Run the gated PostgreSQL
migration and concurrency tests against the repository's compose PostgreSQL:

```bash
docker run --rm --network oac_default \
  -v "$PWD/backend":/app -w /app \
  -e CLIPFACTORY_TEST_POSTGRES_URL='postgresql+psycopg://clipfactory:clipfactory@postgres:5432/clipfactory' \
  <backend-test-image> python -m pytest \
    tests/test_stage43_concurrency.py \
    tests/test_stage43_migration.py::test_stage_4_3_postgresql_alembic_upgrade -q
```

The migration test creates a throwaway database, runs the real Alembic chain to
head (including the frozen Stage 4.0/4.1/4.2 revisions and the Stage 4.3
revision), verifies the selection constraints and index names on PostgreSQL,
downgrades to `20260917_0018`, upgrades again, and drops the database. It uses no
DDL rewriting and alters or removes no production constraint. Boolean columns use
portable `true`/`false` predicates (and PostgreSQL-safe identifier lengths)
rather than SQLite-style integer comparisons.

## Explicit exclusions

Stage 4.3 does not implement or modify Stage 4.0 eligibility, Stage 4.1
generation, Stage 4.2 governance logic/thresholds, plan rewriting or automatic
repair, Gemini/Qwen selection, factual research/fact-checking, automatic
`FINAL_CLIP` queueing, Stage 5 rendering (FFmpeg cuts, captions, reframing, audio
normalization, face tracking), Stage 6 narration execution/TTS, voice/provider/
model/channel selection, B-roll/gameplay, publishing/scheduling/metadata, Stage 7
diversity, source re-ingestion, or Whisper tuning. It configures or probes no
provider availability and behaves identically whether Gemini, Qwen, or Ollama is
available.
