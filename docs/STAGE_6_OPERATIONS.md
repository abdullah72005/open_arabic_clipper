# Stage 6 transformed-short materialization operations

Status: **IN PROGRESS.** The durable persistence, the deterministic
materialization core, and the candidate-scoped queue/executor/worker/API/CLI are
implemented and tested (the executor assembles inputs through genuine persisted
ORM relationships and persists a fenced current row). The Stage 5.2
`TRANSFORMED_SHORT` authored renderer, the idempotent finalizer, the semantic/TTS
provider adapters, the review UI integration, and real manual acceptance are not
yet implemented. Stage 6 remains awaiting human media review and SOL review; it
is not frozen.

## Objective and ownership

Stage 6 builds one genuine transformed Short from one explicitly selected
candidate:

selected Stage 4 plan + usable FINAL_CLIP evidence + current Stage 5.0 execution
contract + current Stage 5.1 source-local visual intelligence + materialized
original value + optional narration/supporting assets + final authored
timeline/audio recipe → reusable Stage 5.2 renderer → technical QC and Stage 6
assessment → human-reviewable transformed artifact.

Runtime ownership is fixed: Stage 5.1 is source-local visual intelligence; Stage 6
owns authored content, timeline, and editing decisions; Stage 5.2 is deterministic
execution, mixing/mastering execution, and encoding/QC. The renderer never writes
scripts, selects strategies, calls semantic providers, decides originality, or
chooses retention edits. Stage 6 renders from the original source plus accepted
lossless authored assets; `CORE_SOURCE_VALIDATION` is optional diagnostics, never
an intermediate prerequisite.

## Implemented components

- Persistence (`alembic` migration `20261014_0023`): a `MATERIALIZATION` job kind,
  the `TRANSFORMED_SHORT` render purpose, the `channel_production_profiles` and
  `clip_materializations` tables, and nullable `materialization_id` FKs on
  `processing_jobs` and `render_executions`. No `PipelineStage`, `PipelineRun`,
  `_NEXT_STAGE` entry, or source-lifecycle change is introduced.
- `backend/app/materialization/policy.py`: versioned policy (`stage6-v1`),
  validation (`stage6-validation-v1`), timeline schema, quality policy, and the
  hard budgets (≤4 raw semantic calls including ≤2 strong-tier, ≤3 narration
  chunks, ≤60 s narration, ≤6 raw synthesis calls, ≤180 s clip, one repair,
  conservative boundary-trim limits, `-16 LUFS` / `-1.5 dBTP` / 48 kHz mastering
  target).
- `types.py`, `inputs.py`, `fingerprints.py`: immutable value objects,
  DB-independent inputs, and stable fingerprints that exclude secrets, quotas,
  attempts, absolute paths, and timestamps. A voice-only change never invalidates
  no-narration output; the narration fingerprint includes the voice identity only
  when narration is actually used.
- `profiles.py`: channel-profile semantic-context matching against the selected
  plan's target context. A mismatch is `UPSTREAM_REVALIDATION_REQUIRED`; voice or
  provider identity is never part of semantic matching.
- `scripting.py`: deterministic authored-copy validation (paraphrase-only,
  generic filler, presentation-only, invented numbers, protected-token mutation),
  a claim ledger (source-supported / source-grounded inference / external
  requires verification), and explicit required/optional slot resolution.
- `timeline.py`: immutable authored timeline and final audio recipe at unit
  source speed, with cumulative boundary quantization, hero protection (no trim,
  narration interruption, source replacement, or obscuring overlay), conservative
  optional non-hero boundary trimming only with agreeing acoustic + word
  evidence, and no competing speech overlap by default.
- `supporting.py`: a strict versioned supporting-asset manifest, rights gating
  (only explicit compatible rights permit automatic use), deterministic AUTO
  selection, and explicit declined reasons. OFF by default; zero originality
  credit; no provider calls.
- `narration.py`: a provider-neutral synthesis interface returning managed
  lossless audio with actual format/duration/hash and request identity, single
  WAV wrapping (never double-wrapping an existing WAV), deterministic WAV
  measurement (empty/silent/truncated/clipped), bounded request building, and a
  deterministic ASR-vs-script comparison.
- `acceptance.py`: categorical quality dimensions with no aggregate score and a
  truthful outcome mapping (`READY_FOR_HUMAN_REVIEW`,
  `MATERIALIZED_WITH_CAUTION`, `NO_OUTPUT`, `DEFERRED_PROVIDER`, `TTS_UNAVAILABLE`,
  `BLOCKED_VERIFICATION`, `REJECTED_QUALITY`, `UPSTREAM_REVALIDATION_REQUIRED`,
  `NEEDS_MANUAL_TRANSCRIPT_REVIEW`, `FAILED`). Successful results are always
  `publication_ready=false` with `human_review_required=true`.

## Channel production profile

A `channel_production_profiles` row is minimal local channel configuration (not
social-account management). It carries a stable `channel_key` and `revision`
(one database-current row per `channel_key`), the semantic context
(`target_market`, `output_language_policy`, `register_intent`,
`narration_allowed`), the persistent TTS provider/model/API/voice/style/settings
(never a random voice per clip), and supporting-visual defaults. Secrets stay in
settings and are never stored on the profile. Neutral defaults are
`UNSPECIFIED` / `SOURCE_LANGUAGE` / `SOURCE_COMPATIBLE`.

## Prerequisite follow-ups (completed)

- Production transcription configuration defaulted to `v1` and overrode the
  provider's `v1beta` default, so the effective Interactions request used the
  wrong surface. The Settings default and the documented/example configuration
  now use `v1beta` (the surface that serves the Interactions API), with a
  regression through the production Settings factory and the real installed SDK
  with a fake HTTP transport. Adjudication and the Stage 2.7/Stage 4 providers
  keep their independently configured `v1` generateContent surface.
- Stage 4.2 governance cleared a heuristic revision on an incomplete review. A
  heuristic revision is now cleared only by a genuinely complete, favorable
  review (fidelity `PASS`, known positive substantive value, retention
  `PRESERVED`, coherence `COHERENT`, template feel `LOW`, unsupported claim
  `NONE`, narration `APPROPRIATE` where relevant), any `UNKNOWN`/concerning
  decision-critical dimension defers governance, and a later weaker review
  restores a previously cleared revision. Policy/validation advanced to
  `stage4.2-v8`.

## Remaining work (explicit)

- Semantic writing/critique provider adapter and the Gemini TTS adapter, with
  shared admission, bounded raw-call accounting, and fake-transport tests.
- The DB-assembly service, `MATERIALIZATION` queue/executor, `RENDER_EXECUTION`
  `TRANSFORMED_SHORT` handoff, idempotent Stage 6 finalizer, fencing,
  cancellation, and resume.
- Reusable-renderer extension for authored cards/output overlays, mixed segment
  assembly, explicit audio tracks/gain envelopes, final mix
  measurement/mastering, and final-purpose QC (authored sections, source
  synchronization, caption/audio coverage).
- Candidate-scoped API (`POST/GET /api/candidates/{id}/materializations`,
  `GET /api/materializations/{id}`, cancel, artifact), CLI, and the Produce /
  Results UI changes.
- Real manual acceptance (approximately three clips, maximum four) with the
  fail-closed provenance and exact-currentness principles reused from Stage 5.2.

Until those exist, no transformed-short MP4 can be produced and no acceptance
claim is made.

## Verification

Run the Stage 6 tests with the documented Docker Python 3.12 runtime (or an
isolated 3.12 environment):

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend \
  python -m pytest tests/test_stage6_materialization_core.py \
                   tests/test_stage6_supporting_narration.py \
                   tests/test_stage6_migration.py -q
```

The host Python 3.10.12 run is a convenience only and is not Python 3.12
verification. PostgreSQL migration validation requires
`CLIPFACTORY_TEST_POSTGRES_URL`; without it the PostgreSQL test is skipped.
