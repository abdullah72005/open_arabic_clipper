# Stage 6 transformed-short materialization operations

Status: **IN PROGRESS.** The durable persistence, the deterministic
materialization core, the candidate-scoped queue/executor/worker, the Stage 5.2
`TRANSFORMED_SHORT` authored renderer, the idempotent finalizer, the hosted
Gemini writing/critique adapter, the Gemini TTS adapter with separately
checkpointed local ASR fidelity validation, the bounded voice audition, and the
read-only acceptance exporter are implemented and tested. The candidate-scoped
API/CLI review surface and the Produce→Results UI parity are implemented: Produce
tracks the durable materialization id across preparation/rendering/finalization
and exposes a transformed short only after the committed assessment permits
review; Results consults the shared Stage 6 readiness evaluator; cancellation is
server-authoritative including an in-flight submission, and id-based resume
revalidates live prerequisites. The remaining work is real manual acceptance over
authorized source media and the human media review. Stage 6 remains awaiting human
media review and SOL review; it is not frozen.

## Stage 6 CLI commands

```bash
# Queue / resume / cancel one candidate-scoped generation.
python -m app.cli materialize <candidate-id> [--priority CANDIDATE] [--profile-key K] [--supporting OFF]
python -m app.cli materialization-resume <candidate-id> [--profile-key K]
python -m app.cli materialization-cancel <candidate-id>
python -m app.cli materialization-status <candidate-id>

# Import a strict JSON production profile (new current revision per channel_key).
python -m app.cli production-profile-import path/to/profile.json

# Read-only exports (never fabricate eligibility).
python -m app.cli stage6-acceptance-export [--discover] [--limit 4] [--materialization-id ID ...]
python -m app.cli stage6-tts-audition                      # 3 voices x 3 scripts, <=9 calls

# Recover surviving audition evidence and preserve the consumed-budget floor.
python -m app.cli stage6-tts-audition-recover --report path/to/captured.json --probe-calls 1
```

An omitted `--profile-key` (and an omitted API `profile_key`) resolves to the
current `channel_key='default'` profile when one exists, otherwise to the neutral
no-voice profile. `stage6-tts-audition` makes zero live calls without a configured
`GEMINI_API_KEY` and reports that honestly.

Audition accounting is truthful: shared HIGH admission is checked before *each*
attempt (a denial spends no attempt), a 429 applies the shared rate/quota cooldown,
and each attempt is classified by the shared pure delivery classification
(`UNCERTAIN` for transport ambiguity/crash-left reservation, `FAILED_KNOWN` for a
known HTTP rejection, `RESPONDED` for a received-but-unusable response). The
durable `consumed-budget.json` floor survives a missing ledger, changed storage
root, or container restart, so recovery never authorizes a fresh nine-call batch.
Local narration fidelity is policy-versioned (`stage6-fidelity-v2`): protected
technical/numeric forms are preserved atomically, and a lone ASR transcript
disagreement is deferred evidence (never an automatic confirmed meaning mismatch)
while independently established audio/content failures still reject. Deferral
never approves a voice or review readiness.

Published export packages are immutable and self-verifying: an existing
`packages/<case>/<package_identity>/` is reused only after its manifest, index, and
every listed file hash re-verify. A corrupt existing package, or an
unreadable/malformed/unsupported catalog, fails closed and preserves the previous
reference and the valid staging evidence.

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

The optional `tts_settings["narration_gain_db"]` requests a bounded constant gain
for every narration placement in the final mix (clamped to `-12..+6 dB` by
`policy.narration_gain_db`; neutral default `0.0`). It participates in the audio
recipe and output fingerprint, and is applied by the frozen renderer as an
explicit `volume=<gain>dB` gain point on the narration track.

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

## Hosted adapters

- `gemini.py`: `GeminiWritingProvider` and `GeminiCritiqueProvider` execute exactly
  one `generateContent` request per invocation (the durable gateway reserves the
  attempt first), v1, temperature 0, routine `gemini-3.5-flash-lite` / strong
  `gemini-3.8-flash` (low thinking), strict structured-output schemas with closed
  enums, and disabled SDK retries. A 429/quota/refusal/malformed/timeout outcome
  is returned as a truthful *deferred* result, never raised into the pipeline.
- The script critique is categorical per block (grounding/entailment, substantive
  added value, semantic fidelity, unsupported facts, code-switch preservation,
  redundancy, dialect/register fit, naturalness, length/necessity). Acceptance
  requires a *known positive* grounding and added-value verdict plus no critical
  concern/failure; a favorable critique never clears a deterministic UNMET finding.
- `tts_gemini.py`: `GeminiNarrationSynthesizer` uses the Interactions API
  (`v1beta`, model `gemini-3.8-flash-tts`) with verbatim input, a `speech_metadata`
  style annotation, and a `speech_config` voice; it honors the returned MIME (a
  WAV is saved as-is; headerless `audio/l16` is wrapped exactly once). Accepted
  audio is written as immutable lossless bytes through StorageService.
- Local fidelity validation (`validate_narration_fidelity`) is a *separate*
  checkpoint: it transcribes the synthesized WAV with the leased faster-whisper
  engine and applies the strict order/occurrence-, negation-, number-, and
  code-switch-aware comparison. Unknown/empty fidelity defers; a clear meaning
  mismatch rejects.
- `tts_audition.py` / `stage6-tts-audition`: exactly three prebuilt voices
  (Charon, Kore, Achird) × three fixed scripts (Egyptian conversational,
  Arabic-English code switching, English), at most nine raw synthesis calls, no
  automatic retries, cached by request identity. The result proposes one
  technically viable PROVISIONAL voice; human quality approval is still required.
- `acceptance_export.py` / `stage6-acceptance-export`: read-only assembly of the
  exact persisted lineage (MP4, index, ASS, timeline, audio recipe, QC/assessment,
  provider attempts and repair usage, SHA-256 manifest). Every COPIED destination
  byte set (MP4, bound authored ASS, narration, supporting assets) is verified
  against its authoritative expected hash before publishing; a missing or
  mismatched copy invalidates acceptance. The snapshot/recheck binds the complete
  render/QC/assessment and referenced-asset identities. It never fabricates
  rights, governance, speech, or eligibility. A diagnostic export never replaces a
  valid package: the last valid package is preserved and the diagnostic package is
  published separately, and the catalog is updated atomically with the published
  package identity.

## Remaining work (explicit)

- Candidate-scoped API/CLI completeness and response types (separate
  lifecycle/semantic/live freshness, component progress, exact render binding,
  findings) are implemented; the singular candidate route,
  `materialize`/`materialization-status`, resume, cancel, and the artifact
  identity contract exist.
- Produce/Results UI parity is implemented: Produce tracks the durable
  materialization id across preparation → render → finalization, polls the
  read-only materialization endpoint across changing render/finalizer job
  identities, and offers a transformed short only after the committed readiness
  response permits review (never from `artifact_available` alone); cancellation is
  server-authoritative (including an in-flight submission) with an explicit
  unconfirmed/retry state; Results consults the shared readiness evaluator and
  distinguishes current review-ready output from pending/deferred/rejected/
  historical/diagnostic media. No automatic publication approval.
- Real manual acceptance (approximately three clips, maximum four) over authorized
  source media, and the human media review. No authorized source media is
  configured in this environment, so no genuine `MANUAL CHECK REQUIRED NOW`
  artifact exists. A hosted `GEMINI_API_KEY` IS configured (and was used for a live
  adapter validation and the bounded 3x3 TTS audition, which returned no
  technically viable voice — see `STATUS.md`); the missing prerequisite is the
  authorized media, not the provider credential.

Until a genuine manual-acceptance run exists, no acceptance claim is made.

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

The SOL-reopened corrections are covered by
`tests/test_stage6_rc_v4_corrections.py` (id-scoped cancel/resume authority and
resume revalidation), `tests/test_stage6_cache_validation.py` (completed-cache vs
committed-assessment unification, missing-assessment finalizer recovery, and
destination-byte export verification/preservation), and
`tests/test_stage6_results_boundary.py` (the Produce→Results HTTP boundary and the
transformed Results review classification). Live-input mutation around resume is
covered by `tests/test_stage6_finalizer_currentness_postgres.py` under
`CLIPFACTORY_TEST_POSTGRES_URL`. Frontend Produce/Results behavior is covered by
`frontend/src/lib/produce-runner.test.ts` and
`frontend/src/components/produce-runner.test.tsx`.
