# Stage 6 production completion implementation plan

> **For V4:** Use `superpowers:executing-plans` task by task. Review directly;
> do not dispatch automatic multi-agent review or subagents. SOL architects and
> reviews; V4 implements. Checkboxes track work, not proof of product acceptance.

**Goal:** One eligible candidate becomes a genuine assessed transformed Short,
followed by a tiny real manual-acceptance set.

**Architecture:** A durable materialization envelope exists before provider work;
owned workers execute reusable component checkpoints, then dispatch the existing
renderer with an optional typed authored payload. A separate MATERIALIZATION job
finalizes exact render evidence and may allocate one bounded repair.

**Tech stack:** Python 3.12, SQLAlchemy 2, PostgreSQL 16, Redis/Celery, FFmpeg/libass,
FastAPI, Next.js/TypeScript, google-genai >=2.25,<3 for the specified TTS boundary.

**Spec:** [SOL production architecture](../specs/2026-10-14-stage-6-production-architecture.md).
Read it completely before editing. Inspected baseline: `098cc2c`.

## Global constraints

- Do not implement Stage 7, change frozen stage ownership, or lower upstream gates.
- No new PipelineStage/PipelineRun/_NEXT_STAGE/source lifecycle/job platform.
- Render source + authored lossless assets directly; no intermediate H.264 input.
- Preserve hero/source meaning, raw evidence, dialect, canonical source ASS.
- Narration optional by plan; supporting OFF by default and rights-cleared only.
- Semantic attempts ≤4, strong ≤2; narration chunks ≤3 and ≤60 seconds,
  synthesis attempts ≤6; output ≤180 seconds; one repair per generation.
- Final audio 48 kHz stereo, -16 LUFS ±1 LU; encoded true peak ≤-1.5 dBTP.
- Unknown critical evidence defers; no aggregate score or fake acceptance evidence.
- Do not commit HANDOFF.md or oac_implementation_evidence.txt.
- `publication_ready=false`; freeze only after human media review and SOL review.
- Preserve existing Stage 3.5/Gemini/governance/binding compatibility corrections.
- Make focused commits. Continue through the full plan unless a demonstrated
  external prerequisite blocks dependent work. Missing internal code is work.

## Task 1 — Envelope-first queue and correct ownership

**Files:** create `backend/app/materialization/repository.py`; modify
`queue.py`, `executor.py`, `service.py`, `types.py` in that package;
`backend/app/workers/tasks.py`; tests
`backend/tests/test_stage6_runtime_ownership.py` (new). Use the existing domain
models/migrations; do not add schema by habit.

**Interfaces:** implement MaterializationOwner, get_or_create_envelope,
claim_materialization from spec section 3; keep public queue_materialization and
MaterializationQueueOutcome, adding materialization_id/render_execution_id as needed.

- [ ] Add a two-session PostgreSQL queue test synchronized AFTER active lookup.
  Stub only broker delivery, use real persisted prerequisite fixtures, and assert:

```python
# Both outcomes identify the shared active generation/job.
assert outcomes[0].materialization_id == outcomes[1].materialization_id
assert outcomes[0].job_id == outcomes[1].job_id
assert count_active_materialization_jobs(candidate_id) == 1
```

  `count_active_materialization_jobs` is a test-local scalar SQL count filtered by
  MATERIALIZATION, candidate's refinement/envelope, and QUEUED/RUNNING status.
- [ ] Add a production executor regression: finish a CANDIDATE generation, enqueue
  a valid FINAL_CLIP generation as a NEW job, execute it. Assert both distinct jobs
  can have local claim 1 and the newer input becomes current. Add a stale-old-worker
  mutation attempt and assert no row/job/reference changes.
- [ ] Add broker failure/retry regression: first dispatch raises; second POST
  redispatches the same job/request, creates no second job, and preserves the marker
  on repeated failure. Read-only GET must not dispatch.
- [ ] Implement spec section 4: candidate scope lock, envelope before work, durable
  request payload/FKs, CAS on envelope.active_job_id, and exact paired job/envelope
  ownership. Eliminate comparisons between different jobs' claim counters.
- [ ] Complete cancellation/resume and joint job/envelope finalization. Include
  fresh status predicates on the ProcessingJob.materialization_id update. Bind tasks
  to envelope identity; reject mismatched candidate/source/kind/FKs.
- [ ] Inspect migrations 0023/0024 on disposable PostgreSQL. Preserve prior rows on
  downgrade: 0024 currently deletes DEFERRED_EVIDENCE materializations. Replace that
  deletion with semantic_outcome=FAILED and lifecycle=BLOCKED; retain DEFERRED_EVIDENCE
  in reason_codes/history, keeping review/cache/publication flags false; add a row-preservation
  regression. This is Stage 6 schema work, not an upstream migration rewrite.
- [ ] Run these regressions under Python 3.12/disposable PostgreSQL, the existing
  sol3/integration tests, formatter/linter/type checks; update obsolete tests that
  assumed globally ordered claim counters. Commit the coherent ownership change.

**Deliverable:** Concurrent/retried requests own one scoped envelope/job; fresh
jobs work; stale/cancelled jobs cannot mutate it. No provider or renderer needed.

## Task 2 — Durable request gateway and content checkpoints

**Files:** create `materialization/components.py`; modify `repository.py`,
`policy.py`, `providers.py`, `executor.py`, `service.py`, `fingerprints.py`, `types.py`;
tests `backend/tests/test_stage6_component_recovery.py` (new).

**Interfaces:** reserve_attempt, load_component, save_attempt_result,
save_local_component, prepare_materialization as defined in spec. Pure providers
remain injectable; transport methods perform one request and do not own DB budgets.

- [ ] Add a fake provider that opens an independent DB session IN its write method
  and asserts its unique RESERVED attempt is already committed. Make it raise after
  recording invocation; restart the executor with a fresh session and inspect the
  durable UNCERTAIN attempt and remaining budget.
- [ ] Add accepted-writing-before-critique crash and accepted-synthesis-before-ASR
  crash tests. Resume must reuse the completed subcomponent without a second write
  or synthesis call. Failed local ASR must not regenerate the audio automatically.
- [ ] Add a full accepted SCRIPT replay test:

```python
before = provider.calls
recovered = load_component(session, materialization_id, "SCRIPT", dependency_fp)
assert recovered["status"] == "ACCEPTED"
assert recovered["result"]["authored_blocks"] == accepted_blocks
assert provider.calls == before
```

  Also run the real executor resume path; a direct load assertion alone is insufficient.
- [ ] Implement the checkpoint/attempt schemas and dependency fingerprints from
  spec sections 5–6. Persist writing/critique responses separately, actual accepted
  script/claims/findings, audio assets/ASR evidence, supporting selection, timeline.
  Use unique attempts plus stable request identity; no input-hash deduplication of
  separate raw calls, no lost JSON in-place mutations, no synthetic responses.
- [ ] Wire the production executor to checkpointed components. Remove production
  all-in-one provider execution followed by end-only persistence. Reuse existing
  pure helpers behind this boundary; keep any fixture compatibility wrapper explicit.
- [ ] Add cancellation-between-writing-and-critique, exhausted-budget, and duplicate
  delivery checks. Assert no next provider call after stop. Preserve repair use.
- [ ] Run focused recovery + existing provider/materialization tests, format/lint/types;
  commit. Do not call Gemini for these tests.

**Deliverable:** A crash cannot erase budget history; replay returns accepted
content, and only unfinished authorized phases execute.

## Task 3 — Frozen authored payload, timeline, and deterministic ASS

**Files:** create `materialization/rendering.py`, `materialization/overlays.py`,
`render/execution/authored.py`; modify materialization timeline/types/fingerprints,
renderer types/validation/timeline/policy/compiler/service; tests
`backend/tests/test_stage6_authored_spec.py` (new).

**Interfaces:** AuthoredRenderPayload, RenderUnit, MediaAssetRef, AudioTrackSpec,
GainPoint, MasteringSpec; optional RenderSpec.authored; build_transformed_render_spec;
compile_authored_render called from existing compile_render. Core defaults unchanged.

- [ ] Add a pure SOURCE→CARD→SOURCE spec fixture with a final-source gap occupied
  by the card. Assert authored visual units are contiguous and source-only occurrences
  correctly have a gap. Core validation must still reject unexplained source gaps.
- [ ] Add cumulative fractional-duration tests that would drift if each duration
  were rounded independently. Assert exact frame totals and sample boundaries from
  delivery fps/48 kHz, including fractional hero duration and trailing authored unit.
- [ ] Implement strict typed payloads and purpose-based validation. SOURCE references
  match actual Stage 5.0 rebound bindings and Stage 5.1 scenes; placeholders create
  no blank units. Required slot failures cannot reach a renderable payload.
- [ ] Implement persisted authored output-time ASS, safe escaping/Arabic shaping,
  static narration text, card layout, safe adjacent annotation placement, actual
  attribution, and readability findings per spec section 8. Reuse source ASS bytes
  unchanged. Do not inject text into filter expressions or fabricate word timestamps.
- [ ] Implement component INPUT/output fingerprints, asset hash verification, exact
  bound materialization ID/output hash, and no-narration voice-change compatibility.
  Test changed script/audio bytes, same-path asset replacement, and changed profile.
- [ ] Preserve source-only serialization/compiler fingerprints where new optional
  fields are absent. Add explicit transformed policy versions without silently
  rewriting core semantics. Run existing Stage 5.2 engine/live-contract tests.
- [ ] Format/lint/type-check and commit this typed seam.

**Deliverable:** A complete executable authored request exists with no renderer
semantic choices and no source-caption retiming ambiguity.

## Task 4 — Real authored video and lossless final audio execution

**Files:** create `render/execution/audio.py`; modify authored.py, compiler.py,
runner.py, qc.py, types.py, executor.py; tests
`backend/tests/test_stage6_authored_render.py` (new). Reuse `stage52_support.py`.

**Interfaces:** compile_authored_render returns the existing CompiledRender with
an optional typed audio-preparation plan. run_compiled_render retains its public
signature and executes the extra preparation phases only for authored payloads.

- [ ] Add a real-FFmpeg tiny SOURCE→CARD→SOURCE render. Use deterministic speech-like
  tones/synthetic video for this TEST only. Assert card frames actually appear,
  source pixels/captions remain in mapped intervals, duration matches, and encode
  manifest reports one final H.264/AAC output.
- [ ] Add narration-with-padding audio fixture at 44.1 kHz source / 24 kHz narration.
  Assert final 48 kHz stereo, sample-correct placement, no source/narration clipping,
  no truncated tail, and intended silent gaps. No INDEX analysis WAV in final mix.
- [ ] Implement video unit concat using frozen framing primitives, canonical source
  ASS before timestamp reset, authored ASS after concat, controlled asset inputs.
  Add muted rights-cleared background only on authored units.
- [ ] Implement spec section 9: float PCM final mix, full-mix analysis, feasible
  linear mastering with explicit settings, lossless master, then one final encode.
  Record every audio measurement and selected parameter. No per-asset normalization,
  automatic amix normalization, hidden dynamic loudnorm fallback, or extra MP4.
- [ ] Reuse shared process-group cancellation and one render deadline for mix,
  measure, master, encode, probe, and QC. Test cancellation during audio preparation
  and exhausted deadline preventing the next process. StorageService owns all paths.
- [ ] Extend purpose-aware QC: intentional static cards/silent text intervals are
  checked as authored intervals; source similarity/frozen checks run on source units
  with their actual mappings. Required source speech/narration still has audio checks.
  Measure final decoded AAC loudness/true peak; missing metrics fail mandatory evidence.
- [ ] Test loudness/peak rejection, Arabic/code-switch authored ASS, and real core
  render compatibility. Do not expand beyond tiny fixtures. Commit.

**Deliverable:** The existing renderer produces an actual mixed/mastered authored
MP4 from original media, with technical QC and no double lossy encode.

## Task 5 — Production render dispatch and finalizer

**Files:** create `materialization/finalization.py`; modify materialization queue,
executor, repository, rendering, service; renderer queue/service/executor; workers;
tests `backend/tests/test_stage6_vertical_slice.py` (new).

**Interfaces:** commit_prepared; optional materialization_id/expected_output_fingerprint
on render queue/spec builder; finalize_materialization returns FinalizationResult.
Reuse MATERIALIZATION job kind for post-render assessment; no new lifecycle platform.

- [ ] Build a tiny persisted integration fixture through existing Stage 4/5 helpers
  and actual Stage 5.2 executor. This is a TEST fixture; acceptance uses genuine
  production selections. Drive prepare job → render job → finalizer job.
- [ ] Implement AWAITING_RENDER handoff, exact durable render binding, dispatch
  recovery, post-render callback, and explicit resume recovery after a lost callback.
  No Celery task waits synchronously for another task. Re-delivered callbacks coalesce.
- [ ] Pass the exact artifact purpose through render envelope demotion/cache/read
  helpers, including `_get_or_create_envelope`; assert a preexisting core artifact
  remains current when a transformed render is created. Set stage6_implemented on
  transformed rows and preserve publication_ready=false.
- [ ] Add the critical stale-during-provider test and stale-during-render test:
  mutate candidate/plan/FINAL_CLIP/profile in another session before commit; assert
  no current review-ready reference, accepted history retained where valid.
- [ ] Extend frozen render publication locking/revalidation for exact bound
  materialization/profile/output hash. Finalizer locks/revalidates again and builds
  AssessmentContext entirely from actual stored evidence.
- [ ] Assert idempotent finalization produces identical assessment fingerprint and
  zero provider/render calls; missing metrics/critical semantic evidence defers;
  publication_ready stays false; pre-render state never requests final media review.
- [ ] Implement CAS one-repair allocation and the fixed repair action vocabulary.
  New content/audio recipe changes output hash and render identity, preserves ledger,
  and cannot revive the old render callback. At most one content-repaired render.
- [ ] Run relevant PostgreSQL ownership/currentness, real renderer/lifecycle tests,
  lint/types. Commit the working production vertical slice.

**Deliverable:** One candidate can finish the actual production render/finalization
path. Do not stop here if hosted adapters/UI/acceptance remain independently doable.

## Task 6 — Grounded hosted scripts and explainable critique

**Files:** create `materialization/gemini.py`; modify providers.py, components.py,
scripting.py, acceptance.py, settings.py; tests
`backend/tests/test_stage6_gemini_boundary.py` (new).

**Interfaces:** GeminiWritingProvider implements existing writing protocol;
GeminiCritiqueProvider implements the expanded categorical critique contract.
Both expose single-request transports called through the durable gateway.

- [ ] Add real-installed-SDK fake-HTTP tests asserting v1 generateContent, selected
  model, structured schema, bounded actual evidence, temperature/thinking settings,
  and one raw POST with retries disabled. Add malformed/incomplete/429/refusal cases.
- [ ] Implement routine/strong model choices and routing exactly from spec section
  10. No premium pass for every clip, provider availability probes, or automatic Qwen.
- [ ] Expand script critique dimensions and strict acceptance. Test useful grounded
  inference, paraphrase/filler, unsupported numbers/facts, invalid references,
  Arabic register/code-switch issues, and UNKNOWN critical critique. Preserve
  deterministic hard failures, actual claim ledger, and required contract slots.
- [ ] Wire explicit materialization mode/settings into provider construction;
  do not infer allow_hosted from whether a critique object happens to be present.
  Missing providers defer honestly; deterministic cache hits make zero calls.
- [ ] Run fake boundary/semantic tests and upstream request/governance regressions;
  format/lint/types; commit. A few actual writing calls are reserved for the real
  acceptance clips, not repeated test validation.

**Deliverable:** Production writes useful grounded original prose with auditable
critique and correctly bounded provider use.

## Task 7 — TTS, local audio verification, and persistent provisional voice

**Files:** create `materialization/tts_gemini.py`; modify narration.py, components.py,
profiles.py, settings.py, `backend/pyproject.toml`, CLI; tests
`backend/tests/test_stage6_tts_boundary.py` (new). Update .env.example and environment docs.

**Interfaces:** GeminiNarrationSynthesizer implements NarrationSynthesizer; accepted
NarrationAudio points to real immutable StorageService-managed bytes. Persist synthesis
before separate local ASR validation; both are component checkpoints.

- [ ] Check current official documentation again and install compatible SDK in the
  Python 3.12 test image. Set google-genai >=2.25,<3; test the effective API/body through
  the installed SDK rather than only asserting a Python dict.
- [ ] Implement Interactions v1beta, verbatim accepted text, style metadata and fixed
  voice configuration, correct WAV/raw-PCM handling, no retries, bounded audio limits.
  Test WAV isn't double-wrapped, raw PCM metadata, wrong/missing MIME, silent/truncated
  response, and uncertain request recovery. Preserve other clients' independent v1.
- [ ] Implement local Whisper fidelity validation through existing heavy-model leases;
  no Gemini transcription for generated audio by default. Store ASR/runtime evidence.
  Test repeated-name Arabic role reversal, negation scope, number attribution,
  code switching, and allowed orthographic/nonlexical variation. Do not use a loose
  scalar ratio as proof of meaning.
- [ ] Add profile JSON import/read and a bounded CLI audition: exactly Charon/Kore/
  Achird × the three scripts in spec, at most nine raw calls. Cache samples by request
  identity. Export audio/checks/manifests and report quality honestly.
- [ ] Persist one technically viable PROVISIONAL channel voice before rendering the
  acceptance clips. Human quality approval is still required. If none is viable,
  report the concrete provider/quality evidence and complete independent work.
- [ ] Run narration/provider/upstream SDK regressions, lint/types; commit.

**Deliverable:** Real optional narration is traceable, consistent per channel,
audio-backed validated, and never forced into narration-NONE plans.

## Task 8 — Candidate API/CLI and truthful Produce/Results

**Files:** `backend/app/api/app.py`, `backend/app/cli.py`, materialization service/queue;
`frontend/src/lib/{api-client,produce}.ts`, `frontend/src/components/produce-runner.tsx`,
`frontend/src/app/results/page.tsx`, related existing tests and new API tests.

**Interfaces:** exact routes/commands in spec section 13; retain singular candidate
route compatibility. Default omitted supporting mode inherits the OFF-default profile.

- [ ] Test queue/read/resume/cancel/artifact boundaries, stale/historical responses,
  cache/resume flags, and exact bound artifact identity. GET makes no writes/dispatch.
- [ ] Implement response types with separate lifecycle/semantic outcome/live freshness,
  component progress, render binding, final findings, and human review requirement.
- [ ] Replace Produce's core-render terminal step with materialize→render→finalize
  polling for the SAME candidate. No candidate retry loop or bulk orchestration.
  A CANDIDATE transcript fallback cannot substitute for final transcript readiness.
- [ ] Make Cancel cancel server work before stopping polling. Results explicitly
  distinguishes core validation from transformed media and never says publish-ready.
- [ ] Run focused API/CLI and frontend tests/types/lint/build as documented; update
  README/operations/environment/API docs with actual commands and behavior. Commit.

**Deliverable:** An operator can produce, inspect, resume, cancel, and review one
actual transformed candidate through the normal application boundary.

## Task 9 — Genuine manual-acceptance package and stop

**Files:** create `materialization/acceptance_export.py`; CLI; update STATUS.md,
docs/STAGE_6_OPERATIONS.md, ARCHITECTURE.md, PIPELINE.md, README as appropriate.
Keep generated media in managed storage, not git.

- [ ] Identify at most three/four authorized source moments with genuine current
  production eligibility. Do not manually seed governance success or synthetic
  acceptance speech. If none are available, report exact unavailable source/eligibility
  evidence; do not turn Stage 6 into source search or upstream gate remediation.
- [ ] Produce the specified tiny set through normal candidate APIs/jobs: source-led
  no narration, useful context/commentary, Arabic/code-switch narration; optional
  supporting case only if useful and rights-cleared. Use one provisional voice.
- [ ] Export videos, readable index, lineage/scripts/ASS/audio/assets/timeline/QC/
  findings/hash manifest and reserved/confirmed/uncertain provider counts. Record
  no-output cases truthfully, never label a core render transformed.
- [ ] Run focused final integration/currentness checks under 3.12/PostgreSQL/real
  FFmpeg and relevant frontend checks; report exact results/skips. Do not rerun huge
  suites or provider samples without a concrete remaining concern.
- [ ] Update status with implemented versus pending HUMAN review, commit coherent
  docs/code, and report files/migrations/provider choices/test results/exact absolute
  host paths/limitations/deferred Stage 7/git status/commit hashes.
- [ ] Say `MANUAL CHECK REQUIRED NOW` only when actual assessed videos are ready.
  Stop for human media review; do not freeze Stage 6 or begin Stage 7.

Final report ends exactly `READY FOR SOL REVIEW` or `IMPLEMENTATION INCOMPLETE`.

## Execution commands and evidence discipline

Read README for the current container test command. A minimal existing-container
pattern (install test tools in an isolated temporary target if missing) is:

```bash
docker compose exec -T backend python --version
docker compose exec -T backend python -m pip install --target /tmp/stage6-test-deps \
  'pytest>=8.3,<9' 'pytest-asyncio>=0.24,<1' ruff mypy
docker compose exec -T -e PYTHONPATH=/tmp/stage6-test-deps:/app backend \
  python -m pytest tests/test_stage6_integration.py -q
```

Use the project dependency installation/build flow for SDK/package changes so worker
and backend actually share compatible runtime dependencies. Disposable PostgreSQL
fixtures derive an admin URL from the supplied test URL, create a unique database,
and destroy only that database. Never run downgrade/recreate against the application
database. Do not print secrets. Test skips are separate from passes.

## SOL review checkpoints

The implementation order is fixed; checkpoints are independently testable commits,
not invitations to stop after every helper task. If interrupted, leave a truthful
status and resume the next task. Escalate only an architectural contradiction with
concrete evidence or a real external prerequisite. Ordinary names, migrations,
thresholds, API shape, and module placement are already decided above.

SOL will review: runtime ownership/recovery evidence, actual renderer integration,
grounding/naturalness and narration necessity, exact final artifact lineage, and the
tiny acceptance package. Human media judgment remains mandatory.
