# Stage 5.2 operations — render execution, audio, and technical QC

Stage 5.2 is a dependable deterministic media execution engine. It consumes a
current executable **Stage 5.0** render contract and a current ready **Stage 5.1**
visual-composition plan (with its canonical ASS), plus a versioned delivery
profile, and produces an encoded artifact, a normalized execution manifest,
deterministic technical QC, and a durable fenced result.

It is explicit and candidate-scoped, extends the existing Celery/`ProcessingJob`
platform with a `RENDER_EXECUTION` job kind, adds **no** `PipelineStage`, **no**
`PipelineRun`, and **no** `_NEXT_STAGE` entry, and never touches the source
lifecycle.

## Purpose

The current artifact purpose is `CORE_SOURCE_VALIDATION`: it executes only the
ordered `SOURCE_MEDIA` occurrences from the contract. Authored material
(narration, textual annotations, verification evidence, transitions) is
deliberately omitted and every omitted block/slot requirement is persisted and
exposed. Omitted authored time is never filled with invented black screens,
placeholder narration, planning drafts, or fabricated content.

A source-core artifact is **not** publication-final. `publication_ready=false`,
`stage6_implemented=false`, and the final authored timeline stays unfrozen.
`QC_PASS`, successful encoding, and `FINAL_TRANSCRIPT_READY` are never publishing
eligibility or approval.

## Ordering relative to Stage 6

Stage 5.1 supplies source-local visual intelligence. Stage 6 will materialize
authored content and final timing. The reusable Stage 5.2 engine executes a
supplied timeline. A typed seam exists for a future supplied materialized
timeline; the low-level engine is independent of candidate/database discovery
and never authors media. There is **no mandatory intermediate H.264 file**: one
final video encode and one final audio encode are produced per render, directly
from the managed original source.

## Source-local and output-local timing

Each occurrence maps `source [start, end)` to `output [start, end)` at unit
speed: `output_time = occurrence.output_start + source_time - occurrence.source_start`.
Source 30.5 in a `[30,33)` occurrence starting at output 2.0 maps to 2.5. The
removed source gap never remains in video, audio, or captions.

`Compiler`/`timeline` use one shared source timestamp origin from the source
media timing, preserve genuine source A/V offsets, use explicit `trim`/`atrim`
windows and `setpts`/`asetpts=PTS-STARTPTS` per visual scene, and accumulate
output-frame/sample boundaries **cumulatively** (never independent per-scene
rounding). Video is joined per scene inside each occurrence; audio is joined
only at real occurrence boundaries, so visual splitting cannot repeatedly round
or pad audio. No blanket `-shortest`, no unbounded async audio correction.

## Executing the frozen visual plan

- `SOURCE_AS_IS`: accepted scale-to-contain + centered pad.
- `STATIC_CROP` / `MULTI_SUBJECT_FIT` / `CENTER_FALLBACK`: the accepted
  normalized crop geometry, clamped with the frozen `clamp_crop` math, rounded
  to even chroma-safe dimensions.
- `TRACKED_CROP`: consumes `cx`, `cy`, `height_fraction`, uses the accepted
  smoothstep (`u*u*(3-2u)`), holds endpoints outside interior intervals, and
  never interpolates across a hard scene boundary.
- `BACKGROUND_FILL`: accepted `gblur=sigma=36:steps=2` +
  `eq=brightness=-0.18:saturation=0.70` contained sharp foreground over
  scaled/cropped self-fill.

### Tracked-crop execution mechanism

The installed FFmpeg's `crop` filter does **not** expose runtime width/height
commands (verified: a `sendcmd` schedule on `crop` leaves the crop size
unchanged). The smallest proven equivalent is a per-frame dynamic
`scale` (`eval=frame`) that maps the accepted `height_fraction` to the output
height, followed by a per-frame `crop` position. The accepted normalized crop
path is not redesigned; zoom and pan both execute mathematically. A test proves
real changing `height_fraction` works in the installed build.

Rotation is applied exactly once (FFmpeg autorotates on decode); the compiler
sets `rotate=0` on the output. Exotic non-square pixel geometry fails closed
(`EXOTIC_PIXEL_ASPECT`) rather than silently changing the crop coordinate space.

## Canonical ASS execution

Canonical Stage 5.1 ASS is burned **before** `setpts` while timestamps are
source-local, preserving the original ASS bytes and dynamic highlight states.
The ASS is localized under a controlled filename in the attempt directory; it is
never re-serialized, re-escaped, re-ordered, or re-planned, and `bidi.py` is
never re-applied. Canonical transcript text stays logical Unicode; the derived
ASS already carries the accepted run-level visual order. The ASS asset must be
managed, exist, and match its SHA-256; missing/corrupt/stale ASS fails closed.

`ass.event_count` counts caption events, not dynamic Dialogue states. Validation
requires at least one Dialogue row per caption event (never equality).

## Delivery profile and audio

A separate versioned Stage 5.2 delivery profile (independent of the Stage 5.0
output profile) is code-defined:

- MP4, `libx264`, CPU-only, `preset veryfast`, CRF 20, `yuv420p`;
- square-pixel 1080x1920 at the contract target frame rate (set with `-r`);
- AAC 192 kb/s, 48 kHz;
- mono/stereo preserved; supported wider layouts use an explicit stereo downmix;
- `+faststart`, explicit video/audio mapping, no subtitle/data/attachment copy.

Original source audio is used (never `speech-analysis.wav`), trimmed to matching
occurrences, rebased, resampled/channel-mapped explicitly, joined, and encoded
once. There is no crossfade, time-stretch, gain pumping, publication loudness
normalization, narration, ducking, or Stage 6 mixing. A clear seam remains:
source + authored audio → explicit mix → one final mastering pass → encode.

## Persistence and request identity

- Table `render_executions`, one row per candidate + deterministic input
  fingerprint; scoped partial unique current index on
  `(candidate, artifact_purpose, delivery_profile_key)`.
- `JobKind.RENDER_EXECUTION`, `processing_jobs.render_execution_id`.
- Lifecycle: `QUEUED`, `RENDERING`, `QC_RUNNING`, `COMPLETE`, `BLOCKED`,
  `FAILED`, `CANCELLED`. QC severity/status is separate: `PASS`, `WARN`, `FAIL`.
- `COMPLETE` requires no hard QC failure (`PASS`/`WARN`). `BLOCKED` is a
  non-executable input; `FAILED` is process/technical-QC failure. Only current
  valid `COMPLETE` results are cache-eligible. Cache/artifact/lifecycle
  consistency is enforced by check constraints.

The request fingerprint covers exact upstream identities/fingerprints, source
stat identity, the validated visual payload, canonical ASS SHA-256 and policy
bindings, ordered timeline/occurrences, artifact purpose, audio execution
policy, delivery profile/encoder options, actual FFmpeg/libass/font runtime
identity, compiler/timeline/QC policy versions, and every output-affecting
execution choice. It excludes job/attempt/claim ids, timestamps, progress,
transient lock availability, absolute deployment paths, API keys, TTS
provider/model/voice labels, publishing metadata, and unrelated analytics.

The request is frozen at queue time. Retries/force create a new attempt, never a
new semantic request identity, and never mix an earlier successful artifact with
a later failure's QC.

## Queueing, fencing, cancellation

- Queue under a short transaction: lock/revalidate, savepoint uniqueness
  recovery, atomic `active_job_id` compare-and-swap, commit before dispatch.
- Concurrent identical requests converge; `force` never duplicates an already
  active equivalent render.
- Duplicate/redelivered Celery invocations are fenced by an atomic
  `QUEUED/FAILED -> RUNNING` job claim that advances `claim_version`. All
  worker-owned writes are guarded by exact job id, claim version, RUNNING
  status, `active_job_id`, and request identity, so an old worker can never
  overwrite a newer run.
- Queued cancellation is honored before any work. Cancellation is observed
  through fresh scalar queries while FFmpeg/QC runs. Child process groups are
  terminated and escalated to `SIGKILL` after a bounded grace, then reaped; only
  owned temporary files are cleaned; no success pointer is published; no next
  stage is scheduled.
- A single PostgreSQL session advisory lock on a dedicated connection caps
  simultaneous render attempts across workers. Admission is non-blocking with a
  bounded wait, verifies dedicated-connection liveness, and releases on every
  exit path. It never holds candidate row locks during encoding.
- Bounded retries (maximum three) apply only to explicit transient failures
  (admission/broker/storage). Stale/malformed plans, missing/corrupt ASS,
  unsupported profiles, deterministic compile errors, QC failures, and
  cancellation are never auto-retried. Render retries do **not** route through
  the generic source `/sources/{id}/retry`.

## Artifact safety and cache

Layout (storage is the sole path owner):

```
sources/{source_id}/render-executions/{render_execution_id}/
  {job_id}-{claim_version}/
    output.mp4
    execution.json      # normalized manifest
    qc.json             # technical QC
    captions.ass        # canonical ASS, verbatim
    filtergraph.txt     # generated graph
```

Temporary output is attempt-local and immutable per claim. The artifact pointer
is published only after process success, artifact hash/probe/decode/QC, source
freshness recheck, and a fenced finalization transaction. A crash may leave an
unreferenced immutable artifact; that is safer than an incorrect authoritative
pointer. A missing/corrupt cached artifact is not a cache hit. A matching good
artifact may be reused without a fresh encode only after exact validation.
Determinism means identical normalized decisions and complete runtime-bound
identity; the actual output SHA-256 is stored.

## Technical QC

Hard checks: managed artifact exists/non-zero; ffprobe succeeds; expected
video/audio streams exist; expected codecs/pixel format/dimensions/SAR/DAR;
rotation state; finite plausible duration; FPS sanity; sampled decoded frames
(including around joins) succeed; audio decodes; sample rate/channel layout
match; timeline/frame/sample count agreement; A/V duration consistency; caption
intervals inside occurrences. Appearance checks (bounded low-resolution samples):
near-total black, frozen non-caption content (ignoring caption color changes),
total silence, and peak/clipping risk. Timing tolerance is
`max(0.10 s, 2/fps + 1024/48000)` for container/audio-video duration; cumulative
multi-span drift is tested tightly.

QC never proves lip sync, good cropping, readability, or visual taste, and does
not perform OCR or run a face/vision model. QC failure leaves no usable success
pointer and is never cache-eligible. Warnings remain visible in API/CLI/report.

## API and CLI

- `POST /api/candidates/{id}/render-execution` (bounded delivery-profile choice
  + `force`; current purpose `CORE_SOURCE_VALIDATION`; publication-final requests
  are rejected)
- `GET  /api/candidates/{id}/render-execution`
- `GET  /api/render-executions/{id}`
- `GET  /api/render-executions/{id}/artifact` (served through StorageService
  validation + `FileResponse`; missing/corrupt is 404)
- `POST /jobs/{job_id}/cancel` (existing)

CLI:

```
python -m app.cli render-execution CANDIDATE_ID [--delivery-profile KEY] [--force]
python -m app.cli render-execution-status CANDIDATE_ID
python -m app.cli render-execution-artifact RENDER_EXECUTION_ID
```

The Stage 5.2 handoff (`GET /api/candidates/{id}/stage5-2-handoff`) now exposes
the authoritative Stage 5.0 output profile (previously empty); this is a
read-only correction and does not alter Stage 5.1 output or fingerprints.

## Configuration

Add the `CLIPFACTORY_RENDER_*` variables from `.env.example` to bound encoder
threads, non-complex filtergraph threads (`CLIPFACTORY_RENDER_FILTER_THREADS`),
complex filtergraph threads (`CLIPFACTORY_RENDER_FILTER_COMPLEX_THREADS`),
global concurrent renders, cancellation poll cadence, admission wait,
render/source/output time ceilings, and QC sampling. The source/output duration
ceilings are enforced at spec build and compile time.

## Remediation against code review (stage5.2-v3)

The engine/policy/fingerprint/schema versions are `stage5.2-v3` (`v3`
fingerprint). Prior `v1`/`v2` rows are not reused; a request carrying the new
policy identity creates a fresh envelope.

A third focused pass closed five review findings without changing any
policy/fingerprint version: an authoritative cancellation now wins over the
heartbeat-loss latch; the bound `FINAL_CLIP` refinement is locked through
publication; full ownership (including the latch and admission) is rechecked at
the final boundary; historical reactivation is committed before returning; and a
single absolute attempt deadline covers encode, output probing, and QC. A fourth
focused pass closed two more: the bound selected Stage 4.1 `TransformationPlan`
is locked through publication (its writer does not take the candidate lock), and
shared-deadline exhaustion is propagated as `QCTimeout` through every frame/audio
QC path and rechecked before returning the verdict (never a `WARN`), with the
runner recomputing its probe budget after hashing. A fifth focused pass closed
the failing-output-probe cancellation gap: a cancel observed while `ffprobe` runs
is honored on the probe exception exit (timeout, nonzero exit, launch failure, or
malformed output) in both the runner and the executor's runner-error boundary, so
it can no longer strand the execution as `RENDERING`.

- **Attempt state.** A forced rerender enters a fenced `RENDERING` transition
  that atomically clears `cache_eligible`, `qc_status`, the artifact pointer,
  manifest, QC result, and output fingerprint, so the
  `ck_render_executions_cache_consistency` check always holds and an earlier
  success is never mixed with a later failure.
- **Ownership fencing.** Every worker-owned mutation is one UPDATE fenced on
  `active_job_id`, executing job id, `claim_version`, and `RUNNING` status. The
  job claim predicate itself includes status/heartbeat eligibility. A failed
  transaction is rolled back before failure persistence (no
  `PendingRollbackError`), and a superseded worker cannot fail/cancel a newer
  run.
- **Cancellation.** A dedicated cancellation fence accepts a job already flipped
  to `CANCELLED` (as an API/session cancel does) while still requiring the exact
  execution, `active_job_id`, executing job, and `claim_version`. Running
  cancellation during rendering and during QC finalizes the owned row as
  `CANCELLED`, clears cache eligibility and the artifact pointer, releases
  `active_job_id`, and never touches a newer run. The regression cancels the real
  job on a separate committed connection before the worker marks cancellation.
  Cancellation wins over the heartbeat-loss latch: an API/session cancel makes the
  heartbeat's `RUNNING` update match zero rows and set the latch, but the stop path
  still finalizes the owned execution as `CANCELLED` rather than failing it with a
  `RENDER_OWNERSHIP_LOST` code the `RUNNING`-only fence cannot apply.
  Cancellation during a *failing* output probe is also honored: if the operator
  cancels while `ffprobe` runs and the probe then times out (after consuming the
  shared budget), exits nonzero, fails to launch, or returns malformed output, the
  runner re-checks cancellation/ownership on the probe exception exit and raises
  the cancellation path instead of propagating `RENDER_TIMEOUT`/`QC_PROBE_FAILED`.
  The executor re-checks again at its runner-error boundary, so a cancel committed
  between the runner's last poll and its error exit cannot leave the execution
  `RENDERING` with `active_job_id` retained. Without a stop the truthful
  timeout/probe-error classification is preserved.
- **Ownership loss.** The heartbeat-loss latch is sticky and authoritative in the
  stop predicate, which freshly verifies the job is still `RUNNING`, the claim
  and `active_job_id` still match, admission is still held, and the latch is
  unset. Expensive encoding/QC work stops (and the child is reaped) on heartbeat
  loss, terminal status, claim replacement, active-job reassignment, or admission
  loss; lost ownership records `RENDER_OWNERSHIP_LOST` distinctly from
  cancellation. The same full ownership check (latch, `RUNNING`/claim/active-job,
  admission) runs again at the final publication boundary, so a heartbeat or
  admission loss observed after QC's last stop poll cannot publish success.
- **Final currentness.** Publication is one short transaction that locks the
  candidate, its bound contract, the bound selected Stage 4.1
  `TransformationPlan`, the visual plan, and the bound `FINAL_CLIP` refinement,
  rebuilds the frozen request, compares fingerprints, re-checks source stat
  identity and the exact consumed ASS bytes, re-checks job/envelope/admission
  ownership, and completes the job *in the same transaction* that writes the
  result. Only the contract and selection writers take the candidate lock first;
  the selected plan and the `FINAL_CLIP` refinement are joined explicitly because
  their writers take no candidate lock — Stage 4.1's `_persist_plans` updates
  existing `TransformationPlan` rows (including `plan_output_fingerprint` and
  `is_current`) and commits without it, and `apply_manual_transcript` likewise.
  A planning rerun or a manual edit therefore either commits before the
  transaction (and fails the rebuild) or blocks on the locked row and serializes
  after publication. An upstream invalidation, cancellation, observed
  ownership/admission loss, or claim change committed before publication therefore
  blocks/cancels/supersedes instead of publishing a stale `COMPLETE`; no upstream
  lock is held during encoding or QC. Deterministic PostgreSQL race tests cover
  concurrent invalidation, cancellation, claim supersession, a bound-refinement
  update overlapping publication, a selected-plan update overlapping publication,
  and lock serialization of both.
- **Publication errors.** A publication/database failure is not swallowed as
  duplicate delivery: it propagates after rollback and is recorded truthfully, so
  it cannot report a false success or strand a `RUNNING` job. Terminal-state
  helpers no longer suppress broad exceptions.
- **Source-local origin.** Probe stream starts are normalized to the earliest
  stream start (the same origin FFmpeg applies to input timestamps); the common
  container offset is removed while the genuine relative A/V offset is preserved
  (video-5s/audio-5.25s → video 0/audio 0.25). Missing/unusable stream-timing
  evidence fails closed with `SOURCE_STREAMS_UNSUPPORTED`; hermetic tests inject
  explicit stream facts through an override seam instead of relying on a silent
  production fallback.
- **Shared origin.** The compiler preserves genuine per-stream start offsets via
  `adelay`/`apad` (audio) and `tpad` (video) around the occurrence origin instead
  of independently rebasing both streams to zero.
- **Runtime identity.** The loaded libass is resolved to its real shared object
  (filename + content hash, never `--enable-libass`); the effective primary and
  Latin fallback font files are content-hashed (never a basename or empty hash);
  library versions are parsed in full. Discovery is cached under a signature that
  includes the binary/font/library stat, so a changed dependency under the same
  path invalidates it.
- **Cache validity.** Cache reuse validates artifact existence/size/digest,
  lifecycle/QC, current upstream dependencies, runtime identity, and QC identity.
  A changed QC policy invalidates the request/verdict.
- **Admission.** Dedicated-connection liveness is verified by an actual
  ping/advisory-lock query; a severed backend is reported not-held and the active
  child is reaped.
- **Dispatch/history.** A broker dispatch failure is recorded and redispatched on
  the next request; an equivalent historical request reactivates its row
  atomically within its purpose/profile scope, and commits that promotion before
  returning so a cached/active outcome cannot leave an uncommitted current-row
  flip that a request-session close would roll back.
- **QC / attempt deadline.** Per-stream start/duration and A/V delta are measured;
  silence is judged only against the selected source occurrences (bounded), so a
  legitimately silent selected span is not a hard failure. One absolute monotonic
  deadline covers encode, post-encode output probing, and QC; each bounded
  subprocess is sized to the remaining budget, and once the deadline is exhausted
  no further process starts. A frame or audio call that consumes its shared budget
  and raises `TimeoutExpired` is classified as exhaustion (not an ordinary decode
  failure), QC re-checks cancellation/deadline after every call — including
  exception exits — and once more before returning the verdict, so exhaustion is
  always a distinct `QCTimeout` persisted as a hard `QC_TIMEOUT` failure, never
  downgraded into a `WARN` success. Post-encode probing recomputes its budget
  *after* hashing (hashing can consume the remaining budget), verifies the
  deadline after probing, and raises `RENDER_TIMEOUT` and is skipped when the
  budget is already exhausted. Cancellation/ownership is polled before and after
  every bounded subprocess (including each source-audio call) and around output
  probing; cancellation keeps precedence over an exhausted deadline, and no
  source-audio or probe call starts after a stop is observed. QC is persisted on
  FAIL.
- **Migration.** The `20260918_0022` downgrade deletes Stage 5.2 rows/jobs
  itself (preserving all prior-stage data/jobs) so it works on a populated
  database without test-side cleanup.

## Runtime checks

Required: Python 3.12, FFmpeg with libass, `ffprobe`, and (for gated tests) a
disposable PostgreSQL. Set `PYTHONPATH=/app` when running tests so a stale
installed package is not imported. Do not reset an operator database; the
PostgreSQL-gated tests create and drop disposable databases.

## Manual acceptance

`python -m app.render.execution.acceptance --source <media> --output
storage/benchmarks/stage-5-2/manual-acceptance` produces four short playable
MP4s plus `README.md`, `manifest.json`, and per-case QC. Cases 01/02/04 use the
real source A/V with demonstration crop/scene specifications (not the frozen
Stage 5.1 plan) and are labeled as such; case 03 burns the canonical ASS built
through production `serialize_ass`. The gated
`tests/test_stage52_live_contract.py` exercises *persisted* Stage 5.0/5.1
bindings plus the canonical ASS through the real Celery task entry point to a
persisted managed artifact at
`storage/benchmarks/stage-5-2/live-contract/live-source-validation.mp4`; it binds
a uniquely named disposable database and never resets the supplied or operator
database. The canonical active-word regression
(`tests/test_stage52_render.py`) verifies the encoded highlight and stable
surrounding layout for `أنا كنت content creator لمدة سنتين` without OCR. Human
inspection of lip sync, cut timing, crop smoothness, black flashes, caption
timing/highlighting, mixed BiDi, sharpness, background fill, audio level, and
multi-span synchronization is required. Automatic QC is not human approval.
