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
threads, filtergraph threads, global concurrent renders, cancellation poll
cadence, admission wait, render/source/output time ceilings, and QC sampling.

## Runtime checks

Required: Python 3.12, FFmpeg with libass, `ffprobe`, and (for gated tests) a
disposable PostgreSQL. Set `PYTHONPATH=/app` when running tests so a stale
installed package is not imported. Do not reset an operator database; the
PostgreSQL-gated tests create and drop disposable databases.

## Manual acceptance

`python -m app.render.execution.acceptance --source <media> --output
storage/benchmarks/stage-5-2/manual-acceptance` produces four short playable
MP4s plus `README.md`, `manifest.json`, and per-case QC. Human inspection of lip
sync, cut timing, crop smoothness, black flashes, caption timing/highlighting,
mixed BiDi, sharpness, background fill, audio level, and multi-span
synchronization is required. Automatic QC is not human approval.
