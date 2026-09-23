# Stage 5.1 - Visual composition, framing, and caption plan

Stage 5.1 turns one current executable Stage 5.0 render contract into a durable,
deterministic, CPU-local, provider-free **plan**: anonymous face tracks, one
per-scene framing mode, a compact crop-keyframe path, FINAL_CLIP-only caption
events, one ASS document, and materialization-required overlay placements. It is
explicit, candidate-scoped work that extends the existing Celery/`ProcessingJob`
platform with a `VISUAL_COMPOSITION` job kind.

**Stage 5.1 produces a plan, not a rendered video.** There is no final FFmpeg
render, no encoding, no final audio mix, no TTS, no Stage 5.2, and no Stage 6.
The ASS document is a plan artifact that a later stage may consume; all real
rendering in Stage 5.1 is limited to PNG validation previews.

## Boundaries

Stage 5.1 must never:

- produce a final production FFmpeg render, encode libx264/aac/loudnorm, persist
  a final MP4, mix final audio, run render QC, or analyze black frames of a
  production render;
- select a TTS provider/model/voice, generate narration, write hooks, or
  generate B-roll/gameplay/images;
- call Gemini, Qwen/Ollama, a hosted vision API, or any network service at
  analysis time;
- scan a whole source, run active-speaker/audiovisual ML, split-screen, face
  recognition/identity/biometrics, or platform evasion;
- mutate Stage 4 or Stage 5.0 rows, materialize Stage 6 work, publish, schedule,
  or produce metadata;
- add a `PipelineStage`, an `_NEXT_STAGE` entry, or a `PipelineRun`.

Network calls = 0; Qwen loads = 0; Gemini calls = 0. The only external
processes are FFmpeg/ffprobe through safe argument arrays (bounded read-only
ffprobe display geometry, per-span frame sampling, per-span scene-cut detection)
and the CPU ONNX detector. Preview rendering is PNG-only. No new library is
introduced: the detector uses the already installed `onnxruntime`, and frame
decoding uses FFmpeg CLI plus `numpy`.

## Position and required inputs

```
Stage 4.3 selection
  -> Stage 5.0 executable render contract (READY_FOR_RENDER_PLANNING / MATERIALIZATION_REQUIRED)
  -> Stage 5.1 deterministic visual-composition plan (READY_FOR_VISUAL_EXECUTION)
  -> future Stage 5.2 / Stage 6
```

Queueing requires all of:

1. a current retained candidate whose disposition is `CANDIDATE` or
   `CANDIDATE_NEEDS_REFINEMENT`;
2. a current, live-effective, executable Stage 5.0 render contract:
   status `READY_FOR_RENDER_PLANNING` or `MATERIALIZATION_REQUIRED`,
   `contract_ready=true`, `live_freshness=CURRENT`, and `effective=true`.

`executor.py` and `service.py` re-resolve the contract and fail closed if it is
missing, not current, or not executable. Stage 5.1 consumes the contract's
bound source spans, ordered blocks, materialization slots, `caption_input`, and
display/output profile; it never re-reads Stage 4 plans or Stage 3.5 rows as
truth.

## Statuses

Semantic statuses are separate from the processing lifecycle:

| Semantic status | Meaning |
| --- | --- |
| `READY_FOR_VISUAL_EXECUTION` | Deterministic plan produced; `plan_ready=true` and cache-eligible. |
| `BLOCKED` | The contract is not current/executable, there are no valid bound source spans, or analysis scope was exceeded. |
| `FAILED` | An unexpected internal planning error (recorded as `INTERNAL_ERROR`). |

Execution lifecycle: `QUEUED`, `ANALYZING`, `COMPLETE`, `FAILED`, `CANCELLED`.
Closed plan-level reason codes include `CONTRACT_NOT_CURRENT`,
`CONTRACT_NOT_EXECUTABLE`, `NO_BOUND_SOURCE_SPANS`, `SOURCE_MEDIA_UNAVAILABLE`,
`ANALYSIS_SCOPE_EXCEEDED`, `DETECTOR_UNAVAILABLE`, `CAPTION_EVIDENCE_MISSING`,
`CAPTION_COLLISION_UNRESOLVED`, `BIDI_CONTROL_NEUTRALIZED`,
`TRACKS_RESET_AT_CUT`, and `FALLBACK_APPLIED`; framing, caption, and overlay
modules add their own bounded closed evidence codes.

## Bounded analysis

`analysis.py` is the only module that starts a decoder. It analyzes **only the
union of the selected bound source spans** plus a bounded `scene_context_seconds`
margin (default 0.5 s) on each side. The margin exists solely so scene-cut
detection can see lead-in/lead-out around a cut; it is never emitted in a scene,
keyframe, caption, or output range, and is never sampled as an output frame.

- one FFmpeg child process per span reads raw RGB frames incrementally from a
  pipe; a whole source is never scanned or buffered;
- scene cuts come from one bounded FFmpeg `select='gt(scene,N)'` +
  `metadata=mode=print` pass per span, then `segment_scenes` turns interior cuts
  into contiguous scenes and deterministically merges scenes shorter than
  `min_scene_seconds`;
- only the selected union is checked against `max_analysis_seconds` (600 s); if
  the context-expanded footprint would exceed `max_analysis_frames` (1500) at
  `analysis_fps` (2.0), the effective fps is deterministically reduced, never
  below `min_fps` 1.0, and the reduction is recorded as
  `ANALYSIS_FPS_REDUCED:<from>-><to>` and in the analysis fingerprint; if even
  the floor cannot fit, `AnalysisScopeExceeded` becomes a `BLOCKED` plan with
  `ANALYSIS_SCOPE_EXCEEDED` (never a crash);
- frames are downscaled to `analysis_frame_max_dimension` (640) using FFmpeg's
  `scale='min(640,iw)':-2`; the pinned frame-time mapping is computed by the
  sampler (FFmpeg timestamps are not trusted): `source_time = span_start +
  sample_index / effective_fps`.

"Scope caps" are enforced through the versioned `Stage51Config` and are part of
the plan input fingerprint. Deterministic synthetic fixtures (tiny generated
media and small synthetic frames) are used to validate the sampler, scene
detector, planner, and detector without private media; the real YuNet model test
asserts that the model loads and runs without raising on a synthetic frame
rather than asserting a face.

## Framing modes and deterministic precedence

`framing.py` is pure, CPU-local, and provider-free. Every scene gets exactly one
`FramingDecision` with a closed mode and closed evidence codes.

| Mode | Meaning |
| --- | --- |
| `SOURCE_AS_IS` | Source is already within the 9:16 tolerance; no crop. |
| `STATIC_CROP` | One fixed person crop for the scene. |
| `TRACKED_CROP` | Smoothed, rate-limited path following one persistent face. |
| `MULTI_SUBJECT_FIT` | Two or more persistent faces fit inside one crop. |
| `BACKGROUND_FILL` | Source cannot be cropped safely (taller, screen content, too-wide/small subject); fill rather than cut important content. |
| `CENTER_FALLBACK` | Conservative center crop when no safe decision is possible. |

Selection is short-circuiting in this deterministic order:

1. invalid display geometry -> `CENTER_FALLBACK` (`INVALID_GEOMETRY`); an
   unexpected arithmetic/type/value failure inside selection also degrades to
   `CENTER_FALLBACK` (`FRAMING_FAILURE`);
2. display aspect within 0.05 of 9:16 -> `SOURCE_AS_IS`
   (`SOURCE_ALREADY_VERTICAL`);
3. display aspect narrower than 9:16 (source taller than target) ->
   `BACKGROUND_FILL` (`SOURCE_TALLER_THAN_TARGET`);
4. no stable persistent track -> caller-supplied screen-content evidence yields
   `BACKGROUND_FILL` (`SCREEN_CONTENT`), otherwise `CENTER_FALLBACK`
   (`NO_FACE` or `FACE_TRACK_UNSTABLE`, plus `DETECTOR_UNAVAILABLE` when the
   detector is disabled);
5. largest persistent face height below `face_min_height_fraction` (0.06) ->
   `BACKGROUND_FILL` (`SUBJECT_TOO_SMALL`);
6. two or more persistent faces (considered up to three) -> `MULTI_SUBJECT_FIT`
   when they fit, otherwise `BACKGROUND_FILL` (`MULTIPLE_FACES`,
   `IMPORTANT_CONTENT_WOULD_BE_CROPPED`, optionally `SUBJECT_TOO_WIDE`);
7. one persistent face -> `TRACKED_CROP` when center motion exceeds 0.02, else
   `STATIC_CROP` (`SINGLE_PERSISTENT_FACE`).

Hero scenes add `HERO_PROTECTION_APPLIED` and `protected=true`; hero protection
halves the allowed pan rate and freezes zoom changes larger than a 0.02
tolerance. Framing is per scene; tracks never cross a hard cut, and a cut adds
`TRACKS_RESET_AT_CUT`.

## Crop keyframes and smoothing

Every emitted crop is a clamped 9:16 window represented by a compact normalized
`CropKeyframe` (center + height fraction), never one keyframe per decoded frame.

- static modes emit exactly two identical keyframes (start hold + end hold);
- `TRACKED_CROP` applies deterministic exponential target smoothing
  (`alpha = 1 - exp(-6 * dt)`), a dead zone of 0.18 x crop-height with release
  hysteresis, a maximum pan velocity of 0.55 x crop-height/s eased with a
  smoothstep curve, a maximum zoom rate of 0.15/s, a minimum hold of 0.5 s, and
  a detection-loss hold (default 1.0 s) that falls back to the conservative
  center target;
- crops are clamped (shift, never rescale) fully inside the frame;
- a per-scene keyframe cap of 40 is enforced by removing the keyframe with the
  smallest local path delta and marking the path `KEYFRAMES_SIMPLIFIED`;
- tracked samples are capped at 240 stored per scene by proportional decimation
  while persistence always counts all associated samples.

## Caption architecture

`captions.py` is FINAL_CLIP-only: caption text comes from the exact
`caption_input` word timestamps (the Stage 5.0 `FINAL_CLIP` evidence), and only
words that fall inside a selected bound span become events. A span with no
usable word evidence emits a `CAPTION_EVIDENCE_MISSING` marker and never
fabricates text.

- **Segmentation.** Splits on Latin/Arabic punctuation
  (period, exclamation, question, comma, semicolon, colon, ellipsis, and the
  Arabic comma/semicolon/question-mark/full-stop) or an inter-word pause of at
  least 0.45 s; respects maximum
  event duration 3.6 s, minimum event duration 0.7 s, at most 7 words per
  event (configurable; compact 3-7 word chunks when phrase boundaries allow),
  and script-aware estimated width with greedy wrapping into at most two lines.
  Short fragments are merged when the result still fits.
- **Layout.** Estimated per-character advance (Arabic 0.55 em, alphanumeric
  0.56 em, space 0.28 em, punctuation 0.30 em, other 0.50 em) with a 1.12
  safety factor; usable width is the smaller of 0.86 x 1080 and the frame width
  inside the safe-zone left/right insets.
- **Timing.** Event start is the first word start; end is the last word end,
  at least the minimum duration, capped at the maximum duration and at
  `span.end + 0.30 s`; overlapping drafts are trimmed deterministically.
- **Placement.** Default band is `LOWER` (ASS alignment 2). A scene switches its
  whole set of events to `UPPER` (alignment 8) when the lower band is
  persistently intersected by a protected face box covering more than 15% of
  the band area. Scene-level hysteresis holds `UPPER` until the scene is
  completely collision-free. If both bands collide, the plan records
  `CAPTION_COLLISION_UNRESOLVED` and the hero picks the lesser collision;
  captions are never dropped for a face.
- **Active-word emphasis.** Each caption event carries the exact FINAL_CLIP
  word timings copied verbatim (logical Unicode order). `ass.py` emits one
  stationary Dialogue state per spoken word tiling `[event.start, event.end]`,
  with the word currently being spoken wrapped in a configurable ASS color
  override and every other word left at the style's primary color. The active
  word changes exactly at the word boundary; the whole block keeps the same
  style, alignment, margins, and text, so it never bounces or reflows. If any
  per-word timing is missing, degenerate, non-increasing, starts after the
  event, or shorter than 0.05 s, the event degrades to a single static phrase
  line instead of inventing precise highlighting. There are no karaoke wipes,
  random animations, emoji, content rewriting, provider calls, or synthesized
  timing.

`ass.py` serializes exactly one ASS document per caption plan.

### Overlays

`overlays.py` turns the executable Stage 5.0 materialization slots and ordered
blocks into immutable materialization-required `OverlayRequirement` objects.
Stage 5.1 never invents, renders, or carries publication text: every
requirement has `text=None` and `status=MATERIALIZATION_REQUIRED`. A draft line
carried by a slot is preserved only as a non-authoritative `authoring_reference`
(`draft_only=true`, `authoritative=false`). Placement uses the real closed
Stage 4.1/5.0 vocabulary (`PlanBlockType`, `DeliveryIntent`,
`SourceExcerptRole`, `NarrationPurpose`) and `interrupts_source`; free-form
provider `placement`/`purpose` strings are deliberately not used as closed
inputs. Desired zones are `TOP_HOOK`, `UPPER_THIRD`, `LOWER_THIRD`, and
`CENTER`; conflict resolution priority is protected face > safe zones > caption
stability > the originally desired zone, and every requirement carries
collision constraints (`MUST_NOT_COVER_PROTECTED_FACE`,
`MUST_RESPECT_SAFE_ZONES`, `MUST_NOT_OVERLAP_SCENE_CAPTION_ZONE`, and
`ZONE_CONFLICT_UNRESOLVED_NO_FREE_ZONE` when nothing is viable). Timing is a
block-relative `timeline_hint`; no global timeline is frozen.

## Arabic, English, and mixed BiDi

- Caption text is stored and emitted in **canonical logical Unicode order**: the
  exact source word tokens joined by single spaces, with no reordering,
  replacement, or injected direction controls.
- Shaping and bidirectional layout are delegated to the real renderer:
  FFmpeg's libass `ass` filter (libass with FriBidi and HarfBuzz). Stage 5.1
  never performs manual BiDi rewriting.
- `tests/test_stage51_bidi_render.py` proves this with **real rendered
  regression testing**, not string inspection: it builds an ASS document with
  the production serializer, renders it through the actual libass `ass` filter
  against a lavfi source, decodes the PNG, and asserts ink plus byte
  determinism for Arabic-only, English-only, embedded-English, mixed
  numeric/Arabic, mixed-punctuation, and representative mixed strings. It also
  renders a naive character-reversed document as a negative control and
  asserts the images differ. The same suite proves dynamic active-word
  emphasis: for each fixture it renders every word state of one event at its
  real time, asserts the block bounding box is stationary, asserts that the
  per-word state lines with a no-op color render pixel-identically to the plain
  static render (so the override tags cause no reflow, reorder, or reshaping),
  and asserts the accent mask is present in every state and advances between
  states. The tests skip only when ffmpeg, the libass `ass`
  filter, or a usable installed font is genuinely absent; the project Docker
  test image provides all three.
- Frontend `<bdi>` handling is unrelated to video subtitles and does not solve
  subtitle rendering; Stage 5.0 explicitly requires real ASS/libass validation
  in Stage 5.1, which these tests provide.

## ASS escaping and safety

`ass.py` escapes source text so libass renders it literally, never as override
blocks or drawing commands:

- a source `{` is emitted `\{` and a source `}` as `\}`;
- a backslash is emitted literally, except that when immediately followed by
  `N`, `n`, or `h` a zero-width word joiner is inserted so the pair renders
  literally instead of changing layout;
- Unicode bidi controls U+202A-U+202E and U+2066-U+2069 are removed and recorded
  as `BIDI_CONTROL_NEUTRALIZED`;
- tabs/newlines and U+2028/U+2029 become spaces; other C0/C1 control characters
  are removed.

The document uses `PlayResX: 1080`, `PlayResY: 1920`, `WrapStyle: 2`, and the
`CaptionLower`/`CaptionUpper` styles. The real-render test
`test_escaped_drawing_command_does_not_execute` confirms a hostile
`{\p1}...{\p0}` drawing command does not execute from source text.

## Safe zones and output geometry

The deterministic output profile is 1080 x 1920 (9:16). Safe-zone insets are
fractions of the output frame; pixel getters return rounded integers so FFmpeg
and ASS consume the same numbers. The default `SHORTS_VERTICAL_SAFE_ZONE_V1`
(and the conservative `GENERIC_CONSERVATIVE_SAFE_ZONE_V1` fallback) are:

| Inset | Fraction | Pixels |
| --- | --- | --- |
| top | 0.12 | 230 |
| bottom | 0.25 | 480 |
| left | 0.05 | 54 |
| right | 0.15 | 162 |

`caption_bottom_gap_px` and `caption_top_gap_px` are both 24, so the ASS lower
margin is 504 and the upper margin is 254. The safe-zone key participates in
the input fingerprint through the output profile.

Display geometry is derived from encoded dimensions, rotation (snapped to the
nearest 90 degrees), and pixel aspect ratio; display pixels are treated as
square after rotation and a non-square pixel aspect is recorded as exotic
evidence without distorting the display coordinate space.

## Fonts

`backend/Dockerfile` installs `fonts-noto-core` (a legally distributable
Arabic+Latin family; no proprietary font is bundled). The default
`CLIPFACTORY_VISUAL_CAPTION_FONT_FAMILY` is `Noto Sans Arabic`. The ASS style
uses the configured family; libass resolves it through fontconfig, which falls
back to another Arabic-capable installed family (ultimately `DejaVu Sans` in the
render test) if the exact family is absent. A missing font never fails a plan;
it only affects the eventual renderer.

## Caption style defaults

The default `CaptionStyle` is shared by the real plan ASS and the standalone
BiDi/black fixtures, so a validation fixture cannot look good while the real
composition uses a different size. Defaults: font size 88, white primary text,
black outline width 7, shadow 3, at most 2 lines, at 1080 x 1920 (a 1.25 line
factor, 110 px line pitch). That keeps captions immediately readable on a
360 x 640 preview without covering the subject. Dynamic emphasis is on by
default: the active word uses the accent color `&H0000FFFF` (yellow) while
inactive words stay white. The style (including active color, dynamic-emphasis
enable, and max words per event) is part of the `stage51_config_payload()`
fingerprint, so any value change invalidates prior plans at the correct
boundary. TTS voice/provider/model are never fingerprint inputs.

## Detector

`detector.py` runs the vendored OpenCV Zoo YuNet 2023mar ONNX model through the
already installed `onnxruntime` CPU execution provider. No OpenCV, MediaPipe, or
PIL; no network.

- **Identity (fingerprinted):** model file
  `face_detection_yunet_2023mar.onnx`, sha256
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`,
  fixed input size 640 x 640, score threshold 0.6, NMS IoU 0.3,
  `CPUExecutionProvider`, intra-op threads 2.
- The vendored graph has a **fixed 640x640 input** (empirically a 320x320
  tensor raises `INVALID_ARGUMENT`), which is why the policy default is 640.
  Preprocessing letterboxes RGB to a BGR NCHW blob and postprocessing decodes
  the model's already-sigmoid-activated outputs with a geometric-mean score.
- The model file is sha256-verified before a session is created. A disabled,
  missing, or hash-mismatched detector becomes `NullFaceDetector`: `detect()`
  returns no boxes and `ready()` reports False. Detector unavailability never
  raises out of analysis; it produces `DETECTOR_UNAVAILABLE` and conservative
  fallback framing.
- Detections are **anonymous boxes plus a confidence score only**. Track
  association is deterministic and geometry-only (IoU, then centroid distance,
  then size continuity); a track bridges at most `track_max_gap_samples`
  missing samples and never crosses a scene. There is no recognition,
  embedding, biometric matching, or named-person identification.

## Persistence

One table, `visual_composition_plans`, added by migration
`20260918_0021` (`down_revision=20260918_0020`):

- one row per `(clip_candidate_id, input_fingerprint)`;
- exactly one database-current row per candidate, enforced by a partial unique
  index `uq_visual_composition_plans_current` (`is_current`);
- FKs to `render_contracts`, `transformation_plan_selections`,
  `transformation_plans`, and `candidate_refinements` (all `ON DELETE SET NULL`);
  status/execution-status enums and bool check constraints;
- semantic `status` (`READY_FOR_VISUAL_EXECUTION` / `BLOCKED` / `FAILED`) is
  separate from `execution_status`
  (`QUEUED` / `ANALYZING` / `COMPLETE` / `FAILED` / `CANCELLED`);
- `plan_ready` is true only for `READY_FOR_VISUAL_EXECUTION`;
  `cache_eligible` is true only for `READY_FOR_VISUAL_EXECUTION`.

The migration also adds the `VISUAL_COMPOSITION` job kind and a nullable
`processing_jobs.visual_composition_plan_id` FK (`ON DELETE SET NULL`). It adds
no `PipelineStage`, no `PipelineRun`, and never changes source lifecycle values.
The downgrade removes only Stage 5.1 schema and preserves every Stage 1-5.0 row.

Only `READY_FOR_VISUAL_EXECUTION` plans carry scenes, captions, the ASS asset,
and overlays. `BLOCKED` and `FAILED` plans persist truthful reason codes and
empty evidence and are never cache-eligible.

## Fingerprints and cache

Canonical fingerprints are composed with `canonical_fingerprint`:

- `input_fingerprint` covers candidate identity/currentness/disposition and
  analysis fingerprint; source media identity; the Stage 5.0 contract input and
  output fingerprints plus status/ready/current/live-freshness/effective;
  selected plan, selection, and final refinement identities; display geometry
  and rotation; the ordered bound source spans; the live caption-source
  fingerprint and caption payload; the output profile; the safe-zone key and
  version; every policy/schema/fingerprint/config version and the full versioned
  `stage51_config_payload()`; and detector identity;
- `output_fingerprint` covers scenes, captions, ASS sha256/event/line counts and
  policy version, overlays, and readiness;
- `analysis_fingerprint`, `framing_fingerprint`, `ass_fingerprint`,
  `caption_source_fingerprint`, and `source_media_fingerprint` are separate.

**What invalidates a plan:** any change to the executable Stage 5.0 contract
(status, fingerprints, currentness/effectiveness), source media identity,
display geometry/rotation, bound source spans, live caption source, output
profile, safe zone, Stage 5.1 policy/config/schema (including the caption style,
active-word color/emphasis policy, and chunk size), detector identity, or the
candidate/analysis identity.

**What does NOT invalidate a plan:** TTS provider/model/voice, future narration
audio/text, publishing title/schedule/metadata, codec/encoder settings, final
render artifacts, and analytics configuration. None of those are Stage 5.1
inputs. `build_stage51_input_payload` even accepts an
`excluded_runtime_context` argument purely so call sites can document
Stage 5.1-excluded runtime inputs, and it is never placed in the fingerprint.

Read currentness recomputes the full input fingerprint from live rows using the
persisted geometry (stat-only source identity, never ffprobe, never a provider).
`CURRENT` requires exact equality; a mismatch is `STALE`; a recomputation error
is `UNVERIFIABLE`. `effective` is true only for a current, ready, `CURRENT` row.

## Jobs, concurrency, and cancellation

- Queueing validates prerequisites, reuses a current cache-eligible plan unless
  `force=true`, and creates one durable plan envelope plus one `VISUAL_COMPOSITION`
  `ProcessingJob`. Cache validation is stat-only: `_NoProbe` raises if
  queue-time freshness ever tries to touch media.
- At most one active job per envelope is guaranteed by an atomic
  `active_job_id IS NULL` compare-and-swap; a unique-constraint race on the
  envelope row is recovered inside a savepoint.
- The executor fences duplicate/redelivered Celery invocations with an atomic
  `QUEUED -> RUNNING` claim that advances the durable `claim_version` token;
  the loser sets `skipped_duplicate` and performs no planning work. Liveness is a
  renewable `heartbeat_at`; only a genuinely abandoned claim is reclaimable.
  Every persistence path is fenced on the claim token and `status == RUNNING`.
- Cancellation is cooperative and polled before and after planning and before
  persistence (the planner also polls it during frame sampling); it keeps the
  job and plan `CANCELLED`, never schedules anything, and preserves the row.
- Every exit path releases owned detector/frame-sampler/scene-detector resources
  and cleans temporary files, including cache hits.
- Failed jobs retain bounded, sanitized diagnostics (a repository-owned
  prerequisite message or the exception type only; never media, prompts, keys,
  or payloads).

## API and CLI

API endpoints:

- `POST /api/candidates/{candidate_id}/visual-composition?force=<bool>` - queue
  one candidate-scoped run (202); returns plan/job/queued/cached/active state.
- `GET /api/candidates/{candidate_id}/visual-composition` - current plan with
  `live_freshness` and `effective`.
- `GET /api/visual-compositions/{plan_id}` - one plan row with its own
  current/freshness/effective state.
- `GET /api/candidates/{candidate_id}/stage5-2-handoff` - read-only Stage 5.2
  handoff.
- `POST/GET /api/candidates/{candidate_id}/render-contract` and
  `GET /api/render-contracts/{contract_id}` and
  `GET /api/candidates/{candidate_id}/stage5-1-handoff` are the Stage 5.0
  contract endpoints that remain the required input.

CLI commands:

```bash
python -m app.cli visual-composition CANDIDATE_ID [--force]
python -m app.cli visual-composition-status CANDIDATE_ID
python -m app.cli stage5-2-handoff CANDIDATE_ID
python -m app.cli stage5-1-handoff CANDIDATE_ID
python -m app.cli render-contract CANDIDATE_ID
python -m app.cli render-contract-status CANDIDATE_ID
```

Celery task `clipfactory.run_visual_composition(plan_id, job_id, force)` uses the
existing explicit candidate-scoped job wrapper: it creates no `PipelineRun` and
schedules nothing automatically.

## Operations and preview

Preview rendering (`preview.py`) is **PNG-only validation** through safe FFmpeg
argument arrays. It composes a scaled/cropped background and burns the already
serialized ASS captions onto a bounded set of representative source frames, one
PNG per frame. It never encodes a video: a denylist refuses `libx264`, `libx265`,
`libvpx`, `aac`, `loudnorm`, and `mpeg4` before any FFmpeg call. Previews are
gated by `CLIPFACTORY_VISUAL_PREVIEW_ENABLED`; when disabled, or when FFmpeg is
missing, they raise `PreviewError` and never fall back to video.

`BACKGROUND_FILL` renders the source scaled to cover, then a strong blur plus a
modest dim/desaturation (`gblur=sigma=36:steps=2`, then
`eq=brightness=-0.18:saturation=0.70`) so the background reads as a background
rather than a second copy of the scene, with the sharp contained foreground
overlaid on top. No external imagery or B-roll is ever introduced.

The default preview directory is
`storage/benchmarks/visual-composition/{candidate_id}/{contract_id}` (the
storage `benchmarks/` category). Callers may pass an explicit
`output_directory`. Preview images are workspace-local operational artifacts and
are never committed. There is currently no API or CLI command that triggers
preview rendering; it is a programmatic seam used by validation tests.

## Configuration

Every Stage 5.1 setting is derived from `Settings.stage51_config()` and the full
payload participates in the input fingerprint. All are bounded by the field
constraints shown; see `.env.example` for the comments.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLIPFACTORY_VISUAL_COMPOSITION_ENABLED` | `true` | Stage 5.1 enable flag. When `false`, `validate_candidate_for_composition` and `queue_visual_composition` fail closed with a queue error. |
| `CLIPFACTORY_VISUAL_ANALYSIS_FPS` | `2.0` | Nominal analysis frame rate (reduced deterministically if needed). |
| `CLIPFACTORY_VISUAL_ANALYSIS_MAX_FRAMES` | `1500` | Hard analysis frame budget per plan. |
| `CLIPFACTORY_VISUAL_ANALYSIS_MAX_SECONDS` | `600.0` | Hard selected-span seconds cap. |
| `CLIPFACTORY_VISUAL_ANALYSIS_FRAME_MAX_DIMENSION` | `640` | Max decoded frame dimension. |
| `CLIPFACTORY_VISUAL_SCENE_CUT_THRESHOLD` | `0.35` | FFmpeg scene-change threshold. |
| `CLIPFACTORY_VISUAL_DETECTOR_ENABLED` | `true` | Enable the vendored face detector. |
| `CLIPFACTORY_VISUAL_DETECTOR_MODEL_PATH` | vendored `face_detection_yunet_2023mar.onnx` | Detector model path; sha256-verified. |
| `CLIPFACTORY_VISUAL_DETECTOR_INPUT_SIZE` | `640` | Detector input size; the vendored export is fixed at 640. |
| `CLIPFACTORY_VISUAL_DETECTOR_SCORE_THRESHOLD` | `0.6` | Detection score threshold. |
| `CLIPFACTORY_VISUAL_CAPTION_FONT_FAMILY` | `Noto Sans Arabic` | ASS caption font family. |
| `CLIPFACTORY_VISUAL_CAPTION_MAX_LINES` | `2` | Maximum caption lines per event. |
| `CLIPFACTORY_VISUAL_CAPTION_ACTIVE_COLOR` | `&H0000FFFF` | ASS `AABBGGRR` color for the actively spoken word. |
| `CLIPFACTORY_VISUAL_CAPTION_DYNAMIC_EMPHASIS` | `true` | Highlight the actively spoken word; false renders plain static captions. |
| `CLIPFACTORY_VISUAL_CAPTION_MAX_WORDS_PER_EVENT` | `7` | Maximum words per caption chunk. |
| `CLIPFACTORY_VISUAL_PREVIEW_ENABLED` | `true` | Allow PNG validation previews. |

## Versions

- policy `stage5.1-v1`
- schema `stage5.1-schema-v1`
- fingerprint `1`
- framing policy `stage5.1-framing-v1`
- caption layout policy `stage5.1-caption-layout-v3`
- ASS policy `stage5.1-ass-v3`
- background fill policy `stage5.1-background-fill-v1`
- safe-zone profile `shorts-reels-safe-zone-v1`
- output geometry `1080 x 1920` (9:16; `SHORTS_1080X1920` is the Stage 5.0
  render profile whose contract Stage 5.1 consumes)

## Validation

```bash
# focused Stage 5.1 (default)
docker run --rm -e PYTHONPATH=/app -v "$PWD/backend:/app" -w /app oac-backend-test \
  python -m pytest tests/ -k stage51 -q

# PostgreSQL-gated migration test only
docker run --rm --network oac_default \
  -e PYTHONPATH=/app \
  -e CLIPFACTORY_TEST_POSTGRES_URL=postgresql+psycopg://clipfactory:clipfactory@postgres:5432/clipfactory \
  -v "$PWD/backend:/app" -w /app oac-backend-test \
  python -m pytest tests/test_stage51_migration.py -q

# formatting and lint
ruff format app tests alembic && ruff check app tests alembic
```

Full backend result: **1608 passed, 9 skipped** in Docker Python 3.12 (mount the
repository root, not only `backend/`, so the couple of tests that read
`compose.yaml` resolve). The
PostgreSQL-gated suites (migration, models, and the real live-contract
end-to-end test) pass **11 passed, 1 skipped** against a disposable PostgreSQL;
the live-contract run exercises migration → seeded selection/FINAL_CLIP rows →
real `create_render_contract` on real media → queue → executor with the real
FFprobe/FFmpeg/YuNet seams → Stage 5.2 handoff → faithful preview PNGs, and
asserts a cache hit on re-queue and `STALE` after a media stat change. The real
libass render tests run in the project image, which ships `ffmpeg` and
`fonts-noto-core`; `Noto Sans Arabic` resolves and the BiDi renders were
re-verified under it. Preview rendering composes the plan's real framing
(interpolated crop keyframes per mode, planned contain-over-blurred-self for
`BACKGROUND_FILL`, bounded scale/pad for `SOURCE_AS_IS`) rather than a fixed
center crop. No test makes a live provider, network, TTS, or final-render call.

A scene that has face detections but no track meeting the persistence threshold
now uses `BACKGROUND_FILL` (keeping every subject visible); `CENTER_FALLBACK` is
reserved for zero-detection/no-evidence scenes and invalid geometry. FFmpeg
input seeking (`-ss` before `-i`) was verified byte-identical to output seeking
at 66/68/80 s on the AV1 validation source, so seeking is not a limitation.

Known limitations: preview rendering has no API/CLI entry point. Stage 5.2 and
Stage 6 are
not implemented, and a Stage 5.1 plan is not a rendered video, a transcript, or
publishing readiness.
