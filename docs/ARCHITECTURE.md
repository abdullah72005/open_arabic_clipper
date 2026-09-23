# Architecture

The browser dashboard talks to FastAPI. FastAPI persists sources, jobs, and
pipeline runs in PostgreSQL, then schedules heavyweight work through Celery.
Redis is the Celery broker/result backend. A worker has concurrency one and
late acknowledgements so local media work is conservative and retryable.

`StorageService` is the sole application owner of filesystem paths below
`CLIPFACTORY_STORAGE_ROOT` (mounted as `./storage` in Compose). It creates
per-source folders, validates paths, checks capacity, and writes atomically.
The backend image uses Python 3.12 and installs FFmpeg/ffprobe from Debian.

Compose services are `postgres`, `redis`, `backend`, `worker`, `frontend`, and
the optional `ollama` service under the `reconstruction` profile. Postgres and
Redis use named volumes; source media uses the visible local `storage/` mount so
an operator can inspect or back it up. Ollama stores models in the
`ollama_models` volume and exposes no public port.

Stage 3 lives in `app/candidates/`: deterministic proposal generation, content
classification, scoring/uncertainty, hooks, novelty, a Stage 3-specific semantic
provider boundary (deterministic/adaptive-Gemini/local-Qwen), fingerprints, and
a durable `CandidateAnalysisExecutor`. The runner and worker add a
`CANDIDATE_ANALYSIS` stage and job kind; `cancellation` is generalized from the
Stage 2.7 reconstruction marker without changing Stage 2.7 behavior.
`candidate_analyses` and `clip_candidates` persist the result, and small API/CLI
surfaces expose provenance, queueing, and bounded inspection. There is no
separate queue, service process, orchestration subsystem, or analytics engine.

## Stage 4.0 transformation eligibility

Stage 4.0 lives in `app/transformation/` and is explicit, candidate-scoped work
after a candidate has a usable Stage 3.5 refinement. It decides whether a
credible substantive transformation path exists, produces at most three
recommended (and three useful rejected) strategy directions for Stage 4.1, and
returns `NO_TRANSFORMATION_STRATEGY_WORTH_USING` as a normal successful outcome
when no natural strategy clears the hard gates. It extends the existing
Celery/`ProcessingJob` platform with a `TRANSFORMATION_ELIGIBILITY` job kind and
a nullable `processing_jobs.transformation_analysis_id` FK. It adds **no**
`PipelineStage`, **no** `PipelineRun`, **no** `_NEXT_STAGE` entry, and no source
lifecycle change. `transformation_eligibility_analyses` (one current row per
candidate) and `transformation_strategy_candidates` (one current row per
`(analysis, strategy_type)`) persist the bounded result.

Deterministic logic owns prerequisites, transcript/context sufficiency,
transformation necessity, source-moment structure, content-to-strategy
suitability, presentation-only zero credit, hard gates, validation, final
eligibility, safe fallback, and deterministic ranking. An optional Stage
4-specific provider (hosted Gemini routine/strong tiers or an explicit
`local_only` Qwen) may only assess candidates that survived those gates, runs at
most one hosted call per analysis through the shared HIGH admission gate, and
never overrides a hard blocker. Rights/provenance risk and
originality/transformation risk stay separate; the platform-risk snapshot is
decision support, never a legal or monetization guarantee. Stage 4.1 receives a
typed read-only handoff only.

## Stage 4.1 transformation plan generation

Stage 4.1 lives in `app/transformation/planning/` and is explicit,
candidate-scoped work after a current, non-stale Stage 4.0 analysis with at least
one current recommended strategy. It consumes only that handoff, accepts a usable
`CANDIDATE` Stage 3.5 refinement (never requires `FINAL_CLIP`), and produces zero
to three concrete validated plans — normally at most one per current recommended
strategy. It does not select, approve, authorize, or mark a winning plan, and it
adds **no** `PipelineStage`, **no** `PipelineRun`, **no** `_NEXT_STAGE` entry,
and no source lifecycle change. It extends the existing Celery/`ProcessingJob`
platform with a `TRANSFORMATION_PLANNING` job kind and a nullable
`processing_jobs.transformation_plan_set_id` FK. `transformation_plan_sets` (one
durable planning envelope per candidate) and `transformation_plans` (one stable
row per `(plan_set, strategy_candidate)`) persist the result; a zero-plan,
deferred, degraded, or provider-unavailable plan set is recorded truthfully
without fake plan rows. Bounded structured blocks are stored as validated JSON on
the plan rather than a block table.

Deterministic logic owns readiness, input bounds, routing, source-span
resolution, hero placement, duration arithmetic, substantive-value validation,
paraphrase/cosmetic rejection, narration/TTS separation, verification dependency
enforcement, material distinction, persistence eligibility, and cache/fingerprint
composition. An optional planning provider (hosted Gemini routine/strong tiers or
an explicit `local_only` Qwen) may only turn an already-approved strategy into a
concrete plan; the provider selects source spans by indexed word references or a
full-window sentinel and never supplies timestamps, quotes, TTS choices, or
rendering instructions. Hosted calls are batched per tier (at most one call per
tier and two per plan-set run) through the shared HIGH admission gate with
temperature 0 and strict structured validation. A missing key/outage/429/quota/
safety refusal/malformed output never fails the source: accepted per-strategy
checkpoints survive, only unfinished work stays non-cache-eligible, and a later
normal request retries it. Narration is an abstract semantic requirement; Stage
4.1 decides WHAT narration communicates, channel configuration will decide WHO
the persistent narrator is, and Stage 6 generates speech.

## Stage 4.2 retention, originality, and platform-risk governance

Stage 4.2 lives in `app/transformation/governance/` and is explicit,
candidate-scoped work after a current, non-stale, complete Stage 4.1 plan set
with at least one current plan. It independently governs every current plan and
answers whether it is good and safe enough to be considered by Stage 4.3. It is
a critic/governor: it never generates, mutates, repairs, or selects a plan, adds
**no** `PipelineStage`, **no** `PipelineRun`, **no** `_NEXT_STAGE` entry, and no
source lifecycle change. It extends the existing Celery/`ProcessingJob` platform
with a `TRANSFORMATION_GOVERNANCE` job kind and a nullable
`processing_jobs.transformation_governance_set_id` FK. `transformation_governance_sets`
(one durable envelope per Stage 4.1 plan set) and
`transformation_governance_results` (one stable row per `(set, plan)`) persist
independent per-plan results without touching the immutable Stage 4.1 plan.

Deterministic logic owns integrity revalidation, evidence derivation, independent
categorical dimensions (no overall score), hard gates, status precedence,
platform-risk interpretation, reason/remediation codes, and fingerprint
composition. Each plan becomes `APPROVED_FOR_SELECTION`,
`APPROVED_WITH_CAUTION`, `BLOCKED_PENDING_VERIFICATION`, `REVISION_REQUIRED`,
`REJECTED_BY_GOVERNOR`, or `GOVERNANCE_DEFERRED`, with an explicit
`eligible_for_stage4_3` boolean that is a filter, not a ranking. Semantic
fidelity, presentation-only transformation, fabricated critical claims, and
misleading hooks are hard failures no other dimension offsets. An optional
provider (hosted Gemini routine/strong, or explicit `local_only` Qwen) supplies
only bounded observable semantic findings and can never assign a final status or
platform classification. Every raw hosted call acquires the shared HIGH
admission gate; at most two raw calls per governance run. Accepted per-plan
critiques are checkpointed per provider-input fingerprint and reused on exact
dependency match. The candidate summary is `PLANS_ELIGIBLE_FOR_SELECTION`,
`NO_GOVERNOR_APPROVED_PLAN`, or `GOVERNANCE_DEFERRED`. Rights, platform
originality, and spam/repetition stay separate; account/channel repetition is
deferred to Stage 7. The read-only Stage 4.2 -> 4.3 handoff contains no winner,
`selected_plan_id`, render-ready state, or publication approval; deterministic
selection is implemented separately by Stage 4.3 (below). See
`docs/STAGE_4_2_OPERATIONS.md`.

## Stage 4.3 deterministic final-plan selection

Stage 4.3 lives in `app/transformation/selection/` and is explicit,
candidate-scoped, synchronous, transaction-safe, and provider-free. It consumes
the authoritative `build_stage4_3_handoff()` freshness contract plus the current
Stage 4.1 plan rows and commits each candidate to exactly zero or one current
survivor. It never replans, re-governs, rewrites, researches, refines
transcripts, or renders; it adds **no** Celery task, `ProcessingJob` kind,
queue/executor, `PipelineStage`, `PipelineRun`, or `_NEXT_STAGE` entry.

`policy.py` owns the closed statuses (`PLAN_SELECTED`,
`PLAN_SELECTED_WITH_CAUTION`, `NO_SELECTABLE_PLAN`, `SELECTION_DEFERRED`,
`STALE_SELECTION_INPUT`), the conservative caution allowlist, resolved
verification states, explicit categorical order maps, and the lexicographic
comparison vector (no weighted aggregate score). Only `VERIFIED_CURRENT`
governance may select; clean approvals arbitrate before cautions; internally
inconsistent approved evidence fails closed as deferred and is never repaired.
`fingerprints.py` canonically covers candidate/stage40/stage41/stage42 identity,
refinement identity, every current plan and governance result, and the Stage 4.3
policy, while excluding TTS voice/provider/model, render configuration, and
publishing configuration. `service.py` performs verdict extraction, arbitration,
and transactional persistence; `handoff.py` exposes a read-only execution handoff
with live readiness (`READY_FOR_FINAL_REFINEMENT`,
`REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK`, `READY_FOR_EXECUTION_PREP`,
`BLOCKED`) that is never persisted.

Persistence is one new table, `transformation_plan_selections` (one row per
candidate + input fingerprint, nullable `selected_plan_id`, one database-current
row per candidate via a partial unique index). Concurrent POSTs converge through
candidate-row locking, savepoint uniqueness recovery, and post-lock freshness
revalidation. The selected Stage 4.2 evidence is snapshotted immutably because
governance rows may be refreshed later; Stage 4.1 plans and Stage 4.2 rows are
never mutated. Selection is not render or publication readiness, and narration
remains semantic only. See `docs/STAGE_4_3_OPERATIONS.md`.

## Stage 5.0 execution preflight and render contract

Stage 5.0 is a new deterministic, provider-free `app/render` package that turns
the current Stage 4.3 selection into one durable, evidence-bound execution/render
contract per candidate/input fingerprint. It is synchronous and
transaction-safe: no Celery task, `ProcessingJob` kind, queue/executor,
`PipelineStage`, `PipelineRun`, or `_NEXT_STAGE` entry. It never renders,
generates content, captions, tracks faces, synthesizes speech, or publishes.

`policy.py` owns the versions, closed statuses/outcomes/reason codes,
tolerances, protected semantic operators (negation/exclusivity/modality, English
+ Arabic), contraction expansions, filler tokens, the `SHORTS_1080X1920` render
profile registry, and `Stage50Config`/`stage50_config_payload()`. `types.py`
holds frozen value objects (`FinalClipEvidence`, `BoundSourceSpan`,
`ContractBlock`, `MaterializationSlot`, media identity/facts, `ContractDraft`).
`compatibility.py` evaluates the current `FINAL_CLIP` deterministically against
the frozen Stage 4.1 plan excerpts (bounded alignment, wording/entity/number/
operator comparison, timing and boundary bands, payoff/hook and grounding-quote
coverage, meaning-critical unresolved spans) under a fixed outcome precedence.
`binding.py` rebinds each `SOURCE_EXCERPT` to current `FINAL_CLIP` word evidence
without ever mutating the plan. `media.py` performs stat-only managed-source
identity and one bounded read-only ffprobe probe through an injectable seam with
cached reuse. `fingerprints.py` composes canonical input/output/caption-source/
media-identity/probe fingerprints that exclude TTS voice/provider/model, caption
font/animation, crop, B-roll, codec tuning, publishing metadata, and final render
artifact hashes. `service.py` runs preflight, assembles the contract, and
persists/reuses it; `handoff.py` exposes the read-only Stage 5.1 handoff with
`stage5_1_implemented=false`, `stage5_2_implemented=false`,
`stage6_implemented=false`.

`READY_FOR_RENDER_PLANNING`/`MATERIALIZATION_REQUIRED` persist a non-empty
contract; every other status persists a reusable, non-executable preflight row.
Persistence is one new table, `render_contracts` (unique candidate + input
fingerprint, one database-current row per candidate via a partial unique index,
`contract_ready`/status coupling check), added by migration `20260918_0020`.
Stage 5.0 performs no BiDi manipulation and generates no subtitles; **Stage 5.1
must add real rendered ASS/libass mixed Arabic–English caption regression
tests**. See `docs/STAGE_5_0_OPERATIONS.md`.

## Stage 5.1 visual composition, framing, and caption plan

Stage 5.1 is a new deterministic, CPU-local, provider-free `app/composition`
package that turns one current executable Stage 5.0 render contract into one
durable visual-composition plan per candidate/input fingerprint. It is explicit
and candidate-scoped and extends the existing Celery/`ProcessingJob` platform
with a `VISUAL_COMPOSITION` job kind, an executor, and a queue; it adds no
`PipelineStage`, no `PipelineRun`, and no `_NEXT_STAGE` entry. It never produces
a final render, encodes a video, mixes audio, synthesizes speech, publishes, or
advances the source lifecycle.

`policy.py` owns the versions, closed framing/evidence/status/reason enums, the
safe-zone profiles, caption style, detector identity, and `Stage51Config` /
`stage51_config_payload()`. `types.py` holds frozen value objects
(`DisplayGeometry`, `FaceDetection`, `FaceTrack`, `CropKeyframe`, `Scene`,
`CaptionEvent`, `OverlayRequirement`, `PlannerInputs`,
`VisualCompositionPlan`). `analysis.py` is the only decoder: it samples only the
union of selected bound spans plus a bounded cut-alignment margin with one FFmpeg
child per span, computes a bounded fps-reduced analysis plan, and detects scene
cuts per span. `geometry.py` does rotation/pixel-aspect display math and one
bounded read-only ffprobe probe. `detector.py` runs the vendored, sha256-verified
OpenCV Zoo YuNet ONNX model through the installed `onnxruntime` CPU provider and
never recognizes or identifies people. `tracking.py` builds anonymous
per-scene tracks by deterministic geometry-only association. `framing.py`
selects the per-scene mode by deterministic precedence and builds compact,
clamped, smoothed crop keyframes. `captions.py` builds FINAL_CLIP-only caption
events with segmentation/layout/timing and scene-level safe-zone placement.
`ass.py` serializes exactly one escaped ASS document. `overlays.py` derives
materialization-required overlay placements. `planner.py` orchestrates the plan;
`fingerprints.py` composes canonical input/output/analysis/framing/ASS/
caption-source/media fingerprints that exclude TTS, publishing, codec, final
render, and analytics inputs; `service.py` resolves inputs, runs the planner, and
persists/reuses the plan; `queue.py` and `executor.py` provide the durable job
platform; `preview.py` renders PNG-only validation previews; `handoff.py`
exposes the read-only Stage 5.2 handoff.

Persistence is one new table, `visual_composition_plans` (unique candidate +
input fingerprint, one database-current row per candidate via a partial unique
index), plus the `VISUAL_COMPOSITION` job kind and a nullable
`processing_jobs.visual_composition_plan_id` FK, added by migration
`20260918_0021`. `READY_FOR_VISUAL_EXECUTION` plans carry scenes, captions, the
ASS asset, and overlays; `BLOCKED`/`FAILED` plans persist truthful reason codes
and are never cache-eligible. Shaping of mixed Arabic/English/numbers captions
is delegated to the real libass/FriBidi/HarfBuzz renderer with real rendered
regression tests, never manual BiDi rewriting. See
`docs/STAGE_5_1_OPERATIONS.md`.

