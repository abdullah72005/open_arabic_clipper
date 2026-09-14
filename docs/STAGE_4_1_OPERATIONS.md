# Stage 4.1 operations: transformation plan generation

Stage 4.1 turns each current recommended Stage 4.0 strategy direction into zero
to three concrete, validated structured transformation plans. It preserves the
strongest source moment as an early hero, requires every original block to add
something the raw excerpt did not, and never selects or approves a winning plan.
A zero-plan, deferred, or provider-unavailable result is a **normal successful
semantic outcome**, not a pipeline failure.

## Position in the funnel

```
Stage 3 candidate
  -> Stage 3.5 audio-verified CANDIDATE refinement
  -> explicit Stage 4.0 eligibility + recommended strategy directions
  -> explicit Stage 4.1 concrete structured transformation plans
  -> future Stage 4.2 retention/originality governor
  -> future Stage 4.3 final plan selection
```

Stage 4.1 is explicit, candidate-scoped work after a current, non-stale Stage 4.0
analysis with at least one current recommended strategy. It is **not** in the
automatic `_NEXT_STAGE` chain, never advances every source, adds no
`PipelineStage`, no `PipelineRun`, and no source lifecycle change, and never
requests `FINAL_CLIP` refinement. It extends the existing Celery/`ProcessingJob`
platform with a `TRANSFORMATION_PLANNING` job kind plus a nullable
`processing_jobs.transformation_plan_set_id` FK (FK `ON DELETE SET NULL`).

## Input gate and bounded inputs

Queueing requires all of:

1. a current retained `CANDIDATE`/`CANDIDATE_NEEDS_REFINEMENT` candidate;
2. a usable Stage 3.5 refinement (`CANDIDATE` is accepted; `FINAL_CLIP` is never
   required);
3. a current, non-stale Stage 4.0 handoff with `ready_for_stage4_1=true` and at
   least one current recommended strategy.

A stale Stage 4.0 analysis is a prerequisite conflict, never permission to mix
current transcript data with old strategies. The selected analysis and
refinement are reloaded by persisted identity, and the exact nearby context is
recovered through Stage 4.0's own bounded input assembly. Only bounded evidence
is assembled: the effective refined transcript, indexed Stage 3.5 word
timestamps, bounded nearby context, Stage 3 evidence, the Stage 4.0 source
moment, aggregate assessments, platform risk, preservation and verification
evidence, dialect/entities/unresolved/code-switch evidence, provenance/risk, and
the target/narration semantic context. A whole video or whole multi-hour
transcript is never included, and Stage 4.0 eligibility/ranking is never
recomputed.

## Plan-set schema

`transformation_plan_sets` — one durable planning envelope per candidate:

- `id`, `source_video_id`, unique `clip_candidate_id`, unique
  `transformation_analysis_id`, `refinement_id`, refinement priority/quality;
- `execution_status` (`QUEUED`/`PLANNING`/`COMPLETE`/`PROVIDER_DEGRADED`/
  `FAILED`/`CANCELLED`);
- `planning_outcome` (`PLANS_GENERATED`,
  `PLANS_GENERATED_WITH_VERIFICATION_REQUIRED`, `PLANNING_DEFERRED`,
  `NO_VALID_PLAN_FROM_STRATEGY`, `PROVIDER_UNAVAILABLE`) and bounded
  `outcome_reasons`;
- Stage 4.0 input/output identity snapshot (`stage40_snapshot`);
- target/audience and narration-policy semantic snapshot (`target_context`);
- provider mode, stable configured identity, status, sanitized evidence;
- per-strategy attempt/checkpoint state (`strategy_attempts`);
- input/output fingerprints, policy/schema/validation versions, cache
  eligibility, `active_job_id`, bounded metrics, processing duration, timestamps.

Execution lifecycle and semantic outcome are deliberately separate concepts. A
no-plan, provider-unavailable, or deferred plan set may be recorded as a
successful processing job while truthfully recording the semantic result.

## Plan schema

`transformation_plans` — one stable row per
`(plan_set_id, transformation_strategy_candidate_id)`:

- UUID, plan-set/strategy FKs, `plan_key`, `is_current`, `status`
  (`PLAN_GENERATED`/`PLAN_GENERATED_WITH_VERIFICATION_REQUIRED`),
  `generation_rank` (never an approval rank), strategy type/intensity, Stage 4.0
  strategy fingerprint/snapshot, source dialect snapshot, target audience/
  language/register intent;
- ordered structured `blocks`, hero block index, hero source start/end,
  hero appearance time, preservation constraints, original-value kinds/reasons,
  narration need/requirements, external-fact dependencies, required context,
  derived durations and source/original balance, hook/payoff preservation
  evidence, degraded/fallback rules, Stage 4.0 risk/assessment snapshot,
  planner confidence, generation origin, sanitized planning-provider evidence,
  per-strategy provider-input and plan-output fingerprints, policy version,
  timestamps.

Only validated generated plans are persisted. `NO_VALID_PLAN`, transient
failure, omission, or malformed output are recorded in the plan set's bounded
per-strategy attempt data rather than as fabricated empty plan rows. Repeated
identical work reuses the plan UUID; upstream changes recompute the semantic
fingerprint without creating duplicates.

## Block contract

Closed block set: `SOURCE_EXCERPT`, `ORIGINAL_VALUE`, `TRANSITION`,
`TEXTUAL_ANNOTATION`, `FACT_VERIFICATION_PLACEHOLDER`. Blocks are strict typed
value objects serialized to validated JSON on the plan (no block table).

Each block has a stable index, type, purpose, estimated duration, placement,
whether it interrupts source, preservation constraints, dependency IDs, and the
fields required for its type.

- `SOURCE_EXCERPT`: source role (`HERO`/`HOOK`/`PAYOFF`/`SUPPORT`), indexed word
  references when available, deterministically resolved source-time start/end,
  source text derived from Stage 3.5 evidence (never trusted from provider text),
  and a continuity rationale.
- `ORIGINAL_VALUE`: a `SubstantiveValueKind`, a specific semantic intent, why the
  block adds information unavailable from the raw excerpt, grounding references
  to Stage 4.0 strategy/context/evidence, delivery intent (on-screen text /
  narration / flexible), and an optional short evaluation-only draft line
  (`draft_only=true` when present).
- `TEXTUAL_ANNOTATION`: substantive explanatory/analytical text only. Captions,
  labels repeating speech, emojis, borders, crops, and B-roll are not
  substantive annotations.
- `TRANSITION`: non-substantive, tightly duration-bounded, zero originality
  credit.
- `FACT_VERIFICATION_PLACEHOLDER`: asserts no external fact; names the claim/
  dependency, why it is needed, and its intended use;
  `must_verify_before_execution=true`; links to real dependent content by the
  integer block indexes of dependent substantive blocks. Those blocks carry the
  placeholder's `claim_dependency` value in `dependency_ids`, or the narration
  references it through `verification_dependency_ids`. Bogus, nonexistent, or
  unlinked references are rejected.

One giant script, finished voiceover, shot list, frame timeline, FFmpeg
operation, or publication-ready copy is never persisted.

## Source hero and span rules

The provider never invents source timestamps or quotes. It selects source spans
by start/end word index into a bounded indexed list of Stage 3.5 word timestamps,
or by a full-refined-window sentinel when word evidence is unavailable. Actual
timestamps and excerpt text are resolved and persisted deterministically.

- Every valid plan contains at least one source excerpt; exactly one is the hero.
- The hero is block 0 or 1 and lies within the selected refinement bounds.
- Spans are positive, ordered, bounded, and snapped to actual word evidence when
  present; invalid indices or out-of-window spans reject only that provider item.
- When word timing coverage is insufficient, only a safe full-window excerpt is
  allowed.
- Every source excerpt must meet `min_source_excerpt_seconds` (0.4 s); a
  near-zero-duration token excerpt, including the hero, is rejected.
- Plan-local excerpts never mutate Stage 3.5 bounds or transcripts.
- Partially overlapping source spans are rejected, not only exact duplicates;
  chronological, distinct, non-overlapping excerpts are preserved.
- The hero must actually appear early: the cap uses **true elapsed block
  duration** before the hero — including any preceding source `SUPPORT` excerpt
  — at 3 s, with a stricter 1.5 s cap for short, dense, joke, or payoff-first
  moments. A long source-support preamble followed by the hero at block 1 is
  rejected. The authored-material cap is retained as an additional protection,
  and a 10–20 s preamble is rejected.
- Blocks per plan are capped at eight and plans at three. Duration/config limits
  are versioned and fingerprinted.
- Hero appearance time (the true elapsed duration before the hero), source/
  original/narration duration totals, and ratios are computed deterministically
  and persisted with elapsed/authored evidence for audit; provider arithmetic is
  never trusted.

Valid structures include source-first, a very short original frame followed by
source, source/counterpoint/source/synthesis, and mostly uninterrupted source
followed by a concise value block. No universal arrangement is hard-coded.

## Original-value validation

Every substantive original block must answer: *what does the viewer learn or
understand here that the source excerpt alone did not provide?* If the answer is
absent, the block is rejected. Deterministic Arabic/English-aware checks reject:

- empty or generic semantic intent;
- known fake-hook and filler markers;
- presentation-only terms used as the claimed contribution;
- paraphrase scaffolding such as "he is saying…", "what she means is…", "in
  other words…", "يعني", "ما يقوله هو…";
- high lexical containment/similarity to the source with no added dimension;
- a draft line that merely repeats the source;
- unsupported facts presented as true, or claims implying verification already
  happened;
- source distortion;
- a stated value kind inconsistent with the approved Stage 4.0 strategy.

Presentation-only changes always receive zero substantive credit and can never
make an otherwise repost-like plan valid. Deeper plan-quality judgment is
deferred to Stage 4.2; no embeddings, extra model, or subjective optimization
loop is used.

### Hard TTS/rendering/evasion boundary

Every provider-controlled free-text field that can persist into a plan is also
checked against bounded, explicit policy markers: purpose, semantic intent,
why-unavailable, draft line, continuity rationale, preservation constraints,
verification rationale, intended use, grounding refs, and narration language/
register. Rejected output includes TTS provider/model/voice selection, speaker
identity, frame-level rendering instructions (`ffmpeg`, timeline, shot list,
storyboard, keyframe, render instructions), cosmetic-only transformation claims
(mirroring, pitch shifting, speed tricks, watermark removal/obfuscation), and
platform-detection/copyright-evasion tactics. Markers are explicit phrases, so
ordinary semantic wording such as "model" in "explain the model" is never
blocked. Legitimate narration semantics (purpose, language, register, duration,
placement, verification dependency) are preserved.

## Narration abstraction (future TTS contract)

Narration is an abstract semantic requirement only. Need states: `NONE`,
`OPTIONAL`, `RECOMMENDED`, `REQUIRED`. Purposes: `CONTEXT`, `ANALYSIS`,
`COUNTERPOINT`, `EXPLANATION`, `TAKEAWAY`, `HOOK`, `TRANSITION`. A requirement
may include purpose, language, register, estimated duration, placement/block
reference, maximum allowed source interruption, whether it overlaps source audio,
whether it replaces silence, whether it is essential, and verification
dependency IDs.

- Narration is never automatically required; funny/source-led plans normally use
  `NONE`; a `NONE` plan is fully valid.
- If removing an optional narrated block would remove the plan's only
  substantive contribution, that narration cannot be labeled optional: the plan
  must mark it essential/required or be rejected.
- A substantive block that requires narration delivery while `NarrationNeed.NONE`
  is recorded is rejected, and a narration-disallowed context cannot accept an
  essential narration-only contribution.
- Narration cannot replace or dialect-shift quoted source speech; no speech is
  synthesized.

Stage 4.1 **does not select**: TTS provider, TTS model, TTS voice ID, speaker
identity, random per-video voice, or a Gemini TTS voice name. Provider/model
fields describing the Stage 4.1 planning model appear only in planning-provider
evidence, never in narration/TTS semantics.

- Stage 4.1 decides **what** narration communicates.
- Stage 6 decides **how** speech is generated.
- Channel configuration decides **who** the persistent narrator sounds like
  (provider, model, voice, style, language policy, pacing, fallback voice).

There is no channel/account/target-market persistence model and no
`ChannelTTSConfig`; only a typed resolver seam (`PlanningContextResolver`)
projects planning-semantic context from a mapping when a future channel
configuration exists. Planning fingerprints include only narration-enabled and
language/register policy, and exclude provider/model/voice/fallback/rendering/
speech-engine settings.

## Target market versus source dialect

Planning context defaults are neutral: target market `UNSPECIFIED`, output
language policy `SOURCE_LANGUAGE`, register intent `SOURCE_COMPATIBLE`, narration
allowed. Target-market context may affect comprehension assumptions,
original-block framing, output language, and narration register. It must never
alter the source transcript, quoted speech, or source dialect. For example, an
Egyptian source with a future GCC target keeps Egyptian source excerpts
byte-for-byte from Stage 3.5 evidence while an original narration requirement may
request broadly understandable Arabic; it never fabricates Gulf slang or
relabels the source dialect. Changing semantic target market/language/register
invalidates the plan input fingerprint; changing TTS voice/provider/model does
not.

## Factual verification

A strategy marked `REQUIRES_EXTERNAL_FACT_VERIFICATION` must yield at least one
concrete dependency. The planning provider creates a structured placeholder that
names the claim/dependency and why it is needed. Linkage is deterministic and
persisted: the placeholder lists the integer block indexes of dependent
substantive blocks, those blocks carry the placeholder's `claim_dependency` value
in `dependency_ids`, and narration may depend on a claim through
`verification_dependency_ids`. Bogus, missing, nonexistent, or unlinked
references are rejected; the plan status becomes verification-required; the
dependency blocks execution until verified. `must_verify_before_execution=true`
is preserved. No placeholder text may imply verification already happened. There
is no external browse/search/research subsystem, no Gemini search grounding, and
no automatic web request. External facts are never supplied by the provider.

## Gemini behavior

Official Gemini documentation was checked **2026-09-14** against the installed
`google-genai` SDK (2.x, `generate_content` structured-output surface):

- Gemini API models: <https://ai.google.dev/gemini-api/docs/models>
- Structured output: <https://ai.google.dev/gemini-api/docs/structured-output>

Configured:

- routine model `gemini-3.5-flash-lite`, strong model `gemini-3.8-flash`;
- API version `v1`; temperature `0`; strong thinking level `low`;
- bounded output tokens suitable for at most three eight-block plans;
- shared Gemini admission priority `HIGH`.

Routing is deterministic: strong only for genuinely complex strategy work
(`ANALYSIS`, `COUNTERPOINT`, `COMPARISON`, `NEWS_CONTEXT`, `DEBATE_CONTEXT`,
`CLAIM_CONTEXT_CONCLUSION`, `STRONG` intensity, or an external
verification-dependent argument). Third-party provenance alone never forces
every plan onto the strong model.

Requests are batched only within one candidate: pending strategy requests are
grouped by routine versus strong, at most one call per tier and at most two
hosted calls per plan-set run, never across unrelated candidates; at most the
three approved current strategies are included. Each strategy item is parsed
independently, so one malformed item never invalidates accepted siblings. The
two-call ceiling is a hard **raw** `generate_content` budget: Stage 4.1 performs
no per-tier retry inside it, the provider refuses a third raw call, and persisted
metrics report actual raw hosted calls (`hosted_raw_calls`) in addition to outer
tier invocations (`routine_calls`/`strong_calls`).

Provider output must identify the exact requested Stage 4.0 strategy ID/key.
Unknown, rejected, stale, duplicate, or omitted strategy identities are invalid
(an explicit `NO_VALID_PLAN` with a bounded reason is a terminal, cacheable
semantic result; mere omission or malformed output is not).

The system instruction prohibits long generic introductions,
paraphrase-as-originality, fabricated facts, rewritten source speech, source
dialect shifting, publication-ready full scripts, voice/provider/model
selection, frame-level rendering instructions, final-plan approval, and
platform-detection evasion.

Gemini clients are lazy, constructed only at first call, closed on every exit
path, and the key is scrubbed on release. Credentials, transcript content,
prompts, and remote response bodies never appear in errors or logs. Gemini calls
are skipped on cache hits and when all current strategies are safely
deterministic.

## Qwen

`CLIPFACTORY_LOCAL_QWEN_ENABLED=false` by default. `adaptive` planning uses
Gemini or safe deterministic behavior and never silently falls back to Qwen.
`local_only` uses Qwen/Ollama only when explicitly enabled and never calls
Gemini. There is no hidden model load; local inference uses the shared
heavy-model lease; local failures degrade safely; Qwen cannot claim audio
recovery or external fact verification, and it uses a separate planning prompt/
schema (never the Stage 4.0 directions-only output).

## Provider degradation

Accepted cached plan results remain accepted and are never overwritten by a
transient failure. A missing key, 429, timeout, outage, quota denial, safety
refusal, or malformed output never fails the source: remaining work is recorded
as deferred/unavailable, accepted per-strategy checkpoints survive (and are
re-persisted on every run, including repeated forced reruns), only unfinished
work stays non-cache-eligible, and a later normal request retries only unfinished
strategy work. Deferred and no-valid-plan outcomes are successful semantic
outcomes.

## Durable execution, lease release, and cancellation

A duplicate/redelivered Celery invocation of the same `(plan_set_id, job_id)` is
fenced by an atomic `QUEUED -> RUNNING` `ProcessingJob` claim performed inside the
executor, so provider work runs exactly once; the loser records
`skipped_duplicate` and performs no provider work. Legitimate retries re-claim a
`FAILED` job, and a run abandoned by a crashed worker is reclaimable after a
bounded staleness window. The plan-set `active_job_id` compare-and-swap remains
as an additional guard.

The `local_only` executor retains the exact lease-bound provider wrapper and
releases it on every exit path (success, provider failure, cancellation, cache
hit, and exception), so the shared heavy-model lease always exits and its renewer
stops; a subsequent lease is never blocked. Hosted clients are closed and keys
scrubbed on the same paths.

## Fingerprints, cache, and concurrency

The plan-set input fingerprint covers all output-relevant semantics: candidate
identity/current disposition; Stage 4.0 analysis ID/input/output fingerprint and
policy/schema/validation identity; current recommended strategy IDs, keys, ranks,
and fingerprints; selected refinement ID/priority/status/quality/output
fingerprint; exact bounded transcript, refined bounds, indexed word evidence,
unresolved/entity/dialect/code-switch evidence; bounded context actually sent;
Stage 4.0 source moment, assessments, platform risk, preservation and
verification evidence; target-market/language/register semantic context;
narration-enabled/language/register semantic policy; Stage 4.1 policy/config/
validation/schema; and planning provider mode, tier route, model, API, prompt
hash, schema, temperature, thinking, and budgets.

It excludes the Gemini key value or presence, admission/cooldown/outage state, TTS
provider/model/voice/fallback voice, speech-generation settings, rendering/
FFmpeg configuration, publishing schedule, and unrelated frontend settings.
Per-strategy provider-input fingerprints let an accepted validated plan be reused
when its fingerprint still matches, including after transient provider loss or a
force request; only missing/transiently invalid strategies are retried. The
output fingerprint covers the ordered current validated plan representation and
terminal per-strategy outcomes, excluding transient timing, metrics, and provider
availability.

Queue-time cache validation and handoff freshness use the same settings-derived
config, provider mode, and stable configured provider identity as execution, so
transient availability never invalidates accepted plans while a real model/
prompt/schema/config change does. Concurrent queue requests are transaction-safe
(savepoint uniqueness recovery plus an atomic `active_job_id IS NULL`
compare-and-swap), yielding one plan set and at most one active job with no
escaped `IntegrityError`/`500`.

Cancellation is cooperative and polled with a fresh scalar query before planning,
before and after admission, before and after every actual provider call, before
persistence finalization, and before a successful return. It keeps the job and
plan set `CANCELLED`, schedules no next stage, preserves accepted per-strategy
checkpoints, closes Gemini clients, releases Qwen leases, scrubs keys, and never
persists newly completed work after cancellation.

## API and CLI

- `POST /api/candidates/{candidate_id}/transformation-plans` — queue one
  candidate-scoped planning run (returns plan-set/job/cache/active state; accepts
  no TTS configuration).
- `GET /api/candidates/{candidate_id}/transformation-plans` — read the current
  plan set and plans.
- `GET /api/transformation-plan-sets/{plan_set_id}` — read one plan set.
- `GET /api/candidates/{candidate_id}/stage4-2-handoff` — typed read-only Stage
  4.2 handoff.

```bash
python -m app.cli transformation-plan-generate CANDIDATE_ID [--force]
python -m app.cli transformation-plans CANDIDATE_ID
python -m app.cli transformation-plan-handoff CANDIDATE_ID
```

Celery task `clipfactory.run_transformation_planning` uses the existing explicit
candidate-scoped job wrapper: it creates no `PipelineRun` and schedules nothing
automatically.

## Stage 4.2 handoff

Read-only structured handoff containing candidate/source identity; Stage 4.0
analysis identity and fingerprints; selected refinement identity/quality; plan
set identity/status/outcome/current/stale/cache state; provider mode/status/
identity; and every current plan's ID/key/fingerprint/status/rank/confidence,
Stage 4.0 strategy ID/key/type/fingerprint/intensity, target audience/language/
register, source dialect snapshot, exact ordered blocks, hero source span and
appearance time, hook/payoff preservation evidence, substantive value kinds/
reasons, narration need and abstract requirements, verification dependencies,
duration/source-original balance, required context, preservation/fallback
constraints, Stage 4.0 risk and assessment snapshot, and planning-provider
attribution. It always contains:

```json
{ "stage4_2_implemented": false, "stage4_3_implemented": false }
```

It contains no selected, approved, winner, authoritative-plan, governor verdict,
or render-ready state, so Stage 4.2 never needs to reconstruct block order or
plan semantics from prose.

## Scope exclusions

Stage 4.1 does **not** implement Stage 4.2 scoring/retention/originality
verdicts, Stage 4.3 selection/approval, a final plan winner, publication-ready
full scripting, TTS generation or Gemini TTS, voice selection, channel/account/
TTS schema, per-video voices, rendering/FFmpeg operations, captions/gameplay/
B-roll/face tracking, publishing/scheduling/metadata/analytics, Stage 7
diversity, external browsing/research, search grounding, platform-detection
evasion, ASR/ingest changes, or automatic pipeline advancement.

## Validation

```bash
# focused Stage 4.1
docker run --rm -e PYTHONPATH=/app -v "$PWD/backend:/app" -w /app oac-backend-test \
  python -m pytest tests/ -k stage41 -q

# full backend suite (with repository compose/.env mounted for path-based tests)
docker run --rm -e PYTHONPATH=/app -v "$PWD/backend:/app" \
  -v "$PWD/compose.yaml:/compose.yaml:ro" -v "$PWD/.env.example:/.env.example:ro" \
  -w /app oac-backend-test python -m pytest tests/ -q

# coverage gate (CI)
coverage run --source=app -m pytest && coverage report --fail-under=79

# formatting and lint
ruff format app tests alembic && ruff check app tests alembic
```

All Stage 4.1 tests are deterministic and hermetic: providers are mocked and no
test makes a live Gemini, Qwen, web, TTS, or rendering call even when a key is
present. The hardening pass advanced planning versions to
`stage4.1-v2` / `stage4.1-schema-v2` / `stage4.1-validation-v2` (real
verification-block linkage, hero elapsed-time cap, minimum excerpt duration,
narration-contract checks, overlapping-span rejection, and the
TTS/rendering/evasion boundary), so prior plans invalidate through the input
fingerprint. Known limitation: the repository's strict `mypy` configuration
already reports the same class of `no-any-return`/`untyped-decorator` findings in
the frozen Stage 4.0 provider modules and the FastAPI app; Stage 4.1 matches that
existing convention rather than introducing a new lint regime.
