# Stage 4.0 operations: transformation eligibility and strategy discovery

Stage 4.0 decides, for each refined Stage 3 candidate, whether it has a credible
substantive transformation path that preserves the strongest source-retention
moment, and produces a small bounded set of strategy directions for Stage 4.1.
`NO_TRANSFORMATION_STRATEGY_WORTH_USING` is a **normal successful outcome**, not
an error, and may contain zero recommended strategies.

## Position in the funnel

```
Stage 3 candidate -> Stage 3.5 CANDIDATE refinement -> explicit Stage 4.0 analysis
  -> later plan stages -> FINAL_CLIP refinement only for worthwhile selected clips
```

Stage 4.0 is explicit, candidate-scoped work after a usable, audio-backed Stage
3.5 refinement. It is **not** in the automatic `_NEXT_STAGE` chain, never
advances every source, adds no `PipelineStage`, no `PipelineRun`, and no source
lifecycle change, and never requests `FINAL_CLIP` refinement automatically. It
extends the existing Celery/`ProcessingJob` platform with a
`TRANSFORMATION_ELIGIBILITY` job kind plus a nullable
`processing_jobs.transformation_analysis_id` FK.

## Input resolution

Queueing requires a current retained Stage 3 candidate and at least one usable
Stage 3.5 refinement:

1. Prefer a genuinely usable `FINAL_CLIP` refinement when already ready.
2. Otherwise use a usable `CANDIDATE` refinement.
3. A queued, failed, cancelled, empty, or unresolved final row never hides a
   usable candidate row.
4. No Stage 3.5 refinement means a prerequisite error — never analysis of the
   coarse INDEX transcript.
5. A completed refinement whose transcript remains too uncertain produces
   `INSUFFICIENT_TRANSCRIPT_CONFIDENCE`.

The analysis consumes the effective refined transcript, refined boundaries,
source-time word timestamps, confidence/unresolved spans, entity evidence,
dialect/code-switch evidence, a small bounded nearby context assembled from only
relevant transcript segments, and the Stage 3 content types/hooks/scores/idea and
topic summaries, provenance, rights, and originality evidence. It does not
fingerprint or send the whole raw transcript.

## Eligibility outcomes

`ELIGIBLE_FOR_TRANSFORMATION`, `ELIGIBLE_WITH_CAUTION`, `TRANSFORMATION_REQUIRED`,
`NO_TRANSFORMATION_STRATEGY_WORTH_USING`, `INSUFFICIENT_TRANSCRIPT_CONFIDENCE`,
`INSUFFICIENT_CONTEXT`, `UNRESOLVED_POLICY_OR_PROVENANCE_RISK`.

`NO_TRANSFORMATION_STRATEGY_WORTH_USING` is a succeeded job and a cache-eligible
completed analysis, exposed through API/CLI/handoff with an empty recommended
strategy list; it is never an exception or pipeline failure. Deterministic
precedence: identity/bounds validation, transcript blockers, missing-context
blockers, explicit elevated/conflicting provenance, transformation necessity,
content-suitable direction generation, hard strategy gates, then no-strategy,
transformation-required (surviving third-party path), eligible, or
eligible-with-caution. Unknown/third-party provenance alone is never
`UNRESOLVED_POLICY_OR_PROVENANCE_RISK`, which is reserved for explicit stored
conflicts.

Processing lifecycle is tracked separately: `QUEUED`, `ANALYZING`, `COMPLETE`,
`PROVIDER_DEGRADED`, `FAILED`, `CANCELLED`.

## Strategy directions

Closed set: `CONTEXT_HOOK`, `HOOK_PLUS_TAKEAWAY`, `EXPLANATORY`, `COMMENTARY`,
`ANALYSIS`, `SUMMARY`, `COMPARISON`, `COUNTERPOINT`, `REACTION_FRAMING`,
`QUESTION_EXPLANATION_TAKEAWAY`, `CLAIM_CONTEXT_CONCLUSION`, `DEBATE_CONTEXT`,
`NEWS_CONTEXT`, `SOURCE_AS_EVIDENCE`, `SOURCE_LED_MINIMAL`. At most three
recommended and three useful rejected directions per analysis; no minimum is
forced. Content suitability (preference order, least intrusive first):

| Content | Suitable directions |
| --- | --- |
| interview/opinion | `SOURCE_AS_EVIDENCE`, `ANALYSIS`, `COUNTERPOINT`, `CONTEXT_HOOK` |
| educational/tutorial | `EXPLANATORY`, `COMPARISON`, `HOOK_PLUS_TAKEAWAY` |
| funny | `SOURCE_LED_MINIMAL`, `CONTEXT_HOOK`, `REACTION_FRAMING` (preservation-first) |
| story/emotional | `CONTEXT_HOOK`, `HOOK_PLUS_TAKEAWAY`, `SOURCE_LED_MINIMAL` |
| debate | `CLAIM_CONTEXT_CONCLUSION`, `COUNTERPOINT`, `DEBATE_CONTEXT` |
| news/current event | `NEWS_CONTEXT`, `SOURCE_AS_EVIDENCE`, `EXPLANATORY` |
| reaction-worthy | `REACTION_FRAMING`, `ANALYSIS`, `CONTEXT_HOOK` |

`SUMMARY`, `COMMENTARY`, and `REACTION_FRAMING` must contribute candidate-specific
new value; their names alone are never enough. Directions are not scripts and
contain no completed narration, hook text, edit timeline, shot list, TTS text, or
rendering instruction.

### Substantive value must be grounded

A deterministic recommendation requires a concrete, candidate-specific value-add
basis supported by available evidence: a specific missing contextual gap,
inference, explanation target, comparison basis, counterpoint,
verification/correction requirement, synthesis, authored thesis, useful takeaway,
or source-as-evidence framing. Static strategy scores, Stage 3 quality, transcript
length, hook presence, or generic template wording never establish value on their
own. When no such evidence exists, Stage 4.0 returns
`NO_TRANSFORMATION_STRATEGY_WORTH_USING` (or, for a serious complex case, defers
to selective provider discovery); it never invents facts or requires external
research. Value-focus text quotes or names the concrete evidence rather than
emitting a generic phrase, e.g. `Advance the stated thesis: "<candidate claim>"`.
The same grounding check applies to provider directions: an ungrounded or
generic focus is rejected.


## Retention preservation

Every recommended direction keeps the strongest source moment as the hero. The
analysis persists constraints, not an edit timeline: keep the source hook/payoff
early, do not add a long preamble, preserve comic timing/narrative order, keep
the key claim intact, and place concise context or implication after the source
moment. Short/high-density/payoff-driven moments receive high source-moment
damage risk when a direction requires setup before the interesting moment, and
the corresponding direction is rejected with `HOOK_PAYOFF_DAMAGE`.

## Originality and anti-slop

Each recommended direction identifies at least one substantive-value kind:
missing context, inference, explanation, comparison, counterpoint,
verification/correction, synthesis, authored thesis, useful takeaway, or source
used as evidence. Presentation-only changes (captions, crop, reframe, zoom,
punch-ins, borders, emojis, gameplay, B-roll, background loops, music, speed
changes, simple cuts, filters) receive zero credit and can never be the basis of
a recommended direction. Hard gates reject directions that have no identified
substantive value, rely only on presentation, are mostly paraphrase, are generic
filler, materially distort the source, create a fake dramatic hook, damage the
hook/payoff, require unavailable context, cannot reach the required originality,
or depend on an unverified external fact without marking it. Hard filtering
precedes ranking; ranking is deterministic and transparent (sufficient
value/originality, retention, lower damage, added-value density, lower
filler/redundancy, lower sufficient intensity, stable enum tie-break). There is
no single "transformation score".

## Transformation intensity

`MINIMAL`, `MODERATE`, `STRONG`. `MINIMAL` must still be substantive (brief
authored context, source used as evidence, concise original inference or
takeaway). More editing is not automatically better; the least intrusive
sufficient intensity is preferred. A third-party clip never passes solely because
it will later receive polished presentation.

## Third-party and platform risk

Rights/provenance risk and originality/transformation risk are separate.
Third-party/unknown sources remain analyzable, usually receive higher
transformation necessity, must clear stronger substantive-originality
requirements, may produce `TRANSFORMATION_REQUIRED`, and may still produce
`NO_TRANSFORMATION_STRATEGY_WORTH_USING` when the source cannot be transformed
naturally without ruining retention. The persisted platform-risk snapshot covers
YouTube reused-content, YouTube inauthentic/repetitive/mass-produced, Facebook
unoriginal-content, spam/template-heavy, and source-dominance risk with reasons
and limitations. It is decision support only — not a monetization guarantee,
copyright opinion, or legal conclusion. No classifier simulation, watermarking,
mirroring, speed/filter tricks, or detector evasion is implemented.

Official guidance was checked **2026-09-14**:

- YouTube channel monetization policies (reused content; inauthentic/repetitive
  content) — <https://support.google.com/youtube/answer/1311392>
- YouTube spam, deceptive practices, and scams policies —
  <https://support.google.com/youtube/answer/2801973>
- Meta: Rewarding Original Creators on Facebook —
  <https://about.fb.com/news/2026/03/rewarding-original-creators-on-facebook/>
- Meta: Combating unoriginal content / Original Content Guidelines —
  <https://creators.facebook.com/blog/combating-unoriginal-content/>

Only durable principles are encoded, never guesses about hidden classifiers.

## Provider behavior

`CLIPFACTORY_TRANSFORMATION_PROVIDER_MODE` is `adaptive` by default
(`deterministic` / `adaptive` / `local_only`). `deterministic` makes zero Gemini
and zero Qwen calls. `adaptive` selects exactly one hosted tier before the call
and never falls back to Qwen. `local_only` uses Qwen/Ollama only when
`CLIPFACTORY_LOCAL_QWEN_ENABLED=true` and never Gemini. With a missing key the
analysis still completes deterministically.

Official facts checked **2026-09-14** and confirmed against the installed
`google-genai` SDK (2.x, `generate_content` structured-output surface; the
Interactions API is the preferred future interface but the established
repository adapter uses `generate_content`):

- Routine discovery: `gemini-3.5-flash-lite` (stable low-cost structured-output).
- Complex/high-value claim, debate, news, or transformation-required case:
  `gemini-3.8-flash` with low thinking, chosen by a pure deterministic router.
- One tier per call, never both automatically; temperature 0; strict structured
  output; lazy SDK client closed on every exit path; key unwrapped only at client
  creation and never logged/serialized/committed.

Gemini is skipped for missing/uncertain transcripts, insufficient context,
obvious no-strategy/low-quality cases, cache hits, and matching accepted results.
Every hosted request passes through the shared Gemini admission controller at
`HIGH`; at most one hosted strategy-discovery call per analysis. On 429 the
controller cooldown is honored; on missing key/outage/malformed output/safety
refusal/quota denial the deterministic results stand, accepted matching hosted
results are preserved, the analysis never fails, the run stays non-cache-eligible
only while selected hosted work is unfinished, and a later retry reuses accepted
work.

## Fingerprints and idempotency

The input fingerprint covers candidate UUID/key/current disposition; Stage 3
analysis fingerprint and policy version; relevant Stage 3 scores, classifications,
hooks, idea/topic evidence; provenance, rights, and originality snapshots;
selected refinement UUID/priority/status/quality; refinement output fingerprint;
the exact effective transcript; refined boundaries; transcript confidence;
relevant unresolved/entity/dialect/code-switch evidence; bounded nearby context
actually used; deterministic policy/config/version; platform-risk policy version;
provider mode and pure provider-tier route; and provider/model/API/prompt/schema/
temperature/thinking/budget identity. It excludes whole-source raw ASR, unrelated
segments, renderer/publishing/frontend settings, future Stage 4.1–4.3 settings,
the Gemini key value, and transient admission/cooldown/outage state. The output
fingerprint covers eligibility and the complete current strategy representation,
excluding transient metrics/timing/availability. Identical repeated requests
return the same analysis/strategy IDs with no duplicate provider call; force may
recompute deterministic assembly but never repeats an accepted hosted call with
the same provider-input fingerprint. A changed transcript, refined boundary,
Stage 3 evidence, deterministic/transformation policy, selected provider/model,
prompt, or schema invalidates correctly; unrelated rendering/publishing changes
do not.

Queue-time cache validation uses the same settings-derived Stage 4.0
configuration, provider mode, and **stable configured provider identity** that
execution uses, so an adaptive or `local_only` cached analysis is reused instead
of re-queued and never triggers a second provider call. The Stage 4.1 handoff
evaluates freshness against that same configured identity, so a just-completed
analysis is not immediately stale and a relevant policy/config/provider/model/
prompt change is detected.

The configured identity is derived from configuration only (no key, no client,
no network) and is identical whether or not the provider is currently available.
Transient Gemini/key/provider unavailability therefore does not invalidate
accepted analysis: the cache stays a hit, no replacement job is created, and a
force rerun reuses the accepted hosted result instead of overwriting it. A real
model, prompt, schema, provider-mode, or policy/config change still changes the
identity and invalidates correctly.

### Concurrency safety

Concurrent queue requests for one candidate are transaction-safe. Analysis
creation recovers from a `clip_candidate_id` uniqueness race inside a savepoint,
and active-job claiming is a single conditional
`UPDATE ... SET active_job_id = :job WHERE id = :id AND active_job_id IS NULL`
compare-and-swap that both PostgreSQL and SQLite evaluate atomically (a losing
writer re-evaluates the predicate after the winner commits and observes zero
rows). Simultaneous requests therefore yield one analysis and at most one active
Stage 4.0 job, reuse the active job instead of duplicating it, and never surface
a uniqueness `IntegrityError` or 500 during normal concurrent queueing.

## Cancellation and resource safety

The exact executing job is polled before analysis, before and after provider
admission, before and after every provider call, before persistence finalization,
and once before a successful return. Cancellation keeps the job `CANCELLED`,
preserves valid accepted provider evidence/checkpoints, schedules nothing else,
produces no completed/eligible state, closes provider clients, releases Qwen
leases, scrubs the Gemini key, and keeps transcript data out of errors/logs.

## Persistence, API, CLI, handoff

- `transformation_eligibility_analyses`: one reusable current row per candidate
  (refinement identity/quality, execution status, eligibility outcome/reasons,
  independent assessments, source-moment evidence, platform-risk snapshot,
  intensity, provider mode/identity/status/evidence, fingerprints, cache
  eligibility, active job, metrics, duration, timestamps).
- `transformation_strategy_candidates`: one stable row per
  `(analysis_id, strategy_type)` (deterministic strategy key, current marker,
  disposition, rank, intensity, bounded direction/value/preservation/verification
  fields, independent assessments, confidence, origin/provider evidence, strategy
  fingerprint). Matching types upsert with stable UUIDs; absent strategies are
  marked non-current only after a successful finalized analysis; nothing is
  deleted on ordinary reruns.
- Migration `20260914_0014` downgrade deletes only Stage 4.0 jobs, removes the
  Stage 4.0 job FK, drops the Stage 4.0 tables, narrows the job-kind constraint,
  and preserves every Stage 1–3.7 row and lifecycle value.

API: `POST /api/candidates/{id}/transformation-analyses`, `GET
/api/candidates/{id}/transformation-analysis`, `GET
/api/transformation-analyses/{id}`, `GET /api/candidates/{id}/stage4-1-handoff`.
CLI: `transformation-analyze`, `transformation-analysis`,
`transformation-handoff`.

The Stage 4.1 handoff exposes candidate identity; analysis ID/output fingerprint;
selected refinement and quality; effective transcript; transcript
confidence/status; refined bounds; word timing evidence; unresolved spans; Stage 3
content types/scores/hooks/idea/topic; provenance/rights/originality;
dialect/code-switch/entity evidence; eligibility outcome/reasons; transformation
necessity/potential; platform-risk evidence; recommended and useful rejected
strategies with intensity/retention/damage/dominance/added-value/originality and
verification requirements; provider/routing evidence; cache/current/stale state;
and `stage4_1_implemented=false`. If current upstream input no longer matches the
analysis input fingerprint, the handoff reports stale/not-ready rather than
mixing current data with old strategies. `NO_TRANSFORMATION_STRATEGY_WORTH_USING`
returns a valid handoff with `ready_for_stage4_1=false`, not 500/404.

## Scope boundaries

Out of scope: Stage 4.1 plan generation, Stage 4.2 critic/governor, Stage 4.3
selection/approval, scripts, edit timelines, shot lists, TTS, narration
rendering, Stage 5 rendering, supporting-visual generation, external factual
research, Gemini search grounding, publishing, scheduling, metadata generation,
Stage 7 diversity/history, platform classifier simulation, detection evasion,
watermark/mirror/speed/filter tricks, benchmarks, ASR/ingest changes, and any
requirement that the operator appear on camera or supply commentary.

## Known limitations

- Strategy discovery is a deterministic heuristic plus optional provider
  assessment; it is decision support, not a quality guarantee.
- Ranking uses fixed transparent criteria; there is deliberately no learned
  "transformation score".
- `NO_TRANSFORMATION_STRATEGY_WORTH_USING` is preferred over a weak forced
  direction; operators may rerun with a different mode when evidence changes.
- The platform-risk snapshot reflects durable public principles only; it makes no
  claim about hidden classifiers or approval outcomes.
