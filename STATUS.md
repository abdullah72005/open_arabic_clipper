# Runtime status

## Stage 4.3 deterministic final-plan selection (2026-09-18)

Stage 4.3 deterministically commits each current Stage 4.2 governance set to
exactly zero or one current survivor. It is explicit, candidate-scoped,
synchronous, transaction-safe, and **provider-free**: no Celery task, no
`ProcessingJob` kind, no executor/queue, no `PipelineStage`, no `PipelineRun`,
and **no** `_NEXT_STAGE` entry. It never replans, re-governs, rewrites,
researches, refines transcripts, or renders.

- **Authoritative input.** `build_stage4_3_handoff()` (frozen Stage 4.2 -> 4.3
  contract) plus the current Stage 4.1 plan rows. Only governance freshness
  `VERIFIED_CURRENT` may produce a selected plan; `STALE` is
  `STALE_SELECTION_INPUT`; `NOT_CURRENT`/`UNVERIFIABLE` are `SELECTION_DEFERRED`.
- **Closed outcomes.** `PLAN_SELECTED`, `PLAN_SELECTED_WITH_CAUTION`,
  `NO_SELECTABLE_PLAN`, `SELECTION_DEFERRED`, `STALE_SELECTION_INPUT`. Every
  outcome is a successful semantic result once the candidate exists;
  candidate-not-found stays 404. Completed verification-blocked/revision/
  rejected mixes are `NO_SELECTABLE_PLAN`; deferred semantic governance keeps
  the answer open (`SELECTION_DEFERRED`).
- **Eligibility and caution.** Clean selection requires
  `APPROVED_FOR_SELECTION`, `eligible_for_stage4_3=true`, empty hard gates, a
  matching plan fingerprint, resolved verification
  (`GROUNDED_IN_SOURCE`/`NOT_APPLICABLE`, `unresolved=false`), and strong
  semantic fidelity. `APPROVED_WITH_CAUTION` is auto-selectable only when every
  warning is in the conservative allowlist (`SOURCE_DOMINANCE_CONCERN` with a
  MODERATE source-dominance dimension; `TEMPLATE_MASS_PRODUCED_FEEL` with a
  MODERATE template dimension), retention is not damaged/mixed, coherence is
  acceptable, and there is no unsupported-claim or unknown warning. Clean
  approvals always arbitrate before cautions. Internally inconsistent approved
  evidence (non-empty hard gates, fingerprint mismatch, unresolved essential
  verification, missing decision-critical fields) fails closed as deferred and
  is never "repaired".
- **Deterministic hierarchy, no aggregate score.** Lexicographic comparison over
  persisted evidence: semantic fidelity, retention/damage, substantive
  originality, platform reuse risk (worst level then HIGH/MODERATE count),
  coherence, filler/redundancy, narration burden, proportionality/intensity (when
  value is not worse), then Stage 4.1 generation rank, planner confidence, and
  stable plan identity. Planner rank/confidence never override Stage 4.2
  evidence. The first material distinction, approval tier, eligible/excluded
  plan IDs, and per-alternative dispositions are persisted as explainability
  evidence.
- **Persistence and concurrency.** One new table `transformation_plan_selections`
  (one row per candidate + selection input fingerprint; nullable
  `selected_plan_id`, never a plan copy), a unique
  `(clip_candidate_id, input_fingerprint)`, and a PostgreSQL/SQLite partial
  unique index allowing at most one `is_current=true` row per candidate.
  Concurrent POSTs converge through candidate-row locking, savepoint uniqueness
  recovery, and post-lock freshness revalidation. The selected governance
  evidence is snapshotted immutably; Stage 4.1 plans and Stage 4.2 rows are never
  mutated. A changed Stage 4.2 fingerprint, Stage 4.1 plan fingerprint,
  verification state, warning, governor policy/profile, relevant target context,
  or Stage 4.3 policy version creates a new current row and marks the prior one
  historical. TTS voice/provider/model, render configuration, and publishing
  configuration never invalidate selection.
- **Selection vs execution readiness.** Readiness is live, never persisted:
  `READY_FOR_FINAL_REFINEMENT`, `REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK`,
  `READY_FOR_EXECUTION_PREP`, `BLOCKED`. A CANDIDATE-grade plan is selectable and
  requires final refinement; a current usable `FINAL_CLIP` on the same identity
  is execution-prep ready; a newer usable `FINAL_CLIP` forces a compatibility
  check rather than render readiness. Stage 4.3 never enqueues Stage 3.5 work and
  selection is not render/publication readiness.
- **Handoff.** `GET /api/candidates/{id}/execution-handoff` returns one selected
  plan or none with exact unchanged ordered blocks, hero span, preservation
  constraints, original-value kinds/reasons, abstract narration semantics (never
  a TTS provider/model/voice), source dialect, verification, platform-risk
  snapshots, refinement identities, compatibility requirement, and
  `stage5_implemented=false`/`stage6_tts_implemented=false`. It never emits
  platform-safety or monetization guarantees.
- **API/CLI.** `POST`/`GET /api/candidates/{id}/transformation-selection`,
  `GET /api/transformation-selections/{id}`, and
  `GET /api/candidates/{id}/execution-handoff`; CLI `transformation-selection`
  and `transformation-selection-handoff`. Both are synchronous and provider-free.
  The Stage 4.2 -> 4.3 handoff metadata now truthfully reports
  `stage4_3_implemented=true` without changing its evidence semantics.
- **Versions.** Policy `stage4.3-v1`, schema `stage4.3-schema-v1`, fingerprint
  version `1`; migration `20260918_0019`.

Deterministic verification: 63 focused Stage 4.3 tests (selection, arbitration,
eligibility, freshness, FINAL_CLIP readiness, API/CLI, model constraints,
migration) plus the full backend suite in Docker Python 3.12; Stage 4.3
PostgreSQL concurrency/partial-unique validation is gated on
`CLIPFACTORY_TEST_POSTGRES_URL`. No live provider calls in the automated suite.
See [docs/STAGE_4_3_OPERATIONS.md](docs/STAGE_4_3_OPERATIONS.md).

## Stage 4.2 retention/originality/platform-risk governor (2026-09-17)

Stage 4.2 independently governs every current Stage 4.1 transformation plan and
answers whether it is good and safe enough to be considered by Stage 4.3. It is
explicit, candidate-scoped, critic-only work: it never generates, mutates,
repairs, or selects a plan, adds **no** `PipelineStage`, **no** `PipelineRun`,
**no** `_NEXT_STAGE` entry, and never advances the source lifecycle.

- **No lifecycle change.** It extends the Celery/`ProcessingJob` platform with a
  `TRANSFORMATION_GOVERNANCE` job kind and a nullable
  `processing_jobs.transformation_governance_set_id` FK. New persistence is
  `transformation_governance_sets` (one envelope per Stage 4.1 plan set) and
  `transformation_governance_results` (one stable row per plan), with check
  constraints binding `eligible_for_stage4_3` to the approved statuses. Stage 4.2
  is critic-only and never selects a plan; Stage 4.3 performs deterministic
  selection separately in `transformation_plan_selections`.
- **Immutable input.** Queueing requires a current retained candidate, a usable
  `CANDIDATE`/`FINAL_CLIP` Stage 3.5 refinement (never requires `FINAL_CLIP`), a
  current non-stale complete/degraded Stage 4.1 plan set with at least one current
  plan, and exact fingerprint matches. Stale input is refused before provider
  work; malformed persisted content is never reinterpreted or repaired. Critical
  integrity (identity, current flags, closed block types/indexes, hero/source-span
  integrity, chronology/overlap, narration abstraction, verification linkage,
  prohibited TTS/render/evasion text, plan output fingerprint) is revalidated
  deterministically.
- **Execution vs semantic outcome.** Lifecycle
  (`QUEUED`/`GOVERNING`/`COMPLETE`/`PROVIDER_DEGRADED`/`FAILED`/`CANCELLED`) is
  separate from per-plan status
  (`APPROVED_FOR_SELECTION`/`APPROVED_WITH_CAUTION`/`BLOCKED_PENDING_VERIFICATION`/
  `REVISION_REQUIRED`/`REJECTED_BY_GOVERNOR`/`GOVERNANCE_DEFERRED`) and the
  candidate outcome (`PLANS_ELIGIBLE_FOR_SELECTION`/`NO_GOVERNOR_APPROVED_PLAN`/
  `GOVERNANCE_DEFERRED`). A rejection, revision, verification block,
  all-ineligible result, or provider deferral is a successful semantic result.
  `eligible_for_stage4_3` is true only for the two approved statuses (a filter,
  not a ranking), enforced by a database check constraint. Candidate precedence:
  any approved/caution → eligible; else any deferred → deferred; else
  no-governor-approved-plan with separate verification/revision/rejection counts.
- **Deterministic dimensions, no aggregate score.** Independent categorical
  assessments with bounded reason codes/evidence are persisted for retention
  preservation, source-moment damage, substantive originality, source dominance,
  semantic fidelity, generic filler, redundant commentary, narration burden,
  verification completeness, template/mass-produced feel, YouTube
  reused/inauthentic/spam risk, Facebook unoriginal/spam risk, coherence, and
  transformation proportionality. Severity classes `HARD_FAIL`/
  `BLOCKING_CONDITION`/`REVISION`/`WARNING`/`ADVISORY` drive deterministic
  precedence: integrity/prohibited/fidelity failures and no-value/presentation-
  only/fabricated-claim/misleading-hook failures reject; unresolved essential
  verification blocks; repairable retention/narration/filler/template/over-edit
  damage requires revision; missing required semantic evidence defers; otherwise
  caution or approval. A 12-second generic preamble before the hero fails;
  narration interrupting a punchline is high source-moment damage; narration
  `NONE` is valid and never penalized. Platform-specific risk alone never
  globally hard-rejects.
- **Rights/verification separation.** Claim state is explicit
  (`GROUNDED_IN_SOURCE`/`EXTERNAL_REQUIRED_UNRESOLVED`/`UNSUPPORTED_OR_FABRICATED`/
  `NOT_APPLICABLE`); Stage 4.2 never fabricates a placeholder, claims verification
  happened, browses, or adds research infrastructure. Copyright/rights, platform
  originality/reuse, and spam/repetition risk stay separate in persistence and
  handoff. Account/channel repetition is `DEFERRED_TO_STAGE_7`.
- **Platform-policy profile.** Immutable code-defined profile
  `stage4.2-platform-policy-2026-09-17-v1` (checked 2026-09-17; official sources
  re-confirmed: YouTube channel monetization, YouTube spam, Meta rewarding
  original creators, Meta cracking down on spam, Facebook original-content
  guidance). It encodes durable reused-content/inauthentic-content/spam concepts
  on YouTube and original/unoriginal/spam concepts on Facebook, and contains no
  evasion logic, monetization guarantee, algorithm-safety claim, or phrase such
  as "safe for YouTube"/"will not be flagged". Policy/checked date participate in
  the input fingerprint.
- **Providers optional.** `deterministic` makes zero Gemini/Qwen calls; `adaptive`
  (default) runs deterministic gates first and calls hosted Gemini only for
  genuinely ambiguous/high-value plans, batched into one bounded plan-set request
  (max three) with an optional single second strong call for plans whose first
  valid critique stays `UNKNOWN`, hard-capped at two raw hosted calls per run and
  never retried inside that ceiling. Every raw call acquires shared HIGH
  admission; missing key/denial/429/timeout/outage/refusal/malformed output never
  fails the candidate — the plan defers, the set degrades to `PROVIDER_DEGRADED`,
  and accepted per-plan checkpoints are reused on a later request. `local_only`
  uses Qwen only when `CLIPFACTORY_LOCAL_QWEN_ENABLED=true` and never Gemini.
  Provider output is parsed with a tolerant top-level shape and strict per-item
  validation; one malformed plan item never invalidates valid siblings. Gemini is
  never asked whether a platform will flag/monetize/recommend and can never
  assign a status or platform classification; platform-guarantee text, TTS/voice
  identity, rendering instructions, and evasion tactics are discarded.
- **Cache/concurrency.** Per-plan critiques are checkpointed under their
  provider-input fingerprint and reused on exact match including forced reruns;
  TTS provider/model/voice, render settings, publishing settings, and metadata do
  not invalidate governance; platform-policy/config/mode/provider identity changes
  do. Unfinished/deferred work makes the set non-cache-eligible. `QUEUED`/`FAILED`
  → `RUNNING` job claims advance durable `claim_version`, duplicate/redelivered
  invocations perform no provider work, liveness is a renewable heartbeat, stale
  reclaim requires an abandoned heartbeat (not old `started_at`), cancellation is
  polled against the exact executing job before/after each provider call and
  before persistence, stale workers cannot persist/cancel/fail/finalize a newer
  claim, and providers/Gemini key are released on every exit path.
- **API/CLI/handoff.** `POST /api/candidates/{id}/transformation-governance`,
  `GET /api/candidates/{id}/transformation-governance`,
  `GET /api/transformation-governance-sets/{id}`, and
  `GET /api/candidates/{id}/stage4-3-handoff`; CLI `transformation-govern`,
  `transformation-governance`, `transformation-governance-handoff`; Celery
  `clipfactory.run_transformation_governance`. The typed read-only Stage 4.3
  handoff exposes per-plan governance with `stage4_3_implemented=false` and no
  winner/selected plan/render-ready/publication state.

Deterministic verification: 108 focused Stage 4.2 tests (governor, providers,
service, executor/queue/cache/concurrency/cancellation, API/CLI/handoff/migration)
plus the full backend suite in Docker Python 3.12 with the repository compose/.env
files mounted, with no live provider calls in the automated suite. Versions:
governor `stage4.2-v5`, schema `stage4.2-schema-v1`, validation
`stage4.2-validation-v5`. See [docs/STAGE_4_2_OPERATIONS.md](docs/STAGE_4_2_OPERATIONS.md).

### Stage 4.2 corrective patch (2026-09-17)

- **Verification grounding.** A substantive authored block is grounded only when
  it declares source/evidence references; an ungrounded authored claim (numeric
  or not, e.g. "The merger closes next Monday") is an unresolved verification
  dependency and becomes `BLOCKED_PENDING_VERIFICATION`, never a silent
  `GROUNDED_IN_SOURCE`/`APPROVED_FOR_SELECTION`. Legitimate source-grounded
  explanation/source-as-evidence plans are unaffected.
- **Stale RUNNING recovery.** Queueing re-dispatches a genuinely abandoned
  Stage 4.2 `RUNNING` job (heartbeat older than the shared stale window) so the
  executor's atomic claim performs the claim-version-bumping reclaim; a live
  job is never reclaimed, stale workers cannot persist/finalize a newer claim,
  and redelivered invocations make no duplicate provider call.
- **Handoff freshness.** The read-only Stage 4.3 handoff exposes explicit
  `current`/`stale` governance state and never reports a stale result as
  currently eligible for Stage 4.3.
- **Second-call checkpoint.** A bounded strong second critique that resolves an
  initially unknown plan is persisted as that plan's checkpoint, so a forced
  rerun with unchanged inputs reuses it with zero additional hosted calls; a
  failed second call preserves the first accepted checkpoint and stays
  truthfully deferred.
- **Strict hero thresholds** (`short_moment_seconds`,
  `high_moment_density_floor`) now actually govern strict hero-window
  classification and remain fingerprint-invalidating.
- **Provider boundary** persists only closed accepted finding codes (arbitrary
  text is discarded, forbidden text still rejects) and drops block indexes
  outside the requested plan's block range.
- **Real source-grounding resolution (P1).** Grounding references are resolved
  against real plan source evidence (source-excerpt block indexes, word-index
  ranges, source/time spans within the refined window, or a quote of real source
  excerpt text). Provider labels such as `strategy`, `source_excerpt`, or any
  arbitrary string are not proof.
- **Claim-to-source support resolution (P1).** A structural citation alone is
  never sufficient, and lexical overlap establishes relevance only, never factual
  entailment. For factual claims, deterministic `SUPPORTED` is limited to a
  supported direct quotation or a near-exact restatement whose content tokens are
  all present in the cited wording; a clearly-tied non-factual interpretation
  whose only non-cited tokens are framing words is also supported. Shared
  entity/topic alone, a changed predicate, partial overlap, or inferred factual
  conclusions are `AMBIGUOUS` → selective semantic review, and when the provider
  is unavailable `GOVERNANCE_DEFERRED` (or blocked when external verification is
  essential) — never approved. Unrelated citations to factual claims are
  `BLOCKED_PENDING_VERIFICATION`. A claim like "Microsoft files bankruptcy" cited
  to "Microsoft announced a new product launch" is not grounded and not eligible.
  No deterministic entailment is claimed; Gemini stays non-authoritative and no
  web lookup/fact-checking is added. Support policy version
  `stage4.2-v3`/`stage4.2-validation-v3` invalidates prior governance.
- **Direct-quote semantic-fidelity correction (P1).** A valid quote or matching
  contiguous phrase is no longer by itself sufficient for `GROUNDED_IN_SOURCE`.
  A direct-quote/reference-framed claim is grounded only when every added material
  token is harmless attribution/interpretive framing; an added factual predicate,
  changed predicate, event, date, or number (e.g. `"Microsoft launches product"`
  plus `"files bankruptcy"`) is `AMBIGUOUS` → `SUPPORT_UNVERIFIED`, requiring
  semantic review, `GOVERNANCE_DEFERRED` when the provider is unavailable, or
  `BLOCKED_PENDING_VERIFICATION` when an essential external dependency exists.
  Exact quotes with attribution, near-exact restatements, source-grounded
  non-factual interpretation, and clearly-supported claims (zero hosted calls) are
  preserved. Versions advanced to `stage4.2-v4`/`stage4.2-validation-v4`.
- **Meaning-changing semantic-operator protection (P1).** Deterministic support
  comparison now protects negation, exclusivity, and modality/certainty/
  obligation operators (English and Arabic) from stopword/presentation filtering;
  apostrophe contractions are expanded (`can't` → `cannot`, `won't` → `will not`,
  …) and Arabic punctuation is normalized so tokens compare exactly. A claim that
  adds, drops, or changes a protected operator relative to its cited wording
  (`only`, `cannot`, `لا`, `فقط`, `will`/`may`/`must`, `قد`/`سوف`, …) can never
  become `GROUNDED_IN_SOURCE` through lexical overlap alone, even inside a valid
  quote; it becomes `AMBIGUOUS` → `SUPPORT_UNVERIFIED`/`GOVERNANCE_DEFERRED`
  pending semantic review. Permitted attribution framing and faithful identical
  restatements remain grounded with zero hosted calls. Versions advanced to
  `stage4.2-v5`/`stage4.2-validation-v5`.
- **Fail-closed Stage 4.3 handoff freshness (P1).** The handoff exposes
  `freshness` ∈ {`VERIFIED_CURRENT`, `STALE`, `NOT_CURRENT`, `UNVERIFIABLE`}.
  Only `VERIFIED_CURRENT` may retain per-plan `eligible_for_stage4_3`; a missing
  fingerprint, unresolved inputs, or a fingerprint recomputation error is
  `UNVERIFIABLE`, and every non-verified state forces `eligible_for_stage4_3=false`.

## Stage 4.1 transformation plan generation (2026-09-14)

Stage 4.1 turns each current recommended Stage 4.0 strategy direction into zero
to three concrete structured transformation plans. It is explicit,
candidate-scoped work after a current, non-stale Stage 4.0 analysis with at least
one current recommended strategy, accepts a usable `CANDIDATE` Stage 3.5
refinement, and is **never** added to the automatic `_NEXT_STAGE` chain.

- **No lifecycle change.** It extends the existing Celery/`ProcessingJob`
  platform with a `TRANSFORMATION_PLANNING` job kind and a nullable
  `processing_jobs.transformation_plan_set_id` FK. It adds no `PipelineStage`,
  no `PipelineRun`, and never advances the source lifecycle. Stage 4.2 and Stage
  4.3 are implemented as separate explicit, candidate-scoped work.
- **Input gate.** Queueing requires a current retained candidate, a usable
  Stage 3.5 refinement, and a current, non-stale Stage 4.0 handoff
  (`ready_for_stage4_1=true`) with at least one current recommended strategy. A
  stale Stage 4.0 analysis is a prerequisite conflict, never a reason to mix
  current transcript data with old strategies. Only bounded inputs are sent: the
  effective refined transcript, indexed word evidence, bounded nearby context,
  Stage 3 evidence, the Stage 4.0 source moment/assessments/risk, dialect/
  entities/unresolved evidence, and target/narration semantic context.
- **Plan schema.** `transformation_plan_sets` is one durable envelope per
  candidate; `transformation_plans` is one stable row per
  `(plan_set, strategy_candidate)`. Execution lifecycle
  (`QUEUED`/`PLANNING`/`COMPLETE`/`PROVIDER_DEGRADED`/`FAILED`/`CANCELLED`) is
  separate from semantic outcome (`PLANS_GENERATED`,
  `PLANS_GENERATED_WITH_VERIFICATION_REQUIRED`, `PLANNING_DEFERRED`,
  `NO_VALID_PLAN_FROM_STRATEGY`, `PROVIDER_UNAVAILABLE`). A zero-plan, deferred,
  or provider-unavailable plan set is a successful semantic result with no fake
  plan rows. The `NO_VALID_PLAN_FROM_STRATEGY` conclusion is truthful only when
  every attempted strategy is a clean, validated explicit no-valid outcome; any
  invalid/malformed/deferred/omitted/provider-failure work makes the outcome
  `PLANNING_DEFERRED` (or `PROVIDER_UNAVAILABLE`) and keeps it non-cache-eligible
  rather than asserting a conclusion the provider never validly reached. Blocks
  are validated JSON on the plan (closed set:
  `SOURCE_EXCERPT`, `ORIGINAL_VALUE`, `TRANSITION`, `TEXTUAL_ANNOTATION`,
  `FACT_VERIFICATION_PLACEHOLDER`).
- **Deterministic first.** Deterministic logic owns readiness, bounds, routing,
  source-span resolution (indexed Stage 3.5 word references or a safe
  full-window sentinel; provider timestamps/quotes are never trusted), hero
  placement, duration arithmetic, substantive-value/paraphrase/cosmetic
  rejection, narration/TTS separation, verification dependency enforcement,
  material distinction, persistence eligibility, and fingerprint/cache
  composition. Every valid plan has exactly one hero source excerpt and at least
  one source excerpt; blocks are capped at eight and plans at three. The hero
  must actually appear early: the cap uses **true elapsed block duration** before
  the hero (including any preceding source `SUPPORT` excerpt) at 3 s, strictly
  1.5 s for short, dense, joke, or payoff-first moments, and the authored-material
  cap is retained as an additional protection. Every source excerpt must meet
  `min_source_excerpt_seconds` (0.4 s), and partially overlapping source spans are
  rejected, not only exact duplicates; chronology and non-overlap are preserved.
  `hero_appearance_time` and the persisted elapsed/authored evidence remain
  accurate and auditable.
- **Hard TTS/rendering/evasion boundary.** Every provider-controlled string or
  string list that can persist into plan rows, per-strategy checkpoints,
  outcomes, or the Stage 4.2 handoff is checked against bounded, explicit policy
  markers **before any no-valid-plan early return**: plan preservation
  constraints, planner notes, no-valid reasons, block purpose, semantic intent,
  why-unavailable, draft line, continuity rationale, verification rationale,
  intended use, grounding refs, dependency ids, claim dependency, and narration
  language/register/dependency ids. TTS provider-plus-voice selection (for
  example "Use Gemini voice Charon"), named-speaker identity (for example "Have
  Morgan Freeman narrate"), named-person imitation/identity-selection phrased
  without any voice/narration/TTS word (for example "Make it sound like Morgan
  Freeman", "In the style of Morgan Freeman", "Impersonate Morgan Freeman",
  "Morgan Freeman's voice"), explicit bilingual (English and Arabic) per-video
  narrator/voice/speaker identity selection for single- and multi-token
  identities (for example "Narrated by Charon", "Use Charon as narrator", "Set
  Charon as the speaker", "Use narrator Charon to explain the claim", "Use a
  voice called Charon", "استخدم صوت شيرون للسرد", "شيرون كراوٍ", "اختر محمد رمضان
  راوياً"), TTS
  provider/model/voice selection, frame-level
  rendering instructions, cosmetic-only transformation claims, and
  platform/detection-evasion tactics (mirroring, pitch shifting, speed tricks,
  watermark removal/obfuscation) are rejected. Legitimate narration semantics
  (purpose, language, register, duration, placement, dependency) are preserved,
  ordinary semantic words such as "model" are not blocked, ordinary wording such
  as "The narrator named several causes" is not treated as selection, and
  ordinary discussion of a named person (for example "Reference Morgan Freeman's
  career as context") is not treated as identity selection. The planner service
  runs
  this boundary for **every** parsed provider result, including explicit
  `no_valid_plan=true` payloads, before any field can be persisted,
  checkpointed, used as an attempt/outcome reason, reused from cache, or exposed
  through the Stage 4.2 handoff; a no-valid payload with forbidden text is a safe
  invalid/malformed result (never an explicit decline) and stores only the
  bounded rejection code, while a clean bounded no-valid reason remains a
  legitimate `NO_VALID_PLAN_FROM_STRATEGY` outcome.
- **Narration contract.** A plan whose substantive block requires narration
  delivery while `NarrationNeed.NONE` is recorded is rejected, and a
  narration-disallowed context cannot accept an essential narration-only
  contribution. Valid `NONE` plans with no narration-delivery block remain valid.
- **Providers are optional.** `deterministic` makes zero Gemini/Qwen calls
  (conservative grounded fallback only for `SOURCE_LED_MINIMAL`/
  `SOURCE_AS_EVIDENCE`); `adaptive` (default) batches per tier, at most one
  hosted call per tier and two per plan-set run, through the shared HIGH
  admission gate (`gemini-3.5-flash-lite` routine, `gemini-3.8-flash`
  low-thinking for genuinely complex strategies), temperature 0, strict
  structured output, and never falls back to Qwen; `local_only` uses Qwen only
  when `CLIPFACTORY_LOCAL_QWEN_ENABLED=true`. The two-call ceiling is a hard
  **raw** `generate_content` budget: there is no per-tier retry inside it, the
  provider refuses a third raw call, and persisted metrics report actual raw
  hosted calls only: deterministic and `local_only`/Qwen planning report
  `hosted_raw_calls=0`, and Gemini reports its real `generate_content` count
  (never a generic provider-invocation count). Missing key/outage/429/quota/safety
  refusal/malformed output never fails the source: accepted per-strategy
  checkpoints survive (including across repeated forced reruns), only unfinished
  work stays non-cache-eligible, and a later normal request retries it. Planning
  mode is separately configurable
  (`CLIPFACTORY_TRANSFORMATION_PLANNING_MODE`) and decoupled from Stage 4.0 mode.
- **Narration and TTS.** Narration is an abstract semantic requirement (need,
  purpose, language, register, duration, placement, dependencies) and never
  selects a TTS provider, model, or voice; channel configuration will decide the
  persistent narrator and Stage 6 generates speech. Narration is never
  automatically required; a `NONE` plan is fully valid.
- **Dialect and target.** Source dialect is preserved byte-for-byte from Stage
  3.5 evidence; a future target market may shape original-block framing and a
  broadly understandable narration register but must never alter source speech
  or relabel the dialect. No channel/account/target-market persistence model was
  added.
- **Verification.** A strategy marked `REQUIRES_EXTERNAL_FACT_VERIFICATION`
  must yield a structured placeholder that names the claim and links to real
  dependent content. Linkage is deterministic and persisted: a placeholder lists
  the integer block indexes of dependent substantive blocks, those blocks carry
  the placeholder's claim id in `dependency_ids`, and narration depends on a
  claim through `verification_dependency_ids`. Every malformed verification
  reference shape is rejected at the strict provider parse boundary rather than
  treated as empty: non-list strings, booleans, objects, floats, supplied nulls,
  nested values, and malformed items anywhere in a list (including past the
  bounded valid-item cap). Valid integer indexes are preserved. Bogus, missing,
  nonexistent, self-referential, non-substantive, or unlinked references are
  rejected; `must_verify_before_execution=true` is preserved. No fact is
  fabricated and there is no browsing/research subsystem.
- **Durable execution.** A duplicate/redelivered Celery invocation of the same
  `(plan_set_id, job_id)` is fenced by an atomic `QUEUED -> RUNNING`
  `ProcessingJob` claim, so provider work runs exactly once. Legitimate retries
  re-claim a `FAILED` job and clear stale terminal failure metadata (`error_code`,
  `error_message`, prior `completed_at`). Every claim advances a durable
  per-execution ownership token (`processing_jobs.claim_version`), and every
  worker-owned planning-state mutation is fenced on the token **and**
  `status == RUNNING`, so neither plan-set claim, persistence, cancellation,
  failure, nor finalization can commit after an API cancellation. A reclaimed
  stale worker can neither persist, cancel, fail, nor finalize a newer run.
  Liveness is a renewable durable heartbeat (`processing_jobs.heartbeat_at`)
  renewed from a background thread while the worker executes, so a live worker
  stalled inside a provider call is never reclaimed merely because its original
  `started_at` is old; only a genuinely abandoned heartbeat is reclaimable. A
  cancelled API job is truthfully reconciled to `CANCELLED` on the plan set
  without a newer claim being overwritten. Failed planning jobs retain bounded
  sanitized operator diagnostics (prerequisite messages or the exception type
  only; never keys, transcripts, or provider payloads), and failure recording
  rolls back a pending-rollback session and re-checks durable ownership first.
  The local-only executor retains and releases the exact lease-bound provider
  wrapper, so the shared heavy-model lease always exits on success, provider
  failure, cancellation, cache hit, and exception.
- **API/CLI/handoff.** Minimal endpoints queue/read one plan set and expose a
  typed read-only Stage 4.2 handoff (`stage4_2_implemented=false`,
  `stage4_3_implemented=false`) with exact ordered blocks, hero span, narration
  semantics, verification dependencies, and Stage 4.0 risk.

Deterministic verification: 172 focused Stage 4.1 tests (63 original + 24
review-remediation + 19 final re-review remediation + 20 final-closure
remediation + 8 provider no-valid boundary closure + 11 final truthfulness and
speaker-boundary closure + 20 bilingual speaker-boundary closure + 7 Arabic
multi-token closure) plus the full
backend suite in Docker Python 3.12 with the repository compose/.env files
mounted, with no live provider calls in the automated suite. Planning
policy/schema/validation versions advanced to
`stage4.1-v8`/`stage4.1-schema-v8`/`stage4.1-validation-v8` so prior plans and
checkpoints invalidate. See
[docs/STAGE_4_1_OPERATIONS.md](docs/STAGE_4_1_OPERATIONS.md).

## Stage 4.0 transformation eligibility and strategy discovery (2026-09-14)

Stage 4.0 decides, per refined Stage 3 candidate, whether a credible substantive
transformation path exists and emits a small bounded set of strategy directions
for Stage 4.1. It is explicit, candidate-scoped work after a usable Stage 3.5
refinement and is **not** added to the automatic `_NEXT_STAGE` chain.

- **No lifecycle change.** It extends the existing Celery/`ProcessingJob`
  platform with a `TRANSFORMATION_ELIGIBILITY` job kind and a nullable
  `processing_jobs.transformation_analysis_id` FK. It adds no `PipelineStage`,
  no `PipelineRun`, and never advances the source lifecycle; Stage 4.1 is
  implemented separately (see above) and adds none of these either.
- **Input resolution.** Queueing requires a current retained candidate and a
  usable, audio-backed Stage 3.5 refinement. A usable `FINAL_CLIP` row is
  preferred when already ready, otherwise a usable `CANDIDATE` row is used; a
  queued/failed/cancelled/empty final row never hides a usable candidate row and
  final refinement is never enqueued automatically. No Stage 3.5 refinement
  means a prerequisite error, never analysis of the coarse INDEX transcript.
- **Deterministic first.** Hard gates own transcript/context sufficiency,
  transformation necessity, source-moment structure, content-to-strategy
  suitability, presentation-only zero credit, anti-slop validation, retention,
  originality, filler, and final eligibility. A deterministic recommendation also
  requires a concrete, candidate-specific value-add basis supported by available
  evidence (a specific context gap, inference, explanation target, comparison,
  counterpoint, verification, synthesis, thesis, takeaway, or source-as-evidence
  framing); static scores, quality, length, hooks, or generic templates never
  establish value on their own, and absent evidence yields
  `NO_TRANSFORMATION_STRATEGY_WORTH_USING` or defers to selective provider
  discovery. There is no single "transformation score"; ranking uses ordered
  transparent criteria after hard filtering.
  `NO_TRANSFORMATION_STRATEGY_WORTH_USING` is a successful, cache-eligible
  completed result that may contain zero recommended strategies.
- **Bounded strategy directions.** At most three recommended and three useful
  rejected directions from the closed 15-value set, each carrying independent
  assessments, preservation requirements, verification flags, and a stable
  strategy fingerprint. Directions are not scripts and contain no narration,
  timeline, TTS, or rendering fields. Value-focus text quotes or names concrete
  candidate evidence, and ungrounded provider directions are rejected.
- **Providers are optional.** `deterministic` makes zero Gemini/Qwen calls;
  `adaptive` (default) selects one hosted tier per analysis
  (`gemini-3.5-flash-lite` routine, `gemini-3.8-flash` low-thinking for
  complex/transformation-required cases) through the shared HIGH admission gate,
  at most one hosted call, never a Qwen fallback; `local_only` uses Qwen only
  when `CLIPFACTORY_LOCAL_QWEN_ENABLED=true`. A missing key/outage/429/malformed
  output preserves deterministic results, never fails the analysis, and keeps
  the run non-cache-eligible only while selected hosted work is unfinished.
- **Rights vs originality stay separate.** Unknown/third-party provenance remains
  analyzable and usually raises transformation necessity
  (`TRANSFORMATION_REQUIRED`); `UNRESOLVED_POLICY_OR_PROVENANCE_RISK` is reserved
  for explicit stored conflicts. The platform-risk snapshot is decision support,
  not legal or monetization advice; official platform guidance was checked
  2026-09-14.
- **Cache, freshness, concurrency.** Queue-time cache validation and Stage 4.1
  handoff freshness use the same settings-derived config/mode and **stable
  configured provider identity** as execution. That identity is derived from
  configuration only, so transient Gemini/key/provider unavailability does not
  invalidate accepted analysis (cache stays a hit, no replacement job, force
  reruns reuse accepted hosted output; handoff stays current), while a real
  model/prompt/schema/provider/config change still invalidates. Concurrent queue
  requests are transaction-safe (savepoint uniqueness recovery plus an atomic
  `active_job_id IS NULL` compare-and-swap), yielding one analysis and at most one
  active job with no escaped `IntegrityError`/500.
- **API/CLI/handoff.** Minimal endpoints queue/read one analysis and expose a
  typed read-only Stage 4.1 handoff (`stage4_1_implemented=true`) that now also
  exposes source-moment evidence, aggregate assessments, candidate coarse bounds/
  segment indexes, and sanitized provider evidence, and reports stale instead of
  mixing current transcript data with old strategies.

Deterministic verification: 72 focused Stage 4.0 tests (passing both with and
without a Gemini key present) plus the full backend suite in Docker Python 3.12
with the repository compose/.env files mounted, with no live provider calls in
the automated suite. See
[docs/STAGE_4_0_OPERATIONS.md](docs/STAGE_4_0_OPERATIONS.md).

## Stage 3.7 performance instrumentation (2026-09-12)

Stage-level durable metrics now distinguish remote metadata lookup, transfer,
artifact bytes, bounded directory-scan cost, postprocessing applicability,
ffprobe, WAV cache validation/extraction, and transcription execution settings.
The established 29.25-minute authorized WebM baseline remains INGEST 609.43 s,
TRANSCRIPTION 1080.27 s, and CANDIDATE_ANALYSIS 2.35 s. The authorized
non-destructive replay of the thread-only cached-WAV experiment
(`large-v3-turbo`, CPU int8, beam 5, automatic language, timestamps, VAD off,
batch one, eight CPU threads) took 1024.90 s, 5.1% faster than baseline. It
preserved 4,712 word timestamps and all 24 retained deterministic Stage 3
candidates. It is selected; rollback-safe defaults remain CPU threads eight,
batch one, and VAD off. No batch or VAD test was justified. Full remote video is
still acquired before analysis; audio-first/deferred-video acquisition was not
implemented. See [benchmarks](docs/BENCHMARKS.md).
INDEX batch size is constrained to one until a compatible batch adapter and
semantic replay validation exist.

Stage 3 candidate analysis extends the local-first ingest/transcription
foundation through `READY_FOR_REFINEMENT`. Stage 2.7.1 reaches
`READY_FOR_ANALYSIS`; worker orchestration then runs the durable
`CANDIDATE_ANALYSIS` stage, which discovers coarse clip moments from the
imperfect INDEX transcript and advances the source to `READY_FOR_REFINEMENT`.
Stage 3 semantic mode defaults to deterministic (zero Gemini calls, zero Qwen
loads). Stage 3.5 then provides explicit, candidate-scoped, audio-verified
transcript and boundary refinement (candidate 5/5 s, final 8/8 s context, max
150 s) through the existing job system, with optional selective Gemini
transcription/adjudication behind a shared priority/budget gate. Rendering,
publishing, review UI, and authorization remain out of scope; Stage 4.0 is
implemented (explicit, candidate-scoped); Stage 4.1, Stage 4.2, and Stage 4.3 are implemented.

## Stage 3.5 candidate-scoped refinement (2026-09-12)

Stage 3.5 turns one explicitly requested Stage 3 coarse candidate into a
trustworthy, precisely bounded, audio-verified transcript while spending
expensive compute only on requested candidates/final clips:

- **Candidate-scoped, not an automatic stage.** Work is explicit after
  `READY_FOR_REFINEMENT` and never added to the automatic `_NEXT_STAGE` chain; a
  source may stay `READY_FOR_REFINEMENT` while individual candidates refine
  independently. It extends the existing Celery/`ProcessingJob` system with a
  `CANDIDATE_REFINEMENT` job kind and a `candidate_refinements` row per
  `(clip_candidate_id, priority)`; no new `PipelineStage`, `PipelineRun`, queue,
  scheduler, service, or billing system. Only `CANDIDATE`/`FINAL_CLIP` are valid;
  `INDEX` is rejected.
- **Bounded audio only.** A StorageService-owned window service validates the
  cached mono 16 kHz artifact against the source hash, then extracts only the
  bounded candidate interval from the original source with a safe FFmpeg
  argument list (candidate pre/post 5/5 s, final 8/8 s, max 150 s, path
  `sources/{source_id}/candidate-refinements/{candidate_id}/{priority}.wav`),
  validates duration/non-empty output/hashes, replaces atomically, and removes
  partial temp files. Context bounds and refined bounds persist separately;
  context audio never becomes the final clip. The whole source is never
  retranscribed or uploaded.
- **Targeted local Whisper backbone.** `large-v3-turbo`, word timestamps,
  automatic language, no VAD, no condition-on-previous-text, beam 5 (candidate)
  / 8 (final), deterministic temperature fallback; reuses `WhisperEngine`,
  child-process isolation, peak-memory reporting, and the shared heavy-model
  lease. Clip-relative word times convert once to source time and validate
  within the context window. Raw ASR, timestamps, words, and Stage 2 evidence are
  never rewritten.
- **Omitted-English recovery from audio.** A targeted audio result may restore
  previously omitted Latin-bearing speech (e.g. INDEX `أنا عملت امبارح` → audio
  `أنا عملت deploy للbackend امبارح`) and persists a code-switch recovery metric
  only when it survives validation. A text-only model/`local_only` Qwen path can
  never add omitted English and never calls Gemini. Source dialect is preserved.
- **Hosted providers are optional and selective.** `gemini-3.5-transcribe`
  through the documented Interactions API in verbatim mode with word timestamps
  (never smart mode, custom vocabulary never combined with timestamps, one
  bounded clip per request, remote file deleted in `finally`, URI never
  persisted, SDK client lazy and closed on exit, key never logged). Candidate
  mode calls hosted only for material uncertainty; final mode calls whenever
  configured unless an authoritative manual transcript already satisfies the
  strict gate. `gemini-3.8-flash` adjudication chooses one supplied reading or
  `UNRESOLVED` and can never rewrite the transcript. Official docs checked
  2026-09-12; SDK `google-genai` 2.23.0.
- **Shared Gemini priority gate.** A Redis-backed atomic fixed-window admission
  controller (`CRITICAL`/`HIGH`/`MEDIUM`/`LOW`/`AVOID`, configurable reserves
  that cannot exceed capacity) decides before any upload/generation; Redis
  failure fails closed for hosted work while local continues; 429/`Retry-After`
  establishes a bounded cooldown. Stage 3 semantic calls pass through MEDIUM;
  Stage 2.7 maps FINAL_CLIP→CRITICAL, CANDIDATE→MEDIUM, INDEX→AVOID.
- **Deterministic acceptance, entities, and boundaries.** Protected
  names/numbers/dates/URLs/technical/Latin tokens, dialect preservation,
  translation/MSA conversion, edit/insertion ratios, repetition, material
  ASR disagreement, and entity normalization are checked. Candidate mode may
  keep a flagged ambiguity; final meaning-critical ambiguity becomes
  `NEEDS_MANUAL_TRANSCRIPT_REVIEW` and is never resolved arbitrarily. Boundaries
  are refined conservatively within the radius, never default to the full
  context window, and retain the coarse edge on weak evidence.
- **Caching, idempotency, cancellation.** Top-level and component fingerprints
  (audio/local/hosted/adjudication/consensus/boundary) always distinguish
  candidate from final; components are reused only on exact dependency match;
  degraded runs stay non-cache-eligible; manual text always wins and is never
  cleared. The exact executing job is polled around every expensive step and
  during Whisper; cancellation stays `CANCELLED`, preserves checkpoints,
  deletes remote/temp files, releases leases/providers, and never produces a
  ready state.
- **API/CLI/handoff.** Queue/get/list one candidate refinement or a bounded
  candidate-grade batch (score-ordered, default 5, max 10, no bulk FINAL_CLIP),
  submit manual text/resolutions, and fetch a typed read-only Stage 4 handoff
  (refined transcript/exact bounds + Stage 3 evidence; never mislabels
  candidate-grade output as final-ready). Stage 4.0, Stage 4.1, Stage 4.2, and
  Stage 4.3 are implemented, and `FINAL_TRANSCRIPT_READY` is not publishing
  readiness.

Deterministic verification adds Stage 3.5 tests plus the full existing suite:
780 backend tests pass (Docker Python 3.12), coverage 87% (gate 79%). Stage 3.5
defaults to adaptive with no live provider calls in the automated suite. See
[docs/STAGE_3_5_OPERATIONS.md](docs/STAGE_3_5_OPERATIONS.md).

## Stage 3 candidate analysis (2026-09-12)

Stage 3 discovers the best potential clip moments cheaply from an imperfect
INDEX transcript, preserves promising moments when transcription is uncertain,
and defers expensive audio/transcript refinement to Stage 3.5:

- **Pipeline position.** `AUDIO_ANALYSIS -> READY_FOR_ANALYSIS ->
  CANDIDATE_ANALYSIS -> READY_FOR_REFINEMENT`. `AUDIO_ANALYSIS` still records
  `READY_FOR_ANALYSIS`; worker orchestration then queues `CANDIDATE_ANALYSIS`.
  Failure or cancellation leaves the source at `READY_FOR_ANALYSIS`; success
  advances it to `READY_FOR_REFINEMENT`. A dedicated job kind and retry mapping
  were added without a new queue, scheduler, service process, or orchestration
  subsystem. Existing sources at `READY_FOR_ANALYSIS` are analyzable through
  `POST /api/sources/{id}/candidate-analysis` or `python -m app.cli
  candidate-analysis`.
- **Source provenance.** `SourceVideo.rights_status` is reused; `media_origin`
  (default `OTHER` = unclassified) and a bounded `provenance_metadata` JSON
  object were added. Provenance is set at creation or through an explicit
  `PATCH /api/sources/{id}/provenance`; duplicate ingest never mutates it.
  Provenance changes invalidate Stage 3 only. Unknown/third-party provenance
  never blocks local analysis, and rights risk is separate from
  originality/transformation risk.
- **Deterministic discovery.** A pure one-pass service selects analysis text in
  operator/final/corrected/raw priority, forms coarse windows from timestamps,
  silence/pause midpoints, sentence punctuation, speaker changes, question/answer
  and contrast/topic transitions, story/payoff cues, and RMS energy changes, with
  a fixed fallback only when boundaries are unavailable. Proposals are built over
  bounded atoms; a segment longer than the maximum coarse window is split at
  deterministic word/timestamp boundaries (or a proportional-character fallback
  when only text exists), every atom is then hard-bounded so no pathological
  single word/timestamp span survives, and source duration is the hard outer
  bound. Stable atom-span identity keeps multiple windows from one segment
  non-colliding across reruns. The funnel is broad discovery → loose configurable
  raw safety cap (96/hour, 960/source; CPU/memory protection only) → full
  deterministic scoring/classification/novelty/refinement → rank by `clip_score`
  → tight shortlist caps (15 s minimum, 35–75 s preferred, 120 s
  maximum, 24 proposals/hour, 240/source, 60 retained) → optional provider work;
  these are caps, not targets. Overlapping similar proposals merge; distinct ideas
  stay separate; zero good moments yields zero accepted candidates.
- **Scoring separation.** Independent normalized scores are persisted
  (`clip_score`, `short_form_score`, `moment_density_score`,
  `boredom_risk_score`, `ending_quality_score`, `loopability_score`,
  `engagement_confidence`, `transcript_confidence`, `audio_confidence`,
  `boundary_confidence`, `uncertainty_severity`, `idea_novelty_score`,
  `topic_novelty_score`, `recent_semantic_similarity_risk`). Content-quality
  scores are computed by one shared aggregate used by both deterministic and
  provider-enriched paths; `clip_score` never includes transcript
  cleanliness/confidence, unresolved INDEX state, word/boundary confidence,
  code-switch uncertainty, provider availability, or audio confidence. A
  zero-adjustment provider response preserves the deterministic `clip_score`, and
  an INDEX-deferred candidate keeps the same content score as its clean
  equivalent while still routing to `CANDIDATE_NEEDS_REFINEMENT`. A filler moment
  with perfect transcript confidence stays low quality. After a long segment is
  split, word/acoustic/low-confidence and protected/code-switch evidence is
  filtered to each candidate's actual coarse time span.
- **Refinement-needed behavior.** Strong non-redundant content with material
  uncertainty becomes `CANDIDATE_NEEDS_REFINEMENT` with bounded reason codes
  (`LOW_TRANSCRIPT_CONFIDENCE`, `UNRESOLVED_INDEX_TEXT`,
  `LOW_CONFIDENCE_WORD_SPAN`, `CODE_SWITCH_UNCERTAINTY`,
  `PROTECTED_ENTITY_UNCERTAINTY`, `LOW_BOUNDARY_CONFIDENCE`); code switching
  alone never triggers refinement. Content below threshold becomes `DO_NOT_CLIP`;
  the weaker of strongly redundant candidates becomes
  `DO_NOT_CLIP_RECENTLY_REDUNDANT`.
- **Content and hooks.** A closed 14-value content ontology with a deterministic
  Arabic/English cue classifier (provider values must be declared enum members),
  and at most three source-faithful hooks per retained proposal from eight hook
  types. Deterministic cue/classification/hook matching runs against an
  analysis-only normalized view (safe Unicode normalization, English
  case-folding, Arabic diacritic/tatweel removal, conservative alif/ya unify)
  without rewriting stored text, timestamps, numbers, names, URLs, technical
  forms, protected tokens, or hook display text. Cue vocabularies stay small with
  common Egyptian, Gulf/Saudi, Levantine, MSA/Fusha, and English variants.
  Provider hooks are strictly validated; invented/changed protected tokens or
  numbers are rejected and deterministic hooks are kept.
- **Novelty.** Same-source overlap+similarity merges, same-source canonical
  idea/topic signatures catch non-overlapping repeats, and a bounded recent
  cross-source corpus (500 current non-rejected candidates) provides duplication
  risk. Deterministic novelty is the cheap first pass and initial redundancy
  filter; clearly redundant candidates never consume provider quota. After
  accepted provider enrichment, novelty/disposition is recomputed for eligible
  retained candidates using improved summaries, without reviving a weak candidate
  or un-redundanting an already-redundant one. Provider-free mode uses stable
  Arabic/English tokenization, stopwords, unigrams/bigrams, and TF-IDF cosine. No
  vector database or embedding service. Recurring channel/history diversity is
  deferred to Stage 7.
- **Semantic providers.** A Stage 3-specific protocol/schema (not the Stage 2.7
  reconstruction prompt). `deterministic` (default) makes zero Gemini/Qwen calls;
  `adaptive` selectively batches strongest-first through Gemini only when a key is
  configured and never falls back to Qwen; `local_only` uses Qwen/Ollama only when
  `CLIPFACTORY_LOCAL_QWEN_ENABLED=true` and never Gemini. Hard caps: 32
  provider-evaluated candidates/source, 8/request, 4 calls/source, bounded
  input/output tokens, temperature 0. Rate limits stop later calls; malformed
  items are isolated. Any requested candidate missing a valid accepted result
  makes the run non-cache-eligible/retryable (`PROVIDER_PARTIAL`); accepted
  evaluations are persisted on candidate rows and a later rerun reuses them,
  retrying only the missing/invalid candidates whose provider-input fingerprint
  still matches. Provider failure degrades safely and never fails the pipeline.
- **Persistence.** `candidate_analyses` (one-to-one source summary) and
  `clip_candidates` (deterministic `candidate_key`, current/stale marker, bounded
  evidence, queryable scores, separate rights/originality risk, signatures,
  provider evidence) with database constraints. The input fingerprint covers every
  output-affecting config value plus semantic mode and stable provider identity;
  the output fingerprint covers the complete persisted candidate representation.
  Upsert preserves UUIDs for unchanged intervals; stale marking happens only after
  a successful finalization; historical candidates are never deleted. The Stage 3
  migration downgrade removes Stage-3-only rows/history and maps the source
  lifecycle back to `READY_FOR_ANALYSIS` before narrowing constraints, preserving
  all pre-Stage-3 data.
- **Minimal read-only view.** The source detail page shows a "Clip candidates"
  card (summary line plus compact rows with time-range seek, score, disposition,
  content type, excerpt, refinement reasons, and a show-rejected toggle) backed by
  the candidate-analysis/candidates endpoints. No candidate editing or
  refinement UI.
- **Stage 3.5 handoff.** Accepted/refinement-needed candidates persist the
  segment indexes, INDEX excerpt, low-confidence spans, unresolved evidence,
  refinement reasons/severity, dialect evidence, code-switch/protected tokens,
  boundary confidence, fingerprints, and provenance/originality snapshot so Stage
  3.5 can extract audio, refine transcription/boundaries, and reach
  publication-quality text.

Deterministic verification covers the focused Stage 3 candidate/API/migration
tests plus the full existing suite: 703 backend tests pass (Docker Python 3.12).
Stage 3 defaults to deterministic and makes no live provider calls in the
automated suite. See
[docs/STAGE_3_OPERATIONS.md](docs/STAGE_3_OPERATIONS.md).

## Stage 2.7.1 dialect-aware preservation (2026-09-10)

Stage 2.7.1 adds conservative source-level Arabic dialect awareness and exact
Arabic-English code-switch preservation:

- **Dialect profiles.** `EGYPTIAN`, `SAUDI`, `GULF`, `LEVANTINE`, `MSA`, and
  `UNKNOWN_ARABIC`. `None` means Arabic profiling is not applicable (no Arabic
  evidence); `UNKNOWN_ARABIC` means Arabic is present but the regional/formal
  profile is uncertain, mixed, unsupported, or insufficiently evidenced and
  always favors no change. Dialect describes the speech actually present in the
  source; it is not a target audience or localization choice, and there is no
  deployment-wide dialect default.
- **Lightweight deterministic detection.** A pure detector (no network, no
  LLM, no model loading, no audio decoding) classifies the source during Stage
  2.5 normalization from immutable raw segment text. Marker scoring is bounded
  to a representative sample of at most 48 segments, but Arabic applicability
  scans every immutable raw segment, so an Arabic segment outside the sample can
  never make the source `None`/not-applicable. A known profile needs at least
  two distinct markers, a weighted score of at least 4, a lead of at least 2
  over the runner-up, and a score of at least 1.5×max(runner_up, 1); MSA
  additionally needs strong formal evidence and no meaningful competing
  colloquial score. Competing, weak, or mixed evidence resolves to
  `UNKNOWN_ARABIC`; non-Arabic material resolves to `None`. Detected confidence
  is bounded to [0.80, 0.99] with a deterministic formula; unknown/not-applicable
  use 0.0. Evidence is bounded and stores no transcript bodies.
- **Operator override.** An optional `dialect_profile_override` may be supplied
  when a source is created through URL ingest JSON or multipart upload. It wins
  with confidence 1.0, participates in normalization fingerprints, and an
  explicit `UNKNOWN_ARABIC` override means "do not force a dialect." The initial
  implementation accepts the override at source creation only; there is no
  post-ingest editing workflow in this stage. Duplicate source creation never
  mutates an existing source's override.
- **Egyptian Stage 2.5 isolation.** The Egyptian lexicon and its optional
  provider apply only when the effective profile is confidently or explicitly
  EGYPTIAN. SAUDI, GULF, LEVANTINE, MSA, UNKNOWN_ARABIC, and non-Arabic material
  pass through unchanged with zero optional Stage 2.5 provider calls, and
  candidate-null segments are never sent to the optional correction provider.
- **Code-switch and protected tokens.** Exact Latin words/names,
  abbreviations, technical forms, and Western/Arabic-Indic numbers (including
  compound numeric/date forms) are extracted as ordered protected tokens and
  preserved through Stage 2.5 and every accepted Stage 2.7 candidate; a
  candidate that removes, replaces, reorders, changes case, Arabicizes, or
  invents a protected token is rejected, and slash/`+`/`#`/URL technical forms
  are kept as exact atomic tokens (URLs include query, fragment, parameter, and
  percent-encoded syntax). `code_switch_suspected` is true only when a
  segment itself contains both Arabic-script evidence and Latin-letter-bearing
  protected-token evidence (numbers alone and English-only segments are not
  flagged). Omitted-English audio recovery is
  deferred to Stage 3.5.
- **Shared provider contract.** Both the local Qwen/OpenAI-compatible provider
  and the hosted Gemini provider receive the same dialect-neutral,
  preservation-first reconstruction instruction and validated profile-specific
  addenda (the local provider's old Egyptian-only base instruction was removed
  and Gemini's divergent neutral instruction was folded into the shared
  builder). The dormant `dialect_profile` request field is populated from the
  target segment's inherited effective profile, and local aggregate request
  planning sizes the exact profile-specific instruction.
- **Fingerprints.** Normalization input/output fingerprints are versioned and
  now cover the stored operator override, detector policy version, effective
  profile/confidence/selection, the dialect-aware correction identity
  (correction policy, detector policy, Egyptian lexicon version, output
  thresholds), and the preservation policy version. Reconstruction whole-output
  and per-target fingerprints are versioned and include per-segment dialect
  identity and code-switch tokens. Dialect/profile/override changes invalidate
  Stage 2.5 and Stage 2.7 derived work at their correct boundaries without
  retranscribing audio; per-target checkpoint reuse respects dialect identity.
- **INDEX remains cheap.** Whole-source ingestion still performs zero Qwen and
  zero Gemini calls, never constructs/probes providers for dialect detection,
  preserves uncertainty and code-switch evidence, and reaches
  `READY_FOR_ANALYSIS`. Dialect presence or code switching is never itself a
  reason to spend Gemini quota. CANDIDATE/FINAL_CLIP targeted refinement
  consumes the stored target profile through the shared request contract, and
  `refine_transcript_window` never redetects or overwrites the source dialect.

The known regression benchmark findings below remain historical evidence and are
not readiness proof.

## Stage 2.7 finalization: INDEX ingestion and targeted refinement (2026-09-10)

Whole-source transcription is indexing, not publication. Normal whole-source
ingestion now runs at **INDEX** priority and defers every provider
reconstruction truthfully instead of spending Qwen/Gemini time across arbitrary
source segments:

- **Refinement priorities.** `INDEX` (whole-source default, indexing quality),
  `CANDIDATE` (shortlisted window, semantic quality), and `FINAL_CLIP` (selected
  clip, publication/caption quality). Priority participates in reconstruction
  fingerprints, so an INDEX result can never satisfy a CANDIDATE/FINAL_CLIP
  request and one window can never satisfy another.
- **INDEX is cheap and evidence-preserving.** Normal ingestion pays for ASR +
  Stage 2.5 + cheap uncertainty bookkeeping. It makes zero Qwen calls (Qwen is
  not loaded) and zero Gemini calls, preserves raw ASR/Stage 2.5/timestamps/word
  and acoustic evidence, marks deferred spans unresolved (never a provider
  failure), and still reaches the analysis-ready success state.
- **Local Qwen disabled by default.** `CLIPFACTORY_LOCAL_QWEN_ENABLED=false` is
  the default; the operator re-enables local Qwen explicitly for targeted work.
  Ollama integration, local-provider config, provider tests, and `local_only`
  routing remain intact. Intentional `local_only` still uses Qwen (never Gemini)
  when configured.
- **Gemini is reserved for targeted candidate/final work.** No blanket
  whole-source cleanup; the bounded routing/budget gates and per-job target cap
  are unchanged.
- **Reusable targeted refinement.** `refine_transcript_window(source_id,
  start_time, end_time, priority)` refines only the bounded requested region,
  reuses the Stage 2.5/2.7 provider/routing/validation/checkpoint mechanisms,
  preserves raw ASR/timestamps and manual overrides, and returns a structured
  outcome (refined text, confidence, status, provider/routing evidence,
  unresolved state, target/window identity). Bounds: max window 300 s, max 32
  targets (both configurable).
- **Local-wall ceiling is authoritative over the Gemini backlog.** Once the
  ceiling expires, no new local-origin escalation is enqueued and queued
  escalations are invalidated (`local_escalations_dropped`); no Gemini call is
  made for the backlog, safe Stage 2.5/current text is preserved, and targets are
  marked unresolved/manual review.
- **Uncertainty handoff.** Persisted `refinement_priority`, derived
  `needs_refinement` and `code_switch_suspected` (Stage 2.7.1 evidence-based;
  no omitted-English audio recovery, which is deferred to Stage 3.5), plus the
  existing status/confidence/focus-span/provider/routing evidence, give future
  stages what they need to decide targeted refinement.

The known regression benchmark findings below remain historical evidence and are
not readiness proof.

Stage 2.7 has not yet passed its required private, authorized unseen-audio
benchmark. No quality, latency, RAM, VRAM, or Stage 3 readiness claim is made
until that evaluation manifest and human review are available. The known
Chernobyl diagnostic run is regression evidence only and is never counted as
unseen readiness.

## Stage 2.7 completion gate

Stage 2.7 is complete only when every item below has current, direct evidence.
A missing or indirect proof is a failed gate.

| # | Gate | Evidence | Result |
| --- | --- | --- | --- |
| 1 | Local provider health `AVAILABLE`; live worker invokes it | `python -m app.cli reconstruction-health` | PASS (digest `2a654d98e6fb…eefd`) |
| 2 | Provider regression tests and real audio prove multi-word repair | provider tests; benchmark comparison rows | FAIL (provisional: one repair applied under 0.82 gate; strict unseen set still missing; pre-fix reports non-comparable) |
| 3 | Raw ASR text and all timestamps unchanged through downstream stages | `test_reconstruction_persistence.py` deep-equality | PASS |
| 4 | Forced retranscription reruns every stale transcript-derived stage | `test_pipeline_fingerprints.py` | PASS |
| 5 | Media/audio and transcript quality separate; bad sample no longer reports high transcript quality | `test_transcript_quality.py` | PASS |
| 6 | Unavailable provider/model visible in persistence, health, API, CLI, UI | `test_reconstruction_status.py`, API/UI tests | PASS |
| 7 | Real unseen Egyptian benchmark improves materially | private unseen-audio benchmark | FAIL (no unseen-audio set) |
| 8 | Regression ≤2%, preserved-correct ≥98%, hallucinated = 0 | benchmark aggregate | FAIL (no valid aggregate; pre-fix aggregates invalid evidence) |
| 9 | Chernobyl first 30 seconds manually re-tested | diagnostic comparison rows | FAIL (pre-fix reports non-comparable; must re-run on the committed evaluator and fingerprint) |
| 10 | All Stage 2/2.5/2.6/2.7 backend and frontend tests pass | pytest + vitest | PASS (354 backend, 11 frontend) |
| 11 | README, STATUS, AGENTS, ENVIRONMENT, architecture, pipeline, benchmark, local setup, troubleshooting match installation | documentation | PASS |
| 12 | Infrastructure reliability: persistent unsafe state, lease-loss handling, capture provenance, child memory telemetry | lifecycle/provenance tests | PASS |
| 13 | Small practical acceptance review on current `large-v3-turbo` sources | human review below | PASS (semantically usable; meaning-changing errors rare) |
| 14 | Final report ends with exactly one terminal status line | below | — |

The Stage 2.7 correctness foundation was fixed and committed on 2026-09-07:
provider output is validated at a strict boundary, confidence is the one scalar
the model returns, failures are isolated per segment, prompts are budgeted over
the complete chat envelope, evaluation separates runtime status, deterministic
text comparison, literal exact comparison, and human labels, and reconstruction
fingerprints include the full runtime identity. Because the evaluator and
confidence pipeline changed, **every benchmark report produced before the
correctness-foundation commit is non-comparable**; their semantic and exact
aggregates are invalid evidence and their run IDs are historical artifacts only.

Earlier benchmark findings in `docs/BENCHMARKS.md` remain as history:
`qwen3:8b` was infeasible on the 7.4 GiB machine (out-of-memory kill during
load), the one-pass small-context protocol with `qwen3.5:4b` was feasible
end-to-end, `qwen3:4b` cannot be parsed (thinking mode exhausts the 256-token
budget), and `qwen2.5:7b` could not be pulled reliably. None of this is unseen
readiness evidence. A fresh, correctly-fingerprinted run is required before any
new model or threshold decision.

See the task report for the latest local verification evidence. Copy
`.env.example` to `.env` before starting Compose.

## Heavy-model lifecycle status (2026-09-07)

The memory and heavy-model lifecycle plan is implemented and its checkpoint
passed: a read-only `diagnose-memory` command and `scripts/diagnose-memory.sh`
label host/WSL/cgroup/process/container/swap memory; a Redis-backed
`clipfactory:heavy-model` lease serializes Whisper and Ollama across workers and
the CLI; Whisper runs in a spawned child process that is reaped before the lease
releases; Ollama unload is verified by polling `/api/ps` and recorded as
`unload_outcome`; and Celery runs `--pool=solo` (pre-fork daemonic workers
cannot spawn the child) with concurrency 1 and `PYTHONPATH=/app`.

The operator applied `[wsl2] memory=11GB swap=4GB`; Linux/Docker now report
about 10.69 GiB effective capacity. Three sequential measured trials passed:
Whisper and Ollama never overlapped, `ollama ps` was empty after each
reconstruction, unload confirmed in under a second, swap growth was effectively
zero, and no OOM occurred. This validates the lifecycle only; no model-quality
or unseen-readiness claim is made. The 8B model has not been downloaded or
benchmarked.

## Stage 2.7 model and ASR quality status (2026-09-07)

The immutable ASR capture contract and complete benchmark evidence recording were
implemented and tested. A known-regression corpus was captured on
`large-v3-turbo`. Findings:

- `qwen3:8b` (Q4_K_M) now loads without OOM under 10.69 GiB (it was OOM-killed
  at 7.44 GiB), but a full 8B replay hung, so the three-run reliability gate is
  not met.
- `qwen3.5:4b` repaired 0 of the three known phrases (`فيور 25 نوفمبر`,
  `آخره يشيلت نصر واحد`, `فيه 71`) on the frozen capture.
- Full `large-v3` recovered `فيور` (one ASR_AUDIO fix) but missed `اتناشر`;
  below the two-fix gate, so `large-v3-turbo` remains the default.
- The unseen-corpus human references are not yet available; the stricter
  model/ASR benchmark gate cannot be authorized. A stricter 8B/ASR reliability
  and quality evaluation remains open. This is historical benchmark evidence;
  Stage 2.7.1 dialect awareness is implemented and separately verified by
  deterministic tests, not by these model-quality findings.

## Infrastructure and telemetry fixes (2026-09-08)

Persistent unsafe-model state, lease-loss handling, capture provenance
verification, and child memory telemetry were implemented and tested:

- Unsafe model residency is a persistent Redis marker with no TTL
  (`clipfactory:heavy-model:unsafe`). It survives worker restart, CLI exit, and
  lease TTL expiry, and blocks every new heavy-model acquisition until an
  operator runs `python -m app.cli recover-heavy-model`, which clears it only
  after confirming the model is no longer resident.
- Acquisition and operator recovery are each a single Redis-side Lua script.
  Acquisition checks the unsafe marker and takes the lease atomically, so a
  marker written while a waiter retries still blocks it; recovery clears the
  stale lease and the unsafe marker in one script with no gap in which a new
  acquisition can occur, and never deletes a valid newly acquired lease.
- A lost or unrenewable lease cancels and reaps the active Whisper child,
  records persistent unsafe state, and fails the stage closed; overlapping
  heavy jobs cannot start after lease expiry.
- Immutable ASR replay verifies clip id, source id, original-media SHA-256,
  exact clip audio SHA-256, exact start/end bounds, schema version, and decoder
  identity before replay; a matching clip id alone never authorizes a run.
  Multiple clips from one source keep distinct clip hashes and bounds.
- Whisper child peak memory is read in the child after model work. Abnormal
  child exits report `UNKNOWN` peak instead of a fabricated value.

## Small practical acceptance review (2026-09-08)

Two real authorized Arabic sources transcribed with `large-v3-turbo` were
reviewed segment by segment for meaning-changing transcript errors:

| Source | Segments | Usable | Minor variants | Meaning-changing | Nonsense/garbage | Number/name/fact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chernobyl narrative (159 s) | 76 | 49 | 8 | 1 | 18 | 0 |
| Cuba/Granma narrative (122 s) | 61 | 44 | 2 | 0 | 13 | 2 |
| **Total** | **137** | **93** | **10** | **1** | **31** | **2** |

The vast majority of segments are semantically usable; a downstream Stage 3
model can recover the speaker's meaning from the final transcript. The
meaning-changing errors are localized to high-fan-out dialectal speech and
numbers (for example `إخلاء` → `إخلاق`, `اتناشر` → `نصر`, `70` → `71`), and the
known-regression phrases remain unrepaired by `qwen3.5:4b`. A third
authorized Arabic source (Guatemala narrative) exists but was transcribed with
the obsolete `small` model and is not evidence for the current stack.

The production stack is unchanged: ASR `large-v3-turbo`, reconstruction
`qwen3.5:4b`. The infrastructure gates pass and the small practical acceptance
review shows the current transcript is semantically usable with rare
meaning-changing errors, so the unrepaired known-regression phrases no longer
block infrastructure readiness. The strict unseen-audio benchmark and a stricter
8B/ASR reliability and quality evaluation remain open; until then, manual
correction or Stage-3 exclusion of harmful transcript segments remains the short
practical quality path. Stage 2.7.1 dialect awareness is implemented and is not
the same gate as model-quality authorization.

## Hosted Gemini adaptive routing (2026-09-09)

Stage 2.7 now supports an optional hosted Gemini reconstruction provider with
deterministic routing. `qwen3.5:4b` remains the normal private quota-free path.
Three routing modes exist: `local_only` (never calls Gemini), `adaptive`
(default), and `gemini_only` (skips Qwen). The default is `ADAPTIVE` because a
configured live Gemini key passed the tiny smoke test; a missing key falls back
to local behavior exactly.

Routing is a single deterministic policy (`route_adaptive`) with centralized
constants. Targets that Stage 2.5 already trusts (`NO_LLM`) consume no provider
calls. Mild localized uncertainty uses Qwen. Clearly difficult segments
(contiguous very-low-probability words, large low-confidence spans, severe
routing scores, or uncertainty overlapping a protected number/name) use one
Gemini request directly. In ADAPTIVE mode, unresolved, malformed, failed, or
near-accepted Qwen results escalate to one Gemini attempt. Every Gemini
candidate passes the same shared validation and confidence/acceptance gates;
Gemini is never automatically authoritative.

Gemini is treated as scarce: `CLIPFACTORY_GEMINI_MAX_TARGETS_PER_JOB` (default
5) caps Gemini reconstruction targets per job, accepted local results never
call Gemini, full transcripts are never sent (only bounded windows), a 429/quota
exhaustion stops further Gemini calls for that job, and identical completed work
reuses its fingerprint without a duplicate Gemini call. `ADAPTIVE` and
`GEMINI_ONLY` may send short transcript snippets and bounded context to Google
Gemini; this is configuration, not rights/provenance policy. The cap reserves
Gemini capacity but is not an account-wide billing/quota manager.

The initial live smoke used the operator's configured `GEMINI_API_KEY` (never
printed, logged, or committed) against `gemini-3.6-flash`, verifying
authentication, model availability, structured-output parsing, and readable
usage metadata. The corrective pass superseded the model with
`gemini-3.8-flash` (see below). The operator retains the local-first default in
`local_only` by setting `CLIPFACTORY_RECONSTRUCTION_ROUTING_MODE=local_only`.

## Adaptive Gemini corrective pass (2026-09-09)

> Superseded in part by the performance corrective pass below: `NO_LLM` trust no
> longer depends on the correction method alone; clean well-covered unchanged
> results and trusted repairs that resolved their uncertainty are the trust
> basis.

A focused corrective pass finalized the adaptive router before Sol review:

- **Stage 2.5 trust routing.** `NO_LLM` now requires affirmative Stage 2.5
  evidence (`correction_method` not `unchanged`/`pending` at
  `correction_confidence >= 0.90`) plus clean acoustics and no protected-token
  ambiguity. High Whisper probabilities alone can no longer suppress contextual
  checking; a confidently wrong segment with an unchanged low-trust correction
  stays eligible for Qwen. Isolated low-confidence words are labeled accurately.
- **Strongest-first budget.** `CLIPFACTORY_GEMINI_MAX_TARGETS_PER_JOB` (default
  `5`) is spent on the five strongest eligible targets by deterministic routing
  severity, not transcript order. Direct-Gemini targets are ranked and allocated
  first, then Qwen, then ranked local escalations; ties break by segment index.
- **Lazy heavy-model lease.** The Ollama lease is acquired only around actual
  local inference and local release. `NO_LLM`, `GEMINI_ONLY`, and direct-Gemini
  work never acquire it; a direct-Gemini call runs before any optional local
  fallback lease. Real local inference stays lease-protected.
- **Secret handling.** The Gemini key is a Pydantic `SecretStr`, masked in repr,
  `model_dump`, JSON, and validation errors, and unwrapped only when building the
  SDK client.
- **Stable fingerprints and cache eligibility.** Fingerprints cover stable
  dependency identity only; transient availability is excluded. A temporary
  outage never invalidates accepted output, and a first-run degraded fallback is
  retried after provider recovery. `cache_eligible` distinguishes reusable runs.
- **Dialect-neutral Gemini prompting.** Gemini preserves the dialect/register
  evident in the source and context; it never defaults to Egyptian. A future
  `dialect_profile` request hint receives a narrow profile-preservation addendum.
- **Model and thinking.** Default is `gemini-3.8-flash` with
  `CLIPFACTORY_GEMINI_THINKING_LEVEL=low` (bounded reasoning), passed through
  `ThinkingConfig`, with `thoughts_token_count` retained in usage.
- **Retry classification.** Permanent 400-class request/schema failures,
  authentication, model-not-found, 429, malformed output, and refusal are never
  retried; connection/timeout/eligible 5xx get at most one bounded retry.
- **SDK cleanup.** `GeminiReconstructionProvider.release()` closes the owned SDK
  client independently of local model release; cleanup failure is a sanitized
  warning that never replaces a valid result.
- **Observability.** Routes, auth-vs-rate-limit counts, final-provider (the
  accepted text source, not a failed attempted provider), and usage metadata are
  now accurate.

Live verification: one tiny `gemini-3.8-flash` structured-output smoke at
`thinking_level=low` succeeded (`دي موقراطية` → `ديمقراطية`; usage
`270` prompt / `81` candidate / `75` thoughts), and a real difficult known
Stage 2.7 phrase (`فيور 25 نوفمبر`, which `qwen3.5:4b` could not repair) routed
`GEMINI_DIRECT` and Gemini repaired it to `في يوم 25 نوفمبر`, which passes the
shared validation at HIGH (`phonetic 0.888`, score `0.933`). The key was never
printed or logged. The default mode remains `ADAPTIVE`; `local_only` is the
fully-local override.

## Performance and correctness corrective pass (2026-09-09)

A real three-minute source exposed a release-blocking failure: Stage 2.5 left
all 79 segments `unchanged` (confidence `0.0`), the adaptive router treated
every one as Qwen-eligible, orchestration made one sequential Qwen request per
segment, and after 36–40 minutes reconstruction had not completed while Ollama
used ~1148% CPU and the machine hit 100% CPU/RAM/disk. The corrective pass fixes
routing, batching, ceilings, cancellation, caching, and hardware controls:

- **Restored `NO_LLM` routing.** Clean, well-covered unchanged Stage 2.5
  results now route to `NO_LLM` (no provider calls); high Whisper probabilities
  plus adequate coverage plus zero low spans are affirmative trust. Trusted
  Stage 2.5 repairs resolve the raw-ASR words their `changes` cover, and a clean
  remainder is not reprocessed; independent unresolved spans still route to an
  LLM. Missing evidence remains a conservative local check, never Gemini-direct.
  A deterministic 79-segment clean fixture makes zero Qwen and zero Gemini calls.
- **Bounded local micro-batching.** `reconstruction_provider_batch_windows`
  (8) and `reconstruction_provider_batch_characters` (24 000) are now
  functional: local targets are processed in deterministic micro-batches with
  exact segment-ID mapping and per-candidate validation isolation.
- **Hard local work ceilings.** `CLIPFACTORY_LOCAL_RECONSTRUCTION_MAX_TARGETS_PER_JOB`
  (default 64, spent strongest-first) and
  `CLIPFACTORY_LOCAL_RECONSTRUCTION_MAX_WALL_SECONDS` (default 1200) guarantee a
  four-hour source cannot create unbounded local inference. Skipped targets are
  marked unresolved/manual review (`local_target_budget_exhausted` /
  `local_time_budget_exhausted`) and never auto-escalate to Gemini.
- **Cooperative cancellation.** Cancellation is checked before and after every
  provider batch. A cancelled job stays `CANCELLED`, is never overwritten as
  successful, and never schedules the next stage; the current bounded request
  is allowed to finish.
- **Per-target reuse and checkpoints.** Each completed target persists its
  stable dependency fingerprint and cache eligibility. A restart after
  cancellation or a late provider failure reuses accepted targets without
  re-calling a provider; only failed or eligible-unresolved work is retried.
- **Corrected dependency fingerprints.** Output fingerprints now include every
  route-relevant input (Stage 2.5 method/confidence/applied state and change
  digest, word probabilities, acoustic evidence, language, dialect carrier,
  bounded context) plus routing version/thresholds, local batch and ceiling
  settings, Gemini API version and temperature. Fingerprint version is `4`.
- **No Gemini metadata probes.** Gemini availability is configuration-level
  with a lazily constructed SDK client; the bounded generation request is the
  availability check. Cache hits, all-`NO_LLM`, and `LOCAL_ONLY` jobs make zero
  Gemini network calls and ideally construct no client. Generation is
  deterministic (`temperature=0`, explicit API version `v1`); `gemini-3.8-flash`
  and `thinking_level=low` are unchanged.
- **Secret and exception hygiene.** Gemini error chains are sanitized with
  `from None`, client-close errors too; a fake-secret traceback test proves the
  sentinel appears in neither the exception nor its formatted traceback.
- **Provider cleanup.** Owned provider resources close exactly once on every
  exit path (cache hit, success, cancellation, timeout, provider failure, and
  orchestration failure); the lazy lease is held only for real local inference.
- **Ollama hardware safeguards.** Compose pins `OLLAMA_NUM_PARALLEL=1`,
  `OLLAMA_MAX_LOADED_MODELS=1`, a bounded queue, `OLLAMA_CONTEXT_LENGTH=4096`,
  `OLLAMA_CPUS=6`, `OLLAMA_MEM_LIMIT=6g`, and `OLLAMA_MEMSWAP_LIMIT=8g`,
  operator-tunable via the environment.
- **Progress and observability.** Lightweight progress (totals, local eligible/
  completed, Gemini eligible/completed, unresolved, phase, budget remaining,
  cancellation requested) is persisted in `reconstruction_metadata`.

Deterministic verification uses fake providers and stored transcript structures
only. A clean 79-segment fixture makes zero provider calls; a ten-target local
fixture with a four-target batch limit makes exactly three local calls; a
synthetic ~6 000-segment transcript completes with at most 64 local targets (8
batches) and zero Gemini calls; a partial-Gemini-failure restart reuses the four
accepted targets and issues exactly one new Gemini request; cancellation between
batches prevents later provider calls and keeps the job `CANCELLED`. Real Qwen,
Whisper, video replay, and repeated live Gemini runs are prohibited in this
corrective pass, so no real wall-time claim is made; the deterministic call-count
improvements and enforced upper bounds above are the acceptance evidence.

## Sol-review blocker fixes (2026-09-09)

A second corrective pass fixed the four validated release blockers plus the
cache-hit lifecycle defect without redesigning Stage 2.7:

- **Aggregate local batches are context-safe.** The local provider now evaluates
  the exact combined chat envelope that will be sent (system instruction, full
  `{"targets": [...]}` payload, chat-framing and safety reserves, scaled output
  budget) and greedily splits groups so no sent request ever exceeds
  `max_context_tokens`; window/character ceilings remain secondary bounds. A
  regression proves requests that each fit alone are split into two calls and
  every sent envelope fits.
- **Degraded reconstruction retries normally.** `PipelineRunner` consults an
  optional executor `skip_is_allowed`; the reconstruction executor allows a skip
  only when the stored run is fully cache-eligible. A degraded run (unresolved/
  provider-failed/rate-limited/local-ceiling) re-enters the executor on a later
  non-force request, reuses accepted per-target work with zero provider calls,
  and retries only eligible unfinished targets.
- **Cancellation works on every route.** One centralized poll runs before and
  after each local batch, Gemini-only attempt, direct-Gemini attempt, and
  escalation, plus once immediately before a successful return. Cancellation
  landing after the final local batch is never missed, and a cancelled Gemini
  job can no longer finish successfully or schedule the next stage.
- **Ollama CPU default is six.** `OLLAMA_CPUS` defaults to `6` (operator
  overridable) in Compose, `.env.example`, and documentation; memory, swap,
  queue, loaded-model, and parallelism safeguards are unchanged.
- **Fresh cache-hit cleanup.** The executor now releases owned provider
  resources on every exit path, so a fresh cache-hit worker scrubs the Gemini
  key and closes owned SDK clients without building a client or making a network
  call; a regression proves the key is scrubbed with zero generation.

Bounded verification (fake providers, stored transcripts) passes: 411 backend
tests plus the 21 immutable-ASR capture tests; Ruff and formatting clean; scoped
mypy shows no new errors. One tiny live Gemini structured-output smoke was
attempted through the production settings path and failed with a sanitized
`PROVIDER_ERROR` category (external/API-side); it was not retried and no key was
printed, logged, or committed.

## Local-batch boundary and cross-session cancellation fixes (2026-09-09)

A third corrective pass fixed the two remaining P1 blockers without redesigning
Stage 2.7:

- **Every actual local request is a visible orchestration unit.** Aggregate
  context-safe planning moved from a hidden provider loop into orchestration.
  The provider exposes a pure `plan_aggregate_batches` helper (no HTTP) that
  splits each window/character micro-batch into actual request units whose exact
  combined chat envelope fits `max_context_tokens`; `reconstruct_segments`
  executes exactly one HTTP call per unit. Orchestration polls cooperative
  cancellation and the local wall-time budget before every actual request and
  checkpoints after every actual request. If one actual request fails after
  earlier ones succeeded, only its targets fail/escalate/unresolve; the earlier
  targets' accepted candidates stay checkpointed and reusable.
- **Cancellation status is read fresh across sessions.** `PipelineRunner` binds
  the exact executing reconstruction job id to the executor, and
  `_job_cancelled` reads that job's status with a fresh scalar column query
  (bypassing the worker session's ORM identity map), so a `CANCELLED` commit
  made by the API in a separate session is observed by the worker on its next
  poll without a commit or refresh. Queued and active cancellation, `CANCELLED`
  job/run persistence, and no-next-stage scheduling are preserved.

Deterministic verification (fake transports, fake clocks, SQLite sessions) now
includes: a service-level eight-target micro-batch context-split into multiple
actual requests with every sent envelope fitting; cancellation after the first
actual request preventing the second; the fake clock hitting the wall-time
ceiling after the first actual request; a second actual-request failure
preserving the first request's accepted checkpointed candidates; the provider
refusing to hide an over-budget combined request behind a loop; and a two-session
test where an API-session `CANCELLED` commit stops the worker before any further
provider call with the run and job left `CANCELLED`.

STAGE 2.7 MUST CONTINUE

## Planning-isolation fix (2026-09-09)

A final corrective pass closed the last remaining P1 blocker: a later oversized
target (one that cannot fit `max_context_tokens` even after bounded shrinking)
could previously abort the whole local phase during **eager** planning of all
future micro-batches, before any valid unit ran or checkpointed. Planning is now
per-micro-batch and lazy: each window/character micro-batch is planned
immediately before it executes. A planner rejection is caught at the micro-batch
boundary, only the irreducible target is isolated on its own
(fallback/unresolved, or Gemini escalation only when policy and budget permit),
and earlier/later valid units still run and checkpoint. Deterministic tests
prove: a later irreducible target does not abort the earlier valid unit (its
call happens, its result is checkpointed, only the oversized target is affected);
an irreducible target between two valid units leaves both sides running; and in
adaptive mode the irreducible target escalates to Gemini exactly once while the
valid unit runs locally. Cancellation, wall-time, aggregate-envelope, per-target
reuse, and cross-session cancellation guarantees are unchanged and green.

## Wall-ceiling and Gemini 503 verification (2026-09-09)

A narrow corrective pass closed a hard-ceiling bypass and sharpened Gemini
503 observability. The local wall-time ceiling is now checked **before** each
micro-batch is planned, not only before each actual request: once it expires,
the batch is never planned, no target is classified unfit, no local
attempt/failure is recorded for it, and no `local_context_unfit` Gemini
escalation is enqueued — the existing tail handling marks remaining targets
`local_time_budget_exhausted`. Deterministic fake-clock tests prove: a later
irreducible target after the ceiling yields zero local failures/attempts and
zero Gemini calls (local-only and adaptive); isolation of a middle irreducible
target still runs both valid sides when budget is available. Gemini HTTP 503 is
classified as the precise sanitized `SERVICE_UNAVAILABLE` category, is retried
exactly once (initial + one bounded retry, two attempts maximum), and after
exhaustion produces a safe fallback with `failure:SERVICE_UNAVAILABLE`
evidence; 429/401/403/malformed-output remain single-attempt and non-retryable.
Fake-SDK tests also prove a sentinel fake key never appears in exceptions,
tracebacks, or metadata.

STAGE 2.7 MUST CONTINUE
