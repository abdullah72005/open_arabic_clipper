# Stage 4.2 operations — retention, originality, and platform-risk governor

Stage 4.2 is an explicit, candidate-scoped **critic/governor**. It evaluates
every current Stage 4.1 transformation plan independently and answers one
question:

> Is this concrete plan good and safe enough to be considered by Stage 4.3?

The governing principle is: **preserve the source moment, add genuine authorial
value, maintain semantic fidelity, and use the least intrusive sufficient
transformation.**

Stage 4.2 never generates, mutates, repairs, or selects a plan. It adds no
`PipelineStage`, no `PipelineRun`, no `_NEXT_STAGE` entry, and never advances the
source lifecycle. It is never automatic: it runs only when explicitly queued for
one candidate through the existing Celery/`ProcessingJob` platform with a
`TRANSFORMATION_GOVERNANCE` job kind.

## Funnel boundary

```
Stage 4.0  transformation eligibility and strategy discovery
    ↓
Stage 4.1  0–3 concrete structured transformation plans
    ↓
Stage 4.2  independent governance of every current plan   ← this stage
    ↓
Stage 4.3  future final plan selection or candidate rejection (NOT implemented)
```

## Immutable Stage 4.1 input

Stage 4.2 consumes the read-only Stage 4.1 handoff and reloads identities from
persistence. It never accepts a client-supplied plan body. Queueing requires:

- a current retained candidate;
- a usable `CANDIDATE` or already-usable `FINAL_CLIP` Stage 3.5 refinement
  (a `FINAL_CLIP` plan is never required);
- a current, non-stale, complete/degraded Stage 4.1 plan set;
- at least one current Stage 4.1 plan;
- exact matching plan/refinement/analysis fingerprints.

Stale input is refused **before any provider work**. Critical plan integrity is
revalidated deterministically (identity, current flags, closed block types and
indexes, hero/source-span integrity, narration abstraction, verification
linkage, absence of TTS/render/evasion text, plan output fingerprint). Malformed
persisted content is never reinterpreted or repaired.

## Execution lifecycle vs semantic outcomes

Execution lifecycle (`transformation_governance_sets.execution_status`), separate
from the semantic outcome:

| Status | Meaning |
| --- | --- |
| `QUEUED` | queued |
| `GOVERNING` | claimed and running |
| `COMPLETE` | finished, all semantic work done |
| `PROVIDER_DEGRADED` | finished but optional provider work is unfinished |
| `FAILED` | server/pipeline failure only |
| `CANCELLED` | cooperatively cancelled |

A semantic rejection, revision, verification block, all-plans-ineligible result,
or provider deferral is **not** a pipeline/server failure.

Per-plan statuses (`transformation_governance_results.status`):

| Status | Meaning | `eligible_for_stage4_3` |
| --- | --- | --- |
| `APPROVED_FOR_SELECTION` | sound, eligible | true |
| `APPROVED_WITH_CAUTION` | eligible with explicit warnings/platform-risk evidence | true |
| `BLOCKED_PENDING_VERIFICATION` | sound but ineligible until essential external verification is resolved | false |
| `REVISION_REQUIRED` | potentially salvageable, current immutable plan ineligible | false |
| `REJECTED_BY_GOVERNOR` | must not proceed | false |
| `GOVERNANCE_DEFERRED` | required semantic evidence unavailable/invalid; no false approve or reject | false |

`eligible_for_stage4_3` is a **filter, not a ranking**. A database check
constraint enforces the exact status/eligibility relationship.

Candidate-level outcomes (`transformation_governance_sets.governance_outcome`),
deterministic precedence:

1. At least one approved/approved-with-caution → `PLANS_ELIGIBLE_FOR_SELECTION`.
2. Otherwise any current plan deferred for unfinished evidence →
   `GOVERNANCE_DEFERRED`.
3. Otherwise → `NO_GOVERNOR_APPROVED_PLAN` (a normal successful semantic result;
   its summary keeps separate verification-blocked, revision-required and
   rejected counts).

A set may hold an eligible plan and a deferred sibling: the candidate outcome
stays `PLANS_ELIGIBLE_FOR_SELECTION`, the deferred plan remains ineligible, the
execution/provider state is degraded, and the set stays non-cache-eligible until
the unfinished work resolves. A survivor is never forced.

## Deterministic decision precedence

Explicit severity classes: `HARD_FAIL`, `BLOCKING_CONDITION`, `REVISION`,
`WARNING`, `ADVISORY`. Final status is assigned only by deterministic code;
Gemini can never assign it. Approximate precedence:

1. integrity, prohibited/evasion, semantic-fidelity hard failures;
2. no substantive value, presentation-only transformation, fabricated critical
   claim, intrinsically misleading hook;
3. unresolved essential verification;
4. repairable retention/narration/coherence/filler/redundancy/proportionality;
5. semantic-evidence deferral;
6. caution;
7. approval.

Examples:

- semantic distortion → reject;
- presentation-only/no substantive contribution → reject;
- fabricated numeric claim → reject;
- real but unresolved external dependency → block;
- removable generic intro or pacing-damaging narration → revision;
- moderate source dominance/template/platform risk without a shared hard failure
  → caution;
- strong and complete plan → approve.

Platform-specific risk alone never hard-blocks content globally. Shared
underlying failures (no added value, deceptive framing, fabricated claims) reject
the plan independently.

## Independent dimensions (no overall score)

No overall score, viral score, monetization score, or weighted aggregate exists.
Independent categorical assessments with bounded evidence/reason codes are
persisted:

1. retention preservation; 2. source-moment damage; 3. substantive
originality/added value; 4. source dominance; 5. semantic fidelity; 6. generic
filler risk; 7. redundant commentary risk; 8. narration burden; 9. verification
completeness; 10. plan-level template/mass-produced feel; 11. YouTube
reused-content risk; 12. YouTube inauthentic/mass-produced risk; 13. YouTube
spam/deceptive-practices risk (where observable); 14. Facebook unoriginal-content
risk; 15. Facebook spam/repetitive risk (where observable); 16. plan
coherence/watchability; 17. transformation proportionality.

Stage 3 engagement evidence is carried only as context. Retention percentages,
views, CTR, RPM, virality, and platform enforcement are never predicted.

### Retention preservation

Hero timing (true elapsed and authored duration before the hero), interruption
before/during payoff, split of question/answer, joke/punchline, argument or
emotional momentum, commentary that reveals the payoff early, source
fragmentation, switching burden, narration placement, and dragging a concise
moment are all recomputed from the immutable plan. A 12-second generic preamble
before the hero is a deterministic failure. Severe but structurally repairable
pacing damage is `REVISION_REQUIRED`; high originality never offsets destroying
the source moment.

### Substantive originality

Credit is given only for genuine context, explanation, inference, comparison,
counterpoint, synthesis, verification/correction, authored thesis, useful
takeaway, or source-as-evidence framing. Captions, crop/reframe, zoom, borders,
emojis, gameplay, generic B-roll, loops, music, speed changes, filters, cuts,
transitions, watermarking, and pitch changes receive **zero** credit. Word count,
narration duration, and edit count are never originality.

### Source dominance

There is no universal source-ratio gate. Ratios are evidence interpreted with
strategy, content type, source-moment structure, actual semantic contribution,
transformation intensity, and provenance/originality risk. A high-source-ratio
`SOURCE_AS_EVIDENCE` or preservation-first joke plan can pass when its concise
authored contribution is genuinely distinct; a low-source-ratio plan can still
fail as filler. Third-party/unknown provenance raises the required substantive
transformation standard but never automatically blocks governance.

### Semantic fidelity hard gate

Rejects material distortion, reversed/removed context, exaggerated certainty,
speculation-as-fact, literalized sarcasm/jokes, manufactured controversy,
implied answers to other questions, generated-material attribution to the
speaker, unrelated source evidence, and fake/misleading hooks. Fidelity failure
cannot be offset by retention or originality.

### Filler, redundancy, template feel

Deterministically detects paraphrase-only commentary, "he is saying…"
restatements, generic "this changes everything" framing, empty "why this
matters" scaffolding, forced moral lessons, vague takeaways, repetitive
transitions, mechanically imposed hook/source/takeaway templates, and
presentation-only wrapping. `REJECTED_BY_GOVERNOR` for no real contribution or
presentation-only repackaging; `REVISION_REQUIRED` when a genuinely promising
strategy is currently implemented as removable filler, paraphrase, or template
scaffolding. Plan intent is judged, not final script naturalness.

### Narration burden

Narration `NONE` is valid and never penalized. Assessed: necessity, semantic
contribution, redundancy, duration, placement, interruption, whether a shorter
textual annotation would suffice, and whether the source payoff should simply
end the clip. Closed findings: appropriate, excessive, redundant,
position-damaging, not needed, unknown. No TTS provider, model, voice ID, speaker
identity, or channel voice is ever selected or mentioned.

## Verification

Claim state is explicit:

- `GROUNDED_IN_SOURCE` — every substantive authored block declares at least one
  grounding reference that deterministically resolves to real plan source
  evidence **and** the cited source wording supplies conservative lexical support
  for the claim (shared claim-specific content tokens, a contiguous shared phrase,
  or direct-quote/reference framing); may proceed;
- `EXTERNAL_REQUIRED_UNRESOLVED` — an essential correctly linked external
  dependency, an unresolvable grounding reference (including provider-declared
  labels such as `strategy` or `source_excerpt`), or an unsupported factual claim
  whose citation is real but unrelated (e.g. "The merger closes next Monday"
  citing a source that contains no merger information); →
  `BLOCKED_PENDING_VERIFICATION` when the rest is sound;
- `SUPPORT_UNVERIFIED` — the citation resolves but the deterministic lexical
  check cannot confirm claim support; requires semantic review (defer or block,
  never a silent approval);
- `UNSUPPORTED_OR_FABRICATED` → `REJECTED_BY_GOVERNOR`;
- `NOT_APPLICABLE`.

Provider-declared grounding labels are never proof, and a structural citation
alone is never proof. Absence of a verification placeholder is never proof of
grounding. Because no deterministic system can prove semantic entailment, the
support check is deliberately conservative and three-way:

- **supported** → `GROUNDED_IN_SOURCE`;
- **ambiguous** (weak/partial lexical overlap, or no factual signal) → semantic
  judgment: the selective provider may assess observable claim-to-source
  support/fidelity when available; when unavailable the plan stays truthful
  (`GOVERNANCE_DEFERRED`, or `BLOCKED_PENDING_VERIFICATION` when external fact
  verification is essential) and is never approved;
- **unsupported** (no meaningful claim-to-evidence overlap or a factual signal
  with an unrelated citation) → `EXTERNAL_REQUIRED_UNRESOLVED` /
  `BLOCKED_PENDING_VERIFICATION`.

Deterministic code owns the final status and hard gates; Gemini is
non-authoritative, may only report bounded observable support/fidelity findings,
and is never a fact-checking or web-lookup system. Clearly-supported claims are
resolved deterministically without any hosted call. Legitimate source-grounded
explanation, inference, quotation, and source-as-evidence plans that reference
real supporting evidence are unaffected.

Stage 4.2 never creates missing placeholders, never claims verification occurred,
never browses at runtime, and adds no research/fact-checking infrastructure.
Gemini is skipped for an obviously verification-blocked plan when critique would
not change its truthful state.

## Platform-policy profile

Immutable, code-defined profile (never a dynamic engine, never runtime scraping):

- governor policy: `stage4.2-v2`
- schema: `stage4.2-schema-v1`
- validation: `stage4.2-validation-v2`
- platform profile: `stage4.2-platform-policy-2026-09-17-v1`
- checked date: `2026-09-17`

Official sources (re-confirmed 2026-09-17):

- YouTube channel monetization policies — <https://support.google.com/youtube/answer/1311392>
- YouTube spam policy — <https://support.google.com/youtube/answer/2801973>
- Meta: Rewarding Original Creators on Facebook — <https://about.fb.com/news/2021/07/rewarding-original-creators-on-facebook/>
- Meta: Cracking Down on Spammy Content on Facebook — <https://about.fb.com/news/2023/08/cracking-down-on-spammy-content-on-facebook/>
- Facebook original-content business guidance — <https://www.facebook.com/business/help/1136636083752902>

Durable concepts encoded:

**YouTube**

- borrowed material requires significant original contribution or meaningful
  difference;
- critical review, explanation, commentary and substantive editing can add value;
- minimal changes remain reused-content risk even with permission;
- reused-content review is separate from copyright;
- generic, repetitive, template-like or mass-produced material carries
  inauthentic-content risk;
- automated high-volume minimal variation, scraped reposting, deceptive
  presentation and technical detection evasion are spam/integrity risks;
- some channel-wide review factors cannot be evaluated per plan and are deferred.

**Facebook**

- creator-produced material is original;
- third-party material may qualify when it presents genuinely new information,
  analysis or substantial storyline improvement;
- facial reaction, stitching or narrating what is already visible without
  meaningful addition remains unoriginal;
- borders, captions and speed changes are minor edits and receive no originality
  credit;
- spam-network behavior, unrelated metadata, coordinated engagement and
  account-level flooding are outside Stage 4.2.

Persisted platform output is shared core evidence plus platform-specific
deterministic interpretation:

```json
{
  "policy_profile_version": "stage4.2-platform-policy-2026-09-17-v1",
  "policy_checked_at": "2026-09-17",
  "youtube": {
    "reused_content": {"level": "LOW|MODERATE|HIGH|UNDETERMINED", "reason_codes": [], "evidence": []},
    "inauthentic_mass_produced": {},
    "spam_deceptive_practices": {}
  },
  "facebook": {"unoriginal_content": {}, "spam_repetitive": {}},
  "generic": {"source_dominance": {}, "substantive_transformation": {}, "template_mass_produced_feel": {}},
  "account_level_repetition": "DEFERRED_TO_STAGE_7",
  "limitations": ["Decision support only; not a legal, copyright, monetization, recommendation, or enforcement guarantee."]
}
```

The profile deliberately contains no phrase equivalent to "safe for YouTube",
"safe for Facebook", "guaranteed monetizable", "algorithm safe", "will not be
flagged", or "bypasses detection". There is no evasion logic, monetization
guarantee, or algorithm-safety claim. Copyright/rights/licence risk, platform
originality/reuse risk, and spam/repetitive-content risk stay separate in
persistence and handoff.

## Gemini, Qwen, and provider boundary

Stage 4.2 has its own semantic mode (`stage4.2`, default `adaptive`) and its own
`CLIPFACTORY_TRANSFORMATION_GOVERNANCE_*` settings, so Stage 4.2 changes never
invalidate frozen Stage 4.0/4.1 caches.

Deterministic gates run first. Gemini is **not** called for: cache hits, stale or
malformed inputs, already hard-rejected plans, trivial deterministic failures,
clearly verification-blocked plans where critique adds nothing, or plans whose
deterministic evidence is sufficient for a truthful result. For remaining
ambiguous/high-value plans:

1. one bounded plan-set request containing all pending current plans (max three);
2. independent per-plan identity/output preserved;
3. strong tier for the bounded set when any pending plan deterministically
   requires strong semantic reasoning, otherwise routine;
4. at most one second strong call for specific plans whose first valid critique
   remains genuinely `UNKNOWN`/materially ambiguous;
5. hard raw hosted-call ceiling of two per governance run, no per-call retry.

As of 2026-09-17 the configured stable models are `gemini-3.5-flash-lite`
(routine) and `gemini-3.8-flash` (low thinking, strong). Every raw hosted call
acquires shared Gemini admission at `HIGH` (Redis-backed, fail-closed) and honors
cooldown/rate-limit recording. Missing key, admission denial, 429, timeout,
outage, safety refusal, or malformed output never fails the candidate/source: the
plan defers, the set degrades to `PROVIDER_DEGRADED`, and a later request retries
unfinished work.

Gemini is asked only for observable semantic characteristics (fidelity, context
distortion, source/commentary relationship, value distinctness,
paraphrase/redundancy, unsupported-claim signals, source-moment interruption,
coherence, narration necessity/placement, generic/template-shaped risk). It is
never asked whether a platform will flag, monetize, or recommend a plan, and it
can never output a final governor status or platform-risk classification.

Provider output is parsed with a tolerant top-level shape and strict independent
per-item validation: unknown plan IDs, duplicate items, mismatched fingerprints,
invalid enum values, invented spans, unbounded strings/lists, fabricated
verification completion, final statuses, platform-safety guarantees, TTS/voice
identity, rendering instructions, and evasion tactics are discarded. One
malformed plan item never invalidates valid siblings.

Qwen is unchanged: `CLIPFACTORY_LOCAL_QWEN_ENABLED=false` by default, no implicit
model download or load, used only in explicit `local_only`, never an adaptive
Gemini-to-Qwen fallback, no Gemini calls in `local_only`, and shared
heavy-model lease with guaranteed provider release.

## Checkpoints, cache, fingerprints

Each accepted semantic critique is persisted under its own provider-input
fingerprint on `transformation_governance_sets.plan_attempts` and reused on exact
dependency match, including forced reruns. Valid siblings from a batched response
are checkpointed; only invalid siblings defer. After a provider call, accepted
critiques are checkpointed promptly under the job claim-version fence.

The governance input fingerprint includes candidate/source identity, plan-set
identity and input/output fingerprints, each current plan's full
governance-relevant representation, hero/hook/payoff evidence, narration
semantics, verification dependencies, strategy type/intensity and Stage 4.0
strategy/risk evidence, refinement identity and bounded evidence, rights,
provenance and originality evidence, semantic target-market/language/register
context, Stage 4.2 policy/config/schema/validation identity, platform-policy
profile/version/date, and provider mode/model/API/prompt-schema hash/temperature/
thinking/budgets.

It explicitly excludes: the Gemini key value; transient admission counters/
cooldown/outage state; TTS provider/model/voice ID/fallback voice; caption font;
render resolution/settings; publishing schedule/settings; metadata title/
thumbnail; and unrelated frontend settings. **A TTS-only, render-settings-only,
or publishing-settings-only change does not invalidate governance.**

Cache behavior: exact complete inputs reuse the set/results without another
provider call; blocked/revision/rejection outcomes are cacheable when all
evidence is complete; any unfinished/deferred semantic work makes the set
non-cache-eligible and a later ordinary request retries it while reusing accepted
per-plan checkpoints; force recomputes deterministic logic but never repeats an
accepted matching provider critique. Output fingerprints cover the complete
ordered governance results and candidate summary, excluding transient timing,
metrics, and availability.

## Concurrency and cancellation

Reusing the Stage 4.1 architecture unchanged:

- one active governance job per governance set;
- atomic active-job claim (`active_job_id IS NULL` compare-and-swap);
- durable `ProcessingJob.claim_version` advanced on every claim;
- `QUEUED`/`FAILED` → `RUNNING` claim; duplicate/redelivered invocation performs
  no provider work;
- renewable heartbeat while blocked in provider calls;
- stale reclaim based on abandoned heartbeat, not merely old `started_at`;
- fresh scalar cancellation reads against the exact executing job, checked
  before and after each provider call and before final persistence;
- final persistence fenced on the current claim and still-`RUNNING` status;
- stale workers cannot persist, cancel, fail, or finalize a newer claim;
- cancellation keeps `CANCELLED`, clears ownership, schedules nothing, selects
  nothing, and preserves checkpointed accepted work;
- provider resources and the Gemini key are released/scrubbed on every exit path,
  including cache hits and cancellation.

## API, CLI, and jobs

- `POST /api/candidates/{candidate_id}/transformation-governance` — explicit
  queue, optional `force`, `202`, cache/active semantics.
- `GET /api/candidates/{candidate_id}/transformation-governance` — current set
  and per-plan results.
- `GET /api/transformation-governance-sets/{governance_set_id}` — direct read.
- `GET /api/candidates/{candidate_id}/stage4-3-handoff` — read-only future Stage
  4.3 handoff.
- CLI: `transformation-govern`, `transformation-governance`,
  `transformation-governance-handoff`.
- Celery: `clipfactory.run_transformation_governance`. It creates no
  `PipelineRun` and schedules no next stage.

## Stage 4.3 handoff

`build_stage4_3_handoff` returns a typed read-only structure with candidate,
rights/provenance, selected refinement, Stage 4.0/4.1 identities, the governance
set summary, and every current plan with its independent governance result.
`stage4_3_implemented` is `false`. There is **no winner, no `selected_plan_id`,
no render-ready state, no publication approval, and no automatic queue action**.
Stage 4.1 generation order is preserved; no governor preference rank is added.

Freshness is fail-closed and truthful. `governance_set.freshness` is exactly one
of:

- `VERIFIED_CURRENT` — the stored input fingerprint recomputes to the same value
  from current inputs; per-plan `eligible_for_stage4_3` is preserved as persisted;
- `STALE` — the recomputed fingerprint differs from the stored one;
- `NOT_CURRENT` — the governance set is not in a completed/degraded state or has
  no semantic outcome;
- `UNVERIFIABLE` — the stored input fingerprint is missing, inputs cannot
  resolve, or fingerprint recomputation raises.

For `STALE`, `NOT_CURRENT`, or `UNVERIFIABLE`, every handoff plan reports
`eligible_for_stage4_3=false`; only an explicitly verified current result may
retain eligibility. `governance_set.current`/`stale` and per-plan
`governance.freshness`/`stale` mirror this state.

## Reason codes and remediation

Closed reason codes include `PLAN_INTEGRITY_INVALID`, `NO_SUBSTANTIVE_VALUE`,
`PRESENTATION_ONLY_TRANSFORMATION`, `REDUNDANT_PARAPHRASE_ONLY`,
`GENERIC_FILLER_ONLY`, `SEMANTIC_DISTORTION`, `CONTEXT_REVERSAL`,
`FALSE_ATTRIBUTION`, `SARCASM_LITERALIZED`, `SPECULATION_PRESENTED_AS_FACT`,
`UNRELATED_SOURCE_EVIDENCE`, `FAKE_OR_MISLEADING_HOOK`,
`UNSUPPORTED_CRITICAL_CLAIM`, `EXTERNAL_VERIFICATION_REQUIRED`,
`SOURCE_MOMENT_SEVERELY_DAMAGED`, `PAYOFF_INTERRUPTED`, `EXCESSIVE_PREAMBLE`,
`OVER_FRAGMENTED`, `NARRATION_EXCESSIVE`, `NARRATION_REDUNDANT`,
`NARRATION_POSITION_DAMAGING`, `SOURCE_DOMINANCE_CONCERN`,
`TEMPLATE_MASS_PRODUCED_FEEL`, `PLATFORM_REUSE_RISK`,
`PROVIDER_SEMANTIC_REVIEW_UNAVAILABLE`, `PROVIDER_OUTPUT_INVALID`,
`PLATFORM_EVASION_TACTIC`, `TTS_IDENTITY_FORBIDDEN`.

Revision-required plans persist structured remediation (action code, target
block indexes, priority, bounded repository-controlled note). Actions include
removing a redundant intro, shortening a preamble, moving explanation after the
hero, removing paraphrase, reducing narration, preserving payoff, resolving
verification, and replacing generic takeaway intent with genuine analysis. No
blocks are rewritten, no replacement text is created, and Stage 4.1 is never
enqueued.

## Observability

Bounded metrics: governance sets completed, plans evaluated, deterministic hard
rejects, verification-blocked plans, plans semantically reviewed, approvals,
cautions, revisions, rejections, deferred plans, no-survivor candidates,
platform-risk distribution, narration-excessive findings, semantic-fidelity
failures, provider/raw hosted calls, plans per provider call, checkpoint/set
cache hits, provider failures, and wall time. No analytics dashboards or Stage 11
infrastructure.

## Boundaries (not implemented here)

Stage 4.3 selection; winner selection; `selected_plan_id`; automatic plan
rewriting; an automatic Stage 4.1↔4.2 repair loop; plan mutation; narration
script generation; TTS synthesis/provider/model/voice selection; channel voice
identity; rendering/FFmpeg; captions/face tracking/B-roll/gameplay retrieval;
external research/web fact-checking; search grounding; publishing/scheduling/
metadata; account/channel history queries; Stage 7 diversity logic; platform
classifier simulation; detection evasion; watermark/mirroring/pitch/speed tricks;
ASR/ingestion changes; Gemini benchmarking; a single opaque overall score.

Account/channel-level repetition is explicitly
`DEFERRED_TO_STAGE_7`.

## Statuses and handoff summary

Execution: `QUEUED`, `GOVERNING`, `COMPLETE`, `PROVIDER_DEGRADED`, `FAILED`,
`CANCELLED`.
Per-plan: `APPROVED_FOR_SELECTION`, `APPROVED_WITH_CAUTION`,
`BLOCKED_PENDING_VERIFICATION`, `REVISION_REQUIRED`, `REJECTED_BY_GOVERNOR`,
`GOVERNANCE_DEFERRED`.
Candidate: `PLANS_ELIGIBLE_FOR_SELECTION`, `NO_GOVERNOR_APPROVED_PLAN`,
`GOVERNANCE_DEFERRED`.

Counts on the governance set keep verification-blocked, revision-required and
rejected plans separate so `NO_GOVERNOR_APPROVED_PLAN` never implies they all
failed for the same reason.

## Validation

```bash
# focused Stage 4.2
docker run --rm -e PYTHONPATH=/app -v "$PWD/backend:/app" -w /app oac-backend-test \
  python -m pytest tests/ -k stage42 -q

# full backend suite (mount the repository compose/.env files for path-based tests)
docker run --rm -e PYTHONPATH=/app -v "$PWD/backend:/app" \
  -v "$PWD/compose.yaml:/compose.yaml:ro" -v "$PWD/.env.example:/.env.example:ro" \
  -w /app oac-backend-test python -m pytest tests/ -q

# formatting and lint
ruff format app tests alembic && ruff check app tests alembic
```

All Stage 4.2 tests are deterministic and hermetic: providers are mocked and no
test makes a live Gemini, Qwen, web, TTS, or rendering call even when a key is
present. Known limitation: the repository's strict `mypy` configuration already
reports `no-any-return`/untyped-decorator findings in the frozen Stage 4.0/4.1
modules; Stage 4.2 matches that convention rather than introducing a new lint
regime.

## Corrective patch (2026-09-17)

1. **Verification grounding.** See the Verification section: an ungrounded
   authored claim (numeric or not) is an unresolved verification dependency and
   cannot silently reach `APPROVED_FOR_SELECTION`.
2. **Stale `RUNNING` recovery.** When queueing and the only active job is a
   `RUNNING` job whose heartbeat is older than the shared stale window
   (`JOB_CLAIM_STALE_SECONDS`), the queue re-dispatches the same job id so the
   executor's atomic `QUEUED`/`FAILED`/stale-`RUNNING` → `RUNNING` claim performs
   the claim-version-bumping reclaim. A live/heartbeating job is never reclaimed;
   the superseded worker's `claim_version` fence prevents it from persisting,
   cancelling, failing, or finalizing the newer claim; and redelivered
   invocations of an already-terminal job are fenced out with no duplicate
   provider call. The queue outcome carries
   `skipped_reason="RECOVERED_STALE_RUNNING"`.
3. **Handoff freshness.** `build_stage4_3_handoff` invokes the governance
   staleness check and exposes `governance_set.current` and
   `governance_set.stale`, plus a per-plan `governance.stale`. When stale, every
   plan's `eligible_for_stage4_3` is reported `false`; a stale governance result
   is never represented as currently eligible. `stage4_3_implemented` stays
   `false` and no winner is chosen.
4. **Second-call checkpointing.** A bounded strong second critique that resolves
   an initially `UNKNOWN` plan is applied and persisted as that plan's checkpoint
   under its provider-input fingerprint, so a forced rerun with unchanged inputs
   reuses it and makes zero additional hosted calls. If the second call fails, the
   first accepted critique is preserved and the plan stays truthfully
   `GOVERNANCE_DEFERRED`.
5. **Strict hero thresholds.** `Stage42Config.short_moment_seconds` and
   `Stage42Config.high_moment_density_floor` are the only thresholds used by
   `is_strict_hero_window`; both appear in `governance_config_payload` and thus
   invalidate the input fingerprint when changed.
6. **Provider boundary.** Only closed `ACCEPTED_PROVIDER_FINDING_CODES` are
   persisted (arbitrary text is discarded; forbidden text in any raw finding code
   still rejects the critique), and block indexes outside the requested plan's
   block range are dropped. Bounded sanitized summaries and valid provider
   semantic enums are preserved.
7. **Real source-grounding resolution (P1).** Grounding references are resolved
   against the immutable plan's own source excerpts (block indexes, word-index
   ranges, source/time spans within the refined window, or quoted source-excerpt
   text). Provider labels (`strategy`, `source_excerpt`, arbitrary strings) are
   not proof; unresolvable grounding is `EXTERNAL_REQUIRED_UNRESOLVED` /
   `BLOCKED_PENDING_VERIFICATION`.
8. **Fail-closed handoff freshness (P1).** `freshness` is
   `VERIFIED_CURRENT`/`STALE`/`NOT_CURRENT`/`UNVERIFIABLE`; only
   `VERIFIED_CURRENT` retains eligibility. A missing governance input
   fingerprint, unresolved inputs, or a fingerprint recomputation error is
   `UNVERIFIABLE` and forces every handoff plan `eligible_for_stage4_3=false`.
   `stage4_3_implemented` stays `false` and no winner is selected.
9. **Claim-to-source support resolution (P1).** A structural citation alone is
   no longer sufficient for `GROUNDED_IN_SOURCE`. A conservative deterministic
   lexical check classifies each substantive block as supported, ambiguous, or
   unsupported against its actually-cited source wording; unrelated citations,
   generic source text, and merely adjacent timing cannot ground a factual claim.
   Unsupported factual claims are `BLOCKED_PENDING_VERIFICATION`; ambiguous
   support routes to the selective provider when available and otherwise stays
   truthfully `GOVERNANCE_DEFERRED`/blocked, never approved. Supported factual
   statements, direct quotes, and non-factual explanation/inference tied to cited
   evidence remain grounded. Gemini stays non-authoritative and no web
   lookup/fact-checking is added. Policy/validation versions advanced to
   `stage4.2-v2`/`stage4.2-validation-v2` so prior governance is invalidated.
