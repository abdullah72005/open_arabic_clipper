# Stage 1, Stage 2, Stage 2.5, Stage 2.7, and Stage 3 pipeline

1. An operator submits a permitted public URL or uploads a local file.
2. The API creates a durable source and queued ingest job.
3. The worker acquires the source into managed storage and records its hash.
4. The worker calls `ffprobe` with an argument array and stores typed media
   metadata.
5. The worker extracts cached mono 16 kHz WAV audio and runs local faster-whisper.
6. Raw source/segment text, segment ordering, timestamps, and word timestamps are persisted.
7. Stage 2.5 derives conservative dialect-aware correction into separate
   corrected/final fields. A pure deterministic detector (no network, no LLM, no
   model loading, no audio decoding) classifies the source from immutable raw
   segment text into `EGYPTIAN`, `SAUDI`, `GULF`, `LEVANTINE`, `MSA`, or
   `UNKNOWN_ARABIC` (`None` = no Arabic evidence). The Egyptian lexicon and its
   optional provider apply only for confidently/explicitly EGYPTIAN sources;
   other profiles pass through unchanged. Exact Latin/technical/number tokens
   are preserved and `code_switch_suspected` evidence is persisted. It never
   realigns audio or overwrites raw evidence.
8. Stage 2.7 derives bounded contextual reconstruction through the managed local
   Ollama provider. It preserves raw text, segment timestamps, and word
   timestamps; it never creates, removes, merges, splits, or retimes segments.
   Local and Gemini providers share one dialect-neutral, preservation-first
   instruction plus validated profile-specific addenda, and the shared request
   carries the target segment's inherited effective dialect profile.
9. Timestamp-aware chunks use final operator text when present, otherwise
   corrected text; silence/quality signals are persisted separately.
10. The source reaches `READY_FOR_ANALYSIS`.
11. Stage 3 candidate analysis runs from the imperfect INDEX transcript, derives
    coarse deterministic proposals, scores content separately from transcript
    confidence, classifies content types, generates bounded source-faithful
    hooks, enforces same-source/cross-source novelty, and persists accepted and
    rejected candidates. It then advances the source to `READY_FOR_REFINEMENT`.
    Failure or cancellation leaves it at `READY_FOR_ANALYSIS`.

Stage 3 consumes imperfect INDEX-quality text. Candidate quality and transcript
confidence are separate; a strong moment with material uncertainty survives as
`CANDIDATE_NEEDS_REFINEMENT` for Stage 3.5, which owns targeted audio/transcript
refinement and exact boundaries. Source dialect is source evidence, not target
audience; code-switched text is preserved and omitted-English recovery is not
implemented in Stage 3. Unknown/third-party provenance never blocks local
analysis. Stage 3 semantic mode defaults to deterministic; `adaptive` uses
Gemini selectively when configured and `local_only` uses Qwen/Ollama only when
`CLIPFACTORY_LOCAL_QWEN_ENABLED=true`. See
`docs/STAGE_3_OPERATIONS.md` for the full design.

Each stage is persisted and idempotent. Stage runs persist canonical input and
output fingerprints; a changed upstream evidence reruns downstream derived
stages. Completed stages are skipped on resume; failures retain job and pipeline
error data for an operator retry. Unknown rights are allowed through local
ingest/probe only. Any future candidate generation, rendering, or publishing
must first pass an explicit authorization policy; it must reject `UNKNOWN`
rights by default.

Correction uses at most two neighboring segments on either side, but emits one
result per target segment only. Default operation is the local versioned lexicon.
An optional configured OpenAI-compatible local provider receives bounded batches
with stable IDs and may only approve a declared lexicon candidate; invalid,
missing, unsafe, or low-confidence output falls back to raw/lexicon text. Manual
operator text is feedback data only and does not train a model online.

Reconstruction defaults to the managed local Ollama provider (`qwen3.5:4b`) with
routing-driven, schema-validated two-pass candidates and per-candidate scores.
An unavailable or misconfigured provider persists a truthful status and leaves
Stage 2.5 final; it never blocks `READY_FOR_ANALYSIS`.

## Stage 3.5 candidate-scoped refinement

Stage 3.5 is explicit, candidate-scoped work after the source reaches
`READY_FOR_REFINEMENT`. It is **not** part of the automatic `_NEXT_STAGE` chain
and never advances every source; a source may stay `READY_FOR_REFINEMENT` while
individual candidates refine independently. It processes **candidate audio only**
— it never rediscovers clips, retranscribes a whole source, or uploads a whole
source — and extends the existing Celery/`ProcessingJob` system with a
`CANDIDATE_REFINEMENT` job kind plus a `candidate_refinements` row per
`(clip_candidate_id, priority)`. Only `CANDIDATE` (semantic quality) and
`FINAL_CLIP` (publication/caption quality) are valid; `INDEX` is rejected.

It extracts a bounded context window (candidate 5/5 s, final 8/8 s, max 150 s)
from the original source, runs targeted local faster-whisper as the mandatory
backbone (word timestamps, automatic language, no VAD, no
condition-on-previous-text, beam 5/8), converts clip-relative times to source
time once, and validates them inside the context window. Omitted English/code
switching is recovered only from actual targeted audio transcription; a
text-only model or `local_only` Qwen pass can never add omitted English.
Optional hosted `gemini-3.5-transcribe` (Interactions API, verbatim, word
timestamps) and `gemini-3.8-flash` adjudication are selective and gated by a
shared Redis priority/budget controller. Deterministic entity/ambiguity and
boundary refinement produce `CANDIDATE_REFINED`, `FINAL_TRANSCRIPT_READY`,
`NEEDS_MANUAL_TRANSCRIPT_REVIEW`, `PROVIDER_DEGRADED`, `REFINEMENT_FAILED`, or
`CANCELLED`. `FINAL_TRANSCRIPT_READY` is transcript readiness, not publishing
readiness. Stage 4 receives a typed read-only Stage 3.5 handoff; Stage 4.0, Stage
4.1, Stage 4.2, and Stage 4.3 are implemented. See
`docs/STAGE_3_5_OPERATIONS.md`.

## Stage 4.0 transformation eligibility

Stage 4.0 is explicit, candidate-scoped work after a candidate has a completed,
audio-backed Stage 3.5 refinement. It is **not** part of the automatic
`_NEXT_STAGE` chain and never analyzes every candidate. Queueing requires a
current retained Stage 3 candidate plus a usable Stage 3.5 refinement; a usable
`FINAL_CLIP` refinement is preferred when already ready, otherwise a usable
`CANDIDATE` refinement is used, and a queued/failed/cancelled/empty final row
never hides a usable candidate row. Final refinement is never enqueued here. If
no Stage 3.5 refinement exists, queueing fails with a prerequisite error rather
than analyzing the coarse INDEX transcript.

Stage 4.0 assesses whether the strongest source-retention moment has a credible
substantive transformation path, then emits a bounded set of strategy
directions. It never rewrites source speech, never localizes, and never treats
dialect as a target market. Deterministic gates reject directions that are
presentation-only, paraphrase, generic filler, distorted, fake-hooked,
retention-damaging, context-starved, under-original, or dependent on an
unverified external fact without marking it. The result is one of
`ELIGIBLE_FOR_TRANSFORMATION`, `ELIGIBLE_WITH_CAUTION`,
`TRANSFORMATION_REQUIRED`, `NO_TRANSFORMATION_STRATEGY_WORTH_USING` (a successful
completed analysis), `INSUFFICIENT_TRANSCRIPT_CONFIDENCE`, `INSUFFICIENT_CONTEXT`,
or `UNRESOLVED_POLICY_OR_PROVENANCE_RISK`. Zero recommended directions is a valid
success. Stage 4.1 receives a typed read-only handoff and is responsible for
planning; Stage 4.0 produces directions, not scripts. See
`docs/STAGE_4_0_OPERATIONS.md`.

## Stage 4.1 transformation plan generation

Stage 4.1 is explicit, candidate-scoped work after a current, non-stale Stage 4.0
analysis with at least one current recommended strategy. It is **not** part of
the automatic `_NEXT_STAGE` chain and never advances every candidate. It accepts
a usable `CANDIDATE` Stage 3.5 refinement and never requires `FINAL_CLIP`. It
produces zero to three concrete plans (normally at most one per recommended
strategy) and never selects or approves a winner.

Every valid plan has exactly one hero source excerpt placed as block 0 or 1;
source spans are selected by indexed Stage 3.5 word references (or a safe
full-window sentinel when word coverage is insufficient), and actual timestamps
and excerpt text are resolved deterministically, never trusted from the provider.
Authored material before the hero is capped (a stricter 1.5 s cap for short,
dense, joke, or payoff-first moments), blocks are capped at eight, and the source
hook/payoff is preserved. Original-value blocks must state what the viewer learns
beyond the excerpt; presentation-only edits, paraphrase scaffolding, fake hooks,
distortion, unsupported facts, and value kinds inconsistent with the Stage 4.0
strategy are rejected before persistence. External facts are carried as
verification placeholders that block dependent blocks; nothing is fabricated.
Narration is an abstract semantic requirement only (need, purpose, language,
register, duration, placement, dependencies) and never selects a TTS provider,
model, or voice. The result is `PLANS_GENERATED`,
`PLANS_GENERATED_WITH_VERIFICATION_REQUIRED`, `PLANNING_DEFERRED`,
`NO_VALID_PLAN_FROM_STRATEGY`, or `PROVIDER_UNAVAILABLE`; a deferred,
provider-unavailable, or zero-plan outcome is a successful semantic result, not a
source/pipeline failure. Stage 6 will generate speech; channel configuration will
decide the persistent narrator. See `docs/STAGE_4_1_OPERATIONS.md`.

## Stage 4.2 retention, originality, and platform-risk governance

Stage 4.2 is explicit, candidate-scoped work after a current, non-stale,
complete Stage 4.1 plan set with at least one current plan. It is **not** part of
the automatic `_NEXT_STAGE` chain and never advances every candidate. It
independently governs every current plan and never generates, mutates, repairs,
or selects one.

Deterministic logic revalidates plan integrity and derives independent
categorical dimensions — retention preservation, source-moment damage,
substantive originality, source dominance, semantic fidelity, generic filler and
redundant commentary, narration burden, verification completeness, plan-level
template/mass-produced feel, observable YouTube/Facebook reuse/spam risk,
coherence, and transformation proportionality — with no overall score. Hard
failures (semantic distortion, context reversal, false attribution, literalized
sarcasm, speculation-as-fact, unrelated source evidence, fake hooks, fabricated
claims, presentation-only transformation, no substantive value) are never offset
by another dimension. Essential unresolved external verification blocks a plan.
Repairable retention/narration/filler/template/proportionality damage requires
revision. Semantic-evidence deferral never falsely approves or rejects. Each plan
becomes `APPROVED_FOR_SELECTION`, `APPROVED_WITH_CAUTION`,
`BLOCKED_PENDING_VERIFICATION`, `REVISION_REQUIRED`, `REJECTED_BY_GOVERNOR`, or
`GOVERNANCE_DEFERRED`, with `eligible_for_stage4_3` a filter, not a ranking. The
candidate outcome is `PLANS_ELIGIBLE_FOR_SELECTION`, `NO_GOVERNOR_APPROVED_PLAN`,
or `GOVERNANCE_DEFERRED`, preserving separate verification-blocked,
revision-required, and rejected counts.

The platform-policy profile is immutable and code-defined
(`stage4.2-platform-policy-2026-09-17-v1`, checked 2026-09-17); it encodes durable
YouTube reused/inauthentic/spam concepts and Facebook original/unoriginal/spam
concepts and never claims algorithm safety or monetization. Copyright/rights,
platform originality, and spam/repetition remain separate. Account/channel-level
repetition is `DEFERRED_TO_STAGE_7`. The read-only Stage 4.2 -> 4.3 handoff
contains no winner; deterministic selection is Stage 4.3. See
`docs/STAGE_4_2_OPERATIONS.md`.

## Stage 4.3 deterministic final-plan selection

Stage 4.3 is explicit, candidate-scoped, **synchronous**, transaction-safe, and
provider-free. It consumes the authoritative `build_stage4_3_handoff()` freshness
contract plus current Stage 4.1 plan rows and commits each candidate to exactly
zero or one current survivor: `PLAN_SELECTED`, `PLAN_SELECTED_WITH_CAUTION`,
`NO_SELECTABLE_PLAN`, `SELECTION_DEFERRED`, or `STALE_SELECTION_INPUT`. Only
`VERIFIED_CURRENT` governance can select; `STALE` yields no selection,
`NOT_CURRENT`/`UNVERIFIABLE` defer, and internally inconsistent approved evidence
fails closed as deferred and is never repaired. A conservative caution allowlist
(`SOURCE_DOMINANCE_CONCERN`/`TEMPLATE_MASS_PRODUCED_FEEL` at MODERATE only)
allows automatic caution selection; clean approvals always arbitrate first.
Arbitration is a readable lexicographic comparison over Stage 4.2 evidence with
no weighted aggregate score, ending in stable plan identity. It adds no Celery
task, `ProcessingJob` kind, queue/executor, `PipelineStage`, `PipelineRun`, or
`_NEXT_STAGE` entry, and it never replans, re-governs, rewrites, researches,
refines transcripts, or renders.

One table, `transformation_plan_selections`, persists one row per candidate +
selection input fingerprint with a nullable `selected_plan_id`, an immutable
snapshot of the selected governance evidence, per-alternative dispositions, and
one database-current row per candidate enforced by a partial unique index.
Concurrent POSTs converge through candidate-row locking, savepoint uniqueness
recovery, and post-lock freshness revalidation. Selection and execution readiness
are separate: readiness is computed live by the read-only execution handoff
(`READY_FOR_FINAL_REFINEMENT`, `REQUIRES_FINAL_REFINEMENT_COMPATIBILITY_CHECK`,
`READY_FOR_EXECUTION_PREP`, `BLOCKED`) and CANDIDATE-grade planning remains
selectable without requiring `FINAL_CLIP`. Narration remains semantic only.
See `docs/STAGE_4_3_OPERATIONS.md`.

## Stage 5.0 execution preflight and render contract

Stage 5.0 is explicit, candidate-scoped, **synchronous**, transaction-safe, and
provider-free. It consumes the read-only Stage 4.3 `build_execution_handoff()`
plus live rows, validates publication-quality `FINAL_CLIP` evidence, performs
bounded managed-source media preflight, and persists one current
`RenderContract` per candidate/input fingerprint. Statuses are `BLOCKED`,
`FINAL_CLIP_REFINEMENT_REQUIRED`, `UPSTREAM_REVALIDATION_REQUIRED`,
`INVALID_SOURCE_BINDING`, `SOURCE_MEDIA_UNAVAILABLE`,
`READY_FOR_RENDER_PLANNING`, and `MATERIALIZATION_REQUIRED`; only the last two
are executable. It never renders, captions, tracks faces, synthesizes speech,
publishes, replans, re-governs, re-selects, or advances the source lifecycle, and
it adds no Celery task, `ProcessingJob` kind, queue/executor, `PipelineStage`,
`PipelineRun`, or `_NEXT_STAGE` entry.

FINAL_CLIP compatibility is deterministic over persisted evidence: bounded token
alignment, protected semantic operators (negation/exclusivity/modality, English
+ Arabic), entity/number changes, recovered code-switch tokens, change-ratio
bands, complete-thought/window-clipping checks, timing-drift bands, payoff/hook
coverage, grounding-quote preservation, and meaning-critical unresolved spans.
Material/unresolved outcomes become `UPSTREAM_REVALIDATION_REQUIRED`; spans that
no longer bind become `INVALID_SOURCE_BINDING`. Source excerpts rebind to current
`FINAL_CLIP` word timings without mutating the frozen Stage 4.1 plan; caption and
quote material always uses `final_clip_text`. `caption_input` preserves the exact
`FINAL_CLIP` transcript byte-for-byte with `logical_order_preserved=true` and no
BiDi manipulation; no subtitle file is generated in Stage 5.0.

One table, `render_contracts`, keeps one row per candidate + input fingerprint
with one database-current row per candidate via a partial unique index; cached
probe facts are reused for an unchanged media identity. `input_fingerprint`
excludes TTS voice/provider/model, caption font/animation, crop, B-roll, codec
tuning, publishing metadata, and final render artifact hashes. See
`docs/STAGE_5_0_OPERATIONS.md`.
