# Stage 3.5 candidate-scoped refinement operations and design

Stage 3.5 turns one explicitly requested Stage 3 coarse candidate into a
trustworthy, precisely bounded, audio-verified transcript. It processes
**candidate audio only**: it never rediscovers clips, never retranscribes a whole
source, and never uploads a whole source to a hosted provider. It is not part of
the automatic `_NEXT_STAGE` chain; a source may remain `READY_FOR_REFINEMENT`
while individual candidates have independent refinement states.

## Quality ladder and scope

- Whole source: `INDEX` (Stage 2.7 indexing quality; not a Stage 3.5 priority).
- Shortlisted candidate: `CANDIDATE` semantic quality.
- Explicitly selected final clip: `FINAL_CLIP` publication/caption quality.

Stage 3.5 accepts **only** `CANDIDATE` and `FINAL_CLIP`. `INDEX` is rejected by
every Stage 3.5 entry point. A source is never advanced automatically into
refinement; work is explicit and bounded.

`FINAL_TRANSCRIPT_READY` means the strict final transcript/timing gate passed
and no meaning-critical unresolved ambiguity remains. It does **not** mean the
clip is ready to publish. Publishing eligibility is evaluated separately.

## Pipeline position

```
... -> READY_FOR_ANALYSIS -> CANDIDATE_ANALYSIS -> READY_FOR_REFINEMENT
                                                       |
                                        explicit, candidate-scoped Stage 3.5
                                        (ProcessingJob kind CANDIDATE_REFINEMENT)
```

Candidate refinement is an extension of the existing Celery/`ProcessingJob`
system, **not** a new `PipelineStage` and **not** a `PipelineRun`. It owns a
`candidate_refinements` row per `(clip_candidate_id, priority)`.

## Candidate lifecycle

`QUEUED -> REFINING -> CANDIDATE_REFINED | NEEDS_MANUAL_TRANSCRIPT_REVIEW |
PROVIDER_DEGRADED | REFINEMENT_FAILED | CANCELLED`

- `CANDIDATE_REFINED`: semantic-quality transcript and refined boundaries exist.
- `PROVIDER_DEGRADED`: the optional hosted provider was unavailable, denied, or
  failed while safe local evidence exists; readiness and provider availability
  are separate concepts. The run is **not** cache-eligible and a later normal
  request retries only unfinished components.
- `NEEDS_MANUAL_TRANSCRIPT_REVIEW`: a meaning-critical unresolved entity/phrase,
  or a final transcript that cannot be safely timing-aligned.
- `REFINEMENT_FAILED`: required local/storage/invariant work failed with no safe
  result. Optional Gemini failure never produces this while local evidence is
  safe.
- `CANCELLED`: cooperative cancellation was observed. Cancellation never
  produces a ready state.

## Final-clip lifecycle

A `FINAL_CLIP` request may consume a valid candidate-grade refinement as its
seed bounds and evidence, but it runs its own stricter policy and has a distinct
top-level fingerprint. A direct explicit `FINAL_CLIP` request is permitted with
no candidate-grade row and seeds from the Stage 3 coarse bounds.

## Bounded audio-window service

`app/refinement/audio_window.py` owns extraction and is the only path from a
coarse candidate to a short mono 16 kHz WAV:

- Candidate pre/post context: **5 s / 5 s** (default).
- Final-clip pre/post context: **8 s / 8 s** (default).
- Maximum extracted refinement window: **150 s**.
- Boundary adjustment/search radius: **5 s**.

For coarse candidate `[start, end]` the context window is
`[max(0, start - pre), min(source_duration, end + post)]`; an oversized window is
shrunk symmetrically around the coarse midpoint. Context bounds and refined clip
bounds are persisted separately: context audio never automatically becomes the
final clip window.

Extraction uses a safe FFmpeg argument list, reads the **original source media**
(the original source and its hash stay authoritative), validates the cached
`AudioArtifact` against the source hash, writes atomically through a temp file,
validates duration/non-empty output/hashes, and removes partial temp files on
error. The deterministic path is
`storage/sources/{source_id}/candidate-refinements/{candidate_id}/{priority}.wav`.
Each candidate has at most one cached WAV per quality level, replaced atomically
when its audio-input fingerprint changes. Remote Gemini files are deleted in
`finally` on every exit path and their URIs are never persisted.

## Targeted local Whisper

Targeted local `faster-whisper` is the mandatory backbone and always available
as fallback. It reuses `WhisperEngine`, child-process isolation, peak-memory
reporting, and the shared heavy-model lease.

Defaults:

- Model `large-v3-turbo`; candidate beam `5`, final beam `8`.
- Word timestamps enabled.
- `language=None` automatic detection so embedded English can survive.
- `condition_on_previous_text=False`; VAD disabled so boundary speech is not
  trimmed.
- Deterministic temperature fallback preserved.

All local word/segment times begin relative to the extracted audio and are
converted **once** to source time by adding `context_start`, then validated
(`context_start <= word.start < word.end <= context_end`). Zero/non-positive
duration, non-finite values, and a genuine context escape are rejected
separately; a rejected record retains a bounded diagnostic (reason, record index,
original relative/source bounds, and context bounds) in its evidence JSON
without changing the original values or widening the window. Stage 3.5 never
rewrites the source `Transcript.raw_text`, immutable raw segments, or Stage 2
word timestamps. Context/hotword terms may be passed only when already evidenced
by the source transcript/operator input; suspected missing English is never
invented as a prompt.

## FINAL_CLIP context versus selected evidence

`FINAL_CLIP` observes the whole extraction context but only the settled refined
clip bounds are executable. The producer persists the complete bounded raw word
observation (all original records, including invalid ones) in the context
`TARGETED_LOCAL_ASR` evidence. The context record keeps its original validation
state; in the known zero-duration-outside-clip fixture it remains `REJECTED`.

A pure, versioned helper (`app/refinement/scope.py`) derives a separately
validated clip-only view from those observations at the actual refined bounds.
The one new nonblocking context case is a finite zero-duration record whose point
is inside the extraction context and strictly outside the clip. It is recorded
as context-only invalid evidence and never becomes a caption word. All of these
still refuse: zero duration inside or on a clip boundary; reversed, non-finite,
malformed, source-mapping-inconsistent or unknown-location timing; a context
escape; a selected word that crosses the clip edge; missing or incomplete raw
observations; unaccounted transcript text; and ambiguous tokenization.

For new scoped `FINAL_CLIP` outcomes, `final_transcript` and `word_timestamps`
are the accepted clip view, with a canonical ordered clip index and the separate
parent raw-record index. The full context automatic text is still preserved in
`automatic_transcript` and in the context evidence record. The view envelope is
persisted at `provider_evidence["final_clip_view"]` with exact context/clip
bounds, parent observation fingerprint, raw-to-clip mapping, clip text/words,
policy/validation state, reason, and policy/view fingerprints. The output
fingerprint binds the scope view, so changed bounds, raw observations, scope
policy or manual text invalidate downstream planning/caption artifacts.

Manual and source-segment operator text remain authoritative. They are projected
against the same canonical clip words through the existing bounded display
policy; unsupported text is preserved verbatim and the row stays in review. No
ASR draft is substituted over it. Legacy rows that retain only the first
timestamp diagnostic and a filtered word list are **not** retroactively upgraded:
they require a supported rebuild and remain reviewable. There is no general
multi-token alignment in this correction, and a valid selected view alone is not
`FINAL_TRANSCRIPT_READY`; confidence, consensus, ambiguity and manual-authority
gates still apply. Human verification of the spoken wording remains required.

## Evidence, consensus, and acceptance

Bounded `EvidenceRecord`s (INDEX raw, Stage 2.5, Stage 2.7, targeted local ASR,
hosted ASR, adjudication, operator) are deduplicated by stable fingerprint.
Final-text priority:

1. Candidate-refinement operator/manual text.
2. Existing source-segment operator text for the spans it covers.
3. Validated high-confidence adjudication.
4. Validated local/hosted ASR agreement.
5. Best validated audio-backed ASR result.
6. Existing Stage 2.7 accepted text.
7. Stage 2.5 corrected text.
8. Raw INDEX ASR.

Automated work never overwrites or removes manual text. Deterministic acceptance
checks cover finite/ordered/in-window timestamps, protected names/numbers/dates/
URLs/abbreviations/technical/Latin tokens, source-dialect preservation,
unexpected translation or MSA conversion, extreme edit/insertion ratios,
repeated/hallucinated text, material audio-backed disagreement, unsupported
omitted-English insertion, entity normalization/disagreement, and output bounds.

A text-only model is never the sole evidence for a word omitted by ASR. New
English/code-switch content enters accepted text only from actual targeted audio
transcription or explicit operator input.

## Omitted-English and code-switch recovery

Stage 3.5 owns omitted-English recovery using multilingual targeted audio
transcription — not text guessing. Latin words/names, abbreviations,
technical/product terms, mixed Arabic-English grammar, meaningful casing, and
the Arabic source dialect are preserved.

Example: audio `أنا عملت deploy للbackend امبارح`, INDEX `أنا عملت امبارح`.
Targeted audio ASR may restore `deploy`/`backend` (a `code_switch_recoveries`
metric/evidence record is persisted only when an audio-backed result adds
previously omitted Latin-bearing speech and it survives validation). A text-only
reconstruction given only the INDEX string may not add them.

## Entities and ambiguity

Practical important entities (people/product/place names, dates, times, numbers,
ages, percentages, money, scores, statistics, abbreviations, technical terms)
are extracted and compared. Normalization is comparison-only; the spoken/display
form is preserved in evidence.

- Candidate mode may finish `CANDIDATE_REFINED` with a flagged ambiguity,
  exposing unresolved spans and reduced confidence.
- Final mode never selects arbitrarily between conflicting numbers, dates, names,
  or technical terms. A material disagreement routes to adjudication when quota
  permits, otherwise to `NEEDS_MANUAL_TRANSCRIPT_REVIEW`.

## Hosted Gemini transcription and adjudication (optional)

Two providers, deliberately separate from the Stage 2.7 text-reconstruction
provider:

- **Transcription** — `gemini-3.5-transcribe` through the current documented
  **Interactions API**, verbatim mode (never smart transcription), word
  timestamps requested in the normal pass. Custom vocabulary is never combined
  with timestamps in the same pass; a separate explicitly routed pass would be
  required if ever justified, and each candidate is never double-called
  automatically. Source dialect is used only as a language/accent hint, never as
  a rewriting instruction.
- **Adjudication** — `gemini-3.8-flash` with structured output. It chooses one
  supplied reading or `UNRESOLVED` with confidence and a bounded reason, and may
  never rewrite the transcript or introduce a new sentence. Several small
  ambiguities from one candidate may share one bounded request; unrelated
  candidates are never batched.

Routing: targeted local Whisper always runs first. Candidate mode consults
hosted transcription only for material uncertainty (likely code-switch omission,
low-confidence speech, important entity uncertainty, substantial ASR
disagreement, insufficient boundary evidence). Final mode consults it whenever
configured unless an authoritative manual transcript plus timing evidence already
satisfies the strict gate. Hosted absence/denial/failure falls back to local
evidence.

Official documentation was checked on **2026-09-12** (Gemini API audio
transcription, Gemini 3.5 Transcribe, Gemini 3.8 Flash, Files API, Interactions
API). The installed SDK is `google-genai` 2.23.0; the Interactions surface used
here (`client.interactions.create` with
`generation_config.transcription_config`, `VerbatimTranscriptionMode`,
`WordInfo`, and `client.files.upload`/`delete`) was verified against that
installed package. The Files API retains uploads for up to 48 hours unless
deleted; ClipFactory deletes remote files best-effort immediately. Only one
bounded refinement WAV is uploaded per request; a single remote upload is reused
within a job when both transcription and adjudication need it; the SDK client is
created lazily, closed on every exit, and the API key is unwrapped only at client
construction and never logged/exposed/committed. `adaptive` and `gemini_only`
may send short candidate audio to Google Gemini; `local_only` sends nothing.

## Boundaries

`app/refinement/boundary.py` is a conservative deterministic refiner using
targeted word timestamps, existing source word timestamps, silence intervals,
pause gaps, sentence punctuation, first/last complete spoken word, and Stage 3
semantic coverage within the configured radius and duration limits.
`0 <= refined_start < refined_end <= source_duration`; the clip stays inside the
extracted context window; the full context bounds are never used as final bounds
by default; weak evidence conservatively retains the coarse edge. Candidate and
final rows may have different refined bounds. No rendering or framing decisions
are made.

## Global Gemini priority and budget

`app/refinement/admission.py` provides a small shared Redis-backed admission
controller (no new service, no billing system):

- `CRITICAL`: final-clip transcription, final meaning-critical adjudication,
  final entity verification.
- `HIGH`: future transformation/script work.
- `MEDIUM`: candidate transcription/adjudication and Stage 3 semantic work.
- `LOW`: future optional metadata/critics (disabled by default).
- `AVOID`: INDEX or arbitrary whole-source cleanup (always denied).

Admission is an atomic fixed-window request using a single Redis Lua script,
decided **before** any upload or generation. Configurable window duration, total
calls per window, reserved CRITICAL/HIGH calls, whether LOW is permitted, and a
provider cooldown limit. Reserves may never exceed total capacity. `CRITICAL`
may use the full window; `HIGH` cannot consume the CRITICAL reserve; `MEDIUM`
cannot consume either reserve; `LOW` runs only when explicitly allowed; `AVOID`
is always denied. Lower-priority denial degrades/skips without waiting or
failing the pipeline. Redis unavailability fails closed for hosted work while
local processing continues. A provider 429/quota response establishes a bounded
shared cooldown using a validated `Retry-After` when available; real exhaustion
truthfully denies even CRITICAL rather than spinning. Transient counters and
cooldowns are not fingerprint inputs; the static admission policy is. The
controller supplements the existing per-job caps rather than replacing them, and
is used by Stage 3 semantic calls (MEDIUM) and Stage 2.7 reconstruction
(FINAL_CLIP → CRITICAL, CANDIDATE → MEDIUM, INDEX → AVOID).

## Qwen

All Qwen/Ollama integrations are preserved. `CLIPFACTORY_LOCAL_QWEN_ENABLED=false`
remains the default. Default candidate/final refinement never routes through
Qwen before targeted Whisper or merely because Qwen is local. Adaptive refinement
uses targeted Whisper plus selective Gemini and never silently falls back to
Qwen. Explicit `local_only` refinement may use bounded Qwen reconstruction after
targeted Whisper only when the operator enables Qwen; because Qwen is text-only,
it can never claim audio recovery of omitted English and never calls Gemini.
Qwen/Ollama remains protected by the heavy-model lease and releases resources.

## Fingerprints, checkpoints, and idempotency

Top-level input fingerprints include the source ID and authoritative
source/audio hashes, candidate UUID/key and stable Stage 3 span identity and
analysis fingerprint, coarse/seed bounds, exact context bounds, priority, audio
hash, local model/settings/runtime identity, hosted provider/model/API/mode/
settings, reasoning model/API/schema/thinking/temperature, dialect
profile/confidence/selection evidence, transcript revision and relevant segment
evidence/manual text, routing/boundary/entity/validation/consensus/admission
policy and prompt/schema versions, and Qwen identity only when actually selected.
Unrelated Stage 4/renderer/frontend settings are excluded. Candidate and final
top-level fingerprints always differ.

Per-component checkpoints exist for audio extraction, local ASR, hosted
transcription, adjudication, accepted consensus, and boundary result. A rerun
reuses a component only when its exact dependency fingerprint matches. Local
checkpoint reuse requires the matching persisted `TARGETED_LOCAL_ASR` record in
`ACCEPTED` state; a rejected, missing, malformed, or mismatched record is never
promoted to accepted, and reuse preserves that record's own transcript, words,
window, model/settings, and confidence rather than aggregate final fields.
Accepted hosted/local/adjudication work survives retry, restart, cancellation,
and transient outage; a provider outage never overwrites previously accepted
evidence. Degraded runs remain non-cache-eligible so normal retries reconsider
only unfinished work. Queueing is idempotent: a matching ready/cache-eligible
result returns without a job or provider call; matching queued/running work
returns the existing job; degraded work queues only unfinished components; a
changed relevant input reuses the same refinement row and invalidates only
affected components. Duplicate refinement rows are never created, and concurrent
duplicate provider spending is prevented by atomically claiming the unique row
for a job.

## Cancellation and resources

The exact executing `ProcessingJob` is polled before extraction, before/after
local ASR, before upload, before/after every hosted call, before adjudication,
before boundary/final persistence, and once before a successful return. While
local Whisper runs, the model-process parent observes cancellation and
terminates/reaps the child. On cancellation the job/refinement stay `CANCELLED`,
no ready state is produced, no further expensive work is scheduled, completed
valid checkpoints are preserved, partial local temp files are removed, any
uploaded Gemini file is deleted, heavy-model leases/providers are released, and
secret material is scrubbed. Every executor exit path releases owned providers.

## Manual review (backend only)

A meaning-critical unresolved span blocks `FINAL_TRANSCRIPT_READY` and produces
`NEEDS_MANUAL_TRANSCRIPT_REVIEW`. Bounded unresolved spans persist a stable span
ID, source-time bounds where known, tiny context, candidate readings, evidence
IDs/providers, confidence, reason, entity type, meaning-critical flag, and
resolution state/operator resolution. `POST /api/refinements/{id}/manual`
submits authoritative corrected text and explicitly resolves named ambiguity IDs;
submitting text does not silently clear every unresolved span. Manual text always
wins on rerun. For scoped `FINAL_CLIP` rows this projection runs over the
canonical clip view, not the full context word sequence. Final readiness uses
one deterministic display projection over the existing ordered timed words:
unchanged text and exactly one isolated one-token replacement are supported,
retaining that word's original index/start/end. An interior replacement needs
unchanged immediate neighbours; an edge replacement needs two unchanged adjacent
tokens on its available side. Insertions, deletions, reordering, multiple or
unanchored replacements, ambiguous tokenization, and missing/invalid/nonmonotonic
timing are refused. Stored word records are validated as-is: missing or null
bounds are not defaulted to zero, explicit indices are not discarded, and
malformed records are not filtered out. The manual text is retained and the row
stays `NEEDS_MANUAL_TRANSCRIPT_REVIEW` with an actionable
`manual-display-alignment` unresolved span; it never falls back to ASR copy.
Changing manual text also changes the scoped output fingerprint so dependent
plans/contracts must be rebuilt. No review UI is built in Stage 3.5.

## Stage 4 handoff

`GET /api/candidates/{id}/stage4-handoff` returns a typed read-only handoff:
candidate ID/key/source ID, Stage 3 coarse bounds and segment/span identity,
exact refined start/end, transcript quality level, effective final transcript,
source-time word timestamps, confidence/readiness/status, unresolved spans and
manual-review requirement, dialect profile/confidence, code-switch and
recovered-English evidence, entity evidence, provider/routing evidence, Stage 3
content scores/types/hooks/novelty, provenance/rights/originality, and
refinement input/output fingerprints. It prefers a valid `FINAL_CLIP` row;
otherwise it exposes candidate-grade output truthfully and never labels
candidate quality as final-ready. Stage 4 transformation planning is **not**
implemented.

## Persistence, API, and CLI

Persistence: `candidate_refinements` (one row per candidate/priority) plus
`processing_jobs.candidate_refinement_id` and the `CANDIDATE_REFINEMENT` job
kind. The migration `20260913_0012` safely widens the job-kind constraint, adds
the table/indexes/constraints, and downgrades by removing Stage 3.5 jobs and
refinements before narrowing the constraint, preserving Stage 1–3 data.

API:

- `POST /api/candidates/{id}/refinements?priority=CANDIDATE|FINAL_CLIP[&force=]`
- `GET /api/refinements/{id}`
- `GET /api/candidates/{id}/refinements`
- `GET /api/candidates/{id}/stage4-handoff`
- `POST /api/refinements/{id}/manual`
- `POST /api/sources/{id}/candidate-refinements/batch[?limit=&force=]`
  (candidate-grade only, score-ordered, default limit 5, hard maximum 10; no
  bulk FINAL_CLIP)
- Existing `POST /jobs/{id}/cancel` works for candidate refinement.

CLI: `python -m app.cli candidate-refine CANDIDATE_ID [--priority ...]`,
`candidate-refine-batch SOURCE_ID [--limit N]`, `candidate-refinements
CANDIDATE_ID`, `candidate-handoff CANDIDATE_ID`,
`candidate-recovery --expectation MANIFEST.json [--apply] [--require-reuse]`.

## Local observation recovery

`candidate-recovery` is an operator-only, offline path that admits one already
captured, hash-pinned local faster-whisper observation as a raw `FINAL_CLIP`
checkpoint. It is not a capture framework, migration, or new job platform. It
never loads a model, constructs a provider, dispatches a job, or writes final
clip text.

Preview is the default and writes nothing. `--apply` revalidates under the
source/refinement locks, writes a uniquely named audit snapshot under
`storage/temporary/diagnostics/recovery-admissions/`, and replaces only the
`TARGETED_LOCAL_ASR` checkpoint plus its `audio_extraction`/`local_asr`
component fingerprints and recovery provenance. It keeps the context evidence
`REJECTED`, leaves manual/final text, word timestamps, jobs, source lifecycle and
publishing approvals untouched, and never marks the row ready or cache eligible.
Repeated identical apply is a no-op.

The expectation manifest pins the bundle path (storage-relative), source/
candidate/refinement/original-job identities, coarse/context/refined bounds,
candidate key, raw-span fingerprint, source/context-WAV/three-bundle-file
hashes, and the approved 67-token text hash. Validation is independent of
assertions made by the bundle: it recomputes the operator raw-span identity,
hashes the current source media and context WAV, and compares the captured
decoder options and source-evidenced hotwords with the current production
`FINAL_CLIP` contract. Any mismatch refuses before mutation.

The 2026-10-28 selected-window bundle is pinned as:

```json
{
  "bundle_relative_path": "diagnostics/local-recovery-b3fcfc05-20261028T044442Z-5edff6f75ef1",
  "source_id": "cafc529e-b0d8-401b-b439-2721a57de73c",
  "candidate_id": "b3fcfc05-638a-49f6-89f2-cdab89291871",
  "refinement_id": "b2ee68f5-1ec1-40c8-a9b0-3fb5b1d7c345",
  "original_job_id": "fc9ded35-74b0-46ab-b3d0-bee91b38e37c",
  "coarse_start": 384.28,
  "coarse_end": 408.66,
  "context_start": 376.28,
  "context_end": 416.66,
  "refined_start": 384.29999999999995,
  "refined_end": 408.67999999999995,
  "candidate_key": "9ebcd6855ab02d1017b5f7e7e45ae1948a2d4623363339c5bce4fcaf3f96fd3e",
  "raw_span_fingerprint": "60c630dc55c1e7d41b6024c45effb2d245e79e7182ccd7a4fd9cc66be929e4b7",
  "source_sha256": "77043cae0ceb6d15c27a73ea16842503e3e8e56044bc28a284772305ac63f706",
  "audio_sha256": "7ea64c4ac920a1ce8e308d02bdad7c32a119361617fdc8c537cb8a2ce1bce96e",
  "decoder_result_sha256": "d3c4a3090b1f820b96b0bef19f1994b1123f9912270088f14b4f749c39b82fbc",
  "result_sha256": "db4e7193e70357971c29510f5b9d11962073b1d88f57461df01c89ef8aeaf466",
  "prior_evidence_sha256": "c78633884b7a0bb96683178395c269af818c0a83be5cbfd4344f32654f15994e",
  "approved_text_sha256": "6cc761f9da32f3469c0cd0eca1554fd3a59fd47bf6afc7e088ce5332e74c7f9e",
  "approved_token_count": 67,
  "raw_word_record_count": 107,
  "expected_rejection_reason": "nonpositive word duration",
  "context_only_invalid_record_indexes": [105]
}
```

Required environment for this bundle: `large-v3-turbo`, CPU, compute type
`auto`, CPU compute type `int8`, final beam 8, temperature
`0,0.2,0.4,0.6,0.8,1.0`, `local_only` routing, local Qwen disabled, and no
Gemini key. The source region must still yield no source-evidenced hotword hint;
otherwise the decoder/context contract check refuses.

Operational sequence (only after independent code review and authorization):

1. `python -m app.cli candidate-recovery --expectation MANIFEST.json` (preview;
   require `derived_view.state=ACCEPTED`, `approved_text_matches=true`,
   `writes=0`, and `reuse_ready_after_apply=true`).
2. `python -m app.cli candidate-recovery --expectation MANIFEST.json --apply`
   (require `applied=true` and a snapshot path).
3. `python -m app.cli candidate-recovery --expectation MANIFEST.json
   --require-reuse` (necessary read-only preflight; exit 0 and
   `reuse_ready=true` are required before dispatch). This checks the current
   checkpoint only; it is **not** the runtime guard and does not constrain the
   later job.
4. Run the one-shot guarded continuation:

   ```bash
   python -m app.cli candidate-recovery-continue --expectation MANIFEST.json
   ```

   This captures the normal queue dispatch and executes exactly one
   `CANDIDATE_REFINEMENT` job for the pinned candidate/refinement, `FINAL_CLIP`,
   `force=false`, in this process with no broker/backlog consumer. The
   production factory still owns the decoder/input identity, and the normal task
   still owns job claim, failure/cancellation and cleanup. A process-local
   deny-only ASR collaborator is installed on the task's executor factory: if
   accepted/rejected observation reuse misses, it raises before any Whisper or
   hosted provider work. It never returns a fake transcript.

   The command fails closed unless
   `CLIPFACTORY_REFINEMENT_ROUTING_MODE=local_only`,
   `CLIPFACTORY_LOCAL_QWEN_ENABLED=false`, and no Gemini key is configured.
   Success requires `succeeded=true`, `denial_calls=0`,
   `metrics.component_reuse=1`, and no `metrics.local_asr_runs`. A reuse miss
   after a successful preflight fails the job with `denial_calls=1`; do not
   proceed to manual submission or assume a retry is free.
5. Submit the exact approved 67-token text through the existing manual endpoint,
   then run the existing readiness/currentness and Stage 4/caption checks.

Admission is recovery evidence, not a model run and not consensus. It does not
set `FINAL_TRANSCRIPT_READY`, `cache_eligible`, manual text, or publishing
approval.

## Configuration

All Stage 3.5 knobs are `CLIPFACTORY_`-prefixed (see `.env.example` and
`docs/ENVIRONMENT.md`). Notable defaults: routing `adaptive`, candidate context
5 s/5 s, final context 8 s/8 s, max window 150 s, boundary radius 5 s, candidate
beam 5, final beam 8, batch default 5 / max 10, transcription model
`gemini-3.5-transcribe`, adjudication model `gemini-3.8-flash`, admission window
60 s with 30 total calls, 8 CRITICAL reserve, 6 HIGH reserve, LOW disabled,
60 s provider cooldown.

## Verification

Deterministic services and mocked providers cover candidate/final refinement,
bounded extraction, code-switch recovery, entity ambiguity, boundaries, hosted
optionality/degradation, admission reserves, Qwen disabled/`local_only`,
caching/checkpoints/idempotency/cancellation, API/migration/handoff, and
Stage 2/2.5/2.7/2.7.1/3 regressions. No automated test makes a live network call
or loads hours of media.
