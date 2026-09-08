# Stage 2.7 Completion Remediation Design

## Decision status

This design is ready for operator review. It supersedes the provider-default,
fallback-state, routing, stage-cache, quality-score, and benchmark sections of
`docs/superpowers/specs/2026-09-05-stage-2-7-contextual-reconstruction-design.md`.
Unchanged safety, timestamp, raw-evidence, and Stage 3 exclusions remain in force.

Stage 2.7 remains incomplete until the implementation and real-audio gates in this
document pass. No Stage 3 work belongs in this change.

## Verified current-state failures

Repository and live PostgreSQL inspection on 2026-09-06 confirmed the four reported
failures:

1. `Settings.reconstruction_provider` and `.env.example` default to `disabled`.
   The Chernobyl source therefore persisted `reconstruction_method=stage2_5_fallback`
   while every reconstruction result remained unchanged.
2. forced retranscription clears only `Transcript.input_fingerprint`. The worker then
   queues normalization without `force`; `PipelineRunner` sees a historical successful
   run and skips it. The live source consequently contains new `text` values while its
   old derived `raw_text` keys are absent and its old final text remains visible.
3. `ContextualReconstructor` sends every segment through the same request shape. The
   request omits Stage 2.5 text, word probabilities, language, uncertainty reasons, and
   repeated-entity evidence. `WindowConfig` batch limits are unused, the candidate
   margin is hard-coded to `0.12`, and unresolved low-confidence spans receive no
   truthful state.
4. `assess_source` blends transcript confidence with speech density and silence into
   `overall_source_quality_score`. The problematic 159.36-second source reports
   `overall=0.877` and `audio=0.985` even though the contextual provider never ran.

The first 7.28 seconds of the live Chernobyl sample provide decisive acoustic evidence:

- `يام`: 0.552 word probability;
- `النطقة`: 0.697;
- `دقل`: 0.537;
- `للزعو`: 0.381.

The current transcript still contains `عملية إخلاق مؤقت`,
`ثلاث يام بس لتطهير النطقة`, and `يا جماعة مفيش دقل للزعو`.

## Selected approach

Keep the existing Stage 2.7 package and two-pass provider boundary, but complete it
instead of replacing it. Add an Ollama-managed implementation of the existing
OpenAI-compatible provider protocol, explicit routing evidence, real dependency
fingerprints, truthful reconstruction states, and a real benchmark runner.

This approach is safer than a whole-transcript rewrite and smaller than adding forced
alignment. It preserves stable segment identity while allowing constrained multi-word
repair inside each original segment.

## Non-negotiable invariants

- Raw Whisper `text`, segment order, segment start/end, nested word timestamps, and
  flattened word timestamps are never changed by normalization or reconstruction.
- A reconstruction may replace text only within one existing segment. It never creates,
  removes, merges, splits, reorders, or retimes segments.
- Final-text priority remains manual override, applied HIGH reconstruction, Stage 2.5,
  then raw ASR.
- Provider prompts contain transcript evidence only. They contain no web results, source
  title facts, or outside story knowledge.
- The provider may preserve Egyptian Arabic, repetitions, fillers, numbers, names, and
  code switching. It may not summarize, formalize, translate, or add clauses.
- Missing provider/model state is visible and lowers transcript quality. It never
  masquerades as successful reconstruction.
- GPU remains optional. CPU-only execution must work.
- Only owned or operator-authorized media may enter real benchmarks.

## Local provider and deployment

### Runtime choice

Add an `ollama` provider kind while retaining `openai_compatible` for other local
servers and `disabled` only as an explicit diagnostic mode. Product defaults become:

```text
CLIPFACTORY_RECONSTRUCTION_PROVIDER=ollama
CLIPFACTORY_RECONSTRUCTION_PROVIDER_BASE_URL=http://ollama:11434
CLIPFACTORY_RECONSTRUCTION_PROVIDER_MODEL=qwen3:8b
CLIPFACTORY_RECONSTRUCTION_PROVIDER_TIMEOUT_SECONDS=180
CLIPFACTORY_RECONSTRUCTION_RELEASE_AFTER_RUN=true
```

Add an Ollama Compose service under the `reconstruction` profile. Do not automatically
download model weights during ordinary `docker compose up`. Setup is explicit:

```bash
docker compose --profile reconstruction up -d ollama
docker compose exec ollama ollama pull qwen3:8b
docker compose --profile reconstruction up --build -d
docker compose exec backend python -m app.cli reconstruction-health
```

`reconstruction-health` must verify endpoint reachability, exact model presence, and
model digest. It must exit non-zero for missing configuration, endpoint, or model.

The 2026-09-06 machine has 7.4 GiB RAM, 2.0 GiB swap, 22 logical Intel Core Ultra 9
185H CPUs, and no CUDA. Ollama lists `qwen3:8b` at 5.2 GB and `qwen3.5:9b` at 6.6 GB.
Use `qwen3:8b` as the provisional practical model. Benchmark `qwen3.5:9b` only after
unloading Whisper and stopping avoidable services; record OOM or harmful swap pressure
as an infeasible result rather than weakening the safety gates. `qwen3.5:4b` is a
resource fallback candidate, not an automatic quality winner.

Model promotion is evidence-driven. The final `.env.example` model is the best model
that fits and passes the same frozen sample. If no candidate passes, retain the
provisional model and keep Stage 2.7 open.

### Provider lifecycle

Extend the provider protocol with:

```python
class ProviderAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    MISCONFIGURED = "MISCONFIGURED"

@dataclass(frozen=True)
class ProviderHealth:
    availability: ProviderAvailability
    provider: str
    model: str | None
    model_digest: str | None
    detail: str

class ReconstructionProvider(Protocol):
    def health(self) -> ProviderHealth: ...
    def generate_candidates(self, requests: list[GenerationRequest]) -> dict[int, list[ReconstructionCandidate]]: ...
    def resolve_candidates(self, requests: list[ResolutionRequest]) -> dict[int, ResolutionChoice]: ...
    def release(self) -> None: ...
```

`OllamaReconstructionProvider` uses `/v1/chat/completions` for structured output,
`/api/tags` for model/digest discovery, and `/api/generate` with `keep_alive: 0` for
release. `OpenAICompatibleReconstructionProvider.release()` is a no-op and its health
probe uses `/v1/models`. Provider health runs before candidate work. Release runs in a
`finally` block.

`WhisperEngine.transcribe` must consume the segment iterator, drop the local model
reference, and run garbage collection in `finally`. ASR and Ollama are not required to
remain resident together.

## Context and acoustic routing

### Request evidence

Each `WindowSegment` and provider target carries:

- stable segment ID/index, start, and end;
- immutable raw text;
- Stage 2.5 corrected text;
- language;
- previous and following raw plus corrected context;
- every word with text, start, end, and probability when available;
- aggregate acoustic metrics;
- repeated exact entities/spans from source-local memory;
- `routing_priority`, `routing_score`, `routing_reasons`, and `focus_spans`.

Windows retain the current 8-second/5-segment target and 15-second/8-segment hard
maximum. Batch bounds become active: 8 windows and 24,000 UTF-8 characters by default,
with hard maxima of 16 and 48,000. CPU latency and 7.4 GiB RAM favor these smaller
defaults.

### Routing score

Add `routing.py` with immutable `WordRisk`, `RoutingEvidence`, and `RoutingDecision`
types. Use configurable thresholds, not a single cutoff:

```text
low_word_probability       = 0.72
very_low_word_probability  = 0.50
low_mean_word_probability  = 0.78
high_low_word_ratio        = 0.25
high_routing_score         = 0.45

mean_risk       = 1 - mean(word probabilities)
low_ratio       = words below 0.72 / words with probabilities
very_low_ratio  = words below 0.50 / words with probabilities
run_density     = longest consecutive run below 0.72 / words with probabilities
segment_risk    = max(0, 1 - exp(avg_logprob)) when avg_logprob exists
routing_score   = 0.35*mean_risk + 0.25*low_ratio
                + 0.20*very_low_ratio + 0.10*run_density
                + 0.10*segment_risk
```

Priority is `RECONSTRUCT` when any of these is true:

- routing score is at least 0.45;
- any word is below 0.45;
- at least two words are below 0.72;
- low-word ratio is at least 0.25;
- mean word probability is below 0.78;
- Stage 2.5 left a correction uncertainty marker.

All other Arabic speech is `CONTEXT_CHECK`, not forbidden. This is the high-confidence
escape hatch: the model may still propose a phrase when raw acoustics are confident but
local discourse is incoherent. Empty/non-speech segments are `SKIP`. Manual overrides
remain visible as context but are never automatic targets.

Sort provider batches by priority while restoring exact source order before persistence.
Pass A sees every `RECONSTRUCT` and `CONTEXT_CHECK` target. Prompt wording tells the
model to inspect `focus_spans` first, return zero candidates for coherent text, and
prefer unchanged text under doubt.

### Multi-word resolution

Pass A may propose up to two new phrases for the target segment. Each proposal includes
change spans and evidence segment IDs. Application code adds raw and Stage 2.5 forms.

Pass B returns scores for every candidate, not only the selected candidate. Server code
recomputes each candidate's deterministic score and calculates the real top-versus-second
margin. Remove the hard-coded `margin=0.12`.

Acoustic confidence affects routing and the edit penalty. It never blocks correction by
itself. HIGH acceptance still requires phonetic similarity, semantic coherence,
protected-token preservation, no unsupported entity change, and a real score margin.
The known phrase `مفيش دقل للزعو` may therefore become `مفيش داعي للذعر` without any
production phrase map.

## Truthful reconstruction state

Persist this enum at transcript and segment level:

```python
class ReconstructionStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    APPLIED = "APPLIED"
    UNCHANGED_HIGH_CONFIDENCE = "UNCHANGED_HIGH_CONFIDENCE"
    LOW_CONFIDENCE_UNRESOLVED = "LOW_CONFIDENCE_UNRESOLVED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    FAILED = "FAILED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
```

Definitions:

- `NOT_REQUIRED`: empty/non-speech target or explicitly unsupported language.
- `APPLIED`: deterministic gates accepted a HIGH novel candidate.
- `UNCHANGED_HIGH_CONFIDENCE`: provider ran, selected raw/Stage 2.5, and no routing or
  validation uncertainty remains.
- `LOW_CONFIDENCE_UNRESOLVED`: evidence identified uncertainty but no candidate safely
  cleared HIGH.
- `PROVIDER_UNAVAILABLE`: configured provider/model health failed before or during the
  request. Stage 2.5 stays final.
- `FAILED`: an unexpected application/persistence failure prevented a valid result.
- `MANUAL_OVERRIDE`: operator text is final for this segment; automatic evidence remains.

Transcript status uses worst-first precedence:

```text
FAILED > PROVIDER_UNAVAILABLE > LOW_CONFIDENCE_UNRESOLVED > APPLIED
> MANUAL_OVERRIDE > UNCHANGED_HIGH_CONFIDENCE > NOT_REQUIRED
```

Keep `reconstruction_method` for implementation identity such as
`ollama:qwen3:8b`, not outcome. Add `provider_availability`, model digest, routed count,
unresolved count, and status counts to bounded reconstruction metadata. API, CLI,
health, source detail, and debug views expose the explicit status.

Provider absence may allow the media pipeline to reach `READY_FOR_ANALYSIS`, but the
source must display a degraded transcript banner and `manual_review_required=true`.
Neither API nor UI may label the reconstruction itself successful.

## Dependency fingerprints and forced reprocessing

### Persisted stage inputs

Add nullable `PipelineRun.input_fingerprint` and `PipelineRun.output_fingerprint`.
Historic rows remain nullable and therefore stale once. Add:

```python
@dataclass(frozen=True)
class StageExecutionResult:
    output_fingerprint: str

class StageExecutor(Protocol):
    def input_fingerprint(self, source: SourceVideo) -> str: ...
    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult: ...
```

`PipelineRunner` obtains the current input fingerprint before its skip decision. It
skips only when the latest successful run has the same non-empty input fingerprint and
`force` is false. It stores both fingerprints on success.

### Fingerprint chain

- audio extraction: source content hash plus FFmpeg extraction version;
- transcription: audio content hash plus all `TranscriptionOptions`;
- normalization: `transcription_revision`, transcription output fingerprint, ordered
  raw segment IDs/text/timestamps/word evidence, corrector version, and correction
  settings/provider identity;
- reconstruction: normalization output fingerprint, ordered raw and Stage 2.5 text,
  reconstruction algorithms/settings/prompts/provider/model/digest;
- audio analysis: audio hash, transcription revision, analysis algorithm version;
- source quality: audio-analysis output fingerprint, reconstruction output fingerprint,
  and quality algorithm version.

Add `Transcript.transcription_revision` and `normalization_fingerprint`. Increment the
revision only after Whisper produces a complete successful result. A forced ASR run
passes `force=True` into `TranscriptionExecutor.execute`, bypasses its internal cache,
and increments the revision even when text happens to match. Derived fingerprints then
become stale automatically. API and CLI stop mutating fingerprint columns to empty
sentinels.

When a stage finishes, the worker queues its normal successor with `force=False`.
Fingerprint mismatch causes required downstream execution. An identical non-forced
request skips safely. A force flag is no longer forwarded manually through the whole
chain.

`AudioAnalysisExecutor` may reuse expensive audio features when `audio_hash` matches,
but it must recompute transcript-derived speech rate and source quality whenever their
fingerprints change.

## Separate transcript and media quality

Add these persisted fields to `SourceQualityAssessment`:

```text
transcript_quality_score: float
low_confidence_word_ratio: float
unresolved_segment_ratio: float
manual_review_required: bool
input_fingerprint: str
```

Retain `audio_quality_score`. Keep `overall_source_quality_score` only for backward
compatibility and set it to `min(audio_quality_score, transcript_quality_score)` so a
good waveform cannot hide unusable text. UI labels it `Conservative source floor`, not
`Transcript quality`.

Compute segment transcript quality from the available acoustic baseline and state:

```text
raw_acoustic = weighted public Whisper confidence, or 0 when absent

APPLIED                    = 0.35*raw_acoustic + 0.65*reconstruction_confidence
UNCHANGED_HIGH_CONFIDENCE  = raw_acoustic
LOW_CONFIDENCE_UNRESOLVED  = min(raw_acoustic, 0.45)
PROVIDER_UNAVAILABLE       = min(raw_acoustic, 0.40) when routed, else raw_acoustic
FAILED                     = min(raw_acoustic, 0.25)
MANUAL_OVERRIDE            = 0.95 for that reviewed segment
NOT_REQUIRED               = excluded from speech-weighted average
```

Before reconstruction, an applied Stage 2.5 correction may substitute
`0.50*raw_acoustic + 0.50*correction_confidence`; unchanged Stage 2.5 text gets no
artificial confidence boost. Weight segment scores by timed word count, falling back to
duration, then an equal weight. Clamp to `[0, 1]`.

`low_confidence_word_ratio` counts probabilities below 0.72. The unresolved ratio counts
speech segments in `LOW_CONFIDENCE_UNRESOLVED`, `PROVIDER_UNAVAILABLE`, or `FAILED`.
`manual_review_required` is true when any such segment exists or provider availability
is not `AVAILABLE` for a configured Arabic transcript.

The quality reason list records raw acoustic confidence, low-word ratio, unresolved
ratio, correction/reconstruction confidence, provider availability, reconstruction
status, and manual-review state. Speech density and silence remain audio reasons only.

## API and UI behavior

- Add reconstruction status and split quality fields to transcript/source responses.
- Add provider health to `/system/health`; unavailable configured Ollama is `DEGRADED`,
  not silently omitted.
- Always render a transcript status card, even when no correction was applied.
- Render distinct `Audio quality`, `Transcript quality`, low-confidence-word ratio,
  unresolved-segment ratio, provider/model, and manual-review requirement.
- Show per-segment status, routing reasons, focus-span probabilities, candidate,
  confidence, and flags inside correction details.
- Add `Reconstruct transcript` and `Force retranscription` actions with job refresh.
- Keep timestamp seek bound to original segment start.

## Real pipeline benchmark

Replace the current report-reader command with an actual runner. A private manifest
references storage-owned authorized media and human reference rows; it does not embed a
precomputed report. The runner must execute:

1. raw `large-v3-turbo` ASR;
2. Stage 2.5 normalization/correction;
3. Stage 2.7 through the configured live provider;
4. deterministic comparison plus a human-review worksheet.

The manifest contains source path, authorization, split, topic/categories, clip bounds,
and reference segment text. The output stores raw, Stage 2.5, Stage 2.7, reference,
status, confidence, timings, peak RSS, peak VRAM when present, model ID, and digest under
`storage/benchmarks/`. Console output remains aggregate and path-only.

Create a private full-reference manifest for source
`37c14f55-eacb-4d9f-8775-47a721cba5a9` (159.36 seconds) if the operator can review the
whole clip. At minimum, freeze and manually inspect the first 30 seconds, including the
three supplied failures. This known sample is a required regression set, not the only
unseen acceptance set. The broader gate still requires at least five non-overlapping
clips, 2–5 minutes, three topics, two recordings, and all existing category coverage.

Compare `qwen3:8b` and `qwen3.5:9b` with identical prompts, thresholds, and frozen input
when the latter completes without OOM or severe swap pressure. Record `qwen3.5:9b` as
infeasible when it cannot fit. Compare `qwen3.5:4b` only as a practical fallback.
Selection order is semantic accuracy, hallucination/regression rate, reliability, then
speed.

Report counts for `improved`, `unchanged_correct`, `unchanged_wrong`, `regressed`,
`hallucinated`, and `unresolved`. Report CER/WER as secondary diagnostics. A human must
confirm semantic labels; string distance never decides readiness.

## Completion gate

Stage 2.7 is complete only when all items have current evidence:

1. real local provider health is `AVAILABLE` and the live worker invokes it;
2. provider regression tests and real audio prove multi-word repair;
3. raw ASR text and all timestamps remain unchanged through downstream stages;
4. forced retranscription reruns every stale transcript-derived stage;
5. media/audio and transcript quality are separate and the bad sample no longer reports
   high transcript quality;
6. unavailable provider/model is visible in persistence, health, API, CLI, and UI;
7. the real unseen Egyptian benchmark improves materially;
8. regression rate is at most 2%, preserved-correct rate at least 98%, and hallucinated
   facts/names/numbers/clauses equal zero;
9. the Chernobyl first 30 seconds and preferably full 159.36 seconds are manually
   re-tested;
10. all Stage 2, 2.5, 2.6, and 2.7 backend/frontend tests pass;
11. README, STATUS, AGENTS, ENVIRONMENT, architecture, pipeline, benchmark, local setup,
    and troubleshooting docs match the measured installation;
12. final report ends with exactly `READY FOR STAGE 3` only if every item passes;
    otherwise it ends with exactly `STAGE 2.7 MUST CONTINUE`.

## Research snapshot

- Qwen documents Qwen3-8B and Egyptian Arabic among 119 supported languages/dialects:
  <https://qwenlm.github.io/blog/qwen3/>
- Ollama lists `qwen3:8b` at 5.2 GB and `qwen3:14b` at 9.3 GB:
  <https://ollama.com/library/qwen3>
- Ollama lists `qwen3.5:9b` at 6.6 GB and `qwen3.5:4b` at 3.4 GB:
  <https://ollama.com/library/qwen3.5>
- Ollama supports `response_format` through `/v1/chat/completions`:
  <https://docs.ollama.com/api/openai-compatibility>
- Ollama recommends JSON schema plus temperature zero for deterministic structured
  output: <https://docs.ollama.com/capabilities/structured-outputs>
- Ollama's native generate API supports `keep_alive: 0` to unload a model:
  <https://docs.ollama.com/api/generate>
