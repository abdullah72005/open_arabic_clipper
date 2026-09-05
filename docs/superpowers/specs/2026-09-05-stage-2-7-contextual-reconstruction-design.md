# Stage 2.7 Contextual Egyptian Transcript Reconstruction Design

## Decision status

This document closes Stage 2.7 architecture decisions. Implementation must not
start until the operator approves this spec and its linked implementation plan.

Stage 2.7 upgrades transcript quality after Stage 2.5. It does not start Stage 3.
The design keeps every ASR segment ID and timestamp, keeps Stage 2.5 output as a
fallback, and adds a separately persisted contextual reconstruction layer.

## Scope

Stage 2.7 must:

- reconstruct multi-word Egyptian Arabic ASR failures using 5–15 seconds or
  3–8 neighboring segments of transcript context;
- preserve raw ASR text, segment order, timestamps, word timestamps, natural
  Egyptian grammar, repetitions, fillers, code switching, names, and numbers;
- generate a small candidate set, resolve it contextually, and retain raw text
  whenever evidence is weak;
- use existing faster-whisper acoustic indicators without relying on private
  decoding APIs or adding forced alignment;
- maintain source-local entity consistency without external knowledge;
- persist confidence, reason, quality flags, and performance evidence needed by
  operators and a future Stage 3;
- benchmark against unseen, operator-authorized Egyptian audio rather than claim
  success from deterministic text fixtures.

Stage 2.7 must not:

- select clips, score clip candidates, render, publish, or authorize media;
- alter audio alignment, segment IDs, starts, ends, or segment order;
- translate, formalize, summarize, clean up style, remove fillers, or improve a
  story;
- infer facts, names, numbers, clauses, or explanations not supported by speech;
- train online from manual overrides;
- require CUDA, a remote service, or a bundled LLM to keep the pipeline usable.

## Existing baseline

Stage 2.5 currently corrects one segment at a time with two neighboring segments
on each side. Its optional provider may only choose between raw text and an exact
versioned lexicon candidate. This is safe but structurally unable to reconstruct
unseen multi-word errors.

Current persistence already separates raw, corrected, operator, and final text.
Current faster-whisper 1.2.1 serialization stores segment `avg_logprob`,
`no_speech_prob`, and word `probability`; it does not store segment tokens,
compression ratio, or fallback temperature.

## Considered approaches

### Selected: two-pass text reconstruction with deterministic acceptance

For every original segment, construct a bounded overlapping transcript window.
Pass A proposes no more than two novel reconstructions. Application code adds the
raw and Stage 2.5 forms, deduplicates to at most three total candidates, then Pass
B ranks those candidates. A deterministic validator and confidence gate decide
whether the winner is safe enough to persist and auto-apply.

This approach can repair unseen phrases, preserves segment identity, batches
model work, exposes an audit trail, and lets the system reject fluent but
unsupported output.

### Rejected: one full-transcript rewrite

A single whole-source rewrite gives broad context but makes segment mapping,
retry isolation, payload size, hallucination detection, and partial recovery
unsafe. Re-aligning the rewritten paragraph would violate the timestamp rule.

### Rejected: Whisper n-best or forced alignment as the primary design

Installed faster-whisper exposes tokens, segment log probability, no-speech
probability, compression ratio, fallback temperature, and word probabilities.
Its public `transcribe` result returns the chosen sequence, not a stable n-best
contract. CTranslate2 internally produces hypotheses, but reaching them would
couple ClipFactory to private faster-whisper internals. A forced aligner would add
a new model, resource cost, and alignment failure modes before evidence shows it
is needed. Stage 2.7 therefore consumes stable scalar evidence only.

## Pipeline architecture

The durable pipeline becomes:

```text
audio
  -> faster-whisper raw transcript
  -> Stage 2.5 normalization/correction
  -> Stage 2.7 contextual reconstruction
  -> timestamp-aware chunks and audio analysis
  -> READY_FOR_ANALYSIS
```

`PipelineStage.CONTEXTUAL_RECONSTRUCTION` sits between
`TRANSCRIPT_NORMALIZATION` and `AUDIO_ANALYSIS`. It has its own executor,
pipeline run, job kind, fingerprint, retry surface, CLI command, and API trigger.
New sources run it automatically. Existing sources may run it without repeating
ASR through an explicit reconstruction job.

Provider absence or a failed window never corrupts or blocks the usable Stage 2.5
transcript. The stage persists fallback results and operational flags, then lets
the source continue to `READY_FOR_ANALYSIS`.

## Module boundaries

New code lives under `app.transcription.reconstruction`:

- `types.py`: strict immutable request, candidate, decision, acoustic, flag, and
  confidence types;
- `windows.py`: deterministic duration/segment bounded context windows;
- `phonetics.py`: Arabic comparison normalization and weighted phonetic distance;
- `entities.py`: source-local entity memory built only from transcript evidence;
- `providers.py`: Pass A and Pass B protocols, prompts, JSON schemas, and the
  OpenAI-compatible HTTP adapter;
- `validation.py`: identity, protected-token, length, insertion, and evidence
  validation;
- `confidence.py`: deterministic scoring and HIGH/MEDIUM/LOW policy;
- `service.py`: batch orchestration, fallback, fingerprinting, and result
  assembly;
- `benchmark.py`: unseen-audio comparison and review-report generation.

Stage 2.5 `correction.py` and its lexicon stay intact. Reconstruction consumes its
output but does not broaden its provider permissions.

## Context-window algorithm

Each original segment is a target exactly once. Its model input is an overlapping
read-only window; only the target segment has an output slot.

For target index `i`:

1. Start with segment `i`.
2. Alternately add the closest previous and following segment, preserving source
   order and favoring the shorter current side.
3. Stop when the window spans at least 8 seconds or contains 5 segments.
4. Never include more than 15 seconds or 8 segments. A target segment longer than
   15 seconds is allowed alone.
5. At source edges, use available context. A short source may contain fewer than
   3 segments or less than 5 seconds.

Every context item carries stable `segment_index`, start/end, immutable raw text,
Stage 2.5 corrected text, and acoustic indicators. Manual override text is not
model input, so automatic reconstruction remains deterministic and cannot learn
online from an operator edit.

Requests are grouped by both count and serialized size: at most 16 target windows
and 48,000 UTF-8 characters per provider request. These defaults are configurable
within hard maxima of 32 windows and 96,000 characters. There is never one HTTP
request per one-second segment.

## Transcript-derived source context

No web lookup, metadata description, title-based fact injection, or external
knowledge enters reconstruction. Global context contains only:

- detected transcript language;
- exact entity forms extracted from raw text;
- repeated exact Arabic 2–4 token spans;
- exact Latin/code-switched spans and numeric strings;
- nearby raw and Stage 2.5 text from the bounded window.

No generated topic summary is used. This avoids converting an early model guess
into source-wide pseudo-evidence.

## Source-local entity memory

`SourceEntityMemory` is rebuilt for each reconstruction run and never shared
between sources. It accepts:

- exact Latin runs and digit strings from raw segments;
- an Arabic span nominated by the provider only when that exact span appears in
  raw text;
- repeated Arabic forms only after two source occurrences;
- a canonical form only when it already appears verbatim in the source.

The memory records surface form, exact supporting segment IDs, occurrence count,
and normalized comparison key. It may preserve or prefer an observed repeated
form. It may not invent, complete, translate, or fetch an entity. A proposed name
change without an observed canonical form is rejected and flagged
`POSSIBLE_ENTITY_ERROR`.

## Pass A: plausible reconstruction

`ReconstructionProvider.generate_candidates` receives batches of target windows.
For each target it may return zero, one, or two novel candidates. The prompt
requires:

- spoken Egyptian wording rather than formal Arabic;
- phonetic relationship to raw speech;
- exact preservation of supported Latin strings and numbers;
- no new facts, names, clauses, explanations, or stylistic cleanup;
- one non-empty output for the same target segment ID;
- evidence segment IDs and explicit raw-to-proposed change spans.

Application code validates candidate shape before Pass B. It always inserts raw
text and Stage 2.5 corrected text, deduplicates by Arabic comparison form, and
keeps at most three candidates total. Provider order has no acceptance meaning.

## Pass B: contextual resolution

`ReconstructionProvider.resolve_candidates` receives the same window, validated
candidates with opaque IDs, source entity memory, and acoustic indicators. It
returns exactly one candidate ID plus bounded scores for semantic coherence,
Egyptian naturalness, discourse continuity, and entity consistency.

The application, not the provider, computes phonetic similarity, edit size,
protected-token checks, overall score, and candidate margin. Raw remains a valid
winner. Missing IDs, duplicate IDs, invalid scores, malformed JSON, or a choice
outside the candidate set invalidates that target and selects Stage 2.5 fallback.

Both passes use temperature 0 and strict JSON schema. Prompt and schema versions
are included in the reconstruction fingerprint.

## Arabic phonetic plausibility

Comparison normalization is never display normalization. It removes diacritics,
tatweel, punctuation, and whitespace differences; normalizes Alef forms and
Yeh/Alef Maqsura; treats dropped Hamza and final Teh Marbuta with reduced cost;
and supports comparison across one-to-three-token merge/split spans.

A weighted Damerau-Levenshtein scorer uses reduced substitution cost, not total
equivalence, for common Egyptian/Arabic confusables such as emphatic/plain pairs,
Hamza carriers, and likely connected-speech reductions. Exact spaces cost zero
inside a merge/split span. Unrelated consonant changes, new content tokens, Latin
changes, and digit changes keep full cost.

Production code contains no phrase-specific replacement for the supplied
examples. Those phrases appear only in regression/evaluation data.

## Acoustic evidence

Stage 2.7 extends raw serialization to retain public faster-whisper fields:

- segment `tokens`;
- `avg_logprob`;
- `compression_ratio`;
- `no_speech_prob`;
- decoder `temperature`;
- word start/end/text/`probability`.

Per-segment acoustic confidence is:

```text
0.50 * mean(word probabilities, when present)
+ 0.35 * clamp(exp(avg_logprob), 0, 1)
+ 0.15 * clamp(1 - no_speech_prob, 0, 1)
```

When word probabilities are absent, their weight is redistributed proportionally
between the other available terms. Missing all terms produces `None`, not a false
zero.

These values say whether raw ASR was confident; they do not directly prove a
candidate phrase. They therefore add a penalty to large changes against
high-confidence raw text but never independently approve a correction.

No automatic audio re-decode is included. A later evidence-backed change may add
one, but candidate-biased prompts or hotwords must not be treated as an objective
acoustic score.

## Safety validation

A novel candidate is rejected before ranking if any rule fails:

- target ID differs, is missing, duplicated, reordered, or has empty text;
- start/end or any timestamp-like value is returned by the provider;
- normalized Latin-token or digit sequence changes;
- an Arabic entity changes to a form absent from source entity memory;
- output character length is outside 0.60–1.60 times baseline length;
- token count delta exceeds `max(3, ceil(0.40 * baseline_tokens))`;
- more than two content tokens have no phonetic alignment to a raw/baseline span;
- a new sentence boundary or clause-sized insertion lacks an aligned raw span;
- reported changes do not point to substrings in raw/baseline and candidate text;
- phonetic similarity is below 0.55.

The raw and Stage 2.5 fallback candidates bypass novel-candidate rejection because
they are already persisted evidence. Provider confidence never bypasses a rule.

## Confidence and application policy

For a valid candidate, the server computes:

```text
score =
    0.35 * deterministic_phonetic_similarity
  + 0.25 * provider_semantic_coherence
  + 0.15 * provider_discourse_continuity
  + 0.10 * provider_egyptian_naturalness
  + 0.10 * deterministic_entity_consistency
  + 0.05 * provider_selection_confidence
  - 0.20 * raw_acoustic_confidence * normalized_edit_ratio
```

The candidate margin is winner score minus runner-up score.

- `HIGH`: score at least 0.86, margin at least 0.12, phonetic similarity at
  least 0.72, semantic coherence at least 0.80, and every safety rule passes.
  A changed HIGH candidate is auto-applied.
- `MEDIUM`: score at least 0.74, margin at least 0.08, edit ratio at most 0.20,
  token delta at most one, phonetic similarity at least 0.85, and every safety
  rule passes. It is stored as a review proposal but does not replace final text.
- `LOW`: every other outcome. It is stored as unresolved and does not replace
  final text.

Only HIGH enters `final_text`. This resolves the brief's stricter final-priority
rule in favor of safety. Thresholds are versioned configuration and may move only
after the unseen-audio benchmark is rerun and documented.

## Quality flags

Each segment persists a deduplicated list drawn from:

- `HIGH_ASR_UNCERTAINTY`: acoustic confidence below 0.60, any word probability
  below 0.35, or no-speech probability above 0.50;
- `MULTIWORD_RECONSTRUCTION`: an accepted change touches more than one source
  token or changes a word boundary;
- `POSSIBLE_ENTITY_ERROR`: entity evidence conflicts or cannot support a change;
- `CONTEXT_DEPENDENT_CORRECTION`: contextual winner differs from the best
  deterministic phonetic candidate;
- `LOW_CONFIDENCE_UNRESOLVED`: uncertain raw/baseline remains final or a
  MEDIUM/LOW proposal is rejected;
- `RECONSTRUCTION_PROVIDER_ERROR`: one or both provider passes failed for the
  target window.

Future Stage 3 receives these strings without interpreting unknown values as
safe. Stage 3 itself is not implemented here.

## Persistence and final-text priority

Existing `Transcript.corrected_text` remains the Stage 2.5 automatic correction.
Stage 2.7 adds transcript-level `contextual_reconstructed_text`,
`reconstruction_fingerprint`, `reconstruction_confidence`,
`reconstructed_segment_ratio`, `reconstruction_method`,
`reconstruction_version`, `reconstruction_processing_duration`, and a bounded
JSON `reconstruction_metadata` summary.

Every segment keeps existing fields and gains:

- `contextual_reconstructed_text`: accepted HIGH text, otherwise Stage 2.5 text;
- `reconstruction_candidate_text`: winning novel proposal, or `null`;
- `reconstruction_applied`: boolean;
- `reconstruction_confidence`: numeric score;
- `reconstruction_confidence_level`: `HIGH`, `MEDIUM`, or `LOW`;
- `reconstruction_method` and `reconstruction_version`;
- `reconstruction_changes`: validated spans and evidence IDs;
- `quality_flags`: the strings above.

Full rejected candidate lists and provider responses are not stored in the main
transcript JSON. The benchmark stores its own bounded evaluation artifact through
`StorageService`; production logs contain counts and hashes, not transcript text.

Final segment priority is exact:

```python
if operator_text:
    final_text = operator_text
elif reconstruction_applied and reconstruction_confidence_level == "HIGH":
    final_text = contextual_reconstructed_text
elif corrected_text:
    final_text = corrected_text
else:
    final_text = raw_text
```

Manual overrides remain untouched across reconstruction when the segment index
and raw text still match. Chunks, transcript `final_text`, search, and display are
rebuilt atomically from this priority.

## Idempotency and reprocessing

The SHA-256 reconstruction fingerprint covers:

- transcription input fingerprint;
- ordered raw segment IDs/text/timestamps;
- Stage 2.5 correction version and ordered corrected text;
- window/entity/phonetic/confidence algorithm versions;
- provider kind, exact model identifier, Pass A/Pass B prompt/schema versions;
- all thresholds and batch bounds.

It excludes API keys and manual overrides. Matching successful fingerprints skip
provider work and only refresh final/chunk state when manual text changed.

The runner gains an explicit forced-run path that creates a new pipeline run
instead of treating any historical success as permanently current. A forced
reconstruction job reruns only Stage 2.7. A forced retranscription reruns the
dependent normalization, reconstruction, and analysis chain.

## Provider and deployment policy

`openai_compatible` remains the only Stage 2.7 transport. It uses standard
library HTTP plus strict Pydantic response validation; no OpenAI SDK dependency
is added. Ollama is a supported local endpoint because its OpenAI-compatible chat
API accepts JSON-schema structured outputs. It is not added as a required Compose
service and no model is downloaded automatically.

Provider configuration remains disabled by default, preserving offline pipeline
operation. With no provider, Stage 2.7 emits safe fallback fields and flags.
Reference evaluation uses local Ollama `qwen3:8b` at temperature 0 because it is
multilingual and small enough for common CPU/RAM setups. This is a benchmark
candidate, not a promoted product default. Exact model identifier and digest are
recorded in results. Promotion requires the acceptance gate below; failure leaves
Stage 2.7 status as `MUST CONTINUE`.

## Error handling

- Provider batches receive two attempts total for connection reset, timeout, or
  HTTP 429/5xx using bounded exponential delays of 0.5 then 1.0 seconds.
- Schema, identity, or safety errors are not retried with the same response.
- Pass A failure selects Stage 2.5 for every affected target.
- Pass B failure stores valid Pass A winner only as LOW review evidence and
  selects Stage 2.5 final text.
- One failed batch does not discard successful batches.
- Database mutation happens only after all target results exist; one transaction
  replaces derived reconstruction annotations and chunks.
- Database failure uses normal durable pipeline retry. Raw and Stage 2.5 evidence
  remain unchanged.
- Logs include source ID, batch number, provider/model, duration, counts, and
  error class; they exclude transcript and API-key content.

## API, CLI, and operator UI

Existing transcript routes remain backward compatible. Transcript responses add
the new aggregate and per-segment fields.

New operator entry points:

- `POST /api/sources/{id}/reconstruct?force=false` queues a reconstruction job;
- `python -m app.cli reconstruct SOURCE_ID --force` does the same;
- `python -m app.cli benchmark-reconstruction MANIFEST.json` runs the private
  unseen-audio evaluation and prints JSON.

Source detail keeps final text primary. Correction details show raw, Stage 2.5,
Stage 2.7 candidate/applied text, confidence level/score, flags, method, and
manual override. Timestamp seeking still uses the original segment start.

## Unseen-audio benchmark

Fixture tests are necessary regression checks but cannot complete Stage 2.7.
Completion requires a private, versioned manifest under storage ownership with:

- at least 5 non-overlapping real Egyptian clips;
- 2–5 minutes total evaluated speech;
- at least 3 topics and 2 source recordings;
- slang, fast speech, Arabic/English code switching, names/entities, and narrative
  speech represented across the set;
- operator authorization recorded for every source;
- frozen raw ASR, frozen Stage 2.5 text, segment IDs/timestamps, human-heard
  reference text, and review labels for every evaluated segment;
- a `test` split never used to author prompts, thresholds, lexicon entries, or
  phrase-specific code.

The runner emits side-by-side raw, Stage 2.5, Stage 2.7, and reference text plus a
review worksheet. Human review classifies each segment as `improved`,
`unchanged_correct`, `unchanged_wrong`, `regressed`, or `hallucinated`, and scores
downstream comprehensibility from 1 through 5. Exact/normalized token distance is
reported only as a secondary signal because Egyptian spelling varies.

Completion gate:

- Stage 2.7 semantic-correct rate improves by at least 10 percentage points over
  Stage 2.5 on the frozen test split;
- at least 25% of Stage 2.5-wrong segments become `improved`;
- regression rate is at most 2% of Stage 2.5-correct segments;
- zero hallucinated facts, names, numbers, or new clauses;
- at least 98% of Stage 2.5-correct segments remain correct;
- mean comprehensibility improves and does not fall in any required category;
- the four supplied real failure families improve in fixture/provider regression
  tests without production phrase maps;
- added wall time per source minute, throughput, peak RAM, and peak VRAM are
  measured on the actual provider/model/hardware;
- all Stage 2, 2.5, and 2.6 automated suites remain green.

Performance has no fabricated pass number before measurement. Results must state
whether throughput is operationally acceptable. Any out-of-memory result, severe
latency that prevents normal use, or failed quality gate keeps Stage 2.7 open.

## Automated test strategy

Unit tests cover window bounds, source edges, Arabic normalization, weighted
phonetic spans, merge/split boundaries, entity extraction, provider schemas,
candidate limits, identity validation, protected tokens, insertion limits,
confidence thresholds, flags, fallback, and fingerprints.

Integration tests cover migration upgrade/downgrade, new stage order, forced
reprocessing, partial provider failure, transaction rollback, manual-override
priority, chunk rebuild, API/CLI contracts, UI display, and immutable raw IDs,
timestamps, word timestamps, and order.

Regression fixtures include the supplied `ديموقراطية`, `كان بيقودها`, farmer,
and `United Fruit Company` examples plus unusual valid slang that must remain
unchanged. Fake providers return plausible candidates in tests; production code
contains no matching phrase dictionary.

The real-audio benchmark is an explicit operator gate, not a network-dependent CI
test. CI validates its manifest schema, metric math, review completeness, and
gate decisions using small synthetic records.

## Documentation and completion reporting

Implementation updates `README.md`, `.env.example`, `docs/ARCHITECTURE.md`,
`docs/PIPELINE.md`, `docs/BENCHMARKS.md`, `docs/ENVIRONMENT.md`,
`docs/TROUBLESHOOTING.md`, `STATUS.md`, and `AGENTS.md` with actual commands,
settings, benchmark evidence, limits, and Stage 3 non-goals.

The final implementation report ends with exactly one of:

```text
READY FOR STAGE 3
```

or:

```text
STAGE 2.7 MUST CONTINUE
```

`READY FOR STAGE 3` is forbidden until every automated check and unseen-audio
quality gate above has current evidence.

## Research evidence

- Installed runtime inspection confirms faster-whisper 1.2.1 and CTranslate2
  4.8.2. The public faster-whisper segment includes tokens, average log
  probability, compression ratio, no-speech probability, words, and temperature:
  <https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py>
- Ollama documents JSON-schema structured outputs and OpenAI-compatible
  `response_format` support:
  <https://docs.ollama.com/capabilities/structured-outputs>
- Ollama documents its `/v1/chat/completions` compatibility:
  <https://docs.ollama.com/api/openai-compatibility>
- The reference Qwen3 family advertises 100+ languages/dialects; this supports
  evaluation, not promotion without ClipFactory's own Egyptian benchmark:
  <https://ollama.com/library/qwen3>

