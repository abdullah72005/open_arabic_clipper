# Stage 2.5 Egyptian Arabic Transcript Correction Design

## Scope

Stage 2.5 adds conservative contextual text correction after faster-whisper and
before downstream analysis. It improves Egyptian Arabic ASR errors while keeping
raw ASR evidence, segment identity, ordering, timestamps, English terms, and
speaker meaning intact. It does not perform alignment, model fine-tuning, clip
selection, rendering, publishing, or Stage 3 candidate discovery.

## Existing baseline

The current worker uses faster-whisper 1.2.1 with the `small` model, automatic
CPU/CUDA selection, CPU `int8`, CUDA `float16`, beam size 5, automatic language
detection, word timestamps enabled, faster-whisper's default temperature
fallback, previous-text conditioning enabled by library default, and VAD,
initial prompt, and hotwords disabled. Stage 2 stores raw source text, timestamped
segment JSON, word timestamps, detected language, and conservative whitespace
normalization.

The untouched Stage 2 regression selection passes 44 tests. Current
normalization leaves all three supplied Egyptian failures unchanged, so the
known-example baseline is 0/3 corrected.

## Architecture

The pipeline remains:

```text
audio -> faster-whisper -> immutable raw segments
      -> contextual Egyptian correction -> corrected segment text
      -> timestamp-aware chunks and Stage 2 analysis
```

Correction is a separate operation within transcript normalization. A focused
`app.transcription.correction` package owns Arabic similarity, lexicon loading,
provider contracts, prompt construction, output validation, confidence gating,
and fixture benchmarking. The pipeline executor only supplies ordered segments,
persists returned annotations, rebuilds chunks from final display text, and
records aggregate confidence indicators.

## Persistence model

Existing `Transcript.raw_text`, raw segment `text`, raw segment `start`/`end`,
and `word_segments` remain authoritative and are never overwritten.

Each segment JSON gains:

- `raw_text`: exact raw segment `text` copied explicitly for stable consumers
- `corrected_text`: accepted automatic output or raw text after fallback
- `correction_applied`: whether accepted text differs from raw text
- `correction_confidence`: bounded system confidence from 0 through 1
- `correction_method`: `unchanged`, `lexicon`, `llm`, `llm+lexicon`, or
  `operator`
- `correction_version`: correction configuration/version string
- `correction_changes`: validated change explanations
- `operator_text`: nullable operator override retained as evaluation feedback
- `final_text`: operator text when present, otherwise corrected text

Transcript columns add corrected/final text and system indicators:

- `corrected_text`
- `final_text`
- `raw_transcript_confidence`
- `correction_confidence`
- `corrected_segment_ratio`
- `uncertain_segment_ratio`
- `correction_method`
- `correction_version`

These values are confidence signals, not claims of ground-truth accuracy.
An Alembic migration backfills corrected/final text from normalized text and
uses conservative zero/default indicators for old rows.

## Lexicon and candidate generation

`app/transcription/lexicons/egyptian_ar.json` stores canonical colloquial
phrases, known ASR confusions, priority, and notes. Entries cover supplied
regressions plus broader Egyptian fillers, connected speech, slang, negation,
questions, football, and technical vocabulary. Application logic does not embed
a replacement table.

Candidate generation normalizes Unicode, whitespace, punctuation, Arabic
diacritics, Alef variants, Yeh/Alef Maqsura, and context-safe Teh Marbuta forms
for comparison only. It uses character edit similarity and declared confusion
hints. Original spelling remains available for output. A candidate alone never
authorizes replacement.

Deterministic automatic correction requires either an exact declared confusion
or a high similarity score plus a small edit. Names, numeric tokens, and Latin
tokens are protected. Large changes require stronger contextual/provider
evidence. Ambiguous candidates remain unchanged.

## Context, batching, and provider

The correction engine processes bounded batches, preserving integer segment
indexes. Every target receives up to two preceding and two following segments.
Only the target segment's text may be emitted. Provider responses must contain
the exact requested IDs once each and cannot add, remove, merge, or reorder
segments.

No existing `LLMProvider` exists. Stage 2.5 introduces a small
`CorrectionProvider` protocol and an OpenAI-compatible HTTP implementation that
works with local endpoints such as Ollama. Runtime settings control endpoint,
model, API key, timeout, batch size, and enablement. The default remains the
deterministic local corrector so ClipFactory works without network access or a
new required service. A configured provider augments candidate decisions; it
does not bypass deterministic safety checks.

The provider prompt includes these exact constraints:

> You are correcting speech-recognition errors in an Egyptian Arabic transcript.
> Preserve the speaker's exact meaning and dialect. Make only changes strongly
> justified by phonetics and surrounding context. Do not translate. Do not
> formalize. Do not summarize. Do not add information. Preserve
> English/code-switched words. If the raw text is already plausible, return it
> unchanged.

Output uses strict JSON with segment IDs, corrected text, changed flag,
confidence, and structured changes. Schema, types, ranges, IDs, protected
tokens, length growth, numbers, and names are validated. Any transport error,
invalid JSON, missing/duplicate ID, unsafe edit, or low-confidence result falls
back to raw text.

## Confidence policy

Thresholds are configuration-driven:

- high confidence: apply an allowed correction
- medium confidence: apply only a small edit or exact lexicon confusion with
  context support
- low confidence: preserve raw text and mark segment uncertain

System confidence combines provider confidence, candidate similarity, lexicon
priority, edit size, protected-token checks, and context support. Provider
self-confidence is never sufficient alone.

## Whisper review

faster-whisper 1.2.1 supports temperature fallback, VAD, previous-text
conditioning, initial prompts, and hotwords. Current defaults are retained
because no controlled authorized audio corpus proves a safer global change.
Stage 2.5 exposes output-affecting `condition_on_previous_text`, `vad_filter`,
`initial_prompt`, and `hotwords` settings and includes them in cache
fingerprints. Prompt/hotword support therefore exists for controlled operator
benchmarks without silently biasing English or code-switched sources.

## API and operator UI

Existing transcript fields and routes remain compatible. Transcript responses
add corrected/final text and aggregate indicators. Segment search uses
`final_text` while raw text remains inspectable.

The source detail view shows final corrected text by default and offers a debug
comparison for changed segments: raw, automatic correction, confidence, method,
and operator override. A narrow segment-edit endpoint validates and persists
manual text without changing raw/automatic output or timestamps. Clearing the
override restores automatic text. This stores feedback only; it does not train
models online.

## Benchmark and tests

A versioned JSON fixture corpus contains the three supplied failures and
realistic cases for filler words, connected speech, slang, negation, questions,
football, business/technical language, Arabic-English code switching, names,
numbers, already-correct Arabic, and English-only speech.

The same fixture runner records baseline normalization and Stage 2.5 results:
improved, unchanged, worsened, automatic correction rate, uncertain rate,
latency, peak memory, exact-reference score, normalized token edit distance,
and manual semantic-correctness labels. The report explicitly notes dialect
spelling limits of WER-like metrics.

Tests cover:

- Arabic normalization and near-match scoring
- lexicon parsing and conservative candidate generation
- context windows and bounded provider batches
- strict provider response validation and raw fallback
- confidence thresholds, protected English/names/numbers, and low confidence
- supplied regressions and diverse fixture categories
- immutable timestamps, segment IDs/order, and raw evidence
- migration/backfill, idempotence, retry/recovery, API compatibility, manual
  overrides, UI comparison, English-only behavior, and code switching
- all existing Stage 2 backend/frontend checks

## Failure handling and recovery

Correction is deterministic for a fixed raw transcript, lexicon version,
provider model, prompt version, and thresholds. These values contribute to a
correction fingerprint/version. Re-running normalization replaces derived
automatic annotations and chunks atomically while preserving operator overrides.
Provider failures degrade to raw text instead of failing transcription. Database
errors retain normal pipeline retry semantics.

## Completion evidence

Completion requires passing backend and frontend suites, migration upgrade and
downgrade checks, fixture benchmark improvement without English/code-switch
regression, timestamp/order/raw-preservation assertions, documented before/after
Whisper settings, measured correction latency/memory, updated `AGENTS.md`,
`STATUS.md`, README/environment docs when commands or settings change, and a
requirement-by-requirement audit. Stage 3 remains untouched.
