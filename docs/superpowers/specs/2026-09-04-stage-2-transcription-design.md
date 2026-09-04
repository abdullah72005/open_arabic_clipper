# Stage 2 Transcription Design

## Purpose

Extend ClipFactory's durable ingest/probe foundation into a local-first,
timestamp-accurate transcription pipeline ready for later clip analysis. The
system must handle Egyptian Arabic, Modern Standard Arabic, English, and mixed
Arabic/English speech without a paid API.

## Pipeline and orchestration

The persisted lifecycle becomes:

`INGEST -> PROBE -> AUDIO_EXTRACTION -> TRANSCRIPTION -> TRANSCRIPT_NORMALIZATION -> AUDIO_ANALYSIS -> READY_FOR_ANALYSIS`.

Each stage keeps the existing `PipelineRun` and `ProcessingJob` semantics:
stages are idempotent, completed work is skipped, failures persist a usable
error, and retry or worker restart resumes from the first incomplete stage.
Transcription work runs only in Celery's `transcription` queue with default
concurrency one, preventing simultaneous loading of large Whisper models. The
existing `media` queue handles extraction and inexpensive FFmpeg analysis.

Source ownership and authorization remain unchanged. Stage 2 gathers evidence
only; it does not select clips, render, publish, or make authorization
decisions.

## Storage and data model

`StorageService` remains the only owner of filesystem paths. Per-source audio
is an atomically written mono, 16 kHz PCM WAV artifact. Its persisted analysis
record stores relative output path, SHA-256 hash, sample rate, duration, and
the source content hash used to produce it. A valid matching artifact is
reused; source changes invalidate it.

`Transcript` has one current, reusable result per source and stores:

- source ID, detected or forced language, nullable language probability;
- Whisper model and material transcription options plus a deterministic input
  fingerprint;
- raw text and conservative display-normalized text;
- duration, processing duration, and created/updated times;
- timestamped segment and optional word data, retaining raw Whisper values.

Every stored segment has start/end seconds, text, `avg_logprob`,
`no_speech_prob`, and words when available. Word entries retain start/end,
word, and probability when provided. `TranscriptChunk` records reusable
analysis windows with times, text, segment references, and bounded preceding
and following context.

`AudioAnalysis` stores silence intervals, fixed-window RMS/energy statistics,
silence ratio, long pauses, energy changes, speech density, and approximate
speech rate. `SourceQualityAssessment` stores transcript/language confidence,
speech density, silence ratio, audio-quality and repetition scores, nullable
visual/candidate-density scores, an overall score, machine-readable reasons,
and version. It is advisory only and never skips a source.

## Transcription service

`faster-whisper` is wrapped behind a narrow service interface so normal tests
use deterministic fakes and do not download models. Configuration exposes
model, device, compute type, beam size, optional forced language, and queue
concurrency. Accepted model identifiers are `tiny`, `base`, `small`, `medium`,
and `large-v3` when supported by installed faster-whisper.

Default policy is `small`: CPU uses `int8`; a detected usable NVIDIA CUDA
device uses `float16`. Device and compute type remain explicit configuration
overrides. Auto-detection is the default for every source, including mixed
Arabic/English speech. Forced language is an opt-in operator override. The
detected language and confidence are persisted, while source text remains
untouched rather than being transliterated or translated.

At worker startup, hardware selection is safe: CUDA availability is probed;
failure or an unsupported configuration falls back to CPU rather than failing
the API process. Worker errors for missing audio streams, invalid WAV output,
insufficient disk capacity, unavailable model runtime, and transcription
failure are structured and retryable only when retry can help.

The first supported local run records a benchmark with source audio duration,
wall time, real-time factor, audio minutes per wall-clock minute, model,
device, and compute type. A practical small-model benchmark is required on
this CPU-only environment; documentation must state the actual measurement
and that quality validation requires authorized Arabic material.

## Cache and normalization

Transcript reuse is keyed by source content hash, audio artifact hash, Whisper
model, forced-language setting, beam size, word-timestamp setting, device-
independent material options, and normalization/analysis versions. Styling or
other later presentation settings cannot invalidate it. `retranscribe` creates
a new transcription job and may specify a model; `force_retranscribe` bypasses
the matching transcript cache but preserves prior durable job history.

Normalization is display-only and conservative: collapse repeated whitespace,
normalize line endings and punctuation spacing, remove strictly adjacent
duplicate ASR artifacts, and apply only safe Arabic Unicode canonicalization.
It never formalizes Egyptian Arabic, translates, changes wording, or damages
embedded English, numbers, names, or punctuation. Raw and normalized text are
both returned by the API.

## Analysis and chunking

FFmpeg `silencedetect` output is parsed into start/end/duration intervals and
cached against the audio hash. Fixed time windows calculate inexpensive
RMS/volume and energy deltas. Transcript timestamps derive speech density and
approximate speech rate; segment gaps derive long pauses. No neural audio or
visual scoring is added in Stage 2.

Chunking chooses sentence/segment boundaries closest to configurable target
duration and text size, never splits at arbitrary characters. A chunk includes
its segment references and a bounded textual context before and after it. Long
sources are processed incrementally so a 4-hour source does not require a
single in-memory transcript-analysis payload.

## API, CLI, and UI

API endpoints are:

- `GET /api/sources/{id}/transcript`
- `GET /api/sources/{id}/transcript/segments`
- `GET /api/sources/{id}/transcript/search?q=`
- `POST /api/sources/{id}/retranscribe`

Responses use UTF-8 and include timestamps. Search is case-insensitive for
Latin text and Unicode-aware for Arabic text, returning matching segment
timestamps. The existing route prefix conventions are retained when mapping
these paths into FastAPI.

CLI commands are `clipfactory transcribe SOURCE_ID`, `clipfactory transcript
SOURCE_ID`, and `clipfactory retranscribe SOURCE_ID --model medium`.

The source-detail UI displays pipeline progress, detected language/confidence,
model, processing duration, searchable transcript text, and timestamped
segments. Arabic or mixed text gets `dir="auto"` with appropriate bidi-safe
containers. A timestamp seeks an available source-video player; absence of a
browser-playable source does not break transcript access.

## Error handling and observability

Jobs report honest stage-level progress: extracting audio, estimated
transcription progress based on completed audio duration, normalizing,
analyzing silence, and ready for analysis. Exact Whisper percentage is never
claimed when unavailable. Structured logs include source/job IDs, stage,
model, device, timing, cache decision, and retryability; they never log full
private transcript text by default.

## Testing and validation

Test-first coverage includes audio extraction/cache/missing-stream handling,
Whisper serialization, Arabic Unicode normalization, mixed Arabic/English
preservation and auto-detection, cache fingerprints, force retranscription,
silence parsing, audio features, chunking, quality persistence, state
transitions, retry/restart recovery, API search, CLI dispatch, and RTL UI
rendering. Fixtures include deterministic Egyptian Arabic and mixed-language
transcript data. Unit tests never download multi-gigabyte models.

An optional, documented integration marker runs faster-whisper against a
legitimate small local audio/video fixture when the model runtime is available.
It records actual throughput and validates stored timestamps, language, and
`READY_FOR_ANALYSIS`. Documentation explains how operators can run the same
validation with authorized Egyptian Arabic material.

## Constraints

- Python 3.12, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL, Redis, Celery,
  Next.js, TypeScript, Tailwind, FFmpeg/ffprobe, and local faster-whisper.
- No paid transcription API, GPU requirement, DRM/login/paywall/CAPTCHA bypass,
  automatic authorization, candidate selection, rendering, or publishing.
- New observable behavior follows TDD; docs and operational configuration are
  updated with implementation.
