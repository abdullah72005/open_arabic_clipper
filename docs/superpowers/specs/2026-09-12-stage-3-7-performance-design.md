# Stage 3.7 Performance Optimization Design

## Goal

Reduce time to `READY_FOR_REFINEMENT` through measured, quality-neutral execution
improvements while retaining the existing whole-source INDEX transcript contract.

## Evidence and baseline

The representative, operator-provided source is
`4037a813-6fe6-4c83-96ff-e5cd4bf210ce` (`-mxx1e-P5sHE.webm`). Durable pipeline
records show: INGEST 609.426861 s, PROBE 0.213602 s, AUDIO_EXTRACTION 14.710578 s,
TRANSCRIPTION 1080.269895 s, AUDIO_ANALYSIS 7.718860 s, and
CANDIDATE_ANALYSIS 2.350317 s. It is a 1,754.9-second remote WebM with a 561 MiB
managed original and a 54 MiB cached 16 kHz WAV.

The worker exposes 22 CPUs with no cgroup CPU quota. The active container reports
7.8 GiB memory and 4 GiB swap. The installed `faster-whisper` is 1.2.1. It supports
`cpu_threads` on `WhisperModel` and exposes `BatchedInferencePipeline`, whose
defaults differ from the application contract (notably VAD), so any batch path must
pass the existing decoding values explicitly.

Historical INGEST records contain only stage duration; they cannot establish how
the 609 seconds divide between yt-dlp metadata, transfer, and postprocessing.

## Non-goals and invariants

- Do not change the model, beam size, automatic language detection, temperature
  fallback, `condition_on_previous_text`, word timestamps, or default VAD-off
  behavior.
- Do not change raw ASR/word timestamp persistence, Stage 2.5, INDEX Stage 2.7
  deferral, dialect and protected-token handling, Stage 3 candidate logic, Stage
  3.5 refinement settings, manual overrides, cancellation, child cleanup, or
  heavy-model lease behavior.
- Do not add providers, GPU work, Qwen/Gemini changes, a hardware autotuner,
  rendering, publishing, or frontend BiDi work.
- Keep managed local upload as its current one streaming write plus streaming hash
  and atomic rename. Do not add a second file copy or hash pass.
- Keep remote full-video acquisition before analysis. Audio-first and deferred
  full-video lifecycle designs are explicitly excluded.

## Durable timing design

Add an additive JSON metrics field to `PipelineRun`, with an Alembic migration.
Extend `StageExecutionResult` with an optional metrics mapping; `PipelineRunner`
persists a sanitized copy upon success and logs a structured stage summary. Existing
executors remain source-compatible because metrics default to an empty mapping.

INGEST metrics distinguish source kind, cache/reuse outcome, metadata lookup wall
time, download wall time, final filesystem artifact bytes, application
copy/storage-write time, source hashing time, and postprocessing state. Remote
artifact bytes come from `Path.stat`, never yt-dlp progress parsing. yt-dlp merge or
remux time is emitted only when a reliably observable application-requested
postprocessor ran; the current single-default-format command records it as not
applicable rather than inventing a duration.

PROBE, AUDIO_EXTRACTION, and TRANSCRIPTION execution results record their measured
wall time and their cache/reuse state. Extraction additionally reports cache
validation separately from a fresh FFmpeg extraction, making the existing WAV hash
validation visible without weakening it. Normal runner fingerprint skips remain
unchanged and do not invoke the extractor.

## Ingestion implementation boundary

`YtDlpAdapter.inspect` and `_run_download` receive independent monotonic timing.
The adapter returns acquisition metrics alongside the existing path and filename;
it retains public-host checks, the uncredentialed safe argument list, configured
size bounds, bounded diagnostics, and directory-size enforcement. No fragile
progress scraping, directory-watcher expansion, full-file remote content hash, or
additional application copy is introduced.

Local adapter metrics report the single storage write and stream-hash path. Probe
and extraction keep the storage service as the owner of paths.

## ASR execution design

Add two bounded environment settings:

- `CLIPFACTORY_WHISPER_CPU_THREADS`: `0` means CTranslate2 automatic/default
  behavior; positive values are explicitly passed to child-side model construction.
- `CLIPFACTORY_WHISPER_INDEX_BATCH_SIZE`: defaults to `1` and is bounded to the
  small experiment range. Batch size one uses the existing `WhisperModel.transcribe`
  path unchanged.

The selected execution settings, resolved device/compute type, child elapsed time,
child peak RSS, lease wait, and memory snapshots are recorded in transcription
metrics for reproducibility. Output-affecting settings remain in
`TranscriptionOptions`, persisted transcript options, and the transcription input
fingerprint. Thread count is execution-only unless validation proves it changes
output; batch size and VAD are fingerprinted because they can change segmentation
and text.

Before enabling batching, run compatibility tests against 1.2.1 for automatic
language, all existing decoding arguments, explicit `vad_filter=False`, word
timestamps, serializable result materialization, cancellation, reaping, and peak
RSS. Only test batch 2 after a safe thread result; test batch 4 only if batch 2 is
safe and materially worthwhile. Keep batch 1 if that evidence is absent or any
semantic/candidate regression appears.

## Benchmark and acceptance method

Never overwrite or force-rerun the representative source's persisted baseline.
Use its cached WAV with a narrow benchmark path that exports an isolated transcript
snapshot, applies existing normalization and deterministic Stage 3 analysis in
memory or isolated benchmark persistence, and compares it with the durable
baseline. Capture elapsed wall time, audio-minutes per wall-minute, child peak RSS,
memory/swap/CPU evidence, timestamp validity, transcript confidence/evidence,
Arabic/English/protected-token/code-switch differences, and candidate recall/order.

Run only the specified progressive ladder: existing baseline A, measured thread
variant B, then batch 2 C only if compatible, batch 4 D only if justified, and one
conservative VAD E only if prior variants are inadequate. Stop when a safe gain is
demonstrated. Select the fastest semantic-acceptance result; do not treat a shorter
time as sufficient by itself.

## Tests and documentation

Add deterministic focused tests before implementation for metric persistence,
remote timing boundaries, safe cache reporting, local upload non-duplication,
settings/options/fingerprint semantics, default compatibility, batch adapter
argument preservation if implemented, cancellation/reaping, lease behavior, peak
RSS, Arabic-English preservation, and deterministic Stage 3 replay compatibility.

Update `STATUS.md`, `.env.example`, environment documentation, and
`docs/BENCHMARKS.md` with actual test evidence. The final benchmark report must
state actual hardware/resources, the historical baseline, every configuration
actually tested, rollback settings, rejection reasons, and that full remote video
continues to be downloaded early.
