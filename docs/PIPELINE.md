# Stage 1, Stage 2, and Stage 2.5 pipeline

1. An operator submits a permitted public URL or uploads a local file.
2. The API creates a durable source and queued ingest job.
3. The worker acquires the source into managed storage and records its hash.
4. The worker calls `ffprobe` with an argument array and stores typed media
   metadata.
5. The worker extracts cached mono 16 kHz WAV audio and runs local faster-whisper.
6. Raw source/segment text, segment ordering, timestamps, and word timestamps are persisted.
7. Stage 2.5 derives conservative contextual Egyptian correction into separate
   corrected/final fields. It never realigns audio or overwrites raw evidence.
8. Timestamp-aware chunks use final operator text when present, otherwise
   corrected text; silence/quality signals are persisted.
9. The source reaches `READY_FOR_ANALYSIS`.

Each stage is persisted and idempotent. Completed stages are skipped on resume;
failures retain job and pipeline error data for an operator retry. Unknown
rights are allowed through local ingest/probe only. Any future candidate
generation, rendering, or publishing must first pass an explicit authorization
policy; it must reject `UNKNOWN` rights by default.

Correction uses at most two neighboring segments on either side, but emits one
result per target segment only. Default operation is the local versioned lexicon.
An optional configured OpenAI-compatible local provider receives bounded batches
with stable IDs and may only approve a declared lexicon candidate; invalid,
missing, unsafe, or low-confidence output falls back to raw/lexicon text. Manual
operator text is feedback data only and does not train a model online.
