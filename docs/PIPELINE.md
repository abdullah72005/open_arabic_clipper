# Stage 1 pipeline

1. An operator submits a permitted public URL or uploads a local file.
2. The API creates a durable source and queued ingest job.
3. The worker acquires the source into managed storage and records its hash.
4. The worker calls `ffprobe` with an argument array and stores typed media
   metadata.
5. The source reaches `READY_FOR_TRANSCRIPTION`.

Each stage is persisted and idempotent. Completed stages are skipped on resume;
failures retain job and pipeline error data for an operator retry. Unknown
rights are allowed through local ingest/probe only. Any future candidate
generation, rendering, or publishing must first pass an explicit authorization
policy; it must reject `UNKNOWN` rights by default.
