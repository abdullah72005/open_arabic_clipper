# Runtime status

Stage 2 extends the local-first ingest and metadata-probing foundation through
`READY_FOR_ANALYSIS`. It prepares cached mono 16 kHz WAV audio, transcribes
locally with faster-whisper, conservatively normalizes Arabic/English/mixed text,
persists timestamped semantic chunks, and records silence, RMS, speech-density,
speech-rate, and source-quality metadata. Automatic clip selection, rendering,
publishing, and authorization remain out of scope.

See the task report for the latest local verification evidence. Copy
`.env.example` to `.env` before starting Compose.
