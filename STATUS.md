# Runtime status

Stage 2.5 extends the local-first ingest/transcription foundation through
`READY_FOR_ANALYSIS`. It prepares cached mono 16 kHz WAV audio, transcribes
locally with faster-whisper, preserves raw timestamped ASR evidence, derives
conservative contextual Egyptian Arabic correction into separate corrected/final
fields, persists timestamped semantic chunks, and records silence, RMS,
speech-density, speech-rate, and system confidence indicators. The default
provider is local lexicon `egyptian-ar-v1`; OpenAI-compatible local LLM use is
optional and confidence-gated. The 14-case fixture benchmark improved 3 cases,
left 11 unchanged, made 0 worse, and did not change English-only/code-switch
fixtures. Automatic clip selection, rendering, publishing, and authorization
remain out of scope. Rights and provenance are retained for future publishing
review, but never block local ingestion, transcription, correction, analysis,
or review of technically accessible public media.

See the task report for the latest local verification evidence. Copy
`.env.example` to `.env` before starting Compose.
