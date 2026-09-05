# Runtime status

Stage 2.7 extends the local-first ingest/transcription foundation through
`READY_FOR_ANALYSIS`. It prepares cached mono 16 kHz WAV audio, transcribes
locally with faster-whisper, preserves raw timestamped ASR evidence, derives
conservative contextual Egyptian Arabic correction into separate Stage 2.5
fields, then applies an optional two-pass Stage 2.7 contextual reconstruction
without altering raw text, timestamps, word timestamps, or manual feedback.
Final text priority is manual override, then HIGH-confidence Stage 2.7, then
Stage 2.5, then raw ASR. The local OpenAI-compatible reconstruction provider is
disabled by default; a missing or invalid provider response falls back to Stage
2.5 and the source still reaches analysis. Automatic clip selection, rendering,
publishing, and authorization remain out of scope.

Stage 2.7 has not yet passed its required private, authorized unseen-audio
benchmark. No quality, latency, RAM, VRAM, or Stage 3 readiness claim is made
until that evaluation manifest and human review are available.

See the task report for the latest local verification evidence. Copy
`.env.example` to `.env` before starting Compose.
