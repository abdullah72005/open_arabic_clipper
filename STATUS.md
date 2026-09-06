# Runtime status

Stage 2.7 extends the local-first ingest/transcription foundation through
`READY_FOR_ANALYSIS`. It prepares cached mono 16 kHz WAV audio, transcribes
locally with faster-whisper, preserves raw timestamped ASR evidence, derives
conservative contextual Egyptian Arabic correction into separate Stage 2.5
fields, then applies a bounded two-pass Stage 2.7 contextual reconstruction
through the managed local Ollama provider without altering raw text, timestamps,
word timestamps, or manual feedback. Final text priority is manual override,
then HIGH-confidence Stage 2.7, then Stage 2.5, then raw ASR. The default
reconstruction provider is local Ollama (`qwen3:8b`); a missing or invalid
provider response falls back to Stage 2.5, records a truthful unavailable
status, and the source still reaches analysis. Automatic clip selection,
rendering, publishing, and authorization remain out of scope.

Stage 2.7 has not yet passed its required private, authorized unseen-audio
benchmark. No quality, latency, RAM, VRAM, or Stage 3 readiness claim is made
until that evaluation manifest and human review are available. The known
Chernobyl diagnostic run is regression evidence only and is never counted as
unseen readiness.

## Stage 2.7 completion gate

Stage 2.7 is complete only when every item below has current, direct evidence.
A missing or indirect proof is a failed gate.

| # | Gate | Evidence | Result |
| --- | --- | --- | --- |
| 1 | Local provider health `AVAILABLE`; live worker invokes it | `python -m app.cli reconstruction-health` | PASS (digest `500a1f067a9f…b41`) |
| 2 | Provider regression tests and real audio prove multi-word repair | provider tests; benchmark comparison rows | FAIL (no model applied a reconstruction) |
| 3 | Raw ASR text and all timestamps unchanged through downstream stages | `test_reconstruction_persistence.py` deep-equality | PASS |
| 4 | Forced retranscription reruns every stale transcript-derived stage | `test_pipeline_fingerprints.py` | PASS |
| 5 | Media/audio and transcript quality separate; bad sample no longer reports high transcript quality | `test_transcript_quality.py` | PASS |
| 6 | Unavailable provider/model visible in persistence, health, API, CLI, UI | `test_reconstruction_status.py`, API/UI tests | PASS |
| 7 | Real unseen Egyptian benchmark improves materially | private unseen-audio benchmark | FAIL (no unseen-audio set) |
| 8 | Regression ≤2%, preserved-correct ≥98%, hallucinated = 0 | benchmark aggregate | FAIL (no aggregate) |
| 9 | Chernobyl first 30 seconds manually re-tested | diagnostic comparison rows | FAIL (three known phrases reviewed; models infeasible) |
| 10 | All Stage 2/2.5/2.6/2.7 backend and frontend tests pass | pytest + vitest | PASS (204 backend, 11 frontend) |
| 11 | README, STATUS, AGENTS, ENVIRONMENT, architecture, pipeline, benchmark, local setup, troubleshooting match installation | documentation | PASS |
| 12 | Final report ends with exactly one terminal status line | below | — |

Benchmark findings recorded in `docs/BENCHMARKS.md`: `qwen3:8b` is infeasible on
the 7.4 GiB machine (out-of-memory kill during load); `qwen3.5:4b` loads but
did not apply a reconstruction because the unbatched two-pass request exceeded
the model context. Neither is unseen readiness evidence.

See the task report for the latest local verification evidence. Copy
`.env.example` to `.env` before starting Compose.

STAGE 2.7 MUST CONTINUE

