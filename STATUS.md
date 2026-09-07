# Runtime status

Stage 2.7 extends the local-first ingest/transcription foundation through
`READY_FOR_ANALYSIS`. It prepares cached mono 16 kHz WAV audio, transcribes
locally with faster-whisper, preserves raw timestamped ASR evidence, derives
conservative contextual Egyptian Arabic correction into separate Stage 2.5
fields, then applies a bounded one-pass Stage 2.7 contextual reconstruction
through the managed local Ollama provider without altering raw text, timestamps,
word timestamps, or manual feedback. Final text priority is manual override,
then HIGH-confidence Stage 2.7, then Stage 2.5, then raw ASR. The default
reconstruction provider is local Ollama (`qwen3.5:4b`); a missing or invalid
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
| 1 | Local provider health `AVAILABLE`; live worker invokes it | `python -m app.cli reconstruction-health` | PASS (digest `2a654d98e6fb…eefd`) |
| 2 | Provider regression tests and real audio prove multi-word repair | provider tests; benchmark comparison rows | FAIL (provisional: one repair applied under 0.82 gate; strict unseen set still missing; pre-fix reports non-comparable) |
| 3 | Raw ASR text and all timestamps unchanged through downstream stages | `test_reconstruction_persistence.py` deep-equality | PASS |
| 4 | Forced retranscription reruns every stale transcript-derived stage | `test_pipeline_fingerprints.py` | PASS |
| 5 | Media/audio and transcript quality separate; bad sample no longer reports high transcript quality | `test_transcript_quality.py` | PASS |
| 6 | Unavailable provider/model visible in persistence, health, API, CLI, UI | `test_reconstruction_status.py`, API/UI tests | PASS |
| 7 | Real unseen Egyptian benchmark improves materially | private unseen-audio benchmark | FAIL (no unseen-audio set) |
| 8 | Regression ≤2%, preserved-correct ≥98%, hallucinated = 0 | benchmark aggregate | FAIL (no valid aggregate; pre-fix aggregates invalid evidence) |
| 9 | Chernobyl first 30 seconds manually re-tested | diagnostic comparison rows | FAIL (pre-fix reports non-comparable; must re-run on the committed evaluator and fingerprint) |
| 10 | All Stage 2/2.5/2.6/2.7 backend and frontend tests pass | pytest + vitest | PASS (204 backend, 11 frontend) |
| 11 | README, STATUS, AGENTS, ENVIRONMENT, architecture, pipeline, benchmark, local setup, troubleshooting match installation | documentation | PASS |
| 12 | Final report ends with exactly one terminal status line | below | — |

The Stage 2.7 correctness foundation was fixed and committed on 2026-09-07:
provider output is validated at a strict boundary, confidence is the one scalar
the model returns, failures are isolated per segment, prompts are budgeted over
the complete chat envelope, evaluation separates runtime status, deterministic
text comparison, literal exact comparison, and human labels, and reconstruction
fingerprints include the full runtime identity. Because the evaluator and
confidence pipeline changed, **every benchmark report produced before the
correctness-foundation commit is non-comparable**; their semantic and exact
aggregates are invalid evidence and their run IDs are historical artifacts only.

Earlier benchmark findings in `docs/BENCHMARKS.md` remain as history:
`qwen3:8b` was infeasible on the 7.4 GiB machine (out-of-memory kill during
load), the one-pass small-context protocol with `qwen3.5:4b` was feasible
end-to-end, `qwen3:4b` cannot be parsed (thinking mode exhausts the 256-token
budget), and `qwen2.5:7b` could not be pulled reliably. None of this is unseen
readiness evidence. A fresh, correctly-fingerprinted run is required before any
new model or threshold decision.

See the task report for the latest local verification evidence. Copy
`.env.example` to `.env` before starting Compose.

## Heavy-model lifecycle status (2026-09-07)

The memory and heavy-model lifecycle plan is implemented and its checkpoint
passed: a read-only `diagnose-memory` command and `scripts/diagnose-memory.sh`
label host/WSL/cgroup/process/container/swap memory; a Redis-backed
`clipfactory:heavy-model` lease serializes Whisper and Ollama across workers and
the CLI; Whisper runs in a spawned child process that is reaped before the lease
releases; Ollama unload is verified by polling `/api/ps` and recorded as
`unload_outcome`; and Celery runs `--pool=solo` (pre-fork daemonic workers
cannot spawn the child) with concurrency 1 and `PYTHONPATH=/app`.

The operator applied `[wsl2] memory=11GB swap=4GB`; Linux/Docker now report
about 10.69 GiB effective capacity. Three sequential measured trials passed:
Whisper and Ollama never overlapped, `ollama ps` was empty after each
reconstruction, unload confirmed in under a second, swap growth was effectively
zero, and no OOM occurred. This validates the lifecycle only; no model-quality
or unseen-readiness claim is made. The 8B model has not been downloaded or
benchmarked.

## Stage 2.7 model and ASR quality status (2026-09-07)

The immutable ASR capture contract and complete benchmark evidence recording were
implemented and tested. A known-regression corpus was captured on
`large-v3-turbo`. Findings:

- `qwen3:8b` (Q4_K_M) now loads without OOM under 10.69 GiB (it was OOM-killed
  at 7.44 GiB), but a full 8B replay hung, so the three-run reliability gate is
  not met.
- `qwen3.5:4b` repaired 0 of the three known phrases (`فيور 25 نوفمبر`,
  `آخره يشيلت نصر واحد`, `فيه 71`) on the frozen capture.
- Full `large-v3` recovered `فيور` (one ASR_AUDIO fix) but missed `اتناشر`;
  below the two-fix gate, so `large-v3-turbo` remains the default.
- The unseen-corpus human references are not yet available; Stage 2.7.1 cannot
  be authorized. A stricter 8B/ASR reliability and quality evaluation remains
  open.

STAGE 2.7 MUST CONTINUE

