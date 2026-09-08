# Stage 2.7 Model and ASR Quality Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the current 4B reconstructor with one practical 8B Qwen candidate on immutable ASR, separately test full Whisper `large-v3` against `large-v3-turbo` for audio-dependent errors, and promote only a configuration that wins human-reviewed quality gates without memory unreliability.

**Architecture:** Split audio decoding from text reconstruction. Capture immutable ASR once per ASR configuration, replay the exact capture through each reconstruction model, and retain every candidate and rejection reason. Human-reviewed rows determine quality; automated metrics support diagnosis. Reconstruction and ASR promotions have independent gates.

**Tech Stack:** faster-whisper/CTranslate2 CPU `int8`, Ollama OpenAI-compatible API, qwen3.5:4b, qwen3:8b practical quantization, JSONL benchmark artifacts, pytest, Ruff, mypy, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-09-07-stage-2-7-accuracy-runtime-recovery-design.md`

## Prerequisites and Constraints

- Start only after both earlier plans are reviewed and the three-run lifecycle checkpoint passes with effective capacity at least 10 GiB.
- Use only authorized private audio through the storage service. Never commit audio, raw private transcripts, or human-reference text.
- Do not alter confidence thresholds, score weights, validation rules, prompts, Stage 2.5 rules, or lexicons during an A/B run.
- Do not hard-code any known phrase. Known phrases are test observations and human references only.
- Compare only current `qwen3.5:4b` and `qwen3:8b` initially. Do not download a model catalog.
- Never run two models concurrently. Acquire the heavy-model lease for every ASR and reconstruction run.
- Quality decisions require human review. A normalized string heuristic cannot label hallucinations.
- Failed/OOM runs remain in the report; never discard them from reliability statistics.

---

## Task 1: Add an immutable ASR capture contract

**Files:**

- Create: `backend/app/transcription/reconstruction/capture.py`
- Modify: `backend/app/transcription/reconstruction/benchmark.py`
- Modify: `backend/app/cli.py`
- Test: `backend/tests/test_reconstruction_capture.py`
- Test: `backend/tests/test_reconstruction_audio_benchmark.py`
- Test: `backend/tests/test_cli.py`

- [ ] Define a versioned `ASRCapture` schema containing: capture ID, authorized storage-relative source identity/hash, clip boundaries, Whisper model, faster-whisper and CTranslate2 versions, device/compute type, all decoding options, language, raw segments, segment/word timestamps, acoustic evidence, wall time, and memory snapshots.
- [ ] Add validation tests for stable IDs, ordered/non-overlapping timestamps, immutable raw text, complete decoder identity, source mismatch, and schema-version rejection.
- [ ] Add `benchmark-reconstruction --capture-asr <manifest>` and `--from-capture <capture>` modes. Capture mode invokes Whisper exactly once and writes through the storage service; replay mode must never construct a transcriber.
- [ ] Compute the capture hash from canonical schema content. Preserve each replay as a new run referencing the capture hash; do not modify the capture file.
- [ ] Store raw private captures under the existing ignored benchmark storage tree. Add a test or repository check proving fixture/example paths do not accidentally commit private artifacts.
- [ ] Run focused tests.
- [ ] Commit: `Add immutable ASR benchmark captures`

## Task 2: Record complete model, prompt, memory, and decision evidence

**Files:**

- Modify: `backend/app/transcription/reconstruction/benchmark.py`
- Modify: `backend/app/transcription/reconstruction/types.py`
- Modify: `backend/app/cli.py`
- Test: `backend/tests/test_reconstruction_audio_benchmark.py`
- Test: `backend/tests/test_cli.py`

- [ ] Extend each run report with configured model, live digest, Ollama quantization/size metadata, load success/failure, load time, reconstruction wall time, average/max serialized prompt bytes, average/max estimated input tokens, peak effective/system/container/process RAM where measurable, peak swap, OOM evidence, unload confirmation, and unresolved rate.
- [ ] Extend each row with raw, Stage 2.5, candidate, candidate confidence, deterministic validation outcome, apply/reject reason, Stage 2.7, reference-present flag, human correctness label, human safety label, and root-cause taxonomy.
- [ ] Add invariant tests: row status totals equal speech-row count; all referenced rows are included; accepted plus rejected candidates equals candidates returned; exact metrics never consume human labels; missing metrics are explicit `null` plus reason rather than zero.
- [ ] Update CLI summary to print the decision-driving human metrics first, followed by operational metrics.
- [ ] Run focused tests and inspect a synthetic JSONL report.
- [ ] Commit: `Record complete Stage 2.7 benchmark evidence`

## Task 3: Build and freeze the reviewed corpus

**Files:**

- Modify locally only: authorized manifest/reference files under `storage/benchmarks/stage-2-7/`
- Modify: `docs/BENCHMARKS.md` with aggregate/redacted metadata only

- [ ] Locate the same problematic real clip and reviewed timestamps for:

  - raw `فيور 25 نوفمبر` versus spoken `في يوم 25 نوفمبر`;
  - raw `آخره يشيلة نصر واحد` versus spoken `آخره يشيل اتناشر واحد`;
  - the reviewed raw `71` versus spoken phrase around `70 واحد`;
  - the three existing Chernobyl connected-speech failures.

- [ ] Add at least one separate unseen authorized Egyptian clip. The operator must manually reference every evaluated speech row and assign correctness/safety labels. If no unseen clip or reference is available, stop and report `STAGE 2.7 MUST CONTINUE`; do not invent it.
- [ ] Assign exactly one root cause to each baseline mismatch: `ASR_AUDIO`, `STAGE25`, `MODEL_CANDIDATE`, `VALIDATION_OR_GATE`, `PERSISTENCE`, or `EVALUATOR`.
- [ ] Capture one `large-v3-turbo` baseline with the current decoder settings. Hash it and make it read-only for reconstruction comparison.
- [ ] Run a dry report that proves complete reference coverage and no unknown labels before loading any LLM.
- [ ] Commit only redacted corpus metadata/documentation: `Freeze Stage 2.7 reviewed corpus`

## Task 4: Establish the 4B baseline on the frozen capture

**Files:**

- Modify: `docs/BENCHMARKS.md`
- Modify: `STATUS.md`

- [ ] Confirm `ollama show qwen3.5:4b` and record exact digest and quantization. Confirm memory preflight and no resident Whisper/Ollama model.
- [ ] Replay the same frozen capture three sequential times using the unchanged one-pass prompt and confidence policy. Acquire/release the heavy-model lease for each run.
- [ ] Record all metrics required by the spec, including accepted corrections and correct candidates rejected by validation/gate.
- [ ] Verify deterministic outputs or explain differences. Do not average away an OOM, provider failure, or unload warning.
- [ ] Have the operator review changed outputs. Store private row labels with artifacts and only aggregate/redacted conclusions in docs.
- [ ] Commit: `Record frozen 4B reconstruction baseline`

## Task 5: Run one practical 8B reconstruction candidate

**Files:**

- Modify: `.env.example` only if the candidate is ultimately promoted
- Modify: `backend/app/core/settings.py` only if the candidate is ultimately promoted
- Modify: `docs/BENCHMARKS.md`
- Modify: `STATUS.md`

- [ ] After preflight passes, inspect/install `qwen3:8b` through Ollama. Record exact digest, storage size, and quantization reported by Ollama. Prefer the normal practical Ollama quantization that fits; do not silently substitute an unrecorded tag.
- [ ] Run one smoke request through the existing OpenAI-compatible provider with the production prompt budget. On load failure/OOM, collect kernel/container/Ollama evidence, verify unload, and retry only after confirming no stale model and no concurrent heavy task.
- [ ] If the smoke succeeds, replay the exact frozen capture three sequential times. Do not rerun Whisper and do not alter prompt/gate settings between 4B and 8B.
- [ ] Compare human-reviewed results using these reconstruction promotion gates:

  - at least two additional correct text-recoverable multi-word repairs over 4B;
  - at least 50% of reviewed `MODEL_CANDIDATE` rows improve;
  - zero new factual/name/number hallucinations;
  - zero meaningful regressions;
  - all three runs complete without OOM, stale residency, or swap growth above 1 GiB per run;
  - every requested output is valid or cleanly contained as per-segment fallback.

- [ ] Promote the 8B model default only if every gate passes. Otherwise retain `qwen3.5:4b` and explicitly document whether the reason was quality, reliability, or both.
- [ ] Do not try unrelated models in this task. If `qwen3:8b` cannot be installed because of a transient network failure, record that as incomplete evidence rather than a hardware ceiling.
- [ ] Commit: `Benchmark 8B contextual reconstruction`

## Task 6: A/B full `large-v3` against `large-v3-turbo`

**Files:**

- Modify: `.env.example` only if full `large-v3` is promoted
- Modify: `backend/app/core/settings.py` only if full `large-v3` is promoted
- Modify: `docs/BENCHMARKS.md`
- Modify: `STATUS.md`

- [ ] Select only rows classified `ASR_AUDIO` plus their 5–15 second local context windows. Keep source audio, timestamps, language, beam size, VAD behavior, temperature policy, CPU device, and `int8` compute type identical. The only intended variable is Whisper model ID.
- [ ] Run turbo first, release it, verify worker exit/memory reclamation, then run full `large-v3`. Never overlap either ASR model with Ollama.
- [ ] Have the operator compare raw outputs to audio/reference. Record protected-number/name correctness, connected-speech repairs, regressions, WER/CER, load time, decode wall time, peak RAM/swap, and OOM/unload results. Treat WER/CER as supporting metrics, not a substitute for row review.
- [ ] Only if full `large-v3` wins the targeted screen, capture the full known and unseen clips once with it and compare downstream Stage 2.5 plus the selected reconstruction model.
- [ ] Promote full `large-v3` only when it:

  - fixes at least two additional human-confirmed `ASR_AUDIO` errors across known and unseen sets;
  - introduces zero new protected-number/name, factual, or clause regressions;
  - makes downstream Stage 2.5/2.7 reviewed accuracy no worse;
  - completes three sequential full-clip runs without OOM, stale residency, or unbounded swap;
  - has its measured latency explicitly accepted as a quality-first tradeoff.

- [ ] If it fails any gate, keep `large-v3-turbo` as default. Optionally document full `large-v3` as a future targeted uncertain-span mode; do not implement a new routing system in this task.
- [ ] Commit: `Compare full and turbo Whisper models`

## Task 7: Final selection, verification, and honest terminal state

**Files:**

- Modify: `docs/ENVIRONMENT.md`
- Modify: `docs/BENCHMARKS.md`
- Modify: `docs/STAGE_2_7_OPERATIONS.md`
- Modify: `STATUS.md`
- Modify: `AGENTS.md` if operational defaults or lifecycle architecture changed materially

- [ ] Run one end-to-end authorized clip with the selected ASR and reconstruction defaults through persisted segment and transcript output. Confirm raw/timestamps unchanged and final-text priority intact.
- [ ] Run:

  ```bash
  docker compose run --rm --no-deps backend sh -lc \
    "pytest -q tests/test_reconstruction_*.py tests/test_runtime_memory.py tests/test_heavy_model_lease.py \
     tests/test_transcription.py tests/test_pipeline.py tests/test_pipeline_fingerprints.py \
     tests/test_stage2_pipeline_e2e.py && \
     ruff format --check app tests && ruff check app tests && \
     mypy app/runtime app/transcription app/pipeline/stages.py app/workers"
  ```

- [ ] Report exact test counts, branch, commits, and clean/dirty worktree. List every remaining reviewed transcript error and its root-cause category.
- [ ] Completion outcome A: if a stronger 8B model is reliable and materially better with no meaningful safety loss, update its operational default and end the final report exactly `READY FOR STAGE 2.7.1` only if the unseen corpus and all Stage 2.7 gates also pass.
- [ ] Completion outcome B: if corrected memory, sequential unloading, and practical quantization conclusively fail three-run reliability, keep 4B, document the hardware ceiling with measurements, and end exactly `STAGE 2.7 MUST CONTINUE`.
- [ ] If evidence is incomplete, network installation fails, human references are missing, or quality remains below gate, end exactly `STAGE 2.7 MUST CONTINUE`. Never label incomplete evidence a hardware ceiling.
- [ ] Document next options without implementing them: existing hosted-provider abstraction, targeted uncertain-span retranscription, and further full `large-v3` evaluation.
- [ ] Commit: `Finalize Stage 2.7 runtime and model evidence`

