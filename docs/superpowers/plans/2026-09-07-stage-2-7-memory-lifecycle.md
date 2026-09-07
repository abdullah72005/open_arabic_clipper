# Stage 2.7 Memory and Heavy-Model Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain the 7.4 GiB runtime limit with measurements, safely expose about 11 GiB when the operator applies host configuration, and guarantee that Whisper and Ollama reconstruction models cannot be resident concurrently.

**Architecture:** Add one read-only memory snapshot abstraction and one Redis-backed cross-process heavy-model lease. Use recyclable Celery children as the hard CTranslate2 reclamation boundary, explicit Ollama unload plus bounded verification, and preflight gates for 8B experiments. Host-side WSL changes remain operator-controlled.

**Tech Stack:** Python 3.12, Linux `/proc` and cgroup v2, Redis, Celery, Docker Compose, Ollama HTTP/API, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-07-stage-2-7-accuracy-runtime-recovery-design.md`

## Prerequisite and Constraints

- Begin only after the correctness-foundation plan is reviewed and its test checkpoint passes.
- Do not download or benchmark an 8B model in this plan.
- Never automatically edit `%UserProfile%/.wslconfig`, restart Docker Desktop, run `wsl --shutdown`, or allocate all 16 GB of host RAM.
- Never infer host RAM from Linux `/proc/meminfo`; label host, WSL VM, cgroup, process, container, Ollama, and swap measurements separately.
- Do not use a process-local mutex as the only exclusion mechanism. Worker and CLI processes must coordinate through Redis.
- All normal lease-release and Ollama-unload paths must be in `finally` blocks and
  tested. A Whisper lease cannot be released until its spawned model subprocess is
  reaped; an unsafe unload timeout must block another start rather than release early.
- Use TDD and one task-level commit at a time.

---

## Task 1: Capture and document the actual memory limit cause

**Files:**

- Create: `scripts/diagnose-memory.sh`
- Modify: `docs/ENVIRONMENT.md`
- Modify: `docs/STAGE_2_7_OPERATIONS.md`

- [ ] Collect read-only Linux evidence:

  ```bash
  free -h
  swapon --show --bytes
  sed -n '1,25p' /proc/meminfo
  test -f /sys/fs/cgroup/memory.max && cat /sys/fs/cgroup/memory.max
  test -f /sys/fs/cgroup/memory.current && cat /sys/fs/cgroup/memory.current
  test -f /sys/fs/cgroup/memory.peak && cat /sys/fs/cgroup/memory.peak
  docker info --format '{{json .MemTotal}}'
  docker inspect clipfactory-worker --format '{{json .HostConfig.Memory}}' 2>/dev/null || true
  docker stats --no-stream
  ollama ps
  ```

- [ ] Collect host-side evidence by giving the operator these PowerShell commands; do not fabricate results if the agent cannot run them:

  ```powershell
  Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
  Get-Content "$env:USERPROFILE\.wslconfig" -ErrorAction SilentlyContinue
  wsl --status
  wsl -d Ubuntu -- free -h
  ```

- [ ] Determine the narrowest active ceiling: explicit `.wslconfig`, Docker Desktop VM setting, container cgroup limit, or observed WSL dynamic allocation. Record raw values and the conclusion in `docs/ENVIRONMENT.md`. If Windows evidence is unavailable, label the cause unresolved and stop; do not state a guess as fact.
- [ ] Add `scripts/diagnose-memory.sh` as a read-only, `set -eu` script that prints labeled Linux/cgroup/Docker/Ollama facts and tolerates missing optional files. Do not parse localized human-readable output into control decisions.
- [ ] Document the recommended host configuration only when the measured cause is WSL/Docker capacity:

  ```ini
  [wsl2]
  memory=11GB
  swap=4GB
  ```

  Explain that the operator writes `%UserProfile%\.wslconfig`, runs `wsl --shutdown`, restarts Docker Desktop, and reruns diagnostics. Document 12 GB only as an optional measured choice when Windows remains healthy.
- [ ] Commit: `Document Stage 2.7 memory ceiling`

## Task 2: Add a testable runtime memory snapshot

**Files:**

- Create: `backend/app/runtime/__init__.py`
- Create: `backend/app/runtime/memory.py`
- Create: `backend/tests/test_runtime_memory.py`
- Modify: `backend/app/cli.py`
- Test: `backend/tests/test_cli.py`

- [ ] Write tests using a temporary fake proc/cgroup tree for finite cgroup limit, `max`, missing files, cgroup v1 fallback if supported, malformed optional values, swap totals, process RSS, and effective capacity `min(linux_total, finite_cgroup_limit)`.
- [ ] Define immutable `MemorySnapshot` fields: timestamp, Linux total/available bytes, swap total/free bytes, cgroup limit/current/peak bytes or `None`, process RSS bytes, and effective capacity bytes. Do not call current usage “peak.”
- [ ] Implement pure parsers separated from filesystem access so tests do not depend on the developer machine.
- [ ] Add `clipfactory diagnose-memory --json` that emits machine-readable fields and a human mode with GiB labels. It is read-only and exits nonzero only when required Linux inputs cannot be read.
- [ ] Add an `assert_8b_preflight(snapshot, minimum_bytes=10*1024**3)` helper returning an actionable error that names the measured effective capacity and points to `docs/ENVIRONMENT.md`.
- [ ] Run:

  ```bash
  docker compose run --rm --no-deps backend pytest -q tests/test_runtime_memory.py tests/test_cli.py
  ```

- [ ] Commit: `Add runtime memory diagnostics`

## Task 3: Serialize heavy models across workers and CLI commands

**Files:**

- Create: `backend/app/runtime/heavy_model_lease.py`
- Create: `backend/tests/test_heavy_model_lease.py`
- Modify: `backend/app/core/settings.py`

- [ ] Write fake-Redis tests for acquire success, busy timeout, unique owner token, TTL renewal, ownership-checked Lua release, expired lease, and release after an exception.
- [ ] Define `HeavyModelLeaseBusy` as retryable and include owner-purpose metadata without secrets. Use one key: `clipfactory:heavy-model`.
- [ ] Implement a context manager with `SET key token NX PX ttl`, a bounded acquisition deadline, renewal before TTL expiry for long runs, and an atomic compare-and-delete release script. Never delete a lease owned by another token.
- [ ] Add bounded settings for lease TTL, renewal interval, and blocking timeout. Require TTL to exceed two renewal intervals.
- [ ] Rerun the focused tests, Ruff, and mypy for the new module.
- [ ] Commit: `Serialize heavy model usage with Redis`

## Task 4: Enforce lease boundaries in durable pipeline stages

**Files:**

- Modify: `backend/app/workers/tasks.py`
- Modify: `backend/app/pipeline/stages.py`
- Modify: `backend/app/cli.py`
- Test: `backend/tests/test_pipeline.py`
- Test: `backend/tests/test_cli.py`
- Test: `backend/tests/test_stage2_pipeline_e2e.py`

- [ ] Add integration tests proving the transcription stage holds the heavy-model lease from before the isolated Whisper subprocess starts until after that subprocess is reaped, and releases it before contextual reconstruction can begin.
- [ ] Add tests proving reconstruction holds the same lease around Ollama health/inference/release, and a benchmark CLI cannot enter while a worker fake owns it.
- [ ] Inject the lease factory into executors/commands; do not instantiate hidden Redis clients in domain logic. Keep tests deterministic with a no-op or fake lease.
- [ ] On lease contention, raise the retryable busy error so the existing Celery retry mechanism reschedules the stage. Never start the model after acquisition failure.
- [ ] Emit structured `heavy_model_acquired`/`heavy_model_released` events with purpose, wait duration, owner-process ID, and before/after `MemorySnapshot`; never log source media content.
- [ ] Rerun focused pipeline/CLI tests.
- [ ] Commit: `Enforce sequential heavy model stages`

## Task 5: Make native Whisper reclamation reliable

**Files:**

- Create: `backend/app/runtime/model_process.py`
- Create: `backend/tests/test_model_process.py`
- Modify: `backend/app/transcription/engine.py`
- Modify: `backend/app/workers/celery_app.py`
- Modify: `compose.yaml`
- Test: `backend/tests/test_transcription.py`
- Test: `backend/tests/test_pipeline.py`

- [ ] Add a spawned-process runner test proving the model factory executes in the child PID, a typed serializable result returns to the parent, child exceptions return a bounded error envelope without a traceback containing secrets, timeouts terminate only the exact child PID, and the parent always `join()`s/reaps it.
- [ ] Implement the runner with Python's `multiprocessing` `spawn` context and a top-level child entrypoint. The parent must consume the bounded queue/pipe result while the child is alive, enforce a configured timeout, verify a non-live exit state, and only then permit lease release. Never use shell process-name killing.
- [ ] Route worker and benchmark Whisper calls through this runner. Instantiate faster-whisper and load model weights inside the child; do not construct or import a resident model in the coordinator before spawning.
- [ ] Add engine tests proving every generator/result is fully consumed before the child replies and local references/garbage collection run on success and exceptions. If CTranslate2 exposes a supported unload API in the installed version, call it; do not call an undocumented method.
- [ ] Configure the worker with concurrency 1, prefetch multiplier 1, and `--max-tasks-per-child=1` in Compose/Celery settings. Add a configuration test that reads the effective Celery values rather than string-matching documentation.
- [ ] Confirm Stage 2 stages remain individually durable tasks. The isolated Whisper subprocess must exit before its coordinator releases the Redis lease, and a later task may load Ollama only after acquiring that lease. Do not combine transcription and reconstruction into one task.
- [ ] Add structured before-load, after-transcribe, after-cleanup snapshots. Explain that child exit is the hard reclamation boundary when native allocators retain arenas.
- [ ] Run `pytest -q tests/test_model_process.py tests/test_transcription.py tests/test_pipeline.py`.
- [ ] Commit: `Recycle workers after heavy model stages`

## Task 6: Verify Ollama unload instead of assuming it

**Files:**

- Modify: `backend/app/transcription/reconstruction/ollama.py`
- Modify: `backend/app/transcription/reconstruction/providers.py`
- Test: `backend/tests/test_reconstruction_ollama.py`
- Test: `backend/tests/test_reconstruction_provider.py`

- [ ] Add fake HTTP tests proving release sends the supported `keep_alive: 0` unload request for the configured model, then polls the Ollama process/model listing until the model disappears.
- [ ] Test immediate success, delayed disappearance, bounded timeout, transport failure, and release from an inference exception.
- [ ] Represent unload outcome as diagnostics: requested, confirmed, elapsed seconds, warning. A timeout must warn and keep the heavy-model lease from being reused until bounded cleanup handling finishes; it must not corrupt already produced text.
- [ ] Do not kill Ollama or all containers automatically.
- [ ] Rerun Ollama/provider tests.
- [ ] Commit: `Verify reconstruction model unload`

## Task 7: Measure the lifecycle and close the checkpoint

**Files:**

- Modify: `docs/ENVIRONMENT.md`
- Modify: `docs/BENCHMARKS.md`
- Modify: `docs/STAGE_2_7_OPERATIONS.md`
- Modify: `STATUS.md`

- [ ] Ask the operator to apply the documented WSL configuration if required, then rerun `diagnose-memory`. Confirm Linux/Docker effective capacity is approximately 11 GiB before proceeding.
- [ ] With no model resident, run exactly one authorized transcription task. Record process/container/system memory and swap before load, at observed peak, after model cleanup, and after worker child exit.
- [ ] Then run exactly one 4B reconstruction task. Record the same measurements and confirm Whisper is absent before Ollama loads and Ollama is absent after release.
- [ ] Repeat the sequence three times. Failure conditions: any overlap of Whisper and Ollama, OOM event, stale Ollama model after timeout, effective capacity below 10 GiB, or swap growth greater than 1 GiB per run without recovery.
- [ ] Run:

  ```bash
  docker compose run --rm --no-deps backend sh -lc \
    "pytest -q tests/test_runtime_memory.py tests/test_heavy_model_lease.py tests/test_model_process.py tests/test_transcription.py \
     tests/test_reconstruction_ollama.py tests/test_pipeline.py tests/test_stage2_pipeline_e2e.py && \
     ruff format --check app tests && ruff check app tests && \
     mypy app/runtime app/transcription/engine.py app/transcription/reconstruction/ollama.py app/workers"
  ```

- [ ] Record exact test counts, memory values, and failures. Do not call memory release verified from unit tests alone.
- [ ] Commit: `Verify sequential model memory lifecycle`
- [ ] Stop for review. Do not begin model downloads or quality claims until three measured lifecycle runs pass.
