# Stage 3.7 Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the time from authorized source to `READY_FOR_REFINEMENT` through measurable ingest and whole-source ASR execution improvements without changing INDEX transcript or candidate semantics.

**Architecture:** Persist an additive per-attempt JSON metrics record through `StageExecutionResult` and `PipelineRunner`; instrument yt-dlp, probe, extraction, and transcription at their existing boundaries. Keep baseline decoding in the current spawned child and add bounded CPU-thread and batch controls, only enabling the compatible faster-whisper batch adapter after deterministic contract tests and a non-destructive representative comparison.

**Tech Stack:** Python 3.12, SQLAlchemy 2, Alembic, PostgreSQL, Celery, faster-whisper 1.2.1, FFmpeg, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-12-stage-3-7-performance-design.md`

## Global Constraints

- Preserve raw ASR segments and words, Stage 2.5, INDEX Stage 2.7 deferral, Stage 2.7.1, Stage 3, Stage 3.5, manual overrides, cancellation, child cleanup, and heavy-model lease behavior.
- Keep default INDEX decoding: `large-v3-turbo`, CPU int8 without CUDA, beam 5, automatic language, timestamps, current temperature fallback, previous-text conditioning, and VAD off.
- Do not add remote content hashing, extra upload copies, audio-first/deferred video lifecycle, new providers, GPU work, model substitutions, or Stage 4 work.
- Persist only non-sensitive numeric/enum metrics. Never persist source URLs, yt-dlp diagnostics, provider secrets, or transcript text in metrics.
- Use TDD: each production change begins with a focused failing test, followed by the minimal implementation and focused green test.

## Task 1: Durable stage metrics boundary

**Files:**

- Create: `backend/alembic/versions/20260912_0013_stage_3_7_pipeline_metrics.py`
- Modify: `backend/app/models/pipeline_run.py`
- Modify: `backend/app/pipeline/executor.py`
- Modify: `backend/app/pipeline/runner.py`
- Modify: `backend/tests/test_pipeline.py`

- [ ] Write failing tests that construct a successful executor result with `metrics={"wall_seconds": 1.25, "cache": "miss"}` and assert the completed `PipelineRun.metrics` exactly preserves that JSON; assert existing two-field results persist `{}`; assert a runner fingerprint skip creates no execution and does not overwrite the prior run metrics.
- [ ] Add nullable-safe, non-null JSON `metrics` with `server_default='{}'` to `pipeline_runs` in the model and migration. Upgrade populates existing rows with `{}`; downgrade drops only this column.
- [ ] Change `StageExecutionResult` to:

  ```python
  @dataclass(frozen=True)
  class StageExecutionResult:
      output_fingerprint: str
      value: object | None = None
      metrics: Mapping[str, object] = field(default_factory=dict)
  ```

  Copy the mapping with `dict()` before persistence; reject non-JSON-serializable values with a `StageExecutionError` before the success transition.
- [ ] In `PipelineRunner.run`, assign validated metrics immediately before setting `SUCCEEDED`, then emit one structured `pipeline_stage_metrics` log with source ID, stage, attempt, skipped flag, and metrics. Keep failure/cancellation transitions unchanged.
- [ ] Run the new migration against an empty test database and execute:

  ```bash
  docker compose run --rm --no-deps backend pytest -q tests/test_pipeline.py
  ```

- [ ] Commit: `Add durable pipeline stage metrics`

## Task 2: Instrument acquisition without weakening yt-dlp safety

**Files:**

- Modify: `backend/app/services/source_adapters.py`
- Modify: `backend/app/pipeline/stages.py`
- Modify: `backend/tests/test_api_sources.py`
- Modify: `backend/tests/test_transcription.py`

- [ ] Write failing unit tests with injected monotonic clock/subprocess fakes that assert metadata and download times are independently present; final bytes equal the completed file's `stat().st_size`; local acquisition reports one storage write; URL validation, proxy use, max-size enforcement, bounded diagnostics, and public-address revalidation remain unchanged.
- [ ] Add a frozen `AcquisitionMetrics` value to `AcquiredSource`: source kind, metadata/download/storage-write/hash wall times, final artifact bytes, `cache_reuse`, and `postprocess_state`. Local acquisition records only its measured atomic streamed write. Remote acquisition measures `inspect()` and `_run_download()` independently and obtains bytes after `_downloaded_path()` from filesystem facts.
- [ ] Record `postprocess_state="not_requested"` and `postprocess_seconds=None` for the existing default-format command. Do not parse progress output or claim a merge/remux duration unless a future application-requested postprocessor exposes a reliable duration.
- [ ] Have `IngestExecutor.execute` return those metric values through `StageExecutionResult`; retain its input/output fingerprints and the existing remote replacement of `source_uri`.
- [ ] Run:

  ```bash
  docker compose run --rm --no-deps backend pytest -q tests/test_api_sources.py tests/test_transcription.py
  ```

- [ ] Commit: `Instrument source acquisition timing`

## Task 3: Make probe and WAV cache work visible

**Files:**

- Modify: `backend/app/media/audio.py`
- Modify: `backend/app/pipeline/stages.py`
- Modify: `backend/tests/test_audio.py`
- Modify: `backend/tests/test_transcription.py`

- [ ] Write failing tests proving a valid cached WAV reports `cache_reuse="hit"`, its validation hash duration, and zero FFmpeg duration; a stale/missing WAV reports `cache_reuse="miss"` and an extraction duration; normal runner fingerprint skipping does not invoke `AudioExtractor.extract`.
- [ ] Return a small extraction outcome containing the artifact and non-sensitive metrics. Keep `_is_valid()`'s SHA-256 check exactly as the corruption guard; do not remove it or add another file read.
- [ ] Time `FFprobe.probe()` in `ProbeExecutor.execute`, reporting `ffprobe_seconds`; time `AudioExtractor.extract()` in `AudioExtractionExecutor.execute`, reporting extraction/cache metrics. Preserve their fingerprints, errors, storage ownership, and output fingerprints.
- [ ] Run:

  ```bash
  docker compose run --rm --no-deps backend pytest -q tests/test_audio.py tests/test_transcription.py tests/test_pipeline.py
  ```

- [ ] Commit: `Report media cache and probe timing`

## Task 4: Add default-safe ASR execution configuration

**Files:**

- Modify: `backend/app/core/settings.py`
- Modify: `.env.example`
- Modify: `backend/app/transcription/service.py`
- Modify: `backend/app/transcription/engine.py`
- Modify: `backend/app/pipeline/stages.py`
- Modify: `backend/tests/test_transcription.py`
- Modify: `docs/ENVIRONMENT.md`

- [ ] Write failing settings/options tests for `whisper_cpu_threads=0`, `whisper_index_batch_size=1`, default VAD-off, and existing model/beam/language/timestamp settings. Assert batch size changes the transcription fingerprint, while automatic thread zero remains recorded but does not change an otherwise identical output fingerprint.
- [ ] Add bounded settings `CLIPFACTORY_WHISPER_CPU_THREADS` (`ge=0`, safe default `0`) and `CLIPFACTORY_WHISPER_INDEX_BATCH_SIZE` (`ge=1`, `le=4`, safe default `1`). Include batch size in `TranscriptionOptions`; include it in `asdict()` fingerprint and persisted `transcription_options`.
- [ ] Extend the child-only model factory boundary to pass `cpu_threads` to `faster_whisper.WhisperModel`. Never import or construct faster-whisper in the worker parent. Add execution metrics for requested/effective CPU threads, batch size, resolved device/compute type, child elapsed seconds, peak RSS (or `UNKNOWN`), and before/after memory snapshots.
- [ ] Preserve `batch_size=1` using the existing `WhisperModel.transcribe` call and exactly its current arguments. Keep thread count out of the input fingerprint unless a measured comparison demonstrates output variance.
- [ ] Run focused settings, spawned-child, timestamp, cancellation, and lease tests:

  ```bash
  docker compose run --rm --no-deps backend pytest -q tests/test_transcription.py tests/test_model_process.py tests/test_heavy_model_lease.py
  ```

- [ ] Commit: `Add bounded ASR execution settings`

## Task 5: Add the conditional batch adapter only after its contract is red

**Files:**

- Modify: `backend/app/transcription/engine.py`
- Modify: `backend/tests/test_transcription.py`

- [ ] Write a failing fake-batch-pipeline test for batch size 2 that verifies all required arguments: `language=None`, beam 5, full temperature tuple, `condition_on_previous_text=True`, `word_timestamps=True`, `vad_filter=False`, initial prompt/hotwords, and `batch_size=2`. Assert serialized segments/words have the same timestamp shape as the normal path.
- [ ] Write failing spawned-child tests showing batch execution fully materializes generators before replying, propagates cancellation through `ModelProcessRunner`, reaps the child, and reports peak RSS on normal exit.
- [ ] In the child only, construct `BatchedInferencePipeline(model_obj)` only when `options.index_batch_size > 1`; invoke its public `transcribe` signature with explicit preserved values. Keep the normal model path for one. Do not alter Stage 3.5, which has separate transcription options.
- [ ] If faster-whisper 1.2.1 cannot honor one required argument or timestamp result, delete this branch and leave batch size restricted to one; document the rejection rather than approximating the contract.
- [ ] Run:

  ```bash
  docker compose run --rm --no-deps backend pytest -q tests/test_transcription.py tests/test_model_process.py
  ```

- [ ] Commit: `Support bounded compatible Whisper batching`

## Task 6: Non-destructive representative comparison

**Files:**

- Modify: `backend/app/transcription/benchmark.py`
- Modify: `backend/app/cli.py`
- Modify: `backend/tests/test_transcription.py`
- Modify: `backend/tests/test_cli.py`

- [ ] Write failing tests for a read-only benchmark command that accepts a source ID, resolves only its existing `AudioArtifact`, refuses absent media, runs no pipeline transition, does not mutate `Transcript`, `ClipCandidate`, `CandidateAnalysis`, or `PipelineRun`, and writes only a private benchmark JSON artifact.
- [ ] Implement `benchmark-index-source SOURCE_ID --label NAME` using the current engine/options. It writes the raw benchmark output plus normalized/candidate comparison summaries under storage-owned `benchmarks/stage-3-7/<run-id>/`; it may use isolated SQLite/session data but must not persist to the production database.
- [ ] Report duration, elapsed time, `audio_minutes_per_wall_minute = duration_seconds / 60 / wall_seconds * 60`, child peak RSS, options, timestamp validity, segment/word counts, protected-token/code-switch summaries, and candidate key/score/disposition differences. Use existing normalizer and deterministic candidate executor; do not implement a new scoring system.
- [ ] Run focused CLI/benchmark tests and inspect production record counts before/after a dry fake-engine test.
- [ ] Commit: `Add isolated INDEX comparison benchmark`

## Task 7: Measure only the bounded representative ladder

**Files:**

- Modify: `docs/BENCHMARKS.md`
- Modify: `STATUS.md`
- Modify: `docs/ENVIRONMENT.md`

- [ ] Capture read-only resource evidence in the worker/container: `nproc`, `/sys/fs/cgroup/cpu.max`, `python -c 'import os; print(os.cpu_count())'`, memory snapshot, swap totals/free, Celery worker configuration, and any concurrent Ollama residency.
- [ ] Retain baseline A from durable records; never force/rewrite source `4037a813-6fe6-4c83-96ff-e5cd4bf210ce`. Test B on its cached WAV with the best conservative measured thread value. Test C (batch 2) only if Task 5 passed and B is accepted. Test D (batch 4) only if C is safe and worth further testing. Test E with one VAD setting only if no accepted B/C/D gain is adequate.
- [ ] For every actual run, preserve the JSON benchmark artifact and record only observed wall time, throughput, RSS, memory/swap/CPU evidence, transcript/candidate comparison, and outcome. Stop the ladder at the first adequate accepted gain.
- [ ] If one controlled post-instrumentation remote ingest is authorized and necessary, use one new authorized source acquisition only; otherwise document historical ingest components as unreconstructable. Never redownload the representative source.
- [ ] Document chosen and rollback settings, rejected actual variants, that ingestion's network transfer is not an application defect when measured, and that full video remains acquired early.
- [ ] Commit: `Document Stage 3.7 measurements`

## Task 8: Final verification and delivery

**Files:** all modified files above.

- [ ] Run Alembic upgrade against the Docker database and verify the `pipeline_runs.metrics` column exists.
- [ ] Run the focused regression suite:

  ```bash
  docker compose run --rm --no-deps backend sh -lc \
    "pytest -q tests/test_pipeline.py tests/test_api_sources.py tests/test_storage.py tests/test_audio.py tests/test_transcription.py tests/test_model_process.py tests/test_heavy_model_lease.py tests/test_stage3_candidates.py tests/test_stage35_refinement.py && \
     ruff format --check app tests && ruff check app tests && \
     mypy app/core/settings.py app/pipeline app/services/source_adapters.py app/media/audio.py app/transcription app/runtime"
  ```

- [ ] Inspect `git diff --check`, `git diff --stat`, `git status --short`, and the migration revision chain. Confirm no media, source artifacts, secrets, or unrelated changes are staged.
- [ ] Commit coherent implementation/docs changes. Report only actual tested variants and verified results using the exact required Stage 3.7 report structure.
