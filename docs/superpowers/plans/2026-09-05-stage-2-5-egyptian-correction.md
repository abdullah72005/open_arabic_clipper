# Stage 2.5 Egyptian Arabic Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add contextual Egyptian Arabic ASR correction without changing raw evidence, timestamps, English/code-switching, or operator control.

**Architecture:** A pure correction module loads a versioned lexicon, compares Arabic-aware candidates in bounded segment context, and optionally calls a validated OpenAI-compatible local provider. The pipeline stores raw, corrected, and final text separately; API/UI exposes evidence and manual overrides.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Alembic, faster-whisper 1.2.1, pytest, Next.js, TypeScript, Tailwind CSS.

**Spec:** `docs/superpowers/specs/2026-09-05-stage-2-5-egyptian-correction-design.md`

## Global Constraints

- `Transcript.raw_text`, segment `text`, starts, ends, and word timestamps are immutable ASR evidence.
- Text-only correction cannot merge, reorder, create, delete, or realign segments.
- Preserve dialect, meaning, names, numbers, Latin/code-switched words; low confidence keeps raw text.
- Default remains local-first/offline; no LLM endpoint is required and Stage 3 stays out of scope.
- Backend verification runs Python 3.12 in Docker as `python -m pytest` against mounted source.
- Every production behavior starts red, then minimal green, then refactor.

## File Structure

- `backend/app/transcription/correction.py`: correction data types, Arabic comparison, confidence gate, contextual selection.
- `backend/app/transcription/providers.py`: provider protocol, prompt, HTTP adapter, response validator.
- `backend/app/transcription/lexicons/egyptian_ar.json`: canonical phrases, confusions, priority, notes.
- `backend/app/transcription/fixtures/egyptian_ar_correction.json`: verified diverse regression corpus.
- `backend/app/transcription/correction_benchmark.py`: fixture metrics, latency, and memory report.
- Transcript model/migration/pipeline/settings/engine/quality: durable derived state and safe options.
- API/frontend transcript files: compatibility, overrides, and debug comparison.
- Focused correction/benchmark tests plus existing Stage 2 tests: correctness and regression coverage.

### Task 1: Correction domain and lexicon

**Files:** Create `backend/app/transcription/correction.py`, `backend/app/transcription/lexicons/egyptian_ar.json`, `backend/tests/test_correction.py`.

**Interfaces:** `CorrectionConfig`; `SegmentCorrection(segment_index, raw_text, corrected_text, applied, confidence, method, version, changes, uncertain)`; `ContextualCorrector.correct(segments) -> list[SegmentCorrection]`.

- [ ] **Step 1: Write failing test.**

```python
def test_known_egyptian_confusion_keeps_timestamp() -> None:
    result = ContextualCorrector.from_default_lexicon().correct([{"start": 0.0, "end": 1.0, "text": "خطي بالك"}])
    assert result[0].corrected_text == "خلي بالك"
```

Add failure cases for supplied long/confusion examples, low confidence, unchanged Arabic, English-only, code switching, names, and numbers.

- [ ] **Step 2: Verify red.** Run `docker compose run --rm --no-deps -v /mnt/c/Users/abdul/OneDrive/Documents/VScode/oac/backend:/app backend sh -c 'python -m pip install -q pytest pytest-asyncio httpx && python -m pytest -q tests/test_correction.py'`. Expected: import failure because correction module is absent.

- [ ] **Step 3: Implement green.**

```python
class ContextualCorrector:
    def correct(self, segments: list[Mapping[str, object]]) -> list[SegmentCorrection]:
        return [self._correct_one(index, segments) for index in range(len(segments))]
```

Load lexicon with `importlib.resources`; normalize only for comparison; protect numeric/Latin tokens; apply only exact declared confusion or high-similarity small edit; use two neighbors each side.

- [ ] **Step 4: Verify green.** Run Task 1 test command. Expected: supplied examples correct and input segment dictionaries unchanged.

- [ ] **Step 5: Commit.** Run `git add backend/app/transcription/correction.py backend/app/transcription/lexicons/egyptian_ar.json backend/tests/test_correction.py && git commit -m "Add conservative Egyptian transcript correction"`.

### Task 2: Provider contract and strict fallback

**Files:** Create `backend/app/transcription/providers.py`; modify correction module/test.

**Interfaces:** `CorrectionProvider.correct_batch(requests) -> list[dict[str, object]]`; `OpenAICompatibleCorrectionProvider(base_url, model, api_key, timeout_seconds, request)`; optional provider constructor parameter.

- [ ] **Step 1: Write failing test.**

```python
def test_invalid_provider_ids_use_safe_lexicon_result() -> None:
    provider = FakeProvider([{"segment_id": 9, "corrected_text": "invented", "changed": True, "confidence": 0.99, "changes": []}])
    result = ContextualCorrector.from_default_lexicon(provider=provider).correct([{"start": 0.0, "end": 1.0, "text": "خطي بالك"}])
    assert result[0].corrected_text == "خلي بالك"
```

Also cover two-before/two-after context, malformed JSON, duplicate/missing IDs, unsafe number/Latin edits, low confidence, and transport errors.

- [ ] **Step 2: Verify red.** Run Task 1 test command. Expected: provider protocol/validation missing.

- [ ] **Step 3: Implement green.**

```python
class CorrectionProvider(Protocol):
    def correct_batch(self, requests: list[dict[str, object]]) -> list[dict[str, object]]: ...
```

Use `urllib.request` and OpenAI-compatible `/v1/chat/completions`; encode exact spec prompt; parse JSON; reject non-list, bad/duplicate/missing IDs, bad ranges/types, protected-token changes, oversized rewrites. Any provider failure returns no provider correction.

- [ ] **Step 4: Verify green.** Run Task 1 test command. Expected: unsafe provider data can never change raw evidence.

- [ ] **Step 5: Commit.** Run `git add backend/app/transcription/providers.py backend/app/transcription/correction.py backend/tests/test_correction.py && git commit -m "Add validated contextual correction provider"`.

### Task 3: Fixture benchmark

**Files:** Create `backend/app/transcription/fixtures/egyptian_ar_correction.json`, `backend/app/transcription/correction_benchmark.py`, `backend/tests/test_correction_benchmark.py`.

**Interfaces:** `run_correction_fixture_benchmark(path, corrector) -> CorrectionBenchmarkReport`; report dictionary contains improved, unchanged, worsened, automatic correction rate, uncertain rate, exact match rate, normalized token error rate, wall-clock seconds, peak memory bytes.

- [ ] **Step 1: Write failing test.**

```python
def test_fixture_benchmark_improves_egyptian_without_english_regression() -> None:
    report = run_correction_fixture_benchmark(default_fixture_path(), ContextualCorrector.from_default_lexicon())
    assert report.improved >= 3
    assert report.worsened == 0
    assert report.by_category["english_only"].worsened == 0
```

- [ ] **Step 2: Verify red.** Run Task 1 command with `tests/test_correction_benchmark.py`. Expected: corpus/runner absent.

- [ ] **Step 3: Implement green.** Corpus fields: raw, expected, previous, next, category, semantic_review. Include supplied examples plus filler, connected speech, slang, negation, question, football, business/tech, Arabic-English, names, numbers, correct Arabic, English-only. Compare baseline `normalize_transcript(raw)`, measure `time.perf_counter` and `tracemalloc`, label all metrics fixture/system indicators.

- [ ] **Step 4: Verify green.** Run `docker compose run --rm --no-deps -v /mnt/c/Users/abdul/OneDrive/Documents/VScode/oac/backend:/app backend sh -c 'python -m pip install -q pytest pytest-asyncio httpx && python -m pytest -q tests/test_correction_benchmark.py && python -m app.transcription.correction_benchmark --fixture app/transcription/fixtures/egyptian_ar_correction.json --baseline && python -m app.transcription.correction_benchmark --fixture app/transcription/fixtures/egyptian_ar_correction.json'`. Expected: measured JSON and zero worsened fixtures.

- [ ] **Step 5: Commit.** Run `git add backend/app/transcription/fixtures/egyptian_ar_correction.json backend/app/transcription/correction_benchmark.py backend/tests/test_correction_benchmark.py && git commit -m "Add Egyptian ASR correction benchmark"`.

### Task 4: Persistence, pipeline, quality, Whisper settings

**Files:** Modify transcript model, `pipeline/stages.py`, settings, transcription service/engine, quality service, Stage 2 tests; create `backend/alembic/versions/20260905_0005_stage_2_5_correction.py`.

**Interfaces:** `TranscriptionOptions` gains temperature/previous-text/VAD/initial-prompt/hotwords; `TranscriptNormalizationExecutor(session, corrector=None)` is idempotent; transcript gains corrected/final text and confidence/ratio/method/version fields.

- [ ] **Step 1: Write failing test.**

```python
def test_normalization_persists_raw_corrected_final_and_timestamp(...) -> None:
    transcript = TranscriptNormalizationExecutor(session=session).execute(source)
    assert transcript.segments[0]["raw_text"] == "خطي بالك"
    assert transcript.segments[0]["corrected_text"] == "خلي بالك"
    assert transcript.segments[0]["final_text"] == "خلي بالك"
    assert transcript.segments[0]["start"] == 0.0
```

Add fingerprint/engine option tests, migration upgrade/downgrade, raw/order/word timestamp preservation, retry/idempotence, legacy fields, and confidence-not-accuracy tests.

- [ ] **Step 2: Verify red.** Run Task 1 command with `tests/test_transcription.py tests/test_models.py tests/test_migrations.py tests/test_quality.py`. Expected: derived fields/options/migration absent.

- [ ] **Step 3: Implement green.** Store each segment raw/corrected/applied/confidence/method/version/changes/operator/final. Join final segment text; build chunks from final text. Preserve override only if index/raw match. Backfill migration. Keep Whisper defaults unchanged: fallback temperatures `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`, previous text true, VAD false, prompt/hotwords absent.

- [ ] **Step 4: Verify green.** Run Step 2 command. Expected: all existing Stage 2 behavior preserved.

- [ ] **Step 5: Commit.** Run `git add backend/app/models/transcript.py backend/alembic/versions/20260905_0005_stage_2_5_correction.py backend/app/pipeline/stages.py backend/app/core/settings.py backend/app/transcription/service.py backend/app/transcription/engine.py backend/app/services/source_quality.py backend/tests && git commit -m "Persist contextual transcript corrections"`.

### Task 5: API, manual feedback, debug UI

**Files:** Modify `backend/app/api/app.py`, API tests, frontend API client/tests, source detail page.

**Interfaces:** POST and DELETE `/api/sources/{source_id}/transcript/segments/{segment_index}/override`; optional transcript segment raw/corrected/final metadata.

- [ ] **Step 1: Write failing test.**

```python
def test_operator_override_preserves_evidence(client: TestClient, source: SourceVideo) -> None:
    response = client.post(f"/api/sources/{source.id}/transcript/segments/0/override", json={"text": "خلي بالك يا أحمد"})
    assert response.json()["raw_text"] == "خطي بالك"
    assert response.json()["corrected_text"] == "خلي بالك"
    assert response.json()["final_text"] == "خلي بالك يا أحمد"
```

Add clear override, invalid index/text, final-text search, API compatibility, and TS client route tests.

- [ ] **Step 2: Verify red.** Run Task 1 command with `tests/test_api_sources.py`; then `(cd frontend && npm test -- --runInBand src/lib/api-client.test.ts)`. Expected: endpoint/client absent.

- [ ] **Step 3: Implement green.** Validate index/text bounds; mutate only operator/final text; recompute transcript final/normalized text and chunks transactionally; search final text. UI shows final text and expandable raw/automatic/confidence/method/version with editable override, preserving direction and seek.

- [ ] **Step 4: Verify green.** Run backend API test then `cd frontend && npm test && npm run lint && npm run build`. Expected: override/clear retains immutable evidence and UI compiles.

- [ ] **Step 5: Commit.** Run `git add backend/app/api/app.py backend/tests/test_api_sources.py frontend/src/lib/api-client.ts frontend/src/lib/api-client.test.ts frontend/src/app/sources/[id]/page.tsx && git commit -m "Add transcript correction debug and override"`.

### Task 6: Documentation and completion verification

**Files:** Modify README, `.env.example`, AGENTS, STATUS, benchmarks, pipeline docs, benchmark test.

**Interfaces:** Document all `CLIPFACTORY_CORRECTION_*` and added `CLIPFACTORY_WHISPER_*` knobs, provider opt-in, benchmark command, metrics, default behavior, non-goals.

- [ ] **Step 1: Write failing final-report test.**

```python
def test_benchmark_report_includes_latency_and_memory(capsys) -> None:
    main(["--fixture", str(default_fixture_path())])
    assert {"wall_clock_seconds", "peak_memory_bytes"} <= json.loads(capsys.readouterr().out).keys()
```

- [ ] **Step 2: Verify red or existing coverage.** Run Task 1 command with benchmark test; record whether required keys already pass.

- [ ] **Step 3: Document observed baseline/post results.** Run same fixture in baseline/correction modes. Update docs with actual counts, rates, latency/memory, metric limits, remaining cases, selected provider/default, unchanged Whisper defaults, and controlled opt-in requirement.

- [ ] **Step 4: Run full quality gates.** Run `docker compose config`; Docker backend `coverage run --source=app -m pytest`, Ruff format/check, mypy; frontend `npm ci && npm test && npm run lint && npm run build`. Confirm migration upgrade/downgrade and raw/timestamp/order evidence.

- [ ] **Step 5: Commit.** Run `git add README.md .env.example AGENTS.md STATUS.md docs/BENCHMARKS.md docs/PIPELINE.md backend/tests/test_correction_benchmark.py && git commit -m "Document Stage 2.5 correction operations"`.

## Plan Self-Review

- Coverage: Tasks 1–3 cover correction, context, lexicon, provider, fixtures, quality, latency, memory, Arabic/English/code-switch safety. Task 4 covers persistence, recovery, timestamps, settings, quality. Task 5 covers API/UI/manual feedback. Task 6 documents and verifies every completion criterion.
- No placeholders: every task gives files, interfaces, a red test, command, expected result, green implementation, verification, and commit.
- Type consistency: correction classes and provider precede pipeline/API consumers; final text is the only downstream display/search/chunk field.
