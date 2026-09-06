# Stage 2.7 Contextual Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, two-pass, multi-segment Egyptian Arabic transcript reconstruction while preserving raw alignment and Stage 2.5 fallback behavior.

**Architecture:** A dedicated durable stage builds overlapping context windows for each stable segment, asks an optional local OpenAI-compatible provider for and then ranks a bounded candidate set, and applies only HIGH results after deterministic phonetic, entity, acoustic, and hallucination gates. Detailed reconstruction state is derived and persisted separately; manual text remains highest priority and Stage 2.5 remains the fail-open baseline.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, SQLAlchemy 2, Alembic, PostgreSQL, Celery, Redis, faster-whisper 1.2.1, CTranslate2 4.8.2, pytest, Next.js, TypeScript, Tailwind CSS.

**Spec:** `docs/superpowers/specs/2026-09-05-stage-2-7-contextual-reconstruction-design.md`

## Global Constraints

- Do not implement Stage 3 candidate discovery, rendering, publishing, or automatic authorization.
- Preserve every raw segment `text`, stable list index, start/end, word timestamp, and source order.
- Preserve Egyptian colloquial wording, fillers, repetitions, English/code switching, names, and numbers.
- Never add external topic/fact context; entity memory comes only from the same raw transcript.
- Only a HIGH Stage 2.7 decision may replace Stage 2.5 text in `final_text`; manual override remains first.
- Provider absence, malformed output, timeout, or unsafe output falls back to Stage 2.5 and records flags.
- No CUDA, LLM endpoint, Ollama service, or model download is required for normal pipeline completion.
- Python checks run in the Docker Python 3.12 runtime; host Python 3.10 is not acceptance evidence.
- Every behavior change follows red-green-refactor and each task ends with focused passing checks.
- Real-audio quality claims require the private unseen benchmark gate; fixture success is insufficient.

---

## File Structure

Create:

- `backend/app/transcription/reconstruction/__init__.py`: public reconstruction exports.
- `backend/app/transcription/reconstruction/types.py`: immutable domain types and enums.
- `backend/app/transcription/reconstruction/windows.py`: window and acoustic evidence builders.
- `backend/app/transcription/reconstruction/phonetics.py`: Arabic weighted comparison.
- `backend/app/transcription/reconstruction/entities.py`: transcript-only source entity memory.
- `backend/app/transcription/reconstruction/providers.py`: two-pass protocol and HTTP adapter.
- `backend/app/transcription/reconstruction/validation.py`: deterministic candidate validation.
- `backend/app/transcription/reconstruction/confidence.py`: score, margin, and level gate.
- `backend/app/transcription/reconstruction/service.py`: batching, fallback, flags, and fingerprint.
- `backend/app/transcription/reconstruction/benchmark.py`: private benchmark runner and gate.
- `backend/app/transcription/fixtures/egyptian_ar_reconstruction.json`: text-only regressions.
- `backend/alembic/versions/20260905_0008_stage_2_7_reconstruction.py`: columns and enum constraints.
- focused tests matching each module under `backend/tests/`.

Modify:

- faster-whisper engine serialization for public acoustic fields;
- transcript model, pipeline enums/runner/stages/tasks, settings, API, CLI, quality service, and tests;
- storage service/category so private benchmark manifests remain below the configured root;
- frontend API types/client/source detail and tests;
- environment, architecture, pipeline, benchmark, operations, status, and project-memory docs.

Do not place reconstruction logic in `pipeline/stages.py`, `api/app.py`, or the existing 286-line
`correction.py`. Those files orchestrate focused services only.

### Task 1: Preserve complete public faster-whisper acoustic evidence

**Files:**
- Modify: `backend/app/transcription/engine.py`
- Modify: `backend/tests/test_transcription.py`

**Interfaces:**
- Consumes: faster-whisper `Segment.tokens`, `compression_ratio`, and `temperature`.
- Produces: raw segment dictionaries containing those fields in addition to current evidence.

- [ ] **Step 1: Write failing serialization assertions.** Extend the existing fake Whisper segment
with `tokens=[50364, 1234, 50414]`, `compression_ratio=1.17`, and `temperature=0.2`, then assert:

```python
assert result.segments[0]["tokens"] == [50364, 1234, 50414]
assert result.segments[0]["compression_ratio"] == 1.17
assert result.segments[0]["temperature"] == 0.2
assert result.segments[0]["start"] == 0.0
assert result.segments[0]["end"] == 1.25
```

- [ ] **Step 2: Verify red.** Run:

```bash
docker compose exec -T backend python -m pytest -q tests/test_transcription.py -k "engine"
```

Expected: assertions fail because the three new keys are absent.

- [ ] **Step 3: Serialize only public evidence fields.** Change `_serialize_segment` to include:

```python
"tokens": [int(token) for token in getattr(segment, "tokens", None) or []],
"compression_ratio": _optional_float(getattr(segment, "compression_ratio", None)),
"temperature": _optional_float(getattr(segment, "temperature", None)),
```

Do not change text, timestamps, or word serialization.

- [ ] **Step 4: Verify green.** Run Task 1 command. Expected: engine tests pass.

- [ ] **Step 5: Commit.**

```bash
git add backend/app/transcription/engine.py backend/tests/test_transcription.py
git commit -m "Preserve Whisper acoustic evidence"
```

### Task 2: Define reconstruction types, acoustic score, and context windows

**Files:**
- Create: `backend/app/transcription/reconstruction/__init__.py`
- Create: `backend/app/transcription/reconstruction/types.py`
- Create: `backend/app/transcription/reconstruction/windows.py`
- Create: `backend/tests/test_reconstruction_windows.py`

**Interfaces:**
- Produces: `ConfidenceLevel`, `QualityFlag`, `AcousticEvidence`, `WindowSegment`,
  `ReconstructionWindow`, `ReconstructionCandidate`, `ResolutionScores`,
  `SegmentReconstruction`, `ReconstructionResult`, `WindowConfig`.
- Produces: `acoustic_evidence(segment) -> AcousticEvidence`.
- Produces: `build_reconstruction_window(segments, target_index, config) -> ReconstructionWindow`.

- [ ] **Step 1: Write window and acoustic tests.** Cover center target, source edge, short source,
one segment longer than 15 seconds, 8-segment cap, 15-second cap, absent word probabilities,
and exact score math:

```python
def test_window_expands_both_sides_without_changing_target_identity() -> None:
    segments = [
        {"start": float(i * 2), "end": float(i * 2 + 2), "text": f"s{i}"}
        for i in range(9)
    ]
    window = build_reconstruction_window(segments, 4, WindowConfig())
    assert window.target_segment_index == 4
    assert 3 <= len(window.segments) <= 8
    assert window.segments[-1].end - window.segments[0].start <= 15
    assert [item.segment_index for item in window.segments] == sorted(
        item.segment_index for item in window.segments
    )

def test_acoustic_score_uses_documented_weights() -> None:
    evidence = acoustic_evidence(
        {"avg_logprob": -0.2, "no_speech_prob": 0.1,
         "words": [{"probability": 0.8}, {"probability": 0.6}]}
    )
    expected = 0.50 * 0.7 + 0.35 * math.exp(-0.2) + 0.15 * 0.9
    assert evidence.confidence == pytest.approx(expected)
```

- [ ] **Step 2: Verify red.** Run:

```bash
docker compose exec -T backend python -m pytest -q tests/test_reconstruction_windows.py
```

Expected: import failure because reconstruction package is absent.

- [ ] **Step 3: Implement immutable types and defaults.** Use `str, Enum` for values persisted in
JSON and frozen dataclasses for internal contracts. `WindowConfig` defaults are exact:

```python
@dataclass(frozen=True)
class WindowConfig:
    target_seconds: float = 8.0
    target_segments: int = 5
    max_seconds: float = 15.0
    max_segments: int = 8
    provider_batch_windows: int = 16
    provider_batch_characters: int = 48_000
```

`ConfidenceLevel` values are `HIGH`, `MEDIUM`, `LOW`. `QualityFlag` contains all six values from
the design spec.

- [ ] **Step 4: Implement deterministic expansion.** Keep source order in returned segments,
alternate nearest left/right additions, stop at the first target bound, and refuse any addition
that would exceed either hard cap. Short sources and an oversized target return all safely
available evidence without synthetic segments.

- [ ] **Step 5: Verify green.** Run Task 2 command. Expected: all window/acoustic tests pass.

- [ ] **Step 6: Commit.**

```bash
git add backend/app/transcription/reconstruction backend/tests/test_reconstruction_windows.py
git commit -m "Add reconstruction windows and evidence types"
```

### Task 3: Add Arabic phonetic scoring and transcript-only entity memory

**Files:**
- Create: `backend/app/transcription/reconstruction/phonetics.py`
- Create: `backend/app/transcription/reconstruction/entities.py`
- Create: `backend/tests/test_reconstruction_phonetics.py`
- Create: `backend/tests/test_reconstruction_entities.py`

**Interfaces:**
- Produces: `normalize_phonetic(text: str) -> str`.
- Produces: `phonetic_similarity(source: str, candidate: str) -> float`.
- Produces: `aligned_content_insertions(source: str, candidate: str) -> int`.
- Produces: `build_entity_memory(segments) -> SourceEntityMemory`.
- `SourceEntityMemory.supports_change(old, new, evidence_ids) -> bool`.
- `SourceEntityMemory.with_observed_nomination(surface_form, evidence_ids, segments)` returns a
  new memory only when every cited raw segment contains the exact form.

- [ ] **Step 1: Write phonetic tests.** Include Alef forms, Yeh/Alef Maqsura, Teh Marbuta,
dropped Hamza, weighted emphatic/plain pairs, transposition, connected token boundaries,
unrelated phrases, and Latin/digit strictness:

```python
@pytest.mark.parametrize(
    ("left", "right", "minimum"),
    [
        ("دخم", "ضخمة", 0.72),
        ("صاب", "صعب", 0.72),
        ("مش لعين ياكل", "مش لاقيين ياكلوا", 0.72),
        ("في الوقت", "فيالوقت", 0.95),
    ],
)
def test_egyptian_near_matches_score_high(left: str, right: str, minimum: float) -> None:
    assert phonetic_similarity(left, right) >= minimum

def test_unrelated_story_insertion_scores_low() -> None:
    assert phonetic_similarity("الراجل وصل", "الرئيس شرح تاريخ الشركة") < 0.55
```

- [ ] **Step 2: Write entity tests.** Prove exact Latin/numeric extraction, Arabic two-occurrence
rule, evidence IDs, source isolation, and rejection of an unseen name:

```python
def test_entity_memory_never_accepts_unseen_canonical_name() -> None:
    memory = build_entity_memory([
        {"text": "قابلت جاكوب امبارح"},
        {"text": "United Fruit Company سنة 1950"},
    ])
    assert memory.supports_change("جاكوب", "جاكوب أربنز", (0,)) is False
    assert memory.contains_exact("United Fruit Company") is True
```

- [ ] **Step 3: Verify red.** Run:

```bash
docker compose exec -T backend python -m pytest -q \
  tests/test_reconstruction_phonetics.py tests/test_reconstruction_entities.py
```

Expected: module import failures.

- [ ] **Step 4: Implement weighted comparison.** Use Unicode NFC, remove Arabic diacritics and
tatweel, normalize only comparison forms, apply reduced edit costs for documented letter classes,
and compare concatenated one-to-three-token spans. Preserve full costs for Latin and digits.
Clamp final similarity to `[0, 1]`.

- [ ] **Step 5: Implement entity memory.** Store exact observed forms plus segment IDs. Extract
Latin runs and digit strings directly. Admit Pass A Arabic nominations only through
`with_observed_nomination`, which verifies exact raw source occurrence at every evidence ID;
mark repeated Arabic spans after two occurrences. Do not
import a named-entity model or external lookup.

- [ ] **Step 6: Verify green.** Run Task 3 command. Expected: all phonetic/entity tests pass.

- [ ] **Step 7: Commit.**

```bash
git add backend/app/transcription/reconstruction/phonetics.py \
  backend/app/transcription/reconstruction/entities.py \
  backend/tests/test_reconstruction_phonetics.py \
  backend/tests/test_reconstruction_entities.py
git commit -m "Add Egyptian phonetic and entity evidence"
```

### Task 4: Implement strict two-pass provider contracts

**Files:**
- Create: `backend/app/transcription/reconstruction/providers.py`
- Create: `backend/tests/test_reconstruction_provider.py`

**Interfaces:**
- Produces: `GenerationRequest`, `ResolutionRequest`, `ResolutionChoice`.
- Produces protocol:

```python
class ReconstructionProvider(Protocol):
    def generate_candidates(
        self, requests: list[GenerationRequest]
    ) -> dict[int, list[ReconstructionCandidate]]: ...

    def resolve_candidates(
        self, requests: list[ResolutionRequest]
    ) -> dict[int, ResolutionChoice]: ...
```

- Produces: `OpenAICompatibleReconstructionProvider` using `/v1/chat/completions`.

- [ ] **Step 1: Write Pass A contract tests.** Capture outgoing body and assert temperature 0,
strict `json_schema` response format, no manual text, stable target/context IDs, raw/Stage 2.5
text, acoustic evidence, exact entity forms, and no more than two provider candidates accepted.

- [ ] **Step 2: Write Pass B and failure tests.** Assert opaque candidate IDs, exact ID coverage,
valid score ranges, rejection of missing/duplicate/unrequested IDs, malformed JSON, returned
timestamps, timeout, HTTP 429/5xx retry count two, and no retry for schema errors:

```python
with pytest.raises(ProviderResponseError, match="candidate ID"):
    provider.resolve_candidates([request_with_unknown_choice])
assert retrying_http.calls == 2
assert retrying_http.delays == [0.5]
```

- [ ] **Step 3: Verify red.** Run:

```bash
docker compose exec -T backend python -m pytest -q tests/test_reconstruction_provider.py
```

Expected: provider module is absent.

- [ ] **Step 4: Implement versioned prompts and Pydantic schemas.** Pass A permits zero through
two novel candidates with `text`, `changes`, `evidence_segment_ids`, and optional exact
`entity_mentions`. Pass B permits exactly
one candidate ID with `semantic_coherence`, `egyptian_naturalness`, `discourse_continuity`,
`entity_consistency`, and `selection_confidence`, each constrained to 0–1. Include the schema in
both `response_format` and prompt content.

- [ ] **Step 5: Implement HTTP and retry boundary.** Use injected standard-library request and
delay callables. Retry connection reset, timeout, 429, and 5xx once after 0.5 seconds. Map all
failures to `ProviderResponseError` without embedding response or transcript text in messages.

- [ ] **Step 6: Verify green.** Run Task 4 command. Expected: all provider tests pass without a
network endpoint.

- [ ] **Step 7: Commit.**

```bash
git add backend/app/transcription/reconstruction/providers.py \
  backend/tests/test_reconstruction_provider.py
git commit -m "Add two-pass reconstruction provider"
```

### Task 5: Enforce candidate safety and confidence gates

**Files:**
- Create: `backend/app/transcription/reconstruction/validation.py`
- Create: `backend/app/transcription/reconstruction/confidence.py`
- Create: `backend/tests/test_reconstruction_validation.py`
- Create: `backend/tests/test_reconstruction_confidence.py`

**Interfaces:**
- Produces: `ValidationConfig`, `CandidateValidation`.
- Produces: `validate_candidate(window, candidate, memory, config) -> CandidateValidation`.
- Produces: `score_candidates(...) -> list[CandidateScore]`.
- Produces: `decide_candidate(...) -> ReconstructionDecision`.

- [ ] **Step 1: Write validation table tests.** Each design rule gets an isolated case: wrong ID,
empty output, Latin mutation, number mutation, unseen entity, length ratio, token delta, three
unaligned content insertions, new clause, false change spans, low phonetic score, and a valid
multi-word candidate. Assert the precise rejection code rather than only `accepted is False`.

- [ ] **Step 2: Write confidence-boundary tests.** Exercise values immediately below and at HIGH
and MEDIUM thresholds, candidate margin, high-confidence raw edit penalty, raw winner, provider
self-confidence not bypassing phonetic score, and exact final application policy:

```python
def test_medium_candidate_is_review_only() -> None:
    decision = decide_candidate(candidate_score(0.74, margin=0.08, edit_ratio=0.1))
    assert decision.level is ConfidenceLevel.MEDIUM
    assert decision.applied is False

def test_high_candidate_is_only_automatic_reconstruction() -> None:
    decision = decide_candidate(candidate_score(0.86, margin=0.12, phonetic=0.72))
    assert decision.level is ConfidenceLevel.HIGH
    assert decision.applied is True
```

- [ ] **Step 3: Verify red.** Run:

```bash
docker compose exec -T backend python -m pytest -q \
  tests/test_reconstruction_validation.py tests/test_reconstruction_confidence.py
```

Expected: validation/confidence modules are absent.

- [ ] **Step 4: Implement exact safety constants and reason codes.** Use length range 0.60–1.60,
token delta `max(3, ceil(0.40 * baseline_tokens))`, at most two unaligned content insertions, and
minimum phonetic similarity 0.55. Raw and Stage 2.5 baseline candidates remain eligible fallback.

- [ ] **Step 5: Implement documented scoring equation.** Compute all weights and the
`0.20 * raw_acoustic_confidence * normalized_edit_ratio` penalty server-side. Calculate margin
from sorted server scores. Apply HIGH only at 0.86/0.12/0.72/0.80 bounds. Store MEDIUM at
0.74/0.08/0.20/one-token/0.85 bounds without applying it.

- [ ] **Step 6: Verify green.** Run Task 5 command. Expected: every boundary and rejection case
passes.

- [ ] **Step 7: Commit.**

```bash
git add backend/app/transcription/reconstruction/validation.py \
  backend/app/transcription/reconstruction/confidence.py \
  backend/tests/test_reconstruction_validation.py \
  backend/tests/test_reconstruction_confidence.py
git commit -m "Gate unsafe transcript reconstructions"
```

### Task 6: Orchestrate batched reconstruction, fallback, flags, and fingerprints

**Files:**
- Create: `backend/app/transcription/reconstruction/service.py`
- Create: `backend/tests/test_reconstruction_service.py`
- Modify: `backend/app/transcription/reconstruction/__init__.py`

**Interfaces:**
- Produces:

```python
class ContextualReconstructor:
    def reconstruct(
        self,
        segments: Sequence[Mapping[str, object]],
        *,
        language: str | None,
        transcription_fingerprint: str,
        correction_version: str,
    ) -> ReconstructionResult: ...
```

- Produces: `reconstruction_fingerprint(...) -> str`.

- [ ] **Step 1: Write orchestration tests.** Cover one target result per input index, payload split
at 16 windows and 48,000 characters, raw/Stage 2.5 candidate insertion, three-candidate cap,
deduplication, Pass A partial failure, Pass B failure, HIGH/MEDIUM/LOW outputs, unchanged slang,
manual field exclusion, all six quality flags, stable fingerprints, and input immutability.

- [ ] **Step 2: Write final-priority helper tests.** Lock the reusable helper:

```python
assert select_final_text(operator_text="manual", reconstructed="high", reconstruction_applied=True,
                         level=ConfidenceLevel.HIGH, corrected="stage25", raw="raw") == "manual"
assert select_final_text(operator_text=None, reconstructed="high", reconstruction_applied=True,
                         level=ConfidenceLevel.HIGH, corrected="stage25", raw="raw") == "high"
assert select_final_text(operator_text=None, reconstructed="medium", reconstruction_applied=False,
                         level=ConfidenceLevel.MEDIUM, corrected="stage25", raw="raw") == "stage25"
```

- [ ] **Step 3: Verify red.** Run:

```bash
docker compose exec -T backend python -m pytest -q tests/test_reconstruction_service.py
```

Expected: service is absent.

- [ ] **Step 4: Implement batch orchestration.** Build all windows and entity memory before calls.
Generate batches, validate/dedupe candidates, resolve only safe sets, compute decisions, and
assemble results in exact input order. Catch provider failures per batch and emit fallback records
without swallowing programming errors.

- [ ] **Step 5: Implement deterministic flags and fingerprint.** Hash sorted JSON containing every
design-specified input except API keys/manual text. Ensure transcript text never enters logs. Make
`ContextualReconstructor.disabled()` emit Stage 2.5 text plus LOW metadata without HTTP; add
`LOW_CONFIDENCE_UNRESOLVED` only when acoustic/context evidence is uncertain, not merely because
provider is disabled.

- [ ] **Step 6: Verify green.** Run Tasks 2–6 tests together:

```bash
docker compose exec -T backend python -m pytest -q tests/test_reconstruction_*.py
```

Expected: all reconstruction unit tests pass.

- [ ] **Step 7: Commit.**

```bash
git add backend/app/transcription/reconstruction/service.py \
  backend/app/transcription/reconstruction/__init__.py \
  backend/tests/test_reconstruction_service.py
git commit -m "Orchestrate contextual transcript reconstruction"
```

### Task 7: Persist Stage 2.7 state with reversible migration

**Files:**
- Create: `backend/alembic/versions/20260905_0008_stage_2_7_reconstruction.py`
- Modify: `backend/app/models/transcript.py`
- Modify: `backend/app/core/enums.py`
- Modify: `backend/tests/test_models.py`
- Modify: `backend/tests/test_migrations.py`

**Interfaces:**
- `Transcript` gains `contextual_reconstructed_text`, `reconstruction_fingerprint`,
  `reconstruction_confidence`, `reconstructed_segment_ratio`, `reconstruction_method`,
  `reconstruction_version`, `reconstruction_processing_duration`, and `reconstruction_metadata`.
- `PipelineStage` gains `CONTEXTUAL_RECONSTRUCTION`; `JobKind` gains `RECONSTRUCTION`.

- [ ] **Step 1: Write model and migration tests.** Assert default values, JSON metadata round trip,
new enum values, revision/down-revision, and migration before/after value tuples. Run a real
upgrade to head, downgrade to `20260905_0007`, then upgrade to head against the Compose database.

- [ ] **Step 2: Verify red.** Run:

```bash
docker compose exec -T backend python -m pytest -q tests/test_models.py tests/test_migrations.py
```

Expected: new fields and migration are absent.

- [ ] **Step 3: Implement model columns.** Use `Text`, `String(64)`, `Float`, and `JSON` with
non-null safe defaults. Do not rename or drop Stage 2.5/raw columns.

- [ ] **Step 4: Implement revision `20260905_0008`.** Add transcript columns, alter the non-native
`source_lifecycle_state`, `pipeline_stage`, and `job_kind` enum/check constraints with explicit
before/after tuples, and backfill contextual text from `corrected_text`. Downgrade restores old
enum constraints only after updating reconstruction lifecycle/job values to compatible values,
then drops only Stage 2.7 columns.

- [ ] **Step 5: Verify migration and tests.** Run:

```bash
docker compose exec -T backend alembic upgrade head
docker compose exec -T backend alembic downgrade 20260905_0007
docker compose exec -T backend alembic upgrade head
docker compose exec -T backend python -m pytest -q tests/test_models.py tests/test_migrations.py
```

Expected: all commands exit zero and existing transcript raw/corrected/final values survive.

- [ ] **Step 6: Commit.**

```bash
git add backend/alembic/versions/20260905_0008_stage_2_7_reconstruction.py \
  backend/app/models/transcript.py backend/app/core/enums.py \
  backend/tests/test_models.py backend/tests/test_migrations.py
git commit -m "Persist Stage 2.7 reconstruction state"
```

### Task 8: Wire durable stage, idempotency, forced reruns, and settings

**Files:**
- Modify: `backend/app/core/settings.py`
- Modify: `.env.example`
- Modify: `backend/app/pipeline/runner.py`
- Modify: `backend/app/pipeline/stages.py`
- Modify: `backend/app/workers/tasks.py`
- Modify: `backend/app/services/source_quality.py`
- Modify: `backend/tests/test_settings.py`
- Modify: `backend/tests/test_pipeline.py`
- Modify: `backend/tests/test_transcription.py`
- Modify: `backend/tests/test_quality.py`
- Modify: `backend/tests/test_stage2_pipeline_e2e.py`

**Interfaces:**
- Produces `Settings.reconstructor() -> ContextualReconstructor`.
- Produces `ContextualReconstructionExecutor.execute(source) -> Transcript`.
- Extends `PipelineRunner.run(..., force: bool = False)` and Celery task with the same flag.
- Stage order becomes normalization → contextual reconstruction → audio analysis.

- [ ] **Step 1: Write settings tests.** Assert disabled default, fully configured provider, missing
base URL/model error, exact thresholds/window/batch values, secret exclusion from fingerprint,
and reference model supplied only through environment.

- [ ] **Step 2: Write executor tests.** Use a fake reconstructor to assert immutable raw text,
IDs/order/timestamps/word timestamps; separate Stage 2.5/reconstruction/final fields; manual
priority; atomic chunk rebuild; aggregate fields; fingerprint cache hit; forced cache bypass; and
rollback when chunk persistence fails.

- [ ] **Step 3: Write runner/task tests.** Prove a normal historical success skips, `force=True`
creates a fresh run, force propagates down retranscription dependencies, the new stage occurs in
the exact order, reconstruction job maps correctly, and cached audio analysis still refreshes
source quality after transcript changes.

- [ ] **Step 4: Verify red.** Run:

```bash
docker compose exec -T backend python -m pytest -q \
  tests/test_settings.py tests/test_pipeline.py tests/test_transcription.py \
  tests/test_quality.py tests/test_stage2_pipeline_e2e.py
```

Expected: new settings/stage/force behavior is absent.

- [ ] **Step 5: Implement executor transaction.** Convert service results to segment JSON, preserve
matching `operator_text`, select final text with the shared helper, replace chunks, calculate
aggregates, and commit once. Matching reconstruction fingerprint bypasses provider work but still
refreshes final/chunks when manual text changed.

- [ ] **Step 6: Implement orchestration force semantics.** A forced run creates a new
`PipelineRun`; it never mutates a historical successful run. Explicit retranscription propagates
force through normalization/reconstruction/analysis. Explicit reconstruction starts at only the
new stage. Audio-analysis cache hits call `assess_source` before returning.

- [ ] **Step 7: Verify green.** Run Task 8 command. Expected: focused and existing Stage 2 E2E
tests pass.

- [ ] **Step 8: Commit.**

```bash
git add .env.example backend/app/core/settings.py backend/app/pipeline/runner.py \
  backend/app/pipeline/stages.py backend/app/workers/tasks.py \
  backend/app/services/source_quality.py backend/tests
git commit -m "Run durable contextual reconstruction stage"
```

### Task 9: Expose reconstruction through API, CLI, and operator UI

**Files:**
- Modify: `backend/app/api/app.py`
- Modify: `backend/app/cli.py`
- Modify: `backend/tests/test_api_sources.py`
- Modify: `backend/tests/test_cli.py`
- Modify: `frontend/src/lib/api-client.ts`
- Modify: `frontend/src/lib/api-client.test.ts`
- Modify: `frontend/src/app/sources/[id]/page.tsx`

**Interfaces:**
- Adds `POST /api/sources/{source_id}/reconstruct?force=false`.
- Adds CLI `reconstruct SOURCE_ID --force/--no-force`.
- Transcript JSON exposes aggregate and per-segment Stage 2.7 evidence.

- [ ] **Step 1: Write API tests.** Assert job kind/status, source-not-found, dispatch stage/force,
response backward compatibility, new fields, manual override priority, clearing override returns
HIGH reconstruction before Stage 2.5, search uses final text, and timestamps remain unchanged.

- [ ] **Step 2: Write CLI tests.** Assert help exposes `reconstruct`; a missing source fails;
default uses cached fingerprint; `--force` clears only reconstruction fingerprint and dispatches
the reconstruction stage.

- [ ] **Step 3: Write frontend client tests.** Assert new field parsing and encoded reconstruct URL:

```typescript
await client.reconstructTranscript("source/id", true);
expect(fetcher).toHaveBeenCalledWith(
  "http://api/api/sources/source%2Fid/reconstruct?force=true",
  { method: "POST" }
);
```

- [ ] **Step 4: Verify red.** Run:

```bash
docker compose exec -T backend python -m pytest -q tests/test_api_sources.py tests/test_cli.py
(cd frontend && npm test -- --run src/lib/api-client.test.ts)
```

Expected: reconstruct endpoint/command/client are absent.

- [ ] **Step 5: Implement backend entry points.** Queue `JobKind.RECONSTRUCTION`, dispatch the new
stage with orchestration force enabled, clear only `reconstruction_fingerprint` when operator asks
for a provider rerun, and return existing typed job response.

- [ ] **Step 6: Implement UI evidence display.** Extend TypeScript types. Keep final text primary;
inside details show Raw, Stage 2.5, Stage 2.7 proposal/applied text, level/score, flags,
method/version, and manual text. Render reconstruct button/job error. Keep `onSeek(segment.start)`.

- [ ] **Step 7: Verify green.** Run Task 9 backend/frontend commands plus:

```bash
(cd frontend && npm run lint && npm run build)
```

Expected: API/CLI/client/UI checks pass.

- [ ] **Step 8: Commit.**

```bash
git add backend/app/api/app.py backend/app/cli.py backend/tests/test_api_sources.py \
  backend/tests/test_cli.py frontend/src/lib/api-client.ts \
  frontend/src/lib/api-client.test.ts frontend/src/app/sources/[id]/page.tsx
git commit -m "Expose contextual reconstruction evidence"
```

### Task 10: Add non-production phrase regression fixtures

**Files:**
- Create: `backend/app/transcription/fixtures/egyptian_ar_reconstruction.json`
- Create: `backend/tests/test_reconstruction_regressions.py`

**Interfaces:**
- Fixture fields: `id`, `category`, `previous`, `raw`, `next`, `expected`,
  `semantic_review`, `must_change`, `protected_tokens`.
- Test helper uses fake two-pass provider output; production modules never import fixture data.

- [ ] **Step 1: Create fixture data.** Include all four supplied families exactly, plus at least
four already-valid cases covering unusual slang, filler repetition, code switching, and an
unfamiliar name. Mark expected unchanged behavior explicitly.

- [ ] **Step 2: Write production-isolation and behavior tests.** Assert no production Python/JSON
outside fixtures contains raw-to-expected mapping pairs. Feed fixture candidates through the real
window, validator, phonetic, confidence, and service layers with a fake provider. Assert IDs,
timestamps, Latin tokens, colloquial forms, changed/unchanged policy, and zero invented clauses.

```python
def test_fixture_phrases_are_not_a_production_replacement_map() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in reconstruction_source_files()
    )
    assert '"دخم": "ضخمة"' not in production
    assert '"ياخدها الرئيس": "كان بيقودها الرئيس"' not in production
```

- [ ] **Step 3: Run regression tests.**

```bash
docker compose exec -T backend python -m pytest -q tests/test_reconstruction_regressions.py
```

Expected: all supplied examples traverse generic logic; valid slang remains unchanged.

- [ ] **Step 4: Commit.**

```bash
git add backend/app/transcription/fixtures/egyptian_ar_reconstruction.json \
  backend/tests/test_reconstruction_regressions.py
git commit -m "Add Stage 2.7 reconstruction regressions"
```

### Task 11: Build unseen-audio benchmark and hard completion gate

**Files:**
- Create: `backend/app/transcription/reconstruction/benchmark.py`
- Create: `backend/tests/test_reconstruction_audio_benchmark.py`
- Modify: `backend/app/cli.py`
- Modify: `backend/tests/test_cli.py`
- Modify: `backend/app/services/storage.py`
- Modify: `backend/tests/test_storage.py`

**Interfaces:**
- Produces `BenchmarkManifest`, `ClipReview`, `BenchmarkReport` Pydantic models.
- Produces `run_reconstruction_benchmark(manifest, reconstructor) -> BenchmarkReport`.
- Produces `evaluate_completion_gate(report) -> tuple[bool, list[str]]`.
- `StorageCategory.BENCHMARKS` resolves private evaluation files below
  `<storage-root>/benchmarks`.
- CLI: `benchmark-reconstruction MANIFEST_NAME`.

- [ ] **Step 1: Write storage and manifest-validation tests.** Assert benchmark paths resolve only
below the new storage category and reject absolute/traversal paths. Reject fewer than five clips, duration outside
120–300 seconds, overlapping clip ranges, fewer than three topics/two source recordings, missing
authorization, absent required categories, any missing human reference/review label, and a test
split used in tuning metadata.

- [ ] **Step 2: Write metric/gate tests.** Cover every outcome and denominator, spelling-tolerant
secondary distance, comprehensibility category floor, performance fields, exact model identifier
and digest, ten-point semantic lift, 25% wrong-segment improvement, 2% regression bound, 98%
correct preservation, zero severe hallucinations, and each failure reason.

```python
passed, reasons = evaluate_completion_gate(report_with(hallucinated=1))
assert passed is False
assert "hallucinated facts/names/numbers/clauses must be zero" in reasons
```

- [ ] **Step 3: Verify red.** Run:

```bash
docker compose exec -T backend python -m pytest -q \
  tests/test_reconstruction_audio_benchmark.py tests/test_cli.py tests/test_storage.py
```

Expected: benchmark module/command are absent.

- [ ] **Step 4: Implement benchmark and review worksheet.** Add
`StorageCategory.BENCHMARKS = "benchmarks"`; resolve the CLI argument relative to that category and
load only storage-owned/private
manifest paths, compare frozen raw/Stage 2.5/Stage 2.7/reference rows, require human labels, and
measure wall time, source minutes, throughput, peak process RAM, and provider-reported VRAM when
available. Print JSON without transcript bodies. Emit `READY FOR STAGE 3` only when gate passes;
otherwise emit `STAGE 2.7 MUST CONTINUE`.

- [ ] **Step 5: Verify green with synthetic records.** Run Task 11 command. Expected: all manifest,
metric, privacy, and gate tests pass without private audio or live provider.

- [ ] **Step 6: Curate and run real private test split.** Through `StorageService`, create a private
manifest from operator-authorized sources containing 5+ non-overlapping clips, 2–5 minutes, 3+
topics, 2+ recordings, and all required speech categories. Freeze raw and Stage 2.5 before prompt
tuning. Have the operator fill human-heard references/review labels for every evaluated segment.
Run the configured local reference provider:

```bash
docker compose exec -T worker python -m app.cli benchmark-reconstruction \
  stage-2-7/unseen-test-v1.json
```

Expected: machine-readable actual metrics plus one exact status line. If any gate fails, do not
relax thresholds against the test split; revise on a separate development split and rerun a fresh
unseen test split.

- [ ] **Step 7: Commit code and synthetic tests only.** Never commit private media/reference rows.

```bash
git add backend/app/transcription/reconstruction/benchmark.py \
  backend/tests/test_reconstruction_audio_benchmark.py backend/app/cli.py backend/tests/test_cli.py \
  backend/app/services/storage.py backend/tests/test_storage.py
git commit -m "Add unseen reconstruction quality gate"
```

### Task 12: Document operations and run full completion audit

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/PIPELINE.md`
- Modify: `docs/BENCHMARKS.md`
- Modify: `docs/ENVIRONMENT.md`
- Modify: `docs/TROUBLESHOOTING.md`
- Modify: `STATUS.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Documentation contains exact environment names, commands, provider failure behavior,
fingerprint rules, model/digest, actual benchmark results, performance cost, limits, and non-goals.

- [ ] **Step 1: Update docs from measured behavior.** Record disabled default, local
OpenAI-compatible/Ollama setup, reference `qwen3:8b` evaluation identifier/digest, both prompts at
temperature 0, final priority, flags, reconstruction endpoint/CLI, private benchmark workflow,
actual latency/RAM/VRAM/throughput, quality outcomes, and the exact completion status. Do not copy
unseen transcript bodies into tracked docs.

- [ ] **Step 2: Format and lint backend.** Run:

```bash
docker compose exec -T backend ruff format app tests
docker compose exec -T backend ruff check app tests
docker compose exec -T backend mypy app
```

Expected: formatter completes; Ruff and mypy exit zero.

- [ ] **Step 3: Run complete backend suite with coverage.** Run:

```bash
docker compose exec -T backend coverage erase
docker compose exec -T backend coverage run --source=app -m pytest
docker compose exec -T backend coverage report --show-missing
```

Expected: all Stage 2/2.5/2.6/2.7 tests pass and configured coverage gate remains satisfied.

- [ ] **Step 4: Run frontend and Compose gates.** Run:

```bash
(cd frontend && npm test)
(cd frontend && npm run lint)
(cd frontend && npm run build)
docker compose config
```

Expected: tests, lint, production build, and Compose validation exit zero.

- [ ] **Step 5: Recheck migrations and invariants.** Run upgrade/downgrade/upgrade from Task 7,
then query one reconstructed transcript and assert raw IDs/order/text/start/end/word timestamps are
byte-for-byte equal to the pre-reconstruction snapshot. Confirm manual overrides still win and
provider-disabled processing reaches `READY_FOR_ANALYSIS`.

- [ ] **Step 6: Audit every completion criterion.** Attach authoritative evidence for all thirteen
brief criteria: context use, material multi-word improvement, timestamps, raw text, colloquial
speech, code switching, entities, hallucinations, unseen benchmark, measured regressions, old
suites, `STATUS.md`/`AGENTS.md`, and performance. Fixture-only evidence is marked insufficient.

- [ ] **Step 7: Commit documentation and final evidence.**

```bash
git add README.md .env.example docs AGENTS.md STATUS.md
git commit -m "Document Stage 2.7 reconstruction operations"
```

- [ ] **Step 8: Report status exactly.** End the implementation report with `READY FOR STAGE 3`
only if Task 11 gate and every Task 12 check passed. Otherwise end with
`STAGE 2.7 MUST CONTINUE` and list failed evidence without claiming completion.

## Plan Self-Review

- Spec coverage: Tasks 1–6 cover acoustic evidence, windows, two passes, phonetics, entities,
  safety, confidence, flags, batching, fallback, and fingerprints. Tasks 7–9 cover persistence,
  durable orchestration, final priority, recovery, API/CLI/UI, and Stage 3 handoff data. Tasks
  10–12 cover regression fixtures, unseen real audio, explicit regression/hallucination metrics,
  performance, documentation, and full old-suite verification.
- Scope: one subsystem—Stage 2.7 transcript reconstruction—with supporting persistence,
  operations, evaluation, and UI evidence. Stage 3 remains excluded.
- Placeholder scan: no unresolved marker, deferred implementation, generic validation instruction,
  or unexplained cross-task shorthand remains.
- Type consistency: `ContextualReconstructor.reconstruct`, `ReconstructionResult`, confidence
  enums, flags, provider methods, final-text helper, and benchmark gate names remain identical
  across producer and consumer tasks.
- Data priority: manual override → applied HIGH contextual reconstruction → Stage 2.5 corrected
  text → raw ASR is consistent in service, persistence, API/UI, benchmark, and docs tasks.
