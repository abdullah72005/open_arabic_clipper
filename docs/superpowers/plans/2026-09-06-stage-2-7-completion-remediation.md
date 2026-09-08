# Stage 2.7 Completion Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a real local Egyptian Arabic reconstruction model run in the live pipeline, repair multi-word errors safely, invalidate stale derived stages, report truthful transcript quality, and prove the result on authorized real audio.

**Architecture:** Extend the existing Stage 2.7 two-pass package rather than replacing it. An Ollama-managed OpenAI-compatible provider supplies structured candidates and per-candidate resolution scores; deterministic routing, safety, state, and dependency-fingerprint code remains authoritative. Persist explicit reconstruction state and separate transcript quality, then drive an actual private benchmark through the same production services.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Alembic, Celery, PostgreSQL, Redis, faster-whisper, Ollama OpenAI-compatible API, Pydantic, pytest, Next.js 15, React 19, TypeScript, Vitest.

**Spec:** `docs/superpowers/specs/2026-09-06-stage-2-7-completion-remediation-design.md`

## Global Constraints

- Do not start Stage 3 clip selection, rendering, publishing, or automatic authorization.
- Process only operator-owned or authorized benchmark media.
- Preserve immutable raw Whisper text, segment IDs/order, segment timestamps, word timestamps, numbers, Latin tokens, repetitions, fillers, and code switching.
- Never summarize, translate, formalize Egyptian Arabic, add facts, or use external story knowledge.
- Final text priority is manual override, applied HIGH Stage 2.7, Stage 2.5, then raw ASR.
- Python runtime is 3.12; host Python 3.10 is unsupported. Use Docker for backend checks.
- CPU-only operation must work. GPU remains optional.
- StorageService owns every application and benchmark path.
- No production phrase map may contain the supplied failures or their expected answers.
- Keep provider transcripts and API keys out of logs.
- Each task starts with a failing test, ends with focused verification, and gets one narrow imperative commit.

---

## File Structure

### New files

- `backend/alembic/versions/20260906_0009_stage_2_7_truth_and_fingerprints.py`: reversible columns for reconstruction state, stage fingerprints, transcript revisions, and split quality.
- `backend/app/pipeline/fingerprints.py`: canonical JSON hashing and stage dependency fingerprints.
- `backend/app/transcription/reconstruction/routing.py`: word/span risk extraction and priority decisions.
- `backend/app/transcription/reconstruction/status.py`: segment and transcript status derivation.
- `backend/app/transcription/reconstruction/ollama.py`: Ollama health, digest lookup, OpenAI-compatible inference adapter, and unload call.
- `backend/tests/test_pipeline_fingerprints.py`: stale-stage and forced-ASR regressions.
- `backend/tests/test_reconstruction_routing.py`: acoustic and high-confidence context-check routing.
- `backend/tests/test_reconstruction_status.py`: truthful segment and aggregate state.
- `backend/tests/test_reconstruction_ollama.py`: health and unload behavior without network access.
- `backend/tests/test_transcript_quality.py`: split-quality math and caps.
- `frontend/src/app/sources/[id]/page.test.tsx`: status and split-quality rendering.

### Focused modifications

- `backend/app/core/enums.py`: persisted reconstruction/provider enums.
- `backend/app/core/settings.py`: enabled Ollama defaults and versioned routing/provider settings.
- `backend/app/models/{pipeline_run,transcript,audio_analysis,source_quality_assessment}.py`: new persisted fingerprints and quality/status columns.
- `backend/app/pipeline/{executor,runner,stages}.py`: force-aware execution and fingerprint checks.
- `backend/app/workers/tasks.py`: live provider construction and fingerprint-driven successor chain.
- `backend/app/transcription/{engine,service}.py`: ASR release and revision-producing fingerprints.
- `backend/app/transcription/reconstruction/{types,windows,providers,confidence,service,benchmark}.py`: complete evidence payload, batching, resolution, statuses, and actual benchmark execution.
- `backend/app/services/{health,source_quality,storage}.py`: provider health and separate quality.
- `backend/app/api/app.py`, `backend/app/cli.py`: truthful API/CLI state and force requests without cache sentinels.
- `frontend/src/lib/{api-client.ts,api-client.test.ts}` and source detail: typed state and operator visibility.
- `compose.yaml`, `.env.example`: Ollama profile and active reconstruction defaults.
- `README.md`, `STATUS.md`, `AGENTS.md`, `docs/{ARCHITECTURE,BENCHMARKS,ENVIRONMENT,LOCAL_SETUP,PIPELINE,TROUBLESHOOTING}.md`: exact operations and measured evidence.

---

### Task 1: Persist truthful state and dependency metadata

**Files:**

- Create: `backend/alembic/versions/20260906_0009_stage_2_7_truth_and_fingerprints.py`
- Modify: `backend/app/core/enums.py:1-53`
- Modify: `backend/app/models/pipeline_run.py:17-67`
- Modify: `backend/app/models/transcript.py:17-66`
- Modify: `backend/app/models/audio_analysis.py:17-39`
- Modify: `backend/app/models/source_quality_assessment.py:17-41`
- Modify: `backend/tests/test_models.py`
- Modify: `backend/tests/test_migrations.py`
- Create: `backend/tests/test_reconstruction_status.py`
- Create: `backend/app/transcription/reconstruction/status.py`

**Interfaces:**

- Produces `ProviderAvailability`, `ReconstructionStatus`, and `aggregate_reconstruction_status`.
- Produces nullable historic stage fingerprints, transcript revision/fingerprints, audio-analysis fingerprint, and quality evidence fields used by Tasks 6–9.
- Migration revision is `20260906_0009`; down revision is `20260905_0008`.

- [ ] **Step 1: Write failing enum, model, aggregation, and migration tests**

Add assertions with these exact behaviors:

```python
def test_reconstruction_status_uses_worst_first_precedence() -> None:
    assert aggregate_reconstruction_status([
        ReconstructionStatus.APPLIED,
        ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
    ]) is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
    assert aggregate_reconstruction_status([
        ReconstructionStatus.PROVIDER_UNAVAILABLE,
        ReconstructionStatus.FAILED,
    ]) is ReconstructionStatus.FAILED

def test_stage_2_7_truth_columns_exist() -> None:
    assert {"input_fingerprint", "output_fingerprint"} <= set(PipelineRun.__table__.columns.keys())
    assert {"transcription_revision", "normalization_fingerprint", "reconstruction_status"} <= set(Transcript.__table__.columns.keys())
    assert "input_fingerprint" in AudioAnalysis.__table__.columns
    assert {
        "transcript_quality_score", "low_confidence_word_ratio",
        "unresolved_segment_ratio", "manual_review_required", "input_fingerprint",
    } <= set(SourceQualityAssessment.__table__.columns.keys())
```

Extend migration tests to upgrade from `20260905_0008`, inspect every column above,
downgrade to `20260905_0008`, and confirm only revision `0009` columns disappear.

- [ ] **Step 2: Run tests and confirm missing enums/columns fail**

Run:

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest pytest-asyncio httpx >/dev/null && \
   pytest tests/test_models.py tests/test_migrations.py tests/test_reconstruction_status.py -q"
```

Expected: collection or assertions fail because new state and columns do not exist.

- [ ] **Step 3: Add exact enums and status aggregation**

Add to `core/enums.py`:

```python
class ProviderAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    MISCONFIGURED = "MISCONFIGURED"

class ReconstructionStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    APPLIED = "APPLIED"
    UNCHANGED_HIGH_CONFIDENCE = "UNCHANGED_HIGH_CONFIDENCE"
    LOW_CONFIDENCE_UNRESOLVED = "LOW_CONFIDENCE_UNRESOLVED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    FAILED = "FAILED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
```

Implement `status.py` with this precedence:

```python
_PRECEDENCE = {
    ReconstructionStatus.NOT_REQUIRED: 0,
    ReconstructionStatus.UNCHANGED_HIGH_CONFIDENCE: 1,
    ReconstructionStatus.MANUAL_OVERRIDE: 2,
    ReconstructionStatus.APPLIED: 3,
    ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED: 4,
    ReconstructionStatus.PROVIDER_UNAVAILABLE: 5,
    ReconstructionStatus.FAILED: 6,
}

def aggregate_reconstruction_status(
    values: Sequence[ReconstructionStatus],
) -> ReconstructionStatus:
    return max(values, key=_PRECEDENCE.__getitem__) if values else ReconstructionStatus.NOT_REQUIRED
```

- [ ] **Step 4: Add model fields and reversible migration**

Use non-native SQLAlchemy enums for `Transcript.reconstruction_status`; server default
must be `NOT_REQUIRED`. Add:

```text
pipeline_runs.input_fingerprint VARCHAR(64) NULL
pipeline_runs.output_fingerprint VARCHAR(64) NULL
transcripts.transcription_revision INTEGER NOT NULL DEFAULT 0
transcripts.normalization_fingerprint VARCHAR(64) NOT NULL DEFAULT ''
transcripts.reconstruction_status VARCHAR enum NOT NULL DEFAULT 'NOT_REQUIRED'
audio_analyses.input_fingerprint VARCHAR(64) NOT NULL DEFAULT ''
source_quality_assessments.transcript_quality_score FLOAT NOT NULL DEFAULT 0
source_quality_assessments.low_confidence_word_ratio FLOAT NOT NULL DEFAULT 0
source_quality_assessments.unresolved_segment_ratio FLOAT NOT NULL DEFAULT 0
source_quality_assessments.manual_review_required BOOLEAN NOT NULL DEFAULT true
source_quality_assessments.input_fingerprint VARCHAR(64) NOT NULL DEFAULT ''
```

Add `CheckConstraint` bounds for all new `[0,1]` ratio/score columns and
`transcription_revision >= 0`. Downgrade removes only these fields.

- [ ] **Step 5: Run focused tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/alembic/versions/20260906_0009_stage_2_7_truth_and_fingerprints.py \
  backend/app/core/enums.py backend/app/models backend/app/transcription/reconstruction/status.py \
  backend/tests/test_models.py backend/tests/test_migrations.py \
  backend/tests/test_reconstruction_status.py
git commit -m "Persist truthful Stage 2.7 state"
```

---

### Task 2: Add a managed Ollama provider and active defaults

**Files:**

- Create: `backend/app/transcription/reconstruction/ollama.py`
- Modify: `backend/app/transcription/reconstruction/providers.py:1-249`
- Modify: `backend/app/transcription/reconstruction/types.py:1-88`
- Modify: `backend/app/core/settings.py:16-145`
- Modify: `backend/app/services/health.py`
- Modify: `backend/app/api/app.py:566-625`
- Modify: `backend/app/cli.py:1-164`
- Modify: `backend/tests/test_reconstruction_provider.py`
- Create: `backend/tests/test_reconstruction_ollama.py`
- Modify: `backend/tests/test_settings.py`
- Modify: `backend/tests/test_health.py`
- Modify: `.env.example:27-32`
- Modify: `compose.yaml:1-68`

**Interfaces:**

- Produces `ProviderHealth` and managed provider methods `health()` and `release()`.
- Settings produce `OllamaReconstructionProvider` by default and retain explicit
  `openai_compatible`/`disabled` modes.
- API health consumes a zero-transcript provider health probe.

- [ ] **Step 1: Write failing provider lifecycle and settings tests**

Use injected byte-request callables; no test opens a socket:

```python
from dataclasses import dataclass
from urllib.parse import urlsplit

@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    body: bytes | None

class StubHttp:
    def __init__(self, responses: dict[tuple[str, str], bytes]) -> None:
        self.responses = responses
        self.calls: list[RecordedRequest] = []

    def __call__(
        self, method: str, url: str, body: bytes | None,
        headers: dict[str, str], timeout: float,
    ) -> bytes:
        del headers, timeout
        path = urlsplit(url).path
        self.calls.append(RecordedRequest(method, path, body))
        return self.responses[(method, path)]

def test_ollama_health_requires_exact_model_and_returns_digest() -> None:
    request = StubHttp({
        ("GET", "/api/tags"): b'{"models":[{"name":"qwen3:8b","digest":"sha256:abc"}]}',
    })
    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434", model="qwen3:8b", timeout_seconds=3, request=request
    )
    assert provider.health() == ProviderHealth(
        ProviderAvailability.AVAILABLE, "ollama", "qwen3:8b", "sha256:abc", "model available"
    )

def test_ollama_release_unloads_configured_model() -> None:
    request = StubHttp({("POST", "/api/generate"): b'{"done":true}'})
    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434", model="qwen3:8b", timeout_seconds=3, request=request
    )
    provider.release()
    assert json.loads(request.calls[-1].body or b"{}") == {
        "model": "qwen3:8b", "keep_alive": 0
    }

def test_reconstruction_defaults_to_managed_local_provider() -> None:
    settings = Settings(_env_file=None)
    assert settings.reconstruction_provider == "ollama"
    assert settings.reconstruction_provider_model == "qwen3:8b"
    assert settings.reconstruction_provider_timeout_seconds == 180
    assert settings.reconstruction_release_after_run is True
```

Add health tests proving a missing model yields a `DEGRADED` reconstruction check with
detail `configured model qwen3:8b is not installed`.

- [ ] **Step 2: Run focused tests and confirm failure**

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest pytest-asyncio httpx >/dev/null && \
   pytest tests/test_reconstruction_provider.py tests/test_reconstruction_ollama.py \
   tests/test_settings.py tests/test_health.py -q"
```

Expected: FAIL because Ollama provider, lifecycle types, and defaults are absent.

- [ ] **Step 3: Implement provider health and lifecycle contracts**

Add these exact immutable values to `types.py`:

```python
@dataclass(frozen=True)
class ProviderHealth:
    availability: ProviderAvailability
    provider: str
    model: str | None
    model_digest: str | None
    detail: str
```

Change `HttpRequest` to
`Callable[[str, str, bytes | None, dict[str, str], float], bytes]`, where the first
argument is the HTTP method. Extend `ReconstructionProvider` with `health()` and
`release()`. The generic
OpenAI-compatible provider calls `/v1/models`, marks connection/JSON/absent-model
failures `UNAVAILABLE`, returns the exact matching model ID, and makes `release()` a
no-op. Keep secrets out of `ProviderHealth.detail`.

- [ ] **Step 4: Implement Ollama management without changing inference protocol**

`OllamaReconstructionProvider` subclasses or delegates to
`OpenAICompatibleReconstructionProvider`. It uses the same `/v1/chat/completions`
generation and resolution methods. Implement:

```python
def health(self) -> ProviderHealth:
    payload = self._json_request("GET", "/api/tags", None)
    match = next((item for item in payload.get("models", []) if item.get("name") == self.model), None)
    if match is None:
        return ProviderHealth(ProviderAvailability.UNAVAILABLE, "ollama", self.model, None,
                              f"configured model {self.model} is not installed")
    return ProviderHealth(ProviderAvailability.AVAILABLE, "ollama", self.model,
                          str(match.get("digest") or "") or None, "model available")

def release(self) -> None:
    if self.release_after_run:
        self._json_request("POST", "/api/generate", {"model": self.model, "keep_alive": 0})
```

Map timeouts, connection refusal, non-2xx responses, invalid JSON, and missing `models`
to `ProviderHealth(UNAVAILABLE, ...)`; do not expose response bodies.

- [ ] **Step 5: Activate safe product defaults and Compose profile**

Use `Literal["disabled", "openai_compatible", "ollama"]` with default `ollama`, base
URL `http://ollama:11434`, model `qwen3:8b`, timeout `180.0`, release `True`. Add an
`ollama/ollama` service named `ollama`, volume `ollama_models:/root/.ollama`, no public
port, profile `reconstruction`, and no automatic model-pull command. Add
`ollama_models:` to volumes. `.env.example` must exactly match settings.

- [ ] **Step 6: Surface provider health through CLI and system health**

Add `python -m app.cli reconstruction-health`. Print JSON with availability, provider,
model, digest, and detail. Raise `typer.Exit(1)` unless availability is `AVAILABLE`.
Add the same probe as `reconstruction_provider` in `/system/health`; map `AVAILABLE` to
`HEALTHY`, everything else to `DEGRADED`.

- [ ] **Step 7: Run focused tests and validate Compose**

Run Step 2 plus:

```bash
docker compose config --profiles
docker compose --profile reconstruction config --services
```

Expected: tests PASS; output includes profile/service `reconstruction`/`ollama`.

- [ ] **Step 8: Commit**

```bash
git add backend/app/transcription/reconstruction backend/app/core/settings.py \
  backend/app/services/health.py backend/app/api/app.py backend/app/cli.py backend/tests \
  .env.example compose.yaml
git commit -m "Enable managed local reconstruction"
```

---

### Task 3: Release ASR and provider memory deterministically

**Files:**

- Modify: `backend/app/transcription/engine.py:27-113`
- Modify: `backend/app/transcription/reconstruction/service.py:29-84`
- Modify: `backend/tests/test_transcription.py`
- Modify: `backend/tests/test_reconstruction_service.py`

**Interfaces:**

- Consumes managed `ReconstructionProvider.release()` from Task 2.
- Guarantees provider release after success or expected provider failure.

- [ ] **Step 1: Write failing release tests**

```python
class LazyWhisperModel:
    exhausted = False

    def transcribe(self, path: str, **kwargs: object) -> tuple[Iterator[object], object]:
        del path, kwargs
        def rows() -> Iterator[object]:
            yield SimpleNamespace(start=0.0, end=1.0, text=" كلام", words=[])
            self.exhausted = True
        return rows(), SimpleNamespace(language="ar", language_probability=0.99, duration=1.0)

def test_whisper_materializes_segments_before_collecting_model() -> None:
    model = LazyWhisperModel()
    collected: list[bool] = []
    engine = WhisperEngine(
        model_factory=lambda *_: model,
        cuda_available=lambda: False,
        collect_garbage=lambda: collected.append(model.exhausted) or 0,
    )
    result = engine.transcribe(Path("speech.wav"), TranscriptionOptions(
        model="small", device="cpu", compute_type="int8", beam_size=5,
    ))
    assert result.segments
    assert collected == [True]

class ReleasingBrokenProvider:
    release_calls = 0

    def health(self) -> ProviderHealth:
        return ProviderHealth(ProviderAvailability.AVAILABLE, "test", "test", "sha256:x", "ok")

    def generate_candidates(self, requests: list[GenerationRequest]) -> dict[int, list[ReconstructionCandidate]]:
        raise ProviderResponseError("invalid JSON")

    def resolve_candidates(self, requests: list[ResolutionRequest]) -> dict[int, ResolutionChoice]:
        raise AssertionError("resolution must not run")

    def release(self) -> None:
        self.release_calls += 1

def test_reconstructor_releases_provider_after_failure() -> None:
    provider = ReleasingBrokenProvider()
    ContextualReconstructor(provider).reconstruct([{
        "start": 0.0, "end": 1.0, "text": "كلام", "corrected_text": "كلام"
    }], language="ar",
        transcription_fingerprint="asr", correction_version="stage25")
    assert provider.release_calls == 1
```

- [ ] **Step 2: Run tests and confirm release assertions fail**

Run:

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest >/dev/null && \
   pytest tests/test_transcription.py tests/test_reconstruction_service.py -q"
```

- [ ] **Step 3: Materialize then release Whisper**

Inject `collect_garbage: Callable[[], int] = gc.collect`. Keep `model` local. Consume all
segments and words before return, then execute `del model` and `collect_garbage()` in a
`finally` block. Do not release before the lazy faster-whisper iterator finishes.

- [ ] **Step 4: Release reconstruction provider in one finally block**

Probe and use the provider inside `try`, preserve typed unavailable fallback behavior,
then call `self._provider.release()` in `finally`. If release fails, record a bounded
metadata warning but do not discard valid reconstruction results. Never include request
text in the warning.

- [ ] **Step 5: Run tests and commit**

Run Step 2. Expected: PASS.

```bash
git add backend/app/transcription/engine.py backend/app/transcription/reconstruction/service.py \
  backend/tests/test_transcription.py backend/tests/test_reconstruction_service.py
git commit -m "Release local inference resources"
```

---

### Task 4: Route uncertain spans with complete context evidence

**Files:**

- Create: `backend/app/transcription/reconstruction/routing.py`
- Create: `backend/tests/test_reconstruction_routing.py`
- Modify: `backend/app/transcription/reconstruction/types.py:10-88`
- Modify: `backend/app/transcription/reconstruction/windows.py:16-127`
- Modify: `backend/app/transcription/reconstruction/providers.py:17-249`
- Modify: `backend/app/core/settings.py`
- Modify: `backend/tests/test_reconstruction_windows.py`
- Modify: `backend/tests/test_reconstruction_provider.py`

**Interfaces:**

- Produces `RoutingPriority`, `WordRisk`, `RoutingEvidence`, `RoutingDecision`, and
  `route_segment(segment, config)`.
- `GenerationRequest` consumes full `ReconstructionWindow`, language, entity forms, and
  routing decision rather than flattened raw-only strings.

- [ ] **Step 1: Write failing Chernobyl acoustic-routing tests**

```python
def test_multiple_weak_words_force_reconstruction_priority() -> None:
    segment = {"avg_logprob": -0.2188539335, "words": [
        {"word": "يا", "start": 5.8, "end": 6.0, "probability": 0.5286},
        {"word": "جماعة", "start": 6.0, "end": 6.46, "probability": 0.9901},
        {"word": "مفيش", "start": 6.46, "end": 6.78, "probability": 0.8688},
        {"word": "دقل", "start": 6.78, "end": 6.98, "probability": 0.5373},
        {"word": "للزعو", "start": 6.98, "end": 7.28, "probability": 0.3807},
    ]}
    decision = route_segment(segment, RoutingConfig())
    assert decision.priority is RoutingPriority.RECONSTRUCT
    assert {span.text.strip() for span in decision.focus_spans} >= {"دقل", "للزعو"}
    assert "multiple_low_probability_words" in decision.reasons

def test_high_confidence_arabic_still_gets_context_check() -> None:
    decision = route_segment({"text": "عبارة محتملة", "words": [
        {"word": "عبارة", "probability": 0.98}, {"word": "محتملة", "probability": 0.98}
    ]}, RoutingConfig(), language="ar")
    assert decision.priority is RoutingPriority.CONTEXT_CHECK
```

Add request serialization assertions for segment IDs, raw and corrected text, previous
and following windows, word probabilities, language, repeated entities, routing reasons,
and focus spans.

- [ ] **Step 2: Run tests and confirm routing/request failures**

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest >/dev/null && \
   pytest tests/test_reconstruction_routing.py tests/test_reconstruction_windows.py \
   tests/test_reconstruction_provider.py -q"
```

- [ ] **Step 3: Implement versioned routing configuration and formula**

Use exact defaults from the spec: 0.72 low, 0.50 very low, 0.78 low mean, 0.25 high
ratio, 0.45 score. Implement the five weighted terms and OR conditions verbatim. Missing
word probabilities must not become false low confidence; use available segment evidence
and route Arabic to `CONTEXT_CHECK`.

- [ ] **Step 4: Expand window/request types without changing timestamps**

Add `WordEvidence(text, start, end, probability)` to `WindowSegment`. Carry raw and
corrected previous/following context as `WindowSegment` values. Pass language, exact
source-local entity forms, and `RoutingDecision`. Provider JSON must not offer any field
for changed timestamps.

- [ ] **Step 5: Activate batch bounds**

Implement `batch_generation_requests(requests, max_windows=8, max_characters=24_000)`.
Count UTF-8 decoded characters from deterministic `json.dumps(..., ensure_ascii=False,
sort_keys=True)`. Reject configured values above hard maxima 16/48,000 in Settings.
Sort `RECONSTRUCT` before `CONTEXT_CHECK`; persist results in original segment order.

- [ ] **Step 6: Tighten prompts**

Pass A system prompt must include these literal rules:

```text
Reconstruct only what was most plausibly spoken in the target segment.
Preserve Egyptian Arabic; do not rewrite into MSA.
Do not summarize, paraphrase stylistically, translate, add facts, names, numbers, or clauses.
Use focus_spans and surrounding raw/corrected evidence. Return zero candidates when unchanged is safer.
```

For `ollama`, include `reasoning_effort="none"`. When the configured model identifier
starts with `qwen3`, also end the system instruction with `/no_think`. Generic
`openai_compatible` requests omit the Ollama-specific reasoning field. Keep temperature
zero in every provider.

- [ ] **Step 7: Run tests and commit**

Run Step 2. Expected: PASS.

```bash
git add backend/app/transcription/reconstruction/routing.py \
  backend/app/transcription/reconstruction/types.py \
  backend/app/transcription/reconstruction/windows.py \
  backend/app/transcription/reconstruction/providers.py backend/app/core/settings.py \
  backend/tests/test_reconstruction_routing.py backend/tests/test_reconstruction_windows.py \
  backend/tests/test_reconstruction_provider.py
git commit -m "Route uncertain transcript spans"
```

---

### Task 5: Resolve multi-word candidates with real margins and truthful outcomes

**Files:**

- Modify: `backend/app/transcription/reconstruction/types.py`
- Modify: `backend/app/transcription/reconstruction/providers.py`
- Modify: `backend/app/transcription/reconstruction/confidence.py:17-54`
- Modify: `backend/app/transcription/reconstruction/validation.py:17-69`
- Modify: `backend/app/transcription/reconstruction/service.py:29-211`
- Modify: `backend/tests/test_reconstruction_provider.py`
- Modify: `backend/tests/test_reconstruction_confidence.py`
- Modify: `backend/tests/test_reconstruction_validation.py`
- Modify: `backend/tests/test_reconstruction_service.py`
- Modify: `backend/tests/test_reconstruction_regressions.py`
- Modify: `backend/app/transcription/fixtures/egyptian_ar_reconstruction.json`

**Interfaces:**

- `ResolutionChoice` produces scores for every candidate and selected ID.
- `SegmentReconstruction` produces `status`, routing evidence, model method/version, and
  validated change evidence.
- Consumes status precedence from Task 1 and routing decisions from Task 4.

- [ ] **Step 1: Add failing real-margin and multi-word provider regressions**

Add the three exact reported cases to fixture-only data. A fake provider must propose:

```json
[
  {"raw":"عملية إخلاق مؤقت","expected":"عملية إخلاء مؤقت"},
  {"raw":"ثلاث يام بس لتطهير النطقة","expected":"تلات أيام بس لتطهير المنطقة"},
  {"raw":"يا جماعة مفيش دقل للزعو","expected":"يا جماعة مفيش داعي للذعر"}
]
```

Test that low word probabilities route the latter two, all three retain original
start/end/words, and the accepted HIGH result becomes final. Also test that a selected
candidate with a computed 0.04 margin is not applied even if provider-reported selection
confidence is 1.0.

- [ ] **Step 2: Run focused tests and confirm failure**

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest >/dev/null && \
   pytest tests/test_reconstruction_provider.py tests/test_reconstruction_confidence.py \
   tests/test_reconstruction_validation.py tests/test_reconstruction_service.py \
   tests/test_reconstruction_regressions.py -q"
```

- [ ] **Step 3: Return and validate scores for every candidate**

Change Pass B schema to one resolution per target containing `selected_candidate_id`
and `candidate_scores`. Reject missing, duplicate, foreign, or incomplete candidate IDs.
Require one score record for every supplied candidate. Compute deterministic final scores
for all valid candidates, sort by `(-score, candidate_id)`, and calculate:

```python
margin = ranked[0].score - ranked[1].score if len(ranked) > 1 else ranked[0].score
```

Delete the hard-coded `margin=0.12` call.

- [ ] **Step 4: Preserve multi-word safety while allowing the examples**

Keep protected Latin/digit equality, 0.60–1.60 length ratio, token-delta bound, entity
evidence, and phonetic floor. Calculate phonetic similarity across aligned 1–3-token
spans. Do not add any phrase-specific production rule. Mark accepted boundary changes
`MULTIWORD_RECONSTRUCTION`.

- [ ] **Step 5: Derive exact segment statuses**

Use:

```python
if operator_text:
    status = ReconstructionStatus.MANUAL_OVERRIDE
elif provider_health.availability is not ProviderAvailability.AVAILABLE:
    status = ReconstructionStatus.PROVIDER_UNAVAILABLE
elif applied:
    status = ReconstructionStatus.APPLIED
elif routing.priority is RoutingPriority.SKIP:
    status = ReconstructionStatus.NOT_REQUIRED
elif routing.priority is RoutingPriority.RECONSTRUCT or candidate_rejected:
    status = ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
else:
    status = ReconstructionStatus.UNCHANGED_HIGH_CONFIDENCE
```

Expected provider failures return `PROVIDER_UNAVAILABLE`; unexpected application errors
remain exceptions so the durable stage becomes `FAILED`. Do not use
`stage2_5_fallback` as an outcome.

- [ ] **Step 6: Persist method and audit evidence in result types**

Set `reconstruction_method` to `provider:model`, for example `ollama:qwen3:8b`.
Store status, candidate, applied flag, confidence, confidence level, routing score/reasons,
focus spans, validated changes, and flags. Bound focus evidence to words in the target.

- [ ] **Step 7: Run tests and commit**

Run Step 2. Expected: PASS.

```bash
git add backend/app/transcription/reconstruction backend/app/transcription/fixtures \
  backend/tests/test_reconstruction_*.py
git commit -m "Apply contextual multi-word reconstruction"
```

---

### Task 6: Persist reconstruction state atomically and expose it through API and CLI

**Files:**

- Modify: `backend/app/pipeline/stages.py:280-403`
- Modify: `backend/app/api/app.py:85-113,331-386,501-526,550-560`
- Modify: `backend/app/cli.py:65-120`
- Modify: `backend/tests/test_reconstruction_persistence.py`
- Modify: `backend/tests/test_api_sources.py`
- Modify: `backend/tests/test_cli.py`

**Interfaces:**

- Consumes Task 5 `ReconstructionResult` and status aggregation.
- Produces transcript-level status/method/metadata and per-segment debug fields for Tasks
  8–9.

- [ ] **Step 1: Write failing unavailable, applied, manual, and timestamp tests**

Assert a provider-unavailable execution persists:

```python
assert transcript.reconstruction_status is ReconstructionStatus.PROVIDER_UNAVAILABLE
assert transcript.reconstruction_method == "ollama:qwen3:8b"
assert transcript.reconstruction_metadata["provider_availability"] == "UNAVAILABLE"
assert transcript.reconstruction_metadata["unresolved_segments"] > 0
assert transcript.segments[0]["reconstruction_status"] == "PROVIDER_UNAVAILABLE"
```

Snapshot raw `text`, start/end, nested words, and flattened `word_segments` before
reconstruction and assert deep equality afterward. Verify manual override remains final
and its segment status is `MANUAL_OVERRIDE`.

- [ ] **Step 2: Run focused tests and confirm failure**

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest pytest-asyncio httpx >/dev/null && \
   pytest tests/test_reconstruction_persistence.py tests/test_api_sources.py tests/test_cli.py -q"
```

- [ ] **Step 3: Persist complete segment and aggregate state**

In `ContextualReconstructionExecutor`, copy original segment dictionaries and add only
derived keys. Never assign `text`, `start`, `end`, `words`, or transcript
`word_segments`. Aggregate status with Task 1. Metadata must include counts by status,
provider availability, model, digest, routed/unresolved/applied counts, batch count,
release warning, and algorithm versions; never include transcript text or API keys.

- [ ] **Step 4: Make API/CLI force requests stop mutating caches**

Delete the blocks that set `input_fingerprint = ""` and
`reconstruction_fingerprint = ""`. Queue jobs with the requested `force` value only.
Expose `reconstruction_status` in `TranscriptResponse` and transcript CLI JSON. Add a
`--force/--no-force` option to reconstruct and retranscribe with existing defaults.

- [ ] **Step 5: Run tests and commit**

Run Step 2. Expected: PASS.

```bash
git add backend/app/pipeline/stages.py backend/app/api/app.py backend/app/cli.py \
  backend/tests/test_reconstruction_persistence.py backend/tests/test_api_sources.py \
  backend/tests/test_cli.py
git commit -m "Expose truthful reconstruction outcomes"
```

---

### Task 7: Replace historical-success caching with dependency fingerprints

**Files:**

- Create: `backend/app/pipeline/fingerprints.py`
- Create: `backend/tests/test_pipeline_fingerprints.py`
- Modify: `backend/app/pipeline/executor.py`
- Modify: `backend/app/pipeline/runner.py:37-184`
- Modify: `backend/app/pipeline/stages.py:31-467`
- Modify: `backend/app/workers/tasks.py:27-120`
- Modify: `backend/app/transcription/service.py:10-42`
- Modify: `backend/app/transcription/reconstruction/service.py:139-172`
- Modify: `backend/tests/test_pipeline.py`
- Modify: `backend/tests/test_transcription.py`
- Modify: `backend/tests/test_stage2_pipeline_e2e.py`

**Interfaces:**

- Produces `canonical_fingerprint(namespace, version, payload)`,
  `StageExecutionResult(output_fingerprint)`, and force-aware `StageExecutor`.
- Downstream stages consume current upstream output fingerprints and transcript revision.

- [ ] **Step 1: Reproduce the exact stale-normalization bug in a failing test**

Build a source with successful historical TRANSCRIPTION, NORMALIZATION, RECONSTRUCTION,
and AUDIO_ANALYSIS runs. Seed old normalized text. Force transcription with fake turbo
output. Execute the queued successor chain synchronously. Assert:

```python
assert transcript.transcription_revision == 2
assert transcript.segments[0]["text"] == "new turbo raw"
assert transcript.segments[0]["raw_text"] == "new turbo raw"
assert transcript.normalized_text == "new turbo raw"
assert normalization_executor.calls == 1
assert reconstruction_executor.calls == 1
assert quality_assessor.calls == 1
```

Also assert an unchanged non-forced run makes zero executor calls, while a forced ASR
whose text is identical still increments revision and refreshes downstream fingerprints.

- [ ] **Step 2: Run focused tests and confirm stale outputs remain**

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest pytest-asyncio httpx >/dev/null && \
   pytest tests/test_pipeline.py tests/test_pipeline_fingerprints.py \
   tests/test_transcription.py tests/test_stage2_pipeline_e2e.py -q"
```

- [ ] **Step 3: Implement canonical hashes and executor contract**

```python
def canonical_fingerprint(namespace: str, version: str, payload: Mapping[str, object]) -> str:
    body = {"namespace": namespace, "version": version, "payload": payload}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class StageExecutionResult:
    output_fingerprint: str

class StageExecutor(Protocol):
    def input_fingerprint(self, source: SourceVideo) -> str: ...
    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult: ...
```

Update every concrete and test executor. Output fingerprints hash persisted outputs, not
timestamps or processing duration.

- [ ] **Step 4: Make runner skip only matching current input**

Resolve executor before skip. Compute input. Skip only when latest run succeeded, force
is false, input is non-empty, and stored input equals current input. On a forced rerun,
increment attempt even after previous success. Store input before execution and output
after success. Historic `NULL` inputs rerun once.

- [ ] **Step 5: Implement exact dependency chain**

Include fields from the design's fingerprint chain. `TranscriptionExecutor.execute`
bypasses its internal fingerprint cache when force is true and increments
`transcription_revision` after a complete result. Normalization includes that revision,
so forced identical text still reruns. Reconstruction includes normalization fingerprint
and provider digest. Audio analysis includes transcript revision. Quality fingerprint is
computed separately in Task 8.

- [ ] **Step 6: Keep successor force false and trust fingerprints**

Do not forward force to `_NEXT_STAGE`. `run_pipeline_stage` queues the successor with
`force=False`; mismatched dependency fingerprints decide execution. A skipped stage still
queues its successor so a later stale dependency can self-heal.

- [ ] **Step 7: Run focused tests and commit**

Run Step 2. Expected: PASS, including exact stale-normalization regression.

```bash
git add backend/app/pipeline backend/app/workers/tasks.py backend/app/transcription \
  backend/tests/test_pipeline.py backend/tests/test_pipeline_fingerprints.py \
  backend/tests/test_transcription.py backend/tests/test_stage2_pipeline_e2e.py
git commit -m "Invalidate transcript-derived pipeline stages"
```

---

### Task 8: Separate transcript quality from audio quality

**Files:**

- Modify: `backend/app/services/source_quality.py:1-73`
- Modify: `backend/app/pipeline/stages.py:405-467`
- Modify: `backend/tests/test_quality.py`
- Create: `backend/tests/test_transcript_quality.py`

**Interfaces:**

- Produces `assess_transcript_quality(transcript) -> TranscriptQualityEvidence` and a
  fingerprinted `assess_source` result.
- Consumes per-segment reconstruction status/routing/confidence from Task 6.

- [ ] **Step 1: Write failing quality separation and bad-sample tests**

```python
def transcript_with(
    *, probabilities: list[float], statuses: list[str] | None = None,
    provider_availability: str = "AVAILABLE",
) -> Transcript:
    statuses = statuses or ["UNCHANGED_HIGH_CONFIDENCE"]
    segments = [{
        "start": 0.0,
        "end": 3.0,
        "text": " كلام",
        "corrected_text": "كلام",
        "reconstruction_status": statuses[min(index, len(statuses) - 1)],
        "reconstruction_confidence": 0.0,
        "routing_priority": "RECONSTRUCT",
        "words": [{"word": f" w{word}", "probability": probability}
                  for word, probability in enumerate(probabilities)],
    } for index in range(len(statuses))]
    return Transcript(
        whisper_model="large-v3-turbo", transcription_options={}, input_fingerprint="a" * 64,
        raw_text="كلام", normalized_text="كلام", corrected_text="كلام", final_text="كلام",
        language="ar", detected_language_probability=0.99, duration=3.0,
        segments=segments, word_segments=[],
        reconstruction_metadata={"provider_availability": provider_availability},
    )

def perfect_audio(source_id: UUID) -> AudioAnalysis:
    return AudioAnalysis(
        source_video_id=source_id, audio_hash="b" * 64, silence_intervals=[], features=[],
        silence_ratio=0.015, speech_density=0.985, speech_rate=120.0,
    )

def test_good_audio_cannot_mask_unavailable_uncertain_transcript(sqlite_engine: object) -> None:
    transcript = transcript_with(
        probabilities=[0.99, 0.53, 0.38],
        statuses=["PROVIDER_UNAVAILABLE"],
        provider_availability="UNAVAILABLE"
    )
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(source_uri="/imports/bad.mp4")
        session.add(source)
        session.flush()
        transcript.source_video_id = source.id
        analysis = perfect_audio(source.id)
        session.add_all([transcript, analysis])
        session.commit()
        assessment = assess_source(session, source, transcript, analysis)
        assert assessment.audio_quality_score >= 0.95
        assert assessment.transcript_quality_score <= 0.40
        assert assessment.overall_source_quality_score == assessment.transcript_quality_score
        assert assessment.manual_review_required is True

def test_low_confidence_word_ratio_uses_configured_boundary() -> None:
    evidence = assess_transcript_quality(transcript_with(probabilities=[0.71, 0.72, 0.90]))
    assert evidence.low_confidence_word_ratio == pytest.approx(1 / 3)
```

Add tests for APPLIED blending, unresolved cap 0.45, unavailable routed cap 0.40,
FAILED cap 0.25, manual segment 0.95, excluded NOT_REQUIRED, and missing confidence.

- [ ] **Step 2: Run tests and confirm the blended score fails**

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest >/dev/null && \
   pytest tests/test_quality.py tests/test_transcript_quality.py -q"
```

- [ ] **Step 3: Implement segment-weighted transcript quality exactly**

Create an immutable `TranscriptQualityEvidence(score, low_confidence_word_ratio,
unresolved_segment_ratio, manual_review_required, reasons)`. Use the state formulas and
0.72 word threshold from the design. Weight by timed word count, then segment duration,
then one. Clamp every public score to `[0,1]`.

- [ ] **Step 4: Keep audio independent and deprecate aggregate optimism**

Retain `audio_quality_score = clamp(1 - silence_ratio)`. Set
`overall_source_quality_score = min(audio_quality_score, transcript_quality_score)`.
Store speech density/silence only as audio reasons. Store raw acoustic, low-word,
unresolved, correction, reconstruction, provider, status, and manual-review reasons as
transcript reasons.

- [ ] **Step 5: Recompute quality when transcript changes**

If `AudioAnalysis.audio_hash` matches, reuse silence/features but recompute speech rate
and call `assess_source` whenever quality input fingerprint differs. Quality fingerprint
includes audio-analysis output, reconstruction output, and `QUALITY_VERSION=2`.

- [ ] **Step 6: Run tests and commit**

Run Step 2. Expected: PASS.

```bash
git add backend/app/services/source_quality.py backend/app/pipeline/stages.py \
  backend/tests/test_quality.py backend/tests/test_transcript_quality.py
git commit -m "Separate transcript and audio quality"
```

---

### Task 9: Make degradation and quality visible to operators

**Files:**

- Modify: `backend/app/api/app.py:51-113,463-526`
- Modify: `backend/tests/test_api_sources.py`
- Modify: `frontend/src/lib/api-client.ts:1-145`
- Modify: `frontend/src/lib/api-client.test.ts`
- Modify: `frontend/src/app/sources/[id]/page.tsx:14-204`
- Create: `frontend/src/components/transcript-status.tsx`
- Create: `frontend/src/components/transcript-status.test.tsx`

**Interfaces:**

- API adds `QualityResponse` and exposes `GET /api/sources/{id}/quality`.
- Frontend consumes exact enum strings from Task 1.

- [ ] **Step 1: Write failing API and UI tests**

API test calls `GET /api/sources/{id}/quality` and asserts exact JSON:

```python
assert response.json()["reconstruction_status"] == "PROVIDER_UNAVAILABLE"
assert response.json()["quality"] == {
    "audio_quality_score": 0.985,
    "transcript_quality_score": 0.4,
    "low_confidence_word_ratio": 0.2,
    "unresolved_segment_ratio": 0.5,
    "manual_review_required": True,
    "conservative_source_floor": 0.4,
}
```

Render the new pure `TranscriptStatus` component with `renderToStaticMarkup` from the
already-installed `react-dom/server`. An unavailable transcript with no applied
correction must contain visible text `Reconstruction unavailable`, `Transcript quality
40%`, `Audio quality 99%`, and `Manual review required`. Render a focus span through the
same component and assert `دقل · 54%` appears. This avoids new browser-test dependencies.

- [ ] **Step 2: Run tests and confirm missing UI/API state**

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest pytest-asyncio httpx >/dev/null && pytest tests/test_api_sources.py -q"
(cd frontend && npm test -- --run)
```

- [ ] **Step 3: Add typed API quality/status responses**

Return reconstruction status even before/apart from applied candidates. Return split
quality using the persisted assessment; use `null` until assessment exists. Keep current
fields backward compatible. Add provider health/status to transcript metadata without
returning secrets.

- [ ] **Step 4: Render status and split quality unconditionally**

Implement and export pure `TranscriptStatus({ transcript, quality })` from the new
component. At top of transcript card, render status badge, provider/model, Audio quality,
Transcript quality, low-confidence ratio, unresolved ratio, and manual review. Map enum
labels explicitly; unknown future values render their raw string.

Correction details must render for any non-`NOT_REQUIRED` status, not only when a change
was applied. Show routing reasons and focus spans with rounded probability. Keep original
seek behavior and final-text priority.

- [ ] **Step 5: Add force retranscription action**

Add `api.retranscribeTranscript(id, force = true)` and a source-detail button labeled
`Force retranscription`. Queue the job, refresh job state, and never mutate displayed
text optimistically. Existing reconstruct button remains.

- [ ] **Step 6: Run tests, lint, and build**

```bash
(cd frontend && npm test -- --run && npm run lint && npm run build)
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/api/app.py backend/tests/test_api_sources.py frontend/src
git commit -m "Surface transcript reconstruction quality"
```

---

### Task 10: Replace the benchmark report reader with an actual pipeline runner

**Files:**

- Modify: `backend/app/transcription/reconstruction/benchmark.py:1-146`
- Modify: `backend/app/cli.py:150-164`
- Modify: `backend/app/services/storage.py`
- Modify: `backend/tests/test_reconstruction_audio_benchmark.py`
- Modify: `backend/tests/test_cli.py`

**Interfaces:**

- Produces `BenchmarkManifest` with media/reference input, `BenchmarkRunner.run`,
  machine-readable comparison rows, aggregate report, and review worksheet.
- Uses production `WhisperEngine`, `ContextualCorrector`, and `ContextualReconstructor`.

- [ ] **Step 1: Write failing end-to-end benchmark tests with fake engines/provider**

Manifest shape:

```json
{
  "version": "stage-2-7-private-v1",
  "split": "test",
  "sources": [{"id":"chernobyl","path":"sources/.../video.webm","authorized":true}],
  "clips": [{
    "id":"chernobyl-0000-0030","source_id":"chernobyl","topic":"history",
    "start_seconds":0,"end_seconds":30,"categories":["narrative","fast_speech"],
    "reference_segments":[{"segment_index":0,"text":"...","reviewed":true}]
  }]
}
```

Test that the runner calls raw ASR, Stage 2.5, and the actual provider service in order;
writes comparison/report/worksheet via `StorageService.atomic_write`; captures
`improved`, `unchanged_correct`, `unchanged_wrong`, `regressed`, `hallucinated`, and
`unresolved`; and refuses unreviewed/unauthorized/out-of-storage inputs.

- [ ] **Step 2: Run tests and confirm current report-only implementation fails**

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest >/dev/null && \
   pytest tests/test_reconstruction_audio_benchmark.py tests/test_cli.py -q"
```

- [ ] **Step 3: Split manifest inputs from generated report**

Remove `report` from `BenchmarkManifest`. Add validated `BenchmarkSource`,
`ReferenceSegment`, and `BenchmarkClip`. Require storage-relative paths, authorization,
`reviewed=true`, exact segment indexes, non-overlap, 2–5 total minutes, five clips,
three topics, two recordings, and required categories for readiness mode. Add
`--allow-known-regression-set` only for the Chernobyl diagnostic run; it can never pass
the unseen readiness gate.

- [ ] **Step 4: Execute production stages in memory/private workspace**

Extract each bounded clip with FFmpeg safe argument arrays into a benchmark job directory.
Run `large-v3-turbo`, Stage 2.5 corrector, then configured contextual reconstructor.
Assert raw segment IDs/timestamps before and after derivation. Do not write benchmark
rows into production `Transcript` records.

- [ ] **Step 5: Generate deterministic and human-review artifacts**

Write JSONL comparison rows containing raw, Stage 2.5, Stage 2.7, reference, status,
confidence, WER/CER, model/digest, and blank or prefilled human label. Human labels are
authoritative for semantic gates. Aggregate only reviewed rows. Console prints aggregate
metrics and storage-owned artifact paths, never full transcript text.

- [ ] **Step 6: Measure resources**

Measure wall time with `monotonic`, peak process RSS with `resource.getrusage`, and VRAM
only when a supported CUDA query exists. Record swap before/after from `/proc/meminfo`.
Mark a model infeasible on OOM, provider death, or sustained swap growth above 1 GiB.
Do not fabricate a performance pass threshold.

- [ ] **Step 7: Keep readiness gate strict**

Retain semantic lift at least 10 points, at least 25% of wrong spans improved, regression
at most 2%, preservation at least 98%, zero hallucinations, and non-decreasing category
comprehensibility. Add `unresolved` to report. Require provider availability, model
digest, complete human labels, and the exact production prompt/settings fingerprint.

- [ ] **Step 8: Run tests and commit**

Run Step 2. Expected: PASS.

```bash
git add backend/app/transcription/reconstruction/benchmark.py backend/app/cli.py \
  backend/app/services/storage.py backend/tests/test_reconstruction_audio_benchmark.py \
  backend/tests/test_cli.py
git commit -m "Run real reconstruction benchmarks"
```

---

### Task 11: Create and run the private Chernobyl and model-comparison benchmarks

**Files:**

- Create outside Git: `storage/benchmarks/stage-2-7/chernobyl-reference-v1.json`
- Create outside Git: `storage/benchmarks/stage-2-7/unseen-test-v1.json`
- Generate outside Git: `storage/benchmarks/stage-2-7/results/<run-id>/...`
- Modify after measurement: `docs/BENCHMARKS.md`
- Modify after measurement: `docs/ENVIRONMENT.md`
- Modify after measurement: `STATUS.md`

**Interfaces:**

- Consumes Task 10 benchmark runner and Task 2 Ollama service.
- Produces measured evidence only; no private transcript is committed.

- [ ] **Step 1: Start Ollama and install the provisional model**

```bash
docker compose --profile reconstruction up -d ollama
docker compose exec ollama ollama pull qwen3:8b
docker compose --profile reconstruction up --build -d
docker compose exec backend python -m app.cli reconstruction-health
```

Expected: availability `AVAILABLE`, model `qwen3:8b`, non-empty digest. Stop if not.

- [ ] **Step 2: Create the private full-reference Chernobyl manifest**

Reference source UUID `37c14f55-eacb-4d9f-8775-47a721cba5a9`, duration 159.36 seconds,
inside the storage-owned path. Listen to and manually transcribe the full clip when
practical. At minimum, review 0–30 seconds and include exact corrected forms for:

```text
عملية إخلاء مؤقت
تلات أيام بس لتطهير المنطقة
يا جماعة مفيش داعي للذعر
```

Do not infer the rest from topic knowledge. Mark each row `reviewed=true` only after
listening.

- [ ] **Step 3: Run known-sample diagnostic with Qwen3 8B**

```bash
docker compose exec backend python -m app.cli benchmark-reconstruction \
  stage-2-7/chernobyl-reference-v1.json --allow-known-regression-set \
  --model qwen3:8b
```

Manually inspect every 0–30 second row and all changed rows. Record improved, preserved,
unchanged wrong, regressions, hallucinations, unresolved, latency, RSS, swap, and digest.

- [ ] **Step 4: Probe Qwen3.5 9B feasibility, then compare if it fits**

```bash
docker compose exec ollama ollama pull qwen3.5:9b
docker compose exec backend python -m app.cli benchmark-reconstruction \
  stage-2-7/chernobyl-reference-v1.json --allow-known-regression-set \
  --model qwen3.5:9b
```

If OOM, provider death, or swap growth above 1 GiB occurs, record `INFEASIBLE_ON_TARGET`
with evidence. Do not loosen application gates. If feasible, compare semantic accuracy,
regression/hallucination rate, reliability, then speed. Run `qwen3.5:4b` only when both
larger candidates are operationally unusable or as an explicitly labeled fallback.

- [ ] **Step 5: Freeze and run the unseen acceptance set**

Create `unseen-test-v1.json` with at least five non-overlapping clips, 120–300 seconds,
three topics, two recordings, all required categories, authorization, and listened-to
references. Do not tune prompts, thresholds, or lexicon on this split after freezing.
Run winner once, complete manual labels, then rerun aggregation without changing outputs.

- [ ] **Step 6: Update measured docs without private transcript text**

Record hardware, model ID/digest, command, sample topology, aggregate counts/rates,
latency, peak RSS/VRAM, swap, feasibility, known first-30-second outcome, and gate result.
Set STATUS to `READY FOR STAGE 3` only if every completion gate passes; otherwise keep
`STAGE 2.7 MUST CONTINUE` and list exact failed gates.

- [ ] **Step 7: Commit aggregate evidence only**

```bash
git status --short
git add docs/BENCHMARKS.md docs/ENVIRONMENT.md STATUS.md
git commit -m "Record Stage 2.7 benchmark evidence"
```

Before commit, confirm no `storage/benchmarks` path or transcript text appears in staged
changes.

---

### Task 12: Update operations docs and run the full completion audit

**Files:**

- Modify: `README.md`
- Modify: `.env.example`
- Modify: `compose.yaml`
- Modify: `STATUS.md`
- Modify: `AGENTS.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/BENCHMARKS.md`
- Modify: `docs/ENVIRONMENT.md`
- Modify: `docs/LOCAL_SETUP.md`
- Modify: `docs/PIPELINE.md`
- Modify: `docs/TROUBLESHOOTING.md`

**Interfaces:**

- Documents exact final code and measured model. Does not claim readiness without Task
  11 evidence.

- [ ] **Step 1: Add a documentation contract test or search checklist**

Add assertions in `backend/tests/test_settings.py` or a focused documentation test that
required env names exist in `.env.example`, README contains Ollama setup and unavailable
behavior, and STATUS contains exactly one terminal Stage 2.7 line.

- [ ] **Step 2: Update every operational document with exact behavior**

Document active provider defaults, profile/start/pull/health commands, model digest,
unload behavior, 7.4 GiB CPU-only constraint, reconstruction statuses, fingerprint-driven
reruns, force semantics, split quality, private benchmark workflow, and missing-model
recovery. Update AGENTS current scope to Stage 2.7 without adding Stage 3 features.

- [ ] **Step 3: Scan for obsolete claims and placeholders**

Run:

```bash
rg -n "provider is disabled by default|RECONSTRUCTION_PROVIDER=disabled|stage2_5_fallback" \
  README.md STATUS.md AGENTS.md docs .env.example backend/app frontend/src
```

Expected: no obsolete default/fallback claims in current operational docs or production
code; no planning placeholders. Historical 2026-09-05 spec/plan may retain old text only
when the new spec clearly supersedes it.

- [ ] **Step 4: Run complete backend verification**

```bash
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest pytest-asyncio httpx coverage ruff mypy >/dev/null && \
   coverage run --source=app -m pytest && coverage report --fail-under=80 && \
   ruff format --check app tests && ruff check app tests && mypy app tests"
```

Expected: all Stage 2/2.5/2.6/2.7 tests PASS; coverage at least 80%; format, lint, and
strict type checks PASS.

- [ ] **Step 5: Run complete frontend and deployment verification**

```bash
(cd frontend && npm ci && npm test -- --run && npm run lint && npm run build)
docker compose config
docker compose --profile reconstruction config
docker compose exec backend alembic current
docker compose exec backend python -m app.cli reconstruction-health
```

Expected: frontend checks PASS; Compose configurations resolve; Alembic reports
`20260906_0009 (head)`; provider is `AVAILABLE` with model digest.

- [ ] **Step 6: Audit every completion gate against direct evidence**

Create a 12-row checklist in STATUS. For each gate, link a test, command output, API/UI
observation, or benchmark artifact. A missing or indirect proof is a failed gate. Confirm
the known Chernobyl first 30 seconds manually and confirm raw/timestamp deep-equality
tests passed.

- [ ] **Step 7: Set final status and commit**

If all gates pass, end STATUS and final implementation report with exactly:

```text
READY FOR STAGE 3
```

Otherwise end with exactly:

```text
STAGE 2.7 MUST CONTINUE
```

Then commit:

```bash
git add README.md .env.example compose.yaml STATUS.md AGENTS.md docs backend/tests
git commit -m "Document Stage 2.7 operations"
```

---

## Plan Self-Review

### Spec coverage

- Real enabled local provider: Tasks 2, 3, 11.
- Multi-word contextual repair and 5–15-second context: Tasks 4, 5.
- Acoustic word-confidence routing without a single threshold: Task 4.
- High-confidence incoherence escape hatch: Tasks 4, 5.
- Raw/timestamp immutability: Tasks 5, 6, 7, 10, 12.
- Fingerprint-driven downstream invalidation and exact stale bug: Task 7.
- Separate audio/transcript quality and misleading 0.877 case: Task 8.
- Truthful unavailable/degraded state in persistence/API/CLI/UI: Tasks 1, 2, 6, 9.
- Actual Chernobyl and unseen-audio pipeline benchmark: Tasks 10, 11.
- Small realistic model comparison and RAM evidence: Tasks 2, 11.
- Full regression and documentation gate: Task 12.
- Stage 3 exclusion and exact terminal status: Global Constraints and Task 12.

### Type consistency

- `ProviderAvailability` and `ReconstructionStatus` originate in `core/enums.py` and are
  reused by provider, service, persistence, quality, API, and frontend string unions.
- `ProviderHealth` originates in reconstruction types and is returned by every provider.
- `RoutingDecision` originates in routing and travels unchanged through requests and
  persisted evidence.
- `StageExecutionResult.output_fingerprint` is the only executor success contract.
- Transcript quality fields match migration, ORM, API, and frontend names.

### Execution order

Tasks are dependency-ordered. Give a small model one task at a time with both this plan
and the remediation spec. Require it to stop after each task's tests and commit for human
review. Do not let a later task rename an earlier interface without updating this plan
and all consumers first.
