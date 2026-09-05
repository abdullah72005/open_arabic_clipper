# Stage 2 Transcription Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Turn every probed source with usable audio into a cached, timestamped local faster-whisper transcript, audio analysis, chunks, and quality assessment at READY_FOR_ANALYSIS.

**Architecture:** Extend existing durable pipeline rather than replacing it. New services own audio extraction, Whisper conversion, normalization, analysis, and chunking; PipelineRunner owns transitions and Celery only dispatches stage work. API, CLI, and UI read durable state and schedule work, never load models in request threads.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Alembic, Celery, Redis, FFmpeg, faster-whisper, Next.js, TypeScript, Tailwind, pytest, Vitest.

**Spec:** docs/superpowers/specs/2026-09-04-stage-2-transcription-design.md

## Global Constraints

- Default auto-detect handles Arabic, English, and mixed speech; forced language is opt-in.
- Default model is small; CPU uses int8, usable CUDA uses float16.
- StorageService owns paths; FFmpeg gets only argument arrays.
- No paid API, GPU requirement, transcript rewriting, authorization, clip selection, rendering, or publishing.
- Write every behavioral test first and observe expected failure.
- Unit tests cannot download large Whisper models; real runs are separately marked integration tests.

---

## File Structure

- backend/app/core/{enums,settings}.py: lifecycle/job values and runtime configuration.
- backend/app/models/{audio_artifact,transcript,audio_analysis,source_quality_assessment}.py: durable Stage 2 data.
- backend/app/media/{audio,analysis}.py: FFmpeg extraction and metrics.
- backend/app/transcription/{engine,service,normalization,chunking}.py: model adapter and reusable transcript processing.
- backend/app/pipeline/stages.py and backend/app/workers/tasks.py: durable executors and queue routing.
- backend/app/api/app.py and backend/app/cli.py: transcript access and controls.
- frontend/src/lib/api-client.ts and frontend/src/app/sources/[id]/page.tsx: API client and RTL viewer.
- backend/alembic/versions/20260904_0003_stage_2_transcription.py: schema migration.
- backend/tests/test_{audio,analysis,transcription,transcript_api,quality,cli}.py: backend coverage.
- docs/TRANSCRIPTION.md plus README, STATUS, AGENTS, pipeline/setup/troubleshooting docs: operations and benchmark.

### Task 1: Add Stage 2 domain state and migration

**Files:**
- Modify: backend/app/core/enums.py, backend/app/models/__init__.py, backend/app/models/source_video.py, backend/app/pipeline/runner.py
- Create: backend/app/models/audio_artifact.py, backend/app/models/transcript.py, backend/app/models/audio_analysis.py, backend/app/models/source_quality_assessment.py, backend/alembic/versions/20260904_0003_stage_2_transcription.py
- Test: backend/tests/test_models.py, backend/tests/test_migrations.py

**Interfaces:**
- Produces AudioArtifact, Transcript, AudioAnalysis, SourceQualityAssessment related to SourceVideo.
- Adds AUDIO_EXTRACTION, TRANSCRIPTION, TRANSCRIPT_NORMALIZATION, AUDIO_ANALYSIS, READY_FOR_ANALYSIS and JobKind.TRANSCRIPTION.

- [ ] **Step 1: Write failing persistence test**

~~~python
def test_source_keeps_timestamped_transcript(session, source):
    transcript = Transcript(
        source_video=source, language="ar", whisper_model="small",
        transcription_options={"beam_size": 5}, input_fingerprint="x" * 64,
        raw_text="أهلا", normalized_text="أهلا", duration=0.8,
        segments=[{"start": 0.0, "end": 0.8, "text": "أهلا",
                   "avg_logprob": -0.1, "no_speech_prob": 0.01, "words": []}],
    )
    session.add(transcript); session.commit()
    assert source.transcript.segments[0]["start"] == 0.0
~~~

- [ ] **Step 2: Verify RED**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_models.py tests/test_migrations.py -q

Expected: FAIL because Stage 2 models and migration do not exist.

- [ ] **Step 3: Implement model and migration**

~~~python
class Transcript(Base):
    __tablename__ = "transcripts"
    source_video_id: Mapped[UUID] = mapped_column(
        ForeignKey("source_videos.id", ondelete="CASCADE"), unique=True, index=True
    )
    language: Mapped[str | None] = mapped_column(String(32))
    detected_language_probability: Mapped[float | None]
    whisper_model: Mapped[str] = mapped_column(String(64))
    transcription_options: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    input_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    normalized_text: Mapped[str] = mapped_column(Text, default="")
    segments: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    word_segments: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    duration: Mapped[float] = mapped_column(Float, default=0)
~~~

Implement matching artifact, analysis, quality models, source relationships, indexes,
and Alembic upgrade/downgrade. Update next-stage mapping without changing Stage 1 behavior.

- [ ] **Step 4: Verify GREEN**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c 'pytest tests/test_models.py tests/test_migrations.py -q && alembic upgrade head && alembic downgrade -1 && alembic upgrade head'

Expected: PASS; migration reaches head after rollback.

- [ ] **Step 5: Commit**

~~~bash
git add backend/app/core backend/app/models backend/app/pipeline/runner.py backend/alembic/versions backend/tests/test_models.py backend/tests/test_migrations.py
git commit -m "feat: persist Stage 2 transcription state"
~~~

### Task 2: Configure Whisper and deterministic cache keys

**Files:**
- Modify: backend/app/core/settings.py, .env.example, backend/pyproject.toml, backend/Dockerfile, compose.yaml
- Create: backend/app/transcription/__init__.py, backend/app/transcription/service.py
- Test: backend/tests/test_settings.py, backend/tests/test_transcription.py

**Interfaces:** Produces TranscriptionOptions.fingerprint(audio_hash: str) -> str and Settings.transcription_options() -> TranscriptionOptions.

- [ ] **Step 1: Write failing policy tests**

~~~python
def test_fingerprint_changes_only_for_material_options():
    options = TranscriptionOptions("small", "cpu", "int8", 5)
    assert options.fingerprint("a" * 64) == options.fingerprint("a" * 64)
    assert options.fingerprint("a" * 64) != replace(options, beam_size=1).fingerprint("a" * 64)

def test_defaults_are_practical(settings):
    assert settings.whisper_model == "small"
    assert settings.whisper_device == "auto"
    assert settings.whisper_cpu_compute_type == "int8"
~~~

- [ ] **Step 2: Verify RED**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_settings.py tests/test_transcription.py -q

Expected: FAIL because options/settings are missing.

- [ ] **Step 3: Implement typed configuration**

~~~python
@dataclass(frozen=True)
class TranscriptionOptions:
    model: str
    device: str
    compute_type: str
    beam_size: int
    language: str | None = None
    word_timestamps: bool = True

    def fingerprint(self, audio_hash: str) -> str:
        payload = json.dumps(asdict(self) | {"audio_hash": audio_hash}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
~~~

Add bounded model validation, model/device/compute/beam/queue settings, faster-whisper
dependency, and worker queues media,transcription at concurrency one.

- [ ] **Step 4: Verify GREEN**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_settings.py tests/test_transcription.py -q

Expected: PASS without model download.

- [ ] **Step 5: Commit**

~~~bash
git add backend/app/core/settings.py backend/app/transcription backend/pyproject.toml backend/Dockerfile compose.yaml .env.example backend/tests
git commit -m "feat: configure local transcription"
~~~

### Task 3: Extract cached analysis WAV audio

**Files:**
- Create: backend/app/media/audio.py
- Modify: backend/app/services/storage.py
- Test: backend/tests/test_audio.py

**Interfaces:** AudioExtractor.extract(source: SourceVideo) -> AudioArtifact; raises MissingAudioStreamError or retryable stage error.

- [ ] **Step 1: Write failing extraction/cache tests**

~~~python
def test_extractor_writes_mono_16khz_wav_once(fake_runner, source, storage, session):
    artifact = AudioExtractor(storage, fake_runner, session).extract(source)
    again = AudioExtractor(storage, fake_runner, session).extract(source)
    assert artifact.sample_rate == 16_000
    assert artifact.output_path == again.output_path
    assert fake_runner.calls == 1

def test_extractor_reports_missing_audio_stream(fake_runner, source, storage, session):
    fake_runner.fail("Output file #0 does not contain any stream")
    with pytest.raises(MissingAudioStreamError):
        AudioExtractor(storage, fake_runner, session).extract(source)
~~~

- [ ] **Step 2: Verify RED**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_audio.py -q

Expected: FAIL because extractor is missing.

- [ ] **Step 3: Implement safe, idempotent extractor**

~~~python
args = [ffmpeg_binary, "-y", "-i", str(source_path), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(destination)]
subprocess.run(args, check=True, capture_output=True, text=True)
~~~

Resolve paths through StorageService, reserve capacity, atomically persist output hash,
duration/sample rate/source hash, and reuse only valid matching artifacts.

- [ ] **Step 4: Verify GREEN**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_audio.py -q

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add backend/app/media/audio.py backend/app/services/storage.py backend/tests/test_audio.py
git commit -m "feat: add cached speech audio extraction"
~~~

### Task 4: Implement faster-whisper adapter

**Files:**
- Create: backend/app/transcription/engine.py
- Modify: backend/app/transcription/service.py
- Test: backend/tests/test_transcription.py

**Interfaces:** WhisperEngine.transcribe(audio_path, options) -> TranscriptionResult with detected language/probability, raw text, duration, segments, and words.

- [ ] **Step 1: Write failing engine test**

~~~python
def test_engine_falls_back_to_cpu_int8_and_preserves_words(fake_whisper):
    engine = WhisperEngine(fake_whisper, cuda_available=lambda: False)
    result = engine.transcribe(Path("speech.wav"), TranscriptionOptions("small", "auto", "auto", 5))
    assert fake_whisper.model_args == ("small", "cpu", "int8")
    assert result.language == "ar"
    assert result.segments[0]["words"][0]["start"] == 0.0
~~~

- [ ] **Step 2: Verify RED**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_transcription.py -q

Expected: FAIL because engine is missing.

- [ ] **Step 3: Implement lazy device/model adapter**

~~~python
segments, info = model.transcribe(
    str(audio_path), beam_size=options.beam_size, language=options.language,
    word_timestamps=options.word_timestamps,
)
serialized = [{"start": item.start, "end": item.end, "text": item.text,
               "avg_logprob": item.avg_logprob, "no_speech_prob": item.no_speech_prob,
               "words": [{"start": word.start, "end": word.end, "word": word.word,
                          "probability": word.probability} for word in item.words or []]}
              for item in segments]
~~~

Probe CUDA only in worker context, fall back safely for auto mode, preserve raw
Whisper language/text/word values, and inject model factory in tests.

- [ ] **Step 4: Verify GREEN**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_transcription.py -q

Expected: PASS using fake model only.

- [ ] **Step 5: Commit**

~~~bash
git add backend/app/transcription backend/tests/test_transcription.py
git commit -m "feat: add local faster whisper adapter"
~~~

### Task 5: Normalize, chunk, and analyze audio

**Files:**
- Create: backend/app/transcription/normalization.py, backend/app/transcription/chunking.py, backend/app/media/analysis.py
- Test: backend/tests/test_transcription.py, backend/tests/test_analysis.py

**Interfaces:** normalize_transcript(text) -> str, build_chunks(segments, config) -> list[Chunk], parse_silencedetect(output) -> list[SilenceInterval].

- [ ] **Step 1: Write failing text/chunk/silence tests**

~~~python
def test_normalization_preserves_egyptian_and_english_words():
    assert normalize_transcript("  أنا  okay\n\nأنا  ") == "أنا okay أنا"

def test_chunking_prefers_segment_boundary_and_context():
    chunks = build_chunks(SEGMENTS, ChunkConfig(target_seconds=30, context_segments=1))
    assert chunks[0].segment_ids == [0, 1]
    assert chunks[0].following_context == SEGMENTS[2]["text"]

def test_parse_silencedetect_pairs_intervals():
    assert parse_silencedetect("silence_start: 1.0\nsilence_end: 2.5 | silence_duration: 1.5") == [
        SilenceInterval(start=1.0, end=2.5, duration=1.5)
    ]
~~~

- [ ] **Step 2: Verify RED**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_transcription.py tests/test_analysis.py -q

Expected: FAIL because utilities are missing.

- [ ] **Step 3: Implement semantic analysis**

Normalize whitespace/newlines/punctuation spacing, safe Arabic canonical forms, and
only adjacent repeated ASR fragments. Choose sentence/segment boundaries nearest
target duration, never character-slice words. Parse silencedetect and calculate
fixed-window RMS/energy, silence ratio, pauses, speech density/rate, and deltas.

- [ ] **Step 4: Verify GREEN**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_transcription.py tests/test_analysis.py -q

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add backend/app/transcription backend/app/media/analysis.py backend/tests
git commit -m "feat: analyze and normalize transcripts"
~~~

### Task 6: Wire durable workers, retries, cache, and quality

**Files:**
- Create: backend/app/pipeline/stages.py, backend/app/services/source_quality.py
- Modify: backend/app/workers/{tasks,celery_app}.py, backend/app/pipeline/runner.py
- Test: backend/tests/test_pipeline.py, backend/tests/test_quality.py

**Interfaces:** registered executor per new stage; retranscription preserves retries/cache history; assess_source(source, transcript, analysis) -> SourceQualityAssessment.

- [ ] **Step 1: Write failing recovery/quality tests**

~~~python
def test_completed_transcript_reuses_fingerprint(session, source, executors):
    PipelineRunner(session, executors).run(source.id, PipelineStage.TRANSCRIPTION)
    assert PipelineRunner(session, executors).run(source.id, PipelineStage.TRANSCRIPTION).skipped

def test_quality_is_advisory(session, source, transcript):
    assessment = assess_source(source, transcript, ANALYSIS)
    assert assessment.overall_source_quality_score >= 0
    assert source.lifecycle_state is not PipelineStage.FAILED
~~~

- [ ] **Step 2: Verify RED**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_pipeline.py tests/test_quality.py -q

Expected: FAIL because stages/quality service are missing.

- [ ] **Step 3: Implement durable stages**

Run extraction on media; transcription on transcription; then normalization,
analysis, chunk/quality persistence. Persist failure before raising, resume
incomplete work after restart, honor force retranscribe, estimate progress from
completed segment seconds, and calculate versioned confidence/silence/speech/
repetition/audio scores plus reasons without rejecting sources.

- [ ] **Step 4: Verify GREEN**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_pipeline.py tests/test_quality.py -q

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add backend/app/pipeline backend/app/workers backend/app/services/source_quality.py backend/tests
git commit -m "feat: run recoverable transcription pipeline"
~~~

### Task 7: Add transcript API and CLI

**Files:**
- Modify: backend/app/api/app.py, backend/app/cli.py
- Test: backend/tests/test_transcript_api.py, backend/tests/test_cli.py

**Interfaces:** GET /sources/{id}/transcript, GET /sources/{id}/transcript/segments, GET /sources/{id}/transcript/search?q=, POST /sources/{id}/retranscribe; CLI transcribe, transcript, retranscribe.

- [ ] **Step 1: Write failing API/CLI tests**

~~~python
def test_search_returns_mixed_language_timestamp(client, transcript):
    response = client.get(f"/sources/{transcript.source_video_id}/transcript/search?q=hello")
    assert response.status_code == 200
    assert response.json()["segments"][0]["start"] == 12.4

def test_retranscribe_cli_accepts_model(runner):
    result = runner.invoke(app, ["retranscribe", "11111111-1111-1111-1111-111111111111", "--model", "medium"])
    assert result.exit_code == 0
~~~

- [ ] **Step 2: Verify RED**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_transcript_api.py tests/test_cli.py -q

Expected: FAIL because API/CLI are missing.

- [ ] **Step 3: Implement responses/dispatch**

Expose raw/normalized text, language/confidence/model/options/duration/timestamps.
Use SQLAlchemy parameterized filters and bounded pagination. Retranscribe validates
model, supports force, creates durable job, dispatches Celery; CLI calls same services.

- [ ] **Step 4: Verify GREEN**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest tests/test_transcript_api.py tests/test_cli.py -q

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add backend/app/api/app.py backend/app/cli.py backend/tests/test_transcript_api.py backend/tests/test_cli.py
git commit -m "feat: expose searchable transcripts"
~~~

### Task 8: Add RTL transcript viewer

**Files:**
- Modify: frontend/src/lib/api-client.ts, frontend/src/lib/api-client.test.ts, frontend/src/app/sources/[id]/page.tsx, frontend/src/app/globals.css
- Create: frontend/src/app/sources/[id]/transcript-viewer.tsx
- Test: frontend/src/app/sources/[id]/transcript-viewer.test.tsx

**Interfaces:** client methods getTranscript, searchTranscript, retranscribe; TranscriptViewer supports search, metadata, RTL, timestamps, optional seeking.

- [ ] **Step 1: Write failing client/viewer tests**

~~~typescript
it("searches mixed language and retains timestamp", async () => {
  const api = createApiClient("http://api", fetcher);
  await expect(api.searchTranscript("id", "hello")).resolves.toMatchObject({
    segments: [{ start: 4.2, text: "أهلا hello" }],
  });
});

it("uses automatic direction for Arabic", () => {
  render(<TranscriptViewer transcript={arabicTranscript} />);
  expect(screen.getByText("أهلا").closest("section")).toHaveAttribute("dir", "auto");
});
~~~

- [ ] **Step 2: Verify RED**

Run: cd frontend && npm test -- --run src/lib/api-client.test.ts src/app/sources/[id]/transcript-viewer.test.tsx

Expected: FAIL because transcript client/viewer are missing.

- [ ] **Step 3: Implement viewer**

Render live stage/progress, detected language/confidence, model, duration, raw/
normalized text, timestamped segments, and search. Use dir="auto" with bidi-safe
CSS. Timestamp buttons seek a supplied video ref only when one is present.

- [ ] **Step 4: Verify GREEN**

Run: cd frontend && npm test -- --run src/lib/api-client.test.ts src/app/sources/[id]/transcript-viewer.test.tsx

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add frontend/src
git commit -m "feat: add RTL transcript viewer"
~~~

### Task 9: Benchmark, integrate, document, and verify

**Files:**
- Create: docs/TRANSCRIPTION.md, backend/tests/test_transcription_integration.py
- Modify: README.md, AGENTS.md, STATUS.md, docs/PIPELINE.md, docs/LOCAL_SETUP.md, docs/TROUBLESHOOTING.md
- Test: complete backend/frontend suites and optional integration marker.

**Interfaces:** documented config, CPU/GPU policy, Arabic procedure, actual throughput record, queue/retry troubleshooting.

- [ ] **Step 1: Write integration contract**

~~~python
@pytest.mark.integration
def test_local_audio_transcription_records_throughput(local_audio_fixture, configured_service):
    result = configured_service.transcribe(local_audio_fixture)
    assert result.wall_seconds > 0
    assert result.audio_minutes_per_wall_minute > 0
~~~

- [ ] **Step 2: Verify initial integration state**

Run: docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend pytest -m integration -q

Expected: skip with exact prerequisite reason when no authorized fixture/model exists.

- [ ] **Step 3: Implement integration path and docs**

Document configuration/model choices/queues/retries, authorized Arabic-fixture setup,
measured source duration/wall time/RTF/minutes-per-minute/model/device/compute type,
and safe local setup. Replace Stage 1-only wording in required docs.

- [ ] **Step 4: Verify full suite**

Run: docker compose config && docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c 'coverage run --source=app -m pytest && coverage report && ruff format --check app tests && ruff check app tests' && cd frontend && npm test && npm run lint && npm run build

Expected: all checks pass; integration runs or skips only for documented fixture/runtime prerequisite.

- [ ] **Step 5: Commit**

~~~bash
git add README.md AGENTS.md STATUS.md docs backend/tests/test_transcription_integration.py
git commit -m "docs: document Stage 2 transcription operations"
~~~

## Plan Self-Review

- Coverage: Tasks 1–8 implement specified pipeline, persistence, engine, cache, language, analysis, quality, API, CLI, UI, retry, and restart requirements. Task 9 covers integration, benchmark, and documentation.
- Placeholder scan: no TBD, TODO, or deferred implementation steps.
- Interface consistency: persistence and TranscriptionOptions precede media/engine consumers; API/CLI/UI follow durable pipeline behavior.

