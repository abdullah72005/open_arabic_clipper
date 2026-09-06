# ClipFactory / open_arabic_clipper

ClipFactory is a local-first foundation for safely ingesting media, probing its
metadata, and transcribing owned or authorized media. Stage 2 extracts a cached
mono 16 kHz WAV, runs local faster-whisper with automatic Arabic (Egyptian/MSA),
English, and mixed-speech detection, preserves raw ASR evidence, applies
conservative contextual Egyptian-Arabic correction, and records silence/quality
signals through `READY_FOR_ANALYSIS`. It does not
select clips, reframe, render, publish, or automatically authorize content.

Only process material you own or are explicitly authorized to process. URL
ingest downloads permitted public sources directly; an optional outbound proxy
may be configured. The software does not bypass DRM, logins, paywalls,
CAPTCHAs, or platform protections.

## Quick start (Docker)

```bash
cp .env.example .env
docker compose up --build -d
docker compose exec backend alembic upgrade head
docker compose exec backend python -m app.cli health
```

Open `http://localhost:3301` for the dashboard and `http://localhost:8300/docs`
for the API. Stop services with `docker compose down`; add `-v` only when you
intentionally want to remove database and Redis volumes. Local media remains
under `./storage`.

## Development checks

GitHub Actions runs these quality gates on every push and pull request: backend
tests with a coverage report, Ruff format/lint checks, frontend tests/lint/build,
and Docker Compose configuration validation.

```bash
docker compose config
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c \
  "python -m pip install pytest pytest-asyncio httpx coverage ruff && \
   coverage run --source=app -m pytest && ruff format --check app tests && ruff check app tests"
(cd frontend && npm ci && npm test && npm run lint && npm run build)
```

The frontend image is built for the browser API base URL configured by
`NEXT_PUBLIC_API_BASE_URL`; use `http://localhost:8300` for local browser use.

Read [local setup](docs/LOCAL_SETUP.md), [architecture](docs/ARCHITECTURE.md),
[pipeline](docs/PIPELINE.md), and [troubleshooting](docs/TROUBLESHOOTING.md)
before using external media sources.

## Stage 2, 2.5, and 2.7 transcription quality

Workers need FFmpeg/ffprobe and the local `faster-whisper` dependency. Configure
`CLIPFACTORY_WHISPER_MODEL` (`tiny`, `base`, `small`, `medium`, `large-v3`, or
`large-v3-turbo`),
`CLIPFACTORY_WHISPER_DEVICE` (`auto`, `cpu`, or `cuda`), and optionally
`CLIPFACTORY_WHISPER_LANGUAGE` (`ar` or `en`). `auto` uses CUDA only when
available and otherwise uses CPU `int8` inference.

The default decoder is `large-v3-turbo`, with auto device selection, CPU `int8`, beam 5,
word timestamps, faster-whisper fallback temperatures `[0, 0.2, 0.4, 0.6, 0.8,
1]`, previous-text conditioning enabled, VAD disabled, and no prompt/hotwords.
`CLIPFACTORY_WHISPER_TEMPERATURE`,
`CLIPFACTORY_WHISPER_CONDITION_ON_PREVIOUS_TEXT`,
`CLIPFACTORY_WHISPER_VAD_FILTER`, `CLIPFACTORY_WHISPER_INITIAL_PROMPT`, and
`CLIPFACTORY_WHISPER_HOTWORDS` are output-affecting settings and invalidate the
transcript cache. Do not opt into prompt/hotword/VAD changes without an
operator-authorized benchmark covering Arabic, English, and code-switched audio.

Stage 2.5 preserves `raw_text` and every raw segment `text`/timestamp permanently.
It adds `corrected_text`, `final_text`, confidence indicators, correction method,
version, and per-segment correction metadata. The default local corrector uses
the versioned Egyptian phrase lexicon only; it never requires a network or an
LLM. To opt into a local OpenAI-compatible endpoint such as Ollama, configure
`CLIPFACTORY_CORRECTION_PROVIDER=openai_compatible` plus provider base URL and
model. Provider responses are batched, context-bounded, schema-validated, and
may only approve a declared lexicon candidate; they fall back to raw/lexicon
output on any failure or unsafe change.

Stage 2.7 runs after Stage 2.5 and before audio analysis. It retains raw ASR,
Stage 2.5, Stage 2.7, and manual text separately; final text is always manual
override, then an applied HIGH-confidence reconstruction, then Stage 2.5, then
raw ASR. The default provider configuration is local Ollama at
`http://ollama:11434` with `qwen3:8b`; it never downloads a model implicitly.
Start the optional service with `docker compose --profile reconstruction up -d
ollama`, then have the operator explicitly pull the selected model (for example,
`docker compose exec ollama ollama pull qwen3:8b`). Set
`CLIPFACTORY_RECONSTRUCTION_PROVIDER=disabled` to run without a provider, or
use `openai_compatible` with an operator-configured local endpoint. Invalid,
unavailable, or release-failed providers preserve Stage 2.5 output and do not
block `READY_FOR_ANALYSIS`; their status is recorded for review. Use
`python -m app.cli reconstruction-health` to inspect safe provider/model
metadata, and `POST /api/sources/{id}/reconstruct` or
`python -m app.cli reconstruct SOURCE_ID --force` to queue reconstruction.

Stage 2.7 cache reuse is dependency-aware: stage runs persist canonical input
and output fingerprints, and changed upstream evidence reruns downstream work.
Manual force requests queue the requested stage without clearing historical
cache fields. The transcript API exposes reconstruction status and public
derived metadata; `GET /api/sources/{id}/quality` separately reports audio and
transcript/reconstruction quality, with the aggregate conservatively taking the
lower score. The source detail page shows provider availability, unresolved or
manual statuses, split-quality reasons, and bounded routing focus evidence.

Use `GET /api/sources/{id}/transcript` for raw/corrected/final evidence,
`GET /api/sources/{id}/transcript/search?q=...` for timestamped final-text
segments, and `POST /api/sources/{id}/retranscribe` to queue a new local ASR job.
Operators can save or clear a final manual correction with `POST` or `DELETE`
`/api/sources/{id}/transcript/segments/{segment_index}/override`; raw and
automatic text remain unchanged. Arabic transcript panels show a correction debug
view and retain original Unicode code-switched terms. Selecting a segment seeks
storage-owned local playback to its original timestamp.

Operator commands are available from the backend environment: `python -m app.cli
transcribe SOURCE_ID`, `python -m app.cli transcript SOURCE_ID`, and `python -m
app.cli retranscribe SOURCE_ID`. The latter bypasses the cache by default.

Before selecting a deployment default, run `python -m app.cli benchmark
REPRESENTATIVE_AUTHORIZED_AUDIO.wav` on the target machine. It prints the source
duration, wall-clock time, real-time factor, audio-minutes-per-wall-minute,
model, device, and compute type from the actual local run. No representative
licensed Arabic sample is bundled with this repository, so benchmark figures are
intentionally not fabricated.

The current local cached-model benchmark is recorded in
[benchmark results](docs/BENCHMARKS.md).

Run the deterministic correction fixture benchmark in the backend container:

```bash
python -m app.transcription.correction_benchmark \
  --fixture app/transcription/fixtures/egyptian_ar_correction.json --baseline
python -m app.transcription.correction_benchmark \
  --fixture app/transcription/fixtures/egyptian_ar_correction.json
```

Fixture metrics are regression evidence, not ground-truth dialect accuracy. Use
an authorized audio set and manual semantic review before enabling any LLM model
or changing Whisper decoding defaults.

Stage 2.7 readiness requires a private manifest inside storage-owned
`benchmarks/`, with no transcript bodies committed to the repository. Run
`python -m app.cli benchmark-reconstruction stage-2-7/unseen-test-v1.json`.
The runner executes raw `large-v3-turbo` ASR, Stage 2.5, then Stage 2.7 through
the configured live provider, writes a JSONL comparison, a human-review
worksheet, and an aggregate report under `storage/benchmarks/stage-2-7/results/`,
and prints only aggregate metrics and storage-owned artifact paths. Use
`--model` to compare an alternate configured model and
`--allow-known-regression-set` only for the Chernobyl diagnostic run, which can
never pass the unseen readiness gate. Until the strict unseen-audio gate passes,
the status is `STAGE 2.7 MUST CONTINUE`.

See [Stage 2.7 operations](docs/STAGE_2_7_OPERATIONS.md) for the persisted
status contract and operator troubleshooting notes.
