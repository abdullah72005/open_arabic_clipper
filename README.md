# ClipFactory / open_arabic_clipper

ClipFactory is a local-first foundation for safely ingesting media, probing its
metadata, and transcribing owned or authorized media. Stage 2 extracts a cached
mono 16 kHz WAV, runs local faster-whisper with automatic Arabic, English, and
mixed-speech detection, preserves raw ASR evidence, applies conservative
dialect-aware Arabic correction, and records silence/quality signals through
`READY_FOR_ANALYSIS`. Stage 3 then finds promising coarse clip moments cheaply
from that imperfect INDEX transcript, scores content separately from transcript
confidence, preserves strong uncertain moments as `CANDIDATE_NEEDS_REFINEMENT`
for Stage 3.5, and advances the source to `READY_FOR_REFINEMENT`. It does not
extract or retranscribe candidate audio, recover omitted English, select final
boundaries, reframe, render, publish, or automatically authorize content.

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
LLM. The Egyptian lexicon applies only when the source is confidently or
explicitly Egyptian; other dialects and unknown Arabic pass through unchanged.
To opt into a local OpenAI-compatible endpoint such as Ollama, configure
`CLIPFACTORY_CORRECTION_PROVIDER=openai_compatible` plus provider base URL and
model. Provider responses are batched, context-bounded, schema-validated, and
may only approve a declared Egyptian lexicon candidate; they fall back to
raw/lexicon output on any failure or unsafe change.

Stage 2.7.1 adds conservative source-level Arabic dialect awareness. During
Stage 2.5 normalization a pure, deterministic detector (no network, no LLM, no
model loading, no audio decoding) classifies the speech actually present in the
source from immutable raw segment text into one of `EGYPTIAN`, `SAUDI`, `GULF`,
`LEVANTINE`, `MSA`, or `UNKNOWN_ARABIC`. `None` means no Arabic evidence;
`UNKNOWN_ARABIC` means Arabic is present but the profile is uncertain, mixed,
or insufficiently evidenced and always favors no change. The effective profile
and confidence are persisted on the transcript and inherited by every segment.
Detected Latin words, names, abbreviations, technical tokens, and numbers are
preserved exactly (spelling, order, casing, digits) through Stage 2.5 and every
accepted Stage 2.7 candidate; technical forms such as `C++`, `foo/bar`, `#build`,
and full URLs (including query, fragment, and percent-encoded syntax, e.g.
`https://example.com/page?foo=bar#section`) are kept as exact atomic tokens, so
a candidate that fragments or changes them is rejected. `code_switch_suspected`
is true only for a segment that itself contains both Arabic-script evidence and
Latin-letter protected tokens, while numbers alone and English-only segments are
not flagged. An optional `dialect_profile_override` may be supplied when a
source is created through URL ingest or multipart upload and takes precedence
over detection with confidence 1.0; it is stored on the source and included in
normalization fingerprints so a future supported rerun invalidates derived work
without retranscribing audio. Dialect is source evidence, not a target audience
or localization choice, and there is no deployment-wide dialect default.

Stage 2.7 runs after Stage 2.5 and before audio analysis. It retains raw ASR,
Stage 2.5, Stage 2.7, and manual text separately; final text is always manual
override, then an applied HIGH-confidence reconstruction, then Stage 2.5, then
raw ASR. The default provider configuration is local Ollama at
`http://ollama:11434` with `qwen3.5:4b`; it never downloads a model implicitly.
Ollama starts with the rest of the stack on `docker compose up`; have the
operator explicitly pull the selected model once (for example,
`docker compose exec ollama ollama pull qwen3.5:4b`). Set
`CLIPFACTORY_RECONSTRUCTION_PROVIDER=disabled` to run without a provider, or
use `openai_compatible` with an operator-configured local endpoint. Invalid,
unavailable, or release-failed providers preserve Stage 2.5 output and do not
block `READY_FOR_ANALYSIS`; their status is recorded for review. Use
`python -m app.cli reconstruction-health` to inspect safe provider/model
metadata, and `POST /api/sources/{id}/reconstruct` or
`python -m app.cli reconstruct SOURCE_ID --force` to queue reconstruction.
If an unload fails or a heavy-model lease is lost, unsafe state is recorded in
Redis with no TTL and blocks new heavy work; run
`python -m app.cli recover-heavy-model` to clear it after confirming the model
is no longer resident.

Whole-source transcription is indexing, not publication. Normal whole-source
ingestion runs at INDEX priority and defers all provider reconstruction
truthfully (unresolved, never a provider failure), so it pays for ASR + Stage
2.5 + cheap uncertainty bookkeeping only and makes zero Qwen/Gemini calls.
Automatic local Qwen use is disabled by default
(`CLIPFACTORY_LOCAL_QWEN_ENABLED=false`); an operator re-enables it explicitly,
and a reusable `refine_transcript_window(source_id, start_time, end_time,
priority)` service entry point refines only a bounded selected window at
CANDIDATE (semantic) or FINAL_CLIP (publication/caption) quality while
preserving raw ASR, timestamps, and manual overrides.

An optional hosted Gemini provider (`gemini-3.8-flash`, low thinking, temperature
`0`, stable v1 API by default) supports deterministic adaptive routing
(`local_only` / `adaptive` / `gemini_only`;
`CLIPFACTORY_RECONSTRUCTION_ROUTING_MODE`, default `adaptive`). Clean, well-covered
unchanged Stage 2.5 segments and trusted Stage 2.5 repairs route to `NO_LLM` and
use neither LLM; a clean transcript may make zero Qwen and zero Gemini calls.
Normal uncertainty uses Qwen first, clearly difficult targets use one Gemini
request directly, and Qwen failures escalate to Gemini under a finite per-job
budget (`CLIPFACTORY_GEMINI_MAX_TARGETS_PER_JOB`, default `5`) that is spent on
the strongest eligible targets first, not the first five. Local Qwen work is
batched (`CLIPFACTORY_RECONSTRUCTION_PROVIDER_BATCH_WINDOWS`/`_BATCH_CHARACTERS`)
and hard-bounded per job
(`CLIPFACTORY_LOCAL_RECONSTRUCTION_MAX_TARGETS_PER_JOB` default `64`,
`CLIPFACTORY_LOCAL_RECONSTRUCTION_MAX_WALL_SECONDS` default `1200`), so a
four-hour source can never create unbounded local inference; skipped targets are
marked unresolved/manual review and never auto-escalate to Gemini, and once the
local wall-time ceiling expires, queued local-origin Gemini escalations are
invalidated (no Gemini call for the backlog). Each sent
local batch is also split at the provider boundary so its combined chat envelope
never exceeds the configured context. `adaptive` and
`gemini_only` may send short transcript snippets and bounded context to Google
Gemini; set `local_only` for a fully local pipeline. The key is a secret, held
as `SecretStr`, unwrapped only at lazy client construction, and never logged or
committed; there are no per-job Gemini metadata probes, and a fresh cache-hit
worker scrubs the key without any network call. A temporary Gemini
outage never overwrites accepted output: fingerprints are stable identity only
and cache eligibility is tracked separately. Cancelling a reconstruction job is
cooperative and polled on every provider route: the job stays `CANCELLED`, the
next stage is not scheduled, and already-accepted per-target work survives
restart without re-calling a provider. A degraded (not cache-eligible)
reconstruction is automatically retried on a later normal request without
repeating accepted targets.

Stage 2.7 cache reuse is dependency-aware: stage runs persist canonical input
and output fingerprints (plus per-target fingerprints), and changed upstream
evidence reruns downstream work. Manual force requests queue the requested stage
without clearing historical cache fields. The transcript API exposes
reconstruction status and public derived metadata;
`GET /api/sources/{id}/quality` separately reports audio and
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

## Stage 3 candidate analysis

Stage 3 (`CANDIDATE_ANALYSIS`) runs after `READY_FOR_ANALYSIS` and consumes the
imperfect INDEX transcript. It generates bounded deterministic coarse proposals,
scores content-quality separately from transcript confidence, classifies content
types, produces at most three source-faithful hooks, deduplicates same-source and
cross-source repeated ideas, and persists both accepted and rejected proposals.
Strong uncertain moments survive as `CANDIDATE_NEEDS_REFINEMENT` for Stage 3.5;
content quality below threshold is `DO_NOT_CLIP`, and redundant moments are
`DO_NOT_CLIP_RECENTLY_REDUNDANT`.

Stage 3 semantic mode defaults to `deterministic`: zero Gemini calls and zero
Qwen model loads. `adaptive` uses Gemini only when a key is configured, and
`local_only` uses Qwen/Ollama only with `CLIPFACTORY_LOCAL_QWEN_ENABLED=true`.
Missing or misconfigured providers degrade to deterministic output and never
fail the pipeline. Unknown/third-party provenance never blocks local analysis;
rights risk and originality/transformation risk are separate. Source dialect is
source evidence, not target audience; code-switched text is preserved and
omitted-English recovery is deferred to Stage 3.5.

Queue and inspect candidates:

```bash
python -m app.cli candidate-analysis SOURCE_ID [--force]
python -m app.cli candidates SOURCE_ID [--limit 20] [--include-rejected]
```

API: `POST /api/sources/{id}/candidate-analysis`, `GET
/api/sources/{id}/candidates`, `GET /api/candidates/{id}`, and `PATCH
/api/sources/{id}/provenance`. See
[Stage 3 operations](docs/STAGE_3_OPERATIONS.md) for the full design, bounds,
fingerprints, and Stage 3.5 handoff.
