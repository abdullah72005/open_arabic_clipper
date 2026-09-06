# Stage 2.7 operations

Stage 2.7 performs bounded contextual reconstruction after Stage 2.5. It is
local-first and never overwrites raw ASR text, segment timing, word timing, or
Stage 2.5 evidence.

## Provider operation

The default configuration is the optional local Ollama provider at
`http://ollama:11434` using `qwen3:8b`. Starting the Compose profile does not
pull any model. An operator must explicitly obtain the configured model before
reconstruction is available.

```bash
docker compose --profile reconstruction up -d ollama
docker compose exec ollama ollama pull qwen3:8b
docker compose exec backend python -m app.cli reconstruction-health
```

The health command and API expose provider availability, provider name, model,
and model digest only. They do not expose provider response bodies, prompts,
transcript text, credentials, or API keys. If the provider is unavailable,
misconfigured, or fails during release, reconstruction persists a truthful
status and falls back safely to earlier evidence.

## Text and quality truth

Raw ASR, Stage 2.5 corrected text, Stage 2.7 reconstructed text, and manual
operator text are separate evidence. Final text priority is manual override,
then an applied high-confidence reconstruction, then Stage 2.5, then raw ASR.

`GET /api/sources/{source_id}/transcript` includes reconstruction status and
public derived metadata. `GET /api/sources/{source_id}/quality` reports audio
quality separately from transcript/reconstruction quality. The compatibility
aggregate is the lower of those scores, so clean audio cannot hide unresolved
speech evidence. The dashboard presents the same status, reasons, and bounded
routing focus spans.

## Resuming or forcing work

Pipeline reuse requires matching canonical dependency fingerprints. A changed
input reruns the affected stage and downstream derived stages; historic null or
legacy fingerprints never create a cache hit. `--force` requests execution of
the selected stage without erasing persisted cache fields.

```bash
docker compose exec backend python -m app.cli reconstruct SOURCE_ID --force
docker compose exec backend python -m app.cli retranscribe SOURCE_ID --force
```

These commands affect only application-owned derived state. They do not modify
raw text or timing evidence.

## Benchmark boundary

Readiness requires a private, authorized, human-reviewed unseen-audio manifest
under storage-owned `benchmarks/`. Do not treat synthetic fixture metrics or a
known Chernobyl diagnostic set as readiness evidence. The real runner and
unseen-audio gate remain the next implementation/verification work.
