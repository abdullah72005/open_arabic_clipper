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

## Memory diagnostics

`scripts/diagnose-memory.sh` is a read-only host/container diagnostic that
labels host RAM, the WSL VM limit, the container cgroup limit, process and
container usage, and Ollama residency. It never edits configuration. Run it
before any heavy-model trial and after any WSL/Docker change:

```bash
./scripts/diagnose-memory.sh
```

The current machine ceiling is the WSL2 VM allocation (about 7.44 GiB with the
default 50% of a 16 GB host). Raising it requires the operator to write
`%UserProfile%\.wslconfig` with `memory=11GB` and `swap=4GB`, run
`wsl --shutdown`, restart Docker Desktop, and rerun the diagnostic; the
repository never performs that change itself. See `docs/ENVIRONMENT.md` for the
measured values and conclusion.

## Measured sequential lifecycle (2026-09-07, 10.69 GiB envelope)

Three sequential transcription-plus-reconstruction trials ran on an
operator-authorized 51.5 s source (`ca6cb88a…`) with `large-v3-turbo` (int8,
CPU) and `qwen3.5:4b` through the managed Ollama provider. The worker runs
Celery with `--pool=solo --concurrency=1 --max-tasks-per-child=1` and
`PYTHONPATH=/app`; the pre-fork pool is not used because its daemonic workers
cannot spawn the Whisper child process. A Redis-backed heavy-model lease
(`clipfactory:heavy-model`) serializes Whisper and Ollama.

Per-trial container RSS peaks (`docker stats --no-stream`, GiB):

| Trial | Whisper child peak | Whisper after child exit | Ollama peak | Ollama after unload | `ollama ps` after |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 1.97 | ≈ 0.17 | n/a (observed 0.33 idle) | 0.33 | empty |
| 2 | 2.46 | ≈ 0.26 | 5.85 | runtime empty (2.2 page cache) | empty |
| 3 | 3.36 | ≈ 0.47 | 5.93 | runtime empty (2.3 page cache) | empty |

Observations across all three trials:

- No Whisper/Ollama overlap: the lease serialized them, and the Whisper child
  exited before Ollama began loading.
- No OOM event; `MemAvailable` never fell below about 4.6 GiB.
- Swap growth per run was effectively zero (4 GiB swap, `SwapFree` stayed above
  3.99 GiB throughout).
- `ollama ps` was empty after every reconstruction; the persisted
  `unload_outcome` metadata recorded `requested=true`, `confirmed=true`, and
  sub-second elapsed time. The Ollama container's residual ≈ 2.2 GiB RSS is
  kernel page cache attributed to the container, not a resident model runtime.
- The worker process itself stayed small; the container RSS includes the
  spawned child and reclaimable page cache.

The health command and API expose provider availability, provider name, model,
and model digest only. They do not expose provider response bodies, prompts,
transcript text, credentials, or API keys. If the provider is unavailable,
misconfigured, or fails during release, reconstruction persists a truthful
status and falls back safely to earlier evidence.

## Provider confidence contract

Model output is untrusted data at one strict adapter boundary. Provider
confidence is valid only when its JSON value is a non-boolean finite real number
in the inclusive range `[0.0, 1.0]`; `NaN`, infinity, overflowed exponents,
booleans, strings, and out-of-range values are rejected as a contained provider
failure. The one-pass model emits one scalar confidence, so a candidate stores
`provider_confidence` only; no fabricated acoustic, phonetic, semantic, or
margin dimensions are created. Confidence never bypasses deterministic
validation.

The provisional apply policy is `score = provider_confidence -
0.20 * raw_acoustic_confidence * edit_ratio`, with `HIGH` at `score >= 0.82` and
`phonetic_similarity >= 0.72`. The `0.82` threshold is provisional and is not
tuned in the correctness plan.

## Per-segment failure isolation

Reconstruction calls are one target per provider call. A failure for one target
(provider error, network timeout) falls back only that segment to Stage 2.5 with
`PROVIDER_UNAVAILABLE`; earlier successful segments keep their applied
reconstruction. A provider-health failure before any call still falls back all
targets because no call can safely start. Provider release runs in one outer
`finally`; a release failure is logged but never replaces a valid result.

## Prompt budgeting

`max_context_tokens` applies to the complete chat envelope: the stable system
instruction, the serialized user wrapper, a chat-framing reserve, the configured
output budget (256), and a safety reserve. The adapter uses a conservative UTF-8
estimate, then deterministically shrinks following context, previous context,
entities, and word evidence in that order, never dropping the target segment ID
or its raw/Stage 2.5 text. An irreducible request raises a contained provider
error before any HTTP dispatch.

## Runtime identity and fingerprints

Every output-affecting dependency participates in the reconstruction runtime
identity: provider protocol, configured model, live model digest (or an explicit
`digest_unavailable` marker, never a fake tag), one-pass prompt hash and schema
version, full context/output budget, confidence-policy version, deterministic
validation version. Both the executor input fingerprint and the reconstruction
output fingerprint include this identity with canonical JSON ordering. A model,
digest, prompt, policy, validation, or context change invalidates prior
successful Stage 2.7 runs, and a provider-unavailable run has a distinct
fingerprint that cannot collide with an available-model run.

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

Evaluation separates four axes and reports them independently:

1. reconstruction runtime status (`APPLIED`, `LOW_CONFIDENCE_UNRESOLVED`, ...);
2. deterministic text comparison against a human reference (normalization plus
   the reviewed word-pair lexicon);
3. literal exact-string comparison (NFC equality after whitespace collapse only);
4. a validated human semantic/safety label.

Referenced unresolved rows stay in the Stage 2.5 and Stage 2.7 correctness
denominators; runtime status never removes a referenced row from the
denominator. Unreferenced rows are `unreviewed`, never implicitly safe.
Automated changed-but-wrong output is `changed_wrong`; `hallucinated` is a human
safety label. Human labels override semantic/safety counts only and never
override exact counts. Unknown human labels are rejected when loading a manifest
or review worksheet.

## Immutable ASR capture and replay

`benchmark-reconstruction` supports capturing immutable ASR once and replaying
it so reconstruction models are compared on identical raw segments:

```bash
docker compose exec backend python -m app.cli benchmark-reconstruction \
  stage-2-7/known-regression-v1.json --allow-known-regression-set --capture-asr
docker compose exec backend python -m app.cli benchmark-reconstruction \
  stage-2-7/known-regression-v1.json --allow-known-regression-set \
  --model qwen3:8b --from-capture <capture-id>
```

`--capture-asr` runs Whisper exactly once per clip and writes a hashed,
read-only capture under `storage/benchmarks/stage-2-7/captures/`.
`--from-capture` replays a stored capture and never constructs a transcriber.
Whisper and reconstruction runs always hold the `clipfactory:heavy-model`
lease, so models never overlap.
