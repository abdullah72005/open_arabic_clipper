# Stage 2.7 Accuracy and Runtime Recovery Design

## Decision status

Approved for implementation planning on 2026-09-07. This design supersedes the
provider parsing, confidence representation, reconstruction fingerprint,
transcript-level reconstruction persistence, benchmark evaluation, memory lifecycle,
and model-selection sections of the earlier Stage 2.7 designs. Their raw-evidence,
timestamp, authorization, local-first, and Stage 3 exclusion rules remain in force.

Implementation is split into three ordered plans:

1. correctness foundation;
2. memory and heavy-model lifecycle;
3. frozen-input model and targeted-ASR validation.

No model may be promoted until the first two plans pass. Stage 2.7 remains incomplete
until the third plan proves either a reliable quality improvement or the local hardware
ceiling.

## Problem statement

The pipeline is structurally complete but its current quality evidence is not reliable
enough to guide model selection:

- `qwen3.5:4b` runs but repairs few obvious Egyptian connected-speech errors;
- `qwen3:8b` was OOM-killed while Linux exposed about 7.4 GiB RAM and 2 GiB swap;
- the Windows host reportedly has 16 GB physical RAM, so the effective WSL/Docker
  limit must be measured and explained;
- the same examples persist after hours of prompt and threshold work;
- some examples are text-context reconstruction problems, while protected-number
  errors require audio-aware retranscription;
- current provider parsing, benchmark classification, fingerprints, and persistence
  contain defects that can crash fallback, accept invalid confidence, hide errors, or
  keep stale output.

The recovery must improve final transcript quality without weakening protection for
facts, names, numbers, raw ASR text, or timestamps.

## Non-negotiable invariants

- Raw Whisper text, segment order, segment timestamps, word timestamps, and public
  acoustic evidence are immutable.
- Stage 2.5, Stage 2.7 candidate, accepted Stage 2.7 text, manual text, and final text
  remain separate persisted evidence.
- Final-text priority remains manual override, accepted HIGH Stage 2.7, Stage 2.5,
  then raw ASR.
- A reconstruction never creates, deletes, merges, splits, reorders, or retimes a
  segment.
- Text-only reconstruction never guesses a changed digit, number, Latin token, or
  unsupported entity.
- Audio-dependent errors are sent to a measured retranscription experiment rather
  than made reachable by weakening reconstruction validation.
- Provider failures fall back per affected segment and never block local analysis.
- Only authorized, storage-owned audio enters benchmarks.
- CPU-only execution remains supported; GPU is optional.
- Quality outranks speed, but repeated OOM, uncontrolled swap, or concurrent heavy
  models is operational failure.
- No Stage 3 or Stage 2.7.1 work begins from incomplete evidence.

## Failure taxonomy

Every reviewed mismatch must be assigned exactly one primary cause before a fix is
attempted:

1. `ASR_AUDIO`: raw Whisper decoding is wrong and text context cannot safely recover
   the truth, especially changed numbers or facts.
2. `STAGE25`: raw ASR contains a correction supported by the existing conservative
   Stage 2.5 contract, but Stage 2.5 misses it.
3. `MODEL_CANDIDATE`: Stage 2.7 receives sufficient evidence but the model proposes no
   correct candidate.
4. `VALIDATION_OR_GATE`: the model proposes the human-correct candidate but server
   validation or confidence policy rejects it.
5. `PERSISTENCE`: the accepted candidate is lost or replaced before API/final text.
6. `EVALUATOR`: stored texts are correct but automated or human-label aggregation
   reports the wrong outcome.

The diagnostic row records raw, Stage 2.5, provider request fingerprint, candidate,
provider confidence, deterministic validation, decision reason, segment status,
persisted segment text, transcript text, human reference, and primary cause. Prompt or
threshold changes are forbidden until this trace shows `MODEL_CANDIDATE` or
`VALIDATION_OR_GATE` respectively.

## Architecture overview

```text
authorized audio
    |
    v
heavy-model lease --> Whisper in recyclable worker child
    |                    |
    |                    v
    |             immutable ASR capture
    |                    |
    |          worker exits / memory verified
    |                    |
    v                    v
heavy-model lease --> Ollama one-pass reconstruction
                         |
                 strict provider boundary
                         |
              deterministic validation/gate
                         |
              separate persisted evidence
                         |
                 human-reviewed evaluator
```

The immutable ASR capture is reused for model comparisons. Whisper is run again only
for an end-to-end confirmation or a targeted audio-dependent ASR experiment.

## 1. Correctness foundation

### Strict provider boundary

All model-produced fields are untrusted. The OpenAI-compatible adapter must convert
missing JSON, truncated JSON, invalid coverage, invalid types, invalid booleans, and
invalid confidence into `ProviderResponseError`.

Provider confidence is valid only when its JSON value is a non-boolean finite real
number in the inclusive range `[0.0, 1.0]`. `NaN`, positive/negative infinity,
overflowed exponent values, booleans, strings, and out-of-range values are rejected.

`_extract_json_object` may retain robust scanning, but a no-object result is part of
the provider error contract. It must never escape as a bare `ValueError`.

Requests remain one target per provider call. Failure is isolated to that target:
successful earlier results survive, failed targets receive `PROVIDER_UNAVAILABLE`, and
unrouted targets retain their normal status. A provider health failure before any call
still falls back all targets.

### Honest confidence representation

The one-pass model emits one scalar confidence. It does not emit five independently
verified dimensions or a real top-versus-runner-up margin. The candidate therefore
stores `provider_confidence`, not duplicated `ResolutionScores`.

The current provisional numeric behavior is preserved without threshold tuning:

```text
score = provider_confidence - 0.20 * raw_acoustic_confidence * edit_ratio

HIGH:
  score >= 0.82
  phonetic_similarity >= 0.72

MEDIUM:
  score >= 0.74
  provider_confidence >= 0.75
  edit_ratio <= 0.20
  token_delta <= 1
  phonetic_similarity >= 0.85
```

This removes redundant checks that were aliases for the same model scalar while
preserving effective one-pass decisions. All deterministic protected-token, length,
token-delta, phonetic, and entity checks still run before this policy.

Provider confidence remains evidence, not truth. It cannot bypass deterministic
validation.

### Full request budgeting

`max_context_tokens` applies to the complete chat transaction. Budgeting includes:

- system prompt;
- user JSON wrapper and target payload;
- chat framing reserve;
- the configured 256 output tokens;
- a safety reserve.

Without the model tokenizer, the adapter uses a deliberately conservative UTF-8-based
estimate and a fixed reserve. It shrinks following context, previous context, entities,
and word evidence deterministically. It never drops target raw/Stage 2.5 text or stable
segment ID. If the irreducible envelope does not fit, it raises
`ProviderResponseError` before HTTP dispatch.

Measured serialized prompt estimates are stored in benchmark rows so future tokenizer
integration can replace the approximation without changing safety behavior.

### Runtime identity and fingerprints

Every output-affecting reconstruction dependency participates in a stable runtime
identity:

- provider kind and base protocol;
- configured model identifier;
- live model digest when available;
- one-pass prompt hash and schema/protocol version;
- maximum context and output token settings;
- confidence-policy version and thresholds;
- deterministic validation version;
- correction/normalization fingerprint and ordered segment evidence.

`ContextualReconstructor.runtime_identity()` exposes this data. Pipeline input and
output fingerprints include it. A model, digest, prompt, policy, validation, or context
change invalidates successful Stage 2.7 runs. Provider unavailability is represented
truthfully and cannot reuse a prior available-model fingerprint.

### Persistence

Each persisted segment stores the exact `SegmentReconstruction` output. Transcript
`contextual_reconstructed_text` joins each segment's contextual reconstruction, not
its Stage 2.5 field. Manual overrides affect only `final_text`. Chunks and normalized
display text are rebuilt from final text while raw and Stage 2.5 fields remain intact.

### Benchmark evaluation axes

Evaluation separates four concepts:

1. reconstruction runtime status (`APPLIED`, `LOW_CONFIDENCE_UNRESOLVED`, and so on);
2. deterministic text comparison against a human reference;
3. literal exact-string comparison;
4. human semantic/safety label.

Runtime status never removes a referenced row from correctness denominators.

The deterministic comparison uses a narrow, auditable Egyptian variant lexicon. It
does not accept arbitrary prefix/suffix insertions or generic dental substitutions.
Arabic names such as `ثامر` and `تامر`, possessives such as `كتاب` and `كتابي`, and
verb changes such as `عمل` and `يعمل` remain distinct unless an explicit reviewed
word-pair entry says otherwise.

Exact comparison is literal NFC text equality after the documented whitespace policy.
It does not fold alef forms, `ة/ه`, `ى/ي`, punctuation, or diacritics.

Automated changed-but-wrong output is `changed_wrong`, not automatically
`hallucinated`. Hallucination is a human safety label for an invented fact, name,
number, or clause. Human labels are enum-validated and override only semantic report
counts. They never override exact counts.

Every referenced row contributes to Stage 2.5 and Stage 2.7 correctness, even when the
provider abstains or fails. Unreferenced rows are `unreviewed`, not safe. Readiness
requires complete references and human safety labels for every evaluated speech row.

## 2. Memory and heavy-model lifecycle

### Environment diagnosis

Diagnosis records Windows physical memory, WSL configuration, Linux `/proc/meminfo`,
active swap, cgroup limits/current usage, Docker Desktop limits, per-container memory,
and Ollama residency. The report distinguishes host RAM, WSL VM limit, container
cgroup limit, process RSS, aggregate container memory, and swap.

Repository code never silently edits `%UserProfile%/.wslconfig` or shuts down WSL. If
the effective limit is below 10 GiB and host RAM is 16 GB, documentation recommends:

```ini
[wsl2]
memory=11GB
swap=4GB
```

The operator applies it and runs `wsl --shutdown` from Windows. Eleven GiB leaves about
five GiB physical headroom for Windows while allowing a practical quantized 8B trial.
Twelve GiB may be documented as an operator-selected alternative only when measured
Windows pressure remains safe.

### Runtime preflight

A read-only memory probe reports total/available Linux memory, total/free swap, cgroup
limit/current/peak where available, process RSS, and effective capacity. The 8B
benchmark refuses to start when effective memory is below 10 GiB or another heavy-model
lease exists. Normal 4B fallback operation remains available with a warning rather
than a global startup failure.

### Sequential execution guarantee

The existing single Celery concurrency is retained and worker children recycle after
each task with `--max-tasks-per-child=1` and prefetch multiplier one. Transcription and
reconstruction remain separate durable tasks.

Whisper runs in a dedicated spawned subprocess owned by the task/CLI coordinator. The
coordinator acquires the heavy-model lease before spawning, receives a bounded typed
result/error envelope, joins and verifies termination of the subprocess, measures the
post-exit container state, and only then releases the lease. This makes process exit—not
Python garbage collection—the hard CTranslate2 reclamation boundary and prevents a CLI
from entering the small gap between task return and Celery child recycling.

`WhisperEngine` still releases local references and runs supported cleanup in the child.
Celery child recycling remains defense in depth for other native allocations; it is not
the synchronization primitive.

Ollama release remains in `finally` with `keep_alive: 0`. Completion is verified by
polling `ollama ps`/the compatible process listing until the model disappears or a
bounded timeout produces a release warning.

A Redis-backed heavy-model lease named `clipfactory:heavy-model` serializes worker
transcription, reconstruction, and benchmark commands. The lease uses a unique token,
bounded TTL, bounded blocking timeout, ownership-checked release, and a retryable busy
error. It prevents a CLI benchmark from loading Ollama while a worker owns Whisper. A
Whisper lease is never released before the spawned model subprocess has been reaped. An
Ollama lease is never released before unload is confirmed or the bounded failure path
has recorded that the environment is unsafe for another heavy-model start.

### Memory evidence

Memory snapshots are taken before Whisper load, after transcription, after worker
recycle, before Ollama load, after reconstruction, and after Ollama unload. Host-side
benchmark tooling also records `docker stats --no-stream` for worker, backend, and
Ollama because one process's `ru_maxrss` cannot measure the external model server.

Reliability requires three consecutive candidate-model runs without OOM, timeout,
stale Ollama residency, or swap growth above one GiB per run. Any repeated failure is
recorded; it is not hidden by restarting services and reporting only a successful run.

## 3. Frozen-input quality validation

### Immutable ASR capture

The benchmark is split into capture and reconstruction phases.

Capture runs authorized audio through the configured Whisper model once, consumes all
segments, persists immutable raw segment/word evidence through `StorageService`, and
records audio hash plus complete `TranscriptionOptions` fingerprint. It then releases
Whisper before reconstruction begins.

Reconstruction consumes a capture only when its audio/options fingerprint validates.
Both 4B and candidate 8B models receive identical Stage 2.5 text, context, word
evidence, prompt version, validation policy, and human references.

### Required failure corpus

The known corpus includes reviewed timestamps and references for:

- `فيور 25 نوفمبر` versus `في يوم 25 نوفمبر`;
- `آخره يشيلة نصر واحد` versus `آخره يشيل اتناشر واحد`;
- the verified phrase around the raw `71`/spoken `70 واحد` error;
- the three existing Chernobyl connected-speech failures.

The first two are evaluated for text recoverability after trace classification. The
number-changing row is presumed `ASR_AUDIO` unless acoustic evidence proves the raw
number was already represented elsewhere. No reference phrase is injected into the
production prompt or lexicon.

If the newer problematic clip cannot be located from existing DB/storage evidence, the
executor stops before benchmarking and requests its source ID plus reviewed timestamps.
It does not invent a manifest.

### Model shortlist and protocol

Only two reconstruction configurations are compared initially:

1. current `qwen3.5:4b` baseline;
2. already-compatible quantized `qwen3:8b` after memory correction and verified
   sequential unload.

No catalog sweep is allowed. A single alternative Qwen 7B–8B instruct model may replace
`qwen3:8b` only if the latter remains protocol-incompatible or cannot be installed,
and the substitution is documented before download.

Temperature stays zero. One-pass small-context requests, strict total prompt budget,
protected-token validation, and the provisional confidence policy remain identical.
Each model runs three times against the same capture for reliability; only the final
selected model gets one end-to-end Whisper-to-persistence confirmation.

### Targeted ASR experiment

Rows classified `ASR_AUDIO` are tested separately on their exact stored audio interval
with small padding. Compare `large-v3-turbo` to full `large-v3` using the same language,
timestamp, and no-reference-injection rules. Do not add the expected phrase as a
hotword or prompt. This is an experiment, not automatic production routing.

If full `large-v3` materially fixes protected-number/audio-dependent rows within the
new memory envelope, a later explicitly approved plan may change ASR selection or add
targeted retranscription. This recovery does not smuggle that architectural change into
the reconstruction model task.

### `large-v3` versus `large-v3-turbo`

The full `large-v3` decoder is an explicit quality experiment, not an assumed upgrade.
Official Whisper documentation describes turbo as a pruned `large-v3` with four decoder
layers rather than the large model's 32, trading a small amount of accuracy for much
higher speed. That trade is relevant because this product values Egyptian transcription
quality over latency and several known failures are audio-dependent. It does not prove
that full `large-v3` is better on this corpus.

The experiment uses the same faster-whisper release, CPU `int8` compute type, decoding
settings, source audio, reviewed timestamp spans, and evaluator for both ASR models. Run
`large-v3-turbo` first, release it and verify memory reclamation, then run `large-v3`;
they must never be resident together or overlap with Ollama. Start with only the reviewed
problem spans. Run a full-clip comparison only if full `large-v3` wins that screen.

Promote `large-v3` from experiment to operational ASR default only when all are true:

- it fixes at least two additional human-confirmed `ASR_AUDIO` errors on the combined
  known and unseen reviewed sets, including no degradation on protected numbers/names;
- it creates zero new human-labeled hallucinations or factual regressions;
- its downstream Stage 2.5 and Stage 2.7 outputs are no worse on reviewed rows;
- it completes three sequential full-clip runs without OOM, stale model residency, or
  unbounded swap growth inside the corrected memory envelope;
- measured latency is documented and accepted as the cost of the quality gain.

If it does not meet every gate, keep `large-v3-turbo` as the default and retain full
`large-v3` only as an optional targeted retranscription candidate. A number mismatch
such as raw `71` versus spoken `70` cannot be credited to the text-only reconstruction
model because protected-token validation correctly prevents that guess.

### Selection gates

Promote the stronger reconstruction model only when all are true:

- it produces at least two more human-confirmed correct text-recoverable multi-word
  repairs than 4B on the frozen corpus;
- at least 50% of reviewed text-recoverable Stage 2.5-wrong rows improve;
- human-confirmed factual/name/number hallucinations remain zero;
- regression rate does not exceed 2%, with at least 98% of Stage 2.5-correct rows
  preserved;
- referenced unresolved rows are included in denominators and unresolved rate does not
  worsen relative to 4B;
- three consecutive runs finish without OOM or more than one GiB incremental swap;
- accepted text reaches persisted segment, transcript, final text, and API fields.

The unseen readiness corpus remains required for Stage 2.7 completion. The known error
corpus is a regression/calibration set and can never alone authorize Stage 2.7.1.

If the stronger model fails operational gates after correct WSL allocation, sequential
unload, and practical quantization, retain `qwen3.5:4b`. Record the hardware ceiling and
recommend the existing hosted-provider boundary, targeted uncertain-span
retranscription, and full `large-v3` evaluation without implementing cloud use.

## Error handling and recovery

- Invalid provider output: per-target Stage 2.5 fallback plus provider error metadata.
- Provider health unavailable: all routed targets fall back; analysis may continue with
  manual review required.
- Heavy-model lease busy: retryable stage/CLI error; never start concurrently.
- Insufficient 8B memory: benchmark preflight fails with measured capacity and host
  configuration guidance; 4B production fallback remains usable.
- Whisper cleanup not verified: recycle worker child before reconstruction proceeds.
- Ollama unload timeout: mark release warning, refuse another heavy model until lease
  cleanup or operator intervention.
- Missing human reference/source ID: stop quality claims; do not infer labels.
- Benchmark interruption: keep immutable capture and partial run artifact, but exclude
  incomplete run from selection.

## Test strategy

The correctness plan uses unit and integration tests for parser failures, confidence
bounds, partial recovery, full-envelope budgets, evaluator axes, fingerprints, and
persistence. The memory plan uses injected `/proc`/cgroup readers, fake Redis locks,
worker configuration assertions, Ollama unload polling tests, and one measured manual
Docker run. The quality plan uses frozen captures and deterministic fake providers for
automation plus human-reviewed real runs for evidence.

Real audio and transcripts remain private storage artifacts and are never committed.
Only aggregate, redacted metrics and reviewed conclusions enter documentation.

## Delivery boundaries

Each plan is committed separately after its own verification:

1. `Fix Stage 2.7 correctness foundations`
2. `Make heavy model lifecycle memory-safe`
3. `Benchmark stronger Stage 2.7 model`

The implementation agent stops after each plan and reports tests, diff, and git state.
It does not combine all three plans into one unreviewable five-hour session.

## Final status

End the final report with exactly one line:

```text
READY FOR STAGE 2.7.1
```

only if the unseen corpus and every gate pass. Otherwise end with:

```text
STAGE 2.7 MUST CONTINUE
```
