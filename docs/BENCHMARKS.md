# Local transcription benchmark

This repository was benchmarked on 2026-09-04 with an operator-authorized
51.54-second source clip. The first run downloaded the `small` model; the
numbers below are the subsequent cached-model run, so they describe inference
rather than one-time model setup.

| Hardware | Model | Device / compute type | Audio duration | Wall time | Real-time factor | Audio min / wall min |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Intel Core Ultra 9 185H | small | CPU / int8 | 51.54 s | 16.49 s | 0.32 | 3.13 |

The run auto-detected English at 0.85 probability. The source is valuable as a
real local pipeline check, but it is not evidence of Egyptian-Arabic quality.

An operator-authorized Egyptian-Arabic source was processed end-to-end on the
same machine on 2026-09-04. The model auto-detected Arabic at 0.983 probability
and produced 2,268 timestamped segments and 6,573 word timestamps. This is a
real pipeline measurement, not a controlled benchmark, so the longer
45:24.89-minute source and its speech characteristics are recorded separately.

| Hardware | Model | Device / compute type | Audio duration | Wall time | Real-time factor | Audio min / wall min |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Intel Core Ultra 9 185H | small | CPU / int8 | 45:24.89 | 27:04.45 | 0.596 | 1.677 |

Run `python -m app.cli benchmark AUTHORIZED_AUDIO_FILE` on target hardware
before changing the default model or device policy.

## Stage 2.5 Egyptian correction fixture benchmark

On 2026-09-05, the deterministic `egyptian_ar_correction.json` corpus was run
in the backend Python 3.12 container with baseline display normalization and the
default local `egyptian-ar-v1` lexicon corrector. The 14 manually reviewed cases
include Arabic-only, English-only, code-switched technical speech, names,
numbers, filler words, slang, negation, questions, football, and the three
operator-supplied phonetic errors.

| Mode | Exact fixture match | Token error signal | Improved | Unchanged | Worse | Auto rate | Uncertain rate | Wall time | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline normalization | 78.57% | 10.94% | 0 | 14 | 0 | 0.00% | 100.00% | 4.44 ms | 6,760 B |
| Stage 2.5 local lexicon | 100.00% | 0.00% | 3 | 11 | 0 | 21.43% | 78.57% | 6.67 ms | 7,284 B |

Observed fixture overhead is 2.23 ms and 524 B. These micro-benchmark values do
not include faster-whisper inference or an optional LLM endpoint. Standard WER
and exact spelling can misrepresent dialect quality, so each fixture also has a
manual semantic-correctness note. The meaningful result is that all three known
Egyptian errors become clear while the English-only and code-switch cases remain
unchanged.

Known remaining cases: unknown Egyptian phonetic confusions, ambiguous names or
numbers, and corrections that require acoustic evidence beyond nearby text stay
raw until an operator adds and benchmarks a reviewed lexicon entry.

## Stage 2.7 contextual reconstruction gate

The private runner (`python -m app.cli benchmark-reconstruction`) executes raw
`large-v3-turbo` ASR, Stage 2.5 correction, then Stage 2.7 through the managed
local Ollama provider, and writes a JSONL comparison, a human-review worksheet,
and an aggregate report under `storage/benchmarks/stage-2-7/results/<run-id>/`.
It prints only aggregate metrics and storage-owned artifact paths.

On 2026-09-06 and 2026-09-07 the known Chernobyl diagnostic (source
`37c14f55-eacb-4d9f-8775-47a721cba5a9`, first 30 seconds, the three
operator-supplied failure phrases with their reviewed expected forms) was run
with `--allow-known-regression-set` against three models:

| Model | Model digest | Wall time | Peak process RAM | Result |
| --- | --- | ---: | ---: | --- |
| qwen3:8b | `500a1f067a9f…b41` | 163.4 s | 2.16 GiB | infeasible: `llama-server` killed (`signal: killed`) while loading ~5.5 GiB into 7.4 GiB RAM |
| qwen3.5:4b (gate 0.86) | `2a654d98e6fb…eefd` | 628.8 s | 2.18 GiB | feasible: `provider_available=true`, `model_feasible=true`, 0 hallucinations, 0 regressions, 10 unresolved, 1 unchanged-wrong; no reconstruction applied |
| qwen3.5:4b (gate 0.82, semantic evaluator) | `2a654d98e6fb…eefd` | 531.0 s | 2.15 GiB | kept provisionally: 1 improved (`تلاتة يام بس لتطهير المنطقة`), 0 hallucinations, 0 regressions, 9 unresolved, 1 unchanged-wrong |
| qwen3:4b | `359d7dd4bcda…4fae7` | n/a | n/a | unusable: thinking mode emits chain-of-thought that exhausts the 256-token budget before any JSON payload |

The machine (Core Ultra 9 185H, 7.4 GiB RAM, 2 GiB swap, CPU-only) cannot hold
`qwen3:8b` alongside the Whisper pipeline and services; the model load was
out-of-memory killed. `qwen3.5:4b` loads and runs the one-pass small-context
protocol end-to-end.

The evaluator classifies segments by Egyptian normalization plus
phonetic/semantic equivalence: orthographic variants (diacritics, alef forms,
`ة`/`ه`, `ى`/`ي`), prosthetic alef (`يام`/`أيام`), suffix `ة` (`تلات`/`تلاتة`),
and Egyptian dental shifts (`ث`/`ت`, `ذ`/`د`, `ظ`/`ز`) are equivalent; names,
numbers, Latin tokens, and meaning-changing substitutions are not. Exact string
match is retained as a separate metric (`exact_*` in the report and
`exact_status` per row).

On the provisional `score >= 0.82` apply gate, the model applied segment 1 as
`تلاتة يام بس لتطهير المنطقة` (reference `تلات أيام بس لتطهير المنطقة`): the
semantic evaluator reports `improved`, and the exact-match metric reports
`hallucinated` solely because of the dialect spelling variants `تلاتة`/`تلات`
and `يام`/`أيام`. There is no meaning change, regression, or invented fact, so
0.82 is kept provisionally. Raw ASR and Stage 2.5 completed for all runs; the
three known multi-word errors are not all fixed. `qwen2.5:7b` could not be
pulled reliably over the network from this machine. `qwen3:4b` was pulled but
cannot be parsed because its reasoning is emitted as free text before the JSON
object.

These are regression diagnostics only. The model comparison and the strict
unseen-audio acceptance set (at least five non-overlapping clips, two to five
minutes, three topics, two recordings, with human-reviewed references) remain
open work, and a known diagnostic run is never counted as unseen readiness.

The runner's safety gates reject unreviewed, unauthorized, or out-of-storage
inputs; a known regression set can run but can never pass the unseen readiness
gate. Until a feasible model and a completed unseen-audio evaluation pass every
gate, Stage 2.7 remains open.
