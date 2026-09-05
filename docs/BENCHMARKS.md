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

No Stage 2.7 quality or performance value has been recorded yet. Fixture tests
exercise safety and regressions only; they cannot prove real Egyptian-audio
improvement. A private, operator-authorized unseen test manifest must contain at
least five non-overlapping clips totaling two to five minutes, across three
topics and two recordings, including slang, fast speech, code switching,
entities, and narrative speech. Every evaluated segment requires a frozen human
reference and review label.

Store that manifest below the storage service `benchmarks/` category and run:

```bash
python -m app.cli benchmark-reconstruction stage-2-7/unseen-test-v1.json
```

The command prints only aggregate metrics and fails readiness unless it records
the provider model identifier/digest, a 10-point semantic lift, 25% improvement
of Stage 2.5-wrong segments, no more than 2% regression, 98% preservation, zero
hallucinations, and non-decreasing category comprehensibility. Until measured
evidence meets all bounds, Stage 2.7 remains open.
