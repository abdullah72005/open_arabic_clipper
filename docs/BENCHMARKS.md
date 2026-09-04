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
Run `python -m app.cli benchmark AUTHORIZED_AUDIO_FILE` on target hardware
before changing the default model or device policy.
