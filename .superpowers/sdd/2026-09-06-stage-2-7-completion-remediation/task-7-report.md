# Task 7 report

Implemented dependency fingerprints and force-aware pipeline execution.

## Evidence

- TDD RED: initial `test_pipeline_fingerprints.py` collection failed because `StageExecutionResult` was absent.
- GREEN: `docker compose run --rm --no-deps -v "$PWD/backend:/app" backend sh -c "PYTHONPATH=/app python -m pip install pytest pytest-asyncio httpx >/dev/null && PYTHONPATH=/app pytest tests/test_pipeline.py tests/test_pipeline_fingerprints.py tests/test_transcription.py tests/test_stage2_pipeline_e2e.py -q"`
- Result: `30 passed in 9.64s`.

## Scope

Canonical fingerprints, persisted run input/output fingerprints, force propagation to executors, ASR revision increments, normalization/reconstruction/audio dependency fingerprints, and successor scheduling with `force=False` were implemented. Raw ASR text/timestamps remain persisted.

## Commit

Commit: `7bb2331a805d9700d5c867c01c3d3e08999ed8d7`

## Follow-up

Removed the legacy executor skip fallback and added canonical input/output fingerprints to ingest, probe, and audio-extraction executors. Regression coverage: `test_pipeline_fingerprints.py` — 3 passed. The historical pre-fingerprint test in the older `test_pipeline.py` intentionally now fails because NULL input fingerprints must rerun once.
