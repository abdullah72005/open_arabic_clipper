# Task 5 Report: Pipeline and Celery Workers

## Delivered

- Added an idempotent `PipelineRunner` with persisted `PipelineRun` and
  `ProcessingJob` state transitions, executor protocol, retry accounting, and
  stage advancement through `READY_FOR_TRANSCRIPTION`.
- Added the explicit future AUTOPILOT authorization boundary. Sources with
  `UNKNOWN` rights raise `AutopilotAuthorizationError`.
- Added Celery configuration using JSON serialization, late acknowledgement,
  prefetch multiplier one, and default concurrency one.
- Added durable stage-task wrappers that retain the processing-job identity,
  increment retry counts only for retryable exceptions, and a worker heartbeat
  task.
- Added `backend/tests/test_pipeline.py`. The RED run failed because the
  `app.pipeline` package did not exist. The GREEN run passed after the focused
  implementation.

## Verification

Run from `backend/` using the repository virtual environment:

```text
.venv/bin/python -m mypy app/pipeline app/workers
Success: no issues found in 7 source files

.venv/bin/python -m ruff check app tests
All checks passed!

.venv/bin/python -m pytest -q
41 passed in 7.66s
```

## Deferred verification

The containerized Celery worker ping check was deferred because Docker is not
available on this host. It remains required in a Docker-capable environment.
