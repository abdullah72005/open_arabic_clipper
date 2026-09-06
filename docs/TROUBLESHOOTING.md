# Troubleshooting

## Docker cannot pull or build

Run `docker compose config` to validate configuration without starting services.
If Docker reports a credential-helper error, sign in to Docker Desktop or fix
its configured credential store, then retry `docker compose build`. This is an
environment issue, not a reason to relax image or service security.

## A job is stuck or failed

Inspect `docker compose logs worker` and the Jobs page. Check Redis and worker
reachability with `docker compose exec redis redis-cli ping` and
`docker compose exec worker celery -A app.workers.celery_app:celery_app inspect ping`.
Do not delete active sources; cancel the job first, then use the documented API
or CLI retry behavior.

## FFmpeg or ffprobe is unavailable

The container image installs both. Native operators must install FFmpeg and put
`ffmpeg`/`ffprobe` on `PATH`, or set the corresponding `CLIPFACTORY_*_BINARY`
variables.

## Reconstruction reports the provider unavailable

Start the optional Ollama profile and pull the configured model, then run
`python -m app.cli reconstruction-health` to confirm `AVAILABLE` with a model
digest. A `DEGRADED` health result or a persisted
`PROVIDER_UNAVAILABLE`/`LOW_CONFIDENCE_UNRESOLVED` transcript status means the
configured model is absent or unreachable; Stage 2.5 output stays final. Pull
the model explicitly — Compose never downloads model weights implicitly — and
retry with `python -m app.cli reconstruct SOURCE_ID --force`, which re-queues
the stage without clearing persisted cache fields.

## A URL is rejected

Only ordinary permitted public HTTP(S) URLs are accepted. Do not attempt to
circumvent DRM, logins, paywalls, CAPTCHA challenges, or other access controls;
obtain the media and authorization through legitimate means.
