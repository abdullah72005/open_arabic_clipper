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

## A URL is rejected

Only ordinary permitted public HTTP(S) URLs are accepted. Do not attempt to
circumvent DRM, logins, paywalls, CAPTCHA challenges, or other access controls;
obtain the media and authorization through legitimate means.
