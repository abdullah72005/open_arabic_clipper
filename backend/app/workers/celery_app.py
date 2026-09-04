"""Conservative Celery configuration for local media work."""

from celery import Celery  # type: ignore[import-not-found]

from app.core.settings import get_settings

settings = get_settings()
celery_app = Celery("clipfactory", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_default_queue="media",
    task_routes={"clipfactory.run_transcription": {"queue": "transcription"}},
    task_track_started=True,
    timezone="UTC",
)
celery_app.autodiscover_tasks(["app.workers"])
