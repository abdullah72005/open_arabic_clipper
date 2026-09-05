"""Conservative Celery configuration for local media work."""

from celery import Celery  # type: ignore[import-not-found]

from app.core.enums import PipelineStage
from app.core.settings import get_settings

settings = get_settings()
celery_app = Celery("clipfactory", broker=settings.redis_url, backend=settings.redis_url)


def _route_pipeline_stage(
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    options: dict[str, object],
    task: object = None,
    **kw: object,
) -> dict[str, str] | None:
    """Route transcription work to its dedicated queue; everything else stays on media."""
    stage = kwargs.get("stage") if "stage" in kwargs else (args[1] if len(args) >= 2 else None)
    if stage == PipelineStage.TRANSCRIPTION.value:
        return {"queue": "transcription"}
    return None


celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_default_queue="media",
    task_routes=_route_pipeline_stage,
    task_track_started=True,
    timezone="UTC",
)
celery_app.autodiscover_tasks(["app.workers"])
