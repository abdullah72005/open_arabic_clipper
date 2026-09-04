from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from the environment."""

    model_config = SettingsConfigDict(env_prefix="CLIPFACTORY_", extra="ignore")

    environment: str = "development"
    database_url: str = "postgresql+psycopg://clipfactory:clipfactory@postgres:5432/clipfactory"
    redis_url: str = "redis://redis:6379/0"
    storage_root: Path = Path("/var/lib/clipfactory")
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    max_upload_bytes: int = Field(default=2 * 1024 * 1024 * 1024, gt=0)
    max_remote_download_bytes: int = Field(default=2 * 1024 * 1024 * 1024, gt=0)
    url_egress_proxy: str | None = None
    max_concurrent_uploads: int = Field(default=2, gt=0)
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
