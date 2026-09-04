from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.transcription.service import TranscriptionOptions


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
    whisper_model: Literal["tiny", "base", "small", "medium", "large-v3"] = "small"
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute_type: str | None = None
    whisper_cpu_compute_type: str = "int8"
    whisper_cuda_compute_type: str = "float16"
    whisper_beam_size: int = Field(default=5, gt=0, le=20)
    whisper_language: str | None = Field(default=None, min_length=2, max_length=16)
    whisper_word_timestamps: bool = True
    transcription_queue_concurrency: int = Field(default=1, gt=0)
    cors_origins: list[str] = ["http://localhost:3301"]

    def transcription_options(self) -> TranscriptionOptions:
        """Build the output-affecting options passed to the worker-side engine."""

        return TranscriptionOptions(
            model=self.whisper_model,
            device=self.whisper_device,
            compute_type=self.whisper_compute_type or "auto",
            beam_size=self.whisper_beam_size,
            language=self.whisper_language,
            word_timestamps=self.whisper_word_timestamps,
        )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
