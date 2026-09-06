from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.transcription.correction import ContextualCorrector, CorrectionConfig
from app.transcription.providers import CorrectionProvider, OpenAICompatibleCorrectionProvider
from app.transcription.reconstruction import ContextualReconstructor
from app.transcription.reconstruction.ollama import OllamaReconstructionProvider
from app.transcription.reconstruction.providers import (
    OpenAICompatibleReconstructionProvider,
    ReconstructionProvider,
)
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
    whisper_model: Literal["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"] = (
        "large-v3-turbo"
    )
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute_type: str | None = None
    whisper_cpu_compute_type: str = "int8"
    whisper_cuda_compute_type: str = "float16"
    whisper_beam_size: int = Field(default=5, gt=0, le=20)
    whisper_language: str | None = Field(default=None, min_length=2, max_length=16)
    whisper_word_timestamps: bool = True
    whisper_temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    whisper_condition_on_previous_text: bool = True
    whisper_vad_filter: bool = False
    whisper_initial_prompt: str | None = Field(default=None, max_length=4_000)
    whisper_hotwords: str | None = Field(default=None, max_length=4_000)
    correction_context_segments: int = Field(default=2, ge=0, le=5)
    correction_high_confidence: float = Field(default=0.90, ge=0, le=1)
    correction_medium_confidence: float = Field(default=0.75, ge=0, le=1)
    correction_max_small_edit_ratio: float = Field(default=0.25, ge=0, le=1)
    correction_provider_batch_size: int = Field(default=32, gt=0, le=200)
    correction_provider: Literal["disabled", "openai_compatible"] = "disabled"
    correction_provider_base_url: str | None = Field(default=None, max_length=2_048)
    correction_provider_model: str | None = Field(default=None, max_length=256)
    correction_provider_api_key: str | None = Field(default=None, max_length=4_096)
    correction_provider_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    reconstruction_provider: Literal["disabled", "openai_compatible", "ollama"] = "ollama"
    reconstruction_provider_base_url: str | None = Field(
        default="http://ollama:11434", max_length=2_048
    )
    reconstruction_provider_model: str | None = Field(default="qwen3:8b", max_length=256)
    reconstruction_provider_timeout_seconds: float = Field(default=180.0, gt=0, le=300)
    reconstruction_release_after_run: bool = True
    reconstruction_provider_batch_windows: int = Field(default=8, gt=0, le=16)
    reconstruction_provider_batch_characters: int = Field(default=24_000, gt=0, le=48_000)
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
            cpu_compute_type=self.whisper_cpu_compute_type,
            cuda_compute_type=self.whisper_cuda_compute_type,
            temperature=self.whisper_temperature,
            condition_on_previous_text=self.whisper_condition_on_previous_text,
            vad_filter=self.whisper_vad_filter,
            initial_prompt=self.whisper_initial_prompt,
            hotwords=self.whisper_hotwords,
        )

    def correction_config(self) -> CorrectionConfig:
        """Build safe correction thresholds and local-provider batch bounds."""

        return CorrectionConfig(
            context_segments=self.correction_context_segments,
            high_confidence=self.correction_high_confidence,
            medium_confidence=self.correction_medium_confidence,
            max_small_edit_ratio=self.correction_max_small_edit_ratio,
            provider_batch_size=self.correction_provider_batch_size,
        )

    def correction_provider_instance(self) -> CorrectionProvider | None:
        """Return an explicit local LLM integration only when fully configured."""

        if self.correction_provider == "disabled":
            return None
        if not self.correction_provider_base_url or not self.correction_provider_model:
            raise ValueError(
                "correction_provider_base_url and correction_provider_model are required "
                "for an openai_compatible correction provider"
            )
        return OpenAICompatibleCorrectionProvider(
            base_url=self.correction_provider_base_url,
            model=self.correction_provider_model,
            api_key=self.correction_provider_api_key,
            timeout_seconds=self.correction_provider_timeout_seconds,
        )

    def contextual_corrector(self) -> ContextualCorrector:
        """Build local lexicon correction with an optional explicitly configured provider."""

        return ContextualCorrector.from_default_lexicon(
            config=self.correction_config(), provider=self.correction_provider_instance()
        )

    def reconstruction_provider_instance(self) -> ReconstructionProvider | None:
        """Return a local Stage 2.7 provider only when explicitly configured."""

        if self.reconstruction_provider == "disabled":
            return None
        if not self.reconstruction_provider_base_url or not self.reconstruction_provider_model:
            raise ValueError(
                "reconstruction_provider_base_url and reconstruction_provider_model are required "
                "for an openai_compatible reconstruction provider"
            )
        if self.reconstruction_provider == "ollama":
            return OllamaReconstructionProvider(
                base_url=self.reconstruction_provider_base_url,
                model=self.reconstruction_provider_model,
                timeout_seconds=self.reconstruction_provider_timeout_seconds,
                release_after_run=self.reconstruction_release_after_run,
            )
        return OpenAICompatibleReconstructionProvider(
            base_url=self.reconstruction_provider_base_url,
            model=self.reconstruction_provider_model,
            timeout_seconds=self.reconstruction_provider_timeout_seconds,
        )

    def contextual_reconstructor(self) -> ContextualReconstructor:
        """Build Stage 2.7 reconstruction with safe local fallback by default."""

        return ContextualReconstructor(self.reconstruction_provider_instance())


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
