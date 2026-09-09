from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.runtime.heavy_model_lease import HeavyModelLeaseFactory
from app.transcription.correction import ContextualCorrector, CorrectionConfig
from app.transcription.providers import CorrectionProvider, OpenAICompatibleCorrectionProvider
from app.transcription.reconstruction import ContextualReconstructor
from app.transcription.reconstruction.gemini import GeminiReconstructionProvider
from app.transcription.reconstruction.ollama import OllamaReconstructionProvider
from app.transcription.reconstruction.providers import (
    OpenAICompatibleReconstructionProvider,
    ReconstructionProvider,
)
from app.transcription.reconstruction.routing import AdaptiveRoutingConfig, RoutingMode
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
    reconstruction_provider_model: str | None = Field(default="qwen3.5:4b", max_length=256)
    reconstruction_provider_timeout_seconds: float = Field(default=180.0, gt=0, le=300)
    reconstruction_release_after_run: bool = True
    reconstruction_provider_max_context_tokens: int = Field(default=4_096, gt=0, le=32_768)
    reconstruction_provider_output_tokens: int = Field(default=256, gt=0, le=4_096)
    reconstruction_chat_framing_reserve: int = Field(default=64, gt=0, le=4_096)
    reconstruction_safety_reserve: int = Field(default=128, gt=0, le=4_096)
    reconstruction_provider_batch_windows: int = Field(default=8, gt=0, le=16)
    reconstruction_provider_batch_characters: int = Field(default=24_000, gt=0, le=48_000)
    local_reconstruction_max_targets_per_job: int = Field(default=64, ge=0, le=100_000)
    local_reconstruction_max_wall_seconds: float = Field(default=1_200.0, gt=0, le=86_400)
    reconstruction_routing_mode: Literal["local_only", "adaptive", "gemini_only"] = "adaptive"
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CLIPFACTORY_GEMINI_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_model: str = Field(default="gemini-3.8-flash", max_length=256)
    gemini_thinking_level: Literal["low", "medium", "high"] = "low"
    gemini_temperature: float = Field(default=0.0, ge=0, le=2)
    gemini_api_version: str = Field(default="v1", min_length=1, max_length=32)
    gemini_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    gemini_retry_attempts: int = Field(default=1, ge=0, le=3)
    gemini_retry_backoff_seconds: float = Field(default=1.5, gt=0, le=30)
    gemini_max_targets_per_job: int = Field(default=5, ge=0, le=100)
    gemini_max_output_tokens: int = Field(default=1024, gt=0, le=4_096)
    heavy_model_lease_ttl_seconds: float = Field(default=300.0, gt=0)
    heavy_model_lease_renewal_interval_seconds: float = Field(default=60.0, gt=0)
    heavy_model_lease_acquisition_timeout_seconds: float = Field(default=15.0, gt=0)
    transcription_queue_concurrency: int = Field(default=1, gt=0)
    cors_origins: list[str] = ["http://localhost:3301"]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _validate_heavy_model_lease(self) -> "Settings":
        if (
            self.heavy_model_lease_ttl_seconds
            <= 2 * self.heavy_model_lease_renewal_interval_seconds
        ):
            raise ValueError("heavy model lease TTL must exceed two renewal intervals")
        return self

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

    def reconstruction_provider_instance(
        self, model: str | None = None
    ) -> ReconstructionProvider | None:
        """Return a local Stage 2.7 provider only when explicitly configured."""

        if self.reconstruction_provider == "disabled":
            return None
        resolved_model = model or self.reconstruction_provider_model
        if not self.reconstruction_provider_base_url or not resolved_model:
            raise ValueError(
                "reconstruction_provider_base_url and reconstruction_provider_model are required "
                "for an openai_compatible reconstruction provider"
            )
        if self.reconstruction_provider == "ollama":
            return OllamaReconstructionProvider(
                base_url=self.reconstruction_provider_base_url,
                model=resolved_model,
                timeout_seconds=self.reconstruction_provider_timeout_seconds,
                release_after_run=self.reconstruction_release_after_run,
                max_context_tokens=self.reconstruction_provider_max_context_tokens,
                output_tokens=self.reconstruction_provider_output_tokens,
                chat_framing_reserve=self.reconstruction_chat_framing_reserve,
                safety_reserve=self.reconstruction_safety_reserve,
            )
        return OpenAICompatibleReconstructionProvider(
            base_url=self.reconstruction_provider_base_url,
            model=resolved_model,
            timeout_seconds=self.reconstruction_provider_timeout_seconds,
            max_context_tokens=self.reconstruction_provider_max_context_tokens,
            output_tokens=self.reconstruction_provider_output_tokens,
            chat_framing_reserve=self.reconstruction_chat_framing_reserve,
            safety_reserve=self.reconstruction_safety_reserve,
        )

    def contextual_reconstructor(self) -> ContextualReconstructor:
        """Build Stage 2.7 reconstruction with safe fallback and optional hosted Gemini."""

        return ContextualReconstructor(
            self.reconstruction_provider_instance(),
            gemini_provider=self.gemini_provider_instance(),
            routing=AdaptiveRoutingConfig(mode=RoutingMode(self.reconstruction_routing_mode)),
            gemini_budget=self.gemini_max_targets_per_job,
            batch_windows=self.reconstruction_provider_batch_windows,
            batch_characters=self.reconstruction_provider_batch_characters,
            local_max_targets=self.local_reconstruction_max_targets_per_job,
            local_wall_seconds=self.local_reconstruction_max_wall_seconds,
        )

    def gemini_provider_instance(self) -> GeminiReconstructionProvider | None:
        """Return the hosted Gemini provider only when a key is configured."""

        key = self.gemini_api_key
        if key is None or not key.get_secret_value():
            return None
        return GeminiReconstructionProvider(
            api_key=key.get_secret_value(),
            model=self.gemini_model,
            timeout_seconds=self.gemini_timeout_seconds,
            retry_attempts=self.gemini_retry_attempts,
            retry_backoff_seconds=self.gemini_retry_backoff_seconds,
            max_output_tokens=self.gemini_max_output_tokens,
            thinking_level=self.gemini_thinking_level,
            temperature=self.gemini_temperature,
            api_version=self.gemini_api_version,
        )

    @property
    def gemini_api_key_present(self) -> bool:
        """Presence-only check that never exposes the key value."""

        if self.gemini_api_key is None:
            return False
        return bool(self.gemini_api_key.get_secret_value())

    def heavy_model_lease_factory(self) -> HeavyModelLeaseFactory:
        """Build the Redis-backed lease factory that serializes heavy models."""

        from redis import Redis

        return HeavyModelLeaseFactory(
            redis=Redis.from_url(self.redis_url),
            ttl_seconds=self.heavy_model_lease_ttl_seconds,
            renewal_interval_seconds=self.heavy_model_lease_renewal_interval_seconds,
            acquisition_timeout_seconds=self.heavy_model_lease_acquisition_timeout_seconds,
        )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
