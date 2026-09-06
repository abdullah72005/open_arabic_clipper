from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.settings import Settings
from app.transcription.reconstruction.ollama import OllamaReconstructionProvider
from app.transcription.reconstruction.providers import OpenAICompatibleReconstructionProvider


def test_settings_uses_storage_root_from_explicit_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage_root = tmp_path / "clipfactory-storage"
    monkeypatch.setenv("CLIPFACTORY_STORAGE_ROOT", str(storage_root))

    settings = Settings()

    assert settings.storage_root == storage_root


def test_settings_rejects_non_positive_upload_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLIPFACTORY_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("CLIPFACTORY_MAX_UPLOAD_BYTES", "0")

    with pytest.raises(ValidationError, match="greater than 0"):
        Settings()


def test_settings_default_to_large_v3_turbo_transcription() -> None:
    """Default transcription uses the requested high-quality turbo model."""

    settings = Settings()

    assert settings.whisper_model == "large-v3-turbo"
    assert settings.whisper_device == "auto"
    assert settings.whisper_cpu_compute_type == "int8"


def test_settings_accept_forced_transcription_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator can override auto-detection for a known-language source."""

    monkeypatch.setenv("CLIPFACTORY_WHISPER_LANGUAGE", "ar")

    assert Settings().whisper_language == "ar"


def test_settings_builds_auto_transcription_options() -> None:
    """Auto mode defers device selection while preserving configured output options."""

    options = Settings().transcription_options()

    assert options.model == "large-v3-turbo"
    assert options.device == "auto"
    assert options.language is None


def test_settings_transcription_options_honor_compute_type_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cpu/cuda compute-type env vars reach the worker-side engine."""

    monkeypatch.setenv("CLIPFACTORY_WHISPER_CPU_COMPUTE_TYPE", "int8_float16")
    monkeypatch.setenv("CLIPFACTORY_WHISPER_CUDA_COMPUTE_TYPE", "float32")

    options = Settings().transcription_options()

    assert options.cpu_compute_type == "int8_float16"
    assert options.cuda_compute_type == "float32"


def test_settings_explicit_compute_type_overrides_per_device_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFACTORY_WHISPER_COMPUTE_TYPE", "float16")

    options = Settings().transcription_options()

    assert options.compute_type == "float16"


def test_settings_builds_opt_in_local_correction_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local-first default is lexicon-only; compatible LLM use is explicit configuration."""

    monkeypatch.setenv("CLIPFACTORY_CORRECTION_PROVIDER", "openai_compatible")
    monkeypatch.setenv("CLIPFACTORY_CORRECTION_PROVIDER_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("CLIPFACTORY_CORRECTION_PROVIDER_MODEL", "qwen-local")
    monkeypatch.setenv("CLIPFACTORY_CORRECTION_PROVIDER_BATCH_SIZE", "8")

    settings = Settings()

    assert settings.correction_config().provider_batch_size == 8
    provider = settings.correction_provider_instance()
    assert provider is not None


def test_reconstruction_defaults_to_managed_local_provider() -> None:
    settings = Settings(_env_file=None)

    assert settings.reconstruction_provider == "ollama"
    assert settings.reconstruction_provider_base_url == "http://ollama:11434"
    assert settings.reconstruction_provider_model == "qwen3:8b"
    assert settings.reconstruction_provider_timeout_seconds == 180
    assert settings.reconstruction_release_after_run is True
    assert isinstance(settings.reconstruction_provider_instance(), OllamaReconstructionProvider)


def test_reconstruction_retains_explicit_disabled_mode() -> None:
    settings = Settings(_env_file=None, reconstruction_provider="disabled")

    assert settings.reconstruction_provider_instance() is None


def test_reconstruction_retains_explicit_openai_compatible_mode() -> None:
    settings = Settings(
        _env_file=None,
        reconstruction_provider="openai_compatible",
        reconstruction_provider_base_url="http://provider:8080",
        reconstruction_provider_model="local-model",
    )

    assert isinstance(
        settings.reconstruction_provider_instance(),
        OpenAICompatibleReconstructionProvider,
    )
