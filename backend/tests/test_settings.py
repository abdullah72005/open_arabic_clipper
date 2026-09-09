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
    assert settings.reconstruction_provider_model == "qwen3.5:4b"
    assert settings.reconstruction_provider_timeout_seconds == 180
    assert settings.reconstruction_release_after_run is True
    assert settings.reconstruction_provider_max_context_tokens == 4096
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


def test_reconstruction_prompt_budget_settings_have_positive_bounded_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.reconstruction_provider_output_tokens == 256
    assert settings.reconstruction_chat_framing_reserve > 0
    assert settings.reconstruction_safety_reserve > 0


def test_reconstruction_prompt_budget_settings_reject_zero() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(_env_file=None, reconstruction_provider_output_tokens=0)
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(_env_file=None, reconstruction_chat_framing_reserve=0)
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(_env_file=None, reconstruction_safety_reserve=0)


def test_reconstruction_default_mode_is_adaptive() -> None:
    settings = Settings(_env_file=None)

    assert settings.reconstruction_routing_mode == "adaptive"


def test_local_reconstruction_bounds_have_conservative_defaults() -> None:
    """Local Qwen work is ceilinged so a long source cannot run for days."""

    settings = Settings(_env_file=None)

    assert settings.local_reconstruction_max_targets_per_job == 64
    assert settings.local_reconstruction_max_wall_seconds == 1200
    assert settings.reconstruction_provider_batch_windows == 8
    assert settings.reconstruction_provider_batch_characters == 24_000


def test_local_reconstruction_bounds_parse_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFACTORY_LOCAL_RECONSTRUCTION_MAX_TARGETS_PER_JOB", "16")
    monkeypatch.setenv("CLIPFACTORY_LOCAL_RECONSTRUCTION_MAX_WALL_SECONDS", "300")
    monkeypatch.setenv("CLIPFACTORY_RECONSTRUCTION_PROVIDER_BATCH_WINDOWS", "4")
    monkeypatch.setenv("CLIPFACTORY_RECONSTRUCTION_PROVIDER_BATCH_CHARACTERS", "12000")

    settings = Settings(_env_file=None)

    assert settings.local_reconstruction_max_targets_per_job == 16
    assert settings.local_reconstruction_max_wall_seconds == 300
    assert settings.reconstruction_provider_batch_windows == 4
    assert settings.reconstruction_provider_batch_characters == 12_000


def test_local_reconstruction_bounds_reject_zero() -> None:
    # A zero wall budget is never valid; a zero target budget is a valid way to
    # disable local reconstruction entirely.
    with pytest.raises(ValidationError):
        Settings(_env_file=None, local_reconstruction_max_wall_seconds=0)
    zero_targets = Settings(_env_file=None, local_reconstruction_max_targets_per_job=0)
    assert zero_targets.local_reconstruction_max_targets_per_job == 0


def test_gemini_temperature_defaults_to_zero_and_parses() -> None:
    settings = Settings(_env_file=None)

    assert settings.gemini_temperature == 0.0
    assert settings.gemini_api_version == "v1"

    overridden = Settings(_env_file=None, gemini_temperature=0.5, gemini_api_version="v1beta")
    assert overridden.gemini_temperature == 0.5
    assert overridden.gemini_api_version == "v1beta"


def test_ollama_compose_hardware_safeguards_are_configured() -> None:
    """Compose pins Ollama parallelism, loaded models, queue, context, CPU, RAM,
    and swap to safe defaults for the documented ~10.7 GiB WSL environment."""

    import re
    from pathlib import Path

    compose = (Path(__file__).resolve().parents[2] / "compose.yaml").read_text(encoding="utf-8")
    assert re.search(r"OLLAMA_NUM_PARALLEL:\s*\"1\"", compose)
    assert re.search(r"OLLAMA_MAX_LOADED_MODELS:\s*\"1\"", compose)
    assert re.search(r"OLLAMA_MAX_QUEUE:\s*\"\$\{OLLAMA_MAX_QUEUE:-4\}\"", compose)
    assert re.search(r"OLLAMA_CONTEXT_LENGTH:\s*\"\$\{OLLAMA_CONTEXT_LENGTH:-4096\}\"", compose)
    assert re.search(r"cpus:\s*\"\$\{OLLAMA_CPUS:-9\}\"", compose)
    assert re.search(r"mem_limit:\s*\"\$\{OLLAMA_MEM_LIMIT:-6g\}\"", compose)
    assert re.search(r"memswap_limit:\s*\"\$\{OLLAMA_MEMSWAP_LIMIT:-8g\}\"", compose)


def test_gemini_defaults_and_absent_key_do_not_block_startup() -> None:
    settings = Settings(_env_file=None)

    assert settings.gemini_model == "gemini-3.8-flash"
    assert settings.gemini_thinking_level == "low"
    assert settings.gemini_timeout_seconds == 30.0
    assert settings.gemini_max_targets_per_job == 5
    assert settings.gemini_api_key_present is False
    assert settings.gemini_provider_instance() is None


def test_gemini_thinking_level_validates_supported_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFACTORY_GEMINI_THINKING_LEVEL", "high")

    assert Settings(_env_file=None).gemini_thinking_level == "high"

    monkeypatch.setenv("CLIPFACTORY_GEMINI_THINKING_LEVEL", "extreme")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_gemini_key_is_presence_detected_from_prefixed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFACTORY_GEMINI_API_KEY", "test-key-not-committed")

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key_present is True
    assert settings.gemini_provider_instance() is not None


def test_gemini_key_is_presence_detected_from_unprefixed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-committed")

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key_present is True


def test_gemini_key_value_is_never_returned_by_public_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-super-secret-value")

    settings = Settings(_env_file=None)
    identity = settings.contextual_reconstructor().runtime_identity()

    serialized = repr(identity)
    assert "AIza-super-secret-value" not in serialized
    assert "gemini_api_key" not in serialized.casefold()


def test_gemini_key_is_secret_in_repr_and_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-super-secret-value")

    settings = Settings(_env_file=None)

    assert "AIza-super-secret-value" not in repr(settings)
    assert "AIza-super-secret-value" not in str(settings.model_dump())
    assert "AIza-super-secret-value" not in str(settings.model_dump_json())
    assert settings.gemini_api_key_present is True


def test_gemini_key_is_not_echoed_in_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-super-secret-value")

    with pytest.raises(ValidationError) as raised:
        Settings(_env_file=None, gemini_retry_attempts=9)

    assert "AIza-super-secret-value" not in str(raised.value)


def test_gemini_provider_unwraps_secret_only_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPFACTORY_GEMINI_API_KEY", "AIza-super-secret-value")

    settings = Settings(_env_file=None)
    provider = settings.gemini_provider_instance()

    assert provider is not None
    assert "AIza-super-secret-value" not in repr(provider.runtime_identity())
    assert "AIza-super-secret-value" not in repr(provider)
