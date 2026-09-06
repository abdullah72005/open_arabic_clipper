import json
from types import SimpleNamespace

from typer.testing import CliRunner

from app.cli import app
from app.transcription.reconstruction.types import ProviderAvailability, ProviderHealth


def test_stage_2_transcript_commands_are_exposed() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "transcribe" in result.stdout
    assert "retranscribe" in result.stdout
    assert "reconstruct" in result.stdout
    assert "benchmark-reconstruction" in result.stdout
    assert "reconstruction-health" in result.stdout
    assert "transcript" in result.stdout


def test_benchmark_reconstruction_exposes_model_and_regression_flags() -> None:
    result = CliRunner().invoke(app, ["benchmark-reconstruction", "--help"])

    assert result.exit_code == 0
    assert "--model" in result.stdout
    assert "--allow-known-regression-set" in result.stdout


def test_benchmark_reconstruction_limits_diagnostic_override_to_chernobyl_manifest() -> None:
    result = CliRunner().invoke(
        app,
        [
            "benchmark-reconstruction",
            "other/stage-2-7/chernobyl-reference-v1.json",
            "--allow-known-regression-set",
        ],
    )

    assert result.exit_code == 2


def test_reconstruction_health_prints_provider_identity_and_digest(monkeypatch) -> None:
    health = ProviderHealth(
        ProviderAvailability.AVAILABLE,
        "ollama",
        "qwen3:8b",
        "sha256:abc",
        "model available",
    )
    monkeypatch.setattr(
        "app.cli.get_settings",
        lambda: SimpleNamespace(reconstruction_provider_instance=lambda: _Provider(health)),
    )

    result = CliRunner().invoke(app, ["reconstruction-health"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "availability": "AVAILABLE",
        "provider": "ollama",
        "model": "qwen3:8b",
        "digest": "sha256:abc",
        "detail": "model available",
    }


def test_reconstruction_health_exits_nonzero_when_provider_is_unavailable(monkeypatch) -> None:
    health = ProviderHealth(
        ProviderAvailability.UNAVAILABLE,
        "ollama",
        "qwen3:8b",
        None,
        "configured model qwen3:8b is not installed",
    )
    monkeypatch.setattr(
        "app.cli.get_settings",
        lambda: SimpleNamespace(reconstruction_provider_instance=lambda: _Provider(health)),
    )

    result = CliRunner().invoke(app, ["reconstruction-health"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["detail"] == "configured model qwen3:8b is not installed"


class _Provider:
    def __init__(self, health: ProviderHealth) -> None:
        self._health = health

    def health(self) -> ProviderHealth:
        return self._health
