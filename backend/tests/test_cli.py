import json
import re
from types import SimpleNamespace

from typer.testing import CliRunner

from app.cli import app
from app.runtime.memory import MemoryReadError, MemorySnapshot
from app.transcription.reconstruction.types import ProviderAvailability, ProviderHealth

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_HELP_ENV = {"COLUMNS": "120", "LINES": "40"}


def _plain_help(stdout: str) -> str:
    """Strip rich ANSI and wrapping so help assertions survive any terminal width."""

    return re.sub(r"\s+", "", _ANSI.sub("", stdout))


def _help(command: list[str]) -> str:
    result = CliRunner().invoke(app, command, env=_HELP_ENV)
    assert result.exit_code == 0
    return _plain_help(result.stdout)


def test_stage_2_transcript_commands_are_exposed() -> None:
    help_text = _help(["--help"])
    assert "transcribe" in help_text
    assert "retranscribe" in help_text
    assert "reconstruct" in help_text
    assert "benchmark-reconstruction" in help_text
    assert "reconstruction-health" in help_text
    assert "transcript" in help_text


def test_benchmark_reconstruction_exposes_model_and_regression_flags() -> None:
    help_text = _help(["benchmark-reconstruction", "--help"])
    assert "--model" in help_text
    assert "--allow-known-regression-set" in help_text


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


def test_diagnose_memory_json_prints_effective_capacity(monkeypatch) -> None:
    snapshot = MemorySnapshot(0.0, 7803048 * 1024, 1, 1, 1, None, None, None, 1)
    monkeypatch.setattr("app.cli.capture_memory", lambda **kwargs: snapshot)

    result = CliRunner().invoke(app, ["diagnose-memory", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["linux_total"] == 7803048 * 1024
    assert payload["effective_capacity"] == 7803048 * 1024


def test_diagnose_memory_exits_nonzero_when_required_inputs_unreadable(monkeypatch) -> None:
    def raise_error(**kwargs: object) -> MemorySnapshot:
        raise MemoryReadError("meminfo unreadable")

    monkeypatch.setattr("app.cli.capture_memory", raise_error)

    result = CliRunner().invoke(app, ["diagnose-memory", "--json"])

    assert result.exit_code != 0
    assert "meminfo unreadable" in result.stdout


class _Provider:
    def __init__(self, health: ProviderHealth) -> None:
        self._health = health

    def health(self) -> ProviderHealth:
        return self._health
