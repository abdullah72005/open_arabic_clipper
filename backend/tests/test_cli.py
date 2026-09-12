import json
import re
from types import SimpleNamespace
from uuid import UUID

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.core.enums import SemanticProviderMode
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
    assert "benchmark-index-replay" in help_text
    assert "reconstruction-health" in help_text
    assert "recover-heavy-model" in help_text
    assert "transcript" in help_text


def test_benchmark_index_replay_emits_read_only_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    source_id = UUID("4037a813-6fe6-4c83-96ff-e5cd4bf210ce")
    source = SimpleNamespace(
        rights_status="UNKNOWN",
        media_origin="OTHER",
        provenance_metadata={},
        dialect_profile_override=None,
    )
    transcript = SimpleNamespace(duration=10.0)
    artifact = SimpleNamespace(duration=10.0, output_path="source/audio.wav")
    analysis = SimpleNamespace(silence_intervals=[], features=[])
    candidate_analysis = SimpleNamespace(semantic_provider_mode=SemanticProviderMode.DETERMINISTIC)
    candidate = SimpleNamespace(candidate_key="candidate-1", disposition="CANDIDATE")

    class Query:
        def __init__(self, value: object) -> None:
            self.value = value

        def filter(self, *_args: object) -> "Query":
            return self

        def order_by(self, *_args: object) -> "Query":
            return self

        def limit(self, *_args: object) -> "Query":
            return self

        def one_or_none(self) -> object:
            return self.value

        def all(self) -> list[object]:
            return self.value if isinstance(self.value, list) else []

    class Session:
        def __init__(self) -> None:
            self.values = iter(
                [transcript, artifact, analysis, candidate_analysis, [candidate], []]
            )

        def __enter__(self) -> "Session":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def get(self, *_args: object) -> object:
            return source

        def query(self, *_args: object) -> Query:
            return Query(next(self.values))

    class Lease:
        ownership_lost = False

        def __enter__(self) -> "Lease":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    class Settings:
        def transcription_options(self) -> object:
            return object()

        def heavy_model_lease_factory(self) -> object:
            return SimpleNamespace(acquire=lambda **_kwargs: Lease())

        def stage3_config(self) -> object:
            return SimpleNamespace(novelty_corpus_limit=1)

        def contextual_corrector(self) -> object:
            return object()

    class Report:
        def as_dict(self) -> dict[str, object]:
            return {"wall_clock_seconds": 1.0}

    monkeypatch.setattr("app.cli.get_settings", lambda: Settings())
    monkeypatch.setattr("app.cli.create_session_factory", lambda: lambda: Session())
    monkeypatch.setattr(
        "app.cli._storage", lambda: SimpleNamespace(resolve=lambda *_args: "audio.wav")
    )
    monkeypatch.setattr(
        "app.cli.WhisperEngine", lambda: SimpleNamespace(last_child_peak_rss=lambda: 123)
    )
    monkeypatch.setattr(
        "app.cli.transcribe_for_benchmark",
        lambda *_args: (Report(), SimpleNamespace(duration=10.0)),
    )
    monkeypatch.setattr(
        "app.cli.replay_index_candidates",
        lambda **_kwargs: SimpleNamespace(as_dict=lambda: {"retained_candidate_overlap_count": 1}),
    )

    result = CliRunner().invoke(app, ["benchmark-index-replay", str(source_id)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "benchmark": {"wall_clock_seconds": 1.0},
        "child_peak_rss_bytes": 123,
        "replay": {"retained_candidate_overlap_count": 1},
    }


def test_recover_heavy_model_reports_clear_when_no_unsafe_state(monkeypatch) -> None:
    class Factory:
        def unsafe_recorded(self) -> bool:
            return False

    class Settings:
        def heavy_model_lease_factory(self) -> Factory:
            return Factory()

    monkeypatch.setattr("app.cli.get_settings", lambda: Settings())
    result = CliRunner().invoke(app, ["recover-heavy-model"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "CLEAR"


def test_recover_heavy_model_blocks_while_model_still_resident(monkeypatch) -> None:
    class Factory:
        def unsafe_recorded(self) -> bool:
            return True

        def unsafe_reason(self) -> str:
            return "model still resident after unload timeout"

    class Provider:
        def is_model_resident(self) -> bool:
            return True

    class Settings:
        reconstruction_provider_model = "qwen3.5:4b"

        def heavy_model_lease_factory(self) -> Factory:
            return Factory()

        def reconstruction_provider_instance(self) -> Provider:
            return Provider()

    monkeypatch.setattr("app.cli.get_settings", lambda: Settings())
    result = CliRunner().invoke(app, ["recover-heavy-model"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "UNSAFE"


def test_recover_heavy_model_clears_only_after_confirmed_not_resident(monkeypatch) -> None:
    cleared: list[str] = []

    class Factory:
        def unsafe_recorded(self) -> bool:
            return True

        def recover(self) -> None:
            cleared.append("recovered")

    class Provider:
        def is_model_resident(self) -> bool:
            return False

    class Settings:
        def heavy_model_lease_factory(self) -> Factory:
            return Factory()

        def reconstruction_provider_instance(self) -> Provider:
            return Provider()

    monkeypatch.setattr("app.cli.get_settings", lambda: Settings())
    result = CliRunner().invoke(app, ["recover-heavy-model"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "RECOVERED"
    assert cleared == ["recovered"]


def test_recover_heavy_model_resident_changes_no_redis_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery while the model is still resident must fail and leave Redis untouched."""

    from test_heavy_model_lease import FakeClock, FakeRedis

    from app.runtime.heavy_model_lease import HeavyModelLeaseFactory

    _LEASE_KEY = "clipfactory:heavy-model"
    now = FakeClock()
    redis = FakeRedis(now)
    factory = HeavyModelLeaseFactory(
        redis=redis,
        ttl_seconds=300,
        renewal_interval_seconds=60,
        acquisition_timeout_seconds=5,
        snapshotter=lambda: SimpleNamespace(effective_capacity=1),
    )
    factory.mark_unsafe(reason="model still resident after unload timeout")
    redis.set(_LEASE_KEY, "stale-token", nx=True, px=300000)

    class Provider:
        def is_model_resident(self) -> bool:
            return True

    class Settings:
        reconstruction_provider_model = "qwen3.5:4b"

        def heavy_model_lease_factory(self) -> HeavyModelLeaseFactory:
            return factory

        def reconstruction_provider_instance(self) -> Provider:
            return Provider()

    monkeypatch.setattr("app.cli.get_settings", lambda: Settings())
    result = CliRunner().invoke(app, ["recover-heavy-model"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "UNSAFE"
    assert factory.unsafe_recorded() is True
    assert redis.get(_LEASE_KEY) == "stale-token"


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


def test_benchmark_cli_cannot_enter_when_heavy_lease_busy(monkeypatch, tmp_path) -> None:
    from app.runtime.heavy_model_lease import HeavyModelLeaseBusy
    from app.transcription.service import TranscriptionOptions

    class BusyLeaseFactory:
        def acquire(self, *, purpose: str) -> object:
            raise HeavyModelLeaseBusy("heavy-model lease is busy")

    root = tmp_path / "storage"
    manifest_dir = root / "benchmarks" / "stage-2-7"
    manifest_dir.mkdir(parents=True)
    manifest = {
        "version": "stage-2-7-private-v1",
        "split": "test",
        "sources": [{"id": "s", "path": "s/v.webm", "authorized": True}],
        "clips": [
            {
                "id": "c1",
                "source_id": "s",
                "topic": "h",
                "start_seconds": 0,
                "end_seconds": 30,
                "categories": ["narrative"],
                "reference_segments": [{"segment_index": 0, "text": "x", "reviewed": True}],
            }
        ],
        "known_regression_set": True,
    }
    (manifest_dir / "known-regression-v1.json").write_text(json.dumps(manifest), encoding="utf-8")

    class FakeSettings:
        storage_root = root
        reconstruction_provider = "ollama"

        def reconstruction_provider_instance(self, model: str | None = None) -> None:
            return None

        def transcription_options(self) -> TranscriptionOptions:
            return TranscriptionOptions("large-v3-turbo", "cpu", "int8", 5)

        def contextual_corrector(self) -> None:
            return None

        def heavy_model_lease_factory(self) -> BusyLeaseFactory:
            return BusyLeaseFactory()

    monkeypatch.setattr("app.cli.get_settings", lambda: FakeSettings())

    result = CliRunner().invoke(
        app,
        [
            "benchmark-reconstruction",
            "stage-2-7/known-regression-v1.json",
            "--allow-known-regression-set",
        ],
    )

    assert result.exit_code == 1
    assert "busy" in result.stdout


class _Provider:
    def __init__(self, health: ProviderHealth) -> None:
        self._health = health

    def health(self) -> ProviderHealth:
        return self._health
