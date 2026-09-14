"""Shared Stage 4.0 test helpers: settings injection and a fake provider.

Keeps every Stage 4.0 test hermetic: queue/handoff/executor are always driven
through an injected settings object and a fake provider, so a Gemini key present
in the environment can never trigger live work.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.core.enums import (
    ExternalFactRequirement,
    SemanticProviderMode,
    StrategyDisposition,
    SubstantiveValueKind,
    TransformationIntensity,
    TransformationStrategyType,
)
from app.runtime.heavy_model_lease import NoopHeavyModelLeaseFactory
from app.transformation.policy import DEFAULT_CONFIG, Stage40Config
from app.transformation.providers import TransformationStrategyRequest
from app.transformation.types import (
    TransformationProviderResult,
    TransformationProviderStrategy,
)


class FakeStage40Settings:
    """Minimal duck-typed settings for Stage 4.0 executor/queue/handoff."""

    def __init__(
        self,
        *,
        provider: object | None = None,
        mode: SemanticProviderMode = SemanticProviderMode.ADAPTIVE,
        config: Stage40Config = DEFAULT_CONFIG,
        admission: object | None = None,
    ) -> None:
        self._provider = provider
        self._mode = mode
        self._config = config
        self._admission = admission

    def stage40_config(self) -> Stage40Config:
        return self._config

    def transformation_semantic_mode(self) -> SemanticProviderMode:
        return self._mode

    def transformation_provider(self) -> object | None:
        return self._provider

    def heavy_model_lease_factory(self) -> object:
        return NoopHeavyModelLeaseFactory()

    def gemini_admission_controller(self) -> object | None:
        return self._admission


class FakeTransformationProvider:
    """Deterministic fake provider that never touches the network."""

    provider_name = "fake"

    def __init__(
        self,
        strategies: Sequence[TransformationProviderStrategy] = (),
        *,
        model: str = "fake-model",
    ) -> None:
        self.model = model
        self.calls = 0
        self.behavior = "ok"
        self._strategies = tuple(strategies)

    def select_tier(self, requests: Sequence[TransformationStrategyRequest]) -> str:
        return "ROUTINE"

    def discover(
        self, requests: Sequence[TransformationStrategyRequest]
    ) -> dict[str, TransformationProviderResult]:
        self.calls += 1
        if self.behavior in {"rate_limited", "outage"}:
            from app.transformation.providers import TransformationProviderError

            category = "RATE_LIMITED" if self.behavior == "rate_limited" else "PROVIDER_ERROR"
            raise TransformationProviderError(category)
        return {
            requests[0].candidate_id: TransformationProviderResult(
                candidate_id=requests[0].candidate_id,
                strategies=self._strategies,
                notes="",
                confidence=0.7,
            )
        }

    def release(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "fake", "model": self.model}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


def make_provider_strategy(**overrides: Any) -> TransformationProviderStrategy:
    base: dict[str, object] = {
        "strategy_type": TransformationStrategyType.CONTEXT_HOOK,
        "disposition": StrategyDisposition.RECOMMENDED,
        "intensity": TransformationIntensity.MINIMAL,
        "direction_summary": "Add the missing economic context behind the mentorship claim",
        "added_value_focus": "Supply context on why promotion rates fell for junior staff",
        "substantive_value_kind": SubstantiveValueKind.MISSING_CONTEXT,
        "preservation_requirements": ("Keep the source moment as the hero.",),
        "external_verification_requirement": ExternalFactRequirement.NOT_REQUIRED,
        "verification_requirements": (),
        "rejection_reasons": (),
        "confidence": 0.7,
        "retention_preservation": 0.8,
        "source_moment_damage_risk": 0.2,
        "added_value_density": 0.6,
        "originality_potential": 0.7,
        "source_dominance_risk": 0.42,
        "generic_filler_risk": 0.2,
        "redundant_commentary_risk": 0.2,
        "template_staleness_risk": 0.2,
    }
    base.update(overrides)
    return TransformationProviderStrategy(**base)  # type: ignore[arg-type]


def install_stage40_settings(monkeypatch: Any, settings: FakeStage40Settings) -> None:
    """Route queue and handoff runtime identity through one injected settings."""

    monkeypatch.setattr("app.transformation.queue.get_settings", lambda: settings)
    monkeypatch.setattr("app.transformation.handoff.get_settings", lambda: settings)
