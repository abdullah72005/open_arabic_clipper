"""Regression tests for Sol review fixes: global Gemini admission routing."""

from __future__ import annotations

from app.candidates.executor import _AdmissionBoundSemanticProvider
from app.candidates.providers import (
    ProviderErrorCategory,
    SemanticProviderError,
)
from app.core.enums import AdmissionPriority, RefinementPriority
from app.transcription.reconstruction.routing import AdaptiveRoutingConfig, RoutingMode
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
)


class _FakeGemini:
    provider_name = "gemini"
    model = "gemini-test"

    def __init__(self) -> None:
        self.calls = 0

    def health(self) -> ProviderHealth:
        return ProviderHealth(ProviderAvailability.AVAILABLE, "gemini", self.model, "digest", "")

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "gemini", "model": self.model}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def release(self) -> None:
        return None

    def usage_summary(self) -> dict[str, int]:
        return {}

    def reconstruct_segments(self, requests):
        self.calls += len(requests)
        return {
            request.segment_index: ReconstructionCandidate(
                f"c{request.segment_index}", request.raw_text
            )
            for request in requests
        }


class _Admission:
    def __init__(self, *, admitted: bool) -> None:
        self._admitted = admitted
        self.priorities: list[str] = []
        self.rate_limits = 0

    def acquire(self, priority: AdmissionPriority):
        self.priorities.append(priority.value)
        return type("Decision", (), {"admitted": self._admitted})()

    def record_rate_limit(self, retry_after=None) -> None:
        self.rate_limits += 1

    def runtime_identity(self) -> dict[str, object]:
        return {"admission_policy_version": "test"}


_SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "مرحبا", "raw_text": "مرحبا", "corrected_text": "مرحبا"}
]


def _reconstructor(admission, priority: RefinementPriority) -> ContextualReconstructor:
    return ContextualReconstructor(
        None,
        gemini_provider=_FakeGemini(),  # type: ignore[arg-type]
        routing=AdaptiveRoutingConfig(mode=RoutingMode.GEMINI_ONLY),
        gemini_budget=3,
        priority=priority,
        admission=admission,
    )


def test_reconstruction_admission_priority_mapping() -> None:
    assert (
        _reconstructor(_Admission(admitted=True), RefinementPriority.INDEX)._admission_priority()
        == "AVOID"
    )
    assert (
        _reconstructor(
            _Admission(admitted=True), RefinementPriority.CANDIDATE
        )._admission_priority()
        == "MEDIUM"
    )
    assert (
        _reconstructor(
            _Admission(admitted=True), RefinementPriority.FINAL_CLIP
        )._admission_priority()
        == "CRITICAL"
    )


def test_reconstruction_admission_denial_skips_gemini_call() -> None:
    admission = _Admission(admitted=False)
    reconstructor = _reconstructor(admission, RefinementPriority.CANDIDATE)
    gemini = reconstructor._gemini  # type: ignore[attr-defined]
    result = reconstructor.reconstruct(
        _SEGMENTS,
        language="ar",
        transcription_fingerprint="t",
        correction_version="v",
        target_indexes=[0],
    )
    assert admission.priorities == ["MEDIUM"]
    assert gemini.calls == 0  # type: ignore[attr-defined]
    assert result.segments[0].status is not None


def test_reconstruction_admission_allows_gemini_call() -> None:
    admission = _Admission(admitted=True)
    reconstructor = _reconstructor(admission, RefinementPriority.FINAL_CLIP)
    gemini = reconstructor._gemini  # type: ignore[attr-defined]
    reconstructor.reconstruct(
        _SEGMENTS,
        language="ar",
        transcription_fingerprint="t",
        correction_version="v",
        target_indexes=[0],
    )
    assert admission.priorities == ["CRITICAL"]
    assert gemini.calls == 1  # type: ignore[attr-defined]


class _FakeSemantic:
    provider_name = "gemini"
    model = "gemini-test"

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, requests):
        self.calls += 1
        return {}

    def release(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "gemini", "model": self.model}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()


def test_stage3_admission_denial_prevents_evaluate() -> None:
    inner = _FakeSemantic()
    admission = _Admission(admitted=False)
    provider = _AdmissionBoundSemanticProvider(inner, admission)  # type: ignore[arg-type]
    try:
        provider.evaluate([])
    except SemanticProviderError as error:
        assert error.category is ProviderErrorCategory.RATE_LIMITED
    else:  # pragma: no cover - defensive
        raise AssertionError("expected admission denial")
    assert inner.calls == 0
    assert admission.priorities == ["MEDIUM"]


def test_stage3_admission_allows_evaluate() -> None:
    inner = _FakeSemantic()
    admission = _Admission(admitted=True)
    provider = _AdmissionBoundSemanticProvider(inner, admission)  # type: ignore[arg-type]
    provider.evaluate([])
    assert inner.calls == 1
    assert admission.priorities == ["MEDIUM"]
