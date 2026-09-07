from app.core.enums import ReconstructionStatus
from app.transcription.reconstruction.providers import (
    ProviderResponseError,
    ReconstructionRequest,
)
from app.transcription.reconstruction.service import ContextualReconstructor, select_final_text
from app.transcription.reconstruction.types import (
    ConfidenceLevel,
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
    ResolutionScores,
)


class OnePassProvider:
    def __init__(self, candidate: ReconstructionCandidate | None = None) -> None:
        self.release_calls = 0
        self.requests: list[ReconstructionRequest] = []
        self._candidate = candidate or ReconstructionCandidate(
            "provider-0",
            "ضخمة",
            scores=ResolutionScores(1.0, 1.0, 1.0, 1.0, 1.0),
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "ollama", "qwen3.5:4b", "sha256:x", "ok"
        )

    def release(self) -> None:
        self.release_calls += 1

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.requests.extend(requests)
        return {request.segment_index: self._candidate for request in requests}


class ReleaseFailingProvider(OnePassProvider):
    def release(self) -> None:
        raise RuntimeError("release failed")


def test_reconstructor_applies_only_high_contextual_candidate() -> None:
    """A high-scoring candidate becomes automatic final text."""

    provider = OnePassProvider()
    result = ContextualReconstructor(provider).reconstruct(
        [{"start": 0.0, "end": 1.0, "text": "دخم", "corrected_text": "دخم"}],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    segment = result.segments[0]
    assert segment.contextual_reconstructed_text == "ضخمة"
    assert segment.confidence_level is ConfidenceLevel.HIGH
    assert segment.applied is True
    assert result.contextual_reconstructed_text == "ضخمة"
    assert provider.release_calls == 1
    assert segment.reconstruction_method == "ollama:qwen3.5:4b"


def test_operator_text_precedes_provider_candidate_and_is_manual_override() -> None:
    result = ContextualReconstructor(OnePassProvider()).reconstruct(
        [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "دخم",
                "corrected_text": "دخم",
                "operator_text": "يدوي",
            }
        ],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )
    assert result.segments[0].contextual_reconstructed_text == "يدوي"
    assert result.segments[0].status.value == "MANUAL_OVERRIDE"


def test_reconstructor_preserves_result_when_release_fails_after_success() -> None:
    """Cleanup failures stay out of the result path and only add bounded metadata."""

    result = ContextualReconstructor(ReleaseFailingProvider()).reconstruct(
        [{"start": 0.0, "end": 1.0, "text": "دخم", "corrected_text": "دخم"}],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert result.segments[0].contextual_reconstructed_text == "ضخمة"
    assert result.contextual_reconstructed_text == "ضخمة"
    assert result.metadata["release_warning"] == "provider_release_failed"


def test_reconstructor_without_provider_preserves_stage_2_5_text() -> None:
    """Disabled local models leave useful Stage 2.5 output untouched and auditable."""

    result = ContextualReconstructor(None).reconstruct(
        [{"start": 0.0, "end": 1.0, "text": "خطي بالك", "corrected_text": "خلي بالك"}],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert result.segments[0].contextual_reconstructed_text == "خلي بالك"
    assert result.segments[0].applied is False


def test_reconstructor_falls_back_only_for_expected_provider_failures() -> None:
    class BrokenProvider:
        def health(self) -> ProviderHealth:
            return ProviderHealth(ProviderAvailability.AVAILABLE, "test", "test", "sha256:x", "ok")

        def reconstruct_segments(
            self, requests: list[ReconstructionRequest]
        ) -> dict[int, ReconstructionCandidate]:
            raise ProviderResponseError("invalid JSON")

        def release(self) -> None:
            pass

    provider = BrokenProvider()
    result = ContextualReconstructor(provider).reconstruct(
        [{"start": 0.0, "end": 1.0, "text": "خطي بالك", "corrected_text": "خلي بالك"}],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert result.segments[0].contextual_reconstructed_text == "خلي بالك"
    assert result.segments[0].quality_flags[0].value == "RECONSTRUCTION_PROVIDER_ERROR"
    assert result.segments[0].status is ReconstructionStatus.PROVIDER_UNAVAILABLE


def test_reconstructor_surfaces_provider_unavailable_when_model_cannot_run() -> None:
    class UnavailableProvider:
        def health(self) -> ProviderHealth:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE, "ollama", "qwen3:8b", None, "OOM"
            )

        def reconstruct_segments(
            self, requests: list[ReconstructionRequest]
        ) -> dict[int, ReconstructionCandidate]:
            raise AssertionError("must not call provider when unavailable")

        def release(self) -> None:
            pass

    result = ContextualReconstructor(UnavailableProvider()).reconstruct(
        [{"start": 0.0, "end": 1.0, "text": "raw", "corrected_text": "corrected"}],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert result.segments[0].status is ReconstructionStatus.PROVIDER_UNAVAILABLE
    assert result.segments[0].contextual_reconstructed_text == "corrected"


def test_final_text_priority_keeps_manual_text_above_reconstruction() -> None:
    """Operator wording always wins over every automatic transcript layer."""

    assert (
        select_final_text(
            operator_text="manual",
            reconstructed="high",
            reconstruction_applied=True,
            level=ConfidenceLevel.HIGH,
            corrected="stage25",
            raw="raw",
        )
        == "manual"
    )


def test_reconstructor_sends_small_context_window() -> None:
    """The provider receives only local context, not the full transcript."""

    provider = OnePassProvider()
    ContextualReconstructor(provider).reconstruct(
        [
            {"start": 0.0, "end": 1.0, "text": "a", "corrected_text": "a"},
            {"start": 1.0, "end": 2.0, "text": "b", "corrected_text": "b"},
            {"start": 2.0, "end": 3.0, "text": "c", "corrected_text": "c"},
            {"start": 3.0, "end": 4.0, "text": "d", "corrected_text": "d"},
            {"start": 4.0, "end": 5.0, "text": "e", "corrected_text": "e"},
        ],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert len(provider.requests) == 5
    for request in provider.requests:
        assert len(request.previous) <= 2
        assert len(request.following) <= 2
        assert request.segment_index is not None
