from app.core.enums import ReconstructionStatus, RefinementPriority
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
    UnloadOutcome,
)


class OnePassProvider:
    def __init__(self, candidate: ReconstructionCandidate | None = None) -> None:
        self.release_calls = 0
        self.requests: list[ReconstructionRequest] = []
        self._candidate = candidate or ReconstructionCandidate(
            "provider-0",
            "ضخمة",
            provider_confidence=1.0,
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "ollama", "qwen3.5:4b", "sha256:x", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "ollama",
            "model": "qwen3.5:4b",
            "digest": "sha256:x",
            "prompt_hash": "p",
            "schema_version": "s",
            "max_context_tokens": 4096,
            "output_tokens": 256,
            "chat_framing_reserve": 64,
            "safety_reserve": 128,
            "confidence_policy_version": "one-pass-provider-confidence-v1",
            "validation_version": "stage-2-7-validation-v1",
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

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
    result = ContextualReconstructor(provider, priority=RefinementPriority.CANDIDATE).reconstruct(
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
    result = ContextualReconstructor(
        OnePassProvider(), priority=RefinementPriority.CANDIDATE
    ).reconstruct(
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
    assert result.segments[0].contextual_reconstructed_text == "دخم"
    assert result.segments[0].status.value == "MANUAL_OVERRIDE"


def test_reconstructor_preserves_result_when_release_fails_after_success() -> None:
    """Cleanup failures stay out of the result path and only add bounded metadata."""

    result = ContextualReconstructor(
        ReleaseFailingProvider(), priority=RefinementPriority.CANDIDATE
    ).reconstruct(
        [{"start": 0.0, "end": 1.0, "text": "دخم", "corrected_text": "دخم"}],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert result.segments[0].contextual_reconstructed_text == "ضخمة"
    assert result.contextual_reconstructed_text == "ضخمة"
    assert result.metadata["release_warning"] == "provider_release_failed"


def test_reconstructor_records_unload_outcome_without_corrupting_text() -> None:
    """A verified unload warning is recorded; successful reconstruction text stays."""

    class WarningReleaseProvider(OnePassProvider):
        def release(self) -> UnloadOutcome:
            return UnloadOutcome(True, False, 1.0, "model still resident after unload timeout")

    result = ContextualReconstructor(
        WarningReleaseProvider(), priority=RefinementPriority.CANDIDATE
    ).reconstruct(
        [{"start": 0.0, "end": 1.0, "text": "دخم", "corrected_text": "دخم"}],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert result.segments[0].contextual_reconstructed_text == "ضخمة"
    assert result.metadata["release_warning"] == "model still resident after unload timeout"
    assert result.metadata["unload_outcome"]["requested"] is True
    assert result.metadata["unload_outcome"]["confirmed"] is False


def test_reconstructor_without_provider_preserves_stage_2_5_text() -> None:
    """Disabled local models leave useful Stage 2.5 output untouched and auditable."""

    result = ContextualReconstructor(None, priority=RefinementPriority.CANDIDATE).reconstruct(
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

        def runtime_identity(self) -> dict[str, object]:
            return {"provider": "test", "model": "test", "digest": "sha256:x"}

        def refresh_runtime_identity(self) -> dict[str, object]:
            return self.runtime_identity()

        def reconstruct_segments(
            self, requests: list[ReconstructionRequest]
        ) -> dict[int, ReconstructionCandidate]:
            raise ProviderResponseError("invalid JSON")

        def release(self) -> None:
            pass

    provider = BrokenProvider()
    result = ContextualReconstructor(provider, priority=RefinementPriority.CANDIDATE).reconstruct(
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

        def runtime_identity(self) -> dict[str, object]:
            return {"provider": "ollama", "model": "qwen3:8b", "digest": "digest_unavailable"}

        def refresh_runtime_identity(self) -> dict[str, object]:
            return self.runtime_identity()

        def reconstruct_segments(
            self, requests: list[ReconstructionRequest]
        ) -> dict[int, ReconstructionCandidate]:
            raise AssertionError("must not call provider when unavailable")

        def release(self) -> None:
            pass

    result = ContextualReconstructor(
        UnavailableProvider(), priority=RefinementPriority.CANDIDATE
    ).reconstruct(
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


def test_deterministic_validation_failure_is_never_applied() -> None:
    """A protected-token change is rejected before the confidence policy runs."""

    provider = OnePassProvider(
        ReconstructionCandidate("provider-0", "الرئيس 70", provider_confidence=1.0)
    )
    result = ContextualReconstructor(provider, priority=RefinementPriority.CANDIDATE).reconstruct(
        [{"start": 0.0, "end": 1.0, "text": "الرئيس 71", "corrected_text": "الرئيس 71"}],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    segment = result.segments[0]
    assert segment.applied is False
    assert segment.status is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
    assert segment.contextual_reconstructed_text == "الرئيس 71"
    assert segment.confidence == 0.0


def test_unchanged_candidate_text_remains_unchanged() -> None:
    """A provider proposal identical to Stage 2.5 never fabricates a reconstruction."""

    provider = OnePassProvider(
        ReconstructionCandidate("provider-0", "خلي بالك", provider_confidence=0.99)
    )
    result = ContextualReconstructor(provider, priority=RefinementPriority.CANDIDATE).reconstruct(
        [{"start": 0.0, "end": 1.0, "text": "خلي بالك", "corrected_text": "خلي بالك"}],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    segment = result.segments[0]
    assert segment.applied is False
    assert segment.contextual_reconstructed_text == "خلي بالك"


class ScriptedTargetProvider:
    """Emit a fixed candidate per target and raise a recoverable failure for one target."""

    def __init__(
        self,
        candidates: dict[int, ReconstructionCandidate],
        fail_segment: int,
        failure: type[Exception],
    ) -> None:
        self.candidates = candidates
        self.fail_segment = fail_segment
        self.failure = failure
        self.requests: list[ReconstructionRequest] = []

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "ollama", "qwen3.5:4b", "sha256:x", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "ollama",
            "model": "qwen3.5:4b",
            "digest": "sha256:x",
            "prompt_hash": "p",
            "schema_version": "s",
            "max_context_tokens": 4096,
            "output_tokens": 256,
            "chat_framing_reserve": 64,
            "safety_reserve": 128,
            "confidence_policy_version": "one-pass-provider-confidence-v1",
            "validation_version": "stage-2-7-validation-v1",
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.requests.extend(requests)
        for request in requests:
            if request.segment_index == self.fail_segment:
                raise self.failure("target failure")
        return {
            request.segment_index: self.candidates[request.segment_index] for request in requests
        }

    def release(self) -> None:
        pass


def test_one_segment_failure_does_not_erase_successful_targets() -> None:
    """A per-target provider failure falls back only that segment."""

    provider = ScriptedTargetProvider(
        {
            0: ReconstructionCandidate("provider-0", "ضخمة", provider_confidence=1.0),
            2: ReconstructionCandidate("provider-0", "صحيح", provider_confidence=0.99),
        },
        fail_segment=1,
        failure=ProviderResponseError,
    )
    result = ContextualReconstructor(provider, priority=RefinementPriority.CANDIDATE).reconstruct(
        [
            {"start": 0.0, "end": 1.0, "text": "دخم", "corrected_text": "دخم"},
            {"start": 1.0, "end": 2.0, "text": "خطي بالك", "corrected_text": "خلي بالك"},
            {"start": 2.0, "end": 3.0, "text": "صحيح", "corrected_text": "صحيح"},
        ],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert result.segments[0].applied is True
    assert result.segments[0].contextual_reconstructed_text == "ضخمة"
    assert result.segments[1].status is ReconstructionStatus.PROVIDER_UNAVAILABLE
    assert result.segments[1].contextual_reconstructed_text == "خلي بالك"
    assert result.segments[2].contextual_reconstructed_text == "صحيح"


def test_oserror_on_one_segment_falls_back_only_that_segment() -> None:
    """Network timeouts are per-target recoverable failures too."""

    provider = ScriptedTargetProvider(
        {
            0: ReconstructionCandidate("provider-0", "ضخمة", provider_confidence=1.0),
            2: ReconstructionCandidate("provider-0", "صحيح", provider_confidence=0.99),
        },
        fail_segment=1,
        failure=OSError,
    )
    result = ContextualReconstructor(provider, priority=RefinementPriority.CANDIDATE).reconstruct(
        [
            {"start": 0.0, "end": 1.0, "text": "دخم", "corrected_text": "دخم"},
            {"start": 1.0, "end": 2.0, "text": "خطي بالك", "corrected_text": "خلي بالك"},
            {"start": 2.0, "end": 3.0, "text": "صحيح", "corrected_text": "صحيح"},
        ],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert result.segments[0].applied is True
    assert result.segments[1].status is ReconstructionStatus.PROVIDER_UNAVAILABLE
    assert result.segments[2].contextual_reconstructed_text == "صحيح"


def test_reconstructor_sends_small_context_window() -> None:
    """The provider receives only local context, not the full transcript."""

    provider = OnePassProvider()
    ContextualReconstructor(provider, priority=RefinementPriority.CANDIDATE).reconstruct(
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
