from app.transcription.reconstruction.providers import (
    GenerationRequest,
    ProviderResponseError,
    ResolutionChoice,
    ResolutionRequest,
)
from app.transcription.reconstruction.service import ContextualReconstructor, select_final_text
from app.transcription.reconstruction.types import (
    ConfidenceLevel,
    ReconstructionCandidate,
    ResolutionScores,
)


class HighConfidenceProvider:
    def generate_candidates(
        self, requests: list[GenerationRequest]
    ) -> dict[int, list[ReconstructionCandidate]]:
        return {
            request.segment_index: [
                ReconstructionCandidate("provider-0", "ضخمة", evidence_segment_ids=(0,))
            ]
            for request in requests
        }

    def resolve_candidates(self, requests: list[ResolutionRequest]) -> dict[int, ResolutionChoice]:
        return {
            request.segment_index: ResolutionChoice(
                "provider-0", ResolutionScores(1.0, 1.0, 1.0, 1.0, 1.0)
            )
            for request in requests
        }


def test_reconstructor_applies_only_high_contextual_candidate() -> None:
    """A high-scoring source-supported candidate becomes automatic final text."""

    result = ContextualReconstructor(HighConfidenceProvider()).reconstruct(
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
        def generate_candidates(
            self, requests: list[GenerationRequest]
        ) -> dict[int, list[ReconstructionCandidate]]:
            raise ProviderResponseError("invalid JSON")

        def resolve_candidates(
            self, requests: list[ResolutionRequest]
        ) -> dict[int, ResolutionChoice]:
            raise AssertionError("resolution must not run")

    result = ContextualReconstructor(BrokenProvider()).reconstruct(
        [{"start": 0.0, "end": 1.0, "text": "خطي بالك", "corrected_text": "خلي بالك"}],
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert result.segments[0].contextual_reconstructed_text == "خلي بالك"
    assert result.segments[0].quality_flags[0].value == "RECONSTRUCTION_PROVIDER_ERROR"


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
