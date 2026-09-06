from app.core.enums import ProviderAvailability, ReconstructionStatus
from app.transcription.reconstruction.status import aggregate_reconstruction_status


def test_stage_2_7_enum_values_are_stable() -> None:
    assert {value.value for value in ProviderAvailability} == {
        "AVAILABLE",
        "UNAVAILABLE",
        "MISCONFIGURED",
    }
    assert {value.value for value in ReconstructionStatus} == {
        "NOT_REQUIRED",
        "APPLIED",
        "UNCHANGED_HIGH_CONFIDENCE",
        "LOW_CONFIDENCE_UNRESOLVED",
        "PROVIDER_UNAVAILABLE",
        "FAILED",
        "MANUAL_OVERRIDE",
    }


def test_reconstruction_status_uses_worst_first_precedence() -> None:
    assert (
        aggregate_reconstruction_status(
            [
                ReconstructionStatus.APPLIED,
                ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
            ]
        )
        is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
    )
    assert (
        aggregate_reconstruction_status(
            [
                ReconstructionStatus.PROVIDER_UNAVAILABLE,
                ReconstructionStatus.FAILED,
            ]
        )
        is ReconstructionStatus.FAILED
    )


def test_empty_reconstruction_status_is_not_required() -> None:
    assert aggregate_reconstruction_status([]) is ReconstructionStatus.NOT_REQUIRED
