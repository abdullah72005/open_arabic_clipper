from collections.abc import Sequence

from app.core.enums import ReconstructionStatus

_PRECEDENCE = {
    ReconstructionStatus.NOT_REQUIRED: 0,
    ReconstructionStatus.UNCHANGED_HIGH_CONFIDENCE: 1,
    ReconstructionStatus.MANUAL_OVERRIDE: 2,
    ReconstructionStatus.APPLIED: 3,
    ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED: 4,
    ReconstructionStatus.PROVIDER_UNAVAILABLE: 5,
    ReconstructionStatus.FAILED: 6,
}


def aggregate_reconstruction_status(
    values: Sequence[ReconstructionStatus],
) -> ReconstructionStatus:
    """Return the most severe reconstruction outcome in a collection."""

    return max(values, key=_PRECEDENCE.__getitem__) if values else ReconstructionStatus.NOT_REQUIRED
