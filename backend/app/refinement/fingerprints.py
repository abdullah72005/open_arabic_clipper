"""Stage 3.5 fingerprint composition.

Top-level and component fingerprints are stable identity only: transient
availability, counters, and cooldowns are deliberately excluded so a temporary
provider outage never invalidates accepted work.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.pipeline.fingerprints import canonical_fingerprint
from app.refinement.policy import FINGERPRINT_VERSION


def refinement_input_fingerprint(payload: Mapping[str, object]) -> str:
    return str(canonical_fingerprint("candidate-refinement-input", FINGERPRINT_VERSION, payload))


def refinement_output_fingerprint(payload: Mapping[str, object]) -> str:
    return str(canonical_fingerprint("candidate-refinement-output", FINGERPRINT_VERSION, payload))


def component_fingerprint(name: str, payload: Mapping[str, object]) -> str:
    """A per-component checkpoint key (audio/asr/hosted/adjudication/boundary)."""

    return str(canonical_fingerprint(f"candidate-refinement-{name}", FINGERPRINT_VERSION, payload))


def priority_scope(priority: str) -> dict[str, object]:
    return {"priority": priority, "scope": "candidate"}
