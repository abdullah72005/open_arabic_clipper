"""Stage 4.2 retention, originality, and platform-risk governor.

Explicit, candidate-scoped governance of every current Stage 4.1 transformation
plan. Deterministic logic owns integrity, evidence, dimensions, hard gates,
status precedence, platform-risk interpretation, and fingerprints. An optional
provider supplies bounded observable semantic findings only. The governor never
generates, mutates, repairs, or selects a plan, and adds no pipeline stage.
"""

from app.transformation.governance.policy import (
    GOVERNOR_POLICY_VERSION,
    GOVERNOR_SCHEMA_VERSION,
    GOVERNOR_VALIDATION_VERSION,
    PLATFORM_POLICY_CHECKED_AT,
    PLATFORM_POLICY_PROFILE_VERSION,
    Stage42Config,
)

__all__ = [
    "GOVERNOR_POLICY_VERSION",
    "GOVERNOR_SCHEMA_VERSION",
    "GOVERNOR_VALIDATION_VERSION",
    "PLATFORM_POLICY_CHECKED_AT",
    "PLATFORM_POLICY_PROFILE_VERSION",
    "Stage42Config",
]
