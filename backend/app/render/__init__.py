"""Stage 5.0 deterministic execution preflight and render contract.

This package performs no rendering, content generation, provider call, TTS,
caption-file writing, face tracking, or publishing. It validates that the
current Stage 4.3 selection still binds to current FINAL_CLIP transcript
evidence and a present source media artifact, then persists one durable,
evidence-bound execution/render contract per candidate/input fingerprint.
"""

from __future__ import annotations

from app.render.policy import (
    RENDER_CONTRACT_FINGERPRINT_VERSION,
    RENDER_CONTRACT_POLICY_VERSION,
    RENDER_CONTRACT_SCHEMA_VERSION,
)

__all__ = [
    "RENDER_CONTRACT_FINGERPRINT_VERSION",
    "RENDER_CONTRACT_POLICY_VERSION",
    "RENDER_CONTRACT_SCHEMA_VERSION",
]
