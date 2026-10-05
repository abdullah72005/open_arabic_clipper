"""Deterministic Stage 5.2 render-execution fingerprints.

Request/runtime/compiler/QC fingerprints via ``canonical_fingerprint``. The
request fingerprint covers every pre-execution input and output-affecting policy
but deliberately excludes job id, attempt id, claim version, timestamps,
progress, transient lock availability, absolute deployment paths, API keys, TTS
provider/model/voice labels, publishing metadata, and unrelated analytics.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.pipeline.fingerprints import canonical_fingerprint
from app.render.execution.policy import EXECUTION_FINGERPRINT_VERSION


def render_request_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "render-execution-request", EXECUTION_FINGERPRINT_VERSION, dict(payload)
        )
    )


def render_runtime_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "render-execution-runtime", EXECUTION_FINGERPRINT_VERSION, dict(payload)
        )
    )


def render_compiled_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "render-execution-compiled", EXECUTION_FINGERPRINT_VERSION, dict(payload)
        )
    )


def render_qc_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint("render-execution-qc", EXECUTION_FINGERPRINT_VERSION, dict(payload))
    )


def render_output_fingerprint(payload: Mapping[str, object]) -> str:
    return str(
        canonical_fingerprint(
            "render-execution-output", EXECUTION_FINGERPRINT_VERSION, dict(payload)
        )
    )


__all__ = [
    "render_compiled_fingerprint",
    "render_output_fingerprint",
    "render_qc_fingerprint",
    "render_request_fingerprint",
    "render_runtime_fingerprint",
]
