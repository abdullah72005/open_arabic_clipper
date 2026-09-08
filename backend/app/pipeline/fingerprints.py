"""Stable content fingerprints for pipeline dependencies."""

import hashlib
import json
from collections.abc import Mapping, Sequence


def canonical_fingerprint(namespace: str, version: str, payload: Mapping[str, object]) -> str:
    body = {"namespace": namespace, "version": version, "payload": payload}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def reconstruction_output_fingerprint(
    *,
    provider_identity: Mapping[str, object],
    segments: Sequence[Mapping[str, object]],
    language: str | None,
    transcription_fingerprint: str,
    correction_version: str,
) -> str:
    """Fingerprint Stage 2.7 output from stable dependency identity only.

    ``provider_identity`` carries the full stable runtime identity: local
    provider, routing mode and policy thresholds, Gemini provider/model/schema,
    budgets, thinking level, and confidence/validation versions. Transient
    provider availability is execution state, not identity, and is deliberately
    excluded so a temporary outage cannot invalidate accepted output. Cache
    eligibility is tracked separately by the executor. Never include credentials.
    """

    return canonical_fingerprint(
        "reconstruction-output",
        "3",
        {
            "language": language,
            "transcription_fingerprint": transcription_fingerprint,
            "correction_version": correction_version,
            "runtime_identity": dict(provider_identity),
            "segments": [
                {
                    "raw": segment.get("raw_text", segment.get("text", "")),
                    "corrected": segment.get("corrected_text"),
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                }
                for segment in segments
            ],
        },
    )
