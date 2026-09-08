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
    provider_available: bool,
    segments: Sequence[Mapping[str, object]],
    language: str | None,
    transcription_fingerprint: str,
    correction_version: str,
    gemini_available: bool = False,
) -> str:
    """Fingerprint Stage 2.7 output including every output-affecting dependency.

    ``provider_identity`` carries the full runtime identity: local provider,
    routing mode and policy thresholds, Gemini provider/model/schema, budgets,
    and confidence/validation versions. Never include credentials in this data.
    """

    return canonical_fingerprint(
        "reconstruction-output",
        "2",
        {
            "language": language,
            "transcription_fingerprint": transcription_fingerprint,
            "correction_version": correction_version,
            "provider_available": provider_available,
            "gemini_available": gemini_available,
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
