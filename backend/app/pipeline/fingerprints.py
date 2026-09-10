"""Stable content fingerprints for pipeline dependencies."""

from __future__ import annotations

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
    target_indexes: Sequence[int] | None = None,
) -> str:
    """Fingerprint Stage 2.7 output from stable dependency identity only.

    ``provider_identity`` carries the full stable runtime identity: local
    provider, routing mode and policy thresholds (including evidence-coverage and
    clean-average trust rules), refinement priority, Gemini provider/model/schema/
    API version and temperature, budgets, batching and local-work ceilings, and
    confidence/validation versions. Transient provider availability is execution
    state, not identity, and is deliberately excluded so a temporary outage
    cannot invalidate accepted output. Cache eligibility is tracked separately by
    the executor. Never include credentials.

    Version 5 adds the refinement priority/scope identity: ``target_indexes``
    plus a whole-source versus window marker, so a whole-source INDEX result can
    never satisfy a window-scoped CANDIDATE/FINAL_CLIP refinement and one
    requested window can never satisfy another. Version 4 added every
    route-relevant input the adaptive router now consumes (Stage 2.5
    method/confidence/applied state and change digest, word probabilities,
    acoustic evidence).
    """

    return canonical_fingerprint(
        "reconstruction-output",
        "5",
        {
            "language": language,
            "transcription_fingerprint": transcription_fingerprint,
            "correction_version": correction_version,
            "runtime_identity": dict(provider_identity),
            "scope": "window" if target_indexes is not None else "whole_source",
            "target_indexes": tuple(target_indexes) if target_indexes is not None else None,
            "segments": [_segment_dependency(segments, index) for index in range(len(segments))],
        },
    )


def reconstruction_target_fingerprint(
    *,
    provider_identity: Mapping[str, object],
    segments: Sequence[Mapping[str, object]],
    target_index: int,
    language: str | None,
    transcription_fingerprint: str,
    correction_version: str,
) -> str:
    """Fingerprint one reconstruction target's stable route dependencies.

    The target fingerprint covers the target's own content and Stage 2.5 state,
    its bounded surrounding corrected context (so a neighbor edit invalidates
    only the affected targets), and the shared runtime identity. It is used to
    reuse accepted per-target work after a restart without re-calling a provider.
    """

    return canonical_fingerprint(
        "reconstruction-target",
        "1",
        {
            "language": language,
            "transcription_fingerprint": transcription_fingerprint,
            "correction_version": correction_version,
            "runtime_identity": dict(provider_identity),
            "target": _segment_dependency(segments, target_index),
        },
    )


def _segment_dependency(segments: Sequence[Mapping[str, object]], index: int) -> dict[str, object]:
    """Stable route-relevant dependency payload for one transcript segment.

    Bounded context is the two preceding and two following corrected segments,
    matching the provider request envelope.
    """

    segment = segments[index]
    previous = segments[max(0, index - 2) : index]
    following = segments[index + 1 : index + 3]
    return {
        "raw": _text(segment, "raw_text", "text"),
        "corrected": _text(segment, "corrected_text", "text"),
        "start": segment.get("start"),
        "end": segment.get("end"),
        "correction_method": segment.get("correction_method"),
        "correction_confidence": segment.get("correction_confidence"),
        "correction_applied": segment.get("correction_applied"),
        "correction_changes": _stable_changes(segment.get("correction_changes")),
        "words": _word_evidence(segment.get("words")),
        "avg_logprob": segment.get("avg_logprob"),
        "no_speech_prob": segment.get("no_speech_prob"),
        "language": segment.get("language"),
        "dialect_profile": segment.get("dialect_profile"),
        "previous_context": tuple(_text(item, "corrected_text", "text") for item in previous),
        "following_context": tuple(_text(item, "corrected_text", "text") for item in following),
    }


def _text(segment: Mapping[str, object], preferred: str, fallback: str) -> str:
    value = segment.get(preferred)
    if value is None:
        value = segment.get(fallback)
    return str(value or "")


def _stable_changes(changes: object) -> tuple[dict[str, str], ...]:
    if not isinstance(changes, list):
        return ()
    stable: list[dict[str, str]] = []
    for change in changes:
        if not isinstance(change, Mapping):
            continue
        stable.append(
            {
                str(key): str(value)
                for key, value in change.items()
                if str(key).casefold() not in {"reason"}
            }
        )
    return tuple(stable)


def _word_evidence(words: object) -> tuple[dict[str, object], ...]:
    if not isinstance(words, list):
        return ()
    evidence: list[dict[str, object]] = []
    for word in words:
        if not isinstance(word, Mapping):
            continue
        item: dict[str, object] = {"word": str(word.get("word", ""))}
        if isinstance(word.get("probability"), int | float):
            item["probability"] = float(word["probability"])
        if isinstance(word.get("start"), int | float):
            item["start"] = float(word["start"])
        if isinstance(word.get("end"), int | float):
            item["end"] = float(word["end"])
        evidence.append(item)
    return tuple(evidence)
