"""Stable Stage 4.0 fingerprint composition.

Only output-relevant evidence participates. Whole-source raw ASR, unrelated
transcript segments, renderer/publishing/frontend settings, the Gemini key
value, and transient admission/cooldown/outage state are deliberately excluded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.pipeline.fingerprints import canonical_fingerprint
from app.transformation.policy import (
    INPUT_FINGERPRINT_VERSION,
    OUTPUT_FINGERPRINT_VERSION,
    Stage40Config,
    stage40_config_payload,
    stage40_policy_payload,
)
from app.transformation.types import StrategyDraft, TransformationInputs

STRATEGY_FINGERPRINT_VERSION = "1"
PROVIDER_INPUT_FINGERPRINT_VERSION = "1"


def transformation_input_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint("transformation-input", INPUT_FINGERPRINT_VERSION, dict(payload))


def transformation_output_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint("transformation-output", OUTPUT_FINGERPRINT_VERSION, dict(payload))


def strategy_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint(
        "transformation-strategy", STRATEGY_FINGERPRINT_VERSION, dict(payload)
    )


def provider_input_fingerprint(payload: Mapping[str, object]) -> str:
    return canonical_fingerprint(
        "transformation-provider-input", PROVIDER_INPUT_FINGERPRINT_VERSION, dict(payload)
    )


def _bounded_words(words: Sequence[object]) -> list[dict[str, object]]:
    bounded: list[dict[str, object]] = []
    for word in words:
        if not isinstance(word, Mapping):
            continue
        item: dict[str, object] = {"text": str(word.get("text", word.get("word", "")))}
        for key in ("start", "end", "probability"):
            value = word.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                item[key] = round(float(value), 4)
        bounded.append(item)
    return bounded


def _bounded_spans(spans: Sequence[object]) -> list[dict[str, object]]:
    bounded: list[dict[str, object]] = []
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        entry: dict[str, object] = {}
        for key in ("span_id", "start", "end", "resolution_state", "meaning_critical"):
            if key in span:
                entry[key] = span[key]
        bounded.append(entry)
    return bounded


def _bounded_hooks(hooks: Sequence[object]) -> list[dict[str, object]]:
    bounded: list[dict[str, object]] = []
    for hook in hooks:
        if not isinstance(hook, Mapping):
            continue
        entry: dict[str, object] = {}
        for key in ("type", "text", "source_segment_indexes", "strength"):
            if key in hook:
                entry[key] = hook[key]
        bounded.append(entry)
    return bounded


def build_input_fingerprint_payload(
    *,
    inputs: TransformationInputs,
    config: Stage40Config,
    provider_identity: Mapping[str, object],
    provider_mode: str,
    provider_route: str | None,
    strategy_fingerprints: Sequence[str] = (),
) -> dict[str, object]:
    """Compose the full Stage 4.0 input identity from bounded evidence only."""

    return {
        "candidate": {
            "id": inputs.candidate_id,
            "key": inputs.candidate_key,
            "disposition": inputs.disposition,
            "content_type": inputs.content_type.value,
            "secondary_content_types": [item.value for item in inputs.secondary_content_types],
            "coarse_start": inputs.coarse_start,
            "coarse_end": inputs.coarse_end,
        },
        "stage3": {
            "analysis_fingerprint": inputs.stage3_analysis_fingerprint,
            "policy_version": inputs.stage3_policy_version,
            "clip_score": inputs.clip_score,
            "short_form_score": inputs.short_form_score,
            "moment_density_score": inputs.moment_density_score,
            "ending_quality_score": inputs.ending_quality_score,
            "loopability_score": inputs.loopability_score,
            "idea_summary": inputs.idea_summary,
            "topic_summary": inputs.topic_summary,
            "hooks": _bounded_hooks(inputs.hooks),
        },
        "provenance": {
            "rights_status": inputs.rights_status,
            "rights_risk": inputs.rights_risk.value,
            "originality_risk": inputs.originality_risk.value,
            "media_origin": inputs.media_origin,
            "snapshot": dict(inputs.provenance_snapshot),
        },
        "refinement": {
            "priority": inputs.refinement_priority,
            "quality_level": inputs.refinement_quality_level,
            "status": inputs.refinement_status,
            "output_fingerprint": inputs.refinement_output_fingerprint,
            "confidence": inputs.refinement_confidence,
            "refined_start": inputs.refined_start,
            "refined_end": inputs.refined_end,
            "transcript": inputs.transcript,
            "transcript_confidence": inputs.transcript_confidence,
            "words": _bounded_words(inputs.word_timestamps),
            "unresolved_spans": _bounded_spans(inputs.unresolved_spans),
            "entity_evidence": [dict(item) for item in inputs.entity_evidence],
            "dialect_profile": inputs.dialect_profile,
            "dialect_confidence": inputs.dialect_confidence,
            "code_switch": dict(inputs.code_switch),
        },
        "context": list(inputs.context_segments),
        "policy": stage40_policy_payload(),
        "config": stage40_config_payload(config),
        "provider": {
            "mode": provider_mode,
            "route": provider_route,
            "identity": dict(provider_identity),
            "strategy_fingerprints": sorted(strategy_fingerprints),
        },
    }


def build_output_fingerprint_payload(
    *,
    eligibility_outcome: str,
    intensity: str | None,
    strategies: Sequence[StrategyDraft],
    source_moment: Mapping[str, object],
    platform_risk: Mapping[str, object],
    assessments: Mapping[str, object],
) -> dict[str, object]:
    """Fingerprint the complete current eligibility + strategy representation.

    Excludes transient metrics, timing, and provider availability.
    """

    ordered = sorted(strategies, key=lambda item: item.strategy_type.value)
    return {
        "eligibility_outcome": eligibility_outcome,
        "intensity": intensity,
        "source_moment": dict(source_moment),
        "platform_risk": dict(platform_risk),
        "assessments": dict(assessments),
        "strategies": [item.fingerprint_payload() for item in ordered],
    }
