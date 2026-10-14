"""Channel production-profile semantic context resolution and matching.

The profile supplies the target-market/language/register context a transformed
short is authored under. It must match the selected Stage 4 plan's planning
context; a mismatch is an upstream revalidation requirement, never a silent
rewrite of the frozen plan or its target context. TTS voice/provider identity is
deliberately *not* part of semantic-context matching: a voice change affects
Stage 6 narration, never Stage 4 semantics.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.materialization.inputs import ProfileContext
from app.materialization.policy import (
    DEFAULT_OUTPUT_LANGUAGE_POLICY,
    DEFAULT_REGISTER_INTENT,
    DEFAULT_TARGET_MARKET,
)

_SEMANTIC_FIELDS = ("target_market", "output_language_policy", "register_intent")


def profile_from_row(row: object | None) -> ProfileContext:
    """Build a neutral-default :class:`ProfileContext` from a profile row."""

    if row is None:
        return ProfileContext(
            profile_id=None,
            profile_key="default",
            target_market=DEFAULT_TARGET_MARKET,
            output_language_policy=DEFAULT_OUTPUT_LANGUAGE_POLICY,
            register_intent=DEFAULT_REGISTER_INTENT,
            narration_allowed=False,
        )
    tts_settings = getattr(row, "tts_settings", None)
    supporting_defaults = getattr(row, "supporting_visual_defaults", None)
    return ProfileContext(
        profile_id=str(getattr(row, "id", "")) or None,
        profile_key=str(getattr(row, "channel_key", "default")),
        target_market=str(getattr(row, "target_market", DEFAULT_TARGET_MARKET)),
        output_language_policy=str(
            getattr(row, "output_language_policy", DEFAULT_OUTPUT_LANGUAGE_POLICY)
        ),
        register_intent=str(getattr(row, "register_intent", DEFAULT_REGISTER_INTENT)),
        narration_allowed=bool(getattr(row, "narration_allowed", False)),
        tts_provider=str(getattr(row, "tts_provider", "") or ""),
        tts_model=str(getattr(row, "tts_model", "") or ""),
        tts_api_version=str(getattr(row, "tts_api_version", "") or ""),
        tts_voice=str(getattr(row, "tts_voice", "") or ""),
        tts_style=str(getattr(row, "tts_style", "") or ""),
        tts_settings=dict(tts_settings) if isinstance(tts_settings, Mapping) else {},
        supporting_visual_mode=str(
            getattr(getattr(row, "supporting_visual_mode", "OFF"), "value", "OFF")
        ),
        supporting_visual_style=str(
            getattr(getattr(row, "supporting_visual_style", "AUTO"), "value", "AUTO")
        ),
        supporting_visual_defaults=(
            dict(supporting_defaults) if isinstance(supporting_defaults, Mapping) else {}
        ),
    )


def semantic_context_matches(
    profile: ProfileContext, plan_target_context: Mapping[str, object]
) -> tuple[bool, tuple[str, ...]]:
    """Whether the profile's semantic context matches the plan's target context."""

    mismatches: list[str] = []
    for field in _SEMANTIC_FIELDS:
        expected = str(plan_target_context.get(field, _default_for(field)))
        actual = str(getattr(profile, field))
        if actual != expected:
            mismatches.append(field)
    return (not mismatches, tuple(mismatches))


def _default_for(field: str) -> str:
    return {
        "target_market": DEFAULT_TARGET_MARKET,
        "output_language_policy": DEFAULT_OUTPUT_LANGUAGE_POLICY,
        "register_intent": DEFAULT_REGISTER_INTENT,
    }[field]


__all__ = [
    "profile_from_row",
    "semantic_context_matches",
]
