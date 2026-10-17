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
from typing import TYPE_CHECKING

from app.materialization.inputs import ProfileContext
from app.materialization.policy import (
    DEFAULT_OUTPUT_LANGUAGE_POLICY,
    DEFAULT_REGISTER_INTENT,
    DEFAULT_TARGET_MARKET,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

_SEMANTIC_FIELDS = ("target_market", "output_language_policy", "register_intent")

#: Strict import field allow-list. Unknown fields are rejected rather than
#: silently ignored; secrets never appear in a profile.
_PROFILE_IMPORT_FIELDS = frozenset(
    {
        "channel_key",
        "label",
        "target_market",
        "output_language_policy",
        "register_intent",
        "narration_allowed",
        "tts_provider",
        "tts_model",
        "tts_api_version",
        "tts_voice",
        "tts_style",
        "tts_settings",
        "supporting_visual_mode",
        "supporting_visual_style",
        "supporting_visual_defaults",
    }
)
_SUPPORTING_MODES = frozenset({"OFF", "AUTO", "ON"})
_SUPPORTING_STYLES = frozenset(
    {"AUTO", "SUBTLE", "EDITORIAL", "ABSTRACT", "DOCUMENTARY", "MINIMAL"}
)

_MAX_FIELD_LENGTHS = {
    "channel_key": 120,
    "label": 200,
    "target_market": 80,
    "output_language_policy": 80,
    "register_intent": 80,
    "tts_provider": 80,
    "tts_model": 120,
    "tts_api_version": 40,
    "tts_voice": 120,
    "tts_style": 120,
}


class ProfileImportError(ValueError):
    """A production-profile import payload was malformed or unsupported."""


def _bounded_str(payload: Mapping[str, object], key: str, *, default: str = "") -> str:
    value = payload.get(key, default)
    if value is None:
        value = ""
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ProfileImportError(f"{key} must be a string")
    text = str(value).strip()
    limit = _MAX_FIELD_LENGTHS.get(key)
    if limit is not None and len(text) > limit:
        raise ProfileImportError(f"{key} exceeds {limit} characters")
    return text


def _bounded_mapping(payload: Mapping[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProfileImportError(f"{key} must be an object")
    return {str(item_key): item for item_key, item in value.items()}


def import_profile(session: "Session", payload: Mapping[str, object]) -> object:
    """Import one strict production profile as a new current revision.

    Validates a strict JSON object (unknown fields refused, bounded strings, a
    closed supporting-visual vocabulary), demotes the previous current revision
    for the ``channel_key``, and persists a new current row. Secrets never live in
    a profile; the durable identity is the stable ``channel_key``.
    """

    from sqlalchemy import select

    from app.core.enums import SupportingVisualMode, SupportingVisualStyle
    from app.models.channel_production_profile import ChannelProductionProfile

    if not isinstance(payload, Mapping):
        raise ProfileImportError("profile payload must be a JSON object")
    unknown = set(str(key) for key in payload) - _PROFILE_IMPORT_FIELDS
    if unknown:
        raise ProfileImportError(f"unknown profile fields: {sorted(unknown)}")

    channel_key = _bounded_str(payload, "channel_key")
    if not channel_key:
        raise ProfileImportError("channel_key is required")
    mode = _bounded_str(payload, "supporting_visual_mode", default="OFF").upper()
    if mode not in _SUPPORTING_MODES:
        raise ProfileImportError("unsupported supporting_visual_mode")
    style = _bounded_str(payload, "supporting_visual_style", default="AUTO").upper()
    if style not in _SUPPORTING_STYLES:
        raise ProfileImportError("unsupported supporting_visual_style")
    narration_allowed = payload.get("narration_allowed", False)
    if not isinstance(narration_allowed, bool):
        raise ProfileImportError("narration_allowed must be a boolean")

    existing = session.scalars(
        select(ChannelProductionProfile).where(
            ChannelProductionProfile.channel_key == channel_key,
            ChannelProductionProfile.is_current.is_(True),
        )
    ).all()
    revision = max((int(row.revision) for row in existing), default=0) + 1
    for row in existing:
        row.is_current = False
    session.flush()

    profile = ChannelProductionProfile(
        channel_key=channel_key,
        revision=revision,
        is_current=True,
        label=_bounded_str(payload, "label"),
        target_market=_bounded_str(payload, "target_market", default=DEFAULT_TARGET_MARKET)
        or DEFAULT_TARGET_MARKET,
        output_language_policy=_bounded_str(
            payload, "output_language_policy", default=DEFAULT_OUTPUT_LANGUAGE_POLICY
        )
        or DEFAULT_OUTPUT_LANGUAGE_POLICY,
        register_intent=_bounded_str(payload, "register_intent", default=DEFAULT_REGISTER_INTENT)
        or DEFAULT_REGISTER_INTENT,
        narration_allowed=narration_allowed,
        tts_provider=_bounded_str(payload, "tts_provider"),
        tts_model=_bounded_str(payload, "tts_model"),
        tts_api_version=_bounded_str(payload, "tts_api_version"),
        tts_voice=_bounded_str(payload, "tts_voice"),
        tts_style=_bounded_str(payload, "tts_style"),
        tts_settings=_bounded_mapping(payload, "tts_settings"),
        supporting_visual_mode=SupportingVisualMode(mode),
        supporting_visual_style=SupportingVisualStyle(style),
        supporting_visual_defaults=_bounded_mapping(payload, "supporting_visual_defaults"),
    )
    session.add(profile)
    session.commit()
    return profile


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
    "ProfileImportError",
    "import_profile",
    "profile_from_row",
    "semantic_context_matches",
]
