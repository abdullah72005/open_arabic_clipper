"""Deterministic Arabic/English content-type classification."""

from __future__ import annotations

from collections.abc import Sequence

from app.candidates.cues import CONTENT_CUES, PAYOFF_CUES, STORY_BUILDUP_CUES
from app.candidates.policy import DEFAULT_CONFIG, Stage3Config
from app.candidates.text import contains_any_cue, contains_cue, matching_text
from app.candidates.types import ContentClassification
from app.core.enums import ContentType

_ORDER = tuple(ContentType)


def classify_content(
    text: str,
    *,
    config: Stage3Config = DEFAULT_CONFIG,
    provider_primary: ContentType | None = None,
    provider_secondary: Sequence[ContentType] = (),
) -> ContentClassification:
    """Classify candidate text. Provider values may only be declared enum members."""

    scores: dict[str, float] = {}
    matching = matching_text(text)
    for content_type, cues in CONTENT_CUES.items():
        hits = sum(1 for cue in cues if contains_cue(matching, cue))
        if hits:
            scores[content_type.value] = float(hits)
    if "?" in matching or "؟" in matching:
        scores.setdefault(ContentType.REACTION_WORTHY.value, 0.0)
    if contains_any_cue(matching, PAYOFF_CUES):
        scores[ContentType.STORY.value] = scores.get(ContentType.STORY.value, 0.0) + 0.5
    if contains_any_cue(matching, STORY_BUILDUP_CUES):
        scores[ContentType.STORY.value] = scores.get(ContentType.STORY.value, 0.0) + 0.5

    deterministic_primary, secondary = _select(scores, config)
    if provider_primary is not None and isinstance(provider_primary, ContentType):
        primary = provider_primary
        secondary = tuple(
            item
            for item in dict.fromkeys((deterministic_primary, *secondary, *provider_secondary))
            if isinstance(item, ContentType) and item is not primary
        )
    else:
        primary = deterministic_primary
        secondary = tuple(
            item
            for item in dict.fromkeys((*secondary, *provider_secondary))
            if isinstance(item, ContentType) and item is not primary
        )
    secondary = secondary[: config.max_secondary_content_types]
    reasons = tuple(
        f"content:{content_type.value}={scores[content_type.value]:.2f}"
        for content_type in _ORDER
        if content_type.value in scores and scores[content_type.value] > 0
    )
    return ContentClassification(
        primary=primary,
        secondary=secondary,
        scores=scores,
        reasons=reasons,
    )


def _select(
    scores: dict[str, float], config: Stage3Config
) -> tuple[ContentType, tuple[ContentType, ...]]:
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], _ORDER.index(ContentType(item[0]))),
    )
    if not ranked or ranked[0][1] <= 0:
        return ContentType.OTHER, ()
    primary = ContentType(ranked[0][0])
    secondary = tuple(ContentType(name) for name, value in ranked[1:] if value > 0)
    return primary, secondary
