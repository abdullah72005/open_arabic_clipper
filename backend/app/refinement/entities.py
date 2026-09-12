"""Deterministic practical entity extraction and comparison for Stage 3.5.

Extraction preserves the spoken/display form exactly and produces a separate
normalized comparison copy. The module is pure: no network, no LLM, no model
loading, no audio decoding. Classification is shape based and deliberately
conservative: it never guesses a person/product/place name when the token is
not even Latin, and it only claims a technical type for a real technical shape.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum

from app.refinement.types import EntityMention
from app.transcription.arabic import normalize_for_comparison
from app.transcription.dialect import extract_protected_tokens

ENTITY_POLICY_VERSION = "stage3.5-entity-v1"
MAX_ENTITY_CONFLICTS = 32


class EntityType(str, Enum):
    """Practical entity classes that matter for candidate comparison."""

    PERSON = "PERSON"
    PRODUCT = "PRODUCT"
    PLACE = "PLACE"
    DATE = "DATE"
    TIME = "TIME"
    NUMBER = "NUMBER"
    PERCENTAGE = "PERCENTAGE"
    MONEY = "MONEY"
    SCORE = "SCORE"
    ABBREVIATION = "ABBREVIATION"
    TECHNICAL = "TECHNICAL"


_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")
_DATE_PATTERN = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_SCORE_PATTERN = re.compile(r"\d+[-:]\d+")
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
_YEAR_RANGE = (1000, 2100)
_PERCENT_WORDS = frozenset({"percent", "percentage", "بالمية", "بالمائة", "في المية"})
_CURRENCY_SYMBOLS = ("$", "€", "£", "¥", "₪", "﷼")
_CURRENCY_WORDS = frozenset(
    {
        "usd",
        "egp",
        "sar",
        "aed",
        "eur",
        "gbp",
        "dollar",
        "dollars",
        "pound",
        "euro",
        "ريال",
        "دولار",
        "جنيه",
        "درهم",
        "دينار",
        "يورو",
    }
)
_CURRENCY_PREFIX = re.compile(
    r"(?:[$€£¥₪﷼]|(?:usd|egp|sar|aed|eur|gbp|dollar|dollars|pound|euro|ريال|دولار|جنيه|درهم|دينار|يورو))\s*",
    re.IGNORECASE,
)
_MEANING_CRITICAL_TYPES = frozenset(
    {
        EntityType.PERSON.value,
        EntityType.PRODUCT.value,
        EntityType.PLACE.value,
        EntityType.DATE.value,
        EntityType.TIME.value,
        EntityType.NUMBER.value,
        EntityType.PERCENTAGE.value,
        EntityType.MONEY.value,
        EntityType.SCORE.value,
    }
)


def _has_latin_letter(token: str) -> bool:
    return bool(_LATIN_LETTER.search(token))


def _has_arabic_script(token: str) -> bool:
    return bool(_ARABIC_SCRIPT.search(token))


def _western_digits(token: str) -> str:
    return token.translate(_ARABIC_INDIC_DIGITS)


def _latin_letters(token: str) -> str:
    return "".join(character for character in token if character.isascii() and character.isalpha())


def _classify_token(token: str, text: str) -> EntityType:
    """Classify one protected token by deterministic shape rules."""

    stripped = token.strip()
    digits = _western_digits(stripped)
    lowered = stripped.casefold()

    if stripped.endswith("%") or lowered in _PERCENT_WORDS:
        return EntityType.PERCENTAGE
    if re.search(
        rf"{re.escape(stripped)}\s*(?:%|percent|percentage|بالمية|بالمائة|في المية)",
        text,
        re.IGNORECASE,
    ):
        return EntityType.PERCENTAGE

    if any(symbol in stripped for symbol in _CURRENCY_SYMBOLS) or lowered in _CURRENCY_WORDS:
        return EntityType.MONEY
    if _CURRENCY_PREFIX.search(text) is not None and re.search(
        rf"(?:[$€£¥₪﷼]|(?:usd|egp|sar|aed|eur|gbp|dollar|dollars|pound|euro|ريال|دولار|جنيه|درهم|دينار|يورو))\s*{re.escape(stripped)}",
        text,
        re.IGNORECASE,
    ):
        return EntityType.MONEY

    if _TIME_PATTERN.fullmatch(digits):
        return EntityType.TIME
    if _DATE_PATTERN.fullmatch(digits):
        return EntityType.DATE
    if digits.isdigit() and _YEAR_RANGE[0] <= int(digits) <= _YEAR_RANGE[1]:
        return EntityType.DATE
    if _SCORE_PATTERN.fullmatch(digits):
        return EntityType.SCORE

    if _has_latin_letter(stripped) and any(character in stripped for character in "+/#"):
        return EntityType.TECHNICAL
    if _has_latin_letter(stripped) and ("." in stripped or "/" in stripped or "://" in stripped):
        return EntityType.TECHNICAL

    letters = _latin_letters(stripped)
    if len(letters) >= 2 and letters.isupper():
        return EntityType.ABBREVIATION

    if _NUMBER_PATTERN.fullmatch(digits):
        return EntityType.NUMBER

    if _has_latin_letter(stripped):
        return EntityType.PERSON if stripped[0].isupper() else EntityType.PRODUCT

    return EntityType.PRODUCT


def extract_entities(
    text: str,
    *,
    start: float | None = None,
    end: float | None = None,
    evidence_fingerprints: tuple[str, ...] = (),
) -> tuple[EntityMention, ...]:
    """Extract bounded practical entities from raw text.

    Only `extract_protected_tokens` candidates are considered, so the function
    never invents Arabic common-word entities. Display text is preserved exactly
    and `normalized` is populated for comparison only.
    """

    mentions: list[EntityMention] = []
    seen: set[tuple[str, str]] = set()
    for token in extract_protected_tokens(text):
        entity_type = _classify_token(token, text)
        key = (token, entity_type.value)
        if key in seen:
            continue
        seen.add(key)
        mentions.append(
            EntityMention(
                text=token,
                normalized=normalize_entity(token),
                entity_type=entity_type.value,
                start=start,
                end=end,
                evidence_fingerprints=tuple(evidence_fingerprints),
            )
        )
    return tuple(mentions)


def normalize_entity(text: str) -> str:
    """Return a comparison-only normalized form, never a display replacement."""

    value = text.strip()
    if not value:
        return ""
    return str(normalize_for_comparison(_western_digits(value)))


def compare_entity_sets(
    left: Sequence[EntityMention], right: Sequence[EntityMention]
) -> list[dict[str, object]]:
    """Return bounded same-type differences between two entity sets.

    A difference is reported only for entities of the same type whose
    comparison-normalized forms disagree. The function never selects a winner;
    it reports both display forms so a caller can resolve the meaning.
    """

    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for left_entity in left:
        for right_entity in right:
            if left_entity.entity_type != right_entity.entity_type:
                continue
            if left_entity.normalized == right_entity.normalized:
                continue
            key = (
                left_entity.entity_type,
                left_entity.normalized,
                right_entity.normalized,
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "entity_type": left_entity.entity_type,
                    "left": left_entity.text,
                    "right": right_entity.text,
                    "meaning_critical": left_entity.entity_type in _MEANING_CRITICAL_TYPES,
                }
            )
            if len(records) >= MAX_ENTITY_CONFLICTS:
                return records
    return records
