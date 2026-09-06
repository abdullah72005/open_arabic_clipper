"""Arabic-aware phonetic comparison for reconstruction safety gates."""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_PROTECTED = re.compile(r"[A-Za-z]+|[0-9٠-٩]+")
_CONNECTED_SPEECH_EXTENSIONS = frozenset("اويعه")
_CANONICAL = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
        "ء": "",
        "ض": "د",
        "ص": "س",
        "ط": "ت",
        "ظ": "ز",
        "ث": "س",
        "ذ": "ز",
    }
)


def phonetic_similarity(source: str, candidate: str) -> float:
    """Return a bounded comparison score without relaxing Latin or numeric evidence."""

    if _protected_tokens(source) != _protected_tokens(candidate):
        return 0.0
    left = normalize_phonetic(source).replace(" ", "")
    right = normalize_phonetic(candidate).replace(" ", "")
    if left == right:
        return 1.0
    longest = max(len(left), len(right), 1)
    return max(0.0, 1.0 - _weighted_distance(left, right) / longest)


def normalize_phonetic(text: str) -> str:
    """Normalize only forms that commonly vary in Arabic ASR output."""

    normalized = unicodedata.normalize("NFC", text)
    normalized = _DIACRITICS.sub("", normalized).replace("ـ", "")
    normalized = _NON_WORD.sub(" ", normalized).translate(_CANONICAL)
    return _SPACE.sub(" ", normalized).strip().casefold()


def _weighted_distance(left: str, right: str) -> float:
    previous = [0.0]
    for character in right:
        previous.append(previous[-1] + _insertion_cost(character))
    for left_index, left_character in enumerate(left, start=1):
        current = [previous[0] + _insertion_cost(left_character)]
        for right_index, right_character in enumerate(right, start=1):
            substitution = _substitution_cost(left_character, right_character)
            current.append(
                min(
                    current[-1] + _insertion_cost(right_character),
                    previous[right_index] + _insertion_cost(left_character),
                    previous[right_index - 1] + substitution,
                )
            )
        previous = current
    return previous[-1]


def _protected_tokens(text: str) -> tuple[str, ...]:
    return tuple(_PROTECTED.findall(text.casefold()))


def _insertion_cost(character: str) -> float:
    return 0.45 if character in _CONNECTED_SPEECH_EXTENSIONS else 1.0


def _substitution_cost(left: str, right: str) -> float:
    if left == right:
        return 0.0
    return (
        0.45
        if left in _CONNECTED_SPEECH_EXTENSIONS or right in _CONNECTED_SPEECH_EXTENSIONS
        else 1.0
    )
