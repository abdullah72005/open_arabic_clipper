"""Shared Arabic text normalization used only for comparison copies."""

from __future__ import annotations

import re
import unicodedata

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_ARABIC_COMPARISON = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})


def normalize_for_comparison(text: str) -> str:
    """Normalize Arabic spelling and layout only for candidate comparison."""

    canonical = unicodedata.normalize("NFC", text)
    without_diacritics = _ARABIC_DIACRITICS.sub("", canonical)
    without_punctuation = _PUNCTUATION.sub(" ", without_diacritics)
    return (
        _WHITESPACE.sub(" ", without_punctuation.translate(_ARABIC_COMPARISON)).strip().casefold()
    )
