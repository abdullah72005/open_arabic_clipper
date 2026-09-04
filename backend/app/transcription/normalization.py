"""Conservative display normalization for Arabic and mixed transcripts."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION_SPACE = re.compile(r"\s+([,.;:!?،؛؟])")


def normalize_transcript(text: str) -> str:
    """Clean layout artifacts without translating or formalizing spoken language."""

    canonical = unicodedata.normalize("NFC", text)
    collapsed = _WHITESPACE.sub(" ", canonical).strip()
    return _PUNCTUATION_SPACE.sub(r"\1", collapsed)
