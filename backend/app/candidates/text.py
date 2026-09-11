"""Deterministic text selection, cue matching, and feature tokenization."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from functools import lru_cache

from app.transcription.dialect import extract_protected_tokens

_TEXT_PRIORITY = ("operator_text", "final_text", "corrected_text", "raw_text")

_SENTENCE_END = re.compile(r"[.!؟?…]|۔")
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_ARABIC_NORMALIZE = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
    }
)
_WORD = re.compile(r"[\u0600-\u06FF]+|[A-Za-z][A-Za-z0-9+._/-]*|\d+(?:[.,:/-]\d+)*")

_ARABIC_STOPWORDS = frozenset(
    {
        "في",
        "من",
        "على",
        "عن",
        "الي",
        "الا",
        "هذا",
        "هذه",
        "ذلك",
        "التي",
        "الذي",
        "هو",
        "هي",
        "انا",
        "انت",
        "احنا",
        "هم",
        "كان",
        "كانت",
        "يكون",
        "لا",
        "ما",
        "ان",
        "او",
        "ثم",
        "كل",
        "بعد",
        "قبل",
        "حتي",
        "مع",
        "بس",
        "طب",
        "يعني",
        "زي",
        "عشان",
        "علشان",
        "كده",
        "ده",
        "دي",
        "دا",
    }
)
_ENGLISH_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "these",
        "those",
        "it",
        "as",
        "at",
        "by",
        "from",
        "we",
        "you",
        "they",
        "i",
        "he",
        "she",
        "so",
        "if",
        "then",
        "than",
        "there",
        "here",
        "not",
        "no",
        "yes",
        "do",
        "does",
        "did",
        "like",
        "just",
        "really",
        "very",
    }
)
_STOPWORDS = _ARABIC_STOPWORDS | _ENGLISH_STOPWORDS


def analysis_segment_text(segment: Mapping[str, object]) -> str:
    """Select authoritative analysis text for one segment.

    Priority: non-empty ``operator_text`` (manual override), ``final_text``,
    ``corrected_text``, then raw/text fallback. Transcript fields are never
    modified by this selection.
    """

    for key in _TEXT_PRIORITY:
        value = segment.get(key)
        if key == "raw_text" and value is None:
            value = segment.get("text")
        if isinstance(value, str) and value.strip():
            return value
    value = segment.get("text")
    return str(value) if isinstance(value, str) else ""


def analysis_texts(segments: Sequence[Mapping[str, object]]) -> list[str]:
    return [analysis_segment_text(segment) for segment in segments]


def segment_start(segment: Mapping[str, object]) -> float:
    return _number(segment.get("start"))


def segment_end(segment: Mapping[str, object]) -> float:
    return _number(segment.get("end"))


def is_sentence_end(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()))


def is_question(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith(("?", "؟", "؟")) or "؟" in stripped


def has_any(text: str, cues: Sequence[str]) -> bool:
    matching = matching_text(text)
    return contains_any_cue(matching, cues)


def normalize_arabic(text: str) -> str:
    return text.translate(_ARABIC_NORMALIZE)


def matching_text(text: str) -> str:
    """Analysis-only normalized matching view for deterministic cue detection.

    This never replaces stored transcript text, corrected/final text, timestamps,
    numbers, names, URLs, technical forms, protected code-switch tokens, or hook
    display text. It normalizes Unicode safely, case-folds Latin text, removes
    Arabic diacritics and tatweel, and conservatively unifies common alif/ya
    variants so cue matching is robust to orthographic variation.
    """

    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ARABIC_DIACRITICS.sub("", normalized)
    normalized = normalize_arabic(normalized)
    return normalized.casefold()


@lru_cache(maxsize=8192)
def normalized_cue(cue: str) -> str:
    """Cache the matching view of a fixed cue string (pure function of the cue)."""

    return matching_text(cue)


def contains_cue(matching: str, cue: str) -> bool:
    """Cue membership against a precomputed :func:`matching_text` view."""

    return normalized_cue(cue) in matching


def contains_any_cue(matching: str, cues: Sequence[str]) -> bool:
    return any(normalized_cue(cue) in matching for cue in cues)


def tokenize(text: str) -> list[str]:
    """Stable Arabic/English-compatible feature tokens.

    Diacritics/tatweel are removed and Arabic letter variants unified; Latin
    tokens are case-folded; stopwords are removed. Stored candidate text is never
    altered by this feature-only transform.
    """

    protected = {token.casefold() for token in extract_protected_tokens(text)}
    tokens: list[str] = []
    for raw in _WORD.findall(text):
        token = normalize_arabic(_ARABIC_DIACRITICS.sub("", raw)).casefold()
        if not token or token in _STOPWORDS:
            continue
        tokens.append(token)
    if not tokens and protected:
        tokens = sorted(protected)
    return tokens


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def token_feature_counts(text: str) -> dict[str, int]:
    """Unigram + bigram feature counts for novelty comparison."""

    tokens = tokenize(text)
    counts: dict[str, int] = {}
    for token in tokens:
        counts[f"1:{token}"] = counts.get(f"1:{token}", 0) + 1
    for gram in ngrams(tokens, 2):
        key = f"2:{' '.join(gram)}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _number(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0
