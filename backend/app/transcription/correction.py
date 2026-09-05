"""Conservative, lexicon-backed correction for Egyptian Arabic ASR text."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_ARABIC_COMPARISON = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})


@dataclass(frozen=True)
class CorrectionConfig:
    """Thresholds for conservative automatic correction."""

    context_segments: int = 2
    high_confidence: float = 0.90
    medium_confidence: float = 0.75
    max_small_edit_ratio: float = 0.25


@dataclass(frozen=True)
class SegmentCorrection:
    """One derived correction; source segments are never modified."""

    segment_index: int
    raw_text: str
    corrected_text: str
    applied: bool
    confidence: float
    method: str
    version: str
    changes: list[dict[str, str]]
    uncertain: bool


@dataclass(frozen=True)
class LexiconEntry:
    canonical: str
    confusions: tuple[str, ...]
    priority: int
    notes: str


@dataclass(frozen=True)
class CorrectionContext:
    """Bounded neighboring text for one target segment."""

    previous: tuple[str, ...]
    current: str
    following: tuple[str, ...]


class ContextualCorrector:
    """Apply only declared, high-confidence corrections to one segment at a time."""

    def __init__(
        self,
        entries: tuple[LexiconEntry, ...],
        version: str,
        config: CorrectionConfig = CorrectionConfig(),
    ) -> None:
        self._entries = entries
        self._version = version
        self._config = config
        self._by_confusion = {
            normalize_for_comparison(confusion): entry
            for entry in entries
            for confusion in entry.confusions
        }

    @classmethod
    def from_default_lexicon(cls, config: CorrectionConfig = CorrectionConfig()) -> ContextualCorrector:
        """Load growable phrase hints from data rather than application logic."""

        path = Path(__file__).with_name("lexicons") / "egyptian_ar.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = tuple(
            LexiconEntry(
                canonical=str(entry["canonical"]),
                confusions=tuple(str(confusion) for confusion in entry.get("confusions", [])),
                priority=int(entry["priority"]),
                notes=str(entry["notes"]),
            )
            for entry in payload["entries"]
        )
        return cls(entries, str(payload["version"]), config)

    def correct(self, segments: list[Mapping[str, object]]) -> list[SegmentCorrection]:
        """Return one correction per input segment while retaining input ordering."""

        return [self._correct_one(index, segments) for index in range(len(segments))]

    def _correct_one(
        self, index: int, segments: list[Mapping[str, object]]
    ) -> SegmentCorrection:
        raw_text = str(segments[index].get("text", ""))
        entry = self._by_confusion.get(normalize_for_comparison(raw_text))
        if entry is None or entry.priority < 90:
            return self._unchanged(index, raw_text)

        confidence = min(0.99, 0.90 + entry.priority / 1_000)
        if confidence < self._config.high_confidence:
            return self._unchanged(index, raw_text)
        return SegmentCorrection(
            segment_index=index,
            raw_text=raw_text,
            corrected_text=entry.canonical,
            applied=entry.canonical != raw_text,
            confidence=confidence,
            method="lexicon",
            version=self._version,
            changes=[
                {
                    "from": raw_text,
                    "to": entry.canonical,
                    "reason": "declared Egyptian Arabic phonetic ASR confusion",
                }
            ],
            uncertain=False,
        )

    def _unchanged(self, index: int, raw_text: str) -> SegmentCorrection:
        return SegmentCorrection(
            segment_index=index,
            raw_text=raw_text,
            corrected_text=raw_text,
            applied=False,
            confidence=0.0,
            method="unchanged",
            version=self._version,
            changes=[],
            uncertain=True,
        )


def normalize_for_comparison(text: str) -> str:
    """Normalize Arabic spelling and layout only for candidate comparison."""

    canonical = unicodedata.normalize("NFC", text)
    without_diacritics = _ARABIC_DIACRITICS.sub("", canonical)
    without_punctuation = _PUNCTUATION.sub(" ", without_diacritics)
    return _WHITESPACE.sub(" ", without_punctuation.translate(_ARABIC_COMPARISON)).strip().casefold()


def context_window(
    segments: list[Mapping[str, object]], target_index: int, context_segments: int = 2
) -> CorrectionContext:
    """Return neighboring raw text without exposing non-target output slots."""

    if context_segments < 0:
        raise ValueError("context_segments must be non-negative")
    if target_index < 0 or target_index >= len(segments):
        raise IndexError("target_index is outside the segment list")
    previous_start = max(0, target_index - context_segments)
    following_end = min(len(segments), target_index + 1 + context_segments)
    return CorrectionContext(
        previous=tuple(
            str(segment.get("text", "")) for segment in segments[previous_start:target_index]
        ),
        current=str(segments[target_index].get("text", "")),
        following=tuple(
            str(segment.get("text", ""))
            for segment in segments[target_index + 1 : following_end]
        ),
    )
