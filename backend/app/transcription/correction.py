"""Conservative, lexicon-backed correction for Egyptian Arabic ASR text."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from app.transcription.providers import (
    CorrectionProvider,
    CorrectionRequest,
    ProviderCorrection,
    ProviderResponseError,
    validate_provider_results,
)

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
    provider_batch_size: int = 32


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
        provider: CorrectionProvider | None = None,
    ) -> None:
        self._entries = entries
        self._version = version
        self._config = config
        self._provider = provider
        self._by_confusion = {
            normalize_for_comparison(confusion): entry
            for entry in entries
            for confusion in entry.confusions
        }

    @classmethod
    def from_default_lexicon(
        cls,
        config: CorrectionConfig = CorrectionConfig(),
        provider: CorrectionProvider | None = None,
    ) -> ContextualCorrector:
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
        return cls(entries, str(payload["version"]), config, provider)

    def correct(self, segments: Sequence[Mapping[str, object]]) -> list[SegmentCorrection]:
        """Return one correction per input segment while retaining input ordering."""

        provider_results = self._provider_results(segments)
        return [
            self._correct_one(index, segments, provider_results.get(index))
            for index in range(len(segments))
        ]

    def _provider_results(
        self, segments: Sequence[Mapping[str, object]]
    ) -> dict[int, ProviderCorrection]:
        if self._provider is None or not segments:
            return {}
        requests = [
            CorrectionRequest(
                segment_index=index,
                previous=context_window(segments, index, self._config.context_segments).previous,
                raw_text=str(segment.get("text", "")),
                following=context_window(segments, index, self._config.context_segments).following,
                candidate_text=self._candidate_text(str(segment.get("text", ""))),
            )
            for index, segment in enumerate(segments)
        ]
        results: dict[int, ProviderCorrection] = {}
        for start in range(0, len(requests), self._config.provider_batch_size):
            batch = requests[start : start + self._config.provider_batch_size]
            try:
                results.update(
                    validate_provider_results(
                        {request.segment_index for request in batch},
                        self._provider.correct_batch(batch),
                    )
                )
            except (ProviderResponseError, OSError):
                continue
        return results

    def _candidate_text(self, raw_text: str) -> str | None:
        entry = self._by_confusion.get(normalize_for_comparison(raw_text))
        return entry.canonical if entry is not None else None

    def _correct_one(
        self,
        index: int,
        segments: Sequence[Mapping[str, object]],
        provider_result: ProviderCorrection | None,
    ) -> SegmentCorrection:
        raw_text = str(segments[index].get("text", ""))
        entry = self._by_confusion.get(normalize_for_comparison(raw_text))
        if provider_result is not None and self._provider_result_is_safe(raw_text, provider_result):
            if entry is not None and provider_result.corrected_text == entry.canonical:
                return self._provider_correction(index, raw_text, provider_result, "llm+lexicon")
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

    def _provider_result_is_safe(self, raw_text: str, result: ProviderCorrection) -> bool:
        if result.confidence < self._config.high_confidence:
            return False
        if result.changed != (result.corrected_text != raw_text):
            return False
        if _protected_tokens(raw_text) != _protected_tokens(result.corrected_text):
            return False
        if (
            _normalized_edit_ratio(raw_text, result.corrected_text)
            > self._config.max_small_edit_ratio
        ):
            return False
        return len(result.corrected_text) <= len(raw_text) + max(4, int(len(raw_text) * 0.35))

    def _provider_correction(
        self, index: int, raw_text: str, result: ProviderCorrection, method: str
    ) -> SegmentCorrection:
        return SegmentCorrection(
            segment_index=index,
            raw_text=raw_text,
            corrected_text=result.corrected_text,
            applied=result.changed,
            confidence=float(result.confidence),
            method=method,
            version=self._version,
            changes=result.changes,
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
    return (
        _WHITESPACE.sub(" ", without_punctuation.translate(_ARABIC_COMPARISON)).strip().casefold()
    )


def _normalized_edit_ratio(raw_text: str, corrected_text: str) -> float:
    """Bound provider rewrites so uncertain Arabic names and facts stay intact."""
    raw = normalize_for_comparison(raw_text)
    corrected = normalize_for_comparison(corrected_text)
    longest = max(len(raw), len(corrected), 1)
    previous = list(range(len(corrected) + 1))
    for raw_index, raw_character in enumerate(raw, start=1):
        current = [raw_index]
        for corrected_index, corrected_character in enumerate(corrected, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[corrected_index] + 1,
                    previous[corrected_index - 1] + (raw_character != corrected_character),
                )
            )
        previous = current
    return previous[-1] / longest


def context_window(
    segments: Sequence[Mapping[str, object]], target_index: int, context_segments: int = 2
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
            str(segment.get("text", "")) for segment in segments[target_index + 1 : following_end]
        ),
    )


def _protected_tokens(text: str) -> tuple[str, ...]:
    """Names/numbers and English technical terms must survive automatic edits exactly."""

    return tuple(re.findall(r"[A-Za-z]+|[0-9٠-٩]+", text.casefold()))
