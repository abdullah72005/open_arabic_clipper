"""Deterministic safety checks for model-proposed transcript reconstruction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.transcription.reconstruction.entities import SourceEntityMemory
from app.transcription.reconstruction.phonetics import phonetic_similarity
from app.transcription.reconstruction.types import ReconstructionCandidate

_PROTECTED = re.compile(r"[A-Za-z]+|[0-9٠-٩]+")


@dataclass(frozen=True)
class CandidateValidation:
    accepted: bool
    reason: str | None
    phonetic_similarity: float
    edit_ratio: float
    token_delta: int


def validate_candidate(
    raw_text: str, candidate: ReconstructionCandidate, memory: SourceEntityMemory
) -> CandidateValidation:
    """Reject changes that cannot be supported without audio realignment."""

    score = phonetic_similarity(raw_text, candidate.text)
    edit_ratio = _edit_ratio(raw_text, candidate.text)
    token_delta = abs(len(raw_text.split()) - len(candidate.text.split()))
    if not candidate.text.strip():
        return CandidateValidation(False, "empty_text", score, edit_ratio, token_delta)
    if _protected_tokens(raw_text) != _protected_tokens(candidate.text):
        return CandidateValidation(
            False, "protected_tokens_changed", score, edit_ratio, token_delta
        )
    if not 0.60 <= len(candidate.text) / max(len(raw_text), 1) <= 1.60:
        return CandidateValidation(False, "length_ratio", score, edit_ratio, token_delta)
    if token_delta > max(3, _ceil_fraction(len(raw_text.split()))):
        return CandidateValidation(False, "token_delta", score, edit_ratio, token_delta)
    if score < 0.55:
        return CandidateValidation(False, "phonetic_similarity", score, edit_ratio, token_delta)
    return CandidateValidation(True, None, score, edit_ratio, token_delta)


def _protected_tokens(text: str) -> tuple[str, ...]:
    return tuple(_PROTECTED.findall(text.casefold()))


def _ceil_fraction(tokens: int) -> int:
    return max(1, (tokens * 40 + 99) // 100)


def _edit_ratio(left: str, right: str) -> float:
    longest = max(len(left), len(right), 1)
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / longest
