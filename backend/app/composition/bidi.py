"""Deterministic derived-asset BiDi ordering for Stage 5.1 captions.

The deployed libass build does not reorder mixed-direction text (empirically
verified: pure Arabic renders in logical left-to-right order, and Unicode
paragraph marks / isolates have no effect), so the *derived* ASS asset must
carry the visual word order itself. This module is pure and provider-free: no
network, no model, no rendering.

Guarantees:

- canonical transcript text is never read from or written back to here; callers
  pass an already-tokenized caption and receive a permutation of token indexes;
- characters inside a token are never reversed, so Arabic shaping (HarfBuzz)
  and Arabic letters keep their canonical logical order;
- only whole tokens are moved, using Unicode bidirectional classes, to produce
  the visual left-to-right reading order for the paragraph base direction.

This is a word/run-level application of the Unicode Bidirectional Algorithm,
sufficient for captions made of Arabic words, Latin runs, numbers, and
punctuation. It is deliberately not a token reversal: RTL tokens keep their
internal order and Latin runs stay internally left-to-right.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

RTL = "RTL"
LTR = "LTR"

_RTL_CLASSES = frozenset({"R", "AL"})
_LTR_CLASSES = frozenset({"L"})
_NUMBER_CLASSES = frozenset({"EN", "AN"})


def _strong_classes(text: str) -> set[str]:
    return {unicodedata.bidirectional(char) for char in text if char}


def paragraph_direction(text: str) -> str:
    """Return the paragraph base direction from the first strong character."""

    for char in text:
        bidi_class = unicodedata.bidirectional(char)
        if bidi_class in _RTL_CLASSES:
            return RTL
        if bidi_class in _LTR_CLASSES:
            return LTR
    return LTR


def token_direction(token: str) -> str | None:
    """Classify one token as ``RTL``, ``LTR``, or ``None`` (neutral/number)."""

    classes = _strong_classes(token)
    if classes & _RTL_CLASSES:
        return RTL
    if classes & _LTR_CLASSES:
        return LTR
    return None


def _resolved_directions(tokens: Sequence[str], base: str) -> list[str]:
    directions: list[str | None] = [token_direction(token) for token in tokens]
    # Numbers read left-to-right; attach them to the surrounding strong context
    # rather than treating them as neutral punctuation.
    for index, token in enumerate(tokens):
        if directions[index] is None and _strong_classes(token) & _NUMBER_CLASSES:
            directions[index] = LTR
    # Resolve remaining neutrals to the previous strong run, else the next.
    previous: str | None = None
    for index in range(len(tokens)):
        if directions[index] is None:
            directions[index] = previous
        else:
            previous = directions[index]
    following: str | None = None
    for index in range(len(tokens) - 1, -1, -1):
        if directions[index] is None:
            directions[index] = following
        else:
            following = directions[index]
    return [direction or base for direction in directions]


def visual_order(tokens: Sequence[str], base: str) -> list[int]:
    """Return token indexes in visual left-to-right order for ``base``.

    Runs are ordered right-to-left for an RTL paragraph and left-to-right for
    an LTR paragraph; tokens inside an RTL run are reversed, tokens inside an
    LTR run keep their order.
    """

    if not tokens:
        return []
    directions = _resolved_directions(tokens, base)
    runs: list[tuple[str, list[int]]] = []
    for index, direction in enumerate(directions):
        if runs and runs[-1][0] == direction:
            runs[-1][1].append(index)
        else:
            runs.append((direction, [index]))

    ordered: list[int] = []
    run_sequence = reversed(runs) if base == RTL else runs
    for direction, indexes in run_sequence:
        if direction == RTL:
            ordered.extend(reversed(indexes))
        else:
            ordered.extend(indexes)
    return ordered


def visual_tokens(tokens: Sequence[str], base: str) -> list[str]:
    """Return the tokens reordered into visual left-to-right order."""

    return [tokens[index] for index in visual_order(tokens, base)]


def atomic_units(tokens: Sequence[str], base: str) -> list[list[int]]:
    """Return wrapping units: each maximal LTR run is one unit, RTL is per-token.

    An LTR run such as ``content creator`` wraps as a single unit when it fits,
    so it is never split across lines merely because it is two word tokens.
    """

    if not tokens:
        return []
    directions = _resolved_directions(tokens, base)
    units: list[list[int]] = []
    index = 0
    while index < len(tokens):
        if directions[index] == LTR:
            end = index
            while end + 1 < len(tokens) and directions[end + 1] == LTR:
                end += 1
            units.append(list(range(index, end + 1)))
            index = end + 1
        else:
            units.append([index])
            index += 1
    return units


__all__ = [
    "LTR",
    "RTL",
    "atomic_units",
    "paragraph_direction",
    "token_direction",
    "visual_order",
    "visual_tokens",
]
