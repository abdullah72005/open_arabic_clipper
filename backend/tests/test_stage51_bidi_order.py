"""Deterministic derived-asset BiDi ordering and run-aware wrapping tests.

These tests pin the *intended human visual reading order* for each mixed
Arabic/English fixture as an explicit left-to-right token sequence, and check
that the pure derived-asset transform produces exactly that order. They do not
compare logical strings, which cannot prove visual order.
"""

from __future__ import annotations

import pytest

from app.composition.bidi import (
    LTR,
    RTL,
    atomic_units,
    paragraph_direction,
    token_direction,
    visual_order,
    visual_tokens,
)
from app.composition.captions import wrap_text
from app.composition.policy import CaptionStyle

# (canonical logical text, expected base direction, expected visual L->R order)
VISUAL_CASES: list[tuple[str, str, str]] = [
    (
        "أنا كنت content creator لمدة سنتين",
        RTL,
        "سنتين لمدة content creator كنت أنا",
    ),
    (
        "أنا بستخدم Python و Django كل يوم",
        RTL,
        "يوم كل Django و Python بستخدم أنا",
    ),
    ("السعر 150 EGP بس", RTL, "بس 150 EGP السعر"),
    (
        "جربت GPT-5 وبعدها رجعت للشغل",
        RTL,
        "للشغل رجعت وبعدها GPT-5 جربت",
    ),
    (
        "This is اختبار بسيط with English",
        LTR,
        "This is بسيط اختبار with English",
    ),
    ("أنا سعيد، لأن Python رائع!", RTL, "رائع! Python لأن سعيد، أنا"),
    ("خصم 20% على السعر", RTL, "السعر على 20% خصم"),
]


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("text", "base", "expected"),
    VISUAL_CASES,
)
def test_visual_order_matches_intended_reading_sequence(
    text: str, base: str, expected: str
) -> None:
    tokens = text.split(" ")
    assert paragraph_direction(text) == base
    assert " ".join(visual_tokens(tokens, base)) == expected
    assert " ".join(tokens[index] for index in visual_order(tokens, base)) == expected


def test_visual_order_is_a_permutation_of_canonical_tokens() -> None:
    text = "أنا كنت content creator لمدة سنتين"
    tokens = text.split(" ")
    assert sorted(visual_tokens(tokens, RTL)) == sorted(tokens)


def test_token_direction_classifies_arabic_latin_numbers_and_punctuation() -> None:
    assert token_direction("أنا") == RTL
    assert token_direction("content") == LTR
    assert token_direction("GPT-5") == LTR
    assert token_direction("20%") is None
    assert token_direction("%") is None


def test_ltr_run_is_one_wrapping_unit() -> None:
    tokens = "أنا كنت content creator لمدة سنتين".split(" ")
    units = atomic_units(tokens, RTL)
    assert [2, 3] in units  # content creator stays together


def test_wrap_keeps_latin_run_on_one_line() -> None:
    style = CaptionStyle()
    lines, overflow = wrap_text("أنا كنت content creator لمدة سنتين", style, 864.0)
    joined = list(lines)
    assert any("content creator" in line for line in joined)
    assert not any(line.endswith("content") for line in joined)
    assert overflow in (True, False)


def test_wrap_does_not_split_latin_run_when_it_fits() -> None:
    style = CaptionStyle()
    lines, _ = wrap_text("أنا أراجع content creator الآن", style, 864.0)
    assert any("content creator" in line for line in lines)
