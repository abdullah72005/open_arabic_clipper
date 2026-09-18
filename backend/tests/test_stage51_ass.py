"""Pure Stage 5.1 ASS serialization, escaping, and storage tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.composition.ass import (
    aggregate_reason_codes,
    ass_asset_relative_path,
    escape_ass_text,
    escape_ass_text_with_evidence,
    serialize_ass,
    write_ass_asset,
)
from app.composition.captions import CaptionPlan, build_caption_plan
from app.composition.policy import (
    CaptionPlacementZone,
    CaptionStyle,
    PlanReasonCode,
    SafeZoneProfile,
    Stage51Config,
    safe_zone_for,
)
from app.composition.types import CaptionEvent
from app.services.storage import StorageService

SOURCE_ID = "11111111-1111-1111-1111-111111111111"
_WJ = "\u2060"
_BIDI_CONTROL_CODEPOINTS = tuple(range(0x202A, 0x202F)) + tuple(range(0x2066, 0x206A))


def _safe_zone() -> SafeZoneProfile:
    return safe_zone_for("SHORTS_VERTICAL_SAFE_ZONE_V1")


def _words(tokens: Sequence[str]) -> list[dict[str, object]]:
    return [
        {"index": index, "text": token, "start": index * 0.5, "end": index * 0.5 + 0.4}
        for index, token in enumerate(tokens)
    ]


def _plan() -> CaptionPlan:
    return build_caption_plan(
        [(0, 0.0, 0.9, 0, 1, "HERO")],
        _words(["hello", "world"]),
        CaptionStyle(),
        Stage51Config(),
    )


def _event(
    start: float,
    end: float,
    text: str,
    zone: str = CaptionPlacementZone.LOWER.value,
) -> CaptionEvent:
    return CaptionEvent(
        event_id=f"caption-{start}",
        block_index=0,
        word_start_index=int(start * 10),
        word_end_index=int(end * 10),
        start=start,
        end=end,
        text=text,
        lines=(text,),
        placement_zone=zone,
        placement_reason="DEFAULT_LOWER_ZONE",
    )


def _assert_no_bidi_controls(text: str) -> None:
    assert all(ord(character) not in _BIDI_CONTROL_CODEPOINTS for character in text)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("source", "expected"),
    [
        ("hello", "hello"),
        ("a{b", "a\\{b"),
        ("a}b", "a\\}b"),
        ("{p}", "\\{p\\}"),
        ("\\", "\\"),
        ("\\N", "\\" + _WJ + "N"),
        ("\\n", "\\" + _WJ + "n"),
        ("\\h", "\\" + _WJ + "h"),
        ("\\x", "\\x"),
        ("{\\p1}x{\\p0}", "\\{\\p1\\}x\\{\\p0\\}"),
        ("a\tb", "a b"),
        ("a\x00b", "ab"),
        ("a\x1fb", "ab"),
        ("a\x85b", "ab"),
        ("a\u2028b", "a b"),
    ],
)
def test_escape_ass_text_is_exact(source: str, expected: str) -> None:
    assert escape_ass_text(source) == expected
    _assert_no_bidi_controls(escape_ass_text(source))


def test_escape_ass_text_neutralizes_bidi_and_records_reason() -> None:
    escaped, reasons = escape_ass_text_with_evidence("a\u202bb\u2067c")
    assert escaped == "abc"
    assert reasons == (PlanReasonCode.BIDI_CONTROL_NEUTRALIZED.value,)


def test_escape_ass_text_never_leaves_an_unescaped_brace() -> None:
    for source in ["{", "}", "{\\p1}", "}{", "a{b}c", "\\N{", "}{\\p1}{\\p0}"]:
        escaped = escape_ass_text(source)
        for index, character in enumerate(escaped):
            if character in "{}":
                assert index > 0 and escaped[index - 1] == "\\"


def test_serialize_ass_is_deterministic() -> None:
    plan = _plan()
    assert serialize_ass(plan, CaptionStyle(), _safe_zone()) == serialize_ass(
        plan, CaptionStyle(), _safe_zone()
    )


def test_serialize_ass_header_and_styles() -> None:
    data = serialize_ass(_plan(), CaptionStyle(), _safe_zone())
    text = data.decode("utf-8")

    assert data.startswith(b"[Script Info]\n")
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data
    assert text.endswith("\n")
    assert "ScriptType: v4.00+" in text
    assert "PlayResX: 1080" in text
    assert "PlayResY: 1920" in text
    assert "WrapStyle: 2" in text
    assert "Style: CaptionLower" in text
    assert "Style: CaptionUpper" in text
    assert ",2,54,162,504,1" in text
    assert ",8,54,162,254,1" in text


def test_serialize_ass_uses_style_configuration() -> None:
    style = CaptionStyle(
        font_family="DejaVu Sans",
        font_size=64,
        primary_color="&H00112233",
        outline_width=7,
        alignment_lower=2,
        alignment_upper=8,
    )
    text = serialize_ass(_plan(), style, _safe_zone()).decode("utf-8")

    assert "CaptionLower,DejaVu Sans,64,&H00112233" in text
    assert ",1,7,0,2,54,162,504,1" in text


def test_serialize_ass_sorts_events_by_start_then_word_index() -> None:
    plan = CaptionPlan(
        policy_version="test",
        events=(
            _event(2.0, 2.5, "second"),
            _event(1.0, 1.5, "first"),
        ),
    )
    text = serialize_ass(plan, CaptionStyle(), _safe_zone()).decode("utf-8")
    dialogue_lines = [line for line in text.splitlines() if line.startswith("Dialogue:")]

    assert len(dialogue_lines) == 2
    assert dialogue_lines[0].endswith("first")
    assert dialogue_lines[1].endswith("second")


def test_serialize_ass_places_upper_zone_with_an8() -> None:
    plan = CaptionPlan(
        policy_version="test",
        events=(_event(1.0, 1.5, "up", zone=CaptionPlacementZone.UPPER.value),),
    )
    text = serialize_ass(plan, CaptionStyle(), _safe_zone()).decode("utf-8")

    assert "CaptionUpper" in text
    assert "{\\an8}up" in text


def test_serialize_ass_escapes_source_injection() -> None:
    hostile = "{\\p1}m 0 0 l 800 0 800 800 0 800{\\p0}"
    plan = CaptionPlan(policy_version="test", events=(_event(1.0, 1.5, hostile),))
    text = serialize_ass(plan, CaptionStyle(), _safe_zone()).decode("utf-8")

    assert "{\\p1}" not in text
    assert "\\{\\p1\\}" in text
    _assert_no_bidi_controls(text)


def test_aggregate_reason_codes_includes_neutralization() -> None:
    plan = CaptionPlan(
        policy_version="test",
        events=(_event(1.0, 1.5, "a\u202bb"),),
        reason_codes=(),
    )
    assert PlanReasonCode.BIDI_CONTROL_NEUTRALIZED.value in aggregate_reason_codes(plan)


def test_ass_asset_relative_path_layout() -> None:
    assert ass_asset_relative_path(SOURCE_ID, "fp") == (
        f"sources/{SOURCE_ID}/visual-composition/fp.ass"
    )


def test_write_ass_asset_writes_atomically_and_reuses_identical_bytes(tmp_path: Path) -> None:
    storage = StorageService(tmp_path)
    data = b"[Script Info]\n"
    path, digest = write_ass_asset(storage, SOURCE_ID, "fp1", data)

    assert path == storage.source_directory(SOURCE_ID) / "visual-composition" / "fp1.ass"
    assert path.read_bytes() == data
    assert digest == hashlib.sha256(data).hexdigest()

    stat_before = path.stat().st_mtime_ns
    reused_path, reused_digest = write_ass_asset(storage, SOURCE_ID, "fp1", data)
    assert reused_path == path
    assert reused_digest == digest
    assert path.stat().st_mtime_ns == stat_before
    assert not list(path.parent.glob("*.tmp"))


def test_write_ass_asset_overwrites_when_digest_differs(tmp_path: Path) -> None:
    storage = StorageService(tmp_path)
    path, _ = write_ass_asset(storage, SOURCE_ID, "fp1", b"old")
    rewritten, digest = write_ass_asset(storage, SOURCE_ID, "fp1", b"new")

    assert rewritten == path
    assert path.read_bytes() == b"new"
    assert digest == hashlib.sha256(b"new").hexdigest()
