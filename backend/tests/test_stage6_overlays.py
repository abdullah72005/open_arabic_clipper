"""Stage 6 deterministic authored ASS overlay and readability evidence."""

from __future__ import annotations

from fractions import Fraction

from app.materialization.overlays import (
    AUTHORED_ASS_POLICY_VERSION,
    build_authored_ass,
)
from app.materialization.types import AuthoredTimeline, TimelineSegment


def _timeline(*segments: TimelineSegment) -> AuthoredTimeline:
    return AuthoredTimeline(
        segments=segments,
        frame_rate=Fraction(30, 1),
        sample_rate=48000,
        output_duration=7.0,
        output_frame_count=210,
        output_sample_count=336000,
        schema_version="stage6-timeline-v2",
    )


def test_authored_ass_renders_cards_in_output_time() -> None:
    timeline = _timeline(
        TimelineSegment(
            segment_id="s0",
            plan_block_index=0,
            kind="SOURCE",
            output_start=0.0,
            output_end=3.0,
            is_hero=True,
        ),
        TimelineSegment(
            segment_id="s1",
            plan_block_index=1,
            kind="CARD",
            output_start=3.0,
            output_end=4.0,
            text="The mechanism is that mentorship loss compounds",
        ),
    )
    result = build_authored_ass(timeline, play_res_x=1080, play_res_y=1920)
    text = result.data.decode("utf-8")
    assert result.policy_version == AUTHORED_ASS_POLICY_VERSION
    assert result.event_count == 1
    assert "Dialogue: 0,0:00:03.00,0:00:04.00,Card" in text
    assert "mentorship" in text
    assert result.sha256


def test_authored_ass_escapes_and_wraps_arabic() -> None:
    timeline = _timeline(
        TimelineSegment(
            segment_id="s1",
            plan_block_index=1,
            kind="CARD",
            output_start=1.0,
            output_end=5.0,
            text="أحمد قابل محمد اليوم في المكتب بعد انتهاء الاجتماع الطويل جدا",
        )
    )
    result = build_authored_ass(timeline, play_res_x=1080, play_res_y=1920)
    text = result.data.decode("utf-8")
    assert "أحمد" in text
    assert "\\N" in text  # wrapped into multiple lines
    assert result.line_count >= 2


def test_authored_ass_flags_overlong_text() -> None:
    long_text = " ".join(["word"] * 40)
    timeline = _timeline(
        TimelineSegment(
            segment_id="s1",
            plan_block_index=2,
            kind="CARD",
            output_start=1.0,
            output_end=5.0,
            text=long_text,
        )
    )
    result = build_authored_ass(timeline, play_res_x=1080, play_res_y=1920)
    assert any(
        finding.get("dimension") == "VISUAL_CAPTION_READABILITY"
        and finding.get("level") == "FAIL"
        and "AUTHORED_TEXT_REQUIRES_SHORTENING" in finding.get("reason_codes", [])
        for finding in result.findings
    )
    # Complete copy is preserved; the card is refused rather than silently
    # truncated to fit three lines.
    visible = result.data.decode("utf-8")
    assert visible.count("word") == 40


def test_annotation_without_a_free_zone_is_a_centred_standalone_card() -> None:
    timeline = _timeline(
        TimelineSegment(
            segment_id="s1",
            plan_block_index=3,
            kind="TEXTUAL_ANNOTATION",
            output_start=1.0,
            output_end=3.0,
            text="note",
        )
    )
    result = build_authored_ass(timeline, play_res_x=1080, play_res_y=1920)
    text = result.data.decode("utf-8")
    assert "{\\an5\\pos(" in text
    # Centred inside the shared safe area, not the 10% bottom-inset/48px card.
    assert "{\\an8}" not in text
    assert "Style: Card,Noto Sans Arabic,88," in text
