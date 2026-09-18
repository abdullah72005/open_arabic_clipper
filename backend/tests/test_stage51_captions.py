"""Pure Stage 5.1 FINAL_CLIP caption-plan construction tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.composition.captions import (
    BIDI_CONTROL_NEUTRALIZED,
    CAPTION_COLLISION_UNRESOLVED,
    CAPTION_EVIDENCE_MISSING,
    CaptionPlan,
    SceneFaceBox,
    build_caption_plan,
    estimate_text_width,
    wrap_text,
)
from app.composition.policy import CaptionPlacementZone, CaptionStyle, Stage51Config
from app.composition.types import BoundSpan, CaptionEvent


def _words(
    tokens: Sequence[str],
    *,
    start: float = 0.0,
    step: float = 0.5,
    duration: float = 0.4,
) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "text": token,
            "start": start + index * step,
            "end": start + index * step + duration,
        }
        for index, token in enumerate(tokens)
    ]


def _timed_words(
    entries: Sequence[tuple[str, float, float]],
) -> list[dict[str, object]]:
    return [
        {"index": index, "text": text, "start": begin, "end": finish}
        for index, (text, begin, finish) in enumerate(entries)
    ]


def _plan(
    spans: Sequence[object],
    words: Sequence[dict[str, object]],
    *,
    style: CaptionStyle | None = None,
    scene_face_boxes: Sequence[object] = (),
    previous_scene_zones: dict[int, str] | None = None,
    hero_block_index: int | None = None,
) -> CaptionPlan:
    return build_caption_plan(
        spans,
        words,
        style or CaptionStyle(),
        Stage51Config(),
        scene_face_boxes=scene_face_boxes,
        previous_scene_zones=previous_scene_zones,
        hero_block_index=hero_block_index,
    )


def _event_texts(plan: CaptionPlan) -> list[str]:
    return [event.text for event in plan.events]


def test_words_in_explicit_range_form_one_exact_logical_event() -> None:
    plan = _plan([(0, 0.0, 1.4, 0, 2, "HERO")], _words(["hello", "world", "again"]))

    assert len(plan.events) == 1
    event = plan.events[0]
    assert event.text == "hello world again"
    assert event.lines == ("hello world again",)
    assert event.word_start_index == 0
    assert event.word_end_index == 2
    assert event.start == 0.0


def test_fallback_selects_words_whose_times_fall_inside_span() -> None:
    plan = _plan(
        [(0, 0.5, 1.9, None, None, "SUPPORT")],
        _words(["zero", "one", "two", "three"]),
    )

    assert len(plan.events) == 1
    assert plan.events[0].text == "one two three"
    assert plan.events[0].word_start_index == 1
    assert plan.events[0].word_end_index == 3


def test_span_without_word_evidence_emits_marker_and_never_fabricates() -> None:
    plan = _plan([(4, 100.0, 101.0, None, None, "SUPPORT")], _words(["hello", "world"]))

    assert plan.events == ()
    assert len(plan.missing_evidence) == 1
    assert plan.missing_evidence[0].block_index == 4
    assert plan.missing_evidence[0].reason == CAPTION_EVIDENCE_MISSING
    assert CAPTION_EVIDENCE_MISSING in plan.reason_codes


def test_punctuation_splits_events_without_splitting_every_word() -> None:
    plan = _plan([(0, 0.0, 1.9, 0, 3, "HERO")], _words(["hello", "world.", "next", "one"]))

    assert _event_texts(plan) == ["hello world.", "next one"]
    assert len(plan.events) < 4


def test_inter_word_pause_splits_events() -> None:
    words = _timed_words(
        [
            ("a", 0.0, 0.4),
            ("b", 0.5, 0.9),
            ("c", 2.0, 2.4),
            ("d", 2.5, 2.9),
        ]
    )
    plan = _plan([(0, 0.0, 2.9, 0, 3, "HERO")], words)

    assert _event_texts(plan) == ["a b", "c d"]


def test_max_words_per_event_is_enforced() -> None:
    tokens = [f"w{index}" for index in range(15)]
    plan = _plan(
        [(0, 0.0, 1.5, 0, 14, "HERO")],
        _words(tokens, step=0.1, duration=0.08),
    )

    assert len(plan.events) >= 2
    for event in plan.events:
        word_count = event.word_end_index - event.word_start_index + 1
        assert word_count <= 12


def test_event_end_is_clamped_to_span_tail() -> None:
    words = _timed_words([("short", 0.0, 0.1)])
    plan = _plan([(0, 0.0, 0.2, 0, 0, "HERO")], words)

    assert len(plan.events) == 1
    assert plan.events[0].end == pytest.approx(0.5)


def test_overlapping_events_are_trimmed_so_none_overlap() -> None:
    words = _timed_words([("first", 0.0, 0.4), ("second", 0.3, 0.7)])
    plan = _plan(
        [
            (0, 0.0, 0.4, 0, 0, "HERO"),
            (1, 0.3, 0.7, 1, 1, "SUPPORT"),
        ],
        words,
        hero_block_index=0,
    )

    ordered = sorted(plan.events, key=lambda event: event.start)
    assert len(ordered) == 2
    assert ordered[0].end <= ordered[1].start


def test_default_placement_is_lower_zone() -> None:
    plan = _plan([(0, 0.0, 0.9, 0, 1, "HERO")], _words(["hello", "world"]))

    assert plan.events[0].placement_zone == CaptionPlacementZone.LOWER.value
    assert plan.events[0].placement_reason == "DEFAULT_LOWER_ZONE"
    assert plan.scene_zones == {}


def test_scene_face_boxes_accept_seven_field_tuple() -> None:
    plan = _plan(
        [(0, 0.0, 0.9, 0, 1, "HERO")],
        _words(["hello", "world"]),
        scene_face_boxes=[(0, 0.0, 2.0, 0.30, 0.70, 0.40, 0.05)],
    )

    assert plan.scene_zones == {0: CaptionPlacementZone.UPPER.value}


def test_bound_span_objects_are_accepted() -> None:
    span = BoundSpan(
        block_index=0,
        start=0.0,
        end=0.9,
        word_start_index=0,
        word_end_index=1,
        source_role="HERO",
        is_hero=True,
        caption_text="",
    )
    plan = build_caption_plan([span], _words(["hello", "world"]), CaptionStyle(), Stage51Config())

    assert plan.events[0].text == "hello world"


def test_persistent_lower_collision_switches_whole_scene_to_upper() -> None:
    face = SceneFaceBox(scene_index=0, start=0.0, end=2.0, x=0.30, y=0.70, w=0.40, h=0.05)
    plan = _plan(
        [(0, 0.0, 0.9, 0, 1, "HERO")],
        _words(["hello", "world"]),
        scene_face_boxes=[face],
    )

    assert plan.scene_zones == {0: CaptionPlacementZone.UPPER.value}
    assert plan.events[0].placement_zone == CaptionPlacementZone.UPPER.value
    assert plan.events[0].placement_reason == "SCENE_COLLISION_SWITCH_UPPER"
    assert plan.events[0].collision_evidence["lower_fraction"] > 0.15


def test_upper_scene_reverts_to_lower_only_when_collision_free() -> None:
    face = SceneFaceBox(scene_index=0, start=0.0, end=2.0, x=0.40, y=0.40, w=0.20, h=0.10)
    plan = _plan(
        [(0, 0.0, 0.9, 0, 1, "HERO")],
        _words(["hello", "world"]),
        scene_face_boxes=[face],
        previous_scene_zones={0: CaptionPlacementZone.UPPER.value},
    )

    assert plan.scene_zones == {0: CaptionPlacementZone.LOWER.value}
    assert plan.events[0].placement_zone == CaptionPlacementZone.LOWER.value
    assert plan.events[0].placement_reason == "SCENE_COLLISION_FREE_HYSTERESIS_LOWER"


def test_upper_scene_holds_when_lower_collision_is_below_threshold() -> None:
    face = SceneFaceBox(scene_index=0, start=0.0, end=2.0, x=0.30, y=0.72, w=0.10, h=0.01)
    plan = _plan(
        [(0, 0.0, 0.9, 0, 1, "HERO")],
        _words(["hello", "world"]),
        scene_face_boxes=[face],
        previous_scene_zones={0: CaptionPlacementZone.UPPER.value},
    )

    assert plan.scene_zones == {0: CaptionPlacementZone.UPPER.value}
    assert plan.events[0].placement_reason == "SCENE_HYSTERESIS_HOLD_UPPER"


def test_hero_span_with_both_zones_colliding_picks_lesser_and_flags() -> None:
    face = SceneFaceBox(scene_index=0, start=0.0, end=2.0, x=0.0, y=0.10, w=1.0, h=0.70)
    plan = _plan(
        [(0, 0.0, 0.9, 0, 1, "HERO")],
        _words(["hello", "world"]),
        scene_face_boxes=[face],
        hero_block_index=0,
    )

    assert CAPTION_COLLISION_UNRESOLVED in plan.reason_codes
    event = plan.events[0]
    assert event.placement_reason == CAPTION_COLLISION_UNRESOLVED
    assert event.placement_zone in {
        CaptionPlacementZone.LOWER.value,
        CaptionPlacementZone.UPPER.value,
    }


def test_canonical_text_preserves_tokens_and_bidi_controls() -> None:
    token = "abc\u202bdef"
    plan = _plan([(0, 0.0, 0.4, 0, 0, "HERO")], _words([token]))

    assert len(plan.events) == 1
    assert plan.events[0].text == token
    assert BIDI_CONTROL_NEUTRALIZED in plan.reason_codes


def test_unordered_explicit_indexes_are_rejected() -> None:
    with pytest.raises(AssertionError):
        _plan([(0, 0.0, 0.9, 2, 0, "HERO")], _words(["a", "b", "c"]))


def test_explicit_index_missing_from_timestamps_is_rejected() -> None:
    with pytest.raises(AssertionError):
        _plan([(0, 0.0, 0.9, 0, 5, "HERO")], _words(["a", "b"]))


def test_wrap_text_greedily_wraps_and_reports_overflow() -> None:
    style = CaptionStyle()
    lines, overflow = wrap_text("alpha beta", style, 864.0)
    assert lines == ("alpha beta",)
    assert overflow is False

    narrow = CaptionStyle(max_line_width_fraction=0.05, max_lines=1)
    lines, overflow = wrap_text("alpha beta", narrow, 54.0)
    assert lines == ("alpha", "beta")
    assert overflow is True


def test_over_wide_word_occupies_its_own_line() -> None:
    lines, overflow = wrap_text("mmmmmmmmmm", CaptionStyle(), 1.0)
    assert lines == ("mmmmmmmmmm",)
    assert overflow is False


def test_estimated_width_is_script_aware() -> None:
    style = CaptionStyle()
    assert estimate_text_width("مرحبا", style) > 0.0
    assert estimate_text_width("hello", style) > 0.0


def test_plan_sorts_events_by_start_then_index() -> None:
    words = _timed_words([("b", 1.0, 1.4), ("a", 0.0, 0.4)])
    plan = _plan(
        [
            (0, 1.0, 1.4, 0, 0, "SUPPORT"),
            (1, 0.0, 0.4, 1, 1, "HERO"),
        ],
        words,
    )

    assert [event.start for event in plan.events] == sorted(event.start for event in plan.events)
    assert all(isinstance(event, CaptionEvent) for event in plan.events)


def test_plan_as_dict_is_json_friendly() -> None:
    plan = _plan([(0, 0.0, 0.9, 0, 1, "HERO")], _words(["hello", "world"]))

    payload = plan.as_dict()
    assert payload["policy_version"] == "stage5.1-caption-layout-v1"
    events = payload["events"]
    assert isinstance(events, list)
    assert events[0]["text"] == "hello world"
