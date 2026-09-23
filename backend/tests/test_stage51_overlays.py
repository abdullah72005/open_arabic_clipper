"""Pure Stage 5.1 overlay placement-requirement tests.

No network, no provider, no model, no FFmpeg. These tests pin the
materialization-only contract: Stage 5.1 derives *placement* requirements from
the real closed Stage 4.1/5.0 vocabulary and never invents or authorizes text.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from app.composition.overlays import (
    CONSTRAINT_CAPTION_ZONE,
    CONSTRAINT_NO_FREE_ZONE,
    CONSTRAINT_PROTECTED_FACE,
    CONSTRAINT_SAFE_ZONES,
    OVERLAY_BASE_CONSTRAINTS,
    ProtectedRegion,
    build_overlay_requirements,
    resolve_zone_conflicts,
)
from app.composition.policy import OverlayStatus, OverlayZone, safe_zone_for
from app.composition.types import OverlayRequirement

MATERIALIZATION_REQUIRED = OverlayStatus.MATERIALIZATION_REQUIRED.value


def _block(
    index: int,
    block_type: str,
    *,
    placement: str = "sequential",
    purpose: str = "",
    interrupts_source: bool = False,
    start: float = 0.0,
    end: float = 2.0,
    authoritative: bool = False,
) -> dict[str, object]:
    return {
        "block_index": index,
        "block_type": block_type,
        "purpose": purpose,
        "placement": placement,
        "interrupts_source": interrupts_source,
        "timeline": {"start": start, "end": end, "authoritative": authoritative},
    }


def _authored_text_slot(
    block_index: int,
    block_type: str,
    *,
    delivery_intent: str = "ON_SCREEN_TEXT",
    draft_line: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "reason_code": "AUTHORED_TEXT_MATERIALIZATION_REQUIRED",
        "purpose": "On-screen context annotation",
        "delivery_intent": delivery_intent,
        "placement": "sequential",
    }
    if draft_line is not None:
        payload["authoring_reference"] = {
            "draft_line": draft_line,
            "draft_only": True,
            "authoritative": False,
        }
    return {
        "slot_id": f"block-{block_index}",
        "slot_kind": "AUTHORED_TEXT",
        "block_index": block_index,
        "block_type": block_type,
        "required": True,
        "reason_code": "AUTHORED_TEXT_MATERIALIZATION_REQUIRED",
        "payload": payload,
    }


def _narration_slot(purposes: Sequence[str]) -> dict[str, object]:
    return {
        "slot_id": "narration",
        "slot_kind": "AUTHORED_NARRATION",
        "block_index": None,
        "block_type": "NARRATION_REQUIREMENT",
        "required": True,
        "reason_code": "NARRATION_MATERIALIZATION_REQUIRED",
        "payload": {"need": "REQUIRED", "purposes": list(purposes), "essential": True},
    }


def _walk(value: object, path: str = "") -> Iterator[tuple[str, object]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for position, item in enumerate(value):
            yield from _walk(item, f"{path}[{position}]")
    else:
        yield path, value


def _requirements(
    slots: Sequence[dict[str, object]],
    blocks: Sequence[dict[str, object]] = (),
    **kwargs: object,
) -> tuple[OverlayRequirement, ...]:
    return build_overlay_requirements(slots, blocks, **kwargs)  # type: ignore[arg-type]


def test_validated_textual_annotation_slot_yields_materialization_required() -> None:
    slots = [
        {
            "slot_id": "block-0",
            "slot_kind": "SOURCE_MEDIA",
            "block_index": 0,
            "block_type": "SOURCE_EXCERPT",
            "required": False,
            "reason_code": "",
            "payload": {},
        },
        _authored_text_slot(1, "TEXTUAL_ANNOTATION"),
        {
            "slot_id": "block-2",
            "slot_kind": "TRANSITION",
            "block_index": 2,
            "block_type": "TRANSITION",
            "required": False,
            "reason_code": "",
            "payload": {},
        },
        {
            "slot_id": "block-3",
            "slot_kind": "VERIFICATION_EVIDENCE",
            "block_index": 3,
            "block_type": "FACT_VERIFICATION_PLACEHOLDER",
            "required": False,
            "reason_code": "",
            "payload": {},
        },
    ]
    blocks = [
        _block(0, "SOURCE_EXCERPT"),
        _block(1, "TEXTUAL_ANNOTATION", purpose="On-screen context annotation", start=2.0, end=5.0),
        _block(2, "TRANSITION"),
        _block(3, "FACT_VERIFICATION_PLACEHOLDER"),
    ]

    result = _requirements(slots, blocks)

    assert len(result) == 1
    requirement = result[0]
    assert requirement.text is None
    assert requirement.status == MATERIALIZATION_REQUIRED
    assert requirement.block_type == "TEXTUAL_ANNOTATION"
    assert requirement.slot_kind == "AUTHORED_TEXT"
    assert requirement.desired_zone == OverlayZone.UPPER_THIRD.value
    assert "TEXTUAL_ANNOTATION_TO_UPPER_THIRD" in requirement.placement_reason
    assert requirement.collision_constraints == OVERLAY_BASE_CONSTRAINTS
    assert requirement.timeline_hint == {"start": 2.0, "end": 5.0, "authoritative": False}


def test_slot_without_text_keeps_draft_non_authoritative() -> None:
    draft = "DRAFT_INVENTED_ANNOTATION_XYZ"
    requirement = _requirements(
        [_authored_text_slot(1, "TEXTUAL_ANNOTATION", draft_line=draft)],
        [_block(1, "TEXTUAL_ANNOTATION")],
    )[0]

    serialized = requirement.as_dict()
    assert requirement.text is None
    assert requirement.status == MATERIALIZATION_REQUIRED

    occurrences = [
        (path, value)
        for path, value in _walk(serialized)
        if isinstance(value, str) and draft in value
    ]
    assert occurrences, "sandbox draft must be carried as an explicit reference"
    for path, _ in occurrences:
        assert path.endswith("authoring_reference.draft_line"), path

    reference = serialized["authoring_reference"]
    assert isinstance(reference, dict)
    assert reference["draft_line"] == draft
    assert reference["draft_only"] is True
    assert reference["authoritative"] is False
    assert requirement.placement_reason
    assert draft not in requirement.placement_reason


def test_contract_narration_slot_is_placement_only() -> None:
    requirement = _requirements([_narration_slot(["HOOK"])])[0]

    assert requirement.block_index is None
    assert requirement.block_type == "NARRATION_REQUIREMENT"
    assert requirement.slot_kind == "AUTHORED_NARRATION"
    assert requirement.text is None
    assert requirement.status == MATERIALIZATION_REQUIRED
    assert requirement.desired_zone == OverlayZone.TOP_HOOK.value
    assert "NARRATION_PURPOSE_HOOK_TO_TOP_HOOK" in requirement.placement_reason
    assert requirement.timeline_hint is None


def test_original_value_narration_delivery_is_placement_only() -> None:
    slot = {
        "slot_id": "block-1",
        "slot_kind": "AUTHORED_NARRATION",
        "block_index": 1,
        "block_type": "ORIGINAL_VALUE",
        "required": True,
        "reason_code": "NARRATION_MATERIALIZATION_REQUIRED",
        "payload": {"delivery_intent": "NARRATION", "purpose": "Authored contribution"},
    }
    requirement = _requirements([slot], [_block(1, "ORIGINAL_VALUE")])[0]

    assert requirement.slot_kind == "AUTHORED_NARRATION"
    assert requirement.text is None
    assert requirement.status == MATERIALIZATION_REQUIRED
    assert requirement.desired_zone == OverlayZone.CENTER.value


def test_caption_collision_moves_zone_deterministically() -> None:
    slots = [_authored_text_slot(1, "ORIGINAL_VALUE", delivery_intent="ON_SCREEN_TEXT")]
    blocks = [_block(1, "ORIGINAL_VALUE")]

    first = _requirements(
        slots,
        blocks,
        scene_caption_zones={0: "LOWER"},
        block_scene_indexes={1: 0},
    )
    second = _requirements(
        slots,
        blocks,
        scene_caption_zones={0: "LOWER"},
        block_scene_indexes={1: 0},
    )

    assert len(first) == 1
    requirement = first[0]
    assert requirement.desired_zone != OverlayZone.LOWER_THIRD.value
    assert "ZONE_MOVED_CAPTION_CONFLICT" in requirement.placement_reason
    assert CONSTRAINT_CAPTION_ZONE in requirement.collision_constraints
    assert [item.as_dict() for item in first] == [item.as_dict() for item in second]


def test_all_zones_conflict_stays_materialization_required() -> None:
    giant = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    requirement = _requirements(
        [_authored_text_slot(1, "ORIGINAL_VALUE", delivery_intent="ON_SCREEN_TEXT")],
        [_block(1, "ORIGINAL_VALUE")],
        protected_regions=[giant],
    )[0]

    assert requirement.text is None
    assert requirement.status == MATERIALIZATION_REQUIRED
    assert requirement.desired_zone == OverlayZone.LOWER_THIRD.value
    assert "ZONE_CONFLICT_UNRESOLVED" in requirement.placement_reason
    assert CONSTRAINT_NO_FREE_ZONE in requirement.collision_constraints
    assert CONSTRAINT_PROTECTED_FACE in requirement.collision_constraints
    assert CONSTRAINT_SAFE_ZONES in requirement.collision_constraints


def test_hero_overlay_cannot_cover_protected_face() -> None:
    lower_face = {"x": 0.10, "y": 0.68, "w": 0.20, "h": 0.05}
    requirement = _requirements(
        [_authored_text_slot(1, "ORIGINAL_VALUE", delivery_intent="ON_SCREEN_TEXT")],
        [_block(1, "ORIGINAL_VALUE")],
        protected_regions=[lower_face],
        hero_block_index=1,
    )[0]

    assert requirement.desired_zone != OverlayZone.LOWER_THIRD.value
    assert "ZONE_MOVED_PROTECTED_FACE" in requirement.placement_reason
    assert "HERO_BLOCK" in requirement.placement_reason
    assert CONSTRAINT_PROTECTED_FACE in requirement.collision_constraints

    region = ProtectedRegion(x=0.10, y=0.68, w=0.20, h=0.05)
    resolution = resolve_zone_conflicts(
        OverlayZone.LOWER_THIRD,
        protected_regions=[region],
        safe_zone=safe_zone_for(None),
    )
    assert resolution.zone == OverlayZone.TOP_HOOK
    assert resolution.unresolved is False


def test_repeated_calls_produce_identical_as_dict_output() -> None:
    slots = [
        _authored_text_slot(1, "TEXTUAL_ANNOTATION", draft_line="DRAFT_LINE_ABC"),
        _authored_text_slot(4, "ORIGINAL_VALUE", delivery_intent="ON_SCREEN_TEXT"),
        _narration_slot(["CONTEXT"]),
    ]
    blocks = [
        _block(1, "TEXTUAL_ANNOTATION", start=0.0, end=2.0),
        _block(4, "ORIGINAL_VALUE", start=2.0, end=6.0, interrupts_source=True),
    ]
    kwargs: dict[str, object] = {
        "scene_caption_zones": {0: "LOWER"},
        "block_scene_indexes": {4: 0},
        "protected_regions": [{"x": 0.4, "y": 0.70, "w": 0.2, "h": 0.05}],
    }

    first = [item.as_dict() for item in _requirements(slots, blocks, **kwargs)]
    second = [item.as_dict() for item in _requirements(slots, blocks, **kwargs)]

    assert first == second
    assert len(first) == 3
    assert all(item["text"] is None for item in first)
    assert all(item["status"] == MATERIALIZATION_REQUIRED for item in first)
