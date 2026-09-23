"""Deterministic Stage 5.1 overlay placement requirements.

This module is pure and provider-free: no network, no provider call, no model
loading, no audio decoding, no rendering. It turns the executable Stage 5.0
materialization slots plus the ordered contract blocks into immutable
:class:`~app.composition.types.OverlayRequirement` value objects.

Hard guarantees:

- Stage 5.1 never invents, renders, or carries publication text. Every returned
  requirement has ``text = None`` and ``status = MATERIALIZATION_REQUIRED``.
- An ``authoring_reference`` draft line carried by a slot payload is preserved
  only as explicitly non-authoritative metadata (``draft_only=True``,
  ``authoritative=False``). It is never copied into ``text``, never treated as a
  hook, and never placed anywhere a later stage could mistake it for copy.
- ``desired_zone`` is derived from the *real closed* Stage 4.1/Stage 5.0
  vocabulary (``PlanBlockType``, ``DeliveryIntent``, ``SourceExcerptRole``,
  ``NarrationPurpose``) and ``interrupts_source``. Stage 4.1 ``placement`` and
  ``purpose`` are free-form provider strings and are therefore deliberately
  *not* used as closed inputs; only the real enums and the boolean are mapped.
- Timing is carried as a block-relative ``timeline_hint``. Global offsets are
  never frozen.
- Every requirement carries collision constraints for Stage 5.2/6; a zone that
  conflicts with a caption band or a protected face region is moved
  deterministically, or the requirement stays materialization-required with an
  explicit unresolved placement constraint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.composition.policy import (
    CaptionPlacementZone,
    OverlayStatus,
    OverlayZone,
    SafeZoneProfile,
    safe_zone_for,
)
from app.composition.types import OverlayRequirement

# --- Closed vocabulary -------------------------------------------------------

BLOCK_TYPE_TEXTUAL_ANNOTATION = "TEXTUAL_ANNOTATION"
BLOCK_TYPE_ORIGINAL_VALUE = "ORIGINAL_VALUE"
BLOCK_TYPE_NARRATION_REQUIREMENT = "NARRATION_REQUIREMENT"

SLOT_KIND_AUTHORED_TEXT = "AUTHORED_TEXT"
SLOT_KIND_AUTHORED_NARRATION = "AUTHORED_NARRATION"

DELIVERY_ON_SCREEN_TEXT = "ON_SCREEN_TEXT"
DELIVERY_NARRATION = "NARRATION"

PURPOSE_HOOK = "HOOK"
PURPOSE_CONTEXT = "CONTEXT"
PURPOSE_EXPLANATION = "EXPLANATION"
PURPOSE_TAKEAWAY = "TAKEAWAY"
PURPOSE_ANALYSIS = "ANALYSIS"
PURPOSE_COUNTERPOINT = "COUNTERPOINT"

ROLE_HOOK = "HOOK"

# --- Collision constraints ---------------------------------------------------

CONSTRAINT_PROTECTED_FACE = "MUST_NOT_COVER_PROTECTED_FACE"
CONSTRAINT_SAFE_ZONES = "MUST_RESPECT_SAFE_ZONES"
CONSTRAINT_CAPTION_ZONE = "MUST_NOT_OVERLAP_SCENE_CAPTION_ZONE"
CONSTRAINT_NO_FREE_ZONE = "ZONE_CONFLICT_UNRESOLVED_NO_FREE_ZONE"

OVERLAY_BASE_CONSTRAINTS: tuple[str, ...] = (
    CONSTRAINT_PROTECTED_FACE,
    CONSTRAINT_SAFE_ZONES,
    CONSTRAINT_CAPTION_ZONE,
)

# --- Deterministic zone bands ------------------------------------------------

#: Fractional vertical band occupied by the top hook zone below the safe inset.
TOP_HOOK_BAND_HEIGHT = 0.15
#: Fractional vertical band occupied by the lower third above the safe inset.
LOWER_THIRD_BAND_HEIGHT = 0.20
#: Fixed fractional vertical bands for the upper third and center.
UPPER_THIRD_BAND = (0.30, 0.50)
CENTER_BAND = (0.42, 0.62)

_EPSILON = 1e-9

#: Stable tie-break order; also the priority used when relocating a zone.
_ZONE_ORDER: tuple[OverlayZone, ...] = (
    OverlayZone.TOP_HOOK,
    OverlayZone.UPPER_THIRD,
    OverlayZone.LOWER_THIRD,
    OverlayZone.CENTER,
)

#: Which caption band each overlay zone competes with (``None`` never conflicts).
_ZONE_CAPTION_CONFLICT: Mapping[OverlayZone, str | None] = {
    OverlayZone.TOP_HOOK: CaptionPlacementZone.UPPER.value,
    OverlayZone.UPPER_THIRD: CaptionPlacementZone.UPPER.value,
    OverlayZone.LOWER_THIRD: CaptionPlacementZone.LOWER.value,
    OverlayZone.CENTER: None,
}

# --- Placement reason codes --------------------------------------------------

REASON_SOURCE_ROLE_HOOK = "SOURCE_ROLE_HOOK_TO_TOP_HOOK"
REASON_NARRATION_PURPOSE_HOOK = "NARRATION_PURPOSE_HOOK_TO_TOP_HOOK"
REASON_TEXTUAL_ANNOTATION = "TEXTUAL_ANNOTATION_TO_UPPER_THIRD"
REASON_NARRATION_PURPOSE_CONTEXT = "NARRATION_PURPOSE_CONTEXT_TO_UPPER_THIRD"
REASON_NARRATION_PURPOSE_TAKEAWAY = "NARRATION_PURPOSE_TAKEAWAY_TO_LOWER_THIRD"
REASON_ORIGINAL_VALUE_NARRATION = "ORIGINAL_VALUE_NARRATION_TO_CENTER"
REASON_ORIGINAL_VALUE_ON_SCREEN = "ORIGINAL_VALUE_ON_SCREEN_TEXT_TO_LOWER_THIRD"
REASON_ORIGINAL_VALUE_INTERRUPTS = "ORIGINAL_VALUE_INTERRUPTS_SOURCE_TO_CENTER"
REASON_ORIGINAL_VALUE_DEFAULT = "ORIGINAL_VALUE_DEFAULT_TO_LOWER_THIRD"
REASON_NARRATION_DEFAULT = "NARRATION_REQUIREMENT_DEFAULT_TO_CENTER"
REASON_FALLBACK = "BLOCK_TYPE_FALLBACK_TO_CENTER"

RESOLUTION_DESIRED_AVAILABLE = "DESIRED_ZONE_AVAILABLE"
RESOLUTION_DESIRED_CAPTION_UNRESOLVED = "DESIRED_ZONE_CAPTION_UNRESOLVED"
RESOLUTION_MOVED_PROTECTED_FACE = "ZONE_MOVED_PROTECTED_FACE"
RESOLUTION_MOVED_SAFE_ZONE = "ZONE_MOVED_SAFE_ZONE"
RESOLUTION_MOVED_CAPTION = "ZONE_MOVED_CAPTION_CONFLICT"
RESOLUTION_MOVED_ALTERNATE = "ZONE_MOVED_ALTERNATE"
RESOLUTION_UNRESOLVED = "ZONE_CONFLICT_UNRESOLVED"


@dataclass(frozen=True)
class ProtectedRegion:
    """One anonymous protected face/eyes region in display-normalized coordinates.

    Coordinates use the :class:`app.composition.types.FaceDetection` convention:
    origin top-left, ``x`` right, ``y`` down, all normalized by display size.
    ``scene_index`` is optional; ``None`` means the region applies globally.
    """

    x: float
    y: float
    w: float
    h: float
    scene_index: int | None = None
    start: float = 0.0
    end: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "scene_index": self.scene_index,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "w": round(self.w, 6),
            "h": round(self.h, 6),
        }


@dataclass(frozen=True)
class ZoneResolution:
    """Deterministic outcome of resolving one desired overlay zone."""

    zone: OverlayZone
    reason_code: str
    unresolved: bool
    constraints: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "zone": self.zone.value,
            "reason_code": self.reason_code,
            "unresolved": self.unresolved,
            "constraints": list(self.constraints),
        }


# --- Generic accessors -------------------------------------------------------


def _get(source: object, key: str, default: object = None) -> object:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _coerce_zone(value: object) -> OverlayZone:
    if isinstance(value, OverlayZone):
        return value
    if isinstance(value, str):
        try:
            return OverlayZone(value)
        except ValueError:
            pass
    return OverlayZone.CENTER


def _normalize_caption_zone(value: object) -> str | None:
    if isinstance(value, CaptionPlacementZone):
        return value.value
    if isinstance(value, str):
        try:
            return CaptionPlacementZone(value).value
        except ValueError:
            return None
    return None


def _coerce_region(value: object) -> ProtectedRegion:
    if isinstance(value, ProtectedRegion):
        return value
    return ProtectedRegion(
        x=_as_float(_get(value, "x")),
        y=_as_float(_get(value, "y")),
        w=_as_float(_get(value, "w")),
        h=_as_float(_get(value, "h")),
        scene_index=_as_optional_int(_get(value, "scene_index")),
        start=_as_float(_get(value, "start")),
        end=_as_float(_get(value, "end")),
    )


# --- Zone geometry -----------------------------------------------------------


def _zone_band(zone: OverlayZone, safe: SafeZoneProfile) -> tuple[float, float, float, float]:
    """Return the normalized ``(left, top, right, bottom)`` band for a zone."""

    left = safe.left
    right = 1.0 - safe.right
    if zone is OverlayZone.TOP_HOOK:
        top = safe.top
        bottom = min(1.0 - safe.bottom, top + TOP_HOOK_BAND_HEIGHT)
    elif zone is OverlayZone.UPPER_THIRD:
        top, bottom = UPPER_THIRD_BAND
    elif zone is OverlayZone.LOWER_THIRD:
        bottom = 1.0 - safe.bottom
        top = max(safe.top, bottom - LOWER_THIRD_BAND_HEIGHT)
    else:
        top, bottom = CENTER_BAND
    return (left, top, right, bottom)


def _zone_covers_any_region(
    zone: OverlayZone, regions: Sequence[ProtectedRegion], safe: SafeZoneProfile
) -> bool:
    band_left, band_top, band_right, band_bottom = _zone_band(zone, safe)
    for region in regions:
        region_left = region.x
        region_top = region.y
        region_right = region.x + region.w
        region_bottom = region.y + region.h
        overlap_w = min(band_right, region_right) - max(band_left, region_left)
        overlap_h = min(band_bottom, region_bottom) - max(band_top, region_top)
        if overlap_w > _EPSILON and overlap_h > _EPSILON:
            return True
    return False


def _zone_within_safe(zone: OverlayZone, safe: SafeZoneProfile) -> bool:
    left, top, right, bottom = _zone_band(zone, safe)
    return (
        left >= safe.left - _EPSILON
        and right <= 1.0 - safe.right + _EPSILON
        and top >= safe.top - _EPSILON
        and bottom <= 1.0 - safe.bottom + _EPSILON
    )


def resolve_zone_conflicts(
    desired_zone: OverlayZone | str,
    *,
    caption_zone: str | None = None,
    protected_regions: Sequence[ProtectedRegion] = (),
    safe_zone: SafeZoneProfile | None = None,
) -> ZoneResolution:
    """Resolve a desired overlay zone against caption, face, and safe-zone rules.

    The deterministic priority is face/eyes protection > safe zones > caption
    stability > the originally desired zone. A zone is viable only when it
    covers no protected region and stays inside the safe insets. Among viable
    zones, caption-free zones are preferred, then the desired zone, then the
    fixed :data:`_ZONE_ORDER`. When no zone is viable the requirement cannot be
    placed yet and the desired zone is kept with an explicit unresolved
    constraint.
    """

    safe = safe_zone if safe_zone is not None else safe_zone_for(None)
    desired = _coerce_zone(desired_zone)
    caption = _normalize_caption_zone(caption_zone)

    face_free: dict[OverlayZone, bool] = {
        zone: not _zone_covers_any_region(zone, protected_regions, safe) for zone in _ZONE_ORDER
    }
    safe_ok: dict[OverlayZone, bool] = {zone: _zone_within_safe(zone, safe) for zone in _ZONE_ORDER}
    caption_safe: dict[OverlayZone, bool] = {
        zone: caption is None or _ZONE_CAPTION_CONFLICT[zone] != caption for zone in _ZONE_ORDER
    }

    viable = [zone for zone in _ZONE_ORDER if face_free[zone] and safe_ok[zone]]
    if not viable:
        return ZoneResolution(
            zone=desired,
            reason_code=RESOLUTION_UNRESOLVED,
            unresolved=True,
            constraints=(CONSTRAINT_NO_FREE_ZONE,),
        )

    caption_viable = [zone for zone in viable if caption_safe[zone]]
    pool = caption_viable if caption_viable else viable

    if desired in pool:
        if caption_safe[desired]:
            return ZoneResolution(desired, RESOLUTION_DESIRED_AVAILABLE, False)
        return ZoneResolution(
            desired,
            RESOLUTION_DESIRED_CAPTION_UNRESOLVED,
            False,
            constraints=(CONSTRAINT_CAPTION_ZONE,),
        )

    chosen = pool[0]
    if not face_free[desired]:
        reason = RESOLUTION_MOVED_PROTECTED_FACE
    elif not safe_ok[desired]:
        reason = RESOLUTION_MOVED_SAFE_ZONE
    elif not caption_safe[desired]:
        reason = RESOLUTION_MOVED_CAPTION
    else:
        reason = RESOLUTION_MOVED_ALTERNATE
    return ZoneResolution(chosen, reason, False)


# --- Zone derivation ---------------------------------------------------------


def _contracts_overlay(slot_kind: str, block_type: str, delivery_intent: str) -> bool:
    """Whether a materialization slot produces an overlay placement requirement.

    Only contract-authored slots qualify: ``AUTHORED_TEXT`` for on-screen
    ``TEXTUAL_ANNOTATION``/non-narration ``ORIGINAL_VALUE`` blocks, and
    ``AUTHORED_NARRATION`` (including the contract-level narration slot).
    ``SOURCE_MEDIA``, ``TRANSITION``, and ``VERIFICATION_EVIDENCE`` slots are
    ignored.
    """

    if slot_kind == SLOT_KIND_AUTHORED_NARRATION:
        return True
    if slot_kind != SLOT_KIND_AUTHORED_TEXT:
        return False
    if block_type == BLOCK_TYPE_TEXTUAL_ANNOTATION:
        return True
    if block_type == BLOCK_TYPE_ORIGINAL_VALUE:
        return delivery_intent != DELIVERY_NARRATION
    return False


def _derive_zone(
    *,
    block_type: str,
    delivery_intent: str,
    interrupts_source: bool,
    source_role: str,
    narration_purposes: tuple[str, ...],
) -> tuple[OverlayZone, str]:
    if source_role == ROLE_HOOK:
        return OverlayZone.TOP_HOOK, REASON_SOURCE_ROLE_HOOK
    if PURPOSE_HOOK in narration_purposes:
        return OverlayZone.TOP_HOOK, REASON_NARRATION_PURPOSE_HOOK
    if block_type == BLOCK_TYPE_TEXTUAL_ANNOTATION:
        return OverlayZone.UPPER_THIRD, REASON_TEXTUAL_ANNOTATION
    if PURPOSE_CONTEXT in narration_purposes or PURPOSE_EXPLANATION in narration_purposes:
        return OverlayZone.UPPER_THIRD, REASON_NARRATION_PURPOSE_CONTEXT
    if (
        PURPOSE_TAKEAWAY in narration_purposes
        or PURPOSE_ANALYSIS in narration_purposes
        or PURPOSE_COUNTERPOINT in narration_purposes
    ):
        return OverlayZone.LOWER_THIRD, REASON_NARRATION_PURPOSE_TAKEAWAY
    if block_type == BLOCK_TYPE_ORIGINAL_VALUE:
        if delivery_intent == DELIVERY_NARRATION:
            return OverlayZone.CENTER, REASON_ORIGINAL_VALUE_NARRATION
        if delivery_intent == DELIVERY_ON_SCREEN_TEXT:
            return OverlayZone.LOWER_THIRD, REASON_ORIGINAL_VALUE_ON_SCREEN
        if interrupts_source:
            return OverlayZone.CENTER, REASON_ORIGINAL_VALUE_INTERRUPTS
        return OverlayZone.LOWER_THIRD, REASON_ORIGINAL_VALUE_DEFAULT
    if block_type == BLOCK_TYPE_NARRATION_REQUIREMENT:
        return OverlayZone.CENTER, REASON_NARRATION_DEFAULT
    return OverlayZone.CENTER, REASON_FALLBACK


def _authoring_reference(payload: Mapping[str, object]) -> dict[str, object] | None:
    """Return a sanitized, always non-authoritative authoring reference."""

    draft: object = None
    raw = payload.get("authoring_reference")
    if isinstance(raw, Mapping):
        draft = raw.get("draft_line")
    if not isinstance(draft, str) or not draft.strip():
        draft = payload.get("draft_line")
    if not isinstance(draft, str) or not draft.strip():
        return None
    return {"draft_line": draft, "draft_only": True, "authoritative": False}


def _timeline_hint(block: Mapping[str, object] | None) -> dict[str, object] | None:
    if block is None:
        return None
    timeline = _get(block, "timeline")
    if isinstance(timeline, Mapping):
        start = timeline.get("start")
        end = timeline.get("end")
        authoritative = timeline.get("authoritative")
    else:
        start = _get(block, "timeline_start")
        end = _get(block, "timeline_end")
        authoritative = _get(block, "timeline_authoritative")
    if start is None and end is None:
        return None
    return {
        "start": _as_float(start),
        "end": _as_float(end),
        "authoritative": _as_bool(authoritative),
    }


def _index_blocks(blocks: Sequence[Mapping[str, object]]) -> dict[int, Mapping[str, object]]:
    indexed: dict[int, Mapping[str, object]] = {}
    for block in blocks:
        block_index = _as_optional_int(_get(block, "block_index"))
        if block_index is not None:
            indexed[block_index] = block
    return indexed


def _regions_for_scene(
    regions: Sequence[ProtectedRegion], scene_index: int | None
) -> tuple[ProtectedRegion, ...]:
    if scene_index is None:
        return tuple(regions)
    return tuple(
        region
        for region in regions
        if region.scene_index is None or region.scene_index == scene_index
    )


# --- Public builder ----------------------------------------------------------


def build_overlay_requirements(
    materialization_slots: Sequence[Mapping[str, object]],
    blocks: Sequence[Mapping[str, object]] = (),
    *,
    safe_zone: SafeZoneProfile | None = None,
    scene_caption_zones: Mapping[int, str] | None = None,
    protected_regions: Sequence[Mapping[str, object] | ProtectedRegion] = (),
    block_scene_indexes: Mapping[int, int] | None = None,
    hero_block_index: int | None = None,
) -> tuple[OverlayRequirement, ...]:
    """Build immutable overlay placement requirements from contract slots.

    ``materialization_slots`` are the Stage 5.0 execution slots; ``blocks`` are
    the ordered contract blocks carrying ``block_type``, ``purpose``,
    ``placement``, ``interrupts_source``, and a block-relative timeline hint.
    ``scene_caption_zones`` maps a scene index to its resolved caption band
    (``LOWER``/``UPPER``) and ``protected_regions`` carries anonymous face/eyes
    boxes so a zone never covers a protected face. Only currently-unmaterialized
    authored text/narration slots produce a requirement; every returned
    requirement keeps ``text=None`` and ``MATERIALIZATION_REQUIRED``.
    """

    resolved_safe = safe_zone if safe_zone is not None else safe_zone_for(None)
    caption_zones = dict(scene_caption_zones or {})
    block_by_index = _index_blocks(blocks)
    regions = tuple(_coerce_region(region) for region in protected_regions)
    scene_map = {int(key): int(value) for key, value in (block_scene_indexes or {}).items()}

    entries: list[tuple[int, int, str, OverlayRequirement]] = []
    for order, slot in enumerate(materialization_slots):
        slot_kind = _as_str(_get(slot, "slot_kind"))
        block_type = _as_str(_get(slot, "block_type"))
        payload = _as_mapping(_get(slot, "payload"))
        block_index = _as_optional_int(_get(slot, "block_index"))
        block = block_by_index.get(block_index) if block_index is not None else None
        delivery_intent = _as_str(payload.get("delivery_intent")) or _as_str(
            _get(block, "delivery_intent")
        )
        if not _contracts_overlay(slot_kind, block_type, delivery_intent):
            continue

        interrupts_source = _as_bool(_get(block, "interrupts_source"))
        source_role = _as_str(_get(block, "source_role"))
        if not source_role:
            source_binding = _as_mapping(_get(block, "source_binding"))
            source_role = _as_str(source_binding.get("source_role"))
        narration_purposes = _as_str_tuple(payload.get("purposes"))

        desired, zone_reason = _derive_zone(
            block_type=block_type,
            delivery_intent=delivery_intent,
            interrupts_source=interrupts_source,
            source_role=source_role,
            narration_purposes=narration_purposes,
        )

        scene_index = _as_optional_int(_get(block, "scene_index"))
        if block_index is not None and block_index in scene_map:
            scene_index = scene_map[block_index]
        caption_zone = caption_zones.get(scene_index) if scene_index is not None else None
        scene_regions = _regions_for_scene(regions, scene_index)

        resolution = resolve_zone_conflicts(
            desired,
            caption_zone=caption_zone,
            protected_regions=scene_regions,
            safe_zone=resolved_safe,
        )

        placement_reason = f"{zone_reason};{resolution.reason_code}"
        if hero_block_index is not None and block_index == hero_block_index:
            placement_reason = f"{placement_reason};HERO_BLOCK"

        constraints = OVERLAY_BASE_CONSTRAINTS + resolution.constraints
        requirement_id = f"overlay-{_as_str(_get(slot, 'slot_id')) or order}"
        requirement = OverlayRequirement(
            requirement_id=requirement_id,
            block_index=block_index,
            block_type=block_type,
            slot_kind=slot_kind,
            status=OverlayStatus.MATERIALIZATION_REQUIRED.value,
            desired_zone=resolution.zone.value,
            text=None,
            authoring_reference=_authoring_reference(payload),
            timeline_hint=_timeline_hint(block),
            collision_constraints=constraints,
            placement_reason=placement_reason,
        )
        sort_block = 0 if block_index is not None else 1
        entries.append(
            (sort_block, block_index if block_index is not None else 0, requirement_id, requirement)
        )

    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(requirement for _, _, _, requirement in entries)


__all__ = [
    "CENTER_BAND",
    "CONSTRAINT_CAPTION_ZONE",
    "CONSTRAINT_NO_FREE_ZONE",
    "CONSTRAINT_PROTECTED_FACE",
    "CONSTRAINT_SAFE_ZONES",
    "LOWER_THIRD_BAND_HEIGHT",
    "OVERLAY_BASE_CONSTRAINTS",
    "ProtectedRegion",
    "TOP_HOOK_BAND_HEIGHT",
    "UPPER_THIRD_BAND",
    "ZoneResolution",
    "build_overlay_requirements",
    "resolve_zone_conflicts",
]
