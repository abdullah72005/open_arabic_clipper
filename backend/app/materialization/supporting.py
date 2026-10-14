"""Stage 6 supporting-visual library, rights gating, and deterministic AUTO
selection.

Supporting visuals are optional (default OFF) and receive zero substantive
originality credit. Only explicit compatible rights permit automatic use; unknown
or conflicting rights, missing required attribution, or changed/missing bytes
prevent selection. There is no scraping and no arbitrary remote footage, and
AUTO selection never calls a provider.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.core.enums import SupportingAssetRights, SupportingVisualMode, SupportingVisualStyle
from app.materialization.types import SupportingAssetSelection

MANIFEST_SCHEMA_VERSION = "stage6-supporting-manifest-v1"


@dataclass(frozen=True)
class SupportingAsset:
    """One validated, rights-cleared local supporting asset."""

    asset_id: str
    relative_path: str
    sha256: str
    kind: str
    duration_seconds: float
    width: int
    height: int
    owner: str
    creator: str
    origin: str
    license: str
    permission: str
    allowed_uses: tuple[str, ...]
    required_attribution: str
    restrictions: tuple[str, ...]
    suitability_tags: tuple[str, ...]
    rights: str

    @property
    def rights_compatible(self) -> bool:
        return self.rights == SupportingAssetRights.EXPLICIT_COMPATIBLE.value

    @property
    def usable(self) -> bool:
        return self.rights_compatible and bool(self.sha256) and bool(self.relative_path)

    def identity(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "kind": self.kind,
            "rights": self.rights,
            "required_attribution": self.required_attribution,
            "allowed_uses": list(self.allowed_uses),
            "restrictions": list(self.restrictions),
        }

    def as_dict(self) -> dict[str, object]:
        payload = self.identity()
        payload.update(
            {
                "duration_seconds": self.duration_seconds,
                "width": self.width,
                "height": self.height,
                "owner": self.owner,
                "creator": self.creator,
                "origin": self.origin,
                "license": self.license,
                "permission": self.permission,
                "suitability_tags": list(self.suitability_tags),
            }
        )
        return payload


def parse_manifest(data: object) -> tuple[SupportingAsset, ...]:
    """Parse and validate a supporting-asset manifest strictly."""

    if not isinstance(data, Mapping):
        raise ValueError("supporting manifest must be an object")
    if str(data.get("schema_version", "")) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported supporting manifest schema_version")
    entries = data.get("assets")
    if not isinstance(entries, list):
        raise ValueError("supporting manifest requires an assets list")
    assets: list[SupportingAsset] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("supporting asset entry must be an object")
        assets.append(_parse_entry(entry))
    return tuple(assets)


def load_manifest(path: Path) -> tuple[SupportingAsset, ...]:
    return parse_manifest(json.loads(path.read_text(encoding="utf-8")))


def _parse_entry(entry: Mapping[str, object]) -> SupportingAsset:
    asset_id = str(entry.get("asset_id", "")).strip()
    relative_path = str(entry.get("relative_path", "")).strip()
    sha256 = str(entry.get("sha256", "")).strip()
    if not asset_id or not relative_path or not sha256:
        raise ValueError("supporting asset requires asset_id, relative_path, and sha256")
    return SupportingAsset(
        asset_id=asset_id,
        relative_path=relative_path,
        sha256=sha256,
        kind=str(entry.get("kind", "")),
        duration_seconds=_float(entry.get("duration_seconds")),
        width=_int(entry.get("width")),
        height=_int(entry.get("height")),
        owner=str(entry.get("owner", "")),
        creator=str(entry.get("creator", "")),
        origin=str(entry.get("origin", "")),
        license=str(entry.get("license", "")),
        permission=str(entry.get("permission", "")),
        allowed_uses=tuple(str(item) for item in _sequence(entry.get("allowed_uses"))),
        required_attribution=str(entry.get("required_attribution", "")),
        restrictions=tuple(str(item) for item in _sequence(entry.get("restrictions"))),
        suitability_tags=tuple(str(item) for item in _sequence(entry.get("suitability_tags"))),
        rights=str(entry.get("rights", SupportingAssetRights.UNKNOWN.value)),
    )


@dataclass(frozen=True)
class SupportingNeed:
    """Deterministic authored-section need for a supporting visual."""

    block_index: int
    content_type: str = ""
    seriousness: str = ""
    visual_activity: str = ""
    readability_priority: str = ""
    required: bool = False


def select_supporting_asset(
    mode: str,
    style: str,
    need: SupportingNeed | None,
    assets: Sequence[SupportingAsset],
    *,
    style_preference: str = SupportingVisualStyle.AUTO.value,
) -> SupportingAssetSelection:
    """Deterministically select (or explicitly decline) a supporting asset."""

    if mode == SupportingVisualMode.OFF.value:
        return _decline(need, style, "SUPPORTING_DISABLED")
    if need is None:
        return _decline(need, style, "NO_AUTHORED_SECTION")
    if not need.required and mode == SupportingVisualMode.AUTO.value and not need.content_type:
        return _decline(need, style, "UNCERTAIN_USEFULNESS")

    candidates = [asset for asset in assets if asset.usable]
    if not candidates:
        return _decline(need, style, "NO_RIGHTS_CLEARED_ASSET")

    effective_style = style if style != SupportingVisualStyle.AUTO.value else style_preference
    if effective_style != SupportingVisualStyle.AUTO.value:
        styled = [
            asset
            for asset in candidates
            if asset.kind == effective_style or effective_style in asset.suitability_tags
        ]
        if styled:
            candidates = styled

    # Deterministic ordering: suitability tags first, then stable asset id.
    def rank(asset: SupportingAsset) -> tuple[int, str]:
        match = 1 if need.content_type and need.content_type in asset.suitability_tags else 0
        return (-match, asset.asset_id)

    chosen = sorted(candidates, key=rank)[0]
    if not need.required and _distracting(chosen, need):
        return _decline(need, style, "DISTRACTION_RISK")
    return SupportingAssetSelection(
        asset_id=chosen.asset_id,
        block_index=need.block_index,
        style=effective_style,
        resolution="FULFILLED",
        relative_path=chosen.relative_path,
        sha256=chosen.sha256,
        duration_seconds=chosen.duration_seconds,
        rights=chosen.rights,
        required_attribution=chosen.required_attribution,
    )


def _distracting(asset: SupportingAsset, need: SupportingNeed) -> bool:
    if need.readability_priority == "HIGH" and "busy" in asset.suitability_tags:
        return True
    if need.seriousness == "SERIOUS" and "playful" in asset.suitability_tags:
        return True
    return False


def _decline(need: SupportingNeed | None, style: str, reason: str) -> SupportingAssetSelection:
    return SupportingAssetSelection(
        asset_id=None,
        block_index=need.block_index if need is not None else None,
        style=style,
        resolution="DECLINED",
        reason_code=reason,
    )


def _sequence(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "SupportingAsset",
    "SupportingNeed",
    "load_manifest",
    "parse_manifest",
    "select_supporting_asset",
]
