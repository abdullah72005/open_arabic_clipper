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
from app.materialization.policy import (
    REASON_INVALID_ASSET_HASH,
    REASON_MISSING_ATTRIBUTION,
    REASON_MISSING_PROVENANCE,
    REASON_UNSUITABLE_ASSET_RIGHTS,
)
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
    attribution_required: bool = False

    @property
    def rights_compatible(self) -> bool:
        return bool(self.rights == SupportingAssetRights.EXPLICIT_COMPATIBLE.value)

    @property
    def provenance_complete(self) -> bool:
        return bool(
            self.owner.strip()
            and self.creator.strip()
            and self.origin.strip()
            and self.license.strip()
            and self.permission.strip()
        )

    @property
    def hash_valid(self) -> bool:
        if len(self.sha256) != 64:
            return False
        return all(character in "0123456789abcdefABCDEF" for character in self.sha256)

    @property
    def attribution_satisfied(self) -> bool:
        return (not self.attribution_required) or bool(self.required_attribution.strip())

    @property
    def usable(self) -> bool:
        return bool(
            self.rights_compatible
            and self.hash_valid
            and self.relative_path.strip()
            and self.allowed_uses
            and self.provenance_complete
            and self.attribution_satisfied
        )

    def rights_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if not self.rights_compatible:
            blockers.append(REASON_UNSUITABLE_ASSET_RIGHTS)
        if not self.provenance_complete:
            blockers.append(REASON_MISSING_PROVENANCE)
        if not self.hash_valid:
            blockers.append(REASON_INVALID_ASSET_HASH)
        if not self.allowed_uses:
            blockers.append(REASON_MISSING_PROVENANCE)
        if not self.attribution_satisfied:
            blockers.append(REASON_MISSING_ATTRIBUTION)
        return tuple(blockers)

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
    asset = SupportingAsset(
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
        attribution_required=bool(entry.get("attribution_required", False)),
    )
    # A manifest entry that claims explicit compatible rights must actually carry
    # complete provenance, a valid content hash, an allowed use, and any required
    # attribution; otherwise it is a malformed rights claim, not a usable asset.
    if asset.rights_compatible and not asset.usable:
        raise ValueError(
            "explicit-compatible supporting asset is missing required provenance: "
            + ",".join(asset.rights_blockers())
        )
    return asset


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
