"""Typed Stage 6 authored render payload and its structural validation.

These are renderer-owned value objects: Stage 6 produces them, Stage 5.2
executes them. No provider text ever enters an FFmpeg expression; every field is
a typed numeric or allow-listed instruction, and :func:`validate_authored_payload`
rejects malformed, non-finite, boolean-as-integer, duplicate, or ambiguous input
before compilation.

Authored timing is accumulated from RAW ``Fraction`` durations and quantized at
cumulative boundaries exactly once, so independent per-unit rounding can never
drift. Source spans remain in the authoritative source media time domain.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

AUTHORED_SCHEMA_VERSION = "stage6-authored-v1"
AUTHORED_COMPILER_VERSION = "stage6-authored-compiler-v1"
AUTHORED_TIMELINE_POLICY_VERSION = "stage6-authored-timeline-v1"
AUTHORED_QC_POLICY_VERSION = "stage6-authored-qc-v1"

UNIT_SOURCE = "SOURCE"
UNIT_CARD = "CARD"
ALLOWED_UNIT_KINDS = frozenset({UNIT_SOURCE, UNIT_CARD})

ASSET_ROLE_SOURCE_VIDEO = "SOURCE_VIDEO"
ASSET_ROLE_NARRATION = "NARRATION_AUDIO"
ASSET_ROLE_SUPPORTING_IMAGE = "SUPPORTING_IMAGE"
ASSET_ROLE_SUPPORTING_VIDEO = "SUPPORTING_VIDEO"
ALLOWED_ASSET_ROLES = frozenset(
    {
        ASSET_ROLE_SOURCE_VIDEO,
        ASSET_ROLE_NARRATION,
        ASSET_ROLE_SUPPORTING_IMAGE,
        ASSET_ROLE_SUPPORTING_VIDEO,
    }
)

MASTERING_MODE_LINEAR = "LINEAR_PREFERRED"
ALLOWED_MASTERING_MODES = frozenset({MASTERING_MODE_LINEAR})

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")

_REASON_SCHEMA = "AUTHORED_SCHEMA_INVALID"
_REASON_UNIT = "AUTHORED_UNIT_INVALID"
_REASON_ASSET = "AUTHORED_ASSET_INVALID"
_REASON_TRACK = "AUTHORED_TRACK_INVALID"
_REASON_MASTERING = "AUTHORED_MASTERING_INVALID"
_REASON_FRAMES = "AUTHORED_FRAME_TOTAL_INVALID"


class AuthoredPayloadError(ValueError):
    """An authored payload was malformed, ambiguous, or non-executable."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


# ---------------------------------------------------------------------------
# Strict primitive coercion (never silently coerce)
# ---------------------------------------------------------------------------


def _require_mapping(value: object, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuthoredPayloadError(reason, "expected object")
    return value


def _require_int(value: object, reason: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuthoredPayloadError(reason, "expected integer")
    if minimum is not None and value < minimum:
        raise AuthoredPayloadError(reason, "integer below minimum")
    return value


def _require_float(value: object, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuthoredPayloadError(reason, "expected number")
    result = float(value)
    if not math.isfinite(result):
        raise AuthoredPayloadError(reason, "non-finite number")
    return result


def _require_str(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuthoredPayloadError(reason, "expected non-empty string")
    return value


def _require_id(value: object, reason: str) -> str:
    text = _require_str(value, reason)
    if not _ID_RE.match(text):
        raise AuthoredPayloadError(reason, "malformed identifier")
    return text


def _require_hash(value: object, reason: str) -> str:
    text = _require_str(value, reason)
    if not _HASH_RE.match(text):
        raise AuthoredPayloadError(reason, "expected sha256 hex")
    return text


def _require_optional_int(value: object, reason: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    return _require_int(value, reason, minimum=minimum)


def _require_optional_float(value: object, reason: str) -> float | None:
    if value is None:
        return None
    return _require_float(value, reason)


def _reject_unknown_keys(data: Mapping[str, Any], allowed: frozenset[str], reason: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise AuthoredPayloadError(reason, f"unknown keys: {sorted(unknown)}")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaAssetRef:
    asset_id: str
    role: str
    storage_relative_path: str
    sha256: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int | None = None
    channels: int | None = None
    width: int | None = None
    height: int | None = None

    _KEYS = frozenset(
        {
            "asset_id",
            "role",
            "storage_relative_path",
            "sha256",
            "size_bytes",
            "duration_seconds",
            "sample_rate",
            "channels",
            "width",
            "height",
        }
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "role": self.role,
            "storage_relative_path": self.storage_relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "duration_seconds": round(self.duration_seconds, 6),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "width": self.width,
            "height": self.height,
        }

    @staticmethod
    def from_dict(data: object) -> MediaAssetRef:
        mapping = _require_mapping(data, _REASON_ASSET)
        _reject_unknown_keys(mapping, MediaAssetRef._KEYS, _REASON_ASSET)
        role = _require_str(mapping.get("role"), _REASON_ASSET)
        if role not in ALLOWED_ASSET_ROLES:
            raise AuthoredPayloadError(_REASON_ASSET, f"unknown asset role {role}")
        return MediaAssetRef(
            asset_id=_require_id(mapping.get("asset_id"), _REASON_ASSET),
            role=role,
            storage_relative_path=_require_str(mapping.get("storage_relative_path"), _REASON_ASSET),
            sha256=_require_hash(mapping.get("sha256"), _REASON_ASSET),
            size_bytes=_require_int(mapping.get("size_bytes"), _REASON_ASSET, minimum=1),
            duration_seconds=_require_float(mapping.get("duration_seconds"), _REASON_ASSET),
            sample_rate=_require_optional_int(mapping.get("sample_rate"), _REASON_ASSET, minimum=1),
            channels=_require_optional_int(mapping.get("channels"), _REASON_ASSET, minimum=1),
            width=_require_optional_int(mapping.get("width"), _REASON_ASSET, minimum=1),
            height=_require_optional_int(mapping.get("height"), _REASON_ASSET, minimum=1),
        )


@dataclass(frozen=True)
class RenderUnit:
    unit_id: str
    kind: str
    block_index: int
    start_frame: int
    end_frame: int
    occurrence_id: str | None = None
    background_asset_id: str | None = None
    background_start_seconds: float | None = None

    _KEYS = frozenset(
        {
            "unit_id",
            "kind",
            "block_index",
            "start_frame",
            "end_frame",
            "occurrence_id",
            "background_asset_id",
            "background_start_seconds",
        }
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind,
            "block_index": self.block_index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "occurrence_id": self.occurrence_id,
            "background_asset_id": self.background_asset_id,
            "background_start_seconds": self.background_start_seconds,
        }

    @staticmethod
    def from_dict(data: object) -> RenderUnit:
        mapping = _require_mapping(data, _REASON_UNIT)
        _reject_unknown_keys(mapping, RenderUnit._KEYS, _REASON_UNIT)
        kind = _require_str(mapping.get("kind"), _REASON_UNIT)
        if kind not in ALLOWED_UNIT_KINDS:
            raise AuthoredPayloadError(_REASON_UNIT, f"unknown unit kind {kind}")
        occurrence_id = mapping.get("occurrence_id")
        background_asset_id = mapping.get("background_asset_id")
        return RenderUnit(
            unit_id=_require_id(mapping.get("unit_id"), _REASON_UNIT),
            kind=kind,
            block_index=_require_int(mapping.get("block_index"), _REASON_UNIT),
            start_frame=_require_int(mapping.get("start_frame"), _REASON_UNIT, minimum=0),
            end_frame=_require_int(mapping.get("end_frame"), _REASON_UNIT, minimum=0),
            occurrence_id=None
            if occurrence_id is None
            else _require_id(occurrence_id, _REASON_UNIT),
            background_asset_id=None
            if background_asset_id is None
            else _require_id(background_asset_id, _REASON_UNIT),
            background_start_seconds=_require_optional_float(
                mapping.get("background_start_seconds"), _REASON_UNIT
            ),
        )


@dataclass(frozen=True)
class GainPoint:
    sample_index: int
    gain_db: float

    _KEYS = frozenset({"sample_index", "gain_db"})

    def as_dict(self) -> dict[str, object]:
        return {"sample_index": self.sample_index, "gain_db": round(self.gain_db, 6)}

    @staticmethod
    def from_dict(data: object) -> GainPoint:
        mapping = _require_mapping(data, _REASON_TRACK)
        _reject_unknown_keys(mapping, GainPoint._KEYS, _REASON_TRACK)
        return GainPoint(
            sample_index=_require_int(mapping.get("sample_index"), _REASON_TRACK, minimum=0),
            gain_db=_require_float(mapping.get("gain_db"), _REASON_TRACK),
        )


@dataclass(frozen=True)
class AudioTrackSpec:
    track_id: str
    asset_id: str
    source_start_seconds: float
    source_end_seconds: float
    output_start_sample: int
    output_end_sample: int
    gain_points: tuple[GainPoint, ...] = ()

    _KEYS = frozenset(
        {
            "track_id",
            "asset_id",
            "source_start_seconds",
            "source_end_seconds",
            "output_start_sample",
            "output_end_sample",
            "gain_points",
        }
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "asset_id": self.asset_id,
            "source_start_seconds": round(self.source_start_seconds, 6),
            "source_end_seconds": round(self.source_end_seconds, 6),
            "output_start_sample": self.output_start_sample,
            "output_end_sample": self.output_end_sample,
            "gain_points": [point.as_dict() for point in self.gain_points],
        }

    @staticmethod
    def from_dict(data: object) -> AudioTrackSpec:
        mapping = _require_mapping(data, _REASON_TRACK)
        _reject_unknown_keys(mapping, AudioTrackSpec._KEYS, _REASON_TRACK)
        raw_points = mapping.get("gain_points") or []
        if not isinstance(raw_points, (list, tuple)):
            raise AuthoredPayloadError(_REASON_TRACK, "gain_points must be a list")
        return AudioTrackSpec(
            track_id=_require_id(mapping.get("track_id"), _REASON_TRACK),
            asset_id=_require_id(mapping.get("asset_id"), _REASON_TRACK),
            source_start_seconds=_require_float(mapping.get("source_start_seconds"), _REASON_TRACK),
            source_end_seconds=_require_float(mapping.get("source_end_seconds"), _REASON_TRACK),
            output_start_sample=_require_int(
                mapping.get("output_start_sample"), _REASON_TRACK, minimum=0
            ),
            output_end_sample=_require_int(
                mapping.get("output_end_sample"), _REASON_TRACK, minimum=0
            ),
            gain_points=tuple(GainPoint.from_dict(point) for point in raw_points),
        )


@dataclass(frozen=True)
class MasteringSpec:
    target_lufs: float
    ceiling_dbtp: float
    render_guard_dbtp: float
    sample_rate: int
    channels: int
    mode: str = MASTERING_MODE_LINEAR
    repair_peak_limit_db: float = 0.0

    _KEYS = frozenset(
        {
            "target_lufs",
            "ceiling_dbtp",
            "render_guard_dbtp",
            "sample_rate",
            "channels",
            "mode",
            "repair_peak_limit_db",
        }
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "target_lufs": round(self.target_lufs, 6),
            "ceiling_dbtp": round(self.ceiling_dbtp, 6),
            "render_guard_dbtp": round(self.render_guard_dbtp, 6),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "mode": self.mode,
            "repair_peak_limit_db": round(self.repair_peak_limit_db, 6),
        }

    @staticmethod
    def from_dict(data: object) -> MasteringSpec:
        mapping = _require_mapping(data, _REASON_MASTERING)
        _reject_unknown_keys(mapping, MasteringSpec._KEYS, _REASON_MASTERING)
        mode = mapping.get("mode", MASTERING_MODE_LINEAR)
        if mode not in ALLOWED_MASTERING_MODES:
            raise AuthoredPayloadError(_REASON_MASTERING, f"unknown mastering mode {mode}")
        return MasteringSpec(
            target_lufs=_require_float(mapping.get("target_lufs"), _REASON_MASTERING),
            ceiling_dbtp=_require_float(mapping.get("ceiling_dbtp"), _REASON_MASTERING),
            render_guard_dbtp=_require_float(mapping.get("render_guard_dbtp"), _REASON_MASTERING),
            sample_rate=_require_int(mapping.get("sample_rate"), _REASON_MASTERING, minimum=1),
            channels=_require_int(mapping.get("channels"), _REASON_MASTERING, minimum=1),
            mode=str(mode),
            repair_peak_limit_db=_require_float(
                mapping.get("repair_peak_limit_db", 0.0), _REASON_MASTERING
            ),
        )


@dataclass(frozen=True)
class AssAssetRef:
    relative_path: str
    sha256: str
    policy_version: str

    _KEYS = frozenset({"relative_path", "sha256", "policy_version"})

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "policy_version": self.policy_version,
        }

    @staticmethod
    def from_dict(data: object) -> AssAssetRef:
        mapping = _require_mapping(data, _REASON_SCHEMA)
        _reject_unknown_keys(mapping, AssAssetRef._KEYS, _REASON_SCHEMA)
        return AssAssetRef(
            relative_path=_require_str(mapping.get("relative_path"), _REASON_SCHEMA),
            sha256=_require_hash(mapping.get("sha256"), _REASON_SCHEMA),
            policy_version=_require_str(mapping.get("policy_version"), _REASON_SCHEMA),
        )


@dataclass(frozen=True)
class AuthoredRenderPayload:
    schema_version: str
    materialization_id: str
    materialization_output_fingerprint: str
    repair_revision: int
    units: tuple[RenderUnit, ...]
    assets: tuple[MediaAssetRef, ...]
    tracks: tuple[AudioTrackSpec, ...]
    authored_ass: AssAssetRef
    mastering: MasteringSpec
    frame_count: int
    sample_count: int
    source_protection_evidence: Mapping[str, object] = field(default_factory=dict)

    _KEYS = frozenset(
        {
            "schema_version",
            "materialization_id",
            "materialization_output_fingerprint",
            "repair_revision",
            "units",
            "assets",
            "tracks",
            "authored_ass",
            "mastering",
            "frame_count",
            "sample_count",
            "source_protection_evidence",
        }
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "materialization_id": self.materialization_id,
            "materialization_output_fingerprint": self.materialization_output_fingerprint,
            "repair_revision": self.repair_revision,
            "units": [unit.as_dict() for unit in self.units],
            "assets": [asset.as_dict() for asset in self.assets],
            "tracks": [track.as_dict() for track in self.tracks],
            "authored_ass": self.authored_ass.as_dict(),
            "mastering": self.mastering.as_dict(),
            "frame_count": self.frame_count,
            "sample_count": self.sample_count,
            "source_protection_evidence": dict(self.source_protection_evidence),
        }

    @staticmethod
    def from_dict(data: object) -> AuthoredRenderPayload:
        mapping = _require_mapping(data, _REASON_SCHEMA)
        _reject_unknown_keys(mapping, AuthoredRenderPayload._KEYS, _REASON_SCHEMA)
        raw_units = mapping.get("units")
        raw_assets = mapping.get("assets")
        raw_tracks = mapping.get("tracks")
        if not isinstance(raw_units, (list, tuple)):
            raise AuthoredPayloadError(_REASON_SCHEMA, "units must be a list")
        if not isinstance(raw_assets, (list, tuple)):
            raise AuthoredPayloadError(_REASON_SCHEMA, "assets must be a list")
        if not isinstance(raw_tracks, (list, tuple)):
            raise AuthoredPayloadError(_REASON_SCHEMA, "tracks must be a list")
        evidence = mapping.get("source_protection_evidence")
        return AuthoredRenderPayload(
            schema_version=_require_str(mapping.get("schema_version"), _REASON_SCHEMA),
            materialization_id=_require_id(mapping.get("materialization_id"), _REASON_SCHEMA),
            materialization_output_fingerprint=_require_hash(
                mapping.get("materialization_output_fingerprint"), _REASON_SCHEMA
            ),
            repair_revision=_require_int(mapping.get("repair_revision"), _REASON_SCHEMA, minimum=0),
            units=tuple(RenderUnit.from_dict(unit) for unit in raw_units),
            assets=tuple(MediaAssetRef.from_dict(asset) for asset in raw_assets),
            tracks=tuple(AudioTrackSpec.from_dict(track) for track in raw_tracks),
            authored_ass=AssAssetRef.from_dict(mapping.get("authored_ass")),
            mastering=MasteringSpec.from_dict(mapping.get("mastering")),
            frame_count=_require_int(mapping.get("frame_count"), _REASON_SCHEMA, minimum=1),
            sample_count=_require_int(mapping.get("sample_count"), _REASON_SCHEMA, minimum=1),
            source_protection_evidence=dict(evidence) if isinstance(evidence, Mapping) else {},
        )


# ---------------------------------------------------------------------------
# Cumulative timing (Fraction, quantized once at boundaries)
# ---------------------------------------------------------------------------


def cumulative_frame_bounds(durations: Sequence[Fraction], frame_rate: Fraction) -> tuple[int, ...]:
    """Cumulative frame boundaries for ordered raw durations.

    Raw durations are accumulated with exact ``Fraction`` arithmetic and each
    cumulative boundary is quantized once, so independent per-unit rounding can
    never drift. Returns ``len(durations) + 1`` boundaries starting at zero.
    """

    if frame_rate <= 0:
        raise AuthoredPayloadError(_REASON_FRAMES, "frame rate must be positive")
    bounds = [0]
    total = Fraction(0)
    for duration in durations:
        if duration < 0:
            raise AuthoredPayloadError(_REASON_FRAMES, "negative duration")
        total += Fraction(duration)
        bounds.append(int(round(total * frame_rate)))
    return tuple(bounds)


def cumulative_sample_bounds(durations: Sequence[Fraction], sample_rate: int) -> tuple[int, ...]:
    """Cumulative sample boundaries for ordered raw durations at a sample rate."""

    if sample_rate <= 0:
        raise AuthoredPayloadError(_REASON_FRAMES, "sample rate must be positive")
    bounds = [0]
    total = Fraction(0)
    for duration in durations:
        if duration < 0:
            raise AuthoredPayloadError(_REASON_FRAMES, "negative duration")
        total += Fraction(duration)
        bounds.append(int(round(total * sample_rate)))
    return tuple(bounds)


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def validate_authored_payload(payload: AuthoredRenderPayload) -> None:
    """Validate a complete, contiguous, executable authored payload.

    Raises :class:`AuthoredPayloadError` with a reason code. Authored visual
    units must be contiguous and complete (from frame 0 to ``frame_count``), at
    least one SOURCE unit and the complete hero must be present, each SOURCE unit
    references exactly one occurrence and a matching frame interval, and audio
    tracks are bounded, non-overlapping, and gain-monotonic.
    """

    if payload.schema_version != AUTHORED_SCHEMA_VERSION:
        raise AuthoredPayloadError(_REASON_SCHEMA, "unsupported authored schema version")
    if not payload.units:
        raise AuthoredPayloadError(_REASON_UNIT, "at least one visual unit is required")

    assets = {asset.asset_id: asset for asset in payload.assets}
    if len(assets) != len(payload.assets):
        raise AuthoredPayloadError(_REASON_ASSET, "duplicate asset id")

    unit_ids: set[str] = set()
    occurrences: set[str] = set()
    source_units = 0
    cursor = 0
    for unit in payload.units:
        if unit.unit_id in unit_ids:
            raise AuthoredPayloadError(_REASON_UNIT, "duplicate unit id")
        unit_ids.add(unit.unit_id)
        if unit.start_frame != cursor:
            raise AuthoredPayloadError(_REASON_UNIT, "visual units are not contiguous")
        if unit.end_frame <= unit.start_frame:
            raise AuthoredPayloadError(_REASON_UNIT, "empty visual unit")
        cursor = unit.end_frame
        if unit.kind == UNIT_SOURCE:
            source_units += 1
            if unit.occurrence_id is None:
                raise AuthoredPayloadError(_REASON_UNIT, "SOURCE unit requires an occurrence")
            if unit.occurrence_id in occurrences:
                raise AuthoredPayloadError(_REASON_UNIT, "duplicate SOURCE occurrence")
            occurrences.add(unit.occurrence_id)
            if unit.background_asset_id is not None:
                raise AuthoredPayloadError(_REASON_UNIT, "SOURCE unit cannot have a background")
        elif unit.kind == UNIT_CARD:
            if unit.occurrence_id is not None:
                raise AuthoredPayloadError(_REASON_UNIT, "CARD unit cannot reference an occurrence")
            if unit.background_asset_id is not None:
                asset = assets.get(unit.background_asset_id)
                if asset is None:
                    raise AuthoredPayloadError(_REASON_UNIT, "unknown background asset")
                if asset.role not in {
                    ASSET_ROLE_SUPPORTING_IMAGE,
                    ASSET_ROLE_SUPPORTING_VIDEO,
                }:
                    raise AuthoredPayloadError(_REASON_UNIT, "background asset is not supporting")
    if cursor != payload.frame_count:
        raise AuthoredPayloadError(_REASON_FRAMES, "visual units do not cover frame_count")
    if source_units == 0:
        raise AuthoredPayloadError(_REASON_UNIT, "at least one SOURCE unit is required")

    _validate_tracks(payload, assets)

    mastering = payload.mastering
    if mastering.channels != 2:
        raise AuthoredPayloadError(_REASON_MASTERING, "mastering must be stereo")
    if mastering.sample_rate != 48_000:
        raise AuthoredPayloadError(_REASON_MASTERING, "mastering must be 48 kHz")
    if not (mastering.render_guard_dbtp <= mastering.ceiling_dbtp < 0):
        raise AuthoredPayloadError(_REASON_MASTERING, "guard must sit below the ceiling below 0")


def _validate_tracks(payload: AuthoredRenderPayload, assets: Mapping[str, MediaAssetRef]) -> None:
    track_ids: set[str] = set()
    intervals: list[tuple[int, int]] = []
    for track in payload.tracks:
        if track.track_id in track_ids:
            raise AuthoredPayloadError(_REASON_TRACK, "duplicate track id")
        track_ids.add(track.track_id)
        if track.asset_id not in assets:
            raise AuthoredPayloadError(_REASON_TRACK, "unknown track asset")
        if track.source_end_seconds <= track.source_start_seconds:
            raise AuthoredPayloadError(_REASON_TRACK, "inverted source interval")
        if track.output_end_sample <= track.output_start_sample:
            raise AuthoredPayloadError(_REASON_TRACK, "inverted output interval")
        if track.output_end_sample > payload.sample_count:
            raise AuthoredPayloadError(_REASON_TRACK, "track extends past the output")
        previous = track.output_start_sample - 1
        for point in track.gain_points:
            if point.sample_index <= previous:
                raise AuthoredPayloadError(_REASON_TRACK, "gain points must increase")
            if not (track.output_start_sample <= point.sample_index <= track.output_end_sample):
                raise AuthoredPayloadError(_REASON_TRACK, "gain point outside the track")
            previous = point.sample_index
        intervals.append((track.output_start_sample, track.output_end_sample))
    intervals.sort()
    for index in range(1, len(intervals)):
        if intervals[index][0] < intervals[index - 1][1]:
            raise AuthoredPayloadError(_REASON_TRACK, "overlapping tracks")


__all__ = [
    "ALLOWED_ASSET_ROLES",
    "ALLOWED_MASTERING_MODES",
    "ALLOWED_UNIT_KINDS",
    "ASSET_ROLE_NARRATION",
    "ASSET_ROLE_SOURCE_VIDEO",
    "ASSET_ROLE_SUPPORTING_IMAGE",
    "ASSET_ROLE_SUPPORTING_VIDEO",
    "AUTHORED_COMPILER_VERSION",
    "AUTHORED_QC_POLICY_VERSION",
    "AUTHORED_SCHEMA_VERSION",
    "AUTHORED_TIMELINE_POLICY_VERSION",
    "AssAssetRef",
    "AudioTrackSpec",
    "AuthoredPayloadError",
    "AuthoredRenderPayload",
    "GainPoint",
    "MASTERING_MODE_LINEAR",
    "MasteringSpec",
    "MediaAssetRef",
    "RenderUnit",
    "UNIT_CARD",
    "UNIT_SOURCE",
    "cumulative_frame_bounds",
    "cumulative_sample_bounds",
    "validate_authored_payload",
]
