"""Stage 6 frozen authored payload, contiguous units, and cumulative timing."""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from app.render.execution.authored import (
    ASSET_ROLE_NARRATION,
    ASSET_ROLE_SOURCE_VIDEO,
    AUTHORED_SCHEMA_VERSION,
    AuthoredPayloadError,
    AuthoredRenderPayload,
    cumulative_frame_bounds,
    cumulative_sample_bounds,
    validate_authored_payload,
)
from app.render.execution.timeline import TimelineError, validate_occurrence_order
from app.render.execution.types import SceneSpec, TimelineOccurrence

_HASH = "a" * 64


def _asset(asset_id: str, role: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "asset_id": asset_id,
        "role": role,
        "storage_relative_path": f"sources/{asset_id}.bin",
        "sha256": _HASH,
        "size_bytes": 1024,
        "duration_seconds": 7.0,
    }
    payload.update(extra)
    return payload


def _unit(
    unit_id: str,
    kind: str,
    block_index: int,
    start_frame: int,
    end_frame: int,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "kind": kind,
        "block_index": block_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "occurrence_id": None,
        "background_asset_id": None,
        "background_start_seconds": None,
    }
    payload.update(extra)
    return payload


def _valid_payload_dict() -> dict[str, object]:
    return {
        "schema_version": AUTHORED_SCHEMA_VERSION,
        "materialization_id": "mat-1",
        "materialization_output_fingerprint": _HASH,
        "repair_revision": 0,
        "units": [
            _unit("u0", "SOURCE", 0, 0, 90, occurrence_id="occ-0"),
            _unit("u1", "CARD", 1, 90, 120),
            _unit("u2", "SOURCE", 2, 120, 210, occurrence_id="occ-2"),
        ],
        "assets": [
            _asset("src", ASSET_ROLE_SOURCE_VIDEO, sample_rate=48000, channels=2),
            _asset("narr", ASSET_ROLE_NARRATION, sample_rate=24000, channels=1),
        ],
        "tracks": [
            {
                "track_id": "t-source",
                "asset_id": "src",
                "source_start_seconds": 21.0,
                "source_end_seconds": 28.0,
                "output_start_sample": 0,
                "output_end_sample": 336000,
                "gain_points": [
                    {"sample_index": 0, "gain_db": 0.0},
                    {"sample_index": 336000, "gain_db": 0.0},
                ],
            }
        ],
        "authored_ass": {
            "relative_path": "mat/authored.ass",
            "sha256": _HASH,
            "policy_version": "stage6-ass-v1",
        },
        "mastering": {
            "target_lufs": -16.0,
            "ceiling_dbtp": -1.5,
            "render_guard_dbtp": -1.8,
            "sample_rate": 48000,
            "channels": 2,
            "mode": "LINEAR_PREFERRED",
            "repair_peak_limit_db": 0.0,
        },
        "frame_count": 210,
        "sample_count": 336000,
        "source_protection_evidence": {"hero_preserved": True},
    }


def _payload(**overrides: object) -> AuthoredRenderPayload:
    data = _valid_payload_dict()
    data.update(overrides)
    return AuthoredRenderPayload.from_dict(data)


def test_valid_source_card_source_payload_round_trips() -> None:
    payload = _payload()
    validate_authored_payload(payload)
    reparsed = AuthoredRenderPayload.from_dict(payload.as_dict())
    assert reparsed == payload
    assert [unit.kind for unit in payload.units] == ["SOURCE", "CARD", "SOURCE"]
    # Visual units are contiguous from 0 to frame_count.
    assert [(u.start_frame, u.end_frame) for u in payload.units] == [(0, 90), (90, 120), (120, 210)]


def test_authored_source_gap_is_allowed_but_core_requires_contiguous_occurrences() -> None:
    payload = _payload()
    validate_authored_payload(payload)

    scenes = (SceneSpec(0, 0, 21.0, 28.0, "STATIC", "NONE"),)
    first = TimelineOccurrence("occ-0", 0, 21.0, 28.0, 0.0, 3.0, "HERO", True, scenes)
    # The card occupies 3.0-4.0 in output time; the trailing source starts at 4.0.
    trailing = TimelineOccurrence("occ-2", 2, 28.0, 35.0, 4.0, 7.0, "CONTEXT", False, scenes)
    # Core source-only validation still rejects the unexplained gap.
    with pytest.raises(TimelineError):
        validate_occurrence_order((first, trailing))


def test_cumulative_fractional_durations_do_not_drift() -> None:
    durations = [F(7, 2), F(1, 3), F(1, 3)]
    frames = cumulative_frame_bounds(durations, F(30, 1))
    assert frames == (0, 105, 115, 125)
    samples = cumulative_sample_bounds(durations, 48000)
    assert samples == (0, 168000, 184000, 200000)

    # Independent per-unit rounding would drift: round(1/3*30)=10 per card, so a
    # naive accumulation yields 105,115,125 only if boundaries are cumulative.
    naive = [0, 105, 115, 125]
    assert frames == tuple(naive)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update({"frame_count": True}),
        lambda d: d.update({"units": [d["units"][0]] + [d["units"][1], d["units"][1]]}),
        lambda d: d.update({"units": [dict(d["units"][0], kind="BROLL")]}),
        lambda d: d.update({"materialization_output_fingerprint": "deadbeef"}),
        lambda d: d.update({"tracks": [dict(d["tracks"][0], gain_db=0.0)]}),
        lambda d: d.update(
            {
                "tracks": [
                    dict(
                        d["tracks"][0],
                        gain_points=[
                            {"sample_index": 100, "gain_db": 0.0},
                            {"sample_index": 50, "gain_db": -3.0},
                        ],
                    )
                ]
            }
        ),
    ],
)
def test_malformed_payloads_are_rejected(mutate) -> None:  # type: ignore[no-untyped-def]
    data = _valid_payload_dict()
    mutate(data)
    with pytest.raises((AuthoredPayloadError, TypeError, KeyError)):
        payload = AuthoredRenderPayload.from_dict(data)
        validate_authored_payload(payload)


def test_unknown_keys_are_rejected() -> None:
    data = _valid_payload_dict()
    data["surprise"] = 1
    with pytest.raises(AuthoredPayloadError):
        AuthoredRenderPayload.from_dict(data)


def test_missing_source_unit_is_rejected() -> None:
    payload = _payload(
        units=[
            _unit("u0", "CARD", 0, 0, 90),
            _unit("u1", "CARD", 1, 90, 210),
        ]
    )
    with pytest.raises(AuthoredPayloadError):
        validate_authored_payload(payload)


def test_non_contiguous_units_are_rejected() -> None:
    payload = _payload(
        units=[
            _unit("u0", "SOURCE", 0, 0, 90, occurrence_id="occ-0"),
            _unit("u1", "CARD", 1, 95, 210),
        ]
    )
    with pytest.raises(AuthoredPayloadError):
        validate_authored_payload(payload)


def test_inverted_track_interval_is_rejected() -> None:
    data = _valid_payload_dict()
    tracks = list(data["tracks"])  # type: ignore[arg-type]
    tracks[0] = dict(tracks[0], source_end_seconds=20.0)  # type: ignore[arg-type]
    payload = _payload(tracks=tracks)
    with pytest.raises(AuthoredPayloadError):
        validate_authored_payload(payload)


def test_narration_asset_role_is_accepted() -> None:
    data = _valid_payload_dict()
    assert any(a["role"] == ASSET_ROLE_NARRATION for a in data["assets"])  # type: ignore[index]
