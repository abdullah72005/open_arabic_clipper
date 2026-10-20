"""Stage 6 recipe -> typed audio spec translation (gains + ducking + mastering).

Pure mapping regressions for the SOL defect where the persisted audio recipe's
per-entry gains were replaced with 0 dB and authorized ducking regions were
silently dropped.
"""

from __future__ import annotations

import pytest

from app.materialization.rendering import _ducking_by_source, _tracks
from app.render.execution.authored import GainPoint
from app.render.execution.policy import AUTHORED_AUDIO_RECIPE_UNSUPPORTED
from app.render.execution.service import RenderExecutionError

_SR = 48000


def _recipe(*entries: dict[str, object], ducking: list[dict[str, object]] | None = None) -> dict:
    return {
        "entries": list(entries),
        "ducking_regions": ducking or [],
        "target_integrated_lufs": -16.0,
        "true_peak_ceiling_dbtp": -1.5,
        "sample_rate": _SR,
        "channels": 2,
    }


def test_requested_source_gain_is_preserved() -> None:
    recipe = _recipe(
        {
            "entry_id": "src-0",
            "kind": "SOURCE",
            "output_start": 0.0,
            "output_end": 3.0,
            "source_start": 10.0,
            "source_end": 13.0,
            "gain_db": -4.5,
        }
    )
    tracks = _tracks(recipe, {}, _SR)
    assert len(tracks) == 1
    assert tracks[0].gain_points == (GainPoint(sample_index=0, gain_db=-4.5),)
    assert tracks[0].source_start_seconds == 10.0
    assert tracks[0].output_end_sample == 3 * _SR


def test_authorized_ducking_becomes_bounded_ramp_points() -> None:
    recipe = _recipe(
        {
            "entry_id": "src-0",
            "kind": "SOURCE",
            "output_start": 0.0,
            "output_end": 5.0,
            "source_start": 0.0,
            "source_end": 5.0,
            "gain_db": 0.0,
        },
        ducking=[
            {"source_entry_id": "src-0", "start": 2.0, "end": 3.0, "ducking_db": -12.0},
        ],
    )
    tracks = _tracks(recipe, {}, _SR)
    points = tracks[0].gain_points
    # Attack 80 ms before the region, release 180 ms after it, base elsewhere.
    assert points[0] == GainPoint(sample_index=0, gain_db=0.0)
    by_sample = {point.sample_index: point.gain_db for point in points}
    assert by_sample[round((2.0 - 0.08) * _SR)] == 0.0  # ramp start (base)
    assert by_sample[2 * _SR] == -12.0
    assert by_sample[3 * _SR] == -12.0
    assert by_sample[round((3.0 + 0.18) * _SR)] == 0.0
    assert points[-1] == GainPoint(sample_index=5 * _SR, gain_db=0.0)
    # Strictly increasing sample indices (monotonic envelope).
    indices = [point.sample_index for point in points]
    assert indices == sorted(indices)
    assert len(indices) == len(set(indices))


def test_ducking_region_outside_entry_is_rejected() -> None:
    recipe = _recipe(
        {
            "entry_id": "src-0",
            "kind": "SOURCE",
            "output_start": 0.0,
            "output_end": 2.0,
            "source_start": 0.0,
            "source_end": 2.0,
            "gain_db": 0.0,
        },
        ducking=[
            {"source_entry_id": "src-0", "start": 1.5, "end": 3.0, "ducking_db": -12.0},
        ],
    )
    with pytest.raises(RenderExecutionError):
        _tracks(recipe, {}, _SR)


def test_positive_ducking_is_rejected() -> None:
    recipe = _recipe(
        {
            "entry_id": "src-0",
            "kind": "SOURCE",
            "output_start": 0.0,
            "output_end": 4.0,
            "source_start": 0.0,
            "source_end": 4.0,
            "gain_db": 0.0,
        },
        ducking=[{"source_entry_id": "src-0", "start": 1.0, "end": 2.0, "ducking_db": 3.0}],
    )
    with pytest.raises(RenderExecutionError) as error:
        _tracks(recipe, {}, _SR)
    assert error.value.reason_code == AUTHORED_AUDIO_RECIPE_UNSUPPORTED


def test_overlapping_narration_is_rejected() -> None:
    from app.render.execution.authored import ASSET_ROLE_NARRATION, MediaAssetRef

    assets = {
        "chunk-1": MediaAssetRef(
            asset_id="narr-chunk-1",
            role=ASSET_ROLE_NARRATION,
            storage_relative_path="x.wav",
            sha256="a" * 64,
            size_bytes=10,
            duration_seconds=3.0,
        ),
        "chunk-2": MediaAssetRef(
            asset_id="narr-chunk-2",
            role=ASSET_ROLE_NARRATION,
            storage_relative_path="y.wav",
            sha256="b" * 64,
            size_bytes=10,
            duration_seconds=3.0,
        ),
    }
    recipe = _recipe(
        {
            "entry_id": "n-0",
            "kind": "NARRATION",
            "output_start": 0.0,
            "output_end": 3.0,
            "narration_chunk_id": "chunk-1",
            "gain_db": 0.0,
        },
        {
            "entry_id": "n-1",
            "kind": "NARRATION",
            "output_start": 2.0,
            "output_end": 5.0,
            "narration_chunk_id": "chunk-2",
            "gain_db": 0.0,
        },
    )
    with pytest.raises(RenderExecutionError) as error:
        _tracks(recipe, assets, _SR)
    assert error.value.reason_code == AUTHORED_AUDIO_RECIPE_UNSUPPORTED


def test_ducking_index_rejects_overlapping_regions() -> None:
    recipe = _recipe(
        {
            "entry_id": "src-0",
            "kind": "SOURCE",
            "output_start": 0.0,
            "output_end": 6.0,
            "source_start": 0.0,
            "source_end": 6.0,
        },
        ducking=[
            {"source_entry_id": "src-0", "start": 1.0, "end": 3.0, "ducking_db": -6.0},
            {"source_entry_id": "src-0", "start": 2.0, "end": 4.0, "ducking_db": -6.0},
        ],
    )
    with pytest.raises(RenderExecutionError):
        _ducking_by_source(recipe, _SR)


def test_authorized_peak_limit_reaches_mastering_spec() -> None:
    from app.materialization.rendering import _authorized_peak_limit_db

    class _Row:
        metrics: dict[str, object] = {}

    assert _authorized_peak_limit_db(_Row(), {"repair_peak_limit_db": 2.0}) == 2.0
    assert _authorized_peak_limit_db(_Row(), {}) == 0.0
    with pytest.raises(RenderExecutionError):
        _authorized_peak_limit_db(_Row(), {"repair_peak_limit_db": 5.0})
