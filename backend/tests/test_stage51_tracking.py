"""Pure Stage 5.1 per-scene face-tracking tests."""

from __future__ import annotations

import pytest

from app.composition.policy import Stage51Config
from app.composition.tracking import build_tracks, persistent_tracks
from app.composition.types import FaceDetection


def _det(x: float, y: float, w: float, h: float, score: float = 0.9) -> FaceDetection:
    return FaceDetection(x=x, y=y, w=w, h=h, score=score)


def _config(
    track_max_gap_samples: int = 2,
    track_min_persistence_samples: int = 3,
    track_min_persistence_seconds: float = 0.0,
) -> Stage51Config:
    return Stage51Config(
        track_max_gap_samples=track_max_gap_samples,
        track_min_persistence_samples=track_min_persistence_samples,
        track_min_persistence_seconds=track_min_persistence_seconds,
    )


def test_two_separated_tracks_are_kept_apart() -> None:
    frames = [
        (
            index * 0.5,
            (_det(0.1, 0.4, 0.2, 0.2, 0.9), _det(0.7, 0.4, 0.2, 0.2, 0.8)),
        )
        for index in range(4)
    ]
    tracks = build_tracks(0, frames, _config())

    assert len(tracks) == 2
    assert {track.track_id for track in tracks} == {1, 2}
    assert all(track.stable for track in tracks)
    assert all(len(track.samples) == 4 for track in tracks)
    assert all(track.scene_index == 0 for track in tracks)


def test_track_bridges_gap_within_max_gap_samples() -> None:
    frames = [
        (0.0, (_det(0.4, 0.4, 0.2, 0.2),)),
        (0.5, ()),
        (1.0, ()),
        (1.5, (_det(0.41, 0.4, 0.2, 0.2),)),
    ]
    tracks = build_tracks(0, frames, _config(track_min_persistence_samples=2))

    assert len(tracks) == 1
    assert len(tracks[0].samples) == 2
    assert [sample.source_time for sample in tracks[0].samples] == [0.0, 1.5]


def test_track_exceeding_gap_budget_starts_a_new_track() -> None:
    frames = [
        (0.0, (_det(0.4, 0.4, 0.2, 0.2),)),
        (0.5, ()),
        (1.0, ()),
        (1.5, ()),
        (2.0, (_det(0.4, 0.4, 0.2, 0.2),)),
    ]
    tracks = build_tracks(0, frames, _config(track_max_gap_samples=2))

    assert len(tracks) == 2
    assert tracks[0].track_id == 1
    assert tracks[1].track_id == 2
    assert [sample.source_time for sample in tracks[1].samples] == [2.0]


def test_low_persistence_track_is_unstable_and_excluded() -> None:
    frames = [
        (0.0, (_det(0.4, 0.4, 0.2, 0.2),)),
        (0.5, (_det(0.4, 0.4, 0.2, 0.2),)),
    ]
    config = _config(track_min_persistence_samples=3)
    tracks = build_tracks(0, frames, config)

    assert len(tracks) == 1
    assert tracks[0].persistence == 2.0
    assert tracks[0].stable is False
    assert persistent_tracks(tracks, config) == ()


def test_short_duration_track_is_unstable() -> None:
    frames = [
        (0.0, (_det(0.4, 0.4, 0.2, 0.2),)),
        (0.1, (_det(0.4, 0.4, 0.2, 0.2),)),
        (0.2, (_det(0.4, 0.4, 0.2, 0.2),)),
    ]
    config = _config(track_min_persistence_samples=2, track_min_persistence_seconds=1.0)
    tracks = build_tracks(0, frames, config)

    assert tracks[0].persistence >= 3
    assert (tracks[0].last_time - tracks[0].first_time) < 1.0
    assert tracks[0].stable is False


def test_iou_association_beats_centroid_distance() -> None:
    # Track 1 is a large box up-right of the new detection; track 2 is a tiny box
    # whose centroid is much closer, but whose IoU with the detection is smaller.
    first = (
        _det(0.5, 0.5, 0.3, 0.3, 0.9),
        _det(0.4, 0.5, 0.02, 0.02, 0.5),
    )
    second = (_det(0.4, 0.4, 0.2, 0.2, 0.95),)
    config = _config()
    tracks = build_tracks(0, [(0.0, first), (0.5, second)], config)

    by_id = {track.track_id: track for track in tracks}
    assert len(by_id[1].samples) == 2
    assert len(by_id[2].samples) == 1
    assert by_id[1].samples[-1].cx == 0.5
    assert by_id[1].samples[-1].cy == 0.5


def test_samples_are_decimated_to_scene_budget() -> None:
    frames = [(index * 0.25, (_det(0.4, 0.4, 0.2, 0.2),)) for index in range(300)]
    tracks = build_tracks(0, frames, _config())

    assert len(tracks) == 1
    track = tracks[0]
    assert track.persistence == 300.0
    assert track.decimated is True
    assert len(track.samples) == 240
    assert len(track.samples) <= 240


def test_scene_within_budget_is_not_marked_decimated() -> None:
    frames = [(index * 0.25, (_det(0.4, 0.4, 0.2, 0.2),)) for index in range(10)]
    tracks = build_tracks(0, frames, _config())

    assert tracks[0].decimated is False
    assert len(tracks[0].samples) == 10


def test_track_ids_reset_per_scene() -> None:
    frames = [(index * 0.5, (_det(0.4, 0.4, 0.2, 0.2),)) for index in range(4)]
    first_scene = build_tracks(0, frames, _config())
    second_scene = build_tracks(1, frames, _config())

    assert [track.track_id for track in first_scene] == [1]
    assert [track.track_id for track in second_scene] == [1]
    assert first_scene[0].scene_index == 0
    assert second_scene[0].scene_index == 1


def test_persistent_tracks_sorted_by_face_height_then_id() -> None:
    frames = [
        (
            index * 0.5,
            (_det(0.1, 0.4, 0.1, 0.1, 0.9), _det(0.7, 0.2, 0.3, 0.4, 0.8)),
        )
        for index in range(4)
    ]
    config = _config()
    tracks = build_tracks(0, frames, config)
    persistent = persistent_tracks(tracks, config)

    assert len(persistent) == 2
    assert persistent[0].mean_face_height_fraction > persistent[1].mean_face_height_fraction
    assert persistent[0].track_id == 2


def test_empty_input_returns_no_tracks() -> None:
    assert build_tracks(0, [], _config()) == ()
    assert build_tracks(0, [(0.0, ())], _config()) == ()


def test_internal_box_convention_has_self_iou_one_and_true_geometry() -> None:
    from app.composition.tracking import (
        _centroid_distance,
        _detection_box,
        _iou,
        _size_difference,
    )

    detection = _det(0.4, 0.2, 0.2, 0.4)
    box = _detection_box(detection)

    assert box == pytest.approx((0.4, 0.2, 0.6, 0.6))
    assert _iou(box, box) == 1.0
    assert abs(_centroid_distance(box, box)) == pytest.approx(0.0)
    assert _size_difference(box, box) == pytest.approx(0.0)
    # A translated identical box has the same size and no overlap.
    shifted = (0.7, 0.2, 0.9, 0.6)
    assert _iou(box, shifted) == 0.0
    assert abs(_size_difference(box, shifted)) == pytest.approx(0.0)
    assert abs(_centroid_distance(box, shifted) - 0.3) < 1e-12


def test_translation_consistent_association_keeps_one_track() -> None:
    frames = [
        (0.0, (_det(0.30, 0.40, 0.10, 0.20),)),
        (0.5, (_det(0.32, 0.40, 0.10, 0.20),)),
        (1.0, (_det(0.34, 0.40, 0.10, 0.20),)),
    ]
    tracks = build_tracks(0, frames, _config())

    assert len(tracks) == 1
    assert len(tracks[0].samples) == 3


def test_recorded_two_shot_to_close_up_jump_is_not_associated() -> None:
    """Exact saved boxes from the failed acceptance plan.

    Right two-shot face: cx=0.729885, cy=0.508918, w=0.060453, h=0.13346.
    Close-up face: cx=0.568219, cy=0.383399, w=0.091679, h=0.240932.
    Correct centroid distance is 0.20467 (> the 0.13346 box diameter); the old
    XYWH/edge confusion measured 0.08137 and falsely joined them.
    """

    from app.composition.tracking import _centroid_distance, _detection_box

    right = _det(
        0.729885 - 0.060453 / 2.0,
        0.508918 - 0.13346 / 2.0,
        0.060453,
        0.13346,
    )
    close = _det(
        0.568219 - 0.091679 / 2.0,
        0.383399 - 0.240932 / 2.0,
        0.091679,
        0.240932,
    )
    distance = _centroid_distance(_detection_box(right), _detection_box(close))
    assert abs(distance - 0.20467) < 1e-4
    assert distance > 0.13346

    tracks = build_tracks(
        1,
        [(584.66, (right,)), (585.66, (close,))],
        _config(track_min_persistence_samples=1),
    )
    assert len(tracks) == 2
