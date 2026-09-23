"""Deterministic per-scene anonymous face tracking for Stage 5.1.

Tracking is pure and geometry-only: it never uses appearance descriptors,
embeddings, or identity, and it never associates across scenes. The caller
supplies samples already grouped per scene and sorted by source time, so a scene
boundary is a hard track boundary by construction.

Association is a deterministic greedy assignment. For each detection the best
active track is chosen by a lexicographic key: highest IoU first, then smallest
centroid distance, then best box-size continuity. A detection may only join a
track that either overlaps it (IoU > 0) or whose last box is within one box
diameter; otherwise it starts a new anonymous track. A track bridges at most
``track_max_gap_samples`` missing samples before it closes.

Tracks are always returned as evidence, even when unstable or decimated. Stored
samples are capped at ``_MAX_STORED_SAMPLES_PER_SCENE`` per scene (240) by a
deterministic proportional decimation; ``persistence`` always reflects the true
number of associated samples before decimation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from app.composition.policy import Stage51Config
from app.composition.types import FaceDetection, FaceTrack, TrackSample

_MAX_STORED_SAMPLES_PER_SCENE = 240

Box: TypeAlias = tuple[float, float, float, float]
SamplesByTime: TypeAlias = Sequence[tuple[float, tuple[FaceDetection, ...]]]


def _iou(first: Box, second: Box) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _centroid_distance(first: Box, second: Box) -> float:
    ax = first[0] + (first[2] - first[0]) / 2.0
    ay = first[1] + (first[3] - first[1]) / 2.0
    bx = second[0] + (second[2] - second[0]) / 2.0
    by = second[1] + (second[3] - second[1]) / 2.0
    return math.hypot(ax - bx, ay - by)


def _size_difference(first: Box, second: Box) -> float:
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    larger = max(area_a, area_b)
    if larger <= 0.0:
        return 0.0
    return abs(area_a - area_b) / larger


@dataclass
class _WorkingTrack:
    track_id: int
    box: Box
    samples: list[TrackSample] = field(default_factory=list)
    gap: int = 0
    closed: bool = False

    def add(self, source_time: float, detection: FaceDetection) -> None:
        self.samples.append(
            TrackSample(
                source_time=source_time,
                cx=detection.cx,
                cy=detection.cy,
                w=detection.w,
                h=detection.h,
                score=detection.score,
            )
        )
        self.box = (detection.x, detection.y, detection.w, detection.h)
        self.gap = 0


def _detection_box(detection: FaceDetection) -> Box:
    return (detection.x, detection.y, detection.w, detection.h)


def _association_key(detection_box: Box, track_box: Box) -> tuple[float, float, float]:
    iou = _iou(detection_box, track_box)
    distance = _centroid_distance(detection_box, track_box)
    size = _size_difference(detection_box, track_box)
    return (iou, -distance, -size)


def _acceptable(detection_box: Box, track_box: Box) -> bool:
    if _iou(detection_box, track_box) > 0.0:
        return True
    return _centroid_distance(detection_box, track_box) <= max(track_box[2], track_box[3])


def _uniform_indices(count: int, keep: int) -> list[int]:
    if keep >= count:
        return list(range(count))
    if keep <= 1:
        return [0]
    return [round(index * (count - 1) / (keep - 1)) for index in range(keep)]


def _allocate_sample_budget(counts: Sequence[int], cap: int) -> list[int]:
    """Deterministically allocate a per-scene stored-sample budget."""

    total = sum(counts)
    if total <= cap:
        return list(counts)

    allocation = [1 if count > 0 else 0 for count in counts]
    nonempty = [index for index, count in enumerate(counts) if count > 0]
    if len(nonempty) >= cap:
        # Pathological many-track case: keep one sample per track up to the cap.
        for position, index in enumerate(nonempty):
            allocation[index] = 1 if position < cap else 0
        return allocation

    remaining = cap - len(nonempty)
    weights = [count - 1 for count in counts]
    weight_total = sum(weights)
    if remaining > 0 and weight_total > 0:
        quotas = [remaining * weight / weight_total for weight in weights]
        additions = [
            min(int(quotas[index]), counts[index] - allocation[index])
            for index in range(len(counts))
        ]
        leftover = remaining - sum(additions)
        order = sorted(
            range(len(counts)),
            key=lambda index: (-(quotas[index] - int(quotas[index])), index),
        )
        for index in order:
            if leftover <= 0:
                break
            room = counts[index] - allocation[index] - additions[index]
            if room > 0:
                take = min(room, leftover)
                additions[index] += take
                leftover -= take
        allocation = [allocation[index] + additions[index] for index in range(len(counts))]
    return allocation


def build_tracks(
    scene_index: int,
    samples_by_time: SamplesByTime,
    config: Stage51Config,
) -> tuple[FaceTrack, ...]:
    """Build anonymous deterministic tracks for one scene."""

    working: list[_WorkingTrack] = []
    next_track_id = 1

    for source_time, detections in samples_by_time:
        assigned: set[int] = set()
        for detection in detections:
            detection_box = _detection_box(detection)
            best: _WorkingTrack | None = None
            best_key: tuple[float, float, float] | None = None
            for track in working:
                if track.closed or track.track_id in assigned:
                    continue
                if not _acceptable(detection_box, track.box):
                    continue
                key = _association_key(detection_box, track.box)
                if best_key is None or key > best_key:
                    best_key = key
                    best = track
            if best is None:
                new_track = _WorkingTrack(track_id=next_track_id, box=detection_box)
                next_track_id += 1
                new_track.add(source_time, detection)
                working.append(new_track)
                assigned.add(new_track.track_id)
            else:
                best.add(source_time, detection)
                assigned.add(best.track_id)

        for track in working:
            if track.closed or track.track_id in assigned:
                continue
            track.gap += 1
            if track.gap > config.track_max_gap_samples:
                track.closed = True

    counts = [len(track.samples) for track in working]
    allocation = _allocate_sample_budget(counts, _MAX_STORED_SAMPLES_PER_SCENE)

    tracks: list[FaceTrack] = []
    for track, stored_count in zip(working, allocation, strict=True):
        original = track.samples
        if stored_count < len(original):
            indices = _uniform_indices(len(original), stored_count)
            stored = tuple(original[index] for index in indices)
            decimated = True
        else:
            stored = tuple(original)
            decimated = False

        persistence = float(len(original))
        mean_score = sum(sample.score for sample in original) / persistence if original else 0.0
        mean_height = sum(sample.h for sample in original) / persistence if original else 0.0
        first_time = original[0].source_time
        last_time = original[-1].source_time
        duration = last_time - first_time
        stable = (
            persistence >= config.track_min_persistence_samples
            and duration >= config.track_min_persistence_seconds
        )
        tracks.append(
            FaceTrack(
                track_id=track.track_id,
                scene_index=scene_index,
                first_time=first_time,
                last_time=last_time,
                samples=stored,
                persistence=persistence,
                mean_score=mean_score,
                mean_face_height_fraction=mean_height,
                stable=stable,
                decimated=decimated,
            )
        )

    tracks.sort(key=lambda track: track.track_id)
    return tuple(tracks)


def persistent_tracks(tracks: Sequence[FaceTrack], config: Stage51Config) -> tuple[FaceTrack, ...]:
    """Return stable tracks strongest (largest face) first, then by track id."""

    stable = [track for track in tracks if track.stable]
    stable.sort(key=lambda track: (-track.mean_face_height_fraction, track.track_id))
    return tuple(stable)


__all__ = [
    "build_tracks",
    "persistent_tracks",
]
