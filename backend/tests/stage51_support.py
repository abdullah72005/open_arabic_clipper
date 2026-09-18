"""Shared Stage 5.1 test helpers: seeded executable contract plus fake seams.

Every test is hermetic: no ffprobe, no FFmpeg, no face model, no network, and no
provider. Display geometry, frame sampling, scene-cut detection, and face
detection are all injected fakes.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from stage43_support import FakeGovernanceSettings, install_selection_settings
from stage50_support import Stage50Fixture, seed_stage50

from app.composition.analysis import SampledFrame, Span, merge_cuts
from app.composition.policy import Stage51Config, stage51_config_payload
from app.composition.types import DisplayGeometry, FaceDetection
from app.core.settings import get_settings
from app.render.service import RenderContractView, create_render_contract

Frame = npt.NDArray[np.uint8]


class FakeDisplayProbe:
    """Injectable display-geometry seam that never touches ffprobe."""

    def __init__(self, geometry: DisplayGeometry | None = None) -> None:
        self.geometry = geometry or DisplayGeometry(
            encoded_width=1920,
            encoded_height=1080,
            rotation_degrees=0,
            display_width=1920,
            display_height=1080,
        )
        self.calls = 0

    def probe(self, path: Path) -> DisplayGeometry:
        self.calls += 1
        return self.geometry


class FakeFrameSampler:
    """Deterministic sampler yielding tiny synthetic frames and honoring cancel."""

    def __init__(self, *, yield_frames: bool = True) -> None:
        self.requested: list[Span] = []
        self.calls = 0
        self.released = 0
        self.yield_frames = yield_frames

    def samples(
        self,
        spans: Sequence[Span],
        effective_fps: float,
        max_dimension: int,
        cancel_check: object = None,
    ) -> Iterator[SampledFrame]:
        self.calls += 1
        self.requested = list(spans)
        if not self.yield_frames:
            return iter(())
        frames: list[SampledFrame] = []
        for block_index, start, end in spans:
            duration = end - start
            count = max(1, int(math.ceil(duration * effective_fps - 1e-9)))
            for index in range(count):
                if callable(cancel_check) and cancel_check():
                    from app.composition.analysis import AnalysisCancelled

                    raise AnalysisCancelled("cancelled")
                frames.append(
                    SampledFrame(
                        span_block_index=block_index,
                        sample_index=index,
                        source_time=start + index / effective_fps,
                        rgb_frame=np.zeros((4, 4, 3), dtype=np.uint8),
                    )
                )
        return iter(frames)

    def release(self) -> None:
        self.released += 1


class FakeDetector:
    """Detector seam that is either unavailable or returns a fixed anonymous box."""

    def __init__(self, *, ready: bool = False, boxes: tuple[FaceDetection, ...] = ()) -> None:
        self._ready = ready
        self._boxes = boxes
        self.released = 0

    def ready(self) -> bool:
        return self._ready

    def identity(self) -> dict[str, object]:
        return {"name": "fake-yunet", "available": self._ready}

    def detect(self, rgb_frame: Frame) -> tuple[FaceDetection, ...]:
        return self._boxes if self._ready else ()

    def release(self) -> None:
        self.released += 1


class FakeSceneCutDetector:
    """Scene-cut seam clamped and merged exactly like the real detector."""

    def __init__(self, cuts: Sequence[float] = ()) -> None:
        self._cuts = tuple(cuts)

    def cuts(
        self,
        path: Path | str,
        span_start: float,
        span_end: float,
        threshold: float,
        min_scene_seconds: float,
    ) -> tuple[float, ...]:
        return tuple(merge_cuts(self._cuts, span_start, span_end, min_scene_seconds))


class FakeStage51Settings:
    """Duck-typed settings covering the Stage 5.1 executor/queue surface."""

    def __init__(
        self,
        *,
        config: Stage51Config | None = None,
        storage_root: Path | None = None,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
    ) -> None:
        self.config = config or Stage51Config()
        self.storage_root = storage_root or get_settings().storage_root
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary

    def stage51_config(self) -> Stage51Config:
        return self.config

    def stage51_config_payload(self) -> dict[str, object]:
        return stage51_config_payload(self.config)


@dataclass
class Stage51Fixture:
    stage50: Stage50Fixture
    contract: RenderContractView
    settings: FakeStage51Settings
    display_probe: FakeDisplayProbe = field(default_factory=FakeDisplayProbe)


def seed_stage51(
    session: Any,
    monkeypatch: Any,
    *,
    result_specs: list[dict[str, object]] | None = None,
    narration: str = "NONE",
    planning_on_final: bool = True,
    config: Stage51Config | None = None,
) -> Stage51Fixture:
    """Seed a Stage 5.0 fixture and persist one executable render contract."""

    governance = FakeGovernanceSettings()
    install_selection_settings(monkeypatch, governance)
    stage50 = seed_stage50(
        session,
        settings=governance,
        result_specs=result_specs,
        narration=narration,
        planning_on_final=planning_on_final,
    )
    view = create_render_contract(
        session,
        stage50.selection.candidate.id,
        storage=stage50.storage,
        prober=stage50.prober,
    )
    assert view is not None
    assert view.row.contract_ready is True, view.row.reason_codes
    return Stage51Fixture(
        stage50=stage50,
        contract=view,
        settings=FakeStage51Settings(
            config=config or Stage51Config(), storage_root=stage50.storage.storage_root
        ),
    )


def candidate_of(fixture: Stage51Fixture) -> Any:
    return fixture.stage50.selection.candidate


def executable_payload(fixture: Stage51Fixture) -> Mapping[str, object]:
    return dict(fixture.contract.row.contract_payload or {})


__all__ = [
    "FakeDetector",
    "FakeDisplayProbe",
    "FakeFrameSampler",
    "FakeSceneCutDetector",
    "FakeStage51Settings",
    "Stage51Fixture",
    "candidate_of",
    "executable_payload",
    "seed_stage51",
]
