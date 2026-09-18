"""Stage 5.1 planner orchestration tests with injected fakes.

Every test is hermetic: no FFmpeg, no ffprobe, no network, no model provider,
and no database. Frames are synthetic tiny arrays; face detection is a fake
keyed by the shared sample-time stream.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from app.composition.analysis import SampledFrame, Span, merge_cuts
from app.composition.planner import build_visual_composition_plan
from app.composition.policy import (
    PlanReasonCode,
    Stage51Config,
    VisualCompositionStatus,
)
from app.composition.types import (
    BoundSpan,
    DisplayGeometry,
    FaceDetection,
    PlannerInputs,
    VisualCompositionPlan,
)
from app.services.storage import StorageService

Frame: TypeAlias = npt.NDArray[np.uint8]
CancelCheck: TypeAlias = object

_EXPECTED_KEYS = {
    "identity",
    "geometry",
    "scenes",
    "captions",
    "ass",
    "overlays",
    "materialization",
    "protection",
    "source_local_timeline",
    "readiness",
    "flags",
    "warnings",
}

_SOURCE_ID = "11111111-1111-1111-1111-111111111111"


class FakeFrameSampler:
    """Deterministic sampler that also publishes its sample times to the fake detector."""

    def __init__(self, *, times: list[float] | None = None) -> None:
        self.requested: list[Span] = []
        self.effective_fps = 0.0
        self.max_dimension = 0
        self.cancel_check: CancelCheck | None = None
        self.times: list[float] = times if times is not None else []

    def samples(
        self,
        spans: Sequence[Span],
        effective_fps: float,
        max_dimension: int,
        cancel_check: object = None,
    ) -> Iterator[SampledFrame]:
        self.requested = list(spans)
        self.effective_fps = effective_fps
        self.max_dimension = max_dimension
        self.cancel_check = cancel_check
        self.times.clear()
        frames: list[SampledFrame] = []
        for block_index, start, end in spans:
            duration = end - start
            count = max(0, int(math.ceil(duration * effective_fps - 1e-9)))
            for index in range(count):
                source_time = start + index / effective_fps
                self.times.append(source_time)
                frames.append(
                    SampledFrame(
                        span_block_index=block_index,
                        sample_index=index,
                        source_time=source_time,
                        rgb_frame=np.zeros((4, 4, 3), dtype=np.uint8),
                    )
                )
        return iter(frames)


class FakeFaceDetector:
    """Face detector returning fixed boxes for the shared sample-time stream."""

    def __init__(
        self,
        *,
        ready: bool = True,
        times: list[float] | None = None,
        default_boxes: tuple[FaceDetection, ...] = (),
        boxes_by_time: Mapping[float, tuple[FaceDetection, ...]] | None = None,
    ) -> None:
        self._ready = ready
        self.times: list[float] = times if times is not None else []
        self._default_boxes = default_boxes
        self._boxes_by_time = dict(boxes_by_time or {})
        self._index = 0

    def ready(self) -> bool:
        return self._ready

    def identity(self) -> dict[str, object]:
        return {"name": "fake-yunet", "available": self._ready}

    def detect(self, rgb_frame: Frame) -> tuple[FaceDetection, ...]:
        if not self._ready:
            return ()
        index = self._index
        self._index += 1
        if index < len(self.times):
            source_time = self.times[index]
            for key, boxes in self._boxes_by_time.items():
                if abs(key - source_time) < 1e-6:
                    return boxes
        return self._default_boxes


class FakeSceneCutDetector:
    """Scene-cut seam clamped and merged exactly like the real one."""

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


def _face(
    *,
    x: float = 0.4,
    y: float = 0.3,
    w: float = 0.2,
    h: float = 0.3,
    score: float = 0.9,
) -> FaceDetection:
    return FaceDetection(x=x, y=y, w=w, h=h, score=score)


def _geometry() -> DisplayGeometry:
    return DisplayGeometry(
        encoded_width=1920,
        encoded_height=1080,
        rotation_degrees=0,
        display_width=1920,
        display_height=1080,
    )


def _words(
    tokens: Sequence[str],
    *,
    start: float = 10.0,
    step: float = 0.5,
    duration: float = 0.4,
) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "text": token,
            "start": round(start + index * step, 4),
            "end": round(start + index * step + duration, 4),
        }
        for index, token in enumerate(tokens)
    ]


def _span(
    block_index: int,
    start: float,
    end: float,
    *,
    word_start: int | None,
    word_end: int | None,
    role: str = "HERO",
    is_hero: bool = True,
) -> BoundSpan:
    return BoundSpan(
        block_index=block_index,
        start=start,
        end=end,
        word_start_index=word_start,
        word_end_index=word_end,
        source_role=role,
        is_hero=is_hero,
        caption_text="",
    )


def _source_block() -> dict[str, object]:
    return {
        "block_index": 0,
        "block_type": "SOURCE_EXCERPT",
        "purpose": "Hero source moment",
        "placement": "sequential",
        "interrupts_source": False,
        "estimated_duration_seconds": 10.0,
        "timeline": {"start": 10.0, "end": 20.0, "authoritative": False},
        "preservation_constraints": [],
        "dependency_ids": [],
        "source_role": "HERO",
        "slot_kind": "SOURCE_MEDIA",
    }


def _authored_slot(draft_line: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "reason_code": "AUTHORED_TEXT_MATERIALIZATION_REQUIRED",
        "purpose": "On-screen context annotation",
        "delivery_intent": "ON_SCREEN_TEXT",
        "placement": "sequential",
    }
    if draft_line is not None:
        payload["authoring_reference"] = {
            "draft_line": draft_line,
            "draft_only": True,
            "authoritative": False,
        }
    return {
        "slot_id": "authored-annotation",
        "slot_kind": "AUTHORED_TEXT",
        "block_index": None,
        "block_type": "TEXTUAL_ANNOTATION",
        "required": True,
        "reason_code": "AUTHORED_TEXT_MATERIALIZATION_REQUIRED",
        "payload": payload,
    }


def _inputs(
    *,
    spans: tuple[BoundSpan, ...],
    words: Sequence[Mapping[str, object]],
    source_id: str = _SOURCE_ID,
    materialization_slots: Sequence[Mapping[str, object]] | None = None,
    hero_block_index: int | None = 0,
) -> PlannerInputs:
    return PlannerInputs(
        candidate_id="candidate-1",
        source_id=source_id,
        contract_id="contract-1",
        contract_input_fingerprint="contract-in",
        contract_output_fingerprint="contract-out",
        contract_status="READY_FOR_RENDER_PLANNING",
        contract_ready=True,
        selected_plan_id="plan-1",
        selection_id="selection-1",
        final_refinement_id="final-1",
        source_media_relative_path=f"sources/{source_id}/source.mp4",
        source_media_identity={
            "source_id": source_id,
            "content_hash": "",
            "relative_path": f"sources/{source_id}/source.mp4",
            "size_bytes": 2048,
            "mtime_ns": 1,
        },
        display_geometry=_geometry(),
        frames_per_second=30.0,
        spans=spans,
        blocks=(_source_block(),),
        hero_block_index=hero_block_index,
        preservation_constraints=(),
        retention={},
        governance={},
        narration={},
        materialization_slots=(
            tuple(materialization_slots)
            if materialization_slots is not None
            else (_authored_slot(),)
        ),
        caption_input={
            "transcript_text": "FINAL CLIP ONLY",
            "word_timestamps": [dict(word) for word in words],
        },
        caption_source_fingerprint="caption-fp",
        output_profile={
            "profile_key": "SHORTS_1080X1920",
            "width": 1080,
            "height": 1920,
        },
        safe_zone_key="SHORTS_VERTICAL_SAFE_ZONE_V1",
        language="ar",
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
        code_switch_evidence={},
        warnings=(),
    )


def _build(
    inputs: PlannerInputs,
    *,
    config: Stage51Config | None = None,
    storage: StorageService | None = None,
    detector: FakeFaceDetector | None = None,
    cuts: Sequence[float] = (),
    contract_ok: bool = True,
    contract_current: bool = True,
) -> VisualCompositionPlan:
    times: list[float] = []
    resolved_config = config if config is not None else Stage51Config()
    sampler = FakeFrameSampler(times=times)
    resolved_detector = (
        detector
        if detector is not None
        else FakeFaceDetector(times=times, default_boxes=(_face(),))
    )
    resolved_detector.times = times
    return build_visual_composition_plan(
        inputs=inputs,
        config=resolved_config,
        storage=storage,
        frame_sampler=sampler,
        scene_cut_detector=FakeSceneCutDetector(cuts),
        detector=resolved_detector,
        contract_ok=contract_ok,
        contract_current=contract_current,
    )


def test_happy_path_produces_ready_visual_composition_plan() -> None:
    words = _words(["hello", "world", "again"])
    spans = (_span(0, 10.0, 20.0, word_start=0, word_end=2),)
    plan = _build(_inputs(spans=spans, words=words), cuts=(15.0,))

    assert plan.status == VisualCompositionStatus.READY_FOR_VISUAL_EXECUTION.value
    assert plan.plan_ready is True
    assert plan.cache_eligible is True
    assert set(plan.payload) == _EXPECTED_KEYS
    assert plan.payload["scenes"]
    assert plan.payload["captions"]["events"]
    assert plan.payload["ass"]["sha256"]
    assert plan.payload["flags"] == {
        "stage5_2_implemented": False,
        "stage6_implemented": False,
    }
    assert plan.payload["source_local_timeline"]["final_timeline_frozen"] is False
    assert plan.payload["readiness"]["stage5_2_handoff_eligible"] is True
    assert plan.payload["readiness"]["source_framing_ready"] is True
    assert plan.payload["readiness"]["source_captions_ready"] is True


def test_no_bound_source_spans_blocks_plan() -> None:
    plan = _build(_inputs(spans=(), words=()))

    assert plan.status == VisualCompositionStatus.BLOCKED.value
    assert plan.plan_ready is False
    assert plan.cache_eligible is False
    assert PlanReasonCode.NO_BOUND_SOURCE_SPANS.value in plan.reason_codes


def test_analysis_scope_exceeded_blocks_plan() -> None:
    config = Stage51Config(max_analysis_seconds=1.0)
    spans = (_span(0, 10.0, 20.0, word_start=0, word_end=0),)
    plan = _build(_inputs(spans=spans, words=_words(["only"])), config=config)

    assert plan.status == VisualCompositionStatus.BLOCKED.value
    assert PlanReasonCode.ANALYSIS_SCOPE_EXCEEDED.value in plan.reason_codes


def test_reversed_span_blocks_plan() -> None:
    spans = (_span(0, 20.0, 10.0, word_start=0, word_end=0),)
    plan = _build(_inputs(spans=spans, words=_words(["only"])))

    assert plan.status == VisualCompositionStatus.BLOCKED.value
    assert PlanReasonCode.CONTRACT_NOT_EXECUTABLE.value in plan.reason_codes


def test_non_current_or_non_executable_contract_blocks_plan() -> None:
    inputs = _inputs(spans=(_span(0, 10.0, 20.0, word_start=0, word_end=0),), words=_words(["x"]))

    stale = _build(inputs, contract_current=False)
    assert stale.status == VisualCompositionStatus.BLOCKED.value
    assert PlanReasonCode.CONTRACT_NOT_CURRENT.value in stale.reason_codes

    broken = _build(inputs, contract_ok=False)
    assert broken.status == VisualCompositionStatus.BLOCKED.value
    assert PlanReasonCode.CONTRACT_NOT_EXECUTABLE.value in broken.reason_codes


def test_detector_not_ready_still_produces_plan() -> None:
    times: list[float] = []
    spans = (_span(0, 10.0, 20.0, word_start=0, word_end=0),)
    inputs = _inputs(spans=spans, words=_words(["only"]))
    detector = FakeFaceDetector(ready=False, times=times)
    sampler = FakeFrameSampler(times=times)

    plan = build_visual_composition_plan(
        inputs=inputs,
        config=Stage51Config(),
        storage=None,
        frame_sampler=sampler,
        scene_cut_detector=FakeSceneCutDetector(),
        detector=detector,
    )

    assert plan.status == VisualCompositionStatus.READY_FOR_VISUAL_EXECUTION.value
    assert plan.plan_ready is True
    assert PlanReasonCode.DETECTOR_UNAVAILABLE.value in plan.reason_codes
    assert plan.payload["scenes"]
    assert plan.metrics["face_detections"] == 0


def test_analyzed_seconds_equal_selected_span_seconds() -> None:
    spans = (_span(0, 10.0, 20.0, word_start=0, word_end=0),)
    inputs = _inputs(spans=spans, words=_words(["only"]))
    times: list[float] = []
    sampler = FakeFrameSampler(times=times)
    detector = FakeFaceDetector(times=times, default_boxes=(_face(),))

    plan = build_visual_composition_plan(
        inputs=inputs,
        config=Stage51Config(scene_context_seconds=1.5),
        storage=None,
        frame_sampler=sampler,
        scene_cut_detector=FakeSceneCutDetector(),
        detector=detector,
    )

    assert plan.metrics["analyzed_source_seconds"] == sum(
        span.end - span.start for span in inputs.spans
    )
    assert sampler.requested == [(0, 10.0, 20.0)]
    assert sampler.requested[0][1] >= 10.0
    assert sampler.requested[0][2] <= 20.0


def test_captions_reference_final_clip_word_indexes_and_flag_missing_evidence() -> None:
    words = _words(["alpha", "beta"])
    spans = (
        _span(0, 10.0, 20.0, word_start=0, word_end=1),
        _span(1, 30.0, 35.0, word_start=None, word_end=None, role="SUPPORT", is_hero=False),
    )
    plan = _build(_inputs(spans=spans, words=words, hero_block_index=0))
    captions = plan.payload["captions"]

    events = captions["events"]
    assert [event["block_index"] for event in events] == [0]
    assert events[0]["word_start_index"] == 0
    assert events[0]["word_end_index"] == 1
    assert events[0]["text"] == "alpha beta"

    missing = captions["missing_evidence"]
    assert [marker["block_index"] for marker in missing] == [1]
    assert PlanReasonCode.CAPTION_EVIDENCE_MISSING.value in plan.reason_codes


def test_authored_slots_are_verbatim_and_never_authoritative_text() -> None:
    spans = (_span(0, 10.0, 20.0, word_start=0, word_end=0),)
    plan = _build(
        _inputs(
            spans=spans,
            words=_words(["only"]),
            materialization_slots=(_authored_slot("DRAFT AUTHORED LINE"),),
        )
    )

    slots = plan.payload["materialization"]["slots"]
    assert slots[0]["payload"]["authoring_reference"]["draft_line"] == "DRAFT AUTHORED LINE"
    assert slots[0]["payload"]["authoring_reference"]["authoritative"] is False
    assert plan.payload["readiness"]["authored_assets_pending"] is True

    for overlay in plan.payload["overlays"]:
        assert overlay["text"] is None
        reference = overlay["authoring_reference"]
        if reference is not None:
            assert reference["authoritative"] is False
            assert reference["draft_only"] is True
    assert '"text": "DRAFT AUTHORED LINE"' not in json.dumps(plan.payload)


def test_ass_asset_is_written_through_storage(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    source_id = str(uuid.uuid4())
    spans = (_span(0, 10.0, 20.0, word_start=0, word_end=0),)
    plan = _build(
        _inputs(spans=spans, words=_words(["only"]), source_id=source_id),
        storage=storage,
    )

    asset = plan.payload["ass"]
    assert asset["asset_path"] == (
        f"sources/{source_id}/visual-composition/{plan.ass_fingerprint}.ass"
    )
    assert (tmp_path / "storage" / str(asset["asset_path"])).is_file()
    assert asset["event_count"] == 1
    assert asset["line_count"] >= 1


def test_deterministic_payload_and_fingerprints() -> None:
    words = _words(["hello", "world", "again"])
    spans = (_span(0, 10.0, 20.0, word_start=0, word_end=2),)
    first = _build(_inputs(spans=spans, words=words), cuts=(15.0,))
    second = _build(_inputs(spans=spans, words=words), cuts=(15.0,))

    assert json.dumps(first.payload, sort_keys=True) == json.dumps(second.payload, sort_keys=True)
    assert first.input_fingerprint == second.input_fingerprint
    assert first.output_fingerprint == second.output_fingerprint
    assert first.analysis_fingerprint == second.analysis_fingerprint
    assert first.framing_fingerprint == second.framing_fingerprint
    assert first.ass_fingerprint == second.ass_fingerprint
