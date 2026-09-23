"""Stage 5.1 visual-composition plan builder.

Pure-ish orchestration from already-resolved Stage 5.0 contract facts
(:class:`~app.composition.types.PlannerInputs`) plus injected seams
(``frame_sampler``, ``scene_cut_detector``, ``detector``) to a persisted-ready
:class:`~app.composition.types.VisualCompositionPlan`.

The planner owns deterministic orchestration only. It never touches the
database, never calls ffprobe itself, never makes a network call, and never
imports a hosted or local model provider. It decodes only the selected bound
source spans plus a bounded scene-context margin used solely for cut alignment;
that context is always clipped out of every emitted scene, keyframe, caption,
and output range.

Cache eligibility is deliberately narrow: ``cache_eligible`` is ``True`` if and
only if ``status == READY_FOR_VISUAL_EXECUTION``. A degraded analysis (an
unavailable detector, a deterministic fallback framing mode, a reduced analysis
fps, or missing caption word evidence) is recorded truthfully in reason codes,
warnings, and metrics but is not, by itself, a reason to withhold a
deterministic semantic plan. Only ``BLOCKED``/``FAILED`` results are never
cache-eligible.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from app.composition.analysis import (
    AnalysisScopeExceeded,
    FrameSampler,
    SceneCutDetector,
    merge_spans,
    plan_analysis,
    segment_scenes,
)
from app.composition.ass import (
    ass_asset_relative_path,
    serialize_ass,
    write_ass_asset,
)
from app.composition.captions import CaptionPlan, SceneFaceBox, build_caption_plan
from app.composition.detector import FaceDetector
from app.composition.fingerprints import (
    analysis_fingerprint as _analysis_fingerprint,
)
from app.composition.fingerprints import (
    ass_fingerprint as _ass_fingerprint,
)
from app.composition.fingerprints import (
    build_stage51_input_payload,
    source_media_identity_fingerprint,
    visual_composition_input_fingerprint,
    visual_composition_output_fingerprint,
)
from app.composition.fingerprints import (
    caption_plan_fingerprint as _caption_plan_fingerprint,
)
from app.composition.fingerprints import (
    framing_fingerprint as _framing_fingerprint,
)
from app.composition.framing import build_scene, select_framing_mode
from app.composition.overlays import ProtectedRegion, build_overlay_requirements
from app.composition.policy import (
    ASS_POLICY_VERSION,
    CAPTION_LAYOUT_POLICY_VERSION,
    DETECTOR_IDENTITY,
    FINGERPRINT_VERSION,
    FRAMING_POLICY_VERSION,
    OUTPUT_ASPECT,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SAFE_ZONE_PROFILE_VERSION,
    SCHEMA_VERSION,
    VISUAL_COMPOSITION_POLICY_VERSION,
    CaptionStyle,
    FramingEvidence,
    FramingMode,
    PlanReasonCode,
    SafeZoneProfile,
    Stage51Config,
    VisualCompositionStatus,
    config_fingerprint_dict,
    safe_zone_for,
)
from app.composition.tracking import build_tracks
from app.composition.types import (
    BoundSpan,
    CompositionMetrics,
    FaceDetection,
    OverlayRequirement,
    PlannerInputs,
    Scene,
    VisualCompositionPlan,
)
from app.services.storage import StorageService

CancelCheck = Callable[[], bool]

_INTERNAL_ERROR = "INTERNAL_ERROR"
_HERO_MODE_FROZEN = "HERO_MODE_FROZEN"
_OVERLAY_CANNOT_COVER_FACE = "OVERLAY_CANNOT_COVER_FACE"
_EPSILON = 1e-6


@dataclass(frozen=True)
class _SceneSpan:
    """One clipped scene window inside a selected bound span."""

    scene_index: int
    block_index: int
    start: float
    end: float
    cut_start: bool
    is_hero: bool


def build_visual_composition_plan(
    *,
    inputs: PlannerInputs,
    config: Stage51Config,
    storage: StorageService | None,
    frame_sampler: FrameSampler,
    scene_cut_detector: SceneCutDetector,
    detector: FaceDetector,
    cancel_check: CancelCheck | None = None,
    contract_ok: bool = True,
    contract_current: bool = True,
) -> VisualCompositionPlan:
    """Build a deterministic Stage 5.1 visual-composition plan.

    ``contract_ok``/``contract_current`` are explicit seams: the queue layer
    already validates contract currentness and executability, so the planner
    records ``CONTRACT_NOT_CURRENT``/``CONTRACT_NOT_EXECUTABLE`` only when a
    caller deliberately reports the contract as unusable.
    """

    detector_identity: Mapping[str, object] = dict(DETECTOR_IDENTITY)

    if not contract_current:
        return _blocked(
            inputs,
            config,
            detector_identity,
            PlanReasonCode.CONTRACT_NOT_CURRENT,
            ("render contract is not current",),
        )
    if not contract_ok:
        return _blocked(
            inputs,
            config,
            detector_identity,
            PlanReasonCode.CONTRACT_NOT_EXECUTABLE,
            ("render contract is not executable",),
        )
    if not inputs.spans:
        return _blocked(
            inputs,
            config,
            detector_identity,
            PlanReasonCode.NO_BOUND_SOURCE_SPANS,
            ("no bound source spans were provided",),
        )
    invalid = [span for span in inputs.spans if not _span_is_valid(span)]
    if invalid:
        warnings = tuple(
            f"INVALID_SPAN_RANGE:block={span.block_index}:start={span.start}:end={span.end}"
            for span in invalid
        )
        return _blocked(
            inputs,
            config,
            detector_identity,
            PlanReasonCode.CONTRACT_NOT_EXECUTABLE,
            warnings,
        )

    try:
        return _build(
            inputs=inputs,
            config=config,
            storage=storage,
            frame_sampler=frame_sampler,
            scene_cut_detector=scene_cut_detector,
            detector=detector,
            cancel_check=cancel_check,
            detector_identity=detector_identity,
        )
    except AnalysisScopeExceeded as error:
        return _blocked(
            inputs,
            config,
            detector_identity,
            PlanReasonCode.ANALYSIS_SCOPE_EXCEEDED,
            (f"analysis scope exceeded: {error}",),
        )
    except Exception as error:  # noqa: BLE001 - explicit FAILED fallback for the worker
        return _failed(inputs, config, detector_identity, error)


def _build(
    *,
    inputs: PlannerInputs,
    config: Stage51Config,
    storage: StorageService | None,
    frame_sampler: FrameSampler,
    scene_cut_detector: SceneCutDetector,
    detector: FaceDetector,
    cancel_check: CancelCheck | None,
    detector_identity: Mapping[str, object],
) -> VisualCompositionPlan:
    analysis_start = time.monotonic()
    style = _caption_style(config)
    safe_zone = safe_zone_for(inputs.safe_zone_key)
    selected = merge_spans([(span.block_index, span.start, span.end) for span in inputs.spans])
    analysis_plan = plan_analysis(
        selected,
        analysis_fps=config.analysis_fps,
        max_analysis_seconds=config.max_analysis_seconds,
        max_analysis_frames=config.max_analysis_frames,
        scene_context_seconds=config.scene_context_seconds,
    )

    hero_blocks = {span.block_index: span.is_hero for span in inputs.spans if span.is_hero}
    if inputs.hero_block_index is not None:
        hero_blocks[inputs.hero_block_index] = True

    source_path = _source_path(inputs, storage)
    scene_spans: list[_SceneSpan] = []
    for block_index, start, end in selected:
        expanded_start = max(0.0, start - config.scene_context_seconds)
        expanded_end = end + config.scene_context_seconds
        cuts = scene_cut_detector.cuts(
            source_path,
            expanded_start,
            expanded_end,
            config.scene_cut_threshold,
            config.min_scene_seconds,
        )
        raw_scenes = segment_scenes(
            expanded_start,
            expanded_end,
            cuts,
            min_scene_seconds=config.min_scene_seconds,
        )
        for position, (raw_start, raw_end) in enumerate(raw_scenes):
            clip_start = max(raw_start, start)
            clip_end = min(raw_end, end)
            if clip_end - clip_start <= 0.0:
                continue
            scene_spans.append(
                _SceneSpan(
                    scene_index=len(scene_spans),
                    block_index=block_index,
                    start=clip_start,
                    end=clip_end,
                    cut_start=position > 0,
                    is_hero=hero_blocks.get(block_index, False),
                )
            )

    try:
        detector_ready = bool(detector.ready())
    except Exception:  # noqa: BLE001 - availability must never fail the plan
        detector_ready = False

    frames_by_scene: dict[int, list[tuple[float, tuple[FaceDetection, ...]]]] = {
        scene.scene_index: [] for scene in scene_spans
    }
    sampled_frames = 0
    face_detections = 0
    detection_wall = 0.0
    decode_start = time.monotonic()
    for frame in frame_sampler.samples(
        selected,
        analysis_plan.effective_fps,
        config.analysis_frame_max_dimension,
        cancel_check,
    ):
        sampled_frames += 1
        scene = _scene_for_time(scene_spans, frame.span_block_index, frame.source_time)
        if scene is None or not detector_ready:
            continue
        detection_start = time.monotonic()
        detections = detector.detect(frame.rgb_frame)
        detection_wall += time.monotonic() - detection_start
        face_detections += len(detections)
        frames_by_scene[scene.scene_index].append((frame.source_time, detections))
    decode_wall = max(0.0, (time.monotonic() - decode_start) - detection_wall)

    scenes: list[Scene] = []
    track_count = 0
    mode_seconds: dict[str, float] = {mode.value: 0.0 for mode in FramingMode}
    no_face_seconds = 0.0
    fallback_reasons: set[str] = set()
    for scene in scene_spans:
        samples = sorted(frames_by_scene[scene.scene_index], key=lambda item: item[0])
        tracks = build_tracks(scene.scene_index, samples, config)
        track_count += len(tracks)
        decision = select_framing_mode(
            block_index=scene.block_index,
            scene_start=scene.start,
            scene_end=scene.end,
            tracks=tracks,
            config=config,
            geometry=inputs.display_geometry,
            is_hero=scene.is_hero,
        )
        duration = max(0.0, scene.end - scene.start)
        mode_seconds[decision.mode] = mode_seconds.get(decision.mode, 0.0) + duration
        if FramingEvidence.NO_FACE.value in decision.evidence:
            no_face_seconds += duration
        if decision.mode == FramingMode.CENTER_FALLBACK.value:
            fallback_reasons.add(PlanReasonCode.FALLBACK_APPLIED.value)
        scenes.append(
            build_scene(
                scene_index=scene.scene_index,
                block_index=scene.block_index,
                scene_start=scene.start,
                scene_end=scene.end,
                cut_start=scene.cut_start,
                mode=decision,
                tracks=tracks,
                config=config,
                geometry=inputs.display_geometry,
                is_hero=scene.is_hero,
            )
        )

    scene_boxes = _scene_face_boxes(scenes)
    caption_plan = build_caption_plan(
        inputs.spans,
        _word_timestamps(inputs.caption_input),
        style,
        config,
        scene_face_boxes=scene_boxes,
        hero_block_index=inputs.hero_block_index,
    )

    reasons: set[str] = set(caption_plan.reason_codes)
    for composition_scene in scenes:
        reasons.update(composition_scene.framing.evidence)
    if not detector_ready:
        reasons.add(PlanReasonCode.DETECTOR_UNAVAILABLE.value)
        fallback_reasons.add(PlanReasonCode.DETECTOR_UNAVAILABLE.value)
    if any(scene.cut_start for scene in scene_spans):
        reasons.add(PlanReasonCode.TRACKS_RESET_AT_CUT.value)
    if caption_plan.missing_evidence:
        fallback_reasons.add(PlanReasonCode.CAPTION_EVIDENCE_MISSING.value)

    overlay_requirements = build_overlay_requirements(
        inputs.materialization_slots,
        inputs.blocks,
        safe_zone=safe_zone,
        scene_caption_zones=caption_plan.scene_zones,
        protected_regions=_protected_regions(scene_boxes),
        block_scene_indexes=_block_scene_map(scenes),
        hero_block_index=inputs.hero_block_index,
    )

    readiness = _readiness(
        VisualCompositionStatus.READY_FOR_VISUAL_EXECUTION.value,
        scenes,
        inputs.materialization_slots,
    )
    line_count = sum(len(event.lines) if event.lines else 1 for event in caption_plan.events)
    metrics = CompositionMetrics(
        analyzed_source_seconds=analysis_plan.total_selected_seconds,
        sampled_frames=sampled_frames,
        scene_count=len(scenes),
        track_count=track_count,
        face_detections=face_detections,
        tracked_crop_seconds=mode_seconds[FramingMode.TRACKED_CROP.value],
        static_crop_seconds=mode_seconds[FramingMode.STATIC_CROP.value],
        multi_subject_seconds=mode_seconds[FramingMode.MULTI_SUBJECT_FIT.value],
        background_fill_seconds=mode_seconds[FramingMode.BACKGROUND_FILL.value],
        source_as_is_seconds=mode_seconds[FramingMode.SOURCE_AS_IS.value],
        no_face_seconds=no_face_seconds,
        center_fallback_seconds=mode_seconds[FramingMode.CENTER_FALLBACK.value],
        caption_events=len(caption_plan.events),
        caption_lines=line_count,
        analysis_wall_seconds=max(0.0, time.monotonic() - analysis_start),
        decode_wall_seconds=decode_wall,
        detection_wall_seconds=detection_wall,
        cache_reuse=False,
        fallback_reasons=tuple(sorted(fallback_reasons)),
    )

    warnings = list(inputs.warnings)
    if analysis_plan.reduced_fps:
        warnings.append(
            f"ANALYSIS_FPS_REDUCED:{config.analysis_fps}->{analysis_plan.effective_fps}"
        )

    return _finalize(
        inputs=inputs,
        config=config,
        detector_identity=detector_identity,
        status=VisualCompositionStatus.READY_FOR_VISUAL_EXECUTION.value,
        plan_ready=True,
        scenes=scenes,
        caption_plan=caption_plan,
        style=style,
        safe_zone=safe_zone,
        overlay_requirements=overlay_requirements,
        ordered_block_hints=_ordered_block_hints(inputs.blocks),
        readiness=readiness,
        metrics=metrics,
        reason_codes=tuple(sorted(reasons)),
        warnings=tuple(warnings),
        effective_fps=analysis_plan.effective_fps,
        effective_reduced=analysis_plan.reduced_fps,
        storage=storage,
        write_asset=True,
    )


def _finalize(
    *,
    inputs: PlannerInputs,
    config: Stage51Config,
    detector_identity: Mapping[str, object],
    status: str,
    plan_ready: bool,
    scenes: Sequence[Scene],
    caption_plan: CaptionPlan,
    style: CaptionStyle,
    safe_zone: SafeZoneProfile,
    overlay_requirements: Sequence[OverlayRequirement],
    ordered_block_hints: Sequence[Mapping[str, object]],
    readiness: Mapping[str, object],
    metrics: CompositionMetrics,
    reason_codes: Sequence[str],
    warnings: Sequence[str],
    effective_fps: float,
    effective_reduced: bool,
    storage: StorageService | None,
    write_asset: bool,
) -> VisualCompositionPlan:
    scenes_list = [scene.as_dict() for scene in scenes]
    captions_payload: dict[str, object] = {
        "policy_version": caption_plan.policy_version,
        "style": style.as_dict(),
        "events": [event.as_dict() for event in caption_plan.events],
        "missing_evidence": [marker.as_dict() for marker in caption_plan.missing_evidence],
        "safe_zone": safe_zone.as_dict(),
    }
    caption_fp = _caption_plan_fingerprint(
        {
            "caption_plan": caption_plan.as_dict(),
            "style": style.as_dict(),
            "safe_zone": safe_zone.as_dict(),
        }
    )
    ass_fp = _ass_fingerprint(
        {
            "caption_plan_fingerprint": caption_fp,
            "style": style.as_dict(),
            "safe_zone": safe_zone.as_dict(),
            "policy_version": ASS_POLICY_VERSION,
        }
    )
    ass_bytes = serialize_ass(caption_plan, style, safe_zone)
    ass_sha = hashlib.sha256(ass_bytes).hexdigest()
    asset_path: str | None = None
    if write_asset and storage is not None and inputs.source_id:
        _, written_digest = write_ass_asset(storage, inputs.source_id, ass_fp, ass_bytes)
        ass_sha = written_digest
        asset_path = ass_asset_relative_path(inputs.source_id, ass_fp)
    event_count = len(caption_plan.events)
    line_count = sum(len(event.lines) if event.lines else 1 for event in caption_plan.events)
    ass_payload: dict[str, object] = {
        "asset_path": asset_path,
        "sha256": ass_sha,
        "event_count": event_count,
        "line_count": line_count,
        "policy_version": ASS_POLICY_VERSION,
    }
    overlays_list = [requirement.as_dict() for requirement in overlay_requirements]
    source_media_fp = source_media_identity_fingerprint(inputs.source_media_identity)

    payload: dict[str, object] = {
        "identity": _identity_payload(inputs, source_media_fp),
        "geometry": {
            **inputs.display_geometry.as_dict(),
            "frames_per_second": inputs.frames_per_second,
            "output": {
                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,
                "aspect": round(OUTPUT_ASPECT, 6),
            },
            "safe_zone": safe_zone.as_dict(),
        },
        "scenes": scenes_list,
        "captions": captions_payload,
        "ass": ass_payload,
        "overlays": overlays_list,
        "materialization": _materialization_payload(inputs.materialization_slots),
        "protection": _protection_payload(inputs, scenes),
        "source_local_timeline": {
            "final_timeline_frozen": False,
            "ordered_block_hints": [dict(hint) for hint in ordered_block_hints],
        },
        "readiness": dict(readiness),
        "flags": {
            "stage5_2_implemented": False,
            "stage6_implemented": False,
        },
        "warnings": list(warnings),
    }

    analysis_fp = _analysis_fingerprint(
        {
            "spans": [span.as_dict() for span in inputs.spans],
            "effective_fps": effective_fps,
            "effective_fps_reduced": effective_reduced,
            "scene_context_seconds": config.scene_context_seconds,
            "min_scene_seconds": config.min_scene_seconds,
            "scene_cut_threshold": config.scene_cut_threshold,
            "display_geometry": inputs.display_geometry.as_dict(),
            "detector_identity": dict(detector_identity),
        }
    )
    framing_fp = _framing_fingerprint(
        {
            "scenes": scenes_list,
            "display_geometry": inputs.display_geometry.as_dict(),
            "framing_policy_version": FRAMING_POLICY_VERSION,
            "config": config_fingerprint_dict(config),
            "detector_identity": dict(detector_identity),
        }
    )
    input_fp = visual_composition_input_fingerprint(
        build_stage51_input_payload(
            candidate_id=inputs.candidate_id,
            candidate_key=inputs.candidate_id,
            candidate_is_current=True,
            disposition=inputs.contract_status,
            analysis_fingerprint=analysis_fp,
            source_id=inputs.source_id,
            source_media_identity=dict(inputs.source_media_identity),
            contract_input_fingerprint=inputs.contract_input_fingerprint,
            contract_output_fingerprint=inputs.contract_output_fingerprint,
            contract_status=inputs.contract_status,
            contract_ready=inputs.contract_ready,
            contract_is_current=True,
            contract_live_freshness="CURRENT",
            contract_effective=True,
            selected_plan_id=inputs.selected_plan_id,
            selection_id=inputs.selection_id,
            final_refinement_id=inputs.final_refinement_id,
            display_geometry=inputs.display_geometry.as_dict(),
            rotation_degrees=inputs.display_geometry.rotation_degrees,
            bound_spans=[span.as_dict() for span in inputs.spans],
            caption_source_fingerprint=inputs.caption_source_fingerprint,
            caption_payload=dict(inputs.caption_input),
            output_profile=dict(inputs.output_profile),
            config=config,
            detector_identity=dict(detector_identity),
            safe_zone_key=inputs.safe_zone_key,
        )
    )
    output_fp = visual_composition_output_fingerprint(
        {
            "scenes": scenes_list,
            "captions": captions_payload,
            "ass": {
                "sha256": ass_sha,
                "event_count": event_count,
                "line_count": line_count,
                "policy_version": ASS_POLICY_VERSION,
            },
            "overlays": overlays_list,
            "readiness": dict(readiness),
        }
    )

    return VisualCompositionPlan(
        status=status,
        plan_ready=plan_ready,
        reason_codes=tuple(reason_codes),
        payload=payload,
        metrics=metrics.as_dict(),
        input_fingerprint=input_fp,
        output_fingerprint=output_fp,
        analysis_fingerprint=analysis_fp,
        framing_fingerprint=framing_fp,
        ass_fingerprint=ass_fp,
        caption_source_fingerprint=inputs.caption_source_fingerprint,
        contract_input_fingerprint=inputs.contract_input_fingerprint,
        contract_output_fingerprint=inputs.contract_output_fingerprint,
        source_media_fingerprint=source_media_fp,
        source_media_identity=dict(inputs.source_media_identity),
        cache_eligible=status == VisualCompositionStatus.READY_FOR_VISUAL_EXECUTION.value,
    )


def _blocked(
    inputs: PlannerInputs,
    config: Stage51Config,
    detector_identity: Mapping[str, object],
    reason_code: PlanReasonCode,
    warnings: Sequence[str],
) -> VisualCompositionPlan:
    style = _caption_style(config)
    safe_zone = safe_zone_for(inputs.safe_zone_key)
    return _finalize(
        inputs=inputs,
        config=config,
        detector_identity=detector_identity,
        status=VisualCompositionStatus.BLOCKED.value,
        plan_ready=False,
        scenes=(),
        caption_plan=_empty_caption_plan(style, config),
        style=style,
        safe_zone=safe_zone,
        overlay_requirements=(),
        ordered_block_hints=_ordered_block_hints(inputs.blocks),
        readiness=_readiness(
            VisualCompositionStatus.BLOCKED.value, (), inputs.materialization_slots
        ),
        metrics=CompositionMetrics(),
        reason_codes=(reason_code.value,),
        warnings=tuple(inputs.warnings) + tuple(warnings),
        effective_fps=config.analysis_fps,
        effective_reduced=False,
        storage=None,
        write_asset=False,
    )


def _failed(
    inputs: PlannerInputs,
    config: Stage51Config,
    detector_identity: Mapping[str, object],
    error: BaseException,
) -> VisualCompositionPlan:
    style = _caption_style(config)
    safe_zone = safe_zone_for(inputs.safe_zone_key)
    warnings = tuple(inputs.warnings) + (f"INTERNAL_ERROR:{type(error).__name__}:{error}",)
    return _finalize(
        inputs=inputs,
        config=config,
        detector_identity=detector_identity,
        status=VisualCompositionStatus.FAILED.value,
        plan_ready=False,
        scenes=(),
        caption_plan=_empty_caption_plan(style, config),
        style=style,
        safe_zone=safe_zone,
        overlay_requirements=(),
        ordered_block_hints=_ordered_block_hints(inputs.blocks),
        readiness=_readiness(
            VisualCompositionStatus.FAILED.value, (), inputs.materialization_slots
        ),
        metrics=CompositionMetrics(),
        reason_codes=(_INTERNAL_ERROR,),
        warnings=warnings,
        effective_fps=config.analysis_fps,
        effective_reduced=False,
        storage=None,
        write_asset=False,
    )


def _caption_style(config: Stage51Config) -> CaptionStyle:
    return CaptionStyle(
        font_family=config.caption_font_family,
        max_lines=config.caption_max_lines,
        active_color=config.caption_active_color,
        active_emphasis=config.caption_dynamic_emphasis,
        max_words_per_event=config.caption_max_words_per_event,
    )


def _empty_caption_plan(style: CaptionStyle, config: Stage51Config) -> CaptionPlan:
    return build_caption_plan((), (), style, config)


def _span_is_valid(span: BoundSpan) -> bool:
    return (
        isinstance(span.block_index, int)
        and math.isfinite(span.start)
        and math.isfinite(span.end)
        and span.start >= 0.0
        and span.end > span.start
    )


def _source_path(inputs: PlannerInputs, storage: StorageService | None) -> Path:
    relative = inputs.source_media_relative_path
    if storage is not None and relative:
        return Path(str(storage.storage_root)) / str(relative)
    return Path(relative) if relative else Path(".")


def _scene_for_time(
    scene_spans: Sequence[_SceneSpan], block_index: int, source_time: float
) -> _SceneSpan | None:
    match: _SceneSpan | None = None
    for scene in scene_spans:
        if scene.block_index != block_index:
            continue
        if scene.start - _EPSILON <= source_time <= scene.end + _EPSILON:
            match = scene
    return match


def _scene_face_boxes(scenes: Sequence[Scene]) -> tuple[SceneFaceBox, ...]:
    boxes: list[SceneFaceBox] = []
    for scene in scenes:
        for track in scene.tracks:
            for sample in track.samples:
                boxes.append(
                    SceneFaceBox(
                        scene_index=scene.scene_index,
                        start=sample.source_time,
                        end=sample.source_time,
                        x=sample.cx - sample.w / 2.0,
                        y=sample.cy - sample.h / 2.0,
                        w=sample.w,
                        h=sample.h,
                    )
                )
    return tuple(boxes)


def _protected_regions(boxes: Sequence[SceneFaceBox]) -> tuple[ProtectedRegion, ...]:
    return tuple(
        ProtectedRegion(
            x=box.x,
            y=box.y,
            w=box.w,
            h=box.h,
            scene_index=box.scene_index,
            start=box.start,
            end=box.end,
        )
        for box in boxes
    )


def _block_scene_map(scenes: Sequence[Scene]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for scene in scenes:
        mapping.setdefault(scene.block_index, scene.scene_index)
    return mapping


def _materialization_payload(
    slots: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    slot_list = [dict(slot) for slot in slots]
    required = any(bool(slot.get("required", False)) for slot in slot_list)
    return {"required": required, "slots": slot_list}


def _slot_unmaterialized(slot: Mapping[str, object]) -> bool:
    if slot.get("materialized") is True:
        return False
    if slot.get("status") == "READY":
        return False
    return bool(slot.get("required", False))


def _readiness(
    status: str,
    scenes: Sequence[Scene],
    materialization_slots: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ready = status == VisualCompositionStatus.READY_FOR_VISUAL_EXECUTION.value
    return {
        "source_framing_ready": ready and bool(scenes),
        "source_captions_ready": ready,
        "authored_assets_pending": _authored_assets_pending(materialization_slots),
        "stage5_2_handoff_eligible": ready,
    }


def _protection_payload(inputs: PlannerInputs, scenes: Sequence[Scene]) -> dict[str, object]:
    markers: list[str] = []
    if inputs.hero_block_index is not None or any(scene.framing.protected for scene in scenes):
        markers.append(_HERO_MODE_FROZEN)
    markers.append(_OVERLAY_CANNOT_COVER_FACE)
    return {
        "hero_block_index": inputs.hero_block_index,
        "preservation_constraints": list(inputs.preservation_constraints),
        "retention": dict(inputs.retention),
        "governance": dict(inputs.governance),
        "protection_markers": markers,
    }


def _identity_payload(inputs: PlannerInputs, source_media_fingerprint: str) -> dict[str, object]:
    return {
        "candidate_id": inputs.candidate_id,
        "source_id": inputs.source_id,
        "contract_id": inputs.contract_id,
        "contract_input_fingerprint": inputs.contract_input_fingerprint,
        "contract_output_fingerprint": inputs.contract_output_fingerprint,
        "contract_status": inputs.contract_status,
        "contract_ready": inputs.contract_ready,
        "selected_plan_id": inputs.selected_plan_id,
        "selection_id": inputs.selection_id,
        "final_refinement_id": inputs.final_refinement_id,
        "source_media_identity": dict(inputs.source_media_identity),
        "source_media_fingerprint": source_media_fingerprint,
        "policy_version": VISUAL_COMPOSITION_POLICY_VERSION,
        "schema_version": SCHEMA_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "framing_policy_version": FRAMING_POLICY_VERSION,
        "caption_layout_policy_version": CAPTION_LAYOUT_POLICY_VERSION,
        "ass_policy_version": ASS_POLICY_VERSION,
        "safe_zone_profile_version": SAFE_ZONE_PROFILE_VERSION,
    }


def _ordered_block_hints(
    blocks: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    hints: list[dict[str, object]] = []
    for block in blocks:
        hint: dict[str, object] = {
            "block_index": block.get("block_index"),
            "block_type": block.get("block_type"),
            "placement": block.get("placement"),
            "interrupts_source": block.get("interrupts_source"),
            "source_role": _block_source_role(block),
        }
        timeline = block.get("timeline")
        if isinstance(timeline, Mapping):
            hint["timeline"] = {
                "start": timeline.get("start"),
                "end": timeline.get("end"),
                "authoritative": timeline.get("authoritative"),
            }
        hints.append(hint)
    return hints


def _block_source_role(block: Mapping[str, object]) -> object:
    source_role = block.get("source_role")
    if isinstance(source_role, str) and source_role:
        return source_role
    binding = block.get("source_binding")
    if isinstance(binding, Mapping):
        nested = binding.get("source_role")
        if isinstance(nested, str):
            return nested
    return source_role


def _word_timestamps(
    caption_input: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    raw = caption_input.get("word_timestamps")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(cast(Mapping[str, object], item) for item in raw if isinstance(item, Mapping))


def _authored_assets_pending(slots: Sequence[Mapping[str, object]]) -> bool:
    return any(_slot_unmaterialized(slot) for slot in slots)


__all__ = ["build_visual_composition_plan"]
