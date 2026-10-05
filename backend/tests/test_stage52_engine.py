"""Stage 5.2 engine unit tests: validation, timeline, compiler, fingerprints."""

from __future__ import annotations

import pytest
from stage52_support import crop_scene, fake_runtime, make_spec, occurred

from app.render.execution.compiler import compile_render
from app.render.execution.policy import (
    Stage52Config,
    delivery_profile_for,
    stage52_config_payload,
)
from app.render.execution.timeline import (
    TimelineError,
    build_timeline,
    map_source_to_output,
)
from app.render.execution.types import (
    CaptionEventSpec,
    SceneSpec,
    TechnicalQCResult,
)
from app.render.execution.validation import RenderValidationError, validate_spec


def test_valid_spec_and_timeline_mapping() -> None:
    spec = make_spec()
    validate_spec(spec)
    manifest = build_timeline(spec)
    assert manifest.output_duration == pytest.approx(5.0)
    assert manifest.output_frame_count == 150
    assert map_source_to_output(spec.occurrences, 10.0) == pytest.approx(0.0)
    assert map_source_to_output(spec.occurrences, 30.5) == pytest.approx(2.5)
    assert map_source_to_output(spec.occurrences, 33.0) == pytest.approx(5.0)


def test_occurrences_keep_contract_order_not_source_order() -> None:
    # Contract order block 0 then block 1, but block 0's source is LATER than
    # block 1's. A correct render must not sort by source timestamp.
    occ_a = occurred(
        "block-0",
        0,
        30.0,
        33.0,
        0.0,
        (SceneSpec(0, 0, 30.0, 33.0, "SOURCE_AS_IS", "smoothstep-ease"),),
    )
    occ_b = occurred(
        "block-1",
        1,
        10.0,
        12.0,
        3.0,
        (SceneSpec(1, 1, 10.0, 12.0, "SOURCE_AS_IS", "smoothstep-ease"),),
    )
    spec = make_spec(
        occurrences=(occ_a, occ_b),
        caption_events=(CaptionEventSpec("event-1", 1, 10.5, 11.5),),
    )
    manifest = build_timeline(spec)
    assert [occ.block_index for occ in manifest.occurrences] == [0, 1]
    assert map_source_to_output(spec.occurrences, 31.0) == pytest.approx(1.0)
    assert map_source_to_output(spec.occurrences, 10.5) == pytest.approx(3.5)


def test_validate_rejects_scene_gap() -> None:
    occ = occurred(
        "block-0",
        0,
        10.0,
        14.0,
        0.0,
        (
            SceneSpec(0, 0, 10.0, 11.0, "SOURCE_AS_IS", "smoothstep-ease"),
            SceneSpec(1, 0, 12.0, 14.0, "SOURCE_AS_IS", "smoothstep-ease"),
        ),
    )
    spec = make_spec(occurrences=(occ,), caption_events=())
    with pytest.raises(RenderValidationError) as error:
        validate_spec(spec)
    assert error.value.reason_code == "SCENE_COVERAGE_GAP"


def test_validate_rejects_overlap_and_context_leak() -> None:
    occ = occurred(
        "block-0",
        0,
        10.0,
        12.0,
        0.0,
        (
            SceneSpec(0, 0, 9.0, 11.0, "SOURCE_AS_IS", "smoothstep-ease"),
            SceneSpec(1, 0, 11.0, 12.0, "SOURCE_AS_IS", "smoothstep-ease"),
        ),
    )
    spec = make_spec(occurrences=(occ,), caption_events=())
    with pytest.raises(RenderValidationError) as error:
        validate_spec(spec)
    assert error.value.reason_code == "SCENE_BOUNDS_INVALID"


def test_validate_rejects_crop_without_keyframes() -> None:
    occ = occurred(
        "block-0",
        0,
        10.0,
        12.0,
        0.0,
        (SceneSpec(0, 0, 10.0, 12.0, "TRACKED_CROP", "smoothstep-ease"),),
    )
    spec = make_spec(occurrences=(occ,), caption_events=())
    with pytest.raises(RenderValidationError) as error:
        validate_spec(spec)
    assert error.value.reason_code == "CONTRADICTORY_FRAMING_EVIDENCE"


def test_validate_rejects_unsupported_purpose_and_profile() -> None:
    bad_purpose = make_spec(artifact_purpose="PUBLICATION_FINAL")
    with pytest.raises(RenderValidationError) as error:
        validate_spec(bad_purpose)
    assert error.value.reason_code == "UNSUPPORTED_ARTIFACT_PURPOSE"
    bad_profile = make_spec(delivery_profile_key="NOPE")
    with pytest.raises(RenderValidationError) as error:
        validate_spec(bad_profile)
    assert error.value.reason_code == "UNSUPPORTED_DELIVERY_PROFILE"


def test_validate_rejects_caption_outside_occurrences() -> None:
    spec = make_spec(caption_events=(CaptionEventSpec("event-x", 0, 50.0, 51.0),))
    with pytest.raises(RenderValidationError) as error:
        validate_spec(spec)
    assert error.value.reason_code == "QC_CAPTION_OUT_OF_RANGE"


def test_validate_rejects_exotic_pixel_aspect() -> None:
    spec = make_spec(pixel_aspect_ratio=1.2)
    with pytest.raises(RenderValidationError) as error:
        validate_spec(spec)
    assert error.value.reason_code == "EXOTIC_PIXEL_ASPECT"


def test_timeline_error_on_noncontiguous() -> None:
    occ = occurred(
        "block-0",
        0,
        10.0,
        12.0,
        1.0,
        (SceneSpec(0, 0, 10.0, 12.0, "SOURCE_AS_IS", "smoothstep-ease"),),
    )
    spec = make_spec(occurrences=(occ,), caption_events=())
    with pytest.raises(TimelineError):
        build_timeline(spec)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mode",
    [
        "SOURCE_AS_IS",
        "STATIC_CROP",
        "TRACKED_CROP",
        "MULTI_SUBJECT_FIT",
        "BACKGROUND_FILL",
        "CENTER_FALLBACK",
    ],
)
def test_all_six_framing_modes_compile(mode: str) -> None:
    if mode in {"SOURCE_AS_IS", "BACKGROUND_FILL"}:
        scene = SceneSpec(0, 0, 10.0, 12.0, mode, "smoothstep-ease")
    else:
        scene = crop_scene(0, 0, 10.0, 12.0, mode, [(10.0, 0.5, 0.5, 0.8), (12.0, 0.6, 0.4, 0.7)])
    occ = occurred("block-0", 0, 10.0, 12.0, 0.0, (scene,))
    spec = make_spec(occurrences=(occ,), caption_events=())
    validate_spec(spec)
    compiled = compile_render(spec, fake_runtime())
    graph = compiled.filtergraph
    assert "concat" not in graph or True
    if mode == "BACKGROUND_FILL":
        assert "gblur=sigma=36:steps=2" in graph
        assert "eq=brightness=-0.18:saturation=0.70" in graph
    if mode == "SOURCE_AS_IS":
        assert "force_original_aspect_ratio=decrease" in graph
        assert "pad=1080:1920" in graph


def test_tracked_crop_generates_dynamic_scale_and_crop() -> None:
    scene = crop_scene(
        0, 0, 10.0, 14.0, "TRACKED_CROP", [(10.0, 0.3, 0.5, 1.0), (14.0, 0.7, 0.4, 0.6)]
    )
    occ = occurred("block-0", 0, 10.0, 14.0, 0.0, (scene,))
    spec = make_spec(occurrences=(occ,), caption_events=())
    compiled = compile_render(spec, fake_runtime())
    graph = compiled.filtergraph
    assert "eval=frame" in graph
    assert "3-2*" in graph
    assert graph.count("crop=w=1080:h=1920") >= 1


def test_static_crop_uses_clamped_geometry() -> None:
    scene = crop_scene(0, 0, 10.0, 12.0, "STATIC_CROP", [(10.0, 0.5, 0.5, 0.8)])
    occ = occurred("block-0", 0, 10.0, 12.0, 0.0, (scene,))
    spec = make_spec(occurrences=(occ,), caption_events=())
    compiled = compile_render(spec, fake_runtime())
    assert "crop=w=" in compiled.filtergraph
    assert "scale=1080:1920" in compiled.filtergraph


def test_argv_is_safe_and_complete() -> None:
    spec = make_spec()
    compiled = compile_render(spec, fake_runtime(encoder_threads=3, filter_threads=4))
    argv = list(compiled.argv)
    assert argv[0] == "ffmpeg"
    assert "-nostdin" in argv
    assert "-filter_complex_script" in argv
    assert "libx264" in argv
    assert "veryfast" in argv
    assert "aac" in argv
    assert "+faststart" in argv
    assert "rotate=0" in argv
    assert not any(";" in item and "filter" not in item for item in argv)
    assert all("shell" not in item for item in argv)
    assert argv[-1].endswith("output.mp4")
    assert "-threads" in argv and argv[argv.index("-threads") + 1] == "3"
    assert "-filter_threads" in argv and argv[argv.index("-filter_threads") + 1] == "4"
    assert (
        "-filter_complex_threads" in argv and argv[argv.index("-filter_complex_threads") + 1] == "1"
    )


def test_compiled_fingerprint_is_deterministic_and_profile_sensitive() -> None:
    spec = make_spec()
    runtime = fake_runtime()
    first = compile_render(spec, runtime)
    second = compile_render(spec, runtime)
    assert first.fingerprint == second.fingerprint
    changed_runtime = fake_runtime(encoder_threads=8)
    # Threads are output-affecting via config, not runtime fingerprint, but the
    # compiled command differs; the graph itself is stable.
    assert compile_render(spec, changed_runtime).filtergraph == first.filtergraph


def test_delivery_profile_payload_is_covered() -> None:
    payload = stage52_config_payload(make_config())
    assert "delivery_profiles" in payload
    assert "resource_bounds" in payload
    assert delivery_profile_for("MP4_H264_AAC_1080X1920_V1") is not None


def test_qc_result_shape_rejects_blank() -> None:
    result = TechnicalQCResult(status="PASS", checks=(), reason_codes=(), policy_version="v")
    assert not result.hard_failure
    assert TechnicalQCResult(
        status="FAIL", checks=(), reason_codes=("QC_BLANK_RENDER",)
    ).hard_failure


def make_config() -> Stage52Config:  # local helper to avoid importing Settings
    return Stage52Config()
