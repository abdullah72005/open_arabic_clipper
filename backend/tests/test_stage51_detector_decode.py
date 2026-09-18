"""Pure Stage 5.1 face-detector preprocessing/decoding tests (no model needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from app.composition.detector import (
    NullFaceDetector,
    decode_yunet,
    detector_from_config,
    letterbox,
)
from app.composition.policy import Stage51Config
from app.composition.types import FaceDetection


def _yunet_outputs(count: int) -> dict[str, npt.NDArray[Any]]:
    return {
        "cls_8": np.zeros((1, count, 1), dtype=np.float32),
        "obj_8": np.zeros((1, count, 1), dtype=np.float32),
        "bbox_8": np.zeros((1, count, 4), dtype=np.float32),
    }


def _set_candidate(
    outputs: dict[str, npt.NDArray[Any]],
    index: int,
    cls: float,
    obj: float,
    deltas: tuple[float, ...],
) -> None:
    outputs["cls_8"][0, index, 0] = cls
    outputs["obj_8"][0, index, 0] = obj
    outputs["bbox_8"][0, index, :] = deltas


def test_letterbox_shape_scale_padding_and_purity() -> None:
    frame: npt.NDArray[np.uint8] = np.zeros((50, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # pure red in RGB

    blob, scale, pad_x, pad_y = letterbox(frame, 64)

    assert blob.shape == (1, 3, 64, 64)
    assert blob.dtype == np.float32
    assert scale == pytest.approx(0.64)
    assert pad_x == 0
    assert pad_y == 16
    assert float(blob.min()) >= 0.0
    assert float(blob.max()) <= 255.0
    # The vendored export requires raw [0, 255] values; a /255 normalization
    # silently collapses objectness to zero. Pin the raw range explicitly.
    assert float(blob.max()) > 1.0
    # BGR order: red frame has zero blue/green channels and a full red channel.
    assert float(blob[0, 0].max()) == pytest.approx(0.0)
    assert float(blob[0, 1].max()) == pytest.approx(0.0)
    assert float(blob[0, 2].max()) == pytest.approx(255.0)

    again, _, _, _ = letterbox(frame, 64)
    assert np.array_equal(blob, again)


def test_letterbox_pads_non_square_frame_with_zeros() -> None:
    frame: npt.NDArray[np.uint8] = np.full((50, 100, 3), 200, dtype=np.uint8)
    blob, _, _, pad_y = letterbox(frame, 64)
    assert pad_y == 16
    assert np.all(blob[0, :, :pad_y, :] == 0.0)
    assert np.all(blob[0, :, 64 - pad_y :, :] == 0.0)


def test_decode_positive_candidate_maps_back_to_frame_pixels() -> None:
    frame: npt.NDArray[np.uint8] = np.zeros((50, 100, 3), dtype=np.uint8)
    _, scale, pad_x, pad_y = letterbox(frame, 64)

    outputs = _yunet_outputs(64)
    # Cell (col=4, row=5) at stride 8 -> letterbox center (32, 40), 8x8 box.
    _set_candidate(outputs, 5 * 8 + 4, cls=0.81, obj=0.64, deltas=(0.0, 0.0, 0.0, 0.0))

    detections = decode_yunet(
        outputs,
        input_size=64,
        score_threshold=0.6,
        nms_iou=0.3,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        frame_w=100,
        frame_h=50,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.x == pytest.approx(0.4375, abs=1e-6)
    assert detection.y == pytest.approx(0.625, abs=1e-6)
    assert detection.w == pytest.approx(0.125, abs=1e-6)
    assert detection.h == pytest.approx(0.25, abs=1e-6)
    assert detection.score == pytest.approx(0.72, abs=1e-6)


def test_decode_threshold_rejects_low_score_candidate() -> None:
    outputs = _yunet_outputs(64)
    _set_candidate(outputs, 44, cls=0.2, obj=0.2, deltas=(0.0, 0.0, 0.0, 0.0))

    detections = decode_yunet(
        outputs,
        input_size=64,
        score_threshold=0.6,
        nms_iou=0.3,
        scale=0.64,
        pad_x=0,
        pad_y=16,
        frame_w=100,
        frame_h=50,
    )

    assert detections == ()


def test_decode_nms_suppresses_overlapping_duplicate() -> None:
    outputs = _yunet_outputs(64)
    # Adjacent cells with 16x16 boxes overlap by IoU ~= 0.33 > 0.3.
    _set_candidate(
        outputs,
        5 * 8 + 4,
        cls=1.0,
        obj=1.0,
        deltas=(0.0, 0.0, 0.6931471805599453, 0.6931471805599453),
    )
    _set_candidate(
        outputs,
        5 * 8 + 5,
        cls=0.81,
        obj=1.0,
        deltas=(0.0, 0.0, 0.6931471805599453, 0.6931471805599453),
    )

    detections = decode_yunet(
        outputs,
        input_size=64,
        score_threshold=0.6,
        nms_iou=0.3,
        scale=0.64,
        pad_x=0,
        pad_y=16,
        frame_w=100,
        frame_h=50,
    )

    assert len(detections) == 1
    assert detections[0].score == pytest.approx(1.0, abs=1e-6)
    assert detections[0].x == pytest.approx(0.375, abs=1e-6)
    assert detections[0].w == pytest.approx(0.25, abs=1e-6)


def test_decode_is_deterministic_for_identical_inputs() -> None:
    outputs = _yunet_outputs(64)
    _set_candidate(outputs, 44, cls=0.9, obj=0.9, deltas=(0.0, 0.0, 0.0, 0.0))

    def _run() -> tuple[FaceDetection, ...]:
        return decode_yunet(
            outputs,
            input_size=64,
            score_threshold=0.6,
            nms_iou=0.3,
            scale=0.64,
            pad_x=0,
            pad_y=16,
            frame_w=100,
            frame_h=50,
        )

    assert _run() == _run()


def test_null_detector_never_reports_or_detects() -> None:
    detector = NullFaceDetector()
    assert detector.ready() is False
    assert detector.identity() == {"name": "null", "available": False}
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == ()


def test_detector_from_config_returns_null_when_disabled() -> None:
    detector = detector_from_config(Stage51Config(detector_enabled=False))
    assert isinstance(detector, NullFaceDetector)
    assert detector.ready() is False


def test_detector_from_config_returns_null_for_missing_path() -> None:
    config = Stage51Config(detector_enabled=True, detector_model_path="/nonexistent/model.onnx")
    detector = detector_from_config(config)
    assert isinstance(detector, NullFaceDetector)


def test_detector_from_config_returns_null_for_sha_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "fake.onnx"
    path.write_bytes(b"not the vendored model")
    config = Stage51Config(detector_enabled=True, detector_model_path=str(path))
    assert isinstance(detector_from_config(config), NullFaceDetector)


def test_decode_handles_non_square_frame_in_other_orientation() -> None:
    # Portrait frame: 100x50 -> scale 0.64, horizontal padding.
    frame: npt.NDArray[np.uint8] = np.zeros((100, 50, 3), dtype=np.uint8)
    _, scale, pad_x, pad_y = letterbox(frame, 64)
    assert scale == pytest.approx(0.64)
    assert pad_x == 16
    assert pad_y == 0

    outputs = _yunet_outputs(64)
    # Cell (col=5, row=4): letterbox center (40, 32).
    _set_candidate(outputs, 4 * 8 + 5, cls=1.0, obj=1.0, deltas=(0.0, 0.0, 0.0, 0.0))
    detections = decode_yunet(
        outputs,
        input_size=64,
        score_threshold=0.6,
        nms_iou=0.3,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        frame_w=50,
        frame_h=100,
    )
    assert len(detections) == 1
    # x1 = (36 - 16) / 0.64 = 31.25 ; y1 = (28 - 0) / 0.64 = 43.75.
    assert detections[0].x == pytest.approx(0.625, abs=1e-6)
    assert detections[0].y == pytest.approx(0.4375, abs=1e-6)


def test_face_detection_is_anonymous_geometry_only() -> None:
    detection = FaceDetection(x=0.1, y=0.2, w=0.3, h=0.4, score=0.9)
    assert set(detection.as_dict()) == {"x", "y", "w", "h", "score"}
