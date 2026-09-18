"""Gated real-model Stage 5.1 detector test.

This runs by default in the Docker test image (onnxruntime and the vendored
model are present). It skips with a clear reason only when genuinely absent.
No face is asserted on synthetic noise; the contract is that the real model is
loadable, reports its verified identity, and returns a deterministic tuple.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from app.composition.detector import detector_from_config
from app.composition.policy import DETECTOR_IDENTITY, Stage51Config
from app.composition.types import FaceDetection

pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")

_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "composition"
    / "assets"
    / "face_detection_yunet_2023mar.onnx"
)


def _synthetic_frame() -> npt.NDArray[np.uint8]:
    """Deterministic gradient plus seeded noise; no real face is implied."""

    height, width = 360, 640
    generator = np.random.default_rng(20250918)
    noise = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    gradient_x: npt.NDArray[np.uint8] = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
    gradient_y: npt.NDArray[np.uint8] = np.linspace(0, 255, height, dtype=np.uint8)[:, None, None]
    blended = 0.6 * noise.astype(np.float64) + 0.2 * gradient_x + 0.2 * gradient_y
    return cast(npt.NDArray[np.uint8], np.clip(blended, 0, 255).astype(np.uint8))


def test_real_yunet_model_loads_and_detects_without_raising() -> None:
    if not _MODEL_PATH.is_file():
        pytest.skip(f"vendored YuNet model is absent at {_MODEL_PATH}")

    config = Stage51Config(
        detector_enabled=True,
        detector_model_path=str(_MODEL_PATH),
        detector_input_size=640,
        detector_score_threshold=float(DETECTOR_IDENTITY["score_threshold"]),
        detector_nms_iou=float(DETECTOR_IDENTITY["nms_iou"]),
    )
    detector = detector_from_config(config)

    assert detector.ready() is True
    identity = detector.identity()
    assert identity["sha256"] == DETECTOR_IDENTITY["sha256"]
    assert identity["input_size"] == DETECTOR_IDENTITY["input_size"] == 640

    frame = _synthetic_frame()
    first = detector.detect(frame)
    second = detector.detect(frame)

    assert isinstance(first, tuple)
    assert all(isinstance(detection, FaceDetection) for detection in first)
    assert isinstance(len(first), int)
    assert len(first) >= 0
    assert first == second


def test_real_yunet_input_is_fixed_at_640() -> None:
    if not _MODEL_PATH.is_file():
        pytest.skip(f"vendored YuNet model is absent at {_MODEL_PATH}")

    config = Stage51Config(
        detector_enabled=True,
        detector_model_path=str(_MODEL_PATH),
        detector_input_size=640,
        detector_score_threshold=0.6,
        detector_nms_iou=0.3,
    )
    detector = detector_from_config(config)
    assert detector.ready() is True
    # A deterministic blank frame is enough to exercise the fixed [1,3,640,640]
    # graph end to end; the 320x320 input is known to raise INVALID_ARGUMENT.
    blank: npt.NDArray[np.uint8] = np.zeros((240, 240, 3), dtype=np.uint8)
    result = detector.detect(blank)
    assert isinstance(result, tuple)
    assert isinstance(len(result), int)
