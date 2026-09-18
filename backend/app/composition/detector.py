"""Vendored CPU face detection for Stage 5.1 visual composition.

This module is deliberately small and provider-free:

- it runs the OpenCV Zoo YuNet 2023mar ONNX model through the already installed
  ``onnxruntime`` CPU execution provider (no OpenCV, no MediaPipe, no network);
- it never performs recognition, embedding, biometric matching, or named-person
  identification. A detection is only an anonymous box plus a confidence score;
- model-availability problems never raise out of ``detect()``: the detector
  degrades to an empty result while ``ready()`` reports the truth.

The vendored model has a *fixed* 640x640 input shape (empirically verified: a
320x320 tensor raises INVALID_ARGUMENT), which is why the policy default is 640.

Preprocessing and postprocessing are split so postprocessing is a pure,
unit-testable function:

- ``letterbox`` maps an RGB uint8 HxWx3 frame to a deterministic letterboxed
  BGR float32 NCHW blob scaled to the model's expected [0, 255] range and
  returns the exact scale/pad needed to map detection coordinates back to
  original frame pixels;
- ``decode_yunet`` decodes the raw model outputs with hand-built arrays and no
  hidden state.

Empirical note (verified against the vendored export): the exported graph
already applies sigmoid to the classification and objectness logits, so the raw
``cls_*``/``obj_*`` outputs arrive inside [0, 1] and must NOT be passed through
another sigmoid. The score is the geometric mean of the two.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from app.composition.policy import DETECTOR_IDENTITY, Stage51Config
from app.composition.types import FaceDetection

_logger = logging.getLogger(__name__)

Frame: TypeAlias = npt.NDArray[np.uint8]
Blob: TypeAlias = npt.NDArray[np.float32]
OutputArrays: TypeAlias = dict[str, npt.NDArray[Any]]

_MODEL_INPUT_NAME = "input"
_STRIDES = (8, 16, 32)
_INTRA_OP_THREADS = 2
_INTER_OP_THREADS = 1

_SESSION_LOCK = threading.Lock()
_SESSION_CACHE: dict[tuple[str, int], Any] = {}


class FaceDetector(Protocol):
    """Anonymous face detector boundary used by the Stage 5.1 analyzer."""

    def detect(self, rgb_frame: Frame) -> tuple[FaceDetection, ...]:
        """Return anonymous detections in display-normalized coordinates."""

    def identity(self) -> dict[str, object]:
        """Return a deterministic, fingerprintable detector identity."""

    def ready(self) -> bool:
        """Return True only when detection can actually run."""


class NullFaceDetector:
    """Unavailable detector that never raises and never returns a detection."""

    def detect(self, rgb_frame: Frame) -> tuple[FaceDetection, ...]:
        return ()

    def identity(self) -> dict[str, object]:
        return {"name": "null", "available": False}

    def ready(self) -> bool:
        return False


def _load_onnxruntime() -> Any | None:
    try:
        import onnxruntime  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - exercised only without the runtime
        return None
    return onnxruntime


def _sha256_file(path: str) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _resize_bilinear(image: Frame, out_h: int, out_w: int) -> Blob:
    """Deterministic half-pixel bilinear resize using numpy only."""

    in_h, in_w = int(image.shape[0]), int(image.shape[1])
    if in_h == out_h and in_w == out_w:
        return cast(Blob, image.astype(np.float32))

    rows = (np.arange(out_h, dtype=np.float64) + 0.5) * (in_h / out_h) - 0.5
    cols = (np.arange(out_w, dtype=np.float64) + 0.5) * (in_w / out_w) - 0.5
    rows = np.clip(rows, 0.0, in_h - 1)
    cols = np.clip(cols, 0.0, in_w - 1)

    row0 = np.floor(rows).astype(np.intp)
    col0 = np.floor(cols).astype(np.intp)
    row1 = np.minimum(row0 + 1, in_h - 1)
    col1 = np.minimum(col0 + 1, in_w - 1)
    weight_r = (rows - row0)[:, None, None]
    weight_c = (cols - col0)[None, :, None]

    source = image.astype(np.float32)
    top = source[row0][:, col0] * (1.0 - weight_c) + source[row0][:, col1] * weight_c
    bottom = source[row1][:, col0] * (1.0 - weight_c) + source[row1][:, col1] * weight_c
    return cast(Blob, (top * (1.0 - weight_r) + bottom * weight_r).astype(np.float32))


def letterbox(frame: Frame, size: int) -> tuple[Blob, float, int, int]:
    """Pure deterministic letterbox: RGB HxWx3 uint8 -> BGR NCHW float32 blob.

    Aspect ratio is preserved and the remaining area is zero-padded. Values are
    kept in the model's raw [0, 255] BGR range (this YuNet export expects raw
    values, not a /255 normalization). Returns the blob plus the horizontal and
    vertical padding and the uniform scale so the caller can map letterbox
    coordinates back to original frame pixels.
    """

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("letterbox expects an HxWx3 RGB frame")

    frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
    scale = min(size / frame_w, size / frame_h)
    resized_w = max(1, min(size, int(round(frame_w * scale))))
    resized_h = max(1, min(size, int(round(frame_h * scale))))
    pad_x = (size - resized_w) // 2
    pad_y = (size - resized_h) // 2

    resized = _resize_bilinear(frame, resized_h, resized_w)
    blob: Blob = np.zeros((1, 3, size, size), dtype=np.float32)
    # The vendored YuNet 2023mar export expects raw BGR values in [0, 255].
    # Empirically measured against this exact model: a /255 blob collapses the
    # objectness head to ~0 and yields zero detections, while raw 0..255 BGR
    # yields correct boxes. Do NOT normalize here.
    blob[0, 0, pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized[:, :, 2]
    blob[0, 1, pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized[:, :, 1]
    blob[0, 2, pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized[:, :, 0]
    return blob, float(scale), int(pad_x), int(pad_y)


def _iou_box(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
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


def _stride_candidates(
    outputs: OutputArrays,
    stride: int,
    input_size: int,
    score_threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_w: int,
    frame_h: int,
) -> list[tuple[float, float, float, float, float]]:
    cls = outputs.get(f"cls_{stride}")
    obj = outputs.get(f"obj_{stride}")
    bbox = outputs.get(f"bbox_{stride}")
    if cls is None or obj is None or bbox is None:
        return []

    grid = input_size // stride
    count = grid * grid
    if count <= 0:
        return []

    cls_flat = np.asarray(cls, dtype=np.float64).reshape(-1)[:count]
    obj_flat = np.asarray(obj, dtype=np.float64).reshape(-1)[:count]
    boxes = np.asarray(bbox, dtype=np.float64).reshape(-1, 4)[:count]

    clipped_cls = np.clip(cls_flat, 0.0, 1.0)
    clipped_obj = np.clip(obj_flat, 0.0, 1.0)
    scores = np.sqrt(clipped_cls * clipped_obj)

    indices: npt.NDArray[np.intp] = np.arange(count, dtype=np.intp)
    cols: npt.NDArray[np.intp] = indices % grid
    rows: npt.NDArray[np.intp] = indices // grid

    centers_x = (cols + boxes[:, 0]) * float(stride)
    centers_y = (rows + boxes[:, 1]) * float(stride)
    widths = np.exp(boxes[:, 2]) * float(stride)
    heights = np.exp(boxes[:, 3]) * float(stride)

    candidates: list[tuple[float, float, float, float, float]] = []
    selected = np.nonzero(scores >= score_threshold)[0]
    for index in selected:
        half_w = float(widths[index]) / 2.0
        half_h = float(heights[index]) / 2.0
        # Map letterbox coordinates back to original frame pixels.
        x1 = (float(centers_x[index]) - half_w - pad_x) / scale
        y1 = (float(centers_y[index]) - half_h - pad_y) / scale
        x2 = (float(centers_x[index]) + half_w - pad_x) / scale
        y2 = (float(centers_y[index]) + half_h - pad_y) / scale
        x1 = min(max(x1, 0.0), float(frame_w))
        y1 = min(max(y1, 0.0), float(frame_h))
        x2 = min(max(x2, 0.0), float(frame_w))
        y2 = min(max(y2, 0.0), float(frame_h))
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append((float(scores[index]), x1, y1, x2, y2))
    return candidates


def decode_yunet(
    outputs: OutputArrays,
    input_size: int,
    score_threshold: float,
    nms_iou: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_w: int,
    frame_h: int,
) -> tuple[FaceDetection, ...]:
    """Pure YuNet decode + deterministic NMS.

    Outputs are the raw ``cls_*``/``obj_*``/``bbox_*`` tensors. Deltas are decoded
    per prior cell with an anchor-free YuNet layout, scores are the geometric mean
    of the (already sigmoid-activated) class and objectness values, boxes are
    mapped back to original pixels, and NMS keeps the highest-scoring candidate
    while suppressing boxes whose IoU exceeds ``nms_iou``.
    """

    if frame_w <= 0 or frame_h <= 0 or scale <= 0.0:
        return ()

    candidates: list[tuple[float, float, float, float, float]] = []
    for stride in _STRIDES:
        candidates.extend(
            _stride_candidates(
                outputs,
                stride,
                input_size,
                score_threshold,
                scale,
                pad_x,
                pad_y,
                frame_w,
                frame_h,
            )
        )

    # Deterministic ordering: highest score, then top-left position.
    ordered = sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
    kept: list[tuple[float, float, float, float, float]] = []
    for candidate in ordered:
        _, x1, y1, x2, y2 = candidate
        box = (x1, y1, x2, y2)
        if any(_iou_box(box, (k[1], k[2], k[3], k[4])) > nms_iou for k in kept):
            continue
        kept.append(candidate)

    detections = tuple(
        FaceDetection(
            x=x1 / frame_w,
            y=y1 / frame_h,
            w=(x2 - x1) / frame_w,
            h=(y2 - y1) / frame_h,
            score=score,
        )
        for score, x1, y1, x2, y2 in kept
    )
    return detections


def _create_session(model_path: str, input_size: int, intra_op_threads: int) -> Any | None:
    runtime = _load_onnxruntime()
    if runtime is None:
        return None
    expected = DETECTOR_IDENTITY.get("sha256")
    if not isinstance(expected, str) or _sha256_file(model_path) != expected:
        return None
    try:
        options = runtime.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = _INTER_OP_THREADS
        return runtime.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception:  # pragma: no cover - defensive: availability, not correctness
        _logger.warning("failed to create YuNet session for %s", model_path, exc_info=True)
        return None


def _get_session(model_path: str, input_size: int, intra_op_threads: int) -> Any | None:
    key = (model_path, input_size)
    with _SESSION_LOCK:
        if key in _SESSION_CACHE:
            return _SESSION_CACHE[key]
        session = _create_session(model_path, input_size, intra_op_threads)
        _SESSION_CACHE[key] = session
        return session


class OnnxYuNetDetector:
    """Lazy CPU YuNet detector backed by one process-wide ONNX session."""

    def __init__(
        self,
        model_path: str,
        input_size: int,
        score_threshold: float,
        nms_iou: float,
        intra_op_threads: int = _INTRA_OP_THREADS,
    ) -> None:
        self._model_path = model_path
        self._input_size = input_size
        self._score_threshold = score_threshold
        self._nms_iou = nms_iou
        self._intra_op_threads = intra_op_threads

    def _session(self) -> Any | None:
        if not self._model_path:
            return None
        return _get_session(self._model_path, self._input_size, self._intra_op_threads)

    def ready(self) -> bool:
        try:
            return self._session() is not None
        except Exception:  # pragma: no cover - availability must never raise
            return False

    def identity(self) -> dict[str, object]:
        return {
            "name": DETECTOR_IDENTITY.get("name", "opencv_zoo_face_detection_yunet"),
            "model": os.path.basename(self._model_path) if self._model_path else "",
            "sha256": DETECTOR_IDENTITY.get("sha256"),
            "input_size": self._input_size,
            "score_threshold": self._score_threshold,
            "nms_iou": self._nms_iou,
            "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": self._intra_op_threads,
            "available": self.ready(),
        }

    def detect(self, rgb_frame: Frame) -> tuple[FaceDetection, ...]:
        try:
            if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
                return ()
            frame_h, frame_w = int(rgb_frame.shape[0]), int(rgb_frame.shape[1])
            if frame_w <= 0 or frame_h <= 0:
                return ()
            session = self._session()
            if session is None:
                return ()
            blob, scale, pad_x, pad_y = letterbox(rgb_frame, self._input_size)
            raw_outputs = session.run(None, {_MODEL_INPUT_NAME: blob})
            outputs: OutputArrays = {
                output.name: array
                for output, array in zip(session.get_outputs(), raw_outputs, strict=True)
            }
            return decode_yunet(
                outputs,
                self._input_size,
                self._score_threshold,
                self._nms_iou,
                scale,
                pad_x,
                pad_y,
                frame_w,
                frame_h,
            )
        except Exception:
            _logger.warning("face detection failed", exc_info=True)
            return ()


def detector_from_config(config: Stage51Config) -> FaceDetector:
    """Build the configured detector or a safe null detector."""

    if not config.detector_enabled:
        return NullFaceDetector()
    model_path = config.detector_model_path
    if not model_path or not os.path.isfile(model_path):
        return NullFaceDetector()
    expected = DETECTOR_IDENTITY.get("sha256")
    if not isinstance(expected, str) or _sha256_file(model_path) != expected:
        return NullFaceDetector()
    return OnnxYuNetDetector(
        model_path=model_path,
        input_size=config.detector_input_size,
        score_threshold=config.detector_score_threshold,
        nms_iou=config.detector_nms_iou,
    )


__all__ = [
    "FaceDetector",
    "NullFaceDetector",
    "OnnxYuNetDetector",
    "decode_yunet",
    "detector_from_config",
    "letterbox",
]
