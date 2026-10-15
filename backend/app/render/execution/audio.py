"""Deterministic lossless audio preparation and mastering for authored renders.

Stage 6 owns the explicit tracks, gains, and mastering policy; this module builds
the typed, lossless media path that the frozen runner executes:

    explicit mix (float, 48 kHz stereo) -> measurement -> feasible linear
    mastering -> lossless PCM master -> one final H.264/AAC encode.

Nothing here decides provider/renderer semantics: the mix graph is derived from
typed instructions, the mastering rule is a pure function of measured values, and
the master is a deterministic linear application. No dynamic ``loudnorm`` fallback,
per-asset normalization, or stacked lossy generation is ever introduced.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.render.execution.policy import (
    AUDIO_MASTERING_INFEASIBLE,
    AUDIO_MEASUREMENT_INVALID,
    AUDIO_PEAK_REPAIR_NEEDED,
    AUTHORED_AUDIO_RECIPE_UNSUPPORTED,
)

UNMASTERED_RELATIVE_NAME = "mix-unmastered.wav"
MASTER_RELATIVE_NAME = "mix-master.wav"
MASTER_MIX_FILTERGRAPH_NAME = "authored-mix-filtergraph.txt"
AUDIO_SAMPLE_FORMAT = "pcm_f32le"

#: The measured input LRA is preserved with this bounded, versioned LRA target.
_LOUDNORM_LRA_MIN = 1.0
_LOUDNORM_LRA_MAX = 50.0

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


class AudioPreparationError(RuntimeError):
    """A deterministic audio recipe or mastering step could not be executed."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(reason_code if not detail else f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class LoudnessMeasurement:
    """The measured facts of one loudnorm analysis pass."""

    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float
    normalization_type: str

    def as_dict(self) -> dict[str, object]:
        return {
            "input_i": round(self.input_i, 6),
            "input_tp": round(self.input_tp, 6),
            "input_lra": round(self.input_lra, 6),
            "input_thresh": round(self.input_thresh, 6),
            "target_offset": round(self.target_offset, 6),
            "normalization_type": self.normalization_type,
        }


@dataclass(frozen=True)
class MasteringDecision:
    """The deterministic (mechanical) linear mastering decision."""

    feasible: bool
    reason_code: str | None
    applied_gain_db: float
    target_lufs: float
    true_peak_dbtp: float
    lra: float
    predicted_lufs: float
    mode: str = "LINEAR"

    def as_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "reason_code": self.reason_code,
            "applied_gain_db": round(self.applied_gain_db, 6),
            "target_lufs": round(self.target_lufs, 6),
            "true_peak_dbtp": round(self.true_peak_dbtp, 6),
            "lra": round(self.lra, 6),
            "predicted_lufs": round(self.predicted_lufs, 6),
            "mode": self.mode,
        }


@dataclass(frozen=True)
class AudioPreparationPlan:
    """Typed pre-encode audio plan returned by the authored compiler.

    The runner executes ``mix_argv`` (producing ``unmastered_name``), measures it,
    selects the mechanical mastering decision, and executes the mastering command
    producing ``master_name``; the final encode command maps ``master_name``.
    """

    mix_argv: tuple[str, ...]
    mix_filtergraph: str
    mix_filtergraph_name: str
    unmastered_name: str
    master_name: str
    mastering: Any
    sample_count: int
    sample_rate: int
    source_relative_paths: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Numeric formatting
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


# ---------------------------------------------------------------------------
# Loudnorm measurement
# ---------------------------------------------------------------------------


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def parse_loudnorm_output(text: str) -> LoudnessMeasurement | None:
    """Extract the loudnorm JSON block from mixed ffmpeg output (last block wins)."""

    if not text:
        return None
    parsed: Mapping[str, Any] | None = None
    for match in _JSON_BLOCK_RE.finditer(text):
        try:
            candidate = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, Mapping):
            parsed = candidate
    if parsed is None:
        return None
    input_i = _coerce_float(parsed.get("input_i"))
    input_tp = _coerce_float(parsed.get("input_tp"))
    input_lra = _coerce_float(parsed.get("input_lra"))
    input_thresh = _coerce_float(parsed.get("input_thresh"))
    target_offset = _coerce_float(parsed.get("target_offset"))
    if input_i is None or input_tp is None or input_lra is None or input_thresh is None:
        return None
    return LoudnessMeasurement(
        input_i=input_i,
        input_tp=input_tp,
        input_lra=input_lra,
        input_thresh=input_thresh,
        target_offset=target_offset if target_offset is not None else 0.0,
        normalization_type=str(parsed.get("normalization_type") or ""),
    )


def build_measure_argv(binary: str, path: str, mastering: Any) -> list[str]:
    """Bound the measurement pass; analysis output goes to null."""

    return [
        binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        path,
        "-af",
        (
            f"loudnorm=I={_num(mastering.target_lufs)}:"
            f"TP={_num(mastering.render_guard_dbtp)}:"
            f"LRA={_num(_bounded_lra(11.0))}:print_format=json"
        ),
        "-f",
        "null",
        "-",
    ]


def _bounded_lra(value: float) -> float:
    return min(_LOUDNORM_LRA_MAX, max(_LOUDNORM_LRA_MIN, float(value)))


def select_mastering(measurement: LoudnessMeasurement, mastering: Any) -> MasteringDecision:
    """Mechanical linear mastering selection from measured values.

    ``g = min(target - measured_i, guard - measured_tp)``; the requested linear
    target ``measured_i + g`` must remain within the existing +/-1 LU tolerance.
    An infeasible target is reported as :data:`AUDIO_PEAK_REPAIR_NEEDED` (the
    single bounded Stage 6 repair may authorize a peak limiter), never silently
    handled with dynamic normalization.
    """

    target = float(mastering.target_lufs)
    guard = float(mastering.render_guard_dbtp)
    ideal_gain = target - measurement.input_i
    guard_gain = guard - measurement.input_tp
    gain = min(ideal_gain, guard_gain)
    predicted = measurement.input_i + gain
    feasible = (target - 1.0) <= predicted <= (target + 1.0)
    reason = None if feasible else AUDIO_PEAK_REPAIR_NEEDED
    return MasteringDecision(
        feasible=feasible,
        reason_code=reason,
        applied_gain_db=gain,
        target_lufs=predicted,
        true_peak_dbtp=guard,
        lra=_bounded_lra(measurement.input_lra),
        predicted_lufs=predicted,
    )


def build_master_argv(
    binary: str,
    unmastered_path: str,
    master_path: str,
    measurement: LoudnessMeasurement,
    decision: MasteringDecision,
    mastering: Any,
) -> list[str]:
    """One lossless linear mastering application using measured values.

    ``linear=true`` with all measured values makes loudnorm apply exactly the
    single linear gain the decision computed; the pass prints its JSON so the
    runner can verify the reported ``normalization_type`` is ``linear`` and that
    the limitation is never used.
    """

    lra = decision.lra
    filter_expr = (
        f"loudnorm=I={_num(decision.target_lufs)}:"
        f"TP={_num(decision.true_peak_dbtp)}:"
        f"LRA={_num(lra)}:"
        "linear=true:"
        f"measured_I={_num(measurement.input_i)}:"
        f"measured_TP={_num(measurement.input_tp)}:"
        f"measured_LRA={_num(_bounded_lra(measurement.input_lra))}:"
        f"measured_thresh={_num(measurement.input_thresh)}:"
        f"offset={_num(measurement.target_offset)}:"
        "print_format=json"
    )
    return [
        binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        unmastered_path,
        "-af",
        filter_expr,
        "-ar",
        "48000",
        "-ac",
        "2",
        "-c:a",
        AUDIO_SAMPLE_FORMAT,
        "-y",
        master_path,
    ]


def verify_linear_mastering(master_stderr: str) -> None:
    """Fail closed unless the mastering pass reported a genuinely linear mode."""

    match = parse_loudnorm_output(master_stderr)
    if match is None or match.normalization_type.lower() != "linear":
        raise AudioPreparationError(
            AUDIO_MASTERING_INFEASIBLE,
            "mastering did not report linear normalization",
        )


def measurement_or_fail(text: str) -> LoudnessMeasurement:
    measurement = parse_loudnorm_output(text)
    if measurement is None:
        raise AudioPreparationError(AUDIO_MEASUREMENT_INVALID, "no finite loudness measurement")
    return measurement


def reject_decision(decision: MasteringDecision) -> None:
    if not decision.feasible:
        raise AudioPreparationError(
            decision.reason_code or AUDIO_MASTERING_INFEASIBLE,
            "linear mastering target is not within tolerance",
        )


def unsupported_recipe(detail: str) -> AudioPreparationError:
    return AudioPreparationError(AUTHORED_AUDIO_RECIPE_UNSUPPORTED, detail)


__all__ = [
    "AUDIO_SAMPLE_FORMAT",
    "AudioPreparationError",
    "AudioPreparationPlan",
    "LoudnessMeasurement",
    "MASTER_MIX_FILTERGRAPH_NAME",
    "MASTER_RELATIVE_NAME",
    "MasteringDecision",
    "UNMASTERED_RELATIVE_NAME",
    "build_master_argv",
    "build_measure_argv",
    "measurement_or_fail",
    "parse_loudnorm_output",
    "reject_decision",
    "select_mastering",
    "unsupported_recipe",
    "verify_linear_mastering",
]
