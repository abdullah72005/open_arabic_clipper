"""Deterministic mandatory hook/payoff evidence resolution and coverage.

Stage 4.0 records ``hook_index``/``payoff_index`` as positions in the
authoritative, ordered source transcript segments (never word indexes and never
positions in a filtered planning window). This module is the single pure
implementation used by Stage 4.1 validation, Stage 4.2 integrity revalidation,
Stage 4.3 selection guarding, and Stage 5.0 compatibility.

An absent index (``None``) is optional evidence. A declared index that is
malformed, out of range, has invalid bounds, or conflicts with the authoritative
segment identity is *unresolved*, never satisfied. A resolved segment must be
covered by the resolved source excerpts; a null payoff index can never turn a
missing hook into a payoff failure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MANDATORY_EVIDENCE_CONTRACT_VERSION = "stage4-mandatory-evidence-v1"
MANDATORY_COVERAGE_TOLERANCE_SECONDS = 0.75

HOOK_NOT_COVERED = "HOOK_NOT_COVERED"
PAYOFF_NOT_COVERED = "PAYOFF_NOT_COVERED"
HOOK_EVIDENCE_MALFORMED = "HOOK_EVIDENCE_MALFORMED"
PAYOFF_EVIDENCE_MALFORMED = "PAYOFF_EVIDENCE_MALFORMED"
HOOK_EVIDENCE_UNRESOLVED = "HOOK_EVIDENCE_UNRESOLVED"
PAYOFF_EVIDENCE_UNRESOLVED = "PAYOFF_EVIDENCE_UNRESOLVED"
HOOK_EVIDENCE_MISSING = "HOOK_EVIDENCE_MISSING"
PAYOFF_EVIDENCE_MISSING = "PAYOFF_EVIDENCE_MISSING"

_MALFORMED = "MALFORMED"
_UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class MandatoryEvidence:
    """Bounded authoritative hook/payoff segment evidence for one source."""

    source_id: str = ""
    hook_declared: bool = False
    payoff_declared: bool = False
    hook_segment: tuple[int, float, float] | None = None
    payoff_segment: tuple[int, float, float] | None = None
    hook_error: str | None = None
    payoff_error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": MANDATORY_EVIDENCE_CONTRACT_VERSION,
            "source_id": self.source_id,
            "hook_declared": self.hook_declared,
            "payoff_declared": self.payoff_declared,
            "hook_segment": (
                {
                    "index": self.hook_segment[0],
                    "start": round(self.hook_segment[1], 6),
                    "end": round(self.hook_segment[2], 6),
                }
                if self.hook_segment is not None
                else None
            ),
            "payoff_segment": (
                {
                    "index": self.payoff_segment[0],
                    "start": round(self.payoff_segment[1], 6),
                    "end": round(self.payoff_segment[2], 6),
                }
                if self.payoff_segment is not None
                else None
            ),
            "hook_error": self.hook_error,
            "payoff_error": self.payoff_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object] | None) -> "MandatoryEvidence":
        if not isinstance(value, Mapping):
            return cls()
        return cls(
            source_id=str(value.get("source_id") or ""),
            hook_declared=bool(value.get("hook_declared")),
            payoff_declared=bool(value.get("payoff_declared")),
            hook_segment=_segment(value.get("hook_segment")),
            payoff_segment=_segment(value.get("payoff_segment")),
            hook_error=_error(value.get("hook_error")),
            payoff_error=_error(value.get("payoff_error")),
        )


def _segment(value: object) -> tuple[int, float, float] | None:
    if not isinstance(value, Mapping):
        return None
    index = value.get("index")
    start = value.get("start")
    end = value.get("end")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return None
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return None
    if isinstance(end, bool) or not isinstance(end, (int, float)):
        return None
    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        return None
    if float(end) < float(start):
        return None
    return int(index), float(start), float(end)


def _error(value: object) -> str | None:
    if value in {_MALFORMED, _UNRESOLVED}:
        return str(value)
    return None


def _resolve_one(
    value: object, segments: Sequence[Mapping[str, object]]
) -> tuple[bool, tuple[int, float, float] | None, str | None]:
    if value is None:
        return False, None, None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return True, None, _MALFORMED
    if value >= len(segments):
        return True, None, _UNRESOLVED
    segment = segments[value]
    if not isinstance(segment, Mapping):
        return True, None, _UNRESOLVED
    start = segment.get("start")
    end = segment.get("end")
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return True, None, _UNRESOLVED
    if isinstance(end, bool) or not isinstance(end, (int, float)):
        return True, None, _UNRESOLVED
    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        return True, None, _UNRESOLVED
    if float(end) < float(start):
        return True, None, _UNRESOLVED
    return True, (int(value), float(start), float(end)), None


def resolve_mandatory_evidence(
    source_id: str,
    source_moment: Mapping[str, object] | None,
    source_segments: Sequence[Mapping[str, object]],
) -> MandatoryEvidence:
    """Resolve Stage 4.0 indexes against the ordered authoritative segments."""

    moment = source_moment if isinstance(source_moment, Mapping) else {}
    hook_declared, hook_segment, hook_error = _resolve_one(
        moment.get("hook_index"), source_segments
    )
    payoff_declared, payoff_segment, payoff_error = _resolve_one(
        moment.get("payoff_index"), source_segments
    )
    return MandatoryEvidence(
        source_id=source_id,
        hook_declared=hook_declared,
        payoff_declared=payoff_declared,
        hook_segment=hook_segment,
        payoff_segment=payoff_segment,
        hook_error=hook_error,
        payoff_error=payoff_error,
    )


def point_covered(spans: Sequence[tuple[float, float]], point: float, tolerance: float) -> bool:
    return any(min(a, b) - tolerance <= point <= max(a, b) + tolerance for a, b in spans)


def spans_cover_endpoints(
    spans: Sequence[tuple[float, float]],
    start: float,
    end: float,
    *,
    tolerance: float = MANDATORY_COVERAGE_TOLERANCE_SECONDS,
) -> bool:
    """Existing Stage 5 coverage requirement: both segment endpoints covered."""

    return point_covered(spans, start, tolerance) and point_covered(spans, end, tolerance)


def spans_cover_range(
    spans: Sequence[tuple[float, float]],
    start: float,
    end: float,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Require the complete cited range to be covered by a contiguous union."""

    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        return False
    intervals = sorted(
        (max(0.0, min(a, b)), max(a, b))
        for a, b in spans
        if math.isfinite(float(a)) and math.isfinite(float(b))
    )
    cursor = start
    for left, right in intervals:
        if right + tolerance < cursor:
            continue
        if left > cursor + tolerance:
            return False
        cursor = max(cursor, right)
        if cursor + tolerance >= end:
            return True
    return cursor + tolerance >= end


def coverage_reasons(
    hook_index: object,
    payoff_index: object,
    evidence: MandatoryEvidence,
    spans: Sequence[tuple[float, float]],
    *,
    tolerance: float = MANDATORY_COVERAGE_TOLERANCE_SECONDS,
) -> tuple[str, ...]:
    """Distinct, deterministic hook/payoff coverage failures.

    Absent optional evidence produces no reason. Declared mandatory evidence that
    cannot be authoritatively resolved is reported as unresolved/malformed and is
    never treated as satisfied.
    """

    reasons: list[str] = []
    for key, index, declared, segment, error, not_cov, malformed, unresolved, missing in (
        (
            "hook",
            hook_index,
            evidence.hook_declared,
            evidence.hook_segment,
            evidence.hook_error,
            HOOK_NOT_COVERED,
            HOOK_EVIDENCE_MALFORMED,
            HOOK_EVIDENCE_UNRESOLVED,
            HOOK_EVIDENCE_MISSING,
        ),
        (
            "payoff",
            payoff_index,
            evidence.payoff_declared,
            evidence.payoff_segment,
            evidence.payoff_error,
            PAYOFF_NOT_COVERED,
            PAYOFF_EVIDENCE_MALFORMED,
            PAYOFF_EVIDENCE_UNRESOLVED,
            PAYOFF_EVIDENCE_MISSING,
        ),
    ):
        if index is None:
            if declared:
                reasons.append(missing)
            continue
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            reasons.append(malformed)
            continue
        if not declared:
            reasons.append(unresolved)
            continue
        if error == _MALFORMED:
            reasons.append(malformed)
            continue
        if error == _UNRESOLVED or segment is None:
            reasons.append(unresolved)
            continue
        if int(index) != segment[0]:
            reasons.append(unresolved)
            continue
        if not spans_cover_endpoints(spans, segment[1], segment[2], tolerance=tolerance):
            reasons.append(not_cov)
    return tuple(dict.fromkeys(reasons))


__all__ = [
    "HOOK_EVIDENCE_MALFORMED",
    "HOOK_EVIDENCE_MISSING",
    "HOOK_EVIDENCE_UNRESOLVED",
    "HOOK_NOT_COVERED",
    "MANDATORY_COVERAGE_TOLERANCE_SECONDS",
    "MANDATORY_EVIDENCE_CONTRACT_VERSION",
    "MandatoryEvidence",
    "PAYOFF_EVIDENCE_MALFORMED",
    "PAYOFF_EVIDENCE_MISSING",
    "PAYOFF_EVIDENCE_UNRESOLVED",
    "PAYOFF_NOT_COVERED",
    "coverage_reasons",
    "point_covered",
    "resolve_mandatory_evidence",
    "spans_cover_endpoints",
    "spans_cover_range",
]
