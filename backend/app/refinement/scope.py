"""Pure, versioned FINAL_CLIP selected-evidence derivation.

The FINAL_CLIP ASR pass observes the whole extraction context, but only the
settled refined clip bounds are executable. This module keeps the complete raw
context observation intact and derives a separately validated clip-only view.

It performs no database, filesystem, model or provider work. Invalid context
observations are nonblocking only when they are finite zero-duration points
inside the extraction context and strictly outside the settled clip. Everything
that could become an executable clip word must be finite, strictly positive and
fully inside the clip; partial edge words and unaccounted text still refuse.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.refinement.fingerprints import component_fingerprint
from app.refinement.policy import (
    FINAL_CLIP_SCOPE_POLICY_VERSION,
    MAX_RAW_WORD_OBSERVATIONS,
)
from app.refinement.types import WordTimestamp

_TIMESTAMP_EPSILON = 1e-6

FINAL_CLIP_SCOPE_STATE_ACCEPTED = "ACCEPTED"
FINAL_CLIP_SCOPE_STATE_REJECTED = "REJECTED"
FINAL_CLIP_SCOPE_SPAN_ID = "final-clip-scope"

SCOPE_REASON_INVALID_BOUNDS = "invalid_clip_bounds"
SCOPE_REASON_CLIP_OUTSIDE_CONTEXT = "clip_bounds_outside_context"
SCOPE_REASON_OBSERVATIONS_MISSING = "raw_observations_missing"
SCOPE_REASON_OBSERVATIONS_INCOMPLETE = "raw_observations_incomplete"
SCOPE_REASON_OBSERVATION_FINGERPRINT = "raw_observation_fingerprint_missing"
SCOPE_REASON_OBSERVATION_IDENTITY = "raw_observation_fingerprint_mismatch"
SCOPE_REASON_IDENTITY = "raw_observation_identity_invalid"
SCOPE_REASON_MALFORMED = "malformed_word_timestamp"
SCOPE_REASON_NON_FINITE = "non-finite_word_timestamp"
SCOPE_REASON_REVERSED = "reversed_word_duration"
SCOPE_REASON_CONTEXT_ESCAPE = "word_timestamp_outside_context"
SCOPE_REASON_SOURCE_MAPPING = "source_mapping_mismatch"
SCOPE_REASON_ZERO_INSIDE = "zero_duration_word_inside_clip"
SCOPE_REASON_EDGE_WORD = "partial_word_at_clip_edge"
SCOPE_REASON_ORDER = "nonmonotonic_word_timestamps"
SCOPE_REASON_UNACCOUNTED_TEXT = "unaccounted_transcript_text"
SCOPE_REASON_AMBIGUOUS_TOKENIZATION = "ambiguous_word_tokenization"
SCOPE_REASON_NO_SELECTED = "no_selected_clip_words"


@dataclass(frozen=True)
class ClipViewWord:
    """One executable clip word with its parent raw-record identity."""

    index: int
    parent_record_index: int
    text: str
    start: float
    end: float
    probability: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "parent_record_index": self.parent_record_index,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "probability": self.probability,
        }

    def as_timestamp(self) -> WordTimestamp:
        return WordTimestamp(
            text=self.text,
            start=self.start,
            end=self.end,
            probability=self.probability,
            parent_record_index=self.parent_record_index,
        )


@dataclass(frozen=True)
class FinalClipView:
    """An accepted, separately validated clip-only transcript view."""

    state: str
    policy_version: str
    fingerprint: str
    context_start: float
    context_end: float
    clip_start: float
    clip_end: float
    observation_fingerprint: str
    observation_record_count: int
    text: str
    words: tuple[ClipViewWord, ...]
    context_only_invalid_record_indexes: tuple[int, ...]

    @property
    def mapping(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "clip_index": word.index,
                "parent_record_index": word.parent_record_index,
                "start": word.start,
                "end": word.end,
            }
            for word in self.words
        )

    def envelope(self) -> dict[str, object]:
        return {
            "state": self.state,
            "policy_version": self.policy_version,
            "policy_fingerprint": scope_policy_fingerprint(),
            "fingerprint": self.fingerprint,
            "context_start": self.context_start,
            "context_end": self.context_end,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "observation_fingerprint": self.observation_fingerprint,
            "observation_record_count": self.observation_record_count,
            "selected_text": self.text,
            "selected_word_count": len(self.words),
            "words": [word.as_dict() for word in self.words],
            "mapping": [dict(item) for item in self.mapping],
            "context_only_invalid_record_indexes": list(self.context_only_invalid_record_indexes),
            "reason": None,
        }


@dataclass(frozen=True)
class ScopeRefusal:
    """An explicit refusal to derive an executable clip-only view."""

    state: str
    policy_version: str
    fingerprint: str
    context_start: float | None
    context_end: float | None
    clip_start: float | None
    clip_end: float | None
    observation_fingerprint: str
    observation_record_count: int
    reason: str
    violations: tuple[dict[str, object], ...] = ()

    def envelope(self) -> dict[str, object]:
        return {
            "state": self.state,
            "policy_version": self.policy_version,
            "policy_fingerprint": scope_policy_fingerprint(),
            "fingerprint": self.fingerprint,
            "context_start": self.context_start,
            "context_end": self.context_end,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "observation_fingerprint": self.observation_fingerprint,
            "observation_record_count": self.observation_record_count,
            "selected_text": "",
            "selected_word_count": 0,
            "words": [],
            "mapping": [],
            "context_only_invalid_record_indexes": [],
            "reason": self.reason,
            "violations": [dict(item) for item in self.violations],
        }


def final_clip_scope_state(provider_evidence: object) -> str | None:
    """Return the persisted scoped-view state, if one was recorded."""

    if not isinstance(provider_evidence, Mapping):
        return None
    view = provider_evidence.get("final_clip_view")
    if not isinstance(view, Mapping):
        return None
    state = view.get("state")
    return state if isinstance(state, str) else None


def final_clip_scope_refused(provider_evidence: object) -> bool:
    """True when the persisted FINAL_CLIP scope explicitly refused selection.

    A refusal is structural: it cannot be waived by operator text, a
    resolution string, an optimistic status, or cache ineligibility alone.
    """

    return final_clip_scope_state(provider_evidence) == FINAL_CLIP_SCOPE_STATE_REJECTED


def is_structural_scope_span(span_id: object) -> bool:
    """True for the non-waivable unresolved span created by a scope refusal."""

    return span_id == FINAL_CLIP_SCOPE_SPAN_ID


def scope_policy_fingerprint() -> str:
    """Stable identity of the selected-view policy itself."""

    return component_fingerprint(
        "final-clip-scope-policy",
        {
            "policy_version": FINAL_CLIP_SCOPE_POLICY_VERSION,
            "epsilon": _TIMESTAMP_EPSILON,
            "max_raw_word_observations": MAX_RAW_WORD_OBSERVATIONS,
        },
    )


def raw_observation_fingerprint(
    *,
    raw_observations: Sequence[Mapping[str, object]],
    record_count: int,
    observations_complete: bool,
    context_start: float,
    context_end: float,
) -> str:
    """Fingerprint the complete bounded raw observation, including invalid rows."""

    return component_fingerprint(
        "raw-word-observations",
        {
            "context_start": round(float(context_start), 6),
            "context_end": round(float(context_end), 6),
            "record_count": record_count,
            "complete": observations_complete,
            "records": [dict(item) for item in raw_observations],
        },
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fingerprint(
    *,
    state: str,
    context_start: float | None,
    context_end: float | None,
    clip_start: float | None,
    clip_end: float | None,
    observation_fingerprint: str,
    observation_record_count: int,
    text: str = "",
    words: Sequence[ClipViewWord] = (),
    reason: str | None = None,
    violations: Sequence[Mapping[str, object]] = (),
) -> str:
    return component_fingerprint(
        "final-clip-scope-view",
        {
            "policy_version": FINAL_CLIP_SCOPE_POLICY_VERSION,
            "policy_fingerprint": scope_policy_fingerprint(),
            "state": state,
            "context_start": context_start,
            "context_end": context_end,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "observation_fingerprint": observation_fingerprint,
            "observation_record_count": observation_record_count,
            "text": text,
            "words": [word.as_dict() for word in words],
            "reason": reason,
            "violations": [dict(item) for item in violations],
        },
    )


def _refusal(
    *,
    reason: str,
    context_start: float | None,
    context_end: float | None,
    clip_start: float | None,
    clip_end: float | None,
    observation_fingerprint: str,
    observation_record_count: int,
    violations: Sequence[Mapping[str, object]] = (),
) -> ScopeRefusal:
    return ScopeRefusal(
        state="REJECTED",
        policy_version=FINAL_CLIP_SCOPE_POLICY_VERSION,
        fingerprint=_fingerprint(
            state="REJECTED",
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=observation_record_count,
            reason=reason,
            violations=violations,
        ),
        context_start=context_start,
        context_end=context_end,
        clip_start=clip_start,
        clip_end=clip_end,
        observation_fingerprint=observation_fingerprint,
        observation_record_count=observation_record_count,
        reason=reason,
        violations=tuple(dict(item) for item in violations),
    )


def _violation(record_index: int, reason: str, **detail: object) -> dict[str, object]:
    return {"record_index": record_index, "reason": reason, **detail}


def derive_final_clip_view(
    *,
    raw_observations: Sequence[Mapping[str, object]],
    record_count: int,
    observations_complete: bool,
    observation_fingerprint: str,
    transcript: str,
    context_start: float,
    context_end: float,
    clip_start: float,
    clip_end: float,
) -> FinalClipView | ScopeRefusal:
    """Derive the executable clip-only text/words or refuse explicitly."""

    bounds = (context_start, context_end, clip_start, clip_end)
    if not all(_finite_number(value) is not None for value in bounds):
        return _refusal(
            reason=SCOPE_REASON_INVALID_BOUNDS,
            context_start=None,
            context_end=None,
            clip_start=None,
            clip_end=None,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
        )
    if (
        context_start < 0.0
        or context_end <= context_start
        or clip_start < 0.0
        or clip_end <= clip_start
    ):
        return _refusal(
            reason=SCOPE_REASON_INVALID_BOUNDS,
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
        )
    if (
        clip_start < context_start - _TIMESTAMP_EPSILON
        or clip_end > context_end + _TIMESTAMP_EPSILON
    ):
        return _refusal(
            reason=SCOPE_REASON_CLIP_OUTSIDE_CONTEXT,
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
        )
    if not isinstance(observation_fingerprint, str) or not observation_fingerprint:
        return _refusal(
            reason=SCOPE_REASON_OBSERVATION_FINGERPRINT,
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint="",
            observation_record_count=record_count,
        )
    if not observations_complete:
        return _refusal(
            reason=SCOPE_REASON_OBSERVATIONS_INCOMPLETE,
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
        )
    if record_count <= 0 or len(raw_observations) != record_count:
        return _refusal(
            reason=SCOPE_REASON_OBSERVATIONS_MISSING,
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
        )
    computed_observation_fingerprint = raw_observation_fingerprint(
        raw_observations=raw_observations,
        record_count=record_count,
        observations_complete=observations_complete,
        context_start=context_start,
        context_end=context_end,
    )
    if computed_observation_fingerprint != observation_fingerprint:
        return _refusal(
            reason=SCOPE_REASON_OBSERVATION_IDENTITY,
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
        )

    violations: list[dict[str, object]] = []
    selected: list[ClipViewWord] = []
    context_only_invalid: list[int] = []
    concatenated: list[str] = []
    previous_start: float | None = None

    for position, raw in enumerate(raw_observations):
        record_index = raw.get("record_index")
        text = raw.get("text")
        relative_start = _finite_number(raw.get("relative_start"))
        relative_end = _finite_number(raw.get("relative_end"))
        source_start = _finite_number(raw.get("source_start"))
        source_end = _finite_number(raw.get("source_end"))
        if (
            isinstance(record_index, bool)
            or not isinstance(record_index, int)
            or record_index != position
        ):
            violations.append(_violation(position, SCOPE_REASON_IDENTITY))
            continue
        if not isinstance(text, str) or not text.strip():
            violations.append(_violation(record_index, SCOPE_REASON_MALFORMED))
            continue
        concatenated.append(text)
        if relative_start is None or relative_end is None:
            violations.append(_violation(record_index, SCOPE_REASON_MALFORMED))
            continue
        if source_start is None or source_end is None:
            violations.append(_violation(record_index, SCOPE_REASON_NON_FINITE))
            continue
        if (
            abs(source_start - (relative_start + context_start)) > _TIMESTAMP_EPSILON
            or abs(source_end - (relative_end + context_start)) > _TIMESTAMP_EPSILON
        ):
            violations.append(_violation(record_index, SCOPE_REASON_SOURCE_MAPPING))
            continue
        if source_start < 0.0:
            violations.append(_violation(record_index, SCOPE_REASON_MALFORMED))
            continue
        if source_end < source_start:
            violations.append(_violation(record_index, SCOPE_REASON_REVERSED))
            continue
        if source_end == source_start:
            point = source_start
            if not (
                context_start - _TIMESTAMP_EPSILON <= point <= context_end + _TIMESTAMP_EPSILON
            ):
                violations.append(_violation(record_index, SCOPE_REASON_CONTEXT_ESCAPE))
                continue
            if clip_start - _TIMESTAMP_EPSILON <= point <= clip_end + _TIMESTAMP_EPSILON:
                violations.append(_violation(record_index, SCOPE_REASON_ZERO_INSIDE))
            else:
                context_only_invalid.append(record_index)
            continue
        if previous_start is not None and source_start < previous_start - _TIMESTAMP_EPSILON:
            violations.append(_violation(record_index, SCOPE_REASON_ORDER))
            continue
        previous_start = source_start
        if not (
            context_start - _TIMESTAMP_EPSILON
            <= source_start
            < source_end
            <= context_end + _TIMESTAMP_EPSILON
        ):
            violations.append(_violation(record_index, SCOPE_REASON_CONTEXT_ESCAPE))
            continue
        if source_end <= clip_start + _TIMESTAMP_EPSILON:
            continue
        if source_start >= clip_end - _TIMESTAMP_EPSILON:
            continue
        if (
            source_start >= clip_start - _TIMESTAMP_EPSILON
            and source_end <= clip_end + _TIMESTAMP_EPSILON
        ):
            probability = raw.get("probability")
            selected.append(
                ClipViewWord(
                    index=len(selected),
                    parent_record_index=record_index,
                    text=text,
                    start=source_start,
                    end=source_end,
                    probability=(
                        float(probability)
                        if isinstance(probability, (int, float))
                        and not isinstance(probability, bool)
                        else None
                    ),
                )
            )
            continue
        violations.append(_violation(record_index, SCOPE_REASON_EDGE_WORD))

    if violations:
        return _refusal(
            reason=str(violations[0]["reason"]),
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
            violations=violations,
        )
    if "".join(concatenated).strip() != (transcript or "").strip():
        return _refusal(
            reason=SCOPE_REASON_UNACCOUNTED_TEXT,
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
        )
    if any(len(word.text.split()) != 1 for word in selected):
        return _refusal(
            reason=SCOPE_REASON_AMBIGUOUS_TOKENIZATION,
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
        )
    if not selected:
        return _refusal(
            reason=SCOPE_REASON_NO_SELECTED,
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
        )

    text = "".join(word.text for word in selected).strip()
    if len(text.split()) != len(selected):
        return _refusal(
            reason=SCOPE_REASON_AMBIGUOUS_TOKENIZATION,
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
        )

    words = tuple(selected)
    return FinalClipView(
        state="ACCEPTED",
        policy_version=FINAL_CLIP_SCOPE_POLICY_VERSION,
        fingerprint=_fingerprint(
            state="ACCEPTED",
            context_start=context_start,
            context_end=context_end,
            clip_start=clip_start,
            clip_end=clip_end,
            observation_fingerprint=observation_fingerprint,
            observation_record_count=record_count,
            text=text,
            words=words,
        ),
        context_start=context_start,
        context_end=context_end,
        clip_start=clip_start,
        clip_end=clip_end,
        observation_fingerprint=observation_fingerprint,
        observation_record_count=record_count,
        text=text,
        words=words,
        context_only_invalid_record_indexes=tuple(context_only_invalid),
    )


__all__ = [
    "FINAL_CLIP_SCOPE_SPAN_ID",
    "FINAL_CLIP_SCOPE_STATE_ACCEPTED",
    "FINAL_CLIP_SCOPE_STATE_REJECTED",
    "ClipViewWord",
    "FinalClipView",
    "ScopeRefusal",
    "derive_final_clip_view",
    "final_clip_scope_refused",
    "final_clip_scope_state",
    "is_structural_scope_span",
    "raw_observation_fingerprint",
    "scope_policy_fingerprint",
]
