"""Operator-scoped candidate registration over immutable source evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.candidates.fingerprints import candidate_key
from app.candidates.persistence import apply_candidate_fields
from app.candidates.policy import DEFAULT_CONFIG, POLICY_VERSION, Stage3Config
from app.candidates.service import build_candidate_draft
from app.candidates.text import analysis_segment_text
from app.candidates.types import CandidateDraft, Proposal
from app.core.enums import CandidateDisposition, RefinementReason
from app.models import AudioArtifact, CandidateAnalysis, ClipCandidate, SourceVideo, Transcript
from app.pipeline.fingerprints import canonical_fingerprint

OPERATOR_SELECTION_VERSION = "operator-candidate-selection-v1"
OPERATOR_REGISTRATION_FINGERPRINT_VERSION = "1"
MAX_OPERATOR_REASON_CHARACTERS = 500
_BOUND_TOLERANCE = 1e-6


class CandidateRegistrationError(ValueError):
    """Operator candidate registration was rejected."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class CandidateRegistrationConflict(CandidateRegistrationError):
    """Operator candidate identity conflicts with existing stored evidence."""


@dataclass(frozen=True)
class OperatorCandidateRegistration:
    candidate: ClipCandidate
    created: bool
    ownership: str


@dataclass(frozen=True)
class _ResolvedWindow:
    start_time: float
    end_time: float
    start_segment_index: int
    end_segment_index: int
    segment_indexes: tuple[int, ...]
    raw_text: str
    analysis_text: str


def operator_selection_marker(row: ClipCandidate) -> dict[str, object] | None:
    """Return a stored operator-selection marker, if this row owns one."""

    for source in (row.provenance_snapshot, row.evidence_snapshot):
        if not isinstance(source, Mapping):
            continue
        marker = source.get("operator_selection")
        if isinstance(marker, Mapping):
            return dict(marker)
    return None


def compute_raw_span_fingerprint(
    source: SourceVideo,
    transcript: Transcript,
    audio: AudioArtifact | None,
    start_segment_index: int,
    end_segment_index: int,
) -> str:
    """Fingerprint only immutable source/transcript capture and selected raw words."""

    segments = _segments(transcript)
    selected = _indexed_segments(segments, start_segment_index, end_segment_index)
    start_time = _segment_time(selected[0], "start", "invalid_raw_evidence")
    end_time = _segment_time(selected[-1], "end", "invalid_raw_evidence")
    return _raw_span_fingerprint(
        source,
        transcript,
        audio,
        start_time,
        end_time,
        start_segment_index,
        end_segment_index,
        selected,
    )


def operator_identity_is_current(
    row: ClipCandidate,
    source: SourceVideo,
    transcript: Transcript,
    audio: AudioArtifact | None,
) -> bool:
    """Validate a stored operator pin against the current immutable raw evidence."""

    marker = operator_selection_marker(row)
    if marker is None:
        return False
    if marker.get("version") != OPERATOR_SELECTION_VERSION:
        return False
    start_index = marker.get("start_segment_index")
    end_index = marker.get("end_segment_index")
    if not isinstance(start_index, int) or not isinstance(end_index, int):
        return False
    try:
        current = compute_raw_span_fingerprint(source, transcript, audio, start_index, end_index)
    except CandidateRegistrationError:
        return False
    return marker.get("raw_span_fingerprint") == current


def operator_draft_matches_window(row: ClipCandidate, draft: CandidateDraft) -> bool:
    """Compare a same-key automatic draft scope to the operator pin."""

    marker = operator_selection_marker(row)
    if marker is None:
        return False
    proposal = draft.proposal
    return (
        _close(marker.get("start_time"), proposal.start_time)
        and _close(marker.get("end_time"), proposal.end_time)
        and marker.get("start_segment_index") == proposal.start_segment_index
        and marker.get("end_segment_index") == proposal.end_segment_index
        and list(marker.get("segment_indexes") or [])
        == list(range(proposal.start_segment_index, proposal.end_segment_index + 1))
    )


def register_operator_candidate(
    session: Session,
    source_id: UUID | str,
    start_seconds: float,
    end_seconds: float,
    reason: str,
    *,
    config: Stage3Config = DEFAULT_CONFIG,
) -> OperatorCandidateRegistration:
    """Register one operator-selected immutable source window.

    The caller owns commit/rollback. The source row lock is acquired before any
    transcript/candidate read and is held until the caller's transaction ends.
    """

    reason_text = _validate_reason(reason)
    source = session.scalar(
        select(SourceVideo).where(SourceVideo.id == _as_uuid(source_id)).with_for_update()
    )
    if source is None:
        raise CandidateRegistrationError("source_not_found", "source does not exist")
    transcript = session.scalar(select(Transcript).where(Transcript.source_video_id == source.id))
    if transcript is None:
        raise CandidateRegistrationError("transcript_not_found", "source has no transcript to bind")
    audio = session.scalar(select(AudioArtifact).where(AudioArtifact.source_video_id == source.id))

    window = _resolve_window(transcript, start_seconds, end_seconds)
    key = candidate_key(
        source_id=source.id,
        start_segment_index=window.start_segment_index,
        end_segment_index=window.end_segment_index,
        span_start=window.start_segment_index,
        span_end=window.end_segment_index,
    )
    raw_fingerprint = _raw_span_fingerprint(
        source,
        transcript,
        audio,
        window.start_time,
        window.end_time,
        window.start_segment_index,
        window.end_segment_index,
        _segments(transcript)[window.start_segment_index : window.end_segment_index + 1],
    )

    existing = session.scalar(select(ClipCandidate).where(ClipCandidate.candidate_key == key))
    if existing is not None:
        return _reuse_existing(
            session,
            existing,
            source,
            transcript,
            audio,
            window,
            raw_fingerprint,
            key,
            config,
        )

    marker = _operator_marker(
        source,
        transcript,
        audio,
        window,
        reason_text,
        raw_fingerprint,
        key,
    )
    proposal = Proposal(
        start_segment_index=window.start_segment_index,
        end_segment_index=window.end_segment_index,
        start_time=window.start_time,
        end_time=window.end_time,
        text=window.raw_text,
        boundary_reason="operator_selected",
        segment_indexes=window.segment_indexes,
        span_start=window.start_segment_index,
        span_end=window.end_segment_index,
    )
    draft = build_candidate_draft(
        proposal,
        _segments(transcript),
        source_id=str(source.id),
        rights_status=source.rights_status,
        media_origin=source.media_origin,
        dialect_profile=(transcript.dialect_profile.value if transcript.dialect_profile else None),
        dialect_confidence=transcript.dialect_confidence,
        provenance_metadata=source.provenance_metadata or {},
        config=config,
    )
    draft.candidate_key = key
    draft.disposition = CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT
    draft.refinement_reasons = tuple(
        RefinementReason(value)
        for value in draft.scores.reasons
        if value in RefinementReason._value2member_map_
    )
    draft.refinement_evidence = dict(draft.evidence_snapshot)
    draft.provenance_snapshot = {
        **dict(draft.provenance_snapshot),
        "operator_selection": marker,
    }
    draft.analysis_fingerprint = _registration_fingerprint(
        source.id,
        key,
        raw_fingerprint,
    )

    row = ClipCandidate(source_video_id=source.id, candidate_key=key)
    row.candidate_analysis_id = None
    row.is_current = True
    apply_candidate_fields(row, draft)
    row.analysis_fingerprint = draft.analysis_fingerprint
    row.policy_version = POLICY_VERSION
    session.add(row)
    session.flush()
    return OperatorCandidateRegistration(candidate=row, created=True, ownership="OPERATOR")


def _reuse_existing(
    session: Session,
    row: ClipCandidate,
    source: SourceVideo,
    transcript: Transcript,
    audio: AudioArtifact | None,
    window: _ResolvedWindow,
    raw_fingerprint: str,
    key: str,
    config: Stage3Config,
) -> OperatorCandidateRegistration:
    marker = operator_selection_marker(row)
    if marker is not None:
        if not row.is_current:
            raise CandidateRegistrationConflict(
                "operator_candidate_stale",
                "operator candidate is stale and will not be reactivated",
            )
        if not _marker_matches(
            marker,
            source,
            transcript,
            audio,
            window,
            raw_fingerprint,
            key,
        ) or not operator_identity_is_current(row, source, transcript, audio):
            raise CandidateRegistrationConflict(
                "operator_raw_identity_conflict",
                "operator candidate identity no longer matches immutable raw evidence",
            )
        return OperatorCandidateRegistration(candidate=row, created=False, ownership="OPERATOR")

    if not row.is_current:
        raise CandidateRegistrationConflict(
            "candidate_identity_conflict",
            "existing candidate is stale and will not be adopted",
        )
    if not _automatic_row_is_exact(session, row, source, transcript, window, config):
        raise CandidateRegistrationConflict(
            "candidate_identity_conflict",
            "existing candidate key belongs to a different source window",
        )
    return OperatorCandidateRegistration(candidate=row, created=False, ownership="AUTOMATIC")


def _automatic_row_is_exact(
    session: Session,
    row: ClipCandidate,
    source: SourceVideo,
    transcript: Transcript,
    window: _ResolvedWindow,
    config: Stage3Config,
) -> bool:
    if row.source_video_id != source.id:
        return False
    if row.candidate_analysis_id is None:
        return False
    if session.get(CandidateAnalysis, row.candidate_analysis_id) is None:
        return False
    if not _close(row.start_time, window.start_time) or not _close(row.end_time, window.end_time):
        return False
    if row.start_segment_index != window.start_segment_index:
        return False
    if row.end_segment_index != window.end_segment_index:
        return False
    if list(row.segment_indexes or []) != list(window.segment_indexes):
        return False
    expected = _normalise_excerpt(window.analysis_text)[: config.max_excerpt_characters]
    return _normalise_excerpt(row.transcript_excerpt) == expected


def _operator_marker(
    source: SourceVideo,
    transcript: Transcript,
    audio: AudioArtifact | None,
    window: _ResolvedWindow,
    reason: str,
    raw_fingerprint: str,
    key: str,
) -> dict[str, object]:
    return {
        "version": OPERATOR_SELECTION_VERSION,
        "origin": "OPERATOR",
        "source_id": str(source.id),
        "source_content_hash": source.content_hash,
        "transcript_id": str(transcript.id),
        "transcript_input_fingerprint": transcript.input_fingerprint,
        "transcription_revision": transcript.transcription_revision,
        "audio_artifact_content_hash": audio.content_hash if audio else None,
        "audio_artifact_source_content_hash": audio.source_content_hash if audio else None,
        "start_time": window.start_time,
        "end_time": window.end_time,
        "start_segment_index": window.start_segment_index,
        "end_segment_index": window.end_segment_index,
        "segment_indexes": list(window.segment_indexes),
        "reason": reason,
        "raw_span_fingerprint": raw_fingerprint,
        "candidate_key": key,
    }


def _marker_matches(
    marker: Mapping[str, object],
    source: SourceVideo,
    transcript: Transcript,
    audio: AudioArtifact | None,
    window: _ResolvedWindow,
    raw_fingerprint: str,
    key: str,
) -> bool:
    return (
        marker.get("version") == OPERATOR_SELECTION_VERSION
        and marker.get("origin") == "OPERATOR"
        and marker.get("source_id") == str(source.id)
        and marker.get("source_content_hash") == source.content_hash
        and marker.get("transcript_id") == str(transcript.id)
        and marker.get("transcript_input_fingerprint") == transcript.input_fingerprint
        and marker.get("transcription_revision") == transcript.transcription_revision
        and marker.get("audio_artifact_content_hash") == (audio.content_hash if audio else None)
        and marker.get("audio_artifact_source_content_hash")
        == (audio.source_content_hash if audio else None)
        and _close(marker.get("start_time"), window.start_time)
        and _close(marker.get("end_time"), window.end_time)
        and marker.get("start_segment_index") == window.start_segment_index
        and marker.get("end_segment_index") == window.end_segment_index
        and list(marker.get("segment_indexes") or []) == list(window.segment_indexes)
        and marker.get("raw_span_fingerprint") == raw_fingerprint
        and marker.get("candidate_key") == key
    )


def _registration_fingerprint(source_id: UUID, key: str, raw_fingerprint: str) -> str:
    return canonical_fingerprint(
        "operator-candidate-registration",
        OPERATOR_REGISTRATION_FINGERPRINT_VERSION,
        {
            "source_id": str(source_id),
            "candidate_key": key,
            "operator_selection_version": OPERATOR_SELECTION_VERSION,
            "raw_span_fingerprint": raw_fingerprint,
        },
    )


def _raw_span_fingerprint(
    source: SourceVideo,
    transcript: Transcript,
    audio: AudioArtifact | None,
    start_time: float,
    end_time: float,
    start_segment_index: int,
    end_segment_index: int,
    selected: Sequence[Mapping[str, object]],
) -> str:
    payload = {
        "source_id": str(source.id),
        "source_content_hash": source.content_hash,
        "transcript_id": str(transcript.id),
        "transcript_input_fingerprint": transcript.input_fingerprint,
        "transcription_revision": transcript.transcription_revision,
        "audio_artifact_content_hash": audio.content_hash if audio else None,
        "audio_artifact_source_content_hash": audio.source_content_hash if audio else None,
        "start_time": start_time,
        "end_time": end_time,
        "start_segment_index": start_segment_index,
        "end_segment_index": end_segment_index,
        "segment_indexes": list(range(start_segment_index, end_segment_index + 1)),
        "segments": [
            {
                "index": start_segment_index + offset,
                "start": _segment_time(segment, "start", "invalid_raw_evidence"),
                "end": _segment_time(segment, "end", "invalid_raw_evidence"),
                "raw_text": _raw_text(segment),
                "words": _word_payload(segment),
            }
            for offset, segment in enumerate(selected)
        ],
    }
    return canonical_fingerprint("operator-candidate-raw-span", "1", payload)


def _resolve_window(
    transcript: Transcript,
    start_seconds: float,
    end_seconds: float,
) -> _ResolvedWindow:
    start = _finite_argument(start_seconds, "start")
    end = _finite_argument(end_seconds, "end")
    duration = _finite_argument(transcript.duration, "transcript duration")
    if duration <= 0:
        raise CandidateRegistrationError("invalid_bounds", "transcript duration must be positive")
    if start < 0 or end <= start:
        raise CandidateRegistrationError(
            "invalid_bounds", "start/end must be finite, nonnegative and ordered"
        )
    if start > duration + _BOUND_TOLERANCE or end > duration + _BOUND_TOLERANCE:
        raise CandidateRegistrationError("invalid_bounds", "selected bounds exceed source")

    segments = _segments(transcript)
    if not segments:
        raise CandidateRegistrationError("invalid_bounds", "transcript has no segments")
    for index, segment in enumerate(segments):
        _validate_segment(segment, index)

    start_matches = [
        index
        for index, segment in enumerate(segments)
        if abs(_segment_time(segment, "start", "invalid_bounds") - start) <= _BOUND_TOLERANCE
    ]
    if len(start_matches) != 1:
        raise CandidateRegistrationError(
            "start_bound_not_unique",
            "start must match exactly one real segment start",
        )
    end_matches = [
        index
        for index, segment in enumerate(segments)
        if abs(_segment_time(segment, "end", "invalid_bounds") - end) <= _BOUND_TOLERANCE
    ]
    if len(end_matches) != 1:
        raise CandidateRegistrationError(
            "end_bound_not_unique",
            "end must match exactly one real segment end",
        )

    start_index = start_matches[0]
    end_index = end_matches[0]
    if start_index > end_index:
        raise CandidateRegistrationError("invalid_bounds", "selected segment range is reversed")
    selected = _indexed_segments(segments, start_index, end_index)
    resolved_start = _segment_time(selected[0], "start", "invalid_bounds")
    resolved_end = _segment_time(selected[-1], "end", "invalid_bounds")

    previous_start = -math.inf
    previous_end = -math.inf
    for segment in selected:
        segment_start = _segment_time(segment, "start", "invalid_bounds")
        segment_end = _segment_time(segment, "end", "invalid_bounds")
        if segment_start < previous_start - _BOUND_TOLERANCE:
            raise CandidateRegistrationError(
                "unordered_segments", "selected segments are not ordered by start"
            )
        if segment_end < previous_end - _BOUND_TOLERANCE:
            raise CandidateRegistrationError(
                "unordered_segments", "selected segments are not ordered by end"
            )
        previous_start = segment_start
        previous_end = segment_end

    for segment in selected:
        for word_start, word_end in _word_times(segment):
            if (
                word_start < resolved_start - _BOUND_TOLERANCE
                or word_end > resolved_end + _BOUND_TOLERANCE
            ):
                raise CandidateRegistrationError(
                    "edge_straddling_word",
                    "selected word evidence crosses the registered window edge",
                )

    raw_text = " ".join(
        _raw_text(segment).strip() for segment in selected if _raw_text(segment).strip()
    ).strip()
    if not raw_text:
        raise CandidateRegistrationError(
            "empty_selection_text", "selected window has no raw transcript text"
        )
    analysis_text = " ".join(
        analysis_segment_text(segment).strip()
        for segment in selected
        if analysis_segment_text(segment).strip()
    ).strip()
    return _ResolvedWindow(
        start_time=resolved_start,
        end_time=resolved_end,
        start_segment_index=start_index,
        end_segment_index=end_index,
        segment_indexes=tuple(range(start_index, end_index + 1)),
        raw_text=raw_text,
        analysis_text=analysis_text,
    )


def _segments(transcript: Transcript) -> list[dict[str, object]]:
    value = transcript.segments
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise CandidateRegistrationError(
            "invalid_raw_evidence", "transcript segments are missing or malformed"
        )
    return value


def _indexed_segments(
    segments: Sequence[Mapping[str, object]],
    start_segment_index: int,
    end_segment_index: int,
) -> list[Mapping[str, object]]:
    if (
        start_segment_index < 0
        or end_segment_index < start_segment_index
        or end_segment_index >= len(segments)
    ):
        raise CandidateRegistrationError(
            "invalid_raw_evidence", "selected segment range is outside the transcript"
        )
    return list(segments[start_segment_index : end_segment_index + 1])


def _validate_segment(segment: Mapping[str, object], index: int) -> None:
    segment_start = _segment_time(segment, "start", "invalid_raw_evidence")
    segment_end = _segment_time(segment, "end", "invalid_raw_evidence")
    if segment_end <= segment_start:
        raise CandidateRegistrationError(
            "invalid_raw_evidence", f"segment {index} has a nonpositive duration"
        )
    stored_index = segment.get("index")
    if stored_index is not None and stored_index != index:
        raise CandidateRegistrationError(
            "segment_index_mismatch",
            f"segment {index} declares a different index",
        )


def _segment_time(segment: Mapping[str, object], key: str, code: str) -> float:
    value = segment.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CandidateRegistrationError(code, f"segment {key} is missing or nonnumeric")
    number = float(value)
    if not math.isfinite(number):
        raise CandidateRegistrationError(code, f"segment {key} is not finite")
    return number


def _word_times(segment: Mapping[str, object]) -> list[tuple[float, float]]:
    words = segment.get("words")
    if words is None:
        return []
    if not isinstance(words, list):
        raise CandidateRegistrationError("invalid_word_evidence", "segment words are not a list")
    times: list[tuple[float, float]] = []
    for word in words:
        if not isinstance(word, Mapping):
            raise CandidateRegistrationError(
                "invalid_word_evidence", "segment contains a malformed word record"
            )
        start = word.get("start")
        end = word.get("end")
        if (
            isinstance(start, bool)
            or not isinstance(start, int | float)
            or isinstance(end, bool)
            or not isinstance(end, int | float)
        ):
            raise CandidateRegistrationError(
                "invalid_word_evidence", "word start/end is missing or nonnumeric"
            )
        start_number = float(start)
        end_number = float(end)
        if (
            not math.isfinite(start_number)
            or not math.isfinite(end_number)
            or end_number < start_number
        ):
            raise CandidateRegistrationError("invalid_word_evidence", "word start/end is malformed")
        times.append((start_number, end_number))
    return times


def _word_payload(segment: Mapping[str, object]) -> list[dict[str, object]]:
    words = segment.get("words")
    if not isinstance(words, list):
        return []
    payload: list[dict[str, object]] = []
    for word in words:
        if not isinstance(word, Mapping):
            continue
        item: dict[str, object] = {"word": str(word.get("word") or "")}
        for key in ("start", "end"):
            value = word.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                item[key] = float(value)
        probability = word.get("probability")
        if isinstance(probability, int | float) and not isinstance(probability, bool):
            item["probability"] = float(probability)
        payload.append(item)
    return payload


def _raw_text(segment: Mapping[str, object]) -> str:
    value = segment.get("raw_text")
    if isinstance(value, str) and value.strip():
        return value
    fallback = segment.get("text")
    return fallback if isinstance(fallback, str) else ""


def _validate_reason(reason: str) -> str:
    if not isinstance(reason, str):
        raise CandidateRegistrationError("invalid_reason", "operator reason must be text")
    cleaned = reason.strip()
    if not cleaned:
        raise CandidateRegistrationError("invalid_reason", "operator reason is required")
    if len(cleaned) > MAX_OPERATOR_REASON_CHARACTERS:
        raise CandidateRegistrationError(
            "invalid_reason", "operator reason exceeds the bounded length"
        )
    return cleaned


def _finite_argument(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CandidateRegistrationError("invalid_bounds", f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CandidateRegistrationError("invalid_bounds", f"{label} is not finite")
    return number


def _as_uuid(value: UUID | str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as error:
        raise CandidateRegistrationError("source_not_found", "source id is invalid") from error


def _close(left: object, right: float) -> bool:
    return (
        isinstance(left, int | float)
        and not isinstance(left, bool)
        and math.isfinite(float(left))
        and abs(float(left) - float(right)) <= _BOUND_TOLERANCE
    )


def _normalise_excerpt(text: object) -> str:
    return " ".join(str(text or "").split())
