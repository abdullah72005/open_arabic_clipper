"""Candidate-scoped Stage 3.5 refinement orchestration.

The service turns one coarse Stage 3 candidate into a trustworthy bounded
transcript: it extracts only the candidate context window, runs targeted local
Whisper as the mandatory backbone, optionally consults hosted transcription and
adjudication under the shared admission gate, builds bounded evidence, resolves
omitted English only from audio-backed evidence, refines boundaries, and reports
a truthful lifecycle status. It never retranscribes or uploads a whole source.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import (
    EvidenceKind,
    EvidenceState,
    RefinementPriority,
)
from app.models import (
    AudioAnalysis,
    AudioArtifact,
    CandidateRefinement,
    ClipCandidate,
    SourceVideo,
    Transcript,
)
from app.refinement.audio_window import CandidateAudioWindowService
from app.refinement.boundary import BoundarySignals, refine_boundaries
from app.refinement.display import project_display_text
from app.refinement.entities import compare_entity_sets, extract_entities
from app.refinement.evidence import (
    choose_final_transcript,
    dedupe_evidence,
    select_consensus_text,
    validate_transcript_candidate,
)
from app.refinement.fingerprints import (
    component_fingerprint,
    refinement_input_fingerprint,
    refinement_output_fingerprint,
)
from app.refinement.hosted import HostedProviderError
from app.refinement.policy import (
    CONSENSUS_POLICY_VERSION,
    DEFAULT_CONFIG,
    FINAL_CLIP_SCOPE_POLICY_VERSION,
    REFINEMENT_POLICY_VERSION,
    REFINEMENT_SCHEMA_VERSION,
    REFINEMENT_VALIDATION_VERSION,
    Stage35Config,
)
from app.refinement.scope import (
    FINAL_CLIP_SCOPE_SPAN_ID,
    FinalClipView,
    ScopeRefusal,
    derive_final_clip_view,
)
from app.refinement.types import (
    AdjudicationRequest,
    BoundaryResult,
    EntityMention,
    EvidenceRecord,
    RefinementAudioWindow,
    RefinementCancelled,
    RefinementConfigurationError,
    RefinementContext,
    RefinementOutcome,
    TargetASRResult,
    UnresolvedSpan,
    WordTimestamp,
    admission_priority_for,
    priority_is_stage35,
)
from app.services.storage import StorageCategory, StorageService
from app.transcription.dialect import extract_protected_tokens
from app.transcription.service import TranscriptionOptions

_MIN_TEXT = 1e-6
_TOKEN_BOUNDARY = r"A-Za-z0-9\u0600-\u06FF\u0660-\u0669\u06F0-\u06F9"
_DECODER_OPTION_FIELDS = tuple(field.name for field in fields(TranscriptionOptions))


class RefinementError(RuntimeError):
    """A required Stage 3.5 input or invariant failed with no safe result."""


def _apply_entity_reading(base: str, readings: Sequence[str], selected: str) -> str | None:
    """Substitute one adjudicated entity reading into the base transcript.

    Returns the updated transcript, the unchanged base when the selected reading
    is already present, or ``None`` when no supplied reading can be applied as a
    bounded token substitution. The whole transcript is never replaced.
    """

    if not selected.strip():
        return None
    if not base.strip():
        return None
    pattern_selected = re.compile(
        rf"(?<![{_TOKEN_BOUNDARY}]){re.escape(selected)}(?![{_TOKEN_BOUNDARY}])"
    )
    if pattern_selected.search(base):
        return base
    for reading in readings:
        if reading == selected or not reading.strip():
            continue
        pattern = re.compile(
            rf"(?<![{_TOKEN_BOUNDARY}]){re.escape(reading)}(?![{_TOKEN_BOUNDARY}])"
        )
        updated, count = pattern.subn(lambda _m: selected, base, count=1)
        if count:
            return updated
    return None


def evaluate_timing_alignment(
    manual: str | None,
    automatic: str,
    words: Sequence[WordTimestamp | Mapping[str, object]],
) -> tuple[bool, str | None]:
    """Shared manual-vs-automatic timing-alignment gate.

    Used both by execution and by the manual-review endpoint so unrelated manual
    text can never be marked final-ready.
    """

    if not manual or not manual.strip():
        return True, None
    if not automatic.strip():
        return False, "manual_transcript_without_timing_evidence"
    if not words:
        return False, "manual_transcript_without_word_timestamps"
    projection = project_display_text(
        manual,
        [word.as_dict if isinstance(word, WordTimestamp) else word for word in words],
    )
    if not projection.supported:
        return False, projection.reason or "manual_transcript_cannot_be_aligned_safely"
    return True, None


def compose_refinement_output_fingerprint(
    *,
    input_fingerprint: str,
    priority: RefinementPriority,
    status: str,
    final_text: str,
    refined_start: float,
    refined_end: float,
    unresolved_ids: Sequence[str],
    final_clip_scope_fingerprint: str = "",
) -> str:
    """Compose the durable output identity, including any scoped clip view."""

    payload: dict[str, object] = {
        "input_fingerprint": input_fingerprint,
        "priority": priority.value,
        "status": status,
        "final_transcript": final_text,
        "refined_start": round(refined_start, 6),
        "refined_end": round(refined_end, 6),
        "unresolved": list(unresolved_ids),
    }
    if final_clip_scope_fingerprint:
        payload["final_clip_scope"] = final_clip_scope_fingerprint
    return refinement_output_fingerprint(payload)


@dataclass
class _Region:
    index_text: str
    stage27_text: str
    stage25_text: str
    raw_text: str
    operator_text: str | None
    words: tuple[WordTimestamp, ...]
    segment_indexes: tuple[int, ...] = ()


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _segment_text(segment: dict[str, object], key: str) -> str:
    value = segment.get(key)
    return value if isinstance(value, str) else ""


def _segment_words(segment: dict[str, object]) -> tuple[WordTimestamp, ...]:
    raw_words = segment.get("words")
    if not isinstance(raw_words, list):
        return ()
    words: list[WordTimestamp] = []
    for raw in raw_words:
        if not isinstance(raw, dict):
            continue
        start = raw.get("start")
        end = raw.get("end")
        text = raw.get("word") or raw.get("text")
        if not (_finite(start) and _finite(end)) or not isinstance(text, str):
            continue
        probability = raw.get("probability")
        words.append(
            WordTimestamp(
                text=text,
                start=float(start),  # type: ignore[arg-type]
                end=float(end),  # type: ignore[arg-type]
                probability=float(probability) if _finite(probability) else None,  # type: ignore[arg-type]
            )
        )
    return tuple(words)


def _checkpoint_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _option_value_matches(stored: object, expected: object) -> bool:
    """Compare persisted option values, tolerating JSON tuple/list round-trips."""

    if isinstance(stored, bool) or isinstance(expected, bool):
        return stored is expected
    if isinstance(stored, (tuple, list)) or isinstance(expected, (tuple, list)):
        if not isinstance(stored, (tuple, list)) or not isinstance(expected, (tuple, list)):
            return False
        return len(stored) == len(expected) and all(
            _option_value_matches(item, wanted)
            for item, wanted in zip(stored, expected, strict=True)
        )
    if isinstance(stored, (int, float)) and isinstance(expected, (int, float)):
        return float(stored) == float(expected)
    return stored == expected


def _checkpoint_words(value: object) -> tuple[WordTimestamp, ...] | None:
    if not isinstance(value, list):
        return None
    words: list[WordTimestamp] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        text = item.get("text")
        start = _checkpoint_float(item.get("start"))
        end = _checkpoint_float(item.get("end"))
        if not isinstance(text, str) or not text.strip() or start is None or end is None:
            return None
        if end <= start:
            return None
        probability = item.get("probability")
        probability_value = None if probability is None else _checkpoint_float(probability)
        if probability is not None and probability_value is None:
            return None
        words.append(
            WordTimestamp(
                text=text,
                start=start,
                end=end,
                probability=probability_value,
            )
        )
    return tuple(words)


class CandidateRefinementService:
    """Run one bounded CANDIDATE or FINAL_CLIP refinement."""

    def __init__(
        self,
        *,
        session: Session,
        storage: StorageService,
        config: Stage35Config = DEFAULT_CONFIG,
        audio_service: CandidateAudioWindowService | None = None,
        asr_engine: object | None = None,
        hosted_provider: object | None = None,
        adjudication_provider: object | None = None,
        admission: object | None = None,
        routing_mode: str = "adaptive",
        local_qwen_enabled: bool = False,
        qwen_reconstructor: object | None = None,
        local_identity: dict[str, object] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        if routing_mode not in {"adaptive", "local_only", "gemini_only"}:
            raise RefinementConfigurationError(
                f"unsupported refinement routing mode: {routing_mode}"
            )
        self._session = session
        self._storage = storage
        self._config = config
        self._audio = audio_service or CandidateAudioWindowService(
            storage=storage, session=session, config=config
        )
        self._asr: Any = asr_engine
        self._hosted: Any = hosted_provider
        self._adjudicator: Any = adjudication_provider
        self._admission: Any = admission
        self._routing_mode = routing_mode
        self._local_qwen_enabled = local_qwen_enabled
        self._qwen: Any = qwen_reconstructor
        self._local_identity = local_identity or {}
        self._is_cancelled = is_cancelled or (lambda: False)

    # ------------------------------------------------------------------
    # source loading

    def _source(self, candidate: ClipCandidate) -> SourceVideo:
        source = self._session.get(SourceVideo, candidate.source_video_id)
        if source is None:
            raise RefinementError("source video is missing for candidate refinement")
        return source

    def _transcript(self, source: SourceVideo) -> Transcript:
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if transcript is None:
            raise RefinementError("normalized transcript is required for candidate refinement")
        return transcript

    def _analysis(self, source: SourceVideo) -> AudioAnalysis | None:
        return self._session.scalar(
            select(AudioAnalysis).where(AudioAnalysis.source_video_id == source.id)
        )

    def _artifact(self, source: SourceVideo) -> AudioArtifact:
        artifact = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        if artifact is None:
            raise RefinementError("cached audio artifact is required for candidate refinement")
        return artifact

    # ------------------------------------------------------------------
    # fingerprints

    def _hosted_identity(self) -> dict[str, object]:
        if self._hosted is None:
            return {"provider": "none", "routing_mode": self._routing_mode}
        identity = getattr(self._hosted, "runtime_identity", None)
        base = dict(identity()) if callable(identity) else {}
        base["routing_mode"] = self._routing_mode
        return base

    def _adjudication_identity(self) -> dict[str, object]:
        if self._adjudicator is None:
            return {"provider": "none"}
        identity = getattr(self._adjudicator, "runtime_identity", None)
        return dict(identity()) if callable(identity) else {}

    def _qwen_identity(self) -> dict[str, object]:
        if not (
            self._routing_mode == "local_only"
            and self._local_qwen_enabled
            and self._qwen is not None
        ):
            return {"selected": False}
        identity = getattr(self._qwen, "runtime_identity", None)
        base = dict(identity()) if callable(identity) else {"provider": "ollama"}
        base["selected"] = True
        return base

    def _admission_identity(self) -> dict[str, object]:
        if self._admission is None:
            return {}
        identity = getattr(self._admission, "runtime_identity", None)
        return dict(identity()) if callable(identity) else {}

    def _segment_dependency(
        self, transcript: Transcript, region: _Region
    ) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for index in region.segment_indexes:
            segment = transcript.segments[index]
            payload.append(
                {
                    "index": index,
                    "raw": _segment_text(segment, "raw_text") or _segment_text(segment, "text"),
                    "corrected": _segment_text(segment, "corrected_text"),
                    "final": _segment_text(segment, "final_text"),
                    "operator": _segment_text(segment, "operator_text"),
                    "start": round(float(segment.get("start", 0.0)), 6),
                    "end": round(float(segment.get("end", 0.0)), 6),
                    "words": [
                        [word.text, round(word.start, 6), round(word.end, 6)]
                        for word in region.words
                    ],
                }
            )
        return payload

    def seed_bounds(
        self, candidate: ClipCandidate, priority: RefinementPriority
    ) -> tuple[float, float]:
        """Expose the durable coarse seed used for a refinement request."""

        return self._seed_bounds(candidate, priority)

    def input_fingerprint(self, candidate: ClipCandidate, priority: RefinementPriority) -> str:
        if not priority_is_stage35(priority):
            raise RefinementConfigurationError("Stage 3.5 only supports CANDIDATE/FINAL_CLIP")
        source = self._source(candidate)
        transcript = self._transcript(source)
        artifact = self._artifact(source)
        coarse_start, coarse_end = self._seed_bounds(candidate, priority)
        context_start, context_end = self._audio.context_bounds(
            coarse_start=coarse_start,
            coarse_end=coarse_end,
            source_duration=artifact.duration,
            priority=priority,
        )
        audio_hash = self._audio.audio_input_fingerprint(
            source=source,
            artifact=artifact,
            context_start=context_start,
            context_end=context_end,
            priority=priority,
        )
        return refinement_input_fingerprint(
            {
                "source_id": str(source.id),
                "source_content_hash": source.content_hash,
                "artifact_content_hash": artifact.content_hash,
                "candidate_id": str(candidate.id),
                "candidate_key": candidate.candidate_key,
                "analysis_fingerprint": candidate.analysis_fingerprint,
                "span": [candidate.start_segment_index, candidate.end_segment_index],
                "segment_indexes": list(candidate.segment_indexes or []),
                "coarse_start": round(coarse_start, 6),
                "coarse_end": round(coarse_end, 6),
                "context_start": round(context_start, 6),
                "context_end": round(context_end, 6),
                "priority": priority.value,
                "audio_input_fingerprint": audio_hash,
                "local_identity": self._local_identity,
                "hosted_identity": self._hosted_identity(),
                "adjudication_identity": self._adjudication_identity(),
                "qwen_identity": self._qwen_identity(),
                "admission_identity": self._admission_identity(),
                "dialect_profile": candidate.dialect_profile,
                "dialect_confidence": round(float(candidate.dialect_confidence or 0.0), 6),
                "transcription_revision": transcript.transcription_revision,
                "transcript_input_fingerprint": transcript.input_fingerprint,
                "policy_version": REFINEMENT_POLICY_VERSION,
                "schema_version": REFINEMENT_SCHEMA_VERSION,
                "validation_version": REFINEMENT_VALIDATION_VERSION,
                "consensus_version": CONSENSUS_POLICY_VERSION,
                "final_clip_scope_policy_version": (
                    FINAL_CLIP_SCOPE_POLICY_VERSION
                    if priority is RefinementPriority.FINAL_CLIP
                    else None
                ),
            }
        )

    # ------------------------------------------------------------------
    # execution

    def execute(
        self,
        candidate: ClipCandidate,
        *,
        priority: RefinementPriority,
        force: bool = False,
        prior: CandidateRefinement | None = None,
    ) -> RefinementOutcome:
        if not priority_is_stage35(priority):
            raise RefinementConfigurationError("Stage 3.5 only supports CANDIDATE/FINAL_CLIP")
        source = self._source(candidate)
        transcript = self._transcript(source)
        artifact = self._artifact(source)
        seed_start, seed_end = self._seed_bounds(candidate, priority)
        context_start, context_end = self._audio.context_bounds(
            coarse_start=seed_start,
            coarse_end=seed_end,
            source_duration=artifact.duration,
            priority=priority,
        )
        self._check_cancelled()
        window = self._audio.extract(
            source=source, candidate=candidate, priority=priority, force=force
        )
        audio_path = self._storage.resolve(StorageCategory.SOURCES, window.relative_path)
        if not audio_path.is_file():
            raise RefinementError("extracted refinement audio is unavailable")
        self._check_cancelled()

        ctx = RefinementContext()
        region = self._region(transcript, window.context_start, window.context_end)
        evidence = list(self._existing_evidence(transcript, region))
        source_dialect = transcript.dialect_profile.value if transcript.dialect_profile else None
        index_text = region.index_text

        audio_component = component_fingerprint(
            "audio-extraction",
            {
                "audio_input_fingerprint": window.input_fingerprint,
                "content_hash": window.content_hash,
            },
        )
        ctx.component_fingerprints["audio_extraction"] = audio_component

        local_record, local_error = self._local_asr(
            audio_path=audio_path,
            priority=priority,
            window=window,
            region=region,
            index_text=index_text,
            source_dialect=source_dialect,
            prior=prior,
            audio_component=audio_component,
            force=force,
            ctx=ctx,
        )
        if local_error is not None:
            raise RefinementError(local_error)
        if local_record is not None:
            evidence.append(local_record)

        qwen_text = self._qwen_evidence(
            transcript=transcript,
            region=region,
            priority=priority,
            index_text=index_text,
            source_dialect=source_dialect,
            evidence=evidence,
            ctx=ctx,
        )

        provider_degraded = False
        hosted_record = self._hosted_transcription(
            audio_path=audio_path,
            priority=priority,
            window=window,
            region=region,
            index_text=index_text,
            source_dialect=source_dialect,
            local_record=local_record,
            prior=prior,
            audio_component=audio_component,
            force=force,
            ctx=ctx,
        )
        if isinstance(hosted_record, _Degraded):
            provider_degraded = True
        elif hosted_record is not None:
            evidence.append(hosted_record)

        audio_records = [
            record
            for record in evidence
            if record.kind in {EvidenceKind.TARGETED_LOCAL_ASR, EvidenceKind.HOSTED_ASR}
            and record.state is EvidenceState.ACCEPTED
        ]
        self._check_cancelled()
        consensus_text, consensus_confidence, _ = select_consensus_text(evidence)
        base_text = consensus_text or (local_record.transcript if local_record else "") or qwen_text
        adjudicated_text, adjudication_spans, handled_ambiguities = self._adjudicate(
            audio_path=audio_path,
            priority=priority,
            region=region,
            audio_records=audio_records,
            source_dialect=source_dialect,
            base_text=base_text,
            ctx=ctx,
        )
        if adjudicated_text is not None:
            evidence.append(
                EvidenceRecord(
                    kind=EvidenceKind.ADJUDICATION,
                    fingerprint=component_fingerprint(
                        "adjudication-accepted",
                        {"text": adjudicated_text, "audio": audio_component},
                    ),
                    provider="gemini",
                    model=getattr(self._adjudicator, "model", None),
                    settings={},
                    window_start=window.context_start,
                    window_end=window.context_end,
                    transcript=adjudicated_text,
                    confidence=1.0,
                    state=EvidenceState.ACCEPTED,
                )
            )

        automatic = adjudicated_text or base_text
        manual = prior.manual_transcript if prior is not None else None
        operator_segment_text = region.operator_text

        word_timestamps = self._choose_words(local_record)
        boundary = refine_boundaries(
            BoundarySignals(
                coarse_start=seed_start,
                coarse_end=seed_end,
                context_start=window.context_start,
                context_end=window.context_end,
                word_timestamps=word_timestamps,
                source_word_timestamps=region.words,
                silence_intervals=self._silence_intervals(source),
                radius_seconds=self._config.boundary_search_radius_seconds,
                min_duration_seconds=0.5,
            )
        )
        ctx.component_fingerprints["boundary"] = component_fingerprint(
            "boundary", boundary.as_dict()
        )

        scope = (
            self._final_clip_scope(local_record, window, boundary)
            if priority is RefinementPriority.FINAL_CLIP
            else None
        )
        scoped_view = scope if isinstance(scope, FinalClipView) else None

        if scoped_view is not None:
            # The authoritative automatic rung is the validated clip view. Full
            # context consensus/adjudication cannot be silently reinterpreted as
            # clip text; manual/operator text still wins but must project.
            final_text = choose_final_transcript(
                manual=manual,
                adjudicated=None,
                operator_segment_text=operator_segment_text,
                consensus=None,
                automatic=scoped_view.text,
                stage27_text=None,
                stage25_text=None,
                raw_text=None,
            )
            word_timestamps = tuple(word.as_timestamp() for word in scoped_view.words)
            if final_text.strip() == scoped_view.text:
                timing_aligned, timing_reason = True, None
            else:
                timing_aligned, timing_reason = self._timing_alignment(
                    final_text, scoped_view.text, word_timestamps
                )
        else:
            final_text = choose_final_transcript(
                manual=manual,
                adjudicated=adjudicated_text,
                operator_segment_text=operator_segment_text,
                consensus=consensus_text or None,
                automatic=automatic or None,
                stage27_text=region.stage27_text or None,
                stage25_text=region.stage25_text or None,
                raw_text=region.raw_text or None,
            )
            timing_aligned, timing_reason = self._timing_alignment(
                manual, automatic, word_timestamps
            )

        entities, unresolved = self._entities_and_ambiguity(
            final_text=final_text,
            region=region,
            audio_records=audio_records,
            priority=priority,
            adjudication_spans=adjudication_spans,
            handled_ambiguities=handled_ambiguities,
            provider_degraded=provider_degraded,
        )
        if isinstance(scope, ScopeRefusal):
            unresolved = unresolved + (
                UnresolvedSpan(
                    span_id=FINAL_CLIP_SCOPE_SPAN_ID,
                    start=boundary.start,
                    end=boundary.end,
                    context=(final_text or "")[: self._config.max_context_characters],
                    readings=(),
                    evidence_fingerprints=(
                        local_record.fingerprint if local_record is not None else "",
                    ),
                    providers=("faster-whisper",),
                    confidence=0.0,
                    reason=scope.reason,
                    entity_type=None,
                    meaning_critical=True,
                ),
            )

        omitted_english = self._omitted_english(index_text, automatic)
        if omitted_english:
            ctx.bump("code_switch_recoveries")
            ctx.routing_evidence["code_switch_recovered"] = list(omitted_english)

        meaning_critical = any(span.meaning_critical for span in unresolved)
        if not timing_aligned:
            unresolved = unresolved + (
                UnresolvedSpan(
                    span_id="timing-alignment",
                    start=boundary.start,
                    end=boundary.end,
                    context=manual or final_text or "",
                    readings=(automatic,) if automatic else (),
                    evidence_fingerprints=(),
                    providers=("operator",),
                    confidence=0.0,
                    reason=timing_reason or "timing_alignment_unresolved",
                    entity_type=None,
                    meaning_critical=priority is RefinementPriority.FINAL_CLIP,
                ),
            )
            meaning_critical = any(span.meaning_critical for span in unresolved)

        status = self._status(
            priority=priority,
            local_record=local_record,
            audio_records=audio_records,
            final_text=final_text,
            manual=manual,
            meaning_critical=meaning_critical and timing_aligned,
            provider_degraded=provider_degraded,
            consensus_confidence=consensus_confidence,
            timing_aligned=timing_aligned,
            scope_refused=isinstance(scope, ScopeRefusal),
        )
        confidence = self._confidence(
            priority=priority,
            local_record=local_record,
            hosted_present=any(record.kind is EvidenceKind.HOSTED_ASR for record in audio_records),
            consensus_confidence=consensus_confidence,
            unresolved=unresolved,
        )
        cache_eligible = (
            status in {"CANDIDATE_REFINED", "FINAL_TRANSCRIPT_READY"}
            and not provider_degraded
            and not any(span.meaning_critical for span in unresolved)
        )
        input_fingerprint = self.input_fingerprint(candidate, priority)
        output_fingerprint = compose_refinement_output_fingerprint(
            input_fingerprint=input_fingerprint,
            priority=priority,
            status=status,
            final_text=final_text,
            refined_start=boundary.start,
            refined_end=boundary.end,
            unresolved_ids=[span.span_id for span in unresolved],
            final_clip_scope_fingerprint=scope.fingerprint if scope is not None else "",
        )
        transcript_evidence = dedupe_evidence(evidence, limit=self._config.max_evidence_records)
        provider_evidence: dict[str, object] = {
            "local_asr_accepted": bool(
                local_record and local_record.state is EvidenceState.ACCEPTED
            ),
            "hosted_used": any(record.kind is EvidenceKind.HOSTED_ASR for record in audio_records),
            "adjudication_used": adjudicated_text is not None,
            "provider_degraded": provider_degraded,
            "routing_mode": self._routing_mode,
        }
        if scope is not None:
            provider_evidence["final_clip_view"] = scope.envelope()
        return RefinementOutcome(
            source_id=str(source.id),
            candidate_id=str(candidate.id),
            priority=priority,
            quality_level=priority.value,
            status=status,
            coarse_start=seed_start,
            coarse_end=seed_end,
            context_start=window.context_start,
            context_end=window.context_end,
            refined_start=boundary.start,
            refined_end=boundary.end,
            audio_relative_path=window.relative_path,
            audio_content_hash=window.content_hash,
            audio_input_fingerprint=window.input_fingerprint,
            automatic_transcript=automatic,
            manual_transcript=manual,
            final_transcript=final_text,
            word_timestamps=word_timestamps,
            confidence=confidence,
            dialect_profile=source_dialect,
            dialect_confidence=float(transcript.dialect_confidence or 0.0),
            code_switch_evidence={
                "suspected": bool(candidate.code_switch_suspected),
                "recovered": list(omitted_english),
            },
            transcript_evidence=transcript_evidence,
            entity_evidence=entities,
            unresolved_spans=unresolved[: self._config.max_unresolved_spans],
            provider_evidence=provider_evidence,
            routing_evidence=dict(ctx.routing_evidence),
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fingerprint,
            component_fingerprints=dict(ctx.component_fingerprints),
            cache_eligible=cache_eligible,
            metrics=dict(ctx.metrics),
            processing_duration=0.0,
        )

    # ------------------------------------------------------------------
    # region and existing evidence

    def _region(self, transcript: Transcript, context_start: float, context_end: float) -> _Region:
        indexes = [
            index
            for index, segment in enumerate(transcript.segments)
            if float(segment.get("end", 0.0)) > context_start
            and float(segment.get("start", 0.0)) < context_end
        ]
        index_parts: list[str] = []
        stage27_parts: list[str] = []
        stage25_parts: list[str] = []
        raw_parts: list[str] = []
        operator_parts: list[str] = []
        words: list[WordTimestamp] = []
        for index in indexes:
            segment = transcript.segments[index]
            raw = _segment_text(segment, "raw_text") or _segment_text(segment, "text")
            corrected = _segment_text(segment, "corrected_text")
            stage27 = _segment_text(segment, "contextual_reconstructed_text")
            operator = _segment_text(segment, "operator_text")
            raw_parts.append(raw)
            stage25_parts.append(corrected)
            stage27_parts.append(stage27)
            operator_parts.append(operator)
            index_parts.append(_segment_text(segment, "final_text") or stage27 or corrected or raw)
            words.extend(_segment_words(segment))
        return _Region(
            index_text=" ".join(part for part in index_parts if part).strip(),
            stage27_text=" ".join(part for part in stage27_parts if part).strip(),
            stage25_text=" ".join(part for part in stage25_parts if part).strip(),
            raw_text=" ".join(part for part in raw_parts if part).strip(),
            operator_text=" ".join(part for part in operator_parts if part).strip() or None,
            words=tuple(sorted(words, key=lambda word: (word.start, word.end))),
            segment_indexes=tuple(indexes),
        )

    def _existing_evidence(self, transcript: Transcript, region: _Region) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        base = {
            "window_start": 0.0,
            "window_end": transcript.duration or 0.0,
        }

        def add(kind: EvidenceKind, text: str, confidence: float) -> None:
            if not text.strip():
                return
            records.append(
                EvidenceRecord(
                    kind=kind,
                    fingerprint=component_fingerprint(
                        kind.value.lower(),
                        {"text": text, "revision": transcript.transcription_revision},
                    ),
                    provider="local",
                    model=None,
                    settings={},
                    window_start=base["window_start"],
                    window_end=base["window_end"],
                    transcript=text,
                    confidence=confidence,
                    state=EvidenceState.ACCEPTED,
                )
            )

        add(
            EvidenceKind.INDEX_RAW,
            region.raw_text or region.index_text,
            float(transcript.raw_transcript_confidence or 0.0),
        )
        add(
            EvidenceKind.STAGE25,
            region.stage25_text,
            float(transcript.correction_confidence or 0.0),
        )
        if region.stage27_text:
            add(
                EvidenceKind.STAGE27,
                region.stage27_text,
                float(transcript.reconstruction_confidence or 0.0),
            )
        if region.operator_text:
            add(EvidenceKind.OPERATOR, region.operator_text, 1.0)
        return records

    def _silence_intervals(self, source: SourceVideo) -> tuple[tuple[float, float], ...]:
        analysis = self._analysis(source)
        if analysis is None:
            return ()
        intervals: list[tuple[float, float]] = []
        for raw in analysis.silence_intervals or []:
            if not isinstance(raw, dict):
                continue
            start, end = raw.get("start"), raw.get("end")
            if _finite(start) and _finite(end) and float(end) >= float(start):  # type: ignore[arg-type]
                intervals.append((float(start), float(end)))  # type: ignore[arg-type]
        return tuple(intervals)

    # ------------------------------------------------------------------
    # local ASR

    def _qwen_evidence(
        self,
        *,
        transcript: Transcript,
        region: _Region,
        priority: RefinementPriority,
        index_text: str,
        source_dialect: str | None,
        evidence: list[EvidenceRecord],
        ctx: RefinementContext,
    ) -> str:
        """Optional explicit ``local_only`` Qwen text reconstruction.

        Runs only when the operator explicitly enabled local Qwen and selected
        ``local_only``. It is text-only, so it can never add Latin/code-switch
        content that the INDEX transcript does not already contain.
        """

        if not (
            self._routing_mode == "local_only"
            and self._local_qwen_enabled
            and self._qwen is not None
        ):
            return ""
        self._check_cancelled()
        try:
            result = self._qwen.reconstruct(
                transcript.segments,
                language=transcript.language,
                transcription_fingerprint=transcript.input_fingerprint,
                correction_version=transcript.correction_version,
                target_indexes=region.segment_indexes,
                priority=priority,
            )
        except Exception:
            ctx.bump("provider_failures")
            return ""
        text = getattr(result, "contextual_reconstructed_text", "") or ""
        if not text.strip():
            return ""
        ctx.bump("qwen_runs")
        accepted, reason = self._validate(
            candidate_text=text,
            reference_text=index_text,
            words=(),
            source_dialect=source_dialect,
            allow_unsupported_latin=False,
        )
        evidence.append(
            EvidenceRecord(
                kind=EvidenceKind.STAGE27,
                fingerprint=component_fingerprint(
                    "qwen-reconstruction",
                    {"text": text, "revision": transcript.transcription_revision},
                ),
                provider="ollama",
                model=getattr(self._qwen, "model", None),
                settings={},
                window_start=0.0,
                window_end=float(transcript.duration or 0.0),
                transcript=text,
                confidence=float(transcript.reconstruction_confidence or 0.0),
                state=EvidenceState.ACCEPTED if accepted else EvidenceState.REJECTED,
                reason=reason,
            )
        )
        return text if accepted else ""

    def _local_asr(
        self,
        *,
        audio_path: Path,
        priority: RefinementPriority,
        window: RefinementAudioWindow,
        region: _Region,
        index_text: str,
        source_dialect: str | None,
        prior: CandidateRefinement | None,
        audio_component: str,
        force: bool,
        ctx: RefinementContext,
    ) -> tuple[EvidenceRecord | None, str | None]:
        if self._asr is None:
            return None, None
        reused = self._reuse_local(prior, audio_component, priority, force)
        if reused is not None:
            ctx.bump("component_reuse")
            ctx.component_fingerprints["local_asr"] = reused.fingerprint
            return reused, None
        context_terms = self._context_terms(region, index_text)
        observation_reuse = self._reuse_rejected_local_observations(
            prior,
            audio_component,
            priority,
            force,
            window=window,
            context_terms=context_terms,
        )
        if observation_reuse is not None:
            ctx.bump("component_reuse")
            ctx.component_fingerprints["local_asr"] = observation_reuse.fingerprint
            return observation_reuse, None
        cancel_event = threading.Event()
        self._check_cancelled()
        try:
            result: TargetASRResult = self._asr.transcribe(
                audio_path,
                context_start=window.context_start,
                context_end=window.context_end,
                priority=priority,
                cancel_event=cancel_event,
                context_terms=context_terms,
            )
        except RefinementCancelled:
            raise
        except Exception as error:
            return None, f"targeted local ASR failed: {type(error).__name__}"
        ctx.bump("local_asr_runs")
        ctx.component_fingerprints["local_asr"] = result.fingerprint or component_fingerprint(
            "local-asr", {"transcript": result.transcript}
        )
        accepted, reason = self._validate(
            candidate_text=result.transcript,
            reference_text=index_text,
            words=result.word_timestamps,
            source_dialect=source_dialect,
            allow_unsupported_latin=True,
        )
        if result.rejected_reason:
            accepted, reason = False, result.rejected_reason
        entity_metadata: dict[str, object] = {}
        if result.rejected_detail is not None:
            entity_metadata["timestamp_diagnostic"] = result.rejected_detail
        if result.raw_word_observations:
            entity_metadata["raw_word_observations"] = list(result.raw_word_observations)
            entity_metadata["raw_word_record_count"] = result.raw_word_record_count
            entity_metadata["raw_word_observations_complete"] = (
                result.raw_word_observations_complete
            )
            entity_metadata["raw_word_observations_fingerprint"] = (
                result.raw_word_observations_fingerprint
            )
        return (
            EvidenceRecord(
                kind=EvidenceKind.TARGETED_LOCAL_ASR,
                fingerprint=result.fingerprint
                or component_fingerprint("local-asr", {"transcript": result.transcript}),
                provider=result.provider,
                model=result.model,
                settings=dict(result.settings),
                window_start=window.context_start,
                window_end=window.context_end,
                transcript=result.transcript,
                confidence=result.confidence,
                state=EvidenceState.ACCEPTED if accepted else EvidenceState.REJECTED,
                word_timestamps=result.word_timestamps,
                reason=reason,
                entity_metadata=entity_metadata,
            ),
            None,
        )

    def _reuse_local(
        self,
        prior: CandidateRefinement | None,
        audio_component: str,
        priority: RefinementPriority,
        force: bool,
    ) -> EvidenceRecord | None:
        if prior is None or force or prior.quality_level != priority.value:
            return None
        fingerprints = prior.component_fingerprints or {}
        if fingerprints.get("audio_extraction") != audio_component:
            return None
        fingerprint = fingerprints.get("local_asr")
        if not isinstance(fingerprint, str) or not fingerprint:
            return None

        local_records = [
            item
            for item in prior.transcript_evidence or []
            if isinstance(item, dict) and item.get("kind") == EvidenceKind.TARGETED_LOCAL_ASR.value
        ]
        if len(local_records) != 1:
            return None
        record = local_records[0]
        if record.get("state") != EvidenceState.ACCEPTED.value:
            return None
        if record.get("fingerprint") != fingerprint:
            return None

        transcript = record.get("transcript")
        provider = record.get("provider")
        model = record.get("model")
        settings = record.get("settings")
        confidence = _checkpoint_float(record.get("confidence"))
        window_start = _checkpoint_float(record.get("window_start"))
        window_end = _checkpoint_float(record.get("window_end"))
        words = _checkpoint_words(record.get("word_timestamps"))
        entity_metadata = record.get("entity_metadata", {})
        code_switch_tokens = record.get("code_switch_tokens", [])
        if (
            not isinstance(transcript, str)
            or not transcript.strip()
            or not isinstance(provider, str)
            or not provider.strip()
            or (model is not None and not isinstance(model, str))
            or not isinstance(settings, dict)
            or confidence is None
            or not 0.0 <= confidence <= 1.0
            or window_start is None
            or window_end is None
            or window_end <= window_start
            or abs(window_start - float(prior.context_start)) > _MIN_TEXT
            or abs(window_end - float(prior.context_end)) > _MIN_TEXT
            or words is None
            or not isinstance(entity_metadata, dict)
            or not isinstance(code_switch_tokens, list)
            or not all(isinstance(token, str) for token in code_switch_tokens)
        ):
            return None

        return EvidenceRecord(
            kind=EvidenceKind.TARGETED_LOCAL_ASR,
            fingerprint=fingerprint,
            provider=provider,
            model=model,
            settings=dict(settings),
            window_start=window_start,
            window_end=window_end,
            transcript=transcript,
            confidence=confidence,
            state=EvidenceState.ACCEPTED,
            word_timestamps=words,
            entity_metadata=dict(entity_metadata),
            code_switch_tokens=tuple(code_switch_tokens),
        )

    def _reuse_rejected_local_observations(
        self,
        prior: CandidateRefinement | None,
        audio_component: str,
        priority: RefinementPriority,
        force: bool,
        *,
        window: RefinementAudioWindow,
        context_terms: Sequence[str],
    ) -> EvidenceRecord | None:
        """Reuse complete raw observations without promoting rejected evidence.

        R3's accepted-checkpoint path is unchanged. This FINAL_CLIP-only path
        restores the original rejected context record as *observations* so
        deterministic scope validation can run again; it never changes the
        record to ACCEPTED and refuses legacy rows that lack complete raw
        observations. Reuse also requires the stored observation's decoder and
        context dependencies to match the current request exactly.
        """

        if priority is not RefinementPriority.FINAL_CLIP:
            return None
        if prior is None or force or prior.quality_level != priority.value:
            return None
        fingerprints = prior.component_fingerprints or {}
        if fingerprints.get("audio_extraction") != audio_component:
            return None
        fingerprint = fingerprints.get("local_asr")
        if not isinstance(fingerprint, str) or not fingerprint:
            return None
        local_records = [
            item
            for item in prior.transcript_evidence or []
            if isinstance(item, dict) and item.get("kind") == EvidenceKind.TARGETED_LOCAL_ASR.value
        ]
        if len(local_records) != 1:
            return None
        record = local_records[0]
        if record.get("state") != EvidenceState.REJECTED.value:
            return None
        if record.get("fingerprint") != fingerprint:
            return None
        if not self._rejected_observation_identity_matches(
            record, priority=priority, window=window, context_terms=context_terms
        ):
            return None
        entity_metadata = record.get("entity_metadata")
        if not isinstance(entity_metadata, dict):
            return None
        observations = entity_metadata.get("raw_word_observations")
        record_count = entity_metadata.get("raw_word_record_count")
        complete = entity_metadata.get("raw_word_observations_complete")
        observation_fingerprint = entity_metadata.get("raw_word_observations_fingerprint")
        if (
            not isinstance(observations, list)
            or not observations
            or not all(isinstance(item, dict) for item in observations)
            or isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count <= 0
            or complete is not True
            or not isinstance(observation_fingerprint, str)
            or not observation_fingerprint
        ):
            return None
        transcript = record.get("transcript")
        provider = record.get("provider")
        model = record.get("model")
        settings = record.get("settings")
        confidence = _checkpoint_float(record.get("confidence"))
        window_start = _checkpoint_float(record.get("window_start"))
        window_end = _checkpoint_float(record.get("window_end"))
        words = _checkpoint_words(record.get("word_timestamps"))
        if (
            not isinstance(transcript, str)
            or not transcript.strip()
            or not isinstance(provider, str)
            or not provider.strip()
            or (model is not None and not isinstance(model, str))
            or not isinstance(settings, dict)
            or confidence is None
            or not 0.0 <= confidence <= 1.0
            or window_start is None
            or window_end is None
            or window_end <= window_start
            or abs(window_start - float(prior.context_start)) > _MIN_TEXT
            or abs(window_end - float(prior.context_end)) > _MIN_TEXT
            or words is None
        ):
            return None
        return EvidenceRecord(
            kind=EvidenceKind.TARGETED_LOCAL_ASR,
            fingerprint=fingerprint,
            provider=provider,
            model=model,
            settings=dict(settings),
            window_start=window_start,
            window_end=window_end,
            transcript=transcript,
            confidence=confidence,
            state=EvidenceState.REJECTED,
            word_timestamps=words,
            reason=record.get("reason") if isinstance(record.get("reason"), str) else None,
            entity_metadata=dict(entity_metadata),
        )

    def local_observation_reuse_ready(
        self,
        prior: CandidateRefinement | None,
        *,
        audio_component: str,
        priority: RefinementPriority,
        window: RefinementAudioWindow,
        context_terms: Sequence[str],
    ) -> bool:
        """Read-only preflight for rejected-observation reuse; never calls a model."""

        return (
            self._reuse_rejected_local_observations(
                prior,
                audio_component,
                priority,
                False,
                window=window,
                context_terms=context_terms,
            )
            is not None
        )

    def _rejected_observation_identity_matches(
        self,
        record: dict[str, object],
        *,
        priority: RefinementPriority,
        window: RefinementAudioWindow,
        context_terms: Sequence[str],
    ) -> bool:
        """True only when stored observations match the current local request.

        A record matching its own stored fingerprint proves it was not corrupted;
        it does not prove the current executor asked for the same decoder options
        or bounded context. The complete effective ``TranscriptionOptions``
        identity supplied by the production factory is compared field by field,
        including compute/fallback precision and temperature. Missing identity
        information is never treated as a match. A mismatch falls through to the
        normal fresh backbone path instead of relabelling old observations under
        a new input identity.
        """

        identity = self._local_identity or {}
        provider = identity.get("provider")
        model = identity.get("model")
        if not isinstance(provider, str) or provider != record.get("provider"):
            return False
        if not isinstance(model, str) or model != record.get("model"):
            return False
        settings = record.get("settings")
        if not isinstance(settings, dict):
            return False
        legacy_device = identity.get("device")
        if legacy_device is not None and settings.get("device") != legacy_device:
            return False
        beam = identity.get(
            "final_beam_size"
            if priority is RefinementPriority.FINAL_CLIP
            else "candidate_beam_size"
        )
        if beam is not None and not _option_value_matches(settings.get("beam_size"), beam):
            return False
        decoder_options = identity.get("decoder_options")
        if not isinstance(decoder_options, Mapping):
            return False
        expected_options = decoder_options.get(priority.value)
        if not isinstance(expected_options, Mapping):
            return False
        hint = " ".join(term.strip() for term in context_terms if term and term.strip()) or None
        for field_name in _DECODER_OPTION_FIELDS:
            if field_name not in settings or field_name not in expected_options:
                return False
            expected_value = expected_options[field_name]
            if field_name == "hotwords" and hint is not None:
                expected_value = hint
            if not _option_value_matches(settings[field_name], expected_value):
                return False
        stored_start = _checkpoint_float(settings.get("context_start"))
        stored_end = _checkpoint_float(settings.get("context_end"))
        if stored_start is None:
            stored_start = _checkpoint_float(record.get("window_start"))
        if stored_end is None:
            stored_end = _checkpoint_float(record.get("window_end"))
        if stored_start is None or stored_end is None:
            return False
        return (
            abs(stored_start - window.context_start) <= _MIN_TEXT
            and abs(stored_end - window.context_end) <= _MIN_TEXT
        )

    def _final_clip_scope(
        self,
        local_record: EvidenceRecord | None,
        window: RefinementAudioWindow,
        boundary: BoundaryResult,
    ) -> FinalClipView | ScopeRefusal | None:
        """Derive the executable clip view from complete raw observations."""

        if local_record is None:
            return None
        metadata = local_record.entity_metadata or {}
        raw_observations = metadata.get("raw_word_observations")
        if not isinstance(raw_observations, list) or not raw_observations:
            # Legacy rows preserve only the filtered words and first diagnostic.
            return None
        records = [item for item in raw_observations if isinstance(item, dict)]
        record_count = metadata.get("raw_word_record_count")
        complete = metadata.get("raw_word_observations_complete")
        observation_fingerprint = metadata.get("raw_word_observations_fingerprint")
        return derive_final_clip_view(
            raw_observations=records,
            record_count=(
                record_count
                if isinstance(record_count, int) and not isinstance(record_count, bool)
                else 0
            ),
            observations_complete=complete is True and len(records) == len(raw_observations),
            observation_fingerprint=(
                observation_fingerprint if isinstance(observation_fingerprint, str) else ""
            ),
            transcript=local_record.transcript,
            context_start=window.context_start,
            context_end=window.context_end,
            clip_start=boundary.start,
            clip_end=boundary.end,
        )

    # ------------------------------------------------------------------
    # hosted transcription

    def _hosted_transcription(
        self,
        *,
        audio_path: Path,
        priority: RefinementPriority,
        window: RefinementAudioWindow,
        region: _Region,
        index_text: str,
        source_dialect: str | None,
        local_record: EvidenceRecord | None,
        prior: CandidateRefinement | None,
        audio_component: str,
        force: bool,
        ctx: RefinementContext,
    ) -> "EvidenceRecord | _Degraded | None":
        if self._routing_mode == "local_only" or self._hosted is None:
            return None
        reused = self._reuse_hosted(prior, audio_component, priority, force, ctx)
        if reused is not None:
            return reused
        if not self._wants_hosted(priority, local_record, ctx):
            return None
        available = getattr(self._hosted, "available", None)
        if callable(available) and not available():
            return None
        if not self._admitted(priority, ctx):
            return _Degraded()
        self._check_cancelled()
        try:
            result = self._hosted.transcribe(
                audio_path,
                language_codes=[source_dialect] if source_dialect else None,
                dialect_profile=source_dialect,
                custom_vocabulary=(),
            )
        except HostedProviderError as error:
            self._handle_hosted_error(error, ctx)
            return _Degraded()
        except Exception:
            ctx.bump("provider_failures")
            return _Degraded()
        ctx.bump("hosted_transcription_calls")
        # Provider offsets are window-relative: convert once to source time.
        words = tuple(
            WordTimestamp(
                text=word.text,
                start=word.start + window.context_start,
                end=word.end + window.context_start,
                probability=word.probability,
            )
            for word in result.word_timestamps
            if window.context_start <= word.start + window.context_start < window.context_end
        )
        accepted, reason = self._validate(
            candidate_text=result.transcript,
            reference_text=index_text,
            words=words,
            source_dialect=source_dialect,
            allow_unsupported_latin=True,
        )
        if result.rejected_reason:
            accepted, reason = False, result.rejected_reason
        ctx.component_fingerprints["hosted_transcription"] = component_fingerprint(
            "hosted-transcription", {"transcript": result.transcript, "audio": audio_component}
        )
        return EvidenceRecord(
            kind=EvidenceKind.HOSTED_ASR,
            fingerprint=component_fingerprint(
                "hosted-transcription", {"transcript": result.transcript, "audio": audio_component}
            ),
            provider="gemini",
            model=result.model,
            settings=dict(result.usage),
            window_start=window.context_start,
            window_end=window.context_end,
            transcript=result.transcript,
            confidence=result.confidence,
            state=EvidenceState.ACCEPTED if accepted else EvidenceState.REJECTED,
            word_timestamps=words,
            reason=reason,
        )

    def _reuse_hosted(
        self,
        prior: CandidateRefinement | None,
        audio_component: str,
        priority: RefinementPriority,
        force: bool,
        ctx: RefinementContext,
    ) -> EvidenceRecord | None:
        if prior is None or force or prior.quality_level != priority.value:
            return None
        fingerprints = prior.component_fingerprints or {}
        if fingerprints.get("audio_extraction") != audio_component:
            return None
        fingerprint = fingerprints.get("hosted_transcription")
        if not isinstance(fingerprint, str) or not fingerprint:
            return None
        for item in prior.transcript_evidence or []:
            if not isinstance(item, dict) or item.get("kind") != EvidenceKind.HOSTED_ASR.value:
                continue
            if item.get("state") != EvidenceState.ACCEPTED.value:
                continue
            ctx.bump("component_reuse")
            return EvidenceRecord(
                kind=EvidenceKind.HOSTED_ASR,
                fingerprint=fingerprint,
                provider=str(item.get("provider", "gemini")),
                model=item.get("model") if isinstance(item.get("model"), str) else None,
                settings={},
                window_start=float(item.get("window_start", prior.context_start)),
                window_end=float(item.get("window_end", prior.context_end)),
                transcript=str(item.get("transcript", "")),
                confidence=float(item.get("confidence", 0.0)),
                state=EvidenceState.ACCEPTED,
            )
        return None

    def _wants_hosted(
        self,
        priority: RefinementPriority,
        local_record: EvidenceRecord | None,
        ctx: RefinementContext,
    ) -> bool:
        if self._routing_mode == "gemini_only":
            return True
        if priority is RefinementPriority.FINAL_CLIP:
            return True
        # Candidate mode: only material uncertainty triggers hosted work.
        if local_record is None or local_record.state is not EvidenceState.ACCEPTED:
            return True
        uncertainty = self._uncertainty_score(local_record, ctx)
        ctx.routing_evidence["local_uncertainty"] = round(uncertainty, 6)
        return uncertainty >= self._config.hosted_min_uncertainty

    def _uncertainty_score(self, local_record: EvidenceRecord, ctx: RefinementContext) -> float:
        # Low word confidence, short/low-coverage transcript, and code-switch
        # suspicion raise uncertainty. Bounded to [0, 1].
        confidence = max(0.0, min(1.0, local_record.confidence))
        token_count = len(local_record.transcript.split())
        coverage = min(1.0, token_count / 8.0) if token_count else 0.0
        code_switch = 0.25 if local_record.code_switch_tokens else 0.0
        return min(1.0, max(0.0, (1.0 - confidence) * 0.6 + (1.0 - coverage) * 0.3 + code_switch))

    def _admitted(self, priority: RefinementPriority, ctx: RefinementContext) -> bool:
        if self._admission is None:
            return True
        try:
            decision = self._admission.acquire(admission_priority_for(priority))
        except Exception:
            ctx.bump("admission_denials")
            return False
        if not getattr(decision, "admitted", False):
            ctx.bump("admission_denials")
            ctx.routing_evidence["admission_reason"] = str(getattr(decision, "reason", "denied"))
            return False
        return True

    def _handle_hosted_error(self, error: HostedProviderError, ctx: RefinementContext) -> None:
        ctx.bump("provider_failures")
        category = getattr(error, "category", None)
        ctx.routing_evidence["hosted_error"] = getattr(category, "value", str(category))
        if category is not None and getattr(category, "value", "") == "RATE_LIMITED":
            ctx.bump("provider_rate_limits")
            recorder = getattr(self._admission, "record_rate_limit", None)
            if callable(recorder):
                try:
                    recorder()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # adjudication

    def _adjudicate(
        self,
        *,
        audio_path: Path,
        priority: RefinementPriority,
        region: _Region,
        audio_records: Sequence[EvidenceRecord],
        source_dialect: str | None,
        base_text: str,
        ctx: RefinementContext,
    ) -> tuple[str | None, tuple[UnresolvedSpan, ...], set[str]]:
        if self._routing_mode == "local_only" or self._adjudicator is None:
            return None, (), set()
        conflicts = self._adjudication_requests(region, audio_records, source_dialect)
        if not conflicts:
            return None, (), set()
        if not self._admitted(priority, ctx):
            return None, (), set()
        self._check_cancelled()
        try:
            results = self._adjudicator.adjudicate(conflicts, audio_path=audio_path)
        except HostedProviderError as error:
            self._handle_hosted_error(error, ctx)
            return None, (), set()
        except Exception:
            ctx.bump("provider_failures")
            return None, (), set()
        ctx.bump("adjudication_calls")
        # Adjudication never replaces a whole transcript with a bare entity:
        # entity readings are applied as bounded substitutions into the
        # audio-backed base transcript. Only an explicit whole-transcript
        # disagreement may replace the base.
        current = base_text
        changed = False
        spans: list[UnresolvedSpan] = []
        handled: set[str] = set()
        for request in conflicts:
            handled.add(request.ambiguity_id)
            result = results.get(request.ambiguity_id)
            if (
                result is None
                or result.selected_reading is None
                or result.confidence < self._config.adjudication_min_confidence
            ):
                spans.append(
                    UnresolvedSpan(
                        span_id=request.ambiguity_id,
                        start=None,
                        end=None,
                        context=request.context,
                        readings=request.candidate_readings,
                        evidence_fingerprints=tuple(record.fingerprint for record in audio_records),
                        providers=("gemini",),
                        confidence=float(getattr(result, "confidence", 0.0) or 0.0),
                        reason=str(getattr(result, "rejected_reason", None) or "unresolved"),
                        entity_type=None,
                        meaning_critical=request.meaning_critical,
                    )
                )
                continue
            selected = result.selected_reading
            if request.ambiguity_id == "asr-disagreement":
                # Explicit whole-transcript disagreement may replace the base.
                if selected.strip() and selected.strip() != current:
                    current = selected
                    changed = True
                continue
            updated = _apply_entity_reading(current, request.candidate_readings, selected)
            if updated is None:
                spans.append(
                    UnresolvedSpan(
                        span_id=request.ambiguity_id,
                        start=None,
                        end=None,
                        context=request.context,
                        readings=request.candidate_readings,
                        evidence_fingerprints=tuple(record.fingerprint for record in audio_records),
                        providers=("gemini",),
                        confidence=float(getattr(result, "confidence", 0.0) or 0.0),
                        reason="adjudication_not_applicable",
                        entity_type=None,
                        meaning_critical=request.meaning_critical,
                    )
                )
                continue
            if updated != current:
                current = updated
                changed = True
        return (current if changed else None), tuple(spans), handled

    def _adjudication_requests(
        self,
        region: _Region,
        audio_records: Sequence[EvidenceRecord],
        source_dialect: str | None,
    ) -> list[AdjudicationRequest]:
        requests: list[AdjudicationRequest] = []
        # Material ASR disagreement on the same span.
        if len(audio_records) >= 2:
            texts = tuple(dict.fromkeys(record.transcript for record in audio_records))
            if len(texts) >= 2:
                requests.append(
                    AdjudicationRequest(
                        ambiguity_id="asr-disagreement",
                        context=region.index_text[: self._config.max_context_characters],
                        candidate_readings=texts[: self._config.max_candidate_readings],
                        evidence_summary=tuple(
                            {"provider": record.provider, "text": record.transcript}
                            for record in audio_records
                        ),
                        dialect_profile=source_dialect,
                        meaning_critical=True,
                    )
                )
        left = extract_entities(region.index_text)
        right = extract_entities(" ".join(record.transcript for record in audio_records))
        for index, conflict in enumerate(compare_entity_sets(left, right)):
            readings = tuple(
                dict.fromkeys(
                    value
                    for value in (str(conflict.get("left", "")), str(conflict.get("right", "")))
                    if value
                )
            )
            if len(readings) < 2:
                continue
            requests.append(
                AdjudicationRequest(
                    ambiguity_id=f"entity-{index}",
                    context=region.index_text[: self._config.max_context_characters],
                    candidate_readings=readings[: self._config.max_candidate_readings],
                    evidence_summary=tuple(
                        {"provider": record.provider, "text": record.transcript}
                        for record in audio_records
                    ),
                    dialect_profile=source_dialect,
                    meaning_critical=bool(conflict.get("meaning_critical", True)),
                )
            )
        return requests[: self._config.max_unresolved_spans]

    # ------------------------------------------------------------------
    # entities, status, confidence

    def _entities_and_ambiguity(
        self,
        *,
        final_text: str,
        region: _Region,
        audio_records: Sequence[EvidenceRecord],
        priority: RefinementPriority,
        adjudication_spans: tuple[UnresolvedSpan, ...],
        handled_ambiguities: set[str],
        provider_degraded: bool,
    ) -> tuple[tuple[EntityMention, ...], tuple[UnresolvedSpan, ...]]:
        entities = extract_entities(final_text)
        index_entities = extract_entities(region.index_text)
        conflicts = compare_entity_sets(index_entities, entities)
        spans: list[UnresolvedSpan] = list(adjudication_spans)
        handled = set(handled_ambiguities)
        for index, conflict in enumerate(conflicts):
            span_id = f"entity-{index}"
            if span_id in handled:
                continue
            readings = tuple(
                dict.fromkeys(
                    value
                    for value in (str(conflict.get("left", "")), str(conflict.get("right", "")))
                    if value
                )
            )
            meaning_critical = bool(conflict.get("meaning_critical", True))
            spans.append(
                UnresolvedSpan(
                    span_id=span_id,
                    start=None,
                    end=None,
                    context=region.index_text[: self._config.max_context_characters],
                    readings=readings,
                    evidence_fingerprints=tuple(record.fingerprint for record in audio_records),
                    providers=tuple(record.provider for record in audio_records),
                    confidence=0.0,
                    reason="entity_conflict",
                    entity_type=str(conflict.get("entity_type", "")) or None,
                    meaning_critical=meaning_critical,
                )
            )
        if provider_degraded and priority is RefinementPriority.FINAL_CLIP:
            spans.append(
                UnresolvedSpan(
                    span_id="provider-degraded",
                    start=None,
                    end=None,
                    context="",
                    readings=(),
                    evidence_fingerprints=(),
                    providers=("gemini",),
                    confidence=0.0,
                    reason="provider_degraded",
                    entity_type=None,
                    meaning_critical=False,
                )
            )
        return tuple(entities), tuple(spans)

    def _status(
        self,
        *,
        priority: RefinementPriority,
        local_record: EvidenceRecord | None,
        audio_records: Sequence[EvidenceRecord],
        final_text: str,
        manual: str | None,
        meaning_critical: bool,
        provider_degraded: bool,
        consensus_confidence: float,
        timing_aligned: bool,
        scope_refused: bool = False,
    ) -> str:
        local_accepted = local_record is not None and local_record.state is EvidenceState.ACCEPTED
        if priority is RefinementPriority.FINAL_CLIP and scope_refused:
            # A scoped timing refusal is structural evidence about the audio, not
            # an ambiguity that operator wording can resolve. Aligned manual text
            # cannot become final-ready while the selected view is rejected.
            return "NEEDS_MANUAL_TRANSCRIPT_REVIEW"
        if manual and manual.strip() and timing_aligned:
            return (
                "FINAL_TRANSCRIPT_READY"
                if priority is RefinementPriority.FINAL_CLIP
                else "CANDIDATE_REFINED"
            )
        if priority is RefinementPriority.FINAL_CLIP:
            if meaning_critical:
                return "NEEDS_MANUAL_TRANSCRIPT_REVIEW"
            consensus_ok = (
                bool(final_text.strip())
                and timing_aligned
                and consensus_confidence >= self._config.final_min_transcript_confidence
            )
            hosted_present = any(record.kind is EvidenceKind.HOSTED_ASR for record in audio_records)
            strict = consensus_ok and (hosted_present or local_accepted)
            if strict and not provider_degraded:
                return "FINAL_TRANSCRIPT_READY"
            if provider_degraded and local_accepted:
                return "PROVIDER_DEGRADED"
            return "NEEDS_MANUAL_TRANSCRIPT_REVIEW"
        if local_accepted or audio_records:
            return "CANDIDATE_REFINED"
        if provider_degraded:
            return "PROVIDER_DEGRADED"
        return "REFINEMENT_FAILED"

    def _confidence(
        self,
        *,
        priority: RefinementPriority,
        local_record: EvidenceRecord | None,
        hosted_present: bool,
        consensus_confidence: float,
        unresolved: Sequence[UnresolvedSpan],
    ) -> float:
        base = consensus_confidence
        if base <= 0.0 and local_record is not None:
            base = local_record.confidence
        if hosted_present and local_record is not None:
            base = max(base, 0.6)
        penalty = 0.1 * sum(1 for span in unresolved if span.meaning_critical)
        penalty += 0.03 * sum(1 for span in unresolved if not span.meaning_critical)
        return max(0.0, min(1.0, base - penalty))

    # ------------------------------------------------------------------
    # helpers

    def _validate(
        self,
        *,
        candidate_text: str,
        reference_text: str,
        words: Sequence[WordTimestamp],
        source_dialect: str | None,
        allow_unsupported_latin: bool,
    ) -> tuple[bool, str | None]:
        return validate_transcript_candidate(
            candidate_text=candidate_text,
            reference_text=reference_text,
            word_timestamps=words,
            protected_tokens=extract_protected_tokens(reference_text),
            source_dialect=source_dialect,
            candidate_dialect=None,
            max_edit_ratio=self._config.max_edit_ratio,
            min_phonetic_similarity=self._config.min_phonetic_similarity,
            repeats_max_ratio=self._config.repetitions_max_ratio,
            allow_unsupported_latin=allow_unsupported_latin,
        )

    def _context_terms(self, region: _Region, index_text: str) -> tuple[str, ...]:
        return source_evidenced_context_terms(index_text)

    def _choose_words(self, local_record: EvidenceRecord | None) -> tuple[WordTimestamp, ...]:
        if local_record is not None and local_record.word_timestamps:
            return local_record.word_timestamps
        return ()

    def _omitted_english(self, index_text: str, automatic: str) -> tuple[str, ...]:
        if not automatic.strip():
            return ()
        reference = {token.casefold() for token in extract_protected_tokens(index_text)}
        recovered = [
            token
            for token in extract_protected_tokens(automatic)
            if any(character.isascii() and character.isalpha() for character in token)
            and token.casefold() not in reference
        ]
        return tuple(dict.fromkeys(recovered))

    def _timing_alignment(
        self, manual: str | None, automatic: str, words: Sequence[WordTimestamp]
    ) -> tuple[bool, str | None]:
        return evaluate_timing_alignment(manual, automatic, words)

    def _seed_bounds(
        self, candidate: ClipCandidate, priority: RefinementPriority
    ) -> tuple[float, float]:
        coarse_start = float(candidate.start_time)
        coarse_end = float(candidate.end_time)
        if priority is RefinementPriority.FINAL_CLIP:
            prior = self._session.scalar(
                select(CandidateRefinement).where(
                    CandidateRefinement.clip_candidate_id == candidate.id,
                    CandidateRefinement.priority == RefinementPriority.CANDIDATE,
                )
            )
            if (
                prior is not None
                and prior.refined_start is not None
                and prior.refined_end is not None
                and prior.refined_end > prior.refined_start
                and prior.status
                in {
                    "CANDIDATE_REFINED",
                    "FINAL_TRANSCRIPT_READY",
                    "NEEDS_MANUAL_TRANSCRIPT_REVIEW",
                    "PROVIDER_DEGRADED",
                }
            ):
                return float(prior.refined_start), float(prior.refined_end)
        return coarse_start, coarse_end

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise RefinementCancelled("candidate refinement cancelled")

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Degraded:
    """Marker type returned when optional hosted work was skipped or failed."""


def source_evidenced_context_terms(index_text: str) -> tuple[str, ...]:
    """Bounded ASCII-letter hotwords already present in the source region."""

    terms = extract_protected_tokens(index_text)
    return tuple(
        term
        for term in terms
        if any(character.isascii() and character.isalpha() for character in term)
    )[:8]


def candidate_region_text(transcript: Transcript, start: float, end: float) -> str:
    """Small helper used by the handoff/API to expose the INDEX excerpt."""

    parts = [
        _segment_text(segment, "final_text")
        or _segment_text(segment, "contextual_reconstructed_text")
        or _segment_text(segment, "corrected_text")
        or _segment_text(segment, "raw_text")
        or _segment_text(segment, "text")
        for segment in transcript.segments
        if float(segment.get("end", 0.0)) > start and float(segment.get("start", 0.0)) < end
    ]
    return " ".join(part for part in parts if part).strip()
