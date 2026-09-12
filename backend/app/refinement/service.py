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
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
    REFINEMENT_POLICY_VERSION,
    REFINEMENT_SCHEMA_VERSION,
    REFINEMENT_VALIDATION_VERSION,
    Stage35Config,
)
from app.refinement.types import (
    AdjudicationRequest,
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

_MIN_TEXT = 1e-6


class RefinementError(RuntimeError):
    """A required Stage 3.5 input or invariant failed with no safe result."""


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
        adjudicated_text, adjudication_spans, handled_ambiguities = self._adjudicate(
            audio_path=audio_path,
            priority=priority,
            region=region,
            audio_records=audio_records,
            source_dialect=source_dialect,
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

        self._check_cancelled()
        consensus_text, consensus_confidence, _ = select_consensus_text(evidence)
        automatic = consensus_text or (local_record.transcript if local_record else "") or qwen_text
        manual = prior.manual_transcript if prior is not None else None
        operator_segment_text = region.operator_text
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

        entities, unresolved = self._entities_and_ambiguity(
            final_text=final_text,
            region=region,
            audio_records=audio_records,
            priority=priority,
            adjudication_spans=adjudication_spans,
            handled_ambiguities=handled_ambiguities,
            provider_degraded=provider_degraded,
        )

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

        omitted_english = self._omitted_english(index_text, automatic)
        if omitted_english:
            ctx.bump("code_switch_recoveries")
            ctx.routing_evidence["code_switch_recovered"] = list(omitted_english)

        meaning_critical = any(span.meaning_critical for span in unresolved)
        timing_aligned, timing_reason = self._timing_alignment(manual, automatic, word_timestamps)
        if not timing_aligned:
            unresolved = unresolved + (
                UnresolvedSpan(
                    span_id="timing-alignment",
                    start=boundary.start,
                    end=boundary.end,
                    context=manual or "",
                    readings=(automatic,) if automatic else (),
                    evidence_fingerprints=(),
                    providers=("operator",),
                    confidence=0.0,
                    reason=timing_reason or "timing_alignment_unresolved",
                    entity_type=None,
                    meaning_critical=priority is RefinementPriority.FINAL_CLIP,
                ),
            )

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
        output_fingerprint = refinement_output_fingerprint(
            {
                "input_fingerprint": input_fingerprint,
                "priority": priority.value,
                "status": status,
                "final_transcript": final_text,
                "refined_start": round(boundary.start, 6),
                "refined_end": round(boundary.end, 6),
                "unresolved": [span.span_id for span in unresolved],
            }
        )
        transcript_evidence = dedupe_evidence(evidence, limit=self._config.max_evidence_records)
        provider_evidence = {
            "local_asr_accepted": bool(
                local_record and local_record.state is EvidenceState.ACCEPTED
            ),
            "hosted_used": any(record.kind is EvidenceKind.HOSTED_ASR for record in audio_records),
            "adjudication_used": adjudicated_text is not None,
            "provider_degraded": provider_degraded,
            "routing_mode": self._routing_mode,
        }
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
        cancel_event = threading.Event()
        self._check_cancelled()
        try:
            result: TargetASRResult = self._asr.transcribe(
                audio_path,
                context_start=window.context_start,
                context_end=window.context_end,
                priority=priority,
                cancel_event=cancel_event,
                context_terms=self._context_terms(region, index_text),
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
        if not prior.automatic_transcript.strip():
            return None
        return EvidenceRecord(
            kind=EvidenceKind.TARGETED_LOCAL_ASR,
            fingerprint=fingerprint,
            provider="faster-whisper",
            model=None,
            settings={},
            window_start=prior.context_start,
            window_end=prior.context_end,
            transcript=prior.automatic_transcript,
            confidence=prior.confidence,
            state=EvidenceState.ACCEPTED,
            word_timestamps=tuple(
                WordTimestamp(
                    text=str(item.get("text", "")),
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    probability=item.get("probability"),
                )
                for item in prior.word_timestamps or []
                if isinstance(item, dict)
            ),
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
        selected: list[str] = []
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
            selected.append(result.selected_reading)
        return (" ".join(selected).strip() or None), tuple(spans), handled

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
    ) -> str:
        local_accepted = local_record is not None and local_record.state is EvidenceState.ACCEPTED
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
        terms = extract_protected_tokens(index_text)
        return tuple(
            term
            for term in terms
            if any(character.isascii() and character.isalpha() for character in term)
        )[:8]

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
        if not manual or not manual.strip():
            return True, None
        if not automatic.strip():
            return False, "manual_transcript_without_timing_evidence"
        if not words:
            return False, "manual_transcript_without_word_timestamps"
        from app.transcription.arabic import normalize_for_comparison

        manual_tokens = normalize_for_comparison(manual).split()
        automatic_tokens = normalize_for_comparison(automatic).split()
        if not automatic_tokens:
            return False, "manual_transcript_without_timing_evidence"
        overlap = len(set(manual_tokens) & set(automatic_tokens))
        ratio = overlap / max(1, len(set(manual_tokens)))
        if ratio < 0.3:
            return False, "manual_transcript_cannot_be_aligned_safely"
        return True, None

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
