"""Candidate-scoped Stage 3.5 refinement service/executor tests."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.attributes import flag_modified

from app.composition.captions import build_caption_plan
from app.composition.policy import CaptionStyle, Stage51Config
from app.core.enums import (
    CandidateDisposition,
    EvidenceKind,
    EvidenceState,
    JobKind,
    JobStatus,
    MediaOriginType,
    RefinementPriority,
    RefinementStatus,
    RightsStatus,
)
from app.db.base import Base
from app.models import (
    AudioAnalysis,
    AudioArtifact,
    CandidateRefinement,
    ClipCandidate,
    ProcessingJob,
    SourceVideo,
    Transcript,
)
from app.refinement.asr import TargetedASREngine
from app.refinement.audio_window import RefinementAudioWindow
from app.refinement.executor import (
    CandidateRefinementCancelled,
    CandidateRefinementExecutor,
    build_candidate_refinement_executor,
)
from app.refinement.hosted import (
    HostedErrorCategory,
    HostedProviderError,
    HostedTranscriptionResult,
)
from app.refinement.policy import DEFAULT_CONFIG, FINAL_CLIP_SCOPE_POLICY_VERSION
from app.refinement.queue import apply_manual_transcript
from app.refinement.scope import raw_observation_fingerprint
from app.refinement.service import CandidateRefinementService
from app.refinement.types import (
    AdjudicationResult,
    RefinementCancelled,
    TargetASRResult,
    WordTimestamp,
)
from app.render.service import _caption_display, _caption_source_payload
from app.services.storage import StorageCategory, StorageService
from app.transcription.dialect import ArabicDialectProfile
from app.transcription.engine import TranscriptionResult
from app.transcription.service import TranscriptionOptions
from app.transformation.inputs import (
    TransformationInputError,
    build_transformation_inputs,
    resolve_effective_refinement,
)
from app.transformation.planning.inputs import build_planning_inputs
from app.transformation.planning.policy import DEFAULT_CONFIG as PLANNING_CONFIG
from app.transformation.policy import Stage40Config

_INDEX_TEXT = "أنا عملت امبارح"
_RECOVERED_TEXT = "أنا عملت deploy للbackend امبارح"


def _wav_bytes(*, seconds: float = 1.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


@dataclass
class _FakeAudio:
    storage: StorageService

    def context_bounds(
        self,
        *,
        coarse_start: float,
        coarse_end: float,
        source_duration: float,
        priority: RefinementPriority,
    ) -> tuple[float, float]:
        pre, post = (8.0, 8.0) if priority is RefinementPriority.FINAL_CLIP else (5.0, 5.0)
        return max(0.0, coarse_start - pre), min(source_duration, coarse_end + post)

    def audio_input_fingerprint(
        self,
        *,
        source: SourceVideo,
        artifact: AudioArtifact,
        context_start: float,
        context_end: float,
        priority: RefinementPriority,
    ) -> str:
        return (
            f"audio:{priority.value}:{context_start:.3f}:{context_end:.3f}:{artifact.content_hash}"
        )

    def extract(
        self,
        *,
        source: SourceVideo,
        candidate: ClipCandidate,
        priority: RefinementPriority,
        force: bool = False,
    ) -> RefinementAudioWindow:
        relative = f"{source.id}/candidate-refinements/{candidate.id}/{priority.value}.wav"
        path = self.storage.resolve(StorageCategory.SOURCES, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_wav_bytes())
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        start, end = self.context_bounds(
            coarse_start=candidate.start_time,
            coarse_end=candidate.end_time,
            source_duration=60.0,
            priority=priority,
        )
        return RefinementAudioWindow(
            source_id=str(source.id),
            candidate_id=str(candidate.id),
            priority=priority,
            coarse_start=candidate.start_time,
            coarse_end=candidate.end_time,
            context_start=start,
            context_end=end,
            relative_path=relative,
            content_hash=content_hash,
            duration=1.0,
            input_fingerprint=self.audio_input_fingerprint(
                source=source,
                artifact=_artifact_stub(),
                context_start=start,
                context_end=end,
                priority=priority,
            ),
        )


def _artifact_stub() -> AudioArtifact:
    return AudioArtifact(content_hash="audiohash", source_content_hash="src-hash")


class _FakeASR:
    def __init__(
        self, transcript: str, words: tuple[WordTimestamp, ...], *, confidence: float = 0.95
    ) -> None:
        self._transcript = transcript
        self._words = words
        self._confidence = confidence
        self.calls = 0
        self.context_terms: list[tuple[str, ...]] = []

    def transcribe(
        self,
        audio_path: Path,
        *,
        context_start: float,
        context_end: float,
        priority: RefinementPriority,
        cancel_event=None,
        context_terms=(),
    ) -> TargetASRResult:
        self.calls += 1
        self.context_terms.append(tuple(context_terms))
        return TargetASRResult(
            provider="faster-whisper",
            model="fake",
            language="ar",
            language_probability=0.99,
            transcript=self._transcript,
            word_timestamps=self._words,
            confidence=self._confidence,
            runtime_identity={"model": "fake"},
            fingerprint=f"local-{self._transcript}",
        )


class _RejectedASR(_FakeASR):
    def transcribe(
        self,
        audio_path: Path,
        *,
        context_start: float,
        context_end: float,
        priority: RefinementPriority,
        cancel_event=None,
        context_terms=(),
    ) -> TargetASRResult:
        self.calls += 1
        return TargetASRResult(
            provider="faster-whisper",
            model="fake",
            language="ar",
            language_probability=0.99,
            transcript=self._transcript,
            word_timestamps=self._words,
            confidence=self._confidence,
            runtime_identity={"model": "fake"},
            fingerprint=f"local-{self._transcript}",
            rejected_reason="nonpositive word duration",
            rejected_detail={
                "reason": "nonpositive word duration",
                "record_index": 3,
                "relative_start": 1.0,
                "relative_end": 1.0,
                "source_start": context_start + 1.0,
                "source_end": context_start + 1.0,
                "context_start": context_start,
                "context_end": context_end,
            },
        )


class _FakeHosted:
    provider_name = "gemini"
    model = "gemini-3.5-transcribe"

    def __init__(
        self,
        transcript: str,
        *,
        words: tuple[WordTimestamp, ...] = (),
        error: Exception | None = None,
        available: bool = True,
    ) -> None:
        self._transcript = transcript
        self._words = words
        self._error = error
        self._available = available
        self.calls = 0
        self.released = False

    def available(self) -> bool:
        return self._available

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "gemini", "model": self.model}

    def transcribe(
        self, audio_path: Path, *, language_codes=None, dialect_profile=None, custom_vocabulary=()
    ) -> HostedTranscriptionResult:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return HostedTranscriptionResult(
            transcript=self._transcript,
            language="ar",
            language_probability=0.9,
            word_timestamps=self._words,
            confidence=0.9,
        )

    def release(self) -> None:
        self.released = True


class _FakeAdjudicator:
    model = "gemini-3.8-flash"

    def __init__(self, selections: dict[str, str | None] | None = None) -> None:
        self._selections = selections or {}
        self.calls = 0

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "gemini", "model": self.model}

    def adjudicate(self, requests, *, audio_path=None):
        self.calls += 1
        results: dict[str, AdjudicationResult] = {}
        for request in requests:
            selected = self._selections.get(request.ambiguity_id)
            results[request.ambiguity_id] = AdjudicationResult(
                ambiguity_id=request.ambiguity_id,
                selected_reading=selected,
                confidence=0.9 if selected else 0.0,
                reason="test",
            )
        return results


class _FakeAdmission:
    def __init__(self, *, admitted: bool = True) -> None:
        self._admitted = admitted
        self.acquire_calls = 0
        self.rate_limits = 0

    def acquire(self, priority):
        self.acquire_calls += 1
        return type("Decision", (), {"admitted": self._admitted, "reason": "TEST"})()

    def record_rate_limit(self, retry_after=None) -> None:
        self.rate_limits += 1

    def runtime_identity(self) -> dict[str, object]:
        return {"admission_policy_version": "test"}


class _FakeQwen:
    model = "qwen3.5:4b"

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "ollama", "model": self.model}

    def reconstruct(self, segments, **kwargs):
        self.calls += 1
        return type("Result", (), {"contextual_reconstructed_text": self._text})()


@pytest.fixture
def session(sqlite_engine) -> Session:
    Base.metadata.create_all(sqlite_engine)
    factory = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    return factory()


@pytest.fixture
def storage(tmp_path: Path) -> StorageService:
    root = tmp_path / "storage"
    root.mkdir(exist_ok=True)
    return StorageService(root)


def _segment(text: str, start: float, end: float, *, words: tuple[WordTimestamp, ...] = ()) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text,
        "raw_text": text,
        "corrected_text": text,
        "final_text": text,
        "words": [
            {
                "word": word.text,
                "start": word.start,
                "end": word.end,
                "probability": word.probability,
            }
            for word in words
        ],
    }


def _make_candidate(
    session: Session,
    *,
    index_text: str = _INDEX_TEXT,
    start: float = 10.0,
    end: float = 16.0,
    words: tuple[WordTimestamp, ...] | None = None,
) -> ClipCandidate:
    source = SourceVideo(
        source_uri="/tmp/source.mp4",
        original_filename="source.mp4",
        content_hash="src-hash",
        rights_status=RightsStatus.OWNED,
        media_origin=MediaOriginType.OTHER,
    )
    session.add(source)
    session.flush()
    session.add(
        Transcript(
            source_video_id=source.id,
            language="ar",
            whisper_model="large-v3-turbo",
            duration=60.0,
            raw_text=index_text,
            segments=[_segment(index_text, start, end, words=words or ())],
            dialect_profile=ArabicDialectProfile.EGYPTIAN,
            dialect_confidence=0.9,
            raw_transcript_confidence=0.6,
            correction_confidence=0.6,
            reconstruction_confidence=0.5,
            transcription_revision=1,
            input_fingerprint="tf-1",
            correction_version="v1",
        )
    )
    session.add(
        AudioAnalysis(
            source_video_id=source.id,
            audio_hash="ah",
            input_fingerprint="afp",
            silence_intervals=[{"start": 9.0, "end": 9.6, "duration": 0.6}],
            features=[],
            silence_ratio=0.1,
            speech_density=0.9,
            speech_rate=2.0,
        )
    )
    session.add(
        AudioArtifact(
            source_video_id=source.id,
            output_path=f"{source.id}/speech-analysis.wav",
            content_hash="audiohash",
            source_content_hash="src-hash",
            sample_rate=16000,
            duration=60.0,
        )
    )
    candidate = ClipCandidate(
        source_video_id=source.id,
        candidate_key="ck-1",
        is_current=True,
        disposition=CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
        start_time=start,
        end_time=end,
        start_segment_index=0,
        end_segment_index=0,
        segment_indexes=[0],
        transcript_excerpt=index_text,
        clip_score=0.8,
        boundary_confidence=0.3,
        refinement_reasons=["CODE_SWITCH_UNCERTAINTY"],
        hooks=[],
        analysis_fingerprint="caf",
    )
    session.add(candidate)
    session.commit()
    session.refresh(candidate)
    return candidate


def _service(session: Session, storage: StorageService, **overrides) -> CandidateRefinementService:
    defaults = {
        "session": session,
        "storage": storage,
        "config": DEFAULT_CONFIG,
        "audio_service": _FakeAudio(storage),
        "asr_engine": None,
        "hosted_provider": None,
        "adjudication_provider": None,
        "admission": None,
        "routing_mode": "adaptive",
        "local_identity": {"provider": "faster-whisper", "model": "fake"},
    }
    defaults.update(overrides)
    return CandidateRefinementService(**defaults)


def _source_time_words() -> tuple[WordTimestamp, ...]:
    return (
        WordTimestamp("أنا", 10.4, 10.7, 0.9),
        WordTimestamp("عملت", 10.7, 11.0, 0.9),
        WordTimestamp("deploy", 11.0, 11.5, 0.9),
        WordTimestamp("للbackend", 11.5, 12.1, 0.9),
        WordTimestamp("امبارح", 12.1, 12.6, 0.9),
    )


def test_local_only_recovers_omitted_english(session: Session, storage: StorageService) -> None:
    candidate = _make_candidate(session)
    hosted = _FakeHosted(_RECOVERED_TEXT)
    outcome = _service(
        session,
        storage,
        asr_engine=_FakeASR(_RECOVERED_TEXT, _source_time_words()),
        hosted_provider=hosted,
        admission=_FakeAdmission(),
    ).execute(candidate, priority=RefinementPriority.CANDIDATE)

    assert outcome.status == RefinementStatus.CANDIDATE_REFINED.value
    assert "deploy" in outcome.final_transcript
    assert "backend" in outcome.final_transcript
    assert set(outcome.code_switch_evidence["recovered"]) >= {"deploy", "backend"}
    assert outcome.metrics.get("code_switch_recoveries") == 1
    assert hosted.calls == 0  # clean, high-confidence local result needs no hosted call


def test_text_only_qwen_cannot_invent_english(session: Session, storage: StorageService) -> None:
    candidate = _make_candidate(session)
    hosted = _FakeHosted(_RECOVERED_TEXT)
    qwen = _FakeQwen(_RECOVERED_TEXT)
    outcome = _service(
        session,
        storage,
        asr_engine=_FakeASR(_INDEX_TEXT, _source_time_words()[:2]),
        hosted_provider=hosted,
        routing_mode="local_only",
        local_qwen_enabled=True,
        qwen_reconstructor=qwen,
    ).execute(candidate, priority=RefinementPriority.CANDIDATE)

    assert "deploy" not in outcome.final_transcript
    assert qwen.calls == 1
    assert hosted.calls == 0
    rejected = [
        record
        for record in outcome.transcript_evidence
        if record.kind is EvidenceKind.STAGE27 and record.state is EvidenceState.REJECTED
    ]
    assert rejected and rejected[0].reason == "unsupported_omitted_english"


def test_adaptive_never_uses_qwen(session: Session, storage: StorageService) -> None:
    candidate = _make_candidate(session)
    qwen = _FakeQwen(_RECOVERED_TEXT)
    _service(
        session,
        storage,
        asr_engine=_FakeASR(_INDEX_TEXT, _source_time_words()[:2]),
        routing_mode="adaptive",
        local_qwen_enabled=True,
        qwen_reconstructor=qwen,
    ).execute(candidate, priority=RefinementPriority.CANDIDATE)
    assert qwen.calls == 0


def test_final_blocks_on_meaning_critical_number_conflict(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session, index_text="فيه 71 شخص", start=10.0, end=16.0)
    adjudicator = _FakeAdjudicator({})  # leaves ambiguity unresolved
    outcome = _service(
        session,
        storage,
        asr_engine=_FakeASR("فيه 70 شخص", (WordTimestamp("70", 12.0, 12.4, 0.9),)),
        hosted_provider=_FakeHosted("فيه 70 شخص", words=(WordTimestamp("70", 2.0, 2.4, 0.9),)),
        adjudication_provider=adjudicator,
        admission=_FakeAdmission(),
    ).execute(candidate, priority=RefinementPriority.FINAL_CLIP)

    assert outcome.status == RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW.value
    assert any(span.meaning_critical for span in outcome.unresolved_spans)


def test_candidate_retains_flag_while_final_blocks(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session, index_text="فيه 71 شخص", start=10.0, end=16.0)
    common = dict(
        asr_engine=_FakeASR("فيه 70 شخص", (WordTimestamp("70", 12.0, 12.4, 0.9),)),
        hosted_provider=_FakeHosted("فيه 70 شخص", words=(WordTimestamp("70", 2.0, 2.4, 0.9),)),
        admission=_FakeAdmission(),
    )
    candidate_outcome = _service(session, storage, **common).execute(
        candidate, priority=RefinementPriority.CANDIDATE
    )
    assert candidate_outcome.status == RefinementStatus.CANDIDATE_REFINED.value
    assert candidate_outcome.unresolved_spans

    final_outcome = _service(session, storage, **common).execute(
        candidate, priority=RefinementPriority.FINAL_CLIP
    )
    assert final_outcome.status == RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW.value


def test_provider_degradation_keeps_safe_local_result(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session)
    admission = _FakeAdmission()
    hosted = _FakeHosted("", error=HostedProviderError(HostedErrorCategory.RATE_LIMITED))
    outcome = _service(
        session,
        storage,
        asr_engine=_FakeASR(_INDEX_TEXT, _source_time_words()[:2]),
        hosted_provider=hosted,
        admission=admission,
        routing_mode="gemini_only",
    ).execute(candidate, priority=RefinementPriority.CANDIDATE)
    assert outcome.status in {
        RefinementStatus.CANDIDATE_REFINED.value,
        RefinementStatus.PROVIDER_DEGRADED.value,
    }
    assert outcome.cache_eligible is False
    assert admission.rate_limits == 1


def test_no_hosted_provider_works_locally(session: Session, storage: StorageService) -> None:
    candidate = _make_candidate(session)
    outcome = _service(
        session,
        storage,
        asr_engine=_FakeASR(_INDEX_TEXT, _source_time_words()[:2]),
    ).execute(candidate, priority=RefinementPriority.CANDIDATE)
    assert outcome.status == RefinementStatus.CANDIDATE_REFINED.value


def test_boundaries_stay_within_context_and_not_full_window(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session, start=10.0, end=16.0)
    outcome = _service(
        session,
        storage,
        asr_engine=_FakeASR(_INDEX_TEXT, _source_time_words()),
    ).execute(candidate, priority=RefinementPriority.CANDIDATE)
    assert (
        outcome.context_start <= outcome.refined_start < outcome.refined_end <= outcome.context_end
    )
    assert outcome.refined_start >= outcome.context_start
    assert outcome.refined_end <= outcome.context_end
    assert (outcome.refined_start, outcome.refined_end) != (
        outcome.context_start,
        outcome.context_end,
    )


def test_manual_transcript_always_wins(session: Session, storage: StorageService) -> None:
    candidate = _make_candidate(session)
    prior = CandidateRefinement(
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        priority=RefinementPriority.CANDIDATE,
        status=RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW,
        coarse_start=10.0,
        coarse_end=16.0,
        context_start=5.0,
        context_end=21.0,
        manual_transcript="نص يدوي معتمد",
    )
    session.add(prior)
    session.commit()
    outcome = _service(
        session,
        storage,
        asr_engine=_FakeASR(_INDEX_TEXT, _source_time_words()[:2]),
    ).execute(candidate, priority=RefinementPriority.CANDIDATE, prior=prior)
    assert outcome.final_transcript == "نص يدوي معتمد"


def test_candidate_and_final_fingerprints_differ(session: Session, storage: StorageService) -> None:
    candidate = _make_candidate(session)
    service = _service(session, storage, asr_engine=_FakeASR(_INDEX_TEXT, ()))
    candidate_fp = service.input_fingerprint(candidate, RefinementPriority.CANDIDATE)
    final_fp = service.input_fingerprint(candidate, RefinementPriority.FINAL_CLIP)
    assert candidate_fp != final_fp


def test_cancellation_before_work_makes_no_extension(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session)
    audio = _FakeAudio(storage)
    service = _service(
        session,
        storage,
        asr_engine=_FakeASR(_INDEX_TEXT, ()),
        audio_service=audio,
        is_cancelled=lambda: True,
    )
    with pytest.raises(RefinementCancelled):
        service.execute(candidate, priority=RefinementPriority.CANDIDATE)


# ----------------------------------------------------------------------
# executor


def _executor(
    session: Session, storage: StorageService, **overrides
) -> CandidateRefinementExecutor:
    defaults = dict(
        session=session,
        storage=storage,
        config=DEFAULT_CONFIG,
        audio_service=_FakeAudio(storage),
        asr_engine=_FakeASR(_RECOVERED_TEXT, _source_time_words()),
        local_identity={"provider": "faster-whisper", "model": "fake"},
    )
    defaults.update(overrides)
    return CandidateRefinementExecutor(**defaults)


def _refinement_row(
    session: Session, candidate: ClipCandidate, priority=RefinementPriority.CANDIDATE
) -> CandidateRefinement:
    row = CandidateRefinement(
        source_video_id=candidate.source_video_id,
        clip_candidate_id=candidate.id,
        priority=priority,
        coarse_start=candidate.start_time,
        coarse_end=candidate.end_time,
        context_start=candidate.start_time - 5,
        context_end=candidate.end_time + 5,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def test_executor_cache_hit_skips_asr(session: Session, storage: StorageService) -> None:
    candidate = _make_candidate(session)
    row = _refinement_row(session, candidate)
    executor = _executor(session, storage)
    executor.execute(row.id)
    session.refresh(row)
    assert row.status is RefinementStatus.CANDIDATE_REFINED
    assert row.cache_eligible is True

    counting_asr = _FakeASR(_RECOVERED_TEXT, _source_time_words())
    second = _executor(session, storage, asr_engine=counting_asr)
    second.execute(row.id)
    assert counting_asr.calls == 0


def test_executor_cancellation_never_marks_ready(session: Session, storage: StorageService) -> None:
    candidate = _make_candidate(session)
    row = _refinement_row(session, candidate)
    job = ProcessingJob(
        source_video_id=candidate.source_video_id,
        kind=JobKind.CANDIDATE_REFINEMENT,
        status=JobStatus.CANCELLED,
        candidate_refinement_id=row.id,
    )
    session.add(job)
    session.commit()
    executor = _executor(session, storage)
    executor.set_active_job(job.id)
    with pytest.raises(CandidateRefinementCancelled):
        executor.execute(row.id)
    session.refresh(row)
    assert row.status is RefinementStatus.CANCELLED


def test_executor_reuses_accepted_hosted_evidence_on_outage(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session)
    row = _refinement_row(session, candidate)
    # Seed a valid component checkpoint that matches the deterministic audio id.
    audio_component = None
    executor = _executor(session, storage)
    executor.execute(row.id)
    session.refresh(row)
    audio_component = (row.component_fingerprints or {}).get("audio_extraction")
    assert audio_component

    # Pretend hosted evidence was accepted on the prior run.
    row.component_fingerprints = {
        **dict(row.component_fingerprints or {}),
        "hosted_transcription": "hosted-fp",
    }
    row.transcript_evidence = [
        {
            "kind": "HOSTED_ASR",
            "state": "ACCEPTED",
            "provider": "gemini",
            "transcript": _RECOVERED_TEXT,
            "confidence": 0.9,
            "window_start": row.context_start,
            "window_end": row.context_end,
        }
    ]
    row.cache_eligible = False
    session.commit()

    failing = _FakeHosted("", error=HostedProviderError(HostedErrorCategory.PROVIDER_ERROR))
    second = _executor(session, storage, hosted_provider=failing, routing_mode="gemini_only")
    second.execute(row.id)
    assert failing.calls == 0


# ----------------------------------------------------------------------
def _local_checkpoint(
    *,
    fingerprint: str = "local-fp",
    state: str = "ACCEPTED",
    transcript: str = "record local text",
    provider: object = "faster-whisper",
    model: object = "large-v3-turbo",
    settings: object | None = None,
    window_start: float = 2.0,
    window_end: float = 24.0,
    confidence: object = 0.77,
    word_timestamps: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "kind": "TARGETED_LOCAL_ASR",
        "state": state,
        "reason": None if state == "ACCEPTED" else "word timestamp outside requested window",
        "fingerprint": fingerprint,
        "provider": provider,
        "model": model,
        "settings": {"beam_size": 8} if settings is None else settings,
        "window_start": window_start,
        "window_end": window_end,
        "transcript": transcript,
        "confidence": confidence,
        "word_timestamps": word_timestamps
        if word_timestamps is not None
        else [{"text": "record", "start": 10.4, "end": 10.8, "probability": 0.9}],
    }


def _local_prior(
    *,
    records: list[dict[str, object]],
    audio_component: str = "audio-fp",
    automatic_transcript: str = "aggregate text",
    confidence: float = 0.2,
) -> SimpleNamespace:
    return SimpleNamespace(
        quality_level=RefinementPriority.FINAL_CLIP.value,
        component_fingerprints={
            "audio_extraction": audio_component,
            "local_asr": "local-fp",
        },
        context_start=2.0,
        context_end=24.0,
        automatic_transcript=automatic_transcript,
        confidence=confidence,
        word_timestamps=[{"text": "aggregate", "start": 10.5, "end": 10.9, "probability": 0.1}],
        transcript_evidence=records,
    )


def test_rejected_local_checkpoint_is_not_reused(session: Session, storage: StorageService) -> None:
    prior = _local_prior(records=[_local_checkpoint(state="REJECTED")])
    service = _service(session, storage)

    reused = service._reuse_local(
        prior,
        prior.component_fingerprints["audio_extraction"],
        RefinementPriority.FINAL_CLIP,
        False,
    )

    assert reused is None


def test_local_checkpoint_requires_matching_accepted_record(
    session: Session, storage: StorageService
) -> None:
    service = _service(session, storage)
    cases = {
        "missing": [],
        "wrong_kind": [
            {
                **_local_checkpoint(),
                "kind": "HOSTED_ASR",
            }
        ],
        "fingerprint_mismatch": [_local_checkpoint(fingerprint="different-local-fingerprint")],
        "window_mismatch": [_local_checkpoint(window_start=1.0, window_end=24.0)],
        "empty_transcript": [_local_checkpoint(transcript="   ")],
        "malformed_words": [
            _local_checkpoint(
                word_timestamps=[{"text": "bad", "start": "not-a-number", "end": 1.0}]
            )
        ],
        "invalid_confidence": [_local_checkpoint(confidence="high")],
    }

    for name, records in cases.items():
        prior = _local_prior(records=records)
        reused = service._reuse_local(
            prior,
            prior.component_fingerprints["audio_extraction"],
            RefinementPriority.FINAL_CLIP,
            False,
        )
        assert reused is None, name


def test_local_checkpoint_reuses_its_own_evidence(
    session: Session, storage: StorageService
) -> None:
    record = _local_checkpoint()
    prior = _local_prior(records=[record], automatic_transcript="aggregate different")
    service = _service(session, storage)

    reused = service._reuse_local(
        prior,
        prior.component_fingerprints["audio_extraction"],
        RefinementPriority.FINAL_CLIP,
        False,
    )

    assert reused is not None
    assert reused.state is EvidenceState.ACCEPTED
    assert reused.transcript == "record local text"
    assert reused.fingerprint == "local-fp"
    assert reused.provider == "faster-whisper"
    assert reused.model == "large-v3-turbo"
    assert reused.settings == {"beam_size": 8}
    assert reused.window_start == 2.0
    assert reused.window_end == 24.0
    assert reused.confidence == 0.77
    assert [word.text for word in reused.word_timestamps] == ["record"]


def test_executor_does_not_reuse_rejected_local_checkpoint(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session)
    row = _refinement_row(session, candidate)
    _executor(session, storage).execute(row.id)
    session.refresh(row)

    rejected = next(
        item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR"
    )
    rejected["state"] = "REJECTED"
    rejected["reason"] = "word timestamp outside requested window"
    row.transcript_evidence = list(row.transcript_evidence)
    flag_modified(row, "transcript_evidence")
    row.automatic_transcript = _RECOVERED_TEXT
    row.cache_eligible = False
    session.commit()

    counting = _FakeASR(_RECOVERED_TEXT, _source_time_words())
    _executor(session, storage, asr_engine=counting).execute(row.id)

    assert counting.calls == 1
    session.refresh(row)
    local = next(item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR")
    assert local["state"] == "ACCEPTED"
    assert local["fingerprint"] == f"local-{_RECOVERED_TEXT}"


def test_executor_reuses_accepted_local_record_without_provider(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session)
    row = _refinement_row(session, candidate)
    _executor(session, storage).execute(row.id)
    session.refresh(row)

    local = next(item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR")
    local_words = [dict(word) for word in local["word_timestamps"]]
    local["transcript"] = "record local transcript"
    local["confidence"] = 0.77
    local["model"] = "record-model"
    local["settings"] = {"beam_size": 8, "record": True}
    row.transcript_evidence = list(row.transcript_evidence)
    flag_modified(row, "transcript_evidence")
    row.automatic_transcript = "aggregate deliberately different"
    row.confidence = 0.2
    row.word_timestamps = [{"text": "aggregate", "start": 10.5, "end": 10.9, "probability": 0.1}]
    row.cache_eligible = False
    session.commit()

    counting = _FakeASR(_RECOVERED_TEXT, _source_time_words())
    _executor(session, storage, asr_engine=counting).execute(row.id)

    assert counting.calls == 0
    session.refresh(row)
    reused = next(item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR")
    assert reused["state"] == "ACCEPTED"
    assert reused["transcript"] == "record local transcript"
    assert reused["confidence"] == 0.77
    assert reused["model"] == "record-model"
    assert reused["settings"] == {"beam_size": 8, "record": True}
    assert reused["word_timestamps"] == local_words


def test_rejected_local_evidence_retains_bounded_timestamp_diagnostic(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session)
    outcome = _service(
        session,
        storage,
        asr_engine=_RejectedASR(_INDEX_TEXT, (WordTimestamp("bad", 10.0, 10.0, 0.5),)),
    ).execute(candidate, priority=RefinementPriority.CANDIDATE)

    local = next(
        record
        for record in outcome.transcript_evidence
        if record.kind is EvidenceKind.TARGETED_LOCAL_ASR
    )
    assert local.state is EvidenceState.REJECTED
    assert local.reason == "nonpositive word duration"
    diagnostic = local.entity_metadata["timestamp_diagnostic"]
    assert diagnostic["reason"] == "nonpositive word duration"
    assert diagnostic["record_index"] == 3
    assert diagnostic["relative_start"] == 1.0
    assert diagnostic["relative_end"] == 1.0
    assert diagnostic["context_start"] == outcome.context_start
    assert diagnostic["context_end"] == outcome.context_end


# Sol review regressions


def test_entity_adjudication_preserves_full_utterance(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session, index_text="فيه 71 شخص", start=10.0, end=16.0)
    outcome = _service(
        session,
        storage,
        asr_engine=_FakeASR("فيه 70 شخص", (WordTimestamp("70", 12.0, 12.4, 0.9),)),
        hosted_provider=_FakeHosted("فيه 70 شخص", words=(WordTimestamp("70", 2.0, 2.4, 0.9),)),
        adjudication_provider=_FakeAdjudicator({"entity-0": "71"}),
        admission=_FakeAdmission(),
    ).execute(candidate, priority=RefinementPriority.FINAL_CLIP)

    assert outcome.final_transcript.strip() != "71"
    assert "شخص" in outcome.final_transcript
    assert "71" in outcome.final_transcript
    assert len(outcome.final_transcript.split()) > 1


def test_whole_transcript_adjudication_may_replace(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session, index_text="نص أول", start=10.0, end=16.0)
    outcome = _service(
        session,
        storage,
        asr_engine=_FakeASR("نص أول", (WordTimestamp("نص", 10.5, 10.9, 0.9),)),
        hosted_provider=_FakeHosted("نص ثاني", words=(WordTimestamp("نص", 0.5, 0.9, 0.9),)),
        adjudication_provider=_FakeAdjudicator({"asr-disagreement": "نص ثاني"}),
        admission=_FakeAdmission(),
    ).execute(candidate, priority=RefinementPriority.FINAL_CLIP)

    assert outcome.final_transcript.strip() == "نص ثاني"


def test_executor_persists_extracted_audio_metadata(
    session: Session, storage: StorageService
) -> None:
    candidate = _make_candidate(session)
    row = _refinement_row(session, candidate)
    _executor(session, storage).execute(row.id)
    session.refresh(row)
    assert row.audio_relative_path
    assert row.audio_relative_path.endswith("/CANDIDATE.wav")
    assert row.audio_content_hash
    assert row.audio_input_fingerprint


# ----------------------------------------------------------------------
# FINAL_CLIP context-versus-selected-evidence scope correction


_SCOPE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "final_clip_scope"
_SCOPE_CONTEXT_START = 376.28
_SCOPE_CONTEXT_END = 416.66
_SCOPE_CLIP_START = 384.29999999999995
_SCOPE_CLIP_END = 408.67999999999995


def _scope_payload(name: str) -> dict[str, object]:
    return json.loads((_SCOPE_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _scope_target() -> TargetASRResult:
    response = _scope_payload("decoder-result.json")["response"]
    result = TranscriptionResult(
        language=response["language"],
        language_probability=response["language_probability"],
        raw_text=response["raw_text"],
        duration=response["duration"],
        segments=response["segments"],
        word_segments=response["word_segments"],
    )

    class _CapturedWhisper:
        def transcribe(self, audio_path, options, cancel_event=None):  # type: ignore[no-untyped-def]
            del audio_path, options, cancel_event
            return result

    return TargetedASREngine(
        options_for=lambda _priority: TranscriptionOptions(
            model="large-v3-turbo",
            device="cpu",
            compute_type="auto",
            beam_size=8,
            condition_on_previous_text=False,
            vad_filter=False,
        ),
        engine=_CapturedWhisper(),
    ).transcribe(
        Path("window.wav"),
        context_start=_SCOPE_CONTEXT_START,
        context_end=_SCOPE_CONTEXT_END,
        priority=RefinementPriority.FINAL_CLIP,
    )


class _CapturedASR:
    def __init__(self, target: TargetASRResult) -> None:
        self._target = target
        self.calls = 0

    def transcribe(self, audio_path, **kwargs):  # type: ignore[no-untyped-def]
        del audio_path, kwargs
        self.calls += 1
        return self._target


class _FixtureAudio(_FakeAudio):
    def context_bounds(self, *, coarse_start, coarse_end, source_duration, priority):  # type: ignore[no-untyped-def]
        del source_duration
        pre, post = (8.0, 8.0) if priority is RefinementPriority.FINAL_CLIP else (5.0, 5.0)
        return max(0.0, coarse_start - pre), coarse_end + post


def _scope_local_identity(target: TargetASRResult | None) -> dict[str, object]:
    if target is None:
        return {"provider": "faster-whisper", "model": "fake"}
    settings = target.settings or {}
    options = {
        name: settings[name]
        for name in TranscriptionOptions.__dataclass_fields__
        if name in settings
    }
    return {
        "provider": target.provider,
        "model": target.model,
        "device": settings.get("device"),
        "compute_type": settings.get("compute_type"),
        "condition_on_previous_text": settings.get("condition_on_previous_text"),
        "vad_filter": settings.get("vad_filter"),
        "final_beam_size": settings.get("beam_size"),
        "decoder_options": {
            RefinementPriority.CANDIDATE.value: dict(options),
            RefinementPriority.FINAL_CLIP.value: dict(options),
        },
    }


def _execute_scoped_final(
    session: Session,
    storage: StorageService,
    *,
    target: TargetASRResult | None = None,
    asr: object | None = None,
    row: CandidateRefinement | None = None,
    local_identity: dict[str, object] | None = None,
) -> tuple[CandidateRefinement, ClipCandidate]:
    candidate = (
        session.get(ClipCandidate, row.clip_candidate_id)
        if row is not None
        else _make_candidate(session, start=384.28, end=408.66)
    )
    assert candidate is not None
    resolved = row or _refinement_row(session, candidate, priority=RefinementPriority.FINAL_CLIP)
    if target is None and asr is None:
        target = _scope_target()
    identity_target = target or getattr(asr, "_target", None)
    executor = CandidateRefinementExecutor(
        session=session,
        storage=storage,
        config=DEFAULT_CONFIG,
        audio_service=_FixtureAudio(storage),
        asr_engine=asr or _CapturedASR(target),  # type: ignore[arg-type]
        local_identity=local_identity or _scope_local_identity(identity_target),
    )
    executor.execute(resolved.id)
    session.refresh(resolved)
    return resolved, candidate


def _scope_envelope(row: CandidateRefinement) -> dict[str, object]:
    envelope = (row.provider_evidence or {}).get("final_clip_view")
    assert isinstance(envelope, dict)
    return envelope


def _seed_scope_stage41(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
    candidate: ClipCandidate,
    row: CandidateRefinement,
) -> tuple[object, object]:
    """Retarget a real Stage 4.0 analysis/strategy seed onto the scoped row."""

    from stage41_support import FakeStage41Settings, install_stage41_settings, seed_stage41

    from app.transformation.executor import build_transformation_executor

    settings = FakeStage41Settings()
    install_stage41_settings(monkeypatch, settings)
    _source, _candidate, _refinement, analysis, _strategies = seed_stage41(
        session, settings=settings
    )
    analysis.clip_candidate_id = candidate.id
    analysis.source_video_id = candidate.source_video_id
    analysis.refinement_id = row.id
    analysis.refinement_priority = row.priority.value
    analysis.refinement_quality_level = row.quality_level
    analysis.input_fingerprint = build_transformation_executor(session, settings).input_fingerprint(
        candidate
    )
    session.commit()
    return settings, analysis


def test_final_clip_fixture_persists_scoped_view_without_promoting_context(
    session: Session, storage: StorageService
) -> None:
    row, _candidate = _execute_scoped_final(session, storage)

    local = next(item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR")
    assert local["state"] == "REJECTED"
    assert local["reason"] == "nonpositive word duration"
    assert local["entity_metadata"]["raw_word_record_count"] == 107
    assert len(local["entity_metadata"]["raw_word_observations"]) == 107

    assert row.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
    assert row.cache_eligible is False
    assert len(row.automatic_transcript.split()) == 107
    assert len(row.final_transcript.split()) == 67
    assert len(row.word_timestamps) == 67
    assert row.refined_start == pytest.approx(_SCOPE_CLIP_START)
    assert row.refined_end == pytest.approx(_SCOPE_CLIP_END)

    expected_words = _scope_payload("result.json")["selected_words"]
    assert len(expected_words) == 67
    for offset, (persisted, expected) in enumerate(
        zip(row.word_timestamps, expected_words, strict=True)
    ):
        assert persisted["text"] == expected["text"]
        assert persisted["start"] == expected["start"]
        assert persisted["end"] == expected["end"]
        assert persisted["parent_record_index"] == 17 + offset

    assert not row.final_transcript.startswith("معي")
    assert "هفهم" not in row.final_transcript

    envelope = _scope_envelope(row)
    assert envelope["state"] == "ACCEPTED"
    assert envelope["policy_version"] == FINAL_CLIP_SCOPE_POLICY_VERSION
    assert envelope["context_only_invalid_record_indexes"] == [105]
    assert envelope["observation_record_count"] == 107
    assert len(envelope["words"]) == 67
    assert row.output_fingerprint


def test_final_clip_scope_refusal_keeps_review_and_explicit_reason(
    session: Session, storage: StorageService
) -> None:
    target = _scope_target()
    observations = copy.deepcopy(list(target.raw_word_observations))
    observations[105]["source_start"] = 404.0
    observations[105]["source_end"] = 404.0
    observations[105]["relative_start"] = 404.0 - _SCOPE_CONTEXT_START
    observations[105]["relative_end"] = 404.0 - _SCOPE_CONTEXT_START
    scope_fingerprint = raw_observation_fingerprint(
        raw_observations=observations,
        record_count=target.raw_word_record_count,
        observations_complete=True,
        context_start=_SCOPE_CONTEXT_START,
        context_end=_SCOPE_CONTEXT_END,
    )
    modified = replace(
        target,
        raw_word_observations=tuple(observations),
        raw_word_observations_fingerprint=scope_fingerprint,
    )

    row, _candidate = _execute_scoped_final(session, storage, target=modified)

    envelope = _scope_envelope(row)
    assert envelope["state"] == "REJECTED"
    assert envelope["reason"] == "zero_duration_word_inside_clip"
    assert row.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
    assert row.cache_eligible is False
    assert any(
        span.get("span_id") == "final-clip-scope"
        and span.get("reason") == "zero_duration_word_inside_clip"
        for span in row.unresolved_spans
    )


def test_scope_refusal_cannot_be_waived_by_manual_text_resolution_or_rerun(
    session: Session, storage: StorageService
) -> None:
    target = _scope_target()
    observations = copy.deepcopy(list(target.raw_word_observations))
    observations[105]["source_start"] = 404.0
    observations[105]["source_end"] = 404.0
    observations[105]["relative_start"] = 404.0 - _SCOPE_CONTEXT_START
    observations[105]["relative_end"] = 404.0 - _SCOPE_CONTEXT_START
    scope_fingerprint = raw_observation_fingerprint(
        raw_observations=observations,
        record_count=target.raw_word_record_count,
        observations_complete=True,
        context_start=_SCOPE_CONTEXT_START,
        context_end=_SCOPE_CONTEXT_END,
    )
    modified = replace(
        target,
        raw_word_observations=tuple(observations),
        raw_word_observations_fingerprint=scope_fingerprint,
    )

    row, candidate = _execute_scoped_final(session, storage, target=modified)
    assert _scope_envelope(row)["state"] == "REJECTED"

    # Operator wording that maps to the retained fallback words cannot supply
    # the missing positive-duration timing, and a named resolution cannot waive
    # the structural refusal.
    manual = " ".join(word["text"].strip() for word in row.word_timestamps)
    apply_manual_transcript(
        session,
        row.id,
        manual,
        resolutions={"final-clip-scope": "operator says the timing is fine"},
    )
    session.refresh(row)
    assert row.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
    assert row.cache_eligible is False
    assert row.final_transcript == manual
    scope_span = next(
        span for span in row.unresolved_spans if span.get("span_id") == "final-clip-scope"
    )
    assert scope_span["resolution_state"] == "UNRESOLVED"
    assert scope_span["operator_resolution"] is None
    assert resolve_effective_refinement(session, candidate) is None
    with pytest.raises(TransformationInputError):
        build_transformation_inputs(session, candidate, row, Stage40Config())

    # An ordinary rerun with unchanged observations must not turn the refused
    # scope into an optimistic final-ready row.
    counting = _CapturedASR(modified)
    _execute_scoped_final(session, storage, asr=counting, row=row)
    session.refresh(row)
    assert counting.calls == 0
    assert row.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
    assert row.cache_eligible is False
    assert _scope_envelope(row)["state"] == "REJECTED"
    assert resolve_effective_refinement(session, candidate) is None
    with pytest.raises(TransformationInputError):
        build_transformation_inputs(session, candidate, row, Stage40Config())


def test_scoped_output_reaches_planning_caption_and_stage51_consumers(
    session: Session, storage: StorageService
) -> None:
    row, candidate = _execute_scoped_final(session, storage)

    inputs = build_transformation_inputs(session, candidate, row, Stage40Config())
    assert inputs.transcript == row.final_transcript
    assert (inputs.refined_start, inputs.refined_end) == (row.refined_start, row.refined_end)
    assert [dict(word) for word in inputs.word_timestamps] == row.word_timestamps
    assert inputs.refinement_output_fingerprint == row.output_fingerprint

    payload = _caption_source_payload(row)
    assert payload["final_transcript"] == row.final_transcript
    rendered_words = [dict(word) for word in payload["word_timestamps"]]
    assert len(rendered_words) == len(row.word_timestamps)
    for index, (rendered, persisted) in enumerate(
        zip(rendered_words, row.word_timestamps, strict=True)
    ):
        assert rendered["index"] == index
        assert rendered["text"] == persisted["text"].strip()
        assert rendered["start"] == pytest.approx(persisted["start"])
        assert rendered["end"] == pytest.approx(persisted["end"])
        assert rendered["probability"] == persisted["probability"]
    assert payload["final_refinement_output_fingerprint"] == row.output_fingerprint

    caption_plan = build_caption_plan(
        [(0, row.refined_start, row.refined_end, None, None, "SOURCE_EXCERPT")],
        payload["word_timestamps"],
        CaptionStyle(),
        Stage51Config(),
    )
    assert caption_plan.missing_evidence == ()
    assert caption_plan.events
    event_tokens = [token for event in caption_plan.events for token in event.text.split()]
    assert event_tokens == row.final_transcript.split()
    for event in caption_plan.events:
        for timing in event.word_timings:
            assert row.refined_start <= timing.start + 1e-6
            assert timing.end <= row.refined_end + 1e-6


def test_supported_manual_correction_uses_same_clip_view_and_invalidates_output(
    session: Session, storage: StorageService, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, candidate = _execute_scoped_final(session, storage)
    original_words = copy.deepcopy(row.word_timestamps)
    original_output = row.output_fingerprint

    tokens = row.final_transcript.split()
    corrected = " ".join([*tokens[:-1], "خامس"])
    apply_manual_transcript(session, row.id, corrected)
    session.refresh(row)

    assert row.status is RefinementStatus.FINAL_TRANSCRIPT_READY
    assert row.cache_eligible is True
    assert row.manual_transcript == corrected
    assert row.final_transcript == corrected
    assert row.word_timestamps == original_words
    assert row.output_fingerprint != original_output

    caption_text, caption_words, projection = _caption_display(row)
    assert caption_text == corrected
    assert projection is not None and projection.supported
    assert caption_words[-1]["text"] == "خامس"
    assert caption_words[-1]["start"] == original_words[-1]["start"]
    assert caption_words[-1]["end"] == original_words[-1]["end"]

    payload = _caption_source_payload(row)
    plan = build_caption_plan(
        [(0, row.refined_start, row.refined_end, None, None, "SOURCE_EXCERPT")],
        payload["word_timestamps"],
        CaptionStyle(),
        Stage51Config(),
    )
    event_tokens = " ".join(event.text for event in plan.events).split()
    assert event_tokens == corrected.split()

    # Stage 4.0/4.1 must consume the same effective word view as captions: the
    # corrected token reaches source-selection/quote evidence without losing any
    # original index, time or parent raw-record mapping.
    stage40 = build_transformation_inputs(session, candidate, row, Stage40Config())
    assert stage40.transcript == payload["final_transcript"] == corrected
    assert [dict(word) for word in stage40.word_timestamps] == [
        dict(word) for word in payload["word_timestamps"]
    ]

    settings, analysis = _seed_scope_stage41(session, monkeypatch, candidate, row)
    planning = build_planning_inputs(
        session,
        candidate,
        analysis,  # type: ignore[arg-type]
        row,
        settings,  # type: ignore[arg-type]
        PLANNING_CONFIG,
    )
    assert planning.transcript == corrected
    assert [word.text for word in planning.words] == event_tokens
    assert planning.words[-1].text == "خامس"
    assert planning.words[-1].start == pytest.approx(original_words[-1]["start"])
    assert planning.words[-1].end == pytest.approx(original_words[-1]["end"])
    assert planning.words[-1].index == len(planning.words) - 1
    for parsed, effective in zip(planning.words, stage40.word_timestamps, strict=True):
        assert parsed.text == effective["text"]
        assert parsed.start == pytest.approx(effective["start"])
        assert parsed.end == pytest.approx(effective["end"])
        assert effective["parent_record_index"] is not None
    assert planning.words[-1].start == pytest.approx(payload["word_timestamps"][-1]["start"])
    assert (
        payload["word_timestamps"][-1]["parent_record_index"]
        == original_words[-1]["parent_record_index"]
    )


def test_unsupported_manual_correction_preserves_text_and_stays_blocked(
    session: Session, storage: StorageService
) -> None:
    row, _candidate = _execute_scoped_final(session, storage)
    tokens = row.final_transcript.split()
    corrected = " ".join([tokens[0], "مختلف", "آخر", *tokens[3:]])
    apply_manual_transcript(session, row.id, corrected)
    session.refresh(row)

    assert row.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
    assert row.final_transcript == corrected
    assert row.cache_eligible is False
    caption_text, _words, projection = _caption_display(row)
    assert caption_text == corrected
    assert projection is not None and projection.supported is False


def test_scoped_manual_raw_sequence_with_nonrecord_item_is_refused(
    session: Session, storage: StorageService
) -> None:
    row, candidate = _execute_scoped_final(session, storage)
    tokens = row.final_transcript.split()
    corrected = " ".join([*tokens[:-1], "خامس"])
    apply_manual_transcript(session, row.id, corrected)
    session.refresh(row)
    assert row.status is RefinementStatus.FINAL_TRANSCRIPT_READY
    assert len(row.word_timestamps) == 67

    stored = list(row.word_timestamps)
    stored.insert(10, "not-a-word-record")
    row.word_timestamps = stored
    session.commit()
    session.refresh(row)

    assert len(row.word_timestamps) == 68
    assert row.final_transcript == corrected
    assert row.manual_transcript == corrected

    with pytest.raises(TransformationInputError):
        build_transformation_inputs(session, candidate, row, Stage40Config())

    caption_text, caption_words, projection = _caption_display(row)
    assert caption_text == corrected
    assert projection is not None
    assert projection.supported is False
    assert projection.reason == "manual_transcript_invalid_word_evidence"
    assert len(caption_words) == 68

    payload = _caption_source_payload(row)
    assert payload["display_projection_status"] == "UNSUPPORTED"
    assert payload["display_projection_reason"] == "manual_transcript_invalid_word_evidence"


def test_rejected_observations_are_reused_without_promotion_or_new_asr(
    session: Session, storage: StorageService
) -> None:
    row, _candidate = _execute_scoped_final(session, storage)
    assert row.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW

    counting = _CapturedASR(_scope_target())
    _execute_scoped_final(session, storage, asr=counting, row=row)
    session.refresh(row)

    assert counting.calls == 0
    local = next(item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR")
    assert local["state"] == "REJECTED"
    assert _scope_envelope(row)["state"] == "ACCEPTED"


@pytest.mark.parametrize(
    "identity_change",
    [
        {"model": "different-decoder-model"},
        {"final_beam_size": 5},
        {"device": "cuda"},
    ],
)
def test_rejected_observation_reuse_requires_current_decoder_identity(
    session: Session,
    storage: StorageService,
    identity_change: dict[str, object],
) -> None:
    row, _candidate = _execute_scoped_final(session, storage)
    original_input = row.input_fingerprint

    target = _scope_target()
    identity = _scope_local_identity(target)
    identity.update(identity_change)
    counting = _CapturedASR(target)
    _execute_scoped_final(
        session,
        storage,
        asr=counting,
        row=row,
        local_identity=identity,
    )
    session.refresh(row)

    assert counting.calls == 1
    assert row.input_fingerprint != original_input
    local = next(item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR")
    assert local["state"] == "REJECTED"
    assert local["entity_metadata"]["raw_word_observations"]


def _production_settings(**overrides: object):
    from app.core.settings import Settings

    values: dict[str, object] = {
        "whisper_model": "large-v3-turbo",
        "whisper_device": "cpu",
        "whisper_compute_type": None,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def _production_final_executor(
    session: Session,
    storage: StorageService,
    settings: object,
    monkeypatch: pytest.MonkeyPatch,
    *,
    asr: object,
) -> CandidateRefinementExecutor:
    """Build the real production factory with external resources stubbed."""

    from app.core.settings import Settings
    from app.runtime.heavy_model_lease import NoopHeavyModelLeaseFactory

    monkeypatch.setattr(
        Settings,
        "heavy_model_lease_factory",
        lambda self: NoopHeavyModelLeaseFactory(),
    )
    monkeypatch.setattr(Settings, "hosted_transcription_provider", lambda self: None)
    monkeypatch.setattr(Settings, "adjudication_provider", lambda self: None)
    monkeypatch.setattr(Settings, "gemini_admission_controller", lambda self: None)
    executor = build_candidate_refinement_executor(session, storage, settings)
    executor._audio_service = _FixtureAudio(storage)  # type: ignore[attr-defined]
    executor._asr = asr  # type: ignore[attr-defined]
    return executor


def test_production_final_configuration_reuses_rejected_observations(
    session: Session, storage: StorageService, monkeypatch: pytest.MonkeyPatch
) -> None:
    row, _candidate = _execute_scoped_final(session, storage)
    assert len(row.word_timestamps) == 67
    local = next(item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR")
    assert local["state"] == "REJECTED"

    counting = _CapturedASR(_scope_target())
    executor = _production_final_executor(
        session,
        storage,
        _production_settings(),
        monkeypatch,
        asr=counting,
    )
    executor.execute(row.id)
    session.refresh(row)

    assert counting.calls == 0
    local = next(item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR")
    assert local["state"] == "REJECTED"
    assert _scope_envelope(row)["state"] == "ACCEPTED"


@pytest.mark.parametrize(
    ("overrides", "attribute", "expected"),
    [
        ({"whisper_compute_type": "float32"}, "compute_type", "float32"),
        ({"whisper_cpu_compute_type": "float32"}, "cpu_compute_type", "float32"),
        ({"whisper_temperature": (0.9,)}, "temperature", (0.9,)),
    ],
)
def test_production_decoder_option_change_forces_fresh_backbone(
    session: Session,
    storage: StorageService,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    attribute: str,
    expected: object,
) -> None:
    row, _candidate = _execute_scoped_final(session, storage)
    settings = _production_settings(**overrides)
    requested = settings.targeted_transcription_options("FINAL_CLIP")
    assert getattr(requested, attribute) == expected

    counting = _CapturedASR(_scope_target())
    executor = _production_final_executor(
        session,
        storage,
        settings,
        monkeypatch,
        asr=counting,
    )
    executor.execute(row.id)
    session.refresh(row)

    assert counting.calls == 1
    local = next(item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR")
    assert local["state"] == "REJECTED"


def test_rejected_candidate_retry_keeps_fresh_backbone_recovery(
    session: Session, storage: StorageService
) -> None:
    raw = TranscriptionResult(
        language="ar",
        language_probability=0.99,
        raw_text=_INDEX_TEXT,
        duration=16.0,
        segments=[],
        word_segments=[
            {"word": "أنا", "start": 0.4, "end": 0.8, "probability": 0.95},
            {"word": " عملت", "start": 1.0, "end": 1.4, "probability": 0.95},
            {"word": " امبارح", "start": 2.0, "end": 2.0, "probability": 0.95},
        ],
    )

    class _ZeroDurationWhisper:
        def transcribe(self, audio_path, options, cancel_event=None):  # type: ignore[no-untyped-def]
            del audio_path, options, cancel_event
            return raw

    rejected = TargetedASREngine(
        options_for=lambda _priority: TranscriptionOptions(
            model="fake", device="cpu", compute_type="int8", beam_size=5
        ),
        engine=_ZeroDurationWhisper(),
    ).transcribe(
        Path("fake.wav"),
        context_start=5.0,
        context_end=21.0,
        priority=RefinementPriority.CANDIDATE,
    )
    candidate = _make_candidate(session)
    row = _refinement_row(session, candidate, priority=RefinementPriority.CANDIDATE)
    _executor(session, storage, asr_engine=_CapturedASR(rejected)).execute(row.id)
    session.refresh(row)
    assert row.status is RefinementStatus.REFINEMENT_FAILED
    bad_local = next(
        item for item in row.transcript_evidence if item["kind"] == "TARGETED_LOCAL_ASR"
    )
    assert bad_local["state"] == "REJECTED"
    assert bad_local["entity_metadata"]["raw_word_observations"]

    corrected = _FakeASR(_RECOVERED_TEXT, _source_time_words())
    _executor(session, storage, asr_engine=corrected).execute(row.id)
    session.refresh(row)

    assert corrected.calls == 1
    assert row.status is RefinementStatus.CANDIDATE_REFINED
