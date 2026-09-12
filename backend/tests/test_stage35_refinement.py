"""Candidate-scoped Stage 3.5 refinement service/executor tests."""

from __future__ import annotations

import hashlib
import io
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

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
from app.refinement.audio_window import RefinementAudioWindow
from app.refinement.executor import CandidateRefinementCancelled, CandidateRefinementExecutor
from app.refinement.hosted import (
    HostedErrorCategory,
    HostedProviderError,
    HostedTranscriptionResult,
)
from app.refinement.policy import DEFAULT_CONFIG
from app.refinement.service import CandidateRefinementService
from app.refinement.types import (
    AdjudicationResult,
    RefinementCancelled,
    TargetASRResult,
    WordTimestamp,
)
from app.services.storage import StorageCategory, StorageService
from app.transcription.dialect import ArabicDialectProfile

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
