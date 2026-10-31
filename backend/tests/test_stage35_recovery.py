"""Operator-only local observation recovery tests.

The captured decoder response is used as an offline fixture. These tests never
load a model, call a provider, or touch the operational database/storage. They
exercise the real production conversion, scope derivation, rejected-observation
reuse, manual projection, and Stage 4/caption consumers over isolated SQLite.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.candidates.fingerprints import candidate_key
from app.candidates.registration import (
    OPERATOR_SELECTION_VERSION,
    compute_raw_span_fingerprint,
)
from app.core.enums import (
    CandidateDisposition,
    JobKind,
    JobStatus,
    MediaOriginType,
    RefinementPriority,
    RefinementStatus,
    RightsStatus,
)
from app.db.base import Base
from app.models import (
    AudioArtifact,
    CandidateRefinement,
    ClipCandidate,
    ProcessingJob,
    SourceVideo,
    Transcript,
)
from app.refinement.audio_window import CandidateAudioWindowService
from app.refinement.executor import CandidateRefinementExecutor, build_local_identity
from app.refinement.policy import DEFAULT_CONFIG
from app.refinement.queue import apply_manual_transcript
from app.refinement.recovery import (
    LocalObservationExpectation,
    LocalObservationRecoveryError,
    LocalObservationRecoveryService,
    RecoveryApplyResult,
)
from app.render.service import _caption_display, _caption_source_payload
from app.services.storage import StorageCategory, StorageService
from app.transcription.dialect import ArabicDialectProfile
from app.transcription.service import TranscriptionOptions
from app.transformation.inputs import build_transformation_inputs
from app.transformation.policy import Stage40Config

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "final_clip_scope"
_CONTEXT_START = 376.28
_CONTEXT_END = 416.66
_COARSE_START = 384.28
_COARSE_END = 408.66
_CLIP_START = 384.29999999999995
_CLIP_END = 408.67999999999995
_APPROVED_TEXT_SHA256 = "6cc761f9da32f3469c0cd0eca1554fd3a59fd47bf6afc7e088ce5332e74c7f9e"
_APPROVED_TOKEN_COUNT = 67
_RAW_RECORD_COUNT = 107
_OPTIONS = TranscriptionOptions(
    model="large-v3-turbo",
    device="cpu",
    compute_type="auto",
    beam_size=8,
    language=None,
    word_timestamps=True,
    cpu_compute_type="int8",
    cuda_compute_type="float16",
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    condition_on_previous_text=False,
    vad_filter=False,
    initial_prompt=None,
    hotwords=None,
    cpu_threads=0,
    index_batch_size=1,
)
_SETTINGS_STUB = SimpleNamespace(
    whisper_model="large-v3-turbo",
    whisper_device="cpu",
    refinement_candidate_beam_size=5,
    refinement_final_beam_size=8,
)


def _payload(name: str) -> dict[str, object]:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _options_for(_priority: RefinementPriority) -> TranscriptionOptions:
    return _OPTIONS


def _local_identity() -> dict[str, object]:
    return build_local_identity(_options_for, _SETTINGS_STUB)


def _wav_bytes(*, seconds: float = 1.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _segments(response: dict[str, object]) -> list[dict[str, object]]:
    words = list(response["word_segments"])  # type: ignore[arg-type]
    groups = 16
    size = (len(words) + groups - 1) // groups
    chunks = [words[offset : offset + size] for offset in range(0, len(words), size)]
    segments: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks):
        first = chunk[0]
        last = chunk[-1]
        start = _CONTEXT_START + float(first["start"])
        end = _CONTEXT_START + float(last["end"])
        if end <= start:
            end = start + 0.01
        text = "".join(str(word["word"]) for word in chunk).strip()
        segments.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "text": text,
                "raw_text": text,
                "corrected_text": text,
                "final_text": text,
                "words": [
                    {
                        "word": str(word["word"]),
                        "start": _CONTEXT_START + float(word["start"]),
                        "end": _CONTEXT_START + float(word["end"]),
                        "probability": word.get("probability"),
                    }
                    for word in chunk
                ],
            }
        )
    segments[0]["start"] = _CONTEXT_START
    segments[-1]["end"] = _CONTEXT_END
    return segments


def _marker(
    source: SourceVideo,
    transcript: Transcript,
    artifact: AudioArtifact,
    candidate_key_value: str,
    raw_span_fingerprint: str,
) -> dict[str, object]:
    return {
        "version": OPERATOR_SELECTION_VERSION,
        "origin": "OPERATOR",
        "source_id": str(source.id),
        "source_content_hash": source.content_hash,
        "transcript_id": str(transcript.id),
        "transcript_input_fingerprint": transcript.input_fingerprint,
        "transcription_revision": transcript.transcription_revision,
        "audio_artifact_content_hash": artifact.content_hash,
        "audio_artifact_source_content_hash": artifact.source_content_hash,
        "start_time": _COARSE_START,
        "end_time": _COARSE_END,
        "start_segment_index": 0,
        "end_segment_index": 15,
        "segment_indexes": list(range(16)),
        "reason": "Operator-selected complete richer-wife exchange",
        "raw_span_fingerprint": raw_span_fingerprint,
        "candidate_key": candidate_key_value,
    }


def _bundle_files(
    storage: StorageService,
    *,
    source: SourceVideo,
    transcript: Transcript,
    artifact: AudioArtifact,
    candidate: ClipCandidate,
    refinement: CandidateRefinement,
    job: ProcessingJob,
    marker: dict[str, object],
    audio_sha: str,
    raw_span_fingerprint: str,
    audio_input_fingerprint: str,
    audio_path: str,
) -> tuple[str, dict[str, str]]:
    relative = f"diagnostics/synthetic-recovery-{uuid4().hex}"
    bundle = storage.resolve(StorageCategory.TEMPORARY, relative)
    bundle.mkdir(parents=True, exist_ok=True)

    decoder = copy.deepcopy(_payload("decoder-result.json"))
    decoder["context_start"] = _CONTEXT_START
    decoder["context_end"] = _CONTEXT_END
    decoder["audio_sha256"] = audio_sha
    decoder["options"] = {
        **copy.deepcopy(decoder["options"]),
        "model": _OPTIONS.model,
        "device": _OPTIONS.device,
        "compute_type": _OPTIONS.compute_type,
        "beam_size": _OPTIONS.beam_size,
        "language": _OPTIONS.language,
        "word_timestamps": _OPTIONS.word_timestamps,
        "cpu_compute_type": _OPTIONS.cpu_compute_type,
        "cuda_compute_type": _OPTIONS.cuda_compute_type,
        "temperature": list(_OPTIONS.temperature),
        "condition_on_previous_text": _OPTIONS.condition_on_previous_text,
        "vad_filter": _OPTIONS.vad_filter,
        "initial_prompt": _OPTIONS.initial_prompt,
        "hotwords": _OPTIONS.hotwords,
        "cpu_threads": _OPTIONS.cpu_threads,
        "index_batch_size": _OPTIONS.index_batch_size,
    }

    result = copy.deepcopy(_payload("result.json"))
    result["candidate_id"] = str(candidate.id)
    result["raw_span_fingerprint"] = raw_span_fingerprint
    result["job"] = {
        **dict(result["job"]),  # type: ignore[arg-type]
        "id": str(job.id),
        "status": JobStatus.SUCCEEDED.value,
    }
    result_refinement = result["refinement"]
    result_refinement.update(  # type: ignore[union-attr]
        {
            "id": str(refinement.id),
            "source_video_id": str(source.id),
            "clip_candidate_id": str(candidate.id),
            "priority": RefinementPriority.FINAL_CLIP.value,
            "quality_level": RefinementPriority.FINAL_CLIP.value,
            "status": RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW.value,
            "coarse_start": _COARSE_START,
            "coarse_end": _COARSE_END,
            "context_start": _CONTEXT_START,
            "context_end": _CONTEXT_END,
            "refined_start": _CLIP_START,
            "refined_end": _CLIP_END,
            "audio_relative_path": audio_path,
            "audio_content_hash": audio_sha,
            "audio_input_fingerprint": audio_input_fingerprint,
            "cache_eligible": False,
            "metrics": {"local_asr_runs": 1},
        }
    )

    prior = copy.deepcopy(_payload("prior-evidence.json"))
    prior["candidate_id"] = str(candidate.id)
    prior["raw_span_fingerprint"] = raw_span_fingerprint
    prior["source_hash"] = source.content_hash
    prior["transcript_id"] = str(transcript.id)
    prior["transcript_input_fingerprint"] = transcript.input_fingerprint
    prior["transcription_revision"] = transcript.transcription_revision
    prior["audio_sha256"] = audio_sha
    prior["audio_path"] = audio_path
    prior["operator_marker"] = marker
    prior_refinement = prior["refinement"]
    prior_refinement.update(  # type: ignore[union-attr]
        {
            "id": str(refinement.id),
            "source_video_id": str(source.id),
            "clip_candidate_id": str(candidate.id),
            "priority": RefinementPriority.FINAL_CLIP.value,
            "quality_level": RefinementPriority.FINAL_CLIP.value,
            "coarse_start": _COARSE_START,
            "coarse_end": _COARSE_END,
            "context_start": _CONTEXT_START,
            "context_end": _CONTEXT_END,
            "refined_start": _CLIP_START,
            "refined_end": _CLIP_END,
            "audio_relative_path": audio_path,
            "audio_content_hash": audio_sha,
            "audio_input_fingerprint": audio_input_fingerprint,
        }
    )

    files = {
        "decoder-result.json": decoder,
        "result.json": result,
        "prior-evidence.json": prior,
    }
    hashes: dict[str, str] = {}
    for name, payload in files.items():
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        (bundle / name).write_bytes(data)
        hashes[name] = _sha(data)
    return relative, hashes


def _expectation(
    *,
    source: SourceVideo,
    candidate: ClipCandidate,
    refinement: CandidateRefinement,
    job: ProcessingJob,
    marker: dict[str, object],
    bundle_relative_path: str,
    bundle_hashes: dict[str, str],
    audio_sha: str,
) -> LocalObservationExpectation:
    return LocalObservationExpectation(
        bundle_relative_path=bundle_relative_path,
        source_id=source.id,
        candidate_id=candidate.id,
        refinement_id=refinement.id,
        original_job_id=job.id,
        coarse_start=_COARSE_START,
        coarse_end=_COARSE_END,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        refined_start=_CLIP_START,
        refined_end=_CLIP_END,
        candidate_key=str(marker["candidate_key"]),
        raw_span_fingerprint=str(marker["raw_span_fingerprint"]),
        source_sha256=source.content_hash,
        audio_sha256=audio_sha,
        decoder_result_sha256=bundle_hashes["decoder-result.json"],
        result_sha256=bundle_hashes["result.json"],
        prior_evidence_sha256=bundle_hashes["prior-evidence.json"],
        approved_text_sha256=_APPROVED_TEXT_SHA256,
        approved_token_count=_APPROVED_TOKEN_COUNT,
        raw_word_record_count=_RAW_RECORD_COUNT,
        expected_rejection_reason="nonpositive word duration",
        context_only_invalid_record_indexes=(105,),
    )


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


def _seed_recovery(
    session: Session, storage: StorageService, tmp_path: Path
) -> tuple[
    LocalObservationExpectation,
    ClipCandidate,
    CandidateRefinement,
    ProcessingJob,
    SourceVideo,
    str,
]:
    source_bytes = b"synthetic-recovery-source-bytes"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(source_bytes)
    source_sha = _sha(source_bytes)

    response = _payload("decoder-result.json")["response"]
    real_result = _payload("result.json")
    real_refinement = real_result["refinement"]
    real_job = real_result["job"]

    source = SourceVideo(
        id=UUID(str(real_refinement["source_video_id"])),
        source_uri=str(source_path),
        original_filename="source.bin",
        content_hash=source_sha,
        rights_status=RightsStatus.OWNED,
        media_origin=MediaOriginType.OTHER,
    )
    session.add(source)
    session.flush()

    transcript = Transcript(
        id=uuid4(),
        source_video_id=source.id,
        language="ar",
        whisper_model="large-v3-turbo",
        duration=500.0,
        raw_text=str(response["raw_text"]),
        segments=_segments(response),
        dialect_profile=ArabicDialectProfile.EGYPTIAN,
        dialect_confidence=0.9,
        raw_transcript_confidence=0.83,
        correction_confidence=0.9,
        reconstruction_confidence=0.9,
        transcription_revision=1,
        input_fingerprint="synthetic-transcript-input",
        correction_version="v1",
    )
    session.add(transcript)
    session.flush()

    artifact_bytes = _wav_bytes(seconds=1.0)
    artifact_relative = f"{source.id}/speech-analysis.wav"
    artifact_path = storage.resolve(StorageCategory.SOURCES, artifact_relative)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(artifact_bytes)
    artifact = AudioArtifact(
        source_video_id=source.id,
        output_path=artifact_relative,
        content_hash=_sha(artifact_bytes),
        source_content_hash=source.content_hash,
        sample_rate=16000,
        duration=500.0,
    )
    session.add(artifact)
    session.flush()

    key = candidate_key(
        source_id=source.id,
        start_segment_index=0,
        end_segment_index=15,
        span_start=0,
        span_end=15,
    )
    raw_span_fingerprint = compute_raw_span_fingerprint(source, transcript, artifact, 0, 15)
    marker = _marker(source, transcript, artifact, key, raw_span_fingerprint)

    candidate = ClipCandidate(
        id=UUID(str(real_result["candidate_id"])),
        source_video_id=source.id,
        candidate_key=key,
        is_current=True,
        disposition=CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
        start_time=_COARSE_START,
        end_time=_COARSE_END,
        start_segment_index=0,
        end_segment_index=15,
        segment_indexes=list(range(16)),
        transcript_excerpt=" ".join(str(segment["raw_text"]) for segment in transcript.segments),
        clip_score=0.8,
        boundary_confidence=0.3,
        analysis_fingerprint="synthetic-analysis",
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
        code_switch_suspected=False,
        provenance_snapshot={"operator_selection": marker},
    )
    session.add(candidate)
    session.flush()

    audio_bytes = _wav_bytes(seconds=1.0)
    audio_relative = f"{source.id}/candidate-refinements/{candidate.id}/FINAL_CLIP.wav"
    audio_path = storage.resolve(StorageCategory.SOURCES, audio_relative)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(audio_bytes)
    audio_sha = _sha(audio_bytes)
    audio_service = CandidateAudioWindowService(
        storage=storage, session=session, config=DEFAULT_CONFIG
    )
    audio_input_fingerprint = audio_service.audio_input_fingerprint(
        source=source,
        artifact=artifact,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        priority=RefinementPriority.FINAL_CLIP,
    )
    audio_path.with_name(f"{audio_path.name}.fingerprint").write_text(
        audio_input_fingerprint, encoding="utf-8"
    )

    refinement = CandidateRefinement(
        id=UUID(str(real_refinement["id"])),
        source_video_id=source.id,
        clip_candidate_id=candidate.id,
        priority=RefinementPriority.FINAL_CLIP,
        status=RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW,
        quality_level=RefinementPriority.FINAL_CLIP.value,
        coarse_start=_COARSE_START,
        coarse_end=_COARSE_END,
        context_start=_CONTEXT_START,
        context_end=_CONTEXT_END,
        refined_start=_CLIP_START,
        refined_end=_CLIP_END,
        audio_relative_path=audio_relative,
        audio_content_hash=audio_sha,
        audio_input_fingerprint=audio_input_fingerprint,
        automatic_transcript=str(real_refinement["automatic_transcript"]),
        manual_transcript=None,
        final_transcript=str(real_refinement["final_transcript"]),
        word_timestamps=copy.deepcopy(real_refinement["word_timestamps"]),
        confidence=float(real_refinement["confidence"]),
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
        code_switch_evidence=copy.deepcopy(real_refinement["code_switch_evidence"]),
        transcript_evidence=copy.deepcopy(real_refinement["transcript_evidence"]),
        entity_evidence=copy.deepcopy(real_refinement["entity_evidence"]),
        unresolved_spans=copy.deepcopy(real_refinement["unresolved_spans"]),
        provider_evidence=copy.deepcopy(real_refinement["provider_evidence"]),
        routing_evidence=copy.deepcopy(real_refinement["routing_evidence"]),
        input_fingerprint=str(real_refinement["input_fingerprint"]),
        output_fingerprint=str(real_refinement["output_fingerprint"]),
        component_fingerprints=copy.deepcopy(real_refinement["component_fingerprints"]),
        policy_version=str(real_refinement["policy_version"]),
        schema_version=str(real_refinement["schema_version"]),
        validation_version=str(real_refinement["validation_version"]),
        cache_eligible=False,
        metrics=copy.deepcopy(real_refinement["metrics"]),
        processing_duration=float(real_refinement["processing_duration"]),
    )
    session.add(refinement)
    session.flush()

    job = ProcessingJob(
        id=UUID(str(real_job["id"])),
        source_video_id=source.id,
        candidate_refinement_id=refinement.id,
        kind=JobKind.CANDIDATE_REFINEMENT,
        status=JobStatus.SUCCEEDED,
    )
    session.add(job)
    session.commit()

    bundle_relative_path, bundle_hashes = _bundle_files(
        storage,
        source=source,
        transcript=transcript,
        artifact=artifact,
        candidate=candidate,
        refinement=refinement,
        job=job,
        marker=marker,
        audio_sha=audio_sha,
        raw_span_fingerprint=raw_span_fingerprint,
        audio_input_fingerprint=audio_input_fingerprint,
        audio_path=audio_relative,
    )
    expectation = _expectation(
        source=source,
        candidate=candidate,
        refinement=refinement,
        job=job,
        marker=marker,
        bundle_relative_path=bundle_relative_path,
        bundle_hashes=bundle_hashes,
        audio_sha=audio_sha,
    )
    return expectation, candidate, refinement, job, source, bundle_relative_path


def _service(session: Session, storage: StorageService) -> LocalObservationRecoveryService:
    return LocalObservationRecoveryService(
        session=session,
        storage=storage,
        options_for=_options_for,
        local_identity=_local_identity(),
        config=DEFAULT_CONFIG,
    )


def _snapshot_row(row: CandidateRefinement) -> dict[str, object]:
    return {
        "status": row.status.value,
        "quality_level": row.quality_level,
        "automatic_transcript": row.automatic_transcript,
        "manual_transcript": row.manual_transcript,
        "final_transcript": row.final_transcript,
        "word_timestamps": copy.deepcopy(row.word_timestamps),
        "transcript_evidence": copy.deepcopy(row.transcript_evidence),
        "provider_evidence": copy.deepcopy(row.provider_evidence),
        "component_fingerprints": copy.deepcopy(row.component_fingerprints),
        "cache_eligible": row.cache_eligible,
        "metrics": copy.deepcopy(row.metrics),
        "active_job_id": row.active_job_id,
    }


def test_preview_reports_delta_without_writes(
    session: Session, storage: StorageService, tmp_path: Path
) -> None:
    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    before = _snapshot_row(refinement)
    temporary_root = storage.category_root(StorageCategory.TEMPORARY)
    before_files = sorted(
        path.relative_to(temporary_root).as_posix()
        for path in temporary_root.rglob("*")
        if path.is_file()
    )

    preview = _service(session, storage).preview(expectation)

    assert preview.writes == 0
    assert preview.snapshot_relative_path is None
    assert preview.already_admitted is False
    assert preview.reuse_ready is False
    assert preview.reuse_ready_after_apply is True
    assert preview.reconstructed["raw_word_record_count"] == _RAW_RECORD_COUNT
    assert preview.reconstructed["raw_word_observations_complete"] is True
    assert preview.reconstructed["rejected_reason"] == "nonpositive word duration"
    assert preview.derived_view["state"] == "ACCEPTED"
    assert preview.derived_view["selected_word_count"] == _APPROVED_TOKEN_COUNT
    assert preview.derived_view["selected_text_sha256"] == _APPROVED_TEXT_SHA256
    assert preview.derived_view["context_only_invalid_record_indexes"] == [105]
    assert preview.approved_text_matches is True
    assert preview.proposed_delta["cache_eligible"] == {"from": False, "to": False}
    assert "final_transcript" in preview.proposed_delta["untouched"]
    assert _snapshot_row(refinement) == before
    after_files = sorted(
        path.relative_to(temporary_root).as_posix()
        for path in temporary_root.rglob("*")
        if path.is_file()
    )
    assert after_files == before_files


def test_apply_admits_rejected_complete_checkpoint_and_is_idempotent(
    session: Session, storage: StorageService, tmp_path: Path
) -> None:
    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    before = _snapshot_row(refinement)
    before_jobs = session.query(ProcessingJob).count()

    result = _service(session, storage).apply(expectation)
    session.commit()
    session.refresh(refinement)

    assert isinstance(result, RecoveryApplyResult)
    assert result.applied is True
    assert result.already_admitted is False
    assert result.snapshot_relative_path is not None
    snapshot = storage.resolve(StorageCategory.TEMPORARY, result.snapshot_relative_path)
    assert snapshot.is_file()
    snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert snapshot_payload["previous_refinement"]["status"] == before["status"]
    assert snapshot_payload["previous_refinement"]["final_transcript"] == before["final_transcript"]

    local = [
        item for item in refinement.transcript_evidence if item.get("kind") == "TARGETED_LOCAL_ASR"
    ]
    assert len(local) == 1
    record = local[0]
    assert record["state"] == "REJECTED"
    assert record["reason"] == "nonpositive word duration"
    metadata = record["entity_metadata"]
    assert metadata["raw_word_record_count"] == _RAW_RECORD_COUNT
    assert metadata["raw_word_observations_complete"] is True
    assert len(metadata["raw_word_observations"]) == _RAW_RECORD_COUNT
    assert metadata["recovery"]["kind"] == "local-observation-recovery-v1"
    assert metadata["recovery"]["no_model_call"] is True
    assert metadata["recovery"]["bundle_hashes"]
    assert metadata["recovery"]["original_job_id"] == str(expectation.original_job_id)
    assert metadata["timestamp_diagnostic"]["record_index"] == 105

    assert refinement.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
    assert refinement.cache_eligible is False
    assert refinement.active_job_id is None
    assert refinement.manual_transcript == before["manual_transcript"]
    assert refinement.final_transcript == before["final_transcript"]
    assert refinement.automatic_transcript == before["automatic_transcript"]
    assert refinement.word_timestamps == before["word_timestamps"]
    assert refinement.metrics == before["metrics"]
    assert session.query(ProcessingJob).count() == before_jobs
    assert refinement.component_fingerprints["local_asr"] == record["fingerprint"]
    assert (
        refinement.component_fingerprints["audio_extraction"]
        != (before["component_fingerprints"]["audio_extraction"])
    )

    repeated = _service(session, storage).apply(expectation)
    session.commit()
    session.refresh(refinement)
    assert repeated.applied is False
    assert repeated.already_admitted is True
    assert repeated.snapshot_relative_path is None
    assert repeated.preview.reuse_ready is True
    assert _snapshot_row(refinement)["transcript_evidence"] == copy.deepcopy(
        refinement.transcript_evidence
    )


def test_normal_executor_reuses_admitted_checkpoint_without_model_call(
    session: Session, storage: StorageService, tmp_path: Path
) -> None:
    expectation, candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    _service(session, storage).apply(expectation)
    session.commit()
    session.refresh(refinement)

    class _FailOnCallASR:
        calls = 0

        def transcribe(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            type(self).calls += 1
            raise AssertionError("ASR must not be called when the checkpoint is reusable")

    asr = _FailOnCallASR()
    audio_service = CandidateAudioWindowService(
        storage=storage, session=session, config=DEFAULT_CONFIG
    )
    executor = CandidateRefinementExecutor(
        session=session,
        storage=storage,
        config=DEFAULT_CONFIG,
        audio_service=audio_service,
        asr_engine=asr,
        routing_mode="local_only",
        local_identity=_local_identity(),
    )
    executor.execute(refinement.id)
    session.refresh(refinement)

    assert asr.calls == 0
    assert refinement.status is RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
    assert refinement.cache_eligible is False
    assert len(refinement.final_transcript.split()) == _APPROVED_TOKEN_COUNT
    assert refinement.refined_start == pytest.approx(_CLIP_START)
    assert refinement.refined_end == pytest.approx(_CLIP_END)
    assert refinement.metrics.get("local_asr_runs") is None
    assert refinement.metrics.get("component_reuse") == 1
    local = [
        item for item in refinement.transcript_evidence if item.get("kind") == "TARGETED_LOCAL_ASR"
    ]
    assert len(local) == 1
    assert local[0]["state"] == "REJECTED"
    assert local[0]["entity_metadata"]["raw_word_observations"]
    assert local[0]["entity_metadata"]["recovery"]["no_model_call"] is True
    envelope = refinement.provider_evidence["final_clip_view"]
    assert envelope["state"] == "ACCEPTED"
    assert envelope["context_only_invalid_record_indexes"] == [105]
    assert envelope["selected_word_count"] == _APPROVED_TOKEN_COUNT
    assert "معي" not in refinement.final_transcript
    assert "هفهم" not in refinement.final_transcript
    assert "الجو" not in refinement.final_transcript


def test_manual_endpoint_then_stage4_and_caption_agree_without_context_leak(
    session: Session, storage: StorageService, tmp_path: Path
) -> None:
    expectation, candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    _service(session, storage).apply(expectation)
    session.commit()
    session.refresh(refinement)

    class _FailOnCallASR:
        def transcribe(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            raise AssertionError("ASR must not be called")

    executor = CandidateRefinementExecutor(
        session=session,
        storage=storage,
        config=DEFAULT_CONFIG,
        audio_service=CandidateAudioWindowService(
            storage=storage, session=session, config=DEFAULT_CONFIG
        ),
        asr_engine=_FailOnCallASR(),
        routing_mode="local_only",
        local_identity=_local_identity(),
    )
    executor.execute(refinement.id)
    session.refresh(refinement)

    approved = "".join(
        str(word["text"]) for word in _payload("result.json")["selected_words"]
    ).strip()
    assert _sha(approved.encode("utf-8")) == _APPROVED_TEXT_SHA256

    apply_manual_transcript(session, refinement.id, approved)
    session.refresh(refinement)

    assert refinement.status is RefinementStatus.FINAL_TRANSCRIPT_READY
    assert refinement.cache_eligible is True
    assert refinement.final_transcript == approved
    assert refinement.manual_transcript == approved

    caption_text, caption_words, projection = _caption_display(refinement)
    assert caption_text == approved
    assert projection is not None
    assert projection.supported is True
    assert projection.status == "UNCHANGED"
    assert projection.reason is None
    assert len(caption_words) == _APPROVED_TOKEN_COUNT

    payload = _caption_source_payload(refinement)
    assert payload["final_transcript"] == approved
    assert payload["display_projection_status"] == "UNCHANGED"
    assert payload["display_projection_reason"] is None

    inputs = build_transformation_inputs(session, candidate, refinement, Stage40Config())
    assert inputs.transcript == approved
    assert len(inputs.word_timestamps) == _APPROVED_TOKEN_COUNT
    for forbidden in ("معي", "هفهم", "الجو"):
        assert forbidden not in inputs.transcript


def _rewrite_and_repin(
    expectation: LocalObservationExpectation,
    path: Path,
    payload: dict[str, object],
    *,
    field: str,
) -> LocalObservationExpectation:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return replace(expectation, **{field: _sha(data)})


@pytest.mark.parametrize("tamper", ["decoder_hash", "result_hash", "prior_hash"])
def test_hash_tampering_refuses_before_writes(
    session: Session,
    storage: StorageService,
    tmp_path: Path,
    tamper: str,
) -> None:
    expectation, _candidate, refinement, _job, _source, bundle_relative = _seed_recovery(
        session, storage, tmp_path
    )
    bundle = storage.resolve(StorageCategory.TEMPORARY, bundle_relative)
    before = _snapshot_row(refinement)
    before_files = sorted(
        path.relative_to(storage.category_root(StorageCategory.TEMPORARY)).as_posix()
        for path in storage.category_root(StorageCategory.TEMPORARY).rglob("*")
        if path.is_file()
    )

    decoder_path = bundle / "decoder-result.json"
    result_path = bundle / "result.json"
    prior_path = bundle / "prior-evidence.json"
    decoder = json.loads(decoder_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))

    if tamper == "decoder_hash":
        decoder["context_end"] = 416.67
        path, payload = decoder_path, decoder
    elif tamper == "result_hash":
        result["selected_words"] = result["selected_words"][:-1]
        path, payload = result_path, result
    else:
        prior["operator_marker"]["reason"] = "tampered"
        path, payload = prior_path, prior
    path.write_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    with pytest.raises(LocalObservationRecoveryError) as error:
        _service(session, storage).apply(expectation)
    assert error.value.code == "bundle_hash_mismatch"
    session.rollback()
    session.refresh(refinement)
    assert _snapshot_row(refinement) == before
    after_files = sorted(
        path.relative_to(storage.category_root(StorageCategory.TEMPORARY)).as_posix()
        for path in storage.category_root(StorageCategory.TEMPORARY).rglob("*")
        if path.is_file()
    )
    assert after_files == before_files


@pytest.mark.parametrize(
    ("tamper", "code"),
    [
        ("candidate_id", "bundle_candidate_mismatch"),
        ("refinement_id", "bundle_refinement_identity"),
        ("job_id", "bundle_job_mismatch"),
        ("coarse_bounds", "candidate_bounds"),
        ("context_bounds", "bundle_refinement_bounds"),
        ("source_hash", "prior_source_mismatch"),
        ("audio_hash", "prior_audio_mismatch"),
        ("decoder_options", "decoder_contract_mismatch"),
        ("incomplete_raw", "raw_record_count"),
        ("zero_inside_clip", "derived_view_refused"),
        ("candidate_bounds", "candidate_bounds"),
        ("candidate_segments", "candidate_segments"),
    ],
)
def test_semantic_evidence_changes_refuse_at_named_gate(
    session: Session,
    storage: StorageService,
    tmp_path: Path,
    tamper: str,
    code: str,
) -> None:
    expectation, candidate, refinement, _job, _source, bundle_relative = _seed_recovery(
        session, storage, tmp_path
    )
    bundle = storage.resolve(StorageCategory.TEMPORARY, bundle_relative)
    before = _snapshot_row(refinement)
    before_files = sorted(
        path.relative_to(storage.category_root(StorageCategory.TEMPORARY)).as_posix()
        for path in storage.category_root(StorageCategory.TEMPORARY).rglob("*")
        if path.is_file()
    )

    decoder_path = bundle / "decoder-result.json"
    result_path = bundle / "result.json"
    prior_path = bundle / "prior-evidence.json"
    decoder = json.loads(decoder_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))

    if tamper == "candidate_id":
        result["candidate_id"] = str(uuid4())
        expectation = _rewrite_and_repin(expectation, result_path, result, field="result_sha256")
    elif tamper == "refinement_id":
        result["refinement"]["id"] = str(uuid4())
        expectation = _rewrite_and_repin(expectation, result_path, result, field="result_sha256")
    elif tamper == "job_id":
        result["job"]["id"] = str(uuid4())
        expectation = _rewrite_and_repin(expectation, result_path, result, field="result_sha256")
    elif tamper == "coarse_bounds":
        expectation = replace(expectation, coarse_start=384.29)
    elif tamper == "context_bounds":
        expectation = replace(expectation, context_end=416.67)
    elif tamper == "source_hash":
        prior["source_hash"] = "0" * 64
        expectation = _rewrite_and_repin(
            expectation, prior_path, prior, field="prior_evidence_sha256"
        )
    elif tamper == "audio_hash":
        prior["audio_sha256"] = "0" * 64
        expectation = _rewrite_and_repin(
            expectation, prior_path, prior, field="prior_evidence_sha256"
        )
    elif tamper == "decoder_options":
        decoder["options"]["beam_size"] = 5
        expectation = _rewrite_and_repin(
            expectation, decoder_path, decoder, field="decoder_result_sha256"
        )
    elif tamper == "incomplete_raw":
        decoder["response"]["word_segments"] = decoder["response"]["word_segments"][:-1]
        expectation = _rewrite_and_repin(
            expectation, decoder_path, decoder, field="decoder_result_sha256"
        )
    elif tamper == "zero_inside_clip":
        record = decoder["response"]["word_segments"][105]
        record["start"] = 404.0 - _CONTEXT_START
        record["end"] = 404.0 - _CONTEXT_START
        expectation = _rewrite_and_repin(
            expectation, decoder_path, decoder, field="decoder_result_sha256"
        )
    elif tamper == "candidate_bounds":
        candidate.start_time = 384.29
        session.commit()
    elif tamper == "candidate_segments":
        candidate.segment_indexes = [0]
        session.commit()

    with pytest.raises(LocalObservationRecoveryError) as error:
        _service(session, storage).apply(expectation)
    assert error.value.code == code
    session.rollback()
    session.refresh(refinement)
    assert _snapshot_row(refinement) == before
    after_files = sorted(
        path.relative_to(storage.category_root(StorageCategory.TEMPORARY)).as_posix()
        for path in storage.category_root(StorageCategory.TEMPORARY).rglob("*")
        if path.is_file()
    )
    assert after_files == before_files


def test_active_job_and_conflicting_checkpoint_refuse(
    session: Session, storage: StorageService, tmp_path: Path
) -> None:
    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    active = ProcessingJob(
        source_video_id=refinement.source_video_id,
        candidate_refinement_id=refinement.id,
        kind=JobKind.CANDIDATE_REFINEMENT,
        status=JobStatus.RUNNING,
    )
    session.add(active)
    session.commit()

    with pytest.raises(LocalObservationRecoveryError) as error:
        _service(session, storage).apply(expectation)
    assert error.value.code == "refinement_active"
    session.rollback()

    session.delete(active)
    session.commit()
    _service(session, storage).apply(expectation)
    session.commit()
    session.refresh(refinement)

    changed = copy.deepcopy(refinement.transcript_evidence)
    local = next(item for item in changed if item.get("kind") == "TARGETED_LOCAL_ASR")
    local["entity_metadata"]["raw_word_observations"][0]["text"] = "مختلف"
    refinement.transcript_evidence = changed
    session.commit()

    with pytest.raises(LocalObservationRecoveryError) as error:
        _service(session, storage).apply(expectation)
    assert error.value.code == "checkpoint_conflict"


def test_snapshot_failure_prevents_mutation(
    session: Session, storage: StorageService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    before = _snapshot_row(refinement)

    def _fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(storage, "atomic_write", _fail)
    with pytest.raises(LocalObservationRecoveryError) as error:
        _service(session, storage).apply(expectation)
    assert error.value.code == "snapshot_failed"
    session.rollback()
    session.refresh(refinement)
    assert _snapshot_row(refinement) == before


def test_manual_conflict_refuses(session: Session, storage: StorageService, tmp_path: Path) -> None:
    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    refinement.manual_transcript = "نص مختلف"
    session.commit()

    with pytest.raises(LocalObservationRecoveryError) as error:
        _service(session, storage).apply(expectation)
    assert error.value.code == "manual_conflict"


def test_cli_previews_applies_and_requires_reuse(
    session: Session,
    storage: StorageService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from app import cli

    expectation, _candidate, refinement, _job, _source, _bundle = _seed_recovery(
        session, storage, tmp_path
    )
    manifest = tmp_path / "expectation.json"
    manifest.write_text(json.dumps(expectation.as_dict(), ensure_ascii=False), encoding="utf-8")
    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    settings = SimpleNamespace(
        storage_root=storage.storage_root,
        targeted_transcription_options=lambda _priority: _OPTIONS,
        stage35_config=lambda: DEFAULT_CONFIG,
        whisper_model="large-v3-turbo",
        whisper_device="cpu",
        refinement_candidate_beam_size=5,
        refinement_final_beam_size=8,
    )
    monkeypatch.setattr(cli, "create_session_factory", lambda: factory)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    runner = CliRunner()
    preview = runner.invoke(cli.app, ["candidate-recovery", "--expectation", str(manifest)])
    assert preview.exit_code == 0, preview.stdout
    preview_payload = json.loads(preview.stdout)
    assert preview_payload["writes"] == 0
    assert preview_payload["derived_view"]["state"] == "ACCEPTED"
    assert preview_payload["reuse_ready"] is False
    assert preview_payload["reuse_ready_after_apply"] is True

    applied = runner.invoke(
        cli.app,
        [
            "candidate-recovery",
            "--expectation",
            str(manifest),
            "--apply",
            "--require-reuse",
        ],
    )
    assert applied.exit_code == 0, applied.stdout
    applied_payload = json.loads(applied.stdout)
    assert applied_payload["applied"] is True
    assert applied_payload["snapshot_relative_path"]
    assert applied_payload["preview"]["reuse_ready"] is True

    verified = runner.invoke(
        cli.app,
        [
            "candidate-recovery",
            "--expectation",
            str(manifest),
            "--require-reuse",
        ],
    )
    assert verified.exit_code == 0, verified.stdout
    verified_payload = json.loads(verified.stdout)
    assert verified_payload["reuse_ready"] is True
    session.expire_all()
    assert session.get(CandidateRefinement, refinement.id) is not None
