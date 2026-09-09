"""Concrete durable Stage 2 executors used only by worker processes."""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Protocol, cast

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.enums import JobKind, JobStatus, ReconstructionStatus
from app.core.settings import get_settings
from app.media.analysis import parse_silencedetect, silence_ratio, windowed_rms
from app.media.audio import AudioExtractor
from app.media.ffprobe import FFprobe
from app.models import (
    AudioAnalysis,
    AudioArtifact,
    ProcessingJob,
    SourceVideo,
    Transcript,
    TranscriptChunk,
)
from app.pipeline.executor import StageExecutionResult
from app.pipeline.fingerprints import canonical_fingerprint, reconstruction_target_fingerprint
from app.pipeline.runner import StageExecutionError
from app.runtime.heavy_model_lease import (
    HeavyModelLeaseBusy,
    HeavyModelLeaseFactory,
    NoopHeavyModelLeaseFactory,
)
from app.runtime.memory import MemorySnapshot, capture_memory
from app.services.source_adapters import YtDlpAdapter
from app.services.source_quality import assess_source, quality_input_fingerprint
from app.services.storage import StorageCategory, StorageService
from app.transcription.chunking import ChunkConfig, build_chunks
from app.transcription.correction import ContextualCorrector
from app.transcription.engine import TranscriptionResult, WhisperEngine
from app.transcription.normalization import normalize_transcript
from app.transcription.reconstruction import ContextualReconstructor
from app.transcription.reconstruction.providers import (
    ReconstructionProvider,
    ReconstructionRequest,
)
from app.transcription.reconstruction.service import select_final_text
from app.transcription.reconstruction.status import aggregate_reconstruction_status
from app.transcription.reconstruction.types import (
    ConfidenceLevel,
    ProviderHealth,
    QualityFlag,
    ReconstructionCandidate,
    RequestSizeDiagnostics,
    SegmentReconstruction,
    UnloadOutcome,
    WordEvidence,
)
from app.transcription.service import TranscriptionOptions

_logger = logging.getLogger("clipfactory.stages")


class TranscriptionExecutor:
    """Run local Whisper once per cache fingerprint and persist raw evidence."""

    def __init__(
        self,
        *,
        session: Session,
        engine: WhisperEngine,
        options: TranscriptionOptions | None = None,
        storage: StorageService | None = None,
        lease_factory: HeavyModelLeaseFactory | NoopHeavyModelLeaseFactory | None = None,
        snapshotter: Callable[[], MemorySnapshot] = capture_memory,
    ) -> None:
        self._session = session
        self._engine = engine
        self._options = options or get_settings().transcription_options()
        self._storage = storage
        self._lease_factory = lease_factory or NoopHeavyModelLeaseFactory()
        self._snapshotter = snapshotter

    def input_fingerprint(self, source: SourceVideo) -> str:
        artifact = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        if artifact is None:
            return ""
        return self._options.fingerprint(artifact.content_hash)

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        artifact = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        if artifact is None:
            raise StageExecutionError("speech-analysis audio is missing")
        fingerprint = self._options.fingerprint(artifact.content_hash)
        existing = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if existing is not None and existing.input_fingerprint == fingerprint and not force:
            return StageExecutionResult(existing.input_fingerprint, existing)
        audio_path = Path(artifact.output_path)
        if not audio_path.is_absolute():
            storage = self._storage or StorageService(get_settings().storage_root)
            audio_path = storage.resolve(StorageCategory.SOURCES, audio_path)
        started_at = monotonic()
        self._emit_snapshot("before_load")
        cancel_event = threading.Event()
        with self._lease_factory.acquire(
            purpose="whisper", on_ownership_lost=cancel_event.set
        ) as heavy_lease:
            result = self._engine.transcribe(audio_path, self._options, cancel_event=cancel_event)
            if heavy_lease.ownership_lost:
                raise HeavyModelLeaseBusy(
                    "heavy-model lease was lost during Whisper work; retry the stage"
                )
        self._emit_snapshot("after_transcribe", child_peak_rss=self._engine.last_child_peak_rss())
        transcript = existing or Transcript(source_video_id=source.id)
        self._apply(transcript, result, fingerprint, monotonic() - started_at)
        self._emit_snapshot("after_cleanup")
        transcript.transcription_revision = (transcript.transcription_revision or 0) + 1
        if existing is None:
            self._session.add(transcript)
        self._session.commit()
        self._session.refresh(transcript)
        return StageExecutionResult(
            canonical_fingerprint(
                "transcription-output",
                "1",
                {
                    "raw_text": transcript.raw_text,
                    "segments": transcript.segments,
                    "word_segments": transcript.word_segments,
                    "revision": transcript.transcription_revision,
                },
            ),
            transcript,
        )

    def _emit_snapshot(self, label: str, child_peak_rss: int | None = None) -> None:
        snapshot = self._snapshotter()
        peak_reported = child_peak_rss if child_peak_rss is not None else "UNKNOWN"
        _logger.info(
            "transcription_memory_snapshot",
            extra={
                "snapshot": label,
                "effective_capacity": snapshot.effective_capacity,
                "process_rss": snapshot.process_rss,
                "linux_available": snapshot.linux_available,
                "cgroup_current": snapshot.cgroup_current,
                "cgroup_peak": snapshot.cgroup_peak,
                "child_peak_rss": peak_reported,
            },
        )

    def _apply(
        self,
        transcript: Transcript,
        result: TranscriptionResult,
        fingerprint: str,
        processing_duration: float,
    ) -> None:
        transcript.language = result.language
        transcript.detected_language_probability = result.language_probability
        transcript.whisper_model = self._options.model
        transcript.transcription_options = {
            "model": self._options.model,
            "device": self._options.device,
            "compute_type": self._options.compute_type,
            "beam_size": self._options.beam_size,
            "language": self._options.language,
            "word_timestamps": self._options.word_timestamps,
            "temperature": self._options.temperature,
            "condition_on_previous_text": self._options.condition_on_previous_text,
            "vad_filter": self._options.vad_filter,
            "initial_prompt": self._options.initial_prompt,
            "hotwords": self._options.hotwords,
        }
        transcript.input_fingerprint = fingerprint
        transcript.raw_text = result.raw_text
        transcript.normalized_text = result.raw_text
        transcript.corrected_text = result.raw_text
        transcript.contextual_reconstructed_text = ""
        transcript.final_text = result.raw_text
        transcript.raw_transcript_confidence = _raw_transcript_confidence(result.segments)
        transcript.correction_confidence = 0.0
        transcript.corrected_segment_ratio = 0.0
        transcript.uncertain_segment_ratio = 1.0 if result.segments else 0.0
        transcript.correction_method = "pending"
        transcript.correction_version = "pending"
        transcript.reconstruction_fingerprint = ""
        transcript.reconstruction_confidence = 0.0
        transcript.reconstructed_segment_ratio = 0.0
        transcript.reconstruction_method = "pending"
        transcript.reconstruction_version = "pending"
        transcript.reconstruction_processing_duration = None
        transcript.reconstruction_metadata = {}
        transcript.segments = result.segments
        transcript.word_segments = result.word_segments
        transcript.duration = result.duration
        transcript.processing_duration = processing_duration


class IngestExecutor:
    """Mark a source accepted by the API as ready for its media probe."""

    def __init__(self, url_adapter: YtDlpAdapter | None = None) -> None:
        settings = get_settings()
        self._url_adapter = url_adapter or YtDlpAdapter(
            StorageService(settings.storage_root), egress_proxy=settings.url_egress_proxy
        )

    def input_fingerprint(self, source: SourceVideo) -> str:
        return canonical_fingerprint("ingest-input", "1", {"source_uri": source.source_uri or ""})

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        if not source.source_uri:
            raise StageExecutionError("source URI is missing")
        if source.source_uri.startswith(("http://", "https://")):
            acquired = self._url_adapter.acquire(source.id, source.source_uri)
            source.source_uri = str(acquired.path)
            source.original_filename = acquired.original_filename
        return StageExecutionResult(
            canonical_fingerprint(
                "ingest-output",
                "1",
                {
                    "source_uri": source.source_uri or "",
                    "original_filename": source.original_filename or "",
                },
            ),
            source,
        )


class ProbeExecutor:
    """Validate a local ingested media file with the safe ffprobe adapter."""

    def __init__(self, probe: FFprobe) -> None:
        self._probe = probe

    def input_fingerprint(self, source: SourceVideo) -> str:
        return canonical_fingerprint(
            "probe-input",
            "1",
            {"source_uri": source.source_uri or "", "content_hash": source.content_hash or ""},
        )

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        source_path = Path(source.source_uri)
        if not source_path.is_file():
            raise StageExecutionError("source media file is unavailable for probing")
        try:
            metadata = self._probe.probe(source_path)
            return StageExecutionResult(
                canonical_fingerprint("probe-output", "1", {"metadata": asdict(metadata)}),
                metadata,
            )
        except Exception as error:
            raise StageExecutionError("ffprobe failed to validate source media") from error


class AudioExtractionExecutor:
    """Prepare the cached WAV before local transcription."""

    def __init__(self, extractor: AudioExtractor) -> None:
        self._extractor = extractor

    def input_fingerprint(self, source: SourceVideo) -> str:
        return canonical_fingerprint(
            "audio-extraction-input",
            "1",
            {"source_uri": source.source_uri or "", "content_hash": source.content_hash or ""},
        )

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        artifact = self._extractor.extract(source)
        return StageExecutionResult(
            canonical_fingerprint(
                "audio-extraction-output",
                "1",
                {
                    "content_hash": artifact.content_hash,
                    "output_path": artifact.output_path,
                },
            ),
            artifact,
        )


class TranscriptNormalizationExecutor:
    """Normalize a persisted transcript without rewriting its raw ASR evidence."""

    def __init__(self, *, session: Session, corrector: ContextualCorrector | None = None) -> None:
        self._session = session
        self._corrector = corrector or ContextualCorrector.from_default_lexicon()

    def input_fingerprint(self, source: SourceVideo) -> str:
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if transcript is None:
            return ""
        return canonical_fingerprint(
            "normalization-input",
            "1",
            {
                "transcription_fingerprint": transcript.input_fingerprint,
                "transcription_revision": transcript.transcription_revision,
                "correction_version": "egyptian-ar-v1",
            },
        )

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if transcript is None:
            raise StageExecutionError("transcript is missing")
        previous_overrides = {
            index: (
                str(segment.get("raw_text", segment.get("text", ""))),
                segment.get("operator_text"),
            )
            for index, segment in enumerate(transcript.segments)
            if segment.get("operator_text")
        }
        corrections = self._corrector.correct(transcript.segments)
        normalized_segments: list[dict[str, object]] = []
        for segment, correction in zip(transcript.segments, corrections, strict=True):
            previous = previous_overrides.get(correction.segment_index)
            operator_text = (
                str(previous[1])
                if previous is not None and previous[0] == correction.raw_text
                else None
            )
            final_text = operator_text or correction.corrected_text
            normalized_segments.append(
                {
                    **segment,
                    "raw_text": correction.raw_text,
                    "corrected_text": correction.corrected_text,
                    "correction_applied": correction.applied,
                    "correction_confidence": correction.confidence,
                    "correction_method": correction.method,
                    "correction_version": correction.version,
                    "correction_changes": correction.changes,
                    "operator_text": operator_text,
                    "final_text": final_text,
                    "normalized_text": normalize_transcript(final_text),
                }
            )
        transcript.segments = normalized_segments
        transcript.corrected_text = " ".join(
            str(segment["corrected_text"]) for segment in normalized_segments
        ).strip()
        transcript.final_text = " ".join(
            str(segment["final_text"]) for segment in normalized_segments
        ).strip()
        transcript.normalized_text = normalize_transcript(transcript.final_text)
        transcript.normalization_fingerprint = canonical_fingerprint(
            "normalization-output",
            "1",
            {
                "segments": normalized_segments,
                "transcription_revision": transcript.transcription_revision,
            },
        )
        total_segments = len(normalized_segments)
        applied = [segment for segment in normalized_segments if segment["correction_applied"]]
        uncertain = [
            segment
            for segment in normalized_segments
            if segment["correction_method"] == "unchanged"
        ]
        transcript.raw_transcript_confidence = _raw_transcript_confidence(normalized_segments)
        transcript.correction_confidence = (
            sum(float(segment["correction_confidence"]) for segment in applied) / len(applied)
            if applied
            else 0.0
        )
        transcript.corrected_segment_ratio = (
            len(applied) / total_segments if total_segments else 0.0
        )
        transcript.uncertain_segment_ratio = (
            len(uncertain) / total_segments if total_segments else 0.0
        )
        transcript.correction_method = (
            "mixed"
            if len({str(segment["correction_method"]) for segment in normalized_segments}) > 1
            else str(normalized_segments[0]["correction_method"])
            if normalized_segments
            else "unchanged"
        )
        transcript.correction_version = (
            str(normalized_segments[0]["correction_version"])
            if normalized_segments
            else "egyptian-ar-v1"
        )
        transcript.contextual_reconstructed_text = transcript.corrected_text
        transcript.reconstruction_fingerprint = ""
        transcript.reconstruction_confidence = 0.0
        transcript.reconstructed_segment_ratio = 0.0
        transcript.reconstruction_method = "pending"
        transcript.reconstruction_version = "pending"
        transcript.reconstruction_processing_duration = None
        transcript.reconstruction_metadata = {}
        self._session.execute(
            delete(TranscriptChunk).where(TranscriptChunk.transcript_id == transcript.id)
        )
        self._session.add_all(
            TranscriptChunk(
                transcript_id=transcript.id,
                sequence=sequence,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text=chunk.text,
                segment_indexes=chunk.segment_indexes,
                preceding_context=chunk.preceding_context,
                following_context=chunk.following_context,
            )
            for sequence, chunk in enumerate(build_chunks(transcript.segments, ChunkConfig()))
        )
        self._session.commit()
        self._session.refresh(transcript)
        return StageExecutionResult(transcript.normalization_fingerprint, transcript)


class _Lease(Protocol):
    def __enter__(self) -> "_Lease": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool: ...
    @property
    def ownership_lost(self) -> bool: ...
    def retain(self) -> None: ...


class _LeaseFactory(Protocol):
    def acquire(self, *, purpose: str, on_ownership_lost: object | None = None) -> _Lease: ...


class _LeaseBoundReconstructionProvider:
    """Wrap the local reconstruction provider with a lazy heavy-model lease.

    The heavy-model/Ollama lease is acquired only when actual local inference or
    model release happens. NO_LLM, GEMINI_ONLY, and direct-Gemini work never
    acquire it; a direct-Gemini call runs before any optional local fallback
    lease. The lease is held lazily from the first local call through model
    release, preserving the existing Whisper/Ollama serialization guarantee.
    """

    def __init__(self, inner: ReconstructionProvider, lease_factory: _LeaseFactory) -> None:
        self._inner = inner
        self._lease_factory = lease_factory
        self._lease: _Lease | None = None
        self.provider_name = getattr(inner, "provider_name", "ollama")
        self.model = getattr(inner, "model", None)

    def _enter_lease(self) -> _Lease:
        if self._lease is None:
            self._lease = self._lease_factory.acquire(purpose="ollama")
            self._lease.__enter__()
        return self._lease

    def health(self) -> ProviderHealth:
        return self._inner.health()

    def runtime_identity(self) -> dict[str, object]:
        return self._inner.runtime_identity()

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self._inner.refresh_runtime_identity()

    def last_request_sizes(self) -> tuple[RequestSizeDiagnostics, ...]:
        lookup = getattr(self._inner, "last_request_sizes", None)
        return lookup() if lookup is not None else ()

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        lease = self._enter_lease()
        result = self._inner.reconstruct_segments(requests)
        if lease.ownership_lost:
            raise HeavyModelLeaseBusy(
                "heavy-model lease was lost during reconstruction; retry the stage"
            )
        return result

    def release(self) -> object:
        if self._lease is None:
            return None
        lease = self._lease
        self._lease = None
        try:
            outcome = cast(object, self._inner.release())
            if isinstance(outcome, UnloadOutcome) and outcome.warning is not None:
                lease.retain()
            if lease.ownership_lost:
                raise HeavyModelLeaseBusy(
                    "heavy-model lease was lost during reconstruction; retry the stage"
                )
            return outcome
        finally:
            lease.__exit__(None, None, None)


class ContextualReconstructionExecutor:
    """Persist bounded Stage 2.7 derivations without rewriting prior transcript evidence."""

    def __init__(
        self,
        *,
        session: Session,
        reconstructor: ContextualReconstructor,
        lease_factory: HeavyModelLeaseFactory | NoopHeavyModelLeaseFactory | None = None,
    ) -> None:
        self._session = session
        self._reconstructor = reconstructor
        self._lease_factory = lease_factory or NoopHeavyModelLeaseFactory()

    def input_fingerprint(self, source: SourceVideo) -> str:
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if transcript is None:
            return ""
        return canonical_fingerprint(
            "reconstruction-input",
            "1",
            {
                "normalization_fingerprint": transcript.normalization_fingerprint,
                "transcription_revision": transcript.transcription_revision,
                "correction_version": transcript.correction_version,
                "runtime_identity": self._reconstructor.refresh_runtime_identity(),
            },
        )

    def skip_is_allowed(self, source: SourceVideo) -> bool:
        """Whether a matching succeeded reconstruction run may be skipped.

        A fully cache-eligible run is reusable and may be skipped normally. A
        degraded run (unresolved/provider-failure/rate-limit/local-ceiling
        targets, so ``cache_eligible`` is false) must re-enter the executor on a
        later normal request so it can retry only its unfinished targets after
        provider recovery.
        """

        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if transcript is None:
            return False
        return transcript.reconstruction_metadata.get("cache_eligible") is True

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        reconstructor = self._reconstructor
        local = getattr(reconstructor, "_provider", None)
        if local is not None:
            reconstructor = reconstructor.with_local_provider(
                _LeaseBoundReconstructionProvider(local, self._lease_factory)
            )
        try:
            return self._execute_with(reconstructor, source, force=force)
        finally:
            self._release_reconstructor(reconstructor)

    def _release_reconstructor(self, reconstructor: ContextualReconstructor) -> None:
        """Idempotently release owned provider resources on every executor exit.

        This covers executor early returns (fresh cache hit and cached-output
        preservation during an outage) that happen before
        ``ContextualReconstructor.reconstruct()`` runs its own cleanup, so a
        fresh cache-hit worker scrubs the Gemini key and closes owned clients
        without ever performing a generation or network call. Unowned injected
        clients are never closed.
        """

        provider = getattr(reconstructor, "_provider", None)
        if provider is not None:
            try:
                provider.release()
            except Exception:
                pass
        gemini = getattr(reconstructor, "_gemini", None)
        if gemini is not None:
            try:
                gemini.release()
            except Exception:
                pass

    def _execute_with(
        self,
        reconstructor: ContextualReconstructor,
        source: SourceVideo,
        *,
        force: bool = False,
    ) -> StageExecutionResult:
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if transcript is None:
            raise StageExecutionError("normalized transcript is missing")
        started_at = monotonic()
        stored_eligible = transcript.reconstruction_metadata.get("cache_eligible") is True
        if transcript.reconstruction_fingerprint and not force:
            check = reconstructor.output_fingerprint(
                transcript.segments,
                language=transcript.language,
                transcription_fingerprint=transcript.input_fingerprint,
                correction_version=transcript.correction_version,
            )
            if transcript.reconstruction_fingerprint == check.fingerprint and stored_eligible:
                return StageExecutionResult(check.fingerprint, transcript)
            if stored_eligible and not check.resolved:
                # Transient provider outage: identity cannot be confirmed. Keep the
                # accepted output instead of overwriting it with degraded fallback.
                return StageExecutionResult(check.fingerprint, transcript)
        reconstructor.refresh_runtime_identity()
        identity = reconstructor.runtime_identity()
        reconstructor = reconstructor.with_orchestration(
            is_cancelled=lambda: self._job_cancelled(source.id),
            checkpoint=lambda results, progress: self._persist_checkpoint(
                transcript, results, progress, identity
            ),
        )
        resolved = self._reusable_targets(
            transcript,
            identity,
            language=transcript.language,
            transcription_fingerprint=transcript.input_fingerprint,
            correction_version=transcript.correction_version,
        )
        result = reconstructor.reconstruct(
            transcript.segments,
            language=transcript.language,
            transcription_fingerprint=transcript.input_fingerprint,
            correction_version=transcript.correction_version,
            resolved=resolved,
        )
        if (
            transcript.reconstruction_fingerprint == result.fingerprint
            and stored_eligible
            and not force
        ):
            # A previously degraded run that is now successful shares the stable
            # fingerprint but must still be persisted so it becomes cache-eligible.
            return StageExecutionResult(result.fingerprint, transcript)

        persisted_segments: list[dict[str, object]] = []
        statuses: list[ReconstructionStatus] = []
        for segment, reconstruction in zip(transcript.segments, result.segments, strict=True):
            status = _segment_reconstruction_status(segment, reconstruction)
            statuses.append(status)
            persisted_segments.append(
                self._apply_segment(
                    segment,
                    reconstruction,
                    status,
                    identity,
                    transcript.segments,
                    language=transcript.language,
                    transcription_fingerprint=transcript.input_fingerprint,
                    correction_version=transcript.correction_version,
                )
            )

        transcript.segments = persisted_segments
        transcript.contextual_reconstructed_text = " ".join(
            str(segment["contextual_reconstructed_text"]) for segment in persisted_segments
        ).strip()
        transcript.final_text = " ".join(
            str(segment["final_text"]) for segment in persisted_segments
        ).strip()
        transcript.normalized_text = normalize_transcript(transcript.final_text)
        total = len(result.segments)
        applied = [item for item in result.segments if item.applied]
        flags = sorted({flag.value for item in result.segments for flag in item.quality_flags})
        transcript.reconstruction_fingerprint = result.fingerprint
        transcript.reconstruction_status = aggregate_reconstruction_status(statuses)
        transcript.reconstruction_confidence = (
            sum(item.confidence for item in applied) / len(applied) if applied else 0.0
        )
        transcript.reconstructed_segment_ratio = len(applied) / total if total else 0.0
        transcript.reconstruction_method = (
            "stage2_5_fallback"
            if getattr(self._reconstructor, "_provider", object()) is None
            and getattr(self._reconstructor, "_gemini", object()) is None
            else _reconstruction_method(result.segments, result.metadata)
        )
        transcript.reconstruction_version = "stage2.7-v1"
        transcript.reconstruction_processing_duration = monotonic() - started_at
        status_counts = {status.value: statuses.count(status) for status in set(statuses)}
        runtime_identity = result.metadata.get("runtime_identity")
        routing_counts = result.metadata.get("routing_counts")
        gemini_usage = result.metadata.get("gemini_usage")
        gemini_runtime = (
            runtime_identity.get("gemini")
            if isinstance(runtime_identity, dict)
            and isinstance(runtime_identity.get("gemini"), dict)
            else None
        )
        metadata = {
            "segments": total,
            "applied_segments": len(applied),
            "routed_segments": sum(
                1 for item in result.segments if getattr(item, "routing_score", None) is not None
            ),
            "unresolved_segments": sum(
                1 for status in statuses if status is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
            ),
            "batch_count": 1 if total else 0,
            "status_counts": status_counts,
            "quality_flags": flags,
            "provider_availability": (
                "UNAVAILABLE"
                if any(status is ReconstructionStatus.PROVIDER_UNAVAILABLE for status in statuses)
                else "AVAILABLE"
            ),
            "model": (
                runtime_identity.get("model", transcript.reconstruction_method)
                if isinstance(runtime_identity, dict)
                else transcript.reconstruction_method
            ),
            "model_digest": (
                runtime_identity.get("digest") if isinstance(runtime_identity, dict) else None
            ),
            "algorithm_versions": {"reconstruction": "stage2.7-v1"},
        }
        if isinstance(routing_counts, dict):
            metadata["routing_counts"] = routing_counts
        if isinstance(gemini_usage, dict):
            metadata["gemini_usage"] = gemini_usage
        if isinstance(gemini_runtime, dict):
            metadata["gemini_model"] = gemini_runtime.get("model")
            metadata["gemini_model_digest"] = gemini_runtime.get("digest")
        metadata["cache_eligible"] = result.metadata.get("cache_eligible") is True
        metadata.update(result.metadata)
        transcript.reconstruction_metadata = metadata
        self._session.execute(
            delete(TranscriptChunk).where(TranscriptChunk.transcript_id == transcript.id)
        )
        self._session.add_all(
            TranscriptChunk(
                transcript_id=transcript.id,
                sequence=sequence,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text=chunk.text,
                segment_indexes=chunk.segment_indexes,
                preceding_context=chunk.preceding_context,
                following_context=chunk.following_context,
            )
            for sequence, chunk in enumerate(build_chunks(persisted_segments, ChunkConfig()))
        )
        self._session.commit()
        self._session.refresh(transcript)
        return StageExecutionResult(result.fingerprint, transcript)

    def _job_cancelled(self, source_id: object) -> bool:
        """Cooperative cancellation: the latest reconstruction job was cancelled."""

        job = self._session.scalar(
            select(ProcessingJob)
            .where(
                ProcessingJob.source_video_id == source_id,
                ProcessingJob.kind == JobKind.RECONSTRUCTION,
            )
            .order_by(ProcessingJob.created_at.desc())
        )
        return job is not None and job.status is JobStatus.CANCELLED

    def _reusable_targets(
        self,
        transcript: Transcript,
        identity: dict[str, object],
        *,
        language: str | None,
        transcription_fingerprint: str,
        correction_version: str,
    ) -> dict[int, SegmentReconstruction]:
        """Reuse accepted per-target work whose dependency fingerprint is unchanged.

        A restart after cancellation or a late provider failure keeps the
        already-accepted Gemini/local results without re-calling a provider; only
        eligible unresolved work is reconsidered.
        """

        resolved: dict[int, SegmentReconstruction] = {}
        for index, segment in enumerate(transcript.segments):
            if segment.get("operator_text"):
                continue
            stored_fingerprint = segment.get("reconstruction_target_fingerprint")
            if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
                continue
            if segment.get("reconstruction_cache_eligible") is not True:
                continue
            current_fingerprint = reconstruction_target_fingerprint(
                provider_identity=identity,
                segments=transcript.segments,
                target_index=index,
                language=language,
                transcription_fingerprint=transcription_fingerprint,
                correction_version=correction_version,
            )
            if stored_fingerprint != current_fingerprint:
                continue
            rebuilt = _segment_from_stored(index, segment)
            if rebuilt is not None:
                resolved[index] = rebuilt
        return resolved

    def _persist_checkpoint(
        self,
        transcript: Transcript,
        results: dict[int, SegmentReconstruction],
        progress: dict[str, object],
        identity: dict[str, object],
    ) -> None:
        """Durably persist completed per-target results mid-run.

        Checkpointed batches survive cancellation and worker restarts so a
        restart does not discard every preceding expensive result. The segment
        list is rebuilt and reassigned so SQLAlchemy detects the JSON mutation.
        """

        if not results:
            return
        segments = list(transcript.segments)
        for index, reconstruction in results.items():
            if not 0 <= index < len(segments):
                continue
            status = _segment_reconstruction_status(segments[index], reconstruction)
            segments[index] = self._apply_segment(
                segments[index],
                reconstruction,
                status,
                identity,
                segments,
                language=transcript.language,
                transcription_fingerprint=transcript.input_fingerprint,
                correction_version=transcript.correction_version,
            )
        transcript.segments = segments
        metadata = dict(transcript.reconstruction_metadata)
        metadata["partial"] = True
        metadata["cache_eligible"] = False
        metadata["progress"] = progress
        if isinstance(progress.get("routing_counts"), dict):
            metadata["routing_counts"] = progress["routing_counts"]
        transcript.reconstruction_metadata = metadata
        self._session.commit()

    def _apply_segment(
        self,
        segment: dict[str, object],
        reconstruction: SegmentReconstruction,
        status: ReconstructionStatus,
        identity: dict[str, object],
        segments: list[dict[str, object]],
        *,
        language: str | None,
        transcription_fingerprint: str,
        correction_version: str,
    ) -> dict[str, object]:
        """Merge one reconstruction outcome into its persisted segment record."""

        raw = str(segment.get("raw_text", segment.get("text", "")))
        corrected = str(segment.get("corrected_text", raw))
        operator_text = segment.get("operator_text")
        operator = str(operator_text) if operator_text else None
        final_text = select_final_text(
            operator_text=operator,
            reconstructed=reconstruction.contextual_reconstructed_text,
            reconstruction_applied=reconstruction.applied,
            level=reconstruction.confidence_level,
            corrected=corrected,
            raw=raw,
        )
        target_fingerprint = reconstruction_target_fingerprint(
            provider_identity=identity,
            segments=segments,
            target_index=reconstruction.segment_index,
            language=language,
            transcription_fingerprint=transcription_fingerprint,
            correction_version=correction_version,
        )
        return {
            **segment,
            "contextual_reconstructed_text": reconstruction.contextual_reconstructed_text,
            "reconstruction_candidate_text": reconstruction.candidate_text,
            "reconstruction_applied": reconstruction.applied,
            "reconstruction_confidence": reconstruction.confidence,
            "reconstruction_confidence_level": reconstruction.confidence_level.value,
            "reconstruction_quality_flags": [flag.value for flag in reconstruction.quality_flags],
            "routing_score": reconstruction.routing_score,
            "routing_reasons": list(reconstruction.routing_reasons),
            "focus_spans": [
                {
                    "word": word.text,
                    "start": word.start,
                    "end": word.end,
                    "probability": word.probability,
                }
                for word in reconstruction.focus_spans
            ],
            "reconstruction_status": status.value,
            "reconstruction_method": reconstruction.reconstruction_method,
            "final_text": final_text,
            "normalized_text": normalize_transcript(final_text),
            "reconstruction_route": reconstruction.route,
            "routing_evidence": list(reconstruction.routing_evidence),
            "local_attempted": reconstruction.local_attempted,
            "local_result_state": reconstruction.local_result_state,
            "gemini_attempted": reconstruction.gemini_attempted,
            "gemini_result_state": reconstruction.gemini_result_state,
            "final_provider": reconstruction.final_provider,
            "escalation_reason": reconstruction.escalation_reason,
            "near_acceptance": reconstruction.near_acceptance,
            "reconstruction_target_fingerprint": target_fingerprint,
            "reconstruction_cache_eligible": _target_cache_eligible(reconstruction),
        }


def _reconstruction_method(
    segments: tuple[SegmentReconstruction, ...], metadata: dict[str, object] | None = None
) -> str:
    if metadata and isinstance(metadata.get("reconstruction_method"), str):
        return str(metadata["reconstruction_method"])
    methods = {item.reconstruction_method for item in segments if item.reconstruction_method}
    if len(methods) == 1:
        return next(iter(methods))
    if all(getattr(segment, "candidate_text", None) is None for segment in segments):
        return "stage2_5_fallback"
    if any(
        "RECONSTRUCTION_PROVIDER_ERROR"
        in {flag.value for flag in getattr(segment, "quality_flags", ())}
        for segment in segments
    ):
        return "provider_fallback"
    return "contextual_reconstruction"


def _segment_reconstruction_status(
    segment: dict[str, object], reconstruction: SegmentReconstruction
) -> ReconstructionStatus:
    operator_text = segment.get("operator_text")
    if operator_text:
        return ReconstructionStatus.MANUAL_OVERRIDE
    value = getattr(reconstruction, "status", None)
    if isinstance(value, ReconstructionStatus):
        return value
    if isinstance(value, str):
        try:
            return ReconstructionStatus(value)
        except ValueError:
            pass
    if any(flag.value == "RECONSTRUCTION_PROVIDER_ERROR" for flag in reconstruction.quality_flags):
        return ReconstructionStatus.PROVIDER_UNAVAILABLE
    if reconstruction.applied:
        return ReconstructionStatus.APPLIED
    if reconstruction.confidence_level.value == "LOW":
        return ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
    return ReconstructionStatus.UNCHANGED_HIGH_CONFIDENCE


def _target_cache_eligible(reconstruction: SegmentReconstruction) -> bool:
    """Per-target reuse eligibility: a terminal outcome that needs no provider.

    Provider-accepted HIGH results are reusable without a provider call; NO_LLM
    and manual outcomes never call providers anyway and are recorded as eligible
    so their stored records are self-consistent.
    """

    if reconstruction.route in {"MANUAL", "NO_LLM"}:
        return True
    return reconstruction.applied and reconstruction.confidence_level is ConfidenceLevel.HIGH


def _segment_from_stored(index: int, segment: Mapping[str, object]) -> SegmentReconstruction | None:
    """Rebuild a reusable SegmentReconstruction from persisted checkpoint fields."""

    route = str(segment.get("reconstruction_route") or "")
    if route not in {"LOCAL", "GEMINI_DIRECT", "LOCAL_THEN_GEMINI"}:
        return None
    if segment.get("reconstruction_applied") is not True:
        return None
    if segment.get("reconstruction_confidence_level") != ConfidenceLevel.HIGH.value:
        return None
    status = _parse_reconstruction_status(segment.get("reconstruction_status"))
    if status is None or status in {
        ReconstructionStatus.PROVIDER_UNAVAILABLE,
        ReconstructionStatus.FAILED,
        ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED,
    }:
        return None
    raw = str(segment.get("raw_text", segment.get("text", "")))
    corrected = str(segment.get("corrected_text", raw))
    reconstructed = str(segment.get("contextual_reconstructed_text") or corrected)
    confidence_value = segment.get("reconstruction_confidence")
    confidence = float(confidence_value) if isinstance(confidence_value, int | float) else 0.0
    flags_list: list[QualityFlag] = []
    for flag in _as_list(segment.get("reconstruction_quality_flags")):
        if isinstance(flag, str):
            try:
                flags_list.append(QualityFlag(flag))
            except ValueError:
                continue
    flags = tuple(flags_list)
    routing_reasons = tuple(
        str(reason) for reason in _as_list(segment.get("routing_reasons")) if reason
    )
    routing_evidence = tuple(
        str(evidence) for evidence in _as_list(segment.get("routing_evidence")) if evidence
    )
    focus = tuple(
        WordEvidence(
            str(span.get("word", "")),
            _number(span.get("start")),
            _number(span.get("end")),
            _number(span.get("probability")),
        )
        for span in _as_list(segment.get("focus_spans"))
        if isinstance(span, Mapping)
    )
    routing_score = segment.get("routing_score")
    return SegmentReconstruction(
        index,
        raw,
        corrected,
        reconstructed,
        segment.get("reconstruction_candidate_text"),
        True,
        confidence,
        ConfidenceLevel.HIGH,
        flags,
        status,
        routing_score=float(routing_score) if isinstance(routing_score, int | float) else None,
        routing_reasons=routing_reasons,
        focus_spans=focus,
        validated_changes=tuple(
            change
            for change in _as_list(segment.get("validated_changes"))
            if isinstance(change, Mapping)
        ),
        reconstruction_method=segment.get("reconstruction_method"),
        candidate_id=segment.get("candidate_id"),
        decision_reason=segment.get("decision_reason"),
        explanation=str(segment.get("explanation") or ""),
        route=route,
        routing_evidence=routing_evidence,
        local_attempted=bool(segment.get("local_attempted")),
        local_result_state=segment.get("local_result_state"),
        gemini_attempted=bool(segment.get("gemini_attempted")),
        gemini_result_state=segment.get("gemini_result_state"),
        final_provider=segment.get("final_provider"),
        escalation_reason=segment.get("escalation_reason"),
        near_acceptance=bool(segment.get("near_acceptance")),
    )


def _as_list(value: object) -> list[object]:
    return [item for item in value] if isinstance(value, list) else []


def _parse_reconstruction_status(value: object) -> ReconstructionStatus | None:
    if isinstance(value, ReconstructionStatus):
        return value
    if isinstance(value, str):
        try:
            return ReconstructionStatus(value)
        except ValueError:
            return None
    return None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _raw_transcript_confidence(segments: list[dict[str, object]]) -> float:
    logprobs: list[float] = []
    for segment in segments:
        value = segment.get("avg_logprob")
        if isinstance(value, int | float):
            logprobs.append(float(value))
    return max(0.0, min(1.0, 1.0 + sum(logprobs) / len(logprobs))) if logprobs else 0.0


SilenceCommandRunner = Callable[[list[str]], str]


def _run_silence_command(args: list[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stderr


class AudioAnalysisExecutor:
    """Persist silence and speech-density signals, then refresh source quality."""

    def __init__(
        self,
        *,
        session: Session,
        storage: StorageService,
        ffmpeg_binary: str = "ffmpeg",
        command_runner: SilenceCommandRunner = _run_silence_command,
    ) -> None:
        self._session = session
        self._storage = storage
        self._ffmpeg_binary = ffmpeg_binary
        self._command_runner = command_runner

    def input_fingerprint(self, source: SourceVideo) -> str:
        artifact = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if artifact is None or transcript is None:
            return ""
        return canonical_fingerprint(
            "audio-analysis-input",
            "1",
            {
                "audio_hash": artifact.content_hash,
                "transcription_revision": transcript.transcription_revision,
            },
        )

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        artifact = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if artifact is None or transcript is None:
            raise StageExecutionError("audio artifact and normalized transcript are required")
        existing = self._session.scalar(
            select(AudioAnalysis).where(AudioAnalysis.source_video_id == source.id)
        )
        current_input = self.input_fingerprint(source)
        if existing is not None and existing.input_fingerprint == current_input and not force:
            quality = source.quality_assessment
            if quality is not None and quality.input_fingerprint == quality_input_fingerprint(
                transcript, existing
            ):
                return StageExecutionResult(current_input, existing)
            duration = max(artifact.duration, transcript.duration)
            existing.speech_rate = (
                len(transcript.word_segments) * 60.0 / duration if duration else 0.0
            )
            self._session.commit()
            self._session.refresh(existing)
            assess_source(self._session, source, transcript, existing)
            return StageExecutionResult(current_input, existing)
        if existing is not None and existing.audio_hash == artifact.content_hash and not force:
            duration = max(artifact.duration, transcript.duration)
            existing.input_fingerprint = current_input
            existing.speech_rate = (
                len(transcript.word_segments) * 60.0 / duration if duration else 0.0
            )
            self._session.commit()
            self._session.refresh(existing)
            assess_source(self._session, source, transcript, existing)
            return StageExecutionResult(current_input, existing)
        audio_path = self._storage.resolve(StorageCategory.SOURCES, artifact.output_path)
        args = [
            self._ffmpeg_binary,
            "-i",
            str(audio_path),
            "-af",
            "silencedetect=n=-35dB:d=0.4",
            "-f",
            "null",
            "-",
        ]
        try:
            intervals = parse_silencedetect(self._command_runner(args))
        except (OSError, subprocess.CalledProcessError) as err:
            raise StageExecutionError("ffmpeg audio analysis failed") from err
        duration = max(artifact.duration, transcript.duration)
        ratio = silence_ratio(intervals, duration)
        analysis = existing or AudioAnalysis(source_video_id=source.id)
        analysis.audio_hash = artifact.content_hash
        analysis.input_fingerprint = current_input
        analysis.silence_intervals = [interval.__dict__ for interval in intervals]
        analysis.features = windowed_rms(audio_path)
        if not analysis.features:
            analysis.features = [{"start": 0.0, "end": duration, "rms": 0.0}]
        analysis.silence_ratio = ratio
        analysis.speech_density = 1.0 - ratio
        analysis.speech_rate = len(transcript.word_segments) * 60.0 / duration if duration else 0.0
        if existing is None:
            self._session.add(analysis)
        self._session.commit()
        self._session.refresh(analysis)
        assess_source(self._session, source, transcript, analysis)
        return StageExecutionResult(
            canonical_fingerprint(
                "audio-analysis-output",
                "1",
                {
                    "audio_hash": analysis.audio_hash,
                    "features": analysis.features,
                    "silence_intervals": analysis.silence_intervals,
                    "transcription_revision": transcript.transcription_revision,
                },
            ),
            analysis,
        )
