"""Operator-only admission of a hash-pinned local decoder observation.

This module admits one already-captured, identity-verified local faster-whisper
observation as a raw Stage 3.5 FINAL_CLIP checkpoint. It never loads a model,
constructs a provider, dispatches a job, or derives executable clip text. A
normal subsequent refinement job can reuse the admitted raw observations through
the existing rejected-observation path and derive the validated clip view.

The input bundle is an explicitly operator-trusted, hash-pinned legacy
observation. The service validates every pinned hash, the current immutable
source/transcript/operator identity, the current decoder/context contract, and
the exact derived clip view before any mutation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.candidates.registration import (
    CandidateRegistrationError,
    operator_identity_is_current,
    operator_selection_marker,
)
from app.core.enums import (
    EvidenceKind,
    EvidenceState,
    JobKind,
    JobStatus,
    RefinementPriority,
    RefinementStatus,
)
from app.models import (
    AudioArtifact,
    CandidateRefinement,
    ClipCandidate,
    ProcessingJob,
    SourceVideo,
    Transcript,
)
from app.refinement.asr import (
    convert_transcription_result,
    effective_transcription_options,
)
from app.refinement.audio_window import CandidateAudioWindowService
from app.refinement.fingerprints import component_fingerprint
from app.refinement.policy import DEFAULT_CONFIG, Stage35Config
from app.refinement.scope import FinalClipView, derive_final_clip_view
from app.refinement.service import (
    CandidateRefinementService,
    candidate_region_text,
    source_evidenced_context_terms,
)
from app.refinement.types import TargetASRResult
from app.services.hashing import sha256_file
from app.services.storage import StorageCategory, StorageService
from app.transcription.engine import TranscriptionResult
from app.transcription.service import TranscriptionOptions

_BOUND_TOLERANCE = 1e-6
_MAX_MANIFEST_BYTES = 64 * 1024
_RECOVERY_PROVENANCE_KIND = "local-observation-recovery-v1"


class LocalObservationRecoveryError(ValueError):
    """A recovery bundle or current row failed validation before mutation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class LocalObservationExpectation:
    """The operator-supplied pinned identity for one recovery bundle."""

    bundle_relative_path: str
    source_id: UUID
    candidate_id: UUID
    refinement_id: UUID
    original_job_id: UUID
    coarse_start: float
    coarse_end: float
    context_start: float
    context_end: float
    refined_start: float
    refined_end: float
    candidate_key: str
    raw_span_fingerprint: str
    source_sha256: str
    audio_sha256: str
    decoder_result_sha256: str
    result_sha256: str
    prior_evidence_sha256: str
    approved_text_sha256: str
    approved_token_count: int
    raw_word_record_count: int = 107
    expected_rejection_reason: str = "nonpositive word duration"
    context_only_invalid_record_indexes: tuple[int, ...] = (105,)

    def as_dict(self) -> dict[str, object]:
        return {
            "bundle_relative_path": self.bundle_relative_path,
            "source_id": str(self.source_id),
            "candidate_id": str(self.candidate_id),
            "refinement_id": str(self.refinement_id),
            "original_job_id": str(self.original_job_id),
            "coarse_start": self.coarse_start,
            "coarse_end": self.coarse_end,
            "context_start": self.context_start,
            "context_end": self.context_end,
            "refined_start": self.refined_start,
            "refined_end": self.refined_end,
            "candidate_key": self.candidate_key,
            "raw_span_fingerprint": self.raw_span_fingerprint,
            "source_sha256": self.source_sha256,
            "audio_sha256": self.audio_sha256,
            "decoder_result_sha256": self.decoder_result_sha256,
            "result_sha256": self.result_sha256,
            "prior_evidence_sha256": self.prior_evidence_sha256,
            "approved_text_sha256": self.approved_text_sha256,
            "approved_token_count": self.approved_token_count,
            "raw_word_record_count": self.raw_word_record_count,
            "expected_rejection_reason": self.expected_rejection_reason,
            "context_only_invalid_record_indexes": list(self.context_only_invalid_record_indexes),
        }


@dataclass(frozen=True)
class RecoveryPreview:
    """Read-only description of the exact checkpoint admission delta."""

    bundle_relative_path: str
    bundle_hashes: dict[str, str]
    source_id: str
    candidate_id: str
    refinement_id: str
    original_job_id: str
    coarse_start: float
    coarse_end: float
    context_start: float
    context_end: float
    refined_start: float
    refined_end: float
    decoder_identity: dict[str, object]
    context_terms: tuple[str, ...]
    reconstructed: dict[str, object]
    derived_view: dict[str, object]
    approved_text_sha256: str
    approved_text_matches: bool
    provenance: dict[str, object]
    proposed_delta: dict[str, object]
    reuse_ready: bool
    reuse_ready_after_apply: bool
    already_admitted: bool
    writes: int = 0
    snapshot_relative_path: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "bundle_relative_path": self.bundle_relative_path,
            "bundle_hashes": dict(self.bundle_hashes),
            "source_id": self.source_id,
            "candidate_id": self.candidate_id,
            "refinement_id": self.refinement_id,
            "original_job_id": self.original_job_id,
            "coarse_start": self.coarse_start,
            "coarse_end": self.coarse_end,
            "context_start": self.context_start,
            "context_end": self.context_end,
            "refined_start": self.refined_start,
            "refined_end": self.refined_end,
            "decoder_identity": dict(self.decoder_identity),
            "context_terms": list(self.context_terms),
            "reconstructed": dict(self.reconstructed),
            "derived_view": dict(self.derived_view),
            "approved_text_sha256": self.approved_text_sha256,
            "approved_text_matches": self.approved_text_matches,
            "provenance": dict(self.provenance),
            "proposed_delta": dict(self.proposed_delta),
            "reuse_ready": self.reuse_ready,
            "reuse_ready_after_apply": self.reuse_ready_after_apply,
            "already_admitted": self.already_admitted,
            "writes": self.writes,
            "snapshot_relative_path": self.snapshot_relative_path,
        }


@dataclass(frozen=True)
class RecoveryApplyResult:
    """Result of an apply attempt; mutations belong to the caller's transaction."""

    applied: bool
    already_admitted: bool
    snapshot_relative_path: str | None
    preview: RecoveryPreview

    def as_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "already_admitted": self.already_admitted,
            "snapshot_relative_path": self.snapshot_relative_path,
            "preview": self.preview.as_dict(),
        }


@dataclass(frozen=True)
class _ValidatedRecovery:
    expectation: LocalObservationExpectation
    bundle_relative_path: str
    bundle_hashes: dict[str, str]
    decoder: dict[str, object]
    result: dict[str, object]
    prior: dict[str, object]
    source: SourceVideo
    transcript: Transcript
    artifact: AudioArtifact
    candidate: ClipCandidate
    refinement: CandidateRefinement
    original_job: ProcessingJob
    marker: dict[str, object]
    target: TargetASRResult
    view: FinalClipView
    audio_component: str
    audio_input_fingerprint: str
    audio_relative_path: str
    context_terms: tuple[str, ...]
    effective_options: TranscriptionOptions
    provenance: dict[str, object]
    proposed_delta: dict[str, object]
    already_admitted: bool
    reuse_ready: bool
    reuse_ready_after_apply: bool


def expectation_from_manifest(payload: Mapping[str, object]) -> LocalObservationExpectation:
    """Parse and validate a bounded operator expectation manifest."""

    if not isinstance(payload, Mapping):
        raise LocalObservationRecoveryError("invalid_manifest", "manifest must be an object")
    indexes = payload.get("context_only_invalid_record_indexes", [105])
    if not isinstance(indexes, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in indexes
    ):
        raise LocalObservationRecoveryError(
            "invalid_manifest", "context_only_invalid_record_indexes must be integers"
        )
    return LocalObservationExpectation(
        bundle_relative_path=_manifest_text(payload, "bundle_relative_path"),
        source_id=_manifest_uuid(payload, "source_id"),
        candidate_id=_manifest_uuid(payload, "candidate_id"),
        refinement_id=_manifest_uuid(payload, "refinement_id"),
        original_job_id=_manifest_uuid(payload, "original_job_id"),
        coarse_start=_manifest_float(payload, "coarse_start"),
        coarse_end=_manifest_float(payload, "coarse_end"),
        context_start=_manifest_float(payload, "context_start"),
        context_end=_manifest_float(payload, "context_end"),
        refined_start=_manifest_float(payload, "refined_start"),
        refined_end=_manifest_float(payload, "refined_end"),
        candidate_key=_manifest_text(payload, "candidate_key"),
        raw_span_fingerprint=_manifest_text(payload, "raw_span_fingerprint"),
        source_sha256=_manifest_text(payload, "source_sha256"),
        audio_sha256=_manifest_text(payload, "audio_sha256"),
        decoder_result_sha256=_manifest_text(payload, "decoder_result_sha256"),
        result_sha256=_manifest_text(payload, "result_sha256"),
        prior_evidence_sha256=_manifest_text(payload, "prior_evidence_sha256"),
        approved_text_sha256=_manifest_text(payload, "approved_text_sha256"),
        approved_token_count=_manifest_int(payload, "approved_token_count"),
        raw_word_record_count=_manifest_int(payload, "raw_word_record_count", default=107),
        expected_rejection_reason=_manifest_text(
            payload, "expected_rejection_reason", default="nonpositive word duration"
        ),
        context_only_invalid_record_indexes=tuple(indexes),
    )


def load_expectation_manifest(path: Path) -> LocalObservationExpectation:
    """Read a bounded JSON manifest from an operator-supplied path."""

    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise LocalObservationRecoveryError(
                "invalid_manifest", "manifest exceeds the bounded size"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except LocalObservationRecoveryError:
        raise
    except (OSError, ValueError) as error:
        raise LocalObservationRecoveryError(
            "invalid_manifest", "manifest could not be read as JSON"
        ) from error
    if not isinstance(payload, Mapping):
        raise LocalObservationRecoveryError("invalid_manifest", "manifest must be an object")
    return expectation_from_manifest(payload)


class LocalObservationRecoveryService:
    """Validate, preview, and admit one pinned local observation checkpoint."""

    def __init__(
        self,
        *,
        session: Session,
        storage: StorageService,
        options_for,
        local_identity: dict[str, object],
        config: Stage35Config = DEFAULT_CONFIG,
        audio_service: CandidateAudioWindowService | None = None,
    ) -> None:
        self._session = session
        self._storage = storage
        self._options_for = options_for
        self._local_identity = local_identity
        self._config = config
        self._audio = audio_service or CandidateAudioWindowService(
            storage=storage, session=session, config=config
        )
        self._refinement = CandidateRefinementService(
            session=session,
            storage=storage,
            config=config,
            audio_service=self._audio,
            routing_mode="local_only",
            local_identity=local_identity,
        )

    def preview(self, expectation: LocalObservationExpectation) -> RecoveryPreview:
        """Validate and report the proposed delta without any write."""

        validated = self._validate(expectation)
        return self._preview(validated, writes=0, snapshot_relative_path=None)

    def apply(self, expectation: LocalObservationExpectation) -> RecoveryApplyResult:
        """Revalidate under locks, snapshot the old checkpoint, and admit.

        The caller owns commit/rollback. No commit happens here, so a caller
        rollback preserves the prior evidence, manual text, jobs and source.
        """

        source = self._session.scalar(
            select(SourceVideo).where(SourceVideo.id == expectation.source_id).with_for_update()
        )
        if source is None:
            raise LocalObservationRecoveryError("source_not_found", "source does not exist")
        refinement = self._session.get(
            CandidateRefinement, expectation.refinement_id, with_for_update=True
        )
        if refinement is None:
            raise LocalObservationRecoveryError(
                "refinement_not_found", "candidate refinement does not exist"
            )
        validated = self._validate(expectation, source=source, refinement=refinement)
        if validated.already_admitted:
            return RecoveryApplyResult(
                applied=False,
                already_admitted=True,
                snapshot_relative_path=None,
                preview=self._preview(validated, writes=0, snapshot_relative_path=None),
            )
        snapshot_relative_path = self._snapshot(validated)
        self._mutate(validated)
        preview = self._preview(validated, writes=1, snapshot_relative_path=snapshot_relative_path)
        return RecoveryApplyResult(
            applied=True,
            already_admitted=False,
            snapshot_relative_path=snapshot_relative_path,
            preview=replace(
                preview,
                reuse_ready=validated.reuse_ready_after_apply,
                already_admitted=True,
            ),
        )

    # ------------------------------------------------------------------
    # validation

    def _validate(
        self,
        expectation: LocalObservationExpectation,
        *,
        source: SourceVideo | None = None,
        refinement: CandidateRefinement | None = None,
    ) -> _ValidatedRecovery:
        bundle_dir = self._bundle_directory(expectation)
        bundle_hashes = self._bundle_hashes(bundle_dir, expectation)
        decoder = _load_json_object(bundle_dir / "decoder-result.json", "decoder_result")
        result = _load_json_object(bundle_dir / "result.json", "result")
        prior = _load_json_object(bundle_dir / "prior-evidence.json", "prior_evidence")

        source = source or self._session.get(SourceVideo, expectation.source_id)
        if source is None:
            raise LocalObservationRecoveryError("source_not_found", "source does not exist")
        transcript = self._session.scalar(
            select(Transcript).where(Transcript.source_video_id == source.id)
        )
        if transcript is None:
            raise LocalObservationRecoveryError("transcript_not_found", "source has no transcript")
        artifact = self._session.scalar(
            select(AudioArtifact).where(AudioArtifact.source_video_id == source.id)
        )
        if artifact is None:
            raise LocalObservationRecoveryError(
                "artifact_not_found", "source has no cached audio artifact"
            )
        candidate = self._session.get(ClipCandidate, expectation.candidate_id)
        if candidate is None:
            raise LocalObservationRecoveryError("candidate_not_found", "candidate does not exist")
        refinement = refinement or self._session.get(CandidateRefinement, expectation.refinement_id)
        if refinement is None:
            raise LocalObservationRecoveryError(
                "refinement_not_found", "candidate refinement does not exist"
            )
        original_job = self._session.get(ProcessingJob, expectation.original_job_id)
        if original_job is None:
            raise LocalObservationRecoveryError(
                "original_job_not_found", "captured refinement job does not exist"
            )

        self._validate_source(source, transcript, artifact, expectation)
        marker = self._validate_operator_identity(
            source, transcript, artifact, candidate, expectation
        )
        self._validate_bundle_result(result, prior, marker, expectation)
        self._validate_current_rows(
            source,
            transcript,
            artifact,
            candidate,
            refinement,
            original_job,
            marker,
            expectation,
        )

        seed_start, seed_end = self._refinement.seed_bounds(
            candidate, RefinementPriority.FINAL_CLIP
        )
        _require_close(seed_start, expectation.coarse_start, "coarse_bounds_mismatch", "start")
        _require_close(seed_end, expectation.coarse_end, "coarse_bounds_mismatch", "end")
        context_start, context_end = self._audio.context_bounds(
            coarse_start=seed_start,
            coarse_end=seed_end,
            source_duration=artifact.duration,
            priority=RefinementPriority.FINAL_CLIP,
        )
        _require_close(context_start, expectation.context_start, "context_bounds_mismatch", "start")
        _require_close(context_end, expectation.context_end, "context_bounds_mismatch", "end")

        audio_input_fingerprint = self._audio.audio_input_fingerprint(
            source=source,
            artifact=artifact,
            context_start=context_start,
            context_end=context_end,
            priority=RefinementPriority.FINAL_CLIP,
        )
        audio_relative_path = f"{source.id}/candidate-refinements/{candidate.id}/FINAL_CLIP.wav"
        self._validate_context_audio(audio_relative_path, expectation.audio_sha256)
        audio_component = component_fingerprint(
            "audio-extraction",
            {
                "audio_input_fingerprint": audio_input_fingerprint,
                "content_hash": expectation.audio_sha256,
            },
        )

        base_options = self._options_for(RefinementPriority.FINAL_CLIP)
        region_text = candidate_region_text(transcript, context_start, context_end)
        context_terms = source_evidenced_context_terms(region_text)
        effective_options = effective_transcription_options(base_options, context_terms)
        self._validate_decoder_identity(decoder, expectation)
        self._validate_decoder_contract(decoder, effective_options)

        response = _require_mapping(decoder.get("response"), "decoder_response", "response")
        target = convert_transcription_result(
            _transcription_result(response),
            effective_options=effective_options,
            context_start=context_start,
            context_end=context_end,
            context_terms=context_terms,
        )
        self._validate_reconstructed(target, expectation)
        view = self._derive_view(target, expectation)
        self._validate_approved_text(result, view, expectation)

        already_admitted, reuse_ready = self._current_checkpoint_state(
            refinement,
            target,
            audio_component,
            context_start,
            context_end,
            context_terms,
        )
        if not already_admitted and (
            refinement.cache_eligible
            or refinement.status
            in {RefinementStatus.CANDIDATE_REFINED, RefinementStatus.FINAL_TRANSCRIPT_READY}
        ):
            raise LocalObservationRecoveryError(
                "checkpoint_ready_conflict",
                "refinement is already ready and cannot be replaced by recovery",
            )
        reuse_ready_after_apply = self._simulated_reuse_ready(
            target,
            audio_component,
            context_start,
            context_end,
            context_terms,
        )
        provenance = self._provenance(
            expectation,
            bundle_hashes,
            source,
            candidate,
            refinement,
            original_job,
            effective_options,
            context_terms,
            target,
            view,
        )
        proposed_delta = self._proposed_delta(refinement, target, audio_component, provenance)
        return _ValidatedRecovery(
            expectation=expectation,
            bundle_relative_path=expectation.bundle_relative_path,
            bundle_hashes=bundle_hashes,
            decoder=decoder,
            result=result,
            prior=prior,
            source=source,
            transcript=transcript,
            artifact=artifact,
            candidate=candidate,
            refinement=refinement,
            original_job=original_job,
            marker=marker,
            target=target,
            view=view,
            audio_component=audio_component,
            audio_input_fingerprint=audio_input_fingerprint,
            audio_relative_path=audio_relative_path,
            context_terms=context_terms,
            effective_options=effective_options,
            provenance=provenance,
            proposed_delta=proposed_delta,
            already_admitted=already_admitted,
            reuse_ready=reuse_ready,
            reuse_ready_after_apply=reuse_ready_after_apply,
        )

    def _bundle_directory(self, expectation: LocalObservationExpectation) -> Path:
        relative = expectation.bundle_relative_path.strip()
        if not relative or Path(relative).is_absolute():
            raise LocalObservationRecoveryError(
                "invalid_bundle_path", "bundle path must be storage-relative"
            )
        try:
            bundle = self._storage.resolve(StorageCategory.TEMPORARY, relative)
        except Exception as error:
            raise LocalObservationRecoveryError(
                "invalid_bundle_path", "bundle path is outside operator storage"
            ) from error
        if not bundle.is_dir():
            raise LocalObservationRecoveryError(
                "bundle_not_found", "bundle directory does not exist"
            )
        return bundle

    def _bundle_hashes(
        self, bundle_dir: Path, expectation: LocalObservationExpectation
    ) -> dict[str, str]:
        expected = {
            "decoder-result.json": expectation.decoder_result_sha256,
            "result.json": expectation.result_sha256,
            "prior-evidence.json": expectation.prior_evidence_sha256,
        }
        hashes: dict[str, str] = {}
        for name, pinned in expected.items():
            path = bundle_dir / name
            if not path.is_file():
                raise LocalObservationRecoveryError(
                    "bundle_file_missing", f"{name} is missing from the bundle"
                )
            actual = sha256_file(path)
            if actual != pinned:
                raise LocalObservationRecoveryError(
                    "bundle_hash_mismatch", f"{name} does not match the pinned hash"
                )
            hashes[name] = actual
        return hashes

    def _validate_source(
        self,
        source: SourceVideo,
        transcript: Transcript,
        artifact: AudioArtifact,
        expectation: LocalObservationExpectation,
    ) -> None:
        if source.content_hash != expectation.source_sha256:
            raise LocalObservationRecoveryError(
                "source_identity_mismatch", "stored source hash is not the pinned source"
            )
        source_path = Path(source.source_uri)
        if not source_path.is_file():
            raise LocalObservationRecoveryError(
                "source_file_missing", "source media file is unavailable"
            )
        if sha256_file(source_path) != expectation.source_sha256:
            raise LocalObservationRecoveryError(
                "source_file_mismatch", "source media bytes are not the pinned source"
            )
        if artifact.source_content_hash != source.content_hash:
            raise LocalObservationRecoveryError(
                "artifact_stale", "cached audio artifact is stale for this source"
            )
        artifact_path = self._storage.resolve(StorageCategory.SOURCES, artifact.output_path)
        if not artifact_path.is_file():
            raise LocalObservationRecoveryError(
                "artifact_file_missing", "cached audio artifact file is unavailable"
            )
        if sha256_file(artifact_path) != artifact.content_hash:
            raise LocalObservationRecoveryError(
                "artifact_file_mismatch", "cached audio artifact bytes do not match its hash"
            )
        if not transcript.segments:
            raise LocalObservationRecoveryError(
                "transcript_empty", "transcript has no immutable segments"
            )

    def _validate_operator_identity(
        self,
        source: SourceVideo,
        transcript: Transcript,
        artifact: AudioArtifact,
        candidate: ClipCandidate,
        expectation: LocalObservationExpectation,
    ) -> dict[str, object]:
        marker = operator_selection_marker(candidate)
        if marker is None:
            raise LocalObservationRecoveryError(
                "operator_marker_missing", "candidate has no operator-selection marker"
            )
        if marker.get("candidate_key") != expectation.candidate_key:
            raise LocalObservationRecoveryError(
                "candidate_key_mismatch", "operator candidate key is not the pinned key"
            )
        if marker.get("raw_span_fingerprint") != expectation.raw_span_fingerprint:
            raise LocalObservationRecoveryError(
                "raw_span_mismatch", "operator raw-span fingerprint is not pinned"
            )
        if candidate.candidate_key != expectation.candidate_key:
            raise LocalObservationRecoveryError(
                "candidate_key_mismatch", "candidate key is not the pinned key"
            )
        _require_close(candidate.start_time, expectation.coarse_start, "candidate_bounds", "start")
        _require_close(candidate.end_time, expectation.coarse_end, "candidate_bounds", "end")
        if (
            candidate.start_segment_index != marker.get("start_segment_index")
            or candidate.end_segment_index != marker.get("end_segment_index")
            or list(candidate.segment_indexes or []) != list(marker.get("segment_indexes") or [])
        ):
            raise LocalObservationRecoveryError(
                "candidate_segments", "candidate segment identity is not the operator pin"
            )
        try:
            current = operator_identity_is_current(candidate, source, transcript, artifact)
        except CandidateRegistrationError as error:
            raise LocalObservationRecoveryError(
                "operator_identity_invalid", "operator identity cannot be recomputed"
            ) from error
        if not current:
            raise LocalObservationRecoveryError(
                "operator_identity_stale",
                "operator raw-span identity no longer matches immutable evidence",
            )
        return marker

    def _validate_bundle_result(
        self,
        result: Mapping[str, object],
        prior: Mapping[str, object],
        marker: Mapping[str, object],
        expectation: LocalObservationExpectation,
    ) -> None:
        if result.get("candidate_id") != str(expectation.candidate_id):
            raise LocalObservationRecoveryError(
                "bundle_candidate_mismatch", "result bundle names a different candidate"
            )
        job = _require_mapping(result.get("job"), "bundle_job", "job")
        if job.get("id") != str(expectation.original_job_id):
            raise LocalObservationRecoveryError(
                "bundle_job_mismatch", "result bundle names a different job"
            )
        if job.get("status") != JobStatus.SUCCEEDED.value:
            raise LocalObservationRecoveryError(
                "bundle_job_not_successful", "captured job did not succeed"
            )
        if result.get("raw_span_fingerprint") != expectation.raw_span_fingerprint:
            raise LocalObservationRecoveryError(
                "bundle_raw_span_mismatch", "result bundle raw-span is not pinned"
            )
        if result.get("source_transcript_unchanged") is not True:
            raise LocalObservationRecoveryError(
                "bundle_source_changed", "result bundle did not preserve the source transcript"
            )
        if result.get("engine_invocations") != 1:
            raise LocalObservationRecoveryError(
                "bundle_engine_invocations", "result bundle is not one captured ASR run"
            )
        refinement = _require_mapping(result.get("refinement"), "bundle_refinement", "refinement")
        self._validate_bundle_refinement(refinement, expectation)
        metrics = refinement.get("metrics")
        if not isinstance(metrics, Mapping) or metrics.get("local_asr_runs") != 1:
            raise LocalObservationRecoveryError(
                "bundle_asr_metrics", "result bundle did not record exactly one ASR run"
            )
        records = refinement.get("transcript_evidence")
        if not isinstance(records, list):
            raise LocalObservationRecoveryError(
                "bundle_evidence_missing", "result bundle has no transcript evidence"
            )
        local_records = [
            item
            for item in records
            if isinstance(item, Mapping)
            and item.get("kind") == EvidenceKind.TARGETED_LOCAL_ASR.value
        ]
        if len(local_records) != 1:
            raise LocalObservationRecoveryError(
                "bundle_local_evidence", "result bundle must contain one local evidence record"
            )
        local = local_records[0]
        if local.get("state") != EvidenceState.REJECTED.value:
            raise LocalObservationRecoveryError(
                "bundle_local_state", "captured local evidence is not rejected"
            )
        if local.get("reason") != expectation.expected_rejection_reason:
            raise LocalObservationRecoveryError(
                "bundle_rejection_reason", "captured rejection reason is not the expected one"
            )
        metadata = local.get("entity_metadata")
        if isinstance(metadata, Mapping) and metadata.get("raw_word_observations"):
            raise LocalObservationRecoveryError(
                "bundle_not_legacy", "captured local evidence already has raw observations"
            )

        if prior.get("candidate_id") != str(expectation.candidate_id):
            raise LocalObservationRecoveryError(
                "prior_candidate_mismatch", "prior bundle names a different candidate"
            )
        if prior.get("raw_span_fingerprint") != expectation.raw_span_fingerprint:
            raise LocalObservationRecoveryError(
                "prior_raw_span_mismatch", "prior bundle raw-span is not pinned"
            )
        if prior.get("source_hash") != expectation.source_sha256:
            raise LocalObservationRecoveryError(
                "prior_source_mismatch", "prior bundle source hash is not pinned"
            )
        if prior.get("audio_sha256") != expectation.audio_sha256:
            raise LocalObservationRecoveryError(
                "prior_audio_mismatch", "prior bundle audio hash is not pinned"
            )
        prior_marker = _require_mapping(
            prior.get("operator_marker"), "prior_marker", "operator_marker"
        )
        if not _markers_match(marker, prior_marker):
            raise LocalObservationRecoveryError(
                "prior_marker_mismatch", "prior operator marker is not current"
            )
        prior_refinement = _require_mapping(
            prior.get("refinement"), "prior_refinement", "refinement"
        )
        self._validate_bundle_refinement(prior_refinement, expectation)

    def _validate_bundle_refinement(
        self, refinement: Mapping[str, object], expectation: LocalObservationExpectation
    ) -> None:
        expected = {
            "id": str(expectation.refinement_id),
            "source_video_id": str(expectation.source_id),
            "clip_candidate_id": str(expectation.candidate_id),
            "priority": RefinementPriority.FINAL_CLIP.value,
            "quality_level": RefinementPriority.FINAL_CLIP.value,
            "audio_content_hash": expectation.audio_sha256,
        }
        for key, value in expected.items():
            if refinement.get(key) != value:
                raise LocalObservationRecoveryError(
                    "bundle_refinement_identity", f"bundle refinement {key} is not pinned"
                )
        for key, value in (
            ("coarse_start", expectation.coarse_start),
            ("coarse_end", expectation.coarse_end),
            ("context_start", expectation.context_start),
            ("context_end", expectation.context_end),
            ("refined_start", expectation.refined_start),
            ("refined_end", expectation.refined_end),
        ):
            _require_close(refinement.get(key), value, "bundle_refinement_bounds", key)

    def _validate_current_rows(
        self,
        source: SourceVideo,
        transcript: Transcript,
        artifact: AudioArtifact,
        candidate: ClipCandidate,
        refinement: CandidateRefinement,
        original_job: ProcessingJob,
        marker: Mapping[str, object],
        expectation: LocalObservationExpectation,
    ) -> None:
        if candidate.source_video_id != source.id:
            raise LocalObservationRecoveryError(
                "candidate_source_mismatch", "candidate belongs to a different source"
            )
        if not candidate.is_current:
            raise LocalObservationRecoveryError(
                "candidate_stale", "candidate is stale and cannot be recovered"
            )
        if refinement.source_video_id != source.id:
            raise LocalObservationRecoveryError(
                "refinement_source_mismatch", "refinement belongs to a different source"
            )
        if refinement.clip_candidate_id != candidate.id:
            raise LocalObservationRecoveryError(
                "refinement_candidate_mismatch", "refinement belongs to a different candidate"
            )
        if refinement.priority is not RefinementPriority.FINAL_CLIP:
            raise LocalObservationRecoveryError(
                "refinement_priority", "recovery only admits a FINAL_CLIP checkpoint"
            )
        if refinement.quality_level != RefinementPriority.FINAL_CLIP.value:
            raise LocalObservationRecoveryError(
                "refinement_quality", "recovery only admits a FINAL_CLIP checkpoint"
            )
        if refinement.audio_relative_path != (
            f"{source.id}/candidate-refinements/{candidate.id}/FINAL_CLIP.wav"
        ):
            raise LocalObservationRecoveryError(
                "refinement_audio_path", "refinement audio path is not the expected window"
            )
        if refinement.audio_content_hash != expectation.audio_sha256:
            raise LocalObservationRecoveryError(
                "refinement_audio_hash", "refinement audio hash is not pinned"
            )
        for key, value in (
            ("coarse_start", expectation.coarse_start),
            ("coarse_end", expectation.coarse_end),
            ("context_start", expectation.context_start),
            ("context_end", expectation.context_end),
            ("refined_start", expectation.refined_start),
            ("refined_end", expectation.refined_end),
        ):
            _require_close(getattr(refinement, key), value, "refinement_bounds", key)
        _require_close(marker.get("start_time"), expectation.coarse_start, "marker_bounds", "start")
        _require_close(marker.get("end_time"), expectation.coarse_end, "marker_bounds", "end")
        if original_job.source_video_id != source.id:
            raise LocalObservationRecoveryError(
                "original_job_source", "captured job belongs to a different source"
            )
        if original_job.candidate_refinement_id != refinement.id:
            raise LocalObservationRecoveryError(
                "original_job_refinement", "captured job belongs to a different refinement"
            )
        if original_job.kind is not JobKind.CANDIDATE_REFINEMENT:
            raise LocalObservationRecoveryError(
                "original_job_kind", "captured job is not a candidate refinement"
            )
        if original_job.status is not JobStatus.SUCCEEDED:
            raise LocalObservationRecoveryError(
                "original_job_status", "captured job is not successful"
            )
        if refinement.active_job_id:
            raise LocalObservationRecoveryError(
                "refinement_active", "refinement still owns an active job"
            )
        active = self._session.scalar(
            select(ProcessingJob.id).where(
                ProcessingJob.candidate_refinement_id == refinement.id,
                ProcessingJob.status.in_({JobStatus.QUEUED, JobStatus.RUNNING}),
            )
        )
        if active is not None:
            raise LocalObservationRecoveryError(
                "refinement_active", "refinement has an active queued/running job"
            )
        manual = (refinement.manual_transcript or "").strip()
        if manual and not _manual_matches_approved(manual, expectation):
            raise LocalObservationRecoveryError(
                "manual_conflict", "existing manual text is not the approved wording"
            )

    def _validate_decoder_identity(
        self, decoder: Mapping[str, object], expectation: LocalObservationExpectation
    ) -> None:
        _require_close(
            decoder.get("context_start"), expectation.context_start, "decoder_context", "start"
        )
        _require_close(
            decoder.get("context_end"), expectation.context_end, "decoder_context", "end"
        )
        if decoder.get("audio_sha256") != expectation.audio_sha256:
            raise LocalObservationRecoveryError(
                "decoder_audio_mismatch", "decoder observation audio hash is not pinned"
            )

    def _validate_decoder_contract(
        self, decoder: Mapping[str, object], effective_options: TranscriptionOptions
    ) -> None:
        bundle_options = _require_mapping(decoder.get("options"), "decoder_options", "options")
        expected = _options_payload(effective_options)
        for field, value in expected.items():
            if field not in bundle_options:
                raise LocalObservationRecoveryError(
                    "decoder_contract_mismatch", f"bundle is missing decoder option {field}"
                )
            if not _option_value_matches(bundle_options[field], value):
                raise LocalObservationRecoveryError(
                    "decoder_contract_mismatch", f"decoder option {field} changed"
                )
        for field in bundle_options:
            if field not in expected:
                raise LocalObservationRecoveryError(
                    "decoder_contract_mismatch", f"bundle has unknown decoder option {field}"
                )

    def _validate_context_audio(self, relative_path: str, expected_hash: str) -> None:
        path = self._storage.resolve(StorageCategory.SOURCES, relative_path)
        if not path.is_file():
            raise LocalObservationRecoveryError(
                "context_audio_missing", "extracted context WAV is unavailable"
            )
        if sha256_file(path) != expected_hash:
            raise LocalObservationRecoveryError(
                "context_audio_mismatch", "extracted context WAV bytes are not pinned"
            )

    def _validate_reconstructed(
        self, target: TargetASRResult, expectation: LocalObservationExpectation
    ) -> None:
        if target.rejected_reason != expectation.expected_rejection_reason:
            raise LocalObservationRecoveryError(
                "reconstructed_rejection", "raw observation rejection reason changed"
            )
        if target.raw_word_record_count != expectation.raw_word_record_count:
            raise LocalObservationRecoveryError(
                "raw_record_count", "raw observation record count is not pinned"
            )
        if not target.raw_word_observations_complete:
            raise LocalObservationRecoveryError(
                "raw_observations_incomplete", "raw observation is not complete"
            )
        if len(target.raw_word_observations) != target.raw_word_record_count:
            raise LocalObservationRecoveryError(
                "raw_observations_missing", "raw observation is missing records"
            )
        if not target.transcript.strip():
            raise LocalObservationRecoveryError(
                "empty_transcript", "raw observation transcript is empty"
            )

    def _derive_view(
        self, target: TargetASRResult, expectation: LocalObservationExpectation
    ) -> FinalClipView:
        view = derive_final_clip_view(
            raw_observations=list(target.raw_word_observations),
            record_count=target.raw_word_record_count,
            observations_complete=target.raw_word_observations_complete,
            observation_fingerprint=target.raw_word_observations_fingerprint,
            transcript=target.transcript,
            context_start=expectation.context_start,
            context_end=expectation.context_end,
            clip_start=expectation.refined_start,
            clip_end=expectation.refined_end,
        )
        if not isinstance(view, FinalClipView):
            raise LocalObservationRecoveryError(
                "derived_view_refused", f"derived clip view refused: {view.reason}"
            )
        if view.context_only_invalid_record_indexes != (
            expectation.context_only_invalid_record_indexes
        ):
            raise LocalObservationRecoveryError(
                "derived_view_invalid_records", "derived context-only invalid records changed"
            )
        return view

    def _validate_approved_text(
        self,
        result: Mapping[str, object],
        view: FinalClipView,
        expectation: LocalObservationExpectation,
    ) -> None:
        if len(view.words) != expectation.approved_token_count:
            raise LocalObservationRecoveryError(
                "approved_token_count", "derived clip view token count is not approved"
            )
        view_text = _normalise_text(view.text)
        if _text_sha256(view_text) != expectation.approved_text_sha256:
            raise LocalObservationRecoveryError(
                "approved_text_mismatch", "derived clip view is not the approved wording"
            )
        selected = result.get("selected_words")
        if not isinstance(selected, list) or not all(
            isinstance(item, Mapping) for item in selected
        ):
            raise LocalObservationRecoveryError(
                "bundle_selected_words", "result bundle has no selected words"
            )
        if len(selected) != expectation.approved_token_count:
            raise LocalObservationRecoveryError(
                "bundle_selected_count", "bundle selected words are not the approved count"
            )
        selected_text = _normalise_text("".join(str(item.get("text", "")) for item in selected))
        if selected_text != view_text:
            raise LocalObservationRecoveryError(
                "bundle_selected_text", "bundle selected words differ from the derived view"
            )
        if _text_sha256(selected_text) != expectation.approved_text_sha256:
            raise LocalObservationRecoveryError(
                "bundle_selected_sha", "bundle selected words are not the approved wording"
            )

    # ------------------------------------------------------------------
    # current checkpoint state

    def _current_checkpoint_state(
        self,
        refinement: CandidateRefinement,
        target: TargetASRResult,
        audio_component: str,
        context_start: float,
        context_end: float,
        context_terms: Sequence[str],
    ) -> tuple[bool, bool]:
        records = [
            dict(item)
            for item in (refinement.transcript_evidence or [])
            if isinstance(item, Mapping)
            and item.get("kind") == EvidenceKind.TARGETED_LOCAL_ASR.value
        ]
        if len(records) > 1:
            raise LocalObservationRecoveryError(
                "checkpoint_conflict", "refinement has multiple local evidence records"
            )
        if not records:
            return False, False
        record = records[0]
        state = record.get("state")
        if state == EvidenceState.ACCEPTED.value:
            raise LocalObservationRecoveryError(
                "checkpoint_accepted", "refinement already has an accepted local checkpoint"
            )
        metadata = record.get("entity_metadata")
        observations = (
            metadata.get("raw_word_observations") if isinstance(metadata, Mapping) else None
        )
        if not observations:
            return False, False
        if self._local_record_matches(record, target, audio_component, refinement):
            ready = self._simulated_reuse_ready(
                target,
                audio_component,
                context_start,
                context_end,
                context_terms,
            )
            return True, ready
        raise LocalObservationRecoveryError(
            "checkpoint_conflict", "current local checkpoint is a different complete observation"
        )

    def _local_record_matches(
        self,
        record: Mapping[str, object],
        target: TargetASRResult,
        audio_component: str,
        refinement: CandidateRefinement,
    ) -> bool:
        fingerprints = refinement.component_fingerprints or {}
        if fingerprints.get("audio_extraction") != audio_component:
            return False
        if fingerprints.get("local_asr") != target.fingerprint:
            return False
        if record.get("state") != EvidenceState.REJECTED.value:
            return False
        if record.get("fingerprint") != target.fingerprint:
            return False
        if record.get("provider") != target.provider or record.get("model") != target.model:
            return False
        if record.get("transcript") != target.transcript:
            return False
        if not _option_value_matches(record.get("settings"), target.settings):
            return False
        if not _option_value_matches(record.get("word_timestamps"), _words_payload(target)):
            return False
        if record.get("reason") != target.rejected_reason:
            return False
        metadata = record.get("entity_metadata")
        if not isinstance(metadata, Mapping):
            return False
        if not _option_value_matches(
            metadata.get("raw_word_observations"), list(target.raw_word_observations)
        ):
            return False
        return (
            metadata.get("raw_word_record_count") == target.raw_word_record_count
            and metadata.get("raw_word_observations_complete") is True
            and metadata.get("raw_word_observations_fingerprint")
            == target.raw_word_observations_fingerprint
        )

    def _simulated_reuse_ready(
        self,
        target: TargetASRResult,
        audio_component: str,
        context_start: float,
        context_end: float,
        context_terms: Sequence[str],
    ) -> bool:
        record = self._local_record_payload(target, {}, context_start, context_end)
        stand_in = SimpleNamespace(
            quality_level=RefinementPriority.FINAL_CLIP.value,
            component_fingerprints={
                "audio_extraction": audio_component,
                "local_asr": target.fingerprint,
            },
            transcript_evidence=[record],
            context_start=context_start,
            context_end=context_end,
        )
        window = SimpleNamespace(
            context_start=context_start,
            context_end=context_end,
        )
        return self._refinement.local_observation_reuse_ready(
            stand_in,  # type: ignore[arg-type]
            audio_component=audio_component,
            priority=RefinementPriority.FINAL_CLIP,
            window=window,  # type: ignore[arg-type]
            context_terms=context_terms,
        )

    # ------------------------------------------------------------------
    # mutation

    def _snapshot(self, validated: _ValidatedRecovery) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        relative = (
            f"diagnostics/recovery-admissions/{validated.expectation.candidate_id}/"
            f"{timestamp}-{uuid4().hex}.json"
        )
        destination = self._storage.resolve(StorageCategory.TEMPORARY, relative)
        payload = {
            "kind": "local-observation-recovery-snapshot-v1",
            "candidate_id": str(validated.expectation.candidate_id),
            "refinement_id": str(validated.expectation.refinement_id),
            "bundle_relative_path": validated.bundle_relative_path,
            "bundle_hashes": validated.bundle_hashes,
            "expectation": validated.expectation.as_dict(),
            "previous_refinement": _refinement_payload(validated.refinement),
        }
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        try:
            self._storage.atomic_write(destination, [data])
        except Exception as error:
            raise LocalObservationRecoveryError(
                "snapshot_failed", "audit snapshot could not be written"
            ) from error
        return relative

    def _mutate(self, validated: _ValidatedRecovery) -> None:
        refinement = validated.refinement
        record = self._local_record_payload(
            validated.target,
            validated.provenance,
            validated.expectation.context_start,
            validated.expectation.context_end,
        )
        evidence: list[dict[str, object]] = []
        replaced = False
        for item in refinement.transcript_evidence or []:
            if not isinstance(item, Mapping):
                continue
            if item.get("kind") == EvidenceKind.TARGETED_LOCAL_ASR.value:
                if not replaced:
                    evidence.append(record)
                    replaced = True
                continue
            evidence.append(dict(item))
        if not replaced:
            evidence.append(record)
        refinement.transcript_evidence = evidence
        fingerprints = dict(refinement.component_fingerprints or {})
        fingerprints["audio_extraction"] = validated.audio_component
        fingerprints["local_asr"] = validated.target.fingerprint
        refinement.component_fingerprints = fingerprints
        provider = dict(refinement.provider_evidence or {})
        provider["local_asr_accepted"] = False
        provider["recovery"] = validated.provenance
        refinement.provider_evidence = provider
        refinement.status = RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW
        refinement.cache_eligible = False
        refinement.active_job_id = None

    # ------------------------------------------------------------------
    # reporting helpers

    def _preview(
        self,
        validated: _ValidatedRecovery,
        *,
        writes: int,
        snapshot_relative_path: str | None,
    ) -> RecoveryPreview:
        return RecoveryPreview(
            bundle_relative_path=validated.bundle_relative_path,
            bundle_hashes=dict(validated.bundle_hashes),
            source_id=str(validated.source.id),
            candidate_id=str(validated.candidate.id),
            refinement_id=str(validated.refinement.id),
            original_job_id=str(validated.original_job.id),
            coarse_start=validated.expectation.coarse_start,
            coarse_end=validated.expectation.coarse_end,
            context_start=validated.expectation.context_start,
            context_end=validated.expectation.context_end,
            refined_start=validated.expectation.refined_start,
            refined_end=validated.expectation.refined_end,
            decoder_identity=_decoder_identity(validated.effective_options),
            context_terms=validated.context_terms,
            reconstructed=_reconstructed_payload(validated.target),
            derived_view=_derived_view_payload(validated.view),
            approved_text_sha256=validated.expectation.approved_text_sha256,
            approved_text_matches=True,
            provenance=validated.provenance,
            proposed_delta=validated.proposed_delta,
            reuse_ready=validated.reuse_ready,
            reuse_ready_after_apply=validated.reuse_ready_after_apply,
            already_admitted=validated.already_admitted,
            writes=writes,
            snapshot_relative_path=snapshot_relative_path,
        )

    def _provenance(
        self,
        expectation: LocalObservationExpectation,
        bundle_hashes: Mapping[str, str],
        source: SourceVideo,
        candidate: ClipCandidate,
        refinement: CandidateRefinement,
        original_job: ProcessingJob,
        effective_options: TranscriptionOptions,
        context_terms: Sequence[str],
        target: TargetASRResult,
        view: FinalClipView,
    ) -> dict[str, object]:
        historical = {
            "local_asr_fingerprint": _historical_local_fingerprint(refinement),
            "raw_span_fingerprint": expectation.raw_span_fingerprint,
            "selected_text_sha256": expectation.approved_text_sha256,
        }
        return {
            "kind": _RECOVERY_PROVENANCE_KIND,
            "bundle_relative_path": expectation.bundle_relative_path,
            "bundle_hashes": dict(bundle_hashes),
            "source_id": str(source.id),
            "candidate_id": str(candidate.id),
            "refinement_id": str(refinement.id),
            "original_job_id": str(original_job.id),
            "source_sha256": expectation.source_sha256,
            "audio_sha256": expectation.audio_sha256,
            "decoder_identity": {
                **_decoder_identity(effective_options),
                "context_terms": list(context_terms),
            },
            "historical": historical,
            "reconstructed": _reconstructed_payload(target),
            "derived_view": _derived_view_payload(view),
            "approved_text_sha256": expectation.approved_text_sha256,
            "approved_token_count": expectation.approved_token_count,
            "no_model_call": True,
        }

    def _proposed_delta(
        self,
        refinement: CandidateRefinement,
        target: TargetASRResult,
        audio_component: str,
        provenance: Mapping[str, object],
    ) -> dict[str, object]:
        fingerprints = refinement.component_fingerprints or {}
        records = [
            item
            for item in (refinement.transcript_evidence or [])
            if isinstance(item, Mapping)
            and item.get("kind") == EvidenceKind.TARGETED_LOCAL_ASR.value
        ]
        current_local = records[0] if records else None
        return {
            "transcript_evidence": {
                "operation": "replace_single_local_record"
                if current_local is not None
                else "append_local_record",
                "kind": EvidenceKind.TARGETED_LOCAL_ASR.value,
                "state": EvidenceState.REJECTED.value,
                "reason": target.rejected_reason,
                "raw_word_record_count": target.raw_word_record_count,
                "raw_word_observations_complete": target.raw_word_observations_complete,
                "raw_word_observations_fingerprint": target.raw_word_observations_fingerprint,
                "recovery_provenance": True,
            },
            "component_fingerprints": {
                "audio_extraction": {
                    "from": fingerprints.get("audio_extraction"),
                    "to": audio_component,
                },
                "local_asr": {
                    "from": fingerprints.get("local_asr"),
                    "to": target.fingerprint,
                },
            },
            "provider_evidence": {"recovery": dict(provenance)},
            "status": {
                "from": refinement.status.value,
                "to": RefinementStatus.NEEDS_MANUAL_TRANSCRIPT_REVIEW.value,
            },
            "cache_eligible": {"from": refinement.cache_eligible, "to": False},
            "untouched": [
                "manual_transcript",
                "final_transcript",
                "automatic_transcript",
                "word_timestamps",
                "entity_evidence",
                "unresolved_spans",
                "routing_evidence",
                "source_lifecycle",
                "existing_jobs",
                "publishing_approvals",
            ],
        }

    def _local_record_payload(
        self,
        target: TargetASRResult,
        provenance: Mapping[str, object],
        context_start: float,
        context_end: float,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "raw_word_observations": list(target.raw_word_observations),
            "raw_word_record_count": target.raw_word_record_count,
            "raw_word_observations_complete": target.raw_word_observations_complete,
            "raw_word_observations_fingerprint": target.raw_word_observations_fingerprint,
        }
        if target.rejected_detail is not None:
            metadata["timestamp_diagnostic"] = target.rejected_detail
        if provenance:
            metadata["recovery"] = dict(provenance)
        return {
            "kind": EvidenceKind.TARGETED_LOCAL_ASR.value,
            "fingerprint": target.fingerprint,
            "provider": target.provider,
            "model": target.model,
            "settings": dict(target.settings),
            "window_start": context_start,
            "window_end": context_end,
            "transcript": target.transcript,
            "confidence": target.confidence,
            "state": EvidenceState.REJECTED.value,
            "word_timestamps": _words_payload(target),
            "reason": target.rejected_reason,
            "entity_metadata": metadata,
            "code_switch_tokens": [],
        }


def _transcription_result(response: Mapping[str, object]) -> TranscriptionResult:
    language = response.get("language")
    if language is not None and not isinstance(language, str):
        raise LocalObservationRecoveryError("decoder_response", "language is malformed")
    probability = response.get("language_probability")
    if probability is not None and not _is_number(probability):
        raise LocalObservationRecoveryError("decoder_response", "language probability is malformed")
    raw_text = response.get("raw_text")
    duration = response.get("duration")
    segments = response.get("segments")
    word_segments = response.get("word_segments")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise LocalObservationRecoveryError("decoder_response", "raw text is missing")
    if not _is_number(duration):
        raise LocalObservationRecoveryError("decoder_response", "duration is malformed")
    if not isinstance(segments, list) or not all(isinstance(item, Mapping) for item in segments):
        raise LocalObservationRecoveryError("decoder_response", "segments are malformed")
    if not isinstance(word_segments, list) or not all(
        isinstance(item, Mapping) for item in word_segments
    ):
        raise LocalObservationRecoveryError("decoder_response", "word records are malformed")
    return TranscriptionResult(
        language=language,
        language_probability=float(probability) if probability is not None else None,
        raw_text=raw_text,
        duration=float(duration),
        segments=[dict(item) for item in segments],
        word_segments=[dict(item) for item in word_segments],
    )


def _options_payload(options: TranscriptionOptions) -> dict[str, object]:
    return {field: getattr(options, field) for field in TranscriptionOptions.__dataclass_fields__}


def _decoder_identity(options: TranscriptionOptions) -> dict[str, object]:
    return {
        "provider": "faster-whisper",
        "model": options.model,
        "device": options.device,
        "compute_type": options.compute_type,
        "beam_size": options.beam_size,
        "language": options.language,
        "word_timestamps": options.word_timestamps,
        "condition_on_previous_text": options.condition_on_previous_text,
        "vad_filter": options.vad_filter,
        "hotwords": options.hotwords,
        "cpu_compute_type": options.cpu_compute_type,
        "cuda_compute_type": options.cuda_compute_type,
        "temperature": list(options.temperature),
        "cpu_threads": options.cpu_threads,
        "index_batch_size": options.index_batch_size,
    }


def _reconstructed_payload(target: TargetASRResult) -> dict[str, object]:
    return {
        "local_asr_fingerprint": target.fingerprint,
        "observation_fingerprint": target.raw_word_observations_fingerprint,
        "raw_word_record_count": target.raw_word_record_count,
        "raw_word_observations_complete": target.raw_word_observations_complete,
        "rejected_reason": target.rejected_reason,
        "rejected_detail": target.rejected_detail,
        "word_count": len(target.word_timestamps),
        "transcript_sha256": _text_sha256(target.transcript),
    }


def _derived_view_payload(view: FinalClipView) -> dict[str, object]:
    return {
        "state": view.state,
        "policy_version": view.policy_version,
        "fingerprint": view.fingerprint,
        "selected_text_sha256": _text_sha256(_normalise_text(view.text)),
        "selected_word_count": len(view.words),
        "context_only_invalid_record_indexes": list(view.context_only_invalid_record_indexes),
        "clip_start": view.clip_start,
        "clip_end": view.clip_end,
    }


def _words_payload(target: TargetASRResult) -> list[dict[str, object]]:
    return [word.as_dict for word in target.word_timestamps]


def _historical_local_fingerprint(refinement: CandidateRefinement) -> str | None:
    fingerprints = refinement.component_fingerprints or {}
    value = fingerprints.get("local_asr")
    return value if isinstance(value, str) else None


def _refinement_payload(refinement: CandidateRefinement) -> dict[str, object]:
    return {
        "id": str(refinement.id),
        "source_video_id": str(refinement.source_video_id),
        "clip_candidate_id": str(refinement.clip_candidate_id),
        "priority": refinement.priority.value,
        "status": refinement.status.value,
        "quality_level": refinement.quality_level,
        "active_job_id": refinement.active_job_id,
        "coarse_start": refinement.coarse_start,
        "coarse_end": refinement.coarse_end,
        "context_start": refinement.context_start,
        "context_end": refinement.context_end,
        "refined_start": refinement.refined_start,
        "refined_end": refinement.refined_end,
        "audio_relative_path": refinement.audio_relative_path,
        "audio_content_hash": refinement.audio_content_hash,
        "audio_input_fingerprint": refinement.audio_input_fingerprint,
        "automatic_transcript": refinement.automatic_transcript,
        "manual_transcript": refinement.manual_transcript,
        "final_transcript": refinement.final_transcript,
        "word_timestamps": list(refinement.word_timestamps or []),
        "confidence": refinement.confidence,
        "dialect_profile": refinement.dialect_profile,
        "dialect_confidence": refinement.dialect_confidence,
        "code_switch_evidence": dict(refinement.code_switch_evidence or {}),
        "transcript_evidence": list(refinement.transcript_evidence or []),
        "entity_evidence": list(refinement.entity_evidence or []),
        "unresolved_spans": list(refinement.unresolved_spans or []),
        "provider_evidence": dict(refinement.provider_evidence or {}),
        "routing_evidence": dict(refinement.routing_evidence or {}),
        "input_fingerprint": refinement.input_fingerprint,
        "output_fingerprint": refinement.output_fingerprint,
        "component_fingerprints": dict(refinement.component_fingerprints or {}),
        "policy_version": refinement.policy_version,
        "schema_version": refinement.schema_version,
        "validation_version": refinement.validation_version,
        "cache_eligible": refinement.cache_eligible,
        "metrics": dict(refinement.metrics or {}),
        "processing_duration": refinement.processing_duration,
        "created_at": refinement.created_at,
        "updated_at": refinement.updated_at,
    }


def _markers_match(current: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    keys = (
        "version",
        "origin",
        "source_id",
        "source_content_hash",
        "transcript_id",
        "transcript_input_fingerprint",
        "transcription_revision",
        "audio_artifact_content_hash",
        "audio_artifact_source_content_hash",
        "start_segment_index",
        "end_segment_index",
        "segment_indexes",
        "raw_span_fingerprint",
        "candidate_key",
    )
    for key in keys:
        if current.get(key) != expected.get(key):
            return False
    return _close(current.get("start_time"), expected.get("start_time")) and _close(
        current.get("end_time"), expected.get("end_time")
    )


def _manual_matches_approved(manual: str, expectation: LocalObservationExpectation) -> bool:
    return _text_sha256(_normalise_text(manual)) == expectation.approved_text_sha256


def _load_json_object(path: Path, code: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LocalObservationRecoveryError(code, f"{path.name} could not be read") from error
    if not isinstance(payload, dict):
        raise LocalObservationRecoveryError(code, f"{path.name} is not a JSON object")
    return payload


def _require_mapping(value: object, code: str, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LocalObservationRecoveryError(code, f"{label} is missing or malformed")
    return value


def _manifest_text(payload: Mapping[str, object], key: str, *, default: str | None = None) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise LocalObservationRecoveryError("invalid_manifest", f"{key} must be text")
    return value.strip()


def _manifest_uuid(payload: Mapping[str, object], key: str) -> UUID:
    value = payload.get(key)
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError) as error:
        raise LocalObservationRecoveryError("invalid_manifest", f"{key} must be a UUID") from error


def _manifest_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not _is_number(value):
        raise LocalObservationRecoveryError("invalid_manifest", f"{key} must be numeric")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise LocalObservationRecoveryError("invalid_manifest", f"{key} must be finite")
    return number


def _manifest_int(payload: Mapping[str, object], key: str, *, default: int | None = None) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LocalObservationRecoveryError("invalid_manifest", f"{key} must be an integer")
    return value


def _require_close(value: object, expected: float, code: str, label: str) -> None:
    if not _close(value, expected):
        raise LocalObservationRecoveryError(code, f"{label} does not match the pinned value")


def _close(left: object, right: object) -> bool:
    if not _is_number(left) or not _is_number(right):
        return False
    return abs(float(left) - float(right)) <= _BOUND_TOLERANCE  # type: ignore[arg-type]


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _option_value_matches(stored: object, expected: object) -> bool:
    """Compare JSON-round-tripped values without tuple/list/dict drift."""

    if isinstance(stored, bool) or isinstance(expected, bool):
        return stored is expected
    if isinstance(stored, Mapping) or isinstance(expected, Mapping):
        if not isinstance(stored, Mapping) or not isinstance(expected, Mapping):
            return False
        if set(stored) != set(expected):
            return False
        return all(_option_value_matches(stored[key], expected[key]) for key in expected)
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


def _normalise_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _text_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "LocalObservationExpectation",
    "LocalObservationRecoveryError",
    "LocalObservationRecoveryService",
    "RecoveryApplyResult",
    "RecoveryPreview",
    "expectation_from_manifest",
    "load_expectation_manifest",
]
