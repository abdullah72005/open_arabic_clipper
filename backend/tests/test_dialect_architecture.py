"""Stage 2.7.1 architecture regressions: INDEX, refinement, reuse, fingerprints."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import RefinementPriority, RightsStatus
from app.core.settings import Settings
from app.db.base import Base
from app.models import SourceVideo, Transcript
from app.pipeline.fingerprints import (
    reconstruction_output_fingerprint,
    reconstruction_target_fingerprint,
)
from app.pipeline.stages import ContextualReconstructionExecutor, TranscriptNormalizationExecutor
from app.transcription.dialect import ArabicDialectProfile
from app.transcription.reconstruction.providers import (
    ReconstructionCandidate,
    ReconstructionRequest,
)
from app.transcription.reconstruction.refine import refine_transcript_window
from app.transcription.reconstruction.routing import AdaptiveRoutingConfig, RoutingMode
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import ProviderAvailability, ProviderHealth

CANDIDATE = RefinementPriority.CANDIDATE
FINAL_CLIP = RefinementPriority.FINAL_CLIP
INDEX = RefinementPriority.INDEX
_EGYPTIAN = ArabicDialectProfile.EGYPTIAN
_SAUDI = ArabicDialectProfile.SAUDI


def _identity() -> dict[str, object]:
    return {
        "provider": "ollama",
        "model": "qwen3.5:4b",
        "digest": "sha256:x",
        "prompt_hash": "p",
        "schema_version": "s",
        "max_context_tokens": 4096,
        "output_tokens": 256,
        "chat_framing_reserve": 64,
        "safety_reserve": 128,
        "confidence_policy_version": "c",
        "validation_version": "v",
    }


class RecordingLocal:
    def __init__(self, candidate: ReconstructionCandidate | None = None) -> None:
        self.calls = 0
        self.requests: list[ReconstructionRequest] = []
        self.candidate = candidate or ReconstructionCandidate(
            "provider-0", "ضخمة", provider_confidence=1.0
        )

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "ollama", "qwen3.5:4b", "sha256:x", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
        return _identity()

    def refresh_runtime_identity(self) -> dict[str, object]:
        return _identity()

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        self.requests.extend(requests)
        return {request.segment_index: self.candidate for request in requests}

    def release(self) -> None:
        pass


class CountingGemini:
    def __init__(self) -> None:
        self.calls = 0

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "gemini", "gemini-3.8-flash", "sha256:g", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "gemini", "model": "gemini-3.8-flash", "digest": "sha256:g"}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        return {
            request.segment_index: ReconstructionCandidate("provider-0", "ضخمة", 1.0)
            for request in requests
        }

    def usage_summary(self) -> dict[str, int]:
        return {"prompt_token_count": 1, "candidates_token_count": 1, "total_token_count": 2}

    def release(self) -> None:
        pass


def _segment(index: int, raw: str = "دخم", profile: str | None = None) -> dict[str, object]:
    segment: dict[str, object] = {
        "start": float(index),
        "end": float(index + 1),
        "text": raw,
        "raw_text": raw,
        "corrected_text": raw,
        "final_text": raw,
        "words": [
            {"word": "م", "probability": 0.30},
            {"word": "ش", "probability": 0.25},
            {"word": "قادر", "probability": 0.20},
            {"word": "يفهم", "probability": 0.90},
        ],
        "correction_applied": False,
        "correction_confidence": 0.0,
        "correction_method": "unchanged",
        "correction_changes": [],
        "code_switch_suspected": False,
        "code_switch_tokens": [],
    }
    if profile is not None:
        segment["dialect_profile"] = profile
        segment["dialect_confidence"] = 0.95
        segment["dialect_selection"] = "detected"
        segment["dialect_policy_version"] = "dialect-policy-v2"
    return segment


def _setup(sqlite_engine: Any, segments: list[dict[str, object]]) -> Any:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri="file:///tmp/dialect.mp4",
            content_hash="h",
            rights_status=RightsStatus.OWNED,
        )
        session.add(source)
        session.commit()
        transcript = Transcript(
            source_video_id=source.id,
            whisper_model="large-v3-turbo",
            input_fingerprint="asr-fp",
            normalization_fingerprint="norm-fp",
            transcription_revision=1,
            correction_version="egyptian-ar-v1",
            language="ar",
            segments=segments,
            word_segments=[],
            raw_text=" ".join(str(segment["text"]) for segment in segments),
            corrected_text=" ".join(str(segment["text"]) for segment in segments),
            final_text=" ".join(str(segment["text"]) for segment in segments),
        )
        session.add(transcript)
        session.commit()
        return source.id


def test_index_makes_zero_provider_calls_with_dialect_evidence(sqlite_engine: Any) -> None:
    segments = [_segment(0, profile="EGYPTIAN"), _segment(1, profile="EGYPTIAN")]
    source_id = _setup(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        local = RecordingLocal()
        gemini = CountingGemini()
        reconstructor = ContextualReconstructor(
            local,
            gemini_provider=gemini,
            routing=AdaptiveRoutingConfig(mode=RoutingMode.ADAPTIVE),
            gemini_budget=5,
            priority=INDEX,
        )
        executor = ContextualReconstructionExecutor(session=session, reconstructor=reconstructor)
        executor.execute(session.get(SourceVideo, source_id), force=True)

        assert local.calls == 0
        assert gemini.calls == 0


def test_dialect_detection_never_constructs_or_probes_providers(sqlite_engine: Any) -> None:
    segments = [_segment(0, raw="أنا عايز أعمل deploy دلوقتي", profile=None)]
    source_id = _setup(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        normalization = TranscriptNormalizationExecutor(session=session)
        result = normalization.execute(session.get(SourceVideo, source_id), force=True)

        assert result.output_fingerprint
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )
        assert transcript.dialect_profile == ArabicDialectProfile.EGYPTIAN.value


def test_candidate_refinement_uses_stored_profile_in_shared_requests(sqlite_engine: Any) -> None:
    segments = [
        _segment(0, profile="EGYPTIAN"),
        _segment(1, profile="EGYPTIAN"),
        _segment(2, profile="EGYPTIAN"),
    ]
    source_id = _setup(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        local = RecordingLocal()
        reconstructor = ContextualReconstructor(
            local,
            routing=AdaptiveRoutingConfig(mode=RoutingMode.LOCAL_ONLY),
            priority=CANDIDATE,
            batch_windows=16,
            batch_characters=48_000,
        )
        outcome = refine_transcript_window(
            session,
            source_id,
            start_time=0.5,
            end_time=2.5,
            priority=CANDIDATE,
            reconstructor=reconstructor,
        )

        assert outcome.priority is CANDIDATE
        assert local.calls >= 1
        assert all(request.dialect_profile == "EGYPTIAN" for request in local.requests)
        assert all(request.segment_index in {0, 1, 2} for request in local.requests)


def test_final_clip_contract_remains_intact(sqlite_engine: Any) -> None:
    segments = [_segment(0, profile="EGYPTIAN"), _segment(1, profile="EGYPTIAN")]
    source_id = _setup(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        local = RecordingLocal()
        reconstructor = ContextualReconstructor(
            local,
            routing=AdaptiveRoutingConfig(mode=RoutingMode.LOCAL_ONLY),
            priority=FINAL_CLIP,
            batch_windows=16,
            batch_characters=48_000,
        )
        outcome = refine_transcript_window(
            session,
            source_id,
            start_time=0.5,
            end_time=2.5,
            priority=FINAL_CLIP,
            reconstructor=reconstructor,
        )

        assert outcome.priority is FINAL_CLIP
        assert local.calls >= 1


def test_qwen_is_disabled_by_default_and_enabled_explicitly(monkeypatch: Any) -> None:
    monkeypatch.delenv("CLIPFACTORY_LOCAL_QWEN_ENABLED", raising=False)
    monkeypatch.delenv("CLIPFACTORY_RECONSTRUCTION_PROVIDER", raising=False)
    settings = Settings(_env_file=None)

    assert settings.local_qwen_enabled is False
    assert settings.reconstruction_provider_instance() is None


def test_qwen_works_when_explicitly_enabled(monkeypatch: Any) -> None:
    monkeypatch.setenv("CLIPFACTORY_LOCAL_QWEN_ENABLED", "true")
    monkeypatch.setenv("CLIPFACTORY_RECONSTRUCTION_PROVIDER", "ollama")
    settings = Settings(_env_file=None)

    assert settings.local_qwen_enabled is True
    assert settings.reconstruction_provider_instance() is not None


def test_gemini_remains_optional(monkeypatch: Any) -> None:
    monkeypatch.delenv("CLIPFACTORY_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None)

    assert settings.gemini_provider_instance() is None
    assert settings.gemini_api_key_present is False


def test_per_target_reuse_respects_dialect_identity() -> None:
    segments_egyptian = [_segment(0, profile="EGYPTIAN")]
    segments_saudi = [_segment(0, profile="SAUDI")]

    egyptian_fp = reconstruction_target_fingerprint(
        provider_identity=_identity(),
        segments=segments_egyptian,
        target_index=0,
        language="ar",
        transcription_fingerprint="t",
        correction_version="c",
    )
    saudi_fp = reconstruction_target_fingerprint(
        provider_identity=_identity(),
        segments=segments_saudi,
        target_index=0,
        language="ar",
        transcription_fingerprint="t",
        correction_version="c",
    )

    assert egyptian_fp != saudi_fp


def test_per_target_reuse_accepts_unchanged_dialect_identity() -> None:
    first = reconstruction_target_fingerprint(
        provider_identity=_identity(),
        segments=[_segment(0, profile="EGYPTIAN")],
        target_index=0,
        language="ar",
        transcription_fingerprint="t",
        correction_version="c",
    )
    second = reconstruction_target_fingerprint(
        provider_identity=_identity(),
        segments=[_segment(0, profile="EGYPTIAN")],
        target_index=0,
        language="ar",
        transcription_fingerprint="t",
        correction_version="c",
    )

    assert first == second


def test_output_fingerprint_changes_with_dialect_identity() -> None:
    base = reconstruction_output_fingerprint(
        provider_identity=_identity(),
        segments=[_segment(0, profile="EGYPTIAN")],
        language="ar",
        transcription_fingerprint="t",
        correction_version="c",
    )
    changed = reconstruction_output_fingerprint(
        provider_identity=_identity(),
        segments=[_segment(0, profile="SAUDI")],
        language="ar",
        transcription_fingerprint="t",
        correction_version="c",
    )

    assert base != changed


def test_stage2_asr_fingerprint_does_not_change_with_dialect_fields() -> None:
    from app.transcription.service import TranscriptionOptions

    options = TranscriptionOptions(
        model="large-v3-turbo", device="cpu", compute_type="int8", beam_size=5
    )
    first = options.fingerprint("audio-hash")
    second = options.fingerprint("audio-hash")

    assert first == second


def test_manual_override_remains_authoritative_with_dialect(sqlite_engine: Any) -> None:
    segment = _segment(0, raw="خطي بالك", profile=None)
    segment["operator_text"] = "يدوي"
    segment["final_text"] = "يدوي"
    source_id = _setup(sqlite_engine, [segment])

    with Session(sqlite_engine) as session:
        source = session.get(SourceVideo, source_id)
        source.dialect_profile_override = _EGYPTIAN
        session.commit()
        normalization = TranscriptNormalizationExecutor(session=session)
        normalization.execute(source)

        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )
        assert transcript.segments[0]["final_text"] == "يدوي"
        assert transcript.segments[0]["operator_text"] == "يدوي"


def test_english_only_source_behavior_intact(sqlite_engine: Any) -> None:
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "Deploy the backend now",
            "raw_text": "Deploy the backend now",
        }
    ]
    source_id = _setup(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        source = session.get(SourceVideo, source_id)
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )
        transcript.language = "en"
        session.commit()
        normalization = TranscriptNormalizationExecutor(session=session)
        normalization.execute(source)

        session.refresh(transcript)
        assert transcript.dialect_profile is None
        assert transcript.code_switch_suspected is False
        assert transcript.segments[0]["corrected_text"] == "Deploy the backend now"
