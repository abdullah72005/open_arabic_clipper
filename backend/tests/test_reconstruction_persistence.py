import pytest
from sqlalchemy.orm import Session

from app.core.enums import JobKind, PipelineStage, RightsStatus
from app.db.base import Base
from app.models import SourceVideo, Transcript
from app.pipeline.stages import ContextualReconstructionExecutor
from app.transcription.reconstruction.confidence import CONFIDENCE_POLICY_VERSION
from app.transcription.reconstruction.ollama import OllamaReconstructionProvider
from app.transcription.reconstruction.providers import (
    OpenAICompatibleReconstructionProvider,
    ReconstructionCandidate,
    ReconstructionRequest,
)
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
)
from app.transcription.reconstruction.validation import VALIDATION_VERSION


def test_transcript_declares_separate_stage_2_7_derived_fields() -> None:
    """Reconstruction persistence cannot replace raw or Stage 2.5 transcript evidence."""

    columns = Transcript.__table__.c

    assert "contextual_reconstructed_text" in columns
    assert "reconstruction_fingerprint" in columns
    assert "reconstruction_metadata" in columns
    assert "reconstruction_status" in columns
    assert PipelineStage.CONTEXTUAL_RECONSTRUCTION.value == "CONTEXTUAL_RECONSTRUCTION"
    assert JobKind.RECONSTRUCTION.value == "RECONSTRUCTION"


def test_provider_runtime_identity_is_stable_and_complete() -> None:
    """Every output-affecting dependency participates in the runtime identity."""

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=3,
        max_context_tokens=4096,
        output_tokens=256,
        chat_framing_reserve=64,
        safety_reserve=128,
        request=lambda *_args: b"{}",
    )

    identity = provider.runtime_identity()

    assert identity["provider"] == "openai_compatible"
    assert identity["model"] == "qwen3.5:4b"
    assert identity["digest"] == "digest_unavailable"
    assert identity["prompt_hash"]
    assert identity["schema_version"]
    assert identity["max_context_tokens"] == 4096
    assert identity["output_tokens"] == 256
    assert identity["chat_framing_reserve"] == 64
    assert identity["safety_reserve"] == 128
    assert identity["confidence_policy_version"] == CONFIDENCE_POLICY_VERSION
    assert identity["validation_version"] == VALIDATION_VERSION
    assert provider.runtime_identity() == identity


def test_runtime_identity_uses_live_digest_from_models_endpoint() -> None:
    """A live digest replaces the digest_unavailable marker without faking a tag."""

    provider = OpenAICompatibleReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=3,
        request=lambda *_args: b'{"data":[{"id":"qwen3.5:4b","digest":"sha256:live"}]}',
    )

    provider.health()

    assert provider.runtime_identity()["digest"] == "sha256:live"


def test_ollama_runtime_identity_uses_live_tags_digest() -> None:
    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=3,
        release_after_run=False,
        request=lambda *_args: b'{"models":[{"name":"qwen3.5:4b","digest":"sha256:tags"}]}',
    )

    provider.health()

    assert provider.runtime_identity()["digest"] == "sha256:tags"


def _identity(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "provider": "ollama",
        "model": "qwen3.5:4b",
        "digest": "sha256:base",
        "prompt_hash": "prompt-hash",
        "schema_version": "schema-v1",
        "max_context_tokens": 4096,
        "output_tokens": 256,
        "chat_framing_reserve": 64,
        "safety_reserve": 128,
        "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
        "validation_version": VALIDATION_VERSION,
    }
    base.update(changes)
    return base


class IdentityProvider:
    def __init__(self, identity: dict[str, object]) -> None:
        self._identity = dict(identity)

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE,
            str(self._identity.get("provider", "ollama")),
            str(self._identity.get("model", "qwen3.5:4b")),
            str(self._identity.get("digest") or None),
            "ok",
        )

    def runtime_identity(self) -> dict[str, object]:
        return dict(self._identity)

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        return {}

    def release(self) -> None:
        pass


class UnavailableIdentityProvider(IdentityProvider):
    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.UNAVAILABLE,
            str(self._identity.get("provider", "ollama")),
            str(self._identity.get("model", "qwen3.5:4b")),
            None,
            "not installed",
        )


@pytest.mark.parametrize(
    "change",
    [
        {"model": "qwen3:8b"},
        {"digest": "sha256:other"},
        {"prompt_hash": "other-prompt"},
        {"schema_version": "schema-v2"},
        {"max_context_tokens": 2048},
        {"output_tokens": 512},
        {"chat_framing_reserve": 32},
        {"safety_reserve": 256},
        {"confidence_policy_version": "other-policy"},
        {"validation_version": "other-validation"},
    ],
)
def test_reconstruction_output_fingerprint_changes_with_each_identity_component(
    change: dict[str, object],
) -> None:
    segments = [{"start": 0.0, "end": 1.0, "text": "دخم", "corrected_text": "دخم"}]

    baseline = ContextualReconstructor(IdentityProvider(_identity())).reconstruct(
        segments,
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )
    changed = ContextualReconstructor(IdentityProvider(_identity(**change))).reconstruct(
        segments,
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert baseline.fingerprint != changed.fingerprint


def test_unavailable_run_fingerprint_cannot_collide_with_available_run() -> None:
    segments = [{"start": 0.0, "end": 1.0, "text": "دخم", "corrected_text": "دخم"}]

    available = ContextualReconstructor(
        IdentityProvider(_identity(digest="sha256:live"))
    ).reconstruct(
        segments,
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )
    unavailable = ContextualReconstructor(
        UnavailableIdentityProvider(_identity(digest="sha256:live"))
    ).reconstruct(
        segments,
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert available.fingerprint != unavailable.fingerprint


class CandidateProvider(IdentityProvider):
    def __init__(self, candidate: ReconstructionCandidate) -> None:
        super().__init__(_identity())
        self._candidate = candidate

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        return {request.segment_index: self._candidate for request in requests}


def test_executor_persists_actual_stage27_and_manual_only_affects_final(
    sqlite_engine: object,
) -> None:
    """Transcript reconstruction joins real Stage 2.7 output; manual text stays in final."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri="file:///tmp/source.mp4", content_hash="h", rights_status=RightsStatus.OWNED
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
            segments=[
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "دخم",
                    "raw_text": "دخم",
                    "corrected_text": "دخم",
                    "final_text": "دخم",
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "خطي",
                    "raw_text": "خطي",
                    "corrected_text": "تصحيح",
                    "operator_text": "يدوي",
                    "final_text": "يدوي",
                },
            ],
        )
        session.add(transcript)
        session.commit()

        provider = CandidateProvider(
            ReconstructionCandidate("provider-0", "ضخمة", provider_confidence=0.95)
        )
        executor = ContextualReconstructionExecutor(
            session=session, reconstructor=ContextualReconstructor(provider)
        )
        executor.execute(source, force=True)

        session.refresh(transcript)
        segment0 = transcript.segments[0]
        segment1 = transcript.segments[1]
        assert segment0["contextual_reconstructed_text"] == "ضخمة"
        assert segment0["final_text"] == "ضخمة"
        assert segment1["contextual_reconstructed_text"] == "تصحيح"
        assert segment1["final_text"] == "يدوي"
        assert transcript.contextual_reconstructed_text == "ضخمة تصحيح"
        assert transcript.final_text == "ضخمة يدوي"
        assert transcript.reconstruction_metadata["provider_available"] is True
        assert transcript.reconstruction_metadata["runtime_identity"]["model"] == "qwen3.5:4b"
