import json

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
from app.transcription.reconstruction.routing import (
    AdaptiveRoutingConfig,
    RoutingMode,
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

    def refresh_runtime_identity(self) -> dict[str, object]:
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


def test_unavailable_run_shares_stable_fingerprint_but_is_not_cache_eligible() -> None:
    """Availability is execution state, not identity; a degraded run stays retryable."""

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

    assert available.fingerprint == unavailable.fingerprint
    assert available.metadata["cache_eligible"] is True
    assert unavailable.metadata["cache_eligible"] is False


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


class LiveDigestTransport:
    def __init__(self, digest: str) -> None:
        self.digest = digest
        self.urls: list[str] = []

    def __call__(
        self, method: str, url: str, body: bytes | None, headers: dict[str, str], timeout: float
    ) -> bytes:
        self.urls.append(url)
        if url.endswith("/api/tags"):
            return json.dumps({"models": [{"name": "qwen3.5:4b", "digest": self.digest}]}).encode()
        if url.endswith("/v1/chat/completions"):
            chat_body = json.loads(body)
            targets = json.loads(chat_body["messages"][1]["content"])["targets"]
            content = {
                "reconstructions": [
                    {
                        "segment_id": target["segment_id"],
                        "corrected_text": target["corrected_text"],
                        "unchanged": True,
                    }
                    for target in targets
                ]
            }
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}
            ).encode()
        return b"{}"


def _ollama_provider(transport: LiveDigestTransport) -> OllamaReconstructionProvider:
    return OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=3,
        release_after_run=False,
        request=transport,
    )


def test_same_tag_digest_replacement_invalidates_output_fingerprint() -> None:
    """A model replaced under the same tag changes the output fingerprint immediately."""

    transport = LiveDigestTransport("sha256:old")
    provider = _ollama_provider(transport)
    segments = [{"start": 0.0, "end": 1.0, "text": "دخم", "corrected_text": "دخم"}]

    first = ContextualReconstructor(provider).reconstruct(
        segments,
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )
    transport.digest = "sha256:new"
    second = ContextualReconstructor(provider).reconstruct(
        segments,
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )

    assert first.metadata["runtime_identity"]["digest"] == "sha256:old"
    assert second.metadata["runtime_identity"]["digest"] == "sha256:new"
    assert first.fingerprint != second.fingerprint


def test_executor_input_fingerprint_refreshes_live_digest(sqlite_engine: object) -> None:
    """The pipeline reads the live digest before deciding whether Stage 2.7 is current."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri="file:///tmp/source.mp4", content_hash="h", rights_status=RightsStatus.OWNED
        )
        session.add(source)
        session.commit()
        session.add(
            Transcript(
                source_video_id=source.id,
                whisper_model="large-v3-turbo",
                input_fingerprint="asr-fp",
                normalization_fingerprint="norm-fp",
                transcription_revision=1,
                correction_version="egyptian-ar-v1",
            )
        )
        session.commit()

        transport = LiveDigestTransport("sha256:old")
        executor = ContextualReconstructionExecutor(
            session=session, reconstructor=ContextualReconstructor(_ollama_provider(transport))
        )
        fp_old = executor.input_fingerprint(source)
        transport.digest = "sha256:new"
        fp_new = executor.input_fingerprint(source)

        assert fp_old != fp_new


def test_forced_reconstruction_run_uses_refreshed_digest(sqlite_engine: object) -> None:
    """A forced rerun after a same-tag model replacement persists the new digest."""

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
                }
            ],
        )
        session.add(transcript)
        session.commit()

        transport = LiveDigestTransport("sha256:old")
        executor = ContextualReconstructionExecutor(
            session=session, reconstructor=ContextualReconstructor(_ollama_provider(transport))
        )
        executor.execute(source, force=True)
        session.refresh(transcript)
        assert transcript.reconstruction_metadata["runtime_identity"]["digest"] == "sha256:old"

        transport.digest = "sha256:new"
        executor.execute(source, force=True)
        session.refresh(transcript)
        assert transcript.reconstruction_metadata["runtime_identity"]["digest"] == "sha256:new"


class CountingGemini:
    def __init__(self) -> None:
        self.model = "gemini-3.6-flash"
        self.calls = 0

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "gemini", "gemini-3.6-flash", "sha256:g", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "gemini",
            "model": "gemini-3.6-flash",
            "digest": "sha256:g",
            "prompt_hash": "p",
            "schema_version": "s",
            "timeout_seconds": 30.0,
            "retry_attempts": 0,
            "retry_backoff_seconds": 0.0,
            "max_output_tokens": 256,
            "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
            "validation_version": VALIDATION_VERSION,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def release(self) -> None:
        pass

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        return {
            request.segment_index: ReconstructionCandidate(
                "provider-0", "ضخمة", provider_confidence=1.0
            )
            for request in requests
        }

    def usage_summary(self) -> dict[str, int]:
        return {"prompt_token_count": 1, "candidates_token_count": 1, "total_token_count": 2}


def test_executor_cache_hit_avoids_duplicate_gemini_call(sqlite_engine: object) -> None:
    """Completed identical work reuses the fingerprint and never calls Gemini again."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri="file:///tmp/source.mp4", content_hash="h", rights_status=RightsStatus.OWNED
        )
        session.add(source)
        session.commit()
        hard_words = [
            {"word": "م", "probability": 0.30},
            {"word": "ش", "probability": 0.25},
            {"word": "قادر", "probability": 0.20},
            {"word": "يفهم", "probability": 0.90},
        ]
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
                    "text": "مش قادر يفهم",
                    "raw_text": "مش قادر يفهم",
                    "corrected_text": "مش قادر يفهم",
                    "words": hard_words,
                }
            ],
        )
        session.add(transcript)
        session.commit()

        gemini = CountingGemini()
        reconstructor = ContextualReconstructor(
            CandidateProvider(
                ReconstructionCandidate("provider-0", "ضخمة", provider_confidence=1.0)
            ),
            gemini_provider=gemini,
            routing=AdaptiveRoutingConfig(mode=RoutingMode.ADAPTIVE),
            gemini_budget=5,
        )
        executor = ContextualReconstructionExecutor(session=session, reconstructor=reconstructor)
        executor.execute(source, force=True)
        assert gemini.calls == 1
        executor.execute(source, force=False)
        assert gemini.calls == 1


class FlappyGemini(CountingGemini):
    def __init__(self) -> None:
        super().__init__()
        self.available = True

    def health(self) -> ProviderHealth:
        if not self.available:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE, "gemini", "gemini-3.8-flash", None, "offline"
            )
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "gemini", "gemini-3.8-flash", "sha256:g", "ok"
        )


def _hard_segment() -> dict[str, object]:
    return {
        "start": 0.0,
        "end": 1.0,
        "text": "دخم",
        "raw_text": "دخم",
        "corrected_text": "دخم",
        "words": [
            {"word": "م", "probability": 0.30},
            {"word": "ش", "probability": 0.25},
            {"word": "قادر", "probability": 0.20},
            {"word": "يفهم", "probability": 0.90},
        ],
    }


def test_successful_gemini_result_reused_during_mocked_outage(sqlite_engine: object) -> None:
    """A temporary outage never overwrites accepted Gemini output nor re-generates."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri="file:///tmp/source.mp4", content_hash="h", rights_status=RightsStatus.OWNED
        )
        session.add(source)
        session.commit()
        session.add(
            Transcript(
                source_video_id=source.id,
                whisper_model="large-v3-turbo",
                input_fingerprint="asr-fp",
                normalization_fingerprint="norm-fp",
                transcription_revision=1,
                correction_version="egyptian-ar-v1",
                language="ar",
                segments=[_hard_segment()],
            )
        )
        session.commit()

        gemini = FlappyGemini()
        reconstructor = ContextualReconstructor(
            CandidateProvider(
                ReconstructionCandidate("provider-0", "ضخمة", provider_confidence=1.0)
            ),
            gemini_provider=gemini,
            routing=AdaptiveRoutingConfig(mode=RoutingMode.ADAPTIVE),
            gemini_budget=5,
        )
        executor = ContextualReconstructionExecutor(session=session, reconstructor=reconstructor)
        executor.execute(source, force=True)
        session.refresh(source)
        assert gemini.calls == 1
        accepted_text = source.transcript.contextual_reconstructed_text
        assert accepted_text == "ضخمة"

        gemini.available = False
        executor.execute(source, force=False)
        session.refresh(source)
        assert gemini.calls == 1  # no duplicate generation during outage
        assert source.transcript.contextual_reconstructed_text == accepted_text


def test_transient_first_run_fallback_is_retried_after_recovery(sqlite_engine: object) -> None:
    """A first-run degraded fallback is not cached forever once the provider recovers."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri="file:///tmp/source.mp4", content_hash="h", rights_status=RightsStatus.OWNED
        )
        session.add(source)
        session.commit()
        session.add(
            Transcript(
                source_video_id=source.id,
                whisper_model="large-v3-turbo",
                input_fingerprint="asr-fp",
                normalization_fingerprint="norm-fp",
                transcription_revision=1,
                correction_version="egyptian-ar-v1",
                language="ar",
                segments=[_hard_segment()],
            )
        )
        session.commit()

        gemini = FlappyGemini()
        gemini.available = False
        reconstructor = ContextualReconstructor(
            CandidateProvider(
                ReconstructionCandidate("provider-0", "ضخمة", provider_confidence=1.0)
            ),
            gemini_provider=gemini,
            routing=AdaptiveRoutingConfig(mode=RoutingMode.ADAPTIVE),
            gemini_budget=5,
        )
        executor = ContextualReconstructionExecutor(session=session, reconstructor=reconstructor)
        executor.execute(source, force=True)
        session.refresh(source)
        assert gemini.calls == 0
        assert source.transcript.reconstruction_metadata["cache_eligible"] is False

        gemini.available = True
        executor.execute(source, force=False)
        session.refresh(source)
        assert gemini.calls == 1  # degraded first run retried after recovery
        assert source.transcript.reconstruction_metadata["cache_eligible"] is True
        assert source.transcript.contextual_reconstructed_text == "ضخمة"
        executor.execute(source, force=False)
        assert gemini.calls == 1
