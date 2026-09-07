from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.enums import PipelineRunStatus, PipelineStage, RightsStatus
from app.db.base import Base
from app.models import PipelineRun, SourceVideo, Transcript
from app.pipeline.executor import StageExecutionResult
from app.pipeline.fingerprints import canonical_fingerprint
from app.pipeline.runner import PipelineRunner
from app.pipeline.stages import ContextualReconstructionExecutor
from app.transcription.reconstruction.providers import (
    ReconstructionCandidate,
    ReconstructionRequest,
)
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
)


def test_canonical_fingerprint_is_order_independent() -> None:
    assert canonical_fingerprint("x", "1", {"b": 2, "a": 1}) == canonical_fingerprint(
        "x", "1", {"a": 1, "b": 2}
    )


class IdentityProvider:
    def __init__(self, identity: dict[str, object]) -> None:
        self._identity = dict(identity)

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE,
            str(self._identity.get("provider", "ollama")),
            str(self._identity.get("model")),
            None,
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


def _identity(model: str) -> dict[str, object]:
    return {
        "provider": "ollama",
        "model": model,
        "digest": "sha256:live",
        "prompt_hash": "prompt-hash",
        "schema_version": "schema-v1",
        "max_context_tokens": 4096,
        "output_tokens": 256,
        "chat_framing_reserve": 64,
        "safety_reserve": 128,
        "confidence_policy_version": "one-pass-provider-confidence-v1",
        "validation_version": "stage-2-7-validation-v1",
    }


def test_reconstruction_input_fingerprint_tracks_runtime_identity(
    sqlite_engine: object,
) -> None:
    """A model/digest/prompt/policy change invalidates the Stage 2.7 input fingerprint."""

    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri=f"file:///tmp/{uuid4()}.mp4",
            content_hash="content",
            rights_status=RightsStatus.OWNED,
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

        baseline = ContextualReconstructionExecutor(
            session=session,
            reconstructor=ContextualReconstructor(IdentityProvider(_identity("qwen3.5:4b"))),
        )
        changed_model = ContextualReconstructionExecutor(
            session=session,
            reconstructor=ContextualReconstructor(IdentityProvider(_identity("qwen3:8b"))),
        )
        changed_digest = ContextualReconstructionExecutor(
            session=session,
            reconstructor=ContextualReconstructor(
                IdentityProvider({**_identity("qwen3.5:4b"), "digest": "sha256:other"})
            ),
        )

        baseline_fingerprint = baseline.input_fingerprint(source)
        assert baseline_fingerprint
        assert baseline_fingerprint != changed_model.input_fingerprint(source)
        assert baseline_fingerprint != changed_digest.input_fingerprint(source)


class FingerprintExecutor:
    def __init__(self, value: str = "in") -> None:
        self.value = value
        self.calls = 0

    def input_fingerprint(self, source: SourceVideo) -> str:
        return self.value

    def execute(self, source: SourceVideo, *, force: bool = False) -> StageExecutionResult:
        self.calls += 1
        return StageExecutionResult(output_fingerprint=f"out-{self.calls}")


class NoFingerprintExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, source: SourceVideo) -> None:
        self.calls += 1


def test_historical_success_without_canonical_input_never_skips(sqlite_engine: object) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri=f"file:///tmp/{uuid4()}.mp4", rights_status=RightsStatus.OWNED
        )
        session.add(source)
        session.commit()
        session.add(
            PipelineRun(
                source_video_id=source.id,
                stage=PipelineStage.INGEST,
                status=PipelineRunStatus.SUCCEEDED,
            )
        )
        session.commit()
        executor = NoFingerprintExecutor()
        result = PipelineRunner(session, {PipelineStage.INGEST: executor}).run(
            source.id, PipelineStage.INGEST
        )
        assert result.skipped is False
        assert executor.calls == 1


def test_historical_legacy_input_never_skips(sqlite_engine: object) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri=f"file:///tmp/{uuid4()}.mp4", rights_status=RightsStatus.OWNED
        )
        session.add(source)
        session.commit()
        session.add(
            PipelineRun(
                source_video_id=source.id,
                stage=PipelineStage.INGEST,
                status=PipelineRunStatus.SUCCEEDED,
                input_fingerprint="legacy",
            )
        )
        session.commit()
        executor = FingerprintExecutor("legacy")
        assert (
            PipelineRunner(session, {PipelineStage.INGEST: executor})
            .run(source.id, PipelineStage.INGEST)
            .skipped
            is False
        )


def test_runner_skips_only_matching_input_and_force_increments_attempt(
    sqlite_engine: object,
) -> None:
    Base.metadata.create_all(sqlite_engine)
    with Session(sqlite_engine) as session:
        source = SourceVideo(
            source_uri=f"file:///tmp/{uuid4()}.mp4", rights_status=RightsStatus.OWNED
        )
        session.add(source)
        session.commit()
        executor = FingerprintExecutor()
        runner = PipelineRunner(session, {PipelineStage.INGEST: executor})
        first = runner.run(source.id, PipelineStage.INGEST)
        assert first.skipped is False
        assert runner.run(source.id, PipelineStage.INGEST).skipped is True
        forced = runner.run(source.id, PipelineStage.INGEST, force=True)
        assert forced.skipped is False
        runs = session.query(PipelineRun).order_by(PipelineRun.attempt).all()
        assert executor.calls == 2
        assert [r.attempt for r in runs] == [1, 2]
        assert all(r.status is PipelineRunStatus.SUCCEEDED for r in runs)
