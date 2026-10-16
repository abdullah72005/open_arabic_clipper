"""V4 corrections for SOL-reopened Stage 6 root causes.

These assert the corrected behavior for:

* committed-assessment review readiness (lifecycle/outcome/flag separation and
  UNKNOWN freshness on an unavailable verdict);
* id-scoped cancel/resume generation authority (a historical id never resolves
  into a newer current generation);
* exact reverse-bound artifact verification (corrupt/wrong-purpose/pending
  media is never served);
* the effective synthesizer runtime identity participating in narration reuse.

They replace the temporary diagnostic probe and never assert a defect.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from test_stage6_api_cli_acceptance import _seed_completed_materialization
from test_stage6_component_recovery import component_fixture as component_fixture  # noqa: F401

from app.api.app import create_app
from app.core.enums import (
    CandidateDisposition,
    ContentType,
    JobKind,
    MaterializationLifecycle,
    MaterializationOutcome,
    OriginalityRisk,
    RenderArtifactPurpose,
    RenderExecutionLifecycle,
    RightsRisk,
)
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.finalization import (
    ASSESSMENT_POLICY_VERSION,
    _assessment_fingerprint,
)
from app.materialization.queue import (
    MaterializationQueueError,
    cancel_materialization_by_id,
    resume_materialization_by_id,
)
from app.materialization.readiness import (
    _verify_committed_assessment,
    evaluate_readiness,
)
from app.models import ClipCandidate, ClipMaterialization, ProcessingJob, SourceVideo
from app.models.render_execution import RenderExecution
from app.services.storage import StorageService


class _NoopDispatcher:
    def dispatch(self, *_args: object, **_kwargs: object) -> None:
        return None


@pytest.fixture
def harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, sessionmaker[Session], object]]:
    monkeypatch.setattr(
        "app.workers.tasks.run_materialization_job.delay",
        lambda *a, **k: None,
    )
    settings = get_settings()
    storage_root = settings.storage_root
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    app = create_app(
        session_factory=factory,
        storage=StorageService(storage_root),
        dispatcher=_NoopDispatcher(),
    )
    app.state.session_factory = factory
    with TestClient(app) as client:
        yield client, factory, engine
    engine.dispose()


# --- Section 1: committed-assessment review readiness ----------------------


def _readiness(engine: object, materialization_id: uuid.UUID):
    with Session(engine) as session:  # type: ignore[arg-type]
        row = session.get(ClipMaterialization, materialization_id, populate_existing=True)
        assert row is not None
        return evaluate_readiness(
            session, StorageService(get_settings().storage_root), row
        )


def test_readiness_requires_committed_assessment(harness) -> None:
    _client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/committed.mp4")
    view = _readiness(engine, mid)
    # Seeded lifecycle/outcome labels and a bare fingerprint are not a committed
    # publication: the case can never be review-ready.
    assert view.review_ready is False
    assert view.reasons


def test_committed_assessment_fingerprint_binding(harness) -> None:
    _client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/fingerprint.mp4")

    class _Assessment:
        outcome = MaterializationOutcome.READY_FOR_HUMAN_REVIEW.value
        reason_codes: list[str] = []

    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, mid, populate_existing=True)
        assert envelope is not None
        render = session.get(RenderExecution, envelope.render_execution_id)
        assert render is not None
        measured = render.qc_result["measured"]
        expected = _assessment_fingerprint(envelope, render, measured, _Assessment())
        envelope.metrics = {
            "assessment_fingerprint": expected,
            "assessment_policy_version": ASSESSMENT_POLICY_VERSION,
            "render_output_fingerprint": render.output_fingerprint,
        }
        reasons: list[str] = []
        assert _verify_committed_assessment(envelope, render, _Assessment(), reasons) == expected
        assert reasons == []
        # A policy-version mismatch is never a committed publication.
        envelope.metrics["assessment_policy_version"] = "stage6-assessment-v1"
        mismatch: list[str] = []
        assert _verify_committed_assessment(envelope, render, _Assessment(), mismatch) == ""
        assert "ASSESSMENT_POLICY_MISMATCH" in mismatch
        # A fingerprint over different evidence is refused.
        envelope.metrics["assessment_policy_version"] = ASSESSMENT_POLICY_VERSION
        envelope.metrics["assessment_fingerprint"] = "x" * 64
        wrong: list[str] = []
        assert _verify_committed_assessment(envelope, render, _Assessment(), wrong) == ""
        assert "ASSESSMENT_FINGERPRINT_MISMATCH" in wrong


def test_readiness_rejects_awaiting_render_without_assessment(harness) -> None:
    _client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/awaiting.mp4")
    with Session(engine) as session:
        row = session.get(ClipMaterialization, mid)
        assert row is not None
        row.lifecycle = MaterializationLifecycle.AWAITING_RENDER
        row.human_review_required = False
        row.semantic_outcome = None
        row.metrics = {}
        row.cache_eligible = False
        session.commit()
    view = _readiness(engine, mid)
    assert view.review_ready is False
    assert "LIFECYCLE_NOT_COMPLETE" in view.reasons
    assert "OUTCOME_NOT_REVIEWABLE" in view.reasons


def test_readiness_rejects_cancelled_generation(harness) -> None:
    _client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/cancelled.mp4")
    with Session(engine) as session:
        row = session.get(ClipMaterialization, mid)
        assert row is not None
        row.lifecycle = MaterializationLifecycle.CANCELLED
        row.human_review_required = False
        session.commit()
    view = _readiness(engine, mid)
    assert view.review_ready is False
    assert "LIFECYCLE_NOT_COMPLETE" in view.reasons


def test_readiness_historical_is_not_effective(harness) -> None:
    _client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/historical.mp4")
    with Session(engine) as session:
        row = session.get(ClipMaterialization, mid)
        assert row is not None
        row.is_current = False
        session.commit()
    view = _readiness(engine, mid)
    assert view.review_ready is False
    assert view.freshness == "HISTORICAL"
    assert view.is_effective is False


def test_readiness_unavailable_reports_unknown_freshness(
    harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/unavailable.mp4")

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("readiness unavailable")

    monkeypatch.setattr("app.materialization.readiness.evaluate_readiness", _boom)
    response = client.get(f"/api/materializations/{mid}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["live_freshness"] == "UNKNOWN"
    assert payload["is_effective"] is False
    assert payload["readiness"]["review_ready"] is False


# --- Section 3: exact reverse-bound artifact verification ------------------


def _artifact_path(engine: object, materialization_id: uuid.UUID) -> Path:
    with Session(engine) as session:  # type: ignore[arg-type]
        row = session.get(ClipMaterialization, materialization_id)
        assert row is not None
        render = session.get(RenderExecution, row.render_execution_id)
        assert render is not None
        return get_settings().storage_root / render.artifact_reference["relative_path"]


def test_artifact_valid_current_is_served_with_metadata(harness) -> None:
    client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/served.mp4")
    response = client.get(f"/api/materializations/{mid}/artifact")
    assert response.status_code == 200
    assert response.headers["x-clipfactory-artifact-historical"] == "false"


def test_artifact_corrupted_is_not_served(harness) -> None:
    client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/corrupt.mp4")
    path = _artifact_path(engine, mid)
    path.write_bytes(b"corrupted bytes not the accepted artifact")
    assert client.get(f"/api/materializations/{mid}/artifact").status_code == 404
    detail = client.get(f"/api/materializations/{mid}").json()
    assert detail["artifact_available"] is False


def test_artifact_wrong_purpose_is_not_served(harness) -> None:
    client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/purpose.mp4")
    with Session(engine) as session:
        row = session.get(ClipMaterialization, mid)
        assert row is not None
        render = session.get(RenderExecution, row.render_execution_id)
        assert render is not None
        render.artifact_purpose = RenderArtifactPurpose.CORE_SOURCE_VALIDATION
        session.commit()
    assert client.get(f"/api/materializations/{mid}/artifact").status_code == 404


def test_artifact_pending_render_is_not_served(harness) -> None:
    client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/pending.mp4")
    with Session(engine) as session:
        row = session.get(ClipMaterialization, mid)
        assert row is not None
        render = session.get(RenderExecution, row.render_execution_id)
        assert render is not None
        render.lifecycle = RenderExecutionLifecycle.RENDERING
        session.commit()
    assert client.get(f"/api/materializations/{mid}/artifact").status_code == 404


def test_artifact_historical_is_inspectable_with_explicit_metadata(harness) -> None:
    client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/historical-media.mp4")
    with Session(engine) as session:
        row = session.get(ClipMaterialization, mid)
        assert row is not None
        row.is_current = False
        session.commit()
    response = client.get(f"/api/materializations/{mid}/artifact")
    assert response.status_code == 200
    assert response.headers["x-clipfactory-artifact-historical"] == "true"


# --- Section 2: id-scoped cancel/resume generation authority ---------------


def _seed_generations(
    engine: object, *, current_lifecycle: MaterializationLifecycle
) -> tuple[uuid.UUID, uuid.UUID]:
    """Seed a historical and a current envelope for one candidate."""

    historical_id = uuid.uuid4()
    current_id = uuid.uuid4()
    with Session(engine) as session:  # type: ignore[arg-type]
        source = SourceVideo(source_uri="/tmp/rcV4.mp4", content_hash=f"rcV4-{uuid.uuid4()}")
        session.add(source)
        session.flush()
        candidate = ClipCandidate(
            source_video_id=source.id,
            candidate_key=f"rcV4-{uuid.uuid4()}",
            disposition=CandidateDisposition.CANDIDATE,
            start_time=0.0,
            end_time=10.0,
            start_segment_index=0,
            end_segment_index=0,
            primary_content_type=ContentType.STORY,
            rights_risk=RightsRisk.LOW,
            originality_risk=OriginalityRisk.NOT_INDICATED,
        )
        session.add(candidate)
        session.flush()
        session.add(
            ClipMaterialization(
                id=historical_id,
                source_video_id=source.id,
                clip_candidate_id=candidate.id,
                profile_key="default",
                lifecycle=MaterializationLifecycle.COMPLETE,
                semantic_outcome=MaterializationOutcome.READY_FOR_HUMAN_REVIEW,
                human_review_required=True,
                publication_ready=False,
                cache_eligible=False,
                is_current=False,
                input_fingerprint="hist-input",
                output_fingerprint="hist-output",
            )
        )
        session.add(
            ClipMaterialization(
                id=current_id,
                source_video_id=source.id,
                clip_candidate_id=candidate.id,
                profile_key="default",
                lifecycle=current_lifecycle,
                semantic_outcome=None,
                human_review_required=False,
                publication_ready=False,
                cache_eligible=False,
                is_current=True,
                input_fingerprint="cur-input",
                output_fingerprint="cur-output",
            )
        )
        session.commit()
    return historical_id, current_id


def test_cancel_historical_id_is_refused_without_mutation(harness) -> None:
    client, _factory, engine = harness
    historical_id, current_id = _seed_generations(
        engine, current_lifecycle=MaterializationLifecycle.COMPLETE
    )
    response = client.post(f"/api/materializations/{historical_id}/cancel")
    assert response.status_code == 409
    with Session(engine) as session:
        historical = session.get(ClipMaterialization, historical_id)
        current = session.get(ClipMaterialization, current_id)
        assert historical is not None and historical.lifecycle is MaterializationLifecycle.COMPLETE
        assert current is not None and current.lifecycle is MaterializationLifecycle.COMPLETE


def test_cancel_current_id_is_idempotent(harness) -> None:
    client, _factory, engine = harness
    _historical_id, current_id = _seed_generations(
        engine, current_lifecycle=MaterializationLifecycle.COMPLETE
    )
    first = client.post(f"/api/materializations/{current_id}/cancel")
    assert first.status_code == 200
    assert first.json()["lifecycle"] == "CANCELLED"
    second = client.post(f"/api/materializations/{current_id}/cancel")
    assert second.status_code == 200
    assert second.json()["lifecycle"] == "CANCELLED"


def test_resume_historical_id_is_refused(harness) -> None:
    client, _factory, engine = harness
    historical_id, current_id = _seed_generations(
        engine, current_lifecycle=MaterializationLifecycle.CANCELLED
    )
    response = client.post(f"/api/materializations/{historical_id}/resume")
    assert response.status_code == 409
    with Session(engine) as session:
        current = session.get(ClipMaterialization, current_id)
        assert current is not None
        assert current.lifecycle is MaterializationLifecycle.CANCELLED


def test_resume_cancelled_current_creates_new_fenced_job(harness) -> None:
    client, _factory, engine = harness
    _historical_id, current_id = _seed_generations(
        engine, current_lifecycle=MaterializationLifecycle.CANCELLED
    )
    response = client.post(f"/api/materializations/{current_id}/resume")
    assert response.status_code == 202
    payload = response.json()
    assert payload["materialization_id"] == str(current_id)
    assert payload["queued"] is True
    assert payload["job_id"] is not None
    with Session(engine) as session:
        jobs = session.scalars(
            select(ProcessingJob).where(ProcessingJob.materialization_id == current_id)
        ).all()
        assert any(job.kind is JobKind.MATERIALIZATION for job in jobs)
        row = session.get(ClipMaterialization, current_id)
        assert row is not None and row.lifecycle is not MaterializationLifecycle.CANCELLED


def test_narration_effective_synthesizer_identity_invalidates_reuse(component_fixture) -> None:
    _engine, session, inputs, owner = component_fixture
    from test_stage6_component_recovery import (
        CountingNarrationSynthesizer,
        _context,
    )
    from test_stage6_tts_boundary import _request as _tts_request

    from app.materialization.components import (
        CheckpointedNarrationSynthesizer,
        ComponentProviders,
        DurableRequestGateway,
    )
    from app.materialization.providers import DeterministicWritingProvider

    class _ModelSynth(CountingNarrationSynthesizer):
        def __init__(self, model: str) -> None:
            super().__init__()
            self._model = model

        def runtime_identity(self) -> dict[str, object]:
            return {"provider": "counting-tts", "model": self._model, "api_version": "v1"}

    synth_a, synth_b = _ModelSynth("A"), _ModelSynth("B")
    context = _context(
        session, owner, inputs, ComponentProviders(writing=DeterministicWritingProvider())
    )
    request = _tts_request()
    validator = lambda _audio: {"status": "ACCEPTED"}  # noqa: E731
    CheckpointedNarrationSynthesizer(
        synth_a, DurableRequestGateway(context), inputs, validator=validator
    ).synthesize(request)
    CheckpointedNarrationSynthesizer(
        synth_b, DurableRequestGateway(context), inputs, validator=validator
    ).synthesize(request)
    # A changed effective synthesizer runtime identity is never silently served
    # the other model's accepted audio.
    assert len(synth_a.calls) == 1
    assert len(synth_b.calls) == 1


class _AuditionVoice:
    provider_name = "fake-voice"

    def __init__(self, voice: str) -> None:
        self.voice = voice
        self.calls = 0
        self._persist = None

    def available(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "fake-voice", "voice": self.voice, "model": "fake"}

    def synthesize(self, request):  # type: ignore[no-untyped-def]
        import struct

        from app.materialization.narration import wrap_pcm_as_wav

        self.calls += 1
        frames = 2400
        pcm = struct.pack(f"<{frames}h", *([500] * frames))
        wav = wrap_pcm_as_wav(pcm, sample_rate=24_000, channels=1)
        relative, sha = self._persist(request.chunk_id, wav)  # type: ignore[misc]
        return type(
            "_Audio",
            (),
            {
                "relative_path": relative,
                "sha256": sha,
                "duration_seconds": 0.1,
                "sample_rate": 24_000,
                "channels": 1,
                "text": request.text,
            },
        )()

    def release(self) -> None:
        return None


def test_audition_unavailable_does_not_poison_manifest(tmp_path: Path) -> None:
    import json

    from app.materialization.tts_audition import run_audition

    storage = StorageService(tmp_path / "storage")
    settings = get_settings()

    def accept(_audio: object) -> dict[str, object]:
        return {"status": "ACCEPTED"}

    # No provider available: zero synthesis attempts and no durable attempt entry.
    first = run_audition(
        settings=settings,
        storage=storage,
        model="fake-tts",
        voice_factory=lambda voice: None,
        fidelity=accept,
    )
    assert first.raw_calls == 0
    assert first.provisional_voice is None
    manifest = storage.storage_root / "benchmarks" / "stage-6" / "tts-audition" / "ledger.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["attempts"] == []
    # Configuring a provider afterwards still runs the full audition.
    second = run_audition(
        settings=settings,
        storage=storage,
        model="fake-tts",
        voice_factory=lambda voice: _AuditionVoice(voice),
        fidelity=accept,
    )
    assert second.raw_calls == 9
    assert second.provisional_voice == "Charon"


def test_audition_deferred_fidelity_is_revalidated_on_reuse(tmp_path: Path) -> None:
    from app.materialization.tts_audition import run_audition

    storage = StorageService(tmp_path / "storage")
    settings = get_settings()

    def deferred(_audio: object) -> dict[str, object]:
        return {"status": "DEFERRED"}

    first = run_audition(
        settings=settings,
        storage=storage,
        model="fake-tts",
        voice_factory=lambda voice: _AuditionVoice(voice),
        fidelity=deferred,
    )
    assert first.raw_calls == 9
    assert first.provisional_voice is None
    revalidated: list[str] = []

    def accept(audio: object) -> dict[str, object]:
        revalidated.append(str(getattr(audio, "relative_path", "")))
        return {"status": "ACCEPTED"}

    second = run_audition(
        settings=settings,
        storage=storage,
        model="fake-tts",
        voice_factory=lambda voice: _AuditionVoice(voice),
        fidelity=accept,
    )
    # Cached samples are no longer permanently deferred: the local validator runs
    # again over the verified audio without any new synthesis call.
    assert second.raw_calls == 0
    assert len(revalidated) == 9
    assert second.provisional_voice == "Charon"


def test_export_rejects_artifact_bytes_mismatch(harness) -> None:
    import json

    from app.materialization.acceptance_export import export_materializations

    _client, _factory, engine = harness
    mid = _seed_completed_materialization(engine, "sources/rcV4/export-mismatch.mp4")
    _artifact_path(engine, mid).write_bytes(b"tampered after acceptance")
    settings = get_settings()
    with Session(engine) as session:
        exported = export_materializations(session, settings, [mid])
    case = exported["cases"][0]
    assert case["genuine_acceptance"] is False
    index = json.loads((Path(case["directory"]) / "index.json").read_text(encoding="utf-8"))
    assert "EXPORT_ARTIFACT_MISMATCH" in index["genuine_acceptance_reasons"]
    assert index["genuine_acceptance"] is False


def test_cancel_by_id_service_refuses_historical(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'service.sqlite3'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        historical_id, current_id = _seed_generations(
            engine, current_lifecycle=MaterializationLifecycle.COMPLETE
        )
        with pytest.raises(MaterializationQueueError):
            cancel_materialization_by_id(session, historical_id)
        with pytest.raises(MaterializationQueueError):
            resume_materialization_by_id(session, historical_id)
        current = session.get(ClipMaterialization, current_id)
        assert current is not None and current.lifecycle is MaterializationLifecycle.COMPLETE
    engine.dispose()
