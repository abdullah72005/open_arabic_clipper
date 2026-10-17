"""Stage 6 acceptance export, audition, and production-profile import tests."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from app.core.enums import RenderArtifactPurpose, RenderExecutionLifecycle
from app.core.settings import get_settings
from app.db.base import Base
from app.materialization.acceptance_export import (
    discover_eligible_materializations,
    export_materializations,
    export_tts_audition,
)
from app.materialization.profiles import ProfileImportError, import_profile
from app.materialization.tts_audition import AUDITION_SCRIPTS, AUDITION_VOICES, run_audition
from app.models import ClipMaterialization, ProcessingJob
from app.models.render_execution import RenderExecution


@pytest.fixture  # type: ignore[untyped-decorator]
def engine() -> Iterator[Engine]:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _session() -> Session:
    return Session(create_engine(get_settings().database_url))


def test_profile_import_creates_new_current_revision(engine: Engine) -> None:
    with Session(engine) as session:
        first = import_profile(
            session,
            {
                "channel_key": "news-eg",
                "label": "Egyptian News",
                "target_market": "EGYPT",
                "narration_allowed": True,
                "tts_provider": "gemini-tts",
                "tts_voice": "Kore",
                "supporting_visual_mode": "OFF",
            },
        )
        assert first.revision == 1 and first.is_current is True

        second = import_profile(
            session,
            {"channel_key": "news-eg", "narration_allowed": True, "tts_voice": "Charon"},
        )
        assert second.revision == 2 and second.is_current is True
    with Session(engine) as session:
        from app.models.channel_production_profile import ChannelProductionProfile

        rows = session.scalars(
            select(ChannelProductionProfile).where(
                ChannelProductionProfile.channel_key == "news-eg"
            )
        ).all()
        assert sorted(row.revision for row in rows) == [1, 2]
        assert sum(1 for row in rows if row.is_current) == 1


def test_profile_import_rejects_unknown_and_invalid(engine: Engine) -> None:
    with Session(engine) as session:
        with pytest.raises(ProfileImportError):
            import_profile(session, {"channel_key": "x", "secret_key": "nope"})
        with pytest.raises(ProfileImportError):
            import_profile(session, {"channel_key": "x", "narration_allowed": "yes"})
        with pytest.raises(ProfileImportError):
            import_profile(session, {"label": "no channel key"})
        with pytest.raises(ProfileImportError):
            import_profile(session, {"channel_key": "x", "supporting_visual_mode": "MAYBE"})


def _seed_completed_materialization(engine: Engine, storage_relative: str) -> uuid.UUID:
    import hashlib

    settings = get_settings()
    artifact = settings.storage_root / storage_relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"\x00\x00\x00\x18ftypmp42synthetic")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    output_fingerprint = "a" * 64

    materialization_id = uuid.uuid4()
    render_id = uuid.uuid4()
    source_id = uuid.uuid4()
    candidate_id = uuid.uuid4()
    with Session(engine) as session:
        envelope = ClipMaterialization(
            id=materialization_id,
            source_video_id=source_id,
            clip_candidate_id=candidate_id,
            profile_key="default",
            lifecycle="COMPLETE",
            semantic_outcome="READY_FOR_HUMAN_REVIEW",
            human_review_required=True,
            publication_ready=False,
            is_current=True,
            output_fingerprint=output_fingerprint,
            render_execution_id=render_id,
            metrics={"assessment_fingerprint": "f" * 64, "final_integrated_lufs": -16.0},
            authored_blocks=[{"block_index": 0, "copy": "hi"}],
            call_ledger={
                "entries": [
                    {"attempt_id": "1", "status": "RESPONDED"},
                    {"attempt_id": "2", "status": "UNCERTAIN"},
                ]
            },
        )
        render = RenderExecution(
            id=render_id,
            source_video_id=source_id,
            clip_candidate_id=candidate_id,
            materialization_id=materialization_id,
            artifact_purpose=RenderArtifactPurpose.TRANSFORMED_SHORT,
            lifecycle=RenderExecutionLifecycle.COMPLETE,
            is_current=True,
            output_fingerprint="b" * 64,
            request_payload={
                "materialization_id": str(materialization_id),
                "expected_output_fingerprint": output_fingerprint,
                "spec": {"authored": {"repair_revision": 0}},
            },
            artifact_reference={"relative_path": storage_relative, "sha256": artifact_sha},
            qc_result={"measured": {"final_integrated_lufs": -16.0, "final_true_peak_dbtp": -2.0}},
        )
        session.add(envelope)
        session.add(render)
        session.add(
            ProcessingJob(
                source_video_id=source_id,
                kind="MATERIALIZATION",
                materialization_id=materialization_id,
                status="SUCCEEDED",
            )
        )
        session.commit()
    return materialization_id


def test_acceptance_export_writes_index_and_manifest(engine: Engine) -> None:
    materialization_id = _seed_completed_materialization(engine, "sources/s1/final.mp4")
    settings = get_settings()
    with Session(engine) as session:
        result = export_materializations(session, settings, [materialization_id])
        assert result["cases"]
        exported = Path(result["cases"][0]["directory"])
    assert (exported / "index.json").is_file()
    assert (exported / "SHA256SUMS").is_file()
    assert (exported / "final.mp4").is_file()
    index = json.loads((exported / "index.json").read_text(encoding="utf-8"))
    assert index["materialization_id"] == str(materialization_id)
    assert index["provider_attempts"]["reserved"] == 2
    assert index["provider_attempts"]["uncertain"] == 1
    manifest = (exported / "SHA256SUMS").read_text(encoding="utf-8")
    assert "final.mp4" in manifest and "index.json" in manifest
    catalog = (
        settings.storage_root / "benchmarks" / "stage-6" / "manual-acceptance" / "catalog.json"
    )
    assert catalog.is_file()


def test_discover_eligible_returns_review_ready_only(engine: Engine) -> None:
    _seed_completed_materialization(engine, "sources/s1/final.mp4")
    with Session(engine) as session:
        eligible = discover_eligible_materializations(session, limit=4)
    # Seeded lifecycle/outcome labels alone never establish genuine readiness: the
    # synthetic fixture has no bound authored ASS/asset hashes, no render
    # identity/QC fingerprints, and no live prerequisites, so nothing is eligible.
    assert eligible == []


class _FakeVoiceProvider:
    provider_name = "fake-voice"

    def __init__(self, voice: str) -> None:
        self.voice = voice
        self.calls = 0
        self._persist = None

    def available(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": self.provider_name, "voice": self.voice, "model": "fake"}

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


def test_audition_runs_at_most_nine_calls_then_caches(engine: Engine) -> None:
    from app.services.storage import StorageService

    settings = get_settings()
    storage = StorageService(settings.storage_root)
    providers: dict[str, _FakeVoiceProvider] = {}

    def factory(voice: str):
        providers[voice] = _FakeVoiceProvider(voice)
        return providers[voice]

    def accept(_audio: object) -> dict[str, object]:
        return {"status": "ACCEPTED", "findings": []}

    result = run_audition(
        settings=settings,
        storage=storage,
        model="fake-tts",
        voice_factory=factory,
        fidelity=accept,
    )
    assert len(result.samples) == len(AUDITION_VOICES) * len(AUDITION_SCRIPTS)
    assert result.raw_calls == 9
    assert result.provisional_voice == "Charon"

    providers.clear()
    cached = run_audition(
        settings=settings,
        storage=storage,
        model="fake-tts",
        voice_factory=factory,
        fidelity=accept,
    )
    assert cached.raw_calls == 0
    assert all(sample.cached for sample in cached.samples)


def test_export_tts_audition_writes_index(engine: Engine) -> None:
    settings = get_settings()
    result = export_tts_audition(
        settings,
        [{"name": "Kore-egyptian", "relative_path": "", "voice": "Kore"}],
    )
    index = Path(result["root"]) / "index.json"
    assert index.is_file()
    payload = json.loads(index.read_text(encoding="utf-8"))
    assert payload["samples"][0]["voice"] == "Kore"
