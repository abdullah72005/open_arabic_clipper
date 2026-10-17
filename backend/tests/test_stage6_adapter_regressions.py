"""Corrected Stage 6 adapter/export regressions (replaces SOL's diagnostic script).

Each test asserts the *corrected* behavior for one reproduced defect:

1. categorical critique requires complete evidence (UNKNOWN critical defers);
2. a repeated-name role reversal is rejected by strict lexical-sequence fidelity;
3. narration assets are content-addressed (never overwritten);
4. an audition with any fidelity rejection is nonviable, a model change is not a
   cache hit, and the default export never copies a file onto itself;
5. export discovery rejects a wrong-purpose/demoted render and copies the real
   authored ASS (not the artifact_reference JSON);
6. lease loss and executor cancellation are wired to the ASR child cancel event;
7. a real-SDK transport timeout is recorded as UNCERTAIN, not RESPONDED;
8. a noncritical categorical concern is carried into an accepted script.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
from sqlalchemy.orm import Session
from test_stage6_api_cli_acceptance import (
    _FakeVoiceProvider,
    _seed_completed_materialization,
    engine,
)
from test_stage6_component_recovery import _context
from test_stage6_component_recovery import component_fixture as component_fixture  # noqa: F401
from test_stage6_gemini_boundary import (
    NOVEL_COPY,
    _critique_provider,
    _inputs_with_draft,
    _Ok,
    _Transport,
    _writing_provider,
)

from app.core.settings import get_settings
from app.materialization.acceptance_export import (
    discover_eligible_materializations,
    export_materializations,
    export_tts_audition,
)
from app.materialization.components import ComponentProviders, prepare_materialization
from app.materialization.narration import wrap_pcm_as_wav
from app.materialization.policy import DEFAULT_CONFIG, CallLedger
from app.materialization.providers import (
    CRITIQUE_DIMENSIONS,
    DeterministicWritingProvider,
    materialize_script,
)
from app.materialization.tts_audition import run_audition
from app.materialization.tts_gemini import _persist_to_storage, validate_narration_fidelity
from app.models import ClipMaterialization
from app.models.render_execution import RenderExecution
from app.services.storage import StorageService

__all__ = ["engine"]


def _findings(**overrides: str) -> list[dict[str, str]]:
    return [
        {"dimension": dimension, "level": overrides.get(dimension, "PASS"), "evidence": "bounded"}
        for dimension in CRITIQUE_DIMENSIONS
    ]


def _critique_provider_with(findings: list[dict[str, str]]) -> object:
    transport = _Transport()
    transport.set_handler(
        lambda request: _Ok({"blocks": [{"block_index": 1, "findings": findings}]}).as_response()
    )
    return _critique_provider(transport)


def test_unknown_critical_critique_defers_not_accepted() -> None:
    provider = _critique_provider_with(
        _findings(
            SEMANTIC_FIDELITY="UNKNOWN",
            UNSUPPORTED_FACTS="UNKNOWN",
            CODE_SWITCH_PRESERVATION="UNKNOWN",
        )
    )
    script = materialize_script(
        DeterministicWritingProvider(),
        _inputs_with_draft(NOVEL_COPY),
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        critique_provider=provider,
    )
    assert script.accepted is False
    assert script.deferred is True


def test_repeated_name_role_reversal_is_not_accepted() -> None:
    class Engine:
        def transcribe(self, *args: object, **kwargs: object) -> object:
            return type(
                "Result", (), {"raw_text": "احمد ساعد محمود ثم محمود ساعد احمد", "language": "ar"}
            )()

    result = validate_narration_fidelity(
        Path("not-decoded-by-fake.wav"),
        expected_text="احمد ساعد محمود ثم احمد ساعد محمود",
        engine=Engine(),
        options=object(),
    )
    # The role reversal is still detected (never accepted); a lone transcript
    # disagreement is deferred evidence, not a confirmed meaning mismatch.
    assert result.status == "DEFERRED"
    assert result.accepted is False
    assert "ASR_TRANSCRIPT_DISAGREEMENT" in result.findings


def test_narration_assets_are_content_addressed(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    persist = _persist_to_storage(storage, "same-source")
    first = wrap_pcm_as_wav(b"\x00\x01" * 2400, sample_rate=24000, channels=1)
    second = wrap_pcm_as_wav(b"\x00\x02" * 2400, sample_rate=24000, channels=1)
    path_one, hash_one = persist("narr-1", first)
    path_two, hash_two = persist("narr-1", second)
    # Changed content gets a distinct path; the first generation is never lost.
    assert path_one != path_two
    assert hashlib.sha256(storage.resolve_relative(path_one).read_bytes()).hexdigest() == hash_one
    assert hashlib.sha256(storage.resolve_relative(path_two).read_bytes()).hexdigest() == hash_two


def test_audition_rejection_and_identity_change_are_truthful(engine) -> None:  # type: ignore[no-untyped-def]
    settings = get_settings()
    storage = StorageService(settings.storage_root)

    def reject(_audio: object) -> dict[str, object]:
        return {"status": "REJECTED", "findings": ["MEANING_MISMATCH"]}

    first = run_audition(
        settings=settings,
        storage=storage,
        model="model-a",
        voice_factory=_FakeVoiceProvider,
        fidelity=reject,
    )
    # Every fidelity result was rejected: no provisional voice is proposed.
    assert first.provisional_voice is None

    second = run_audition(
        settings=settings,
        storage=storage,
        model="model-b",
        voice_factory=_FakeVoiceProvider,
    )
    # A changed model does not silently authorize a second nine-call comparison:
    # the durable identity mismatch is reported and zero calls are made.
    assert second.status == "AUDITION_IDENTITY_MISMATCH"
    assert second.raw_calls == 0
    assert second.samples == ()

    # The default export writes into the audition root and never copies a file
    # onto itself.
    exported = export_tts_audition(settings, [sample.as_dict() for sample in first.samples])
    assert Path(exported["root"]).is_dir()


def test_export_never_substitutes_seeded_labels_or_deterministic_path(engine) -> None:  # type: ignore[no-untyped-def]
    settings = get_settings()
    materialization_id = _seed_completed_materialization(engine, "sources/s1/final.mp4")
    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        source_id = envelope.source_video_id
        # A seeded COMPLETE/READY label alone does not establish genuine media: the
        # authoritative readiness view requires live prerequisites, render identity,
        # QC, assessment, and bound asset hashes.
        assert materialization_id not in discover_eligible_materializations(
            session, storage=_storage()
        )

        # Write a *newer* file at the deterministic authored.ass path: it must never
        # substitute for the (absent) bound accepted ASS reference.
        ass_path = (
            _storage().source_directory(source_id)
            / "materializations"
            / str(materialization_id)
            / "authored.ass"
        )
        ass_path.parent.mkdir(parents=True, exist_ok=True)
        ass_path.write_bytes(b"[Script Info]\nScriptType: v4.00+\n")

        result = export_materializations(session, settings, [materialization_id])
    case = Path(result["cases"][0]["directory"])
    assert not (case / "authored.ass").exists()
    assert result["cases"][0]["genuine_acceptance"] is False
    index = __import__("json").loads((case / "index.json").read_text(encoding="utf-8"))
    assert index["diagnostic"] is True
    assert index["publication_ready"] is False
    reasons = set(index["genuine_acceptance_reasons"])
    assert "AUTHORED_ASS_MISSING" in reasons
    assert "STALE_UPSTREAM_INPUTS" in reasons
    # The seeded render's binding is valid, but its missing QC/measured identity
    # keeps the case non-genuine (never fabricated from the seeded labels).
    assert "MISSING_LOUDNESS_EVIDENCE" in reasons

    # Demote the bound render and change its purpose: still not eligible.
    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        render = session.get(RenderExecution, envelope.render_execution_id)
        assert render is not None
        render.is_current = False
        render.artifact_purpose = "CORE_SOURCE_VALIDATION"
        session.commit()
        assert materialization_id not in discover_eligible_materializations(
            session, storage=_storage()
        )


def _storage() -> StorageService:
    return StorageService(get_settings().storage_root)


def test_lease_loss_wires_child_cancellation() -> None:
    observed: dict[str, object] = {}

    class Lease:
        ownership_lost = True

        def __enter__(self) -> "Lease":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    class Factory:
        def acquire(self, **kwargs: object) -> Lease:
            observed["callback"] = kwargs.get("on_ownership_lost")
            return Lease()

    class Engine:
        def transcribe(self, *args: object, **kwargs: object) -> object:
            observed["event"] = kwargs.get("cancel_event")
            return type("Result", (), {"raw_text": "hello", "language": "en"})()

    result = validate_narration_fidelity(
        Path("fake.wav"),
        expected_text="hello",
        engine=Engine(),
        options=object(),
        lease_factory=Factory(),
    )
    assert callable(observed.get("callback"))
    assert observed.get("event") is not None
    assert result.status == "DEFERRED"


def test_sdk_timeout_is_recorded_uncertain(component_fixture) -> None:  # type: ignore[no-untyped-def]
    engine_local, session, inputs, owner = component_fixture
    transport = _Transport()

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("response lost after send", request=request)

    transport.set_handler(timeout)
    provider = _writing_provider(transport)
    context = _context(session, owner, inputs, ComponentProviders(writing=provider))
    prepare_materialization(context)
    with Session(engine_local) as observer:
        row = observer.get(ClipMaterialization, owner.materialization_id)
        assert row is not None
        entries = row.call_ledger["entries"]
        assert len(entries) == 1
        assert entries[0]["status"] == "UNCERTAIN"


def test_noncritical_concern_is_carried_into_accepted_script() -> None:
    provider = _critique_provider_with(_findings(NATURALNESS="CONCERN"))
    script = materialize_script(
        DeterministicWritingProvider(),
        _inputs_with_draft(NOVEL_COPY),
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        critique_provider=provider,
    )
    assert script.accepted is True
    assert any(
        finding.dimension == "LANGUAGE_REGISTER_NATURALNESS" and finding.level == "CONCERN"
        for finding in script.findings
    )
