"""Corrected Stage 6 root-cause regressions (RC1-RC6).

These assert the *corrected* behavior for the six root causes SOL re-opened. They
replace the diagnostic probe (which intentionally asserted the defects) and never
copy an unsafe expectation into an accepted test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from google import genai
from google.genai import types
from sqlalchemy.orm import Session
from test_stage6_api_cli_acceptance import _seed_completed_materialization, engine  # noqa: F401
from test_stage6_component_recovery import CountingNarrationSynthesizer, _context
from test_stage6_component_recovery import component_fixture as component_fixture  # noqa: F401
from test_stage6_gemini_boundary import (
    NOVEL_COPY,
    _critique_provider,
    _critique_request,
    _Ok,
    _Transport,
    _writing_provider,
    _writing_request,
)
from test_stage6_tts_boundary import _request as _tts_request
from test_stage6_tts_boundary import _Transport as _TTSResponseTransport

from app.core.settings import get_settings
from app.materialization.acceptance_export import (
    export_materializations,
    genuine_acceptance_reasons,
)
from app.materialization.components import (
    CheckpointedNarrationSynthesizer,
    ComponentProviders,
    DurableRequestGateway,
    prepare_materialization,
)
from app.materialization.gemini import GeminiCritiqueProvider
from app.materialization.providers import (
    CRITIQUE_DIMENSIONS,
    DeterministicWritingProvider,
    MaterializationProviderError,
)
from app.materialization.tts_audition import (
    AUDITION_SCRIPTS,
    AUDITION_VOICES,
    run_audition,
)
from app.materialization.tts_gemini import GeminiNarrationSynthesizer
from app.models import ClipMaterialization
from app.services.storage import StorageService


class _Crash(BaseException):
    """BaseException so a simulated crash escapes the gateway's Exception handling."""


def _complete(dimension_overrides: dict[str, str] | None = None) -> list[dict[str, str]]:
    overrides = dimension_overrides or {}
    return [
        {"dimension": dimension, "level": overrides.get(dimension, "PASS"), "evidence": "e"}
        for dimension in CRITIQUE_DIMENSIONS
    ]


def _ledger_statuses(engine_local: Any, materialization_id: Any) -> list[str]:
    with Session(engine_local) as observer:
        row = observer.get(ClipMaterialization, materialization_id)
        return [str(entry.get("status")) for entry in row.call_ledger["entries"]]


# --- RC1: strict response structure before categorical acceptance -----------


def test_rc1_duplicate_contradictory_block_invalidates_whole_response() -> None:
    transport = _Transport()
    failing = _complete({"SEMANTIC_FIDELITY": "FAIL"})
    passing = _complete()
    duplicate_blocks = [
        {"block_index": 1, "findings": failing},
        {"block_index": 1, "findings": passing},
    ]
    transport.set_handler(lambda request: _Ok({"blocks": duplicate_blocks}).as_response())
    result = _critique_provider(transport).critique(_critique_request())
    # The contradictory duplicate invalidates the entire response; the favorable
    # remainder is never retained.
    assert result.deferred is True
    assert result.reason_code == "MALFORMED_OUTPUT"
    assert result.blocks == {}
    assert result.decision(1) is None


def test_rc1_unexpected_extra_entry_invalidates_whole_response() -> None:
    transport = _Transport()
    payload = {
        "blocks": [
            {"block_index": 1, "findings": _complete()},
            {"block_index": 2, "findings": _complete()},
        ]
    }
    transport.set_handler(lambda request: _Ok(payload).as_response())
    result = _critique_provider(transport).critique(_critique_request())
    assert result.deferred is True and result.reason_code == "MALFORMED_OUTPUT"


def test_rc1_writing_missing_or_duplicate_block_invalidates_whole_response() -> None:
    transport = _Transport()
    transport.set_handler(
        lambda request: _Ok(
            {"copies": [{"block_index": 1, "text": "a"}, {"block_index": 1, "text": "b"}]}
        ).as_response()
    )
    result = _writing_provider(transport).write(_writing_request())
    assert result.deferred is True and result.reason_code == "MALFORMED_OUTPUT"


def test_rc1_installed_sdk_duplicate_block_defers(component_fixture: Any) -> None:
    _engine, session, inputs, owner = component_fixture
    transport = _Transport()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        text = body["contents"][0]["parts"][0]["text"]
        payload = json.loads(text)
        indices = [int(target["block_index"]) for target in payload["targets"]]
        # Duplicate the first requested block with contradictory findings.
        blocks = [{"block_index": indices[0], "findings": _complete()}]
        blocks.append(
            {"block_index": indices[0], "findings": _complete({"GROUNDING_ENTAILMENT": "FAIL"})}
        )
        for index in indices[1:]:
            blocks.append({"block_index": index, "findings": _complete()})
        return _Ok({"blocks": blocks}).as_response()

    transport.set_handler(handler)
    critique = GeminiCritiqueProvider(api_key="fake-key", client_factory=transport.client)
    context = _context(
        session,
        owner,
        inputs,
        ComponentProviders(writing=DeterministicWritingProvider(), critique=critique),
    )
    prepared = prepare_materialization(context)
    # The malformed critique never authorizes acceptance: no authored block is
    # accepted and the generation is not review-ready.
    assert prepared.authored_blocks == []
    assert prepared.semantic_outcome != "READY_FOR_HUMAN_REVIEW"


# --- RC2: separate synthesis identity from validation identity --------------


def test_rc2_changed_asr_runtime_reruns_validation_without_tts(component_fixture: Any) -> None:
    _engine, session, inputs, owner = component_fixture
    synth = CountingNarrationSynthesizer()
    providers = ComponentProviders(writing=DeterministicWritingProvider())
    context = _context(session, owner, inputs, providers)
    gateway = DurableRequestGateway(context)
    calls: list[str] = []

    def validator_a(audio: Any) -> dict[str, object]:
        calls.append("a")
        return {
            "status": "ACCEPTED",
            "transcript": audio.text,
            "asr_identity": {"model": "model-a"},
        }

    validator_a.runtime_identity = lambda: {"model": "model-a"}  # type: ignore[attr-defined]

    def validator_b(audio: Any) -> dict[str, object]:
        calls.append("b")
        return {
            "status": "REJECTED",
            "transcript": "different",
            "asr_identity": {"model": "model-b"},
        }

    validator_b.runtime_identity = lambda: {"model": "model-b"}  # type: ignore[attr-defined]

    request = _tts_request()
    first = CheckpointedNarrationSynthesizer(synth, gateway, inputs, validator=validator_a)
    first.synthesize(request)
    second = CheckpointedNarrationSynthesizer(synth, gateway, inputs, validator=validator_b)
    with pytest.raises(RuntimeError):
        second.synthesize(request)
    # Synthesis is preserved (one TTS call); validation reran under the new ASR
    # runtime and rejected, so the chunk is not accepted.
    assert calls == ["a", "b"]
    assert len(synth.calls) == 1


def test_rc2_missing_validator_defers_never_true(component_fixture: Any) -> None:
    _engine, session, inputs, owner = component_fixture
    synth = CountingNarrationSynthesizer()
    providers = ComponentProviders(writing=DeterministicWritingProvider())
    context = _context(session, owner, inputs, providers)
    gateway = DurableRequestGateway(context)
    wrapped = CheckpointedNarrationSynthesizer(synth, gateway, inputs, validator=None)
    with pytest.raises(RuntimeError):
        wrapped.synthesize(_tts_request())


# --- RC3: no implicit replay of uncertain provider work ---------------------


def test_rc3_uncertain_writing_not_reinvoiced_on_resume(component_fixture: Any) -> None:
    engine_local, session, inputs, owner = component_fixture
    transport = _Transport()

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("response lost after send", request=request)

    transport.set_handler(timeout)
    provider = _writing_provider(transport)

    def make_context() -> Any:
        return _context(session, owner, inputs, ComponentProviders(writing=provider))

    prepare_materialization(make_context())
    prepare_materialization(make_context())
    # The ordinary resume never silently retried the uncertain request.
    assert len(transport.requests) == 1
    assert _ledger_statuses(engine_local, owner.materialization_id) == ["UNCERTAIN"]


def test_rc3_reserved_crash_left_attempt_is_not_reinvoked(component_fixture: Any) -> None:
    engine_local, session, inputs, owner = component_fixture
    calls: list[int] = []

    class CrashingProvider(DeterministicWritingProvider):
        provider_name = "crashing-hosted"

        def write(self, request: Any) -> Any:
            calls.append(1)
            raise _Crash("crash in the reserve->response gap")

    def make_context() -> Any:
        return _context(session, owner, inputs, ComponentProviders(writing=CrashingProvider()))

    with pytest.raises(_Crash):
        prepare_materialization(make_context())
    session.rollback()
    # A crash-left reservation is unresolved uncertainty, not permission to invoke
    # again: the ordinary resume makes no second provider call.
    prepare_materialization(make_context())
    assert calls == [1]
    assert "UNCERTAIN" in _ledger_statuses(engine_local, owner.materialization_id)


# --- RC4: Interactions retries and typed provider exception accounting ------


def test_rc4_interactions_429_is_single_request_and_cooldown(component_fixture: Any) -> None:
    engine_local, session, inputs, owner = component_fixture
    transport = _TTSResponseTransport({"error": {"code": 429, "message": "quota"}})
    transport.status = 429
    provider = GeminiNarrationSynthesizer(api_key="fake-key", client_factory=transport.client)
    notified: list[bool] = []
    providers = ComponentProviders(writing=DeterministicWritingProvider())
    context = _context(session, owner, inputs, providers)
    context.on_rate_limited = lambda: notified.append(True)
    wrapped = CheckpointedNarrationSynthesizer(provider, DurableRequestGateway(context), inputs)
    with pytest.raises(MaterializationProviderError):
        wrapped.synthesize(_tts_request())
    # Exactly one raw HTTP request; a known rejection is FAILED_KNOWN, and the
    # shared cooldown controller was notified.
    assert len(transport.requests) == 1
    assert _ledger_statuses(engine_local, owner.materialization_id) == ["FAILED_KNOWN"]
    assert notified == [True]


def test_rc4_interactions_zero_retry_boundary_is_single_request() -> None:
    transport = _TTSResponseTransport({"error": {"code": 429, "message": "quota"}})
    transport.status = 429

    def client() -> genai.Client:
        return genai.Client(
            api_key="fake-key",
            http_options=types.HttpOptions(
                api_version="v1beta",
                retry_options=types.HttpRetryOptions(attempts=0),
                httpx_client=httpx.Client(transport=httpx.MockTransport(transport.handler)),
            ),
        )

    provider = GeminiNarrationSynthesizer(api_key="fake-key", client_factory=client)
    with pytest.raises(MaterializationProviderError):
        provider.synthesize(_tts_request())
    # Even with a zero-attempt parent retry option, the guarded Interactions
    # boundary produces exactly one request.
    assert len(transport.requests) == 1


def test_rc4_interactions_success_and_503_request_counts() -> None:
    from test_stage6_tts_boundary import _interaction_response

    payload = _interaction_response("audio/l16", b"\x10\x20" * 50, rate=24_000, channels=1)
    ok = _TTSResponseTransport(payload)
    provider = GeminiNarrationSynthesizer(
        api_key="fake-key", client_factory=ok.client, persist=lambda c, d: ("p", "h")
    )
    provider.synthesize(_tts_request())
    assert len(ok.requests) == 1

    unavailable = _TTSResponseTransport({"error": {"code": 503, "message": "unavailable"}})
    unavailable.status = 503
    provider2 = GeminiNarrationSynthesizer(api_key="fake-key", client_factory=unavailable.client)
    with pytest.raises(MaterializationProviderError):
        provider2.synthesize(_tts_request())
    assert len(unavailable.requests) == 1


# --- RC5: durable, serialized nine-attempt audition -------------------------


def test_rc5_reservation_is_persisted_before_http_and_recovers_uncertain(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    manifest = storage.storage_root / "benchmarks" / "stage-6" / "tts-audition" / "ledger.json"
    seen: list[bool] = []
    invocations: list[str] = []

    class Provider:
        def available(self) -> bool:
            return True

        def synthesize(self, request: Any) -> Any:
            invocations.append(request.chunk_id)
            seen.append(manifest.exists() and "RESERVED" in manifest.read_text(encoding="utf-8"))
            raise _Crash("crash after reservation")

    settings = get_settings()
    for _ in range(2):
        with pytest.raises(_Crash):
            run_audition(
                settings=settings,
                storage=storage,
                model="fixed-model",
                voice_factory=lambda voice: Provider(),
            )
    # The reservation was durable (visible) before the first HTTP op, and the
    # crash-left RESERVED attempt is recovered as UNCERTAIN, never re-invoked.
    first_chunk = f"{AUDITION_VOICES[0]}-{AUDITION_SCRIPTS[0].script_id}"
    assert seen[0] is True
    assert invocations.count(first_chunk) == 1


def test_rc5_budget_is_durable_across_missing_file_and_identity_change(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    settings = get_settings()

    class Fake:
        provider_name = "fake-voice"

        def __init__(self, voice: str) -> None:
            self.voice = voice
            self._persist = None

        def available(self) -> bool:
            return True

        def runtime_identity(self) -> dict[str, object]:
            return {"provider": "fake-voice", "voice": self.voice}

        def synthesize(self, request: Any) -> Any:
            import struct

            from app.materialization.narration import inspect_wav, wrap_pcm_as_wav

            frames = 2400
            pcm = struct.pack(f"<{frames}h", *([500] * frames))
            wav = wrap_pcm_as_wav(pcm, sample_rate=24_000, channels=1)
            relative, sha = self._persist(request.chunk_id, wav)
            info = inspect_wav(wav)
            return type(
                "_Audio",
                (),
                {
                    "relative_path": relative,
                    "sha256": sha,
                    "duration_seconds": info.duration_seconds,
                    "sample_rate": info.sample_rate,
                    "channels": info.channels,
                    "text": request.text,
                },
            )()

        def release(self) -> None:
            return None

    def accept(_audio: object) -> dict[str, object]:
        return {"status": "ACCEPTED"}

    first = run_audition(
        settings=settings,
        storage=storage,
        model="model-a",
        voice_factory=Fake,
        fidelity=accept,
    )
    assert first.raw_calls == 9
    # Delete one artifact: the spent attempt is not reset and not re-synthesized.
    storage.resolve_relative(first.samples[0].relative_path).unlink()
    second = run_audition(
        settings=settings,
        storage=storage,
        model="model-a",
        voice_factory=Fake,
        fidelity=accept,
    )
    assert second.raw_calls == 0
    # A changed model reports an identity mismatch and makes zero calls.
    third = run_audition(
        settings=settings,
        storage=storage,
        model="model-b",
        voice_factory=Fake,
        fidelity=accept,
    )
    assert third.status == "AUDITION_IDENTITY_MISMATCH" and third.raw_calls == 0


def test_rc5_malformed_history_fails_closed(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    root = storage.storage_root / "benchmarks" / "stage-6" / "tts-audition"
    root.mkdir(parents=True, exist_ok=True)
    (root / "ledger.json").write_text("{not valid json", encoding="utf-8")

    class Fake:
        def available(self) -> bool:
            return True

        def synthesize(self, request: Any) -> Any:  # pragma: no cover - must not run
            raise AssertionError("must not synthesize on malformed history")

    result = run_audition(
        settings=get_settings(),
        storage=storage,
        model="fixed-model",
        voice_factory=lambda voice: Fake(),
    )
    assert result.status == "AUDITION_HISTORY_INVALID"
    assert result.raw_calls == 0


def test_rc5_concurrent_invocation_reports_active(tmp_path: Path) -> None:
    storage = StorageService(tmp_path / "storage")
    from app.materialization import tts_audition

    release = {"go": False}

    class Slow:
        def available(self) -> bool:
            return True

        def synthesize(self, request: Any) -> Any:
            raise AssertionError("must not synthesize")

    held: dict[str, object] = {}
    started = {"v": False}

    import threading

    def holder() -> None:
        with tts_audition._audition_lock(storage) as acquired:
            held["acquired"] = acquired
            started["v"] = True
            while not release["go"]:
                time.sleep(0.01)

    thread = threading.Thread(target=holder)
    thread.start()
    while not started["v"]:
        time.sleep(0.01)
    result = run_audition(
        settings=get_settings(),
        storage=storage,
        model="fixed-model",
        voice_factory=lambda voice: Slow(),
    )
    release["go"] = True
    thread.join(timeout=2)
    assert result.status == "AUDITION_ACTIVE" and result.raw_calls == 0


# --- RC6: authoritative readiness shared by export and API ------------------


def test_rc6_export_rejects_invalid_evidence_and_missing_assets(engine: Any) -> None:  # noqa: F811
    settings = get_settings()
    storage = StorageService(settings.storage_root)
    materialization_id = _seed_completed_materialization(engine, "sources/s1/final.mp4")
    with Session(engine) as session:
        envelope = session.get(ClipMaterialization, materialization_id)
        assert envelope is not None
        render = session.get(
            __import__("app.models.render_execution", fromlist=["RenderExecution"]).RenderExecution,
            envelope.render_execution_id,
        )
        envelope.repair_count = 1
        envelope.narration_references = [{"relative_path": "missing.wav", "sha256": "f" * 64}]
        render.request_payload = {}
        render.qc_result = {
            "measured": {"final_integrated_lufs": -40.0, "final_true_peak_dbtp": 1.0}
        }
        session.commit()
        reasons = genuine_acceptance_reasons(session, storage, envelope)
        exported = export_materializations(session, settings, [materialization_id])
    case = exported["cases"][0]
    assert case["genuine_acceptance"] is False
    assert reasons  # never genuine from seeded labels
    assert not (Path(case["directory"]) / "authored.ass").exists()


def test_rc6_api_response_exposes_shared_readiness(engine: Any) -> None:  # noqa: F811
    from app.api.app import _materialization_response

    materialization_id = _seed_completed_materialization(engine, "sources/s1/final.mp4")
    with Session(engine) as session:
        row = session.get(ClipMaterialization, materialization_id)
        assert row is not None
        response = _materialization_response(session, row)
    # The API and the exporter share the same authoritative readiness verdict.
    assert response.readiness["review_ready"] is False
    assert response.publication_ready is False
    # The seeded artifact file genuinely exists (so it is available for
    # inspection), but incomplete lineage keeps the case non-genuine.
    assert response.artifact_available is True
    assert response.render_execution_id is not None
    assert response.live_freshness in {"STALE", "HISTORICAL", "CURRENT"}


__all__ = [
    "AUDITION_SCRIPTS",
    "AUDITION_VOICES",
    "NOVEL_COPY",
]
