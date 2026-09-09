"""Focused regressions for the Sol-review blocker pass.

These tests use deterministic fake providers, a fake transport, and stored
transcript structures only. They prove: aggregate local batches are split so the
real sent envelope never exceeds the configured context, degraded
reconstruction re-enters the executor through PipelineRunner without repeating
accepted targets, cancellation is polled on every provider route and after the
final local batch, the Ollama CPU default is six, and a fresh cache-hit releases
the Gemini provider (scrubbing its key) without any generation or network call.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import JobKind, JobStatus, PipelineRunStatus, PipelineStage, RightsStatus
from app.db.base import Base
from app.models import PipelineRun, ProcessingJob, SourceVideo, Transcript
from app.pipeline.executor import ReconstructionCancelled
from app.pipeline.runner import PipelineRunner
from app.pipeline.stages import ContextualReconstructionExecutor
from app.transcription.reconstruction.confidence import CONFIDENCE_POLICY_VERSION
from app.transcription.reconstruction.gemini import (
    GeminiErrorCategory,
    GeminiProviderError,
    GeminiReconstructionProvider,
)
from app.transcription.reconstruction.ollama import OllamaReconstructionProvider
from app.transcription.reconstruction.providers import ReconstructionRequest
from app.transcription.reconstruction.routing import AdaptiveRoutingConfig, RoutingMode
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
    estimate_tokens,
)
from app.transcription.reconstruction.validation import VALIDATION_VERSION


def _words(probabilities: list[float], texts: list[str]) -> list[dict[str, object]]:
    return [
        {"word": text, "probability": probability}
        for text, probability in zip(texts, probabilities)
    ]


def _hard_segment(index: int, raw: str = "دخم") -> dict[str, object]:
    return {
        "start": float(index),
        "end": float(index + 1),
        "text": raw,
        "raw_text": raw,
        "corrected_text": raw,
        "words": _words([0.30, 0.25, 0.20, 0.90], ["م", "ش", "قادر", "يفهم"]),
        "correction_applied": False,
        "correction_confidence": 0.0,
        "correction_method": "unchanged",
        "correction_changes": [],
    }


def _mild_segment(index: int) -> dict[str, object]:
    return {
        "start": float(index),
        "end": float(index + 1),
        "text": "دخم",
        "raw_text": "دخم",
        "corrected_text": "دخم",
        "words": _words([0.98, 0.60, 0.98], [f"كلام{index}", "جديد", "مصري"]),
        "correction_applied": False,
        "correction_confidence": 0.0,
        "correction_method": "unchanged",
        "correction_changes": [],
    }


class FakeLocal:
    def __init__(
        self,
        candidate: ReconstructionCandidate | None = None,
        invalid_for: dict[int, ReconstructionCandidate] | None = None,
    ) -> None:
        self.calls = 0
        self.requests: list[ReconstructionRequest] = []
        self.candidate = candidate or ReconstructionCandidate(
            "provider-0", "ضخمة", provider_confidence=1.0
        )
        self.invalid_for = invalid_for or {}
        self.released = False

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "ollama", "qwen3.5:4b", "sha256:x", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
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
            "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
            "validation_version": VALIDATION_VERSION,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def release(self) -> None:
        self.released = True

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        self.requests.extend(requests)
        result: dict[int, ReconstructionCandidate] = {}
        for request in requests:
            result[request.segment_index] = self.invalid_for.get(
                request.segment_index, self.candidate
            )
        return result


class _Toggle:
    def __init__(self) -> None:
        self.count = 0


class _TogglingLocal(FakeLocal):
    def __init__(self, toggler: _Toggle) -> None:
        super().__init__()
        self._toggler = toggler

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self._toggler.count += 1
        return super().reconstruct_segments(requests)


class _TogglingGemini:
    def __init__(self, toggler: _Toggle) -> None:
        self.model = "gemini-3.8-flash"
        self.calls = 0
        self.requests: list[ReconstructionRequest] = []
        self._toggler = toggler

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "gemini", "gemini-3.8-flash", "sha256:g", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
        return _gemini_identity()

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def release(self) -> None:
        pass

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        self._toggler.count += 1
        self.requests.extend(requests)
        return {
            request.segment_index: ReconstructionCandidate(
                "provider-0", "ضخمة", provider_confidence=1.0
            )
            for request in requests
        }

    def usage_summary(self) -> dict[str, int]:
        return {"prompt_token_count": 1, "candidates_token_count": 1, "total_token_count": 2}


class _FailOnGemini:
    def __init__(self, fail_segment: int | None = None) -> None:
        self.model = "gemini-3.8-flash"
        self.calls = 0
        self.requests: list[ReconstructionRequest] = []
        self.fail_segment = fail_segment

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "gemini", "gemini-3.8-flash", "sha256:g", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
        return _gemini_identity()

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def release(self) -> None:
        pass

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        self.requests.extend(requests)
        if any(request.segment_index == self.fail_segment for request in requests):
            raise GeminiProviderError(GeminiErrorCategory.RATE_LIMITED)
        return {
            request.segment_index: ReconstructionCandidate(
                "provider-0", "ضخمة", provider_confidence=1.0
            )
            for request in requests
        }

    def usage_summary(self) -> dict[str, int]:
        return {"prompt_token_count": 1, "candidates_token_count": 1, "total_token_count": 2}


def _gemini_identity() -> dict[str, object]:
    return {
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "digest": "sha256:g",
        "prompt_hash": "p",
        "schema_version": "s",
        "api_version": "v1",
        "temperature": 0.0,
        "timeout_seconds": 30.0,
        "retry_attempts": 0,
        "retry_backoff_seconds": 0.0,
        "max_output_tokens": 1024,
        "thinking_level": "low",
        "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
        "validation_version": VALIDATION_VERSION,
    }


def _reconstructor(
    local: FakeLocal,
    gemini: Any | None = None,
    mode: RoutingMode = RoutingMode.ADAPTIVE,
    budget: int = 5,
    **kwargs: Any,
) -> ContextualReconstructor:
    return ContextualReconstructor(
        local,
        gemini_provider=gemini,
        routing=AdaptiveRoutingConfig(mode=mode),
        gemini_budget=budget,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Blocker 1: aggregate local batches must fit the real Ollama context envelope
# ---------------------------------------------------------------------------


def test_aggregate_local_batches_are_split_to_fit_context() -> None:
    """Requests that each fit alone are split when combined would exceed context."""

    captured: list[dict[str, object]] = []
    big_previous = ("سياق",) * 700

    def transport(
        _method: str,
        _url: str,
        body: bytes | None,
        _headers: dict[str, str],
        _timeout: float,
    ) -> bytes:
        assert body is not None
        payload = json.loads(body)
        captured.append(payload)
        targets = json.loads(payload["messages"][1]["content"])["targets"]
        content = {
            "reconstructions": [
                {"segment_id": t["segment_id"], "corrected_text": "هدف", "unchanged": True}
                for t in targets
            ]
        }
        return json.dumps(
            {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]},
            ensure_ascii=False,
        ).encode()

    provider = OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=3,
        release_after_run=False,
        max_context_tokens=4096,
        request=transport,
    )
    requests = [
        ReconstructionRequest(
            segment_index=0,
            raw_text="هدف",
            corrected_text="هدف",
            previous=big_previous,
        ),
        ReconstructionRequest(
            segment_index=1,
            raw_text="هدف",
            corrected_text="هدف",
            previous=big_previous,
        ),
    ]

    result = provider.reconstruct_segments(requests)

    # Combined would exceed 4096, so the provider must have made two calls, one
    # target each.
    assert len(captured) == 2
    for call in captured:
        assert len(json.loads(call["messages"][1]["content"])["targets"]) == 1
    # Every actually-sent aggregate envelope fits the configured context.
    for call in captured:
        system_tokens = estimate_tokens(str(call["messages"][0]["content"]))
        user_tokens = estimate_tokens(str(call["messages"][1]["content"]))
        targets = json.loads(call["messages"][1]["content"])["targets"]
        budget = system_tokens + user_tokens + 64 + 256 * len(targets) + 128
        assert budget <= 4096
    # Candidates still map to the exact requested segments.
    assert set(result) == {0, 1}
    assert result[0].text == "هدف"
    assert result[1].text == "هدف"


# ---------------------------------------------------------------------------
# Blocker 2: degraded reconstruction re-enters the executor through the runner
# ---------------------------------------------------------------------------


def _setup_runner_env(
    sqlite_engine: object, segments: list[dict[str, object]]
) -> tuple[object, object]:
    Base.metadata.create_all(sqlite_engine)
    session = Session(sqlite_engine)
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
        segments=segments,
    )
    session.add(transcript)
    session.commit()
    source_id = source.id
    session.close()
    return source_id, sqlite_engine


def test_degraded_reconstruction_retries_through_runner_without_repeating_accepted(
    sqlite_engine: object,
) -> None:
    """A later non-force run re-enters the executor and retries only unfinished targets."""

    segments = [_hard_segment(i) for i in range(5)]
    source_id, engine = _setup_runner_env(sqlite_engine, segments)

    # First run: four Gemini targets accepted, the fifth rate-limited and its
    # local fallback rejected, so the run is degraded (not cache-eligible).
    invalid_local = FakeLocal(
        invalid_for={4: ReconstructionCandidate("provider-0", "رقم 12345", 1.0)}
    )
    fail_gemini = _FailOnGemini(fail_segment=4)
    first_session = Session(engine)
    first_runner = PipelineRunner(
        first_session,
        {
            PipelineStage.CONTEXTUAL_RECONSTRUCTION: ContextualReconstructionExecutor(
                session=first_session,
                reconstructor=_reconstructor(invalid_local, fail_gemini, budget=5),
            )
        },
    )
    first_result = first_runner.run(source_id, PipelineStage.CONTEXTUAL_RECONSTRUCTION)
    assert first_result.skipped is False
    first_session.close()

    with Session(engine) as check:
        transcript = check.scalar(select(Transcript).where(Transcript.source_video_id == source_id))
        assert transcript is not None
        assert transcript.reconstruction_metadata["cache_eligible"] is False
        assert transcript.segments[4]["reconstruction_status"] == "LOW_CONFIDENCE_UNRESOLVED"

    # Second, non-force run with a recovered provider: the runner must enter the
    # executor (not skip), reuse the four accepted targets with zero Gemini
    # calls, and retry only segment 4.
    fresh_local = FakeLocal()
    fresh_gemini = _FailOnGemini(fail_segment=None)
    second_session = Session(engine)
    second_runner = PipelineRunner(
        second_session,
        {
            PipelineStage.CONTEXTUAL_RECONSTRUCTION: ContextualReconstructionExecutor(
                session=second_session,
                reconstructor=_reconstructor(fresh_local, fresh_gemini, budget=5),
            )
        },
    )
    second_result = second_runner.run(source_id, PipelineStage.CONTEXTUAL_RECONSTRUCTION)
    assert second_result.skipped is False  # entered the executor, not skipped
    second_session.close()

    assert fresh_gemini.calls == 1
    assert [request.segment_index for request in fresh_gemini.requests] == [4]
    with Session(engine) as check:
        transcript = check.scalar(select(Transcript).where(Transcript.source_video_id == source_id))
        assert transcript is not None
        assert transcript.reconstruction_metadata["cache_eligible"] is True
        assert transcript.segments[0]["reconstruction_status"] == "APPLIED"
        assert transcript.segments[4]["reconstruction_status"] == "APPLIED"


# ---------------------------------------------------------------------------
# Blocker 3: cancellation on every provider route and after the final batch
# ---------------------------------------------------------------------------


def test_cancellation_between_direct_gemini_targets_stops_second_call() -> None:
    toggler = _Toggle()
    gemini = _TogglingGemini(toggler)
    local = FakeLocal()
    segments = [_hard_segment(i) for i in range(2)]
    reconstructor = _reconstructor(local, gemini, is_cancelled=lambda: toggler.count >= 1)
    with pytest.raises(ReconstructionCancelled):
        reconstructor.reconstruct(
            segments,
            language="ar",
            transcription_fingerprint="asr-v1",
            correction_version="egyptian-ar-v1",
        )
    assert gemini.calls == 1
    assert local.calls == 0


def test_cancellation_after_gemini_only_attempt_leaves_job_cancelled() -> None:
    toggler = _Toggle()
    gemini = _TogglingGemini(toggler)
    segments = [_hard_segment(i) for i in range(2)]
    reconstructor = _reconstructor(
        FakeLocal(), gemini, mode=RoutingMode.GEMINI_ONLY, is_cancelled=lambda: toggler.count >= 1
    )
    with pytest.raises(ReconstructionCancelled):
        reconstructor.reconstruct(
            segments,
            language="ar",
            transcription_fingerprint="asr-v1",
            correction_version="egyptian-ar-v1",
        )
    assert gemini.calls == 1


def test_cancellation_after_final_local_batch_is_detected_before_success() -> None:
    toggler = _Toggle()
    local = _TogglingLocal(toggler)
    segments = [_mild_segment(i) for i in range(4)]
    reconstructor = _reconstructor(local, batch_windows=4, is_cancelled=lambda: toggler.count >= 1)
    with pytest.raises(ReconstructionCancelled):
        reconstructor.reconstruct(
            segments,
            language="ar",
            transcription_fingerprint="asr-v1",
            correction_version="egyptian-ar-v1",
        )
    assert local.calls == 1  # the single/final local batch ran, then cancellation won


class _DbCancelGemini(_FailOnGemini):
    def __init__(self, session: Session, source_id: object) -> None:
        super().__init__()
        self._session = session
        self._source_id = source_id
        self._flipped = False

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        if not self._flipped:
            job = self._session.scalar(
                select(ProcessingJob)
                .where(
                    ProcessingJob.source_video_id == self._source_id,
                    ProcessingJob.kind == JobKind.RECONSTRUCTION,
                )
                .order_by(ProcessingJob.created_at.desc())
            )
            if job is not None:
                job.status = JobStatus.CANCELLED
                self._session.commit()
                self._flipped = True
        return super().reconstruct_segments(requests)


def test_cancelled_direct_gemini_run_keeps_job_and_run_cancelled(sqlite_engine: object) -> None:
    segments = [_hard_segment(i) for i in range(2)]
    source_id, engine = _setup_runner_env(sqlite_engine, segments)
    session = Session(engine)
    gemini = _DbCancelGemini(session, source_id)
    local = FakeLocal()
    runner = PipelineRunner(
        session,
        {
            PipelineStage.CONTEXTUAL_RECONSTRUCTION: ContextualReconstructionExecutor(
                session=session,
                reconstructor=_reconstructor(local, gemini, budget=5),
            )
        },
    )
    with pytest.raises(ReconstructionCancelled):
        runner.run(source_id, PipelineStage.CONTEXTUAL_RECONSTRUCTION)
    assert gemini.calls == 1
    run = session.scalar(
        select(PipelineRun)
        .where(PipelineRun.source_video_id == source_id)
        .order_by(PipelineRun.created_at.desc())
    )
    assert run.status is PipelineRunStatus.CANCELLED
    latest_job = session.scalar(
        select(ProcessingJob)
        .where(
            ProcessingJob.source_video_id == source_id,
            ProcessingJob.kind == JobKind.RECONSTRUCTION,
        )
        .order_by(ProcessingJob.created_at.desc())
    )
    assert latest_job.status is JobStatus.CANCELLED
    session.close()


# ---------------------------------------------------------------------------
# Blocker 4: Ollama CPU default is six
# ---------------------------------------------------------------------------


def test_ollama_cpu_default_is_six() -> None:
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    compose = (root / "compose.yaml").read_text(encoding="utf-8")
    assert re.search(r"cpus:\s*\"\$\{OLLAMA_CPUS:-6\}\"", compose)
    env = (root / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^OLLAMA_CPUS=6$", env, re.MULTILINE)


# ---------------------------------------------------------------------------
# Cache-hit cleanup: a fresh cache hit releases the Gemini provider and scrubs
# its key without generation or network
# ---------------------------------------------------------------------------


class _SdkModels:
    def generate_content(self, model: str, contents: str, config: object) -> object:
        payload = json.loads(contents)
        entries = [
            {
                "segment_id": target["segment_id"],
                "corrected_text": "ضخمة",
                "unchanged": False,
                "confidence": 0.95,
            }
            for target in payload["targets"]
        ]
        return SimpleNamespace(
            parsed={"reconstructions": entries},
            text=None,
            usage_metadata=None,
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
            prompt_feedback=None,
        )


class _SdkClient:
    def __init__(self) -> None:
        self.models = _SdkModels()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _real_gemini(factory: Any) -> GeminiReconstructionProvider:
    return GeminiReconstructionProvider(
        api_key="test-key-not-committed",
        model="gemini-3.8-flash",
        timeout_seconds=10.0,
        retry_attempts=1,
        retry_backoff_seconds=0.0,
        max_output_tokens=1024,
        thinking_level="low",
        temperature=0.0,
        api_version="v1",
        owns_client=True,
        sleep=lambda _seconds: None,
        client_factory=factory,
    )


def test_fresh_cache_hit_releases_gemini_key_without_generation(sqlite_engine: object) -> None:
    """A fresh worker that only hits the cache never builds a client, never
    calls Gemini, and scrubs the API key on release."""

    segments = [_hard_segment(i) for i in range(1)]
    source_id, engine = _setup_runner_env(sqlite_engine, segments)

    # First full run makes the transcript fully cache-eligible.
    first_session = Session(engine)
    first_source = first_session.get(SourceVideo, source_id)
    assert first_source is not None
    first_executor = ContextualReconstructionExecutor(
        session=first_session,
        reconstructor=_reconstructor(FakeLocal(), _real_gemini(lambda: _SdkClient()), budget=5),
    )
    first_executor.execute(first_source, force=True)
    first_session.close()

    # Fresh provider on a cache-hit run: the key must be scrubbed by the
    # executor cleanup with zero generation and zero client construction.
    factory_calls: list[int] = []

    def factory() -> object:
        factory_calls.append(1)
        return _SdkClient()

    fresh_gemini = _real_gemini(factory)
    second_session = Session(engine)
    second_source = second_session.get(SourceVideo, source_id)
    assert second_source is not None
    second_executor = ContextualReconstructionExecutor(
        session=second_session,
        reconstructor=_reconstructor(FakeLocal(), fresh_gemini, budget=5),
    )
    second_executor.execute(second_source, force=False)
    second_session.close()

    assert factory_calls == []
    assert fresh_gemini._client is None  # no SDK client was ever constructed
    assert fresh_gemini._api_key is None  # key scrubbed on executor release
    assert fresh_gemini.usage_summary()["total_token_count"] == 0
