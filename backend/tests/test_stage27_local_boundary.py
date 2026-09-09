"""Focused regressions for the local-batch ownership and cross-session
cancellation blockers.

These tests use deterministic fake providers, a fake transport, fake clocks, and
SQLite sessions only. They prove: aggregate context-safe splitting is planned by
orchestration into visible actual requests (with cancellation, wall-time, and
checkpoints around each), a partial actual-request failure preserves earlier
accepted work, and API-session cancellation is observed by a worker session via
a fresh scalar status read of the exact executing job.
"""

from __future__ import annotations

import json
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
from app.transcription.reconstruction.ollama import OllamaReconstructionProvider
from app.transcription.reconstruction.providers import (
    ProviderResponseError,
    ReconstructionRequest,
)
from app.transcription.reconstruction.routing import AdaptiveRoutingConfig, RoutingMode
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
    estimate_tokens,
)
from app.transcription.reconstruction.validation import VALIDATION_VERSION


def _big_local_segment(index: int, words: int = 110) -> dict[str, object]:
    word_evidence = [{"word": f"كلمة{i}", "probability": 0.98} for i in range(words - 1)]
    word_evidence.append({"word": "مشكوك", "probability": 0.60})
    return {
        "start": float(index),
        "end": float(index + 1),
        "text": "دخم",
        "raw_text": "دخم",
        "corrected_text": "دخم",
        "words": word_evidence,
        "correction_applied": False,
        "correction_confidence": 0.0,
        "correction_method": "unchanged",
        "correction_changes": [],
    }


class FakeLocal:
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
        pass

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        self.requests.extend(requests)
        return {request.segment_index: self.candidate for request in requests}


class _CancelCommitLocal(FakeLocal):
    """On the first actual call, commit CANCELLED through a separate session."""

    def __init__(self, engine: object, source_id: object) -> None:
        super().__init__()
        self._engine = engine
        self._source_id = source_id
        self._flipped = False

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        if not self._flipped:
            self._flipped = True
            with Session(self._engine) as canceller:
                job = canceller.scalar(
                    select(ProcessingJob)
                    .where(
                        ProcessingJob.source_video_id == self._source_id,
                        ProcessingJob.kind == JobKind.RECONSTRUCTION,
                    )
                    .order_by(ProcessingJob.created_at.desc())
                )
                assert job is not None
                job.status = JobStatus.CANCELLED
                canceller.commit()
        return super().reconstruct_segments(requests)


class _Transport:
    """Records actual HTTP bodies; optionally toggles cancellation or fails."""

    def __init__(
        self,
        *,
        toggler: Any | None = None,
        fail_on_call: int | None = None,
        accept_text: bool = False,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.toggler = toggler
        self.fail_on_call = fail_on_call
        self.accept_text = accept_text

    def __call__(
        self,
        method: str,
        url: str,
        body: bytes | None,
        _headers: dict[str, str],
        _timeout: float,
    ) -> bytes:
        if method == "GET" and url.endswith("/api/tags"):
            return json.dumps(
                {"models": [{"name": "qwen3.5:4b", "digest": "sha256:tags"}]}
            ).encode()
        assert body is not None
        self.calls.append(json.loads(body))
        if self.toggler is not None:
            self.toggler.count += 1
        if self.fail_on_call is not None and len(self.calls) >= self.fail_on_call:
            raise ProviderResponseError("transport failure")
        targets = json.loads(self.calls[-1]["messages"][1]["content"])["targets"]
        entries = []
        for target in targets:
            if self.accept_text:
                entries.append(
                    {
                        "segment_id": target["segment_id"],
                        "corrected_text": "ضخمة",
                        "unchanged": False,
                        "confidence": 1.0,
                        "changes": [],
                    }
                )
            else:
                entries.append(
                    {
                        "segment_id": target["segment_id"],
                        "corrected_text": target["raw_text"],
                        "unchanged": True,
                    }
                )
        content = {"reconstructions": entries}
        return json.dumps(
            {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]},
            ensure_ascii=False,
        ).encode()


class _Toggle:
    def __init__(self) -> None:
        self.count = 0


def _ollama_provider(transport: _Transport) -> OllamaReconstructionProvider:
    return OllamaReconstructionProvider(
        base_url="http://ollama:11434",
        model="qwen3.5:4b",
        timeout_seconds=3,
        release_after_run=False,
        max_context_tokens=4096,
        request=transport,
    )


def _reconstructor(
    provider: Any, *, batch_windows: int = 16, batch_characters: int = 1_000_000, **kwargs: Any
) -> ContextualReconstructor:
    return ContextualReconstructor(
        provider,
        gemini_provider=None,
        routing=AdaptiveRoutingConfig(mode=RoutingMode.LOCAL_ONLY),
        gemini_budget=0,
        batch_windows=batch_windows,
        batch_characters=batch_characters,
        **kwargs,
    )


def _run(reconstructor: ContextualReconstructor, segments: list[dict[str, object]]) -> object:
    return reconstructor.reconstruct(
        segments,
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )


def _sent_envelope_fits(call: dict[str, object], max_context: int = 4096) -> bool:
    system_tokens = estimate_tokens(str(call["messages"][0]["content"]))
    user_tokens = estimate_tokens(str(call["messages"][1]["content"]))
    targets = json.loads(call["messages"][1]["content"])["targets"]
    budget = system_tokens + user_tokens + 64 + 256 * len(targets) + 128
    return budget <= max_context


# ---------------------------------------------------------------------------
# Blocker 1: every actual local request is a visible orchestration unit
# ---------------------------------------------------------------------------


def test_service_level_eight_target_micro_batch_is_context_split() -> None:
    """An eight-target window batch is split by orchestration into multiple
    actual provider requests, and every sent aggregate envelope fits."""

    transport = _Transport()
    provider = _ollama_provider(transport)
    segments = [_big_local_segment(i) for i in range(8)]
    result = _run(_reconstructor(provider), segments)
    assert len(transport.calls) > 1  # context-split, not one combined call
    assert all(_sent_envelope_fits(call) for call in transport.calls)
    assert len(result.segments) == 8
    # Exact segment mapping preserved across the split requests.
    for call in transport.calls:
        targets = json.loads(call["messages"][1]["content"])["targets"]
        assert [t["segment_id"] for t in targets] == sorted(t["segment_id"] for t in targets)


def test_cancellation_after_first_actual_request_prevents_second() -> None:
    toggler = _Toggle()
    transport = _Transport(toggler=toggler)
    provider = _ollama_provider(transport)
    segments = [_big_local_segment(i) for i in range(8)]
    reconstructor = _reconstructor(provider, is_cancelled=lambda: toggler.count >= 1)
    with pytest.raises(ReconstructionCancelled):
        _run(reconstructor, segments)
    assert len(transport.calls) == 1  # the second actual request never happens


def test_wall_time_ceiling_between_actual_requests() -> None:
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += 0.5
        return state["now"]

    transport = _Transport()
    provider = _ollama_provider(transport)
    segments = [_big_local_segment(i) for i in range(8)]
    reconstructor = _reconstructor(provider, monotonic=clock, local_wall_seconds=1.0)
    result = _run(reconstructor, segments)
    assert len(transport.calls) == 1  # wall budget expired before the second request
    assert result.metadata["local_budget"]["time_budget_exhausted"] is True
    unresolved = [
        segment
        for segment in result.segments
        if segment.escalation_reason == "local_time_budget_exhausted"
    ]
    assert unresolved


def test_partial_actual_request_failure_preserves_earlier_success() -> None:
    """Request 1 succeeds and is checkpointed; request 2 fails and only its
    targets fail/escalate/unresolve."""

    transport = _Transport(fail_on_call=2, accept_text=True)
    provider = _ollama_provider(transport)
    segments = [_big_local_segment(i, words=110) for i in range(2)]
    checkpoints: list[dict[int, object]] = []
    reconstructor = _reconstructor(
        provider,
        batch_windows=2,
        checkpoint=lambda results, progress: checkpoints.append(dict(results)),
    )
    result = _run(reconstructor, segments)
    assert len(transport.calls) == 2
    # Request 1's candidate was accepted and persisted in a checkpoint.
    assert result.segments[0].applied is True
    assert any(0 in snapshot and getattr(snapshot[0], "applied", False) for snapshot in checkpoints)
    # Only request 2's targets failed.
    assert result.segments[1].applied is False
    assert result.segments[1].local_result_state == "failure"


# ---------------------------------------------------------------------------
# Blocker 2: cancellation status is read fresh across API/worker sessions
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


def test_cross_session_api_cancellation_is_observed_by_worker(sqlite_engine: object) -> None:
    """An API-session CANCELLED commit is visible to the worker's next poll; no
    further provider call occurs, and the job/run stay CANCELLED."""

    segments = [_big_local_segment(i, words=6) for i in range(4)]
    source_id, engine = _setup_runner_env(sqlite_engine, segments)

    worker = Session(engine)
    local = _CancelCommitLocal(engine, source_id)
    runner = PipelineRunner(
        worker,
        {
            PipelineStage.CONTEXTUAL_RECONSTRUCTION: ContextualReconstructionExecutor(
                session=worker,
                reconstructor=ContextualReconstructor(
                    local,
                    gemini_provider=None,
                    routing=AdaptiveRoutingConfig(mode=RoutingMode.LOCAL_ONLY),
                    gemini_budget=0,
                    batch_windows=4,
                    batch_characters=1_000_000,
                ),
            )
        },
    )
    with pytest.raises(ReconstructionCancelled):
        runner.run(source_id, PipelineStage.CONTEXTUAL_RECONSTRUCTION)

    assert local.calls == 1  # no subsequent local call after cancellation observed

    run = worker.scalar(
        select(PipelineRun)
        .where(PipelineRun.source_video_id == source_id)
        .order_by(PipelineRun.created_at.desc())
    )
    job = worker.scalar(
        select(ProcessingJob)
        .where(
            ProcessingJob.source_video_id == source_id,
            ProcessingJob.kind == JobKind.RECONSTRUCTION,
        )
        .order_by(ProcessingJob.created_at.desc())
    )
    assert run.status is PipelineRunStatus.CANCELLED
    assert job.status is JobStatus.CANCELLED
    # The source lifecycle never advanced to the next stage.
    source = worker.get(SourceVideo, source_id)
    assert source is not None
    assert source.lifecycle_state is PipelineStage.INGEST
    worker.close()
