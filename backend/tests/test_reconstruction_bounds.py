"""Focused regression tests for bounded adaptive reconstruction.

These tests prove the corrective Stage 2.7 behaviors with deterministic fake
providers and stored transcript structures only: clean NO_LLM routing, trusted
Stage 2.5 repair semantics, bounded local micro-batching, local work ceilings,
cooperative cancellation, per-target reuse across restart, stable fingerprints,
secret sanitization, provider cleanup, and control-flow scalability. They never
use Ollama, Gemini, Whisper, audio, network, or real sleeps.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import (
    JobKind,
    JobStatus,
    PipelineRunStatus,
    PipelineStage,
    RefinementPriority,
    RightsStatus,
)
from app.db.base import Base
from app.models import PipelineRun, ProcessingJob, SourceVideo, Transcript
from app.pipeline.executor import ReconstructionCancelled
from app.pipeline.runner import PipelineRunner
from app.pipeline.stages import ContextualReconstructionExecutor
from app.runtime.heavy_model_lease import NoopHeavyModelLeaseFactory
from app.transcription.reconstruction.confidence import CONFIDENCE_POLICY_VERSION
from app.transcription.reconstruction.gemini import (
    GeminiErrorCategory,
    GeminiProviderError,
    GeminiReconstructionProvider,
)
from app.transcription.reconstruction.providers import (
    ProviderResponseError,
    ReconstructionRequest,
)
from app.transcription.reconstruction.routing import (
    AdaptiveRoutingConfig,
    ReconstructionRoute,
    RoutingMode,
    route_adaptive,
)
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
    ReconstructionCandidate,
)
from app.transcription.reconstruction.validation import VALIDATION_VERSION


def _words(probabilities: list[float], texts: list[str] | None = None) -> list[dict[str, object]]:
    texts = texts or [f"w{index}" for index in range(len(probabilities))]
    return [
        {"word": text, "probability": probability}
        for text, probability in zip(texts, probabilities)
    ]


def _segment(
    raw: str = "دخم",
    corrected: str | None = None,
    words: list[dict[str, object]] | None = None,
    operator: str | None = None,
    unchanged: bool = True,
    avg_logprob: float | None = None,
) -> dict[str, object]:
    corrected = corrected or raw
    segment: dict[str, object] = {
        "start": 0.0,
        "end": 1.0,
        "text": raw,
        "raw_text": raw,
        "corrected_text": corrected,
        "correction_applied": False,
        "correction_confidence": 0.0,
        "correction_method": "unchanged" if unchanged else "",
        "correction_changes": [],
    }
    if words is not None:
        segment["words"] = words
    if operator is not None:
        segment["operator_text"] = operator
    if avg_logprob is not None:
        segment["avg_logprob"] = avg_logprob
    return segment


def _mild_uncertainty(index: int) -> dict[str, object]:
    return _segment(
        raw="دخم",
        words=_words([0.98, 0.60, 0.98], [f"كلام{index}", "جديد", "مصري"]),
    )


class FakeLocal:
    def __init__(
        self,
        candidate: ReconstructionCandidate | None = None,
        available: bool = True,
        batch_failures: set[int] | None = None,
        fail_all: bool = False,
    ) -> None:
        self.calls = 0
        self.batches: list[list[int]] = []
        self.requests: list[ReconstructionRequest] = []
        self.candidate = candidate or ReconstructionCandidate(
            "provider-0", "ضخمة", provider_confidence=1.0
        )
        self.available = available
        self.batch_failures = batch_failures or set()
        self.fail_all = fail_all
        self.released = False

    def health(self) -> ProviderHealth:
        if not self.available:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE, "ollama", "qwen3.5:4b", None, "not installed"
            )
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
        indexes = [request.segment_index for request in requests]
        self.batches.append(indexes)
        self.requests.extend(requests)
        if self.fail_all or any(index in self.batch_failures for index in indexes):
            raise ProviderResponseError("local failure")
        return {request.segment_index: self.candidate for request in requests}


class _ScriptedLocal(FakeLocal):
    def __init__(self, candidates: dict[int, ReconstructionCandidate]) -> None:
        super().__init__()
        self._candidates = candidates

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        indexes = [request.segment_index for request in requests]
        self.batches.append(indexes)
        self.requests.extend(requests)
        return {
            request.segment_index: self._candidates.get(request.segment_index, self.candidate)
            for request in requests
        }


class _TogglingLocal(FakeLocal):
    def __init__(self, toggler: "_Toggle") -> None:
        super().__init__()
        self._toggler = toggler

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self._toggler.count += 1
        return super().reconstruct_segments(requests)


class _DbCancelLocal(FakeLocal):
    """Flip the latest reconstruction job to CANCELLED after the first batch."""

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


class FakeGemini:
    def __init__(
        self,
        candidate: ReconstructionCandidate | None = None,
        error: GeminiProviderError | None = None,
        fail_segment: int | None = None,
    ) -> None:
        self.model = "gemini-3.8-flash"
        self.calls = 0
        self.requests: list[ReconstructionRequest] = []
        self.candidate = candidate or ReconstructionCandidate(
            "provider-0", "ضخمة", provider_confidence=1.0
        )
        self.error = error
        self.fail_segment = fail_segment
        self.closed = False

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "gemini", "gemini-3.8-flash", "sha256:g", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
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

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def release(self) -> None:
        self.closed = True

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        self.requests.extend(requests)
        if self.error is not None:
            raise self.error
        if any(request.segment_index == self.fail_segment for request in requests):
            raise GeminiProviderError(GeminiErrorCategory.RATE_LIMITED)
        return {request.segment_index: self.candidate for request in requests}

    def usage_summary(self) -> dict[str, int]:
        return {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}


class _Toggle:
    def __init__(self) -> None:
        self.count = 0


def _reconstructor(
    local: FakeLocal | None,
    gemini: FakeGemini | None = None,
    mode: RoutingMode = RoutingMode.ADAPTIVE,
    budget: int = 10,
    **kwargs: Any,
) -> ContextualReconstructor:
    return ContextualReconstructor(
        local,
        gemini_provider=gemini,
        routing=AdaptiveRoutingConfig(mode=mode),
        gemini_budget=budget,
        priority=RefinementPriority.CANDIDATE,
        **kwargs,
    )


def _run(reconstructor: ContextualReconstructor, segments: list[dict[str, object]]) -> object:
    return reconstructor.reconstruct(
        segments,
        language="ar",
        transcription_fingerprint="asr-v1",
        correction_version="egyptian-ar-v1",
    )


# ---------------------------------------------------------------------------
# Routing corrections
# ---------------------------------------------------------------------------


def test_clean_unchanged_stage25_segment_routes_no_llm() -> None:
    decision = route_adaptive(
        _segment(words=_words([0.98, 0.99], ["ده", "كلام"])), AdaptiveRoutingConfig()
    )
    assert decision.route is ReconstructionRoute.NO_LLM


def test_79_clean_unchanged_segments_make_zero_provider_calls() -> None:
    segments = [
        _segment(raw=f"كلام {index}", words=_words([0.97, 0.98], ["ده", "كلام"]))
        for index in range(79)
    ]
    local, gemini = FakeLocal(), FakeGemini()
    reconstructor = _reconstructor(local, gemini)
    result = _run(reconstructor, segments)
    assert local.calls == 0
    assert gemini.calls == 0
    assert all(segment.route == "NO_LLM" for segment in result.segments)
    assert result.metadata["routing_counts"]["no_llm"] == 79


def test_mild_genuine_uncertainty_uses_local() -> None:
    decision = route_adaptive(
        _segment(words=_words([0.98, 0.60, 0.55, 0.98], ["ده", "كلام", "جديد", "مصري"])),
        AdaptiveRoutingConfig(),
    )
    assert decision.route is ReconstructionRoute.LOCAL


def test_trusted_stage25_repair_resolving_low_confidence_routes_no_llm() -> None:
    segment = _segment(raw="كلام مصري", corrected="كلمه مصري", unchanged=False)
    segment["correction_applied"] = True
    segment["correction_confidence"] = 0.95
    segment["correction_method"] = "lexicon"
    segment["correction_changes"] = [{"from": "كلام", "to": "كلمه", "reason": "phonetic"}]
    segment["words"] = _words([0.40, 0.99], ["كلام", "مصري"])
    decision = route_adaptive(segment, AdaptiveRoutingConfig())
    assert decision.route is ReconstructionRoute.NO_LLM
    assert "stage25_trusted_repair_resolved" in decision.evidence


def test_trusted_repair_with_independent_hard_span_routes_llm() -> None:
    segment = _segment(
        raw="كلام ش قادر م",
        corrected="كلمه ش قادر م",
        unchanged=False,
    )
    segment["correction_applied"] = True
    segment["correction_confidence"] = 0.95
    segment["correction_method"] = "lexicon"
    segment["correction_changes"] = [{"from": "كلام", "to": "كلمه", "reason": "phonetic"}]
    segment["words"] = _words([0.40, 0.25, 0.20, 0.30], ["كلام", "ش", "قادر", "م"])
    decision = route_adaptive(segment, AdaptiveRoutingConfig())
    assert decision.route is ReconstructionRoute.GEMINI_DIRECT
    assert "contiguous_very_low_words=3" in decision.evidence


def test_missing_probability_evidence_is_conservative_local_never_hard() -> None:
    segment = _segment(
        words=[{"word": "ده", "probability": None}, {"word": "كلام", "probability": None}]
    )
    decision = route_adaptive(segment, AdaptiveRoutingConfig())
    assert decision.route is ReconstructionRoute.LOCAL
    assert decision.route is not ReconstructionRoute.GEMINI_DIRECT


# ---------------------------------------------------------------------------
# Bounded local micro-batching
# ---------------------------------------------------------------------------


def test_ten_local_targets_with_batch_four_make_exactly_three_calls() -> None:
    local = FakeLocal()
    segments = [_mild_uncertainty(index) for index in range(10)]
    reconstructor = _reconstructor(local, batch_windows=4)
    result = _run(reconstructor, segments)
    assert local.calls == 3
    assert local.batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert all(segment.applied for segment in result.segments)


def test_local_batching_respects_character_limits() -> None:
    local = FakeLocal()
    long_text = "مصري" * 400
    segments = [
        _segment(raw=long_text, words=_words([0.98, 0.60], ["ده", "كلام"])),
        _segment(raw=long_text, words=_words([0.98, 0.60], ["ده", "كلام"])),
    ]
    reconstructor = _reconstructor(local, batch_characters=1500)
    _run(reconstructor, segments)
    assert local.calls == 2
    assert local.batches == [[0], [1]]


def test_one_invalid_batch_candidate_does_not_reject_valid_siblings() -> None:
    local = _ScriptedLocal(
        {
            0: ReconstructionCandidate("provider-0", "ضخمة", provider_confidence=1.0),
            1: ReconstructionCandidate("provider-0", "رقم 12345", provider_confidence=1.0),
            2: ReconstructionCandidate("provider-0", "ضخمة", provider_confidence=1.0),
        }
    )
    segments = [
        _segment(raw="دخم"),
        _segment(raw="خمسة"),
        _segment(raw="دخم"),
    ]
    reconstructor = _reconstructor(local, batch_windows=3)
    result = _run(reconstructor, segments)
    assert local.calls == 1
    assert result.segments[0].applied is True
    assert result.segments[1].applied is False
    assert result.segments[2].applied is True


# ---------------------------------------------------------------------------
# Local work ceilings
# ---------------------------------------------------------------------------


def test_local_target_ceiling_selects_strongest_candidates() -> None:
    local = FakeLocal()
    segments = [
        _segment(words=_words([0.60, 0.97, 0.96, 0.98], ["a1", "a2", "a3", "a4"])),
        _segment(words=_words([0.60, 0.61, 0.97, 0.98], ["b1", "b2", "b3", "b4"])),
        _segment(words=_words([0.60, 0.61, 0.62, 0.98], ["c1", "c2", "c3", "c4"])),
        _segment(words=_words([0.60, 0.61, 0.62, 0.97, 0.98], ["d1", "d2", "d3", "d4", "d5"])),
    ]
    reconstructor = _reconstructor(local, local_max_targets=2)
    result = _run(reconstructor, segments)
    attempted = {index for batch in local.batches for index in batch}
    assert attempted == {2, 3}
    assert result.segments[2].applied is True
    assert result.segments[3].applied is True
    assert result.segments[0].escalation_reason == "local_target_budget_exhausted"
    assert result.segments[1].escalation_reason == "local_target_budget_exhausted"
    assert result.metadata["local_budget"]["target_budget_exhausted"] is True
    assert result.metadata["cache_eligible"] is False


def test_local_wall_time_ceiling_terminates_safely_with_fake_clock() -> None:
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += 0.5
        return state["now"]

    local = FakeLocal()
    segments = [_mild_uncertainty(index) for index in range(4)]
    reconstructor = _reconstructor(local, monotonic=clock, local_wall_seconds=1.0)
    result = _run(reconstructor, segments)
    assert local.calls == 1
    assert result.segments[0].local_attempted is True
    assert result.segments[1].escalation_reason == "local_time_budget_exhausted"
    assert result.segments[2].escalation_reason == "local_time_budget_exhausted"
    assert result.segments[3].escalation_reason == "local_time_budget_exhausted"
    assert result.metadata["local_budget"]["time_budget_exhausted"] is True


# ---------------------------------------------------------------------------
# Cooperative cancellation
# ---------------------------------------------------------------------------


def test_cancellation_between_batches_prevents_later_provider_calls() -> None:
    toggler = _Toggle()
    local = _TogglingLocal(toggler)
    segments = [_mild_uncertainty(index) for index in range(4)]
    reconstructor = _reconstructor(
        local,
        batch_windows=2,
        is_cancelled=lambda: toggler.count >= 1,
    )
    with pytest.raises(ReconstructionCancelled):
        _run(reconstructor, segments)
    assert local.calls == 1


def test_cancellation_releases_providers_and_keeps_partial_checkpoints() -> None:
    toggler = _Toggle()
    local = _TogglingLocal(toggler)
    gemini = FakeGemini()
    checkpoints: list[tuple[dict[int, object], dict[str, object]]] = []
    segments = [_mild_uncertainty(index) for index in range(4)]
    reconstructor = _reconstructor(
        local,
        gemini,
        batch_windows=2,
        is_cancelled=lambda: toggler.count >= 1,
        checkpoint=lambda results, progress: checkpoints.append((dict(results), progress)),
    )
    with pytest.raises(ReconstructionCancelled):
        _run(reconstructor, segments)
    assert local.released is True
    assert gemini.closed is True
    assert checkpoints
    assert checkpoints[-1][1]["cancellation_requested"] is True


def test_cancelled_job_never_becomes_successful_or_schedules_next_stage(
    sqlite_engine: object,
) -> None:
    segments = [_mild_uncertainty(index) for index in range(4)]
    source, _transcript, session = _setup_executor_env(sqlite_engine, segments)
    job = ProcessingJob(source_video_id=source.id, kind=JobKind.RECONSTRUCTION)
    session.add(job)
    session.commit()
    job_id = job.id

    local = _DbCancelLocal(session, source.id)
    reconstructor = _reconstructor(local, batch_windows=2)
    executor = ContextualReconstructionExecutor(session=session, reconstructor=reconstructor)
    runner = PipelineRunner(
        session,
        {PipelineStage.CONTEXTUAL_RECONSTRUCTION: executor},
    )
    with pytest.raises(ReconstructionCancelled):
        runner.run(source.id, PipelineStage.CONTEXTUAL_RECONSTRUCTION, job_id=job_id)

    session.refresh(job)
    run = session.scalar(
        select(PipelineRun)
        .where(PipelineRun.source_video_id == source.id)
        .order_by(PipelineRun.created_at.desc())
    )
    assert job.status is JobStatus.CANCELLED
    assert run.status is PipelineRunStatus.CANCELLED
    # The source lifecycle never advanced to the next pipeline stage.
    assert source.lifecycle_state is PipelineStage.INGEST
    # Checkpointed completed work survives the cancellation.
    assert local.calls == 1


def _setup_executor_env(
    sqlite_engine: object, segments: list[dict[str, object]]
) -> tuple[SourceVideo, Transcript, Session]:
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
    return source, transcript, session


def test_checkpointed_results_survive_cancellation_in_db(sqlite_engine: object) -> None:
    """Cancellation persists the completed batch so a restart reuses it."""

    segments = [_mild_uncertainty(index) for index in range(4)]
    source, _transcript, session = _setup_executor_env(sqlite_engine, segments)
    job = ProcessingJob(source_video_id=source.id, kind=JobKind.RECONSTRUCTION)
    session.add(job)
    session.commit()
    job_id = job.id

    local = _DbCancelLocal(session, source.id)
    reconstructor = _reconstructor(local, batch_windows=2)
    executor = ContextualReconstructionExecutor(session=session, reconstructor=reconstructor)
    runner = PipelineRunner(session, {PipelineStage.CONTEXTUAL_RECONSTRUCTION: executor})
    with pytest.raises(ReconstructionCancelled):
        runner.run(source.id, PipelineStage.CONTEXTUAL_RECONSTRUCTION, job_id=job_id)
    source_id = source.id
    session.close()

    with Session(sqlite_engine) as fresh:
        transcript = fresh.scalar(select(Transcript).where(Transcript.source_video_id == source_id))
        assert transcript is not None
        assert transcript.segments[0].get("reconstruction_route") in {"LOCAL", "GEMINI_DIRECT"}
        assert transcript.segments[0].get("contextual_reconstructed_text")
        assert transcript.segments[1].get("reconstruction_route") in {"LOCAL", "GEMINI_DIRECT"}
        assert transcript.segments[2].get("reconstruction_route") is None
        assert transcript.segments[3].get("reconstruction_route") is None
        assert transcript.reconstruction_metadata.get("cache_eligible") is False
        assert transcript.reconstruction_metadata.get("partial") is True


class _CancelledExecutor:
    def input_fingerprint(self, source: SourceVideo) -> str:
        return ""

    def execute(self, source: SourceVideo, *, force: bool = False) -> object:
        raise ReconstructionCancelled("test cancellation")


def test_worker_does_not_schedule_next_stage_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.db.session import create_session_factory
    from app.workers import tasks

    factory = create_session_factory()
    Base.metadata.create_all(factory().bind)
    with factory() as session:
        source = SourceVideo(
            source_uri="file:///tmp/source.mp4",
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
            segments=[_mild_uncertainty(0)],
        )
        session.add(transcript)
        job = ProcessingJob(source_video_id=source.id, kind=JobKind.RECONSTRUCTION)
        session.add(job)
        session.commit()
        source_id = source.id
        job_id = job.id

    recorded: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        tasks.run_pipeline_stage, "delay", lambda *args, **kwargs: recorded.append(args)
    )
    monkeypatch.setattr(
        tasks,
        "_stage_executors",
        lambda session: {PipelineStage.CONTEXTUAL_RECONSTRUCTION: _CancelledExecutor()},
    )
    with pytest.raises(ReconstructionCancelled):
        tasks.run_pipeline_stage(
            str(source_id), PipelineStage.CONTEXTUAL_RECONSTRUCTION.value, str(job_id)
        )
    assert recorded == []


# ---------------------------------------------------------------------------
# Per-target reuse across restart
# ---------------------------------------------------------------------------


def test_partial_gemini_success_reused_after_fake_429_restart(sqlite_engine: object) -> None:
    hard_words = [
        {"word": "م", "probability": 0.30},
        {"word": "ش", "probability": 0.25},
        {"word": "قادر", "probability": 0.20},
        {"word": "يفهم", "probability": 0.90},
    ]
    segments = [
        {
            "start": float(i),
            "end": float(i + 1),
            "text": "دخم",
            "raw_text": "دخم",
            "corrected_text": "دخم",
            "words": hard_words,
        }
        for i in range(5)
    ]
    source, transcript, session = _setup_executor_env(sqlite_engine, segments)

    # The fifth target's local fallback is also rejected, so it stays unfinished
    # and is the only eligible unresolved work on restart.
    local = _ScriptedLocal(
        {4: ReconstructionCandidate("provider-0", "رقم 12345", provider_confidence=1.0)}
    )
    gemini = FakeGemini(fail_segment=4)
    reconstructor = _reconstructor(local, gemini, budget=5)
    executor = ContextualReconstructionExecutor(session=session, reconstructor=reconstructor)
    executor.execute(source, force=True)
    session.refresh(source)
    assert gemini.calls == 5
    assert source.transcript.reconstruction_metadata["cache_eligible"] is False
    assert source.transcript.segments[4]["reconstruction_status"] == "LOW_CONFIDENCE_UNRESOLVED"
    assert source.transcript.segments[4]["gemini_result_state"] == "failure:RATE_LIMITED"

    fresh_gemini = FakeGemini()
    fresh_local = _ScriptedLocal({})
    retry_reconstructor = _reconstructor(fresh_local, fresh_gemini, budget=5)
    retry_executor = ContextualReconstructionExecutor(
        session=session, reconstructor=retry_reconstructor
    )
    retry_executor.execute(source, force=False)
    session.refresh(source)
    # The four accepted Gemini targets generate zero new Gemini requests.
    assert fresh_gemini.calls == 1
    assert [request.segment_index for request in fresh_gemini.requests] == [4]
    assert source.transcript.reconstruction_metadata["cache_eligible"] is True


# ---------------------------------------------------------------------------
# Fingerprints and identity stability
# ---------------------------------------------------------------------------


def test_fingerprint_changes_with_stage25_and_acoustic_evidence() -> None:
    base = _segment(words=_words([0.98, 0.99], ["ده", "كلام"]))
    changed_method = dict(base)
    changed_method["correction_method"] = "lexicon"
    changed_method["correction_applied"] = True
    changed_method["correction_confidence"] = 0.95
    changed_method["correction_changes"] = [{"from": "ده", "to": "دا"}]

    changed_words = _segment(words=_words([0.80, 0.99], ["ده", "كلام"]))

    changed_acoustic = _segment(words=_words([0.98, 0.99], ["ده", "كلام"]), avg_logprob=-1.2)

    reconstructor = _reconstructor(FakeLocal())
    fp_base = _run(reconstructor, [base]).fingerprint
    assert _run(reconstructor, [changed_method]).fingerprint != fp_base
    assert _run(reconstructor, [changed_words]).fingerprint != fp_base
    assert _run(reconstructor, [changed_acoustic]).fingerprint != fp_base


def test_provider_outage_does_not_change_stable_cached_identity() -> None:
    def raise_outage() -> object:
        raise ConnectionError("gemini offline")

    healthy = GeminiReconstructionProvider(
        api_key="test-key",
        model="gemini-3.8-flash",
        client_factory=lambda: _FakeSdkClient(),
        owns_client=True,
    )
    outage = GeminiReconstructionProvider(
        api_key="test-key",
        model="gemini-3.8-flash",
        client_factory=raise_outage,
        owns_client=True,
    )

    assert healthy.runtime_identity() == outage.runtime_identity()
    assert outage.refresh_runtime_identity() == outage.runtime_identity()
    assert healthy.runtime_identity()["digest"] == outage.runtime_identity()["digest"]


# ---------------------------------------------------------------------------
# Secret safety and lazy Gemini clients
# ---------------------------------------------------------------------------


class _FakeSdkModels:
    def generate_content(self, model: str, contents: str, config: object) -> object:
        return _SdkResponse()


class _SdkResponse:
    parsed = {
        "reconstructions": [
            {"segment_id": 0, "corrected_text": "ضخمة", "unchanged": False, "confidence": 0.95}
        ]
    }
    text = None
    usage_metadata = None
    candidates = [type("C", (), {"finish_reason": type("F", (), {"name": "STOP"})()})()]
    prompt_feedback = None


class _FakeSdkClient:
    def __init__(self) -> None:
        self.models = _FakeSdkModels()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_fake_secret_absent_from_traceback_and_errors() -> None:
    import traceback

    class SecretLeak(Exception):
        code = 401

        def __init__(self) -> None:
            super().__init__("unauthorized request AIzaSENTINELSECRET")

    def leak() -> object:
        raise SecretLeak()

    provider = GeminiReconstructionProvider(
        api_key="AIzaSENTINELSECRET",
        model="gemini-3.8-flash",
        client_factory=leak,
        owns_client=True,
    )
    request = ReconstructionRequest(segment_index=0, raw_text="دخم", corrected_text="دخم")
    with pytest.raises(GeminiProviderError) as raised:
        provider.reconstruct_segments([request])
    tb = traceback.format_exception(type(raised.value), raised.value, raised.value.__traceback__)
    assert "AIzaSENTINELSECRET" not in str(raised.value)
    assert "AIzaSENTINELSECRET" not in repr(raised.value)
    assert "AIzaSENTINELSECRET" not in "".join(tb)


def test_cache_hit_creates_no_gemini_client_or_network_call(sqlite_engine: object) -> None:
    hard_words = [
        {"word": "م", "probability": 0.30},
        {"word": "ش", "probability": 0.25},
        {"word": "قادر", "probability": 0.20},
        {"word": "يفهم", "probability": 0.90},
    ]
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "مش قادر يفهم",
            "raw_text": "مش قادر يفهم",
            "corrected_text": "مش قادر يفهم",
            "words": hard_words,
        }
    ]
    source, _transcript, session = _setup_executor_env(sqlite_engine, segments)

    factory_calls: list[int] = []

    def factory() -> object:
        factory_calls.append(1)
        return _FakeSdkClient()

    gemini = GeminiReconstructionProvider(
        api_key="test-key",
        model="gemini-3.8-flash",
        client_factory=factory,
        owns_client=True,
    )
    local = FakeLocal(
        candidate=ReconstructionCandidate("provider-0", "ضخمة", provider_confidence=1.0)
    )
    reconstructor = _reconstructor(local, gemini, budget=5)
    executor = ContextualReconstructionExecutor(session=session, reconstructor=reconstructor)
    executor.execute(source, force=True)
    assert gemini.usage_summary()["total_token_count"] == 0
    executor.execute(source, force=False)
    assert len(factory_calls) == 1  # client built once, never on the cache hit
    assert gemini.usage_summary()["total_token_count"] == 0


def test_all_no_llm_creates_no_gemini_client_or_local_lease(sqlite_engine: object) -> None:
    segments = [
        {
            "start": float(i),
            "end": float(i + 1),
            "text": f"كلام {i}",
            "raw_text": f"كلام {i}",
            "corrected_text": f"كلام {i}",
            "words": [
                {"word": "ده", "probability": 0.98},
                {"word": "كلام", "probability": 0.99},
            ],
        }
        for i in range(3)
    ]
    for segment in segments:
        segment["correction_applied"] = False
        segment["correction_confidence"] = 0.0
        segment["correction_method"] = "unchanged"
        segment["correction_changes"] = []
    source, _transcript, session = _setup_executor_env(sqlite_engine, segments)

    factory_calls: list[int] = []

    def factory() -> object:
        factory_calls.append(1)
        return _FakeSdkClient()

    gemini = GeminiReconstructionProvider(
        api_key="test-key",
        model="gemini-3.8-flash",
        client_factory=factory,
        owns_client=True,
    )
    local = FakeLocal(
        candidate=ReconstructionCandidate("provider-0", "ضخمة", provider_confidence=1.0)
    )
    leases = NoopHeavyModelLeaseFactory()
    reconstructor = _reconstructor(local, gemini, budget=5)
    executor = ContextualReconstructionExecutor(
        session=session, reconstructor=reconstructor, lease_factory=leases
    )
    executor.execute(source, force=True)
    assert factory_calls == []
    assert leases.events == []
    assert gemini.usage_summary()["total_token_count"] == 0


# ---------------------------------------------------------------------------
# Provider cleanup
# ---------------------------------------------------------------------------


def test_owned_provider_resources_close_once_on_all_exit_paths() -> None:
    # Normal success path closes the owned SDK client exactly once.
    client = _FakeSdkClient()
    provider = GeminiReconstructionProvider(
        api_key="test-key",
        model="gemini-3.8-flash",
        client_factory=lambda: client,
        owns_client=True,
    )
    provider.reconstruct_segments(
        [ReconstructionRequest(segment_index=0, raw_text="دخم", corrected_text="دخم")]
    )
    provider.release()
    assert client.closed is True

    # Failure path still closes the owned client exactly once.
    failing_client = _FakeSdkClient()
    failing_client.models = type(
        "M",
        (),
        {"generate_content": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))},
    )()
    provider2 = GeminiReconstructionProvider(
        api_key="test-key",
        model="gemini-3.8-flash",
        client_factory=lambda: failing_client,
        owns_client=True,
    )
    with pytest.raises(GeminiProviderError):
        provider2.reconstruct_segments(
            [ReconstructionRequest(segment_index=0, raw_text="دخم", corrected_text="دخم")]
        )
    provider2.release()
    assert failing_client.closed is True

    # Cancellation cleanup releases the local provider and closes Gemini.
    toggler = _Toggle()
    local = _TogglingLocal(toggler)
    gemini = FakeGemini()
    segments = [_mild_uncertainty(index) for index in range(4)]
    reconstructor = _reconstructor(
        local,
        gemini,
        batch_windows=2,
        is_cancelled=lambda: toggler.count >= 1,
    )
    with pytest.raises(ReconstructionCancelled):
        _run(reconstructor, segments)
    assert local.released is True
    assert gemini.closed is True


# ---------------------------------------------------------------------------
# Control-flow scalability
# ---------------------------------------------------------------------------


def test_6000_segment_control_flow_is_bounded() -> None:
    segments: list[dict[str, object]] = []
    for index in range(6_000):
        if index < 5_900:
            segments.append(
                _segment(
                    raw=f"كلام {index}",
                    words=_words([0.98, 0.99], ["ده", "كلام"]),
                )
            )
        else:
            segments.append(_mild_uncertainty(index))
    local = FakeLocal()
    gemini = FakeGemini()
    reconstructor = _reconstructor(
        local,
        gemini,
        budget=5,
        batch_windows=8,
        local_max_targets=64,
    )
    result = _run(reconstructor, segments)
    assert len(result.segments) == 6_000
    assert local.calls == 8  # 64 local targets in batches of 8
    assert gemini.calls == 0
    unresolved = [
        segment
        for segment in result.segments
        if segment.escalation_reason
        in {
            "local_target_budget_exhausted",
            "local_time_budget_exhausted",
        }
    ]
    assert len(unresolved) == 100 - 64
