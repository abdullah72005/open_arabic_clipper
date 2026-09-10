"""Stage 2.7 finalization: INDEX/CANDIDATE/FINAL_CLIP priorities, targeted
window refinement, Qwen disabled-by-default, and the local-wall/Gemini-backlog
ceiling fix.

These tests use fake local/Gemini providers, fake clocks, and SQLite sessions
only; no live model or Google calls are made.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import (
    JobStatus,
    PipelineRunStatus,
    PipelineStage,
    ReconstructionStatus,
    RefinementPriority,
    RightsStatus,
)
from app.db.base import Base
from app.models import PipelineRun, ProcessingJob, SourceVideo, Transcript
from app.pipeline.executor import ReconstructionCancelled
from app.pipeline.fingerprints import reconstruction_target_fingerprint
from app.pipeline.runner import PipelineRunner
from app.pipeline.stages import ContextualReconstructionExecutor
from app.transcription.reconstruction.confidence import CONFIDENCE_POLICY_VERSION
from app.transcription.reconstruction.providers import (
    ProviderResponseError,
    ReconstructionCandidate,
    ReconstructionRequest,
)
from app.transcription.reconstruction.refine import RefinementError, refine_transcript_window
from app.transcription.reconstruction.routing import AdaptiveRoutingConfig, RoutingMode
from app.transcription.reconstruction.service import ContextualReconstructor
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
)
from app.transcription.reconstruction.validation import VALIDATION_VERSION

CANDIDATE = RefinementPriority.CANDIDATE
FINAL_CLIP = RefinementPriority.FINAL_CLIP
INDEX = RefinementPriority.INDEX


def _segment(
    index: int,
    raw: str = "دخم",
    corrected: str = "دخم",
    operator_text: str | None = None,
    *,
    words: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    segment: dict[str, object] = {
        "start": float(index),
        "end": float(index + 1),
        "text": raw,
        "raw_text": raw,
        "corrected_text": corrected,
        "final_text": operator_text or corrected,
        "words": words
        or [
            {"word": "ك", "probability": 0.98},
            {"word": "لام", "probability": 0.98},
            {"word": "مشكوك", "probability": 0.60},
        ],
        "correction_applied": False,
        "correction_confidence": 0.0,
        "correction_method": "unchanged",
        "correction_changes": [],
        "dialect_profile": None,
        "dialect_confidence": 0.0,
        "dialect_selection": "unknown",
        "dialect_policy_version": "dialect-policy-v2",
        "code_switch_suspected": False,
        "code_switch_tokens": [],
    }
    if operator_text:
        segment["operator_text"] = operator_text
    return segment


def _code_switch_segment(index: int) -> dict[str, object]:
    raw = "هنعمل deploy بعد الـ review"
    segment = _segment(
        index,
        raw=raw,
        corrected=raw,
        words=[
            {"word": "هنعمل", "probability": 0.98},
            {"word": "deploy", "probability": 0.95},
            {"word": "بعد", "probability": 0.97},
            {"word": "الـ", "probability": 0.93},
            {"word": "review", "probability": 0.94},
        ],
    )
    segment["dialect_profile"] = "EGYPTIAN"
    segment["dialect_confidence"] = 0.95
    segment["dialect_selection"] = "detected"
    segment["code_switch_suspected"] = True
    segment["code_switch_tokens"] = ["deploy", "review"]
    return segment


def _identity(provider: str = "ollama") -> dict[str, object]:
    return {
        "provider": provider,
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


class CountingLocal:
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
        return dict(_identity("ollama"))

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


class CountingGemini:
    def __init__(self) -> None:
        self.model = "gemini-3.8-flash"
        self.calls = 0

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            ProviderAvailability.AVAILABLE, "gemini", self.model, "sha256:g", "ok"
        )

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "gemini",
            "model": self.model,
            "digest": "sha256:g",
            "prompt_hash": "p",
            "schema_version": "s",
            "timeout_seconds": 30.0,
            "retry_attempts": 1,
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


class FailingLocal(CountingLocal):
    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        self.requests.extend(requests)
        raise ProviderResponseError("local provider failure")


class ClockedFailingLocal(CountingLocal):
    """Fails each request and advances the wall clock while doing so."""

    def __init__(self, state: dict[str, float], nows: tuple[float, ...]) -> None:
        super().__init__()
        self.state = state
        self.nows = nows
        self.calls = 0

    def reconstruct_segments(
        self, requests: list[ReconstructionRequest]
    ) -> dict[int, ReconstructionCandidate]:
        self.calls += 1
        self.state["now"] = self.nows[self.calls - 1]
        raise ProviderResponseError("local provider failure")


def _local_only_reconstructor(
    local: CountingLocal,
    *,
    batch_windows: int = 4,
    batch_characters: int = 1_000_000,
    **kwargs: object,
) -> ContextualReconstructor:
    return ContextualReconstructor(
        local,
        gemini_provider=None,
        routing=AdaptiveRoutingConfig(mode=RoutingMode.LOCAL_ONLY),
        gemini_budget=0,
        batch_windows=batch_windows,
        batch_characters=batch_characters,
        priority=CANDIDATE,
        **kwargs,
    )


def _setup_transcript(sqlite_engine: object, segments: list[dict[str, object]]) -> object:
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
            duration=float(len(segments)),
            segments=segments,
        )
        session.add(transcript)
        session.commit()
        source_id = source.id
    return source_id


def _executor(
    session: Session,
    reconstructor: ContextualReconstructor,
) -> ContextualReconstructionExecutor:
    return ContextualReconstructionExecutor(session=session, reconstructor=reconstructor)


# 1. + 2. + 3. Default INDEX skips Qwen and Gemini and defers truthfully


def test_index_executor_makes_zero_qwen_and_zero_gemini_calls(sqlite_engine: object) -> None:
    segments = [_segment(i) for i in range(4)]
    source_id = _setup_transcript(sqlite_engine, segments)
    local = CountingLocal()
    gemini = CountingGemini()

    with Session(sqlite_engine) as session:
        reconstructor = ContextualReconstructor(
            local,
            gemini_provider=gemini,
            routing=AdaptiveRoutingConfig(mode=RoutingMode.ADAPTIVE),
            gemini_budget=10,
        )
        _executor(session, reconstructor).execute(session.get(SourceVideo, source_id), force=True)
        session.commit()
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )

        assert local.calls == 0  # Qwen never invoked
        assert gemini.calls == 0  # Gemini never invoked
        assert transcript is not None
        assert transcript.reconstruction_metadata["priority"] == INDEX.value
        assert transcript.reconstruction_metadata["index_deferred"] is True
        assert transcript.reconstruction_metadata["provider_calls"] == 0
        assert transcript.reconstruction_metadata["gemini_calls"] == 0
        assert transcript.reconstruction_metadata["cache_eligible"] is True
        assert transcript.reconstruction_method == "index_deferred"
        assert transcript.reconstruction_status is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED


def test_index_executor_preserves_stage25_text_and_defers_not_fails(sqlite_engine: object) -> None:
    segments = [_segment(i) for i in range(3)]
    source_id = _setup_transcript(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        _executor(
            session,
            ContextualReconstructor(CountingLocal(), gemini_provider=None),
        ).execute(session.get(SourceVideo, source_id), force=True)
        session.commit()
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )

        assert transcript is not None
        assert transcript.reconstruction_status is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
        for segment in transcript.segments:
            assert segment["contextual_reconstructed_text"] == "دخم"
            assert segment["final_text"] == "دخم"
            assert (
                segment["reconstruction_status"]
                == ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED.value
            )
            assert segment["escalation_reason"] == "index_priority_deferred"
            assert segment["reconstruction_method"] == "index_deferred"
            assert segment["refinement_priority"] == INDEX.value
            assert segment["needs_refinement"] is True
            # Not a provider failure: INDEX intentionally deferred the work.
            assert segment["reconstruction_quality_flags"] == []


def test_index_with_no_providers_configured_completes_successfully() -> None:
    result = ContextualReconstructor(None).reconstruct(
        [_segment(0), _segment(1)],
        language="ar",
        transcription_fingerprint="asr-fp",
        correction_version="egyptian-ar-v1",
    )

    assert result.metadata["index_deferred"] is True
    assert result.metadata["priority"] == INDEX.value
    assert result.metadata["cache_eligible"] is True
    assert all(
        segment.status is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
        for segment in result.segments
    )
    assert result.segments[0].contextual_reconstructed_text == "دخم"


# 5. Intentional LOCAL_ONLY still works when configured


def test_local_only_mode_uses_qwen_and_never_gemini() -> None:
    local = CountingLocal()
    gemini = CountingGemini()
    reconstructor = ContextualReconstructor(
        local,
        gemini_provider=gemini,
        routing=AdaptiveRoutingConfig(mode=RoutingMode.LOCAL_ONLY),
        gemini_budget=10,
        priority=CANDIDATE,
    )

    result = reconstructor.reconstruct(
        [_segment(0)],
        language="ar",
        transcription_fingerprint="asr-fp",
        correction_version="egyptian-ar-v1",
    )

    assert local.calls == 1
    assert gemini.calls == 0
    assert result.segments[0].applied is True
    assert result.segments[0].contextual_reconstructed_text == "ضخمة"


# 6. + 7. + 8. + 9. + 10. Targeted window refinement


def test_candidate_refinement_invokes_bounded_provider_work_on_window(
    sqlite_engine: object,
) -> None:
    segments = [_segment(i) for i in range(6)]
    source_id = _setup_transcript(sqlite_engine, segments)
    local = CountingLocal()
    reconstructor = _local_only_reconstructor(local, batch_windows=4)

    with Session(sqlite_engine) as session:
        outcome = refine_transcript_window(
            session,
            source_id,
            start_time=1.5,
            end_time=3.5,
            priority=CANDIDATE,
            reconstructor=reconstructor,
        )
        session.commit()

        assert outcome.target_indexes == (1, 2, 3)
        assert len(outcome.results) == 3
        assert outcome.accepted_indexes == (1, 2, 3)
        assert local.calls == 1  # one bounded micro-batch for the three targets
        # Only the three window segments were targeted.
        assert {request.segment_index for request in local.requests} == {1, 2, 3}

        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )
        assert transcript is not None
        for index in (1, 2, 3):
            assert transcript.segments[index]["reconstruction_applied"] is True
            assert transcript.segments[index]["final_text"] == "ضخمة"
            assert transcript.segments[index]["refinement_priority"] == CANDIDATE.value
        # Context/neighboring segments were never mutation targets.
        for index in (0, 4, 5):
            assert "contextual_reconstructed_text" not in transcript.segments[index]
            assert "reconstruction_status" not in transcript.segments[index]
        assert transcript.reconstruction_metadata["priority"] == CANDIDATE.value
        # Window-specific detail lives in the outcome, not the source summary.
        assert outcome.metadata["window"] == {"start": 1.5, "end": 3.5}
        assert outcome.metadata["target_indexes"] == [1, 2, 3]
        assert outcome.metadata["applied_in_window"] == 3


def test_final_clip_priority_is_accepted_by_refinement_contract(sqlite_engine: object) -> None:
    segments = [_segment(i) for i in range(4)]
    source_id = _setup_transcript(sqlite_engine, segments)
    local = CountingLocal()

    with Session(sqlite_engine) as session:
        outcome = refine_transcript_window(
            session,
            source_id,
            start_time=0.5,
            end_time=2.5,
            priority=FINAL_CLIP,
            reconstructor=_local_only_reconstructor(local),
        )
        session.commit()

        assert outcome.priority is FINAL_CLIP
        assert outcome.target_indexes == (0, 1, 2)
        assert local.calls >= 1


def test_refinement_preserves_raw_asr_and_all_timestamps(sqlite_engine: object) -> None:
    segments = [_segment(i) for i in range(5)]
    before = [
        (str(segment["raw_text"]), segment["start"], segment["end"], segment["words"])
        for segment in segments
    ]
    source_id = _setup_transcript(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        refine_transcript_window(
            session,
            source_id,
            start_time=1.0,
            end_time=3.0,
            priority=CANDIDATE,
            reconstructor=_local_only_reconstructor(CountingLocal()),
        )
        session.commit()
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )
        after = [
            (str(segment["raw_text"]), segment["start"], segment["end"], segment["words"])
            for segment in transcript.segments
        ]
        assert after == before


def test_manual_override_still_wins_over_refinement(sqlite_engine: object) -> None:
    segments = [
        _segment(0),
        _segment(1, raw="خطي", corrected="تصحيح", operator_text="يدوي"),
        _segment(2),
    ]
    source_id = _setup_transcript(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        outcome = refine_transcript_window(
            session,
            source_id,
            start_time=0.5,
            end_time=2.5,
            priority=CANDIDATE,
            reconstructor=_local_only_reconstructor(CountingLocal()),
        )
        session.commit()
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )

        assert transcript.segments[1]["final_text"] == "يدوي"
        assert (
            transcript.segments[1]["reconstruction_status"]
            == ReconstructionStatus.MANUAL_OVERRIDE.value
        )
        assert 1 not in outcome.accepted_indexes


def test_refinement_rejects_out_of_bounds_window(sqlite_engine: object) -> None:
    segments = [_segment(i) for i in range(3)]
    source_id = _setup_transcript(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        with pytest.raises(RefinementError, match="end_time exceeds"):
            refine_transcript_window(
                session,
                source_id,
                start_time=0.0,
                end_time=100.0,
                priority=CANDIDATE,
                reconstructor=_local_only_reconstructor(CountingLocal()),
            )
        with pytest.raises(RefinementError, match="0 <= start_time"):
            refine_transcript_window(
                session,
                source_id,
                start_time=-1.0,
                end_time=2.0,
                priority=CANDIDATE,
                reconstructor=_local_only_reconstructor(CountingLocal()),
            )


# 11. Fingerprints distinguish priority, window scope, and provider identity


def test_fingerprints_distinguish_priority_and_window_scope() -> None:
    segments = [_segment(i) for i in range(4)]
    base_kwargs = dict(
        language="ar",
        transcription_fingerprint="asr-fp",
        correction_version="egyptian-ar-v1",
    )

    index_result = ContextualReconstructor(None).reconstruct(segments, **base_kwargs)
    candidate_whole = ContextualReconstructor(
        CountingLocal(), gemini_provider=None, priority=CANDIDATE
    ).reconstruct(segments, **base_kwargs)
    window_a = ContextualReconstructor(
        CountingLocal(), gemini_provider=None, priority=CANDIDATE
    ).reconstruct(segments, **base_kwargs, target_indexes=[0, 1])
    window_b = ContextualReconstructor(
        CountingLocal(), gemini_provider=None, priority=CANDIDATE
    ).reconstruct(segments, **base_kwargs, target_indexes=[2, 3])

    fingerprints = {
        index_result.fingerprint,
        candidate_whole.fingerprint,
        window_a.fingerprint,
        window_b.fingerprint,
    }
    assert len(fingerprints) == 4  # INDEX != CANDIDATE != window A != window B


def test_refinement_fingerprints_distinguish_windows(sqlite_engine: object) -> None:
    segments = [_segment(i) for i in range(6)]
    source_id = _setup_transcript(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        first = refine_transcript_window(
            session,
            source_id,
            start_time=0.5,
            end_time=1.5,
            priority=CANDIDATE,
            reconstructor=_local_only_reconstructor(CountingLocal()),
        )
        second = refine_transcript_window(
            session,
            source_id,
            start_time=2.5,
            end_time=3.5,
            priority=FINAL_CLIP,
            reconstructor=_local_only_reconstructor(CountingLocal()),
        )
        assert first.fingerprint != second.fingerprint


# 12. Cancellation still works around targeted provider execution


def test_refinement_cancellation_stops_after_first_provider_unit(sqlite_engine: object) -> None:
    segments = [_segment(i) for i in range(6)]
    source_id = _setup_transcript(sqlite_engine, segments)
    state = {"cancel": False}

    class _CancellingLocal(CountingLocal):
        def reconstruct_segments(
            self, requests: list[ReconstructionRequest]
        ) -> dict[int, ReconstructionCandidate]:
            state["cancel"] = True
            return super().reconstruct_segments(requests)

    local = _CancellingLocal()
    with Session(sqlite_engine) as session:
        with pytest.raises(ReconstructionCancelled):
            refine_transcript_window(
                session,
                source_id,
                start_time=0.0,
                end_time=2.0,
                priority=CANDIDATE,
                reconstructor=_local_only_reconstructor(local, batch_windows=1),
                is_cancelled=lambda: state["cancel"],
            )
        assert local.calls == 1  # no further provider unit after cancellation


# 13. Local wall-time expiry invalidates queued local-origin Gemini escalations


def test_wall_ceiling_expiry_drops_queued_local_gemini_escalations() -> None:
    state = {"now": 0.0}

    def clock() -> float:
        return state["now"]

    segments = [_segment(i) for i in range(3)]
    local = ClockedFailingLocal(state, nows=(0.9, 1.1, 1.1))
    gemini = CountingGemini()
    reconstructor = ContextualReconstructor(
        local,
        gemini_provider=gemini,
        routing=AdaptiveRoutingConfig(mode=RoutingMode.ADAPTIVE),
        gemini_budget=10,
        batch_windows=1,
        batch_characters=1_000_000,
        monotonic=clock,
        local_wall_seconds=1.0,
        priority=CANDIDATE,
    )

    result = reconstructor.reconstruct(
        segments,
        language="ar",
        transcription_fingerprint="asr-fp",
        correction_version="egyptian-ar-v1",
    )

    # Two local requests failed; the first queued a Gemini escalation while the
    # ceiling was still active, and it was invalidated once the ceiling expired.
    assert local.calls == 2
    assert gemini.calls == 0  # no Gemini call for the local backlog
    assert result.metadata["local_budget"]["time_budget_exhausted"] is True
    for segment in result.segments:
        assert segment.gemini_attempted is False
        assert (
            segment.escalation_reason and "local_time_budget_exhausted" in segment.escalation_reason
        )
    # Attempted segments record their provider failure; the never-attempted tail
    # records the ceiling skip. No segment was escalated to Gemini.
    assert result.segments[0].status is ReconstructionStatus.PROVIDER_UNAVAILABLE
    assert result.segments[1].status is ReconstructionStatus.PROVIDER_UNAVAILABLE
    assert result.segments[2].status is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
    counts = result.metadata["routing_counts"]
    assert counts.get("local_escalations_dropped", 0) >= 1


# 14. Mixed-language / code-switch evidence is preserved


def test_code_switch_evidence_preserved_and_signalled(sqlite_engine: object) -> None:
    segments = [_code_switch_segment(0), _segment(1)]
    source_id = _setup_transcript(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        _executor(
            session,
            ContextualReconstructor(CountingLocal(), gemini_provider=None),
        ).execute(session.get(SourceVideo, source_id), force=True)
        session.commit()
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )

        segment = transcript.segments[0]
        # Latin tokens and the mixed-language raw text survive unchanged.
        assert segment["contextual_reconstructed_text"] == "هنعمل deploy بعد الـ review"
        assert segment["code_switch_suspected"] is True
        assert segment["words"][1]["word"] == "deploy"
        # Arabic-only segment is not flagged.
        assert transcript.segments[1]["code_switch_suspected"] is False


# 15. Missing Gemini remains graceful for INDEX and targeted refinement


def test_missing_gemini_refinement_degrades_gracefully(sqlite_engine: object) -> None:
    segments = [_segment(i) for i in range(3)]
    source_id = _setup_transcript(sqlite_engine, segments)
    failing = FailingLocal()

    with Session(sqlite_engine) as session:
        outcome = refine_transcript_window(
            session,
            source_id,
            start_time=0.5,
            end_time=2.5,
            priority=CANDIDATE,
            reconstructor=_local_only_reconstructor(failing),
        )
        session.commit()

        # No Gemini is configured; a local failure degrades to safe unresolved.
        assert failing.calls >= 1
        assert all(
            segment.status is ReconstructionStatus.PROVIDER_UNAVAILABLE
            for segment in outcome.results
        )
        assert outcome.unresolved_indexes == (0, 1, 2)


# Review fixes: cancellation on fast paths


def test_reconstruct_index_fast_path_polls_cancellation() -> None:
    """The INDEX early return polls cancellation before succeeding."""

    reconstructor = ContextualReconstructor(None, is_cancelled=lambda: True)

    with pytest.raises(ReconstructionCancelled):
        reconstructor.reconstruct(
            [_segment(0)],
            language="ar",
            transcription_fingerprint="asr-fp",
            correction_version="egyptian-ar-v1",
        )


def test_reconstruct_providerless_fast_path_polls_cancellation() -> None:
    """The no-provider CANDIDATE/FINAL_CLIP early return polls cancellation."""

    reconstructor = ContextualReconstructor(
        None, gemini_provider=None, is_cancelled=lambda: True, priority=CANDIDATE
    )

    with pytest.raises(ReconstructionCancelled):
        reconstructor.reconstruct(
            [_segment(0)],
            language="ar",
            transcription_fingerprint="asr-fp",
            correction_version="egyptian-ar-v1",
        )


def _cancelled_runner_scenario(
    sqlite_engine: object,
    executor: ContextualReconstructionExecutor,
) -> None:
    segments = [_segment(i) for i in range(3)]
    source_id = _setup_transcript(sqlite_engine, segments)
    with Session(sqlite_engine) as worker:
        executor._job_cancelled = lambda: True  # type: ignore[method-assign]
        runner = PipelineRunner(worker, {PipelineStage.CONTEXTUAL_RECONSTRUCTION: executor})
        with pytest.raises(ReconstructionCancelled):
            runner.run(source_id, PipelineStage.CONTEXTUAL_RECONSTRUCTION)

        run = worker.scalar(
            select(PipelineRun)
            .where(PipelineRun.source_video_id == source_id)
            .order_by(PipelineRun.created_at.desc())
        )
        job = worker.scalar(
            select(ProcessingJob)
            .where(ProcessingJob.source_video_id == source_id)
            .order_by(ProcessingJob.created_at.desc())
        )
        assert run.status is PipelineRunStatus.CANCELLED
        assert job.status is JobStatus.CANCELLED
        # Lifecycle never advanced past INGEST (reconstruction was cancelled).
        source = worker.get(SourceVideo, source_id)
        assert source is not None
        assert source.lifecycle_state is PipelineStage.INGEST


def test_cancellation_during_index_fast_path_keeps_job_cancelled(
    sqlite_engine: object,
) -> None:
    with Session(sqlite_engine) as worker:
        executor = ContextualReconstructionExecutor(
            session=worker,
            reconstructor=ContextualReconstructor(None, gemini_provider=None),
        )
        _cancelled_runner_scenario(sqlite_engine, executor)


def test_cancellation_during_providerless_fast_path_keeps_job_cancelled(
    sqlite_engine: object,
) -> None:
    with Session(sqlite_engine) as worker:
        executor = ContextualReconstructionExecutor(
            session=worker,
            reconstructor=ContextualReconstructor(None, gemini_provider=None, priority=CANDIDATE),
        )
        _cancelled_runner_scenario(sqlite_engine, executor)


# Review fixes: explicit priority override drives identity/fingerprints


def _reconstruct_kwargs() -> dict[str, object]:
    return dict(
        language="ar",
        transcription_fingerprint="asr-fp",
        correction_version="egyptian-ar-v1",
    )


def test_explicit_priority_override_drives_runtime_identity_and_fingerprint() -> None:
    segments = [_segment(i) for i in range(3)]
    kwargs = _reconstruct_kwargs()
    base = ContextualReconstructor(None, gemini_provider=None)  # default INDEX

    index_result = base.reconstruct(segments, **kwargs)
    candidate_result = base.reconstruct(segments, **kwargs, priority=CANDIDATE)
    final_result = base.reconstruct(segments, **kwargs, priority=FINAL_CLIP)

    assert index_result.metadata["runtime_identity"]["priority"] == "INDEX"
    assert candidate_result.metadata["runtime_identity"]["priority"] == "CANDIDATE"
    assert final_result.metadata["runtime_identity"]["priority"] == "FINAL_CLIP"
    assert (
        len(
            {
                index_result.fingerprint,
                candidate_result.fingerprint,
                final_result.fingerprint,
            }
        )
        == 3
    )


def test_explicit_candidate_priority_never_collides_with_index_for_identical_segments() -> None:
    segments = [_segment(i) for i in range(3)]
    kwargs = _reconstruct_kwargs()

    index_result = ContextualReconstructor(CountingLocal(), gemini_provider=None).reconstruct(
        segments, **kwargs
    )
    candidate_result = ContextualReconstructor(CountingLocal(), gemini_provider=None).reconstruct(
        segments, **kwargs, priority=CANDIDATE
    )

    assert candidate_result.fingerprint != index_result.fingerprint
    assert candidate_result.metadata["runtime_identity"]["priority"] == "CANDIDATE"
    assert index_result.metadata["runtime_identity"]["priority"] == "INDEX"


def test_explicit_priority_override_drives_per_target_fingerprint_identity() -> None:
    segments = [_segment(0)]
    index_identity = ContextualReconstructor(None).runtime_identity()
    candidate_identity = ContextualReconstructor(None, priority=CANDIDATE).runtime_identity()

    index_target = reconstruction_target_fingerprint(
        provider_identity=index_identity,
        segments=segments,
        target_index=0,
        language="ar",
        transcription_fingerprint="asr-fp",
        correction_version="egyptian-ar-v1",
    )
    candidate_target = reconstruction_target_fingerprint(
        provider_identity=candidate_identity,
        segments=segments,
        target_index=0,
        language="ar",
        transcription_fingerprint="asr-fp",
        correction_version="egyptian-ar-v1",
    )

    assert index_target != candidate_target


# Review fixes: source-wide state stays truthful after targeted refinement


def test_targeted_refinement_keeps_source_wide_state_truthful(sqlite_engine: object) -> None:
    segments = [_segment(i) for i in range(5)]
    source_id = _setup_transcript(sqlite_engine, segments)

    with Session(sqlite_engine) as session:
        # Whole-source INDEX leaves every segment unresolved/deferred.
        _executor(
            session,
            ContextualReconstructor(None, gemini_provider=None),
        ).execute(session.get(SourceVideo, source_id), force=True)
        session.commit()
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )
        assert transcript is not None
        assert transcript.reconstruction_status is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
        assert transcript.reconstruction_metadata["index_deferred_segments"] == 5

        # Successful CANDIDATE refinement of a two-segment subset.
        outcome = refine_transcript_window(
            session,
            source_id,
            start_time=0.0,
            end_time=2.0,
            priority=CANDIDATE,
            reconstructor=_local_only_reconstructor(CountingLocal()),
        )
        session.commit()
        transcript = session.scalar(
            select(Transcript).where(Transcript.source_video_id == source_id)
        )

        assert outcome.accepted_indexes == (0, 1)
        # Source-wide summary must stay truthful over untouched unresolved segments.
        assert transcript.reconstruction_status is ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED
        assert transcript.reconstruction_confidence > 0.0
        assert transcript.reconstructed_segment_ratio == 2 / 5
        assert transcript.reconstruction_metadata["cache_eligible"] is False
        assert transcript.reconstruction_metadata["index_deferred_segments"] == 3
        # Untouched segments keep their deferred evidence.
        for index in (2, 3, 4):
            assert (
                transcript.segments[index]["reconstruction_status"]
                == ReconstructionStatus.LOW_CONFIDENCE_UNRESOLVED.value
            )
            assert transcript.segments[index]["escalation_reason"] == "index_priority_deferred"
        # Window-specific outcome metadata is separate from the source summary.
        assert outcome.metadata["window"] == {"start": 0.0, "end": 2.0}
        assert outcome.metadata["target_indexes"] == [0, 1]
        assert outcome.metadata["applied_in_window"] == 2
        assert "window" not in transcript.reconstruction_metadata
