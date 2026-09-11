"""Focused Stage 3 candidate-analysis tests.

Deterministic fixtures and fake providers only; no live Gemini or Qwen calls.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.candidates import executor as executor_module
from app.candidates import service as service_module
from app.candidates.classification import classify_content
from app.candidates.executor import CandidateAnalysisCancelled, CandidateAnalysisExecutor
from app.candidates.hooks import generate_hooks, validate_provider_hooks
from app.candidates.policy import DEFAULT_CONFIG, Stage3Config
from app.candidates.providers import (
    DeterministicSemanticProvider,
    ProviderErrorCategory,
    SemanticEvaluationRequest,
    SemanticEvaluationResult,
    SemanticProviderError,
    parse_semantic_entries,
)
from app.candidates.service import CandidateAnalysisService, derive_risks
from app.candidates.text import contains_cue, matching_text
from app.candidates.types import CandidateAnalysisOutcome, HookRecord, Proposal
from app.core.enums import (
    CandidateDisposition,
    ContentType,
    HookType,
    JobStatus,
    MediaOriginType,
    OriginalityRisk,
    PipelineRunStatus,
    PipelineStage,
    RefinementReason,
    RightsRisk,
    RightsStatus,
    SemanticProviderMode,
)
from app.db.base import Base
from app.models import (
    AudioAnalysis,
    CandidateAnalysis,
    ClipCandidate,
    PipelineRun,
    ProcessingJob,
    SourceQualityAssessment,
    SourceVideo,
    Transcript,
)
from app.pipeline.executor import StageCancelled
from app.pipeline.runner import PipelineRunner

_GOOD = [
    "في الحقيقة دي معلومة غريبة جدا ومش متوقع انها تحصل.",
    "معني الكلام ان الموضوع كان نتيجة مباشرة للخطأ الكبير.",
    "تخيل مرة كان في قصة حصلت وقلت لازم احكيها لكم.",
    "بس في الاخر النتيجة طلعت مفاجأة كبيرة جدا لكل الناس.",
    "التحليل الحقيقي بيقول ان السبب كان واضح من الاول.",
    "هههه دي نكتة مضحكة جدا جدا وكل الناس ضحكت.",
]
_WEAK = [
    "المهم يعني اممم طيب كلام فاضي.",
    "اه اه يعني عموما المهم.",
]


def _segment(
    text: str,
    start: float,
    end: float,
    *,
    probability: float = 0.9,
    with_words: bool = True,
    status: str | None = None,
    method: str | None = None,
    needs_refinement: bool | None = None,
    operator_text: str | None = None,
    code_switch: bool | None = None,
) -> dict[str, object]:
    segment: dict[str, object] = {
        "start": start,
        "end": end,
        "raw_text": text,
        "corrected_text": text,
        "final_text": text,
        "avg_logprob": -0.15,
    }
    if with_words:
        words = text.split()
        step = (end - start) / max(1, len(words))
        segment["words"] = [
            {
                "word": word,
                "probability": probability,
                "start": start + index * step,
                "end": start + (index + 1) * step,
            }
            for index, word in enumerate(words)
        ]
    if status is not None:
        segment["reconstruction_status"] = status
    if method is not None:
        segment["reconstruction_method"] = method
    if needs_refinement is not None:
        segment["needs_refinement"] = needs_refinement
    if operator_text is not None:
        segment["operator_text"] = operator_text
    if code_switch is not None:
        segment["code_switch_suspected"] = code_switch
    return segment


def _transcript_segments(
    sentences: Sequence[str],
    *,
    per_segment: float = 6.0,
    probability: float = 0.9,
    **kwargs: object,
) -> list[dict[str, object]]:
    segments = []
    start = 0.0
    for sentence in sentences:
        segments.append(
            _segment(
                sentence,
                start,
                start + per_segment,
                probability=probability,
                **kwargs,  # type: ignore[arg-type]
            )
        )
        start += per_segment
    return segments


@pytest.fixture  # type: ignore[untyped-decorator]
def session(tmp_path: Path) -> Session:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'stage3.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    yield factory()
    engine.dispose()


def _make_source(
    session: Session,
    segments: Sequence[Mapping[str, object]],
    *,
    duration: float | None = None,
    rights: RightsStatus = RightsStatus.OWNED,
    origin: MediaOriginType = MediaOriginType.YOUTUBE_CREATOR_VIDEO,
    provenance: Mapping[str, object] | None = None,
    dialect: str | None = "EGYPTIAN",
    language: str | None = "ar",
) -> SourceVideo:
    resolved_duration = (
        duration if duration is not None else float(segments[-1]["end"])  # type: ignore[arg-type]
    )
    source = SourceVideo(
        source_uri="/tmp/source.mp4",
        content_hash=f"hash-{uuid4()}",
        rights_status=rights,
        media_origin=origin,
        provenance_metadata=dict(provenance or {}),
        lifecycle_state=PipelineStage.READY_FOR_ANALYSIS,
    )
    session.add(source)
    session.flush()
    transcript = Transcript(
        source_video_id=source.id,
        language=language,
        whisper_model="large-v3-turbo",
        input_fingerprint="transcript-fp",
        transcription_revision=3,
        normalization_fingerprint="normalization-fp",
        dialect_profile=dialect,
        dialect_confidence=0.9 if dialect else 0.0,
        reconstruction_status="NOT_REQUIRED",
        reconstruction_version="stage2.7-v1",
        duration=resolved_duration,
        segments=list(segments),
        word_segments=[],
    )
    session.add(transcript)
    session.add(
        AudioAnalysis(
            source_video_id=source.id,
            audio_hash="audio-hash",
            input_fingerprint="audio-fp",
            silence_intervals=[{"start": 1.0, "end": 1.5}],
            features=[{"start": 0.0, "end": resolved_duration, "rms": 0.2}],
        )
    )
    session.add(
        SourceQualityAssessment(
            source_video_id=source.id,
            transcript_quality_score=0.8,
            input_fingerprint="quality-fp",
        )
    )
    session.commit()
    session.refresh(source)
    return source


def _config(**overrides: object) -> Stage3Config:
    base: dict[str, object] = {
        "min_window_seconds": 10.0,
        "preferred_window_min_seconds": 20.0,
        "preferred_window_max_seconds": 60.0,
        "max_window_seconds": 90.0,
    }
    base.update(overrides)
    return DEFAULT_CONFIG.with_overrides(**base)


_RETAINED = [
    CandidateDisposition.CANDIDATE,
    CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
]


def _executor(
    session: Session,
    *,
    provider: object = None,
    mode: SemanticProviderMode = SemanticProviderMode.DETERMINISTIC,
    config: Stage3Config | None = None,
) -> CandidateAnalysisExecutor:
    return CandidateAnalysisExecutor(
        session=session,
        config=config or _config(),
        provider=provider,  # type: ignore[arg-type]
        mode=mode,
    )


class FakeProvider:
    provider_name = "fake"
    model = "fake-model-1"

    def __init__(
        self,
        *,
        results: Mapping[str, SemanticEvaluationResult] | None = None,
        error: SemanticProviderError | None = None,
        on_call: object = None,
        synthesize: bool = False,
        partial: int = 0,
    ) -> None:
        self._results = dict(results or {})
        self._error = error
        self._synthesize = synthesize
        self._partial = partial
        self.calls: list[list[str]] = []
        self.released = False
        self._on_call = on_call
        self.rate_limited = False

    def evaluate(
        self, requests: Sequence[SemanticEvaluationRequest]
    ) -> dict[str, SemanticEvaluationResult]:
        self.calls.append([request.candidate_key for request in requests])
        if callable(self._on_call):
            self._on_call(len(self.calls))
        if self._error is not None:
            self.rate_limited = self._error.category is ProviderErrorCategory.RATE_LIMITED
            raise self._error
        if self._synthesize or self._partial:
            selected = list(requests)[self._partial :]
            return {request.candidate_key: _expected(request.candidate_key) for request in selected}
        return {
            request.candidate_key: self._results[request.candidate_key]
            for request in requests
            if request.candidate_key in self._results
        }

    def release(self) -> None:
        self.released = True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "fake", "model": self.model, "digest": "digest", "temperature": 0.0}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def usage_summary(self) -> dict[str, int]:
        return {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15}


def _expected(key: str) -> SemanticEvaluationResult:
    return SemanticEvaluationResult(
        candidate_key=key,
        primary_content_type=ContentType.STORY,
        secondary_content_types=(ContentType.EMOTIONAL,),
        score_adjustments={"moment_density_score": 0.1},
        idea_summary="فكرة",
        topic_summary="موضوع",
        hooks=(),
        confidence=0.8,
        explanation="ok",
    )


# ---------------------------------------------------------------------------
# discovery


def test_clean_index_transcript_generates_candidates(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    executor = _executor(session)
    executor.execute(source)
    candidates = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    assert candidates
    assert any(
        candidate.disposition
        in {CandidateDisposition.CANDIDATE, CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT}
        for candidate in candidates
    )


def test_boundaries_stay_within_source_duration(session: Session) -> None:
    segments = _transcript_segments(_GOOD)
    source = _make_source(session, segments, duration=1000.0)
    executor = _executor(session)
    executor.execute(source)
    candidates = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    for candidate in candidates:
        assert 0 <= candidate.start_time < candidate.end_time <= 1000.0


def test_overlapping_redundant_proposals_merge(session: Session) -> None:
    segments = _transcript_segments(["نفس الفكرة تتكرر هنا كلمة بكلمة."] * 6)
    source = _make_source(session, segments)
    executor = _executor(session, config=_config(min_window_seconds=5.0))
    executor.execute(source)
    proposals = [
        candidate
        for candidate in session.scalars(
            select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
        )
    ]
    assert len(proposals) < 6


def test_distinct_same_source_ideas_remain_separate(session: Session) -> None:
    sentences = [
        "دراسة جديدة بتقول ان النوم الكافي بيحسن الذاكرة بشكل كبير.",
        "البورصة ارتفعت النهاردة بنسبة كبيرة بسبب اخبار الشركات.",
        "في وصفة سهلة لكيك الشوكولاتة بمكونات بسيطة.",
        "اخبار الرياضة: الفريق كسب الماتش في الاخر.",
    ]
    segments = []
    start = 0.0
    for sentence in sentences:
        segments.append(_segment(sentence, start, start + 30.0))
        start += 30.0
    source = _make_source(session, segments)
    executor = _executor(
        session, config=_config(min_window_seconds=20.0, preferred_window_min_seconds=25.0)
    )
    executor.execute(source)
    candidates = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    assert len(candidates) >= 2


def test_zero_good_moment_fixture_yields_zero_accepted(session: Session) -> None:
    segments = _transcript_segments(_WEAK)
    source = _make_source(session, segments)
    executor = _executor(session)
    executor.execute(source)
    accepted = list(
        session.scalars(
            select(ClipCandidate).where(
                ClipCandidate.source_video_id == source.id,
                ClipCandidate.disposition.in_(_RETAINED),
            )
        )
    )
    assert accepted == []


def test_many_plausible_moments_pruned_before_hosted_calls(session: Session) -> None:
    sentences = []
    for index in range(25):
        sentences.append(f"معلومة رقم {index} غريبة ومفاجأة كبيرة جدا جدا.")
    segments = []
    start = 0.0
    for sentence in sentences:
        segments.append(_segment(sentence, start, start + 12.0))
        start += 12.0
    source = _make_source(session, segments, duration=start)
    provider = FakeProvider()
    executor = _executor(
        session,
        provider=provider,
        mode=SemanticProviderMode.ADAPTIVE,
        config=_config(max_proposals_per_source=6, max_provider_candidates=2),
    )
    executor.execute(source)
    assert provider.calls
    assert sum(len(call) for call in provider.calls) <= 2


# ---------------------------------------------------------------------------
# uncertainty


def test_low_transcript_confidence_does_not_auto_reject_strong_moment(session: Session) -> None:
    segments = _transcript_segments(_GOOD, probability=0.2)
    source = _make_source(session, segments)
    executor = _executor(session)
    executor.execute(source)
    accepted = list(
        session.scalars(
            select(ClipCandidate).where(
                ClipCandidate.source_video_id == source.id,
                ClipCandidate.disposition.in_(_RETAINED),
            )
        )
    )
    assert accepted
    assert any(
        candidate.disposition is CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT
        for candidate in accepted
    )


def test_strong_moment_with_low_confidence_needs_refinement(session: Session) -> None:
    segments = _transcript_segments(_GOOD, probability=0.1)
    source = _make_source(session, segments)
    _executor(session).execute(source)
    refinement = list(
        session.scalars(
            select(ClipCandidate).where(
                ClipCandidate.source_video_id == source.id,
                ClipCandidate.disposition == CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
            )
        )
    )
    assert refinement
    assert RefinementReason.LOW_CONFIDENCE_WORD_SPAN.value in refinement[0].refinement_reasons


def test_weak_moment_with_clean_transcript_stays_rejected(session: Session) -> None:
    segments = _transcript_segments(_WEAK, probability=0.99)
    source = _make_source(session, segments)
    _executor(session).execute(source)
    candidates = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    assert candidates
    assert all(
        candidate.transcript_confidence > 0.5
        and candidate.disposition is CandidateDisposition.DO_NOT_CLIP
        for candidate in candidates
    )


def test_unresolved_stage_27_text_is_valid_input(session: Session) -> None:
    segments = _transcript_segments(
        _GOOD,
        status="LOW_CONFIDENCE_UNRESOLVED",
        method="index_deferred",
        needs_refinement=True,
    )
    source = _make_source(session, segments)
    executor = _executor(session)
    result = executor.execute(source)
    assert result.output_fingerprint
    candidates = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    assert candidates


def test_dialect_metadata_passes_unchanged(session: Session) -> None:
    segments = _transcript_segments(_GOOD)
    source = _make_source(session, segments, dialect="MSA")
    _executor(session).execute(source)
    transcript = session.scalar(select(Transcript).where(Transcript.source_video_id == source.id))
    assert transcript is not None and transcript.dialect_profile is not None
    assert transcript.dialect_profile.value == "MSA"
    candidate = session.scalar(
        select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
    )
    assert candidate is not None and candidate.dialect_profile == "MSA"


def test_arabic_english_terms_survive_candidate_processing(session: Session) -> None:
    segments = _transcript_segments(
        ["بنستخدم machine learning و AI في تحليل البيانات بشكل كبير جدا."],
        per_segment=30.0,
    )
    source = _make_source(session, segments)
    _executor(session).execute(source)
    candidate = session.scalar(
        select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
    )
    assert candidate is not None
    assert "machine learning" in candidate.transcript_excerpt
    assert "AI" in candidate.transcript_excerpt


def test_candidate_local_code_switch_uncertainty_contributes(session: Session) -> None:
    segments = _transcript_segments(_GOOD, probability=0.2)
    segments[2]["code_switch_suspected"] = True
    segments[2]["code_switch_tokens"] = ["machine", "learning"]
    source = _make_source(session, segments)
    _executor(session).execute(source)
    refinement = list(
        session.scalars(
            select(ClipCandidate).where(
                ClipCandidate.source_video_id == source.id,
                ClipCandidate.disposition == CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT,
            )
        )
    )
    assert any(
        RefinementReason.CODE_SWITCH_UNCERTAINTY.value in candidate.refinement_reasons
        for candidate in refinement
    )


def test_omitted_english_recovery_is_never_attempted() -> None:
    source = inspect.getsource(service_module) + inspect.getsource(executor_module)
    assert "refine_transcript_window" not in source


# ---------------------------------------------------------------------------
# scoring / hooks


def test_weak_filler_with_perfect_transcript_confidence_stays_low(session: Session) -> None:
    segments = _transcript_segments(_WEAK, probability=1.0)
    source = _make_source(session, segments)
    _executor(session).execute(source)
    candidate = session.scalar(
        select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
    )
    assert candidate is not None
    assert candidate.clip_score < DEFAULT_CONFIG.retention_threshold
    assert candidate.transcript_confidence >= 0.9


def test_hook_output_is_bounded_typed_and_faithful(session: Session) -> None:
    proposal = Proposal(
        start_segment_index=0,
        end_segment_index=0,
        start_time=0.0,
        end_time=30.0,
        text="هل تعلم ان دي معلومة غريبة؟ النتيجة كانت مفاجأة كبيرة فعلا.",
        boundary_reason="sentence_end",
        segment_indexes=(0,),
    )
    hooks = generate_hooks(proposal, [])
    assert hooks and len(hooks) <= DEFAULT_CONFIG.max_hooks_per_candidate
    assert all(isinstance(hook, HookRecord) for hook in hooks)
    for hook in hooks:
        assert hook.type in HookType
        assert hook.text is None or hook.text in proposal.text
        assert 0.0 <= hook.strength <= 1.0


def test_fabricated_numbers_names_and_protected_tokens_rejected() -> None:
    proposal = Proposal(
        start_segment_index=0,
        end_segment_index=0,
        start_time=0.0,
        end_time=30.0,
        text="الرائد محمد قال ان الرقم 25 مهم جدا.",
        boundary_reason="sentence_end",
        segment_indexes=(0,),
    )
    raw = [
        {"type": "DIRECT_CLAIM", "text": "الرائد محمد قال ان الرقم 99 مهم", "faithfulness": 0.9},
        {"type": "DIRECT_CLAIM", "text": "قال الاسم Peter ان الرقم 25 مهم", "faithfulness": 0.9},
        {"type": "DIRECT_CLAIM", "text": "الرائد محمد قال ان الرقم 25 مهم", "faithfulness": 0.9},
    ]
    accepted = validate_provider_hooks(raw, proposal)
    texts = [hook.text for hook in accepted]
    assert "الرائد محمد قال ان الرقم 99 مهم" not in texts
    assert "قال الاسم Peter ان الرقم 25 مهم" not in texts
    assert "الرائد محمد قال ان الرقم 25 مهم" in texts


# ---------------------------------------------------------------------------
# duplicates / novelty


def test_same_source_duplicate_receives_duplication_risk(session: Session) -> None:
    repeated = "الفكرة دي بتتكرر هنا بنفس الكلمات بالظبط في كل مرة."
    segments = _transcript_segments([repeated, repeated, repeated], per_segment=30.0)
    source = _make_source(session, segments)
    _executor(session).execute(source)
    candidates = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    assert any(
        candidate.disposition is CandidateDisposition.DO_NOT_CLIP_RECENTLY_REDUNDANT
        or candidate.recent_semantic_similarity_risk > 0.0
        for candidate in candidates
    )


def test_cross_source_duplicate_receives_duplication_risk(session: Session) -> None:
    text = "نفس المعلومة بالحرف الواحد عن الذكاء الاصطناعي والتقنية."
    first = _make_source(session, _transcript_segments([text], per_segment=40.0))
    _executor(session).execute(first)
    second = SourceVideo(
        source_uri="/tmp/second.mp4",
        content_hash="hash-2",
        rights_status=RightsStatus.OWNED,
        media_origin=MediaOriginType.YOUTUBE_CREATOR_VIDEO,
        lifecycle_state=PipelineStage.READY_FOR_ANALYSIS,
    )
    session.add(second)
    session.flush()
    session.add(
        Transcript(
            source_video_id=second.id,
            language="ar",
            whisper_model="large-v3-turbo",
            input_fingerprint="transcript-fp",
            transcription_revision=3,
            normalization_fingerprint="normalization-fp",
            dialect_profile="UNKNOWN_ARABIC",
            duration=60.0,
            segments=_transcript_segments([text], per_segment=50.0),
            word_segments=[],
        )
    )
    session.add(
        AudioAnalysis(
            source_video_id=second.id,
            audio_hash="audio-hash-2",
            input_fingerprint="audio-fp",
            silence_intervals=[],
            features=[{"start": 0.0, "end": 60.0, "rms": 0.2}],
        )
    )
    session.commit()
    session.refresh(second)
    _executor(session).execute(second)
    candidate = session.scalar(
        select(ClipCandidate).where(ClipCandidate.source_video_id == second.id)
    )
    assert candidate is not None
    assert candidate.recent_semantic_similarity_risk > 0.5


# ---------------------------------------------------------------------------
# provenance


def test_unknown_and_third_party_provenance_does_not_block_analysis(session: Session) -> None:
    for index, rights in enumerate((RightsStatus.UNKNOWN, RightsStatus.THIRD_PARTY_UNKNOWN)):
        sentences = [f"معلومة رقم {index} {sentence}" for sentence in _GOOD]
        source = _make_source(
            session,
            _transcript_segments(sentences),
            rights=rights,
            origin=MediaOriginType.OTHER,
        )
        _executor(session).execute(source)
        retained = session.scalar(
            select(func.count())
            .select_from(ClipCandidate)
            .where(
                ClipCandidate.source_video_id == source.id,
                ClipCandidate.disposition.in_(_RETAINED),
            )
        )
        assert retained and retained > 0


def test_rights_risk_and_originality_risk_are_separate(session: Session) -> None:
    rights, originality = derive_risks(RightsStatus.OWNED, MediaOriginType.MOVIE_TV)
    assert rights is RightsRisk.LOW
    assert originality is OriginalityRisk.TRANSFORMATION_REQUIRED
    rights, originality = derive_risks(RightsStatus.UNKNOWN, MediaOriginType.OTHER)
    assert rights is RightsRisk.UNDETERMINED
    assert originality is OriginalityRisk.UNDETERMINED


def test_provenance_change_invalidates_stage_3_only(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    executor = _executor(session)
    before = executor.input_fingerprint(source)
    transcript = session.scalar(select(Transcript).where(Transcript.source_video_id == source.id))
    assert transcript is not None
    transcript_input = transcript.input_fingerprint
    normalization = transcript.normalization_fingerprint
    source.media_origin = MediaOriginType.PODCAST_INTERVIEW
    session.commit()
    after = executor.input_fingerprint(source)
    assert before != after
    assert transcript.input_fingerprint == transcript_input
    assert transcript.normalization_fingerprint == normalization


# ---------------------------------------------------------------------------
# provider behavior


def test_missing_gemini_key_makes_zero_network_calls(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    provider = FakeProvider()
    executor = _executor(session, provider=None, mode=SemanticProviderMode.ADAPTIVE)
    executor.execute(source)
    analysis = session.scalar(
        select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
    )
    assert analysis is not None
    assert analysis.metrics.get("provider_calls") == 0
    assert provider.calls == []
    assert analysis.cache_eligible is True


def test_provider_failure_degrades_safely(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    provider = FakeProvider(error=SemanticProviderError(ProviderErrorCategory.PROVIDER_ERROR))
    executor = _executor(session, provider=provider, mode=SemanticProviderMode.ADAPTIVE)
    executor.execute(source)
    analysis = session.scalar(
        select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
    )
    assert analysis is not None
    assert analysis.provider_status == "PROVIDER_DEGRADED"
    assert analysis.cache_eligible is False
    assert session.scalar(
        select(func.count())
        .select_from(ClipCandidate)
        .where(ClipCandidate.source_video_id == source.id)
    )


def test_malformed_model_output_cannot_corrupt_persistence() -> None:
    requests = [
        SemanticEvaluationRequest(candidate_key="a"),
        SemanticEvaluationRequest(candidate_key="b"),
    ]
    content = {
        "evaluations": [
            {"candidate_id": "a", "primary_content_type": "NOT_A_TYPE"},
            {"candidate_id": "b", "primary_content_type": ContentType.FUNNY.value},
            {"candidate_id": "ghost", "primary_content_type": ContentType.STORY.value},
        ]
    }
    parsed = parse_semantic_entries(content, requests)
    assert set(parsed) == {"b"}
    assert parsed["b"].primary_content_type is ContentType.FUNNY


def test_rate_limit_stops_later_hosted_calls(session: Session) -> None:
    sentences = [f"معلومة رقم {index} غريبة ومفاجأة كبيرة جدا." for index in range(6)]
    source = _make_source(session, _transcript_segments(sentences, per_segment=20.0))
    provider = FakeProvider(error=SemanticProviderError(ProviderErrorCategory.RATE_LIMITED))
    executor = _executor(
        session,
        provider=provider,
        mode=SemanticProviderMode.ADAPTIVE,
        config=_config(provider_candidates_per_request=1, max_provider_candidates=6),
    )
    executor.execute(source)
    assert len(provider.calls) == 1
    analysis = session.scalar(
        select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
    )
    assert analysis is not None
    assert analysis.provider_status == "RATE_LIMITED"
    assert analysis.metrics.get("provider_rate_limits") == 1


def test_provider_batches_obey_candidate_and_call_caps(session: Session) -> None:
    sentences = [f"معلومة رقم {index} غريبة ومفاجأة كبيرة جدا." for index in range(8)]
    source = _make_source(session, _transcript_segments(sentences, per_segment=20.0))
    provider = FakeProvider()
    executor = _executor(
        session,
        provider=provider,
        mode=SemanticProviderMode.ADAPTIVE,
        config=_config(
            max_provider_candidates=4,
            provider_candidates_per_request=2,
            max_provider_calls_per_source=2,
        ),
    )
    executor.execute(source)
    assert sum(len(call) for call in provider.calls) <= 4
    assert len(provider.calls) <= 2
    analysis = session.scalar(
        select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
    )
    assert analysis is not None
    assert int(analysis.metrics.get("provider_candidates_evaluated", 0) or 0) <= 4


def test_accepted_provider_evaluations_are_reused_on_retry(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    first = FakeProvider(synthesize=True)
    _executor(session, provider=first, mode=SemanticProviderMode.ADAPTIVE).execute(source)
    assert first.calls
    second = FakeProvider(error=SemanticProviderError(ProviderErrorCategory.PROVIDER_ERROR))
    _executor(session, provider=second, mode=SemanticProviderMode.ADAPTIVE).execute(
        source, force=True
    )
    assert second.calls == []
    analysis = session.scalar(
        select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
    )
    assert analysis is not None
    assert analysis.metrics.get("provider_reused", 0) >= 1


def test_stable_provider_identity_excludes_secrets_and_availability() -> None:
    from app.candidates.gemini import GeminiSemanticProvider

    first = GeminiSemanticProvider(api_key="secret-one", model="gemini-3.8-flash")
    second = GeminiSemanticProvider(api_key="secret-two", model="gemini-3.8-flash")
    identity = first.runtime_identity()
    assert identity == second.runtime_identity()
    assert "secret-one" not in str(identity)
    assert "secret-two" not in str(identity)
    first.release()
    second.release()


def test_qwen_disabled_by_default_and_local_only_requires_enablement() -> None:
    from app.core.settings import Settings

    settings = Settings(_env_file=None)
    assert settings.local_qwen_enabled is False
    assert settings.reconstruction_provider_instance() is None
    assert settings.candidate_semantic_provider() is None
    local_only = Settings(
        _env_file=None,
        candidate_semantic_mode="local_only",
        local_qwen_enabled=False,
    )
    assert local_only.candidate_semantic_provider() is None


def test_explicit_local_only_uses_bounded_local_provider_path(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    provider = FakeProvider()
    provider.provider_name = "ollama"
    executor = _executor(
        session,
        provider=provider,
        mode=SemanticProviderMode.LOCAL_ONLY,
        config=_config(max_provider_candidates=2, provider_candidates_per_request=1),
    )
    executor.execute(source)
    assert provider.calls
    assert sum(len(call) for call in provider.calls) <= 2


# ---------------------------------------------------------------------------
# fingerprints / idempotency


def test_reruns_are_idempotent(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    executor = _executor(session)
    executor.execute(source)
    first_count = session.scalar(
        select(func.count())
        .select_from(ClipCandidate)
        .where(ClipCandidate.source_video_id == source.id)
    )
    first_ids = {
        row.candidate_key: row.id
        for row in session.scalars(
            select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
        )
    }
    executor.execute(source, force=True)
    second_count = session.scalar(
        select(func.count())
        .select_from(ClipCandidate)
        .where(ClipCandidate.source_video_id == source.id)
    )
    second_ids = {
        row.candidate_key: row.id
        for row in session.scalars(
            select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
        )
    }
    assert first_count == second_count
    assert first_ids == second_ids


def test_manual_override_is_reflected_in_candidate_text_and_fingerprint(session: Session) -> None:
    segments = _transcript_segments(_GOOD)
    segments[0]["operator_text"] = "كلام محرر يدويا بواسطة المشغل."
    segments[0]["final_text"] = segments[0]["operator_text"]
    source = _make_source(session, segments)
    executor = _executor(session)
    before = executor.input_fingerprint(source)
    executor.execute(source)
    transcript = session.scalar(select(Transcript).where(Transcript.source_video_id == source.id))
    assert transcript is not None
    replacement = [dict(segment) for segment in transcript.segments]
    replacement[0]["operator_text"] = "تعديل يدوي تاني مختلف تماما."
    transcript.segments = replacement
    session.commit()
    after = executor.input_fingerprint(source)
    assert before != after
    candidate = session.scalar(
        select(ClipCandidate).where(
            ClipCandidate.source_video_id == source.id,
            ClipCandidate.start_segment_index == 0,
        )
    )
    if candidate is not None:
        assert "كلام محرر يدويا" in candidate.transcript_excerpt


def test_stale_candidates_marked_only_after_successful_replacement(session: Session) -> None:
    segments = _transcript_segments(_GOOD)
    source = _make_source(session, segments)
    executor = _executor(session)
    executor.execute(source)
    original = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    assert original

    transcript = session.scalar(select(Transcript).where(Transcript.source_video_id == source.id))
    assert transcript is not None
    transcript.segments = _transcript_segments(_WEAK)
    transcript.duration = transcript.segments[-1]["end"]
    session.commit()
    executor.execute(source, force=True)
    stale = list(
        session.scalars(
            select(ClipCandidate).where(
                ClipCandidate.source_video_id == source.id,
                ClipCandidate.candidate_key.in_([row.candidate_key for row in original]),
                ClipCandidate.is_current.is_(False),
            )
        )
    )
    assert stale
    for row in original:
        session.refresh(row)
    assert all(row.is_current is False for row in stale)


def test_audio_analysis_and_quality_fields_preserved(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    quality_before = session.scalar(
        select(SourceQualityAssessment).where(SourceQualityAssessment.source_video_id == source.id)
    )
    assert quality_before is not None
    quality_snapshot = quality_before.transcript_quality_score
    audio_before = session.scalar(
        select(AudioAnalysis).where(AudioAnalysis.source_video_id == source.id)
    )
    assert audio_before is not None
    audio_snapshot = audio_before.input_fingerprint
    transcript_before = session.scalar(
        select(Transcript).where(Transcript.source_video_id == source.id)
    )
    assert transcript_before is not None
    transcript_snapshot = (transcript_before.raw_text, tuple(transcript_before.segments[0].keys()))
    _executor(session).execute(source)
    session.expire_all()
    quality_after = session.scalar(
        select(SourceQualityAssessment).where(SourceQualityAssessment.source_video_id == source.id)
    )
    audio_after = session.scalar(
        select(AudioAnalysis).where(AudioAnalysis.source_video_id == source.id)
    )
    transcript_after = session.scalar(
        select(Transcript).where(Transcript.source_video_id == source.id)
    )
    assert quality_after is not None and quality_after.transcript_quality_score == quality_snapshot
    assert audio_after is not None and audio_after.input_fingerprint == audio_snapshot
    assert transcript_after is not None
    assert transcript_after.raw_text == transcript_snapshot[0]
    assert tuple(transcript_after.segments[0].keys()) == transcript_snapshot[1]


def test_index_stage_makes_zero_reconstruction_providers(session: Session) -> None:
    from app.core.settings import Settings

    settings = Settings(_env_file=None)
    assert settings.reconstruction_provider_instance() is None
    assert settings.gemini_provider_instance() is None


# ---------------------------------------------------------------------------
# cancellation


def test_cancellation_before_start_raises_and_leaves_source_ready(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    job = ProcessingJob(
        source_video_id=source.id,
        kind="CANDIDATE_ANALYSIS",
        status="CANCELLED",
    )
    session.add(job)
    session.commit()
    executor = _executor(session)
    executor.set_active_job(job.id)

    with pytest.raises(CandidateAnalysisCancelled):
        executor.execute(source)
    session.refresh(source)
    assert source.lifecycle_state in {
        PipelineStage.READY_FOR_ANALYSIS,
        PipelineStage.CANDIDATE_ANALYSIS,
    }
    assert (
        session.scalar(
            select(func.count())
            .select_from(ClipCandidate)
            .where(ClipCandidate.source_video_id == source.id)
        )
        == 0
    )


def test_cancellation_callback_stops_service(session: Session) -> None:
    with pytest.raises(StageCancelled):
        CandidateAnalysisService(
            config=_config(), provider=DeterministicSemanticProvider(), is_cancelled=lambda: True
        ).analyze(
            source_id="s",
            segments=_transcript_segments(_GOOD),
            duration=36.0,
            language="ar",
            dialect_profile="EGYPTIAN",
            dialect_confidence=0.9,
        )


def test_runner_cancelled_job_does_not_advance_source(session: Session) -> None:

    source = _make_source(session, _transcript_segments(_GOOD))
    job = ProcessingJob(
        source_video_id=source.id,
        kind="CANDIDATE_ANALYSIS",
        status=JobStatus.CANCELLED,
    )
    session.add(job)
    session.commit()
    executor = _executor(session)
    runner = PipelineRunner(session, {PipelineStage.CANDIDATE_ANALYSIS: executor})
    result = runner.run(source.id, PipelineStage.CANDIDATE_ANALYSIS, job_id=job.id)
    run = session.get(PipelineRun, result.run_id)
    assert run is not None and run.status is PipelineRunStatus.CANCELLED
    session.refresh(source)
    assert source.lifecycle_state is PipelineStage.READY_FOR_ANALYSIS


# finalization remediation


class ZeroAdjustmentProvider:
    """Peer of FakeProvider returning accepted evaluations with no score change."""

    provider_name = "fake"
    model = "fake-model-1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.released = False

    def evaluate(
        self, requests: Sequence[SemanticEvaluationRequest]
    ) -> dict[str, SemanticEvaluationResult]:
        self.calls.append([request.candidate_key for request in requests])
        return {
            request.candidate_key: SemanticEvaluationResult(
                candidate_key=request.candidate_key,
                primary_content_type=None,
                secondary_content_types=(),
                score_adjustments={},
                idea_summary="",
                topic_summary="",
                hooks=(),
                confidence=0.6,
                explanation="",
            )
            for request in requests
        }

    def release(self) -> None:
        self.released = True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "fake", "model": self.model, "digest": "digest", "temperature": 0.0}

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def usage_summary(self) -> dict[str, int]:
        return {}


def _analyze(
    segments: Sequence[Mapping[str, object]], **overrides: object
) -> CandidateAnalysisOutcome:
    return CandidateAnalysisService(config=_config(**overrides)).analyze(
        source_id="source-1",
        segments=segments,
        duration=float(segments[-1]["end"]),  # type: ignore[arg-type]
        language="ar",
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
    )


def test_index_deferred_candidate_keeps_content_score_and_refinement_status() -> None:
    clean = _analyze(_transcript_segments(_GOOD))
    deferred = _analyze(
        _transcript_segments(
            _GOOD,
            status="LOW_CONFIDENCE_UNRESOLVED",
            method="index_deferred",
            needs_refinement=True,
        )
    )
    clean_scores = {draft.candidate_key: draft.scores.clip_score for draft in clean.candidates}
    assert clean_scores and deferred.candidates
    for draft in deferred.candidates:
        assert draft.scores.clip_score == pytest.approx(clean_scores[draft.candidate_key])
    assert any(
        draft.disposition is CandidateDisposition.CANDIDATE_NEEDS_REFINEMENT
        for draft in deferred.candidates
    )
    assert any(
        RefinementReason.UNRESOLVED_INDEX_TEXT in draft.refinement_reasons
        for draft in deferred.candidates
    )


def test_low_transcript_confidence_preserves_content_score_and_retention() -> None:
    clean = _analyze(_transcript_segments(_GOOD, probability=0.97))
    low = _analyze(_transcript_segments(_GOOD, probability=0.05))
    clean_scores = {draft.candidate_key: draft.scores.clip_score for draft in clean.candidates}
    assert low.candidates
    for draft in low.candidates:
        assert draft.scores.clip_score == pytest.approx(clean_scores[draft.candidate_key])
    assert any(draft.disposition in _RETAINED for draft in low.candidates)
    assert any(draft.scores.transcript_confidence < 0.55 for draft in low.candidates)


def test_provider_zero_adjustments_preserve_deterministic_clip_score() -> None:
    segments = _transcript_segments(_GOOD)
    deterministic = _analyze(segments)
    provider = ZeroAdjustmentProvider()
    enriched = CandidateAnalysisService(
        config=_config(), provider=provider, mode=SemanticProviderMode.ADAPTIVE
    ).analyze(
        source_id="source-1",
        segments=segments,
        duration=36.0,
        language="ar",
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
    )
    assert provider.calls
    baseline = {draft.candidate_key: draft.scores.clip_score for draft in deterministic.candidates}
    for draft in enriched.candidates:
        assert draft.scores.clip_score == pytest.approx(baseline[draft.candidate_key])


def test_partial_provider_output_is_retryable_and_reuses_accepted(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    first = FakeProvider(partial=1)
    _executor(session, provider=first, mode=SemanticProviderMode.ADAPTIVE).execute(source)
    analysis = session.scalar(
        select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
    )
    assert analysis is not None
    assert analysis.cache_eligible is False
    assert analysis.provider_status == "PROVIDER_PARTIAL"
    assert int(analysis.metrics.get("provider_malformed_items", 0) or 0) >= 1

    rows = {
        row.candidate_key: row
        for row in session.scalars(
            select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
        )
    }
    accepted = {
        key for key, row in rows.items() if (row.provider_evidence or {}).get("accepted") is True
    }
    requested_first = {key for call in first.calls for key in call}
    missing = requested_first - accepted
    assert accepted and missing

    second = FakeProvider(synthesize=True)
    _executor(session, provider=second, mode=SemanticProviderMode.ADAPTIVE).execute(
        source, force=True
    )
    requested_second = {key for call in second.calls for key in call}
    assert requested_second == missing
    assert not (requested_second & accepted)
    session.refresh(analysis)
    assert analysis.cache_eligible is True
    assert analysis.provider_status == "PROVIDER_EVALUATED"


def test_no_provider_call_for_clearly_redundant_candidates(session: Session) -> None:
    repeated = "الفكرة دي بتتكرر هنا بنفس الكلمات بالظبط في كل مرة."
    segments = _transcript_segments([repeated] * 6, per_segment=30.0)
    source = _make_source(session, segments)
    provider = FakeProvider(synthesize=True)
    _executor(session, provider=provider, mode=SemanticProviderMode.ADAPTIVE).execute(source)
    rows = {
        row.candidate_key: row
        for row in session.scalars(
            select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
        )
    }
    assert any(
        row.disposition is CandidateDisposition.DO_NOT_CLIP_RECENTLY_REDUNDANT
        for row in rows.values()
    )
    requested = {key for call in provider.calls for key in call}
    assert requested
    for key in requested:
        assert rows[key].disposition is not CandidateDisposition.DO_NOT_CLIP_RECENTLY_REDUNDANT


def test_provider_summaries_can_drive_post_enrichment_novelty(session: Session) -> None:
    sentences = [
        "دراسة جديدة بتقول ان النوم الكافي بيحسن الذاكرة، دي معلومة غريبة ومفاجأة كبيرة.",
        "البورصة ارتفعت النهاردة بسبب اخبار الشركات، دي معلومة غريبة ومفاجأة كبيرة.",
    ]
    segments = _transcript_segments(sentences, per_segment=55.0)
    deterministic = _analyze(segments)
    assert not any(
        draft.disposition is CandidateDisposition.DO_NOT_CLIP_RECENTLY_REDUNDANT
        for draft in deterministic.candidates
    )

    source = _make_source(session, segments)
    provider = FakeProvider(synthesize=True)  # identical idea/topic summaries
    _executor(session, provider=provider, mode=SemanticProviderMode.ADAPTIVE).execute(source)
    assert provider.calls
    rows = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    assert any(
        row.disposition is CandidateDisposition.DO_NOT_CLIP_RECENTLY_REDUNDANT for row in rows
    )


def test_oversized_segment_is_split_into_bounded_stable_candidates(session: Session) -> None:
    phrase = "معلومة غريبة ومفاجأة كبيرة، وفي الآخر النتيجة طلعت مفاجأة."
    long_text = " ".join([phrase] * 12)
    source = _make_source(
        session,
        _transcript_segments([long_text], per_segment=300.0),
        duration=300.0,
    )
    executor = _executor(session, config=_config(max_window_seconds=90.0))
    executor.execute(source)
    rows = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    assert len(rows) >= 2
    assert len({row.candidate_key for row in rows}) == len(rows)
    for row in rows:
        assert row.end_time - row.start_time <= 90.0 + 1e-6
        assert 0.0 <= row.start_time < row.end_time <= 300.0
    first_ids = {row.candidate_key: row.id for row in rows}
    executor.execute(source, force=True)
    rerun = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    assert {row.candidate_key for row in rerun} == set(first_ids)
    assert {row.candidate_key: row.id for row in rerun} == first_ids


def test_candidate_bounds_never_exceed_source_duration(session: Session) -> None:
    phrase = "معلومة غريبة ومفاجأة كبيرة جدا، والنتيجة طلعت مفاجأة."
    source = _make_source(
        session,
        _transcript_segments([" ".join([phrase] * 10)], per_segment=240.0),
        duration=240.0,
    )
    _executor(session, config=_config(max_window_seconds=60.0)).execute(source)
    rows = list(
        session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
    )
    assert rows
    for row in rows:
        assert row.start_time >= 0.0
        assert row.end_time <= 240.0


def test_new_output_affecting_config_and_mode_change_input_fingerprint(session: Session) -> None:
    source = _make_source(session, _transcript_segments(_GOOD))
    baseline = _executor(session).input_fingerprint(source)
    for override in (
        {"novelty_corpus_limit": 123},
        {"provider_context_characters": 99},
        {"provenance_max_keys": 5},
        {"provenance_max_value_length": 10},
        {"provider_max_input_characters": 111},
    ):
        candidate = _executor(session, config=_config(**override)).input_fingerprint(source)
        assert candidate != baseline, override
    adaptive = _executor(session, mode=SemanticProviderMode.ADAPTIVE).input_fingerprint(source)
    assert adaptive != baseline


def test_output_fingerprint_tracks_candidate_content_changes() -> None:
    first = _analyze(_transcript_segments(_GOOD))
    second = _analyze(_transcript_segments(_GOOD))
    assert first.output_fingerprint == second.output_fingerprint
    provider = FakeProvider(synthesize=True)
    enriched = CandidateAnalysisService(
        config=_config(), provider=provider, mode=SemanticProviderMode.ADAPTIVE
    ).analyze(
        source_id="source-1",
        segments=_transcript_segments(_GOOD),
        duration=36.0,
        language="ar",
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
    )
    assert enriched.output_fingerprint != first.output_fingerprint


def test_english_capitalization_is_equivalent_for_deterministic_analysis() -> None:
    upper = classify_content("How To Learn Python Fast", config=_config())
    lower = classify_content("how to learn python fast", config=_config())
    assert upper.primary is lower.primary
    assert upper.primary is ContentType.EDUCATIONAL
    assert upper.scores == lower.scores


def test_arabic_diacritics_and_alif_variants_match_without_changing_stored_text(
    session: Session,
) -> None:
    decorated = "الفَرْق بَيــن الصِيام والصَلاة واضح جدا."
    plain = "الفرق بين الصيام والصلاة واضح جدا."
    assert matching_text(decorated) == matching_text(plain)
    assert contains_cue(matching_text(decorated), "الفرق بين")
    classified = classify_content(decorated, config=_config())
    assert classified.primary is ContentType.EDUCATIONAL

    source = _make_source(session, _transcript_segments([decorated], per_segment=40.0))
    _executor(session).execute(source)
    candidate = session.scalar(
        select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
    )
    assert candidate is not None
    assert decorated in candidate.transcript_excerpt


_DIALECT_FIXTURES = {
    "egyptian": "أنا مش مصدق إزاي ده حصل، المعلومة غريبة ومفاجأة، وفي الآخر طلعت مفاجأة كبيرة.",
    "gulf_saudi": "والله المعلومة هذي غريبة ومفاجأة، وفي الآخر طلعت النتيجة مفاجأة كبيرة.",
    "levantine": "صراحة صار معي شي غريب ومفاجأة، وفي الآخر طلع الحل مفاجأة.",
    "msa": "الحقيقة أن هذه المعلومة غريبة ومفاجأة، وفي الآخر كانت النتيجة مفاجأة كبيرة.",
    "english": (
        "The truth is this surprising fact is unbelievable, and in the end the result "
        "turned out to be a huge surprise."
    ),
}


def test_dialect_fixtures_reach_bounded_shortlist_with_no_provider_calls(
    session: Session,
) -> None:
    for name, text in _DIALECT_FIXTURES.items():
        source = _make_source(session, _transcript_segments([text], per_segment=55.0))
        executor = _executor(session)
        executor.execute(source)
        rows = list(
            session.scalars(select(ClipCandidate).where(ClipCandidate.source_video_id == source.id))
        )
        analysis = session.scalar(
            select(CandidateAnalysis).where(CandidateAnalysis.source_video_id == source.id)
        )
        assert any(row.disposition in _RETAINED for row in rows), name
        assert analysis is not None
        assert analysis.metrics.get("provider_calls") == 0, name
        assert analysis.semantic_provider_mode is SemanticProviderMode.DETERMINISTIC, name


def test_code_switch_tokens_preserved_in_excerpt_evidence_and_hooks(session: Session) -> None:
    text = "بنستخدم machine learning و AI في التحليل، والنتيجة كانت مفاجأة كبيرة."
    segments = _transcript_segments([text], per_segment=40.0)
    segments[0]["code_switch_suspected"] = True
    segments[0]["code_switch_tokens"] = ["machine", "learning", "AI"]
    source = _make_source(session, segments)
    _executor(session).execute(source)
    candidate = session.scalar(
        select(ClipCandidate).where(ClipCandidate.source_video_id == source.id)
    )
    assert candidate is not None
    assert "machine learning" in candidate.transcript_excerpt
    assert "AI" in candidate.transcript_excerpt
    assert candidate.code_switch_suspected is True
    tokens = {
        str(token).casefold() for token in candidate.evidence_snapshot.get("code_switch_tokens", [])
    }
    assert {"machine", "learning", "ai"} <= tokens
    for hook in candidate.hooks:
        if hook.get("text"):
            assert hook["text"] in candidate.transcript_excerpt
