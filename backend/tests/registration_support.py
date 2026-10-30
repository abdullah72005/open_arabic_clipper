"""Shared deterministic fixtures for operator candidate registration tests."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from app.candidates.types import (
    CandidateAnalysisOutcome,
    CandidateDraft,
    CandidateScores,
    ContentClassification,
    Proposal,
)
from app.core.enums import (
    CandidateDisposition,
    ContentType,
    MediaOriginType,
    PipelineStage,
    RightsStatus,
    SemanticProviderMode,
)
from app.models import (
    AudioAnalysis,
    AudioArtifact,
    CandidateAnalysis,
    ClipCandidate,
    SourceQualityAssessment,
    SourceVideo,
    Transcript,
)

SOURCE_ID = UUID("cafc529e-b0d8-401b-b439-2721a57de73c")
EXPECTED_KEY = "9ebcd6855ab02d1017b5f7e7e45ae1948a2d4623363339c5bce4fcaf3f96fd3e"
SELECTED_START = 384.280
SELECTED_END = 408.660
SELECTED_INDEXES = list(range(230, 246))
OPERATOR_REASON = "Operator-selected complete richer-wife exchange"

_SELECTED_TEXTS = [
    "طيب أنا أسألك سؤال",
    "لو حبيت واحدة",
    "واتجوزته",
    "وهي طلعت أغنى منك",
    "هل في الحالة دي أنت هتسيبها؟",
    "مش هحط نفسي في الكيس دي",
    "يعني هقولك على حاجة",
    "خلينا متفقين إن الموضوع بيبقى عيلة كاملة",
    "مش مجرد اتنين",
    "فالعيلة بقى هتبص نظرة أنا مش هحبها",
    "فهتسيبها يعني",
    "لا مش هسيبها",
    "إنت كلامك بيقول لك هتسيبها",
    "اللي هو أنت هتيجي بشونطة هدومك",
    "مش هتيجي بحاجة",
    "حمش",
]


def build_segments(
    *,
    count: int = 260,
    with_index: bool = False,
) -> list[dict[str, object]]:
    step = (SELECTED_END - SELECTED_START) / len(SELECTED_INDEXES)
    segments: list[dict[str, object]] = []
    for index in range(count):
        if index in range(SELECTED_INDEXES[0], SELECTED_INDEXES[-1] + 1):
            offset = index - SELECTED_INDEXES[0]
            start = round(SELECTED_START + offset * step, 6)
            end = round(SELECTED_START + (offset + 1) * step, 6)
            text = _SELECTED_TEXTS[offset]
        else:
            start, end = float(index), float(index + 1)
            text = f"حشو رقم {index}"
        words = text.split()
        word_step = (end - start) / max(1, len(words))
        segment: dict[str, object] = {
            "start": start,
            "end": end,
            "raw_text": text,
            "corrected_text": text,
            "final_text": text,
            "normalized_text": text,
            "reconstruction_status": "UNCHANGED_HIGH_CONFIDENCE",
            "avg_logprob": -0.1,
            "no_speech_prob": 0.000001,
            "words": [
                {
                    "word": word,
                    "start": round(start + position * word_step, 6),
                    "end": round(start + (position + 1) * word_step, 6),
                    "probability": 0.95,
                }
                for position, word in enumerate(words)
            ],
        }
        if with_index:
            segment["index"] = index
        segments.append(segment)
    return segments


def install_source(
    session: Session,
    *,
    segments: Sequence[dict[str, object]] | None = None,
    duration: float = 851.8826875,
    lifecycle: PipelineStage = PipelineStage.READY_FOR_REFINEMENT,
) -> tuple[SourceVideo, Transcript]:
    resolved_segments = list(segments or build_segments())
    source = SourceVideo(
        id=SOURCE_ID,
        source_uri="/managed/source.webm",
        content_hash="operator-source-content-hash",
        rights_status=RightsStatus.OWNED,
        media_origin=MediaOriginType.OTHER,
        provenance_metadata={"operator_fixture": True},
        lifecycle_state=lifecycle,
    )
    session.add(source)
    session.flush()
    transcript = Transcript(
        source_video_id=source.id,
        language="ar",
        whisper_model="large-v3-turbo",
        input_fingerprint="operator-transcript-input",
        transcription_revision=4,
        normalization_fingerprint="operator-normalization-v1",
        dialect_profile="EGYPTIAN",
        dialect_confidence=0.9,
        reconstruction_status="NOT_REQUIRED",
        reconstruction_version="stage2.7-v1",
        duration=duration,
        segments=resolved_segments,
        word_segments=[],
    )
    session.add(transcript)
    session.add(
        AudioArtifact(
            source_video_id=source.id,
            output_path="/managed/audio.wav",
            content_hash="operator-audio-content-hash",
            source_content_hash="operator-audio-source-content-hash",
            sample_rate=16000,
            duration=duration,
        )
    )
    session.add(
        AudioAnalysis(
            source_video_id=source.id,
            audio_hash="operator-audio-hash",
            input_fingerprint="operator-audio-input",
            silence_intervals=[],
            features=[{"start": 0.0, "end": duration, "rms": 0.2}],
        )
    )
    session.add(
        SourceQualityAssessment(
            source_video_id=source.id,
            transcript_quality_score=0.8,
            input_fingerprint="operator-quality-input",
        )
    )
    session.commit()
    session.refresh(source)
    session.refresh(transcript)
    return source, transcript


def install_analysis(
    session: Session,
    source_id: UUID,
    *,
    input_fingerprint: str = "operator-analysis-input",
) -> CandidateAnalysis:
    analysis = CandidateAnalysis(
        source_video_id=source_id,
        input_fingerprint=input_fingerprint,
        output_fingerprint="operator-analysis-output",
        policy_version="stage3-v1",
        scoring_version="stage3-v1",
        semantic_provider_mode=SemanticProviderMode.DETERMINISTIC,
        provider_identity={},
        provider_status="DETERMINISTIC",
        cache_eligible=True,
        metrics={},
    )
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    return analysis


def make_candidate(
    session: Session,
    *,
    source_id: UUID,
    candidate_key: str,
    start_time: float,
    end_time: float,
    start_segment_index: int,
    end_segment_index: int,
    is_current: bool = True,
    analysis_id: UUID | None = None,
    transcript_excerpt: str = "",
    provenance: dict[str, object] | None = None,
    disposition: CandidateDisposition = CandidateDisposition.CANDIDATE,
    analysis_fingerprint: str = "automatic-analysis-fingerprint",
) -> ClipCandidate:
    row = ClipCandidate(
        source_video_id=source_id,
        candidate_analysis_id=analysis_id,
        candidate_key=candidate_key,
        is_current=is_current,
        disposition=disposition,
        start_time=start_time,
        end_time=end_time,
        start_segment_index=start_segment_index,
        end_segment_index=end_segment_index,
        segment_indexes=list(range(start_segment_index, end_segment_index + 1)),
        transcript_excerpt=transcript_excerpt,
        evidence_snapshot={
            "start_segment_index": start_segment_index,
            "end_segment_index": end_segment_index,
            "segment_count": end_segment_index - start_segment_index + 1,
        },
        primary_content_type=ContentType.OTHER,
        refinement_reasons=[],
        refinement_evidence={},
        provenance_snapshot=dict(provenance or {}),
        hooks=[],
        idea_summary=transcript_excerpt,
        topic_summary=transcript_excerpt,
        provider_input_fingerprint="",
        provider_evidence={},
        analysis_fingerprint=analysis_fingerprint,
        policy_version="stage3-v1",
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def make_draft(
    *,
    candidate_key: str,
    start_segment_index: int,
    end_segment_index: int,
    start_time: float,
    end_time: float,
    text: str,
) -> CandidateDraft:
    proposal = Proposal(
        start_segment_index=start_segment_index,
        end_segment_index=end_segment_index,
        start_time=start_time,
        end_time=end_time,
        text=text,
        boundary_reason="automatic_test",
        segment_indexes=tuple(range(start_segment_index, end_segment_index + 1)),
        span_start=start_segment_index,
        span_end=end_segment_index,
    )
    return CandidateDraft(
        candidate_key=candidate_key,
        proposal=proposal,
        content=ContentClassification(
            primary=ContentType.OTHER,
            secondary=(),
            scores={},
            reasons=(),
        ),
        scores=CandidateScores(
            clip_score=0.5,
            short_form_score=0.5,
            moment_density_score=0.5,
            boredom_risk_score=0.1,
            ending_quality_score=0.5,
            loopability_score=0.5,
            word_density=1.0,
            engagement_confidence=1.0,
            transcript_confidence=0.9,
            audio_confidence=0.9,
            boundary_confidence=0.8,
            uncertainty_severity=0.1,
            idea_novelty_score=0.0,
            topic_novelty_score=0.0,
            recent_semantic_similarity_risk=0.0,
        ),
        disposition=CandidateDisposition.CANDIDATE,
        transcript_excerpt=text,
        evidence_snapshot={
            "start_segment_index": start_segment_index,
            "end_segment_index": end_segment_index,
            "segment_count": end_segment_index - start_segment_index + 1,
            "span_start": start_segment_index,
            "span_end": end_segment_index,
        },
        provenance_snapshot={"automatic_test": True},
    )


def make_outcome(*drafts: CandidateDraft) -> CandidateAnalysisOutcome:
    return CandidateAnalysisOutcome(
        input_fingerprint="stage3-input-fingerprint",
        output_fingerprint="stage3-output-fingerprint",
        provider_status="DETERMINISTIC",
        semantic_provider_mode="deterministic",
        cache_eligible=True,
        metrics={"automatic_metric": 1},
        candidates=tuple(drafts),
    )
