"""Stage 6 V4 delivery/readability regressions (offline, no providers).

These pin the failed-acceptance card defects: a tiny bottom-inset card, silent
truncation, missing layout evidence, and an unqualified pacing pass for a
dominant silent ending.
"""

from __future__ import annotations

from fractions import Fraction

from app.materialization.acceptance import AssessmentContext, assess_materialization
from app.materialization.overlays import build_authored_ass
from app.materialization.policy import Stage6Config
from app.materialization.service import _delivery_pacing_findings
from app.materialization.types import AuthoredTimeline, QualityFinding, TimelineSegment

_SOURCE_ID = "cafc529e-b0d8-401b-b439-2721a57de73c"
_CARD_TEXT = "المبالغة في الطلبات لا تعني أن الحب أقل."  # 38 chars, fits 3 lines


def _timeline(card_text: str = _CARD_TEXT, *, narration: str | None = None) -> AuthoredTimeline:
    return AuthoredTimeline(
        segments=(
            TimelineSegment(
                segment_id="seg-1-0",
                plan_block_index=0,
                kind="SOURCE",
                output_start=0.0,
                output_end=4.0,
                source_start=583.16,
                source_end=587.14,
                is_hero=True,
            ),
            TimelineSegment(
                segment_id="seg-2-1",
                plan_block_index=1,
                kind="AUTHORED_CARD",
                output_start=4.0,
                output_end=7.72,
                text=card_text,
                narration_chunk_id=narration,
                delivery="NARRATION" if narration else "STANDALONE_CARD",
                delivery_rationale="test delivery rationale",
            ),
        ),
        frame_rate=Fraction(25, 1),
        sample_rate=48000,
        output_duration=7.72,
        output_frame_count=193,
        output_sample_count=370560,
        schema_version="stage6-timeline-v2",
    )


def test_dominant_silent_card_without_length_assessment_defers_readiness() -> None:
    timeline = _timeline()
    layout = build_authored_ass(timeline, play_res_x=1080, play_res_y=1920)
    assert not layout.failed
    pacing = _delivery_pacing_findings(timeline, ())
    assert len(pacing) == 1
    assert pacing[0].level == "UNKNOWN"
    assert pacing[0].reason_codes == ("AUTHORED_PAUSE_NECESSITY_NOT_ASSESSED",)
    evidence = pacing[0].evidence
    assert evidence["dominant"] is True
    assert evidence["duration_seconds"] == 3.72
    assert evidence["reading_seconds"] > 0
    assert evidence["fraction_of_timeline"] > 0.4
    assert evidence["silence_interval"]["has_source_or_narration_audio"] is False
    assert evidence["delivery_rationale"] == "test delivery rationale"

    context = AssessmentContext(
        readable_copy=True,
        render_complete=True,
        render_qc_status="PASS",
        measured_loudness_lufs=-16.0,
        measured_true_peak_dbtp=-2.0,
        extra_findings=(pacing[0],),
    )
    assessment = assess_materialization(context, output_duration=7.72, config=Stage6Config())
    assert assessment.outcome == "DEFERRED_EVIDENCE"
    assert assessment.human_review_required is False


def test_assessed_length_necessity_allows_an_intentional_silent_card() -> None:
    timeline = _timeline()
    assessed = QualityFinding(
        dimension="PACING_CONTEXT_DENSITY",
        level="SATISFIED",
        reason_codes=("CRITIQUE_LENGTH_NECESSITY_PASS",),
    )
    pacing = _delivery_pacing_findings(timeline, (assessed,))
    assert pacing[0].level == "SATISFIED"
    assert pacing[0].reason_codes == ("AUTHORED_PAUSE_NECESSITY_ASSESSED",)

    context = AssessmentContext(
        readable_copy=True,
        render_complete=True,
        render_qc_status="PASS",
        measured_loudness_lufs=-16.0,
        measured_true_peak_dbtp=-2.0,
        extra_findings=(pacing[0],),
    )
    assessment = assess_materialization(context, output_duration=7.72, config=Stage6Config())
    assert assessment.outcome != "DEFERRED_EVIDENCE"


def test_overlong_card_fails_without_truncating_complete_copy() -> None:
    text = " ".join(["كلمة"] * 40)
    result = build_authored_ass(_timeline(text), play_res_x=1080, play_res_y=1920)
    assert result.failed
    assert any(
        "AUTHORED_TEXT_REQUIRES_SHORTENING" in finding["reason_codes"]
        for finding in result.findings
        if finding.get("level") == "FAIL"
    )
    visible = result.data.decode("utf-8")
    assert visible.count("كلمة") == 40


def test_missing_card_copy_cannot_report_a_satisfied_layout() -> None:
    result = build_authored_ass(_timeline(""), play_res_x=1080, play_res_y=1920)
    assert result.failed
    assert any(
        finding.get("level") == "FAIL"
        and "MISSING_REQUIRED_VALUE" in finding.get("reason_codes", ())
        for finding in result.findings
    )
    assert not any(finding.get("level") == "SATISFIED" for finding in result.findings)


def test_card_layout_is_centred_in_the_shared_safe_area() -> None:
    result = build_authored_ass(_timeline(), play_res_x=1080, play_res_y=1920)
    text = result.data.decode("utf-8")
    # Usable area x: 54..918 -> centre 486; y: 230..1440 -> centre 835.
    assert "{\\an5\\pos(486,835)}" in text
    assert "Style: Card,Noto Sans Arabic,88," in text
    assert ",54,162,230,1" in text
    assert "\\an8" not in text
    assert len(result.layout) == 1
    facts = result.layout[0]
    assert facts["delivery"] == "STANDALONE_CARD"
    assert facts["safe_area"]["usable_width"] == 864
    assert facts["wrapped_lines"] <= 3
