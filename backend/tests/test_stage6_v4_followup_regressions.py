"""Stage 6 V4 follow-up regressions for the independent SOL review (R1-R4).

These are offline contract tests. Recorded/fake provider responses exercise the
review boundary only; they do not establish live semantic judgment or human
video acceptance. R4 additionally encodes the emitted ASS with the actual local
libass renderer and measures glyph ink.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from test_stage6_gemini_boundary import NOVEL_COPY, _inputs_with_draft

from app.materialization.acceptance import AssessmentContext, assess_materialization
from app.materialization.components import (
    _critique_result_from_dict,
    _critique_result_to_dict,
)
from app.materialization.overlays import build_authored_ass
from app.materialization.policy import DEFAULT_CONFIG, CallLedger
from app.materialization.providers import (
    CRITIQUE_DIMENSIONS,
    BlockCritique,
    CritiqueDimensionFinding,
    CritiqueRequest,
    CritiqueResult,
    DeterministicWritingProvider,
    materialize_script,
)
from app.materialization.service import _delivery_pacing_findings, assemble_result
from app.materialization.types import AuthoredTimeline, QualityFinding, TimelineSegment

_CARD_TEXT = "المبالغة في الطلبات لا تعني أن الحب أقل."


class _RecordingCritique:
    provider_name = "recorded"
    hosted_provider = True

    def __init__(self, result: CritiqueResult) -> None:
        self._result = result
        self.calls = 0

    def available(self) -> bool:
        return True

    def runtime_identity(self) -> dict[str, object]:
        return {"provider": "recorded", "model": "m", "api_version": "v1"}

    def critique(self, request: CritiqueRequest) -> CritiqueResult:
        self.calls += 1
        self.request = request
        return self._result

    def release(self) -> None:
        return None


def _complete_block(
    *,
    index: int = 1,
    copy: str = NOVEL_COPY,
    refs: tuple[str, ...] = ("block:0",),
    **overrides: str,
) -> BlockCritique:
    findings: list[CritiqueDimensionFinding] = []
    for dimension in CRITIQUE_DIMENSIONS:
        level = overrides.get(dimension, "PASS")
        if dimension == "INTENT_FULFILLMENT":
            findings.append(
                CritiqueDimensionFinding(
                    dimension,
                    level,
                    copy_quote=copy,
                    grounding_refs=refs,
                    added_relation="The copy names the mechanism behind the drop.",
                )
            )
        else:
            findings.append(CritiqueDimensionFinding(dimension, level))
    return BlockCritique(block_index=index, findings=tuple(findings))


def _static_result(block: BlockCritique | None) -> CritiqueResult:
    return CritiqueResult(
        blocks={} if block is None else {block.block_index: block},
        provider_identity={"provider": "recorded", "model": "m", "api_version": "v1"},
    )


def _materialize(copy: str, critique: _RecordingCritique | None, **overrides: object):
    inputs = _inputs_with_draft(copy)
    if overrides:
        blocks = [dict(item) for item in inputs.blocks]
        blocks[1] = {**blocks[1], **overrides}
        inputs = dataclasses.replace(inputs, blocks=tuple(blocks))
    return materialize_script(
        DeterministicWritingProvider(),
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=critique is not None,
        critique_provider=critique,
    ), inputs


# --- R1: content-free substantive copy and the required intent gate ---------


@pytest.mark.parametrize("copy", ["...", "؟،!", "... !!", "the and"])
def test_content_free_substantive_copy_is_rejected_before_review(copy: str) -> None:
    recorder = _RecordingCritique(_static_result(_complete_block(copy=copy)))
    script, _inputs = _materialize(copy, recorder)

    assert not script.accepted
    assert script.authored_blocks == ()
    assert any(
        finding.dimension == "SUBSTANTIVE_VALUE"
        and finding.level == "UNMET"
        and finding.evidence.get("reason") == "CONTENT_FREE"
        for finding in script.findings
    )
    # A deterministic hard failure is not reviewable and must not spend a call.
    assert recorder.calls == 0


def test_content_free_copy_cannot_reach_final_readiness() -> None:
    recorder = _RecordingCritique(_static_result(_complete_block(copy="...")))
    script, inputs = _materialize("...", recorder)
    result = assemble_result(
        inputs,
        script=script,
        narration=(),
        supporting=(),
        config=DEFAULT_CONFIG,
        ledger=CallLedger(),
        tts_available=False,
        narration_required=False,
    )
    context = dataclasses.replace(
        AssessmentContext.from_dict(result.assessment_context),
        render_complete=True,
        render_qc_status="PASS",
        measured_loudness_lufs=-16.0,
        measured_true_peak_dbtp=-2.0,
    )
    assessment = assess_materialization(
        context, output_duration=result.timeline.output_duration, config=DEFAULT_CONFIG
    )
    assert assessment.outcome != "READY_FOR_HUMAN_REVIEW"
    assert assessment.outcome != "MATERIALIZED_WITH_CAUTION"
    assert not assessment.publication_ready
    assert recorder.calls == 0


def test_essential_substantive_copy_never_skips_intent_review() -> None:
    # A deterministic lexical acceptance must not authorize a required
    # substantive block without a semantic intent/redundancy judgment.
    script, _inputs = _materialize(NOVEL_COPY, None)
    assert not script.accepted
    assert script.deferred
    assert script.authored_blocks == ()


def test_explicit_source_as_evidence_exception_still_routes_to_review() -> None:
    recorder = _RecordingCritique(_static_result(_complete_block()))
    script, _inputs = _materialize(
        NOVEL_COPY,
        recorder,
        substantive_value_kind="SOURCE_AS_EVIDENCE",
        dependency_ids=["dep-1"],
    )
    assert script.accepted
    assert not any(
        "UNSUPPORTED_ESSENTIAL_CLAIM" in finding.reason_codes for finding in script.findings
    )


# --- R2: no boolean-only positive critique synthesis ------------------------


def test_legacy_positive_checkpoint_defers_without_resend() -> None:
    legacy = _critique_result_from_dict(
        {
            "accepted": [{"index": 1, "value": True}],
            "blocks": [],
            "provider_identity": {"provider": "gemini"},
            "deferred": False,
        }
    )
    assert legacy.deferred is True
    assert legacy.reason_code == "MALFORMED_CRITIQUE_EVIDENCE"
    assert legacy.blocks == {}
    assert legacy.accepted == {}

    recorder = _RecordingCritique(legacy)
    script, _inputs = _materialize(NOVEL_COPY, recorder)
    assert script.deferred
    assert script.authored_blocks == ()
    assert recorder.calls == 1
    assert all(
        "Fake critique accepted the supplied relation." != block.adds_beyond_source
        for block in script.authored_blocks
    )


def test_boolean_only_result_is_never_persisted_as_authority() -> None:
    from app.materialization.providers import CritiqueResult as Result

    serialized = _critique_result_to_dict(
        Result(accepted={1: True}, provider_identity={"provider": "gemini"})
    )
    assert serialized["accepted"] == []
    parsed = _critique_result_from_dict(serialized)
    assert parsed.blocks == {}
    assert parsed.decision(1) is None
    assert parsed.verdict(1) is None


def test_incomplete_positive_critique_cannot_authorize_readiness() -> None:
    block = BlockCritique(
        block_index=1,
        findings=(CritiqueDimensionFinding("INTENT_FULFILLMENT", "PASS", copy_quote=NOVEL_COPY),),
    )
    recorder = _RecordingCritique(_static_result(block))
    script, _inputs = _materialize(NOVEL_COPY, recorder)
    assert script.deferred
    assert script.authored_blocks == ()


def test_complete_structured_positive_evidence_is_required_and_used() -> None:
    recorder = _RecordingCritique(_static_result(_complete_block()))
    script, _inputs = _materialize(NOVEL_COPY, recorder)
    assert script.accepted
    assert script.authored_blocks[0].adds_beyond_source == (
        "The copy names the mechanism behind the drop."
    )


# --- R3: length necessity is bound to its own delivered block ---------------


def _timeline(
    *, text: str = _CARD_TEXT, block_index: int = 1, duration: float = 3.72
) -> AuthoredTimeline:
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
                plan_block_index=block_index,
                kind="AUTHORED_CARD",
                output_start=4.0,
                output_end=4.0 + duration,
                text=text,
                delivery="STANDALONE_CARD",
                delivery_rationale="test delivery rationale",
            ),
        ),
        frame_rate=Fraction(25, 1),
        sample_rate=48000,
        output_duration=4.0 + duration,
        output_frame_count=round((4.0 + duration) * 25),
        output_sample_count=round((4.0 + duration) * 48000),
        schema_version="stage6-timeline-v2",
    )


def _card_timeline(text: str, *, duration: float = 4.0) -> AuthoredTimeline:
    return AuthoredTimeline(
        segments=(
            TimelineSegment(
                segment_id="card-1",
                plan_block_index=1,
                kind="CARD",
                output_start=0.0,
                output_end=duration,
                text=text,
            ),
        ),
        frame_rate=Fraction(25, 1),
        sample_rate=48000,
        output_duration=duration,
        output_frame_count=round(duration * 25),
        output_sample_count=round(duration * 48000),
        schema_version="stage6-timeline-v2",
    )


def _length_pass(**evidence: object) -> QualityFinding:
    return QualityFinding(
        dimension="PACING_CONTEXT_DENSITY",
        level="SATISFIED",
        reason_codes=("CRITIQUE_LENGTH_NECESSITY_PASS",),
        evidence=evidence,
    )


def test_sibling_length_pass_does_not_authorize_another_block() -> None:
    assessed = _length_pass(
        block_index=1,
        copy=_CARD_TEXT,
        reviewed_delivery="STANDALONE_CARD",
        reviewed_duration_seconds=3.72,
    )
    pacing = _delivery_pacing_findings(_timeline(block_index=2), (assessed,))
    assert len(pacing) == 1
    assert pacing[0].level == "UNKNOWN"
    assert pacing[0].reason_codes == ("AUTHORED_PAUSE_NECESSITY_NOT_ASSESSED",)


def test_matching_length_evidence_authorizes_its_own_silent_card() -> None:
    assessed = _length_pass(
        block_index=1,
        copy=_CARD_TEXT,
        reviewed_delivery="STANDALONE_CARD",
        reviewed_duration_seconds=3.72,
    )
    pacing = _delivery_pacing_findings(_timeline(), (assessed,))
    assert pacing[0].level == "SATISFIED"
    assert pacing[0].reason_codes == ("AUTHORED_PAUSE_NECESSITY_ASSESSED",)
    assert pacing[0].evidence["length_necessity_reviewed"] is True


@pytest.mark.parametrize(
    "text,duration",
    [("A different copy entirely.", 3.72), (_CARD_TEXT, 4.0)],
)
def test_stale_copy_or_duration_does_not_authorize_the_card(text: str, duration: float) -> None:
    assessed = _length_pass(
        block_index=1,
        copy=_CARD_TEXT,
        reviewed_delivery="STANDALONE_CARD",
        reviewed_duration_seconds=3.72,
    )
    pacing = _delivery_pacing_findings(_timeline(text=text, duration=duration), (assessed,))
    assert pacing[0].level == "UNKNOWN"


def test_flexible_without_narration_is_a_silent_card_requiring_length() -> None:
    copy = "The drop reflects selection bias."
    inputs = _inputs_with_draft(copy)
    blocks = list(inputs.blocks)
    blocks[1] = {
        **blocks[1],
        "delivery_intent": "FLEXIBLE",
        "estimated_duration": 0.5,
    }
    inputs = dataclasses.replace(inputs, blocks=tuple(blocks))
    recorder = _RecordingCritique(
        _static_result(_complete_block(copy=copy, LENGTH_NECESSITY="UNKNOWN"))
    )
    script = materialize_script(
        DeterministicWritingProvider(),
        inputs,
        ledger=CallLedger(),
        config=DEFAULT_CONFIG,
        allow_hosted=True,
        critique_provider=recorder,
    )

    target = recorder.request.targets[0]
    assert target.proposed_delivery["delivery"] == "STANDALONE_CARD"
    assert target.proposed_delivery["silent"] is True
    assert target.proposed_delivery["planned_duration_seconds"] == pytest.approx(
        max(1.2, len(copy) / 14.0)
    )
    assert target.require_length_necessity is True
    assert script.deferred
    assert script.authored_blocks == ()


def test_unknown_length_necessity_defers_before_rendering() -> None:
    recorder = _RecordingCritique(_static_result(_complete_block(LENGTH_NECESSITY="UNKNOWN")))
    script, _inputs = _materialize(NOVEL_COPY, recorder)
    assert script.deferred
    assert script.authored_blocks == ()


# --- R4: measured readability for actual rendered glyph ink -----------------

_FFMPEG = shutil.which("ffmpeg")
_R4_SKIP = pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg/libass measurement unavailable")


def _encode_card(data: bytes, destination: Path, *, width: int = 1080, height: int = 1920) -> Path:
    ass_path = destination / "card.ass"
    ass_path.write_bytes(data)
    mp4 = destination / "card.mp4"
    subprocess.run(
        [
            str(_FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r=25:d=2",
            "-vf",
            f"ass={ass_path}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    return mp4


def _decoded_ink_bbox(
    mp4: Path, *, width: int = 1080, height: int = 1920
) -> tuple[int, int, int, int]:
    decoded = subprocess.run(
        [
            str(_FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0.5",
            "-i",
            str(mp4),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    frame = np.frombuffer(decoded.stdout, np.uint8).reshape(height, width, 3)
    ys, xs = np.nonzero(frame.max(axis=2) > 32)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


@_R4_SKIP
def test_wide_latin_card_is_refused_and_the_encoded_mp4_clips(tmp_path: Path) -> None:
    text = "W" * 17
    result = build_authored_ass(_card_timeline(text), play_res_x=1080, play_res_y=1920)
    assert result.failed
    assert any(
        finding.get("reason_codes") == ["AUTHORED_TEXT_EXCEEDS_SAFE_AREA"]
        for finding in result.findings
    )
    assert not any(finding.get("level") == "SATISFIED" for finding in result.findings)
    assert text in result.data.decode("utf-8")

    bbox = _decoded_ink_bbox(_encode_card(result.data, tmp_path))
    # The measured ink is exactly why the old estimate-based PASS was wrong:
    # it starts at the frame edge and crosses the 54..918 safe area.
    assert bbox[0] < 54
    assert bbox[2] > 918


@_R4_SKIP
def test_long_unbreakable_technical_token_is_refused_without_truncation(
    tmp_path: Path,
) -> None:
    text = "https://example.com/" + "a" * 80
    result = build_authored_ass(_card_timeline(text), play_res_x=1080, play_res_y=1920)
    assert result.failed
    assert any(
        finding.get("reason_codes") == ["AUTHORED_TEXT_EXCEEDS_SAFE_AREA"]
        for finding in result.findings
    )
    assert not any(finding.get("level") == "SATISFIED" for finding in result.findings)
    assert text in result.data.decode("utf-8")


@_R4_SKIP
def test_short_mixed_arabic_english_card_is_measured_inside_the_safe_area(
    tmp_path: Path,
) -> None:
    text = "أنا creator"
    result = build_authored_ass(_card_timeline(text), play_res_x=1080, play_res_y=1920)
    assert not result.failed
    assert any(
        finding.get("reason_codes") == ["AUTHORED_LAYOUT_READABLE"] for finding in result.findings
    )
    facts = result.layout[0]
    assert facts["measurement"] == "LIBASS_RASTER"
    assert facts["measured_ink_bbox"] is not None
    emitted = result.data.decode("utf-8")
    assert "أنا" in emitted
    assert "creator" in emitted

    x1, y1, x2, y2 = _decoded_ink_bbox(_encode_card(result.data, tmp_path))
    assert x1 >= 54
    assert x2 <= 918
    assert y1 >= 230
    assert y2 <= 1440


@_R4_SKIP
def test_short_arabic_positive_card_remains_readable() -> None:
    result = build_authored_ass(_card_timeline(_CARD_TEXT), play_res_x=1080, play_res_y=1920)
    assert not result.failed
    assert any(
        finding.get("reason_codes") == ["AUTHORED_LAYOUT_READABLE"] for finding in result.findings
    )
