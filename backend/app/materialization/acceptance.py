"""Stage 6 final quality assessment and truthful outcome mapping.

Assessment persists categorical dimensions with reasons/evidence/UNKNOWN; there
is no aggregate score. Rights, originality, semantic fidelity, technical quality,
and platform risk stay separate. A successful final state is
``READY_FOR_HUMAN_REVIEW`` with ``human_review_required=true`` and
``publication_ready=false``; technical completion is never publication
authorization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.core.enums import (
    MaterializationOutcome,
    QualityDimension,
    QualityLevel,
)
from app.materialization.policy import (
    REASON_HERO_DAMAGED,
    REASON_LOUDNESS_OUT_OF_POLICY,
    REASON_MANUAL_TRANSCRIPT_REVIEW,
    REASON_MISSING_MEASUREMENT,
    REASON_MISSING_RENDER_EVIDENCE,
    REASON_MISSING_REQUIRED_VALUE,
    REASON_NONFINITE_MEASUREMENT,
    REASON_OVEREDIT_RISK,
    REASON_PROFILE_CONTEXT_MISMATCH,
    REASON_STALE_CRITICAL_EVIDENCE,
    REASON_TRUE_PEAK_EXCEEDED,
    REASON_UNREADABLE_REQUIRED_COPY,
    REASON_UNSUITABLE_ASSET_RIGHTS,
    REASON_UNSUPPORTED_ESSENTIAL_CLAIM,
    STAGE6_QUALITY_POLICY_VERSION,
    Stage6Config,
)
from app.materialization.types import MaterializationAssessment, QualityFinding


@dataclass(frozen=True)
class AssessmentContext:
    """Explicit deterministic context for the Stage 6 assessment."""

    profile_mismatch_fields: tuple[str, ...] = ()
    required_slots_fulfilled: bool = True
    required_narration_required: bool = False
    narration_accepted: bool = True
    tts_available: bool = True
    provider_deferred: bool = False
    unsupported_essential: bool = False
    hero_protected: bool = True
    timeline_conflicts: tuple[Mapping[str, object], ...] = ()
    audio_conflicts: tuple[Mapping[str, object], ...] = ()
    asset_rights_ok: bool = True
    script_audio_match: bool = True
    readable_copy: bool = True
    final_clip_usable: bool = True
    render_qc_status: str | None = None
    render_qc_findings: tuple[Mapping[str, object], ...] = ()
    measured_loudness_lufs: float | None = None
    measured_true_peak_dbtp: float | None = None
    critical_evidence_current: bool = True
    critical_fidelity_unknown: bool = False
    render_complete: bool = True
    repair_count: int = 0
    extra_findings: tuple[QualityFinding, ...] = field(default_factory=tuple)


def _finite(value: float | None) -> bool:
    if value is None:
        return True
    return value == value and value not in (float("inf"), float("-inf"))


def assess_materialization(
    context: AssessmentContext,
    *,
    output_duration: float,
    config: Stage6Config,
) -> MaterializationAssessment:
    """Map deterministic evidence to categorical findings and an outcome."""

    findings: list[QualityFinding] = []
    hard_gates: list[dict[str, object]] = []
    reasons: list[str] = []

    def add(
        dimension: QualityDimension,
        level: QualityLevel,
        reason: str | None,
        evidence: Mapping[str, object] | None = None,
    ) -> None:
        findings.append(
            QualityFinding(
                dimension=dimension.value,
                level=level.value,
                reason_codes=(reason,) if reason else (),
                evidence=dict(evidence or {}),
            )
        )
        if level == QualityLevel.UNMET and reason:
            reasons.append(reason)
            hard_gates.append({"reason_code": reason, "dimension": dimension.value})

    if context.profile_mismatch_fields:
        add(
            QualityDimension.PLATFORM_SUITABILITY,
            QualityLevel.UNMET,
            REASON_PROFILE_CONTEXT_MISMATCH,
            {"fields": list(context.profile_mismatch_fields)},
        )
    if not context.final_clip_usable:
        add(
            QualityDimension.GROUNDING_VERIFICATION,
            QualityLevel.UNKNOWN,
            REASON_MANUAL_TRANSCRIPT_REVIEW,
        )

    if not context.required_slots_fulfilled:
        add(QualityDimension.SLOT_FULFILLMENT, QualityLevel.UNMET, REASON_MISSING_REQUIRED_VALUE)
    else:
        add(QualityDimension.SLOT_FULFILLMENT, QualityLevel.SATISFIED, None)

    if context.unsupported_essential:
        add(
            QualityDimension.GROUNDING_VERIFICATION,
            QualityLevel.UNMET,
            REASON_UNSUPPORTED_ESSENTIAL_CLAIM,
        )
    else:
        add(QualityDimension.GROUNDING_VERIFICATION, QualityLevel.SATISFIED, None)

    if not context.hero_protected or context.timeline_conflicts:
        add(
            QualityDimension.HERO_PRESERVATION,
            QualityLevel.UNMET,
            REASON_HERO_DAMAGED,
            {"conflicts": [dict(item) for item in context.timeline_conflicts]},
        )
    else:
        add(QualityDimension.HERO_PRESERVATION, QualityLevel.SATISFIED, None)

    if not context.readable_copy:
        add(
            QualityDimension.VISUAL_CAPTION_READABILITY,
            QualityLevel.UNMET,
            REASON_UNREADABLE_REQUIRED_COPY,
        )
    else:
        add(QualityDimension.VISUAL_CAPTION_READABILITY, QualityLevel.SATISFIED, None)

    if not context.asset_rights_ok:
        add(
            QualityDimension.PLATFORM_SUITABILITY,
            QualityLevel.UNMET,
            REASON_UNSUITABLE_ASSET_RIGHTS,
        )
    else:
        add(QualityDimension.PLATFORM_SUITABILITY, QualityLevel.SATISFIED, None)

    if not context.script_audio_match:
        add(QualityDimension.FIDELITY_ATTRIBUTION, QualityLevel.UNMET, "SCRIPT_AUDIO_MISMATCH")

    critical_evidence_missing = False
    if context.critical_fidelity_unknown:
        add(QualityDimension.FIDELITY_ATTRIBUTION, QualityLevel.UNKNOWN, None)
        critical_evidence_missing = True

    # Final readiness requires actual, complete render/QC/loudness evidence.
    # Missing or non-finite critical evidence defers; a genuine mastering-policy
    # violation is assessed as unmet.
    if not context.render_complete:
        add(QualityDimension.AUDIO_MASTERING_INTELLIGIBILITY, QualityLevel.UNKNOWN, None)
    elif context.render_qc_status is None:
        add(
            QualityDimension.AUDIO_MASTERING_INTELLIGIBILITY,
            QualityLevel.UNKNOWN,
            REASON_MISSING_RENDER_EVIDENCE,
        )
        critical_evidence_missing = True
    elif context.measured_loudness_lufs is None or context.measured_true_peak_dbtp is None:
        add(
            QualityDimension.AUDIO_MASTERING_INTELLIGIBILITY,
            QualityLevel.UNKNOWN,
            REASON_MISSING_MEASUREMENT,
        )
        critical_evidence_missing = True
    elif not _finite(context.measured_loudness_lufs) or not _finite(
        context.measured_true_peak_dbtp
    ):
        add(
            QualityDimension.AUDIO_MASTERING_INTELLIGIBILITY,
            QualityLevel.UNKNOWN,
            REASON_NONFINITE_MEASUREMENT,
        )
        critical_evidence_missing = True
    else:
        loudness = float(context.measured_loudness_lufs)
        true_peak = float(context.measured_true_peak_dbtp)
        loudness_ok = (
            abs(loudness - config.target_integrated_lufs) <= config.mastering_loudness_tolerance_lu
        )
        peak_ok = true_peak <= config.true_peak_ceiling_dbtp
        if context.render_qc_status == "FAIL":
            add(
                QualityDimension.AUDIO_MASTERING_INTELLIGIBILITY,
                QualityLevel.UNMET,
                "RENDER_QC_FAIL",
                {"qc_status": context.render_qc_status},
            )
        elif not loudness_ok:
            add(
                QualityDimension.AUDIO_MASTERING_INTELLIGIBILITY,
                QualityLevel.UNMET,
                REASON_LOUDNESS_OUT_OF_POLICY,
                {"measured_loudness_lufs": loudness, "target": config.target_integrated_lufs},
            )
        elif not peak_ok:
            add(
                QualityDimension.AUDIO_MASTERING_INTELLIGIBILITY,
                QualityLevel.UNMET,
                REASON_TRUE_PEAK_EXCEEDED,
                {"measured_true_peak_dbtp": true_peak, "ceiling": config.true_peak_ceiling_dbtp},
            )
        else:
            add(
                QualityDimension.AUDIO_MASTERING_INTELLIGIBILITY,
                QualityLevel.SATISFIED
                if context.render_qc_status == "PASS"
                else QualityLevel.CONCERN,
                None,
                {"qc_status": context.render_qc_status},
            )

    if context.audio_conflicts:
        add(
            QualityDimension.OVEREDIT_DISTRACTION,
            QualityLevel.CONCERN,
            None,
            {"conflicts": [dict(item) for item in context.audio_conflicts]},
        )

    if output_duration > config.max_clip_seconds:
        add(
            QualityDimension.PACING_CONTEXT_DENSITY,
            QualityLevel.UNMET,
            REASON_OVEREDIT_RISK,
            {"output_duration": round(output_duration, 6), "cap": config.max_clip_seconds},
        )

    if not context.critical_evidence_current:
        add(
            QualityDimension.GROUNDING_VERIFICATION,
            QualityLevel.UNKNOWN,
            REASON_STALE_CRITICAL_EVIDENCE,
        )

    findings.extend(context.extra_findings)

    unmet = [finding for finding in findings if finding.level == QualityLevel.UNMET.value]
    concern = [finding for finding in findings if finding.level == QualityLevel.CONCERN.value]
    unknown = [finding for finding in findings if finding.level == QualityLevel.UNKNOWN.value]

    outcome, human_review_required = _map_outcome(
        context,
        unmet=unmet,
        concern=concern,
        unknown=unknown,
        critical_evidence_missing=critical_evidence_missing,
    )

    return MaterializationAssessment(
        outcome=outcome.value,
        human_review_required=human_review_required,
        publication_ready=False,
        findings=tuple(findings),
        reason_codes=tuple(dict.fromkeys(reasons)),
        hard_gates=tuple(hard_gates),
        policy_version=STAGE6_QUALITY_POLICY_VERSION,
        metrics={
            "unmet_count": len(unmet),
            "concern_count": len(concern),
            "unknown_count": len(unknown),
            "repair_count": context.repair_count,
            "output_duration": round(output_duration, 6),
        },
    )


def _map_outcome(
    context: AssessmentContext,
    *,
    unmet: Sequence[QualityFinding],
    concern: Sequence[QualityFinding],
    unknown: Sequence[QualityFinding],
    critical_evidence_missing: bool,
) -> tuple[MaterializationOutcome, bool]:
    if context.profile_mismatch_fields:
        return MaterializationOutcome.UPSTREAM_REVALIDATION_REQUIRED, False
    if not context.final_clip_usable:
        return MaterializationOutcome.NEEDS_MANUAL_TRANSCRIPT_REVIEW, False
    if context.unsupported_essential:
        return MaterializationOutcome.BLOCKED_VERIFICATION, False
    if context.required_narration_required and not context.narration_accepted:
        if not context.tts_available:
            return MaterializationOutcome.TTS_UNAVAILABLE, False
        return MaterializationOutcome.DEFERRED_PROVIDER, False
    if context.provider_deferred:
        return MaterializationOutcome.DEFERRED_PROVIDER, False
    if not context.required_slots_fulfilled:
        return MaterializationOutcome.NO_OUTPUT, False
    if critical_evidence_missing:
        # Final review cannot proceed on missing/unknown critical evidence; the
        # materialization is truthfully deferred rather than rejected or approved.
        return MaterializationOutcome.DEFERRED_EVIDENCE, False
    if unmet:
        return MaterializationOutcome.REJECTED_QUALITY, False
    if not context.render_complete:
        # Content is materialized and awaiting the renderer; it is not yet
        # human-review-ready because no render/QC/loudness evidence exists.
        return MaterializationOutcome.MATERIALIZED_WITH_CAUTION, False
    if not context.critical_evidence_current:
        return MaterializationOutcome.UPSTREAM_REVALIDATION_REQUIRED, False
    if concern or unknown:
        return MaterializationOutcome.MATERIALIZED_WITH_CAUTION, True
    return MaterializationOutcome.READY_FOR_HUMAN_REVIEW, True


__all__ = [
    "AssessmentContext",
    "assess_materialization",
]
