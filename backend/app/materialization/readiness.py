"""Authoritative read-only Stage 6 readiness view, shared by API and export.

This module is the single source of truth for whether one materialization is a
genuine, review-ready transformed Short. It never maintains a second, permissive
approximation: it reuses the finalizer's own authoritative checks —

* live prerequisite/dependency fingerprints and effective candidate/profile scope
  (:func:`finalization._revalidate_live_inputs`);
* the exact reverse-bound render identity, purpose, output fingerprint, bound
  materialization id, and repair revision
  (:func:`finalization._validate_render_binding`);
* the successful-render identity and finite decoded loudness/true-peak evidence
  (:func:`finalization._complete_render_identity_error`);
* the recomputed final assessment over actual stored evidence
  (:func:`finalization._build_assessment_context` + ``assess_materialization``);
* the actual on-disk artifact hash (:func:`finalization._artifact_identity`);

and additionally verifies the exact authored ASS and every referenced narration/
supporting asset from the *bound persisted render request*, never a newer file at
a deterministic path. Any missing or mismatched required asset makes the package
incomplete/diagnostic, never a genuine acceptance. ``publication_ready`` always
stays false.

Recomputing what an assessment *would* conclude is not sufficient: review
readiness requires the finalizer's *committed* publication. A generation is
review-ready only when it is the current eligible generation with live-matching
dependencies, is lifecycle ``COMPLETE``, persists a reviewable semantic outcome
with the human-review flag set, and carries a complete persisted final assessment
whose assessment fingerprint/policy/render binding matches the freshly verified
evidence. Pending finalization, cancellation, a missing assessment, or mismatched
publication evidence can therefore never become review-ready through GET/export.

Freshness, lifecycle, and review readiness are kept separate. Pending work may
have current inputs while remaining unreviewable; the absence of a STALE reason is
not proof of currentness (an unavailable verdict reports ``UNKNOWN`` freshness and
``is_effective=False``).

It is strictly read-only: it performs no writes, no dispatch, and no mutation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.enums import MaterializationLifecycle, MaterializationOutcome, RenderArtifactPurpose
from app.materialization.acceptance import assess_materialization
from app.materialization.finalization import (
    ASSESSMENT_POLICY_VERSION,
    _artifact_identity,
    _assessment_fingerprint,
    _build_assessment_context,
    _complete_render_identity_error,
    _output_duration,
    _revalidate_live_inputs,
    _validate_render_binding,
)
from app.materialization.policy import DEFAULT_CONFIG, Stage6Config
from app.models import ClipMaterialization
from app.models.render_execution import RenderExecution
from app.services.storage import StorageService

TRANSFORMED_SHORT = RenderArtifactPurpose.TRANSFORMED_SHORT.value

#: Reviewable outcomes for *export* purposes: a valid READY_FOR_HUMAN_REVIEW, or a
#: MATERIALIZED_WITH_CAUTION whose critical evidence is complete (noncritical
#: concerns only). Both still require human review and are never publish-ready.
_REVIEWABLE_OUTCOMES = frozenset(
    {
        MaterializationOutcome.READY_FOR_HUMAN_REVIEW.value,
        MaterializationOutcome.MATERIALIZED_WITH_CAUTION.value,
    }
)

#: Freshness vocabulary (separate from lifecycle and review readiness).
FRESHNESS_CURRENT = "CURRENT"
FRESHNESS_STALE = "STALE"
FRESHNESS_HISTORICAL = "HISTORICAL"
FRESHNESS_UNKNOWN = "UNKNOWN"


def _sha256_file(path: object) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:  # type: ignore[arg-type]
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferencedAsset:
    """One referenced asset's verification result (narration/supporting/ASS)."""

    kind: str
    relative_path: str
    expected_sha256: str
    actual_sha256: str | None
    present: bool
    verified: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "relative_path": self.relative_path,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "present": self.present,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class ArtifactInspection:
    """Read-only inspection verdict for the reverse-bound transformed artifact.

    ``available`` is true only for the exact reverse-bound ``TRANSFORMED_SHORT``
    render/output/repair identity with a completed render and actual on-disk
    bytes whose size and digest match the recorded reference. Historical verified
    media may remain inspectable (``historical=True``) as explicit historical
    metadata; inspection availability never establishes current review readiness.
    """

    available: bool
    relative_path: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    render_execution_id: str | None = None
    historical: bool = False
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "render_execution_id": self.render_execution_id,
            "historical": self.historical,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Stage6Readiness:
    """The authoritative readiness verdict for one materialization."""

    review_ready: bool
    outcome: str | None
    reasons: tuple[str, ...]
    materialization_id: str
    render_execution_id: str | None
    expected_output_fingerprint: str | None
    artifact_relative_path: str | None
    artifact_sha256: str | None
    freshness: str = FRESHNESS_UNKNOWN
    is_effective: bool = False
    metrics: Mapping[str, object] = field(default_factory=dict)
    assessment_findings: tuple[dict[str, object], ...] = ()
    assets: tuple[ReferencedAsset, ...] = ()

    @property
    def publication_ready(self) -> bool:
        return False

    def as_dict(self) -> dict[str, object]:
        return {
            "review_ready": self.review_ready,
            "outcome": self.outcome,
            "reasons": list(self.reasons),
            "materialization_id": self.materialization_id,
            "render_execution_id": self.render_execution_id,
            "expected_output_fingerprint": self.expected_output_fingerprint,
            "artifact_relative_path": self.artifact_relative_path,
            "artifact_sha256": self.artifact_sha256,
            "freshness": self.freshness,
            "is_effective": self.is_effective,
            "publication_ready": False,
            "metrics": dict(self.metrics),
            "assessment_findings": [dict(item) for item in self.assessment_findings],
            "assets": [asset.as_dict() for asset in self.assets],
        }


def _add_asset(
    storage: StorageService,
    assets: list[ReferencedAsset],
    reasons: list[str],
    kind: str,
    relative: str,
    expected_sha: str,
    missing_reason: str,
    mismatch_reason: str,
) -> None:
    if not relative or not expected_sha:
        reasons.append(missing_reason)
        assets.append(ReferencedAsset(kind, relative, expected_sha, None, False, False))
        return
    try:
        path = storage.resolve_relative(relative)
    except Exception:
        path = None
    if path is None or not path.is_file():
        reasons.append(missing_reason)
        assets.append(ReferencedAsset(kind, relative, expected_sha, None, False, False))
        return
    actual = _sha256_file(path)
    verified = actual == expected_sha.lower()
    if not verified:
        reasons.append(mismatch_reason)
    assets.append(ReferencedAsset(kind, relative, expected_sha, actual, True, verified))


def verify_artifact(
    session: Session,
    storage: StorageService,
    envelope: ClipMaterialization,
) -> ArtifactInspection:
    """Read-only artifact verification shared by the API response and download.

    The requested artifact is served only when the envelope's exact reverse-bound
    ``TRANSFORMED_SHORT`` render (purpose, bound materialization id, expected
    output fingerprint, repair revision) is ``COMPLETE`` and the actual on-disk
    bytes match the recorded size and SHA-256. A corrupt, missing, wrong-purpose,
    or wrong-generation artifact is never served as the requested artifact.
    Historical verified media remains inspectable with explicit historical
    metadata.
    """

    render_id = envelope.render_execution_id
    if render_id is None:
        return ArtifactInspection(False, reason="NO_BOUND_RENDER")
    render = session.get(RenderExecution, render_id, populate_existing=True)
    if render is None:
        return ArtifactInspection(False, reason="RENDER_MISSING")
    if render.artifact_purpose.value != TRANSFORMED_SHORT:
        return ArtifactInspection(False, reason="RENDER_NOT_TRANSFORMED_SHORT")
    if render.lifecycle.value != "COMPLETE":
        return ArtifactInspection(False, reason="RENDER_NOT_COMPLETE")
    # The exact reverse-bound render/output/repair identity: never a substitute
    # row that merely shares a deterministic artifact path.
    binding_error = _validate_render_binding(envelope, render)
    if binding_error is not None:
        return ArtifactInspection(
            False,
            reason=(
                binding_error.reason_codes[0] if binding_error.reason_codes else "BINDING_INVALID"
            ),
        )
    artifact = _artifact_identity(storage, render)
    if isinstance(artifact, str):
        return ArtifactInspection(False, reason=artifact)
    if not _artifact_still_matches(storage, artifact):
        return ArtifactInspection(False, reason="ARTIFACT_CHANGED")
    return ArtifactInspection(
        True,
        relative_path=artifact.relative_path,
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
        render_execution_id=str(render.id),
        historical=not envelope.is_current,
    )


def _measured(render: RenderExecution) -> tuple[object, object]:
    qc = render.qc_result if isinstance(render.qc_result, dict) else {}
    measured = qc.get("measured") if isinstance(qc.get("measured"), dict) else {}
    return measured.get("final_integrated_lufs"), measured.get("final_true_peak_dbtp")


def _artifact_still_matches(storage: StorageService, artifact: object) -> bool:
    relative = getattr(artifact, "relative_path", None)
    size_bytes = getattr(artifact, "size_bytes", None)
    mtime_ns = getattr(artifact, "mtime_ns", None)
    if not isinstance(relative, str) or not isinstance(size_bytes, int) or not isinstance(
        mtime_ns, int
    ):
        return False
    path = (storage.storage_root / relative).resolve()
    try:
        path.relative_to(storage.storage_root)
    except ValueError:
        return False
    if not path.is_file():
        return False
    stat = path.stat()
    return bool(stat.st_size == size_bytes and stat.st_mtime_ns == mtime_ns)


def _compute_freshness(
    envelope: ClipMaterialization,
    inputs: object,
    render: RenderExecution | None,
) -> str:
    if not envelope.is_current:
        return FRESHNESS_HISTORICAL
    if inputs is None:
        return FRESHNESS_STALE
    if render is not None and not render.is_current:
        return FRESHNESS_STALE
    return FRESHNESS_CURRENT


def evaluate_readiness(
    session: Session,
    storage: StorageService,
    envelope: ClipMaterialization,
    *,
    config: Stage6Config = DEFAULT_CONFIG,
) -> Stage6Readiness:
    """Re-derive the authoritative readiness verdict for one materialization."""

    reasons: list[str] = []
    materialization_id = str(envelope.id)

    if not envelope.is_current:
        reasons.append("NOT_CURRENT")
    if envelope.lifecycle is not MaterializationLifecycle.COMPLETE:
        reasons.append("LIFECYCLE_NOT_COMPLETE")

    render_id = envelope.render_execution_id
    render: RenderExecution | None = None
    if render_id is None:
        reasons.append("NO_BOUND_RENDER")
    else:
        render = session.get(RenderExecution, render_id, populate_existing=True)
        if render is None:
            reasons.append("RENDER_MISSING")
        else:
            if not render.is_current:
                reasons.append("RENDER_NOT_CURRENT")
            if render.lifecycle.value != "COMPLETE":
                reasons.append("RENDER_NOT_COMPLETE")
            if render.artifact_purpose.value != TRANSFORMED_SHORT:
                reasons.append("RENDER_NOT_TRANSFORMED_SHORT")
            binding_error = _validate_render_binding(envelope, render)
            if binding_error is not None:
                reasons.extend(binding_error.reason_codes)

    # Live prerequisite/dependency fingerprints and effective candidate/profile scope.
    inputs = _revalidate_live_inputs(session, envelope)
    if inputs is None:
        reasons.append("STALE_UPSTREAM_INPUTS")

    assessment = None
    persisted_fingerprint = ""
    if render is not None:
        measured_integrated, measured_true_peak = _measured(render)
        identity_error = _complete_render_identity_error(
            render, measured_integrated, measured_true_peak
        )
        if identity_error is not None:
            reasons.append(identity_error)
        elif inputs is not None:
            context = _build_assessment_context(
                session,
                envelope,
                render,
                inputs=inputs,
                integrated=float(measured_integrated),  # type: ignore[arg-type]
                true_peak=float(measured_true_peak),  # type: ignore[arg-type]
                config=config,
            )
            assessment = assess_materialization(
                context, output_duration=_output_duration(envelope), config=config
            )
            if not assessment.human_review_required:
                reasons.append("HUMAN_REVIEW_NOT_REQUIRED")
            if assessment.publication_ready:
                reasons.append("PUBLICATION_READY_SET")
            persisted_fingerprint = _verify_committed_assessment(
                envelope, render, assessment, reasons
            )
        else:
            reasons.append("MISSING_FINAL_ASSESSMENT")
    else:
        reasons.append("MISSING_FINAL_ASSESSMENT")

    # Persisted publication-side review eligibility (independent of the recompute).
    if envelope.semantic_outcome is None or (
        envelope.semantic_outcome.value not in _REVIEWABLE_OUTCOMES
    ):
        reasons.append("OUTCOME_NOT_REVIEWABLE")
    if not bool(envelope.human_review_required):
        reasons.append("HUMAN_REVIEW_FLAG_CLEARED")
    if bool(envelope.publication_ready):
        reasons.append("PUBLICATION_READY_SET")

    artifact_relative: str | None = None
    artifact_sha: str | None = None
    if render is not None:
        artifact = _artifact_identity(storage, render)
        if isinstance(artifact, str):
            reasons.append(artifact)
        else:
            artifact_relative = artifact.relative_path
            artifact_sha = artifact.sha256

    # Exact referenced authored ASS and narration/supporting assets.
    assets: list[ReferencedAsset] = []
    if render is not None:
        payload = render.request_payload if isinstance(render.request_payload, dict) else {}
        spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
        authored = spec.get("authored") if isinstance(spec.get("authored"), dict) else {}
        raw_ass = authored.get("authored_ass")
        ass_ref = raw_ass if isinstance(raw_ass, dict) else {}
        _add_asset(
            storage,
            assets,
            reasons,
            "authored_ass",
            str(ass_ref.get("relative_path") or ""),
            str(ass_ref.get("sha256") or ""),
            "AUTHORED_ASS_MISSING",
            "AUTHORED_ASS_HASH_MISMATCH",
        )
    for chunk in envelope.narration_references or []:
        if isinstance(chunk, Mapping):
            _add_asset(
                storage,
                assets,
                reasons,
                "narration",
                str(chunk.get("relative_path") or ""),
                str(chunk.get("sha256") or ""),
                "NARRATION_ASSET_MISSING",
                "NARRATION_ASSET_HASH_MISMATCH",
            )
    for item in envelope.asset_references or []:
        if isinstance(item, Mapping) and str(item.get("relative_path") or ""):
            _add_asset(
                storage,
                assets,
                reasons,
                "supporting",
                str(item.get("relative_path") or ""),
                str(item.get("sha256") or ""),
                "SUPPORTING_ASSET_MISSING",
                "SUPPORTING_ASSET_HASH_MISMATCH",
            )

    deduped = tuple(dict.fromkeys(reasons))
    freshness = _compute_freshness(envelope, inputs, render)
    metrics = dict(envelope.metrics or {})
    return Stage6Readiness(
        review_ready=not deduped,
        outcome=assessment.outcome if assessment is not None else None,
        reasons=deduped,
        materialization_id=materialization_id,
        render_execution_id=str(render_id) if render_id else None,
        expected_output_fingerprint=envelope.output_fingerprint or None,
        artifact_relative_path=artifact_relative,
        artifact_sha256=artifact_sha,
        freshness=freshness,
        is_effective=freshness == FRESHNESS_CURRENT,
        metrics={
            "assessment_fingerprint": (
                persisted_fingerprint or metrics.get("assessment_fingerprint")
            ),
            "assessment_policy_version": metrics.get("assessment_policy_version"),
            "final_integrated_lufs": metrics.get("final_integrated_lufs"),
            "final_true_peak_dbtp": metrics.get("final_true_peak_dbtp"),
        },
        assessment_findings=(
            tuple(finding.as_dict() for finding in assessment.findings)
            if assessment is not None
            else ()
        ),
        assets=tuple(assets),
    )


def _verify_committed_assessment(
    envelope: ClipMaterialization,
    render: RenderExecution,
    assessment: object,
    reasons: list[str],
) -> str:
    """Require the persisted finalizer publication to match the freshly verified evidence.

    Returns the persisted assessment fingerprint when complete, else ``""`` and
    appends a truthful reason. The publication is authoritative only when its
    fingerprint, policy version, and render binding match the freshly recomputed
    assessment; a missing or mismatched publication never authorizes readiness.
    """

    metrics = envelope.metrics if isinstance(envelope.metrics, dict) else {}
    persisted = metrics.get("assessment_fingerprint")
    if not isinstance(persisted, str) or not persisted:
        reasons.append("MISSING_FINAL_ASSESSMENT")
        return ""
    if metrics.get("assessment_policy_version") != ASSESSMENT_POLICY_VERSION:
        reasons.append("ASSESSMENT_POLICY_MISMATCH")
        return ""
    if metrics.get("render_output_fingerprint") not in (None, render.output_fingerprint):
        reasons.append("ASSESSMENT_RENDER_BINDING_MISMATCH")
        return ""
    _, _ = _measured(render)
    qc = render.qc_result if isinstance(render.qc_result, dict) else {}
    measured = qc.get("measured") if isinstance(qc.get("measured"), dict) else {}
    fresh = _assessment_fingerprint(envelope, render, measured, assessment)
    if fresh != persisted:
        reasons.append("ASSESSMENT_FINGERPRINT_MISMATCH")
        return ""
    return persisted


__all__ = [
    "ArtifactInspection",
    "ReferencedAsset",
    "Stage6Readiness",
    "evaluate_readiness",
    "verify_artifact",
]
